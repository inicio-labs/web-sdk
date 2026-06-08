use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use miden_client::Felt;
use miden_client::block::BlockHeader;
use miden_client::note::{BlockNumber, Note, NoteDetails, NoteId, NoteInclusionProof, PswapNote};
use miden_client::pswap::lineage::build_record_from_columns;
use miden_client::pswap::{PswapLineageFilter, PswapLineageRecord, PswapLineageRoundUpdate};
use miden_client::store::input_note_states::{CommittedNoteState, UnverifiedNoteState};
use miden_client::store::{InputNoteRecord, StoreError};
use miden_client::utils::{Deserializable, DeserializationError, Serializable};

use super::IdxdbStore;
use crate::note::utils::{SerializedInputNoteData, serialize_input_note};
use crate::promise::{await_js, await_ok};

mod js_bindings;
use js_bindings::{
    JsPswapRoundUpdate,
    idxdb_apply_pswap_round,
    idxdb_get_pswap_lineage,
    idxdb_list_pswap_lineages,
    idxdb_upsert_pswap_lineage,
};

mod models;
use models::PswapLineageIdxdbObject;

const BY_CREATOR_PREFIX: &str = "ByCreator:";
const ACTIVE_BY_TIP_PREFIX: &str = "ActiveByTipNoteIds:";

impl IdxdbStore {
    pub(crate) async fn upsert_pswap_lineage(
        &self,
        record: &PswapLineageRecord,
    ) -> Result<(), StoreError> {
        // `PswapNote` is a builder-constructed view that miden-standards does
        // not give a `Serializable` impl; its canonical encoded form is the
        // underlying `Note` (`From<PswapNote>` / `TryFrom<&Note>`). Persist the
        // `Note` and rebuild the view in `parse_pswap_lineage`.
        let original_note = Note::from(record.original_pswap.clone());

        let promise = idxdb_upsert_pswap_lineage(
            self.db_id(),
            record.order_id().as_canonical_u64().to_string(),
            original_note.to_bytes(),
            record.current_tip_note_id.to_hex(),
            record.current_depth,
            u64::from(record.remaining_offered.amount()).to_string(),
            u64::from(record.remaining_requested.amount()).to_string(),
            record.state.as_u8(),
            record.created_at_block.as_u32(),
            record.updated_at_block.as_u32(),
            record.creator_account_id().to_hex(),
            record.asset_pair_tag().to_bytes(),
            original_note.id().to_hex(),
        );
        await_ok(promise, "failed to upsert pswap lineage").await
    }

    pub(crate) async fn get_pswap_lineage(
        &self,
        order_id: Felt,
    ) -> Result<Option<PswapLineageRecord>, StoreError> {
        let promise =
            idxdb_get_pswap_lineage(self.db_id(), order_id.as_canonical_u64().to_string());
        let row: Option<PswapLineageIdxdbObject> =
            await_js(promise, "failed to get pswap lineage").await?;

        row.map(parse_pswap_lineage).transpose()
    }

    pub(crate) async fn list_pswap_lineages(
        &self,
        filter: PswapLineageFilter,
    ) -> Result<Vec<PswapLineageRecord>, StoreError> {
        let filter_str = match &filter {
            PswapLineageFilter::All => "All".to_string(),
            PswapLineageFilter::Active => "Active".to_string(),
            PswapLineageFilter::ByCreator(account_id) => {
                format!("{BY_CREATOR_PREFIX}{}", account_id.to_hex())
            },
            PswapLineageFilter::ActiveByTipNoteIds(note_ids) => {
                // Empty input returns no rows; short-circuit before crossing
                // the JS boundary.
                if note_ids.is_empty() {
                    return Ok(Vec::new());
                }
                let ids: Vec<String> = note_ids.iter().map(NoteId::to_hex).collect();
                format!("{ACTIVE_BY_TIP_PREFIX}{}", ids.join(","))
            },
        };

        let promise = idxdb_list_pswap_lineages(self.db_id(), filter_str);
        let rows: Vec<PswapLineageIdxdbObject> =
            await_js(promise, "failed to list pswap lineages").await?;

        rows.into_iter().map(parse_pswap_lineage).collect()
    }

    pub(crate) async fn apply_pswap_round(
        &self,
        update: &PswapLineageRoundUpdate,
    ) -> Result<(), StoreError> {
        // Reconstruct payback + remainder into serialized input-note rows. The
        // JS transaction inserts them (skipping any already present) atomically
        // alongside the lineage advance.
        let at_block_header = update.at_block_header.as_ref();
        let mut input_notes = Vec::new();
        if let Some((payback_note, inclusion_proof)) = &update.payback {
            input_notes.push(serialize_pswap_round_note(
                payback_note,
                inclusion_proof,
                at_block_header,
            ));
        }
        if let Some((remainder_note, inclusion_proof)) = &update.remainder {
            input_notes.push(serialize_pswap_round_note(
                remainder_note,
                inclusion_proof,
                at_block_header,
            ));
        }

        let payload = JsPswapRoundUpdate {
            order_id: update.order_id.as_canonical_u64().to_string(),
            round_depth: update.round_depth,
            // Terminal rounds carry no new tip.
            tip_note_id: update.tip_note_id.map(|id| id.to_hex()),
            remaining_offered: u64::from(update.remaining_offered.amount()).to_string(),
            remaining_requested: u64::from(update.remaining_requested.amount()).to_string(),
            state: update.state.as_u8(),
            updated_at_block: update.at_block.as_u32(),
            input_notes,
        };

        let promise = idxdb_apply_pswap_round(self.db_id(), payload);
        await_ok(promise, "failed to apply pswap round").await
    }
}

/// Rebuilds a [`PswapLineageRecord`] from a persisted row. The immutable
/// fields (creator, assets, note type) are recovered from the serialized
/// `original_pswap`; only the mutable columns are read directly.
fn parse_pswap_lineage(row: PswapLineageIdxdbObject) -> Result<PswapLineageRecord, StoreError> {
    let PswapLineageIdxdbObject {
        original_pswap,
        current_tip_note_id,
        current_depth,
        remaining_offered,
        remaining_requested,
        state,
        created_at_block,
        updated_at_block,
    } = row;

    let note = Note::read_from_bytes(&original_pswap)?;
    let original_pswap = PswapNote::try_from(&note)
        .map_err(|err| StoreError::DataDeserializationError(deser_err(err.to_string())))?;

    let current_tip_note_id = NoteId::try_from_hex(&current_tip_note_id)
        .map_err(|err| StoreError::DataDeserializationError(deser_err(err.to_string())))?;

    let remaining_offered = parse_amount(&remaining_offered)?;
    let remaining_requested = parse_amount(&remaining_requested)?;

    build_record_from_columns(
        original_pswap,
        current_tip_note_id,
        current_depth,
        remaining_offered,
        remaining_requested,
        state,
        BlockNumber::from(created_at_block),
        BlockNumber::from(updated_at_block),
    )
    .map_err(|err| StoreError::DatabaseError(format!("pswap_lineage: {err}")))
}

/// Serializes a reconstructed payback or remainder for the round payload. With
/// `at_block_header` the note lands as `Committed`, otherwise `Unverified`.
fn serialize_pswap_round_note(
    note: &Note,
    inclusion_proof: &NoteInclusionProof,
    at_block_header: Option<&BlockHeader>,
) -> SerializedInputNoteData {
    let metadata = *note.metadata();
    let details = NoteDetails::from(note.clone());
    let attachments = note.attachments().clone();

    let state = match at_block_header {
        Some(header) => CommittedNoteState {
            inclusion_proof: inclusion_proof.clone(),
            metadata,
            block_note_root: header.note_root(),
        }
        .into(),
        None => UnverifiedNoteState {
            metadata,
            inclusion_proof: inclusion_proof.clone(),
        }
        .into(),
    };

    let record = InputNoteRecord::new(details, attachments, None, state);
    serialize_input_note(&record)
}

fn parse_amount(value: &str) -> Result<u64, StoreError> {
    value
        .parse::<u64>()
        .map_err(|_| StoreError::DatabaseError(format!("pswap_lineage: invalid amount {value}")))
}

fn deser_err(msg: String) -> DeserializationError {
    DeserializationError::InvalidValue(msg)
}
