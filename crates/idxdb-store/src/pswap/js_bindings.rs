use alloc::string::String;
use alloc::vec::Vec;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys;

use crate::note::utils::SerializedInputNoteData;

// PSWAP IndexedDB Operations
#[wasm_bindgen(module = "/src/js/pswap.js")]
extern "C" {
    // GETS
    // --------------------------------------------------------------------------------------------

    #[wasm_bindgen(js_name = getPswapLineage)]
    pub fn idxdb_get_pswap_lineage(db_id: &str, order_id: String) -> js_sys::Promise;

    #[wasm_bindgen(js_name = listPswapLineages)]
    pub fn idxdb_list_pswap_lineages(db_id: &str, filter: String) -> js_sys::Promise;

    // INSERTS
    // --------------------------------------------------------------------------------------------

    #[wasm_bindgen(js_name = upsertPswapLineage)]
    #[allow(clippy::too_many_arguments)]
    pub fn idxdb_upsert_pswap_lineage(
        db_id: &str,
        order_id: String,
        original_pswap: Vec<u8>,
        current_tip_note_id: String,
        current_depth: u32,
        remaining_offered: String,
        remaining_requested: String,
        state: u8,
        created_at_block: u32,
        updated_at_block: u32,
        creator_account_id: String,
        asset_pair_tag: Vec<u8>,
        subscription_anchor_note_id: String,
    ) -> js_sys::Promise;

    #[wasm_bindgen(js_name = applyPswapRound)]
    pub fn idxdb_apply_pswap_round(db_id: &str, payload: JsPswapRoundUpdate) -> js_sys::Promise;
}

/// Payload for one atomic PSWAP round, consumed by the `applyPswapRound` JS
/// function under `pswap.js`. All fields describe the lineage AFTER the round;
/// `input_notes` carries the reconstructed payback + remainder (0..2 entries).
#[wasm_bindgen(getter_with_clone)]
#[derive(Clone)]
pub struct JsPswapRoundUpdate {
    #[wasm_bindgen(js_name = "orderId")]
    pub order_id: String,

    #[wasm_bindgen(js_name = "roundDepth")]
    pub round_depth: u32,

    /// New tip's note id; `None` on terminal rounds (the tip stays frozen).
    #[wasm_bindgen(js_name = "tipNoteId")]
    pub tip_note_id: Option<String>,

    /// u64 amount as a decimal string.
    #[wasm_bindgen(js_name = "remainingOffered")]
    pub remaining_offered: String,

    /// u64 amount as a decimal string.
    #[wasm_bindgen(js_name = "remainingRequested")]
    pub remaining_requested: String,

    pub state: u8,

    #[wasm_bindgen(js_name = "updatedAtBlock")]
    pub updated_at_block: u32,

    /// Reconstructed payback + remainder for this round, already serialized.
    #[wasm_bindgen(js_name = "inputNotes")]
    pub input_notes: Vec<SerializedInputNoteData>,
}
