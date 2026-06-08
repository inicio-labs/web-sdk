import { getDatabase, IPswapLineage } from "./schema.js";
import { upsertInputNote } from "./notes.js";
import { logWebStoreError, uint8ArrayToBase64 } from "./utils.js";

// Mirrors `PswapLineageState` discriminants in miden-client. Persisted in the
// `state` column — do not renumber.
const STATE_ACTIVE = 0;
const STATE_FULLY_FILLED = 1;
const STATE_RECLAIMED = 2;

const BY_CREATOR_PREFIX = "ByCreator:";
const ACTIVE_BY_TIP_PREFIX = "ActiveByTipNoteIds:";

/** Mirror of `note::utils::SerializedInputNoteData`. */
interface SerializedInputNoteData {
  detailsCommitment: string;
  noteId?: string;
  noteAssets: Uint8Array;
  attachments: Uint8Array;
  serialNumber: Uint8Array;
  inputs: Uint8Array;
  noteScriptRoot: string;
  noteScript: Uint8Array;
  nullifier: string;
  createdAt: string;
  stateDiscriminant: number;
  state: Uint8Array;
  consumedBlockHeight?: number;
  consumedTxOrder?: number;
  consumerAccountId?: string;
}

/** Payload for one atomic PSWAP round, mirror of Rust's `JsPswapRoundUpdate`.
 *  All fields describe the lineage AFTER the round. `inputNotes` carries the
 *  reconstructed payback + remainder (0..2 entries). */
interface JsPswapRoundUpdate {
  orderId: string;
  roundDepth: number;
  /** New tip; undefined on terminal rounds (tip stays frozen). */
  tipNoteId?: string;
  remainingOffered: string;
  remainingRequested: string;
  state: number;
  updatedAtBlock: number;
  inputNotes: SerializedInputNoteData[];
}

export async function upsertPswapLineage(
  dbId: string,
  orderId: string,
  originalPswap: Uint8Array,
  currentTipNoteId: string,
  currentDepth: number,
  remainingOffered: string,
  remainingRequested: string,
  state: number,
  createdAtBlock: number,
  updatedAtBlock: number,
  creatorAccountId: string,
  assetPairTag: Uint8Array,
  subscriptionAnchorNoteId: string
) {
  try {
    const db = getDatabase(dbId);
    const data: IPswapLineage = {
      orderId,
      originalPswap: uint8ArrayToBase64(new Uint8Array(originalPswap)),
      currentTipNoteId,
      currentDepth,
      remainingOffered,
      remainingRequested,
      state,
      createdAtBlock,
      updatedAtBlock,
      creatorAccountId,
      // Stored with the same base64 encoding the `tags` table uses, so the
      // terminal-state tag removal in `applyPswapRound` matches exactly.
      assetPairTag: uint8ArrayToBase64(new Uint8Array(assetPairTag)),
      subscriptionAnchorNoteId,
    };
    await db.pswapLineages.put(data);
  } catch (error) {
    logWebStoreError(error, `Failed to upsert pswap lineage: ${orderId}`);
  }
}

export async function getPswapLineage(dbId: string, orderId: string) {
  try {
    const db = getDatabase(dbId);
    const row = await db.pswapLineages.get(orderId);
    return row ?? null;
  } catch (error) {
    logWebStoreError(error, `Failed to get pswap lineage: ${orderId}`);
  }
}

export async function listPswapLineages(dbId: string, filter: string) {
  try {
    const db = getDatabase(dbId);
    if (filter === "Active") {
      return await db.pswapLineages
        .where("state")
        .equals(STATE_ACTIVE)
        .toArray();
    }
    if (filter.startsWith(BY_CREATOR_PREFIX)) {
      const creator = filter.substring(BY_CREATOR_PREFIX.length);
      return await db.pswapLineages
        .where("creatorAccountId")
        .equals(creator)
        .toArray();
    }
    if (filter.startsWith(ACTIVE_BY_TIP_PREFIX)) {
      const idsString = filter.substring(ACTIVE_BY_TIP_PREFIX.length);
      const ids = idsString.length > 0 ? idsString.split(",") : [];
      if (ids.length === 0) {
        return [];
      }
      const rows = await db.pswapLineages
        .where("currentTipNoteId")
        .anyOf(ids)
        .toArray();
      return rows.filter((row) => row.state === STATE_ACTIVE);
    }
    // "All" and any unrecognized filter return every row.
    return await db.pswapLineages.toArray();
  } catch (error) {
    logWebStoreError(error, "Failed to list pswap lineages");
  }
}

/**
 * Applies one PSWAP round atomically. In a single Dexie transaction this:
 *  1. rejects an unknown `orderId` or a non-monotonic `roundDepth`,
 *  2. advances the lineage tip/depth/remaining/state,
 *  3. inserts payback + remainder into `input_notes` (skipping any already
 *     present — public paybacks are pre-inserted by the screener),
 *  4. drops the asset-pair Subscription tag on terminal states.
 *
 * Any throw rejects the transaction, so Dexie rolls the whole thing back and
 * the store surfaces an error rather than committing half-applied state.
 */
export async function applyPswapRound(
  dbId: string,
  payload: JsPswapRoundUpdate
) {
  const db = getDatabase(dbId);
  return await db.dexie.transaction(
    "rw",
    db.pswapLineages,
    db.inputNotes,
    db.notesScripts,
    db.tags,
    async (tx) => {
      const row = await tx.pswapLineages.get(payload.orderId);
      if (!row) {
        throw new Error(
          `apply_pswap_round: unknown order_id ${payload.orderId}`
        );
      }
      // Fail-loud monotonic-depth check — the last line of defense against
      // correlator off-by-ones / duplicate deliveries.
      if (payload.roundDepth !== row.currentDepth + 1) {
        throw new Error(
          `apply_pswap_round: round_depth ${payload.roundDepth} does not advance current_depth ${row.currentDepth} by 1`
        );
      }

      await tx.pswapLineages.put({
        ...row,
        // Terminal rounds carry no new tip; the previous tip stays frozen.
        currentTipNoteId: payload.tipNoteId ?? row.currentTipNoteId,
        currentDepth: payload.roundDepth,
        remainingOffered: payload.remainingOffered,
        remainingRequested: payload.remainingRequested,
        state: payload.state,
        updatedAtBlock: payload.updatedAtBlock,
      });

      // INSERT OR IGNORE payback + remainder. The details commitment is the
      // input-note primary key, so a pre-existing row means the screener
      // already inserted it; leave it intact rather than clobbering its state.
      for (const note of payload.inputNotes) {
        const existing = await tx.inputNotes.get(note.detailsCommitment);
        if (!existing) {
          await upsertInputNote(
            dbId,
            note.detailsCommitment,
            note.noteId,
            note.noteAssets,
            note.attachments,
            note.serialNumber,
            note.inputs,
            note.noteScriptRoot,
            note.noteScript,
            note.nullifier,
            note.createdAt,
            note.stateDiscriminant,
            note.state,
            note.consumedBlockHeight,
            note.consumedTxOrder,
            note.consumerAccountId,
            tx
          );
        }
      }

      if (
        payload.state === STATE_FULLY_FILLED ||
        payload.state === STATE_RECLAIMED
      ) {
        await tx.tags
          .where({
            tag: row.assetPairTag,
            sourceNoteId: "",
            sourceAccountId: "",
            sourceSubscriptionNoteId: row.subscriptionAnchorNoteId,
          })
          .delete();
      }
    }
  );
}
