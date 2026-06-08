use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::base64_to_vec_u8_required;

/// A `pswapLineages` row as it crosses the JS boundary. Only the columns
/// consumed by `build_record_from_columns` are declared; the idxdb-internal
/// denormalizations (creator, asset-pair tag, subscription anchor) are ignored
/// here because the record is reconstructed from `original_pswap`.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PswapLineageIdxdbObject {
    #[serde(deserialize_with = "base64_to_vec_u8_required", default)]
    pub original_pswap: Vec<u8>,
    pub current_tip_note_id: String,
    pub current_depth: u32,
    /// u64 amount as a decimal string.
    pub remaining_offered: String,
    /// u64 amount as a decimal string.
    pub remaining_requested: String,
    pub state: u8,
    pub created_at_block: u32,
    pub updated_at_block: u32,
}
