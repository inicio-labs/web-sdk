//! Register allocator: topological walk over CSE'd DAG, free a register when
//! its last consumer is emitted. Reports max-liveness.
//!
//! Filled in by Unit 5.
