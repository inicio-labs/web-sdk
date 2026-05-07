//! Register allocator: liveness analysis on the CSE'd DAG. Reports the
//! peak live register count (separately for base and ext pools) and the
//! peak private-memory footprint, which is the number that matters against
//! WGSL's `var<private>` cap.
//!
//! Algorithm: walk nodes in CSE'd insertion order (which is topological
//! because `CseBuilder` always interns children before parents). For each
//! node, allocate a register from a per-pool free list. After emitting,
//! decrement the use-count of each child; when a child hits zero remaining
//! consumers, return its register to the free list.

use std::collections::HashMap;

use crate::cse::{CseGraph, CseNode, ExtLeafKey, NodeId};

/// Whether a CSE node holds a base-field or extension-field value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Base,
    Ext,
}

/// Bytes-per-register for each pool (matches the WGSL kernel's QuadFelt =
/// 16 B and Felt = 8 B encoding).
pub const BASE_REG_BYTES: usize = 8;
pub const EXT_REG_BYTES: usize = 16;

/// Result of register allocation on a `CseGraph`.
#[derive(Clone, Debug)]
pub struct AllocResult {
    /// Per-node assigned register id, partitioned by `NodeKind`.
    pub reg_for_node: Vec<u32>,
    pub kind_for_node: Vec<NodeKind>,
    /// Number of distinct base registers allocated overall (the size of the
    /// base register file the WGSL kernel needs).
    pub base_reg_count: u32,
    /// Number of distinct ext registers allocated overall.
    pub ext_reg_count: u32,
    /// Peak live base registers at any program point.
    pub base_max_live: u32,
    /// Peak live ext registers at any program point.
    pub ext_max_live: u32,
    /// Peak private-memory footprint in bytes:
    /// `base_max_live * BASE_REG_BYTES + ext_max_live * EXT_REG_BYTES`.
    /// This is what's compared against the device's actual `var<private>`
    /// cap.
    pub peak_private_bytes: u32,
}

/// Classify a `CseNode` as base- or ext-typed (which pool its register
/// belongs to).
fn node_kind(node: &CseNode) -> NodeKind {
    match node {
        CseNode::LeafBase(_)
        | CseNode::AddBase(..)
        | CseNode::SubBase(..)
        | CseNode::NegBase(..)
        | CseNode::MulBase(..) => NodeKind::Base,
        CseNode::LeafExt(_)
        | CseNode::AddExt(..)
        | CseNode::SubExt(..)
        | CseNode::NegExt(..)
        | CseNode::MulExt(..) => NodeKind::Ext,
    }
}

/// Children referenced by a `CseNode`. Used for liveness — when this node
/// is emitted, each child's consumer count decrements.
fn node_children(node: &CseNode) -> Vec<NodeId> {
    match node {
        CseNode::LeafBase(_) => Vec::new(),
        CseNode::LeafExt(ExtLeafKey::Base(b)) => vec![*b],
        CseNode::LeafExt(_) => Vec::new(),
        CseNode::AddBase(x, y)
        | CseNode::SubBase(x, y)
        | CseNode::MulBase(x, y)
        | CseNode::AddExt(x, y)
        | CseNode::SubExt(x, y)
        | CseNode::MulExt(x, y) => vec![*x, *y],
        CseNode::NegBase(x) | CseNode::NegExt(x) => vec![*x],
    }
}

/// Run register allocation over the CSE'd DAG.
///
/// Lifecycle model: each root has one virtual external consumer (the
/// alpha-fold step `acc += alpha_powers_global[k] * value_at(root)`) that
/// fires *immediately* after the root is emitted. So a root with no in-DAG
/// consumers has lifetime = exactly one program point (the moment of
/// emission), and its register frees on the next iteration. This matches
/// what the WGSL kernel will actually do — fold each root's contribution
/// into `acc` as it's computed, then move on.
pub fn allocate(graph: &CseGraph) -> AllocResult {
    let n = graph.nodes.len();

    // Identify roots for fast lookup.
    let mut is_root = vec![false; n];
    for &r in &graph.roots_base {
        is_root[r as usize] = true;
    }
    for &r in &graph.roots_ext {
        is_root[r as usize] = true;
    }

    // remaining_uses[i] = number of consumers of node i still pending.
    // Each in-DAG reference contributes 1; roots get +1 for the virtual
    // alpha-fold consumer (decremented right after the root is emitted).
    let mut remaining_uses: Vec<u32> = vec![0; n];
    for (i, node) in graph.nodes.iter().enumerate() {
        for c in node_children(node) {
            // sanity: child must be earlier in topological order
            debug_assert!(
                (c as usize) < i,
                "CSE graph is not topological: node {i} references child {c}"
            );
            remaining_uses[c as usize] += 1;
        }
    }
    for i in 0..n {
        if is_root[i] {
            remaining_uses[i] += 1;
        }
    }

    // Per-pool free lists + counters.
    let mut base_free: Vec<u32> = Vec::new();
    let mut ext_free: Vec<u32> = Vec::new();
    let mut base_count: u32 = 0;
    let mut ext_count: u32 = 0;
    let mut base_in_use: u32 = 0;
    let mut ext_in_use: u32 = 0;
    let mut base_max_live: u32 = 0;
    let mut ext_max_live: u32 = 0;
    let mut peak_private_bytes: u32 = 0;

    let mut reg_for_node = vec![u32::MAX; n];
    let mut kind_for_node = vec![NodeKind::Base; n];

    // Map NodeId -> the register currently holding its value (so we can free
    // it when its last consumer is emitted).
    let mut live_reg: HashMap<NodeId, (NodeKind, u32)> = HashMap::new();

    for (i, node) in graph.nodes.iter().enumerate() {
        let kind = node_kind(node);
        kind_for_node[i] = kind;

        // Allocate a register for this node from the appropriate pool.
        let reg = match kind {
            NodeKind::Base => match base_free.pop() {
                Some(r) => r,
                None => {
                    let r = base_count;
                    base_count += 1;
                    r
                }
            },
            NodeKind::Ext => match ext_free.pop() {
                Some(r) => r,
                None => {
                    let r = ext_count;
                    ext_count += 1;
                    r
                }
            },
        };
        reg_for_node[i] = reg;
        match kind {
            NodeKind::Base => base_in_use += 1,
            NodeKind::Ext => ext_in_use += 1,
        }
        // Update peak BEFORE freeing children — peak occurs at the moment
        // the new node is allocated and children are still live.
        if base_in_use > base_max_live {
            base_max_live = base_in_use;
        }
        if ext_in_use > ext_max_live {
            ext_max_live = ext_in_use;
        }
        let footprint =
            base_in_use * (BASE_REG_BYTES as u32) + ext_in_use * (EXT_REG_BYTES as u32);
        if footprint > peak_private_bytes {
            peak_private_bytes = footprint;
        }
        live_reg.insert(i as NodeId, (kind, reg));

        // Free children whose last consumer is this node.
        let release = |id: NodeId,
                           live_reg: &mut HashMap<NodeId, (NodeKind, u32)>,
                           base_free: &mut Vec<u32>,
                           ext_free: &mut Vec<u32>,
                           base_in_use: &mut u32,
                           ext_in_use: &mut u32| {
            if let Some((ckind, creg)) = live_reg.remove(&id) {
                match ckind {
                    NodeKind::Base => {
                        base_free.push(creg);
                        *base_in_use -= 1;
                    }
                    NodeKind::Ext => {
                        ext_free.push(creg);
                        *ext_in_use -= 1;
                    }
                }
            }
        };

        for c in node_children(node) {
            let prev = remaining_uses[c as usize];
            debug_assert!(prev > 0, "child {c} consumed too many times");
            remaining_uses[c as usize] = prev - 1;
            if remaining_uses[c as usize] == 0 {
                release(
                    c,
                    &mut live_reg,
                    &mut base_free,
                    &mut ext_free,
                    &mut base_in_use,
                    &mut ext_in_use,
                );
            }
        }

        // If this node is a root, the virtual alpha-fold consumer fires
        // immediately. Decrement its own count and free if zero.
        if is_root[i] {
            let prev = remaining_uses[i];
            debug_assert!(prev > 0, "root {i} virtual consumer underflow");
            remaining_uses[i] = prev - 1;
            if remaining_uses[i] == 0 {
                release(
                    i as NodeId,
                    &mut live_reg,
                    &mut base_free,
                    &mut ext_free,
                    &mut base_in_use,
                    &mut ext_in_use,
                );
            }
        }
    }

    AllocResult {
        reg_for_node,
        kind_for_node,
        base_reg_count: base_count,
        ext_reg_count: ext_count,
        base_max_live,
        ext_max_live,
        peak_private_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cse::CseGraph;
    use crate::recorder::record_processor_air;

    /// Tiny synthetic graph — single chain a → b → c → root. Max live should
    /// be 1 base reg (each child freed immediately when its parent emits).
    /// Wait: actually max live is 2 (parent's reg allocated WHILE child's
    /// reg is still live, peak measured before child is freed).
    #[test]
    fn linear_chain_max_live_is_two() {
        // a leaf, b = -a, c = -b. roots_base = [c].
        let mut nodes = Vec::new();
        nodes.push(CseNode::LeafBase(crate::cse::BaseLeafKey::IsFirstRow));
        nodes.push(CseNode::NegBase(0));
        nodes.push(CseNode::NegBase(1));
        let graph = CseGraph {
            nodes,
            roots_base: vec![2],
            roots_ext: vec![],
        };
        let r = allocate(&graph);
        assert_eq!(r.base_max_live, 2, "peak occurs when parent allocates while child is still live");
        assert_eq!(r.ext_max_live, 0);
        assert_eq!(r.base_reg_count, 2, "free list reuse means we only need 2 distinct base regs");
        assert_eq!(r.ext_reg_count, 0);
    }

    /// Diamond: a, b = -a, c = -a, d = b * c. Both b and c hold 'a' at the
    /// time d is constructed, so base_max_live should be 3 (b, c, d all
    /// alive when d allocates).
    #[test]
    fn diamond_max_live_is_three() {
        let mut nodes = Vec::new();
        nodes.push(CseNode::LeafBase(crate::cse::BaseLeafKey::IsFirstRow)); // a = 0
        nodes.push(CseNode::NegBase(0)); // b = 1
        nodes.push(CseNode::NegBase(0)); // c = 2
        nodes.push(CseNode::MulBase(1, 2)); // d = 3
        let graph = CseGraph {
            nodes,
            roots_base: vec![3],
            roots_ext: vec![],
        };
        let r = allocate(&graph);
        assert_eq!(r.base_max_live, 3, "diamond peak: a freed at b/c, then b+c+d alive when d allocates");
        assert_eq!(r.ext_max_live, 0);
    }

    /// Headline measurement: real Miden AIR. Phase 0a Risk (e) part 2.
    #[test]
    fn report_processor_air_alloc_stats() {
        let recorded = record_processor_air();
        let graph = CseGraph::from_recorded(&recorded);
        let alloc = allocate(&graph);

        eprintln!(
            "ProcessorAir register allocation\n\
             ────────────────────────────────────────\n\
             Base regs allocated:    {}  (peak live: {})\n\
             Ext  regs allocated:    {}  (peak live: {})\n\
             Peak private memory:    {} bytes ({:.1} KB)\n\n\
             WGSL `var<private>` budget interpretation:\n\
               Spec minimum: 16 KB\n\
               Apple Silicon typical: ≥ 16 KB (often 32 KB)\n\
               Comfortable target: ≤ 8 KB (8 * BASE_REG_BYTES * 1024)\n",
            alloc.base_reg_count,
            alloc.base_max_live,
            alloc.ext_reg_count,
            alloc.ext_max_live,
            alloc.peak_private_bytes,
            alloc.peak_private_bytes as f64 / 1024.0,
        );

        // Hard fail: spec-minimum 16 KB private memory is the absolute floor.
        // If we exceed this, even spec-compliant browsers can't run the kernel.
        assert!(
            alloc.peak_private_bytes <= 16 * 1024,
            "Phase 0a HARD FAIL: peak private memory {}B exceeds WGSL spec minimum 16 KB. \
             Per the plan, would need spill-to-workgroup-mem (multi-week redesign) or \
             abandon GPU AIR.",
            alloc.peak_private_bytes
        );
        // Soft fail: comfortable target is 8 KB.
        if alloc.peak_private_bytes > 8 * 1024 {
            eprintln!(
                "WARNING: peak private memory {}B > 8 KB comfortable target — \
                 Phase 0a SOFT FAIL on Risk (e) part 2. Still feasible on Apple \
                 Silicon (typical cap ≥ 16 KB), but tight on lower-spec devices.",
                alloc.peak_private_bytes
            );
        }
    }
}
