//! Common-subexpression elimination via structural hash-cons on
//! `SymbolicExpression` / `SymbolicExpressionExt` DAGs.
//!
//! Strategy: walk both base and ext constraint expressions bottom-up and
//! intern each unique sub-tree as a numbered node in a unified `CseGraph`. An
//! `Arc::clone` of the same sub-expression hashes identically, so the
//! interning catches both Arc-shared and structurally-equal duplicates.
//!
//! Why a unified graph (one keyspace for base + ext) instead of two: the ext
//! `ExtLeaf::Base(SymbolicExpression<F>)` variant lifts an entire base
//! sub-tree into the ext layer, and we want CSE across that boundary so a
//! base sub-expression referenced from both base and ext constraints
//! collapses to one node.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use miden_crypto::{Felt, field::BasedVectorSpace};
use p3_air::{BaseEntry, BaseLeaf, ExtEntry, ExtLeaf, SymbolicExpr};

use crate::recorder::{QuadFelt, RecordedAir};

pub type NodeId = u32;

/// One node in the CSE'd DAG. Each unique sub-expression — base OR ext —
/// gets exactly one `CseNode` entry. Children are referenced by `NodeId`,
/// not by Arc, so the graph is a flat `Vec<CseNode>` indexed by NodeId.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CseNode {
    LeafBase(BaseLeafKey),
    LeafExt(ExtLeafKey),
    AddBase(NodeId, NodeId),
    SubBase(NodeId, NodeId),
    NegBase(NodeId),
    MulBase(NodeId, NodeId),
    AddExt(NodeId, NodeId),
    SubExt(NodeId, NodeId),
    NegExt(NodeId),
    MulExt(NodeId, NodeId),
}

/// Hashable, comparable representation of a `BaseLeaf<Felt>`. The upstream
/// `BaseLeaf` derives only `Clone, Debug`, so we project it onto a key that
/// captures all distinguishing info.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BaseLeafKey {
    /// Trace column reference. `entry_kind` distinguishes Preprocessed/Main/
    /// Periodic/Public; `entry_offset` is the row offset for variants that
    /// have one (Preprocessed/Main); `index` is the column index.
    Variable {
        entry_kind: u8,
        entry_offset: usize,
        index: usize,
    },
    IsFirstRow,
    IsLastRow,
    IsTransition,
    /// Base-field constant, encoded as its canonical u64 representation.
    Constant(u64),
}

/// Hashable, comparable representation of an `ExtLeaf<Felt, QuadFelt>`.
/// `Base` lifts a base-field sub-expression — referenced by the NodeId of
/// the already-interned base CSE node (so cross-layer sharing works).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExtLeafKey {
    Base(NodeId),
    /// Permutation column / challenge / permutation-value variable.
    /// `entry_kind` distinguishes the three; `entry_offset` is meaningful
    /// only for `Permutation { offset }`; `index` is the column / challenge
    /// index.
    ExtVariable {
        entry_kind: u8,
        entry_offset: usize,
        index: usize,
    },
    /// Extension-field constant, encoded as the two base-field canonical
    /// coefficients (degree-2 binomial extension over Goldilocks).
    ExtConstant([u64; 2]),
}

fn base_leaf_key(leaf: &BaseLeaf<Felt>) -> BaseLeafKey {
    match leaf {
        BaseLeaf::Variable(v) => {
            let (kind, offset) = entry_to_key(&v.entry);
            BaseLeafKey::Variable {
                entry_kind: kind,
                entry_offset: offset,
                index: v.index,
            }
        }
        BaseLeaf::IsFirstRow => BaseLeafKey::IsFirstRow,
        BaseLeaf::IsLastRow => BaseLeafKey::IsLastRow,
        BaseLeaf::IsTransition => BaseLeafKey::IsTransition,
        BaseLeaf::Constant(f) => BaseLeafKey::Constant(f.as_canonical_u64()),
    }
}

fn entry_to_key(entry: &BaseEntry) -> (u8, usize) {
    match entry {
        BaseEntry::Preprocessed { offset } => (0, *offset),
        BaseEntry::Main { offset } => (1, *offset),
        BaseEntry::Periodic => (2, 0),
        BaseEntry::Public => (3, 0),
    }
}

fn ext_entry_to_key(entry: &ExtEntry) -> (u8, usize) {
    match entry {
        ExtEntry::Permutation { offset } => (0, *offset),
        ExtEntry::Challenge => (1, 0),
        ExtEntry::PermutationValue => (2, 0),
    }
}

fn quadfelt_coeffs(q: &QuadFelt) -> [u64; 2] {
    let coeffs: &[Felt] = q.as_basis_coefficients_slice();
    debug_assert_eq!(coeffs.len(), 2, "QuadFelt is degree-2 binomial extension");
    [coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64()]
}

/// CSE'd DAG over all base + ext constraints in a recorded AIR.
#[derive(Clone, Debug)]
pub struct CseGraph {
    /// Flat node table. NodeId is the index into this Vec.
    pub nodes: Vec<CseNode>,
    /// Roots of the base-field constraint trees (one per base constraint).
    /// In the same order as `RecordedAir::constraints_base`.
    pub roots_base: Vec<NodeId>,
    /// Roots of the extension-field constraint trees (one per ext constraint).
    /// In the same order as `RecordedAir::constraints_ext`.
    pub roots_ext: Vec<NodeId>,
}

impl CseGraph {
    /// Build the CSE'd DAG from a recording of `ProcessorAir`.
    pub fn from_recorded(recorded: &RecordedAir) -> Self {
        let mut builder = CseBuilder::default();
        let roots_base: Vec<NodeId> = recorded
            .constraints_base
            .iter()
            .map(|c| builder.intern_base(c))
            .collect();
        let roots_ext: Vec<NodeId> = recorded
            .constraints_ext
            .iter()
            .map(|c| builder.intern_ext(c))
            .collect();
        Self {
            nodes: builder.nodes,
            roots_base,
            roots_ext,
        }
    }

    /// Total node count (base + ext + lifted-base + leaves) — the headline
    /// number for Phase 0a Risk (e). Target < 5k, soft fail at 6k, hard fail
    /// at > 10k per the plan.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Histogram of node-kind counts. Useful for (1) understanding which ops
    /// dominate the tape and (2) informing super-opcode candidate selection
    /// in Phase 4.
    pub fn opcode_histogram(&self) -> BTreeMap<&'static str, usize> {
        let mut hist = BTreeMap::new();
        for n in &self.nodes {
            let key = match n {
                CseNode::LeafBase(_) => "LeafBase",
                CseNode::LeafExt(_) => "LeafExt",
                CseNode::AddBase(..) => "AddBase",
                CseNode::SubBase(..) => "SubBase",
                CseNode::NegBase(..) => "NegBase",
                CseNode::MulBase(..) => "MulBase",
                CseNode::AddExt(..) => "AddExt",
                CseNode::SubExt(..) => "SubExt",
                CseNode::NegExt(..) => "NegExt",
                CseNode::MulExt(..) => "MulExt",
            };
            *hist.entry(key).or_insert(0) += 1;
        }
        hist
    }
}

#[derive(Default)]
struct CseBuilder {
    nodes: Vec<CseNode>,
    interner: HashMap<CseNode, NodeId>,
    /// Cache: Arc<SymbolicExpression<Felt>> as raw pointer → NodeId. Skips
    /// re-walking sub-trees that we've already processed. The recorder's
    /// op_flags computation produces heavy Arc-sharing across constraints.
    base_arc_cache: HashMap<usize, NodeId>,
    ext_arc_cache: HashMap<usize, NodeId>,
}

impl CseBuilder {
    fn intern(&mut self, node: CseNode) -> NodeId {
        if let Some(&id) = self.interner.get(&node) {
            return id;
        }
        let id = self.nodes.len() as NodeId;
        self.nodes.push(node.clone());
        self.interner.insert(node, id);
        id
    }

    fn intern_base(&mut self, expr: &SymbolicExpr<BaseLeaf<Felt>>) -> NodeId {
        match expr {
            SymbolicExpr::Leaf(l) => self.intern(CseNode::LeafBase(base_leaf_key(l))),
            SymbolicExpr::Add { x, y, .. } => {
                let xi = self.intern_base_arc(x);
                let yi = self.intern_base_arc(y);
                self.intern(CseNode::AddBase(xi, yi))
            }
            SymbolicExpr::Sub { x, y, .. } => {
                let xi = self.intern_base_arc(x);
                let yi = self.intern_base_arc(y);
                self.intern(CseNode::SubBase(xi, yi))
            }
            SymbolicExpr::Neg { x, .. } => {
                let xi = self.intern_base_arc(x);
                self.intern(CseNode::NegBase(xi))
            }
            SymbolicExpr::Mul { x, y, .. } => {
                let xi = self.intern_base_arc(x);
                let yi = self.intern_base_arc(y);
                self.intern(CseNode::MulBase(xi, yi))
            }
        }
    }

    fn intern_base_arc(&mut self, expr: &Arc<SymbolicExpr<BaseLeaf<Felt>>>) -> NodeId {
        let raw = Arc::as_ptr(expr) as usize;
        if let Some(&id) = self.base_arc_cache.get(&raw) {
            return id;
        }
        let id = self.intern_base(expr);
        self.base_arc_cache.insert(raw, id);
        id
    }

    fn intern_ext(&mut self, expr: &SymbolicExpr<ExtLeaf<Felt, QuadFelt>>) -> NodeId {
        match expr {
            SymbolicExpr::Leaf(ExtLeaf::Base(base_expr)) => {
                let base_id = self.intern_base(base_expr);
                self.intern(CseNode::LeafExt(ExtLeafKey::Base(base_id)))
            }
            SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v)) => {
                let (kind, offset) = ext_entry_to_key(&v.entry);
                self.intern(CseNode::LeafExt(ExtLeafKey::ExtVariable {
                    entry_kind: kind,
                    entry_offset: offset,
                    index: v.index,
                }))
            }
            SymbolicExpr::Leaf(ExtLeaf::ExtConstant(c)) => {
                self.intern(CseNode::LeafExt(ExtLeafKey::ExtConstant(quadfelt_coeffs(c))))
            }
            SymbolicExpr::Add { x, y, .. } => {
                let xi = self.intern_ext_arc(x);
                let yi = self.intern_ext_arc(y);
                self.intern(CseNode::AddExt(xi, yi))
            }
            SymbolicExpr::Sub { x, y, .. } => {
                let xi = self.intern_ext_arc(x);
                let yi = self.intern_ext_arc(y);
                self.intern(CseNode::SubExt(xi, yi))
            }
            SymbolicExpr::Neg { x, .. } => {
                let xi = self.intern_ext_arc(x);
                self.intern(CseNode::NegExt(xi))
            }
            SymbolicExpr::Mul { x, y, .. } => {
                let xi = self.intern_ext_arc(x);
                let yi = self.intern_ext_arc(y);
                self.intern(CseNode::MulExt(xi, yi))
            }
        }
    }

    fn intern_ext_arc(&mut self, expr: &Arc<SymbolicExpr<ExtLeaf<Felt, QuadFelt>>>) -> NodeId {
        let raw = Arc::as_ptr(expr) as usize;
        if let Some(&id) = self.ext_arc_cache.get(&raw) {
            return id;
        }
        let id = self.intern_ext(expr);
        self.ext_arc_cache.insert(raw, id);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny constructed DAG: shared sub-expression must collapse to one node.
    #[test]
    fn cse_collapses_shared_subtree() {
        // (a + b) * (a + b) where (a + b) is shared.
        let a = Arc::new(SymbolicExpr::Leaf(BaseLeaf::Variable(
            p3_air::SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0),
        )));
        let b = Arc::new(SymbolicExpr::Leaf(BaseLeaf::Variable(
            p3_air::SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 1),
        )));
        // Build (a + b) as an Add node, then square it via Mul with two refs to the same node.
        let add = Arc::new(SymbolicExpr::Add {
            x: a.clone(),
            y: b.clone(),
            degree_multiple: 1,
        });
        let sq = SymbolicExpr::Mul {
            x: add.clone(),
            y: add.clone(),
            degree_multiple: 2,
        };

        let mut builder = CseBuilder::default();
        let _root = builder.intern_base(&sq);

        // Expected unique nodes:
        //   1) LeafBase(Variable col 0)
        //   2) LeafBase(Variable col 1)
        //   3) AddBase(0, 1)
        //   4) MulBase(2, 2)
        // = 4 nodes total.
        assert_eq!(
            builder.nodes.len(),
            4,
            "CSE should collapse the shared (a+b) sub-expression to one node"
        );
    }

    /// Two structurally-equal but Arc-distinct sub-trees should also collapse.
    #[test]
    fn cse_collapses_structurally_equal_subtrees() {
        // (a + b) * (a + b) where the two (a + b) subtrees are independently
        // constructed Arcs (no Arc::clone).
        let mk_var = |idx| {
            Arc::new(SymbolicExpr::Leaf(BaseLeaf::Variable(
                p3_air::SymbolicVariable::<Felt>::new(BaseEntry::Main { offset: 0 }, idx),
            )))
        };
        let mk_add = || {
            Arc::new(SymbolicExpr::Add {
                x: mk_var(0),
                y: mk_var(1),
                degree_multiple: 1,
            })
        };
        let sq = SymbolicExpr::Mul {
            x: mk_add(),
            y: mk_add(),
            degree_multiple: 2,
        };

        let mut builder = CseBuilder::default();
        let _root = builder.intern_base(&sq);

        // Same expected count as the Arc-shared test: structural hash sees
        // through the duplication.
        assert_eq!(
            builder.nodes.len(),
            4,
            "CSE should collapse structurally-equal sub-trees even when not Arc-shared"
        );
    }

    /// Build the CSE'd graph from the actual ProcessorAir and report stats.
    /// This is the Phase 0a Risk (e) measurement — the headline number for
    /// the hard-fail gate.
    #[test]
    fn report_processor_air_cse_stats() {
        let recorded = crate::recorder::record_processor_air();
        let graph = CseGraph::from_recorded(&recorded);

        let total = graph.node_count();
        let n_base_constraints = recorded.constraints_base.len();
        let n_ext_constraints = recorded.constraints_ext.len();
        let hist = graph.opcode_histogram();

        eprintln!(
            "ProcessorAir CSE'd DAG\n\
             ────────────────────────────────────────\n\
             Total unique nodes: {total}\n\
             Roots (base): {} (one per base constraint)\n\
             Roots (ext):  {} (one per ext constraint)\n\
             Soft target: <5k. Soft fail at 6k. Hard fail at >10k.\n\n\
             Opcode histogram:",
            n_base_constraints, n_ext_constraints,
        );
        for (op, count) in &hist {
            eprintln!("  {op:>10}: {count}");
        }

        // Hard-fail gate per the plan.
        assert!(
            total < 10_000,
            "Phase 0a HARD FAIL: post-CSE node count {total} >= 10000. \
             Per the plan, abandon GPU AIR and pivot to wasm-simd128 + GPU Merkle."
        );
        // Soft-fail flag (not assertion) — informs Phase 1 scope.
        if total >= 5_000 {
            eprintln!(
                "WARNING: post-CSE node count {total} >= 5000 — Phase 0a SOFT \
                 FAIL. Compile-time WGSL gen (Phase 0b risk g) becomes more \
                 attractive than tape interpreter."
            );
        }
    }
}
