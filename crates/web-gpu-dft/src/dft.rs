//! `TwoAdicSubgroupDft<Felt>` impl for `WebGpuDft`.
//!
//! In the CPU-delegating stub, every method forwards to `Radix2DitParallel<Felt>`. Output
//! shape (associated type `Evaluations`) matches `Radix2DitParallel` exactly so the prover
//! sees no behavioral difference. Output bytes are also identical, which is what makes the
//! Phase 3 unit test trivial.
//!
//! When `feature = "real-gpu"` is enabled, the three overridden methods (`dft_batch`,
//! `idft_batch`, `coset_lde_batch`) dispatch to WGSL kernels instead. All other trait methods
//! flow through the default impls in `p3_dft::traits`.

use alloc::sync::Arc;

use miden_crypto::Felt;
use miden_crypto::stark::dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_matrix::bitrev::BitReversedMatrixView;
use p3_matrix::dense::RowMajorMatrix;

use crate::WebGpuDftInner;

/// Cloneable handle to a WebGPU-backed DFT implementation.
///
/// The handle is `Clone` (cheap — Arc bump) and `Default` (clones from a thread-local global
/// installed by [`crate::install_global`]). See crate-level docs for the rationale behind the
/// `Default` workaround.
#[derive(Clone)]
pub struct WebGpuDft {
    pub(crate) inner: Arc<WebGpuDftInner>,
}

impl WebGpuDft {
    /// Create a new GPU-backed DFT handle.
    ///
    /// In the CPU stub build, this is sync and infallible. In the `real-gpu` build, this is
    /// async and may fail if the browser has no `navigator.gpu` adapter — `TransactionProver::
    /// newGpuProver` propagates the error to JS.
    #[cfg(not(feature = "real-gpu"))]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(WebGpuDftInner::default()),
        }
    }

    /// Create a new GPU-backed DFT handle (real-GPU build, async).
    #[cfg(feature = "real-gpu")]
    pub async fn new() -> Result<Self, crate::GpuInitError> {
        // TODO(real-gpu): request adapter + device, compile pipelines, prewarm twiddle cache.
        // For now this is a stub that compiles but isn't wired to wgpu.
        Err(crate::GpuInitError::AdapterUnavailable)
    }
}

impl Default for WebGpuDft {
    fn default() -> Self {
        // Try the thread-local first (populated by install_global from the async init path).
        if let Some(dft) = crate::global() {
            return dft;
        }
        // Fallback: construct a fresh CPU-stub instance. In the real-gpu build this branch is
        // unreachable on the hot path — every prove call passes its dft explicitly via
        // `prove_with_dft`. We keep a fresh-instance fallback here only so the `Default` trait
        // bound is satisfiable at compile time without a runtime panic during tests.
        #[cfg(not(feature = "real-gpu"))]
        {
            Self::new()
        }
        #[cfg(feature = "real-gpu")]
        {
            panic!(
                "WebGpuDft::default() called before install_global. \
                 This path is unreachable in normal flow — file a bug."
            );
        }
    }
}

impl TwoAdicSubgroupDft<Felt> for WebGpuDft {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<Felt>>;

    fn dft_batch(&self, mat: RowMajorMatrix<Felt>) -> Self::Evaluations {
        // CPU-delegating stub. Real-GPU swaps the body for a WGSL dispatch + readback.
        self.inner.cpu_fallback.dft_batch(mat)
    }

    fn idft_batch(&self, mat: RowMajorMatrix<Felt>) -> RowMajorMatrix<Felt> {
        self.inner.cpu_fallback.idft_batch(mat)
    }

    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<Felt>,
        added_bits: usize,
        shift: Felt,
    ) -> Self::Evaluations {
        self.inner.cpu_fallback.coset_lde_batch(mat, added_bits, shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use miden_crypto::stark::dft::Radix2DitParallel;
    use p3_matrix::Matrix;
    use rand::Rng;

    fn random_matrix(rng: &mut impl Rng, rows: usize, cols: usize) -> RowMajorMatrix<Felt> {
        let values: alloc::vec::Vec<Felt> = (0..rows * cols)
            .map(|_| Felt::new(rng.random::<u64>() % Felt::ORDER))
            .collect();
        RowMajorMatrix::new(values, cols)
    }

    #[test]
    fn dft_batch_matches_cpu_ground_truth() {
        let mut rng = rand::rng();
        let mat = random_matrix(&mut rng, 1 << 14, 8);
        let mat_clone = mat.clone();

        let gpu = WebGpuDft::new();
        let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();

        let gpu_out = gpu.dft_batch(mat).to_row_major_matrix();
        let cpu_out = cpu.dft_batch(mat_clone).to_row_major_matrix();

        assert_eq!(gpu_out.values, cpu_out.values);
    }

    #[test]
    fn idft_batch_matches_cpu_ground_truth() {
        let mut rng = rand::rng();
        let mat = random_matrix(&mut rng, 1 << 14, 8);
        let mat_clone = mat.clone();

        let gpu = WebGpuDft::new();
        let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();

        assert_eq!(gpu.idft_batch(mat).values, cpu.idft_batch(mat_clone).values);
    }

    #[test]
    fn coset_lde_batch_matches_cpu_ground_truth() {
        let mut rng = rand::rng();
        let mat = random_matrix(&mut rng, 1 << 12, 4);
        let mat_clone = mat.clone();
        let shift = Felt::new(7);

        let gpu = WebGpuDft::new();
        let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();

        let gpu_out = gpu.coset_lde_batch(mat, 3, shift).to_row_major_matrix();
        let cpu_out = cpu.coset_lde_batch(mat_clone, 3, shift).to_row_major_matrix();

        assert_eq!(gpu_out.values, cpu_out.values);
    }
}
