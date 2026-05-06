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
use miden_crypto::stark::dft::TwoAdicSubgroupDft;
#[cfg(all(feature = "real-gpu", not(target_arch = "wasm32")))]
use p3_matrix::Matrix;
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
    ///
    /// On native targets, acquires a `wgpu::Device` from the default platform backend
    /// (Metal / Vulkan / D3D12) and stores a `WgpuContext` for use by the trait impls.
    /// On wasm32, currently a no-op shell (the CPU fallback path runs); a future substep
    /// will add a SharedArrayBuffer + Atomics.wait bridge to a dedicated GPU worker so
    /// that the sync trait method can drive async GPU readback to completion.
    #[cfg(feature = "real-gpu")]
    pub async fn new() -> Result<Self, crate::GpuInitError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let ctx = crate::gpu::WgpuContext::new().await?;
            let mut inner = WebGpuDftInner::default();
            inner.wgpu_ctx = Some(ctx);
            return Ok(Self { inner: Arc::new(inner) });
        }
        #[cfg(target_arch = "wasm32")]
        {
            // No SAB bridge yet — fall back to the CPU path. Note: the JS-facing
            // factory `TransactionProver::newGpuProver` ALSO checks for `navigator.gpu`
            // before constructing this; once the SAB bridge is wired here the failure
            // mode flips to "real GPU on Chrome, hard-error elsewhere" by design.
            return Ok(Self { inner: Arc::new(WebGpuDftInner::default()) });
        }
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
        #[cfg(all(feature = "real-gpu", not(target_arch = "wasm32")))]
        if let Some(ref ctx) = self.inner.wgpu_ctx {
            let rows = mat.height();
            let cols = mat.width();
            let bit_rev = ctx.gl_dft_batch(&mat.values, rows, cols);
            // Underlying buffer is bit-reversed-order; wrap so .to_row_major_matrix()
            // un-permutes to natural-order for consumers that want it.
            use p3_matrix::bitrev::BitReversibleMatrix;
            return RowMajorMatrix::new(bit_rev, cols).bit_reverse_rows();
        }
        // CPU fallback: stub default OR real-gpu without an active GPU context (wasm32
        // path until the SAB bridge is wired).
        self.inner.cpu_fallback.dft_batch(mat)
    }

    fn idft_batch(&self, mat: RowMajorMatrix<Felt>) -> RowMajorMatrix<Felt> {
        #[cfg(all(feature = "real-gpu", not(target_arch = "wasm32")))]
        if let Some(ref ctx) = self.inner.wgpu_ctx {
            let rows = mat.height();
            let cols = mat.width();
            let out = ctx.gl_idft_batch(&mat.values, rows, cols);
            return RowMajorMatrix::new(out, cols);
        }
        self.inner.cpu_fallback.idft_batch(mat)
    }

    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<Felt>,
        added_bits: usize,
        shift: Felt,
    ) -> Self::Evaluations {
        #[cfg(all(feature = "real-gpu", not(target_arch = "wasm32")))]
        if let Some(ref ctx) = self.inner.wgpu_ctx {
            let rows = mat.height();
            let cols = mat.width();
            let bit_rev = ctx.gl_coset_lde_batch(&mat.values, rows, cols, added_bits, shift);
            use p3_matrix::bitrev::BitReversibleMatrix;
            // bit_rev.len() == (rows << added_bits) * cols → RowMajorMatrix::new infers height.
            return RowMajorMatrix::new(bit_rev, cols).bit_reverse_rows();
        }
        self.inner.cpu_fallback.coset_lde_batch(mat, added_bits, shift)
    }
}

// Integration test for the real-GPU build: drives the GPU through the trait
// surface that the prover actually uses (`TwoAdicSubgroupDft<Felt>`), not via
// the lower-level `gl_*` functions. Skipped on machines without an adapter.
#[cfg(all(test, feature = "real-gpu", not(target_arch = "wasm32")))]
mod real_gpu_tests {
    use super::*;
    use miden_crypto::stark::dft::Radix2DitParallel;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::Rng;

    #[test]
    fn webgpu_dft_trait_impl_matches_cpu() {
        let dft = match pollster::block_on(WebGpuDft::new()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: WebGpuDft::new() failed ({e})");
                return;
            }
        };

        let mut rng = rand::rng();
        let rows = 64usize;
        let cols = 5usize;
        let data: Vec<Felt> = (0..rows * cols).map(|_| Felt::new(rng.random::<u64>() % Felt::ORDER)).collect();
        let mat = RowMajorMatrix::new(data.clone(), cols);

        // dft_batch via WebGpuDft trait method, materialise to natural order.
        let gpu_natural = dft.dft_batch(mat.clone()).to_row_major_matrix().values;
        let cpu_natural = Radix2DitParallel::<Felt>::default()
            .dft_batch(mat.clone())
            .to_row_major_matrix()
            .values;
        assert_eq!(gpu_natural.len(), cpu_natural.len());
        for i in 0..gpu_natural.len() {
            assert_eq!(
                gpu_natural[i].as_canonical_u64(),
                cpu_natural[i].as_canonical_u64(),
                "trait dft_batch mismatch at i={i}",
            );
        }

        // idft_batch
        let gpu_idft = dft.idft_batch(mat.clone()).values;
        let cpu_idft = Radix2DitParallel::<Felt>::default().idft_batch(mat.clone()).values;
        for i in 0..gpu_idft.len() {
            assert_eq!(
                gpu_idft[i].as_canonical_u64(),
                cpu_idft[i].as_canonical_u64(),
                "trait idft_batch mismatch at i={i}",
            );
        }

        // coset_lde_batch
        let added_bits = 2usize;
        let shift = Felt::new(7);
        let gpu_lde = dft
            .coset_lde_batch(mat.clone(), added_bits, shift)
            .to_row_major_matrix()
            .values;
        let cpu_lde = Radix2DitParallel::<Felt>::default()
            .coset_lde_batch(mat.clone(), added_bits, shift)
            .to_row_major_matrix()
            .values;
        for i in 0..gpu_lde.len() {
            assert_eq!(
                gpu_lde[i].as_canonical_u64(),
                cpu_lde[i].as_canonical_u64(),
                "trait coset_lde_batch mismatch at i={i}",
            );
        }
    }
}

// CPU-stub correctness tests. Only meaningful when the crate is built without
// `real-gpu` — under `real-gpu`, `WebGpuDft::new()` is async + may fail on
// machines without a GPU adapter, and the same byte-for-byte diff is exercised
// by the gpu module's roundtrip tests + real_gpu_tests above.
#[cfg(all(test, not(feature = "real-gpu")))]
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
