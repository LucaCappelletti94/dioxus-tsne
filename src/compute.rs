//! The decomposition computations, kept free of any worker or browser types
//! so they can be tested natively.

use crate::messages::{DecompositionMethod, TsneParams};
use crate::pca::pca;

/// Output of a completed decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct DecomposeOutput {
    /// Row major `n_samples * dimension` embedding.
    pub embedding: Vec<f32>,
    /// Final KL divergence of the t-SNE fit.
    pub kl_divergence: Option<f32>,
}

/// Identifies the affinity graph of a t-SNE run. The `P` distribution depends
/// only on the reduced input and the perplexity, so the epochs, the learning
/// rate and the warm-start seed are excluded. The worker holds a
/// single dataset per instance (a new dataset respawns it), so these fields
/// distinguish only the cases that can occur within one instance: a perplexity
/// or `pca_dims` change between a run and a continuation.
#[derive(PartialEq)]
struct AffinityKey {
    n_samples: usize,
    pca_dims: usize,
    perplexity_bits: u32,
    dimension: usize,
}

/// Caches the affinity graph of the last t-SNE run so a warm-start continuation
/// on the same data and perplexity reuses it, skipping the neighbor search and
/// perplexity calibration that dominate setup. Held by the worker across
/// messages and threaded through [`decompose_cached`].
#[derive(Default)]
pub(crate) struct TsneCache {
    entry: Option<(AffinityKey, bhtsne::SparseAffinities<f32>)>,
}

/// Runs a decomposition, reporting intermediate embeddings
/// through `snapshot`.
///
/// # Arguments
///
/// * `data` - row major matrix, `n_samples * n_features` long.
/// * `n_samples` - number of rows.
/// * `n_features` - number of columns.
/// * `method` - the decomposition to run, with its parameters.
/// * `snapshot` - called with the zero based epoch index and the current row
///   major embedding, every `snapshot_every` epochs of iterative methods.
///
/// # Errors
///
/// Returns a human readable message when the inputs are inconsistent or the
/// parameters do not fit the dataset.
pub fn decompose<C>(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    method: &DecompositionMethod,
    snapshot: C,
) -> Result<DecomposeOutput, String>
where
    C: FnMut(usize, &[f32]) + Send + Sync,
{
    decompose_cached(
        data,
        n_samples,
        n_features,
        method,
        &mut TsneCache::default(),
        snapshot,
    )
}

/// Like [`decompose`], but reuses the cached affinity graph for a warm-start
/// continuation on the same data and perplexity, and refreshes the cache after
/// every t-SNE run. The worker passes its persistent [`TsneCache`] so that
/// clicking Continue skips the neighbor search the previous run already paid.
pub(crate) fn decompose_cached<C>(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    method: &DecompositionMethod,
    cache: &mut TsneCache,
    snapshot: C,
) -> Result<DecomposeOutput, String>
where
    C: FnMut(usize, &[f32]) + Send + Sync,
{
    if n_samples == 0 || n_features == 0 {
        return Err(String::from("the dataset is empty"));
    }
    if data.len() != n_samples * n_features {
        return Err(format!(
            "inconsistent dataset: {} values for {n_samples} samples x {n_features} features",
            data.len()
        ));
    }

    match method {
        DecompositionMethod::Tsne(params) => {
            tsne(data, n_samples, n_features, params, cache, snapshot)
        }
    }
}

/// Runs Barnes-Hut t-SNE with PCA preprocessing, reusing and refreshing the
/// affinity cache. Dispatches to [`tsne_impl`] on the requested output
/// dimensionality.
fn tsne<C>(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    params: &TsneParams,
    cache: &mut TsneCache,
    snapshot: C,
) -> Result<DecomposeOutput, String>
where
    C: FnMut(usize, &[f32]) + Send + Sync,
{
    match params.dimension {
        2 => tsne_impl::<2, _>(data, n_samples, n_features, params, cache, snapshot),
        3 => tsne_impl::<3, _>(data, n_samples, n_features, params, cache, snapshot),
        4 => tsne_impl::<4, _>(data, n_samples, n_features, params, cache, snapshot),
        other => Err(format!(
            "unsupported output dimension {other}, expected 2, 3, or 4"
        )),
    }
}

/// Const-generic t-SNE implementation for output dimension `D`.
///
/// The embedding dimensionality is a const generic on `bhtsne::tSNE` (the final
/// `D`), so the caller must dispatch on the runtime `params.dimension` into this
/// function. `bhtsne`'s `barnes_hut` requires `Dim<D>: Morton<D>`, which is
/// implemented for `D in {2, 3, 4, 5, 6}`; this crate exposes 2, 3, and 4,
/// since the visualization is at most 3-D (a 4-D embedding is projected to
/// 3-D at draw time by rotating in the ZW plane and dropping W).
fn tsne_impl<const D: usize, C>(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    params: &TsneParams,
    cache: &mut TsneCache,
    mut snapshot: C,
) -> Result<DecomposeOutput, String>
where
    bhtsne::Dim<D>: bhtsne::Morton<D> + bhtsne::SpectralBlock,
    C: FnMut(usize, &[f32]) + Send + Sync,
{
    // Mirrors the bhtsne perplexity check, which would otherwise panic.
    if (n_samples as f32 - 1.0) < 3.0 * params.perplexity {
        return Err(format!(
            "perplexity {} is too large for {n_samples} samples, it requires at least {} samples",
            params.perplexity,
            (3.0 * params.perplexity).ceil() as usize + 1
        ));
    }
    // A warm start seeds the output, validated here so the worker reports an
    // error instead of tripping the bhtsne length assert.
    if let Some(init) = &params.initial_embedding
        && init.len() != n_samples * D
    {
        return Err(format!(
            "initial embedding has {} values, expected {} for {n_samples} samples in {}D",
            init.len(),
            n_samples * D,
            D
        ));
    }

    // The standard recipe reduces the input with PCA before fitting. Skipped
    // when the input is already at most pca_dims dimensional.
    let (reduced, n_features) = if params.pca_dims > 0 && n_features > params.pca_dims {
        let result = pca(data, n_samples, n_features, params.pca_dims);
        (Some(result.data), params.pca_dims)
    } else {
        (None, n_features)
    };
    let data = reduced.as_deref().unwrap_or(data);

    let samples: Vec<&[f32]> = data.chunks(n_features).collect();
    let snapshot_every = params.snapshot_every.max(1);

    let mut fit: bhtsne::tSNE<'_, f32, &[f32], D> = bhtsne::tSNE::new(&samples);
    fit.perplexity(params.perplexity)
        .epochs(params.epochs)
        .epoch_callback(move |epoch, embedding| {
            if epoch % snapshot_every == 0 {
                snapshot(epoch, embedding);
            }
        });
    // Leaving the learning rate unset lets bhtsne resolve its size-scaled auto
    // default, max(n_samples / early_exaggeration / 4, 50).
    if let Some(learning_rate) = params.learning_rate {
        fit.learning_rate(learning_rate);
    }
    // Apply the early-exaggeration controls. Setting them explicitly (rather
    // than leaning on bhtsne's defaults) also lets the worker label the phase
    // from the epoch index alone. A warm start overrides the duration to 0
    // below, since exaggeration would re-shock the seeded layout.
    fit.early_exaggeration(params.early_exaggeration)
        .stop_lying_epoch(params.early_exaggeration_epochs);
    if let Some(init) = &params.initial_embedding {
        // Warm start: seed the embedding and continue optimizing instead of
        // restarting. Early exaggeration would pull the seeded layout back
        // together, so it is disabled, and the momentum jumps straight to the
        // final value a converged run was already using.
        fit.initial_embedding(init.clone())
            .stop_lying_epoch(0)
            .momentum_switch_epoch(0);
    } else {
        // Fresh run: seed the fit from the top eigenvectors of the
        // affinity graph's normalized Laplacian (Kobak & Linderman 2021).
        // Superior to PCA seeding whenever it applies, and unlike PCA it
        // needs no feature matrix: it works off exactly the graph bhtsne
        // is about to fit, so it stays available for warm-start /
        // precomputed-neighbor workflows too. bhtsne's builder runs the
        // solver internally after the affinities are built, so we only
        // flip the flag on here; no per-axis whitening is needed either,
        // since the eigenvectors already come back orthonormal, and the
        // usual Kobak & Berens 1e-4 magnitude scaling is applied by the
        // solver itself before it hands the seed to the optimizer. Early
        // exaggeration stays on, as recommended.
        fit.spectral_init();
    }

    // Reuse the cached affinity graph on a warm-start continuation of the same
    // data and perplexity, so bhtsne skips the neighbor search and perplexity
    // calibration. A fresh run (no seed) always rebuilds it below.
    let key = AffinityKey {
        n_samples,
        pca_dims: params.pca_dims,
        perplexity_bits: params.perplexity.to_bits(),
        dimension: D,
    };
    let reuse =
        params.initial_embedding.is_some() && cache.entry.as_ref().is_some_and(|(k, _)| *k == key);
    if reuse && let Some((_, affinities)) = cache.entry.take() {
        fit.with_affinities(affinities);
    }

    // Barnes-Hut approximation for the repulsive forces. (FIt-SNE was tried but
    // its fixed-cost interpolation grid made every epoch slow in the browser,
    // regardless of dataset size.) Theta is hardcoded for now; the setting may
    // come back. A reused affinity graph short-circuits the neighbor search, so
    // this only rebuilds the vantage point tree when the graph is not cached.
    const BARNES_HUT_THETA: f32 = 0.5;
    fit.barnes_hut(BARNES_HUT_THETA, |a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    });

    // Refresh the cache with the affinity graph this run computed (or reused),
    // so the next continuation reuses it.
    if let Some(affinities) = fit.affinities() {
        cache.entry = Some((key, affinities));
    }

    let kl_divergence = fit.kl_divergence();
    Ok(DecomposeOutput {
        embedding: fit.embedding(),
        kl_divergence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::TsneParams;

    /// Two well separated clusters via a deterministic LCG, the first half in
    /// one cluster and the second half in the other.
    fn clustered(n_per_cluster: usize, dim: usize) -> Vec<f32> {
        let mut state = 42u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) - 0.5
        };
        let mut data = Vec::with_capacity(2 * n_per_cluster * dim);
        for cluster in 0..2 {
            let centre = if cluster == 0 { 0.0 } else { 10.0 };
            for _ in 0..n_per_cluster {
                for _ in 0..dim {
                    data.push(centre + next());
                }
            }
        }
        data
    }

    /// Fraction of points whose nearest embedding neighbor shares their cluster.
    fn same_cluster_fraction(embedding: &[f32], n: usize) -> f64 {
        let mut same = 0;
        for i in 0..n {
            let mut best = f32::MAX;
            let mut best_j = usize::MAX;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = embedding[2 * i] - embedding[2 * j];
                let dy = embedding[2 * i + 1] - embedding[2 * j + 1];
                let d = dx * dx + dy * dy;
                if d < best {
                    best = d;
                    best_j = j;
                }
            }
            if (i < n / 2) == (best_j < n / 2) {
                same += 1;
            }
        }
        same as f64 / n as f64
    }

    #[test]
    fn fresh_run_populates_and_rekeys_the_cache() {
        const N: usize = 200;
        const DIM: usize = 8;
        let data = clustered(N / 2, DIM);
        let mut cache = TsneCache::default();

        let params = TsneParams {
            epochs: 60,
            pca_dims: 4,
            perplexity: 20.0,
            ..TsneParams::default()
        };
        decompose_cached(
            &data,
            N,
            DIM,
            &DecompositionMethod::Tsne(params),
            &mut cache,
            |_, _| {},
        )
        .unwrap();

        let (key, _) = cache.entry.as_ref().expect("a t-SNE run caches affinities");
        assert_eq!(key.n_samples, N);
        assert_eq!(key.pca_dims, 4);
        assert_eq!(key.perplexity_bits, 20.0f32.to_bits());
        assert_eq!(key.dimension, 2);

        // A run at a different perplexity rebuilds and rekeys the cache, so a
        // later continuation does not reuse a stale graph.
        let params = TsneParams {
            epochs: 60,
            pca_dims: 4,
            perplexity: 25.0,
            ..TsneParams::default()
        };
        decompose_cached(
            &data,
            N,
            DIM,
            &DecompositionMethod::Tsne(params),
            &mut cache,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(
            cache.entry.as_ref().unwrap().0.perplexity_bits,
            25.0f32.to_bits()
        );
    }

    #[test]
    fn warm_start_reusing_cache_continues_from_seed() {
        const N: usize = 200;
        const DIM: usize = 8;
        let data = clustered(N / 2, DIM);
        let mut cache = TsneCache::default();
        let base = || TsneParams {
            pca_dims: 4,
            perplexity: 20.0,
            ..TsneParams::default()
        };

        // Fresh run to converge a layout and populate the cache.
        let seed = decompose_cached(
            &data,
            N,
            DIM,
            &DecompositionMethod::Tsne(TsneParams {
                epochs: 250,
                ..base()
            }),
            &mut cache,
            |_, _| {},
        )
        .unwrap()
        .embedding;
        assert!(cache.entry.is_some(), "the fresh run must cache affinities");

        // Continue, reusing the cached affinities (same data and perplexity).
        let first = std::sync::Mutex::new(None::<Vec<f32>>);
        let output = decompose_cached(
            &data,
            N,
            DIM,
            &DecompositionMethod::Tsne(TsneParams {
                epochs: 50,
                snapshot_every: 1,
                initial_embedding: Some(seed.clone()),
                ..base()
            }),
            &mut cache,
            |_, embedding| {
                let mut guard = first.lock().unwrap();
                if guard.is_none() {
                    *guard = Some(embedding.to_vec());
                }
            },
        )
        .unwrap();

        let first = first.into_inner().unwrap().expect("a snapshot streamed");
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in seed.as_chunks::<2>().0 {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        let diagonal = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
        let mean_shift = seed
            .as_chunks::<2>()
            .0
            .iter()
            .zip(first.as_chunks::<2>().0.iter())
            .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
            .sum::<f32>()
            / N as f32;
        assert!(
            mean_shift < 0.05 * diagonal,
            "reused warm start jumped: mean shift {mean_shift} vs diagonal {diagonal}"
        );
        assert!(
            same_cluster_fraction(&output.embedding, N) > 0.95,
            "reused warm start degraded separation"
        );
    }
}
