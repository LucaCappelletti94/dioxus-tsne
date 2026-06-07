//! The decomposition computations, kept free of any worker or browser types
//! so they can be tested natively.

use crate::messages::{DecompositionMethod, TsneParams};
use crate::pca::pca;

/// Output of a completed decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct DecomposeOutput {
    /// Row major embedding, `n_samples * 2` long.
    pub embedding: Vec<f32>,
    /// Explained variance ratios when the method was PCA.
    pub explained_variance_ratio: Option<Vec<f32>>,
}

/// Runs a decomposition to two dimensions, reporting intermediate embeddings
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
        DecompositionMethod::Pca => {
            let result = pca(data, n_samples, n_features, 2);
            if result.n_components < 2 {
                return Err(String::from(
                    "PCA to two dimensions needs at least two features",
                ));
            }
            Ok(DecomposeOutput {
                embedding: result.data,
                explained_variance_ratio: Some(result.explained_variance_ratio),
            })
        }
        DecompositionMethod::Tsne(params) => tsne(data, n_samples, n_features, params, snapshot),
    }
}

/// Runs Barnes-Hut t-SNE with PCA preprocessing.
fn tsne<C>(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    params: &TsneParams,
    mut snapshot: C,
) -> Result<DecomposeOutput, String>
where
    C: FnMut(usize, &[f32]) + Send + Sync,
{
    if params.theta <= 0.0 {
        return Err(String::from("theta must be strictly positive"));
    }
    // Mirrors the bhtsne perplexity check, which would otherwise panic.
    if (n_samples as f32 - 1.0) < 3.0 * params.perplexity {
        return Err(format!(
            "perplexity {} is too large for {n_samples} samples, it requires at least {} samples",
            params.perplexity,
            (3.0 * params.perplexity).ceil() as usize + 1
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

    let mut fit = bhtsne::tSNE::new(&samples);
    fit.embedding_dim(2)
        .perplexity(params.perplexity)
        .epochs(params.epochs)
        .learning_rate(params.learning_rate)
        .epoch_callback(move |epoch, embedding| {
            if epoch % snapshot_every == 0 {
                snapshot(epoch, embedding);
            }
        })
        .barnes_hut(params.theta, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });

    Ok(DecomposeOutput {
        embedding: fit.embedding(),
        explained_variance_ratio: None,
    })
}
