//! Principal component analysis on row major matrices.
//!
//! Used both as a preprocessing step before the more expensive decompositions
//! (the standard recipe reduces to 50 dimensions before t-SNE) and as a
//! selectable decomposition of its own when the target is 2 or 3 dimensions.

use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

/// Result of a PCA reduction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcaResult {
    /// Row major projected matrix, `n_samples * n_components` long.
    pub data: Vec<f32>,
    /// Number of retained components, `min(n_components, n_features)`.
    pub n_components: usize,
    /// Fraction of the total variance captured by each retained component,
    /// in decreasing order.
    pub explained_variance_ratio: Vec<f32>,
}

/// Reduces a row major matrix to its top principal components.
///
/// The data is centered, the covariance matrix eigendecomposed and the rows
/// projected onto the eigenvectors of the largest eigenvalues. Computations
/// run in f64 for numerical stability. Eigenvector signs are canonicalized
/// (the entry of largest magnitude is positive) so the projection is
/// deterministic.
///
/// This is the dense covariance approach: O(n_features^2) memory and
/// O(n_features^2 * n_samples) time, fine for the hundreds of input features
/// this crate targets.
///
/// # Arguments
///
/// * `data` - row major matrix, `n_samples * n_features` long.
/// * `n_samples` - number of rows.
/// * `n_features` - number of columns.
/// * `n_components` - requested output dimensionality, clamped to
///   `n_features`.
///
/// # Panics
///
/// Panics when `data.len() != n_samples * n_features` or when the matrix is
/// empty.
pub fn pca(data: &[f32], n_samples: usize, n_features: usize, n_components: usize) -> PcaResult {
    assert_eq!(
        data.len(),
        n_samples * n_features,
        "data length must equal n_samples * n_features"
    );
    assert!(
        n_samples > 0 && n_features > 0,
        "the input matrix must not be empty"
    );

    let n_components = n_components.min(n_features);

    // Row major input into a column major nalgebra matrix, in f64.
    let mut centered = DMatrix::<f64>::from_fn(n_samples, n_features, |i, j| {
        f64::from(data[i * n_features + j])
    });

    // Center each column.
    for j in 0..n_features {
        let mean = centered.column(j).mean();
        centered.column_mut(j).add_scalar_mut(-mean);
    }

    // Covariance matrix of the features.
    let denominator = if n_samples > 1 { n_samples - 1 } else { 1 } as f64;
    let covariance = (centered.transpose() * &centered) / denominator;

    let eigen = SymmetricEigen::new(covariance);

    // Eigenvalues are not guaranteed to be ordered: sort indices by
    // decreasing eigenvalue, clamping tiny negative values due to numerics.
    let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().map(|&l| l.max(0.0)).collect();
    let mut order: Vec<usize> = (0..n_features).collect();
    order.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

    // Projection matrix made of the top eigenvectors, sign canonicalized.
    let mut components = DMatrix::<f64>::zeros(n_features, n_components);
    for (target, &source) in order.iter().take(n_components).enumerate() {
        let column = eigen.eigenvectors.column(source);
        let largest = column
            .iter()
            .copied()
            .max_by(|a, b| a.abs().total_cmp(&b.abs()))
            .unwrap_or(1.0);
        let sign = if largest < 0.0 { -1.0 } else { 1.0 };
        components.column_mut(target).copy_from(&(column * sign));
    }

    let projected = centered * components;

    let total_variance: f64 = eigenvalues.iter().sum();
    let explained_variance_ratio: Vec<f32> = order
        .iter()
        .take(n_components)
        .map(|&source| {
            if total_variance > 0.0 {
                (eigenvalues[source] / total_variance) as f32
            } else {
                0.0
            }
        })
        .collect();

    // Column major nalgebra matrix back to a row major buffer.
    let mut data = Vec::with_capacity(n_samples * n_components);
    for i in 0..n_samples {
        for j in 0..n_components {
            data.push(projected[(i, j)] as f32);
        }
    }

    PcaResult {
        data,
        n_components,
        explained_variance_ratio,
    }
}
