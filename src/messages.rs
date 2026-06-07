//! Messages exchanged between the UI components and the decomposition worker.

use serde::{Deserialize, Serialize};

/// Parameters of the Barnes-Hut t-SNE decomposition.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TsneParams {
    /// Perplexity of the conditional distribution, requires
    /// `n_samples - 1 >= 3 * perplexity`.
    pub perplexity: f32,
    /// Barnes-Hut accuracy trade off, strictly positive, lower is more exact.
    pub theta: f32,
    /// Number of fitting epochs.
    pub epochs: usize,
    /// Gradient descent learning rate.
    pub learning_rate: f32,
    /// Reduce the input to at most this many dimensions with PCA before
    /// fitting, the standard recipe uses 50. The reduction is skipped when
    /// the input dimensionality is already lower.
    pub pca_dims: usize,
    /// Send an embedding snapshot to the UI every this many epochs.
    pub snapshot_every: usize,
}

impl Default for TsneParams {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            theta: 0.5,
            epochs: 1000,
            learning_rate: 200.0,
            pca_dims: 50,
            snapshot_every: 5,
        }
    }
}

/// The decomposition to run.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DecompositionMethod {
    /// Barnes-Hut t-SNE through bhtsne, with PCA preprocessing.
    Tsne(TsneParams),
    /// Plain PCA projection onto the top two components.
    Pca,
}

/// Request sent from the UI to the decomposition worker.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WorkerRequest {
    /// Run a decomposition to two dimensions on the given matrix.
    Decompose {
        /// Row major matrix, `n_samples * n_features` long.
        data: Vec<f32>,
        /// Number of rows.
        n_samples: usize,
        /// Number of columns.
        n_features: usize,
        /// The decomposition to run, with its parameters.
        method: DecompositionMethod,
    },
}

/// Response sent from the decomposition worker back to the UI.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WorkerResponse {
    /// Intermediate embedding, sent every `snapshot_every` epochs.
    Snapshot {
        /// Zero based epoch index the snapshot was taken at.
        epoch: usize,
        /// Row major embedding, `n_samples * 2` long.
        embedding: Vec<f32>,
    },
    /// The decomposition finished.
    Done {
        /// Row major final embedding, `n_samples * 2` long.
        embedding: Vec<f32>,
        /// Fraction of the total variance captured by each output dimension,
        /// reported by the PCA method only.
        explained_variance_ratio: Option<Vec<f32>>,
    },
    /// The decomposition could not run.
    Error {
        /// Human readable reason.
        message: String,
    },
}
