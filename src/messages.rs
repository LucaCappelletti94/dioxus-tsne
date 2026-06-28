//! Messages exchanged between the UI components and the decomposition worker.

use serde::{Deserialize, Serialize};

use crate::ingest::Dataset;

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
    /// Gradient descent learning rate, or `None` to let bhtsne pick the
    /// size-scaled auto value `max(n_samples / early_exaggeration / 4, 50)`.
    #[serde(default)]
    pub learning_rate: Option<f32>,
    /// Reduce the input to at most this many dimensions with PCA before
    /// fitting, the standard recipe uses 50. The reduction is skipped when
    /// the input dimensionality is already lower.
    pub pca_dims: usize,
    /// Send an embedding snapshot to the UI every this many epochs.
    pub snapshot_every: usize,
    /// Row major `n_samples * 2` embedding to warm start the fit from,
    /// continuing a previous run instead of random initialization. When set,
    /// early exaggeration is disabled so the seeded layout is not re-shocked.
    #[serde(default)]
    pub initial_embedding: Option<Vec<f32>>,
}

impl Default for TsneParams {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            theta: 0.5,
            epochs: 1000,
            learning_rate: None,
            pca_dims: 50,
            snapshot_every: 5,
            initial_embedding: None,
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
    /// Parse raw file bytes into a dataset off the main thread (keeping the UI
    /// responsive), report it back, and optionally run a decomposition on it.
    Load {
        /// File name, used to detect the format.
        name: String,
        /// The raw file contents.
        bytes: Vec<u8>,
        /// The decomposition to run on the parsed data, or `None` to only load.
        run: Option<DecompositionMethod>,
    },
    /// Run a decomposition to two dimensions on the given matrix (already parsed
    /// data, a warm start, or a resume after a pause).
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
    /// A `Load` request parsed successfully, the dataset is sent back so the UI
    /// can color and run it (and re-run on Continue).
    Loaded {
        /// The parsed dataset (matrix and label columns).
        dataset: Dataset,
    },
    /// A `Load` request failed to parse.
    LoadError {
        /// Human readable reason.
        message: String,
    },
    /// Intermediate embedding, sent every `snapshot_every` epochs.
    Snapshot {
        /// Zero based epoch index the snapshot was taken at.
        epoch: usize,
        /// Row major embedding, `n_samples * 2` long.
        embedding: Vec<f32>,
        /// Milliseconds elapsed since the run started, measured in the worker.
        elapsed_ms: f64,
        /// Size of rayon's global thread pool in the worker, 1 when the worker
        /// runs single threaded.
        threads: usize,
    },
    /// The decomposition finished.
    Done {
        /// Row major final embedding, `n_samples * 2` long.
        embedding: Vec<f32>,
        /// Fraction of the total variance captured by each output dimension,
        /// reported by the PCA method only.
        explained_variance_ratio: Option<Vec<f32>>,
        /// Milliseconds the whole run took, measured in the worker.
        elapsed_ms: f64,
        /// Size of rayon's global thread pool in the worker, 1 when the worker
        /// runs single threaded.
        threads: usize,
    },
    /// The decomposition could not run.
    Error {
        /// Human readable reason.
        message: String,
    },
}
