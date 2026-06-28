//! Messages exchanged between the UI components and the decomposition worker.

use serde::{Deserialize, Serialize};

use crate::ingest::Dataset;

/// Which phase of a t-SNE run is currently executing, reported to the UI so the
/// status line and progress bar can name what the worker is doing.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsnePhase {
    /// Building the affinity graph: the vantage point tree neighbor search and
    /// the perplexity calibration, before any optimization epoch runs.
    FindingNeighbors,
    /// The early-exaggeration epochs, where the `P` distribution is inflated to
    /// pull clusters apart before the layout settles.
    EarlyExaggeration,
    /// The main optimization, after early exaggeration has stopped.
    Optimizing,
}

impl TsnePhase {
    /// Human readable label shown in the UI status line.
    pub fn label(self) -> &'static str {
        match self {
            TsnePhase::FindingNeighbors => "Finding neighbors",
            TsnePhase::EarlyExaggeration => "Early exaggeration",
            TsnePhase::Optimizing => "Optimizing",
        }
    }
}

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
    /// Multiplier applied to the `P` distribution during early exaggeration,
    /// which pulls clusters apart before the layout settles. bhtsne's default
    /// is 12, a value of 1 disables exaggeration. Ignored on a warm start,
    /// which keeps the seeded layout intact.
    #[serde(default = "default_early_exaggeration")]
    pub early_exaggeration: f32,
    /// Number of early-exaggeration epochs (bhtsne's `stop_lying_epoch`), also
    /// the boundary the status line uses to switch from early exaggeration to
    /// optimizing. 0 disables exaggeration. Ignored on a warm start.
    #[serde(default = "default_early_exaggeration_epochs")]
    pub early_exaggeration_epochs: usize,
    /// Send an embedding snapshot to the UI every this many epochs.
    pub snapshot_every: usize,
    /// Row major `n_samples * 2` embedding to warm start the fit from,
    /// continuing a previous run instead of random initialization. When set,
    /// early exaggeration is disabled so the seeded layout is not re-shocked.
    #[serde(default)]
    pub initial_embedding: Option<Vec<f32>>,
}

/// Default early-exaggeration factor, matching bhtsne.
fn default_early_exaggeration() -> f32 {
    12.0
}

/// Default early-exaggeration duration in epochs, matching bhtsne's
/// `stop_lying_epoch`.
fn default_early_exaggeration_epochs() -> usize {
    250
}

impl Default for TsneParams {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            theta: 0.5,
            epochs: 1000,
            learning_rate: None,
            pca_dims: 50,
            early_exaggeration: default_early_exaggeration(),
            early_exaggeration_epochs: default_early_exaggeration_epochs(),
            snapshot_every: 5,
            initial_embedding: None,
        }
    }
}

/// The decomposition to run. Only Barnes-Hut t-SNE (with PCA preprocessing) is
/// offered; kept as an enum so the worker request stays stable if more methods
/// return later.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DecompositionMethod {
    /// Barnes-Hut t-SNE through bhtsne, with PCA preprocessing.
    Tsne(TsneParams),
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
    /// The run entered a phase that produces no embedding yet, currently only
    /// the neighbor search. Lets the UI name the long affinity-build step
    /// instead of showing a bare spinner.
    Phase {
        /// The phase the worker just entered.
        phase: TsnePhase,
    },
    /// Intermediate embedding, sent every `snapshot_every` epochs.
    Snapshot {
        /// Zero based epoch index the snapshot was taken at.
        epoch: usize,
        /// Row major embedding, `n_samples * 2` long.
        embedding: Vec<f32>,
        /// Which optimization phase this epoch belongs to.
        phase: TsnePhase,
        /// Milliseconds elapsed since the run started, measured in the worker.
        elapsed_ms: f64,
        /// Size of rayon's global thread pool in the worker, 1 when the page is
        /// not cross-origin isolated and the pool could not start.
        threads: usize,
    },
    /// The decomposition finished.
    Done {
        /// Row major final embedding, `n_samples * 2` long.
        embedding: Vec<f32>,
        /// Final KL divergence of the t-SNE fit, a quality metric (lower is a
        /// better embedding).
        kl_divergence: Option<f32>,
        /// Milliseconds the whole run took, measured in the worker.
        elapsed_ms: f64,
        /// Size of rayon's global thread pool in the worker, 1 when the page is
        /// not cross-origin isolated and the pool could not start.
        threads: usize,
    },
    /// The decomposition could not run.
    Error {
        /// Human readable reason.
        message: String,
    },
}
