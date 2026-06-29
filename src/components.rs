//! The reusable Dioxus components.
//!
//! [`Decomposition`] is the entry point: a method neutral (t-SNE, PCA, more to
//! come) visualizer configured through a fluent builder. Everything lives on a
//! single decomposer panel, the plot with a thin toolbar of controls on top, an
//! empty state that doubles as the drag and drop loader (with optional example
//! dataset buttons), and an automatic color legend.

use std::cell::RefCell;
use std::rc::Rc;

use crate::color::{ColorScale, Coloring, Marker, colorize};
use crate::ingest::{Dataset, LabelColumn};
use crate::messages::{DecompositionMethod, TsneParams, TsnePhase, WorkerRequest, WorkerResponse};
use crate::plot::ScatterPlot;
use crate::worker::DecompositionWorker;
use dioxus::html::HasFileData;
use dioxus::prelude::*;
use dioxus_free_icons::Icon;
use dioxus_free_icons::icons::fa_brands_icons::FaGithub;
use dioxus_free_icons::icons::fa_solid_icons::{
    FaBan, FaBullseye, FaCalculator, FaCircleNodes, FaCircleQuestion, FaCompress, FaDownload,
    FaExpand, FaFileArrowUp, FaFire, FaGaugeHigh, FaHashtag, FaHeart, FaImage, FaInfinity,
    FaPalette, FaPause, FaPlay, FaRepeat, FaRotateLeft, FaShareNodes, FaShirt, FaSliders,
    FaSpinner, FaTag, FaTrashCan, FaTriangleExclamation, FaVideo, FaXmark,
};
use gloo_worker::{Spawnable, WorkerBridge};
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

/// Longest legend rendered before truncation.
const MAX_LEGEND_ENTRIES: usize = 20;

/// Epoch budget for an "infinite" run: large enough to feel endless, but still
/// finite (a fit cannot be interrupted mid-run, so a truly unbounded value could
/// never stop).
const INFINITE_EPOCHS: usize = 500_000;

/// "thread" or "threads", to read naturally next to a pool size in the status.
fn thread_word(threads: usize) -> &'static str {
    if threads == 1 { "thread" } else { "threads" }
}

/// Plain-language explanation of a fitting phase, shown when the status line's
/// phase label is hovered.
fn phase_help(phase: TsnePhase) -> &'static str {
    match phase {
        TsnePhase::FindingNeighbors => {
            "Building each point's nearest-neighbor graph in the original high-dimensional space, before the layout starts."
        }
        TsnePhase::EarlyExaggeration => {
            "An opening phase that inflates the pull between neighbors so clusters separate before the layout is fine-tuned."
        }
        TsnePhase::Optimizing => {
            "Refining the 2-D layout by gradient descent, balancing attraction to neighbors against repulsion from everything else."
        }
    }
}

/// Current viewport size in CSS pixels, the logical canvas size for the
/// full-bleed plot. Falls back to a sensible default when the window is
/// unavailable.
fn window_size() -> (u32, u32) {
    web_sys::window()
        .map(|window| {
            let width = window
                .inner_width()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(1024.0);
            let height = window
                .inner_height()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(768.0);
            (width.max(1.0) as u32, height.max(1.0) as u32)
        })
        .unwrap_or((1024, 768))
}

/// The role a parsed column plays once the user has assigned it: a t-SNE input
/// feature, a label only used to color points, or dropped entirely.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColumnRole {
    Feature,
    Label,
    Ignore,
}

/// Live numbers behind the status line, kept structured (not only as the
/// formatted string) so each part can show its own explanation on hover.
#[derive(Clone, Copy, PartialEq)]
enum RunInfo {
    /// A snapshot mid-run.
    Running {
        epoch: usize,
        elapsed_s: f64,
        threads: usize,
    },
    /// The finished fit.
    Done {
        elapsed_s: f64,
        threads: usize,
        kl: Option<f32>,
    },
}

/// Where a column's values live in the parsed [`Dataset`]: either a column of
/// the numeric feature matrix, or one of the text label columns.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColumnSource {
    /// Index into the row-major feature matrix (`feature_names[i]`).
    Feature(usize),
    /// Index into `label_columns`.
    Label(usize),
}

/// A parsed column together with the role the user has assigned it. Numeric
/// columns default to `Feature`, text columns to `Label`.
#[derive(Clone, PartialEq)]
struct ColumnEntry {
    name: String,
    source: ColumnSource,
    role: ColumnRole,
    /// Numeric columns can be features; text columns can only label or be
    /// ignored.
    numeric: bool,
}

/// Builds the default column roles for a freshly parsed dataset: every numeric
/// column a feature, every text column a label, in matrix-then-label order.
fn default_columns(dataset: &Dataset) -> Vec<ColumnEntry> {
    let mut entries = Vec::with_capacity(dataset.feature_names.len() + dataset.label_columns.len());
    for (index, name) in dataset.feature_names.iter().enumerate() {
        entries.push(ColumnEntry {
            name: name.clone(),
            source: ColumnSource::Feature(index),
            role: ColumnRole::Feature,
            numeric: true,
        });
    }
    for (index, column) in dataset.label_columns.iter().enumerate() {
        entries.push(ColumnEntry {
            name: column.name.clone(),
            source: ColumnSource::Label(index),
            role: ColumnRole::Label,
            numeric: false,
        });
    }
    entries
}

/// Assembles the t-SNE input matrix from the columns the user marked as
/// features. Returns `None` when no feature column is selected.
fn build_feature_matrix(dataset: &Dataset, columns: &[ColumnEntry]) -> Option<(Vec<f32>, usize)> {
    let feature_indices: Vec<usize> = columns
        .iter()
        .filter_map(|entry| match (entry.role, entry.source) {
            (ColumnRole::Feature, ColumnSource::Feature(index)) => Some(index),
            _ => None,
        })
        .collect();
    if feature_indices.is_empty() {
        return None;
    }
    let n_features = feature_indices.len();
    let mut data = Vec::with_capacity(dataset.n_samples * n_features);
    for row in 0..dataset.n_samples {
        for &index in &feature_indices {
            data.push(dataset.data[row * dataset.n_features + index]);
        }
    }
    Some((data, n_features))
}

/// The label columns to color by given the current roles: text columns kept as
/// labels, plus any numeric column the user reassigned to a label (its values
/// stringified so the continuous color scale can parse them into a heatmap).
fn effective_label_columns(dataset: &Dataset, columns: &[ColumnEntry]) -> Vec<LabelColumn> {
    let mut labels = Vec::new();
    for entry in columns {
        if entry.role != ColumnRole::Label {
            continue;
        }
        match entry.source {
            ColumnSource::Label(index) => {
                if let Some(column) = dataset.label_columns.get(index) {
                    labels.push(column.clone());
                }
            }
            ColumnSource::Feature(index) => {
                let values = (0..dataset.n_samples)
                    .map(|row| dataset.data[row * dataset.n_features + index].to_string())
                    .collect();
                labels.push(LabelColumn {
                    name: entry.name.clone(),
                    values,
                });
            }
        }
    }
    labels
}

/// The size-scaled learning rate bhtsne resolves when none is set,
/// `max(n_samples / early_exaggeration / 4, 50)` with the standard 12x
/// exaggeration. Used only to preview the auto value in the control's
/// placeholder, the actual value is resolved inside the library.
fn auto_learning_rate(n_samples: usize) -> f32 {
    (n_samples as f32 / 12.0 / 4.0).max(50.0)
}

// Plain language explanations shown as hover tooltips (`title`) and to screen
// readers (`aria-label`), so a newcomer can learn what each control does just by
// hovering it.
const HELP_PERPLEXITY: &str = "Roughly how many close neighbors each point pays attention to. Smaller makes tight local clumps, larger spreads things out. 5 to 50 is typical.";
const HELP_THETA: &str = "Trades t-SNE speed against accuracy. Lower is more accurate but slower, higher is faster but rougher. 0.5 is a sensible default.";
const HELP_EPOCHS: &str = "How many refinement steps to run. More steps polish the layout further but take longer. 1000 is a good default, a few hundred is often enough.";
const HELP_INFINITE: &str = "Keep iterating with no fixed limit until you press pause, so you can watch the layout evolve indefinitely. The epoch count above is ignored.";
const HELP_LEARNING_RATE: &str = "How big each refinement step is. Too small and it gets stuck, too large and it looks chaotic. Leave it empty for 'auto', a value scaled to the dataset size (shown greyed in the box). 200 is a common manual value.";
const HELP_PCA_DIMS: &str = "Before t-SNE the data is first squeezed to this many dimensions with PCA to speed things up and cut noise. 30 by default, a typical range is 30 to 50, 2 or more.";
const HELP_COLUMNS: &str = "What each column does. Feature: fed into t-SNE. Label: kept out of t-SNE and offered as a color (numeric labels become a heatmap). Ignore: dropped entirely.";
const HELP_EARLY_EXAGGERATION: &str = "How hard clusters are pushed apart early in the run, before the layout settles. 12 is the standard value, 1 turns it off. Ignored when continuing a run.";
const HELP_EXAGGERATION_EPOCHS: &str = "How many epochs that early push lasts. 250 is typical. 0 turns early exaggeration off. Ignored when continuing a run.";
const HELP_RUN: &str =
    "Start the layout from scratch and watch the scatter plot animate as it improves.";
const HELP_COLOR_BY: &str =
    "Color the points by one of the dataset's label columns to see where each group lands.";
const HELP_SETTINGS: &str = "Advanced t-SNE settings.";
const HELP_RECORD: &str = "Record the scatter plot animation as a WebM video. Toggle while a run is active to capture the embedding evolving.";
const HELP_CLEAR: &str = "Clear the dataset and result, returning to the start.";
const HELP_LEGEND_EXPORT: &str = "Bake the color legend into the downloaded snapshot. It gets its own strip and the points are framed beside it, so the legend never covers them.";

// Status-line tooltips: short explanations so a newcomer can learn the terms by
// hovering them while the fit runs.
const HELP_STATUS_EPOCH: &str = "An epoch is one full optimization pass over every point. t-SNE nudges the layout a little on each pass, so more epochs means a more settled map.";
const HELP_STATUS_DONE: &str = "The fit has finished: it used up the epoch budget (or you paused it). The scatter plot below is the result.";
const HELP_STATUS_THREADS: &str = "How many CPU cores the fit is using at once. It runs on a SharedArrayBuffer thread pool right in your browser, with no server involved.";
const HELP_STATUS_ELAPSED: &str = "Wall-clock time spent fitting so far.";
const HELP_STATUS_KL: &str = "Kullback-Leibler divergence: how faithfully the 2-D map preserves the original high-dimensional neighborhoods. It is the quantity t-SNE minimizes, so lower is a better fit.";
const HELP_STATUS_LOADING: &str = "Reading and parsing your file off the main thread.";

/// File input `accept` attribute.
const DATA_ACCEPT: &str = ".csv,.tsv,.parquet,.arrow,.feather,.npy";

/// Default URL the worker loader module is served from, the path the reference
/// `build.rs` writes it to (`public/dioxus-decompositions/loader.js`, served at
/// the site root). [`Decomposition::new`] uses this unless overridden with
/// [`Decomposition::worker_url`].
pub const DEFAULT_WORKER_URL: &str = "/dioxus-decompositions/loader.js";

/// Spawns a worker bridge whose callback writes replies into `status` and
/// `embedding`. Kept as a free function so a fresh bridge can be spawned on
/// demand (the pause feature orphans the running worker and installs a new
/// one). The `worker_url` points at the loader module, see the crate docs.
#[allow(clippy::too_many_arguments)]
fn spawn_bridge(
    worker_url: &str,
    status: Signal<String>,
    phase: Signal<Option<TsnePhase>>,
    embedding: Signal<Option<Vec<f32>>>,
    dataset: Signal<Option<Dataset>>,
    ingest_error: Signal<Option<String>>,
    color_source: Signal<String>,
    run_info: Signal<Option<RunInfo>>,
) -> WorkerBridge<DecompositionWorker> {
    DecompositionWorker::spawner()
        // The URL points at a loader module that initializes the wasm-bindgen
        // output itself, see the crate documentation.
        .with_loader(true)
        .callback(move |response| {
            // Signals are Copy, the rebindings make the closure a plain Fn.
            let mut status = status;
            let mut phase = phase;
            let mut embedding = embedding;
            let mut dataset = dataset;
            let mut ingest_error = ingest_error;
            let mut color_source = color_source;
            let mut run_info = run_info;
            match response {
                WorkerResponse::Loaded { dataset: parsed } => {
                    // Parsing happened off the main thread; adopt the result,
                    // coloring by the first label column.
                    ingest_error.set(None);
                    color_source.set(
                        parsed
                            .label_columns
                            .first()
                            .map(|c| format!("column:{}", c.name))
                            .unwrap_or_else(|| String::from("none")),
                    );
                    if status.read().as_str() == "loading" {
                        status.set(String::from("idle"));
                    }
                    dataset.set(Some(parsed));
                }
                WorkerResponse::LoadError { message } => {
                    dataset.set(None);
                    status.set(String::from("idle"));
                    ingest_error.set(Some(message));
                }
                WorkerResponse::Phase { phase: current } => {
                    phase.set(Some(current));
                }
                WorkerResponse::Snapshot {
                    epoch,
                    embedding: snapshot,
                    phase: current,
                    elapsed_ms,
                    threads,
                } => {
                    phase.set(Some(current));
                    status.set(format!(
                        "epoch {epoch} ({:.1}s, {threads} {})",
                        elapsed_ms / 1000.0,
                        thread_word(threads)
                    ));
                    run_info.set(Some(RunInfo::Running {
                        epoch,
                        elapsed_s: elapsed_ms / 1000.0,
                        threads,
                    }));
                    embedding.set(Some(snapshot));
                }
                WorkerResponse::Done {
                    embedding: done,
                    kl_divergence,
                    elapsed_ms,
                    threads,
                } => {
                    let seconds = elapsed_ms / 1000.0;
                    let pool = format!("{threads} {}", thread_word(threads));
                    status.set(match kl_divergence {
                        Some(kl) => format!("done in {seconds:.1}s ({pool}), KL {kl:.4}"),
                        None => format!("done in {seconds:.1}s ({pool})"),
                    });
                    run_info.set(Some(RunInfo::Done {
                        elapsed_s: seconds,
                        threads,
                        kl: kl_divergence,
                    }));
                    embedding.set(Some(done));
                }
                WorkerResponse::Error { message } => {
                    status.set(format!("error: {message}"));
                    run_info.set(None);
                }
            }
        })
        .spawn(worker_url)
}

/// An optional glyph shown on an example button, hinting at the dataset's
/// domain. Kept as a small enum (rather than a concrete icon type) so
/// [`ExampleDataset`] stays `Clone`/`PartialEq` for the props system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleIcon {
    /// Numbers / digits (e.g. MNIST).
    Numbers,
    /// Apparel / clothing (e.g. Fashion-MNIST).
    Apparel,
    /// A graph or citation network (e.g. Cora).
    Network,
}

/// A bundled example dataset offered through a one click load button.
#[derive(Debug, Clone, PartialEq)]
pub struct ExampleDataset {
    /// Human readable name shown on the button.
    pub name: String,
    /// URL the file is served from, any supported format.
    pub url: String,
    /// Optional glyph shown before the name.
    pub icon: Option<ExampleIcon>,
    /// Optional plain-language description shown on hover (tooltip and aria
    /// label). Falls back to a generic "load" message when absent.
    pub description: Option<String>,
}

/// Configuration of the drag and drop loader shown over the empty plot, built
/// fluently.
///
/// ```
/// use dioxus_tsne::DropZone;
///
/// let zone = DropZone::new()
///     .accept(["csv", "parquet"])
///     .prompt("Drop a dataset to begin");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DropZone {
    /// Lowercase extensions (without the dot) accepted on drop. Empty accepts
    /// anything and lets the format sniffing decide.
    accept: Vec<String>,
    /// Text inviting a drop, shown over the empty plot.
    prompt: String,
}

impl Default for DropZone {
    fn default() -> Self {
        Self {
            accept: ["csv", "tsv", "parquet", "arrow", "feather", "npy"]
                .into_iter()
                .map(String::from)
                .collect(),
            prompt: String::from(
                "Drop a CSV, TSV, Parquet, Arrow or NumPy (.npy) file here, or click to browse",
            ),
        }
    }
}

impl DropZone {
    /// A loader accepting CSV, TSV and Parquet with a default prompt.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the accepted file extensions (leading dots and case are
    /// ignored). An empty list accepts anything.
    pub fn accept<I, S>(mut self, extensions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.accept = extensions
            .into_iter()
            .map(|ext| ext.into().trim_start_matches('.').to_ascii_lowercase())
            .collect();
        self
    }

    /// Sets the invitation text shown on the empty plot.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// True when `extension` (without a dot) is accepted.
    fn allows(&self, extension: &str) -> bool {
        self.accept.is_empty() || self.accept.iter().any(|a| a == extension)
    }
}

/// A method neutral decomposition visualizer, configured with a fluent builder.
///
/// `Decomposition::new().render()` is a bare scatter plot driven by the worker.
/// Chain the builder methods to opt into the loader, example buttons, controls
/// and draggable points, then call [`render`](Self::render) to get the
/// [`Element`]. Loaded data is colored automatically by its first label column,
/// with a legend over the plot.
///
/// ```ignore
/// use dioxus_tsne::{Decomposition, ExampleDataset};
///
/// Decomposition::new()
///     .drop_zone()
///     .examples(vec![ExampleDataset { name: "MNIST".into(), url: "/mnist.parquet".into(), icon: None, description: None }])
///     .controls()
///     .draggable_points()
///     .render()
/// # ;
/// ```
///
/// The worker is loaded from [`DEFAULT_WORKER_URL`] by default, the path the
/// reference `build.rs` writes the loader to when the app is served at the site
/// root. Override it with [`worker_url`](Self::worker_url) for a custom output
/// path or a site served under a subpath.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    worker_url: String,
    dataset: Option<Dataset>,
    drop_zone: Option<DropZone>,
    examples: Vec<ExampleDataset>,
    controls: bool,
    draggable: bool,
    styled: bool,
    pixel_ratio: Option<f64>,
    logo: Option<String>,
    repo_url: Option<String>,
    support_url: Option<String>,
}

impl Default for Decomposition {
    fn default() -> Self {
        Self::new()
    }
}

impl Decomposition {
    /// A bare plot driven by the worker at [`DEFAULT_WORKER_URL`], every extra
    /// feature off, the default stylesheet on.
    pub fn new() -> Self {
        Self {
            worker_url: String::from(DEFAULT_WORKER_URL),
            dataset: None,
            drop_zone: None,
            examples: Vec::new(),
            controls: false,
            draggable: false,
            styled: true,
            pixel_ratio: None,
            logo: None,
            repo_url: None,
            support_url: None,
        }
    }

    /// Sets the brand logo shown in the top-left corner (an image URL, e.g. a
    /// bundled asset). Without it the top bar shows the plain title text.
    pub fn logo(mut self, url: impl Into<String>) -> Self {
        self.logo = Some(url.into());
        self
    }

    /// Adds a GitHub button to the top bar linking to the project `url` (opened
    /// in a new tab). Omitted when unset.
    pub fn repository(mut self, url: impl Into<String>) -> Self {
        self.repo_url = Some(url.into());
        self
    }

    /// Adds a heart button to the top bar linking to a support or sponsor page
    /// at `url` (opened in a new tab). Omitted when unset.
    pub fn support(mut self, url: impl Into<String>) -> Self {
        self.support_url = Some(url.into());
        self
    }

    /// Overrides the plot's backing buffer resolution multiplier (over its
    /// logical size) for crisp rendering. Defaults to the device pixel ratio,
    /// raise it to supersample further (e.g. `2.0` on a standard-DPI display).
    /// Clamped to `[1, 4]`.
    pub fn pixel_ratio(mut self, ratio: f64) -> Self {
        self.pixel_ratio = Some(ratio);
        self
    }

    /// Overrides where the worker loader module is served from (default
    /// [`DEFAULT_WORKER_URL`]). Set this if your `build.rs` writes the worker
    /// elsewhere or your site is served under a subpath.
    pub fn worker_url(mut self, url: impl Into<String>) -> Self {
        self.worker_url = url.into();
        self
    }

    /// Preloads an in-memory dataset to decompose, a row major `n_samples *
    /// n_features` matrix, so the component starts ready to run without the
    /// user loading a file. Attach columns to color by with
    /// [`labels`](Self::labels). Replaces any previously set dataset.
    pub fn dataset(mut self, data: Vec<f32>, n_samples: usize, n_features: usize) -> Self {
        self.dataset = Some(Dataset {
            data,
            n_samples,
            n_features,
            feature_names: Vec::new(),
            label_columns: Vec::new(),
        });
        self
    }

    /// Adds a column of per point labels to color the preloaded
    /// [`dataset`](Self::dataset) by (call after it). The labels are colorized
    /// like any label column: a palette and markers per class with a legend, or
    /// the viridis scale for numeric values. The first column added is the one
    /// colored by default. No op if no dataset has been set.
    pub fn labels(mut self, name: impl Into<String>, values: Vec<String>) -> Self {
        if let Some(dataset) = self.dataset.as_mut() {
            dataset.label_columns.push(LabelColumn {
                name: name.into(),
                values,
            });
        }
        self
    }

    /// The loader, controls and draggable points all switched on (no example
    /// datasets, add them with [`examples`](Self::examples)).
    pub fn full() -> Self {
        Self::new().drop_zone().controls().draggable_points()
    }

    /// Makes the empty plot a drag and drop (and click to browse) loader with
    /// the default [`DropZone`].
    pub fn drop_zone(mut self) -> Self {
        self.drop_zone = Some(DropZone::default());
        self
    }

    /// Makes the empty plot a loader configured by `zone`.
    pub fn drop_zone_with(mut self, zone: DropZone) -> Self {
        self.drop_zone = Some(zone);
        self
    }

    /// Adds one click load buttons for the given bundled datasets, shown on the
    /// empty plot beneath the drop prompt.
    pub fn examples(mut self, examples: Vec<ExampleDataset>) -> Self {
        self.examples = examples;
        self
    }

    /// Shows the toolbar on the plot: the method selector, run and continue
    /// buttons, the color by selector, the advanced settings and the status.
    pub fn controls(mut self) -> Self {
        self.controls = true;
        self
    }

    /// Lets the user grab points: dragging during a run pauses it and resuming
    /// continues from the edited layout.
    pub fn draggable_points(mut self) -> Self {
        self.draggable = true;
        self
    }

    /// Whether to inject the default stylesheet (also exported as
    /// [`crate::DEFAULT_STYLE`]). On by default, disable it to bring your own
    /// rules for the `decompositions-*` class names.
    pub fn styled(mut self, styled: bool) -> Self {
        self.styled = styled;
        self
    }

    /// Renders the configured visualizer.
    pub fn render(self) -> Element {
        rsx! {
            DecompositionView { config: self }
        }
    }
}

/// The component behind [`Decomposition::render`]. Holds all the state and
/// renders only the pieces enabled in the config.
#[component]
fn DecompositionView(config: Decomposition) -> Element {
    let Decomposition {
        worker_url,
        dataset: preset_dataset,
        drop_zone,
        examples,
        controls,
        draggable,
        styled,
        pixel_ratio,
        logo,
        repo_url,
        support_url,
    } = config;

    // A preloaded dataset starts colored by its first label column.
    let initial_color_source = preset_dataset
        .as_ref()
        .and_then(|d| d.label_columns.first())
        .map(|c| format!("column:{}", c.name))
        .unwrap_or_else(|| String::from("none"));

    let dataset = use_signal(move || preset_dataset);
    let ingest_error = use_signal(|| None::<String>);
    let status = use_signal(|| String::from("idle"));
    let phase = use_signal(|| None::<TsnePhase>);
    let embedding = use_signal(|| None::<Vec<f32>>);
    // Structured numbers for the status line, set alongside `status` from the
    // worker responses (see `spawn_bridge`).
    let run_info = use_signal(|| None::<RunInfo>);
    // A new run (file parsing or the affinity setup) clears any prior run's
    // numbers so the status line never shows stale info during it. The first
    // snapshot then sets fresh Running numbers.
    use_effect(move || {
        if matches!(status.read().as_str(), "running" | "loading") {
            let mut run_info = run_info;
            run_info.set(None);
        }
    });
    // Canvas size, tracking the viewport so the plot fills the page and refits
    // on resize (a window resize listener is installed below).
    let viewport = use_signal(window_size);
    let defaults = TsneParams::default();
    let mut pca_dims = use_signal(|| defaults.pca_dims);
    let mut perplexity = use_signal(|| defaults.perplexity);
    let mut theta = use_signal(|| defaults.theta);
    let mut epochs = use_signal(|| defaults.epochs);
    // Run without a fixed epoch budget: the fit keeps iterating until the user
    // pauses it. Implemented as a huge epoch count the run never reaches.
    let mut infinite = use_signal(|| false);
    // Bake the color legend into the downloaded snapshot (a reserved strip).
    let mut legend_in_export = use_signal(|| false);
    let mut learning_rate = use_signal(|| defaults.learning_rate);
    let mut early_exaggeration = use_signal(|| defaults.early_exaggeration);
    let mut exaggeration_epochs = use_signal(|| defaults.early_exaggeration_epochs);
    let mut color_source = use_signal(|| initial_color_source);
    let mut color_select = use_signal(|| None::<web_sys::HtmlSelectElement>);
    let mut dragging_over = use_signal(|| false);
    // Per-column roles (feature / label / ignore), reset to type-based defaults
    // whenever a new dataset is parsed (see the effect below).
    let mut columns = use_signal(Vec::<ColumnEntry>::new);

    // Reset the column roles to defaults each time the dataset changes.
    use_effect(move || {
        let defaults = dataset
            .read()
            .as_ref()
            .map(default_columns)
            .unwrap_or_default();
        columns.set(defaults);
    });

    // The label columns to color by under the current roles (text labels plus
    // any numeric column reassigned to a heatmap).
    let effective_labels = use_memo(move || -> Vec<LabelColumn> {
        dataset
            .read()
            .as_ref()
            .map(|d| effective_label_columns(d, &columns.read()))
            .unwrap_or_default()
    });

    // Builds the decomposition method from the current controls, shared by the
    // Run button, the auto-run on loading a dataset, and Continue.
    let build_method = move || -> DecompositionMethod {
        DecompositionMethod::Tsne(TsneParams {
            perplexity: perplexity(),
            theta: theta(),
            epochs: if infinite() {
                INFINITE_EPOCHS
            } else {
                epochs()
            },
            learning_rate: learning_rate(),
            pca_dims: pca_dims(),
            early_exaggeration: early_exaggeration(),
            early_exaggeration_epochs: exaggeration_epochs(),
            ..TsneParams::default()
        })
    };

    // The browser snaps a select back to its first option when the bound value
    // is applied before the matching option exists, which happens when loading a
    // dataset sets both the label column options and the source in one update.
    // Re-syncing the DOM value after renders keeps the displayed value honest.
    use_effect(move || {
        let source = color_source.read().clone();
        let _ = dataset.read();
        if let Some(select) = color_select() {
            select.set_value(&source);
        }
    });

    // The active coloring, recomputed on source or dataset changes. Recoloring
    // is a pure redraw, the embedding is never recomputed.
    let coloring_result = use_memo(move || -> Option<Coloring> {
        let source = color_source.read().clone();
        let n_samples = dataset.read().as_ref()?.n_samples;
        let labels = effective_labels.read();
        let column = labels
            .iter()
            .find(|c| c.name == source.strip_prefix("column:").unwrap_or(&source))?;
        if column.values.len() != n_samples {
            return None;
        }
        Some(colorize(&column.values))
    });
    let colors = use_memo(move || coloring_result.read().as_ref().map(|c| c.colors.clone()));
    let markers = use_memo(move || coloring_result.read().as_ref().map(|c| c.markers.clone()));

    // Legend focus: hovering a class dims every other class to grey
    // temporarily, clicking a class pins that dimming on until it is clicked
    // again or a click lands outside the legend. Hover takes precedence over the
    // pin while it lasts. Only categorical colorings have per-class entries to
    // focus, the continuous scale's legend is just its extremes.
    let legend_hovered = use_signal(|| None::<usize>);
    let legend_pinned = use_signal(|| None::<usize>);
    let highlight = use_memo(move || -> Option<(String, Marker)> {
        let coloring = coloring_result.read();
        let coloring = coloring.as_ref()?;
        if coloring.scale != ColorScale::Categorical {
            return None;
        }
        let index = legend_hovered().or(legend_pinned())?;
        coloring
            .legend
            .get(index)
            .map(|e| (e.color.clone(), e.marker))
    });

    // The worker is busy while parsing ("loading") or running.
    let busy = use_memo(move || {
        let status = status.read();
        matches!(status.as_str(), "running" | "loading") || status.starts_with("epoch ")
    });
    // Text for the spinner shown over the empty plot before the first embedding
    // streams back. It must track the phase too: a fresh run has no embedding
    // through both parsing and the neighbor search, so a hardcoded "Loading the
    // dataset" would mislabel the neighbor search.
    let loading_label = use_memo(move || {
        match (status.read().as_str(), *phase.read()) {
            ("loading", _) => "Loading the dataset",
            (_, Some(TsnePhase::FindingNeighbors)) => "Finding neighbors",
            _ => "Working",
        }
        .to_string()
    });
    // Compute progress for the loading bar: `(indeterminate, fraction)`, or
    // None when hidden. "loading" (parsing in the worker) and "running" (the
    // affinity setup) have no sub-progress (indeterminate); "epoch N" is
    // determinate against the epoch budget. Everything else (idle, done,
    // paused, error) hides the bar.
    let progress = use_memo(move || -> Option<(bool, f32)> {
        let status = status.read();
        if matches!(status.as_str(), "running" | "loading") {
            Some((true, 0.0))
        } else if status.starts_with("epoch ") && infinite() {
            // No fixed budget, so the bar stays indeterminate while it runs.
            Some((true, 0.0))
        } else if let Some(rest) = status.strip_prefix("epoch ") {
            let total = epochs().max(1) as f32;
            // The status carries a trailing elapsed time, "epoch N (1.2s)", so
            // take the leading token for the epoch index.
            let epoch = rest
                .split_whitespace()
                .next()
                .and_then(|token| token.parse::<f32>().ok())
                .unwrap_or(0.0);
            Some((false, (epoch / total).clamp(0.0, 1.0)))
        } else {
            None
        }
    });
    let can_drag = use_memo(move || embedding.read().is_some());
    // Set true when a grab paused a running fit, so releasing resumes it.
    let mut resume_pending = use_signal(|| false);

    // The bridge owns the worker and must live across renders. It is held behind
    // a RefCell so the running worker can be orphaned and replaced by a fresh one
    // when a grab pauses the fit (the orphaned worker self-closes after its
    // current run finishes).
    let bridge = use_hook(|| {
        Rc::new(RefCell::new(spawn_bridge(
            &worker_url,
            status,
            phase,
            embedding,
            dataset,
            ingest_error,
            color_source,
            run_info,
        )))
    });

    // WebM recording of the embedding animation through MediaRecorder. Arming
    // the checkbox only marks intent: capture begins at the first epoch (so the
    // loading and affinity-setup frames are skipped) and stops automatically on
    // convergence, leaving a finished video to download. The recorder and the
    // collected blob chunks live behind RefCell so the event closures can mutate
    // them.
    let recording_armed = use_signal(|| false);
    let recording_active = use_signal(|| false);
    let recorded_url = use_signal(|| None::<String>);
    let recorder = use_hook(|| Rc::new(RefCell::new(None::<web_sys::MediaRecorder>)));
    let recording_chunks = use_hook(|| Rc::new(RefCell::new(Vec::<wasm_bindgen::JsValue>::new())));

    // Check browser support once.
    let recording_supported = use_memo(|| {
        web_sys::MediaRecorder::is_type_supported("video/webm;codecs=vp9")
            || web_sys::MediaRecorder::is_type_supported("video/webm")
    });

    // Settings sidebar open state, toggled by the gear button. A document level
    // pointerdown listener closes it on any press outside both the sidebar and
    // the gear (the gear is excluded so its own toggle is not immediately
    // undone by this listener firing on the same press).
    let settings_open = use_signal(|| false);
    // In-app "About t-SNE" overlay, opened by the help button.
    let about_open = use_signal(|| false);
    use_hook(move || {
        let Some(document) = web_sys::window().and_then(|window| window.document()) else {
            return;
        };
        let closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
            if !settings_open() {
                return;
            }
            let inside = event
                .target()
                .and_then(|target| target.dyn_into::<web_sys::Node>().ok())
                .map(|node| {
                    web_sys::window()
                        .and_then(|window| window.document())
                        .map(|document| {
                            let in_panel = document
                                .get_element_by_id("decompositions-sidebar")
                                .is_some_and(|el| el.contains(Some(&node)));
                            let in_gear = document
                                .get_element_by_id("settings")
                                .is_some_and(|el| el.contains(Some(&node)));
                            in_panel || in_gear
                        })
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            if !inside {
                let mut settings_open = settings_open;
                settings_open.set(false);
            }
        }) as Box<dyn FnMut(web_sys::Event)>);
        let _ = document
            .add_event_listener_with_callback("pointerdown", closure.as_ref().unchecked_ref());
        closure.forget();
    });

    // Keep the canvas size in step with the viewport.
    use_hook(move || {
        let Some(window) = web_sys::window() else {
            return;
        };
        let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
            let mut viewport = viewport;
            viewport.set(window_size());
        }) as Box<dyn FnMut(web_sys::Event)>);
        let _ = window.add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref());
        closure.forget();
    });

    // Loads a file by parsing it in the worker (off the main thread, so the UI
    // stays responsive), optionally running `run` on the result. A load in
    // flight orphans the current worker so the new one starts immediately.
    let load = {
        let bridge = bridge.clone();
        let worker_url = worker_url.clone();
        move |name: String, bytes: Vec<u8>, run: Option<DecompositionMethod>| {
            if busy() {
                *bridge.borrow_mut() = spawn_bridge(
                    &worker_url,
                    status,
                    phase,
                    embedding,
                    dataset,
                    ingest_error,
                    color_source,
                    run_info,
                );
            }
            let mut status = status;
            let mut embedding = embedding;
            let mut ingest_error = ingest_error;
            ingest_error.set(None);
            embedding.set(None);
            status.set(String::from(if run.is_some() {
                "running"
            } else {
                "loading"
            }));
            bridge
                .borrow()
                .send(WorkerRequest::Load { name, bytes, run });
        }
    };

    // Starts a fresh run on the current dataset with the configured method, the
    // shared body of the Run button and of the auto-run when a dataset loads.
    let start_run = {
        let bridge = bridge.clone();
        move || {
            let Some(parsed) = dataset.read().clone() else {
                return;
            };
            let mut status = status;
            let mut ingest_error = ingest_error;
            let Some((data, n_features)) = build_feature_matrix(&parsed, &columns.read()) else {
                ingest_error.set(Some(String::from(
                    "select at least one feature column in settings",
                )));
                status.set(String::from("idle"));
                return;
            };
            ingest_error.set(None);
            status.set(String::from("running"));
            bridge.borrow().send(WorkerRequest::Decompose {
                data,
                n_samples: parsed.n_samples,
                n_features,
                method: build_method(),
            });
        }
    };

    // Sends a warm-started t-SNE run seeded with the current embedding, the
    // shared body of Continue and of resuming after a pause.
    let send_warm_start = {
        let bridge = bridge.clone();
        move || {
            let Some(parsed) = dataset.read().clone() else {
                return;
            };
            let Some(seed) = embedding.read().clone() else {
                return;
            };
            if seed.len() != parsed.n_samples * 2 {
                return;
            }
            let Some((data, n_features)) = build_feature_matrix(&parsed, &columns.read()) else {
                return;
            };
            let selected = DecompositionMethod::Tsne(TsneParams {
                perplexity: perplexity(),
                theta: theta(),
                epochs: if infinite() {
                    INFINITE_EPOCHS
                } else {
                    epochs()
                },
                learning_rate: learning_rate(),
                pca_dims: pca_dims(),
                early_exaggeration: early_exaggeration(),
                early_exaggeration_epochs: exaggeration_epochs(),
                initial_embedding: Some(seed),
                ..TsneParams::default()
            });
            let mut status = status;
            status.set(String::from("running"));
            bridge.borrow().send(WorkerRequest::Decompose {
                data,
                n_samples: parsed.n_samples,
                n_features,
                method: selected,
            });
        }
    };

    // Grabbing a point pauses a running fit: orphan the worker (it self-closes
    // once its current run ends) and install a fresh idle one, freezing the
    // layout for the drag.
    let on_drag_start = {
        let bridge = bridge.clone();
        let worker_url = worker_url.clone();
        move |_index: usize| {
            if busy() {
                *bridge.borrow_mut() = spawn_bridge(
                    &worker_url,
                    status,
                    phase,
                    embedding,
                    dataset,
                    ingest_error,
                    color_source,
                    run_info,
                );
                let mut status = status;
                status.set(String::from("paused"));
                resume_pending.set(true);
            }
        }
    };

    // Releasing a paused point resumes the fit, warm starting from the dragged
    // layout on the fresh worker installed when the grab paused it.
    let on_drag_end = {
        let send_warm_start = send_warm_start.clone();
        move |()| {
            if resume_pending() {
                resume_pending.set(false);
                send_warm_start();
            }
        }
    };

    // Resets the panel to its empty state: stop any run (orphaning its worker),
    // drop the dataset and embedding and clear the status. Tuning parameters and
    // the method are left as the user set them.
    let clear = {
        let bridge = bridge.clone();
        let worker_url = worker_url.clone();
        let recorder = recorder.clone();
        let recording_chunks = recording_chunks.clone();
        move |_| {
            if busy() {
                *bridge.borrow_mut() = spawn_bridge(
                    &worker_url,
                    status,
                    phase,
                    embedding,
                    dataset,
                    ingest_error,
                    color_source,
                    run_info,
                );
            }
            // Stop and discard any recording, armed or finished.
            if let Some(rec) = recorder.borrow_mut().take() {
                let _ = rec.stop();
            }
            recording_chunks.borrow_mut().clear();
            let mut recording_armed = recording_armed;
            let mut recording_active = recording_active;
            let mut recorded_url = recorded_url;
            if let Some(url) = recorded_url.read().clone() {
                let _ = web_sys::Url::revoke_object_url(&url);
            }
            recording_armed.set(false);
            recording_active.set(false);
            recorded_url.set(None);
            let mut dataset = dataset;
            let mut embedding = embedding;
            let mut status = status;
            let mut ingest_error = ingest_error;
            let mut color_source = color_source;
            let mut resume_pending = resume_pending;
            let mut run_info = run_info;
            dataset.set(None);
            embedding.set(None);
            status.set(String::from("idle"));
            ingest_error.set(None);
            run_info.set(None);
            color_source.set(String::from("none"));
            resume_pending.set(false);
        }
    };

    // Sets up the MediaRecorder on the plot canvas and starts capturing. Called
    // at the first epoch so the loading and affinity-setup frames are skipped.
    let start_capture = {
        let recorder = recorder.clone();
        let recording_chunks = recording_chunks.clone();
        move || {
            let mut recording_active = recording_active;
            let Some(canvas) = web_sys::window()
                .and_then(|window| window.document())
                .and_then(|document| document.get_element_by_id("scatter-plot"))
                .and_then(|element| element.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            else {
                return;
            };
            let Ok(stream) = canvas.capture_stream_with_frame_request_rate(30.0) else {
                return;
            };
            let mime = if web_sys::MediaRecorder::is_type_supported("video/webm;codecs=vp9") {
                "video/webm;codecs=vp9"
            } else {
                "video/webm"
            };
            let options = web_sys::MediaRecorderOptions::new();
            options.set_mime_type(mime);
            let Ok(media_recorder) =
                web_sys::MediaRecorder::new_with_media_stream_and_media_recorder_options(
                    &stream, &options,
                )
            else {
                return;
            };

            // Collect dataavailable chunks.
            let chunks = recording_chunks.clone();
            let data_closure = Closure::wrap(Box::new(move |event: web_sys::BlobEvent| {
                if let Some(data) = event.data() {
                    chunks.borrow_mut().push(data.into());
                }
            }) as Box<dyn FnMut(web_sys::BlobEvent)>);
            let _ = media_recorder.add_event_listener_with_callback(
                "dataavailable",
                data_closure.as_ref().unchecked_ref(),
            );
            data_closure.forget();

            // On stop, combine the chunks into a WebM and expose its object URL
            // for the download button (no automatic download).
            let chunks_on_stop = recording_chunks.clone();
            let stop_closure = Closure::wrap(Box::new(move || {
                // Runs from the recorder's stop event, outside any dioxus flush,
                // so setting signals here is safe (unlike from the effect).
                let mut recording_active = recording_active;
                let mut recording_armed = recording_armed;
                let mut recorded_url = recorded_url;
                recording_active.set(false);
                // One shot: disarm so the control becomes the download button.
                recording_armed.set(false);
                let array = js_sys::Array::new();
                {
                    let mut chunks = chunks_on_stop.borrow_mut();
                    for chunk in chunks.iter() {
                        array.push(chunk);
                    }
                    chunks.clear();
                }
                if array.length() == 0 {
                    return;
                }
                let bag = web_sys::BlobPropertyBag::new();
                bag.set_type("video/webm");
                let Ok(blob) = web_sys::Blob::new_with_blob_sequence_and_options(&array, &bag)
                else {
                    return;
                };
                if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
                    recorded_url.set(Some(url));
                }
            }) as Box<dyn FnMut()>);
            let _ = media_recorder
                .add_event_listener_with_callback("stop", stop_closure.as_ref().unchecked_ref());
            stop_closure.forget();

            recording_chunks.borrow_mut().clear();
            let _ = media_recorder.start();
            *recorder.borrow_mut() = Some(media_recorder);
            recording_active.set(true);
        }
    };

    // Stops the recorder, the stop event then assembles the video.
    let stop_capture = {
        let recorder = recorder.clone();
        move || {
            if let Some(rec) = recorder.borrow_mut().take() {
                let _ = rec.stop();
            }
        }
    };

    // Whether a capture is in flight. Non reactive on purpose: the effect below
    // must not read a signal it also writes, or setting one mid-flush re-enters
    // the async executor and panics ("RefCell already borrowed").
    let capturing = use_hook(|| Rc::new(RefCell::new(false)));

    // Drive capture off the run status: begin at the first epoch (skipping the
    // load and affinity setup) while armed, and stop once the animation ends
    // (convergence, error, pause, or clear). start_capture and stop_capture only
    // write signals the effect does not read, and the rest of the state (the
    // active flag, the video URL, disarming) is updated from the recorder's stop
    // event, which runs outside the flush.
    {
        let start_capture = start_capture.clone();
        let stop_capture = stop_capture.clone();
        let capturing = capturing.clone();
        use_effect(move || {
            let status = status.read();
            let armed = recording_armed();
            let running = matches!(status.as_str(), "running" | "loading");
            let animating = status.starts_with("epoch ");
            let mut capturing = capturing.borrow_mut();
            if armed && !*capturing && animating {
                *capturing = true;
                start_capture();
            } else if *capturing && !animating && !running {
                *capturing = false;
                stop_capture();
            }
        });
    }

    // Downloads the finished video, then releases its object URL.
    let download_video = move |_| {
        let mut recorded_url = recorded_url;
        let Some(url) = recorded_url.read().clone() else {
            return;
        };
        if let Some(anchor) = web_sys::window()
            .and_then(|window| window.document())
            .and_then(|document| document.create_element("a").ok())
            .and_then(|element| element.dyn_into::<web_sys::HtmlAnchorElement>().ok())
        {
            anchor.set_href(&url);
            anchor.set_download("decomposition.webm");
            anchor.click();
        }
        let _ = web_sys::Url::revoke_object_url(&url);
        recorded_url.set(None);
    };

    // Downloads the current frame as a PNG. Rendered to a fresh square canvas
    // framed to the embedding (not the full-bleed page) with a transparent
    // background, so the picture is tight and drops onto any backdrop.
    let download_image = move |_| {
        let embedding = embedding.read();
        let Some(points) = embedding.as_ref() else {
            return;
        };
        let colors = colors.read();
        let markers = markers.read();
        let highlight = highlight.read();
        // Bake the legend into the picture only when asked and there is one.
        let legend = legend_in_export()
            .then(|| coloring_result.read().as_ref().map(|c| c.legend.clone()))
            .flatten()
            .filter(|entries| !entries.is_empty());
        let (vw, vh) = viewport();
        let size = vw.min(vh).max(1);
        let Some(data_url) = crate::plot::snapshot_png(
            points,
            colors.as_deref(),
            markers.as_deref(),
            highlight.as_ref().map(|(c, m)| (c.as_str(), *m)),
            legend.as_deref(),
            size,
        ) else {
            return;
        };
        if let Some(anchor) = web_sys::window()
            .and_then(|window| window.document())
            .and_then(|document| document.create_element("a").ok())
            .and_then(|element| element.dyn_into::<web_sys::HtmlAnchorElement>().ok())
        {
            anchor.set_href(&data_url);
            anchor.set_download("tsne.png");
            anchor.click();
        }
    };

    // Transport controls (media-player style). Pause freezes a running fit by
    // orphaning its worker (it self-closes after its current run) and installing
    // a fresh idle one, so Play can warm-start from the frozen layout.
    let pause = {
        let bridge = bridge.clone();
        let worker_url = worker_url.clone();
        move || {
            if busy() {
                *bridge.borrow_mut() = spawn_bridge(
                    &worker_url,
                    status,
                    phase,
                    embedding,
                    dataset,
                    ingest_error,
                    color_source,
                    run_info,
                );
                let mut status = status;
                status.set(String::from("paused"));
            }
        }
    };
    // Play resumes from the current layout when there is one (a paused or
    // finished run), otherwise it starts a fresh run.
    let play = {
        let start_run = start_run.clone();
        let send_warm_start = send_warm_start.clone();
        move || {
            if dataset.read().is_none() {
                return;
            }
            if embedding.read().is_some() {
                send_warm_start();
            } else {
                start_run();
            }
        }
    };
    let toggle_play = {
        let play = play.clone();
        let pause = pause.clone();
        move |_| {
            if busy() {
                pause();
            } else {
                play();
            }
        }
    };
    let restart = {
        let start_run = start_run.clone();
        move |_| start_run()
    };
    // Rec arms recording and acts like Play. Toggling it off while armed or
    // recording stops the capture (which assembles the downloadable video).
    let rec_toggle = {
        let play = play.clone();
        let stop_capture = stop_capture.clone();
        let capturing = capturing.clone();
        move |_| {
            let mut recording_armed = recording_armed;
            if recording_armed() || recording_active() {
                recording_armed.set(false);
                stop_capture();
                *capturing.borrow_mut() = false;
            } else {
                let mut recorded_url = recorded_url;
                if let Some(url) = recorded_url.read().clone() {
                    let _ = web_sys::Url::revoke_object_url(&url);
                }
                recorded_url.set(None);
                recording_armed.set(true);
                if !busy() {
                    play();
                }
            }
        }
    };

    let drop_enabled = drop_zone.is_some();
    let drop_prompt = drop_zone
        .as_ref()
        .map(|z| z.prompt.clone())
        .unwrap_or_default();
    let drop_zone = drop_zone.clone();
    let has_examples = !examples.is_empty();
    // Whether there is any label column to color by under the current roles.
    let has_labels = use_memo(move || !effective_labels.read().is_empty());

    // Plot props gated on the enabled features, hoisted out of the rsx so the
    // conditionals stay plain Rust (`.then`) rather than inline if/else.
    let plot_draggable = draggable.then(|| can_drag.into());
    let on_point_moved = draggable.then(|| {
        EventHandler::new(move |(index, x, y): (usize, f32, f32)| {
            let mut embedding = embedding;
            embedding.with_mut(|current| {
                if let Some(current) = current.as_mut()
                    && 2 * index + 1 < current.len()
                {
                    current[2 * index] = x;
                    current[2 * index + 1] = y;
                }
            });
        })
    });
    let on_drag_start = draggable.then(|| EventHandler::new(on_drag_start));
    let on_drag_end = draggable.then(|| EventHandler::new(on_drag_end));

    rsx! {
        div {
            class: "decompositions-explorer",
            // A click anywhere outside the legend clears a pinned focus (the
            // legend stops propagation of its own clicks).
            onclick: move |_| {
                let mut legend_pinned = legend_pinned;
                if legend_pinned().is_some() {
                    legend_pinned.set(None);
                }
            },
            // The whole page is the drop target.
            ondragover: move |evt| {
                if drop_enabled {
                    evt.prevent_default();
                    dragging_over.set(true);
                }
            },
            ondragleave: move |_| dragging_over.set(false),
            ondrop: {
                let load = load.clone();
                move |evt: Event<DragData>| {
                    let zone = drop_zone.clone();
                    let load = load.clone();
                    async move {
                        let Some(zone) = zone else {
                            return;
                        };
                        evt.prevent_default();
                        dragging_over.set(false);
                        let Some(file) = evt.files().into_iter().next() else {
                            return;
                        };
                        let name = file.name();
                        let extension = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
                        if !zone.allows(&extension) {
                            let mut dataset = dataset;
                            let mut ingest_error = ingest_error;
                            dataset.set(None);
                            ingest_error.set(Some(format!("unsupported file type .{extension}")));
                            return;
                        }
                        match file.read_bytes().await {
                            Ok(bytes) => load(name, bytes.to_vec(), None),
                            Err(error) => {
                                let mut ingest_error = ingest_error;
                                ingest_error.set(Some(error.to_string()));
                            }
                        }
                    }
                }
            },

            if styled {
                style { {crate::DEFAULT_STYLE} }
            }

            // Full-bleed plot.
            div { class: "decompositions-plot-area",
                ScatterPlot {
                    embedding,
                    colors: Some(colors.into()),
                    markers: Some(markers.into()),
                    highlight: Some(highlight.into()),
                    draggable: plot_draggable,
                    on_point_moved,
                    on_drag_start,
                    on_drag_end,
                    width: viewport().0,
                    height: viewport().1,
                    pixel_ratio,
                }
            }

            // Color legend overlay (keeps data colors). For categorical scales
            // the entries focus their class on hover and pin it on click.
            if let Some(active) = coloring_result.read().as_ref() {
                {
                    let interactive = active.scale == ColorScale::Categorical;
                    rsx! {
                        div {
                            id: "legend",
                            class: if interactive { "decompositions-legend decompositions-legend--interactive" } else { "decompositions-legend" },
                            // Keep clicks inside the legend from clearing the pin.
                            onclick: move |evt| evt.stop_propagation(),
                            for (index, entry) in active.legend.iter().take(MAX_LEGEND_ENTRIES).enumerate() {
                                span {
                                    class: if interactive && legend_pinned() == Some(index) {
                                        "decompositions-legend-entry decompositions-legend-entry--pinned"
                                    } else {
                                        "decompositions-legend-entry"
                                    },
                                    onmouseenter: move |_| {
                                        if interactive {
                                            let mut legend_hovered = legend_hovered;
                                            legend_hovered.set(Some(index));
                                        }
                                    },
                                    onmouseleave: move |_| {
                                        if interactive {
                                            let mut legend_hovered = legend_hovered;
                                            legend_hovered.set(None);
                                        }
                                    },
                                    onclick: move |_| {
                                        if interactive {
                                            let mut legend_pinned = legend_pinned;
                                            let next = if legend_pinned() == Some(index) { None } else { Some(index) };
                                            legend_pinned.set(next);
                                        }
                                    },
                                    span {
                                        class: "decompositions-legend-swatch",
                                        style: "color: {entry.color};",
                                        "{entry.marker.glyph()}"
                                    }
                                    "{entry.label}"
                                }
                            }
                            if active.legend.len() > MAX_LEGEND_ENTRIES {
                                span { class: "decompositions-legend-more",
                                    "(+{active.legend.len() - MAX_LEGEND_ENTRIES} more)"
                                }
                            }
                        }
                    }
                }
            }

            // Top bar: brand/logo on the left, settings gear on the right.
            if controls {
                div { class: "decompositions-topbar",
                    div { class: "decompositions-brand",
                        a {
                            class: "decompositions-brand-link",
                            href: "/",
                            title: "Reload",
                            "aria-label": "Reload the page",
                            if let Some(logo) = logo.as_ref() {
                                img { class: "decompositions-logo", src: "{logo}", alt: "dioxus-decompositions" }
                            } else {
                                span { "dioxus-decompositions" }
                            }
                        }
                    }
                    div { class: "decompositions-topbar-actions",
                        button {
                            id: "help",
                            class: "decompositions-iconbtn",
                            title: "What is t-SNE?",
                            "aria-label": "What is t-SNE? Opens an explainer.",
                            onclick: move |_| {
                                let mut about_open = about_open;
                                about_open.set(true);
                            },
                            Icon { icon: FaCircleQuestion, width: 17, height: 17, class: "decompositions-icon" }
                        }
                        if let Some(repo_url) = repo_url.as_ref() {
                            a {
                                id: "repo",
                                class: "decompositions-iconbtn",
                                href: "{repo_url}",
                                target: "_blank",
                                rel: "noopener",
                                title: "Source code on GitHub",
                                "aria-label": "Source code on GitHub. Opens in a new tab.",
                                Icon { icon: FaGithub, width: 17, height: 17, class: "decompositions-icon" }
                            }
                        }
                        if let Some(support_url) = support_url.as_ref() {
                            a {
                                id: "support",
                                class: "decompositions-iconbtn decompositions-heartbtn",
                                href: "{support_url}",
                                target: "_blank",
                                rel: "noopener",
                                title: "Support this project",
                                "aria-label": "Support this project. Opens in a new tab.",
                                Icon { icon: FaHeart, width: 16, height: 16, class: "decompositions-icon" }
                            }
                        }
                        button {
                            id: "settings",
                            class: "decompositions-iconbtn",
                            title: HELP_SETTINGS,
                            "aria-label": HELP_SETTINGS,
                            onclick: move |_| {
                                let mut settings_open = settings_open;
                                settings_open.set(!settings_open());
                            },
                            Icon { icon: FaSliders, width: 16, height: 16, class: "decompositions-icon" }
                        }
                    }
                }
            }

            // Empty state covering the page: example buttons and the drop hint.
            // Gone once a dataset is loaded (so dropping a file clears it), but
            // the loading spinner still shows while a load or run is in flight.
            if (drop_enabled || has_examples)
                && embedding.read().is_none()
                && (dataset.read().is_none() || busy())
            {
                div {
                    id: "dropzone",
                    class: if dragging_over() { "decompositions-empty decompositions-empty--over" } else { "decompositions-empty" },
                    if busy() {
                        div { id: "loading", class: "decompositions-loading",
                            Icon { icon: FaSpinner, width: 32, height: 32, class: "decompositions-icon decompositions-spinner" }
                            span { "{loading_label}" }
                        }
                    } else {
                        if has_examples {
                            div { class: "decompositions-examples",
                                for (index, example) in examples.iter().enumerate() {
                                    button {
                                        id: "load-example-{index}",
                                        class: "decompositions-example",
                                        title: example.description.clone().unwrap_or_else(|| format!("Load the {} example dataset.", example.name)),
                                        "aria-label": example.description.clone().unwrap_or_else(|| format!("Load the {} example dataset.", example.name)),
                                        onclick: {
                                            let url = example.url.clone();
                                            let load = load.clone();
                                            move |_| {
                                                let url = url.clone();
                                                let load = load.clone();
                                                let mut status = status;
                                                status.set(String::from("loading"));
                                                async move {
                                                    let fetched = match gloo_net::http::Request::get(&url).send().await {
                                                        Ok(response) => response.binary().await,
                                                        Err(error) => Err(error),
                                                    };
                                                    match fetched {
                                                        Ok(bytes) => load(url, bytes, Some(build_method())),
                                                        Err(error) => {
                                                            let mut status = status;
                                                            let mut ingest_error = ingest_error;
                                                            status.set(String::from("idle"));
                                                            ingest_error.set(Some(error.to_string()));
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                        match example.icon {
                                            Some(ExampleIcon::Numbers) => rsx! {
                                                Icon { icon: FaCalculator, width: 14, height: 14, class: "decompositions-icon" }
                                            },
                                            Some(ExampleIcon::Apparel) => rsx! {
                                                Icon { icon: FaShirt, width: 14, height: 14, class: "decompositions-icon" }
                                            },
                                            Some(ExampleIcon::Network) => rsx! {
                                                Icon { icon: FaShareNodes, width: 14, height: 14, class: "decompositions-icon" }
                                            },
                                            None => rsx! {},
                                        }
                                        "{example.name}"
                                    }
                                }
                            }
                        }
                        if drop_enabled {
                            label { r#for: "file-input", class: "decompositions-drophint",
                                Icon { icon: FaFileArrowUp, width: 16, height: 16, class: "decompositions-icon" }
                                span { {drop_prompt} }
                            }
                            input {
                                id: "file-input",
                                r#type: "file",
                                class: "decompositions-fileinput",
                                accept: DATA_ACCEPT,
                                onchange: {
                                    let load = load.clone();
                                    move |evt: Event<FormData>| {
                                        let load = load.clone();
                                        async move {
                                            let Some(file) = evt.files().into_iter().next() else {
                                                return;
                                            };
                                            match file.read_bytes().await {
                                                Ok(bytes) => load(file.name(), bytes.to_vec(), None),
                                                Err(error) => {
                                                    let mut ingest_error = ingest_error;
                                                    ingest_error.set(Some(error.to_string()));
                                                }
                                            }
                                        }
                                    }
                                },
                            }
                        }
                        if let Some(error) = ingest_error.read().as_ref() {
                            p { id: "ingest-error", class: "decompositions-error",
                                Icon { icon: FaTriangleExclamation, width: 14, height: 14, class: "decompositions-icon" }
                                "{error.clone()}"
                            }
                        }
                    }
                }
            }

            // Bottom-center media-player transport bar.
            if controls && dataset.read().is_some() {
                div { class: "decompositions-transport",
                    div { class: "decompositions-transport-row",
                        // Play/Pause: hidden while a recording run is in flight,
                        // when the rec/stop button is the only control.
                        if !(busy() && (recording_active() || recording_armed())) {
                            button {
                                id: "play",
                                class: "decompositions-iconbtn decompositions-tp-play",
                                title: if busy() { "Pause" } else { "Play" },
                                "aria-label": if busy() { "Pause" } else { "Play" },
                                onclick: toggle_play,
                                if busy() {
                                    Icon { icon: FaPause, width: 15, height: 15, class: "decompositions-icon" }
                                } else {
                                    Icon { icon: FaPlay, width: 15, height: 15, class: "decompositions-icon" }
                                }
                            }
                        }
                        // Restart (run from scratch) only makes sense while stopped.
                        if !busy() {
                            button {
                                id: "restart",
                                class: "decompositions-iconbtn decompositions-tp-restart",
                                title: HELP_RUN,
                                "aria-label": HELP_RUN,
                                onclick: restart,
                                Icon { icon: FaRotateLeft, width: 14, height: 14, class: "decompositions-icon" }
                            }
                        }
                        // Rec: start a recording run while stopped, or stop it
                        // while recording. Hidden during a normal run.
                        if recording_supported() && (!busy() || recording_active() || recording_armed()) {
                            button {
                                id: "record",
                                class: if recording_active() || recording_armed() { "decompositions-iconbtn decompositions-tp-rec decompositions-rec--active" } else { "decompositions-iconbtn decompositions-tp-rec" },
                                title: HELP_RECORD,
                                "aria-label": HELP_RECORD,
                                onclick: rec_toggle,
                                Icon { icon: FaVideo, width: 15, height: 15, class: "decompositions-icon" }
                            }
                        }
                        // Snapshot the current frame as a PNG. Available whenever
                        // there is a frame to capture, running or stopped.
                        if embedding.read().is_some() {
                            button {
                                id: "snapshot",
                                class: "decompositions-iconbtn decompositions-tp-snap",
                                title: "Download a PNG of the current frame",
                                "aria-label": "Download a PNG of the current frame",
                                onclick: download_image,
                                Icon { icon: FaImage, width: 14, height: 14, class: "decompositions-icon" }
                            }
                        }
                        // Loading bar: contracts away when no run is in flight.
                        div {
                            class: if progress().is_some() { "decompositions-scrubber decompositions-scrubber--active" } else { "decompositions-scrubber" },
                            if let Some((indeterminate, fraction)) = progress() {
                                div {
                                    class: if indeterminate { "decompositions-scrubber-fill decompositions-scrubber-fill--indeterminate" } else { "decompositions-scrubber-fill" },
                                    style: if indeterminate { String::new() } else { format!("width: {}%;", fraction * 100.0) },
                                }
                            }
                        }
                        if recorded_url.read().is_some() && !recording_active() {
                            button {
                                id: "download-video",
                                class: "decompositions-iconbtn decompositions-tp-snap",
                                title: "Download the recorded video",
                                "aria-label": "Download the recorded video",
                                onclick: download_video,
                                Icon { icon: FaDownload, width: 14, height: 14, class: "decompositions-icon" }
                            }
                        }
                        button {
                            id: "clear",
                            class: "decompositions-iconbtn decompositions-tp-clear",
                            title: HELP_CLEAR,
                            "aria-label": HELP_CLEAR,
                            onclick: clear,
                            Icon { icon: FaTrashCan, width: 14, height: 14, class: "decompositions-icon" }
                        }
                    }
                    // Second line: the live status, each part hover-explained so a
                    // newcomer can learn the terms. Hidden when idle.
                    {
                        let status_str = status.read().clone();
                        let info = *run_info.read();
                        let current_phase = *phase.read();
                        let total = epochs();
                        let is_infinite = infinite();
                        if let Some(error) = status_str.strip_prefix("error: ") {
                            rsx! {
                                div { class: "decompositions-statusline decompositions-statusline--error",
                                    Icon { icon: FaTriangleExclamation, width: 12, height: 12, class: "decompositions-icon" }
                                    span { "{error}" }
                                }
                            }
                        } else {
                            match info {
                                Some(RunInfo::Running { epoch, elapsed_s, threads }) => {
                                    let epoch_text = if is_infinite { format!("epoch {epoch}") } else { format!("epoch {epoch}/{total}") };
                                    rsx! {
                                        div { class: "decompositions-statusline",
                                            if let Some(ph) = current_phase {
                                                span { class: "decompositions-statseg", title: phase_help(ph), "{ph.label()}" }
                                            }
                                            span { class: "decompositions-statseg", title: HELP_STATUS_EPOCH, "{epoch_text}" }
                                            span { class: "decompositions-statseg", title: HELP_STATUS_ELAPSED, "{elapsed_s:.1}s" }
                                            span { class: "decompositions-statseg", title: HELP_STATUS_THREADS, "{threads} {thread_word(threads)}" }
                                        }
                                    }
                                }
                                Some(RunInfo::Done { elapsed_s, threads, kl }) => {
                                    rsx! {
                                        div { class: "decompositions-statusline",
                                            span { class: "decompositions-statseg", title: HELP_STATUS_DONE, "Done" }
                                            span { class: "decompositions-statseg", title: HELP_STATUS_ELAPSED, "in {elapsed_s:.1}s" }
                                            span { class: "decompositions-statseg", title: HELP_STATUS_THREADS, "{threads} {thread_word(threads)}" }
                                            if let Some(kl) = kl {
                                                span { class: "decompositions-statseg", title: HELP_STATUS_KL, "KL {kl:.4}" }
                                            }
                                        }
                                    }
                                }
                                None => {
                                    // Before the first snapshot: the loading or
                                    // neighbor-search phase, or nothing when idle.
                                    let segment = match status_str.as_str() {
                                        "loading" => Some(("Loading the dataset".to_string(), HELP_STATUS_LOADING)),
                                        "running" => current_phase
                                            .map(|ph| (ph.label().to_string(), phase_help(ph))),
                                        _ => None,
                                    };
                                    if let Some((text, help)) = segment {
                                        rsx! {
                                            div { class: "decompositions-statusline",
                                                span { class: "decompositions-statseg", title: help, "{text}" }
                                            }
                                        }
                                    } else {
                                        rsx! {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Right settings sidebar with the tuning parameters.
            if controls {
                div {
                    id: "decompositions-sidebar",
                    class: if settings_open() { "decompositions-sidebar decompositions-sidebar--open" } else { "decompositions-sidebar" },
                    p { class: "decompositions-section-title", "Parameters" }
                    label { class: "decompositions-field", r#for: "perplexity", title: HELP_PERPLEXITY, "aria-label": HELP_PERPLEXITY,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaCircleNodes, width: 14, height: 14, class: "decompositions-icon" }
                            "Perplexity"
                        }
                        input {
                            id: "perplexity",
                            r#type: "number",
                            min: "1",
                            step: "1",
                            value: "{perplexity}",
                            onchange: move |evt| {
                                if let Ok(value) = evt.value().parse::<f32>() {
                                    perplexity.set(value.max(1.0));
                                }
                            },
                        }
                    }
                    label { class: "decompositions-field", r#for: "theta", title: HELP_THETA, "aria-label": HELP_THETA,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaBullseye, width: 14, height: 14, class: "decompositions-icon" }
                            "Theta"
                        }
                        input {
                            id: "theta",
                            r#type: "number",
                            min: "0.1",
                            max: "1",
                            step: "0.1",
                            value: "{theta}",
                            onchange: move |evt| {
                                if let Ok(value) = evt.value().parse::<f32>() {
                                    theta.set(value.clamp(0.1, 1.0));
                                }
                            },
                        }
                    }
                    label { class: "decompositions-field", r#for: "epochs", title: HELP_EPOCHS, "aria-label": HELP_EPOCHS,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaRepeat, width: 14, height: 14, class: "decompositions-icon" }
                            "Epochs"
                        }
                        input {
                            id: "epochs",
                            r#type: "number",
                            min: "1",
                            step: "50",
                            value: "{epochs}",
                            disabled: infinite(),
                            onchange: move |evt| {
                                if let Ok(value) = evt.value().parse::<usize>() {
                                    epochs.set(value.max(1));
                                }
                            },
                        }
                    }
                    label { class: "decompositions-field", r#for: "infinite", title: HELP_INFINITE, "aria-label": HELP_INFINITE,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaInfinity, width: 14, height: 14, class: "decompositions-icon" }
                            "Run forever"
                        }
                        input {
                            id: "infinite",
                            r#type: "checkbox",
                            checked: infinite(),
                            onchange: move |evt| infinite.set(evt.checked()),
                        }
                    }
                    label { class: "decompositions-field", r#for: "learning-rate", title: HELP_LEARNING_RATE, "aria-label": HELP_LEARNING_RATE,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaGaugeHigh, width: 14, height: 14, class: "decompositions-icon" }
                            "Learning rate"
                        }
                        input {
                            id: "learning-rate",
                            r#type: "number",
                            min: "1",
                            step: "10",
                            value: learning_rate().map(|v| v.to_string()).unwrap_or_default(),
                            placeholder: match dataset.read().as_ref() {
                                Some(d) => format!("auto ({:.0})", auto_learning_rate(d.n_samples)),
                                None => String::from("auto"),
                            },
                            onchange: move |evt| {
                                let text = evt.value();
                                if text.trim().is_empty() {
                                    learning_rate.set(None);
                                } else if let Ok(value) = text.parse::<f32>() {
                                    learning_rate.set(Some(value.max(1.0)));
                                }
                            },
                        }
                    }
                    label { class: "decompositions-field", r#for: "pca-dims", title: HELP_PCA_DIMS, "aria-label": HELP_PCA_DIMS,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaCompress, width: 14, height: 14, class: "decompositions-icon" }
                            "PCA dimensions"
                        }
                        input {
                            id: "pca-dims",
                            r#type: "number",
                            min: "2",
                            value: "{pca_dims}",
                            onchange: move |evt| {
                                if let Ok(dims) = evt.value().parse::<usize>() {
                                    pca_dims.set(dims.max(2));
                                }
                            },
                        }
                    }
                    label { class: "decompositions-field", r#for: "early-exaggeration", title: HELP_EARLY_EXAGGERATION, "aria-label": HELP_EARLY_EXAGGERATION,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaExpand, width: 14, height: 14, class: "decompositions-icon" }
                            "Early exaggeration"
                        }
                        input {
                            id: "early-exaggeration",
                            r#type: "number",
                            min: "1",
                            step: "1",
                            value: "{early_exaggeration}",
                            onchange: move |evt| {
                                if let Ok(value) = evt.value().parse::<f32>() {
                                    early_exaggeration.set(value.max(1.0));
                                }
                            },
                        }
                    }
                    label { class: "decompositions-field", r#for: "exaggeration-epochs", title: HELP_EXAGGERATION_EPOCHS, "aria-label": HELP_EXAGGERATION_EPOCHS,
                        span { class: "decompositions-field-label",
                            Icon { icon: FaFire, width: 14, height: 14, class: "decompositions-icon" }
                            "Exaggeration epochs"
                        }
                        input {
                            id: "exaggeration-epochs",
                            r#type: "number",
                            min: "0",
                            step: "10",
                            value: "{exaggeration_epochs}",
                            onchange: move |evt| {
                                if let Ok(value) = evt.value().parse::<usize>() {
                                    exaggeration_epochs.set(value);
                                }
                            },
                        }
                    }
                    if has_labels() {
                        p { class: "decompositions-section-title", "Color" }
                        label { class: "decompositions-field", r#for: "color-source", title: HELP_COLOR_BY, "aria-label": HELP_COLOR_BY,
                            span { class: "decompositions-field-label",
                                Icon { icon: FaPalette, width: 14, height: 14, class: "decompositions-icon" }
                                "Color by"
                            }
                            select {
                                id: "color-source",
                                class: "decompositions-select",
                                value: "{color_source}",
                                onmounted: move |evt| {
                                    color_select.set(
                                        evt.data()
                                            .downcast::<web_sys::Element>()
                                            .and_then(|element| {
                                                element.clone().dyn_into::<web_sys::HtmlSelectElement>().ok()
                                            }),
                                    );
                                },
                                onchange: move |evt| color_source.set(evt.value()),
                                option { value: "none", "no color" }
                                for column in effective_labels.read().iter() {
                                    option { value: "column:{column.name}", "{column.name}" }
                                }
                            }
                        }
                        label { class: "decompositions-field", r#for: "legend-export", title: HELP_LEGEND_EXPORT, "aria-label": HELP_LEGEND_EXPORT,
                            span { class: "decompositions-field-label",
                                Icon { icon: FaImage, width: 14, height: 14, class: "decompositions-icon" }
                                "Legend in snapshot"
                            }
                            input {
                                id: "legend-export",
                                r#type: "checkbox",
                                checked: legend_in_export(),
                                onchange: move |evt| legend_in_export.set(evt.checked()),
                            }
                        }
                    }
                    if !columns.read().is_empty() {
                        {
                            let cols = columns.read();
                            let features = cols.iter().filter(|c| c.role == ColumnRole::Feature).count();
                            let labels = cols.iter().filter(|c| c.role == ColumnRole::Label).count();
                            let counts = format!(
                                "{features} {} · {labels} {}",
                                if features == 1 { "feature" } else { "features" },
                                if labels == 1 { "label" } else { "labels" },
                            );
                            drop(cols);
                            rsx! {
                                p {
                                    class: "decompositions-section-title",
                                    title: HELP_COLUMNS,
                                    "aria-label": HELP_COLUMNS,
                                    "Columns "
                                    span { class: "decompositions-section-count", "{counts}" }
                                }
                            }
                        }
                        div { class: "decompositions-columns",
                            for (index, column) in columns.read().iter().enumerate() {
                                div { class: "decompositions-column-row",
                                    span { class: "decompositions-column-label",
                                        match column.role {
                                            ColumnRole::Feature => rsx! {
                                                Icon { icon: FaHashtag, width: 11, height: 11, class: "decompositions-icon decompositions-column-roleicon", title: "Feature (fed into t-SNE)" }
                                            },
                                            ColumnRole::Label => rsx! {
                                                Icon { icon: FaTag, width: 11, height: 11, class: "decompositions-icon decompositions-column-roleicon", title: "Label (color only)" }
                                            },
                                            ColumnRole::Ignore => rsx! {
                                                Icon { icon: FaBan, width: 11, height: 11, class: "decompositions-icon decompositions-column-roleicon", title: "Ignored" }
                                            },
                                        }
                                        span {
                                            class: "decompositions-column-name",
                                            title: "{column.name}",
                                            "{column.name}"
                                        }
                                    }
                                    select {
                                        title: HELP_COLUMNS,
                                        "aria-label": HELP_COLUMNS,
                                        value: match column.role {
                                            ColumnRole::Feature => "feature",
                                            ColumnRole::Label => "label",
                                            ColumnRole::Ignore => "ignore",
                                        },
                                        onchange: move |evt| {
                                            let role = match evt.value().as_str() {
                                                "feature" => ColumnRole::Feature,
                                                "label" => ColumnRole::Label,
                                                _ => ColumnRole::Ignore,
                                            };
                                            columns.write()[index].role = role;
                                        },
                                        if column.numeric {
                                            option { value: "feature", "Feature" }
                                        }
                                        option { value: "label", "Label" }
                                        option { value: "ignore", "Ignore" }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // In-app "About t-SNE" overlay, animated in over the plot.
            if about_open() {
                div {
                    class: "decompositions-about-backdrop",
                    onclick: move |_| {
                        let mut about_open = about_open;
                        about_open.set(false);
                    },
                    div {
                        class: "decompositions-about",
                        role: "dialog",
                        "aria-modal": "true",
                        onclick: move |evt| evt.stop_propagation(),
                        button {
                            class: "decompositions-about-close",
                            title: "Close",
                            "aria-label": "Close",
                            onclick: move |_| {
                                let mut about_open = about_open;
                                about_open.set(false);
                            },
                            Icon { icon: FaXmark, width: 16, height: 16, class: "decompositions-icon" }
                        }
                        h2 { "t-SNE" }
                        p { class: "decompositions-about-sub", "t-distributed Stochastic Neighbor Embedding" }

                        h3 { "What it is" }
                        p {
                            "t-SNE is a nonlinear dimensionality-reduction method for visualizing "
                            "high-dimensional data in two dimensions. It places each point so that "
                            "points near each other in the original space stay near each other in "
                            "the picture, which makes local structure and clusters easy to see "
                            a {
                                href: "https://www.jmlr.org/papers/v9/vandermaaten08a.html",
                                target: "_blank",
                                rel: "noopener",
                                "(van der Maaten & Hinton, 2008)"
                            }
                            "."
                        }
                        p {
                            "This tool runs Barnes-Hut t-SNE "
                            a {
                                href: "https://www.jmlr.org/papers/v15/vandermaaten14a.html",
                                target: "_blank",
                                rel: "noopener",
                                "(van der Maaten, 2014)"
                            }
                            ", an approximation that scales to tens of thousands of points, "
                            "entirely in your browser on a background worker. The input is first "
                            "reduced with PCA (30 dimensions by default) to speed up the neighbor "
                            "search and cut noise, then t-SNE produces the layout you watch evolve. "
                            "The embedding is initialized from the top principal components rather "
                            "than from random noise, which preserves the global layout of the data "
                            "and makes runs reproducible "
                            a {
                                href: "https://doi.org/10.1038/s41467-019-13056-x",
                                target: "_blank",
                                rel: "noopener",
                                "(Kobak & Berens, 2019)"
                            }
                            " "
                            a {
                                href: "https://doi.org/10.1038/s41587-020-00809-z",
                                target: "_blank",
                                rel: "noopener",
                                "(Kobak & Linderman, 2021)"
                            }
                            "."
                        }

                        h3 { "When to use it" }
                        p {
                            "Reach for t-SNE when you want to explore high-dimensional data and ask "
                            "whether it has structure and what clusters together. Common inputs are "
                            "learned embeddings, image or text feature vectors, single-cell gene "
                            "expression, and any table of numeric features per sample. It is an "
                            "exploratory and presentation tool, not a preprocessing step for "
                            "downstream models."
                        }

                        h3 { "How to read it (and what not to read into it)" }
                        p {
                            "t-SNE maps are powerful but easy to over-interpret. The caveats below "
                            "are drawn from "
                            a {
                                href: "https://distill.pub/2016/misread-tsne/",
                                target: "_blank",
                                rel: "noopener",
                                "(Wattenberg et al., 2016)"
                            }
                            ":"
                        }
                        ul {
                            li {
                                b { "Perplexity matters. " }
                                "It sets roughly how many neighbors each point considers, and "
                                "different values give different pictures. 5 to 50 is typical."
                            }
                            li {
                                b { "Cluster sizes are not meaningful. " }
                                "t-SNE expands dense clusters and contracts sparse ones, so a blob's "
                                "area says little about how spread out that group really is."
                            }
                            li {
                                b { "Distances between clusters are often not meaningful. " }
                                "Treat the global arrangement with caution."
                            }
                            li {
                                b { "Let it converge. " }
                                "Stopping early leaves a half-formed layout. Run enough epochs, or "
                                "use \"run forever\" and watch."
                            }
                            li {
                                b { "Runs vary. " }
                                "The optimization is stochastic, so the stable signal is the cluster "
                                "structure, not the exact positions."
                            }
                        }
                        p {
                            "The settings panel exposes the knobs that drive all of this: "
                            "perplexity, the Barnes-Hut accuracy (theta), epochs, learning rate, "
                            "PCA dimensions, and the early-exaggeration phase."
                        }

                        h3 { "Built in Rust" }
                        p {
                            "This whole tool is "
                            a { href: "https://www.rust-lang.org/what/wasm", target: "_blank", rel: "noopener", "Rust compiled to WebAssembly" }
                            ", served as static files with no backend. The interface (rendered by "
                            a { href: "https://dioxuslabs.com", target: "_blank", rel: "noopener", "Dioxus" }
                            "), the file parsing, and "
                            a { href: "https://github.com/frjnn/bhtsne", target: "_blank", rel: "noopener", "t-SNE" }
                            " itself all run in your browser, so the data you load never leaves "
                            "your machine."
                        }
                        p {
                            "t-SNE runs across all of your CPU cores at once with "
                            a { href: "https://github.com/rayon-rs/rayon", target: "_blank", rel: "noopener", "Rayon" }
                            ", which works in the browser through "
                            a { href: "https://github.com/RReverser/wasm-bindgen-rayon", target: "_blank", rel: "noopener", "wasm-bindgen-rayon" }
                            " once the page is cross-origin isolated (the "
                            a { href: "https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Opener-Policy", target: "_blank", rel: "noopener", "COOP" }
                            " and "
                            a { href: "https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Embedder-Policy", target: "_blank", rel: "noopener", "COEP" }
                            " headers)."
                        }
                        p {
                            "Modern web development will be written in Rust. "
                            a { href: "https://xkcd.com/2314/", target: "_blank", rel: "noopener", "Carcinization advances." }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_leaves_every_feature_off() {
        let decomposition = Decomposition::new();
        assert!(decomposition.dataset.is_none());
        assert!(decomposition.drop_zone.is_none());
        assert!(decomposition.examples.is_empty());
        assert!(!decomposition.controls);
        assert!(!decomposition.draggable);
        assert!(decomposition.styled);
    }

    #[test]
    fn full_enables_loader_controls_and_dragging() {
        let decomposition = Decomposition::full();
        assert!(decomposition.drop_zone.is_some());
        assert!(decomposition.controls);
        assert!(decomposition.draggable);
    }

    #[test]
    fn worker_url_defaults_and_overrides() {
        assert_eq!(Decomposition::new().worker_url, DEFAULT_WORKER_URL);
        let custom = Decomposition::new().worker_url("/custom/worker.js");
        assert_eq!(custom.worker_url, "/custom/worker.js");
    }

    #[test]
    fn dataset_and_labels_preload_a_colored_dataset() {
        let decomposition = Decomposition::new()
            .dataset(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2)
            .labels("group", vec!["a".into(), "b".into(), "a".into()]);
        let dataset = decomposition.dataset.as_ref().expect("a preset dataset");
        assert_eq!(dataset.n_samples, 3);
        assert_eq!(dataset.n_features, 2);
        assert_eq!(dataset.data.len(), 6);
        assert_eq!(dataset.label_columns.len(), 1);
        assert_eq!(dataset.label_columns[0].name, "group");
        assert_eq!(dataset.label_columns[0].values, ["a", "b", "a"]);
    }

    #[test]
    fn multiple_label_columns_are_kept_in_order() {
        let decomposition = Decomposition::new()
            .dataset(vec![0.0; 4], 2, 2)
            .labels("first", vec!["x".into(), "y".into()])
            .labels("second", vec!["1".into(), "2".into()]);
        let names: Vec<&str> = decomposition
            .dataset
            .as_ref()
            .unwrap()
            .label_columns
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(names, ["first", "second"]);
    }

    #[test]
    fn labels_without_a_dataset_is_a_noop() {
        let decomposition = Decomposition::new().labels("group", vec!["a".into()]);
        assert!(decomposition.dataset.is_none());
    }

    #[test]
    fn drop_zone_accepts_normalize_extensions() {
        let zone = DropZone::new().accept([".CSV", "Parquet"]);
        assert!(zone.allows("csv"));
        assert!(zone.allows("parquet"));
        assert!(!zone.allows("tsv"));
    }
}
