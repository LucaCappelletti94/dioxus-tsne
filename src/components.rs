//! The reusable Dioxus components.
//!
//! [`Decomposition`] is the entry point: a method neutral (t-SNE, PCA, more to
//! come) visualizer configured through a fluent builder. Everything lives on a
//! single decomposer panel, the plot with a thin toolbar of controls on top, an
//! empty state that doubles as the drag and drop loader (with optional example
//! dataset buttons), and an automatic color legend.

use std::cell::RefCell;
use std::rc::Rc;

use crate::color::{Coloring, colorize};
use crate::ingest::{Dataset, LabelColumn};
use crate::messages::{DecompositionMethod, TsneParams, WorkerRequest, WorkerResponse};
use crate::plot::ScatterPlot;
use crate::worker::DecompositionWorker;
use dioxus::html::HasFileData;
use dioxus::prelude::*;
use dioxus_free_icons::Icon;
use dioxus_free_icons::icons::fa_solid_icons::{
    FaBullseye, FaCircleInfo, FaCircleNodes, FaCircleStop, FaCompress, FaDownload, FaFileArrowUp,
    FaForwardStep, FaGaugeHigh, FaPlay, FaRepeat, FaSliders, FaSpinner, FaTriangleExclamation,
    FaVideo, FaXmark,
};
use gloo_worker::{Spawnable, WorkerBridge};
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

/// Longest legend rendered before truncation.
const MAX_LEGEND_ENTRIES: usize = 20;

/// "thread" or "threads", to read naturally next to a pool size in the status.
fn thread_word(threads: usize) -> &'static str {
    if threads == 1 { "thread" } else { "threads" }
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
const HELP_METHOD: &str = "How to flatten the data to two dimensions: t-SNE groups similar points into clusters, PCA is a faster straight line projection.";
const HELP_PERPLEXITY: &str = "Roughly how many close neighbors each point pays attention to. Smaller makes tight local clumps, larger spreads things out. 5 to 50 is typical.";
const HELP_THETA: &str = "Trades t-SNE speed against accuracy. Lower is more accurate but slower, higher is faster but rougher. 0.5 is a sensible default.";
const HELP_EPOCHS: &str =
    "How many refinement steps to run. More steps polish the layout further but take longer.";
const HELP_LEARNING_RATE: &str = "How big each refinement step is. Too small and it gets stuck, too large and it looks chaotic. 200 is a common value.";
const HELP_PCA_DIMS: &str = "Before t-SNE the data is first squeezed to this many dimensions to speed things up. 50 is the standard choice.";
const HELP_RUN: &str =
    "Start the layout from scratch and watch the scatter plot animate as it improves.";
const HELP_CONTINUE: &str =
    "Run more steps starting from the current layout, keeping any points you have dragged.";
const HELP_COLOR_BY: &str =
    "Color the points by one of the dataset's label columns to see where each group lands.";
const HELP_SETTINGS: &str = "Advanced t-SNE settings.";
const HELP_RECORD: &str = "Record the scatter plot animation as a WebM video. Toggle while a run is active to capture the embedding evolving.";
const HELP_CLEAR: &str = "Clear the dataset and result, returning to the start.";

/// File input `accept` attribute.
const DATA_ACCEPT: &str = ".csv,.tsv,.parquet";

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
    embedding: Signal<Option<Vec<f32>>>,
    dataset: Signal<Option<Dataset>>,
    ingest_error: Signal<Option<String>>,
    color_source: Signal<String>,
) -> WorkerBridge<DecompositionWorker> {
    DecompositionWorker::spawner()
        // The URL points at a loader module that initializes the wasm-bindgen
        // output itself, see the crate documentation.
        .with_loader(true)
        .callback(move |response| {
            // Signals are Copy, the rebindings make the closure a plain Fn.
            let mut status = status;
            let mut embedding = embedding;
            let mut dataset = dataset;
            let mut ingest_error = ingest_error;
            let mut color_source = color_source;
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
                WorkerResponse::Snapshot {
                    epoch,
                    embedding: snapshot,
                    elapsed_ms,
                    threads,
                } => {
                    status.set(format!(
                        "epoch {epoch} ({:.1}s, {threads} {})",
                        elapsed_ms / 1000.0,
                        thread_word(threads)
                    ));
                    embedding.set(Some(snapshot));
                }
                WorkerResponse::Done {
                    embedding: done,
                    explained_variance_ratio,
                    elapsed_ms,
                    threads,
                } => {
                    let seconds = elapsed_ms / 1000.0;
                    let pool = format!("{threads} {}", thread_word(threads));
                    status.set(match explained_variance_ratio {
                        Some(ratios) => format!(
                            "done in {seconds:.1}s ({pool}), explained variance: {}",
                            ratios
                                .iter()
                                .map(|r| format!("{:.1}%", r * 100.0))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                        None => format!("done in {seconds:.1}s ({pool})"),
                    });
                    embedding.set(Some(done));
                }
                WorkerResponse::Error { message } => {
                    status.set(format!("error: {message}"));
                }
            }
        })
        .spawn(worker_url)
}

/// A bundled example dataset offered through a one click load button.
#[derive(Debug, Clone, PartialEq)]
pub struct ExampleDataset {
    /// Human readable name shown on the button.
    pub name: String,
    /// URL the file is served from, any supported format.
    pub url: String,
}

/// Configuration of the drag and drop loader shown over the empty plot, built
/// fluently.
///
/// ```
/// use dioxus_decompositions::DropZone;
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
            accept: ["csv", "tsv", "parquet"]
                .into_iter()
                .map(String::from)
                .collect(),
            prompt: String::from("Drop a CSV, TSV or Parquet file here, or click to browse"),
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
/// use dioxus_decompositions::{Decomposition, ExampleDataset};
///
/// Decomposition::new()
///     .drop_zone()
///     .examples(vec![ExampleDataset { name: "MNIST".into(), url: "/mnist.parquet".into() }])
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
        }
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
    let embedding = use_signal(|| None::<Vec<f32>>);
    let mut method = use_signal(|| String::from("tsne"));
    let defaults = TsneParams::default();
    let mut pca_dims = use_signal(|| defaults.pca_dims);
    let mut perplexity = use_signal(|| defaults.perplexity);
    let mut theta = use_signal(|| defaults.theta);
    let mut epochs = use_signal(|| defaults.epochs);
    let mut learning_rate = use_signal(|| defaults.learning_rate);
    let mut color_source = use_signal(|| initial_color_source);
    let mut color_select = use_signal(|| None::<web_sys::HtmlSelectElement>);
    let mut dragging_over = use_signal(|| false);

    // Builds the decomposition method from the current controls, shared by the
    // Run button, the auto-run on loading a dataset, and Continue.
    let build_method = move || -> DecompositionMethod {
        if method.read().as_str() == "pca" {
            DecompositionMethod::Pca
        } else {
            DecompositionMethod::Tsne(TsneParams {
                perplexity: perplexity(),
                theta: theta(),
                epochs: epochs(),
                learning_rate: learning_rate(),
                pca_dims: pca_dims(),
                ..TsneParams::default()
            })
        }
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
        let guard = dataset.read();
        let parsed = guard.as_ref()?;
        let labels = parsed
            .label_columns
            .iter()
            .find(|c| c.name == source.strip_prefix("column:").unwrap_or(&source))?;
        if labels.values.len() != parsed.n_samples {
            return None;
        }
        Some(colorize(&labels.values))
    });
    let colors = use_memo(move || coloring_result.read().as_ref().map(|c| c.colors.clone()));
    let markers = use_memo(move || coloring_result.read().as_ref().map(|c| c.markers.clone()));

    // The worker is busy while parsing ("loading") or running.
    let busy = use_memo(move || {
        let status = status.read();
        matches!(status.as_str(), "running" | "loading") || status.starts_with("epoch ")
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
            embedding,
            dataset,
            ingest_error,
            color_source,
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

    // Settings popover open state. A native <details> stays open until its own
    // summary is clicked again, so the open state is driven from a signal and a
    // document level pointerdown listener closes it on any press outside the
    // popover (the summary's onclick toggles it, see the markup below).
    let settings_open = use_signal(|| false);
    use_hook(move || {
        let Some(document) = web_sys::window().and_then(|window| window.document()) else {
            return;
        };
        let closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
            if !settings_open() {
                return;
            }
            // Keep the popover open only when the press landed inside it.
            let inside = event
                .target()
                .and_then(|target| target.dyn_into::<web_sys::Node>().ok())
                .and_then(|node| {
                    web_sys::window()
                        .and_then(|window| window.document())
                        .and_then(|document| document.get_element_by_id("settings-popover"))
                        .map(|popover| popover.contains(Some(&node)))
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
                    embedding,
                    dataset,
                    ingest_error,
                    color_source,
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
            status.set(String::from("running"));
            bridge.borrow().send(WorkerRequest::Decompose {
                data: parsed.data,
                n_samples: parsed.n_samples,
                n_features: parsed.n_features,
                method: build_method(),
            });
        }
    };
    let run = {
        let start_run = start_run.clone();
        move |_| start_run()
    };

    // Continuing needs an embedding that still matches the current dataset (the
    // length check rejects a dataset swapped after the last run) and the t-SNE
    // method, since the seed lives in t-SNE's two dimensional output.
    let can_continue = use_memo(move || {
        !busy()
            && method.read().as_str() == "tsne"
            && dataset
                .read()
                .as_ref()
                .zip(embedding.read().as_ref())
                .is_some_and(|(parsed, current)| current.len() == parsed.n_samples * 2)
    });

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
            let selected = DecompositionMethod::Tsne(TsneParams {
                perplexity: perplexity(),
                theta: theta(),
                epochs: epochs(),
                learning_rate: learning_rate(),
                pca_dims: pca_dims(),
                initial_embedding: Some(seed),
                ..TsneParams::default()
            });
            let mut status = status;
            status.set(String::from("running"));
            bridge.borrow().send(WorkerRequest::Decompose {
                data: parsed.data,
                n_samples: parsed.n_samples,
                n_features: parsed.n_features,
                method: selected,
            });
        }
    };

    let continue_run = {
        let send_warm_start = send_warm_start.clone();
        move |_| {
            send_warm_start();
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
                    embedding,
                    dataset,
                    ingest_error,
                    color_source,
                );
                let mut status = status;
                status.set(String::from("paused"));
                resume_pending.set(true);
            }
        }
    };

    // Releasing a paused point resumes the fit, warm starting from the dragged
    // layout on the fresh worker installed when the grab paused it.
    let on_drag_end = move |()| {
        if resume_pending() {
            resume_pending.set(false);
            send_warm_start();
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
                    embedding,
                    dataset,
                    ingest_error,
                    color_source,
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
            dataset.set(None);
            embedding.set(None);
            status.set(String::from("idle"));
            ingest_error.set(None);
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

    let drop_enabled = drop_zone.is_some();
    let drop_prompt = drop_zone
        .as_ref()
        .map(|z| z.prompt.clone())
        .unwrap_or_default();
    let drop_zone = drop_zone.clone();
    let has_examples = !examples.is_empty();
    // Whether the loaded dataset offers label columns to color by.
    let has_labels = use_memo(move || {
        dataset
            .read()
            .as_ref()
            .is_some_and(|d| !d.label_columns.is_empty())
    });

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
        div { class: "decompositions-explorer",
            if styled {
                style { {crate::DEFAULT_STYLE} }
            }
            div { class: "decompositions-panel",
                if controls {
                    div { class: "decompositions-toolbar",
                        select {
                            id: "method",
                            class: "decompositions-select",
                            value: "{method}",
                            title: HELP_METHOD,
                            "aria-label": HELP_METHOD,
                            onchange: move |evt| method.set(evt.value()),
                            option { value: "tsne", "t-SNE" }
                            option { value: "pca", "PCA" }
                        }
                        button {
                            id: "run",
                            title: HELP_RUN,
                            "aria-label": HELP_RUN,
                            disabled: dataset.read().is_none() || busy(),
                            onclick: run,
                            Icon { icon: FaPlay, width: 14, height: 14, class: "decompositions-icon" }
                            "Run"
                        }
                        button {
                            id: "continue",
                            title: HELP_CONTINUE,
                            "aria-label": HELP_CONTINUE,
                            disabled: !can_continue(),
                            onclick: continue_run,
                            Icon { icon: FaForwardStep, width: 14, height: 14, class: "decompositions-icon" }
                            "Continue"
                        }
                        if has_labels() {
                            select {
                                id: "color-source",
                                class: "decompositions-select",
                                value: "{color_source}",
                                title: HELP_COLOR_BY,
                                "aria-label": HELP_COLOR_BY,
                                onmounted: move |evt| {
                                    color_select.set(
                                        evt.data()
                                            .downcast::<web_sys::Element>()
                                            .and_then(|element| {
                                                element
                                                    .clone()
                                                    .dyn_into::<web_sys::HtmlSelectElement>()
                                                    .ok()
                                            }),
                                    );
                                },
                                onchange: move |evt| color_source.set(evt.value()),
                                option { value: "none", "no color" }
                                if let Some(parsed) = dataset.read().as_ref() {
                                    for column in parsed.label_columns.iter() {
                                        option { value: "column:{column.name}", "{column.name}" }
                                    }
                                }
                            }
                        }
                        if method.read().as_str() == "tsne" {
                            details {
                                id: "settings-popover",
                                class: "decompositions-settings",
                                open: settings_open(),
                                summary {
                                    id: "settings",
                                    class: "decompositions-settings-summary",
                                    title: HELP_SETTINGS,
                                    "aria-label": HELP_SETTINGS,
                                    // Drive the open state from the signal instead of the native
                                    // summary toggle, so the outside-click listener can close it too.
                                    onclick: move |evt| {
                                        evt.prevent_default();
                                        let mut settings_open = settings_open;
                                        settings_open.set(!settings_open());
                                    },
                                    Icon { icon: FaSliders, width: 15, height: 15, class: "decompositions-icon" }
                                }
                                div { class: "decompositions-settings-panel",
                                    label { r#for: "perplexity", title: HELP_PERPLEXITY,
                                        Icon { icon: FaCircleNodes, width: 14, height: 14, class: "decompositions-icon" }
                                        "Perplexity"
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
                                    label { r#for: "theta", title: HELP_THETA,
                                        Icon { icon: FaBullseye, width: 14, height: 14, class: "decompositions-icon" }
                                        "Theta"
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
                                    label { r#for: "epochs", title: HELP_EPOCHS,
                                        Icon { icon: FaRepeat, width: 14, height: 14, class: "decompositions-icon" }
                                        "Epochs"
                                        input {
                                            id: "epochs",
                                            r#type: "number",
                                            min: "1",
                                            step: "50",
                                            value: "{epochs}",
                                            onchange: move |evt| {
                                                if let Ok(value) = evt.value().parse::<usize>() {
                                                    epochs.set(value.max(1));
                                                }
                                            },
                                        }
                                    }
                                    label { r#for: "learning-rate", title: HELP_LEARNING_RATE,
                                        Icon { icon: FaGaugeHigh, width: 14, height: 14, class: "decompositions-icon" }
                                        "Learning rate"
                                        input {
                                            id: "learning-rate",
                                            r#type: "number",
                                            min: "1",
                                            step: "10",
                                            // Empty means auto: bhtsne resolves the size-scaled
                                            // default and the placeholder previews it for the
                                            // loaded dataset.
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
                                    label { r#for: "pca-dims", title: HELP_PCA_DIMS,
                                        Icon { icon: FaCompress, width: 14, height: 14, class: "decompositions-icon" }
                                        "PCA dimensions"
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
                                    if recording_supported() {
                                        label {
                                            r#for: "record-video",
                                            title: HELP_RECORD,
                                            Icon { icon: FaVideo, width: 14, height: 14, class: "decompositions-icon" }
                                            "Record video"
                                            input {
                                                id: "record-video",
                                                r#type: "checkbox",
                                                checked: recording_armed(),
                                                // Enabled even before a dataset is loaded, so the
                                                // user can arm recording first. Capture itself only
                                                // begins at the first epoch (see the status effect),
                                                // so the loading frames are skipped, and stops on
                                                // convergence into a downloadable video.
                                                onchange: move |evt| {
                                                    let mut recording_armed = recording_armed;
                                                    let mut recorded_url = recorded_url;
                                                    if evt.checked() {
                                                        if let Some(url) = recorded_url.read().clone() {
                                                            let _ = web_sys::Url::revoke_object_url(&url);
                                                        }
                                                        recorded_url.set(None);
                                                        recording_armed.set(true);
                                                    } else {
                                                        recording_armed.set(false);
                                                        stop_capture();
                                                        // Clear the in-flight flag so re-arming during
                                                        // the same run starts a fresh capture (the effect
                                                        // only resets it once the animation ends).
                                                        *capturing.borrow_mut() = false;
                                                    }
                                                },
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        p { id: "status", class: "decompositions-status",
                            Icon { icon: FaCircleInfo, width: 14, height: 14, class: "decompositions-icon" }
                            "{status}"
                        }
                        if recording_active() {
                            span {
                                id: "recording-indicator",
                                class: "decompositions-recording",
                                title: "Recording the embedding animation",
                                "aria-label": "Recording the embedding animation",
                                Icon { icon: FaCircleStop, width: 14, height: 14, class: "decompositions-icon" }
                                "REC"
                            }
                        } else if recorded_url.read().is_some() {
                            button {
                                id: "download-video",
                                class: "decompositions-download",
                                title: "Download the recorded video",
                                "aria-label": "Download the recorded video",
                                onclick: download_video,
                                Icon { icon: FaDownload, width: 14, height: 14, class: "decompositions-icon" }
                                "Download video"
                            }
                        }
                        if dataset.read().is_some() {
                            button {
                                id: "clear",
                                class: "decompositions-clear",
                                title: HELP_CLEAR,
                                "aria-label": HELP_CLEAR,
                                onclick: clear,
                                Icon { icon: FaXmark, width: 16, height: 16, class: "decompositions-icon" }
                            }
                        }
                    }
                }
                if let Some((indeterminate, fraction)) = progress() {
                    div { class: "decompositions-progress",
                        div {
                            class: if indeterminate { "decompositions-progress-fill decompositions-progress-fill--indeterminate" } else { "decompositions-progress-fill" },
                            style: if indeterminate { String::new() } else { format!("width: {}%;", fraction * 100.0) },
                        }
                    }
                }
                div {
                    class: "decompositions-plot-area",
                    ondragover: move |evt| {
                        if drop_enabled {
                            // The drop event only fires when dragover's default
                            // is prevented.
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
                                let extension = name
                                    .rsplit('.')
                                    .next()
                                    .unwrap_or("")
                                    .to_ascii_lowercase();
                                if !zone.allows(&extension) {
                                    let mut dataset = dataset;
                                    let mut ingest_error = ingest_error;
                                    dataset.set(None);
                                    ingest_error
                                        .set(Some(format!("unsupported file type .{extension}")));
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
                    ScatterPlot {
                        embedding,
                        colors: Some(colors.into()),
                        markers: Some(markers.into()),
                        draggable: plot_draggable,
                        on_point_moved,
                        on_drag_start,
                        on_drag_end,
                        pixel_ratio,
                    }
                    if (drop_enabled || has_examples) && embedding.read().is_none() {
                        div {
                            id: "dropzone",
                            class: if dragging_over() { "decompositions-empty decompositions-empty--over" } else { "decompositions-empty" },
                            if busy() {
                                // The moment loading starts the prompt and example
                                // buttons give way to a spinner, so the click has
                                // an immediate, unmistakable effect.
                                div { id: "loading", class: "decompositions-loading",
                                    Icon { icon: FaSpinner, width: 32, height: 32, class: "decompositions-icon decompositions-spinner" }
                                    span { "Loading the dataset" }
                                }
                            }
                            if drop_enabled && !busy() {
                                label { r#for: "file-input", class: "decompositions-droplabel",
                                    Icon { icon: FaFileArrowUp, width: 32, height: 32, class: "decompositions-icon" }
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
                                                    Ok(bytes) => {
                                                        load(file.name(), bytes.to_vec(), None)
                                                    }
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
                            if has_examples && !busy() {
                                div { class: "decompositions-examples",
                                    for (index, example) in examples.iter().enumerate() {
                                        button {
                                            id: "load-example-{index}",
                                            class: "decompositions-example",
                                            title: "Load the {example.name} example dataset.",
                                            "aria-label": "Load the {example.name} example dataset.",
                                            onclick: {
                                                let url = example.url.clone();
                                                let load = load.clone();
                                                move |_| {
                                                    let url = url.clone();
                                                    let load = load.clone();
                                                    // Flip to loading right away, before the fetch,
                                                    // so the buttons give way to the spinner the
                                                    // instant the click lands.
                                                    let mut status = status;
                                                    status.set(String::from("loading"));
                                                    async move {
                                                        let fetched =
                                                            match gloo_net::http::Request::get(&url)
                                                                .send()
                                                                .await
                                                            {
                                                                Ok(response) => {
                                                                    response.binary().await
                                                                }
                                                                Err(error) => Err(error),
                                                            };
                                                        match fetched {
                                                            Ok(bytes) => {
                                                                // Clicking a dataset parses it off
                                                                // the main thread and runs it right
                                                                // away. `load` orphans any run in
                                                                // flight onto a fresh worker so the
                                                                // new one starts immediately.
                                                                load(
                                                                    url,
                                                                    bytes,
                                                                    Some(build_method()),
                                                                );
                                                            }
                                                            Err(error) => {
                                                                let mut status = status;
                                                                let mut ingest_error = ingest_error;
                                                                status.set(String::from("idle"));
                                                                ingest_error
                                                                    .set(Some(error.to_string()));
                                                            }
                                                        }
                                                    }
                                                }
                                            },
                                            "{example.name}"
                                        }
                                    }
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
                    if let Some(active) = coloring_result.read().as_ref() {
                        div { id: "legend", class: "decompositions-legend",
                            for entry in active.legend.iter().take(MAX_LEGEND_ENTRIES) {
                                span { class: "decompositions-legend-entry",
                                    span {
                                        class: "decompositions-legend-swatch",
                                        style: "color: {entry.color};",
                                        "{entry.marker.glyph()}"
                                    }
                                    "{entry.label}"
                                }
                            }
                            if active.legend.len() > MAX_LEGEND_ENTRIES {
                                span { "(+{active.legend.len() - MAX_LEGEND_ENTRIES} more)" }
                            }
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
