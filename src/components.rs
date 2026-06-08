//! The reusable Dioxus components.

use std::rc::Rc;

use dioxus::prelude::*;
use gloo_worker::Spawnable;
use wasm_bindgen::JsCast;

use crate::color::{Coloring, colorize};
use crate::ingest::{Dataset, parse_dataset};
use crate::messages::{DecompositionMethod, TsneParams, WorkerRequest, WorkerResponse};
use crate::plot::ScatterPlot;
use crate::worker::DecompositionWorker;

/// Longest legend rendered before truncation.
const MAX_LEGEND_ENTRIES: usize = 20;

/// A bundled example dataset offered by the explorer through a load button.
#[derive(Debug, Clone, PartialEq)]
pub struct ExampleDataset {
    /// Human readable name shown on the button.
    pub name: String,
    /// URL the file is served from, any supported format.
    pub url: String,
}

/// The main decomposition UI: load a tabular file, run a decomposition in the
/// worker, follow its progress on the animated scatter plot and colorize the
/// points by label columns, pasted values or a dropped single column file.
///
/// # Props
///
/// * `worker_url` - URL of the wasm-bindgen `--target web` JS output of the
///   worker binary registering [`DecompositionWorker`], see the crate level
///   documentation.
/// * `examples` - bundled example datasets, one load button each, so visitors
///   can try the explorer with a single click.
/// * `styled` - whether to inject the default stylesheet (also exported as
///   [`crate::DEFAULT_STYLE`]), on by default. Disable it to bring your own
///   rules for the `decompositions-*` class names.
#[component]
pub fn DecompositionExplorer(
    worker_url: String,
    #[props(default)] examples: Vec<ExampleDataset>,
    #[props(default = true)] styled: bool,
) -> Element {
    let mut dataset = use_signal(|| None::<Dataset>);
    let mut ingest_error = use_signal(|| None::<String>);
    let status = use_signal(|| String::from("idle"));
    let embedding = use_signal(|| None::<Vec<f32>>);
    let mut method = use_signal(|| String::from("tsne"));
    let defaults = TsneParams::default();
    let mut pca_dims = use_signal(|| defaults.pca_dims);
    let mut perplexity = use_signal(|| defaults.perplexity);
    let mut theta = use_signal(|| defaults.theta);
    let mut epochs = use_signal(|| defaults.epochs);
    let mut learning_rate = use_signal(|| defaults.learning_rate);
    let mut color_source = use_signal(|| String::from("none"));
    let mut pasted_labels = use_signal(String::new);
    let mut color_select = use_signal(|| None::<web_sys::HtmlSelectElement>);

    // The browser snaps a select back to its first option when the bound
    // value is applied before the matching option exists, which happens when
    // loading a dataset sets both the label column options and the source in
    // one update. Re-syncing the DOM value after renders keeps the displayed
    // value honest.
    use_effect(move || {
        let source = color_source.read().clone();
        // Subscribe to dataset changes too: they alter the option list.
        let _ = dataset.read();
        if let Some(select) = color_select() {
            select.set_value(&source);
        }
    });

    // The active coloring, or an error when the value count does not match
    // the dataset. Recomputed on source, paste or dataset changes: recoloring
    // is a pure redraw, the embedding is never recomputed.
    let coloring = use_memo(move || -> Result<Option<Coloring>, String> {
        let source = color_source.read().clone();
        let values: Vec<String> = match source.as_str() {
            "none" => return Ok(None),
            "pasted" => {
                let text = pasted_labels.read().clone();
                if text.trim().is_empty() {
                    return Ok(None);
                }
                text.lines().map(|line| line.trim().to_owned()).collect()
            }
            column => {
                let guard = dataset.read();
                let Some(parsed) = guard.as_ref() else {
                    return Ok(None);
                };
                let Some(labels) = parsed
                    .label_columns
                    .iter()
                    .find(|c| c.name == column.strip_prefix("column:").unwrap_or(column))
                else {
                    return Ok(None);
                };
                labels.values.clone()
            }
        };
        let n_samples = dataset.read().as_ref().map_or(0, |d| d.n_samples);
        if values.len() != n_samples {
            return Err(format!(
                "{} color values for {n_samples} samples",
                values.len()
            ));
        }
        Ok(Some(colorize(&values)))
    });
    let colors = use_memo(move || {
        coloring
            .read()
            .as_ref()
            .ok()
            .and_then(|c| c.as_ref())
            .map(|c| c.colors.clone())
    });
    let markers = use_memo(move || {
        coloring
            .read()
            .as_ref()
            .ok()
            .and_then(|c| c.as_ref())
            .map(|c| c.markers.clone())
    });

    // A decomposition is in flight while the status reports progress.
    let busy = use_memo(move || {
        let status = status.read();
        status.as_str() == "running" || status.starts_with("epoch ")
    });

    // The bridge owns the worker and must live across renders, so it is
    // created once. The callback writes the worker replies into the signals.
    // Signals are Copy, the rebindings make the closure a plain Fn.
    let bridge = use_hook(|| {
        Rc::new(
            DecompositionWorker::spawner()
                // The URL points at a loader module that initializes the
                // wasm-bindgen output itself, see the crate documentation.
                .with_loader(true)
                .callback(move |response| {
                    let mut status = status;
                    let mut embedding = embedding;
                    match response {
                        WorkerResponse::Snapshot {
                            epoch,
                            embedding: snapshot,
                        } => {
                            status.set(format!("epoch {epoch}"));
                            embedding.set(Some(snapshot));
                        }
                        WorkerResponse::Done {
                            embedding: done,
                            explained_variance_ratio,
                        } => {
                            status.set(match explained_variance_ratio {
                                Some(ratios) => format!(
                                    "done, explained variance: {}",
                                    ratios
                                        .iter()
                                        .map(|r| format!("{:.1}%", r * 100.0))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ),
                                None => String::from("done"),
                            });
                            embedding.set(Some(done));
                        }
                        WorkerResponse::Error { message } => {
                            status.set(format!("error: {message}"));
                        }
                    }
                })
                .spawn(&worker_url),
        )
    });

    let run = {
        let bridge = bridge.clone();
        move |_| {
            let Some(parsed) = dataset.read().clone() else {
                return;
            };
            let selected = if method.read().as_str() == "pca" {
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
            };
            let mut status = status;
            status.set(String::from("running"));
            bridge.send(WorkerRequest::Decompose {
                data: parsed.data,
                n_samples: parsed.n_samples,
                n_features: parsed.n_features,
                method: selected,
            });
        }
    };

    rsx! {
        div { class: "decompositions-explorer",
            if styled {
                style { {crate::DEFAULT_STYLE} }
            }
            section {
                label { r#for: "file-input", "Dataset"
                    input {
                        id: "file-input",
                        r#type: "file",
                        accept: ".csv,.tsv,.parquet",
                        onchange: move |evt| async move {
                            let Some(file) = evt.files().into_iter().next() else {
                                return;
                            };
                            match file.read_bytes().await {
                                Ok(bytes) => match parse_dataset(&file.name(), &bytes) {
                                    Ok(parsed) => {
                                        ingest_error.set(None);
                                        dataset.set(Some(parsed));
                                    }
                                    Err(error) => {
                                        dataset.set(None);
                                        ingest_error.set(Some(error.to_string()));
                                    }
                                },
                                Err(error) => {
                                    dataset.set(None);
                                    ingest_error.set(Some(error.to_string()));
                                }
                            }
                        },
                    }
                }
                for (index, example) in examples.iter().enumerate() {
                    button {
                        id: "load-example-{index}",
                        onclick: {
                            let url = example.url.clone();
                            move |_| {
                                let url = url.clone();
                                async move {
                                    let fetched =
                                        match gloo_net::http::Request::get(&url).send().await {
                                            Ok(response) => response.binary().await,
                                            Err(error) => Err(error),
                                        };
                                    match fetched {
                                        Ok(bytes) => match parse_dataset(&url, &bytes) {
                                            Ok(parsed) => {
                                                ingest_error.set(None);
                                                // Examples ship curated label
                                                // columns, color by the first
                                                // one right away so the
                                                // classes show without extra
                                                // clicks.
                                                if let Some(labels) =
                                                    parsed.label_columns.first()
                                                {
                                                    color_source
                                                        .set(format!("column:{}", labels.name));
                                                }
                                                dataset.set(Some(parsed));
                                            }
                                            Err(error) => {
                                                dataset.set(None);
                                                ingest_error.set(Some(error.to_string()));
                                            }
                                        },
                                        Err(error) => {
                                            dataset.set(None);
                                            ingest_error.set(Some(error.to_string()));
                                        }
                                    }
                                }
                            }
                        },
                        "Load {example.name}"
                    }
                }
                if let Some(parsed) = dataset.read().as_ref() {
                    p { id: "dataset-summary", class: "decompositions-summary",
                        "{parsed.n_samples} samples x {parsed.n_features} features"
                        if !parsed.label_columns.is_empty() {
                            ", label columns: "
                            {parsed.label_columns.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ")}
                        }
                    }
                }
                if let Some(error) = ingest_error.read().as_ref() {
                    p { id: "ingest-error", class: "decompositions-error", "{error}" }
                }
            }
            section {
                label { r#for: "method", "Method"
                    select {
                        id: "method",
                        value: "{method}",
                        onchange: move |evt| method.set(evt.value()),
                        option { value: "tsne", "t-SNE" }
                        option { value: "pca", "PCA" }
                    }
                }
                if method.read().as_str() == "tsne" {
                    label { r#for: "perplexity", "Perplexity"
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
                    label { r#for: "theta", "Theta"
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
                    label { r#for: "epochs", "Epochs"
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
                    label { r#for: "learning-rate", "Learning rate"
                        input {
                            id: "learning-rate",
                            r#type: "number",
                            min: "1",
                            step: "10",
                            value: "{learning_rate}",
                            onchange: move |evt| {
                                if let Ok(value) = evt.value().parse::<f32>() {
                                    learning_rate.set(value.max(1.0));
                                }
                            },
                        }
                    }
                    label { r#for: "pca-dims", "PCA dimensions"
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
                }
                button {
                    id: "run",
                    // The worker queues messages, a run during a run would
                    // silently execute afterwards, so the button is gated.
                    disabled: dataset.read().is_none() || busy(),
                    onclick: run,
                    "Run"
                }
                p { id: "status", class: "decompositions-status", "{status}" }
            }
            section {
                label { r#for: "color-source", "Color by"
                    select {
                        id: "color-source",
                        value: "{color_source}",
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
                        option { value: "none", "none" }
                        if let Some(parsed) = dataset.read().as_ref() {
                            for column in parsed.label_columns.iter() {
                                option { value: "column:{column.name}", "{column.name}" }
                            }
                        }
                        option { value: "pasted", "pasted values" }
                    }
                }
                if color_source.read().as_str() == "pasted" {
                    label { r#for: "labels-file", "or a single column file"
                        input {
                            id: "labels-file",
                            r#type: "file",
                            accept: ".csv,.tsv,.txt",
                            onchange: move |evt| async move {
                                let Some(file) = evt.files().into_iter().next() else {
                                    return;
                                };
                                if let Ok(text) = file.read_string().await {
                                    pasted_labels.set(text);
                                }
                            },
                        }
                    }
                    textarea {
                        id: "pasted-labels",
                        rows: "4",
                        placeholder: "one label or score per line",
                        oninput: move |evt| pasted_labels.set(evt.value()),
                    }
                }
                if let Err(error) = coloring.read().as_ref() {
                    p { id: "color-error", class: "decompositions-error", "{error}" }
                }
                if let Ok(Some(active)) = coloring.read().as_ref() {
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
            ScatterPlot {
                embedding,
                colors: Some(colors.into()),
                markers: Some(markers.into()),
            }
        }
    }
}
