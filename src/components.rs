//! The reusable Dioxus components.

use std::rc::Rc;

use dioxus::prelude::*;
use gloo_worker::Spawnable;

use crate::color::{Coloring, colorize};
use crate::ingest::{Dataset, parse_dataset};
use crate::messages::{DecompositionMethod, TsneParams, WorkerRequest, WorkerResponse};
use crate::plot::ScatterPlot;
use crate::worker::DecompositionWorker;

/// Longest legend rendered before truncation.
const MAX_LEGEND_ENTRIES: usize = 20;

/// The main decomposition UI: load a tabular file, run a decomposition in the
/// worker, follow its progress on the animated scatter plot and colorize the
/// points by label columns, pasted values or a dropped single column file.
///
/// # Props
///
/// * `worker_url` - URL of the wasm-bindgen `--target web` JS output of the
///   worker binary registering [`DecompositionWorker`], see the crate level
///   documentation.
/// * `example_url` - optional URL of an example dataset (any supported
///   format). When set, a load button offers it so visitors can try the
///   explorer with a single click.
#[component]
pub fn DecompositionExplorer(
    worker_url: String,
    #[props(default)] example_url: Option<String>,
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
        div {
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
            if let Some(url) = example_url {
                button {
                    id: "load-example",
                    onclick: move |_| {
                        let url = url.clone();
                        async move {
                            let fetched = match gloo_net::http::Request::get(&url).send().await {
                                Ok(response) => response.binary().await,
                                Err(error) => Err(error),
                            };
                            match fetched {
                                Ok(bytes) => match parse_dataset(&url, &bytes) {
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
                        }
                    },
                    "Load example dataset"
                }
            }
            if let Some(parsed) = dataset.read().as_ref() {
                p { id: "dataset-summary",
                    "{parsed.n_samples} samples x {parsed.n_features} features"
                    if !parsed.label_columns.is_empty() {
                        ", label columns: "
                        {parsed.label_columns.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ")}
                    }
                }
            }
            if let Some(error) = ingest_error.read().as_ref() {
                p { id: "ingest-error", color: "red", "{error}" }
            }
            div {
                label { r#for: "method", "Method: " }
                select {
                    id: "method",
                    onchange: move |evt| method.set(evt.value()),
                    option { value: "tsne", selected: true, "t-SNE" }
                    option { value: "pca", "PCA" }
                }
                if method.read().as_str() == "tsne" {
                    label { r#for: "perplexity", " Perplexity: " }
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
                    label { r#for: "theta", " Theta: " }
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
                    label { r#for: "epochs", " Epochs: " }
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
                    label { r#for: "learning-rate", " Learning rate: " }
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
                    label { r#for: "pca-dims", " PCA dimensions: " }
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
                button {
                    id: "run",
                    // The worker queues messages, a run during a run would
                    // silently execute afterwards, so the button is gated.
                    disabled: dataset.read().is_none() || busy(),
                    onclick: run,
                    "Run"
                }
            }
            div {
                label { r#for: "color-source", "Color by: " }
                select {
                    id: "color-source",
                    onchange: move |evt| color_source.set(evt.value()),
                    option { value: "none", selected: true, "none" }
                    if let Some(parsed) = dataset.read().as_ref() {
                        for column in parsed.label_columns.iter() {
                            option { value: "column:{column.name}", "{column.name}" }
                        }
                    }
                    option { value: "pasted", "pasted values" }
                }
                if color_source.read().as_str() == "pasted" {
                    textarea {
                        id: "pasted-labels",
                        rows: "4",
                        placeholder: "one label or score per line",
                        oninput: move |evt| pasted_labels.set(evt.value()),
                    }
                    label { r#for: "labels-file", " or drop a single column file: " }
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
                if let Err(error) = coloring.read().as_ref() {
                    p { id: "color-error", color: "red", "{error}" }
                }
            }
            if let Ok(Some(active)) = coloring.read().as_ref() {
                div { id: "legend",
                    for entry in active.legend.iter().take(MAX_LEGEND_ENTRIES) {
                        span { style: "margin-right: 0.8em;",
                            span {
                                style: "display: inline-block; width: 0.8em; height: 0.8em; margin-right: 0.3em; border-radius: 50%; background: {entry.color};",
                            }
                            "{entry.label}"
                        }
                    }
                    if active.legend.len() > MAX_LEGEND_ENTRIES {
                        span { "(+{active.legend.len() - MAX_LEGEND_ENTRIES} more)" }
                    }
                }
            }
            p { id: "status", "{status}" }
            ScatterPlot { embedding, colors: Some(colors.into()) }
        }
    }
}
