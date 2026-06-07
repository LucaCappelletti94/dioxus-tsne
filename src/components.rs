//! The reusable Dioxus components.

use std::rc::Rc;

use dioxus::prelude::*;
use gloo_worker::Spawnable;

use crate::ingest::{Dataset, parse_dataset};
use crate::messages::{DecompositionMethod, TsneParams, WorkerRequest, WorkerResponse};
use crate::plot::ScatterPlot;
use crate::worker::DecompositionWorker;

/// The main decomposition UI: load a tabular file, run a decomposition in the
/// worker and follow its progress. The animated scatter plot and the coloring
/// controls land next.
///
/// # Props
///
/// * `worker_url` - URL of the wasm-bindgen `--target web` JS output of the
///   worker binary registering [`DecompositionWorker`], see the crate level
///   documentation.
#[component]
pub fn DecompositionExplorer(worker_url: String) -> Element {
    let mut dataset = use_signal(|| None::<Dataset>);
    let mut ingest_error = use_signal(|| None::<String>);
    let status = use_signal(|| String::from("idle"));
    let embedding = use_signal(|| None::<Vec<f32>>);
    let mut method = use_signal(|| String::from("tsne"));
    let mut pca_dims = use_signal(|| 50usize);

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
                button {
                    id: "run",
                    disabled: dataset.read().is_none(),
                    onclick: run,
                    "Run"
                }
            }
            p { id: "status", "{status}" }
            ScatterPlot { embedding }
        }
    }
}
