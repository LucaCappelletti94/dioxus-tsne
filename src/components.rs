//! The reusable Dioxus components.

use std::rc::Rc;

use dioxus::prelude::*;
use gloo_worker::Spawnable;

use crate::ingest::{Dataset, parse_dataset};
use crate::messages::{WorkerRequest, WorkerResponse};
use crate::worker::DecompositionWorker;

/// The main decomposition UI.
///
/// Currently a skeleton proving the worker channel end to end: it spawns the
/// worker, sends a ping on demand and renders the reply. It will grow into the
/// full explorer (data ingestion, decomposition controls, animated scatter
/// plot, coloring).
///
/// # Props
///
/// * `worker_url` - URL of the wasm-bindgen `--target web` JS output of the
///   worker binary registering [`DecompositionWorker`], see the crate level
///   documentation.
#[component]
pub fn DecompositionExplorer(worker_url: String) -> Element {
    let response = use_signal(|| String::from("no response yet"));
    let mut dataset = use_signal(|| None::<Dataset>);
    let mut ingest_error = use_signal(|| None::<String>);

    // The bridge owns the worker and must live across renders, so it is
    // created once. The callback writes the worker replies into the signal.
    // Signals are Copy, the rebinding makes the closure a plain Fn.
    let bridge = use_hook(|| {
        Rc::new(
            DecompositionWorker::spawner()
                .callback(move |outcome| {
                    let WorkerResponse::Pong { payload } = outcome;
                    let mut response = response;
                    response.set(payload);
                })
                .spawn(&worker_url),
        )
    });

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
            button {
                id: "ping",
                onclick: move |_| {
                    bridge.send(WorkerRequest::Ping {
                        payload: String::from("ping from the UI"),
                    })
                },
                "Ping worker"
            }
            p { id: "response", "{response}" }
        }
    }
}
