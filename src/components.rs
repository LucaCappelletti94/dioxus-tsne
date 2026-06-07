//! The reusable Dioxus components.

use std::rc::Rc;

use dioxus::prelude::*;
use gloo_worker::Spawnable;

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
