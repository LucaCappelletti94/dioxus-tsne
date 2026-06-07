//! Entry point of the dedicated worker wasm bundle. This is exactly the worker
//! binary that consumers of dioxus-decompositions add to their own workspace.

use dioxus_decompositions::DecompositionWorker;
use gloo_worker::Registrable;

fn main() {
    console_error_panic_hook::set_once();

    DecompositionWorker::registrar().register();
}
