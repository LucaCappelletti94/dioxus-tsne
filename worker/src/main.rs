//! Entry point of the dedicated worker wasm bundle. This is exactly the worker
//! binary that consumers of dioxus-decompositions add to their own workspace.

use dioxus_tsne::DecompositionWorker;
use gloo_worker::Registrable;
use wasm_bindgen::prelude::wasm_bindgen;

// Re-exporting keeps wasm-bindgen-rayon's `initThreadPool` export alive in the
// final wasm so the loader can prepare the pool before any compute runs.
pub use wasm_bindgen_rayon::init_thread_pool;

/// Registers the gloo worker so it starts handling compute messages.
///
/// The loader calls this only after `initThreadPool` has built the rayon pool.
/// Registering earlier would let a queued compute message touch rayon first,
/// which lazily initializes the global pool to a single thread, after which
/// wasm-bindgen-rayon's real `build_global` fails and the worker is stuck
/// single threaded.
#[wasm_bindgen]
pub fn register_worker() {
    DecompositionWorker::registrar().register();
}

fn main() {
    console_error_panic_hook::set_once();
    // Registration is deferred to the loader once the pool is built, see
    // register_worker.
}
