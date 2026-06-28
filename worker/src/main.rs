//! Entry point of the dedicated worker wasm bundle. This is exactly the worker
//! binary that consumers of dioxus-decompositions add to their own workspace.

use dioxus_decompositions::DecompositionWorker;
use gloo_worker::Registrable;

#[cfg(feature = "threads")]
use wasm_bindgen::prelude::wasm_bindgen;

// Re-exporting keeps wasm-bindgen-rayon's `initThreadPool` export alive in the
// final wasm so the loader can prepare the pool before any compute runs.
#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Registers the gloo worker so it starts handling compute messages.
///
/// In the threaded build the loader calls this only after `initThreadPool` has
/// built the rayon pool. Registering earlier would let a queued compute message
/// touch rayon first, which lazily initializes the global pool to a single
/// thread, after which wasm-bindgen-rayon's real `build_global` fails and the
/// worker is stuck single threaded.
#[cfg(feature = "threads")]
#[wasm_bindgen]
pub fn register_worker() {
    DecompositionWorker::registrar().register();
}

fn main() {
    console_error_panic_hook::set_once();

    // The threaded build defers registration to the loader, see register_worker.
    // The plain build has no pool to wait for, so it registers right away.
    #[cfg(not(feature = "threads"))]
    DecompositionWorker::registrar().register();
}
