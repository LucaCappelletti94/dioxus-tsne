//! The gloo worker hosting the decomposition computations.

use gloo_worker::{HandlerId, Worker, WorkerScope};
use send_wrapper::SendWrapper;

use crate::compute::{TsneCache, decompose_cached};
use crate::ingest::parse_dataset;
use crate::messages::{DecompositionMethod, WorkerRequest, WorkerResponse};

/// Worker running the decompositions off the main thread.
///
/// Consumers register this type in their worker binary, see the crate level
/// documentation.
#[derive(Default)]
pub struct DecompositionWorker {
    /// Affinity graph of the last t-SNE run, reused on a warm-start
    /// continuation so Continue skips the neighbor search. A new dataset
    /// respawns the worker, resetting this to empty.
    cache: TsneCache,
}

impl Worker for DecompositionWorker {
    type Message = ();
    type Input = WorkerRequest;
    type Output = WorkerResponse;

    fn create(_scope: &WorkerScope<Self>) -> Self {
        DecompositionWorker::default()
    }

    fn update(&mut self, _scope: &WorkerScope<Self>, _msg: Self::Message) {}

    fn received(&mut self, scope: &WorkerScope<Self>, input: Self::Input, id: HandlerId) {
        match input {
            WorkerRequest::Load { name, bytes, run } => match parse_dataset(&name, &bytes) {
                Ok(dataset) => {
                    // Hand the parsed dataset back so the UI can color it and
                    // re-run it later, then run here on the worker's own copy.
                    let data = dataset.data.clone();
                    let n_samples = dataset.n_samples;
                    let n_features = dataset.n_features;
                    scope.respond(id, WorkerResponse::Loaded { dataset });
                    if let Some(method) = run {
                        run_decomposition(
                            scope,
                            id,
                            &data,
                            n_samples,
                            n_features,
                            &method,
                            &mut self.cache,
                        );
                    }
                }
                Err(error) => scope.respond(
                    id,
                    WorkerResponse::LoadError {
                        message: error.to_string(),
                    },
                ),
            },
            WorkerRequest::Decompose {
                data,
                n_samples,
                n_features,
                method,
            } => {
                run_decomposition(
                    scope,
                    id,
                    &data,
                    n_samples,
                    n_features,
                    &method,
                    &mut self.cache,
                );
            }
        }
    }
}

/// Runs `decompose`, streaming snapshots and the final response back to `id`.
fn run_decomposition(
    scope: &WorkerScope<DecompositionWorker>,
    id: HandlerId,
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    method: &DecompositionMethod,
    cache: &mut TsneCache,
) {
    // Wall clock start, so each snapshot and the final response can report how
    // long the run has taken. `Date::now` reads the JS clock available in the
    // worker. The `f64` is Copy, so the move closure keeps `start` usable after.
    let start = js_sys::Date::now();
    // Size of rayon's global pool. In the threaded worker build this is the
    // wasm-bindgen-rayon pool size (the core count), otherwise 1. Reported back
    // so the UI can show whether the parallel path is actually active.
    let threads = rayon::current_num_threads();

    // The epoch callback requires Send + Sync captures, while the scope holds
    // JS values. The worker is single threaded, so the wrapper is sound: the
    // callback runs on this very thread.
    let snapshot_scope = SendWrapper::new(scope.clone());
    let outcome = decompose_cached(
        data,
        n_samples,
        n_features,
        method,
        cache,
        move |epoch, embedding| {
            snapshot_scope.respond(
                id,
                WorkerResponse::Snapshot {
                    epoch,
                    embedding: embedding.to_vec(),
                    elapsed_ms: js_sys::Date::now() - start,
                    threads,
                },
            );
        },
    );

    let elapsed_ms = js_sys::Date::now() - start;
    let response = match outcome {
        Ok(output) => WorkerResponse::Done {
            embedding: output.embedding,
            explained_variance_ratio: output.explained_variance_ratio,
            elapsed_ms,
            threads,
        },
        Err(message) => WorkerResponse::Error { message },
    };
    scope.respond(id, response);
}
