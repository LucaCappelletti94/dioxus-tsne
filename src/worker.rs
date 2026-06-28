//! The gloo worker hosting the decomposition computations.

use gloo_worker::{HandlerId, Worker, WorkerScope};
use send_wrapper::SendWrapper;

use crate::compute::{TsneCache, decompose_cached};
use crate::ingest::parse_dataset;
use crate::messages::{DecompositionMethod, TsnePhase, WorkerRequest, WorkerResponse};

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
    // Size of rayon's global pool: the wasm-bindgen-rayon pool size (the capped
    // core count). Reported back so the UI can show the live pool size. A page
    // that is not cross-origin isolated leaves this at 1, surfacing a broken
    // serve setup.
    let threads = rayon::current_num_threads();

    // A fresh t-SNE run (no warm-start seed) rebuilds the affinity graph, so the
    // first thing it does is the neighbor search. Name that phase up front, the
    // worker then blocks in it until the first epoch streams back. A warm-start
    // continuation reuses the cached graph and skips straight to optimizing.
    // The early-exaggeration boundary is `0` for a warm start (exaggeration is
    // disabled) and the configured duration otherwise.
    let stop_lying = match method {
        DecompositionMethod::Tsne(params) if params.initial_embedding.is_none() => {
            scope.respond(
                id,
                WorkerResponse::Phase {
                    phase: TsnePhase::FindingNeighbors,
                },
            );
            params.early_exaggeration_epochs
        }
        _ => 0,
    };

    // The epoch callback requires Send + Sync captures, while the scope holds JS
    // values. rayon parallelizes the gradient within each epoch, but bhtsne
    // invokes the epoch callback from the thread driving the fit (this one, the
    // worker's message handler), so the wrapper is only ever dereferenced on the
    // thread that created it.
    let snapshot_scope = SendWrapper::new(scope.clone());
    let outcome = decompose_cached(
        data,
        n_samples,
        n_features,
        method,
        cache,
        move |epoch, embedding| {
            let phase = if epoch < stop_lying {
                TsnePhase::EarlyExaggeration
            } else {
                TsnePhase::Optimizing
            };
            snapshot_scope.respond(
                id,
                WorkerResponse::Snapshot {
                    epoch,
                    embedding: embedding.to_vec(),
                    phase,
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
            kl_divergence: output.kl_divergence,
            elapsed_ms,
            threads,
        },
        Err(message) => WorkerResponse::Error { message },
    };
    scope.respond(id, response);
}
