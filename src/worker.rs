//! The gloo worker hosting the decomposition computations.

use gloo_worker::{HandlerId, Worker, WorkerScope};
use send_wrapper::SendWrapper;

use crate::compute::decompose;
use crate::messages::{WorkerRequest, WorkerResponse};

/// Worker running the decompositions off the main thread.
///
/// Consumers register this type in their worker binary, see the crate level
/// documentation.
pub struct DecompositionWorker;

impl Worker for DecompositionWorker {
    type Message = ();
    type Input = WorkerRequest;
    type Output = WorkerResponse;

    fn create(_scope: &WorkerScope<Self>) -> Self {
        DecompositionWorker
    }

    fn update(&mut self, _scope: &WorkerScope<Self>, _msg: Self::Message) {}

    fn received(&mut self, scope: &WorkerScope<Self>, input: Self::Input, id: HandlerId) {
        match input {
            WorkerRequest::Decompose {
                data,
                n_samples,
                n_features,
                method,
            } => {
                // The epoch callback requires Send + Sync captures, while the
                // scope holds JS values. The worker is single threaded, so the
                // wrapper is sound: the callback runs on this very thread.
                let snapshot_scope = SendWrapper::new(scope.clone());
                let outcome = decompose(
                    &data,
                    n_samples,
                    n_features,
                    &method,
                    move |epoch, embedding| {
                        snapshot_scope.respond(
                            id,
                            WorkerResponse::Snapshot {
                                epoch,
                                embedding: embedding.to_vec(),
                            },
                        );
                    },
                );

                let response = match outcome {
                    Ok(output) => WorkerResponse::Done {
                        embedding: output.embedding,
                        explained_variance_ratio: output.explained_variance_ratio,
                    },
                    Err(message) => WorkerResponse::Error { message },
                };
                scope.respond(id, response);
            }
        }
    }
}
