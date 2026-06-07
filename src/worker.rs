//! The gloo worker hosting the decomposition computations.

use gloo_worker::{HandlerId, Worker, WorkerScope};

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
            WorkerRequest::Ping { payload } => {
                scope.respond(
                    id,
                    WorkerResponse::Pong {
                        payload: format!("worker echo: {payload}"),
                    },
                );
            }
        }
    }
}
