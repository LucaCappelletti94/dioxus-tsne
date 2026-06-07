//! Messages exchanged between the UI components and the decomposition worker.

use serde::{Deserialize, Serialize};

/// Request sent from the UI to the decomposition worker.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkerRequest {
    /// Connectivity check, echoed back by the worker. Used by the skeleton UI
    /// while the real decomposition requests are being built.
    Ping {
        /// Payload echoed back verbatim.
        payload: String,
    },
}

/// Response sent from the decomposition worker back to the UI.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkerResponse {
    /// Reply to [`WorkerRequest::Ping`].
    Pong {
        /// The payload of the originating ping.
        payload: String,
    },
}
