//! # dioxus-decompositions
//!
//! Reusable [Dioxus](https://dioxuslabs.com) components to run and visualize
//! decompositions like t-SNE, PCA and more, computed off the main thread in a
//! dedicated web worker.
//!
//! ## Architecture
//!
//! Browsers require web workers to be separate scripts, so the compute part of
//! this crate cannot be hidden inside a component: the consuming app builds a
//! tiny worker binary and tells the components where it is served from through
//! the `worker_url` prop.
//!
//! Concretely, a consumer:
//!
//! 1. Adds a worker binary crate registering [`DecompositionWorker`]:
//!
//! ```ignore
//! use dioxus_decompositions::DecompositionWorker;
//! use gloo_worker::Registrable;
//!
//! fn main() {
//!     DecompositionWorker::registrar().register();
//! }
//! ```
//!
//! 2. Builds it to wasm with `wasm-bindgen --target web` and serves the output
//!    (with Dioxus, register it as a folder asset, see the `app` crate of this
//!    repository for a complete example).
//!
//! 3. Passes the URL of the wasm-bindgen JS output to the components via the
//!    `worker_url` prop.

mod color;
mod components;
mod compute;
mod ingest;
mod messages;
mod pca;
mod plot;
mod worker;

pub use color::{ColorScale, Coloring, LegendEntry, colorize};
pub use components::DecompositionExplorer;
pub use compute::{DecomposeOutput, decompose};
pub use ingest::{Dataset, IngestError, LabelColumn, parse_dataset};
pub use messages::{DecompositionMethod, TsneParams, WorkerRequest, WorkerResponse};
pub use pca::{PcaResult, pca};
pub use plot::ScatterPlot;
pub use worker::DecompositionWorker;
