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
//! use dioxus_tsne::DecompositionWorker;
//! use gloo_worker::Registrable;
//!
//! fn main() {
//!     DecompositionWorker::registrar().register();
//! }
//! ```
//!
//! 2. Builds it to wasm, runs wasm-bindgen (`--target web` semantics) on it
//!    and serves the output next to a two line loader module that initializes
//!    it with an explicit wasm URL. The `app` crate of this repository
//!    automates all of this with a `build.rs` using the
//!    `wasm-bindgen-cli-support` library, writing into
//!    `public/dioxus-decompositions/` (served at the site root as
//!    [`DEFAULT_WORKER_URL`]). Copy it. The explicit wasm URL in the loader
//!    matters: dx minifies served JS and strips the `import.meta.url` based
//!    default path inside the wasm-bindgen glue.
//!
//! 3. Builds the UI with the [`Decomposition`] fluent builder, opting into the
//!    features it wants:
//!
//! ```ignore
//! use dioxus_tsne::Decomposition;
//!
//! Decomposition::new()
//!     .drop_zone()
//!     .controls()
//!     .draggable_points()
//!     .render()
//! # ;
//! ```
//!
//! The worker is loaded from [`DEFAULT_WORKER_URL`] unless overridden with
//! [`Decomposition::worker_url`] (for a custom `build.rs` output path or a site
//! served under a subpath).

mod color;
mod components;
mod compute;
mod ingest;
mod messages;
mod pca;
mod plot;
mod worker;

/// The default stylesheet of the components, injected by [`Decomposition`]
/// unless [`Decomposition::styled`] is set to false. Consumers styling the
/// `decompositions-*` class names themselves can serve their own rules instead.
pub const DEFAULT_STYLE: &str = include_str!("style.css");

pub use color::{ColorScale, Coloring, LegendEntry, Marker, colorize};
pub use components::{DEFAULT_WORKER_URL, Decomposition, DropZone, ExampleDataset, ExampleIcon};
pub use compute::{DecomposeOutput, decompose};
pub use ingest::{Dataset, IngestError, LabelColumn, parse_dataset};
pub use messages::{DecompositionMethod, TsneParams, WorkerRequest, WorkerResponse};
pub use pca::{PcaResult, pca};
pub use plot::ScatterPlot;
pub use worker::DecompositionWorker;
