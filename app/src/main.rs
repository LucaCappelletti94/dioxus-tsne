//! The dioxus-decompositions web app: drop a CSV, TSV or Parquet file, reduce
//! with PCA, embed with a selectable decomposition and watch the scatter plot
//! animate as epochs progress. Also serves as the reference consumer of the
//! dioxus-decompositions component library.

use dioxus::prelude::*;
use dioxus_decompositions::{DecompositionExplorer, ExampleDataset};

/// Folder asset holding the wasm-bindgen output of the worker bundle, generated
/// by scripts/build-worker.sh before the app build. A folder asset keeps the
/// inner file names, which gloo-worker relies on to derive the .wasm URL from
/// the .js one.
static WORKER_ASSETS: Asset = asset!("/assets/worker");

/// MNIST subsample (1000 digits, PCA-50, snappy Parquet) offered as a one
/// click example dataset.
static MNIST_EXAMPLE: Asset = asset!("/assets/examples/mnist_1k.parquet");

/// Cora citation dataset (2708 papers, 1433 binary bag of words features,
/// 7 subjects), shipped raw so the in-worker PCA preprocessing does the
/// reduction to 50 dimensions.
static CORA_EXAMPLE: Asset = asset!("/assets/examples/cora.parquet");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        h1 { "dioxus-decompositions" }
        DecompositionExplorer {
            worker_url: format!("{WORKER_ASSETS}/decompositions_worker.js"),
            examples: vec![
                ExampleDataset {
                    name: String::from("MNIST digits (1k, PCA-50)"),
                    url: MNIST_EXAMPLE.to_string(),
                },
                ExampleDataset {
                    name: String::from("Cora papers (2.7k, 1433 raw features)"),
                    url: CORA_EXAMPLE.to_string(),
                },
            ],
        }
    }
}
