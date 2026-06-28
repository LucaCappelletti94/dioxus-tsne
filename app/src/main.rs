//! The dioxus-decompositions web app: drop a CSV, TSV or Parquet file, reduce
//! with PCA, embed with a selectable decomposition and watch the scatter plot
//! animate as epochs progress. Also serves as the reference consumer of the
//! dioxus-decompositions component library.

use dioxus::prelude::*;
use dioxus_decompositions::{Decomposition, ExampleDataset};

/// Full MNIST (70000 digits, PCA-20, snappy Parquet) offered as a one click
/// example dataset, a heavy t-SNE workload that exercises the worker pool.
static MNIST_EXAMPLE: Asset = asset!("/assets/examples/mnist.parquet");

/// Full Fashion-MNIST (70000 Zalando clothing images, PCA-20, snappy Parquet),
/// colored by clothing category.
static FASHION_MNIST_EXAMPLE: Asset = asset!("/assets/examples/fashion_mnist.parquet");

/// Cora citation dataset (2708 papers, 7 subjects), reduced to 50 PCA
/// dimensions from the 1433 binary bag of words features.
static CORA_EXAMPLE: Asset = asset!("/assets/examples/cora.parquet");

/// Minimal page shell styling, the component brings its own default theme.
const APP_STYLE: &str = "
body {
    margin: 0;
    padding: 2rem 1.5rem;
    display: flex;
    justify-content: center;
    background: #ffffff;
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
}
main { width: 100%; max-width: 56rem; }
h1 { font-size: 1.35rem; font-weight: 600; color: #1c2733; margin: 0 0 1rem; }
@media (prefers-color-scheme: dark) {
    body { background: #14181d; }
    h1 { color: #e7ecf1; }
}
";

fn main() {
    console_error_panic_hook::set_once();
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        style { {APP_STYLE} }
        main {
        h1 { "dioxus-decompositions" }
        {
            Decomposition::new()
                .drop_zone()
                .examples(vec![
                    ExampleDataset {
                        name: String::from("MNIST 70k"),
                        url: MNIST_EXAMPLE.to_string(),
                    },
                    ExampleDataset {
                        name: String::from("Fashion-MNIST 70k"),
                        url: FASHION_MNIST_EXAMPLE.to_string(),
                    },
                    ExampleDataset {
                        name: String::from("Cora 2.7k"),
                        url: CORA_EXAMPLE.to_string(),
                    },
                ])
                .controls()
                .draggable_points()
                .render()
        }
        }
    }
}
