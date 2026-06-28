# dioxus-decompositions

Reusable [Dioxus](https://dioxuslabs.com) components to run and visualize t-SNE, PCA and similar decompositions in a web worker. Drop a CSV, TSV or Parquet file and watch the embedding animate.

## Usage

`Decomposition` is a fluent builder over a single panel (plot, toolbar, drag and drop loader, color legend). Everything past the bare plot is opt in.

```rust
use dioxus_decompositions::{Decomposition, ExampleDataset};

Decomposition::new()
    .drop_zone()        // drop a file, or click to browse
    .examples(vec![/* ExampleDataset { name, url } */])
    .controls()         // method, Run/Continue, color by, settings, status
    .draggable_points() // grab points to steer a run
    .render()
```

Or start from an in-memory dataset instead of a file:

```rust
Decomposition::new()
    .dataset(features, n_samples, n_features) // row major
    .labels("cluster", labels)                // colored with a legend
    .render()
```

## Worker

The compute runs in a web worker, which browsers require to be a separate script. Your app builds a tiny binary that registers `DecompositionWorker` and serves it. `Decomposition` loads it from `DEFAULT_WORKER_URL` (`/dioxus-decompositions/loader.js`), overridable with `.worker_url(...)`. Copy `app/build.rs`, which automates the build.

See [CONTRIBUTING.md](CONTRIBUTING.md) to run the reference app in `app/`.
