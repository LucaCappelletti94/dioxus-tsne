# dioxus-decompositions

Reusable [Dioxus](https://dioxuslabs.com) components to run and visualize decompositions like t-SNE, PCA and more, computed off the main thread in a dedicated web worker, plus a web app built on them where you drop a CSV, TSV or Parquet file and watch the embedding animate as epochs progress.

## Workspace layout

- The root crate is the publishable `dioxus-decompositions` component library.
- `app/` is the web app, built and served with `dx`. It is also the reference consumer of the library.
- `worker/` is the worker binary the app serves, registering the library's `DecompositionWorker`. Consumers of the library add an equivalent three line binary to their own workspace.

## Why consumers need a worker binary

Browsers require web workers to be separate scripts, so the compute part cannot be hidden inside a component. The library ships the worker type and the components, the consuming app builds the worker binary to its own wasm bundle and passes the URL it is served from to the components through the `worker_url` prop. The `app` crate automates this with a `build.rs` that compiles the worker and runs wasm-bindgen (through the `wasm-bindgen-cli-support` library, no external CLI needed) into `public/worker/`, which `dx` serves verbatim at the site root. Copy that build script as the starting point for your own integration.

## Development

Requirements: the `wasm32-unknown-unknown` target and the `dx` CLI (Dioxus 0.7).

```
dx serve -p decompositions-app
```

The worker bundle is built automatically by the app's `build.rs`, including on changes to the worker or library code. Set `DECOMPOSITIONS_SKIP_WORKER_BUILD=1` to skip it for type-check-only tooling.

The app ships two bundled examples loadable with one click, so you can try the full flow without bringing your own file:

- MNIST digits (1000 samples, already reduced to 50 PCA dimensions): run and color by the digit column to see the classic t-SNE digit clusters.
- Cora papers (2708 samples, the raw 1433 binary bag of words features): exercises the in-worker PCA preprocessing down to 50 dimensions before t-SNE, color by the subject column. The features alone separate the seven subjects only partially, which is the honest result without the citation graph.
