# dioxus-decompositions

Reusable [Dioxus](https://dioxuslabs.com) components to run and visualize decompositions like t-SNE, PCA and more, computed off the main thread in a dedicated web worker, plus a web app built on them where you drop a CSV, TSV or Parquet file and watch the embedding animate as epochs progress.

## Workspace layout

- The root crate is the publishable `dioxus-decompositions` component library.
- `app/` is the web app, built and served with `dx`. It is also the reference consumer of the library.
- `worker/` is the worker binary the app serves, registering the library's `DecompositionWorker`. Consumers of the library add an equivalent three line binary to their own workspace.

## Why consumers need a worker binary

Browsers require web workers to be separate scripts, so the compute part cannot be hidden inside a component. The library ships the worker type and the components, the consuming app builds the worker binary to its own wasm bundle and passes the URL it is served from to the components through the `worker_url` prop. See the crate documentation for the exact steps and the `app` crate for a complete example.

## Development

Requirements: the `wasm32-unknown-unknown` target, the `dx` CLI (Dioxus 0.7) and the `wasm-bindgen` CLI at the exact version pinned in the workspace `Cargo.toml`.

```
./scripts/build-worker.sh        # build the worker bundle into app/assets/worker/
dx serve -p decompositions-app   # build and serve the app
```

The worker bundle is not rebuilt by `dx serve`, rerun the script after changing the worker or the library compute code. The worker bundle must exist before the app compiles, since the app registers it as a folder asset.

The app ships a bundled example, a 1000 digit MNIST subsample reduced to 50 PCA dimensions: click "Load example dataset", then "Run", then color by the digit column to see the classic t-SNE digit clusters without bringing your own file.
