# Contributing

## Layout

- root: the publishable `dioxus-decompositions` library.
- `app/`: the web app and reference consumer.
- `worker/`: the worker binary registering `DecompositionWorker`. Consumers add an equivalent three line binary.

The app's `build.rs` compiles the worker and runs wasm-bindgen (via `wasm-bindgen-cli-support`, no external CLI) into `app/public/dioxus-decompositions/`, served at the site root.

## Requirements

`wasm32-unknown-unknown`, the `dx` CLI (Dioxus 0.7), and a nightly toolchain for the threaded build.

## Commands

```sh
dx serve -p decompositions-app                                    # debug serve
dx serve -p decompositions-app --release --port 9595 --debug-symbols false  # release serve
dx bundle -p decompositions-app --release --debug-symbols false   # release bundle
cargo test && cargo fmt --all -- --check \
  && DECOMPOSITIONS_SKIP_WORKER_BUILD=1 cargo clippy --workspace --all-targets --target wasm32-unknown-unknown
```

Release builds need `--debug-symbols false`: dx defaults it on, emitting DWARF that its `wasm-opt` cannot parse, leaving the wasm unoptimized. Debug `dx serve` runs no `wasm-opt` and is unaffected. Set `DECOMPOSITIONS_SKIP_WORKER_BUILD=1` to skip the worker build for type-check-only tooling. Run the last command before pushing.

## Multi-threading

The optimizer can run on a rayon pool over `SharedArrayBuffer`, opt in:

```sh
DECOMPOSITIONS_WORKER_THREADS=1 dx bundle -p decompositions-app --release --debug-symbols false
```

The pool is `min(navigator.hardwareConcurrency, 32)`, the cap overridable with `DECOMPOSITIONS_WORKER_THREAD_CAP`. It needs a cross-origin isolated page, so the server must send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. `dx serve` sets neither and tree-shakes the pool export to one thread, so test threading with `dx bundle` behind a static server that adds those headers. The status line reports the live pool size.

## Examples

The app bundles three one-click datasets: MNIST and Fashion-MNIST (70000 samples, 20 PCA dimensions) and Cora (2708 papers, 1433 features reduced to 50 PCA dimensions).
