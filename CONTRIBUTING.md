# Contributing

## Layout

- root: the publishable `dioxus-decompositions` library.
- `app/`: the web app and reference consumer.
- `worker/`: the worker binary registering `DecompositionWorker`. Consumers add an equivalent three line binary.

The app's `build.rs` compiles the worker and runs wasm-bindgen (via `wasm-bindgen-cli-support`, no external CLI) into `app/public/dioxus-decompositions/`, served at the site root.

## Requirements

`wasm32-unknown-unknown`, the `dx` CLI (Dioxus 0.7), and a nightly toolchain. The worker is always built as a multi-threaded rayon pool, which needs nightly atomics + build-std, so nightly is not optional.

## Commands

```sh
dx bundle -p decompositions-app --release --debug-symbols false   # release bundle
python3 scripts/serve_isolated.py                                 # serve it, cross-origin isolated
cargo test && cargo fmt --all -- --check \
  && DECOMPOSITIONS_SKIP_WORKER_BUILD=1 cargo clippy --workspace --exclude decompositions-worker \
       --all-targets --target wasm32-unknown-unknown
```

Release builds need `--debug-symbols false`: dx defaults it on, emitting DWARF that its `wasm-opt` cannot parse, leaving the wasm unoptimized. Set `DECOMPOSITIONS_SKIP_WORKER_BUILD=1` to skip the worker build for type-check-only tooling. The worker is excluded from the stable wasm clippy: it pulls in wasm-bindgen-rayon, which `compile_error!`s without the atomics target feature, so it only builds under the nightly atomics recipe `app/build.rs` drives. Run the last command before pushing.

## Threading and serving

The optimizer always runs on a rayon pool over `SharedArrayBuffer`. A single-threaded worker is unusably slow on real datasets, so it is not a build option (`app/build.rs` has no toggle for it). The pool is `min(navigator.hardwareConcurrency, 32)`, the cap overridable with `DECOMPOSITIONS_WORKER_THREAD_CAP`. The status line reports the live pool size.

The pool needs a cross-origin isolated page, so the server must send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. `dx serve --cross-origin-policy` sends both (verified against dx 0.7.9), so the headers do not need a custom server. The separate concern is dx's handling of the wasm-bindgen-rayon glue: its minifier strips the `import.meta.url` default from the wasm init (worked around by the generated `loader.js`) and can tree-shake the `initThreadPool` export, which would collapse the pool.

The verified end-to-end threaded path is a release `dx bundle` served by `scripts/serve_isolated.py` (port 9595, sets both headers). For a hot-reloading dev loop, `dx serve --cross-origin-policy` supplies the isolation headers, paired with a dx that preserves the pool export.

## Examples

The app bundles three one-click datasets: MNIST and Fashion-MNIST (70000 samples, 20 PCA dimensions) and Cora (2708 papers, 50 PCA dimensions, plus a `degree` column for the heatmap example, computed from the linqs Cora citation graph).
