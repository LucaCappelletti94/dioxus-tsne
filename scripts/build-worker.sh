#!/usr/bin/env bash
# Builds the worker wasm bundle and places the wasm-bindgen output in the app
# assets, where the UI spawns it from (see WORKER_ASSETS in the app crate).
# Requires the wasm-bindgen CLI at the exact version pinned in the workspace
# Cargo.toml.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo build -p decompositions-worker --target wasm32-unknown-unknown --release

wasm-bindgen \
    --target web \
    --no-typescript \
    --out-dir app/assets/worker \
    --out-name decompositions_worker \
    target/wasm32-unknown-unknown/release/decompositions_worker.wasm

echo "worker bundle written to app/assets/worker/"
