import init, { initThreadPool, register_worker } from "./decompositions_worker.js";
// dx bundles the glue into this module, so it must re-export the glue's
// interface: the pool workers re-import this module and call its `default`
// (init) and `wbg_rayon_start_worker` exports.
export * from "./decompositions_worker.js";
export { default } from "./decompositions_worker.js";

if (!self.location.href.startsWith("blob:")) {
    await init({ module_or_path: new URL("./decompositions_worker_bg.wasm", import.meta.url) });
    // bhtsne in WASM is fastest with a small pool, the full core count is much
    // slower (atomics sync overhead), so cap it. See DEFAULT_THREAD_CAP.
    await initThreadPool(Math.min(navigator.hardwareConcurrency, 32));
    // Only start handling compute once the pool is built, otherwise the first
    // run would race the pool init and pin the worker to a single thread.
    register_worker();
}
