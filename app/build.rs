//! Builds the dedicated worker wasm bundle into `public/dioxus-decompositions/`,
//! so plain `dx serve` is the only command needed: dx serves the `public/`
//! directory verbatim at the site root, and this script runs before the app
//! compiles. The output directory matches `dioxus_decompositions::DEFAULT_WORKER_URL`
//! (`/dioxus-decompositions/loader.js`), the URL the component loads the worker
//! from by default.
//!
//! wasm-bindgen runs through the `wasm-bindgen-cli-support` library, locked by
//! Cargo like every other dependency, so no external CLI at a matching version
//! is required. The nested cargo build writes to a target directory under
//! OUT_DIR to avoid locking the workspace target directory.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const WORKER_PACKAGE: &str = "decompositions-worker";
const WORKER_STEM: &str = "decompositions_worker";

/// RUSTFLAGS for the threaded worker build: atomics plus the linker exports
/// wasm-bindgen-rayon needs to share memory across the pool workers. Mirrors the
/// flags documented by wasm-bindgen-rayon.
const THREAD_RUSTFLAGS: &str = "-C target-feature=+atomics,+bulk-memory \
-C link-arg=--shared-memory \
-C link-arg=--max-memory=1073741824 \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls \
-C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align \
-C link-arg=--export=__tls_base \
-C link-arg=--export=__heap_base";

/// Default cap on the rayon pool size. Benchmarking bhtsne in the browser
/// (the full 70k MNIST) shows the wall clock keeps improving up to roughly the
/// physical core count, then regresses once the pool oversubscribes the cores
/// (SMT threads contend and the atomics synchronization overhead dominates), so
/// the pool does not follow the full logical core count that the browser
/// reports. Thirty-two matches a typical high-core desktop, smaller machines
/// use all their cores through the `min` with `hardwareConcurrency`. Override
/// with the `DECOMPOSITIONS_WORKER_THREAD_CAP` environment variable.
const DEFAULT_THREAD_CAP: usize = 32;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=../worker/Cargo.toml");
    println!("cargo:rerun-if-changed=../worker/src");
    println!("cargo:rerun-if-env-changed=DECOMPOSITIONS_WORKER_THREADS");
    println!("cargo:rerun-if-env-changed=DECOMPOSITIONS_WORKER_THREAD_CAP");

    // Escape hatch for tooling that only needs the app to type check.
    if env::var_os("DECOMPOSITIONS_SKIP_WORKER_BUILD").is_some() {
        println!("cargo:warning=skipping the worker bundle build on request");
        return;
    }

    // Opt in to the SharedArrayBuffer rayon thread pool. It needs a nightly
    // atomics + build-std compile and cross-origin isolation headers at serve
    // time, so the default build stays single threaded and stable.
    let threads = env::var_os("DECOMPOSITIONS_WORKER_THREADS").is_some();
    if threads {
        println!(
            "cargo:warning=building the worker as a multi-threaded rayon pool (atomics + build-std)"
        );
    }

    if let Err(error) = build_worker(threads) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn build_worker(threads: bool) -> Result<(), String> {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| String::from("cargo did not provide CARGO_MANIFEST_DIR"))?,
    );
    let workspace_root = manifest_dir
        .join("..")
        .canonicalize()
        .map_err(|error| format!("failed to resolve the workspace root: {error}"))?;
    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| String::from("cargo did not provide OUT_DIR"))?,
    );
    // Keep the threaded and plain builds in separate target directories so
    // toggling the mode does not force a full rebuild of the other.
    let target_dir = out_dir.join(if threads {
        "worker-target-threads"
    } else {
        "worker-target"
    });
    let bindgen_dir = out_dir.join("worker-bindgen");
    let public_dir = manifest_dir.join("public/dioxus-decompositions");

    // The worker does the heavy numeric lifting, an unoptimized build would
    // make even small datasets crawl, so it is always built with the release
    // profile regardless of the app profile. The threaded variant additionally
    // needs a nightly atomics build that rebuilds std (build-std), driven
    // through rustup so the app's own toolchain is left untouched.
    let mut command = if threads {
        let mut command = Command::new("rustup");
        command.args(["run", "nightly", "cargo"]);
        command
    } else {
        Command::new(env::var("CARGO").unwrap_or_else(|_| String::from("cargo")))
    };
    command
        .current_dir(&workspace_root)
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .args([
            "build",
            "--package",
            WORKER_PACKAGE,
            "--bin",
            WORKER_STEM,
            "--target",
            "wasm32-unknown-unknown",
            "--release",
        ]);
    if threads {
        command.env("RUSTFLAGS", THREAD_RUSTFLAGS).args([
            "--features",
            "threads",
            "-Z",
            "build-std=std,panic_abort",
        ]);
    } else {
        command.env_remove("RUSTFLAGS");
    }
    let status = command
        .args(["--target-dir"])
        .arg(&target_dir)
        .status()
        .map_err(|error| format!("failed to launch the worker cargo build: {error}"))?;
    if !status.success() {
        return Err(format!("the worker cargo build failed with {status}"));
    }

    let worker_wasm = target_dir
        .join("wasm32-unknown-unknown")
        .join("release")
        .join(format!("{WORKER_STEM}.wasm"));
    if !worker_wasm.exists() {
        return Err(format!(
            "expected the worker wasm at {}",
            worker_wasm.display()
        ));
    }

    let _ignored = fs::remove_dir_all(&bindgen_dir);
    fs::create_dir_all(&bindgen_dir)
        .map_err(|error| format!("failed to create the bindgen directory: {error}"))?;
    wasm_bindgen_cli_support::Bindgen::new()
        .input_path(&worker_wasm)
        .out_name(WORKER_STEM)
        .typescript(false)
        .web(true)
        .map_err(|error| format!("failed to configure wasm-bindgen for web output: {error}"))?
        .generate(&bindgen_dir)
        .map_err(|error| format!("wasm-bindgen generation failed: {error}"))?;

    fs::create_dir_all(&public_dir)
        .map_err(|error| format!("failed to create the public worker directory: {error}"))?;
    copy_if_changed(
        &bindgen_dir.join(format!("{WORKER_STEM}.js")),
        &public_dir.join(format!("{WORKER_STEM}.js")),
    )?;

    // wasm-opt the worker, the heavy numeric path. dx optimizes the main app
    // bundle but never touches this separately built worker, so without this it
    // ships as raw rustc/wasm-bindgen output. Only in release: the optimization
    // is slow and pointless for debug iteration. `Feature::All` lets wasm-opt
    // accept whatever the toolchain emitted, including the threaded build's
    // atomics, bulk memory and shared memory.
    let worker_bg = bindgen_dir.join(format!("{WORKER_STEM}_bg.wasm"));
    let public_bg = public_dir.join(format!("{WORKER_STEM}_bg.wasm"));
    let release = env::var("PROFILE").map(|p| p == "release").unwrap_or(false);
    if release {
        let optimized = bindgen_dir.join(format!("{WORKER_STEM}_bg.opt.wasm"));
        wasm_opt::OptimizationOptions::new_opt_level_2()
            .enable_feature(wasm_opt::Feature::All)
            .run(&worker_bg, &optimized)
            .map_err(|error| format!("wasm-opt failed on the worker: {error}"))?;
        copy_if_changed(&optimized, &public_bg)?;
    } else {
        copy_if_changed(&worker_bg, &public_bg)?;
    }

    // The threaded build references wasm-bindgen-rayon's worker helper as a JS
    // snippet, served alongside the glue. The plain build has no snippets, so
    // any left over from a previous threaded build are cleared.
    let public_snippets = public_dir.join("snippets");
    let _ignored = fs::remove_dir_all(&public_snippets);
    if threads {
        copy_dir_all(&bindgen_dir.join("snippets"), &public_snippets)?;
    }

    // The loader passes the wasm URL explicitly: dx minifies the JS it
    // serves and the transform strips the import.meta.url based default
    // inside the wasm-bindgen glue, so a bare init() call would receive
    // undefined and fail to instantiate. In threads mode it then spins up the
    // rayon pool over all available cores before any compute message arrives.
    let loader = if threads {
        // dx bundles this loader together with the wasm-bindgen glue into a
        // single module, so its import.meta.url (which wasm-bindgen-rayon hands
        // to the pool workers as the module to re-import) points back here. The
        // pool workers must therefore skip the init and pool setup below (the
        // inlined helper instantiates them with the shared module and memory).
        // Without this guard each pool worker would re-run init and
        // initThreadPool, recursively spawning workers until memory is
        // exhausted. They are told apart by the worker's own location: the gloo
        // worker is created from the served loader URL, the pool workers from a
        // blob URL, so self.location.href is a blob URL only in the pool
        // workers (import.meta.url cannot be used here, the pool workers
        // re-import this same module by its real URL). (The name option
        // wasm-bindgen-rayon sets is dropped by dx's minifier, so it is unusable
        // as the discriminator.)
        let cap = env::var("DECOMPOSITIONS_WORKER_THREAD_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_THREAD_CAP);
        format!(
            "import init, {{ initThreadPool, register_worker }} from \"./{WORKER_STEM}.js\";\n// dx bundles the glue into this module, so it must re-export the glue's\n// interface: the pool workers re-import this module and call its `default`\n// (init) and `wbg_rayon_start_worker` exports.\nexport * from \"./{WORKER_STEM}.js\";\nexport {{ default }} from \"./{WORKER_STEM}.js\";\n\nif (!self.location.href.startsWith(\"blob:\")) {{\n    await init({{ module_or_path: new URL(\"./{WORKER_STEM}_bg.wasm\", import.meta.url) }});\n    // bhtsne in WASM is fastest with a small pool, the full core count is much\n    // slower (atomics sync overhead), so cap it. See DEFAULT_THREAD_CAP.\n    await initThreadPool(Math.min(navigator.hardwareConcurrency, {cap}));\n    // Only start handling compute once the pool is built, otherwise the first\n    // run would race the pool init and pin the worker to a single thread.\n    register_worker();\n}}\n"
        )
    } else {
        format!(
            "import init from \"./{WORKER_STEM}.js\";\n\nawait init({{ module_or_path: new URL(\"./{WORKER_STEM}_bg.wasm\", import.meta.url) }});\n"
        )
    };
    write_if_changed(&public_dir.join("loader.js"), loader.into_bytes())?;
    Ok(())
}

/// Recursively copies a directory, skipping unchanged files so the dx watcher
/// is not retriggered.
fn copy_dir_all(source: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination)
        .map_err(|error| format!("failed to create {}: {error}", destination.display()))?;
    let entries = fs::read_dir(source)
        .map_err(|error| format!("failed to read {}: {error}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("failed to read a directory entry: {error}"))?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .map_err(|error| format!("failed to stat {}: {error}", from.display()))?;
        if file_type.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            copy_if_changed(&from, &to)?;
        }
    }
    Ok(())
}

/// Copies only when the content differs, so unchanged outputs keep their
/// mtime and do not retrigger the dx file watcher.
fn copy_if_changed(source: &Path, destination: &Path) -> Result<(), String> {
    let content = fs::read(source)
        .map_err(|error| format!("failed to read {}: {error}", source.display()))?;
    write_if_changed(destination, content)
}

fn write_if_changed(destination: &Path, content: Vec<u8>) -> Result<(), String> {
    if fs::read(destination).is_ok_and(|existing| existing == content) {
        return Ok(());
    }
    fs::write(destination, content)
        .map_err(|error| format!("failed to write {}: {error}", destination.display()))?;
    Ok(())
}
