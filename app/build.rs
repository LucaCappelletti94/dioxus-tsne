//! Builds the dedicated worker wasm bundle into `public/worker/`, so plain
//! `dx serve` is the only command needed: dx serves the `public/` directory
//! verbatim at the site root, and this script runs before the app compiles.
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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=../worker/Cargo.toml");
    println!("cargo:rerun-if-changed=../worker/src");

    // Escape hatch for tooling that only needs the app to type check.
    if env::var_os("DECOMPOSITIONS_SKIP_WORKER_BUILD").is_some() {
        println!("cargo:warning=skipping the worker bundle build on request");
        return;
    }

    if let Err(error) = build_worker() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn build_worker() -> Result<(), String> {
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
    let target_dir = out_dir.join("worker-target");
    let bindgen_dir = out_dir.join("worker-bindgen");
    let public_dir = manifest_dir.join("public/worker");

    let cargo = env::var("CARGO").unwrap_or_else(|_| String::from("cargo"));
    // The worker does the heavy numeric lifting, an unoptimized build would
    // make even small datasets crawl, so it is always built with the release
    // profile regardless of the app profile.
    let status = Command::new(cargo)
        .current_dir(&workspace_root)
        .env_remove("RUSTFLAGS")
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
            "--target-dir",
        ])
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
    copy_if_changed(
        &bindgen_dir.join(format!("{WORKER_STEM}_bg.wasm")),
        &public_dir.join(format!("{WORKER_STEM}_bg.wasm")),
    )?;
    // The loader passes the wasm URL explicitly: dx minifies the JS it
    // serves and the transform strips the import.meta.url based default
    // inside the wasm-bindgen glue, so a bare init() call would receive
    // undefined and fail to instantiate.
    write_if_changed(
        &public_dir.join("loader.js"),
        format!(
            "import init from \"./{WORKER_STEM}.js\";\n\nawait init({{ module_or_path: new URL(\"./{WORKER_STEM}_bg.wasm\", import.meta.url) }});\n"
        )
        .into_bytes(),
    )?;
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
