# dioxus-tsne

[![CI](https://github.com/LucaCappelletti94/dioxus-tsne/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/dioxus-tsne/actions/workflows/ci.yml)
[![Deploy](https://github.com/LucaCappelletti94/dioxus-tsne/actions/workflows/deploy.yml/badge.svg)](https://github.com/LucaCappelletti94/dioxus-tsne/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-tsne.luca.phd-blue)](https://tsne.luca.phd)

Barnes-Hut t-SNE in the browser, in Rust + WebAssembly. Drop a CSV, TSV, Parquet, Arrow or NumPy (`.npy`) file (or pick an example) and watch the embedding form. The fit runs off the main thread on a `SharedArrayBuffer` Rayon pool, with PCA initialization for stable global structure.

## References

t-SNE: [van der Maaten & Hinton 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html), [van der Maaten 2014](https://www.jmlr.org/papers/v15/vandermaaten14a.html). Reading the maps: [Wattenberg et al. 2016](https://distill.pub/2016/misread-tsne/). PCA init: [Kobak & Berens 2019](https://doi.org/10.1038/s41467-019-13056-x), [Kobak & Linderman 2021](https://doi.org/10.1038/s41587-020-00809-z). Implementation: [bhtsne](https://github.com/frjnn/bhtsne).

See [CONTRIBUTING.md](CONTRIBUTING.md) to build and run it locally.
