//! The dioxus-decompositions web app: a full-bleed, minimal t-SNE explorer.
//! Drop a CSV, TSV or Parquet file (or pick an example) and watch the scatter
//! plot animate as epochs progress.

use dioxus::prelude::*;
use dioxus_tsne::{Decomposition, ExampleDataset, ExampleIcon};

/// Full MNIST (70000 digits, PCA-20, snappy Parquet) offered as a one click
/// example dataset, a heavy t-SNE workload that exercises the worker pool.
static MNIST_EXAMPLE: Asset = asset!("/assets/examples/mnist.parquet");

/// Full Fashion-MNIST (70000 Zalando clothing images, PCA-20, snappy Parquet),
/// colored by clothing category.
static FASHION_MNIST_EXAMPLE: Asset = asset!("/assets/examples/fashion_mnist.parquet");

/// Cora citation dataset (2708 papers, 7 subjects), reduced to 50 PCA
/// dimensions from the 1433 binary bag of words features, with a `degree`
/// column (citation degree) as a continuous-color example.
static CORA_EXAMPLE: Asset = asset!("/assets/examples/cora.parquet");

/// Brand logo shown top-left and used as the favicon.
static LOGO: Asset = asset!("/assets/logo.svg");

/// Page shell: the explorer is fixed and full-bleed, so the body just needs to
/// shed default margins and set the base font.
const APP_STYLE: &str = "
html, body {
    margin: 0;
    height: 100%;
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    background: #ffffff;
}
@media (prefers-color-scheme: dark) {
    html, body { background: #0a0a0a; }
}
";

/// SEO metadata, reused across the standard, Open Graph, and Twitter tags.
const SITE_TITLE: &str = "t-SNE: t-distributed Stochastic Neighbor Embedding";
const SITE_DESCRIPTION: &str = "An interactive t-SNE explorer in the browser. Drop a file and watch high-dimensional data form clusters in real time, powered by Rust and WebAssembly.";
const SITE_URL: &str = "https://tsne.luca.phd/";
const OG_IMAGE: &str = "https://tsne.luca.phd/og-image.png";
/// schema.org structured data describing the app, for rich search results.
const JSON_LD: &str = r#"{"@context":"https://schema.org","@type":"WebApplication","name":"t-SNE","url":"https://tsne.luca.phd/","description":"An interactive t-SNE explorer in the browser, powered by Rust and WebAssembly.","applicationCategory":"DataVisualization","operatingSystem":"Any (modern web browser)","offers":{"@type":"Offer","price":"0","priceCurrency":"USD"},"author":{"@type":"Person","name":"Luca Cappelletti"}}"#;

fn main() {
    console_error_panic_hook::set_once();
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        // Favicons and app icons (served from app/public at the site root).
        document::Link { rel: "icon", r#type: "image/svg+xml", href: "/favicon.svg" }
        document::Link { rel: "icon", r#type: "image/png", sizes: "32x32", href: "/favicon-32x32.png" }
        document::Link { rel: "icon", r#type: "image/png", sizes: "16x16", href: "/favicon-16x16.png" }
        document::Link { rel: "apple-touch-icon", sizes: "180x180", href: "/apple-touch-icon.png" }
        document::Link { rel: "manifest", href: "/site.webmanifest" }

        // SEO: standard meta, canonical, Open Graph, Twitter card, JSON-LD.
        document::Meta { name: "description", content: SITE_DESCRIPTION }
        document::Meta { name: "robots", content: "index, follow" }
        document::Meta { name: "theme-color", content: "#ffffff" }
        document::Link { rel: "canonical", href: SITE_URL }
        document::Meta { property: "og:type", content: "website" }
        document::Meta { property: "og:site_name", content: "t-SNE" }
        document::Meta { property: "og:title", content: SITE_TITLE }
        document::Meta { property: "og:description", content: SITE_DESCRIPTION }
        document::Meta { property: "og:url", content: SITE_URL }
        document::Meta { property: "og:image", content: OG_IMAGE }
        document::Meta { property: "og:image:width", content: "1200" }
        document::Meta { property: "og:image:height", content: "630" }
        document::Meta { name: "twitter:card", content: "summary_large_image" }
        document::Meta { name: "twitter:title", content: SITE_TITLE }
        document::Meta { name: "twitter:description", content: SITE_DESCRIPTION }
        document::Meta { name: "twitter:image", content: OG_IMAGE }
        document::Script { r#type: "application/ld+json", {JSON_LD} }

        style { {APP_STYLE} }
        {
            Decomposition::new()
                .logo(LOGO.to_string())
                .repository("https://github.com/LucaCappelletti94/dioxus-decompositions")
                .support("https://github.com/sponsors/LucaCappelletti94")
                .drop_zone()
                .examples(vec![
                    ExampleDataset {
                        name: String::from("MNIST 70k"),
                        url: MNIST_EXAMPLE.to_string(),
                        icon: Some(ExampleIcon::Numbers),
                        description: Some(String::from(
                            "MNIST: 70,000 handwritten digit images, pre-reduced to 20 PCA \
                             dimensions. Colored by digit (0-9).",
                        )),
                    },
                    ExampleDataset {
                        name: String::from("Fashion-MNIST 70k"),
                        url: FASHION_MNIST_EXAMPLE.to_string(),
                        icon: Some(ExampleIcon::Apparel),
                        description: Some(String::from(
                            "Fashion-MNIST: 70,000 Zalando clothing images, pre-reduced to 20 \
                             PCA dimensions. Colored by clothing category.",
                        )),
                    },
                    ExampleDataset {
                        name: String::from("Cora 2.7k"),
                        url: CORA_EXAMPLE.to_string(),
                        icon: Some(ExampleIcon::Network),
                        description: Some(String::from(
                            "Cora: 2,708 machine-learning papers in a citation network, 1,433 \
                             bag-of-words features reduced to 50 PCA dimensions. Color by subject, \
                             or by 'degree' (citation count) for a heatmap.",
                        )),
                    },
                ])
                .controls()
                .draggable_points()
                .render()
        }
    }
}
