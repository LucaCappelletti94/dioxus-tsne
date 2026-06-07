//! The animated scatter plot, a 2D canvas redrawn on every embedding
//! snapshot. Canvas scales to hundreds of thousands of points where SVG nodes
//! would not.

use dioxus::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

/// Maps a row major `n x 2` embedding into pixel coordinates fitting the
/// viewport, preserving the aspect ratio of the embedding.
///
/// t-SNE coordinates drift in magnitude over the epochs, so every snapshot is
/// rescaled independently: the embedding bounding box is centered and
/// uniformly scaled to fit the viewport minus the margin.
///
/// Pure so the mapping is testable natively.
fn project_to_viewport(points: &[f32], width: f32, height: f32, margin: f32) -> Vec<(f32, f32)> {
    let n = points.len() / 2;
    if n == 0 {
        return Vec::new();
    }

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for point in points.chunks_exact(2) {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }

    let span_x = (max_x - min_x).max(f32::EPSILON);
    let span_y = (max_y - min_y).max(f32::EPSILON);
    let scale = ((width - 2.0 * margin) / span_x).min((height - 2.0 * margin) / span_y);

    // Map relative to the bounding box center so the embedding is centered in
    // the viewport, degenerate spans (single point, collinear axis) included.
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;

    points
        .chunks_exact(2)
        .map(|point| {
            (
                width / 2.0 + (point[0] - center_x) * scale,
                height / 2.0 + (point[1] - center_y) * scale,
            )
        })
        .collect()
}

/// Draws the embedding on the canvas.
/// Default point color when no coloring is active.
const DEFAULT_COLOR: &str = "rgba(31, 119, 180, 0.8)";

fn draw(
    canvas: &HtmlCanvasElement,
    points: &[f32],
    colors: Option<&[String]>,
    width: u32,
    height: u32,
) {
    let Some(context) = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .and_then(|c| c.dyn_into::<CanvasRenderingContext2d>().ok())
    else {
        return;
    };

    context.clear_rect(0.0, 0.0, f64::from(width), f64::from(height));

    let pixels = project_to_viewport(points, width as f32, height as f32, 12.0);
    let n = pixels.len();
    if n == 0 {
        return;
    }

    // Points are batched per color into a single path each, so a coloring
    // costs as many fills as there are distinct colors (the continuous scale
    // is quantized for this very reason).
    let colors = colors.filter(|c| c.len() == n);
    let mut batches: Vec<(&str, Vec<usize>)> = Vec::new();
    match colors {
        Some(colors) => {
            for (index, color) in colors.iter().enumerate() {
                match batches.iter_mut().find(|(c, _)| c == color) {
                    Some((_, indices)) => indices.push(index),
                    None => batches.push((color, vec![index])),
                }
            }
        }
        None => batches.push((DEFAULT_COLOR, (0..n).collect())),
    }

    // Circles look better but cost a path arc each, rectangles keep huge
    // embeddings fluid.
    for (color, indices) in batches {
        context.set_fill_style_str(color);
        if n <= 20_000 {
            const RADIUS: f64 = 3.0;
            context.begin_path();
            for &index in &indices {
                let (x, y) = pixels[index];
                let (x, y) = (f64::from(x), f64::from(y));
                context.move_to(x + RADIUS, y);
                let _ = context.arc(x, y, RADIUS, 0.0, std::f64::consts::TAU);
            }
            context.fill();
        } else {
            const SIDE: f64 = 2.0;
            for &index in &indices {
                let (x, y) = pixels[index];
                context.fill_rect(f64::from(x), f64::from(y), SIDE, SIDE);
            }
        }
    }
}

/// Canvas scatter plot of a row major `n x 2` embedding, redrawn whenever the
/// embedding signal changes and rescaled to the viewport on every redraw.
///
/// # Props
///
/// * `embedding` - the points to draw, cleared when `None`.
/// * `colors` - optional CSS color per point, see [`crate::colorize`]. Points
///   fall back to a single default color when absent or of mismatched length.
/// * `width` - canvas width in pixels, defaults to 800.
/// * `height` - canvas height in pixels, defaults to 600.
#[component]
pub fn ScatterPlot(
    embedding: ReadSignal<Option<Vec<f32>>>,
    #[props(default = None)] colors: Option<ReadSignal<Option<Vec<String>>>>,
    #[props(default = 800)] width: u32,
    #[props(default = 600)] height: u32,
) -> Element {
    let mut canvas = use_signal(|| None::<HtmlCanvasElement>);

    // Redraws when the canvas mounts or the embedding or coloring changes.
    use_effect(move || {
        let Some(canvas) = canvas() else {
            return;
        };
        let colors = colors.map(|c| c.read().clone()).unwrap_or_default();
        match embedding.read().as_ref() {
            Some(points) => draw(&canvas, points, colors.as_deref(), width, height),
            None => {
                if let Some(context) = canvas
                    .get_context("2d")
                    .ok()
                    .flatten()
                    .and_then(|c| c.dyn_into::<CanvasRenderingContext2d>().ok())
                {
                    context.clear_rect(0.0, 0.0, f64::from(width), f64::from(height));
                }
            }
        }
    });

    rsx! {
        canvas {
            id: "scatter-plot",
            class: "decompositions-plot",
            width: "{width}",
            height: "{height}",
            onmounted: move |evt| {
                canvas.set(
                    evt.data()
                        .downcast::<web_sys::Element>()
                        .and_then(|element| {
                            element.clone().dyn_into::<HtmlCanvasElement>().ok()
                        }),
                );
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::project_to_viewport;

    #[test]
    fn fits_in_viewport_with_margin() {
        // Coordinates with wild magnitudes, as t-SNE produces late in the run.
        let points = [-250.0, 1000.0, 480.0, -900.0, 0.0, 30.0];
        let pixels = project_to_viewport(&points, 800.0, 600.0, 12.0);

        assert_eq!(pixels.len(), 3);
        for &(x, y) in &pixels {
            assert!((12.0..=788.0).contains(&x), "x out of bounds: {x}");
            assert!((12.0..=588.0).contains(&y), "y out of bounds: {y}");
        }
    }

    #[test]
    fn preserves_aspect_ratio() {
        // A 2:1 wide rectangle must stay 2:1 in pixel space.
        let points = [0.0, 0.0, 20.0, 0.0, 20.0, 10.0, 0.0, 10.0];
        let pixels = project_to_viewport(&points, 500.0, 500.0, 0.0);

        let width = pixels[1].0 - pixels[0].0;
        let height = pixels[2].1 - pixels[1].1;
        assert!((width / height - 2.0).abs() < 1e-4, "{width} x {height}");
    }

    #[test]
    fn rescales_each_snapshot_independently() {
        // The same shape at two very different magnitudes maps to the same
        // pixels: early small-norm snapshots fill the canvas just as well.
        let small = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let large: Vec<f32> = small.iter().map(|v| v * 1e4).collect();

        let a = project_to_viewport(&small, 640.0, 480.0, 10.0);
        let b = project_to_viewport(&large, 640.0, 480.0, 10.0);
        for (pa, pb) in a.iter().zip(&b) {
            assert!((pa.0 - pb.0).abs() < 1e-2 && (pa.1 - pb.1).abs() < 1e-2);
        }
    }

    #[test]
    fn single_point_is_centered() {
        let pixels = project_to_viewport(&[7.0, 7.0], 400.0, 300.0, 10.0);
        assert_eq!(pixels.len(), 1);
        assert!((pixels[0].0 - 200.0).abs() < 1.0);
        assert!((pixels[0].1 - 150.0).abs() < 1.0);
    }

    #[test]
    fn empty_embedding_yields_no_pixels() {
        assert!(project_to_viewport(&[], 400.0, 300.0, 10.0).is_empty());
    }
}
