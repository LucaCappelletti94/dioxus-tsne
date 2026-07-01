//! The animated scatter plot, a 2D canvas redrawn on every embedding
//! snapshot. Canvas scales to hundreds of thousands of points where SVG nodes
//! would not.

use std::cell::RefCell;
use std::rc::Rc;

use dioxus::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::color::{LegendEntry, Marker};

/// Viewport margin, in canvas pixels, kept clear around the embedding.
const MARGIN: f32 = 12.0;

/// Above this many points the plot draws fast colored dots instead of per-class
/// marker shapes (the shapes overlap into illegibility anyway). The legend keys
/// off the same limit so it shows uniform dots rather than implying shapes that
/// are not actually drawn.
pub(crate) const SHAPED_MARKER_LIMIT: usize = 20_000;

/// Baked-in legend (snapshot export) layout, in logical pixels.
const LEGEND_FONT: &str = "13px system-ui, -apple-system, 'Segoe UI', sans-serif";
const LEGEND_ROW_H: f32 = 20.0;
const LEGEND_PAD: f32 = 14.0;
/// Width of the marker plus the gap before its label.
const LEGEND_MARKER_COL: f32 = 18.0;
const LEGEND_MIN_WIDTH: f32 = 90.0;

/// The data-space to canvas-pixel mapping of one draw.
///
/// t-SNE coordinates drift in magnitude over the epochs, so every snapshot is
/// rescaled independently: the embedding bounding box is centered and
/// uniformly scaled to fit the viewport minus the margin. Holding onto the
/// transform of the last draw lets pointer handlers hit-test and unproject
/// against exactly what is on screen.
///
/// Pure so the mapping is testable natively.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Transform {
    scale: f32,
    center_x: f32,
    center_y: f32,
    width: f32,
    height: f32,
}

impl Transform {
    /// Fits the embedding bounding box into the viewport, preserving aspect
    /// ratio. `None` when the embedding is empty.
    ///
    /// The box is fit to the bulk of the points: any beyond
    /// [`OUTLIER_FIT_FACTOR`] times the median distance from the center of mass
    /// are treated as outliers and left out, so a few far stragglers (common in
    /// the first epochs of a run) do not blow up the viewport and squish the
    /// rest into a ball. Stragglers simply clip off-screen until they settle, a
    /// settled embedding has none and is fit in full.
    pub(crate) fn fit(points: &[f32], width: f32, height: f32, margin: f32) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }
        let n = points.len() / 2;

        // Center of mass.
        let (mut mean_x, mut mean_y) = (0.0f32, 0.0f32);
        for point in points.chunks_exact(2) {
            mean_x += point[0];
            mean_y += point[1];
        }
        mean_x /= n as f32;
        mean_y /= n as f32;
        let dist_sq = |point: &[f32]| (point[0] - mean_x).powi(2) + (point[1] - mean_y).powi(2);

        // Outlier cutoff at OUTLIER_FIT_FACTOR times the median distance. A zero
        // median (most points coincident) disables the cutoff.
        let threshold_sq = {
            let mut distances: Vec<f32> = points.chunks_exact(2).map(&dist_sq).collect();
            let median = n / 2;
            distances.select_nth_unstable_by(median, f32::total_cmp);
            let median_sq = distances[median];
            if median_sq > 0.0 {
                median_sq * OUTLIER_FIT_FACTOR.powi(2)
            } else {
                f32::INFINITY
            }
        };

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for point in points.chunks_exact(2) {
            if dist_sq(point) <= threshold_sq {
                min_x = min_x.min(point[0]);
                max_x = max_x.max(point[0]);
                min_y = min_y.min(point[1]);
                max_y = max_y.max(point[1]);
            }
        }

        let span_x = (max_x - min_x).max(f32::EPSILON);
        let span_y = (max_y - min_y).max(f32::EPSILON);
        let scale = ((width - 2.0 * margin) / span_x).min((height - 2.0 * margin) / span_y);

        Some(Self {
            scale,
            // Map relative to the bounding box center so the embedding is
            // centered, degenerate spans (single point, collinear axis) too.
            center_x: (min_x + max_x) / 2.0,
            center_y: (min_y + max_y) / 2.0,
            width,
            height,
        })
    }

    /// Maps a data-space point to canvas pixels.
    pub(crate) fn project(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.width / 2.0 + (x - self.center_x) * self.scale,
            self.height / 2.0 + (y - self.center_y) * self.scale,
        )
    }

    /// Maps a canvas pixel back to data space, the inverse of [`project`].
    ///
    /// [`project`]: Self::project
    fn unproject(&self, px: f32, py: f32) -> (f32, f32) {
        (
            self.center_x + (px - self.width / 2.0) / self.scale,
            self.center_y + (py - self.height / 2.0) / self.scale,
        )
    }
}

/// Projects a row major `n x 2` embedding into viewport pixels, or an empty
/// vector when the embedding is empty. Thin wrapper over [`Transform`], kept
/// for the projection tests now that drawing drives the transform directly.
#[cfg(test)]
fn project_to_viewport(points: &[f32], width: f32, height: f32, margin: f32) -> Vec<(f32, f32)> {
    match Transform::fit(points, width, height, margin) {
        Some(transform) => points
            .chunks_exact(2)
            .map(|point| transform.project(point[0], point[1]))
            .collect(),
        None => Vec::new(),
    }
}

/// Draws the embedding on the canvas.
/// Default point color when no coloring is active (opacity comes from
/// [`FILL_ALPHA`]).
const DEFAULT_COLOR: &str = "#1f77b4";

/// Color of points outside the focused class while a legend entry is hovered or
/// pinned, so only the focused class keeps its real color and stands out.
const DIMMED_COLOR: &str = "#d8d8d8";

/// Opacity of marker fills, so dense regions read as darker where points
/// overlap. The same-color border is drawn fully opaque.
const FILL_ALPHA: f64 = 0.82;

/// Opaque canvas background, so a recorded video has a solid backdrop rather
/// than a transparent one (which video codecs render as black). Picked at draw
/// time to match the page theme: a white plot under a dark UI was the old
/// dark-mode "white on white" bug.
const PLOT_BACKGROUND_LIGHT: &str = "#ffffff";
const PLOT_BACKGROUND_DARK: &str = "#0a0a0a";

/// Whether the page prefers a dark color scheme, so canvas drawing (which CSS
/// variables cannot reach) can match the theme. Cached: `draw` runs every frame
/// and `matchMedia` is a needless DOM query to repeat at the frame rate. A theme
/// switch mid-session is reflected on the next reload.
fn prefers_dark() -> bool {
    thread_local! {
        static CACHED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    CACHED.with(|cached| {
        if let Some(value) = cached.get() {
            return value;
        }
        let value = web_sys::window()
            .and_then(|window| {
                window
                    .match_media("(prefers-color-scheme: dark)")
                    .ok()
                    .flatten()
            })
            .is_some_and(|query| query.matches());
        cached.set(Some(value));
        value
    })
}

/// Multiple of the median distance from the center of mass beyond which a point
/// is left out of the viewport fit (see [`Transform::fit`]). Large enough that a
/// settled embedding keeps all its points, small enough to discard the far
/// stragglers thrown out in the first epochs of a run.
const OUTLIER_FIT_FACTOR: f32 = 4.0;

/// Appends one marker outline at `(x, y)` to the current path.
fn add_marker_to_path(context: &CanvasRenderingContext2d, marker: Marker, x: f64, y: f64, r: f64) {
    match marker {
        Marker::Circle => {
            context.move_to(x + r, y);
            let _ = context.arc(x, y, r, 0.0, std::f64::consts::TAU);
        }
        Marker::Triangle => {
            // Vertices on the circumscribed circle so sizes match the circle.
            context.move_to(x, y - r);
            context.line_to(x + 0.866 * r, y + 0.5 * r);
            context.line_to(x - 0.866 * r, y + 0.5 * r);
            context.close_path();
        }
        Marker::Square => {
            context.rect(x - r, y - r, 2.0 * r, 2.0 * r);
        }
        Marker::Diamond => {
            context.move_to(x, y - r);
            context.line_to(x + r, y);
            context.line_to(x, y + r);
            context.line_to(x - r, y);
            context.close_path();
        }
        Marker::Plus => {
            // Two overlapping bars, filled as one nonzero winding region.
            let arm = 0.4 * r;
            context.rect(x - r, y - arm, 2.0 * r, 2.0 * arm);
            context.rect(x - arm, y - r, 2.0 * arm, 2.0 * r);
        }
        Marker::TriangleDown => {
            context.move_to(x, y + r);
            context.line_to(x + 0.866 * r, y - 0.5 * r);
            context.line_to(x - 0.866 * r, y - 0.5 * r);
            context.close_path();
        }
    }
}

/// Draws the embedding and returns the transform it used, or `None` when there
/// is nothing to draw. `transform_override` freezes the mapping (during a drag)
/// instead of refitting to the current bounding box.
#[allow(clippy::too_many_arguments)]
fn draw(
    canvas: &HtmlCanvasElement,
    points: &[f32],
    colors: Option<&[String]>,
    markers: Option<&[Marker]>,
    highlight: Option<(&str, Marker)>,
    width: u32,
    height: u32,
    ratio: f64,
    transform_override: Option<Transform>,
    opaque_background: bool,
    legend: Option<&[LegendEntry]>,
) -> Option<Transform> {
    let context = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .and_then(|c| c.dyn_into::<CanvasRenderingContext2d>().ok())?;

    // The backing buffer is `ratio` times the logical size so the plot renders
    // crisply on high-DPI displays. All drawing stays in logical coordinates,
    // the context transform scales it up to device pixels, so marker sizes are
    // unchanged but sharp.
    let _ = context.set_transform(ratio, 0.0, 0.0, ratio, 0.0, 0.0);
    context.clear_rect(0.0, 0.0, f64::from(width), f64::from(height));
    // Paint an opaque background so the canvas is not transparent: a transparent
    // canvas reads as white over the page but is encoded as black when the
    // animation is captured to a video, see the recording feature. The PNG
    // snapshot wants a transparent background instead, so it opts out.
    if opaque_background {
        let background = if prefers_dark() {
            PLOT_BACKGROUND_DARK
        } else {
            PLOT_BACKGROUND_LIGHT
        };
        context.set_fill_style_str(background);
        context.fill_rect(0.0, 0.0, f64::from(width), f64::from(height));
    }

    // Reserve a left strip for a baked-in legend (the snapshot export), and fit
    // the points into the area to its right so they never overlap it.
    let legend = legend.filter(|entries| !entries.is_empty());
    let strip_w = legend.map_or(0.0, |entries| {
        legend_strip_width(&context, entries, width as f32)
    });

    let transform = transform_override
        .or_else(|| Transform::fit(points, width as f32 - strip_w, height as f32, MARGIN))?;
    let pixels: Vec<(f32, f32)> = points
        .chunks_exact(2)
        .map(|point| {
            let (x, y) = transform.project(point[0], point[1]);
            (x + strip_w, y)
        })
        .collect();
    let n = pixels.len();
    if n == 0 {
        return None;
    }

    // Points are batched per (color, marker) into a single path each, so a
    // coloring costs as many fills as there are distinct pairs (the
    // continuous scale is quantized for this very reason).
    let colors = colors.filter(|c| c.len() == n);
    let markers = markers.filter(|m| m.len() == n);
    let mut batches: Vec<(&str, Marker, Vec<usize>)> = Vec::new();
    for index in 0..n {
        let base = colors.map_or(DEFAULT_COLOR, |c| c[index].as_str());
        let marker = markers.map_or(Marker::Circle, |m| m[index]);
        // While a class is focused, every point outside it is greyed so the
        // focused class stands out. A class is the (color, marker) pair, which
        // is unique per category.
        let color = match highlight {
            Some((hc, hm)) if base != hc || marker != hm => DIMMED_COLOR,
            _ => base,
        };
        match batches
            .iter_mut()
            .find(|(c, m, _)| *c == color && *m == marker)
        {
            Some((_, _, indices)) => indices.push(index),
            None => batches.push((color, marker, vec![index])),
        }
    }

    // Draw the greyed points first so the focused class paints on top of them.
    if highlight.is_some() {
        batches.sort_by_key(|(color, _, _)| *color != DIMMED_COLOR);
    }

    // Shaped markers look better but cost a path each, rectangles keep huge
    // embeddings fluid. Markers have a translucent fill and a same-color border
    // so individual points stay distinct while overlaps read as denser.
    context.set_line_width(1.0);
    for (color, marker, indices) in batches {
        context.set_fill_style_str(color);
        if n > SHAPED_MARKER_LIMIT {
            // Above this many points the per-point shapes are illegible anyway
            // (they overlap), so draw fast colored dots. This keeps huge
            // embeddings (and their snapshots and drags) fluid, since a path per
            // point would be far too slow.
            const SIDE: f64 = 2.0;
            context.set_global_alpha(FILL_ALPHA);
            for &index in &indices {
                let (x, y) = pixels[index];
                context.fill_rect(f64::from(x), f64::from(y), SIDE, SIDE);
            }
            context.set_global_alpha(1.0);
        } else {
            const RADIUS: f64 = 3.0;
            context.set_stroke_style_str(color);
            context.begin_path();
            for &index in &indices {
                let (x, y) = pixels[index];
                add_marker_to_path(&context, marker, f64::from(x), f64::from(y), RADIUS);
            }
            context.set_global_alpha(FILL_ALPHA);
            context.fill();
            context.set_global_alpha(1.0);
            context.stroke();
        }
    }

    if let Some(entries) = legend {
        draw_legend(
            &context,
            entries,
            strip_w,
            height as f32,
            n > SHAPED_MARKER_LIMIT,
        );
    }

    Some(transform)
}

/// Width of the reserved legend strip: the longest label plus the marker column
/// and padding, clamped so it never eats more than 40% of the canvas.
fn legend_strip_width(
    context: &CanvasRenderingContext2d,
    entries: &[LegendEntry],
    canvas_width: f32,
) -> f32 {
    context.set_font(LEGEND_FONT);
    let max_label = entries
        .iter()
        .map(|entry| {
            context
                .measure_text(&entry.label)
                .map(|metrics| metrics.width() as f32)
                .unwrap_or(0.0)
        })
        .fold(0.0f32, f32::max);
    (LEGEND_PAD * 2.0 + LEGEND_MARKER_COL + max_label).clamp(LEGEND_MIN_WIDTH, canvas_width * 0.4)
}

/// Draws the legend down the reserved left strip: a colored marker and its label
/// per class, vertically centered, with a "+N more" row when the classes do not
/// all fit the height.
fn draw_legend(
    context: &CanvasRenderingContext2d,
    entries: &[LegendEntry],
    strip_w: f32,
    height: f32,
    uniform_marker: bool,
) {
    // Match the page theme so the labels read on the export's own background.
    let dark = prefers_dark();
    let text_color = if dark { "#f4f4f4" } else { "#1a1a1a" };

    context.set_font(LEGEND_FONT);
    context.set_text_baseline("middle");

    let usable = (height - 2.0 * LEGEND_PAD).max(LEGEND_ROW_H);
    let max_rows = (usable / LEGEND_ROW_H).floor().max(1.0) as usize;
    let (shown, overflow) = if entries.len() > max_rows {
        (max_rows - 1, entries.len() - (max_rows - 1))
    } else {
        (entries.len(), 0)
    };
    let rows = shown + usize::from(overflow > 0);
    let mut y = (height - rows as f32 * LEGEND_ROW_H) / 2.0 + LEGEND_ROW_H / 2.0;
    let marker_x = f64::from(LEGEND_PAD + LEGEND_MARKER_COL / 2.0);
    let label_x = f64::from(LEGEND_PAD + LEGEND_MARKER_COL);

    for entry in entries.iter().take(shown) {
        // Match the plot: above the limit it draws dots, not shapes.
        let marker = if uniform_marker {
            Marker::Circle
        } else {
            entry.marker
        };
        context.set_fill_style_str(&entry.color);
        context.set_stroke_style_str(&entry.color);
        context.set_line_width(1.0);
        context.begin_path();
        add_marker_to_path(context, marker, marker_x, f64::from(y), 5.0);
        context.set_global_alpha(FILL_ALPHA);
        context.fill();
        context.set_global_alpha(1.0);
        context.stroke();

        context.set_fill_style_str(text_color);
        let _ = context.fill_text(&entry.label, label_x, f64::from(y));
        y += LEGEND_ROW_H;
    }
    if overflow > 0 {
        context.set_fill_style_str(text_color);
        let _ = context.fill_text(&format!("+{overflow} more"), label_x, f64::from(y));
    }
    context.set_text_baseline("alphabetic");

    // Faint separator between the legend strip and the plot.
    context.set_stroke_style_str(if dark { "#3a3a3a" } else { "#dddddd" });
    context.set_line_width(1.0);
    context.begin_path();
    context.move_to(f64::from(strip_w), f64::from(LEGEND_PAD));
    context.line_to(f64::from(strip_w), f64::from(height - LEGEND_PAD));
    context.stroke();
}

/// Renders the current embedding to an offscreen square canvas with a
/// transparent background and returns it as a PNG data URL, for the snapshot
/// download. Unlike the on-screen plot (full-bleed and opaque, so it records to
/// video cleanly), this frames the embedding into a square sized to the data,
/// so the saved picture is tight and drops straight onto any background.
///
/// `size` is the square's logical side in pixels, rendered at the device pixel
/// ratio for crispness. Returns `None` if the canvas cannot be created or there
/// is nothing to draw.
pub(crate) fn snapshot_png(
    points: &[f32],
    colors: Option<&[String]>,
    markers: Option<&[Marker]>,
    highlight: Option<(&str, Marker)>,
    legend: Option<&[LegendEntry]>,
    size: u32,
) -> Option<String> {
    let document = web_sys::window()?.document()?;
    let canvas: HtmlCanvasElement = document
        .create_element("canvas")
        .ok()?
        .dyn_into::<HtmlCanvasElement>()
        .ok()?;
    let ratio = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0)
        .clamp(1.0, 4.0);
    canvas.set_width((f64::from(size) * ratio).round() as u32);
    canvas.set_height((f64::from(size) * ratio).round() as u32);
    draw(
        &canvas, points, colors, markers, highlight, size, size, ratio, None, false, legend,
    )?;
    canvas.to_data_url_with_type("image/png").ok()
}

/// Renders the embedding to an SVG string for download.
///
/// Pure string building: no DOM, so it can run in a web worker.
/// `size` is the square viewport side in logical pixels.
/// `progress` is called with a fraction (0.0 to 1.0) after each batch is
/// rendered, so the caller can drive a progress bar.
pub(crate) fn build_svg(
    points: &[f32],
    colors: &[String],
    markers: &[u8],
    highlight: Option<(&str, u8)>,
    legend: &[LegendEntry],
    size: u32,
    mut progress: impl FnMut(f32),
) -> Option<String> {
    use std::fmt::Write;

    let n = points.len() / 2;
    if n == 0 {
        return None;
    }

    // Compute transform and project all points.
    let transform = Transform::fit(points, size as f32, size as f32, MARGIN)?;
    let pixels: Vec<(f32, f32)> = points
        .chunks_exact(2)
        .map(|p| transform.project(p[0], p[1]))
        .collect();

    // Resolve per-point (color, marker).
    let effective_colors: Vec<&str> = (0..n)
        .map(|i| colors.get(i).map(|s| s.as_str()).unwrap_or(DEFAULT_COLOR))
        .collect();
    let effective_markers: Vec<u8> = (0..n)
        .map(|i| markers.get(i).copied().unwrap_or(0))
        .collect();

    // Apply highlight dimming.
    let final_colors: Vec<&str> = effective_colors
        .iter()
        .zip(effective_markers.iter())
        .map(|(c, m)| match highlight {
            Some((hc, hm)) if *c != hc || *m != hm => DIMMED_COLOR,
            _ => *c,
        })
        .collect();

    // Batch by (color, marker).
    let mut batches: Vec<(&str, u8, Vec<usize>)> = Vec::new();
    for index in 0..n {
        let color = final_colors[index];
        let marker = effective_markers[index];
        match batches
            .iter_mut()
            .find(|(c, m, _)| *c == color && *m == marker)
        {
            Some((_, _, indices)) => indices.push(index),
            None => batches.push((color, marker, vec![index])),
        }
    }

    // Greyed points first so the highlighted class paints on top.
    if highlight.is_some() {
        batches.sort_by_key(|(color, _, _)| *color != DIMMED_COLOR);
    }

    let use_simple = n > SHAPED_MARKER_LIMIT;
    let mut buf = String::new();

    // SVG header.
    let _ = write!(
        buf,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">"#,
    );

    // Render batches.
    let total_batches = batches.len();
    for (batch_i, (color, marker, indices)) in batches.iter().enumerate() {
        if use_simple {
            // Simple squares for large point counts.
            let half = 1.0; // 2x2 square centered on point.
            for &index in indices {
                let (x, y) = pixels[index];
                let _ = write!(
                    buf,
                    r#"<rect x="{:.2}" y="{:.2}" width="2" height="2" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                    x - half,
                    y - half,
                    escape_attr(color),
                    FILL_ALPHA,
                    escape_attr(color),
                );
            }
        } else {
            // Shaped markers.
            let radius = 3.0;
            for &index in indices {
                let (x, y) = pixels[index];
                let escaped_color = escape_attr(color);
                match marker {
                    0 => {
                        // Circle.
                        let _ = write!(
                            buf,
                            r#"<circle cx="{:.2}" cy="{:.2}" r="{radius}" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                            x, y, escaped_color, FILL_ALPHA, escaped_color,
                        );
                    }
                    1 => {
                        // Triangle up.
                        let _ = write!(
                            buf,
                            r#"<path d="M{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                            x,
                            y - radius,
                            x + 0.866 * radius,
                            y + 0.5 * radius,
                            x - 0.866 * radius,
                            y + 0.5 * radius,
                            escaped_color,
                            FILL_ALPHA,
                            escaped_color,
                        );
                    }
                    2 => {
                        // Square.
                        let _ = write!(
                            buf,
                            r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                            x - radius,
                            y - radius,
                            2.0 * radius,
                            2.0 * radius,
                            escaped_color,
                            FILL_ALPHA,
                            escaped_color,
                        );
                    }
                    3 => {
                        // Diamond.
                        let _ = write!(
                            buf,
                            r#"<path d="M{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                            x,
                            y - radius,
                            x + radius,
                            y,
                            x,
                            y + radius,
                            x - radius,
                            y,
                            escaped_color,
                            FILL_ALPHA,
                            escaped_color,
                        );
                    }
                    4 => {
                        // Plus.
                        let arm = 0.4 * radius;
                        let _ = write!(
                            buf,
                            r#"<path d="M{:.2},{:.2} h{:.2} v{:.2} h-{:.2} Z M{:.2},{:.2} h{:.2} v{:.2} h-{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                            x - radius,
                            y - arm,
                            2.0 * radius,
                            2.0 * arm,
                            2.0 * radius,
                            x - arm,
                            y - radius,
                            2.0 * arm,
                            2.0 * radius,
                            2.0 * arm,
                            escaped_color,
                            FILL_ALPHA,
                            escaped_color,
                        );
                    }
                    5 => {
                        // Triangle down.
                        let _ = write!(
                            buf,
                            r#"<path d="M{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                            x,
                            y + radius,
                            x + 0.866 * radius,
                            y - 0.5 * radius,
                            x - 0.866 * radius,
                            y - 0.5 * radius,
                            escaped_color,
                            FILL_ALPHA,
                            escaped_color,
                        );
                    }
                    _ => {
                        // Fallback to circle.
                        let _ = write!(
                            buf,
                            r#"<circle cx="{:.2}" cy="{:.2}" r="{radius}" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                            x, y, escaped_color, FILL_ALPHA, escaped_color,
                        );
                    }
                }
            }
        }

        // Report progress after each batch.
        progress((batch_i as f32 + 1.0) / total_batches as f32);
    }

    // Legend sidebar.
    if !legend.is_empty() {
        build_legend_svg(&mut buf, legend, size, use_simple);
    }

    let _ = write!(buf, "</svg>");
    Some(buf)
}

/// Escapes a string for use as an SVG attribute value.
fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Appends the legend sidebar SVG elements to the buffer.
fn build_legend_svg(buf: &mut String, entries: &[LegendEntry], size: u32, uniform_marker: bool) {
    use std::fmt::Write;

    const SVG_LEGEND_FONT_SIZE: f32 = 13.0;
    const LEGEND_SEPARATOR: &str = "#dddddd";
    const LEGEND_TEXT_COLOR: &str = "#1a1a1a";

    let max_rows = ((size as f32 - 2.0 * LEGEND_PAD) / LEGEND_ROW_H).floor() as usize;
    let max_rows = max_rows.max(1);
    let (shown, overflow) = if entries.len() > max_rows {
        (max_rows - 1, entries.len() - (max_rows - 1))
    } else {
        (entries.len(), 0)
    };
    let rows = shown + usize::from(overflow > 0);

    // Estimate strip width: marker column + padding + approximate text width.
    let max_label_len = entries
        .iter()
        .take(shown)
        .map(|e| e.label.len())
        .max()
        .unwrap_or(0);
    let approx_text_w = (max_label_len as f32) * (SVG_LEGEND_FONT_SIZE * 0.6);
    let strip_w = (LEGEND_PAD * 2.0 + LEGEND_MARKER_COL + approx_text_w)
        .clamp(LEGEND_MIN_WIDTH, (size as f32) * 0.4);

    // Separator line.
    let _ = write!(
        buf,
        r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="{}" stroke-width="1"/>"#,
        strip_w,
        LEGEND_PAD,
        strip_w,
        size as f32 - LEGEND_PAD,
        LEGEND_SEPARATOR,
    );

    let marker_x = LEGEND_PAD + LEGEND_MARKER_COL / 2.0;
    let label_x = LEGEND_PAD + LEGEND_MARKER_COL;

    for (i, entry) in entries.iter().take(shown).enumerate() {
        let y = (size as f32 - rows as f32 * LEGEND_ROW_H) / 2.0
            + i as f32 * LEGEND_ROW_H
            + LEGEND_ROW_H / 2.0;
        let escaped_color = escape_attr(&entry.color);
        let escaped_label = escape_attr(&entry.label);

        // Marker glyph in the legend.
        let marker = if uniform_marker {
            0u8
        } else {
            entry.marker as u8
        };
        let legend_radius = 5.0;
        match marker {
            0 => {
                let _ = write!(
                    buf,
                    r#"<circle cx="{:.2}" cy="{:.2}" r="{legend_radius}" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                    marker_x, y, escaped_color, FILL_ALPHA, escaped_color,
                );
            }
            1 => {
                let _ = write!(
                    buf,
                    r#"<path d="M{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                    marker_x,
                    y - legend_radius,
                    marker_x + 0.866 * legend_radius,
                    y + 0.5 * legend_radius,
                    marker_x - 0.866 * legend_radius,
                    y + 0.5 * legend_radius,
                    escaped_color,
                    FILL_ALPHA,
                    escaped_color,
                );
            }
            2 => {
                let _ = write!(
                    buf,
                    r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                    marker_x - legend_radius,
                    y - legend_radius,
                    2.0 * legend_radius,
                    2.0 * legend_radius,
                    escaped_color,
                    FILL_ALPHA,
                    escaped_color,
                );
            }
            3 => {
                let _ = write!(
                    buf,
                    r#"<path d="M{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                    marker_x,
                    y - legend_radius,
                    marker_x + legend_radius,
                    y,
                    marker_x,
                    y + legend_radius,
                    marker_x - legend_radius,
                    y,
                    escaped_color,
                    FILL_ALPHA,
                    escaped_color,
                );
            }
            4 => {
                let arm = 0.4 * legend_radius;
                let _ = write!(
                    buf,
                    r#"<path d="M{:.2},{:.2} h{:.2} v{:.2} h-{:.2} Z M{:.2},{:.2} h{:.2} v{:.2} h-{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                    marker_x - legend_radius,
                    y - arm,
                    2.0 * legend_radius,
                    2.0 * arm,
                    2.0 * legend_radius,
                    marker_x - arm,
                    y - legend_radius,
                    2.0 * arm,
                    2.0 * legend_radius,
                    2.0 * arm,
                    escaped_color,
                    FILL_ALPHA,
                    escaped_color,
                );
            }
            5 => {
                let _ = write!(
                    buf,
                    r#"<path d="M{:.2},{:.2} L{:.2},{:.2} L{:.2},{:.2} Z" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="1"/>"#,
                    marker_x,
                    y + legend_radius,
                    marker_x + 0.866 * legend_radius,
                    y - 0.5 * legend_radius,
                    marker_x - 0.866 * legend_radius,
                    y - 0.5 * legend_radius,
                    escaped_color,
                    FILL_ALPHA,
                    escaped_color,
                );
            }
            _ => {}
        }

        // Label text.
        let _ = write!(
            buf,
            r#"<text x="{:.2}" y="{:.2}" fill="{}" font-size="{SVG_LEGEND_FONT_SIZE}" font-family="system-ui, -apple-system, sans-serif" dominant-baseline="central">{escaped_label}</text>"#,
            label_x, y, LEGEND_TEXT_COLOR,
        );
    }

    if overflow > 0 {
        let overflow_y = (size as f32 - rows as f32 * LEGEND_ROW_H) / 2.0
            + shown as f32 * LEGEND_ROW_H
            + LEGEND_ROW_H / 2.0;
        let _ = write!(
            buf,
            r#"<text x="{:.2}" y="{:.2}" fill="{}" font-size="{SVG_LEGEND_FONT_SIZE}" font-family="system-ui, -apple-system, sans-serif" dominant-baseline="central">+{overflow} more</text>"#,
            label_x, overflow_y, LEGEND_TEXT_COLOR,
        );
    }
}

/// In-progress drag of a single point: the pointer that started it, the index
/// of the dragged point and the transform frozen for the drag's duration.
#[derive(Clone, Copy)]
struct DragState {
    pointer_id: i32,
    index: usize,
    transform: Transform,
}

/// Converts an element-relative CSS pixel into a canvas buffer pixel, also
/// returning the buffer-per-CSS scale (the canvas may be stretched by CSS).
fn to_buffer(
    canvas: &HtmlCanvasElement,
    x: f64,
    y: f64,
    width: u32,
    height: u32,
) -> ((f32, f32), f32) {
    let rect = canvas.get_bounding_client_rect();
    let scale_x = if rect.width() > 0.0 {
        f64::from(width) / rect.width()
    } else {
        1.0
    };
    let scale_y = if rect.height() > 0.0 {
        f64::from(height) / rect.height()
    } else {
        1.0
    };
    (((x * scale_x) as f32, (y * scale_y) as f32), scale_x as f32)
}

/// Canvas scatter plot of a row major `n x 2` embedding, redrawn whenever the
/// embedding signal changes and rescaled to the viewport on every redraw.
///
/// # Props
///
/// * `embedding` - the points to draw, cleared when `None`.
/// * `colors` - optional CSS color per point, see [`crate::colorize`]. Points
///   fall back to a single default color when absent or of mismatched length.
/// * `markers` - optional marker shape per point, see [`crate::colorize`].
///   Points fall back to circles when absent or of mismatched length.
/// * `draggable` - when present and true, points can be dragged with the
///   pointer, each move reported through `on_point_moved` in data space.
/// * `on_point_moved` - called with `(index, x, y)` data-space coordinates as a
///   point is dragged. The owner applies the move to the embedding signal.
/// * `on_drag_start` - called with the grabbed point index when a drag begins,
///   before any move. The owner can pause a running computation here.
/// * `on_drag_end` - called when a drag ends (pointer up or cancel). The owner
///   can resume a paused computation here.
/// * `width` - logical canvas width in pixels, defaults to 800.
/// * `height` - logical canvas height in pixels, defaults to 600.
/// * `pixel_ratio` - backing buffer resolution multiplier over the logical
///   size, for crisp rendering on high-DPI displays. Defaults to the device
///   pixel ratio when absent. Clamped to `[1, 4]`.
#[component]
pub fn ScatterPlot(
    embedding: ReadSignal<Option<Vec<f32>>>,
    #[props(default = None)] colors: Option<ReadSignal<Option<Vec<String>>>>,
    #[props(default = None)] markers: Option<ReadSignal<Option<Vec<Marker>>>>,
    #[props(default = None)] highlight: Option<ReadSignal<Option<(String, Marker)>>>,
    #[props(default = None)] draggable: Option<ReadSignal<bool>>,
    #[props(default = None)] on_point_moved: Option<EventHandler<(usize, f32, f32)>>,
    #[props(default = None)] on_drag_start: Option<EventHandler<usize>>,
    #[props(default = None)] on_drag_end: Option<EventHandler<()>>,
    #[props(default = 800)] width: u32,
    #[props(default = 600)] height: u32,
    #[props(default = None)] pixel_ratio: Option<f64>,
) -> Element {
    // Render the backing buffer at the device pixel ratio (or an explicit
    // override) so the plot is sharp on high-DPI screens.
    let ratio = pixel_ratio
        .or_else(|| web_sys::window().map(|w| w.device_pixel_ratio()))
        .unwrap_or(1.0)
        .clamp(1.0, 4.0);
    let buffer_width = (f64::from(width) * ratio).round() as u32;
    let buffer_height = (f64::from(height) * ratio).round() as u32;

    // `width`/`height` are plain props, not signals, so the redraw effect below
    // would not re-run when the viewport (and thus the canvas size) changes,
    // leaving the resized canvas blank. Funnel them through a memo the effect
    // reads, so a resize repaints at the new size.
    let size = use_memo(use_reactive!(|(width, height)| (width, height)));

    let mut canvas = use_signal(|| None::<HtmlCanvasElement>);
    let mut drag = use_signal(|| None::<DragState>);
    // The transform of the last draw, so pointer handlers hit-test and
    // unproject against what is on screen. A plain RefCell, not a signal, so
    // writing it from the draw effect cannot retrigger that effect.
    let last_transform = use_hook(|| Rc::new(RefCell::new(None::<Transform>)));

    // Redraws when the canvas mounts or the embedding, coloring or drag
    // changes. During a drag the frozen transform is reused so only the
    // dragged point moves; clearing the drag refits once at release.
    let redraw_transform = last_transform.clone();
    use_effect(move || {
        // Read the size first so a resize is always a dependency, even before
        // the canvas has mounted.
        let (width, height) = size();
        let Some(canvas) = canvas() else {
            return;
        };
        let colors = colors.map(|c| c.read().clone()).unwrap_or_default();
        let markers = markers.map(|m| m.read().clone()).unwrap_or_default();
        let highlight = highlight.and_then(|h| h.read().clone());
        let override_transform = drag().map(|state| state.transform);
        let used = match embedding.read().as_ref() {
            Some(points) => draw(
                &canvas,
                points,
                colors.as_deref(),
                markers.as_deref(),
                highlight.as_ref().map(|(c, m)| (c.as_str(), *m)),
                width,
                height,
                ratio,
                override_transform,
                true,
                None,
            ),
            None => {
                if let Some(context) = canvas
                    .get_context("2d")
                    .ok()
                    .flatten()
                    .and_then(|c| c.dyn_into::<CanvasRenderingContext2d>().ok())
                {
                    let _ = context.set_transform(ratio, 0.0, 0.0, ratio, 0.0, 0.0);
                    context.clear_rect(0.0, 0.0, f64::from(width), f64::from(height));
                }
                None
            }
        };
        *redraw_transform.borrow_mut() = used;
    });

    let is_draggable = move || draggable.is_some_and(|d| d());
    let down_transform = last_transform.clone();

    rsx! {
        canvas {
            id: "scatter-plot",
            class: if is_draggable() { "decompositions-plot decompositions-plot--draggable" } else { "decompositions-plot" },
            width: "{buffer_width}",
            height: "{buffer_height}",
            onmounted: move |evt| {
                canvas.set(
                    evt.data()
                        .downcast::<web_sys::Element>()
                        .and_then(|element| {
                            element.clone().dyn_into::<HtmlCanvasElement>().ok()
                        }),
                );
            },
            onpointerdown: move |evt| {
                if !is_draggable() || on_point_moved.is_none() {
                    return;
                }
                let Some(canvas) = canvas() else {
                    return;
                };
                let transform = match *down_transform.borrow() {
                    Some(transform) => transform,
                    None => return,
                };
                let guard = embedding.read();
                let Some(points) = guard.as_ref() else {
                    return;
                };
                let location = evt.data().element_coordinates();
                let ((px, py), scale) = to_buffer(&canvas, location.x, location.y, width, height);
                // Pick the nearest point within roughly 8 on-screen pixels.
                let threshold = 8.0 * scale;
                let mut best: Option<(usize, f32)> = None;
                for (index, point) in points.chunks_exact(2).enumerate() {
                    let (qx, qy) = transform.project(point[0], point[1]);
                    let distance = ((qx - px).powi(2) + (qy - py).powi(2)).sqrt();
                    if distance <= threshold && best.is_none_or(|(_, b)| distance < b) {
                        best = Some((index, distance));
                    }
                }
                let Some((index, _)) = best else {
                    return;
                };
                drop(guard);
                evt.prevent_default();
                let pointer_id = evt.data().pointer_id();
                let _ = canvas.set_pointer_capture(pointer_id);
                drag.set(Some(DragState { pointer_id, index, transform }));
                if let Some(handler) = on_drag_start {
                    handler.call(index);
                }
            },
            onpointermove: move |evt| {
                let Some(state) = drag() else {
                    return;
                };
                if evt.data().pointer_id() != state.pointer_id {
                    return;
                }
                let Some(handler) = on_point_moved else {
                    return;
                };
                let Some(canvas) = canvas() else {
                    return;
                };
                let location = evt.data().element_coordinates();
                let ((px, py), _) = to_buffer(&canvas, location.x, location.y, width, height);
                let (x, y) = state.transform.unproject(px, py);
                handler.call((state.index, x, y));
            },
            onpointerup: move |_| {
                if let Some(state) = drag() {
                    if let Some(canvas) = canvas() {
                        let _ = canvas.release_pointer_capture(state.pointer_id);
                    }
                    drag.set(None);
                    if let Some(handler) = on_drag_end {
                        handler.call(());
                    }
                }
            },
            onpointercancel: move |_| {
                if let Some(state) = drag() {
                    if let Some(canvas) = canvas() {
                        let _ = canvas.release_pointer_capture(state.pointer_id);
                    }
                    drag.set(None);
                    if let Some(handler) = on_drag_end {
                        handler.call(());
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Transform, project_to_viewport};

    #[test]
    fn unproject_round_trips_project() {
        let points = [-250.0, 1000.0, 480.0, -900.0, 0.0, 30.0];
        let transform = Transform::fit(&points, 800.0, 600.0, 12.0).unwrap();
        for point in points.chunks_exact(2) {
            let (px, py) = transform.project(point[0], point[1]);
            let (x, y) = transform.unproject(px, py);
            assert!((x - point[0]).abs() < 1e-2, "x: {x} vs {}", point[0]);
            assert!((y - point[1]).abs() < 1e-2, "y: {y} vs {}", point[1]);
        }
    }

    #[test]
    fn degenerate_span_still_unprojects() {
        // A single point has a zero span: fit must not divide by zero and the
        // round trip must still recover the coordinate.
        let transform = Transform::fit(&[7.0, 7.0], 400.0, 300.0, 10.0).unwrap();
        let (px, py) = transform.project(7.0, 7.0);
        assert!((px - 200.0).abs() < 1.0 && (py - 150.0).abs() < 1.0);
        let (x, y) = transform.unproject(px, py);
        assert!((x - 7.0).abs() < 1e-2 && (y - 7.0).abs() < 1e-2);
    }

    #[test]
    fn empty_embedding_has_no_transform() {
        assert!(Transform::fit(&[], 400.0, 300.0, 10.0).is_none());
    }

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

    #[test]
    fn outliers_do_not_collapse_the_fit() {
        let mut points = Vec::new();
        // A tight cluster of 200 points near the origin.
        for i in 0..200 {
            let t = i as f32 / 200.0;
            points.push(t);
            points.push(-t);
        }
        // A few far stragglers, as the first epochs of a run throw out.
        for _ in 0..4 {
            points.push(5000.0);
            points.push(5000.0);
        }

        let transform = Transform::fit(&points, 800.0, 600.0, 12.0).unwrap();
        // A cluster point lands inside the viewport, fit to the bulk, not the
        // stragglers (which would otherwise squish it to the center).
        let (cx, cy) = transform.project(0.5, -0.5);
        assert!(
            (12.0..=788.0).contains(&cx) && (12.0..=588.0).contains(&cy),
            "cluster point should fill the viewport, got {cx},{cy}"
        );
        // A straggler is pushed well off-screen (clipped) instead.
        let (ox, _) = transform.project(5000.0, 5000.0);
        assert!(ox > 788.0, "straggler should clip off-screen, got {ox}");
    }
}
