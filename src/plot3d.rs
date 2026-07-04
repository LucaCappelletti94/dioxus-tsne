//! WebGL2 3-D scatter plot for t-SNE embeddings, in pure Rust.
//!
//! Renders the embedding as point sprites with an orbit camera. Left or middle
//! drag rotates, right drag pans, the wheel zooms. Everything runs off
//! `web_sys::WebGl2RenderingContext`, no JS glue and no `requestAnimationFrame`
//! loop: the [`use_effect`] in [`ScatterPlot3D`] uploads the vertex buffers and
//! issues the draw call synchronously on every signal change, so the
//! `Float32Array::view` over wasm memory is always consumed within the same
//! tick that produced it.
//!
//! The previous JS-backed renderer left the canvas blank because
//! `mat4Mul(proj, view)` used row-major indexing on column-major arrays,
//! which computes `view * proj` and sends every point through a degenerate
//! `w = 0` after the perspective divide. The [`mat4_mul`] here is the
//! column-major product the projection actually needs.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader,
    WebGlUniformLocation,
};

use crate::color::Marker;

/// Vertex shader. Fixed-size point sprites, colored per vertex. Written in
/// GLSL ES 1.00 syntax, which WebGL2 accepts when no `#version` directive is
/// present and keeps the shader readable.
const VERTEX_SHADER: &str = r#"attribute vec3 a_position;
attribute vec4 a_color;
uniform mat4 u_matrix;
uniform float u_inv_extent;
varying vec4 v_color;
void main() {
    gl_Position = u_matrix * vec4(a_position * u_inv_extent, 1.0);
    gl_PointSize = 5.0;
    v_color = a_color;
}
"#;

/// Fragment shader. Discards fragments outside a unit disk in point-sprite
/// coordinates so points read as filled circles rather than squares.
const FRAGMENT_SHADER: &str = r#"precision mediump float;
varying vec4 v_color;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (length(c) > 0.5) discard;
    gl_FragColor = v_color;
}
"#;

/// Fallback point color when the color signal is missing or malformed.
/// Matches the 2-D scatter's `#1f77b4`.
const DEFAULT_COLOR: [f32; 3] = [
    0x1f as f32 / 255.0,
    0x77 as f32 / 255.0,
    0xb4 as f32 / 255.0,
];

/// Alpha of point fills, matching the 2-D scatter so dense regions read as
/// darker where points overlap.
const FILL_ALPHA: f32 = 0.82;

/// Grey applied to points outside the focused class when the legend
/// highlights one entry, matching the 2-D scatter's `#d8d8d8`.
const DIMMED_COLOR: [f32; 3] = [0.851, 0.851, 0.851];

/// Framebuffer clear color, matching the 2-D plot backgrounds so a switch
/// between 2 and 3 dimensions does not flash a different backdrop.
const BACKGROUND_LIGHT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const BACKGROUND_DARK: [f32; 4] = [
    0x0a as f32 / 255.0,
    0x0a as f32 / 255.0,
    0x0a as f32 / 255.0,
    1.0,
];

/// How close two RGB triples have to be, per channel, to count as the same
/// legend color for the highlight check.
const COLOR_MATCH_EPS: f32 = 0.01;

/// Half-angle of the perspective field of view, in radians.
const FOV_HALF_ANGLE: f32 = 0.4;

/// Perspective clip planes. Fixed rather than scaled with the framing
/// extent because the vertex shader normalizes positions by `1 / extent`
/// (`u_inv_extent`): the projection sees the data in a unit-sized box
/// regardless of how bhtsne has stretched clusters apart, so origin lands
/// at the same NDC-z every frame and neither the "70% of the run,
/// everything clips against far" bug nor the "tight-bulk clips against
/// near" bug that scaling the planes would race against numerically for
/// tiny extents can happen. The pair below spans a comfortable frustum
/// around the default camera distance of `zoom = 1.5`, with room for
/// wheel-zoom and pan.
const NEAR_PLANE: f32 = 0.01;
const FAR_PLANE: f32 = 100.0;

/// Orbit-camera parameter clamps.
const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 10.0;
const MAX_PITCH: f32 = 1.5;

/// Sensitivities: canvas-pixel-to-camera multipliers. Copied from the JS.
const ROT_PER_PIXEL: f32 = 0.005;
const PAN_PER_PIXEL: f32 = 0.01;
const ZOOM_PER_WHEEL: f32 = 0.001;

/// Interval (milliseconds) between key-driven rotation increments. About 60
/// Hz, matches the browser's typical rAF cadence closely enough that the
/// motion looks smooth without the closure churn a full rAF loop would need.
pub(crate) const KEY_TICK_MS: u32 = 16;

/// Radians the camera rotates per tick while a keyboard axis is held. Tuned
/// so a full 2*PI turn takes roughly 1.7 s, slow enough to read every angle
/// but fast enough to feel like the world is spinning under the key.
pub(crate) const KEY_ROT_PER_TICK: f32 = 0.06;

/// Fallback extent when the embedding is too small to bound. The value only
/// matters for the framing distance and is scaled by `zoom` immediately.
const FALLBACK_EXTENT: f32 = 2.0;

/// Multiple of the median distance from the center of mass beyond which a
/// point is left out of the framing bounding box. Matches the 2-D scatter's
/// factor so both plots reject the same set of stragglers, which is what
/// prevents a single point diverging mid-run from zooming the camera all
/// the way out and squishing every well-behaved point into sub-pixel dust.
const OUTLIER_FIT_FACTOR: f32 = 4.0;

/// Orbit-camera state. Held in a signal owned by the parent so the same
/// rotation persists across dimension switches (see
/// [`ScatterPlot3D::camera`] and [`project_to_display`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Pitch, radians. Clamped to `[-MAX_PITCH, MAX_PITCH]` by the drag
    /// handler, unclamped by the key ticker.
    pub rot_x: f32,
    /// Yaw, radians. Unclamped so the user can spin freely.
    pub rot_y: f32,
    /// Roll, radians. Driven by holding the `Z` key.
    pub rot_z: f32,
    /// 4-D rotation angle in the ZW plane, radians. Driven by holding the
    /// `W` key. Consumed both here (when [`ScatterPlot3D`] is mounted with
    /// a 4-D embedding at `display_dim = 4`) and by
    /// [`project_to_display`] on the CPU when the parent projects a 4-D
    /// embedding down to 3-D or 2-D.
    pub rot_w: f32,
    /// Distance multiplier applied to the data extent, so the framing
    /// stays scale-independent as the embedding grows over epochs.
    pub zoom: f32,
    /// View-space pan.
    pub pan_x: f32,
    pub pan_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            rot_x: 0.0,
            rot_y: 0.4,
            rot_z: 0.0,
            rot_w: 0.0,
            zoom: 1.5,
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }
}

/// What a drag is doing. Right-click drags pan, every other button drag
/// rotates.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DragMode {
    Rotate,
    Pan,
}

fn drag_mode(button: MouseButton) -> DragMode {
    if matches!(button, MouseButton::Secondary) {
        DragMode::Pan
    } else {
        DragMode::Rotate
    }
}

/// Snapshot of a drag in progress. The pointer position and camera at
/// pointerdown are frozen here, so pointermove computes total deltas
/// (`current - start`) against the frozen camera. Applying incremental
/// deltas per event would accumulate rounding and desynchronize from the
/// pointer, the bug the previous JS renderer had.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Drag {
    pointer_id: i32,
    mode: DragMode,
    start_x: f64,
    start_y: f64,
    start_camera: Camera,
}

/// Which rotation axis one keydown maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RotationAxis {
    X,
    Y,
    Z,
    /// The 4-D rotation the `W` key drives, in the ZW plane. Only
    /// affects the visible embedding when the parent projects a 4-D
    /// point set with [`project_to_display`].
    W,
}

/// Which of the four rotation axes are currently held down. Kept as flags
/// rather than a set so pressing more than one combines their per-tick
/// increments straightforwardly, matching the "hold two at once inclines
/// the rotation" behavior called out in the UX spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct HeldAxes {
    pub x: bool,
    pub y: bool,
    pub z: bool,
    pub w: bool,
}

impl HeldAxes {
    pub(crate) const fn any(self) -> bool {
        self.x || self.y || self.z || self.w
    }

    pub(crate) fn set(&mut self, axis: RotationAxis, value: bool) {
        match axis {
            RotationAxis::X => self.x = value,
            RotationAxis::Y => self.y = value,
            RotationAxis::Z => self.z = value,
            RotationAxis::W => self.w = value,
        }
    }
}

/// A compiled WebGL2 3-D scatter pipeline bound to one canvas.
struct Renderer {
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    pos_buf: WebGlBuffer,
    col_buf: WebGlBuffer,
    u_matrix: WebGlUniformLocation,
    u_inv_extent: WebGlUniformLocation,
    a_position: u32,
    a_color: u32,
}

impl Renderer {
    /// Compiles the shaders, links the program and creates the vertex
    /// buffers for `canvas`. Returns `None` when WebGL2 is unavailable or
    /// a GL object fails to allocate; the caller falls back to an empty
    /// canvas.
    fn new(canvas: &HtmlCanvasElement) -> Option<Self> {
        let gl = canvas
            .get_context("webgl2")
            .ok()
            .flatten()?
            .dyn_into::<WebGl2RenderingContext>()
            .ok()?;

        let vs = compile_shader(&gl, WebGl2RenderingContext::VERTEX_SHADER, VERTEX_SHADER)?;
        let fs = compile_shader(
            &gl,
            WebGl2RenderingContext::FRAGMENT_SHADER,
            FRAGMENT_SHADER,
        )?;
        let program = link_program(&gl, &vs, &fs)?;
        // Shaders are attached to the program; once linked their standalone
        // handles are no longer needed, so drop them right away.
        gl.delete_shader(Some(&vs));
        gl.delete_shader(Some(&fs));

        let a_position_i = gl.get_attrib_location(&program, "a_position");
        let a_color_i = gl.get_attrib_location(&program, "a_color");
        if a_position_i < 0 || a_color_i < 0 {
            gl.delete_program(Some(&program));
            return None;
        }
        let u_matrix = gl.get_uniform_location(&program, "u_matrix")?;
        let u_inv_extent = gl.get_uniform_location(&program, "u_inv_extent")?;

        let pos_buf = gl.create_buffer()?;
        let col_buf = gl.create_buffer()?;

        gl.enable(WebGl2RenderingContext::DEPTH_TEST);
        gl.enable(WebGl2RenderingContext::BLEND);
        gl.blend_func(
            WebGl2RenderingContext::SRC_ALPHA,
            WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
        );

        Some(Self {
            gl,
            program,
            pos_buf,
            col_buf,
            u_matrix,
            u_inv_extent,
            a_position: a_position_i as u32,
            a_color: a_color_i as u32,
        })
    }

    /// Clears the framebuffer to `bg` and returns. Used when there is
    /// nothing to draw (missing or malformed embedding), so a stale frame
    /// does not linger under the new emptiness.
    fn clear(&self, buffer_w: u32, buffer_h: u32, bg: [f32; 4]) {
        self.gl.viewport(0, 0, buffer_w as i32, buffer_h as i32);
        self.gl.clear_color(bg[0], bg[1], bg[2], bg[3]);
        self.gl.clear(
            WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT,
        );
    }

    /// Uploads `positions` (row-major `n * 3`) and `colors` (row-major
    /// `n * 4` RGBA) and draws one frame.
    ///
    /// The uploads use `Float32Array::view` over wasm linear memory. That is
    /// safe here only because no wasm allocation happens between the view
    /// creation and the `buffer_data_with_array_buffer_view` call: `bufferData`
    /// copies out of the view before the closure returns, so the view is done
    /// by the time control leaves this function. Deferring the upload to a
    /// later task (as the previous RAF-driven JS did) would risk uploading a
    /// detached buffer if wasm memory grew in between.
    #[allow(clippy::too_many_arguments)]
    fn draw(
        &self,
        positions: &[f32],
        colors: &[f32],
        buffer_w: u32,
        buffer_h: u32,
        camera: Camera,
        extent: f32,
        bg: [f32; 4],
    ) {
        self.gl.viewport(0, 0, buffer_w as i32, buffer_h as i32);
        self.gl.clear_color(bg[0], bg[1], bg[2], bg[3]);
        self.gl.clear(
            WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT,
        );
        self.gl.use_program(Some(&self.program));

        let aspect = buffer_w as f32 / buffer_h.max(1) as f32;
        let mvp = mvp_matrix(camera, aspect);
        self.gl
            .uniform_matrix4fv_with_f32_array(Some(&self.u_matrix), false, &mvp);
        // Normalize points into a unit-sized box before the projection sees
        // them, so the perspective divide stays scale-invariant even when
        // bhtsne has spread clusters across many world units.
        self.gl
            .uniform1f(Some(&self.u_inv_extent), 1.0 / extent.max(f32::EPSILON));

        self.upload_attribute(&self.pos_buf, positions, self.a_position, 3);
        self.upload_attribute(&self.col_buf, colors, self.a_color, 4);

        self.gl.draw_arrays(
            WebGl2RenderingContext::POINTS,
            0,
            (positions.len() / 3) as i32,
        );
    }

    /// Uploads `data` into `buffer` and points `location` at it as
    /// `components` floats per vertex.
    ///
    /// SAFETY: `Float32Array::view` aliases wasm linear memory. `bufferData`
    /// copies out immediately, and this function performs no other
    /// allocation between the view and the copy, so the view cannot detach.
    fn upload_attribute(&self, buffer: &WebGlBuffer, data: &[f32], location: u32, components: i32) {
        self.gl
            .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(buffer));
        // SAFETY: see the doc comment above.
        unsafe {
            let view = js_sys::Float32Array::view(data);
            self.gl.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &view,
                WebGl2RenderingContext::DYNAMIC_DRAW,
            );
        }
        self.gl.enable_vertex_attrib_array(location);
        self.gl.vertex_attrib_pointer_with_i32(
            location,
            components,
            WebGl2RenderingContext::FLOAT,
            false,
            0,
            0,
        );
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Explicit teardown so unmounting the component does not leak
        // program + buffer handles until the next JS GC cycle.
        self.gl.delete_buffer(Some(&self.pos_buf));
        self.gl.delete_buffer(Some(&self.col_buf));
        self.gl.delete_program(Some(&self.program));
    }
}

/// Compiles a shader, deleting it and returning `None` on failure.
fn compile_shader(gl: &WebGl2RenderingContext, kind: u32, src: &str) -> Option<WebGlShader> {
    let shader = gl.create_shader(kind)?;
    gl.shader_source(&shader, src);
    gl.compile_shader(&shader);
    if gl
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Some(shader)
    } else {
        gl.delete_shader(Some(&shader));
        None
    }
}

/// Links a program, deleting it and returning `None` on failure.
fn link_program(
    gl: &WebGl2RenderingContext,
    vs: &WebGlShader,
    fs: &WebGlShader,
) -> Option<WebGlProgram> {
    let program = gl.create_program()?;
    gl.attach_shader(&program, vs);
    gl.attach_shader(&program, fs);
    gl.link_program(&program);
    if gl
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Some(program)
    } else {
        gl.delete_program(Some(&program));
        None
    }
}

/// Column-major 4x4 matrix product.
///
/// In column-major storage `a[c * 4 + r] == A[r][c]`, so the mathematical
/// product `C = A * B` where `C[r][c] = sum_k A[r][k] * B[k][c]` becomes
/// `c[c * 4 + r] = sum_k a[k * 4 + r] * b[c * 4 + k]`. The previous
/// JS-backed renderer indexed the same arrays row-major and, when handed
/// column-major projection and view matrices, effectively multiplied
/// `view * proj` instead of `proj * view`, sending every point to
/// `w = 0` at the perspective divide.
fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            r[col * 4 + row] = sum;
        }
    }
    r
}

/// Column-major perspective projection with fixed clip planes. Positions
/// are normalized in the vertex shader (`u_inv_extent`), so the projection
/// itself does not depend on how far apart bhtsne has pushed the
/// clusters; see the [`NEAR_PLANE`] and [`FAR_PLANE`] documentation.
fn projection_matrix(aspect: f32) -> [f32; 16] {
    let tf = FOV_HALF_ANGLE.tan();
    let inv_range = 1.0 / (FAR_PLANE - NEAR_PLANE);
    let q = -(FAR_PLANE + NEAR_PLANE) * inv_range;
    let qn = -2.0 * FAR_PLANE * NEAR_PLANE * inv_range;
    [
        1.0 / (aspect * tf),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / tf,
        0.0,
        0.0,
        0.0,
        0.0,
        q,
        -1.0,
        0.0,
        0.0,
        qn,
        0.0,
    ]
}

/// Column-major view matrix for the orbit camera. The camera sits at world
/// Z `= zoom` (in normalized units, since the vertex shader has already
/// divided positions by the framing extent).
///
/// The rotation is `Ry(yaw) * Rx(pitch) * Rz(roll)`, applied to a position
/// as `Ry * Rx * Rz * pos`. Composed from the three proper axis rotation
/// matrices via [`mat4_mul`], so it stays orthogonal for every combination
/// of the three angles. The hand-crafted matrix that used to live here
/// (ported verbatim from the JS renderer) only happened to be orthogonal
/// when `rot_x = 0`; as soon as pitch and yaw were both non-zero it
/// degenerated to a shear that stretched the world-Z column by
/// `sqrt(1 + sin(rot_x)^2)`, which read as "one axis is squished" once the
/// user dragged past the default view.
fn view_matrix(camera: Camera) -> [f32; 16] {
    let rot = mat4_mul(
        &rotation_y(camera.rot_y),
        &mat4_mul(&rotation_x(camera.rot_x), &rotation_z(camera.rot_z)),
    );
    let mut m = rot;
    // Both factors are pure rotations with an identity translation column,
    // so `mat4_mul` also left the translation column at identity: install
    // pan and camera-Z directly.
    m[12] = -camera.pan_x;
    m[13] = -camera.pan_y;
    m[14] = -camera.zoom;
    m
}

/// Column-major 4x4 rotation around the world X axis by `angle` radians.
fn rotation_x(angle: f32) -> [f32; 16] {
    let c = angle.cos();
    let s = angle.sin();
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, c, s, 0.0, //
        0.0, -s, c, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Column-major 4x4 rotation around the world Y axis by `angle` radians.
fn rotation_y(angle: f32) -> [f32; 16] {
    let c = angle.cos();
    let s = angle.sin();
    [
        c, 0.0, -s, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        s, 0.0, c, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Column-major 4x4 rotation around the world Z axis by `angle` radians.
fn rotation_z(angle: f32) -> [f32; 16] {
    let c = angle.cos();
    let s = angle.sin();
    [
        c, s, 0.0, 0.0, //
        -s, c, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// The MVP matrix (column-major) applied to each normalized vertex.
fn mvp_matrix(camera: Camera, aspect: f32) -> [f32; 16] {
    mat4_mul(&projection_matrix(aspect), &view_matrix(camera))
}

/// Diagonal of the framing bounding box.
///
/// A single point diverging mid-run (bhtsne can send one flying while the
/// bulk of the embedding is still converging) would otherwise blow up the
/// axis-aligned bounding box, pull the camera back proportionally, and
/// collapse every well-behaved point into sub-pixels: the ~70% mark of a
/// run where the "everything disappears" symptom shows up. Matching the
/// 2-D scatter, the fit is robust: only points within
/// [`OUTLIER_FIT_FACTOR`] times the median distance from the center of
/// mass contribute to the bounding box, and non-finite coordinates
/// (NaN, +/-Inf) are skipped entirely because otherwise they poison the
/// mean, the median, and every downstream matrix.
fn data_extent(points: &[f32]) -> f32 {
    if points.len() < 3 {
        return FALLBACK_EXTENT;
    }

    let mut mean_x = 0.0f32;
    let mut mean_y = 0.0f32;
    let mut mean_z = 0.0f32;
    let mut finite_count = 0usize;
    for chunk in points.chunks_exact(3) {
        if chunk[0].is_finite() && chunk[1].is_finite() && chunk[2].is_finite() {
            mean_x += chunk[0];
            mean_y += chunk[1];
            mean_z += chunk[2];
            finite_count += 1;
        }
    }
    if finite_count == 0 {
        return FALLBACK_EXTENT;
    }
    let count = finite_count as f32;
    mean_x /= count;
    mean_y /= count;
    mean_z /= count;

    let dist_sq = |c: &[f32]| {
        let dx = c[0] - mean_x;
        let dy = c[1] - mean_y;
        let dz = c[2] - mean_z;
        dx * dx + dy * dy + dz * dz
    };

    // Median squared distance among finite points. A zero median (every
    // finite point sits on the mean) disables the cutoff so the empty-span
    // path below still runs and yields the epsilon extent.
    let mut distances: Vec<f32> = points
        .chunks_exact(3)
        .filter(|c| c[0].is_finite() && c[1].is_finite() && c[2].is_finite())
        .map(&dist_sq)
        .collect();
    let median = distances.len() / 2;
    distances.select_nth_unstable_by(median, f32::total_cmp);
    let median_sq = distances[median];
    let threshold_sq = if median_sq > 0.0 {
        median_sq * OUTLIER_FIT_FACTOR.powi(2)
    } else {
        f32::INFINITY
    };

    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for chunk in points.chunks_exact(3) {
        if !(chunk[0].is_finite() && chunk[1].is_finite() && chunk[2].is_finite()) {
            continue;
        }
        if dist_sq(chunk) > threshold_sq {
            continue;
        }
        for axis in 0..3 {
            if chunk[axis] < lo[axis] {
                lo[axis] = chunk[axis];
            }
            if chunk[axis] > hi[axis] {
                hi[axis] = chunk[axis];
            }
        }
    }
    if !lo[0].is_finite() {
        return FALLBACK_EXTENT;
    }
    let dx = hi[0] - lo[0];
    let dy = hi[1] - lo[1];
    let dz = hi[2] - lo[2];
    (dx * dx + dy * dy + dz * dz).sqrt().max(f32::EPSILON)
}

/// Rotation-only 3-D transform (column-major 4x4 with an identity
/// translation column). Composes `Ry(rot_y) * Rx(rot_x) * Rz(rot_z)` from
/// the same axis rotation factors [`view_matrix`] uses, so the CPU-side
/// projection here matches whatever the shader would do for the same
/// camera. `rot_w`, `zoom`, `pan_*` are untouched.
fn rotation_matrix(camera: Camera) -> [f32; 16] {
    mat4_mul(
        &rotation_y(camera.rot_y),
        &mat4_mul(&rotation_x(camera.rot_x), &rotation_z(camera.rot_z)),
    )
}

/// Projects a row-major `n * embedding_dim` embedding down to a row-major
/// `n * display_dim` embedding, applying the rotations `camera` describes.
///
/// The pipeline is:
///
/// * If `embedding_dim == 4`, apply the ZW rotation by `rot_w` and drop
///   the resulting `w`, yielding a 3-vector.
/// * If the target is 3-D, stop there and return the 3-vectors row-major.
/// * If the target is 2-D, apply the 3-D rotation
///   `Ry(rot_y) * Rx(rot_x) * Rz(rot_z)` and take the resulting `(x, y)`.
///
/// Returns `None` when the input is empty, not a multiple of
/// `embedding_dim`, or `display_dim > embedding_dim` (the toggle refuses
/// that combination, but the projection must degrade gracefully). Pass
/// through with no work when `display_dim == embedding_dim` and the
/// embedding is 2-D or 3-D, since neither of those cases uses `camera`.
/// The 4-D `display_dim == embedding_dim` path DOES apply the ZW rotation
/// so `rot_w` reads out of the shared camera signal into the visible
/// 3-vector, matching what [`ScatterPlot3D`] used to compute internally
/// when it received the raw 4-D embedding.
pub fn project_to_display(
    points: &[f32],
    embedding_dim: usize,
    display_dim: usize,
    camera: Camera,
) -> Option<Vec<f32>> {
    if embedding_dim == 0
        || display_dim == 0
        || display_dim > embedding_dim
        || !points.len().is_multiple_of(embedding_dim)
    {
        return None;
    }
    // Trivial: nothing to project.
    if embedding_dim == display_dim && embedding_dim != 4 {
        return Some(points.to_vec());
    }

    let cw = camera.rot_w.cos();
    let sw = camera.rot_w.sin();
    let rot3d = rotation_matrix(camera);
    let n = points.len() / embedding_dim;
    let mut out = Vec::with_capacity(n * display_dim);

    for chunk in points.chunks_exact(embedding_dim) {
        // Reduce the row to a 3-vector, applying the ZW rotation on the
        // way down from 4-D.
        let (x3, y3, z3) = match embedding_dim {
            2 => (chunk[0], chunk[1], 0.0),
            3 => (chunk[0], chunk[1], chunk[2]),
            _ => (chunk[0], chunk[1], cw * chunk[2] + sw * chunk[3]),
        };

        match display_dim {
            2 => {
                let x_rot = rot3d[0] * x3 + rot3d[4] * y3 + rot3d[8] * z3;
                let y_rot = rot3d[1] * x3 + rot3d[5] * y3 + rot3d[9] * z3;
                out.push(x_rot);
                out.push(y_rot);
            }
            _ => {
                out.push(x3);
                out.push(y3);
                out.push(z3);
            }
        }
    }
    Some(out)
}

/// Converts a 2-D delta expressed in the current display projection back into
/// a delta on the raw `embedding_dim`-wide row. The inverse of
/// [`project_to_display`] restricted to the two projected axes: the third
/// axis (view depth) is held at zero so the point does not slide in and out
/// of the current camera view.
///
/// * 2-D embedding: pass-through.
/// * 3-D embedding: applies the transpose of the 3-D rotation, spreading the
///   display-space `(dx, dy)` across the raw `(x, y, z)` axes.
/// * 4-D embedding: unrotates through the 3-D pipeline as above, then
///   distributes the resulting `dz3` across the raw `(z, w)` pair through
///   the inverse ZW rotation with `dw3 = 0`.
///
/// Returns a `[f32; 4]` of which only the first `embedding_dim` entries are
/// meaningful; the callers already know the row width they hand back to the
/// embedding, so a fixed-size buffer avoids one heap allocation per pointer
/// move.
pub fn unproject_display_delta(
    dx_display: f32,
    dy_display: f32,
    embedding_dim: usize,
    camera: Camera,
) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    match embedding_dim {
        2 => {
            out[0] = dx_display;
            out[1] = dy_display;
        }
        3 | 4 => {
            let rot3d = rotation_matrix(camera);
            // `project_to_display` reads rows 0 and 1 of `rot3d` as the display
            // basis (with the column-major indexing `rot3d[col * 4 + row]`).
            // Its transpose therefore has those same rows as its first two
            // columns, so unrotating `(dx, dy, 0)` back into the raw basis
            // reduces to a pair of two-term dot products.
            let dxr = rot3d[0] * dx_display + rot3d[1] * dy_display;
            let dyr = rot3d[4] * dx_display + rot3d[5] * dy_display;
            let dz3 = rot3d[8] * dx_display + rot3d[9] * dy_display;
            out[0] = dxr;
            out[1] = dyr;
            if embedding_dim == 3 {
                out[2] = dz3;
            } else {
                // Inverse ZW rotation with `dw3 = 0`. Matches the forward
                // pipeline in `project_to_display` where `z3 = cw*z + sw*w`.
                let cw = camera.rot_w.cos();
                let sw = camera.rot_w.sin();
                out[2] = cw * dz3;
                out[3] = sw * dz3;
            }
        }
        _ => {}
    }
    out
}

/// Parses `#rrggbb` (case-insensitive, leading `#` optional) into linear
/// `[r, g, b]` in `[0, 1]`. Returns `None` for anything else.
fn parse_hex_rgb(s: &str) -> Option<[f32; 3]> {
    let hex = s.strip_prefix('#').unwrap_or(s);
    if hex.len() != 6 {
        return None;
    }
    let bytes = u32::from_str_radix(hex, 16).ok()?;
    Some([
        ((bytes >> 16) & 0xFF) as f32 / 255.0,
        ((bytes >> 8) & 0xFF) as f32 / 255.0,
        (bytes & 0xFF) as f32 / 255.0,
    ])
}

fn color_matches(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() < COLOR_MATCH_EPS
        && (a[1] - b[1]).abs() < COLOR_MATCH_EPS
        && (a[2] - b[2]).abs() < COLOR_MATCH_EPS
}

/// Builds the per-vertex RGBA color buffer.
///
/// Points whose parsed color does not match the highlight color are
/// dimmed to [`DIMMED_COLOR`], so the focused legend entry keeps its real
/// color and everything else reads as context. Points with no color entry
/// (or a malformed one) fall back to [`DEFAULT_COLOR`].
fn colors_rgba(
    n: usize,
    colors: Option<&[String]>,
    highlight: Option<&(String, Marker)>,
) -> Vec<f32> {
    let hl_rgb = highlight.and_then(|(hex, _)| parse_hex_rgb(hex));
    let mut out = Vec::with_capacity(n * 4);
    for index in 0..n {
        let entry = colors.and_then(|v| v.get(index)).map(String::as_str);
        let mut rgb = entry.and_then(parse_hex_rgb).unwrap_or(DEFAULT_COLOR);
        if let Some(target) = hl_rgb
            && !color_matches(rgb, target)
        {
            rgb = DIMMED_COLOR;
        }
        out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], FILL_ALPHA]);
    }
    out
}

/// Whether the OS prefers a dark color scheme. Cached-free because a draw
/// runs on demand rather than every frame; matching the media query per
/// redraw picks up a theme switch without a page reload.
fn prefers_dark() -> bool {
    web_sys::window()
        .and_then(|w| w.match_media("(prefers-color-scheme: dark)").ok().flatten())
        .map(|mql| mql.matches())
        .unwrap_or(false)
}

fn background(dark: bool) -> [f32; 4] {
    if dark {
        BACKGROUND_DARK
    } else {
        BACKGROUND_LIGHT
    }
}

/// 3-D canvas scatter plot with orbit controls.
///
/// Left or middle-click drag rotates, right-click drag pans, wheel zooms.
/// Holding one of the letter keys spins the world at a fixed rate about
/// that axis: `X`, `Y`, `Z` for the three 3-D axes, and `W` for the ZW
/// plane rotation when a 4-D embedding is being rendered.
///
/// # Props
///
/// * `embedding` - the points to draw as a row-major `n * 3` matrix, cleared
///   when `None` or when the length is not a multiple of three. The parent
///   is responsible for projecting higher-D embeddings down to 3-D with
///   [`project_to_display`] first, using the same `camera` signal, so the
///   rotation stays consistent across dimensions.
/// * `camera` - shared rotation and view state. The parent owns the signal
///   so its rot/zoom/pan state survives dimension switches: this
///   component's pointer drag and keyboard handlers write into the same
///   signal that the parent's 2-D and 3-D projection memos read from.
/// * `colors` - optional CSS `#rrggbb` per point, matched against
///   `highlight` (if any) to dim the non-focused class. Fallback color is
///   the same shade the 2-D scatter uses.
/// * `markers` - accepted for prop-signature parity with `ScatterPlot`,
///   currently ignored: 3-D points render as uniform disks (per-vertex
///   shape sprites would need a second shader path).
/// * `highlight` - focused legend entry, matched by color only. Points
///   whose color does not match get dimmed.
/// * `width` / `height` - logical canvas size in CSS pixels.
/// * `pixel_ratio` - backing-buffer resolution multiplier, defaulting to
///   the device pixel ratio, clamped to `[1, 4]`.
#[component]
pub fn ScatterPlot3D(
    embedding: ReadSignal<Option<Vec<f32>>>,
    camera: Signal<Camera>,
    #[props(default = None)] colors: Option<ReadSignal<Option<Vec<String>>>>,
    #[props(default = None)] markers: Option<ReadSignal<Option<Vec<Marker>>>>,
    #[props(default = None)] highlight: Option<ReadSignal<Option<(String, Marker)>>>,
    #[props(default = 800)] width: u32,
    #[props(default = 600)] height: u32,
    #[props(default = None)] pixel_ratio: Option<f64>,
) -> Element {
    // Marker shapes are 2-D only in this crate; the prop is accepted so the
    // dispatch in `DecompositionView` can hand the same set of signals to
    // both scatter components.
    let _ = markers;

    let ratio = pixel_ratio
        .or_else(|| web_sys::window().map(|w| w.device_pixel_ratio()))
        .unwrap_or(1.0)
        .clamp(1.0, 4.0);
    let buffer_width = (f64::from(width) * ratio).round() as u32;
    let buffer_height = (f64::from(height) * ratio).round() as u32;

    // Buffer size is a plain (non-signal) input, so the redraw effect
    // depends on it only through this memo. Without it a resize would not
    // trigger a redraw.
    let size = use_memo(use_reactive!(|(buffer_width, buffer_height)| (
        buffer_width,
        buffer_height,
    )));

    let mut canvas = use_signal(|| None::<HtmlCanvasElement>);
    let mut renderer = use_signal(|| None::<Renderer>);
    let mut drag = use_signal(|| None::<Drag>);

    // Uploads the current embedding and draws one frame. Reactive on the
    // renderer, embedding, colors, highlight, camera and size signals.
    use_effect(move || {
        let (buffer_w, buffer_h) = size();
        let bg = background(prefers_dark());
        let renderer_guard = renderer.read();
        let Some(renderer) = renderer_guard.as_ref() else {
            return;
        };
        let cam = camera();
        let emb_guard = embedding.read();
        let Some(points) = emb_guard.as_ref() else {
            renderer.clear(buffer_w, buffer_h, bg);
            return;
        };
        if points.len() < 3 || points.len() % 3 != 0 {
            renderer.clear(buffer_w, buffer_h, bg);
            return;
        }
        let n = points.len() / 3;
        // Clone the current value out of each optional signal so the reactive
        // read is tracked without having to hold a Ref across the draw call.
        let colors_data: Option<Vec<String>> = colors.and_then(|c| c.read().clone());
        let highlight_data: Option<(String, Marker)> = highlight.and_then(|h| h.read().clone());
        let color_data = colors_rgba(n, colors_data.as_deref(), highlight_data.as_ref());
        // Higher-D embeddings are projected to 3-D in the parent (see
        // `project_to_display`), using the same `camera` signal, so the
        // `rot_w` reads out visibly here without the shader needing a
        // second code path.
        let extent = data_extent(points);
        renderer.draw(points, &color_data, buffer_w, buffer_h, cam, extent, bg);
    });

    rsx! {
        canvas {
            // Same id as the 2-D scatter so the MediaRecorder capture in
            // `DecompositionView` can grab this canvas by id in 3-D mode too.
            // Only one of the two scatters is mounted at a time, so there is
            // no duplicate-id conflict.
            id: "scatter-plot",
            class: {
                let mut c = String::from("decompositions-plot decompositions-plot--orbit");
                if drag().is_some() {
                    c.push_str(" decompositions-plot--grabbing");
                }
                c
            },
            width: "{buffer_width}",
            height: "{buffer_height}",
            onmounted: move |evt| {
                let element = evt
                    .data()
                    .downcast::<web_sys::Element>()
                    .and_then(|el| el.clone().dyn_into::<HtmlCanvasElement>().ok());
                if let Some(c) = element {
                    canvas.set(Some(c.clone()));
                    renderer.set(Renderer::new(&c));
                }
            },
            onpointerdown: move |evt| {
                let Some(canvas) = canvas() else {
                    return;
                };
                let data = evt.data();
                let coords = data.client_coordinates();
                let pointer_id = data.pointer_id();
                let button = data.trigger_button().unwrap_or_default();
                let state = Drag {
                    pointer_id,
                    mode: drag_mode(button),
                    start_x: coords.x,
                    start_y: coords.y,
                    start_camera: camera(),
                };
                // Pin the cursor to `grabbing` inline before pointer capture
                // for the same reason as the 2-D scatter (browsers freeze the
                // visible cursor at capture start; a later CSS class change
                // does not repaint). The inline override is cleared at
                // pointerup so CSS resumes control.
                let _ = canvas.style().set_property("cursor", "grabbing");
                let _ = canvas.set_pointer_capture(pointer_id);
                drag.set(Some(state));
                evt.prevent_default();
            },
            onpointermove: move |evt| {
                let Some(state) = drag() else {
                    return;
                };
                let data = evt.data();
                if data.pointer_id() != state.pointer_id {
                    return;
                }
                let coords = data.client_coordinates();
                let dx = (coords.x - state.start_x) as f32;
                let dy = (coords.y - state.start_y) as f32;
                let mut cam = state.start_camera;
                match state.mode {
                    DragMode::Rotate => {
                        cam.rot_x = (state.start_camera.rot_x - dy * ROT_PER_PIXEL)
                            .clamp(-MAX_PITCH, MAX_PITCH);
                        cam.rot_y = state.start_camera.rot_y - dx * ROT_PER_PIXEL;
                    }
                    DragMode::Pan => {
                        cam.pan_x = state.start_camera.pan_x + dx * PAN_PER_PIXEL;
                        cam.pan_y = state.start_camera.pan_y - dy * PAN_PER_PIXEL;
                    }
                }
                camera.set(cam);
                evt.prevent_default();
            },
            onpointerup: move |evt| {
                if let Some(state) = drag() {
                    if let Some(canvas) = canvas() {
                        let _ = canvas.style().remove_property("cursor");
                        let _ = canvas.release_pointer_capture(state.pointer_id);
                    }
                    drag.set(None);
                    evt.prevent_default();
                }
            },
            onpointercancel: move |evt| {
                if let Some(state) = drag() {
                    if let Some(canvas) = canvas() {
                        let _ = canvas.style().remove_property("cursor");
                        let _ = canvas.release_pointer_capture(state.pointer_id);
                    }
                    drag.set(None);
                    evt.prevent_default();
                }
            },
            onwheel: move |evt| {
                // WheelDelta comes in pixels / lines / pages. `strip_units`
                // hands us the raw scalar in the reported unit, which is
                // fine here because the multiplier is tuned to the JS
                // default (pixel deltas on desktop browsers).
                let dy = evt.data().delta().strip_units().y as f32;
                let mut cam = camera();
                cam.zoom = (cam.zoom + dy * ZOOM_PER_WHEEL).clamp(MIN_ZOOM, MAX_ZOOM);
                camera.set(cam);
                evt.prevent_default();
            },
            oncontextmenu: move |evt| {
                // Suppress the browser context menu so right-click drags
                // can pan without popping it.
                evt.prevent_default();
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Column-major identity.
    fn identity() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Column-major translation.
    fn translate(tx: f32, ty: f32, tz: f32) -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            tx, ty, tz, 1.0,
        ]
    }

    /// Column-major uniform scale.
    fn scale(s: f32) -> [f32; 16] {
        [
            s, 0.0, 0.0, 0.0, //
            0.0, s, 0.0, 0.0, //
            0.0, 0.0, s, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Apply a column-major matrix to a column-vector.
    fn apply(m: &[f32; 16], v: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for row in 0..4 {
            let mut sum = 0.0;
            for col in 0..4 {
                sum += m[col * 4 + row] * v[col];
            }
            out[row] = sum;
        }
        out
    }

    fn approx_eq(a: &[f32; 16], b: &[f32; 16]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn mat4_mul_identity_is_neutral() {
        let t = translate(1.0, 2.0, 3.0);
        assert!(approx_eq(&mat4_mul(&t, &identity()), &t));
        assert!(approx_eq(&mat4_mul(&identity(), &t), &t));
    }

    #[test]
    fn mat4_mul_translate_of_scale_scales_then_translates() {
        // T * S applied to a vector: scale first (S), then translate (T).
        // For v = (1, 1, 1, 1), S = *2 gives (2, 2, 2, 1), then T = +(1,2,3)
        // gives (3, 4, 5, 1).
        let m = mat4_mul(&translate(1.0, 2.0, 3.0), &scale(2.0));
        let out = apply(&m, [1.0, 1.0, 1.0, 1.0]);
        assert!((out[0] - 3.0).abs() < 1e-5, "x = {}", out[0]);
        assert!((out[1] - 4.0).abs() < 1e-5, "y = {}", out[1]);
        assert!((out[2] - 5.0).abs() < 1e-5, "z = {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-5, "w = {}", out[3]);
    }

    #[test]
    fn mat4_mul_is_not_commutative_the_right_way() {
        // T * S applied to origin -> translation (S kills nothing at origin).
        // S * T applied to origin -> S(T(0)) = S(t) = 2 * t.
        let ts = mat4_mul(&translate(1.0, 2.0, 3.0), &scale(2.0));
        let st = mat4_mul(&scale(2.0), &translate(1.0, 2.0, 3.0));
        let out_ts = apply(&ts, [0.0, 0.0, 0.0, 1.0]);
        let out_st = apply(&st, [0.0, 0.0, 0.0, 1.0]);
        assert!((out_ts[0] - 1.0).abs() < 1e-5);
        assert!((out_st[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn mvp_at_defaults_places_origin_in_front_of_camera() {
        // With the default camera (rot_y = 0.4, zoom = 1.5), a point at the
        // normalized-data origin should end up inside the clip volume:
        // |x|, |y|, |z| <= |w|. The pre-fix JS path produced w = 0 here
        // and clipped every point away, which is why the plot was blank.
        let mvp = mvp_matrix(Camera::default(), 1.0);
        let clip = apply(&mvp, [0.0, 0.0, 0.0, 1.0]);
        assert!(
            clip[3].abs() > 1e-3,
            "w should be finite and non-zero, got {}",
            clip[3]
        );
        assert!(clip[0].abs() <= clip[3].abs());
        assert!(clip[1].abs() <= clip[3].abs());
        assert!(clip[2].abs() <= clip[3].abs());
    }

    /// Applies the shader's normalization to `point` (dividing by
    /// `extent`) before returning `MVP * normalized_point`, so tests can
    /// exercise the same transform the GPU sees end-to-end.
    fn transform_world(camera: Camera, aspect: f32, extent: f32, point: [f32; 3]) -> [f32; 4] {
        let inv = 1.0 / extent.max(f32::EPSILON);
        let normalized = [point[0] * inv, point[1] * inv, point[2] * inv, 1.0];
        apply(&mvp_matrix(camera, aspect), normalized)
    }

    #[test]
    fn shader_normalization_keeps_origin_visible_across_extents() {
        // Reproduces the "70% into the run, everything disappears" bug:
        // with fixed absolute clip planes and no shader-side scaling, the
        // origin's clip-space z crossed +1 once `zoom * extent` outran the
        // far plane. The shader now divides positions by `extent`, so
        // origin lands at the same NDC-z whether the embedding fits in a
        // 1e-4 cube (fresh spectral seed) or a 1e6-unit one (a runaway
        // outlier).
        let extents = [1.0e-6f32, 1.0e-3, 1.0, 1.0e3, 1.0e6];
        let zooms = [MIN_ZOOM, 1.5, MAX_ZOOM];
        for &extent in &extents {
            for &zoom in &zooms {
                let camera = Camera {
                    zoom,
                    ..Camera::default()
                };
                let clip = transform_world(camera, 1.0, extent, [0.0, 0.0, 0.0]);
                let w = clip[3].abs();
                assert!(
                    w > 1e-3,
                    "extent={extent}, zoom={zoom}: w collapsed to {}",
                    clip[3]
                );
                for (axis, &value) in clip[..3].iter().enumerate() {
                    assert!(
                        value.abs() <= w,
                        "extent={extent}, zoom={zoom}: clip[{axis}]={value} outside w={w}"
                    );
                }
            }
        }
    }

    #[test]
    fn shader_normalization_keeps_bulk_edge_visible_depth_wise() {
        // A point sitting at the far edge of the framing bounding box
        // (roughly `extent * 0.5` from origin, since the diameter of the
        // bulk equals `extent`) also has to survive the depth clip after
        // being normalized, otherwise outer clusters would z-clip while
        // origin stayed visible. Sample a couple of extents to cover
        // early and late in a run.
        for extent in [1.0e-3f32, 1.0, 1.0e3] {
            let clip = transform_world(Camera::default(), 1.0, extent, [extent * 0.5, 0.0, 0.0]);
            let w = clip[3].abs();
            assert!(w > 0.0);
            // Only the depth axis has to fit; x/y being out of the sides
            // is expected framing behavior a drag can bring in.
            assert!(
                clip[2].abs() <= w,
                "extent={extent}: clip z = {}, w = {}",
                clip[2],
                w
            );
        }
    }

    #[test]
    fn data_extent_returns_bbox_diagonal() {
        // A 3-4-5 box has diagonal sqrt(50).
        let points = [
            0.0, 0.0, 0.0, //
            3.0, 4.0, 5.0, //
            1.0, 2.0, 3.0,
        ];
        let diag = data_extent(&points);
        assert!((diag - 50.0f32.sqrt()).abs() < 1e-4, "diag = {diag}");
    }

    #[test]
    fn data_extent_falls_back_for_empty_input() {
        assert_eq!(data_extent(&[]), FALLBACK_EXTENT);
    }

    #[test]
    fn data_extent_ignores_far_outlier() {
        // Reproduces the "70% through the run everything disappears" bug:
        // 200 points tight around origin plus one straggler at 1e6 must not
        // let the straggler blow up the framing.
        let mut bulk: Vec<f32> = Vec::with_capacity(200 * 3);
        for i in 0..200 {
            let t = i as f32 * 0.05;
            bulk.extend_from_slice(&[t.cos(), t.sin(), 0.25 * t]);
        }
        let clean = data_extent(&bulk);
        let mut with_outlier = bulk.clone();
        with_outlier.extend_from_slice(&[1.0e6, 1.0e6, 1.0e6]);
        let robust = data_extent(&with_outlier);
        assert!(robust.is_finite(), "extent went non-finite: {robust}");
        // The straggler is a factor of 1e5 further out than the bulk, so
        // a naive bbox would produce an extent in the same order of
        // magnitude. Anything within 2x the clean extent proves the
        // outlier was rejected.
        assert!(
            robust < clean * 2.0,
            "outlier blew up extent: {robust} vs {clean}"
        );
    }

    #[test]
    fn data_extent_ignores_nan_and_inf() {
        let points = [
            0.0,
            0.0,
            0.0, //
            1.0,
            1.0,
            1.0, //
            2.0,
            2.0,
            2.0, //
            f32::NAN,
            0.0,
            0.0, //
            f32::INFINITY,
            0.0,
            0.0, //
            f32::NEG_INFINITY,
            0.0,
            0.0,
        ];
        let extent = data_extent(&points);
        assert!(
            extent.is_finite(),
            "non-finite input poisoned the extent: {extent}"
        );
        // The finite bulk is the (0,0,0)..(2,2,2) box, diagonal sqrt(12).
        assert!((extent - 12.0f32.sqrt()).abs() < 1e-4, "extent = {extent}");
    }

    #[test]
    fn data_extent_falls_back_when_all_non_finite() {
        let points = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ];
        assert_eq!(data_extent(&points), FALLBACK_EXTENT);
    }

    #[test]
    fn parse_hex_rgb_accepts_leading_hash() {
        let rgb = parse_hex_rgb("#1f77b4").unwrap();
        assert!((rgb[0] - 0x1f as f32 / 255.0).abs() < 1e-6);
        assert!((rgb[1] - 0x77 as f32 / 255.0).abs() < 1e-6);
        assert!((rgb[2] - 0xb4 as f32 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn parse_hex_rgb_rejects_junk() {
        assert!(parse_hex_rgb("").is_none());
        assert!(parse_hex_rgb("#xyz").is_none());
        assert!(parse_hex_rgb("#1234").is_none());
    }

    #[test]
    fn colors_rgba_dims_off_class_points() {
        let colors = vec!["#1f77b4".to_string(), "#ff7f0e".to_string()];
        let highlight = ("#1f77b4".to_string(), Marker::default());
        let out = colors_rgba(2, Some(&colors), Some(&highlight));
        // First point keeps its color.
        assert!((out[0] - DEFAULT_COLOR[0]).abs() < 1e-5);
        // Second point is dimmed.
        assert!((out[4] - DIMMED_COLOR[0]).abs() < 1e-5);
        // All alphas match FILL_ALPHA.
        assert!((out[3] - FILL_ALPHA).abs() < 1e-6);
        assert!((out[7] - FILL_ALPHA).abs() < 1e-6);
    }

    #[test]
    fn colors_rgba_falls_back_to_default_when_signal_short() {
        let colors: Vec<String> = vec!["#1f77b4".to_string()];
        let out = colors_rgba(3, Some(&colors), None);
        // Second and third points fall back to DEFAULT_COLOR.
        assert!((out[4] - DEFAULT_COLOR[0]).abs() < 1e-5);
        assert!((out[8] - DEFAULT_COLOR[0]).abs() < 1e-5);
    }

    #[test]
    fn view_matrix_is_a_proper_ry_times_rx_at_zero_roll() {
        // Confirms the fix for the "one axis looks squished" symptom: the
        // hand-crafted matrix that used to live here was not orthogonal
        // for non-zero pitch AND yaw simultaneously, so dragging the
        // camera stretched world Z by `sqrt(1 + sin(rot_x)^2)`. `view_matrix`
        // is now a proper `Ry * Rx` product at `rot_z = 0`, byte-for-byte
        // matching what mat4_mul of the two factors gives.
        let camera = Camera {
            rot_x: 0.3,
            rot_y: 0.7,
            rot_z: 0.0,
            rot_w: 0.0,
            zoom: 1.5,
            pan_x: 0.2,
            pan_y: -0.1,
        };
        let mut expected = mat4_mul(&rotation_y(camera.rot_y), &rotation_x(camera.rot_x));
        expected[12] = -camera.pan_x;
        expected[13] = -camera.pan_y;
        expected[14] = -camera.zoom;
        let actual = view_matrix(camera);
        for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "view[{index}] = {a}, expected {b}");
        }
    }

    /// Extracts the 3x3 rotation part of a column-major 4x4 matrix.
    fn rotation_3x3(m: &[f32; 16]) -> [[f32; 3]; 3] {
        let mut r = [[0.0f32; 3]; 3];
        for col in 0..3 {
            for row in 0..3 {
                r[col][row] = m[col * 4 + row];
            }
        }
        r
    }

    fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn view_matrix_rotation_part_is_orthogonal_for_arbitrary_angles() {
        // Locks in the fix: for any combination of pitch, yaw and roll,
        // the rotation part is orthogonal (columns unit length, mutually
        // perpendicular). The pre-fix matrix failed both invariants
        // whenever pitch and yaw were both non-zero.
        for &(rx, ry, rz) in &[
            (0.3f32, 0.7, 0.0),
            (0.5, 0.5, 0.0),
            (0.9, -0.4, 0.6),
            (-1.2, 0.9, -0.3),
            (1.5, 1.5, 1.5),
        ] {
            let camera = Camera {
                rot_x: rx,
                rot_y: ry,
                rot_z: rz,
                rot_w: 0.0,
                zoom: 0.0,
                pan_x: 0.0,
                pan_y: 0.0,
            };
            let r = rotation_3x3(&view_matrix(camera));
            for (col, &v) in r.iter().enumerate() {
                let len = dot3(v, v).sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-5,
                    "rx={rx}, ry={ry}, rz={rz}: col {col} length {len}"
                );
            }
            for a in 0..3 {
                for b in (a + 1)..3 {
                    let d = dot3(r[a], r[b]);
                    assert!(
                        d.abs() < 1e-5,
                        "rx={rx}, ry={ry}, rz={rz}: cols {a}.{b} dot {d}"
                    );
                }
            }
        }
    }

    #[test]
    fn view_matrix_rolls_x_axis_into_the_xy_plane() {
        // With pitch and yaw zero, applying the view matrix to the world
        // X axis should rotate it in the XY plane by rot_z (the pan and
        // camera-Z translation only affect the w-column, so unit vectors
        // stay unaffected by them).
        let rz = 0.75_f32;
        let camera = Camera {
            rot_x: 0.0,
            rot_y: 0.0,
            rot_z: rz,
            rot_w: 0.0,
            zoom: 0.0,
            pan_x: 0.0,
            pan_y: 0.0,
        };
        let m = view_matrix(camera);
        let mut out = [0.0f32; 4];
        for row in 0..4 {
            let mut sum = 0.0;
            for col in 0..4 {
                sum += m[col * 4 + row] * [1.0f32, 0.0, 0.0, 0.0][col];
            }
            out[row] = sum;
        }
        assert!((out[0] - rz.cos()).abs() < 1e-6, "x = {}", out[0]);
        assert!((out[1] - rz.sin()).abs() < 1e-6, "y = {}", out[1]);
        assert!(out[2].abs() < 1e-6, "z = {}", out[2]);
    }

    #[test]
    fn held_axes_set_toggles_the_right_flag() {
        let mut axes = HeldAxes::default();
        assert!(!axes.any());
        axes.set(RotationAxis::X, true);
        assert!(axes.any() && axes.x && !axes.y && !axes.z && !axes.w);
        axes.set(RotationAxis::Y, true);
        axes.set(RotationAxis::Z, true);
        axes.set(RotationAxis::W, true);
        assert!(axes.x && axes.y && axes.z && axes.w);
        axes.set(RotationAxis::X, false);
        axes.set(RotationAxis::W, false);
        assert!(!axes.x && axes.y && axes.z && !axes.w);
    }

    #[test]
    fn project_to_display_pass_through_when_dims_match() {
        let camera = Camera::default();
        let three = [0.0f32, 1.0, 2.0, -1.0, -2.0, -3.0];
        assert_eq!(
            project_to_display(&three, 3, 3, camera).unwrap(),
            three.to_vec()
        );
        let two = [0.5f32, 0.7, -0.5, -0.7];
        assert_eq!(
            project_to_display(&two, 2, 2, camera).unwrap(),
            two.to_vec()
        );
    }

    #[test]
    fn project_to_display_drops_w_via_zw_rotation() {
        let camera_zero = Camera::default();
        let points = [0.0f32, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0];
        // rot_w = 0 gives plain "drop w".
        assert_eq!(
            project_to_display(&points, 4, 3, camera_zero).unwrap(),
            vec![0.0, 1.0, 2.0, -1.0, -2.0, -3.0],
        );
        // rot_w = PI / 2 swaps z and w so the visible z becomes the
        // original w.
        let camera_half = Camera {
            rot_w: std::f32::consts::FRAC_PI_2,
            ..Camera::default()
        };
        let one = [3.0f32, 5.0, 7.0, 11.0];
        let projected = project_to_display(&one, 4, 3, camera_half).unwrap();
        assert!((projected[0] - 3.0).abs() < 1e-6);
        assert!((projected[1] - 5.0).abs() < 1e-6);
        assert!((projected[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn project_to_display_applies_3d_rotation_to_2d_target() {
        // Default camera has `rot_y = 0.4`, so the world X axis rotates
        // to `(cos(0.4), 0, -sin(0.4))`; the visible 2-D is the leading
        // pair `(cos(0.4), 0)`.
        let camera = Camera::default();
        let projected = project_to_display(&[1.0, 0.0, 0.0], 3, 2, camera).unwrap();
        assert!((projected[0] - camera.rot_y.cos()).abs() < 1e-6);
        assert!(projected[1].abs() < 1e-6);
    }

    #[test]
    fn project_to_display_refuses_upcast() {
        let camera = Camera::default();
        assert!(project_to_display(&[1.0, 0.0, 0.0], 3, 4, camera).is_none());
        assert!(project_to_display(&[1.0, 0.0], 2, 3, camera).is_none());
    }

    #[test]
    fn project_to_display_refuses_mismatched_input_length() {
        let camera = Camera::default();
        assert!(project_to_display(&[0.0; 5], 3, 3, camera).is_none());
        assert!(project_to_display(&[0.0; 5], 4, 3, camera).is_none());
    }
}
