//! Mapping of label or score values to point colors and marker shapes.
//!
//! Values are treated as continuous when every entry parses as a float
//! (viridis colormap, circles only) and as categorical otherwise (palette and
//! marker shape by distinct value, so classes differ by shape as well as by
//! color). Recoloring is a pure redraw, the embedding is never recomputed.

use serde::{Deserialize, Serialize};

/// The matplotlib tab10 palette, cycled when there are more categories.
const PALETTE: [&str; 10] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
];

/// Viridis anchors at evenly spaced positions, linearly interpolated.
const VIRIDIS: [(u8, u8, u8); 10] = [
    (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (110, 206, 88),
    (181, 222, 43),
    (253, 231, 37),
];

/// Color used for values that cannot be placed on the continuous scale.
const MISSING: &str = "#999999";

/// Number of quantization levels of the continuous scale. Bounding the
/// distinct colors lets the plot batch points per color.
const LEVELS: usize = 64;

/// All-integer value sets with at most this many distinct values are treated
/// as categories (class indices) rather than as a continuous scale.
const MAX_INTEGER_CATEGORIES: usize = 10;

/// How a set of values was mapped to colors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScale {
    /// Distinct values mapped to a palette.
    Categorical,
    /// Numeric values mapped to the viridis colormap.
    Continuous,
}

/// The marker shape of a point, cycled per category so classes differ by
/// shape as well as by color. With the 10 color palette the (color, marker)
/// pairs stay distinct for 30 categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Marker {
    /// A filled circle, also the shape of every continuous scale point.
    #[default]
    Circle,
    /// A filled triangle pointing up.
    Triangle,
    /// A filled square.
    Square,
    /// A filled diamond.
    Diamond,
    /// A filled plus sign.
    Plus,
    /// A filled triangle pointing down.
    TriangleDown,
}

/// Every marker in cycling order.
pub(crate) const MARKERS: [Marker; 6] = [
    Marker::Circle,
    Marker::Triangle,
    Marker::Square,
    Marker::Diamond,
    Marker::Plus,
    Marker::TriangleDown,
];

impl Marker {
    /// The unicode glyph of the marker, rendered in the legend.
    pub fn glyph(self) -> char {
        match self {
            Marker::Circle => '\u{25CF}',       // ●
            Marker::Triangle => '\u{25B2}',     // ▲
            Marker::Square => '\u{25A0}',       // ■
            Marker::Diamond => '\u{25C6}',      // ◆
            Marker::Plus => '\u{271A}',         // ✚
            Marker::TriangleDown => '\u{25BC}', // ▼
        }
    }
}

/// One legend entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegendEntry {
    /// Human readable label.
    pub label: String,
    /// CSS color of the entry.
    pub color: String,
    /// Marker shape of the entry.
    pub marker: Marker,
}

/// Colors and marker shapes assigned to a set of values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coloring {
    /// One CSS color per input value.
    pub colors: Vec<String>,
    /// One marker shape per input value, cycled per category. Continuous
    /// scales use circles throughout.
    pub markers: Vec<Marker>,
    /// Legend describing the mapping: every category, or the extremes of the
    /// continuous scale.
    pub legend: Vec<LegendEntry>,
    /// The detected scale.
    pub scale: ColorScale,
}

/// Samples the viridis colormap at `t` in `[0, 1]`.
fn viridis(t: f32) -> String {
    let t = t.clamp(0.0, 1.0) * (VIRIDIS.len() - 1) as f32;
    let low = t.floor() as usize;
    let high = (low + 1).min(VIRIDIS.len() - 1);
    let fraction = t - low as f32;
    let channel = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * fraction) as u8;
    let (r, g, b) = (
        channel(VIRIDIS[low].0, VIRIDIS[high].0),
        channel(VIRIDIS[low].1, VIRIDIS[high].1),
        channel(VIRIDIS[low].2, VIRIDIS[high].2),
    );
    format!("#{r:02x}{g:02x}{b:02x}")
}

/// Maps values to point colors and marker shapes, auto detecting the scale.
///
/// Continuous when every trimmed value parses as a float: finite values are
/// normalized to the data range and quantized onto [`LEVELS`] viridis colors,
/// non finite ones get a grey, every point a circle. Exception: all-integer
/// sets with at most [`MAX_INTEGER_CATEGORIES`] distinct values are class
/// indices and map to the palette in ascending order. Categorical otherwise:
/// distinct values in first appearance order, palette cycled beyond its size.
/// Categories also cycle through the marker shapes so classes differ by shape
/// as well as by color.
pub fn colorize(values: &[String]) -> Coloring {
    if values.is_empty() {
        return Coloring {
            colors: Vec::new(),
            markers: Vec::new(),
            legend: Vec::new(),
            scale: ColorScale::Categorical,
        };
    }

    let parsed: Option<Vec<f32>> = values
        .iter()
        .map(|v| v.trim().parse::<f32>().ok())
        .collect();

    match parsed {
        Some(numbers) => {
            // Small sets of integers (class indices, MNIST digits) are
            // categories, not a gradient: palette in ascending order.
            if numbers.iter().all(|v| v.is_finite() && v.fract() == 0.0) {
                let mut distinct: Vec<i64> = numbers.iter().map(|&v| v as i64).collect();
                distinct.sort_unstable();
                distinct.dedup();
                if distinct.len() <= MAX_INTEGER_CATEGORIES {
                    let ranks: Vec<usize> = numbers
                        .iter()
                        .map(|&v| distinct.binary_search(&(v as i64)).unwrap())
                        .collect();
                    let colors = ranks
                        .iter()
                        .map(|&rank| String::from(PALETTE[rank % PALETTE.len()]))
                        .collect();
                    let markers = ranks
                        .iter()
                        .map(|&rank| MARKERS[rank % MARKERS.len()])
                        .collect();
                    let legend = distinct
                        .iter()
                        .enumerate()
                        .map(|(rank, value)| LegendEntry {
                            label: value.to_string(),
                            color: String::from(PALETTE[rank % PALETTE.len()]),
                            marker: MARKERS[rank % MARKERS.len()],
                        })
                        .collect();
                    return Coloring {
                        colors,
                        markers,
                        legend,
                        scale: ColorScale::Categorical,
                    };
                }
            }

            let finite: Vec<f32> = numbers.iter().copied().filter(|v| v.is_finite()).collect();
            let min = finite.iter().copied().fold(f32::MAX, f32::min);
            let max = finite.iter().copied().fold(f32::MIN, f32::max);
            let span = (max - min).max(f32::EPSILON);

            let colors = numbers
                .iter()
                .map(|&v| {
                    if v.is_finite() {
                        // Quantized so the plot can batch points per color.
                        let t = (v - min) / span;
                        let level = (t * (LEVELS - 1) as f32).round() / (LEVELS - 1) as f32;
                        viridis(level)
                    } else {
                        String::from(MISSING)
                    }
                })
                .collect();

            let legend = if finite.is_empty() {
                Vec::new()
            } else {
                vec![
                    LegendEntry {
                        label: format!("{min}"),
                        color: viridis(0.0),
                        marker: Marker::Circle,
                    },
                    LegendEntry {
                        label: format!("{max}"),
                        color: viridis(1.0),
                        marker: Marker::Circle,
                    },
                ]
            };

            Coloring {
                markers: vec![Marker::Circle; numbers.len()],
                colors,
                legend,
                scale: ColorScale::Continuous,
            }
        }
        None => {
            let mut categories: Vec<&str> = Vec::new();
            let indices: Vec<usize> = values
                .iter()
                .map(|value| {
                    categories
                        .iter()
                        .position(|&c| c == value.as_str())
                        .unwrap_or_else(|| {
                            categories.push(value.as_str());
                            categories.len() - 1
                        })
                })
                .collect();
            let colors = indices
                .iter()
                .map(|&index| String::from(PALETTE[index % PALETTE.len()]))
                .collect();
            let markers = indices
                .iter()
                .map(|&index| MARKERS[index % MARKERS.len()])
                .collect();

            let legend = categories
                .iter()
                .enumerate()
                .map(|(index, &category)| LegendEntry {
                    label: if category.is_empty() {
                        String::from("(empty)")
                    } else {
                        String::from(category)
                    },
                    color: String::from(PALETTE[index % PALETTE.len()]),
                    marker: MARKERS[index % MARKERS.len()],
                })
                .collect();

            Coloring {
                colors,
                markers,
                legend,
                scale: ColorScale::Categorical,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ColorScale, MARKERS, MISSING, Marker, colorize, viridis};

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| String::from(*v)).collect()
    }

    #[test]
    fn categorical_assigns_stable_distinct_colors() {
        let coloring = colorize(&strings(&["dog", "cat", "dog", "bird", "cat"]));

        assert_eq!(coloring.scale, ColorScale::Categorical);
        assert_eq!(coloring.colors[0], coloring.colors[2]);
        assert_eq!(coloring.colors[1], coloring.colors[4]);
        assert_ne!(coloring.colors[0], coloring.colors[1]);
        assert_ne!(coloring.colors[0], coloring.colors[3]);

        let labels: Vec<&str> = coloring.legend.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, ["dog", "cat", "bird"]);
        assert_eq!(coloring.legend[0].color, coloring.colors[0]);
    }

    #[test]
    fn numeric_values_use_the_continuous_scale() {
        let coloring = colorize(&strings(&["0.5", "5.0", "10.5"]));

        assert_eq!(coloring.scale, ColorScale::Continuous);
        assert_eq!(coloring.colors[0], viridis(0.0));
        assert_eq!(coloring.colors[2], viridis(1.0));
        assert_ne!(coloring.colors[1], coloring.colors[0]);
        assert_ne!(coloring.colors[1], coloring.colors[2]);

        assert_eq!(coloring.legend.len(), 2);
        assert_eq!(coloring.legend[0].label, "0.5");
        assert_eq!(coloring.legend[1].label, "10.5");
    }

    #[test]
    fn small_integer_sets_are_class_indices() {
        // MNIST style digit labels: categorical with an ascending legend.
        let values: Vec<String> = (0..100).map(|v| format!("{}", v % 10)).collect();
        let coloring = colorize(&values);

        assert_eq!(coloring.scale, ColorScale::Categorical);
        assert_eq!(coloring.legend.len(), 10);
        let labels: Vec<&str> = coloring.legend.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
        // Same digit, same color; different digit, different color.
        assert_eq!(coloring.colors[0], coloring.colors[10]);
        assert_ne!(coloring.colors[0], coloring.colors[1]);
    }

    #[test]
    fn many_distinct_integers_stay_continuous() {
        let values: Vec<String> = (0..11).map(|v| format!("{v}")).collect();
        let coloring = colorize(&values);
        assert_eq!(coloring.scale, ColorScale::Continuous);
    }

    #[test]
    fn continuous_quantization_bounds_distinct_colors() {
        let values: Vec<String> = (0..10_000).map(|v| format!("{v}")).collect();
        let coloring = colorize(&values);

        let mut distinct: Vec<&String> = coloring.colors.iter().collect();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() <= super::LEVELS, "{}", distinct.len());
    }

    #[test]
    fn non_finite_values_are_grey() {
        let coloring = colorize(&strings(&["1.0", "NaN", "2.0", "inf"]));

        assert_eq!(coloring.scale, ColorScale::Continuous);
        assert_eq!(coloring.colors[1], MISSING);
        assert_eq!(coloring.colors[3], MISSING);
        assert_eq!(coloring.colors[0], viridis(0.0));
        assert_eq!(coloring.colors[2], viridis(1.0));
    }

    #[test]
    fn mixed_values_fall_back_to_categorical() {
        let coloring = colorize(&strings(&["1.0", "high", "2.0"]));
        assert_eq!(coloring.scale, ColorScale::Categorical);
        assert_eq!(coloring.legend.len(), 3);
    }

    #[test]
    fn palette_cycles_beyond_its_size() {
        let values: Vec<String> = (0..15).map(|v| format!("cat{v}")).collect();
        let coloring = colorize(&values);
        assert_eq!(coloring.colors[10], coloring.colors[0]);
        assert_eq!(coloring.legend.len(), 15);
    }

    #[test]
    fn categories_cycle_through_distinct_markers() {
        let coloring = colorize(&strings(&["dog", "cat", "dog", "bird"]));

        // Same category, same marker; different categories, different ones.
        assert_eq!(coloring.markers[0], coloring.markers[2]);
        assert_ne!(coloring.markers[0], coloring.markers[1]);
        assert_ne!(coloring.markers[1], coloring.markers[3]);
        // The legend carries the same markers as the points.
        assert_eq!(coloring.legend[0].marker, coloring.markers[0]);
        assert_eq!(coloring.legend[1].marker, coloring.markers[1]);
    }

    #[test]
    fn integer_classes_cycle_through_distinct_markers() {
        let values: Vec<String> = (0..100).map(|v| format!("{}", v % 10)).collect();
        let coloring = colorize(&values);

        assert_eq!(coloring.markers.len(), values.len());
        assert_eq!(coloring.markers[0], coloring.markers[10]);
        assert_ne!(coloring.markers[0], coloring.markers[1]);
        // Ten digits over six shapes: the cycle wraps after the last marker.
        assert_eq!(coloring.legend[0].marker, MARKERS[0]);
        assert_eq!(coloring.legend[6].marker, MARKERS[0]);
    }

    #[test]
    fn continuous_scales_use_circles_only() {
        let coloring = colorize(&strings(&["0.5", "5.0", "10.5"]));
        assert!(coloring.markers.iter().all(|&m| m == Marker::Circle));
        assert!(coloring.legend.iter().all(|e| e.marker == Marker::Circle));
    }

    #[test]
    fn empty_input_yields_empty_coloring() {
        let coloring = colorize(&[]);
        assert!(coloring.colors.is_empty());
        assert!(coloring.markers.is_empty());
        assert!(coloring.legend.is_empty());
    }

    #[test]
    fn empty_category_is_named_in_the_legend() {
        let coloring = colorize(&strings(&["a", "", "a"]));
        assert_eq!(coloring.legend[1].label, "(empty)");
    }

    #[test]
    fn viridis_endpoints_match_the_anchors() {
        assert_eq!(viridis(0.0), "#440154");
        assert_eq!(viridis(1.0), "#fde725");
    }
}
