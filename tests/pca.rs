//! Tests for the PCA module.

use dioxus_tsne::pca;

/// Deterministic LCG so the tests need no RNG dependency.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) - 0.5
    }
}

#[test]
fn line_in_three_dimensions_reduces_to_one_component() {
    // Points exactly on the line t * (1, 2, 3): a single component captures
    // all the variance.
    let data: Vec<f32> = (0..10)
        .flat_map(|t| {
            let t = t as f32;
            [t, 2.0 * t, 3.0 * t]
        })
        .collect();

    let result = pca(&data, 10, 3, 2);

    assert_eq!(result.n_components, 2);
    assert!(
        result.explained_variance_ratio[0] > 0.999,
        "first component must capture all variance, got {:?}",
        result.explained_variance_ratio
    );
    assert!(result.explained_variance_ratio[1] < 1e-3);
}

#[test]
fn full_rank_projection_preserves_pairwise_distances() {
    // With as many components as features PCA is a rotation around the mean,
    // so pairwise distances are preserved.
    const N: usize = 30;
    const D: usize = 4;
    let mut next = lcg(42);
    let data: Vec<f32> = (0..N * D).map(|_| next()).collect();

    let result = pca(&data, N, D, D);

    let dist = |m: &[f32], a: usize, b: usize| -> f32 {
        (0..D)
            .map(|j| (m[a * D + j] - m[b * D + j]).powi(2))
            .sum::<f32>()
            .sqrt()
    };
    for a in 0..N {
        for b in (a + 1)..N {
            let before = dist(&data, a, b);
            let after = dist(&result.data, a, b);
            assert!(
                (before - after).abs() < 1e-4,
                "distance between {a} and {b} changed: {before} vs {after}"
            );
        }
    }

    let ratio_sum: f32 = result.explained_variance_ratio.iter().sum();
    assert!((ratio_sum - 1.0).abs() < 1e-5, "ratios must sum to 1");
}

#[test]
fn components_are_ordered_by_variance() {
    const N: usize = 200;
    let mut next = lcg(7);
    // Three independent axes with very different scales.
    let data: Vec<f32> = (0..N)
        .flat_map(|_| [10.0 * next(), 3.0 * next(), 0.1 * next()])
        .collect();

    let result = pca(&data, N, 3, 3);

    let ratios = &result.explained_variance_ratio;
    assert!(ratios[0] > ratios[1] && ratios[1] > ratios[2], "{ratios:?}");

    // The dominant output column must carry the dominant input variance.
    let column_variance = |c: usize| -> f32 {
        let mean = (0..N).map(|i| result.data[i * 3 + c]).sum::<f32>() / N as f32;
        (0..N)
            .map(|i| (result.data[i * 3 + c] - mean).powi(2))
            .sum::<f32>()
            / (N - 1) as f32
    };
    assert!(column_variance(0) > column_variance(1));
    assert!(column_variance(1) > column_variance(2));
}

#[test]
fn requested_components_are_clamped_to_features() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let result = pca(&data, 2, 2, 50);
    assert_eq!(result.n_components, 2);
    assert_eq!(result.data.len(), 4);
}

#[test]
fn projection_is_deterministic() {
    const N: usize = 20;
    const D: usize = 5;
    let mut next = lcg(3);
    let data: Vec<f32> = (0..N * D).map(|_| next()).collect();

    let first = pca(&data, N, D, 2);
    let second = pca(&data, N, D, 2);
    assert_eq!(first, second);
}

#[test]
fn constant_data_yields_zero_projection() {
    let data = vec![5.0; 12];
    let result = pca(&data, 4, 3, 2);
    assert!(result.data.iter().all(|&v| v.abs() < 1e-6));
    assert!(result.explained_variance_ratio.iter().all(|&r| r == 0.0));
}
