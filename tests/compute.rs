//! Tests for the decomposition compute path, the same code the worker runs.

use std::sync::Mutex;

use dioxus_decompositions::{DecompositionMethod, TsneParams, decompose};

/// Two well separated gaussian-ish clusters via a deterministic LCG.
fn clustered_data(n_per_cluster: usize, dim: usize) -> Vec<f32> {
    let mut state = 42u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) - 0.5
    };
    let mut data = Vec::with_capacity(2 * n_per_cluster * dim);
    for cluster in 0..2 {
        let centre = if cluster == 0 { 0.0 } else { 10.0 };
        for _ in 0..n_per_cluster {
            for _ in 0..dim {
                data.push(centre + next());
            }
        }
    }
    data
}

#[test]
fn tsne_reports_snapshots_and_separates_clusters() {
    const N: usize = 200;
    const DIM: usize = 8;
    let data = clustered_data(N / 2, DIM);

    let snapshots = Mutex::new(Vec::<usize>::new());
    let params = TsneParams {
        epochs: 300,
        snapshot_every: 10,
        perplexity: 20.0,
        // Forces the PCA preprocessing path: 8 features reduced to 4.
        pca_dims: 4,
        ..TsneParams::default()
    };

    let output = decompose(
        &data,
        N,
        DIM,
        &DecompositionMethod::Tsne(params),
        |epoch, embedding| {
            assert_eq!(embedding.len(), N * 2);
            snapshots.lock().unwrap().push(epoch);
        },
    )
    .unwrap();

    assert_eq!(output.embedding.len(), N * 2);
    assert!(output.explained_variance_ratio.is_none());

    let seen = snapshots.into_inner().unwrap();
    assert_eq!(seen, (0..300).step_by(10).collect::<Vec<_>>());

    // The two clusters must be separated in the embedding: the nearest
    // neighbour of nearly every point belongs to the same cluster.
    let mut same = 0;
    for i in 0..N {
        let mut best = f32::MAX;
        let mut best_j = usize::MAX;
        for j in 0..N {
            if i == j {
                continue;
            }
            let dx = output.embedding[2 * i] - output.embedding[2 * j];
            let dy = output.embedding[2 * i + 1] - output.embedding[2 * j + 1];
            let d = dx * dx + dy * dy;
            if d < best {
                best = d;
                best_j = j;
            }
        }
        if (i < N / 2) == (best_j < N / 2) {
            same += 1;
        }
    }
    assert!(
        same as f64 / N as f64 > 0.95,
        "clusters not separated: {same}/{N}"
    );
}

#[test]
fn pca_method_reports_explained_variance() {
    const N: usize = 100;
    const DIM: usize = 5;
    let data = clustered_data(N / 2, DIM);

    let output = decompose(&data, N, DIM, &DecompositionMethod::Pca, |_, _| {
        panic!("PCA must not produce snapshots");
    })
    .unwrap();

    assert_eq!(output.embedding.len(), N * 2);
    let ratios = output.explained_variance_ratio.unwrap();
    assert_eq!(ratios.len(), 2);
    // The cluster separation axis dominates the variance.
    assert!(ratios[0] > 0.9, "{ratios:?}");
}

#[test]
fn excessive_perplexity_is_rejected() {
    let data = clustered_data(10, 4);
    let result = decompose(
        &data,
        20,
        4,
        &DecompositionMethod::Tsne(TsneParams::default()),
        |_, _| {},
    );
    let message = result.unwrap_err();
    assert!(message.contains("perplexity"), "{message}");
}

#[test]
fn non_positive_theta_is_rejected() {
    let data = clustered_data(100, 4);
    let params = TsneParams {
        theta: 0.0,
        ..TsneParams::default()
    };
    let result = decompose(&data, 200, 4, &DecompositionMethod::Tsne(params), |_, _| {});
    assert!(result.unwrap_err().contains("theta"));
}

#[test]
fn inconsistent_shape_is_rejected() {
    let result = decompose(&[1.0, 2.0, 3.0], 2, 2, &DecompositionMethod::Pca, |_, _| {});
    assert!(result.unwrap_err().contains("inconsistent"));
}

#[test]
fn empty_input_is_rejected() {
    let result = decompose(&[], 0, 0, &DecompositionMethod::Pca, |_, _| {});
    assert!(result.unwrap_err().contains("empty"));
}

#[test]
fn pca_with_single_feature_is_rejected() {
    let result = decompose(&[1.0, 2.0, 3.0], 3, 1, &DecompositionMethod::Pca, |_, _| {});
    assert!(result.unwrap_err().contains("two features"));
}
