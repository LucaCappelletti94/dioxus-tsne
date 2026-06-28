//! Tests for the decomposition compute path, the same code the worker runs.

use std::sync::Mutex;

use dioxus_tsne::{DecompositionMethod, TsneParams, decompose};

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

/// Fraction of points whose nearest neighbour in the embedding shares their
/// originating cluster (the first half versus the second half).
fn same_cluster_fraction(embedding: &[f32], n: usize) -> f64 {
    let mut same = 0;
    for i in 0..n {
        let mut best = f32::MAX;
        let mut best_j = usize::MAX;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = embedding[2 * i] - embedding[2 * j];
            let dy = embedding[2 * i + 1] - embedding[2 * j + 1];
            let d = dx * dx + dy * dy;
            if d < best {
                best = d;
                best_j = j;
            }
        }
        if (i < n / 2) == (best_j < n / 2) {
            same += 1;
        }
    }
    same as f64 / n as f64
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
fn tsne_warm_start_continues_from_seed() {
    const N: usize = 200;
    const DIM: usize = 8;
    let data = clustered_data(N / 2, DIM);

    // First run to obtain a converged-ish layout to warm start from.
    let seed = decompose(
        &data,
        N,
        DIM,
        &DecompositionMethod::Tsne(TsneParams {
            epochs: 250,
            pca_dims: 4,
            perplexity: 20.0,
            ..TsneParams::default()
        }),
        |_, _| {},
    )
    .unwrap()
    .embedding;

    // Continue from the seed, capturing the very first streamed snapshot.
    let first = Mutex::new(None::<Vec<f32>>);
    let output = decompose(
        &data,
        N,
        DIM,
        &DecompositionMethod::Tsne(TsneParams {
            epochs: 50,
            snapshot_every: 1,
            pca_dims: 4,
            perplexity: 20.0,
            initial_embedding: Some(seed.clone()),
            ..TsneParams::default()
        }),
        |_, embedding| {
            let mut guard = first.lock().unwrap();
            if guard.is_none() {
                *guard = Some(embedding.to_vec());
            }
        },
    )
    .unwrap();

    let first = first.into_inner().unwrap().expect("a snapshot streamed");

    // The seed bounding box diagonal sets the scale of the layout.
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for point in seed.chunks_exact(2) {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    let diagonal = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();

    // The first continued snapshot stays within a small fraction of the seed:
    // warm start resumes the layout instead of scattering from noise.
    let mean_shift = seed
        .chunks_exact(2)
        .zip(first.chunks_exact(2))
        .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
        .sum::<f32>()
        / N as f32;
    assert!(
        mean_shift < 0.05 * diagonal,
        "warm start jumped: mean shift {mean_shift} vs diagonal {diagonal}"
    );

    // Continuing keeps the clusters separated.
    assert!(
        same_cluster_fraction(&output.embedding, N) > 0.95,
        "warm start degraded separation"
    );
}

#[test]
fn tsne_rejects_wrong_initial_embedding_length() {
    const N: usize = 200;
    let data = clustered_data(N / 2, 4);
    let result = decompose(
        &data,
        N,
        4,
        &DecompositionMethod::Tsne(TsneParams {
            // One value short of the required N * 2.
            initial_embedding: Some(vec![0.0; N * 2 - 1]),
            ..TsneParams::default()
        }),
        |_, _| {},
    );
    assert!(
        result.unwrap_err().contains("initial embedding"),
        "expected an initial embedding length error"
    );
}

#[test]
fn tsne_params_serde_round_trips_with_initial_embedding() {
    let params = TsneParams {
        initial_embedding: Some(vec![1.0, 2.0, 3.0, 4.0]),
        ..TsneParams::default()
    };
    let json = serde_json::to_string(&params).unwrap();
    let back: TsneParams = serde_json::from_str(&json).unwrap();
    assert_eq!(back, params);

    // Requests serialized before the field existed must still decode, the
    // missing field defaulting to None.
    let legacy = r#"{
        "perplexity": 30.0,
        "theta": 0.5,
        "epochs": 1000,
        "learning_rate": 200.0,
        "pca_dims": 50,
        "snapshot_every": 5
    }"#;
    let decoded: TsneParams = serde_json::from_str(legacy).unwrap();
    assert_eq!(decoded.initial_embedding, None);
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
    let result = decompose(
        &[1.0, 2.0, 3.0],
        2,
        2,
        &DecompositionMethod::Tsne(TsneParams::default()),
        |_, _| {},
    );
    assert!(result.unwrap_err().contains("inconsistent"));
}

#[test]
fn empty_input_is_rejected() {
    let result = decompose(
        &[],
        0,
        0,
        &DecompositionMethod::Tsne(TsneParams::default()),
        |_, _| {},
    );
    assert!(result.unwrap_err().contains("empty"));
}
