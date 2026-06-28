//! Tests for the file ingestion module.

use dioxus_tsne::{Dataset, IngestError, parse_dataset};

#[test]
fn csv_with_header_and_label_column() {
    let content =
        "sepal_length,sepal_width,species\n5.1,3.5,setosa\n4.9,3.0,setosa\n6.2,3.4,virginica\n";
    let dataset = parse_dataset("iris.csv", content.as_bytes()).unwrap();

    assert_eq!(dataset.n_samples, 3);
    assert_eq!(dataset.n_features, 2);
    assert_eq!(dataset.feature_names, ["sepal_length", "sepal_width"]);
    // Row major.
    assert_eq!(dataset.data, [5.1, 3.5, 4.9, 3.0, 6.2, 3.4]);
    assert_eq!(dataset.label_columns.len(), 1);
    assert_eq!(dataset.label_columns[0].name, "species");
    assert_eq!(
        dataset.label_columns[0].values,
        ["setosa", "setosa", "virginica"]
    );
}

#[test]
fn tsv_sniffed_without_header() {
    let content = "1.0\t2.0\n3.0\t4.0\n";
    let dataset = parse_dataset("data.tsv", content.as_bytes()).unwrap();

    assert_eq!(dataset.n_samples, 2);
    assert_eq!(dataset.n_features, 2);
    assert_eq!(dataset.feature_names, ["column_0", "column_1"]);
    assert_eq!(dataset.data, [1.0, 2.0, 3.0, 4.0]);
    assert!(dataset.label_columns.is_empty());
}

#[test]
fn csv_all_numeric_first_row_is_data() {
    let content = "1,2\n3,4\n";
    let dataset = parse_dataset("data.csv", content.as_bytes()).unwrap();

    assert_eq!(dataset.n_samples, 2);
    assert_eq!(dataset.data, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn csv_single_row_is_data() {
    let content = "1.5,2.5\n";
    let dataset = parse_dataset("data.csv", content.as_bytes()).unwrap();

    assert_eq!(dataset.n_samples, 1);
    assert_eq!(dataset.data, [1.5, 2.5]);
}

#[test]
fn empty_file_is_rejected() {
    assert!(matches!(
        parse_dataset("data.csv", b""),
        Err(IngestError::Empty)
    ));
}

#[test]
fn header_only_file_is_rejected() {
    // A single non numeric row is data without a header, so it fails as no
    // numeric columns rather than as empty.
    assert!(matches!(
        parse_dataset("data.csv", b"a,b,c\n"),
        Err(IngestError::NoNumericColumns)
    ));
}

#[test]
fn no_numeric_columns_is_rejected() {
    let content = "name,species\nrex,dog\nfelix,cat\n";
    assert!(matches!(
        parse_dataset("data.csv", content.as_bytes()),
        Err(IngestError::NoNumericColumns)
    ));
}

#[test]
fn ragged_csv_is_rejected() {
    let content = "a,b\n1,2\n3\n";
    assert!(matches!(
        parse_dataset("data.csv", content.as_bytes()),
        Err(IngestError::Csv(_))
    ));
}

/// Builds a small Parquet file in memory: a float column, an integer column
/// and a string column.
fn parquet_bytes() -> Vec<u8> {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int32Array, RecordBatch, StringArray};
    use parquet::arrow::ArrowWriter;

    let batch = RecordBatch::try_from_iter([
        (
            "score",
            Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5])) as arrow_array::ArrayRef,
        ),
        (
            "count",
            Arc::new(Int32Array::from(vec![10, 20, 30])) as arrow_array::ArrayRef,
        ),
        (
            "species",
            Arc::new(StringArray::from(vec!["a", "b", "a"])) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap();

    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    buffer
}

#[test]
fn parquet_with_numeric_and_label_columns() {
    let bytes = parquet_bytes();
    let dataset = parse_dataset("data.parquet", &bytes).unwrap();

    assert_eq!(dataset.n_samples, 3);
    assert_eq!(dataset.n_features, 2);
    assert_eq!(dataset.feature_names, ["score", "count"]);
    assert_eq!(dataset.data, [0.5, 10.0, 1.5, 20.0, 2.5, 30.0]);
    assert_eq!(dataset.label_columns.len(), 1);
    assert_eq!(dataset.label_columns[0].name, "species");
    assert_eq!(dataset.label_columns[0].values, ["a", "b", "a"]);
}

#[test]
fn parquet_detected_by_magic_despite_wrong_extension() {
    let bytes = parquet_bytes();
    let dataset = parse_dataset("data.bin", &bytes).unwrap();
    assert_eq!(dataset.n_samples, 3);
}

#[test]
fn parquet_with_nulls_in_numeric_column_is_rejected() {
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch};
    use parquet::arrow::ArrowWriter;

    let batch = RecordBatch::try_from_iter([(
        "score",
        Arc::new(Float64Array::from(vec![Some(0.5), None, Some(2.5)])) as arrow_array::ArrayRef,
    )])
    .unwrap();
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    assert!(matches!(
        parse_dataset("data.parquet", &buffer),
        Err(IngestError::MissingValues { .. })
    ));
}

#[test]
fn dataset_serializes_for_the_worker_boundary() {
    let dataset = Dataset {
        data: vec![1.0, 2.0],
        n_samples: 1,
        n_features: 2,
        feature_names: vec!["a".into(), "b".into()],
        label_columns: Vec::new(),
    };
    let json = serde_json::to_string(&dataset).unwrap();
    let back: Dataset = serde_json::from_str(&json).unwrap();
    assert_eq!(dataset, back);
}

/// Writes the same three-column table as `parquet_bytes` to the Arrow IPC file
/// format (Feather v2).
fn arrow_ipc_bytes() -> Vec<u8> {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int32Array, RecordBatch, StringArray};
    use arrow_ipc::writer::FileWriter;

    let batch = RecordBatch::try_from_iter([
        (
            "score",
            Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5])) as arrow_array::ArrayRef,
        ),
        (
            "count",
            Arc::new(Int32Array::from(vec![10, 20, 30])) as arrow_array::ArrayRef,
        ),
        (
            "species",
            Arc::new(StringArray::from(vec!["a", "b", "a"])) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap();

    let mut buffer = Vec::new();
    let mut writer = FileWriter::try_new(&mut buffer, &batch.schema()).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    buffer
}

#[test]
fn arrow_ipc_with_numeric_and_label_columns() {
    let bytes = arrow_ipc_bytes();
    let dataset = parse_dataset("data.arrow", &bytes).unwrap();

    assert_eq!(dataset.n_samples, 3);
    assert_eq!(dataset.n_features, 2);
    assert_eq!(dataset.feature_names, ["score", "count"]);
    assert_eq!(dataset.data, [0.5, 10.0, 1.5, 20.0, 2.5, 30.0]);
    assert_eq!(dataset.label_columns.len(), 1);
    assert_eq!(dataset.label_columns[0].name, "species");
    assert_eq!(dataset.label_columns[0].values, ["a", "b", "a"]);
}

#[test]
fn arrow_ipc_detected_by_magic_despite_wrong_extension() {
    let bytes = arrow_ipc_bytes();
    let dataset = parse_dataset("data.bin", &bytes).unwrap();
    assert_eq!(dataset.n_samples, 3);
}

/// Builds a NumPy v1 `.npy` from a dtype descr, storage order, shape and raw
/// little-endian body, padding the header per the format.
fn npy_v1(descr: &str, fortran: bool, shape: &[usize], body: &[u8]) -> Vec<u8> {
    let shape_str = match shape {
        [n] => format!("({n},)"),
        dims => {
            let inner: Vec<String> = dims.iter().map(usize::to_string).collect();
            format!("({})", inner.join(", "))
        }
    };
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape_str}, }}",
        if fortran { "True" } else { "False" }
    );
    // Pad with spaces so the 10-byte preamble plus the header (newline included)
    // is a multiple of 64.
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    let mut header = dict.into_bytes();
    header.resize(header.len() + pad, b' ');
    header.push(b'\n');

    let mut out = Vec::from(*b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(u16::try_from(header.len()).unwrap()).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(body);
    out
}

fn f32_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn npy_2d_c_order_is_a_feature_matrix() {
    let body = f32_le(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let dataset = parse_dataset("x.npy", &npy_v1("<f4", false, &[2, 3], &body)).unwrap();

    assert_eq!(dataset.n_samples, 2);
    assert_eq!(dataset.n_features, 3);
    assert_eq!(dataset.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(dataset.feature_names, ["column_0", "column_1", "column_2"]);
    assert!(dataset.label_columns.is_empty());
}

#[test]
fn npy_fortran_order_is_read_into_row_major() {
    // The same logical 2x3 matrix as the C-order case, stored column-major.
    let body = f32_le(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let dataset = parse_dataset("x.npy", &npy_v1("<f4", true, &[2, 3], &body)).unwrap();
    assert_eq!(dataset.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn npy_1d_array_is_a_single_feature_column() {
    let body: Vec<u8> = [10i64, 20, 30, 40]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let dataset = parse_dataset("x.npy", &npy_v1("<i8", false, &[4], &body)).unwrap();

    assert_eq!(dataset.n_samples, 4);
    assert_eq!(dataset.n_features, 1);
    assert_eq!(dataset.data, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn npy_detected_by_magic_despite_wrong_extension() {
    let body = f32_le(&[1.0, 2.0]);
    let dataset = parse_dataset("blob.bin", &npy_v1("<f4", false, &[1, 2], &body)).unwrap();
    assert_eq!(dataset.n_samples, 1);
    assert_eq!(dataset.n_features, 2);
}
