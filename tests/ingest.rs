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
#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
#[test]
fn parquet_detected_by_magic_despite_wrong_extension() {
    let bytes = parquet_bytes();
    let dataset = parse_dataset("data.bin", &bytes).unwrap();
    assert_eq!(dataset.n_samples, 3);
}

#[cfg(feature = "parquet")]
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
