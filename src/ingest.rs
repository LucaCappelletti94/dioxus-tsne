//! Parsing of user supplied tabular files (CSV, TSV, Parquet) into a numeric
//! matrix plus candidate label columns for coloring.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A non numeric column set aside as a candidate source for coloring points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelColumn {
    /// Column name, from the header when present.
    pub name: String,
    /// One value per sample, in file order.
    pub values: Vec<String>,
}

/// Numeric dataset parsed from a user supplied file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    /// Row major numeric matrix, `n_samples * n_features` long.
    pub data: Vec<f32>,
    /// Number of rows.
    pub n_samples: usize,
    /// Number of numeric columns.
    pub n_features: usize,
    /// Names of the numeric columns, in matrix order.
    pub feature_names: Vec<String>,
    /// Non numeric columns, candidate color sources.
    pub label_columns: Vec<LabelColumn>,
}

/// Errors produced while parsing a user supplied file.
#[derive(Debug)]
pub enum IngestError {
    /// The file contains no data rows.
    Empty,
    /// No column is fully numeric, there is nothing to embed.
    NoNumericColumns,
    /// Malformed CSV/TSV content.
    Csv(csv::Error),
    /// Malformed Parquet content.
    #[cfg(feature = "parquet")]
    Parquet(parquet::errors::ParquetError),
    /// A numeric Parquet column contains nulls.
    #[cfg(feature = "parquet")]
    MissingValues {
        /// Name of the offending column.
        column: String,
    },
    /// A Parquet column has a type that can be represented neither as f32 nor
    /// as a string label.
    #[cfg(feature = "parquet")]
    UnsupportedColumnType {
        /// Name of the offending column.
        column: String,
        /// The Arrow data type of the column.
        data_type: String,
    },
    /// The file looks like Parquet but the `parquet` feature is disabled.
    #[cfg(not(feature = "parquet"))]
    ParquetUnsupported,
}

impl fmt::Display for IngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestError::Empty => write!(f, "the file contains no data rows"),
            IngestError::NoNumericColumns => {
                write!(f, "no fully numeric column found, nothing to embed")
            }
            IngestError::Csv(e) => write!(f, "malformed CSV/TSV: {e}"),
            #[cfg(feature = "parquet")]
            IngestError::Parquet(e) => write!(f, "malformed Parquet: {e}"),
            #[cfg(feature = "parquet")]
            IngestError::MissingValues { column } => {
                write!(f, "numeric column {column:?} contains missing values")
            }
            #[cfg(feature = "parquet")]
            IngestError::UnsupportedColumnType { column, data_type } => {
                write!(f, "column {column:?} has unsupported type {data_type}")
            }
            #[cfg(not(feature = "parquet"))]
            IngestError::ParquetUnsupported => {
                write!(f, "Parquet support is disabled in this build")
            }
        }
    }
}

impl std::error::Error for IngestError {}

impl From<csv::Error> for IngestError {
    fn from(e: csv::Error) -> Self {
        IngestError::Csv(e)
    }
}

#[cfg(feature = "parquet")]
impl From<parquet::errors::ParquetError> for IngestError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        IngestError::Parquet(e)
    }
}

#[cfg(feature = "parquet")]
impl From<arrow_schema::ArrowError> for IngestError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        IngestError::Parquet(e.into())
    }
}

/// A parsed column before the row major assembly.
enum Column {
    Numeric(Vec<f32>),
    Text(Vec<String>),
}

/// Parses a user supplied tabular file into a [`Dataset`].
///
/// The format is detected from the content (Parquet magic number) with the
/// file name extension as a fallback, CSV and TSV are told apart by sniffing
/// the delimiter of the first line. A header row is detected by comparing the
/// parseability of the first row against the rest. Columns where every value
/// parses as a float become matrix features, every other column is set aside
/// as a candidate label column for coloring.
///
/// # Arguments
///
/// * `file_name` - name of the file, used as a format hint.
/// * `bytes` - raw content of the file.
///
/// # Errors
///
/// Returns an [`IngestError`] when the content is malformed, empty or contains
/// no numeric column.
pub fn parse_dataset(file_name: &str, bytes: &[u8]) -> Result<Dataset, IngestError> {
    const PARQUET_MAGIC: &[u8] = b"PAR1";

    if bytes.starts_with(PARQUET_MAGIC)
        || file_name
            .rsplit('.')
            .next()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
    {
        #[cfg(feature = "parquet")]
        {
            return parse_parquet(bytes);
        }
        #[cfg(not(feature = "parquet"))]
        {
            return Err(IngestError::ParquetUnsupported);
        }
    }

    parse_delimited(bytes)
}

/// Assembles parsed columns into a [`Dataset`], splitting numeric features
/// from label columns.
fn assemble(names: Vec<String>, columns: Vec<Column>, n_samples: usize) -> Dataset {
    let mut feature_names = Vec::new();
    let mut numeric_columns: Vec<Vec<f32>> = Vec::new();
    let mut label_columns = Vec::new();

    for (name, column) in names.into_iter().zip(columns) {
        match column {
            Column::Numeric(values) => {
                feature_names.push(name);
                numeric_columns.push(values);
            }
            Column::Text(values) => label_columns.push(LabelColumn { name, values }),
        }
    }

    let n_features = numeric_columns.len();
    let mut data = Vec::with_capacity(n_samples * n_features);
    for row in 0..n_samples {
        for column in &numeric_columns {
            data.push(column[row]);
        }
    }

    Dataset {
        data,
        n_samples,
        n_features,
        feature_names,
        label_columns,
    }
}

/// Parses CSV/TSV content.
fn parse_delimited(bytes: &[u8]) -> Result<Dataset, IngestError> {
    let delimiter = sniff_delimiter(bytes);

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delimiter)
        .from_reader(bytes);

    let mut rows: Vec<csv::StringRecord> = Vec::new();
    for record in reader.records() {
        rows.push(record?);
    }
    if rows.is_empty() {
        return Err(IngestError::Empty);
    }

    let n_columns = rows[0].len();
    let parses = |field: &str| field.trim().parse::<f32>().is_ok();

    // A column is numeric over the candidate data rows (all rows but the
    // first) when every value parses as a float. The first row belongs to a
    // header when some such column has a non parseable first value.
    let has_header = rows.len() > 1
        && (0..n_columns)
            .any(|c| rows[1..].iter().all(|row| parses(&row[c])) && !parses(&rows[0][c]));

    let (names, data_rows): (Vec<String>, &[csv::StringRecord]) = if has_header {
        (rows[0].iter().map(str::to_owned).collect(), &rows[1..])
    } else {
        (
            (0..n_columns).map(|c| format!("column_{c}")).collect(),
            &rows[..],
        )
    };
    if data_rows.is_empty() {
        return Err(IngestError::Empty);
    }

    let columns: Vec<Column> = (0..n_columns)
        .map(|c| {
            if data_rows.iter().all(|row| parses(&row[c])) {
                Column::Numeric(
                    data_rows
                        .iter()
                        .map(|row| row[c].trim().parse::<f32>().unwrap())
                        .collect(),
                )
            } else {
                Column::Text(
                    data_rows
                        .iter()
                        .map(|row| row[c].trim().to_owned())
                        .collect(),
                )
            }
        })
        .collect();

    if !columns.iter().any(|c| matches!(c, Column::Numeric(_))) {
        return Err(IngestError::NoNumericColumns);
    }

    Ok(assemble(names, columns, data_rows.len()))
}

/// Tells the delimiter of the first line apart, tab versus comma.
fn sniff_delimiter(bytes: &[u8]) -> u8 {
    let first_line = bytes.split(|&b| b == b'\n').next().unwrap_or_default();
    let tabs = first_line.iter().filter(|&&b| b == b'\t').count();
    let commas = first_line.iter().filter(|&&b| b == b',').count();
    if tabs > 0 && tabs >= commas {
        b'\t'
    } else {
        b','
    }
}

/// Parses Parquet content through the Arrow reader.
#[cfg(feature = "parquet")]
fn parse_parquet(bytes: &[u8]) -> Result<Dataset, IngestError> {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use arrow_schema::DataType;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))?.build()?;

    let mut names: Vec<String> = Vec::new();
    let mut columns: Vec<Column> = Vec::new();
    let mut n_samples = 0;

    for batch in reader {
        let batch = batch?;
        let schema = batch.schema();

        if columns.is_empty() {
            for field in schema.fields() {
                names.push(field.name().clone());
                columns.push(if field.data_type().is_numeric() {
                    Column::Numeric(Vec::new())
                } else {
                    Column::Text(Vec::new())
                });
            }
        }

        n_samples += batch.num_rows();

        for (index, column) in columns.iter_mut().enumerate() {
            let array = batch.column(index);
            match column {
                Column::Numeric(values) => {
                    let casted = arrow_cast::cast(array, &DataType::Float32).map_err(|_| {
                        IngestError::UnsupportedColumnType {
                            column: names[index].clone(),
                            data_type: array.data_type().to_string(),
                        }
                    })?;
                    if casted.null_count() > 0 {
                        return Err(IngestError::MissingValues {
                            column: names[index].clone(),
                        });
                    }
                    values.extend(casted.as_primitive::<Float32Type>().values().iter());
                }
                Column::Text(values) => {
                    let casted = arrow_cast::cast(array, &DataType::Utf8).map_err(|_| {
                        IngestError::UnsupportedColumnType {
                            column: names[index].clone(),
                            data_type: array.data_type().to_string(),
                        }
                    })?;
                    let strings = casted.as_string::<i32>();
                    values.extend((0..strings.len()).map(|i| {
                        if strings.is_null(i) {
                            String::new()
                        } else {
                            strings.value(i).to_owned()
                        }
                    }));
                }
            }
        }
    }

    if n_samples == 0 {
        return Err(IngestError::Empty);
    }
    if !columns.iter().any(|c| matches!(c, Column::Numeric(_))) {
        return Err(IngestError::NoNumericColumns);
    }

    Ok(assemble(names, columns, n_samples))
}
