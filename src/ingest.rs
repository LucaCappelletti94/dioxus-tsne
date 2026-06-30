//! Parsing of user supplied files (CSV, TSV, Parquet, Arrow IPC / Feather, and
//! NumPy `.npy`) into a numeric matrix plus candidate label columns for
//! coloring.

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
    Parquet(parquet::errors::ParquetError),
    /// Malformed Arrow IPC / Feather content.
    Arrow(arrow_schema::ArrowError),
    /// Malformed or unsupported NumPy `.npy` content.
    Npy(String),
    /// Malformed or unreadable spreadsheet (`.xlsx`, `.ods`, `.xls`, `.xlsb`).
    Spreadsheet(String),
    /// A numeric Arrow column (Parquet or IPC) contains nulls.
    MissingValues {
        /// Name of the offending column.
        column: String,
    },
    /// An Arrow column has a type that can be represented neither as f32 nor as
    /// a string label.
    UnsupportedColumnType {
        /// Name of the offending column.
        column: String,
        /// The Arrow data type of the column.
        data_type: String,
    },
}

impl fmt::Display for IngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestError::Empty => write!(f, "the file contains no data rows"),
            IngestError::NoNumericColumns => {
                write!(f, "no fully numeric column found, nothing to embed")
            }
            IngestError::Csv(e) => write!(f, "malformed CSV/TSV: {e}"),
            IngestError::Parquet(e) => write!(f, "malformed Parquet: {e}"),
            IngestError::Arrow(e) => write!(f, "malformed Arrow IPC/Feather: {e}"),
            IngestError::Npy(e) => write!(f, "malformed NumPy .npy: {e}"),
            IngestError::Spreadsheet(e) => write!(f, "malformed spreadsheet: {e}"),
            IngestError::MissingValues { column } => {
                write!(f, "numeric column {column:?} contains missing values")
            }
            IngestError::UnsupportedColumnType { column, data_type } => {
                write!(f, "column {column:?} has unsupported type {data_type}")
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

impl From<parquet::errors::ParquetError> for IngestError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        IngestError::Parquet(e)
    }
}

impl From<arrow_schema::ArrowError> for IngestError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        IngestError::Arrow(e)
    }
}

/// A parsed column before the row major assembly.
enum Column {
    Numeric(Vec<f32>),
    Text(Vec<String>),
}

/// A parsed file: the [`Dataset`] plus, for spreadsheets, the worksheet names
/// (for a sheet picker) and the index of the one parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedFile {
    /// The parsed dataset (matrix and label columns).
    pub dataset: Dataset,
    /// Worksheet names in file order, empty for non-spreadsheet formats.
    pub sheets: Vec<String>,
    /// Index into `sheets` of the worksheet `dataset` came from, 0 otherwise.
    pub sheet: usize,
}

/// Parses a user supplied file into a [`Dataset`], reading the first worksheet
/// of a spreadsheet. See [`parse_file`] for the format detection rules.
///
/// # Errors
///
/// Returns an [`IngestError`] when the content is malformed, empty or contains
/// no numeric column.
pub fn parse_dataset(file_name: &str, bytes: &[u8]) -> Result<Dataset, IngestError> {
    parse_file(file_name, bytes, 0).map(|parsed| parsed.dataset)
}

/// Parses a user supplied file into a [`ParsedFile`].
///
/// The format is detected by magic number, falling back to the extension:
/// Parquet (`PAR1`), Arrow IPC / Feather (`ARROW1`) and NumPy `.npy`
/// (`\x93NUMPY`) by magic, spreadsheets (`.xlsx`, `.ods`, `.xls`, `.xlsb`) by
/// extension only (they share a generic container magic), everything else as
/// delimited text. Fully-numeric columns become features, the rest label
/// columns for coloring. For a spreadsheet, `sheet` selects the worksheet
/// (clamped into range), ignored for every other format.
///
/// # Errors
///
/// Returns an [`IngestError`] when the content is malformed, empty or contains
/// no numeric column.
pub fn parse_file(file_name: &str, bytes: &[u8], sheet: usize) -> Result<ParsedFile, IngestError> {
    const PARQUET_MAGIC: &[u8] = b"PAR1";
    const ARROW_MAGIC: &[u8] = b"ARROW1";
    const NPY_MAGIC: &[u8] = b"\x93NUMPY";

    let extension = file_name.rsplit('.').next().map(str::to_ascii_lowercase);
    let extension = extension.as_deref();

    if matches!(extension, Some("xlsx" | "ods" | "xls" | "xlsb")) {
        let (dataset, sheets, active) = parse_spreadsheet(bytes, sheet)?;
        return Ok(ParsedFile {
            dataset,
            sheets,
            sheet: active,
        });
    }

    let dataset = if bytes.starts_with(NPY_MAGIC) || extension == Some("npy") {
        parse_npy(bytes)?
    } else if bytes.starts_with(PARQUET_MAGIC) || extension == Some("parquet") {
        parse_parquet(bytes)?
    } else if bytes.starts_with(ARROW_MAGIC)
        || matches!(extension, Some("arrow" | "feather" | "ipc"))
    {
        parse_arrow_ipc(bytes)?
    } else {
        parse_delimited(bytes)?
    };

    Ok(ParsedFile {
        dataset,
        sheets: Vec::new(),
        sheet: 0,
    })
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

    let mut rows: Vec<Vec<String>> = Vec::new();
    for record in reader.records() {
        rows.push(record?.iter().map(str::to_owned).collect());
    }

    dataset_from_string_rows(rows)
}

/// Builds a [`Dataset`] from rows of stringified cells, shared by delimited text
/// and spreadsheets. The first row is a header when dropping it makes some column
/// fully numeric. Fully-numeric columns become features, the rest label columns.
/// A cell index past a row's end reads as empty.
fn dataset_from_string_rows(rows: Vec<Vec<String>>) -> Result<Dataset, IngestError> {
    if rows.is_empty() {
        return Err(IngestError::Empty);
    }

    let n_columns = rows[0].len();
    fn cell(row: &[String], c: usize) -> &str {
        row.get(c).map(String::as_str).unwrap_or("")
    }
    let parses = |field: &str| field.trim().parse::<f32>().is_ok();

    // A column is numeric over the candidate data rows (all rows but the
    // first) when every value parses as a float. The first row belongs to a
    // header when some such column has a non parseable first value.
    let has_header = rows.len() > 1
        && (0..n_columns).any(|c| {
            rows[1..].iter().all(|row| parses(cell(row, c))) && !parses(cell(&rows[0], c))
        });

    let (names, data_rows): (Vec<String>, &[Vec<String>]) = if has_header {
        (rows[0].iter().map(String::clone).collect(), &rows[1..])
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
            if data_rows.iter().all(|row| parses(cell(row, c))) {
                Column::Numeric(
                    data_rows
                        .iter()
                        .map(|row| cell(row, c).trim().parse::<f32>().unwrap())
                        .collect(),
                )
            } else {
                Column::Text(
                    data_rows
                        .iter()
                        .map(|row| cell(row, c).trim().to_owned())
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

/// Parses one worksheet of a spreadsheet (`.xlsx`, `.ods`, `.xls`, `.xlsb`),
/// returning the [`Dataset`], all worksheet names, and the index parsed (the
/// `sheet` argument clamped into range). Cells are stringified and fed through
/// the same typing as delimited text. Fully empty rows are dropped so trailing
/// blanks do not poison it.
fn parse_spreadsheet(
    bytes: &[u8],
    sheet: usize,
) -> Result<(Dataset, Vec<String>, usize), IngestError> {
    use calamine::{Reader, open_workbook_auto_from_rs};
    use std::io::Cursor;

    // A `Cursor<&[u8]>` is Read + Seek + Clone, so the auto-detector can probe
    // the formats by cheaply cloning the cursor (a pointer copy) rather than
    // deep-copying the file, which a `Cursor<Vec<u8>>` would do on every probe.
    let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes))
        .map_err(|e| IngestError::Spreadsheet(e.to_string()))?;
    let sheets = workbook.sheet_names();
    if sheets.is_empty() {
        return Err(IngestError::Empty);
    }
    let active = sheet.min(sheets.len() - 1);
    let range = workbook
        .worksheet_range(&sheets[active])
        .map_err(|e| IngestError::Spreadsheet(e.to_string()))?;

    let rows: Vec<Vec<String>> = range
        .rows()
        .map(|row| row.iter().map(cell_to_string).collect::<Vec<String>>())
        .filter(|row| row.iter().any(|cell| !cell.is_empty()))
        .collect();

    let dataset = dataset_from_string_rows(rows)?;
    Ok((dataset, sheets, active))
}

/// Stringifies a spreadsheet cell for the delimited-text column typing. Numbers
/// (and date serials) become parseable floats, booleans and ISO date/time
/// strings become labels, empty and error cells become the empty string.
fn cell_to_string(cell: &calamine::Data) -> String {
    use calamine::Data;
    match cell {
        Data::Int(value) => value.to_string(),
        Data::Float(value) => value.to_string(),
        Data::DateTime(value) => value.as_f64().to_string(),
        Data::Bool(value) => value.to_string(),
        Data::String(value) => value.clone(),
        Data::DateTimeIso(value) | Data::DurationIso(value) => value.clone(),
        Data::Error(_) | Data::Empty => String::new(),
    }
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

/// Builds a [`Dataset`] from a stream of Arrow record batches, the shared path
/// for both Parquet and Arrow IPC. Numeric fields become matrix features, every
/// other field a string label column. Columns are taken from the first batch's
/// schema, which all batches share.
fn dataset_from_batches<I>(batches: I) -> Result<Dataset, IngestError>
where
    I: IntoIterator<Item = Result<arrow_array::RecordBatch, IngestError>>,
{
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use arrow_schema::DataType;

    let mut names: Vec<String> = Vec::new();
    let mut columns: Vec<Column> = Vec::new();
    let mut n_samples = 0;

    for batch in batches {
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

/// Parses Parquet content through the Arrow reader.
fn parse_parquet(bytes: &[u8]) -> Result<Dataset, IngestError> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))?.build()?;
    dataset_from_batches(reader.map(|batch| batch.map_err(IngestError::from)))
}

/// Parses Arrow IPC file content (the Feather v2 format) through the same Arrow
/// reader as Parquet. Only the file format (`ARROW1` magic) is supported, not
/// the streaming format.
fn parse_arrow_ipc(bytes: &[u8]) -> Result<Dataset, IngestError> {
    use arrow_ipc::reader::FileReader;
    use std::io::Cursor;

    let reader = FileReader::try_new(Cursor::new(bytes), None)?;
    dataset_from_batches(reader.map(|batch| batch.map_err(IngestError::from)))
}

/// Parses a NumPy `.npy` array into a feature-only [`Dataset`].
///
/// A `.npy` holds a single homogeneous array with no column names or labels, so
/// the result is a pure feature matrix (`column_0..` names, no label columns). A
/// 2-D array maps to `(n_samples, n_features)`, a 1-D array to a single feature
/// column. Numeric dtypes (float, signed/unsigned integer, bool) are converted
/// to `f32`. Structured (record) dtypes and float16 are rejected.
fn parse_npy(bytes: &[u8]) -> Result<Dataset, IngestError> {
    const MAGIC: &[u8] = b"\x93NUMPY";

    let bad = |m: &str| IngestError::Npy(m.to_owned());
    if !bytes.starts_with(MAGIC) {
        return Err(bad("not a .npy file (bad magic)"));
    }
    let major = *bytes.get(6).ok_or_else(|| bad("truncated header"))?;

    // The header length field is u16 in format v1, u32 in v2 and later.
    let (header_start, header_len) = if major >= 2 {
        let raw = bytes.get(8..12).ok_or_else(|| bad("truncated header"))?;
        (12, u32::from_le_bytes(raw.try_into().unwrap()) as usize)
    } else {
        let raw = bytes.get(8..10).ok_or_else(|| bad("truncated header"))?;
        (10, u16::from_le_bytes(raw.try_into().unwrap()) as usize)
    };

    let header = bytes
        .get(header_start..header_start + header_len)
        .ok_or_else(|| bad("truncated header"))?;
    let header = std::str::from_utf8(header).map_err(|_| bad("non-UTF8 header"))?;

    let descr = npy_field(header, "descr").ok_or_else(|| bad("header missing 'descr'"))?;
    let fortran = npy_field(header, "fortran_order").is_some_and(|v| v.contains("True"));
    let shape = npy_shape(header).ok_or_else(|| bad("header missing 'shape'"))?;
    let (n_samples, n_features) = match shape.as_slice() {
        [n] => (*n, 1usize),
        [n, m] => (*n, *m),
        [] => return Err(bad("0-D arrays are not supported")),
        _ => return Err(bad("only 1-D and 2-D arrays are supported")),
    };
    if n_samples == 0 || n_features == 0 {
        return Err(IngestError::Empty);
    }

    let (kind, size, little) = parse_npy_descr(descr)?;
    let count = n_samples
        .checked_mul(n_features)
        .ok_or_else(|| bad("shape too large"))?;
    let payload = &bytes[header_start + header_len..];
    if payload.len() < count * size {
        return Err(bad("data shorter than the shape implies"));
    }

    let mut data = vec![0f32; count];
    for i in 0..n_samples {
        for j in 0..n_features {
            // Fortran (column-major) storage interleaves differently than the
            // row-major C order, so map the logical index to the stored one.
            let src = if fortran {
                j * n_samples + i
            } else {
                i * n_features + j
            };
            let offset = src * size;
            data[i * n_features + j] =
                read_npy_elem(&payload[offset..offset + size], kind, little)?;
        }
    }

    Ok(Dataset {
        data,
        n_samples,
        n_features,
        feature_names: (0..n_features).map(|c| format!("column_{c}")).collect(),
        label_columns: Vec::new(),
    })
}

/// Returns the value token of a scalar `'key': value` entry in a `.npy` header
/// dict, trimmed. Only valid for comma-free values (descr, fortran_order), not
/// the shape tuple (see [`npy_shape`]).
fn npy_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("'{key}'");
    let after_key = &header[header.find(&needle)? + needle.len()..];
    let after_colon = &after_key[after_key.find(':')? + 1..];
    let end = after_colon.find(',').unwrap_or(after_colon.len());
    Some(after_colon[..end].trim())
}

/// Parses the `'shape': (a, b, ...)` tuple from a `.npy` header.
fn npy_shape(header: &str) -> Option<Vec<usize>> {
    let after_key = &header[header.find("'shape'")?..];
    let open = after_key.find('(')? + 1;
    let close = after_key[open..].find(')')? + open;
    let mut dims = Vec::new();
    for token in after_key[open..close].split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        dims.push(token.parse::<usize>().ok()?);
    }
    Some(dims)
}

/// Decodes a `.npy` dtype string like `<f4`, `|u1` or `>i8` into the kind char
/// (`f`/`i`/`u`/`b`), the item size in bytes, and whether it is little-endian.
fn parse_npy_descr(descr: &str) -> Result<(u8, usize, bool), IngestError> {
    let descr = descr.trim().trim_matches(|c| c == '\'' || c == '"');
    let bytes = descr.as_bytes();
    let (mut index, little) = match bytes.first() {
        // `=` is native order, little-endian on the wasm target.
        Some(b'<' | b'=' | b'|') => (1, true),
        Some(b'>') => (1, false),
        _ => (0, true),
    };
    let kind = *bytes
        .get(index)
        .ok_or_else(|| IngestError::Npy(format!("bad dtype {descr:?}")))?;
    index += 1;
    let size: usize = descr[index..]
        .parse()
        .map_err(|_| IngestError::Npy(format!("bad dtype {descr:?}")))?;
    if !matches!(kind, b'f' | b'i' | b'u' | b'b') {
        return Err(IngestError::Npy(format!(
            "unsupported dtype '{}{size}'",
            kind as char
        )));
    }
    Ok((kind, size, little))
}

/// Reads one `.npy` element (`bytes.len() == size`) as `f32`.
fn read_npy_elem(bytes: &[u8], kind: u8, little: bool) -> Result<f32, IngestError> {
    macro_rules! read {
        ($t:ty, $n:literal) => {{
            let raw: [u8; $n] = bytes.try_into().unwrap();
            if little {
                <$t>::from_le_bytes(raw)
            } else {
                <$t>::from_be_bytes(raw)
            }
        }};
    }
    let value = match (kind, bytes.len()) {
        (b'f', 4) => read!(f32, 4),
        (b'f', 8) => read!(f64, 8) as f32,
        (b'i', 1) => bytes[0] as i8 as f32,
        (b'i', 2) => read!(i16, 2) as f32,
        (b'i', 4) => read!(i32, 4) as f32,
        (b'i', 8) => read!(i64, 8) as f32,
        (b'u', 1) => bytes[0] as f32,
        (b'u', 2) => read!(u16, 2) as f32,
        (b'u', 4) => read!(u32, 4) as f32,
        (b'u', 8) => read!(u64, 8) as f32,
        (b'b', 1) => {
            if bytes[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            return Err(IngestError::Npy(format!(
                "unsupported dtype '{}{}'",
                kind as char,
                bytes.len()
            )));
        }
    };
    Ok(value)
}
