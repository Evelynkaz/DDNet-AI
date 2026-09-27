//! `inspect <file>`: schema, row count and compression codec of a Feather (Arrow IPC file)
//! without decompressing any record batch body — only the small per-batch metadata (a flatbuffer
//! a few hundred bytes long) is read for each block, so this is cheap even on the 508 MB weights
//! file.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow::datatypes::Schema;
use arrow::ipc::convert::try_fb_to_schema;
use arrow::ipc::reader::read_footer_length;
use arrow::ipc::root_as_footer;

/// One field of a Feather file's schema, formatted for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInfo {
    pub name: String,
    pub data_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectReport {
    pub fields: Vec<FieldInfo>,
    pub row_count: u64,
    pub num_batches: usize,
    /// Compression codec name (e.g. `"LZ4_FRAME"`, `"ZSTD"`) taken from the first record batch
    /// that has one; `None` if the file has no record batches or none are compressed.
    pub compression: Option<String>,
}

const TRAILER_LEN: usize = 10;

/// Reads the IPC file footer and every record-batch block's *metadata* (never a batch body) to
/// build an [`InspectReport`].
pub fn inspect_file(path: &Path) -> Result<InspectReport> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    if file_len < TRAILER_LEN as u64 {
        bail!(
            "{}: too small to be an Arrow IPC file ({} bytes)",
            path.display(),
            file_len
        );
    }

    reader.seek(SeekFrom::End(-(TRAILER_LEN as i64)))?;
    let mut trailer = [0u8; TRAILER_LEN];
    reader.read_exact(&mut trailer)?;
    let footer_len = read_footer_length(trailer)
        .with_context(|| format!("{}: not a valid Arrow IPC file (bad footer trailer)", path.display()))?;

    let footer_start = file_len
        .checked_sub(TRAILER_LEN as u64)
        .and_then(|x| x.checked_sub(footer_len as u64))
        .with_context(|| format!("{}: footer length {footer_len} is larger than the file", path.display()))?;
    reader.seek(SeekFrom::Start(footer_start))?;
    let mut footer_buf = vec![0u8; footer_len];
    reader.read_exact(&mut footer_buf)?;
    let footer =
        root_as_footer(&footer_buf).map_err(|e| anyhow::anyhow!("{}: invalid IPC footer: {e:?}", path.display()))?;

    let schema_fb = footer
        .schema()
        .with_context(|| format!("{}: IPC footer has no schema", path.display()))?;
    let schema: Schema =
        try_fb_to_schema(schema_fb).with_context(|| format!("{}: cannot decode schema", path.display()))?;
    let fields = schema
        .fields()
        .iter()
        .map(|f| FieldInfo {
            name: f.name().clone(),
            data_type: format!("{:?}", f.data_type()),
        })
        .collect();

    let blocks = footer
        .recordBatches()
        .map(|v| v.iter().collect::<Vec<_>>())
        .unwrap_or_default();

    let mut row_count: u64 = 0;
    let mut compression: Option<String> = None;
    for block in &blocks {
        // Read only this block's metadata prefix (`metaDataLength` bytes), never its body.
        let meta_len = block.metaDataLength();
        if meta_len <= 0 {
            bail!(
                "{}: record batch block has non-positive metaDataLength ({meta_len})",
                path.display()
            );
        }
        reader.seek(SeekFrom::Start(block.offset() as u64))?;
        let mut meta_buf = vec![0u8; meta_len as usize];
        reader.read_exact(&mut meta_buf)?;

        // Encapsulated IPC messages are prefixed by either a 4-byte continuation marker
        // (0xffffffff) followed by a 4-byte length, or (older, v1) just the 4-byte length; the
        // message flatbuffer itself starts after that prefix. `arrow-ipc`'s own reader applies
        // the same 4-vs-8-byte skip (see `arrow_ipc::reader::parse_message`, private but
        // reproduced here since it isn't exported).
        const CONTINUATION_MARKER: [u8; 4] = [0xff; 4];
        let msg_buf = if meta_buf.len() >= 4 && meta_buf[..4] == CONTINUATION_MARKER {
            &meta_buf[8..]
        } else {
            &meta_buf[4..]
        };
        let message = arrow::ipc::root_as_message(msg_buf)
            .map_err(|e| anyhow::anyhow!("{}: invalid IPC message: {e:?}", path.display()))?;
        let batch = message.header_as_record_batch().with_context(|| {
            format!(
                "{}: record batch block does not contain a RecordBatch message",
                path.display()
            )
        })?;
        row_count += batch.length() as u64;
        if compression.is_none()
            && let Some(c) = batch.compression()
        {
            compression = Some(format!("{:?}", c.codec()));
        }
    }

    Ok(InspectReport {
        fields,
        row_count,
        num_batches: blocks.len(),
        compression,
    })
}
