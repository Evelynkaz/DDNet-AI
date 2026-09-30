//! Out-of-core storage for a demo's timeline (task 8.4d).
//!
//! A demo's frames and samples are produced in order by the streaming pipeline and then read back
//! by the analyses, which only ever look at a neighbourhood of the frame they are working on (a
//! detector looks back and forward a few dozen snapshots, except for runs such as a long freeze,
//! which it follows to their end). Keeping the whole timeline in memory made the pipeline's peak
//! memory grow with demo length times players (14.5 GB on a 3.3 h demo).
//!
//! [`PagedWriter`] appends items in order; every [`PAGE_LEN`] items form a page that is encoded
//! (postcard + zstd level 1) and appended to a [`Spill`] file, an unnamed temporary file. A
//! [`PagedStore`] is the finished, immutable result: the file plus a small page index. A
//! [`PagedReader`] gives random access through a small LRU cache of decoded pages, so the resident
//! part of a store is the cache (a few MB) plus about 12 bytes of index per page, whatever the
//! demo length, and an analysis that scans the timeline touches each page a constant number of times.
//!
//! The spill file is unlinked as soon as it is created (nothing is left behind if the process is
//! killed); on platforms where an open file cannot be unlinked the removal is retried when the
//! last store is dropped.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::types::FrameRec;

/// Items per page.
pub const PAGE_LEN: usize = 256;
/// Decoded pages a [`PagedReader`] keeps.
pub const CACHE_PAGES: usize = 8;
/// zstd level of the spill pages (speed over size: the file is temporary).
const SPILL_LEVEL: i32 = 1;

/// The payload of the unwind a [`PagedReader`] starts when a page cannot be read back or decoded
/// (its `get` has no way to return an error). It is raised with `resume_unwind`, which does not
/// call the panic hook, and turned back into an `io::Error` by [`catch_spill`] (and, in the worker
/// threads of `run`, by [`spill_error`]); any other panic is left alone.
#[derive(Debug)]
pub struct SpillError(pub io::Error);

/// Extracts the i/o error from a caught panic payload, or hands the payload back.
pub fn spill_error(payload: Box<dyn std::any::Any + Send>) -> Result<io::Error, Box<dyn std::any::Any + Send>> {
    payload.downcast::<SpillError>().map(|e| e.0)
}

/// Runs `f`, turning a spill read failure inside it into an error; other panics propagate.
pub fn catch_spill<T>(f: impl FnOnce() -> T) -> io::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(payload) => match spill_error(payload) {
            Ok(e) => Err(e),
            Err(other) => std::panic::resume_unwind(other),
        },
    }
}

/// The temporary file the pages of one demo live in (shared by the demo's stores).
#[derive(Debug)]
pub struct Spill {
    inner: Mutex<SpillInner>,
    path: PathBuf,
}

#[derive(Debug)]
struct SpillInner {
    file: File,
    len: u64,
}

static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);

impl Spill {
    /// Creates an empty spill file in `dir` (the system temporary directory when `None`).
    pub fn create(dir: Option<&Path>) -> io::Result<Arc<Spill>> {
        let dir = dir.map_or_else(std::env::temp_dir, Path::to_path_buf);
        fs::create_dir_all(&dir)?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let path = dir.join(format!(
            "ddai-spill-{}-{}-{nanos}",
            std::process::id(),
            SPILL_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new().read(true).write(true).create_new(true).open(&path)?;
        // Unlink now: the open handle keeps the data; a killed process leaves nothing behind.
        let _ = fs::remove_file(&path);
        Ok(Arc::new(Spill {
            inner: Mutex::new(SpillInner { file, len: 0 }),
            path,
        }))
    }

    fn append(&self, bytes: &[u8]) -> io::Result<u64> {
        let mut g = self.inner.lock().expect("no panics while holding the lock");
        let offset = g.len;
        g.file.seek(SeekFrom::Start(offset))?;
        g.file.write_all(bytes)?;
        g.len += bytes.len() as u64;
        Ok(offset)
    }

    fn read(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut g = self.inner.lock().expect("no panics while holding the lock");
        g.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        g.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Bytes written so far.
    pub fn len(&self) -> u64 {
        self.inner.lock().expect("no panics while holding the lock").len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Spill {
    fn drop(&mut self) {
        // Only matters where the early unlink failed (no-op otherwise).
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Clone, Copy)]
struct PageLoc {
    offset: u64,
    len: u32,
}

/// Appends items and cuts them into spilled pages.
pub struct PagedWriter<T> {
    spill: Arc<Spill>,
    page_len: usize,
    tail: Vec<T>,
    pages: Vec<PageLoc>,
    len: usize,
}

impl<T: Serialize> PagedWriter<T> {
    pub fn new(spill: Arc<Spill>) -> Self {
        Self::with_page_len(spill, PAGE_LEN)
    }

    /// Pages of `page_len` items (at least one); tests use tiny pages to cross page boundaries
    /// constantly.
    pub fn with_page_len(spill: Arc<Spill>, page_len: usize) -> Self {
        PagedWriter {
            spill,
            page_len: page_len.max(1),
            tail: Vec::new(),
            pages: Vec::new(),
            len: 0,
        }
    }

    pub fn push(&mut self, item: T) -> io::Result<()> {
        self.tail.push(item);
        self.len += 1;
        if self.tail.len() == self.page_len {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.tail.is_empty() {
            return Ok(());
        }
        let raw = postcard::to_stdvec(&self.tail).map_err(io::Error::other)?;
        let packed = zstd::encode_all(raw.as_slice(), SPILL_LEVEL)?;
        let offset = self.spill.append(&packed)?;
        self.pages.push(PageLoc {
            offset,
            len: u32::try_from(packed.len()).map_err(io::Error::other)?,
        });
        self.tail.clear();
        Ok(())
    }

    /// Writes the last partial page and returns the readable store.
    pub fn finish(mut self) -> io::Result<PagedStore<T>> {
        self.flush()?;
        Ok(PagedStore {
            spill: self.spill,
            page_len: self.page_len,
            pages: self.pages,
            len: self.len,
            _item: PhantomData,
        })
    }
}

/// A finished store: `len` items in pages of a fixed length (the last one may be shorter).
pub struct PagedStore<T> {
    spill: Arc<Spill>,
    page_len: usize,
    pages: Vec<PageLoc>,
    len: usize,
    _item: PhantomData<fn() -> T>,
}

impl<T> std::fmt::Debug for PagedStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedStore")
            .field("len", &self.len)
            .field("pages", &self.pages.len())
            .finish()
    }
}

impl<T> PagedStore<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes of the spill pages of this store.
    pub fn stored_bytes(&self) -> u64 {
        self.pages.iter().map(|p| u64::from(p.len)).sum()
    }
}

impl<T: DeserializeOwned> PagedStore<T> {
    /// A random-access reader with its own page cache. Not `Sync`: one per thread.
    pub fn reader(&self) -> PagedReader<'_, T> {
        PagedReader::with_cache(self, CACHE_PAGES)
    }

    /// A reader with a custom cache size (at least one page); tests use tiny caches to show the
    /// analyses do not depend on the cache holding more than their working set.
    pub fn reader_with_cache(&self, pages: usize) -> PagedReader<'_, T> {
        PagedReader::with_cache(self, pages)
    }
}

/// An item of a page, keeping the decoded page alive.
pub struct Item<T> {
    page: Rc<Vec<T>>,
    idx: usize,
}

impl<T> std::ops::Deref for Item<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.page[self.idx]
    }
}

/// Random access to a [`PagedStore`] through an LRU cache of decoded pages.
pub struct PagedReader<'a, T> {
    store: &'a PagedStore<T>,
    cache: std::cell::RefCell<Vec<(usize, Rc<Vec<T>>)>>,
    capacity: usize,
    loads: std::cell::Cell<u64>,
}

impl<'a, T: DeserializeOwned> PagedReader<'a, T> {
    fn with_cache(store: &'a PagedStore<T>, pages: usize) -> Self {
        PagedReader {
            store,
            cache: std::cell::RefCell::new(Vec::new()),
            capacity: pages.max(1),
            loads: std::cell::Cell::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.store.len
    }

    pub fn is_empty(&self) -> bool {
        self.store.len == 0
    }

    /// Pages decoded so far (cache misses).
    pub fn page_loads(&self) -> u64 {
        self.loads.get()
    }

    /// The item at `i`.
    ///
    /// # Panics
    ///
    /// When `i >= len()`. When the spill file cannot be read back or a page does not decode (a file
    /// this process wrote itself a moment ago: a broken disk, not bad input) it unwinds with a
    /// [`SpillError`] payload and without calling the panic hook; callers that can fail cleanly
    /// wrap the work in [`catch_spill`].
    pub fn get(&self, i: usize) -> Item<T> {
        assert!(i < self.store.len, "store index {i} out of {}", self.store.len);
        let page_len = self.store.page_len;
        Item {
            page: self.page(i / page_len),
            idx: i % page_len,
        }
    }

    fn page(&self, p: usize) -> Rc<Vec<T>> {
        let mut cache = self.cache.borrow_mut();
        if let Some(pos) = cache.iter().position(|(n, _)| *n == p) {
            // Most recently used goes last.
            let hit = cache.remove(pos);
            let page = Rc::clone(&hit.1);
            cache.push(hit);
            return page;
        }
        let loc = self.store.pages[p];
        let fail = |e: io::Error| -> ! { std::panic::resume_unwind(Box::new(SpillError(e))) };
        let packed = self
            .store
            .spill
            .read(loc.offset, loc.len as usize)
            .unwrap_or_else(|e| fail(e));
        let raw = zstd::decode_all(packed.as_slice()).unwrap_or_else(|e| fail(e));
        let page: Vec<T> =
            postcard::from_bytes(&raw).unwrap_or_else(|e| fail(io::Error::new(io::ErrorKind::InvalidData, e)));
        self.loads.set(self.loads.get() + 1);
        let page = Rc::new(page);
        if cache.len() >= self.capacity {
            cache.remove(0);
        }
        cache.push((p, Rc::clone(&page)));
        page
    }
}

/// The frames of one demo: a [`PagedStore`] plus the tick of every frame (4 bytes per frame, the
/// only per-frame memory; it serves the binary searches by tick).
#[derive(Debug)]
pub struct FrameStore {
    frames: PagedStore<FrameRec>,
    ticks: Vec<i32>,
}

pub type FrameReader<'a> = PagedReader<'a, FrameRec>;

impl FrameStore {
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn tick(&self, k: usize) -> i32 {
        self.ticks[k]
    }

    /// Tick of every frame.
    pub fn ticks(&self) -> &[i32] {
        &self.ticks
    }

    pub fn reader(&self) -> FrameReader<'_> {
        self.frames.reader()
    }

    pub fn reader_with_cache(&self, pages: usize) -> FrameReader<'_> {
        self.frames.reader_with_cache(pages)
    }

    pub fn stored_bytes(&self) -> u64 {
        self.frames.stored_bytes()
    }

    /// A store of `frames` in a fresh spill file in the system temporary directory (tests and
    /// small tools; the pipeline writes through [`FrameStoreWriter`]).
    pub fn from_frames(frames: &[FrameRec]) -> io::Result<FrameStore> {
        Self::from_frames_paged(frames, PAGE_LEN)
    }

    /// [`FrameStore::from_frames`] with pages of `page_len` frames.
    pub fn from_frames_paged(frames: &[FrameRec], page_len: usize) -> io::Result<FrameStore> {
        let mut w = FrameStoreWriter::with_page_len(Spill::create(None)?, page_len);
        for f in frames {
            w.push(f.clone())?;
        }
        w.finish()
    }
}

/// Appends frames.
pub struct FrameStoreWriter {
    frames: PagedWriter<FrameRec>,
    ticks: Vec<i32>,
}

impl FrameStoreWriter {
    pub fn new(spill: Arc<Spill>) -> Self {
        Self::with_page_len(spill, PAGE_LEN)
    }

    pub fn with_page_len(spill: Arc<Spill>, page_len: usize) -> Self {
        FrameStoreWriter {
            frames: PagedWriter::with_page_len(spill, page_len),
            ticks: Vec::new(),
        }
    }

    pub fn push(&mut self, frame: FrameRec) -> io::Result<()> {
        self.ticks.push(frame.tick);
        self.frames.push(frame)
    }

    pub fn finish(self) -> io::Result<FrameStore> {
        Ok(FrameStore {
            frames: self.frames.finish()?,
            ticks: self.ticks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_round_trip_through_pages_in_any_access_order() {
        let spill = Spill::create(None).unwrap();
        let mut w = PagedWriter::new(spill);
        let n = PAGE_LEN * 3 + 17;
        for i in 0..n {
            w.push((i as u32, vec![i as u8; i % 5])).unwrap();
        }
        let store = w.finish().unwrap();
        assert_eq!(store.len(), n);
        for cache in [1, 2, CACHE_PAGES] {
            let r = store.reader_with_cache(cache);
            // Forward, backward, and jumping between far pages.
            let order: Vec<usize> = (0..n)
                .chain((0..n).rev())
                .chain((0..n).map(|i| (i * 7919) % n))
                .collect();
            for i in order {
                let item = r.get(i);
                assert_eq!(item.0, i as u32);
                assert_eq!(item.1, vec![i as u8; i % 5]);
            }
        }
    }

    #[test]
    fn a_sequential_scan_loads_every_page_once() {
        let spill = Spill::create(None).unwrap();
        let mut w = PagedWriter::new(spill);
        for i in 0..PAGE_LEN * 5 {
            w.push(i as u64).unwrap();
        }
        let store = w.finish().unwrap();
        let r = store.reader_with_cache(2);
        for i in 0..store.len() {
            assert_eq!(*r.get(i), i as u64);
            // A look-back of one item (the detectors' `k - 1`) does not reload anything.
            if i > 0 {
                assert_eq!(*r.get(i - 1), i as u64 - 1);
            }
        }
        assert_eq!(r.page_loads(), 5);
    }

    #[test]
    fn tiny_pages_work_like_big_ones() {
        let spill = Spill::create(None).unwrap();
        let mut w = PagedWriter::with_page_len(spill, 3);
        for i in 0..20u32 {
            w.push(i).unwrap();
        }
        let store = w.finish().unwrap();
        let r = store.reader_with_cache(2);
        for i in (0..20).rev().chain(0..20) {
            assert_eq!(*r.get(i), i as u32);
        }
    }

    #[test]
    fn a_spill_file_that_cannot_be_read_back_is_an_error_not_a_panic() {
        let spill = Spill::create(None).unwrap();
        let mut w = PagedWriter::new(Arc::clone(&spill));
        for i in 0..PAGE_LEN * 2 {
            w.push(i as u32).unwrap();
        }
        let store = w.finish().unwrap();
        let r = store.reader();
        assert_eq!(*r.get(3), 3);
        // Damage the file behind the store's back: the next page load fails.
        spill.inner.lock().unwrap().file.set_len(4).unwrap();
        let err = catch_spill(|| *r.get(PAGE_LEN + 1)).unwrap_err();
        assert!(!err.to_string().is_empty());
        // Other panics are not swallowed.
        let other = std::panic::catch_unwind(|| catch_spill(|| -> u32 { panic!("a bug") }));
        assert!(other.is_err());
    }

    #[test]
    fn an_empty_store_is_fine() {
        let spill = Spill::create(None).unwrap();
        let store = PagedWriter::<u8>::new(spill).finish().unwrap();
        assert!(store.is_empty());
        assert_eq!(store.stored_bytes(), 0);
    }

    #[test]
    fn the_spill_file_leaves_no_trace_in_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let spill = Spill::create(Some(dir.path())).unwrap();
        let mut w = PagedWriter::new(Arc::clone(&spill));
        for i in 0..PAGE_LEN * 2 {
            w.push(i as u32).unwrap();
        }
        let store = w.finish().unwrap();
        assert!(store.stored_bytes() > 0);
        assert!(spill.len() >= store.stored_bytes());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
        assert_eq!(*store.reader().get(PAGE_LEN + 1), PAGE_LEN as u32 + 1);
    }
}
