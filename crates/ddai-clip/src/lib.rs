//! `ddai-clip` (task 4.3): the live bot's clips.
//!
//! - [`format`]: the clip format v1 (postcard + zstd), the data types;
//! - [`record`]: the 30 s ring recorder with fixed storage (no allocation per frame);
//! - [`held`]: what became of every block we made (held or escaped, and why; the live diagnosis of task 3.10);
//! - [`incidents`]: `findIncidents` with real events, `mergeOverlapping`, `summarise`;
//! - [`store`]: names, the autoclip scheduling and pruning of `~/aiddnet/data/bot/clips/`;
//! - [`replay`]: the offline bit-exact replay of a clip on `ddai-world`/`ddai-physics`.
//!
//! No chat, no network, no nicknames: a clip holds numbers and the 4.1 tags (`c<id>-<hash>`).

pub mod format;
pub mod held;
pub mod incidents;
pub mod record;
pub mod replay;
pub mod store;

pub use format::{
    BotRec, CharRec, Clip, ClipError, ClipEvent, ClipHeader, ClipReason, DdRec, FORMAT_VERSION, Frame, InputRec,
    KillWhy, MAGIC, MAX_EVENTS, MAX_PROJECTILES, MAX_SENT, MAX_TEES, PlayerTag, ProjRec, RING_FRAMES, SentRec,
    SwitchRec, TeeRec,
};
pub use incidents::{Incident, find_incidents, merge_overlapping, summarise};
pub use record::{ClipMeta, FrameBuilder, Recorder};
