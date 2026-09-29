//! Technique catalogue (`docs/research/block-knowledge.md` §2.1, T1-T18, plus the D-048 wall and
//! ceiling hook) and the tag bits stored in [`crate::types::SampleRec::tags`].

use serde::{Deserialize, Serialize};

/// A block technique. `T1..=T18` follow the catalogue numbering; [`Technique::WallHook`] is the
/// owner's explicit interest (D-048): a hook attached to terrain used to change trajectory under
/// threat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Technique {
    T1 = 1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
    T16,
    T17,
    T18,
    WallHook,
}

impl Technique {
    pub const ALL: [Technique; 19] = [
        Technique::T1,
        Technique::T2,
        Technique::T3,
        Technique::T4,
        Technique::T5,
        Technique::T6,
        Technique::T7,
        Technique::T8,
        Technique::T9,
        Technique::T10,
        Technique::T11,
        Technique::T12,
        Technique::T13,
        Technique::T14,
        Technique::T15,
        Technique::T16,
        Technique::T17,
        Technique::T18,
        Technique::WallHook,
    ];

    /// Short stable code used in reports and the manifest.
    pub fn code(self) -> &'static str {
        match self {
            Technique::T1 => "T1",
            Technique::T2 => "T2",
            Technique::T3 => "T3",
            Technique::T4 => "T4",
            Technique::T5 => "T5",
            Technique::T6 => "T6",
            Technique::T7 => "T7",
            Technique::T8 => "T8",
            Technique::T9 => "T9",
            Technique::T10 => "T10",
            Technique::T11 => "T11",
            Technique::T12 => "T12",
            Technique::T13 => "T13",
            Technique::T14 => "T14",
            Technique::T15 => "T15",
            Technique::T16 => "T16",
            Technique::T17 => "T17",
            Technique::T18 => "T18",
            Technique::WallHook => "WH",
        }
    }

    /// Short English name (the report itself is written in Russian around these codes).
    pub fn name(self) -> &'static str {
        match self {
            Technique::T1 => "hook drag through the edge into freeze",
            Technique::T2 => "hammer into a side freeze wall",
            Technique::T3 => "swing up",
            Technique::T4 => "pull down a jumper",
            Technique::T5 => "body push at the edge",
            Technique::T6 => "hold the way-block spot",
            Technique::T7 => "finish a frozen enemy (perma)",
            Technique::T8 => "hammer a frozen enemy",
            Technique::T9 => "escape from being hooked",
            Technique::T10 => "save when thrown up",
            Technique::T11 => "edge stance",
            Technique::T12 => "second-jump save",
            Technique::T13 => "regain the double jump",
            Technique::T14 => "panic hook to a wall or ceiling",
            Technique::T15 => "hook duel",
            Technique::T16 => "rescue (hammerhit)",
            Technique::T17 => "do not stand by a frozen enemy",
            Technique::T18 => "CLB spawn / hub situations",
            Technique::WallHook => "wall/ceiling hook under threat",
        }
    }

    /// Whether a detector exists (see `technique.rs`); the others cannot be observed from
    /// snapshots of a client demo, the reason is given by [`Technique::not_detected_reason`].
    pub fn detectable(self) -> bool {
        self.not_detected_reason().is_none()
    }

    pub fn not_detected_reason(self) -> Option<&'static str> {
        match self {
            Technique::T6 => Some("needs the CLB way-block zones; the archive has no CLB way-block play"),
            Technique::T7 => {
                Some("indistinguishable from T8 without server state (who stays frozen is not the technique)")
            }
            Technique::T16 => Some("teammates do not exist in block; low priority in the catalogue"),
            Technique::T17 => Some("an avoidance rule (an action not taken), not an event visible in snapshots"),
            Technique::T18 => Some("CLB-only spawn/hub geometry; the archive has (almost) no CLB"),
            _ => None,
        }
    }

    pub fn bit(self) -> u32 {
        1u32 << (self as u32 - 1)
    }

    pub fn from_code(code: &str) -> Option<Technique> {
        Technique::ALL.iter().copied().find(|t| t.code() == code)
    }
}

/// Skill-signal tag bits (above the technique bits).
pub mod signal {
    /// Within `lead_ticks` before a credited block of this player.
    pub const LEADS_TO_BLOCK: u32 = 1 << 24;
    /// Within `lead_ticks` before an unforced (not attributed to an enemy) freeze of this player.
    pub const LEADS_TO_SELF_FREEZE: u32 = 1 << 25;
    /// Within `lead_ticks` before an attributed freeze of this player (someone blocked them).
    pub const LEADS_TO_BLOCKED: u32 = 1 << 26;
}

/// Iterates the techniques set in `tags`.
pub fn techniques_in(tags: u32) -> impl Iterator<Item = Technique> {
    Technique::ALL.into_iter().filter(move |t| tags & t.bit() != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_distinct_and_below_the_signal_bits() {
        let mut seen = 0u32;
        for t in Technique::ALL {
            assert_eq!(seen & t.bit(), 0, "{t:?} overlaps");
            seen |= t.bit();
        }
        assert!(seen < signal::LEADS_TO_BLOCK);
    }

    #[test]
    fn codes_round_trip() {
        for t in Technique::ALL {
            assert_eq!(Technique::from_code(t.code()), Some(t));
        }
        assert_eq!(Technique::from_code("T99"), None);
    }

    #[test]
    fn catalogue_numbering_matches_the_document() {
        assert_eq!(Technique::T1 as u8, 1);
        assert_eq!(Technique::T18 as u8, 18);
        assert_eq!(Technique::T9.code(), "T9");
    }
}
