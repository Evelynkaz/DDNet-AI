//! The bot's look in the game (D-068, task 5.6): nickname, clan and skin sent in `Cl_StartInfo`.
//!
//! - **Clan** `Neuroset` by default.
//! - **Skin**: a random pick from DDNet's stock skins at every start; `--skin-seed <n>` makes the pick deterministic
//!   (tests); `--skin <name>` or `skin = "..."` in `settings.toml` fixes it.
//! - **Nick**: stays a command-line decision (`--name`; `Muha` on Swarfey, where it is the one nick the allow-list
//!   `live-servers.toml` pins). It is deliberately not in `settings.toml`: the single-instance lock and the
//!   allow-list gate key on it before the settings are read.
//!
//! Precedence, highest first: `--skin` / `--clan`, then `--skin-seed` (skin only), then `settings.toml`, then the
//! default (clan) or a random pick (skin). The server never validates a skin name (the client falls back to the default
//! skin for one it does not have), but a name that is not made of the plain characters of a skin file name is refused
//! here so a typo does not go to the wire.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};

use crate::settings::Settings;

/// The clan D-068 fixes.
pub const DEFAULT_CLAN: &str = "Neuroset";

/// The pool of the random pick: the vanilla Teeworlds skin set, which DDNet ships in `data/skins` among about a hundred
/// more (community and special skins). `x_ninja`, the ninja power-up's skin, is left out: it looks like a power-up, not
/// like a player.
pub const DEFAULT_SKINS: [&str; 16] = [
    "default",
    "bluekitty",
    "bluestripe",
    "brownbear",
    "cammo",
    "cammostripes",
    "coala",
    "limekitty",
    "pinky",
    "redbopp",
    "redstripe",
    "saddo",
    "toptri",
    "twinbop",
    "twintri",
    "warpaint",
];

/// DDNet's `MAX_CLAN_LENGTH` is 12 bytes including the terminator.
pub const MAX_CLAN_BYTES: usize = 11;
/// DDNet's `MAX_SKIN_LENGTH` is 24 bytes including the terminator.
pub const MAX_SKIN_BYTES: usize = 23;

/// What goes into `Cl_StartInfo` besides the nick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub clan: String,
    pub skin: String,
}

/// The overrides of one run.
#[derive(Debug, Clone, Copy, Default)]
pub struct Overrides<'a> {
    pub clan: Option<&'a str>,
    pub skin: Option<&'a str>,
    pub skin_seed: Option<u64>,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One of [`DEFAULT_SKINS`]: the same one for the same `seed`; random (OS-seeded, differs per call) for `None`.
pub fn pick_skin(seed: Option<u64>) -> &'static str {
    let n = match seed {
        Some(s) => splitmix64(s),
        None => RandomState::new().build_hasher().finish(),
    };
    DEFAULT_SKINS[(n % DEFAULT_SKINS.len() as u64) as usize]
}

fn check_clan(clan: &str) -> Result<(), String> {
    if clan.len() > MAX_CLAN_BYTES {
        return Err(format!("the clan is longer than {MAX_CLAN_BYTES} bytes"));
    }
    if clan.chars().any(char::is_control) {
        return Err("the clan has control characters".to_string());
    }
    Ok(())
}

fn check_skin(skin: &str) -> Result<(), String> {
    if skin.is_empty() || skin.len() > MAX_SKIN_BYTES {
        return Err(format!("the skin name must be 1..={MAX_SKIN_BYTES} bytes"));
    }
    if !skin
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return Err("the skin name may hold only letters, digits, '_', '-' and '.'".to_string());
    }
    Ok(())
}

/// Works out the clan and the skin of this run (see the module docs for the precedence). An override that cannot go
/// on the wire is an error, not a silent fallback.
pub fn resolve(overrides: Overrides<'_>, settings: &Settings) -> Result<Identity, String> {
    let clan = overrides
        .clan
        .or(settings.clan.as_deref())
        .unwrap_or(DEFAULT_CLAN)
        .to_string();
    check_clan(&clan).map_err(|e| format!("clan: {e}"))?;
    let skin = match (overrides.skin, overrides.skin_seed, settings.skin.as_deref()) {
        (Some(s), _, _) => s.to_string(),
        (None, Some(seed), _) => pick_skin(Some(seed)).to_string(),
        (None, None, Some(s)) => s.to_string(),
        (None, None, None) => pick_skin(None).to_string(),
    };
    check_skin(&skin).map_err(|e| format!("skin: {e}"))?;
    Ok(Identity { clan, skin })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_clan_is_neuroset_and_the_skin_is_a_stock_one() {
        let id = resolve(Overrides::default(), &Settings::default()).unwrap();
        assert_eq!(id.clan, "Neuroset");
        assert!(DEFAULT_SKINS.contains(&id.skin.as_str()));
    }

    #[test]
    fn a_seed_makes_the_pick_deterministic_and_different_seeds_reach_different_skins() {
        let o = |seed| Overrides {
            skin_seed: Some(seed),
            ..Overrides::default()
        };
        let a = resolve(o(7), &Settings::default()).unwrap();
        let b = resolve(o(7), &Settings::default()).unwrap();
        assert_eq!(a, b);
        let seen: std::collections::BTreeSet<&str> = (0..500).map(pick_skin_seeded).collect();
        assert_eq!(
            seen.len(),
            DEFAULT_SKINS.len(),
            "every stock skin is reachable by some seed"
        );
    }

    fn pick_skin_seeded(seed: u64) -> &'static str {
        pick_skin(Some(seed))
    }

    #[test]
    fn without_a_seed_the_pick_is_random_per_start() {
        // 40 draws of 17 skins: the chance that all are equal is 17^-39. A constant implementation fails this.
        let seen: std::collections::BTreeSet<&str> = (0..40).map(|_| pick_skin(None)).collect();
        assert!(seen.len() > 1, "{seen:?}");
        assert!(seen.iter().all(|s| DEFAULT_SKINS.contains(s)));
    }

    #[test]
    fn precedence_is_flag_then_seed_then_settings_then_default() {
        let settings = Settings {
            clan: Some("FromFile".into()),
            skin: Some("pinky".into()),
            ..Settings::default()
        };
        let from_file = resolve(Overrides::default(), &settings).unwrap();
        assert_eq!(
            (from_file.clan.as_str(), from_file.skin.as_str()),
            ("FromFile", "pinky")
        );
        let seeded = resolve(
            Overrides {
                skin_seed: Some(3),
                ..Overrides::default()
            },
            &settings,
        )
        .unwrap();
        assert_eq!(
            seeded.skin,
            pick_skin(Some(3)),
            "a seed on the command line beats the file"
        );
        let flag = resolve(
            Overrides {
                clan: Some("Flag"),
                skin: Some("saddo"),
                skin_seed: Some(3),
            },
            &settings,
        )
        .unwrap();
        assert_eq!((flag.clan.as_str(), flag.skin.as_str()), ("Flag", "saddo"));
    }

    #[test]
    fn an_identity_that_cannot_go_on_the_wire_is_refused() {
        fn bad_clan(c: &str) -> Result<Identity, String> {
            resolve(
                Overrides {
                    clan: Some(c),
                    ..Overrides::default()
                },
                &Settings::default(),
            )
        }
        assert!(bad_clan("TwelveBytes!").is_err());
        assert!(bad_clan("ElevenBytes").is_ok());
        assert!(bad_clan("a\nb").is_err());
        assert!(bad_clan("").is_ok(), "an empty clan means no clan");
        fn bad_skin(s: &str) -> Result<Identity, String> {
            resolve(
                Overrides {
                    skin: Some(s),
                    ..Overrides::default()
                },
                &Settings::default(),
            )
        }
        assert!(bad_skin("").is_err());
        assert!(bad_skin("with space").is_err());
        assert!(bad_skin("../etc").is_err());
        assert!(bad_skin(&"s".repeat(MAX_SKIN_BYTES + 1)).is_err());
        assert!(bad_skin("twintri").is_ok());
    }

    #[test]
    fn the_pool_has_no_duplicates() {
        let set: std::collections::BTreeSet<_> = DEFAULT_SKINS.iter().collect();
        assert_eq!(set.len(), DEFAULT_SKINS.len());
    }
}
