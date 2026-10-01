//! The old bot's tuning constants, ported with their TS source lines (`src/bot/bot.ts` unless
//! another file is named; `docs/research/orig-bot.md` §7-§8). Distances are pixels, times are server
//! ticks (50 per second). Nothing here is a guess: where the port changes a value it says so.

/// `TARGET_MAX_PX` — a player farther than this is never a target (`bot.ts:3048`, §7.1).
pub const TARGET_MAX_PX: f32 = 1600.0;
/// `PATH_NEAR_PX` — from this distance on a player must be reachable to be worth chasing (§7.1 step 2).
pub const PATH_NEAR_PX: f32 = 420.0;
/// `BLOCKING_RANGE_PX` (§7.1: the `blockHoldScore` / `finishing` range).
pub const BLOCKING_RANGE_PX: f32 = 320.0;
/// `ENGAGED_PX` — target-hold bonus is full inside this distance (§7.1).
pub const ENGAGED_PX: f32 = 420.0;
/// `HOLD_FADE_PX` — and fades to zero over this many more pixels (§7.1).
pub const HOLD_FADE_PX: f32 = 400.0;
/// `TARGET_HOLD_SCORE` = `PLANNER_DEFAULTS.targetHold` (§7.1 table: 400).
pub const TARGET_HOLD_SCORE: f32 = 400.0;
/// `PLANNER_DEFAULTS.blockHoldScore` (§7.1 table: 0 — the term exists but is off by default).
pub const BLOCK_HOLD_SCORE: f32 = 0.0;
/// `PLANNER_DEFAULTS.settledFreezeTicks` (§7.2: 0, so every frozen non-"finishing" tee is skipped).
pub const SETTLED_FREEZE_TICKS: i32 = 0;
/// `TARGET_DIST_WEIGHT` — score penalty per pixel of distance (§7.1: 0.25).
pub const TARGET_DIST_WEIGHT: f32 = 0.25;
/// `OUT_OF_REACH_SCORE` (§7.1: -700).
pub const OUT_OF_REACH_SCORE: f32 = 700.0;
/// `FINISH_BLOCK_SCORE` (§7.1: +600).
pub const FINISH_BLOCK_SCORE: f32 = 600.0;
/// `FINISH_BLOCK_TICKS` — a frozen current target is "finishing" for this long (§7.2: 150).
pub const FINISH_BLOCK_TICKS: i32 = 150;
/// `AGGRESSOR_MEMORY_TICKS` = `AT_US_MEMORY_TICKS` (§7.1: 150).
pub const AGGRESSOR_MEMORY_TICKS: i32 = 150;
/// `AGGRESSOR_RANGE_PX` (§7.1: 500).
pub const AGGRESSOR_RANGE_PX: f32 = 500.0;
/// `AT_FRIEND_SCORE` (`bot.ts:190`: 450).
pub const AT_FRIEND_SCORE: f32 = 450.0;
/// `SWING_AT_US_PX` — a swing this close counts as "at us" (`bot.ts:172`: 128).
pub const SWING_AT_US_PX: f32 = 128.0;
/// `AFK_TICKS` — no input change for this long is AFK (§7.4: 500 = 10 s).
pub const AFK_TICKS: i32 = 500;
/// `INPUT_SETTLE_TICKS` — after a death a tee's input changes do not count as activity
/// (`bot.ts:260`: 50), so a respawn is not mistaken for a player waking up.
pub const INPUT_SETTLE_TICKS: i32 = 50;
/// `CROWD_FROZEN_TICKS` (`bot.ts:262`: 250) — parked-in-freeze threshold (kept for the 4.2 hooks).
pub const CROWD_FROZEN_TICKS: i32 = 250;
/// `BLOCK_CREDIT_TICKS` — a freeze counts for the last toucher if the touch was this recent
/// (`bot.ts:174`, §7.4: 50).
pub const BLOCK_CREDIT_TICKS: i32 = 50;
/// `TELEPORT_JUMP_PX` — a move of more than this in <= 4 ticks forgets the touch (`bot.ts:176`: 200).
pub const TELEPORT_JUMP_PX: f32 = 200.0;
/// `HAMMER_REACH_AHEAD_PX` / `HAMMER_REACH_PX` (`bot.ts:184-185`: 21 / 56).
pub const HAMMER_REACH_AHEAD_PX: f32 = 21.0;
pub const HAMMER_REACH_PX: f32 = 56.0;
/// `fire_hammer`'s hit radius around the swing point: `proximity / 2 + proximity` = 14 + 28 px
/// (`character.cpp:520-526`) — the geometry of the hammer veto (the 56 px above is the TS attribution
/// heuristic, not the hit test).
pub const HAMMER_HIT_RADIUS_PX: f32 = 42.0;
/// `REFREEZE_TICKS` — a freeze within this many ticks of thawing is not a new onset (`bot.ts:187`: 6).
pub const REFREEZE_TICKS: i32 = 6;
/// `REACH_ANSWER_TICKS` (§7.5: 25) and `REACH_MAX_NODES` (20 000).
pub const REACH_ANSWER_TICKS: i32 = 25;
pub const REACH_MAX_NODES: usize = 20_000;
/// The reachability cache size (§7.5: at most 64 entries — one per client id here, capped at 64 live).
pub const REACH_CACHE_MAX: usize = 64;
/// `SEAL_ANSWER_TICKS` (§7.2: 6) and `SEAL_NEAR_TILES` (2).
pub const SEAL_ANSWER_TICKS: i32 = 6;
pub const SEAL_NEAR_TILES: i32 = 2;
/// `STRONG/WEAK` hook reach used for the spare-bystander radius: `TUNING.hookLength` = 380 (§6.5).
pub const HOOK_LENGTH_PX: f32 = 380.0;
/// `ropeCatchAlong`'s half-width: `PHYSICAL_SIZE + 6` (`planner.ts:642`, ported as
/// `ddai_planner::fields::ROPE_CATCH_PX`).
pub const ROPE_CATCH_PX: f32 = 28.0 + 6.0;
/// `BYSTANDER_PX` (`bot.ts:339`: 160) — frozen non-target tees this close are planner bystanders.
pub const BYSTANDER_PX: f32 = 160.0;
/// `HELPER_RANGE_PX` (`bot.ts:565`: 140) — a friend/ignored tee this close "helps" a frozen bot.
pub const HELPER_RANGE_PX: f32 = 140.0;
/// `JOIN_RETRY_MS` (`bot.ts:552`: 3000) — how often a bot sitting in the spectators asks to join.
pub const JOIN_RETRY_TICKS: i32 = 150;

// --- unstick (`maybeUnstick`, `bot.ts:4574-4693`, §8.6) ------------------------------------------
/// `STUCK_FROZEN_TICKS` = 9*50.
pub const STUCK_FROZEN_TICKS: i32 = 450;
/// `STUCK_WEDGED_TICKS` = 4*50 (free and not moving, only counted while a target exists).
pub const STUCK_WEDGED_TICKS: i32 = 200;
/// `STUCK_RADIUS` — the anchor moves when the tee strays farther than this (48 px).
pub const STUCK_RADIUS_PX: f32 = 48.0;
/// `FROZEN_HARD_LIMIT_TICKS` = 8*50.
pub const FROZEN_HARD_LIMIT_TICKS: i32 = 400;
/// `FROZEN_IN_TILE_TICKS` = 4*50.
pub const FROZEN_IN_TILE_TICKS: i32 = 200;
/// `TRAPPED_TICKS` = 1.5*50.
pub const TRAPPED_TICKS: i32 = 75;
/// `HELPED_LIMIT_TICKS` (`bot.ts:563`: 30*50).
pub const HELPED_LIMIT_TICKS: i32 = 1500;
/// `KILL_COOLDOWN_TICKS` (`bot.ts:558`: 10*50) — the minimum gap between two `Cl_Kill`s.
pub const KILL_COOLDOWN_TICKS: i32 = 500;
/// `WB_LYING_TICKS` / `WB_KILL_COOLDOWN_TICKS` (`bot.ts:560-561`) — wayblock-only; kept for 4.2.
pub const WB_LYING_TICKS: i32 = 25;
pub const WB_KILL_COOLDOWN_TICKS: i32 = 100;
/// Half a tee (`PHYSICAL_SIZE / 2`): the corner offset of the "standing in freeze tiles" probe.
pub const HALF_TEE_PX: f32 = 14.0;

// --- wander / guard (`bot.ts:4882-4971`, §8.7) -----------------------------------------------------
/// The aim turns at most this many radians per decision while wandering (`bot.ts`: 0.12).
pub const WANDER_AIM_STEP: f32 = 0.12;
/// Aim vector length while wandering (300).
pub const WANDER_AIM_RADIUS: f32 = 300.0;
/// `guard`: the shield holds the wanted input this many ticks (`escapeExists(.., 2)`).
pub const GUARD_HOLD_TICKS: i32 = 2;

// --- brain wiring (new in the Rust bot; 3.5's review F2) -------------------------------------------
/// Tees farther than this from us (and not the target) are not given to the brain: hook reach 380 +
/// the far end of a hook throw + ~10 ticks of travel at 30 px/tick (a generous bound) — `docs` §4.1.
pub const THREAT_RADIUS_PX: f32 = 900.0;
/// At most this many non-target tees are handed to the brain (nearest first): a search over a crowd
/// of 8+ tees collapses (3.5 review F2), and the 6 nearest already cover every tee that can reach us.
pub const MAX_LOCAL_OTHERS: usize = 5;
/// Spared tees (friends, ignored, out of game, AFK) kept in the brain's world as physical bodies, at
/// most this many, **counted apart from [`MAX_LOCAL_OTHERS`]** (task 4.1b, review F8): the world has at
/// most `1 + 5 + 3` other tees, still well below the 8+ where the search collapses.
pub const MAX_SPARE_BODIES: usize = 3;
/// A spared tee is simulated when it is within this distance of us: contact needs the centres 28 px
/// apart (the tee's diameter), and two tees that move toward each other close at up to ~14 px/tick
/// (run speed 10, hook and air boosts above it; ~12 px/tick is a fast but ordinary approach) for the 4-5
/// ticks of an ordinary prediction horizon (`pred_tick` + one or two ticks): 28 + 5 * 14 = 98 px, doubled
/// for the rope-pull and velocity spikes of a hook throw. 200 px is about 7 tee diameters.
///
/// **The planner's own rollouts are longer than the prediction horizon** (27 ticks: `steps: 9` x 3), in
/// which we can cover 270-380 px, so a spared tee farther ahead on our path would be run into inside
/// the plan without being simulated (review 4.1b F3). Within [`SPARE_BODY_AHEAD_PX`] a spared tee is
/// therefore also kept when it stands in a [`SPARE_BODY_LANE_PX`]-wide lane ahead of us, along our
/// velocity or toward the target (the plan's goal). The trade-off: a tee off to the side beyond 200 px,
/// or ahead beyond 380 px, is still not a body (it costs search time, D-042, and the cap of
/// [`MAX_SPARE_BODIES`] bounds the total); the hook and hammer vetoes cover contact by rope and swing
/// whether or not the tee is a body.
pub const SPARE_BODY_RANGE_PX: f32 = 200.0;
/// How far ahead along our velocity / toward the target a spared tee is still kept as a body.
pub const SPARE_BODY_AHEAD_PX: f32 = 380.0;
/// Half-width of that lane: two tee diameters plus a little (the lane is a straight line, our path bends).
pub const SPARE_BODY_LANE_PX: f32 = 64.0;
/// Our speed (px/tick) above which the velocity defines a lane.
pub const SPARE_BODY_MIN_SPEED: f32 = 1.5;
/// The prediction horizon is capped at this many ticks past the snapshot (lag of ~120 ms).
pub const MAX_PREDICT_TICKS: i32 = 12;
/// A fresh reachability search per snapshot at most this often (keeps the bot's own overhead at the
/// `<= 0.5 ms p99` target: one 20 000-node flood fill is ~0.1-0.3 ms).
pub const REACH_CHECKS_PER_SNAPSHOT: u32 = 1;

/// The guard (shield) runs only when a freeze or death tile lies within this many tiles of the tee
/// (see [`crate::mapgrid::MapGrid::hazard_within`]); elsewhere it cannot change the verdict and its
/// 130-tick simulation is wasted time. The TS ran it always.
pub const GUARD_HAZARD_TILES: i32 = 12;
/// Fresh `sealedIn` searches per snapshot (each can cost up to 4 x 90 physics steps ~ 1.5 ms).
pub const SEAL_CHECKS_PER_SNAPSHOT: u32 = 1;
/// A "sealed" answer stays valid this long while the tee is still frozen on the same tile (a tee
/// that cannot get out of a pit does not start being able to 6 ticks later); "not sealed" answers
/// keep the TS 6-tick lifetime.
pub const SEALED_TRUE_TICKS: i32 = 30;

/// The quantile of recent decision times that picks the input slot a decision is aimed at (task
/// 4.1b): **p90**. The driver holds each decision until its intended tick, so a high quantile costs
/// latency (at most one tick, when the estimate crosses a slot) and never makes a decision land early.
/// Measured on this shared VM (hybrid, adaptive margin, 75 s runs, 1v1 | 1v3): p90 landed on the
/// predicted tick in 99.6% | 98.1% of the decisions with 94% | 33% in the first slot, p95 in 97.5% |
/// 97.5% (75% | 37%), p97 in 98.7% | 98.9% (18% | 15%): the higher the quantile the more decisions are
/// aimed at the second slot (exact, but a tick slower). The spread between runs on this host is larger
/// than the difference between p90 and p95; see `docs/formats.md` §21.6.
pub const DEFAULT_ESTIMATE_QUANTILE: f64 = 0.9;
