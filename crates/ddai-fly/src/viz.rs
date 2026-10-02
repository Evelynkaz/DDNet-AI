//! The fly's visualisation stream (task 7.4, FLY.md §10, `docs/formats.md` §27): a compact binary frame
//! per decision (group activity, the eye, the DN z-scores, the decoded logits and action, the
//! proposer-chosen flag) plus a static JSON description of its layout.
//!
//! **Nothing here runs unless somebody watches.** [`crate::brain::FlyBrain::decide`] does not know about
//! the stream: a frame is *pulled* by the driver ([`ddai_brain::Brain::viz_frame`]) only while a viewer
//! is subscribed, and it is built from state the decision left behind (rates, DN rates, the ray grid, the
//! decoded action). With no subscriber the decision path is untouched, so its output cannot depend on
//! the stream and it costs nothing; with one, a frame is a few microseconds into a buffer sized once.
//!
//! **Groups.** The heat strips show the mean rate of *groups* of types, in pipeline order: the visual
//! projection neurons (one row per type, the HS/VS/H2 cells together), the ascending neurons, the central
//! brain by neuropil prefix of the type name (AOTU, PVLP, LAL, PS, GNG, ... small ones merged into
//! "прочие"), and the descending neurons by family (DNa, DNg, DNge, DNp, DNpe, MDN). A group's rate is the
//! neuron-weighted mean of its types' mean rates ([`crate::state::DecisionOutput::per_type_mean_rate`]).
//!
//! **Quantisation.** Everything is one byte: rates `u8` over `[0, rate_max]`, DN z-scores `i8` over
//! `[-z_clip, z_clip]`, eye features `u8` over `[0, 1]`, logits `i8` in units of 1/8. The decoder
//! ([`decode_frame`]) returns the dequantised values; the error is at most half a step.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use ddai_flyg::{Flyg, NeuronRole};
use ddai_planner::hybrid::ProposalOutcome;

use crate::encoder::{CH_SELF_VELOCITY_FLOW, CH_SELF_VELOCITY_X, CH_SELF_VELOCITY_Y, Channel, RayGridFeatures};
use crate::model::FlyModel;

/// `DFLY` v1 magic and version.
pub const MAGIC: &[u8; 4] = b"DFLY";
pub const VERSION: u8 = 1;
/// Bytes before the body.
pub const HEADER_LEN: usize = 44;
/// The spatial eye channels in frame order (a subset of [`Channel`], the same order as the encoder's grids).
pub const EYE_CHANNELS: [Channel; 7] = [
    Channel::OpponentPosition,
    Channel::OpponentApproach,
    Channel::OpponentHook,
    Channel::OtherPlayers,
    Channel::Walls,
    Channel::FreezeDeathTiles,
    Channel::NoHookTiles,
];
/// The own-motion scalar channels in frame order.
pub const EYE_SCALARS: [&str; 3] = [CH_SELF_VELOCITY_X, CH_SELF_VELOCITY_Y, CH_SELF_VELOCITY_FLOW];
/// Logit units: a logit byte is `round(logit * LOGIT_SCALE)`.
pub const LOGIT_SCALE: f32 = 8.0;
/// Aim angle: an `i16` of `round(angle * AIM_SCALE)` radians.
pub const AIM_SCALE: f32 = 10_000.0;
/// Groups with fewer neurons than this are merged into "прочие" (central brain only).
const MIN_CENTRAL_GROUP: u32 = 10;
/// At most this many visual-projection rows; the smallest rest is merged into "VPN прочие".
const MAX_VPN_ROWS: usize = 16;

/// Flag bits of the header.
pub mod flag {
    /// `CHOSEN` is meaningful (the fly is a hybrid's proposer).
    pub const CHOSEN_VALID: u8 = 1;
    /// The search played a plan that came from the fly's proposals in this decision.
    pub const CHOSEN: u8 = 1 << 1;
    pub const JUMP: u8 = 1 << 2;
    pub const HOOK: u8 = 1 << 3;
    pub const FIRE: u8 = 1 << 4;
}

/// Which part of the pipeline a group belongs to (the colour family of the page).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    Vpn,
    An,
    Central,
    Dn,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Vpn => "vpn",
            Family::An => "an",
            Family::Central => "central",
            Family::Dn => "dn",
        }
    }
}

/// One heat-strip row.
#[derive(Debug, Clone, PartialEq)]
pub struct VizGroup {
    pub label: String,
    pub family: Family,
    pub neurons: u32,
}

/// What a frame says about the action (the fly's own decision, not necessarily what is played).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VizAction {
    /// `-1`, `0` or `1`.
    pub direction: i32,
    pub jump: bool,
    pub hook: bool,
    pub fire: bool,
    /// Ring-convention aim angle (radians).
    pub aim_angle: f32,
    /// `[left, stop, right]` probabilities, then jump, hook, fire probabilities.
    pub direction_probs: [f32; 3],
    pub jump_prob: f32,
    pub hook_prob: f32,
    pub fire_prob: f32,
}

/// The static layout of the frames of one fly: groups, DN slots, eye size. Built once.
#[derive(Debug, Clone)]
pub struct VizLayout {
    groups: Vec<VizGroup>,
    /// Flat `(type index, weight)` lists, `group_start[g]..group_start[g + 1]` per group.
    group_start: Vec<u32>,
    group_type: Vec<u32>,
    group_weight: Vec<f32>,
    /// Per output slot: `(type name, side letter)`.
    dn: Vec<(String, char)>,
    /// Per action head of the graph: `(action name, output slots)`.
    heads: Vec<(String, Vec<u32>)>,
    rays: usize,
    bins: usize,
    range_tiles: f32,
    rate_max: f32,
    z_clip: f32,
}

fn leading_letters(name: &str) -> &str {
    let end = name.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(name.len());
    &name[..end]
}

/// `(family, label)` of a type, from its role, superclass and name.
fn classify(name: &str, superclass: &str, role: NeuronRole) -> (Family, String) {
    match role {
        NeuronRole::InputVisual => {
            let label = if name.starts_with("HS") || name.starts_with("VS") || name == "H2" {
                "HS/VS".to_string()
            } else {
                name.to_string()
            };
            (Family::Vpn, label)
        }
        NeuronRole::InputAscending => (Family::An, "AN".to_string()),
        NeuronRole::Output => {
            let p = leading_letters(name);
            (Family::Dn, if p.is_empty() { name.to_string() } else { p.to_string() })
        }
        NeuronRole::Hidden => {
            let label = match superclass {
                "cb_intrinsic" => {
                    let p = leading_letters(name);
                    if p.is_empty() { name.to_string() } else { p.to_string() }
                }
                "descending_neuron" => "DN скрытые".to_string(),
                "ascending_neuron" => "AN скрытые".to_string(),
                "sensory_ascending" => "SApp".to_string(),
                "visual_projection" => "VPN скрытые".to_string(),
                "vnc_intrinsic" => "IN (VNC)".to_string(),
                _ => "прочие".to_string(),
            };
            (Family::Central, label)
        }
    }
}

/// Pipeline order of the central rows: the visual targets first, then the premotor and the descending side.
const CENTRAL_ORDER: [&str; 17] = [
    "VPN скрытые",
    "AOTU",
    "PVLP",
    "PLP",
    "AVLP",
    "LAL",
    "PS",
    "SAD",
    "WED",
    "VES",
    "GNG",
    "CB",
    "CL",
    "AN скрытые",
    "SApp",
    "IN (VNC)",
    "DN скрытые",
];

fn central_rank(label: &str) -> usize {
    CENTRAL_ORDER
        .iter()
        .position(|l| *l == label)
        .unwrap_or(CENTRAL_ORDER.len())
}

impl VizLayout {
    /// Builds the layout of `model` for an eye of `rays` x `bins` reaching `range_tiles`.
    pub fn build(model: &FlyModel, rays: usize, bins: usize, range_tiles: f32, z_clip: f32) -> VizLayout {
        let flyg: &Flyg = model.flyg();
        let num_types = flyg.types.len();
        let mut count = vec![0u32; num_types];
        let mut role: Vec<Option<NeuronRole>> = vec![None; num_types];
        for n in &flyg.neurons {
            let t = n.type_index as usize;
            count[t] += 1;
            role[t].get_or_insert(n.role);
        }
        // (family, label) -> type indices.
        let mut by: BTreeMap<(Family, String), Vec<u32>> = BTreeMap::new();
        for (t, ty) in flyg.types.iter().enumerate() {
            let Some(r) = role[t] else { continue };
            by.entry(classify(&ty.name, &ty.superclass, r))
                .or_default()
                .push(t as u32);
        }
        let total = |types: &[u32]| types.iter().map(|&t| count[t as usize]).sum::<u32>();
        // Small central groups and the tail of the visual rows are merged.
        let small: Vec<(Family, String)> = by
            .iter()
            .filter(|((f, _), v)| *f == Family::Central && total(v) < MIN_CENTRAL_GROUP)
            .map(|(k, _)| k.clone())
            .collect();
        for k in small {
            let v = by.remove(&k).unwrap_or_default();
            by.entry((Family::Central, "прочие".to_string())).or_default().extend(v);
        }
        let mut vpn: Vec<(Family, String)> = by.keys().filter(|(f, _)| *f == Family::Vpn).cloned().collect();
        if vpn.len() > MAX_VPN_ROWS {
            vpn.sort_by_key(|k| std::cmp::Reverse(total(&by[k])));
            for k in vpn.split_off(MAX_VPN_ROWS - 1) {
                let v = by.remove(&k).unwrap_or_default();
                by.entry((Family::Vpn, "VPN прочие".to_string())).or_default().extend(v);
            }
        }
        let mut rows: Vec<((Family, String), Vec<u32>)> = by.into_iter().collect();
        rows.sort_by(|a, b| {
            let ((fa, la), (fb, lb)) = (&a.0, &b.0);
            let merged = |l: &str| l.starts_with("прочие") || l.ends_with("прочие");
            fa.cmp(fb)
                .then_with(|| {
                    if *fa == Family::Central {
                        central_rank(la).cmp(&central_rank(lb))
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .then_with(|| merged(la).cmp(&merged(lb)))
                .then_with(|| la.cmp(lb))
        });
        let mut groups = Vec::with_capacity(rows.len());
        let mut group_start = vec![0u32];
        let mut group_type = Vec::new();
        let mut group_weight = Vec::new();
        for ((family, label), types) in rows {
            let n = total(&types);
            for &t in &types {
                group_type.push(t);
                group_weight.push(count[t as usize] as f32 / n.max(1) as f32);
            }
            group_start.push(group_type.len() as u32);
            groups.push(VizGroup {
                label,
                family,
                neurons: n,
            });
        }
        let side = |s: ddai_flyg::Side| match s {
            ddai_flyg::Side::L => 'L',
            ddai_flyg::Side::R => 'R',
            ddai_flyg::Side::M => 'M',
            ddai_flyg::Side::Unknown => '?',
        };
        let dn: Vec<(String, char)> = model
            .output_neuron_indices()
            .iter()
            .map(|&i| {
                let n = &flyg.neurons[i as usize];
                (flyg.types[n.type_index as usize].name.clone(), side(n.side))
            })
            .collect();
        let heads = flyg
            .output_groups
            .iter()
            .map(|g| {
                let slots = g
                    .members
                    .iter()
                    .filter_map(|m| model.output_slot_for_neuron(m.neuron_index))
                    .map(|s| s as u32)
                    .collect();
                (g.action.clone(), slots)
            })
            .collect();
        VizLayout {
            groups,
            group_start,
            group_type,
            group_weight,
            dn,
            heads,
            rays,
            bins,
            range_tiles,
            rate_max: model.config().r_max,
            z_clip,
        }
    }

    pub fn groups(&self) -> &[VizGroup] {
        &self.groups
    }

    pub fn num_dn(&self) -> usize {
        self.dn.len()
    }

    pub fn rays(&self) -> usize {
        self.rays
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    pub fn rate_max(&self) -> f32 {
        self.rate_max
    }

    pub fn z_clip(&self) -> f32 {
        self.z_clip
    }

    /// Bytes of one frame.
    pub fn frame_len(&self) -> usize {
        HEADER_LEN + self.groups.len() + self.dn.len() + EYE_CHANNELS.len() * self.rays * self.bins + EYE_SCALARS.len()
    }

    /// Mean rate of group `g` from the per-type mean rates.
    pub(crate) fn group_rate(&self, g: usize, per_type_mean_rate: &[f32]) -> f32 {
        let (a, b) = (self.group_start[g] as usize, self.group_start[g + 1] as usize);
        self.group_type[a..b]
            .iter()
            .zip(&self.group_weight[a..b])
            .map(|(&t, &w)| per_type_mean_rate[t as usize] * w)
            .sum()
    }

    /// The static description of the stream, one JSON object (`docs/formats.md` §27.2). `role` is `"fly"`
    /// (the fly plays) or `"proposer"` (the fly proposes to the hybrid); `bundle` is the loaded bundle's
    /// `(name, sha256)`, `None` for an untrained fly; `every` the brain's decimation.
    pub fn meta_json(&self, brain_name: &str, role: &str, bundle: Option<(&str, &str)>, every: u32) -> String {
        let mut s = String::with_capacity(4096);
        let _ = write!(
            s,
            "{{\"v\":{VERSION},\"role\":{},\"name\":{},\"bundle\":",
            json_str(role),
            json_str(brain_name)
        );
        match bundle {
            Some((name, sha)) => {
                let _ = write!(s, "{{\"name\":{},\"sha256\":{}}}", json_str(name), json_str(sha));
            }
            None => s.push_str("null"),
        }
        let _ = write!(
            s,
            ",\"every\":{every},\"rate_max\":{},\"z_clip\":{},\"logit_scale\":{LOGIT_SCALE},\"aim_scale\":{AIM_SCALE},\
\"rays\":{},\"bins\":{},\"range_tiles\":{},\"frame_bytes\":{},\"channels\":[",
            self.rate_max,
            self.z_clip,
            self.rays,
            self.bins,
            self.range_tiles,
            self.frame_len()
        );
        for (i, c) in EYE_CHANNELS.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json_str(c.name()));
        }
        s.push_str("],\"scalars\":[");
        for (i, c) in EYE_SCALARS.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json_str(c));
        }
        s.push_str("],\"groups\":[");
        for (i, g) in self.groups.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(
                s,
                "{{\"label\":{},\"family\":\"{}\",\"neurons\":{}}}",
                json_str(&g.label),
                g.family.name(),
                g.neurons
            );
        }
        s.push_str("],\"dn\":[");
        for (i, (name, side)) in self.dn.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(s, "{{\"label\":{},\"side\":\"{side}\"}}", json_str(name));
        }
        s.push_str("],\"heads\":[");
        for (i, (action, slots)) in self.heads.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(s, "{{\"action\":{},\"dn\":[", json_str(action));
            for (j, slot) in slots.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                let _ = write!(s, "{slot}");
            }
            s.push_str("]}");
        }
        s.push_str("]}");
        s
    }
}

/// A JSON string literal (quotes, backslash and control characters escaped).
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn q_unsigned(x: f32, full_scale: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    (x / full_scale * 255.0).round().clamp(0.0, 255.0) as u8
}

fn q_signed(x: f32, full_scale: f32) -> i8 {
    if x.is_nan() {
        return 0;
    }
    (x / full_scale * 127.0).round().clamp(-127.0, 127.0) as i8
}

/// `ln p` for a probability, bounded so a saturated head still fits a byte.
fn logit_byte(logit: f32) -> i8 {
    if logit.is_nan() {
        return 0;
    }
    (logit * LOGIT_SCALE).round().clamp(-128.0, 127.0) as i8
}

fn logit_of(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// Everything one frame is made of, borrowed from the brain's last decision.
pub struct FrameInputs<'a> {
    pub seq: u32,
    pub tick: u32,
    pub latency_us: u32,
    pub per_type_mean_rate: &'a [f32],
    /// Clipped DN z-scores, one per output slot.
    pub dn_z: &'a [f32],
    pub eye: &'a RayGridFeatures,
    pub action: VizAction,
    pub outcome: Option<ProposalOutcome>,
}

/// Writes one frame into `out`, which must be exactly [`VizLayout::frame_len`] bytes. Allocation-free.
pub fn encode_frame(layout: &VizLayout, inp: &FrameInputs<'_>, out: &mut [u8]) {
    assert_eq!(out.len(), layout.frame_len(), "frame buffer has the wrong size");
    assert_eq!(inp.dn_z.len(), layout.dn.len());
    let a = &inp.action;
    let mut flags = 0u8;
    if let Some(o) = inp.outcome {
        flags |= flag::CHOSEN_VALID;
        if o.chosen {
            flags |= flag::CHOSEN;
        }
    }
    if a.jump {
        flags |= flag::JUMP;
    }
    if a.hook {
        flags |= flag::HOOK;
    }
    if a.fire {
        flags |= flag::FIRE;
    }
    out[0..4].copy_from_slice(MAGIC);
    out[4] = VERSION;
    out[5] = flags;
    out[6] = a.direction.clamp(-1, 1) as i8 as u8;
    out[7] = 0;
    out[8..12].copy_from_slice(&inp.tick.to_le_bytes());
    out[12..16].copy_from_slice(&inp.seq.to_le_bytes());
    out[16..18].copy_from_slice(&(inp.latency_us.min(u32::from(u16::MAX)) as u16).to_le_bytes());
    let aim = if a.aim_angle.is_nan() {
        0
    } else {
        (a.aim_angle * AIM_SCALE)
            .round()
            .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
    };
    out[18..20].copy_from_slice(&aim.to_le_bytes());
    // Direction logits as log-probabilities (a softmax is shift-invariant), the heads as true logits.
    let dp = a.direction_probs;
    let logits = [
        dp[0].max(1e-6).ln(),
        dp[1].max(1e-6).ln(),
        dp[2].max(1e-6).ln(),
        logit_of(a.jump_prob),
        logit_of(a.hook_prob),
        logit_of(a.fire_prob),
    ];
    for (i, l) in logits.iter().enumerate() {
        out[20 + i] = logit_byte(*l) as u8;
    }
    let (chosen_total, decisions_total) = inp.outcome.map_or((0, 0), |o| (o.chosen_total, o.decisions_total));
    out[26..30].copy_from_slice(&chosen_total.to_le_bytes());
    out[30..34].copy_from_slice(&decisions_total.to_le_bytes());
    out[34..36].copy_from_slice(&(layout.groups.len() as u16).to_le_bytes());
    out[36..38].copy_from_slice(&(layout.dn.len() as u16).to_le_bytes());
    out[38..40].copy_from_slice(&(layout.rays as u16).to_le_bytes());
    out[40] = layout.bins as u8;
    out[41] = EYE_CHANNELS.len() as u8;
    out[42] = EYE_SCALARS.len() as u8;
    out[43] = 0;

    let mut at = HEADER_LEN;
    for g in 0..layout.groups.len() {
        out[at] = q_unsigned(layout.group_rate(g, inp.per_type_mean_rate), layout.rate_max);
        at += 1;
    }
    for &z in inp.dn_z {
        out[at] = q_signed(z, layout.z_clip) as u8;
        at += 1;
    }
    for ch in EYE_CHANNELS {
        for &v in inp.eye.spatial(ch) {
            out[at] = q_unsigned(v, 1.0);
            at += 1;
        }
    }
    for ch in [
        Channel::SelfVelocityX,
        Channel::SelfVelocityY,
        Channel::SelfVelocityFlow,
    ] {
        out[at] = q_signed(inp.eye.scalar(ch), 1.0) as u8;
        at += 1;
    }
    debug_assert_eq!(at, out.len());
}

/// A decoded frame (dequantised), for tests and tools.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    pub flags: u8,
    pub direction: i8,
    pub tick: u32,
    pub seq: u32,
    pub latency_us: u16,
    pub aim_angle: f32,
    /// `[left, stop, right]` log-probabilities, then the jump, hook and fire logits.
    pub logits: [f32; 6],
    pub chosen_total: u32,
    pub decisions_total: u32,
    pub groups: Vec<f32>,
    pub dn_z: Vec<f32>,
    /// `[channel][ray * bins + bin]`.
    pub eye: Vec<Vec<f32>>,
    pub scalars: Vec<f32>,
    pub rays: usize,
    pub bins: usize,
}

/// Parses a frame, checking magic, version and that the length matches the counts in its header.
/// `rate_max` and `z_clip` are the layout's (`meta` fields) scales.
pub fn decode_frame(bytes: &[u8], rate_max: f32, z_clip: f32) -> Result<DecodedFrame, String> {
    if bytes.len() < HEADER_LEN {
        return Err(format!(
            "frame is {} bytes, the header alone is {HEADER_LEN}",
            bytes.len()
        ));
    }
    if &bytes[0..4] != MAGIC {
        return Err("bad magic".to_string());
    }
    if bytes[4] != VERSION {
        return Err(format!("unsupported frame version {}", bytes[4]));
    }
    let u16_at = |i: usize| usize::from(u16::from_le_bytes([bytes[i], bytes[i + 1]]));
    let u32_at = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    let (n_groups, n_dn, rays) = (u16_at(34), u16_at(36), u16_at(38));
    let (bins, n_ch, n_sc) = (usize::from(bytes[40]), usize::from(bytes[41]), usize::from(bytes[42]));
    let want = HEADER_LEN + n_groups + n_dn + n_ch * rays * bins + n_sc;
    if bytes.len() != want {
        return Err(format!("frame is {} bytes, its counts say {want}", bytes.len()));
    }
    let mut at = HEADER_LEN;
    let groups = bytes[at..at + n_groups]
        .iter()
        .map(|&b| f32::from(b) / 255.0 * rate_max)
        .collect();
    at += n_groups;
    let dn_z = bytes[at..at + n_dn]
        .iter()
        .map(|&b| f32::from(b as i8) / 127.0 * z_clip)
        .collect();
    at += n_dn;
    let mut eye = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        eye.push(
            bytes[at..at + rays * bins]
                .iter()
                .map(|&b| f32::from(b) / 255.0)
                .collect(),
        );
        at += rays * bins;
    }
    let scalars = bytes[at..at + n_sc]
        .iter()
        .map(|&b| f32::from(b as i8) / 127.0)
        .collect();
    let mut logits = [0.0f32; 6];
    for (i, l) in logits.iter_mut().enumerate() {
        *l = f32::from(bytes[20 + i] as i8) / LOGIT_SCALE;
    }
    Ok(DecodedFrame {
        flags: bytes[5],
        direction: bytes[6] as i8,
        tick: u32_at(8),
        seq: u32_at(12),
        latency_us: u16::from_le_bytes([bytes[16], bytes[17]]),
        aim_angle: f32::from(i16::from_le_bytes([bytes[18], bytes[19]])) / AIM_SCALE,
        logits,
        chosen_total: u32_at(26),
        decisions_total: u32_at(30),
        groups,
        dn_z,
        eye,
        scalars,
        rays,
        bins,
    })
}

/// The decimating, buffer-owning side of a stream: counts decisions, builds a frame for every
/// `every`-th one into a buffer sized once, and keeps no other state. Owned by the brain.
#[derive(Debug, Clone)]
pub struct VizEmitter {
    every: u32,
    /// Decisions seen by [`VizEmitter::due`] since the last frame.
    since: u32,
    buf: Vec<u8>,
}

/// The default decimation: every second decision (12.5 Hz of the 25 Hz decisions).
pub const DEFAULT_EVERY: u32 = 2;

impl VizEmitter {
    pub fn new(frame_len: usize, every: u32) -> VizEmitter {
        VizEmitter {
            every: every.max(1),
            // The first decision after a subscription is shown at once.
            since: every.max(1) - 1,
            buf: vec![0; frame_len],
        }
    }

    pub fn every(&self) -> u32 {
        self.every
    }

    pub fn set_every(&mut self, every: u32) {
        self.every = every.max(1);
        self.since = self.every - 1;
    }

    /// Counts one decision; `true` when a frame is due.
    pub fn due(&mut self) -> bool {
        self.since += 1;
        if self.since >= self.every {
            self.since = 0;
            true
        } else {
            false
        }
    }

    /// The buffer to encode into.
    pub fn buf_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    pub fn frame(&self) -> &[u8] {
        &self.buf
    }
}

#[cfg(test)]
mod tests;
