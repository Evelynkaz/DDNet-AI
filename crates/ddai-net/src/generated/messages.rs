// GENERATED — do not edit by hand.
//
// Produced by `tools/ddnet-protocol-gen/generate.py` from DDNet's own protocol description
// (`datasrc/network.py` + `datasrc/datatypes.py`), commit c9d208138f85755521f16a0096b6fe036c5c8698 ("20.1").
// Regenerate with (from the repository root):
//
//   python3 tools/ddnet-protocol-gen/generate.py ~/aiddnet/build/ddnet-20.1/src
//
// Re-running against the same pinned commit's tree reproduces these files byte-for-byte (the
// script formats its own output with `rustfmt`). See `tools/ddnet-protocol-gen/README.md`.

//! Game messages (`NETMSGTYPE_*`) from `datasrc/network.py`: `Sv_*` (server->client) and
//! `Cl_*` (client->server).
//!
//! D-007 / task constraint: the bot must never send chat. `encode_cl_say` exists (for
//! completeness and its own round-trip test) but is `pub(crate)`, not `pub` — there is no
//! public send path for `Cl_Say` anywhere in this crate's API.

/// `CNetMsg_Sv_Motd` (`NETMSGTYPE_SV_MOTD`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvMotd {
    pub message: String,
}

/// Decodes a `SvMotd` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_motd(unpacker: &mut crate::packer::Unpacker) -> Option<SvMotd> {
    let message = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE);
    if unpacker.error() {
        return None;
    }
    Some(SvMotd { message })
}

/// Encodes a `SvMotd` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_motd(msg: &SvMotd, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.message, 0, true);
}

/// `CNetMsg_Sv_Broadcast` (`NETMSGTYPE_SV_BROADCAST`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvBroadcast {
    pub message: String,
}

/// Decodes a `SvBroadcast` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_broadcast(unpacker: &mut crate::packer::Unpacker) -> Option<SvBroadcast> {
    let message = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE);
    if unpacker.error() {
        return None;
    }
    Some(SvBroadcast { message })
}

/// Encodes a `SvBroadcast` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_broadcast(msg: &SvBroadcast, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.message, 0, true);
}

/// `CNetMsg_Sv_Chat` (`NETMSGTYPE_SV_CHAT`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvChat {
    pub team: i32,
    pub client_id: i32,
    pub message: String,
}

/// Decodes a `SvChat` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_chat(unpacker: &mut crate::packer::Unpacker) -> Option<SvChat> {
    let team = unpacker.get_int();
    if !(-2..=3).contains(&team) {
        return None;
    }
    let client_id = unpacker.get_int();
    if !(-1..=127).contains(&client_id) {
        return None;
    }
    let message = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE_CC);
    if unpacker.error() {
        return None;
    }
    Some(SvChat {
        team,
        client_id,
        message,
    })
}

/// Encodes a `SvChat` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_chat(msg: &SvChat, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.team);
    packer.add_int(msg.client_id);
    packer.add_string(&msg.message, 0, true);
}

/// `CNetMsg_Sv_KillMsg` (`NETMSGTYPE_SV_KILLMSG`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvKillMsg {
    pub killer: i32,
    pub victim: i32,
    pub weapon: i32,
    pub mode_special: i32,
}

/// Decodes a `SvKillMsg` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_kill_msg(unpacker: &mut crate::packer::Unpacker) -> Option<SvKillMsg> {
    let killer = unpacker.get_int();
    if !(0..=127).contains(&killer) {
        return None;
    }
    let victim = unpacker.get_int();
    if !(0..=127).contains(&victim) {
        return None;
    }
    let weapon = unpacker.get_int();
    if !(-3..=5).contains(&weapon) {
        return None;
    }
    let mode_special = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(SvKillMsg {
        killer,
        victim,
        weapon,
        mode_special,
    })
}

/// Encodes a `SvKillMsg` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_kill_msg(msg: &SvKillMsg, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.killer);
    packer.add_int(msg.victim);
    packer.add_int(msg.weapon);
    packer.add_int(msg.mode_special);
}

/// `CNetMsg_Sv_SoundGlobal` (`NETMSGTYPE_SV_SOUNDGLOBAL`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvSoundGlobal {
    pub sound_id: i32,
}

/// Decodes a `SvSoundGlobal` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_sound_global(unpacker: &mut crate::packer::Unpacker) -> Option<SvSoundGlobal> {
    let sound_id = unpacker.get_int();
    if !(0..=40).contains(&sound_id) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvSoundGlobal { sound_id })
}

/// Encodes a `SvSoundGlobal` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_sound_global(msg: &SvSoundGlobal, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.sound_id);
}

/// `CNetMsg_Sv_TuneParams` (`NETMSGTYPE_SV_TUNEPARAMS`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvTuneParams {}

/// Decodes a `SvTuneParams` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_tune_params(unpacker: &mut crate::packer::Unpacker) -> Option<SvTuneParams> {
    if unpacker.error() {
        return None;
    }
    Some(SvTuneParams {})
}

/// Encodes a `SvTuneParams` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_tune_params(_msg: &SvTuneParams, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Unused` (`NETMSGTYPE_UNUSED`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unused {}

/// Decodes a `Unused` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_unused(unpacker: &mut crate::packer::Unpacker) -> Option<Unused> {
    if unpacker.error() {
        return None;
    }
    Some(Unused {})
}

/// Encodes a `Unused` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_unused(_msg: &Unused, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_ReadyToEnter` (`NETMSGTYPE_SV_READYTOENTER`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvReadyToEnter {}

/// Decodes a `SvReadyToEnter` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_ready_to_enter(unpacker: &mut crate::packer::Unpacker) -> Option<SvReadyToEnter> {
    if unpacker.error() {
        return None;
    }
    Some(SvReadyToEnter {})
}

/// Encodes a `SvReadyToEnter` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_ready_to_enter(_msg: &SvReadyToEnter, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_WeaponPickup` (`NETMSGTYPE_SV_WEAPONPICKUP`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvWeaponPickup {
    pub weapon: i32,
}

/// Decodes a `SvWeaponPickup` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_weapon_pickup(unpacker: &mut crate::packer::Unpacker) -> Option<SvWeaponPickup> {
    let weapon = unpacker.get_int();
    if !(0..=5).contains(&weapon) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvWeaponPickup { weapon })
}

/// Encodes a `SvWeaponPickup` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_weapon_pickup(msg: &SvWeaponPickup, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.weapon);
}

/// `CNetMsg_Sv_Emoticon` (`NETMSGTYPE_SV_EMOTICON`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvEmoticon {
    pub client_id: i32,
    pub emoticon: i32,
}

/// Decodes a `SvEmoticon` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_emoticon(unpacker: &mut crate::packer::Unpacker) -> Option<SvEmoticon> {
    let client_id = unpacker.get_int();
    if !(0..=127).contains(&client_id) {
        return None;
    }
    let emoticon = unpacker.get_int();
    if !(0..=15).contains(&emoticon) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvEmoticon { client_id, emoticon })
}

/// Encodes a `SvEmoticon` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_emoticon(msg: &SvEmoticon, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.client_id);
    packer.add_int(msg.emoticon);
}

/// `CNetMsg_Sv_VoteClearOptions` (`NETMSGTYPE_SV_VOTECLEAROPTIONS`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteClearOptions {}

/// Decodes a `SvVoteClearOptions` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_clear_options(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteClearOptions> {
    if unpacker.error() {
        return None;
    }
    Some(SvVoteClearOptions {})
}

/// Encodes a `SvVoteClearOptions` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_clear_options(_msg: &SvVoteClearOptions, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_VoteOptionListAdd` (`NETMSGTYPE_SV_VOTEOPTIONLISTADD`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteOptionListAdd {
    pub num_options: i32,
    pub description0: String,
    pub description1: String,
    pub description2: String,
    pub description3: String,
    pub description4: String,
    pub description5: String,
    pub description6: String,
    pub description7: String,
    pub description8: String,
    pub description9: String,
    pub description10: String,
    pub description11: String,
    pub description12: String,
    pub description13: String,
    pub description14: String,
}

/// Decodes a `SvVoteOptionListAdd` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_option_list_add(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteOptionListAdd> {
    let num_options = unpacker.get_int();
    if !(1..=15).contains(&num_options) {
        return None;
    }
    let description0 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description1 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description2 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description3 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description4 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description5 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description6 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description7 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description8 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description9 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description10 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description11 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description12 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description13 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let description14 = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(SvVoteOptionListAdd {
        num_options,
        description0,
        description1,
        description2,
        description3,
        description4,
        description5,
        description6,
        description7,
        description8,
        description9,
        description10,
        description11,
        description12,
        description13,
        description14,
    })
}

/// Encodes a `SvVoteOptionListAdd` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_option_list_add(msg: &SvVoteOptionListAdd, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.num_options);
    packer.add_string(&msg.description0, 0, true);
    packer.add_string(&msg.description1, 0, true);
    packer.add_string(&msg.description2, 0, true);
    packer.add_string(&msg.description3, 0, true);
    packer.add_string(&msg.description4, 0, true);
    packer.add_string(&msg.description5, 0, true);
    packer.add_string(&msg.description6, 0, true);
    packer.add_string(&msg.description7, 0, true);
    packer.add_string(&msg.description8, 0, true);
    packer.add_string(&msg.description9, 0, true);
    packer.add_string(&msg.description10, 0, true);
    packer.add_string(&msg.description11, 0, true);
    packer.add_string(&msg.description12, 0, true);
    packer.add_string(&msg.description13, 0, true);
    packer.add_string(&msg.description14, 0, true);
}

/// `CNetMsg_Sv_VoteOptionAdd` (`NETMSGTYPE_SV_VOTEOPTIONADD`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteOptionAdd {
    pub description: String,
}

/// Decodes a `SvVoteOptionAdd` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_option_add(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteOptionAdd> {
    let description = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(SvVoteOptionAdd { description })
}

/// Encodes a `SvVoteOptionAdd` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_option_add(msg: &SvVoteOptionAdd, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.description, 0, true);
}

/// `CNetMsg_Sv_VoteOptionRemove` (`NETMSGTYPE_SV_VOTEOPTIONREMOVE`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteOptionRemove {
    pub description: String,
}

/// Decodes a `SvVoteOptionRemove` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_option_remove(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteOptionRemove> {
    let description = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(SvVoteOptionRemove { description })
}

/// Encodes a `SvVoteOptionRemove` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_option_remove(msg: &SvVoteOptionRemove, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.description, 0, true);
}

/// `CNetMsg_Sv_VoteSet` (`NETMSGTYPE_SV_VOTESET`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteSet {
    pub timeout: i32,
    pub description: String,
    pub reason: String,
}

/// Decodes a `SvVoteSet` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_set(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteSet> {
    let timeout = unpacker.get_int();
    if timeout < 0 {
        return None;
    }
    let description = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let reason = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(SvVoteSet {
        timeout,
        description,
        reason,
    })
}

/// Encodes a `SvVoteSet` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_set(msg: &SvVoteSet, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.timeout);
    packer.add_string(&msg.description, 0, true);
    packer.add_string(&msg.reason, 0, true);
}

/// `CNetMsg_Sv_VoteStatus` (`NETMSGTYPE_SV_VOTESTATUS`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteStatus {
    pub yes: i32,
    pub no: i32,
    pub pass: i32,
    pub total: i32,
}

/// Decodes a `SvVoteStatus` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_status(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteStatus> {
    let yes = unpacker.get_int();
    if !(0..=128).contains(&yes) {
        return None;
    }
    let no = unpacker.get_int();
    if !(0..=128).contains(&no) {
        return None;
    }
    let pass = unpacker.get_int();
    if !(0..=128).contains(&pass) {
        return None;
    }
    let total = unpacker.get_int();
    if !(0..=128).contains(&total) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvVoteStatus { yes, no, pass, total })
}

/// Encodes a `SvVoteStatus` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_status(msg: &SvVoteStatus, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.yes);
    packer.add_int(msg.no);
    packer.add_int(msg.pass);
    packer.add_int(msg.total);
}

/// `CNetMsg_Cl_Say` (`NETMSGTYPE_CL_SAY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClSay {
    pub team: i32,
    pub message: String,
}

/// Decodes a `ClSay` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_say(unpacker: &mut crate::packer::Unpacker) -> Option<ClSay> {
    let team = unpacker.get_int();
    if !(0..=1).contains(&team) {
        return None;
    }
    let message = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE_CC);
    if unpacker.error() {
        return None;
    }
    Some(ClSay { team, message })
}

/// Encodes a `ClSay` payload (message body only, no leading msg-id — see
/// `crate::message`).
#[allow(dead_code)] // D-007: exists for completeness/tests only — never called by
// non-test code; there is no public send path for chat.
pub(crate) fn encode_cl_say(msg: &ClSay, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.team);
    packer.add_string(&msg.message, 0, true);
}

/// `CNetMsg_Cl_SetTeam` (`NETMSGTYPE_CL_SETTEAM`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClSetTeam {
    pub team: i32,
}

/// Decodes a `ClSetTeam` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_set_team(unpacker: &mut crate::packer::Unpacker) -> Option<ClSetTeam> {
    let team = unpacker.get_int();
    if !(-1..=1).contains(&team) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(ClSetTeam { team })
}

/// Encodes a `ClSetTeam` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_set_team(msg: &ClSetTeam, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.team);
}

/// `CNetMsg_Cl_SetSpectatorMode` (`NETMSGTYPE_CL_SETSPECTATORMODE`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClSetSpectatorMode {
    pub spectator_id: i32,
}

/// Decodes a `ClSetSpectatorMode` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_set_spectator_mode(unpacker: &mut crate::packer::Unpacker) -> Option<ClSetSpectatorMode> {
    let spectator_id = unpacker.get_int();
    if !(-1..=127).contains(&spectator_id) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(ClSetSpectatorMode { spectator_id })
}

/// Encodes a `ClSetSpectatorMode` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_set_spectator_mode(msg: &ClSetSpectatorMode, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.spectator_id);
}

/// `CNetMsg_Cl_StartInfo` (`NETMSGTYPE_CL_STARTINFO`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClStartInfo {
    pub name: String,
    pub clan: String,
    pub country: i32,
    pub skin: String,
    pub use_custom_color: i32,
    pub color_body: i32,
    pub color_feet: i32,
}

/// Decodes a `ClStartInfo` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_start_info(unpacker: &mut crate::packer::Unpacker) -> Option<ClStartInfo> {
    let name = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let clan = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let country = unpacker.get_int();
    let skin = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let use_custom_color = unpacker.get_int();
    if !(0..=1).contains(&use_custom_color) {
        return None;
    }
    let color_body = unpacker.get_int();
    let color_feet = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(ClStartInfo {
        name,
        clan,
        country,
        skin,
        use_custom_color,
        color_body,
        color_feet,
    })
}

/// Encodes a `ClStartInfo` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_start_info(msg: &ClStartInfo, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.name, 0, true);
    packer.add_string(&msg.clan, 0, true);
    packer.add_int(msg.country);
    packer.add_string(&msg.skin, 0, true);
    packer.add_int(msg.use_custom_color);
    packer.add_int(msg.color_body);
    packer.add_int(msg.color_feet);
}

/// `CNetMsg_Cl_ChangeInfo` (`NETMSGTYPE_CL_CHANGEINFO`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClChangeInfo {
    pub name: String,
    pub clan: String,
    pub country: i32,
    pub skin: String,
    pub use_custom_color: i32,
    pub color_body: i32,
    pub color_feet: i32,
}

/// Decodes a `ClChangeInfo` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_change_info(unpacker: &mut crate::packer::Unpacker) -> Option<ClChangeInfo> {
    let name = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let clan = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let country = unpacker.get_int();
    let skin = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let use_custom_color = unpacker.get_int();
    if !(0..=1).contains(&use_custom_color) {
        return None;
    }
    let color_body = unpacker.get_int();
    let color_feet = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(ClChangeInfo {
        name,
        clan,
        country,
        skin,
        use_custom_color,
        color_body,
        color_feet,
    })
}

/// Encodes a `ClChangeInfo` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_change_info(msg: &ClChangeInfo, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.name, 0, true);
    packer.add_string(&msg.clan, 0, true);
    packer.add_int(msg.country);
    packer.add_string(&msg.skin, 0, true);
    packer.add_int(msg.use_custom_color);
    packer.add_int(msg.color_body);
    packer.add_int(msg.color_feet);
}

/// `CNetMsg_Cl_Kill` (`NETMSGTYPE_CL_KILL`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClKill {}

/// Decodes a `ClKill` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_kill(unpacker: &mut crate::packer::Unpacker) -> Option<ClKill> {
    if unpacker.error() {
        return None;
    }
    Some(ClKill {})
}

/// Encodes a `ClKill` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_kill(_msg: &ClKill, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Cl_Emoticon` (`NETMSGTYPE_CL_EMOTICON`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClEmoticon {
    pub emoticon: i32,
}

/// Decodes a `ClEmoticon` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_emoticon(unpacker: &mut crate::packer::Unpacker) -> Option<ClEmoticon> {
    let emoticon = unpacker.get_int();
    if !(0..=15).contains(&emoticon) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(ClEmoticon { emoticon })
}

/// Encodes a `ClEmoticon` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_emoticon(msg: &ClEmoticon, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.emoticon);
}

/// `CNetMsg_Cl_Vote` (`NETMSGTYPE_CL_VOTE`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClVote {
    pub vote: i32,
}

/// Decodes a `ClVote` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_vote(unpacker: &mut crate::packer::Unpacker) -> Option<ClVote> {
    let vote = unpacker.get_int();
    if !(-1..=1).contains(&vote) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(ClVote { vote })
}

/// Encodes a `ClVote` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_vote(msg: &ClVote, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.vote);
}

/// `CNetMsg_Cl_CallVote` (`NETMSGTYPE_CL_CALLVOTE`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClCallVote {
    pub type_: String,
    pub value: String,
    pub reason: String,
}

/// Decodes a `ClCallVote` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_call_vote(unpacker: &mut crate::packer::Unpacker) -> Option<ClCallVote> {
    let type_ = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let value = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let reason = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(ClCallVote { type_, value, reason })
}

/// Encodes a `ClCallVote` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_call_vote(msg: &ClCallVote, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.type_, 0, true);
    packer.add_string(&msg.value, 0, true);
    packer.add_string(&msg.reason, 0, true);
}

/// `CNetMsg_Cl_IsDDNetLegacy` (`NETMSGTYPE_CL_ISDDNETLEGACY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClIsDDNetLegacy {}

/// Decodes a `ClIsDDNetLegacy` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_is_dd_net_legacy(unpacker: &mut crate::packer::Unpacker) -> Option<ClIsDDNetLegacy> {
    if unpacker.error() {
        return None;
    }
    Some(ClIsDDNetLegacy {})
}

/// Encodes a `ClIsDDNetLegacy` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_is_dd_net_legacy(_msg: &ClIsDDNetLegacy, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_DDRaceTimeLegacy` (`NETMSGTYPE_SV_DDRACETIMELEGACY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvDDRaceTimeLegacy {
    pub time: i32,
    pub check: i32,
    pub finish: i32,
}

/// Decodes a `SvDDRaceTimeLegacy` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_dd_race_time_legacy(unpacker: &mut crate::packer::Unpacker) -> Option<SvDDRaceTimeLegacy> {
    let time = unpacker.get_int();
    let check = unpacker.get_int();
    let finish = unpacker.get_int();
    if !(0..=1).contains(&finish) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvDDRaceTimeLegacy { time, check, finish })
}

/// Encodes a `SvDDRaceTimeLegacy` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_dd_race_time_legacy(msg: &SvDDRaceTimeLegacy, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.time);
    packer.add_int(msg.check);
    packer.add_int(msg.finish);
}

/// `CNetMsg_Sv_RecordLegacy` (`NETMSGTYPE_SV_RECORDLEGACY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvRecordLegacy {
    pub server_time_best: i32,
    pub player_time_best: i32,
}

/// Decodes a `SvRecordLegacy` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_record_legacy(unpacker: &mut crate::packer::Unpacker) -> Option<SvRecordLegacy> {
    let server_time_best = unpacker.get_int();
    let player_time_best = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(SvRecordLegacy {
        server_time_best,
        player_time_best,
    })
}

/// Encodes a `SvRecordLegacy` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_record_legacy(msg: &SvRecordLegacy, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.server_time_best);
    packer.add_int(msg.player_time_best);
}

/// `CNetMsg_Unused2` (`NETMSGTYPE_UNUSED2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unused2 {}

/// Decodes a `Unused2` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_unused2(unpacker: &mut crate::packer::Unpacker) -> Option<Unused2> {
    if unpacker.error() {
        return None;
    }
    Some(Unused2 {})
}

/// Encodes a `Unused2` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_unused2(_msg: &Unused2, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_TeamsStateLegacy` (`NETMSGTYPE_SV_TEAMSSTATELEGACY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvTeamsStateLegacy {}

/// Decodes a `SvTeamsStateLegacy` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_teams_state_legacy(unpacker: &mut crate::packer::Unpacker) -> Option<SvTeamsStateLegacy> {
    if unpacker.error() {
        return None;
    }
    Some(SvTeamsStateLegacy {})
}

/// Encodes a `SvTeamsStateLegacy` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_teams_state_legacy(_msg: &SvTeamsStateLegacy, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Cl_ShowOthersLegacy` (`NETMSGTYPE_CL_SHOWOTHERSLEGACY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClShowOthersLegacy {
    pub show: i32,
}

/// Decodes a `ClShowOthersLegacy` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_show_others_legacy(unpacker: &mut crate::packer::Unpacker) -> Option<ClShowOthersLegacy> {
    let show = unpacker.get_int();
    if !(0..=1).contains(&show) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(ClShowOthersLegacy { show })
}

/// Encodes a `ClShowOthersLegacy` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_show_others_legacy(msg: &ClShowOthersLegacy, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.show);
}

/// `CNetMsg_Sv_MyOwnMessage` (`NETMSGTYPE_SV_MYOWNMESSAGE`).
/// UUID name: `my-own-message@heinrich5991.de`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvMyOwnMessage {
    pub test: i32,
}

/// Decodes a `SvMyOwnMessage` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_my_own_message(unpacker: &mut crate::packer::Unpacker) -> Option<SvMyOwnMessage> {
    let test = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(SvMyOwnMessage { test })
}

/// Encodes a `SvMyOwnMessage` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_my_own_message(msg: &SvMyOwnMessage, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.test);
}

/// `CNetMsg_Cl_ShowDistance` (`NETMSGTYPE_CL_SHOWDISTANCE`).
/// UUID name: `show-distance@netmsg.ddnet.tw`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClShowDistance {
    pub x: i32,
    pub y: i32,
}

/// Decodes a `ClShowDistance` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_show_distance(unpacker: &mut crate::packer::Unpacker) -> Option<ClShowDistance> {
    let x = unpacker.get_int();
    let y = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(ClShowDistance { x, y })
}

/// Encodes a `ClShowDistance` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_show_distance(msg: &ClShowDistance, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.x);
    packer.add_int(msg.y);
}

/// `CNetMsg_Cl_ShowOthers` (`NETMSGTYPE_CL_SHOWOTHERS`).
/// UUID name: `showothers@netmsg.ddnet.tw`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClShowOthers {
    pub show: i32,
}

/// Decodes a `ClShowOthers` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_show_others(unpacker: &mut crate::packer::Unpacker) -> Option<ClShowOthers> {
    let show = unpacker.get_int();
    if !(0..=2).contains(&show) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(ClShowOthers { show })
}

/// Encodes a `ClShowOthers` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_show_others(msg: &ClShowOthers, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.show);
}

/// `CNetMsg_Cl_CameraInfo` (`NETMSGTYPE_CL_CAMERAINFO`).
/// UUID name: `camera-info@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClCameraInfo {
    pub zoom: i32,
    pub deadzone: i32,
    pub follow_factor: i32,
}

/// Decodes a `ClCameraInfo` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_camera_info(unpacker: &mut crate::packer::Unpacker) -> Option<ClCameraInfo> {
    let zoom = unpacker.get_int();
    let deadzone = unpacker.get_int();
    let follow_factor = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(ClCameraInfo {
        zoom,
        deadzone,
        follow_factor,
    })
}

/// Encodes a `ClCameraInfo` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_camera_info(msg: &ClCameraInfo, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.zoom);
    packer.add_int(msg.deadzone);
    packer.add_int(msg.follow_factor);
}

/// `CNetMsg_Sv_TeamsState` (`NETMSGTYPE_SV_TEAMSSTATE`).
/// UUID name: `teamsstate@netmsg.ddnet.tw`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvTeamsState {}

/// Decodes a `SvTeamsState` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_teams_state(unpacker: &mut crate::packer::Unpacker) -> Option<SvTeamsState> {
    if unpacker.error() {
        return None;
    }
    Some(SvTeamsState {})
}

/// Encodes a `SvTeamsState` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_teams_state(_msg: &SvTeamsState, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_DDRaceTime` (`NETMSGTYPE_SV_DDRACETIME`).
/// UUID name: `ddrace-time@netmsg.ddnet.tw`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvDDRaceTime {
    pub time: i32,
    pub check: i32,
    pub finish: i32,
}

/// Decodes a `SvDDRaceTime` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_dd_race_time(unpacker: &mut crate::packer::Unpacker) -> Option<SvDDRaceTime> {
    let time = unpacker.get_int();
    let check = unpacker.get_int();
    let finish = unpacker.get_int();
    if !(0..=1).contains(&finish) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvDDRaceTime { time, check, finish })
}

/// Encodes a `SvDDRaceTime` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_dd_race_time(msg: &SvDDRaceTime, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.time);
    packer.add_int(msg.check);
    packer.add_int(msg.finish);
}

/// `CNetMsg_Sv_Record` (`NETMSGTYPE_SV_RECORD`).
/// UUID name: `record@netmsg.ddnet.tw`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvRecord {
    pub server_time_best: i32,
    pub player_time_best: i32,
}

/// Decodes a `SvRecord` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_record(unpacker: &mut crate::packer::Unpacker) -> Option<SvRecord> {
    let server_time_best = unpacker.get_int();
    let player_time_best = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(SvRecord {
        server_time_best,
        player_time_best,
    })
}

/// Encodes a `SvRecord` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_record(msg: &SvRecord, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.server_time_best);
    packer.add_int(msg.player_time_best);
}

/// `CNetMsg_Sv_KillMsgTeam` (`NETMSGTYPE_SV_KILLMSGTEAM`).
/// UUID name: `killmsgteam@netmsg.ddnet.tw`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvKillMsgTeam {
    pub team: i32,
    pub first: i32,
}

/// Decodes a `SvKillMsgTeam` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_kill_msg_team(unpacker: &mut crate::packer::Unpacker) -> Option<SvKillMsgTeam> {
    let team = unpacker.get_int();
    if !(0..=127).contains(&team) {
        return None;
    }
    let first = unpacker.get_int();
    if !(-1..=127).contains(&first) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvKillMsgTeam { team, first })
}

/// Encodes a `SvKillMsgTeam` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_kill_msg_team(msg: &SvKillMsgTeam, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.team);
    packer.add_int(msg.first);
}

/// `CNetMsg_Sv_YourVote` (`NETMSGTYPE_SV_YOURVOTE`).
/// UUID name: `yourvote@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvYourVote {
    pub voted: i32,
}

/// Decodes a `SvYourVote` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_your_vote(unpacker: &mut crate::packer::Unpacker) -> Option<SvYourVote> {
    let voted = unpacker.get_int();
    if !(-1..=1).contains(&voted) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvYourVote { voted })
}

/// Encodes a `SvYourVote` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_your_vote(msg: &SvYourVote, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.voted);
}

/// `CNetMsg_Sv_RaceFinish` (`NETMSGTYPE_SV_RACEFINISH`).
/// UUID name: `racefinish@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvRaceFinish {
    pub client_id: i32,
    pub time: i32,
    pub diff: i32,
    pub record_personal: i32,
    pub record_server: i32,
}

/// Decodes a `SvRaceFinish` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_race_finish(unpacker: &mut crate::packer::Unpacker) -> Option<SvRaceFinish> {
    let client_id = unpacker.get_int();
    if !(0..=127).contains(&client_id) {
        return None;
    }
    let time = unpacker.get_int();
    let diff = unpacker.get_int();
    let record_personal = unpacker.get_int();
    if !(0..=1).contains(&record_personal) {
        return None;
    }
    let record_server = unpacker.get_int_or_default(0);
    if !(0..=1).contains(&record_server) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvRaceFinish {
        client_id,
        time,
        diff,
        record_personal,
        record_server,
    })
}

/// Encodes a `SvRaceFinish` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_race_finish(msg: &SvRaceFinish, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.client_id);
    packer.add_int(msg.time);
    packer.add_int(msg.diff);
    packer.add_int(msg.record_personal);
    packer.add_int(msg.record_server);
}

/// `CNetMsg_Sv_CommandInfo` (`NETMSGTYPE_SV_COMMANDINFO`).
/// UUID name: `commandinfo@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvCommandInfo {
    pub name: String,
    pub args_format: String,
    pub help_text: String,
}

/// Decodes a `SvCommandInfo` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_command_info(unpacker: &mut crate::packer::Unpacker) -> Option<SvCommandInfo> {
    let name = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let args_format = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let help_text = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(SvCommandInfo {
        name,
        args_format,
        help_text,
    })
}

/// Encodes a `SvCommandInfo` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_command_info(msg: &SvCommandInfo, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.name, 0, true);
    packer.add_string(&msg.args_format, 0, true);
    packer.add_string(&msg.help_text, 0, true);
}

/// `CNetMsg_Sv_CommandInfoRemove` (`NETMSGTYPE_SV_COMMANDINFOREMOVE`).
/// UUID name: `commandinfo-remove@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvCommandInfoRemove {
    pub name: String,
}

/// Decodes a `SvCommandInfoRemove` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_command_info_remove(unpacker: &mut crate::packer::Unpacker) -> Option<SvCommandInfoRemove> {
    let name = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(SvCommandInfoRemove { name })
}

/// Encodes a `SvCommandInfoRemove` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_command_info_remove(msg: &SvCommandInfoRemove, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.name, 0, true);
}

/// `CNetMsg_Sv_VoteOptionGroupStart` (`NETMSGTYPE_SV_VOTEOPTIONGROUPSTART`).
/// UUID name: `sv-vote-option-group-start@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteOptionGroupStart {}

/// Decodes a `SvVoteOptionGroupStart` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_option_group_start(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteOptionGroupStart> {
    if unpacker.error() {
        return None;
    }
    Some(SvVoteOptionGroupStart {})
}

/// Encodes a `SvVoteOptionGroupStart` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_option_group_start(_msg: &SvVoteOptionGroupStart, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_VoteOptionGroupEnd` (`NETMSGTYPE_SV_VOTEOPTIONGROUPEND`).
/// UUID name: `sv-vote-option-group-end@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvVoteOptionGroupEnd {}

/// Decodes a `SvVoteOptionGroupEnd` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_vote_option_group_end(unpacker: &mut crate::packer::Unpacker) -> Option<SvVoteOptionGroupEnd> {
    if unpacker.error() {
        return None;
    }
    Some(SvVoteOptionGroupEnd {})
}

/// Encodes a `SvVoteOptionGroupEnd` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_vote_option_group_end(_msg: &SvVoteOptionGroupEnd, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_CommandInfoGroupStart` (`NETMSGTYPE_SV_COMMANDINFOGROUPSTART`).
/// UUID name: `sv-commandinfo-group-start@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvCommandInfoGroupStart {}

/// Decodes a `SvCommandInfoGroupStart` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_command_info_group_start(unpacker: &mut crate::packer::Unpacker) -> Option<SvCommandInfoGroupStart> {
    if unpacker.error() {
        return None;
    }
    Some(SvCommandInfoGroupStart {})
}

/// Encodes a `SvCommandInfoGroupStart` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_command_info_group_start(_msg: &SvCommandInfoGroupStart, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_CommandInfoGroupEnd` (`NETMSGTYPE_SV_COMMANDINFOGROUPEND`).
/// UUID name: `sv-commandinfo-group-end@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvCommandInfoGroupEnd {}

/// Decodes a `SvCommandInfoGroupEnd` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_command_info_group_end(unpacker: &mut crate::packer::Unpacker) -> Option<SvCommandInfoGroupEnd> {
    if unpacker.error() {
        return None;
    }
    Some(SvCommandInfoGroupEnd {})
}

/// Encodes a `SvCommandInfoGroupEnd` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_command_info_group_end(_msg: &SvCommandInfoGroupEnd, _packer: &mut crate::packer::Packer) {}

/// `CNetMsg_Sv_ChangeInfoCooldown` (`NETMSGTYPE_SV_CHANGEINFOCOOLDOWN`).
/// UUID name: `change-info-cooldown@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvChangeInfoCooldown {
    pub wait_until: i32,
}

/// Decodes a `SvChangeInfoCooldown` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_change_info_cooldown(unpacker: &mut crate::packer::Unpacker) -> Option<SvChangeInfoCooldown> {
    let wait_until = unpacker.get_int();
    if !(0..=1879048191).contains(&wait_until) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvChangeInfoCooldown { wait_until })
}

/// Encodes a `SvChangeInfoCooldown` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_change_info_cooldown(msg: &SvChangeInfoCooldown, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.wait_until);
}

/// `CNetMsg_Sv_MapSoundGlobal` (`NETMSGTYPE_SV_MAPSOUNDGLOBAL`).
/// UUID name: `map-sound-global@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvMapSoundGlobal {
    pub sound_id: i32,
}

/// Decodes a `SvMapSoundGlobal` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_map_sound_global(unpacker: &mut crate::packer::Unpacker) -> Option<SvMapSoundGlobal> {
    let sound_id = unpacker.get_int();
    if unpacker.error() {
        return None;
    }
    Some(SvMapSoundGlobal { sound_id })
}

/// Encodes a `SvMapSoundGlobal` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_map_sound_global(msg: &SvMapSoundGlobal, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.sound_id);
}

/// `CNetMsg_Sv_PreInput` (`NETMSGTYPE_SV_PREINPUT`).
/// UUID name: `preinput@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvPreInput {
    pub direction: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
    pub owner: i32,
    pub intended_tick: i32,
}

/// Decodes a `SvPreInput` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_pre_input(unpacker: &mut crate::packer::Unpacker) -> Option<SvPreInput> {
    let direction = unpacker.get_int();
    let target_x = unpacker.get_int();
    let target_y = unpacker.get_int();
    let jump = unpacker.get_int();
    let fire = unpacker.get_int();
    let hook = unpacker.get_int();
    let wanted_weapon = unpacker.get_int();
    let next_weapon = unpacker.get_int();
    let prev_weapon = unpacker.get_int();
    let owner = unpacker.get_int();
    if !(0..=127).contains(&owner) {
        return None;
    }
    let intended_tick = unpacker.get_int();
    if !(0..=1879048191).contains(&intended_tick) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(SvPreInput {
        direction,
        target_x,
        target_y,
        jump,
        fire,
        hook,
        wanted_weapon,
        next_weapon,
        prev_weapon,
        owner,
        intended_tick,
    })
}

/// Encodes a `SvPreInput` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_pre_input(msg: &SvPreInput, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.direction);
    packer.add_int(msg.target_x);
    packer.add_int(msg.target_y);
    packer.add_int(msg.jump);
    packer.add_int(msg.fire);
    packer.add_int(msg.hook);
    packer.add_int(msg.wanted_weapon);
    packer.add_int(msg.next_weapon);
    packer.add_int(msg.prev_weapon);
    packer.add_int(msg.owner);
    packer.add_int(msg.intended_tick);
}

/// `CNetMsg_Sv_SaveCode` (`NETMSGTYPE_SV_SAVECODE`).
/// UUID name: `save-code@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvSaveCode {
    pub state: i32,
    pub error: String,
    pub save_requester: String,
    pub server_name: String,
    pub generated_code: String,
    pub code: String,
    pub team_members: String,
}

/// Decodes a `SvSaveCode` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_save_code(unpacker: &mut crate::packer::Unpacker) -> Option<SvSaveCode> {
    let state = unpacker.get_int();
    if !(0..=4).contains(&state) {
        return None;
    }
    let error = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let save_requester = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let server_name = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let generated_code = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let code = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    let team_members = unpacker.get_string(crate::packer::SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: true,
    });
    if unpacker.error() {
        return None;
    }
    Some(SvSaveCode {
        state,
        error,
        save_requester,
        server_name,
        generated_code,
        code,
        team_members,
    })
}

/// Encodes a `SvSaveCode` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_save_code(msg: &SvSaveCode, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.state);
    packer.add_string(&msg.error, 0, true);
    packer.add_string(&msg.save_requester, 0, true);
    packer.add_string(&msg.server_name, 0, true);
    packer.add_string(&msg.generated_code, 0, true);
    packer.add_string(&msg.code, 0, true);
    packer.add_string(&msg.team_members, 0, true);
}

/// `CNetMsg_Sv_ServerAlert` (`NETMSGTYPE_SV_SERVERALERT`).
/// UUID name: `server-alert@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvServerAlert {
    pub message: String,
}

/// Decodes a `SvServerAlert` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_server_alert(unpacker: &mut crate::packer::Unpacker) -> Option<SvServerAlert> {
    let message = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE);
    if unpacker.error() {
        return None;
    }
    Some(SvServerAlert { message })
}

/// Encodes a `SvServerAlert` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_server_alert(msg: &SvServerAlert, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.message, 0, true);
}

/// `CNetMsg_Sv_ModeratorAlert` (`NETMSGTYPE_SV_MODERATORALERT`).
/// UUID name: `moderator-alert@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvModeratorAlert {
    pub message: String,
}

/// Decodes a `SvModeratorAlert` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_moderator_alert(unpacker: &mut crate::packer::Unpacker) -> Option<SvModeratorAlert> {
    let message = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE);
    if unpacker.error() {
        return None;
    }
    Some(SvModeratorAlert { message })
}

/// Encodes a `SvModeratorAlert` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_moderator_alert(msg: &SvModeratorAlert, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.message, 0, true);
}

/// `CNetMsg_Cl_EnableSpectatorCount` (`NETMSGTYPE_CL_ENABLESPECTATORCOUNT`).
/// UUID name: `enable-spectator-count@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClEnableSpectatorCount {
    pub enable: i32,
}

/// Decodes a `ClEnableSpectatorCount` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_cl_enable_spectator_count(unpacker: &mut crate::packer::Unpacker) -> Option<ClEnableSpectatorCount> {
    let enable = unpacker.get_int();
    if !(0..=1).contains(&enable) {
        return None;
    }
    if unpacker.error() {
        return None;
    }
    Some(ClEnableSpectatorCount { enable })
}

/// Encodes a `ClEnableSpectatorCount` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_cl_enable_spectator_count(msg: &ClEnableSpectatorCount, packer: &mut crate::packer::Packer) {
    packer.add_int(msg.enable);
}

/// `CNetMsg_Sv_MapInfo` (`NETMSGTYPE_SV_MAPINFO`).
/// UUID name: `map-info@netmsg.ddnet.org`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvMapInfo {
    pub description: String,
}

/// Decodes a `SvMapInfo` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s
/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike
/// object decoding, which clamps — this matches DDNet exactly.
pub fn decode_sv_map_info(unpacker: &mut crate::packer::Unpacker) -> Option<SvMapInfo> {
    let description = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE);
    if unpacker.error() {
        return None;
    }
    Some(SvMapInfo { description })
}

/// Encodes a `SvMapInfo` payload (message body only, no leading msg-id — see
/// `crate::message`).
pub fn encode_sv_map_info(msg: &SvMapInfo, packer: &mut crate::packer::Packer) {
    packer.add_string(&msg.description, 0, true);
}

/// Non-UUID message ids, in `datasrc/network.py` declaration order starting at 1
/// (`datasrc/compile.py`'s `create_enum_table(["NETMSGTYPE_EX", ...])`) — part of
/// the wire format, must track DDNet exactly.
pub mod id {
    pub const NETMSGTYPE_SV_MOTD: i32 = 1;
    pub const NETMSGTYPE_SV_BROADCAST: i32 = 2;
    pub const NETMSGTYPE_SV_CHAT: i32 = 3;
    pub const NETMSGTYPE_SV_KILLMSG: i32 = 4;
    pub const NETMSGTYPE_SV_SOUNDGLOBAL: i32 = 5;
    pub const NETMSGTYPE_SV_TUNEPARAMS: i32 = 6;
    pub const NETMSGTYPE_UNUSED: i32 = 7;
    pub const NETMSGTYPE_SV_READYTOENTER: i32 = 8;
    pub const NETMSGTYPE_SV_WEAPONPICKUP: i32 = 9;
    pub const NETMSGTYPE_SV_EMOTICON: i32 = 10;
    pub const NETMSGTYPE_SV_VOTECLEAROPTIONS: i32 = 11;
    pub const NETMSGTYPE_SV_VOTEOPTIONLISTADD: i32 = 12;
    pub const NETMSGTYPE_SV_VOTEOPTIONADD: i32 = 13;
    pub const NETMSGTYPE_SV_VOTEOPTIONREMOVE: i32 = 14;
    pub const NETMSGTYPE_SV_VOTESET: i32 = 15;
    pub const NETMSGTYPE_SV_VOTESTATUS: i32 = 16;
    pub const NETMSGTYPE_CL_SAY: i32 = 17;
    pub const NETMSGTYPE_CL_SETTEAM: i32 = 18;
    pub const NETMSGTYPE_CL_SETSPECTATORMODE: i32 = 19;
    pub const NETMSGTYPE_CL_STARTINFO: i32 = 20;
    pub const NETMSGTYPE_CL_CHANGEINFO: i32 = 21;
    pub const NETMSGTYPE_CL_KILL: i32 = 22;
    pub const NETMSGTYPE_CL_EMOTICON: i32 = 23;
    pub const NETMSGTYPE_CL_VOTE: i32 = 24;
    pub const NETMSGTYPE_CL_CALLVOTE: i32 = 25;
    pub const NETMSGTYPE_CL_ISDDNETLEGACY: i32 = 26;
    pub const NETMSGTYPE_SV_DDRACETIMELEGACY: i32 = 27;
    pub const NETMSGTYPE_SV_RECORDLEGACY: i32 = 28;
    pub const NETMSGTYPE_UNUSED2: i32 = 29;
    pub const NETMSGTYPE_SV_TEAMSSTATELEGACY: i32 = 30;
    pub const NETMSGTYPE_CL_SHOWOTHERSLEGACY: i32 = 31;
}

/// UUID names for every `ex` game message, in `datasrc/network.py` declaration
/// order (see `crate::uuid::UuidRegistry::from_names`).
pub const EX_NAMES: &[&str] = &[
    "my-own-message@heinrich5991.de",
    "show-distance@netmsg.ddnet.tw",
    "showothers@netmsg.ddnet.tw",
    "camera-info@netmsg.ddnet.org",
    "teamsstate@netmsg.ddnet.tw",
    "ddrace-time@netmsg.ddnet.tw",
    "record@netmsg.ddnet.tw",
    "killmsgteam@netmsg.ddnet.tw",
    "yourvote@netmsg.ddnet.org",
    "racefinish@netmsg.ddnet.org",
    "commandinfo@netmsg.ddnet.org",
    "commandinfo-remove@netmsg.ddnet.org",
    "sv-vote-option-group-start@netmsg.ddnet.org",
    "sv-vote-option-group-end@netmsg.ddnet.org",
    "sv-commandinfo-group-start@netmsg.ddnet.org",
    "sv-commandinfo-group-end@netmsg.ddnet.org",
    "change-info-cooldown@netmsg.ddnet.org",
    "map-sound-global@netmsg.ddnet.org",
    "preinput@netmsg.ddnet.org",
    "save-code@netmsg.ddnet.org",
    "server-alert@netmsg.ddnet.org",
    "moderator-alert@netmsg.ddnet.org",
    "enable-spectator-count@netmsg.ddnet.org",
    "map-info@netmsg.ddnet.org",
];

/// Every non-UUID game message, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // mechanical: field counts vary a lot message to
// message (e.g. `Sv_VoteOptionListAdd`'s 15 strings); this is a short-lived decode
// result, not something kept around in bulk, so boxing isn't worth the ergonomics cost.
pub enum GameMsg {
    SvMotd(SvMotd),
    SvBroadcast(SvBroadcast),
    SvChat(SvChat),
    SvKillMsg(SvKillMsg),
    SvSoundGlobal(SvSoundGlobal),
    SvTuneParams(SvTuneParams),
    Unused(Unused),
    SvReadyToEnter(SvReadyToEnter),
    SvWeaponPickup(SvWeaponPickup),
    SvEmoticon(SvEmoticon),
    SvVoteClearOptions(SvVoteClearOptions),
    SvVoteOptionListAdd(SvVoteOptionListAdd),
    SvVoteOptionAdd(SvVoteOptionAdd),
    SvVoteOptionRemove(SvVoteOptionRemove),
    SvVoteSet(SvVoteSet),
    SvVoteStatus(SvVoteStatus),
    ClSay(ClSay),
    ClSetTeam(ClSetTeam),
    ClSetSpectatorMode(ClSetSpectatorMode),
    ClStartInfo(ClStartInfo),
    ClChangeInfo(ClChangeInfo),
    ClKill(ClKill),
    ClEmoticon(ClEmoticon),
    ClVote(ClVote),
    ClCallVote(ClCallVote),
    ClIsDDNetLegacy(ClIsDDNetLegacy),
    SvDDRaceTimeLegacy(SvDDRaceTimeLegacy),
    SvRecordLegacy(SvRecordLegacy),
    Unused2(Unused2),
    SvTeamsStateLegacy(SvTeamsStateLegacy),
    ClShowOthersLegacy(ClShowOthersLegacy),
}

/// Decodes a non-UUID game message payload by its numbered id (`id::*`).
pub fn decode_game_msg(msg_id: i32, unpacker: &mut crate::packer::Unpacker) -> Option<GameMsg> {
    Some(match msg_id {
        id::NETMSGTYPE_SV_MOTD => GameMsg::SvMotd(decode_sv_motd(unpacker)?),
        id::NETMSGTYPE_SV_BROADCAST => GameMsg::SvBroadcast(decode_sv_broadcast(unpacker)?),
        id::NETMSGTYPE_SV_CHAT => GameMsg::SvChat(decode_sv_chat(unpacker)?),
        id::NETMSGTYPE_SV_KILLMSG => GameMsg::SvKillMsg(decode_sv_kill_msg(unpacker)?),
        id::NETMSGTYPE_SV_SOUNDGLOBAL => GameMsg::SvSoundGlobal(decode_sv_sound_global(unpacker)?),
        id::NETMSGTYPE_SV_TUNEPARAMS => GameMsg::SvTuneParams(decode_sv_tune_params(unpacker)?),
        id::NETMSGTYPE_UNUSED => GameMsg::Unused(decode_unused(unpacker)?),
        id::NETMSGTYPE_SV_READYTOENTER => GameMsg::SvReadyToEnter(decode_sv_ready_to_enter(unpacker)?),
        id::NETMSGTYPE_SV_WEAPONPICKUP => GameMsg::SvWeaponPickup(decode_sv_weapon_pickup(unpacker)?),
        id::NETMSGTYPE_SV_EMOTICON => GameMsg::SvEmoticon(decode_sv_emoticon(unpacker)?),
        id::NETMSGTYPE_SV_VOTECLEAROPTIONS => GameMsg::SvVoteClearOptions(decode_sv_vote_clear_options(unpacker)?),
        id::NETMSGTYPE_SV_VOTEOPTIONLISTADD => GameMsg::SvVoteOptionListAdd(decode_sv_vote_option_list_add(unpacker)?),
        id::NETMSGTYPE_SV_VOTEOPTIONADD => GameMsg::SvVoteOptionAdd(decode_sv_vote_option_add(unpacker)?),
        id::NETMSGTYPE_SV_VOTEOPTIONREMOVE => GameMsg::SvVoteOptionRemove(decode_sv_vote_option_remove(unpacker)?),
        id::NETMSGTYPE_SV_VOTESET => GameMsg::SvVoteSet(decode_sv_vote_set(unpacker)?),
        id::NETMSGTYPE_SV_VOTESTATUS => GameMsg::SvVoteStatus(decode_sv_vote_status(unpacker)?),
        id::NETMSGTYPE_CL_SAY => GameMsg::ClSay(decode_cl_say(unpacker)?),
        id::NETMSGTYPE_CL_SETTEAM => GameMsg::ClSetTeam(decode_cl_set_team(unpacker)?),
        id::NETMSGTYPE_CL_SETSPECTATORMODE => GameMsg::ClSetSpectatorMode(decode_cl_set_spectator_mode(unpacker)?),
        id::NETMSGTYPE_CL_STARTINFO => GameMsg::ClStartInfo(decode_cl_start_info(unpacker)?),
        id::NETMSGTYPE_CL_CHANGEINFO => GameMsg::ClChangeInfo(decode_cl_change_info(unpacker)?),
        id::NETMSGTYPE_CL_KILL => GameMsg::ClKill(decode_cl_kill(unpacker)?),
        id::NETMSGTYPE_CL_EMOTICON => GameMsg::ClEmoticon(decode_cl_emoticon(unpacker)?),
        id::NETMSGTYPE_CL_VOTE => GameMsg::ClVote(decode_cl_vote(unpacker)?),
        id::NETMSGTYPE_CL_CALLVOTE => GameMsg::ClCallVote(decode_cl_call_vote(unpacker)?),
        id::NETMSGTYPE_CL_ISDDNETLEGACY => GameMsg::ClIsDDNetLegacy(decode_cl_is_dd_net_legacy(unpacker)?),
        id::NETMSGTYPE_SV_DDRACETIMELEGACY => GameMsg::SvDDRaceTimeLegacy(decode_sv_dd_race_time_legacy(unpacker)?),
        id::NETMSGTYPE_SV_RECORDLEGACY => GameMsg::SvRecordLegacy(decode_sv_record_legacy(unpacker)?),
        id::NETMSGTYPE_UNUSED2 => GameMsg::Unused2(decode_unused2(unpacker)?),
        id::NETMSGTYPE_SV_TEAMSSTATELEGACY => GameMsg::SvTeamsStateLegacy(decode_sv_teams_state_legacy(unpacker)?),
        id::NETMSGTYPE_CL_SHOWOTHERSLEGACY => GameMsg::ClShowOthersLegacy(decode_cl_show_others_legacy(unpacker)?),
        _ => return None,
    })
}

/// Every UUID (`ex`) game message, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExGameMsg {
    SvMyOwnMessage(SvMyOwnMessage),
    ClShowDistance(ClShowDistance),
    ClShowOthers(ClShowOthers),
    ClCameraInfo(ClCameraInfo),
    SvTeamsState(SvTeamsState),
    SvDDRaceTime(SvDDRaceTime),
    SvRecord(SvRecord),
    SvKillMsgTeam(SvKillMsgTeam),
    SvYourVote(SvYourVote),
    SvRaceFinish(SvRaceFinish),
    SvCommandInfo(SvCommandInfo),
    SvCommandInfoRemove(SvCommandInfoRemove),
    SvVoteOptionGroupStart(SvVoteOptionGroupStart),
    SvVoteOptionGroupEnd(SvVoteOptionGroupEnd),
    SvCommandInfoGroupStart(SvCommandInfoGroupStart),
    SvCommandInfoGroupEnd(SvCommandInfoGroupEnd),
    SvChangeInfoCooldown(SvChangeInfoCooldown),
    SvMapSoundGlobal(SvMapSoundGlobal),
    SvPreInput(SvPreInput),
    SvSaveCode(SvSaveCode),
    SvServerAlert(SvServerAlert),
    SvModeratorAlert(SvModeratorAlert),
    ClEnableSpectatorCount(ClEnableSpectatorCount),
    SvMapInfo(SvMapInfo),
}

/// Decodes a UUID (`ex`) game message payload by its UUID name.
pub fn decode_ex_game_msg(name: &str, unpacker: &mut crate::packer::Unpacker) -> Option<ExGameMsg> {
    Some(match name {
        "my-own-message@heinrich5991.de" => ExGameMsg::SvMyOwnMessage(decode_sv_my_own_message(unpacker)?),
        "show-distance@netmsg.ddnet.tw" => ExGameMsg::ClShowDistance(decode_cl_show_distance(unpacker)?),
        "showothers@netmsg.ddnet.tw" => ExGameMsg::ClShowOthers(decode_cl_show_others(unpacker)?),
        "camera-info@netmsg.ddnet.org" => ExGameMsg::ClCameraInfo(decode_cl_camera_info(unpacker)?),
        "teamsstate@netmsg.ddnet.tw" => ExGameMsg::SvTeamsState(decode_sv_teams_state(unpacker)?),
        "ddrace-time@netmsg.ddnet.tw" => ExGameMsg::SvDDRaceTime(decode_sv_dd_race_time(unpacker)?),
        "record@netmsg.ddnet.tw" => ExGameMsg::SvRecord(decode_sv_record(unpacker)?),
        "killmsgteam@netmsg.ddnet.tw" => ExGameMsg::SvKillMsgTeam(decode_sv_kill_msg_team(unpacker)?),
        "yourvote@netmsg.ddnet.org" => ExGameMsg::SvYourVote(decode_sv_your_vote(unpacker)?),
        "racefinish@netmsg.ddnet.org" => ExGameMsg::SvRaceFinish(decode_sv_race_finish(unpacker)?),
        "commandinfo@netmsg.ddnet.org" => ExGameMsg::SvCommandInfo(decode_sv_command_info(unpacker)?),
        "commandinfo-remove@netmsg.ddnet.org" => {
            ExGameMsg::SvCommandInfoRemove(decode_sv_command_info_remove(unpacker)?)
        }
        "sv-vote-option-group-start@netmsg.ddnet.org" => {
            ExGameMsg::SvVoteOptionGroupStart(decode_sv_vote_option_group_start(unpacker)?)
        }
        "sv-vote-option-group-end@netmsg.ddnet.org" => {
            ExGameMsg::SvVoteOptionGroupEnd(decode_sv_vote_option_group_end(unpacker)?)
        }
        "sv-commandinfo-group-start@netmsg.ddnet.org" => {
            ExGameMsg::SvCommandInfoGroupStart(decode_sv_command_info_group_start(unpacker)?)
        }
        "sv-commandinfo-group-end@netmsg.ddnet.org" => {
            ExGameMsg::SvCommandInfoGroupEnd(decode_sv_command_info_group_end(unpacker)?)
        }
        "change-info-cooldown@netmsg.ddnet.org" => {
            ExGameMsg::SvChangeInfoCooldown(decode_sv_change_info_cooldown(unpacker)?)
        }
        "map-sound-global@netmsg.ddnet.org" => ExGameMsg::SvMapSoundGlobal(decode_sv_map_sound_global(unpacker)?),
        "preinput@netmsg.ddnet.org" => ExGameMsg::SvPreInput(decode_sv_pre_input(unpacker)?),
        "save-code@netmsg.ddnet.org" => ExGameMsg::SvSaveCode(decode_sv_save_code(unpacker)?),
        "server-alert@netmsg.ddnet.org" => ExGameMsg::SvServerAlert(decode_sv_server_alert(unpacker)?),
        "moderator-alert@netmsg.ddnet.org" => ExGameMsg::SvModeratorAlert(decode_sv_moderator_alert(unpacker)?),
        "enable-spectator-count@netmsg.ddnet.org" => {
            ExGameMsg::ClEnableSpectatorCount(decode_cl_enable_spectator_count(unpacker)?)
        }
        "map-info@netmsg.ddnet.org" => ExGameMsg::SvMapInfo(decode_sv_map_info(unpacker)?),
        _ => return None,
    })
}

#[cfg(test)]
mod generated_roundtrip_tests {
    use super::*;

    #[test]
    fn roundtrip_sv_motd() {
        let msg = SvMotd {
            message: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_motd(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_motd(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_broadcast() {
        let msg = SvBroadcast {
            message: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_broadcast(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_broadcast(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_chat() {
        let msg = SvChat {
            team: -2,
            client_id: -1,
            message: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_chat(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_chat(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_kill_msg() {
        let msg = SvKillMsg {
            killer: 0,
            victim: 0,
            weapon: -3,
            mode_special: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_kill_msg(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_kill_msg(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_sound_global() {
        let msg = SvSoundGlobal { sound_id: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_sound_global(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_sound_global(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_tune_params() {
        let msg = SvTuneParams {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_tune_params(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_tune_params(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_unused() {
        let msg = Unused {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_unused(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_unused(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_ready_to_enter() {
        let msg = SvReadyToEnter {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_ready_to_enter(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_ready_to_enter(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_weapon_pickup() {
        let msg = SvWeaponPickup { weapon: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_weapon_pickup(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_weapon_pickup(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_emoticon() {
        let msg = SvEmoticon {
            client_id: 0,
            emoticon: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_emoticon(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_emoticon(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_clear_options() {
        let msg = SvVoteClearOptions {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_clear_options(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_clear_options(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_option_list_add() {
        let msg = SvVoteOptionListAdd {
            num_options: 1,
            description0: "abc".to_string(),
            description1: "abc".to_string(),
            description2: "abc".to_string(),
            description3: "abc".to_string(),
            description4: "abc".to_string(),
            description5: "abc".to_string(),
            description6: "abc".to_string(),
            description7: "abc".to_string(),
            description8: "abc".to_string(),
            description9: "abc".to_string(),
            description10: "abc".to_string(),
            description11: "abc".to_string(),
            description12: "abc".to_string(),
            description13: "abc".to_string(),
            description14: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_option_list_add(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_option_list_add(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_option_add() {
        let msg = SvVoteOptionAdd {
            description: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_option_add(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_option_add(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_option_remove() {
        let msg = SvVoteOptionRemove {
            description: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_option_remove(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_option_remove(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_set() {
        let msg = SvVoteSet {
            timeout: 0,
            description: "abc".to_string(),
            reason: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_set(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_set(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_status() {
        let msg = SvVoteStatus {
            yes: 0,
            no: 0,
            pass: 0,
            total: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_status(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_status(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_say() {
        let msg = ClSay {
            team: 0,
            message: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_say(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_say(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_set_team() {
        let msg = ClSetTeam { team: -1 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_set_team(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_set_team(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_set_spectator_mode() {
        let msg = ClSetSpectatorMode { spectator_id: -1 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_set_spectator_mode(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_set_spectator_mode(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_start_info() {
        let msg = ClStartInfo {
            name: "abc".to_string(),
            clan: "abc".to_string(),
            country: 0,
            skin: "abc".to_string(),
            use_custom_color: 0,
            color_body: 0,
            color_feet: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_start_info(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_start_info(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_change_info() {
        let msg = ClChangeInfo {
            name: "abc".to_string(),
            clan: "abc".to_string(),
            country: 0,
            skin: "abc".to_string(),
            use_custom_color: 0,
            color_body: 0,
            color_feet: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_change_info(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_change_info(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_kill() {
        let msg = ClKill {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_kill(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_kill(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_emoticon() {
        let msg = ClEmoticon { emoticon: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_emoticon(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_emoticon(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_vote() {
        let msg = ClVote { vote: -1 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_vote(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_vote(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_call_vote() {
        let msg = ClCallVote {
            type_: "abc".to_string(),
            value: "abc".to_string(),
            reason: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_call_vote(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_call_vote(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_is_dd_net_legacy() {
        let msg = ClIsDDNetLegacy {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_is_dd_net_legacy(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_is_dd_net_legacy(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_dd_race_time_legacy() {
        let msg = SvDDRaceTimeLegacy {
            time: 0,
            check: 0,
            finish: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_dd_race_time_legacy(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_dd_race_time_legacy(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_record_legacy() {
        let msg = SvRecordLegacy {
            server_time_best: 0,
            player_time_best: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_record_legacy(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_record_legacy(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_unused2() {
        let msg = Unused2 {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_unused2(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_unused2(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_teams_state_legacy() {
        let msg = SvTeamsStateLegacy {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_teams_state_legacy(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_teams_state_legacy(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_show_others_legacy() {
        let msg = ClShowOthersLegacy { show: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_show_others_legacy(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_show_others_legacy(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_my_own_message() {
        let msg = SvMyOwnMessage { test: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_my_own_message(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_my_own_message(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_show_distance() {
        let msg = ClShowDistance { x: 0, y: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_show_distance(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_show_distance(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_show_others() {
        let msg = ClShowOthers { show: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_show_others(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_show_others(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_camera_info() {
        let msg = ClCameraInfo {
            zoom: 0,
            deadzone: 0,
            follow_factor: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_camera_info(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_camera_info(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_teams_state() {
        let msg = SvTeamsState {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_teams_state(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_teams_state(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_dd_race_time() {
        let msg = SvDDRaceTime {
            time: 0,
            check: 0,
            finish: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_dd_race_time(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_dd_race_time(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_record() {
        let msg = SvRecord {
            server_time_best: 0,
            player_time_best: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_record(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_record(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_kill_msg_team() {
        let msg = SvKillMsgTeam { team: 0, first: -1 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_kill_msg_team(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_kill_msg_team(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_your_vote() {
        let msg = SvYourVote { voted: -1 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_your_vote(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_your_vote(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_race_finish() {
        let msg = SvRaceFinish {
            client_id: 0,
            time: 0,
            diff: 0,
            record_personal: 0,
            record_server: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_race_finish(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_race_finish(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_command_info() {
        let msg = SvCommandInfo {
            name: "abc".to_string(),
            args_format: "abc".to_string(),
            help_text: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_command_info(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_command_info(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_command_info_remove() {
        let msg = SvCommandInfoRemove {
            name: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_command_info_remove(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_command_info_remove(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_option_group_start() {
        let msg = SvVoteOptionGroupStart {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_option_group_start(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_option_group_start(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_vote_option_group_end() {
        let msg = SvVoteOptionGroupEnd {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_vote_option_group_end(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_vote_option_group_end(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_command_info_group_start() {
        let msg = SvCommandInfoGroupStart {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_command_info_group_start(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_command_info_group_start(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_command_info_group_end() {
        let msg = SvCommandInfoGroupEnd {};
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_command_info_group_end(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_command_info_group_end(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_change_info_cooldown() {
        let msg = SvChangeInfoCooldown { wait_until: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_change_info_cooldown(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_change_info_cooldown(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_map_sound_global() {
        let msg = SvMapSoundGlobal { sound_id: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_map_sound_global(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_map_sound_global(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_pre_input() {
        let msg = SvPreInput {
            direction: 0,
            target_x: 0,
            target_y: 0,
            jump: 0,
            fire: 0,
            hook: 0,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
            owner: 0,
            intended_tick: 0,
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_pre_input(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_pre_input(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_save_code() {
        let msg = SvSaveCode {
            state: 0,
            error: "abc".to_string(),
            save_requester: "abc".to_string(),
            server_name: "abc".to_string(),
            generated_code: "abc".to_string(),
            code: "abc".to_string(),
            team_members: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_save_code(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_save_code(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_server_alert() {
        let msg = SvServerAlert {
            message: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_server_alert(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_server_alert(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_moderator_alert() {
        let msg = SvModeratorAlert {
            message: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_moderator_alert(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_moderator_alert(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_cl_enable_spectator_count() {
        let msg = ClEnableSpectatorCount { enable: 0 };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_cl_enable_spectator_count(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_cl_enable_spectator_count(&mut unpacker), Some(msg));
    }

    #[test]
    fn roundtrip_sv_map_info() {
        let msg = SvMapInfo {
            description: "abc".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = crate::packer::Packer::new(&mut buf);
        encode_sv_map_info(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = crate::packer::Unpacker::new(packer.data());
        assert_eq!(decode_sv_map_info(&mut unpacker), Some(msg));
    }
}
