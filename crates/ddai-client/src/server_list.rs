//! The master server list, **read-only**, and the restricted auto-pick (task 4.3; `src/bot/serverPick.ts`,
//! `docs/research/orig-bot.md` §11).
//!
//! # What changed from the TS, on purpose
//!
//! The TS `serverPick.ts` fetched `master{1..4}.ddnet.org/ddnet/15/servers.json` and **connected to the
//! most populated public block server** (`pickBlockServer`). This bot never does: it may only connect to
//! the servers on an explicit allow-list (CLAUDE.md, D-043, D-051, D-052) —
//!
//! * the **local server** (loopback, always), and
//! * the entries of the owner's `live-servers.toml` that carry `ready = true` (the owner said the IP or
//!   the proxy is ready) and whose pinned nick is the one we play under.
//!
//! The master list is used for two read-only things only: [`list_block`] shows the operator the block
//! servers with their player counts (`ddnet-ai servers --block`), and [`pick_auto`] chooses *among the
//! allow-listed candidates* by how many players the list says are there. A server that is not an
//! allow-listed candidate can never be returned, however full it is, and with no candidate worth it the
//! answer is the local server. The parser holds no player names: clients are counted, not kept.
//!
//! Fetching is not here (`ddnet-ai` does it, over HTTPS, read-only); this module is pure data so that
//! its tests need no network.

use std::net::{Ipv4Addr, SocketAddr};

use crate::live_servers::{self, LiveServers};

/// The master lists, tried in order.
pub const MASTERS: [&str; 4] = [
    "https://master1.ddnet.org/ddnet/15/servers.json",
    "https://master2.ddnet.org/ddnet/15/servers.json",
    "https://master3.ddnet.org/ddnet/15/servers.json",
    "https://master4.ddnet.org/ddnet/15/servers.json",
];

/// The local server (`ddnet-local.service`).
pub const LOCAL_SERVER: &str = "127.0.0.1:8303";

/// One server of the list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerRow {
    pub address: SocketAddr,
    pub name: String,
    pub map: String,
    pub game_type: String,
    pub location: String,
    pub passworded: bool,
    /// Clients that play (not spectators).
    pub players: u32,
    pub clients: u32,
    pub max_clients: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum MasterError {
    #[error("the master list is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// `ADDR_RE` / `pickAddress`: the first `tw-0.6+udp://ip:port` (what this client speaks), else the first of
/// `tw-0.7+udp`, `ddnet+udp`, `udp`. IPv4 only; an out-of-range octet or port skips the address.
fn pick_address(addresses: &serde_json::Value) -> Option<SocketAddr> {
    let mut fallback = None;
    for a in addresses.as_array()? {
        let Some(text) = a.as_str() else { continue };
        let text = text.trim();
        let Some((scheme, rest)) = text.split_once("://") else {
            continue;
        };
        if !matches!(scheme, "tw-0.6+udp" | "tw-0.7+udp" | "ddnet+udp" | "udp") {
            continue;
        }
        let Some((ip, port)) = rest.rsplit_once(':') else {
            continue;
        };
        let (Ok(ip), Ok(port)) = (ip.parse::<Ipv4Addr>(), port.parse::<u16>()) else {
            continue;
        };
        if port == 0 {
            continue;
        }
        let addr = SocketAddr::from((ip, port));
        if scheme == "tw-0.6+udp" {
            return Some(addr);
        }
        fallback.get_or_insert(addr);
    }
    fallback
}

fn text(v: Option<&serde_json::Value>, max: usize) -> String {
    v.and_then(|v| v.as_str())
        .map(|s| s.chars().take(max).collect())
        .unwrap_or_default()
}

/// `parseMaster`: the rows of a master list. Entries without a usable address, duplicates (the first
/// wins) and anything that is not an object are skipped; nothing about the clients is kept but their number.
pub fn parse_master(json: &str) -> Result<Vec<ServerRow>, MasterError> {
    let doc: serde_json::Value = serde_json::from_str(json)?;
    let mut rows: Vec<ServerRow> = Vec::new();
    let Some(list) = doc.get("servers").and_then(|s| s.as_array()) else {
        return Ok(rows);
    };
    for s in list {
        let Some(obj) = s.as_object() else { continue };
        let Some(address) = obj.get("addresses").and_then(pick_address) else {
            continue;
        };
        if rows.iter().any(|r| r.address == address) {
            continue;
        }
        let info = obj.get("info");
        let get = |k: &str| info.and_then(|i| i.get(k));
        let clients = get("clients").and_then(|c| c.as_array());
        let players = clients.map_or(0, |c| {
            c.iter()
                .filter(|x| x.is_object() && x.get("is_player").and_then(|p| p.as_bool()) != Some(false))
                .count()
        });
        let map = match get("map") {
            Some(m) if m.is_object() => text(m.get("name"), 64),
            other => text(other, 64),
        };
        let name = text(get("name"), 128);
        rows.push(ServerRow {
            address,
            name: if name.is_empty() { address.to_string() } else { name },
            map,
            game_type: text(get("game_type"), 32),
            location: text(obj.get("location"), 16),
            passworded: get("passworded").and_then(|p| p.as_bool()) == Some(true),
            players: players as u32,
            clients: clients.map_or(0, Vec::len) as u32,
            max_clients: get("max_clients")
                .and_then(|m| m.as_u64())
                .and_then(|m| u32::try_from(m).ok())
                .unwrap_or(0),
        });
    }
    Ok(rows)
}

/// `BLOCK_RE = /block|blmap|copy (love|the) box|love box/i` over the name, the map and the game type.
pub fn is_block(r: &ServerRow) -> bool {
    [&r.name, &r.map, &r.game_type].iter().any(|s| {
        let l = s.to_lowercase();
        l.contains("block")
            || l.contains("blmap")
            || l.contains("copy love box")
            || l.contains("copy the box")
            || l.contains("love box")
    })
}

/// No room left for one more (`maxClients > 0 && clients >= maxClients - 1`).
pub fn is_full(r: &ServerRow) -> bool {
    r.max_clients > 0 && r.clients + 1 >= r.max_clients
}

/// The block servers, most players first (ties by address), for the operator.
pub fn list_block(rows: &[ServerRow]) -> Vec<&ServerRow> {
    let mut v: Vec<&ServerRow> = rows.iter().filter(|r| is_block(r)).collect();
    v.sort_by(|a, b| b.players.cmp(&a.players).then(a.address.cmp(&b.address)));
    v
}

/// Whether `addr` may be connected to by `--server auto` under `nick`: loopback, or a `ready` entry of the
/// owner's list pinned to exactly this nick.
pub fn is_auto_candidate(addr: SocketAddr, nick: &str, list: &LiveServers) -> bool {
    live_servers::is_loopback(addr) || list.ready_nick(addr) == Some(nick)
}

/// What `--server auto` decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoPick {
    pub addr: SocketAddr,
    /// Why, for the operator's terminal.
    pub why: String,
}

/// `--server auto`, restricted: the allow-listed ready servers that the master list shows as block servers
/// worth joining (not passworded, not full, at least two players — someone to fight, as in the TS), the most
/// populated first; with none, the local server. `rows` may be empty (the list could not be fetched).
pub fn pick_auto(local: SocketAddr, nick: &str, list: &LiveServers, rows: &[ServerRow]) -> AutoPick {
    let best = rows
        .iter()
        .filter(|r| is_block(r) && !r.passworded && !is_full(r) && r.players >= 2)
        .filter(|r| list.ready_nick(r.address) == Some(nick))
        .max_by(|a, b| a.players.cmp(&b.players).then(b.address.cmp(&a.address)));
    match best {
        Some(r) => AutoPick {
            addr: r.address,
            why: format!(
                "the allow-listed, ready block server {} with {} players",
                r.address, r.players
            ),
        },
        None => AutoPick {
            addr: local,
            why: if list.ready_entries().next().is_none() {
                "no server is marked ready in live-servers.toml: the local server".to_string()
            } else if rows.is_empty() {
                "the master list is not available: the local server".to_string()
            } else {
                "no allow-listed ready server has players now: the local server".to_string()
            },
        },
    }
}

/// [`pick_auto`] that asks for the master list **only when there is a ready entry to choose among** (review F4:
/// with none, the answer is the local server and no request needs to leave the machine). `fetch` is the
/// read-only list query; its error text is the operator's warning.
pub fn pick_auto_fetching(
    local: SocketAddr,
    nick: &str,
    list: &LiveServers,
    fetch: impl FnOnce() -> Result<Vec<ServerRow>, String>,
) -> (AutoPick, Option<String>) {
    if list.ready_entries().next().is_none() {
        return (pick_auto(local, nick, list, &[]), None);
    }
    match fetch() {
        Ok(rows) => (pick_auto(local, nick, list, &rows), None),
        Err(e) => (pick_auto(local, nick, list, &[]), Some(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/master-servers.json");

    fn owner() -> SocketAddr {
        "45.141.57.35:8308".parse().unwrap()
    }

    fn list(ready: bool) -> LiveServers {
        toml::from_str(&format!(
            "[[server]]\naddress = \"45.141.57.35:8308\"\nnick = \"Muha\"\npurpose = \"play\"\nready = {ready}\n"
        ))
        .unwrap()
    }

    fn local() -> SocketAddr {
        LOCAL_SERVER.parse().unwrap()
    }

    #[test]
    fn the_fixture_parses_and_the_junk_in_it_is_skipped() {
        let rows = parse_master(FIXTURE).unwrap();
        // 17 entries: 12 good ones; a duplicate address, no addresses, bad addresses, an entry with no info
        // but a good address, and a string are skipped or kept as the rules say.
        let addrs: Vec<String> = rows.iter().map(|r| r.address.to_string()).collect();
        assert!(addrs.contains(&"45.141.57.35:8308".to_string()), "{addrs:?}");
        assert!(
            addrs.contains(&"203.0.113.30:8303".to_string()),
            "a 0.7-only server falls back: {addrs:?}"
        );
        assert_eq!(
            addrs.iter().filter(|a| a.as_str() == "46.174.54.240:8302").count(),
            1,
            "the duplicate address is dropped"
        );
        assert!(
            !addrs
                .iter()
                .any(|a| a.contains("999") || a.contains("203.0.113.40") || a.contains("203.0.113.41"))
        );
        let a = rows.iter().find(|r| r.address == owner()).unwrap();
        assert_eq!((a.players, a.clients, a.max_clients), (2, 2, 64));
        assert_eq!(
            (a.map.as_str(), a.game_type.as_str(), a.location.as_str()),
            ("Copy Love Box", "DDFightNet fng", "eu:it")
        );
        let g = rows
            .iter()
            .find(|r| r.address.port() == 8303 && r.address.ip().to_string() == "203.0.113.30")
            .unwrap();
        assert_eq!(
            (g.players, g.clients),
            (2, 3),
            "a spectator is a client but not a player"
        );
        assert_eq!(g.map, "Copy Love Box", "the map as a plain string");
        assert!(
            rows.iter()
                .find(|r| r.name.starts_with("Block server E"))
                .unwrap()
                .passworded
        );
    }

    #[test]
    fn the_parser_holds_no_player_names_and_survives_garbage() {
        let rows = parse_master(FIXTURE).unwrap();
        let with_names = r#"{"servers": [{"addresses": ["udp://10.0.0.1:8303"], "info": {"name": "S", "clients": [{"name": "SECRETNICK", "clan": "SECRETCLAN", "is_player": true}]}}]}"#;
        let r = parse_master(with_names).unwrap();
        assert_eq!(r[0].players, 1);
        assert!(
            !format!("{r:?}{rows:?}").contains("SECRET"),
            "clients are counted, never kept"
        );
        assert_eq!(parse_master("{}").unwrap(), vec![]);
        assert_eq!(parse_master(r#"{"servers": 5}"#).unwrap(), vec![]);
        assert_eq!(
            parse_master(r#"{"servers": [1, null, "x", {"addresses": 3}]}"#).unwrap(),
            vec![]
        );
        assert!(parse_master("not json").is_err());
        // An entry with an address and nothing else is a row named by its address.
        let r = parse_master(r#"{"servers": [{"addresses": ["udp://10.0.0.1:8303"]}]}"#).unwrap();
        assert_eq!(r[0].name, "10.0.0.1:8303");
        assert_eq!(r[0].players, 0);
    }

    #[test]
    fn block_servers_are_found_by_name_map_or_game_type_and_listed_by_players() {
        let rows = parse_master(FIXTURE).unwrap();
        let names: Vec<&str> = list_block(&rows).iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names[0], "Block server D", "43 players first");
        assert!(names.contains(&"Server H"), "by the map name (blmap)");
        assert!(names.contains(&"Server I"), "by the game type");
        assert!(
            !names.iter().any(|n| n.contains("CTF") || n.contains("Race")),
            "{names:?}"
        );
    }

    #[test]
    fn auto_picks_only_a_ready_allow_listed_server_and_otherwise_the_local_one() {
        let rows = parse_master(FIXTURE).unwrap();
        // The owner has not said ready: the local server, although public block servers are busier.
        let p = pick_auto(local(), "Muha", &list(false), &rows);
        assert_eq!(p.addr, local());
        assert!(p.why.contains("no server is marked ready"), "{}", p.why);
        // Ready and populated: that one — and not the 43-player public server.
        let p = pick_auto(local(), "Muha", &list(true), &rows);
        assert_eq!(p.addr, owner(), "{}", p.why);
        // Ready but under another nick: refused.
        let p = pick_auto(local(), "Intruder", &list(true), &rows);
        assert_eq!(p.addr, local());
        // Ready but the list is not available.
        let p = pick_auto(local(), "Muha", &list(true), &[]);
        assert_eq!(p.addr, local());
        assert!(p.why.contains("not available"), "{}", p.why);
        // Ready but nobody is there (only the bot would be): the local server.
        let mut quiet = rows.clone();
        quiet.iter_mut().find(|r| r.address == owner()).unwrap().players = 1;
        assert_eq!(pick_auto(local(), "Muha", &list(true), &quiet).addr, local());
        // Passworded or full: not joined.
        let mut pw = rows.clone();
        pw.iter_mut().find(|r| r.address == owner()).unwrap().passworded = true;
        assert_eq!(pick_auto(local(), "Muha", &list(true), &pw).addr, local());
        let mut full = rows.clone();
        let o = full.iter_mut().find(|r| r.address == owner()).unwrap();
        o.max_clients = o.clients + 1;
        assert_eq!(pick_auto(local(), "Muha", &list(true), &full).addr, local());
    }

    #[test]
    fn nothing_outside_the_allow_list_can_ever_come_out_of_auto() {
        let rows = parse_master(FIXTURE).unwrap();
        for ready in [false, true] {
            for nick in ["Muha", "Other", ""] {
                let l = list(ready);
                let p = pick_auto(local(), nick, &l, &rows);
                assert!(
                    is_auto_candidate(p.addr, nick, &l),
                    "{ready} {nick:?}: {} is not a candidate",
                    p.addr
                );
                // And what `auto` returns always passes the connect gate the client enforces anyway.
                assert!(live_servers::check(p.addr, nick, &l).is_ok(), "{ready} {nick:?}");
            }
        }
        // Every row of the list, one at a time as the only ready entry: only an allow-listed one is ever picked.
        for r in &rows {
            let l: LiveServers = toml::from_str(&format!(
                "[[server]]\naddress = \"{}\"\nnick = \"Muha\"\nready = false\n",
                r.address
            ))
            .unwrap();
            assert_eq!(
                pick_auto(local(), "Muha", &l, &rows).addr,
                local(),
                "{} is listed but not ready",
                r.address
            );
        }
        assert!(is_auto_candidate(
            "127.0.0.1:9999".parse().unwrap(),
            "x",
            &LiveServers::default()
        ));
        assert!(!is_auto_candidate(
            "46.174.54.240:8302".parse().unwrap(),
            "Muha",
            &list(true)
        ));
    }

    #[test]
    fn the_connect_gate_refuses_a_listed_server_that_is_not_ready_whoever_asks() {
        // Review F1: an explicit `--server` (play, record, the driver) honours `ready` too.
        let err = live_servers::check(owner(), "Muha", &list(false)).unwrap_err();
        assert!(err.to_string().contains("ready = true"), "{err}");
        assert!(live_servers::check(owner(), "Muha", &list(true)).is_ok());
        assert!(live_servers::check(owner(), "Other", &list(true)).is_err());
        // And what `auto` returns always passes the gate, ready or not.
        for ready in [false, true] {
            let l = list(ready);
            let p = pick_auto(local(), "Muha", &l, &parse_master(FIXTURE).unwrap());
            assert!(live_servers::check(p.addr, "Muha", &l).is_ok());
        }
    }

    #[test]
    fn auto_does_not_ask_for_the_master_list_when_nothing_is_ready() {
        let never = || -> Result<Vec<ServerRow>, String> { panic!("the master list must not be fetched") };
        let (p, warn) = pick_auto_fetching(local(), "Muha", &list(false), never);
        assert_eq!((p.addr, warn), (local(), None));
        let (p, _) = pick_auto_fetching(local(), "Muha", &LiveServers::default(), never);
        assert_eq!(p.addr, local());
        // With a ready entry it does ask, and a failed fetch falls back to the local server with a warning.
        let (p, warn) = pick_auto_fetching(local(), "Muha", &list(true), || Err("down".to_string()));
        assert_eq!((p.addr, warn.as_deref()), (local(), Some("down")));
        let rows = parse_master(FIXTURE).unwrap();
        let (p, _) = pick_auto_fetching(local(), "Muha", &list(true), || Ok(rows));
        assert_eq!(p.addr, owner());
    }
}
