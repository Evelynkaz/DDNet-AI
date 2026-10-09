//! `ddnet-ai servers [--block]` (task 4.3): a **read-only** look at the DDNet master list for the operator —
//! which block servers exist, how many players, and which of them the bot is allowed to connect to. It
//! opens no game connection and sends nothing but one HTTPS GET to a master list (`ddai_client::server_list`).
//! Which servers the bot may *connect* to is the allow-list's decision, never this listing's
//! (`--server auto` picks only among allow-listed ready servers, else the local one).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Args;
use ddai_client::live_servers::LiveServers;
use ddai_client::server_list::{self, ServerRow};

#[derive(Debug, Args)]
pub struct ServersArgs {
    /// Only the block servers (Copy Love Box, blmap, "block" in the name, map or game type).
    #[arg(long)]
    pub block: bool,
    /// Show at most this many rows.
    #[arg(long, default_value_t = 30)]
    pub limit: usize,
    /// The nick the bot would play under (decides which allow-listed entries count as usable).
    #[arg(long, default_value = "Muha")]
    pub name: String,
    /// The owner's allow-list (default `~/aiddnet/data/live-servers.toml`).
    #[arg(long)]
    pub live_servers: Option<PathBuf>,
}

/// One HTTPS GET per master until one answers (8 s each); read-only. Errors are text for the terminal.
pub fn fetch_master() -> Result<Vec<ServerRow>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(8))
        .user_agent("ddnet-ai")
        .build()
        .map_err(|e| e.to_string())?;
    let mut last = String::from("no master answered");
    for url in server_list::MASTERS {
        let body = match client
            .get(url)
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.text())
        {
            Ok(b) => b,
            Err(e) => {
                last = format!("{url}: {e}");
                continue;
            }
        };
        match server_list::parse_master(&body) {
            Ok(rows) if !rows.is_empty() => return Ok(rows),
            Ok(_) => last = format!("{url}: an empty list"),
            Err(e) => last = format!("{url}: {e}"),
        }
    }
    Err(last)
}

pub fn run(args: ServersArgs) -> ExitCode {
    let path = args.live_servers.clone().unwrap_or_else(LiveServers::default_path);
    let list = match LiveServers::load_or_empty(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let rows = match fetch_master() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("could not read the master list: {e}");
            return ExitCode::FAILURE;
        }
    };
    print!("{}", render(&rows, &list, &args.name, args.block, args.limit));
    ExitCode::SUCCESS
}

/// The table, one row per server: address, players/max, flags, name.
pub fn render(rows: &[ServerRow], list: &LiveServers, nick: &str, block_only: bool, limit: usize) -> String {
    let mut shown: Vec<&ServerRow> = if block_only {
        server_list::list_block(rows)
    } else {
        let mut v: Vec<&ServerRow> = rows.iter().collect();
        v.sort_by(|a, b| b.players.cmp(&a.players).then(a.address.cmp(&b.address)));
        v
    };
    let total = shown.len();
    shown.truncate(limit);
    let mut out = format!(
        "{total} {}servers on the list; the bot connects only to the local server and to allow-listed ready ones (marked)\n",
        if block_only { "block " } else { "" }
    );
    for r in shown {
        let mark = if server_list::is_auto_candidate(r.address, nick, list) {
            "READY "
        } else if list.allowed_nick(r.address).is_some() {
            "listed"
        } else {
            "      "
        };
        out += &format!(
            "{mark} {:<22} {:>3}/{:<3} {}{}{}  {}  [{} / {}]\n",
            r.address.to_string(),
            r.players,
            if r.max_clients > 0 {
                r.max_clients.to_string()
            } else {
                "?".into()
            },
            if r.passworded { "pw " } else { "" },
            if server_list::is_full(r) { "full " } else { "" },
            r.location,
            r.name,
            r.map,
            r.game_type
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../ddai-client/tests/fixtures/master-servers.json");

    #[test]
    fn the_listing_marks_what_the_bot_may_use_and_makes_no_connection() {
        let rows = server_list::parse_master(FIXTURE).unwrap();
        let list: LiveServers =
            toml::from_str("[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\n").unwrap();
        let text = render(&rows, &list, "Muha", true, 100);
        let line = text.lines().find(|l| l.contains("93.184.216.35:8308")).unwrap();
        assert!(line.starts_with("READY"), "{line}");
        let other = text.lines().find(|l| l.contains("93.184.216.240:8302")).unwrap();
        assert!(other.starts_with("      "), "a public server is never marked: {other}");
        assert!(text.lines().next().unwrap().contains("block servers on the list"));
        assert!(!text.contains("CTF server"), "--block hides the others");
        // Not ready: only listed.
        let list: LiveServers =
            toml::from_str("[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\n").unwrap();
        let text = render(&rows, &list, "Muha", true, 100);
        assert!(
            text.lines()
                .find(|l| l.contains("93.184.216.35:8308"))
                .unwrap()
                .starts_with("listed")
        );
        // The limit.
        assert_eq!(render(&rows, &list, "Muha", true, 3).lines().count(), 4);
    }
}
