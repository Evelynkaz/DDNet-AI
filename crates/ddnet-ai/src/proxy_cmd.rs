//! `ddnet-ai proxy-check` and the proxy wiring shared by `play` and `record` (task 2.6, D-053 amendment).
//!
//! `proxy-check --proxy <name>` checks **only the proxy**: the TCP connect, the authentication, the `UDP ASSOCIATE`
//! reply and the relay address, and says which relay rule applied (the file's `relay`). In the default mode it opens no
//! UDP socket; in `relay = "public"` mode (task 2.6b) it also sends DNS queries for a neutral name to a public resolver
//! through the relay and prints the median round-trip time. It never sends anything to a game server. It prints `ok`,
//! `UDP not supported` or `auth failed` (and never the proxy's address or credentials).

use clap::Args;
use ddai_client::ClientConfig;
use ddai_client::proxy;
use ddai_client::proxy::RelayMode;
use ddai_client::socks5::{self, ProxyCheck, RelayHost, Socks5Error, Timeouts};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Exit code for "the proxy answered `UDP ASSOCIATE` with 0x07": distinct, so a script can tell it from a
/// network failure.
pub const EXIT_UDP_NOT_SUPPORTED: u8 = 3;
/// Exit code for a rejected username/password.
pub const EXIT_AUTH_FAILED: u8 = 4;

#[derive(Debug, Args)]
pub struct ProxyCheckArgs {
    /// Proxy name: `<secrets-dir>/<name>-proxy.toml` (mode 0600, keys host, port, user, pass, and optionally for_server, relay, session_pick).
    #[arg(long)]
    pub proxy: String,
    /// Base data directory; the proxy file is read from `<data-dir>/secrets`. Defaults to `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// Directory holding `<name>-proxy.toml`, instead of `<data-dir>/secrets`.
    #[arg(long)]
    pub secrets_dir: Option<PathBuf>,
}

/// `~/aiddnet/data` (Linux; `ddai_os::dirs` for Windows and the override), same fallback as the other commands.
fn default_data_dir() -> PathBuf {
    ddai_os::dirs::data_root_or_relative()
}

/// What `proxy-check` prints for a finished check and the exit code that goes with it. Pure, so the wording
/// is tested.
pub fn report(name: &str, result: &Result<ProxyCheck, Socks5Error>) -> (String, u8) {
    match result {
        Ok(c) => {
            let relay = match c.relay_host {
                RelayHost::SameAsProxy => "on the proxy's host".to_string(),
                RelayHost::Substituted => {
                    "announced on another host: NOT trusted, the proxy's own address with the announced port is \
                     used instead (the usual handling of a proxy behind NAT)"
                        .to_string()
                }
                RelayHost::Remote => "on ANOTHER host than the proxy: accepted (a public unicast address)".to_string(),
            };
            let rule = match c.mode {
                RelayMode::ProxyHostOnly => {
                    "relay rule: proxy-host-only (default): datagrams go to the proxy's own host only"
                }
                RelayMode::Public => {
                    "relay rule: public: a relay on another host is used only if it is a public unicast address, \
                     none of the game server's IPs and not its port (the unit's cgroup filter must deny the \
                     server's IPs, deploy/README.md)"
                }
            };
            let auth = if c.authenticated {
                "username/password accepted"
            } else {
                "no authentication needed"
            };
            let mut text = format!(
                "proxy {name:?}: ok: TCP connect, {auth}, UDP ASSOCIATE accepted; relay {relay}, port {}. {rule}.",
                c.relay_port
            );
            if let Some(p) = &c.probe {
                text += &format!(
                    " UDP through the relay works: {} of {} DNS queries to 1.1.1.1:53 answered, median RTT {} ms.",
                    p.replies,
                    p.sent,
                    p.median.as_millis()
                );
            }
            if let Some(s) = &c.sessions {
                let list = s
                    .rtts
                    .iter()
                    .map(|r| r.map_or_else(|| "failed".to_string(), |d| format!("{} ms", d.as_millis())))
                    .collect::<Vec<_>>()
                    .join(", ");
                text += &format!(
                    " Sessions tried (median RTT): {list}; session {} is the one kept.",
                    s.picked + 1
                );
            }
            text += " Nothing was sent to any game server.";
            (text, 0)
        }
        Err(Socks5Error::UdpNotSupported) => (
            format!(
                "proxy {name:?}: UDP not supported: UDP ASSOCIATE was answered with reply 0x07 (command not \
                 supported). This proxy is TCP-only and cannot carry the game (D-053)."
            ),
            EXIT_UDP_NOT_SUPPORTED,
        ),
        Err(Socks5Error::AuthFailed) => (
            format!("proxy {name:?}: auth failed: the proxy rejected the username/password. Not retrying."),
            EXIT_AUTH_FAILED,
        ),
        Err(e) => (format!("proxy {name:?}: failed: {e}"), 1),
    }
}

pub fn run(args: ProxyCheckArgs) -> ExitCode {
    let secrets_dir = args
        .secrets_dir
        .clone()
        .unwrap_or_else(|| proxy::secrets_dir_for(&args.data_dir.clone().unwrap_or_else(default_data_dir)));
    let cfg = match proxy::load_proxy(&secrets_dir, &args.proxy) {
        Ok(c) => c,
        Err(e) => {
            // The error never contains a value from the file (see `ddai_client::proxy`).
            eprintln!("proxy-check: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = socks5::check(&cfg, &Timeouts::default());
    let (text, code) = report(cfg.name(), &result);
    if code == 0 {
        println!("{text}");
    } else {
        eprintln!("{text}");
    }
    ExitCode::from(code)
}

/// Sets `config.proxy` for a connection to `server` from the allow-list entry that admits it: a proxy only if the
/// entry names one (`proxy = "<name>"`), read from `<data-dir>/secrets/<name>-proxy.toml`. The driver re-checks
/// the pairing before every attempt, so this is the convenience, not the gate. An entry that names a proxy whose
/// file cannot be loaded (missing, wrong mode, malformed) is an error: never a silent direct connection.
pub fn attach_proxy(config: &mut ClientConfig, server: SocketAddr, data_dir: &Path) -> Result<(), AttachError> {
    match proxy::resolve_for_server(
        server,
        &config.name,
        &config.live_servers,
        &proxy::secrets_dir_for(data_dir),
    ) {
        Ok(p) => {
            if let Some(p) = &p {
                tracing::info!(proxy = %p.name(), "the allow-list entry for this server names a SOCKS5 proxy");
            }
            config.proxy = p;
            Ok(())
        }
        Err(e) => Err(AttachError {
            message: e.to_string(),
            exit: 1,
        }),
    }
}

/// Why [`attach_proxy`] / [`prepare_client`] refused, and the process exit code that goes with it.
#[derive(Debug)]
pub struct AttachError {
    pub message: String,
    pub exit: u8,
}

impl std::fmt::Display for AttachError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// What `play` does with `--live-servers` and the proxy: an explicit allow-list path replaces the default one
/// that `ClientConfig::default()` loaded (a file that cannot be read is an error, never "allow everything"), and
/// then the favourites of `<data-dir>/launch/favourites.json` are added as `ready` entries (task 5.12), and the proxy is attached from the entry that admits `server` ([`attach_proxy`]).
pub fn prepare_client(
    config: &mut ClientConfig,
    live_servers: Option<&Path>,
    server: SocketAddr,
    data_dir: &Path,
) -> Result<(), AttachError> {
    if let Some(path) = live_servers {
        config.live_servers = ddai_client::live_servers::LiveServers::load_or_empty(path).map_err(|e| AttachError {
            message: e.to_string(),
            exit: 1,
        })?;
    }
    // Task 5.12 (D-099): the owner's favourites (written by the site, validated strictly here again) count as `ready` entries, with
    // the proxy the owner assigned. A favourites file that cannot be trusted adds none (the gate then refuses that server).
    let (merged, why) = ddai_client::live_servers::LiveServers::load_with_favourites(
        std::mem::take(&mut config.live_servers),
        &ddai_client::favourites::default_path(data_dir),
        ddai_client::favourites::Rules::current(),
    );
    if let Some(why) = why {
        tracing::warn!(reason = why, "the favourites file is not used");
    }
    config.live_servers = merged;
    attach_proxy(config, server, data_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_client::socks5::Step;

    #[test]
    fn the_three_outcomes_the_lead_looks_for_read_clearly_and_exit_differently() {
        let ok = report(
            "swarfey",
            &Ok(ProxyCheck {
                relay_host: RelayHost::SameAsProxy,
                relay_port: 4242,
                authenticated: true,
                mode: RelayMode::ProxyHostOnly,
                probe: None,
                sessions: None,
            }),
        );
        let subst = report(
            "swarfey",
            &Ok(ProxyCheck {
                relay_host: RelayHost::Substituted,
                relay_port: 4242,
                authenticated: false,
                mode: RelayMode::ProxyHostOnly,
                probe: None,
                sessions: None,
            }),
        );
        assert_eq!(subst.1, 0);
        assert!(
            subst.0.contains("NOT trusted") && subst.0.contains("4242"),
            "{}",
            subst.0
        );
        assert_eq!(ok.1, 0);
        assert!(
            ok.0.contains(": ok:") && ok.0.contains("4242") && ok.0.contains("Nothing was sent"),
            "{}",
            ok.0
        );
        let no_udp = report("swarfey", &Err(Socks5Error::UdpNotSupported));
        assert_eq!(no_udp.1, EXIT_UDP_NOT_SUPPORTED);
        assert!(no_udp.0.contains("UDP not supported"), "{}", no_udp.0);
        let auth = report("swarfey", &Err(Socks5Error::AuthFailed));
        assert_eq!(auth.1, EXIT_AUTH_FAILED);
        assert!(auth.0.contains("auth failed"), "{}", auth.0);
        let net = report("swarfey", &Err(Socks5Error::Timeout { step: Step::Connect }));
        assert_eq!(net.1, 1);
        assert!(net.0.contains("timed out"), "{}", net.0);
        let codes = [0, EXIT_UDP_NOT_SUPPORTED, EXIT_AUTH_FAILED, 1];
        for (i, a) in codes.iter().enumerate() {
            for b in &codes[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn the_report_names_the_relay_rule_the_probe_and_the_sessions() {
        use ddai_client::socks5::{ProbeReport, SessionReport};
        use std::time::Duration;
        let ms = Duration::from_millis;
        let default_mode = report(
            "p",
            &Ok(ProxyCheck {
                relay_host: RelayHost::SameAsProxy,
                relay_port: 1,
                authenticated: false,
                mode: RelayMode::ProxyHostOnly,
                probe: None,
                sessions: None,
            }),
        );
        assert!(
            default_mode.0.contains("relay rule: proxy-host-only"),
            "{}",
            default_mode.0
        );
        assert!(!default_mode.0.contains("median RTT"), "{}", default_mode.0);
        let public = report(
            "p",
            &Ok(ProxyCheck {
                relay_host: RelayHost::Remote,
                relay_port: 4242,
                authenticated: true,
                mode: RelayMode::Public,
                probe: Some(ProbeReport {
                    sent: 5,
                    replies: 4,
                    median: ms(22),
                }),
                sessions: Some(SessionReport {
                    rtts: vec![Some(ms(40)), None, Some(ms(22))],
                    picked: 2,
                }),
            }),
        );
        assert_eq!(public.1, 0);
        for want in [
            "relay rule: public",
            "ANOTHER host",
            "4 of 5 DNS queries to 1.1.1.1:53 answered, median RTT 22 ms",
            "Sessions tried (median RTT): 40 ms, failed, 22 ms; session 3 is the one kept",
            "Nothing was sent to any game server",
        ] {
            assert!(public.0.contains(want), "{want:?} not in {}", public.0);
        }
        // A refused relay and an unanswered probe are plain failures (exit 1) that say why and print no address.
        for e in [
            Socks5Error::RelayAddress(
                "the announced relay address is a private address (relay = public needs a public unicast address)",
            ),
            Socks5Error::ProbeFailed,
        ] {
            let (text, code) = report("p", &Err(e));
            assert_eq!(code, 1);
            assert!(text.contains("failed"), "{text}");
        }
    }

    // --- task 5.12: the favourites are `ready` entries for the bot's own gate --------------------------------------------

    fn favourites_dir(favs: &serde_json::Value, profile: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("launch")).unwrap();
        std::fs::create_dir_all(dir.path().join("secrets")).unwrap();
        std::fs::write(dir.path().join("launch/favourites.json"), favs.to_string()).unwrap();
        if profile {
            let path = dir.path().join("secrets/hp-proxy.toml");
            std::fs::write(
                &path,
                "host = \"198.51.100.7\"\nport = 1080\nuser = \"u\"\npass = \"p\"\n",
            )
            .unwrap();
            ddai_os::private::restrict_file(&path).unwrap();
        }
        dir
    }

    fn fav_json(address: &str, connection: &str) -> serde_json::Value {
        serde_json::json!({"v":1,"favourites":[{"address":address,"name":"S","nick":"Muha","connection":connection,"consent_at":5,"added_at":5}]})
    }

    fn fresh_config(nick: &str) -> ClientConfig {
        ClientConfig {
            name: nick.to_string(),
            ..ClientConfig::default()
        }
    }

    #[test]
    fn a_favourite_passes_the_bots_gate_with_the_proxy_the_owner_assigned() {
        let dir = favourites_dir(&fav_json("93.184.216.35:8308", "proxy:hp"), true);
        let none = dir.path().join("no-live-servers.toml");
        let addr: SocketAddr = "93.184.216.35:8308".parse().unwrap();
        let mut c = fresh_config("Muha");
        prepare_client(&mut c, Some(&none), addr, dir.path()).expect("prepared");
        assert!(ddai_client::live_servers::check(addr, "Muha", &c.live_servers).is_ok());
        assert_eq!(c.proxy.as_ref().map(|p| p.name()), Some("hp"));
        // Another nick and another address are not admitted.
        assert!(ddai_client::live_servers::check(addr, "Other", &c.live_servers).is_err());
        assert!(
            ddai_client::live_servers::check("93.184.216.35:8309".parse().unwrap(), "Muha", &c.live_servers).is_err()
        );
        // The assigned proxy's file is gone: an error, never a direct connection or another proxy.
        std::fs::remove_file(dir.path().join("secrets/hp-proxy.toml")).unwrap();
        let mut c = fresh_config("Muha");
        let e = prepare_client(&mut c, Some(&none), addr, dir.path()).unwrap_err();
        assert!(c.proxy.is_none() && e.exit == 1, "{e}");
        // A direct favourite has no proxy.
        let dir = favourites_dir(&fav_json("93.184.216.35:8308", "direct"), false);
        let mut c = fresh_config("Muha");
        prepare_client(&mut c, Some(&none), addr, dir.path()).unwrap();
        assert!(c.proxy.is_none());
        assert!(ddai_client::live_servers::check(addr, "Muha", &c.live_servers).is_ok());
    }

    #[test]
    fn a_favourites_file_that_cannot_be_trusted_adds_nothing_to_the_bots_gate() {
        let addr: SocketAddr = "93.184.216.35:8308".parse().unwrap();
        let none = tempfile::tempdir().unwrap().path().join("no-live-servers.toml");
        for bad in [
            serde_json::json!({"v":1,"favourites":[{"address":"93.184.216.35:8308","name":"S","nick":"Muha","connection":"direct","consent_at":0,"added_at":5}]}),
            serde_json::json!({"v":1,"favourites":[{"address":"93.184.216.35:8308","name":"S","nick":"Muha","connection":"direct","consent_at":5,"added_at":5,"ready":true}]}),
            serde_json::json!({"v":2,"favourites":[]}),
            serde_json::json!("junk"),
        ] {
            let dir = favourites_dir(&bad, false);
            let mut c = fresh_config("Muha");
            prepare_client(&mut c, Some(&none), addr, dir.path()).unwrap();
            assert!(
                ddai_client::live_servers::check(addr, "Muha", &c.live_servers).is_err(),
                "{bad}"
            );
        }
        // No file at all: nothing added either.
        let dir = tempfile::tempdir().unwrap();
        let mut c = fresh_config("Muha");
        prepare_client(&mut c, Some(&none), addr, dir.path()).unwrap();
        assert!(ddai_client::live_servers::check(addr, "Muha", &c.live_servers).is_err());
    }
}
