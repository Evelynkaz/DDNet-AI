//! `ddnet-ai proxy-check` and the proxy wiring shared by `play` and `record` (task 2.6, D-053 amendment).
//!
//! `proxy-check --proxy <name>` checks **only the proxy**: the TCP connect, the authentication, the `UDP ASSOCIATE`
//! reply and the relay address. It opens no UDP socket and sends nothing to any game server. It prints `ok`,
//! `UDP not supported` or `auth failed` (and never the proxy's address or credentials).

use clap::Args;
use ddai_client::ClientConfig;
use ddai_client::proxy;
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
    /// Proxy name: `<secrets-dir>/<name>-proxy.toml` (mode 0600, keys host, port, user, pass).
    #[arg(long)]
    pub proxy: String,
    /// Base data directory; the proxy file is read from `<data-dir>/secrets`. Defaults to `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// Directory holding `<name>-proxy.toml`, instead of `<data-dir>/secrets`.
    #[arg(long)]
    pub secrets_dir: Option<PathBuf>,
}

/// `~/aiddnet/data`, same fallback as the other commands.
fn default_data_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("aiddnet").join("data"),
        _ => PathBuf::from("data"),
    }
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
            };
            let auth = if c.authenticated {
                "username/password accepted"
            } else {
                "no authentication needed"
            };
            (
                format!(
                    "proxy {name:?}: ok: TCP connect, {auth}, UDP ASSOCIATE accepted; relay {relay}, port {}. \
                     Nothing was sent to any game server.",
                    c.relay_port
                ),
                0,
            )
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
        Err(e) => {
            // A proxy that is issued for another server is a final refusal like any other `ProxyRefused`: exit 4,
            // which the unit's `RestartPreventExitStatus` lists, so systemd does not keep restarting it.
            let exit = if matches!(e, proxy::ProxyResolveError::NotForServer { .. }) {
                EXIT_PROXY_REFUSED
            } else {
                1
            };
            Err(AttachError {
                message: e.to_string(),
                exit,
            })
        }
    }
}

/// Exit code for a final proxy refusal (`GaveUpCategory::ProxyRefused` maps to the same code in the bot).
pub const EXIT_PROXY_REFUSED: u8 = 4;

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
/// then the proxy is attached from the entry that admits `server` ([`attach_proxy`]).
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
            }),
        );
        let subst = report(
            "swarfey",
            &Ok(ProxyCheck {
                relay_host: RelayHost::Substituted,
                relay_port: 4242,
                authenticated: false,
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
}
