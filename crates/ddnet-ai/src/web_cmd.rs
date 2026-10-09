//! Implementation of the `ddnet-ai web` and `ddnet-ai web-passwd` subcommands (task 5.1). All the
//! real logic (security headers, sessions, argon2id, the WebSocket protocol, ...) lives in the
//! `ddai-web` crate; this is thin CLI glue plus the `~/aiddnet/data` default per `CLAUDE.md`'s
//! folder layout.

use clap::Args;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tracing_subscriber::prelude::*;

/// `~/aiddnet/data` on Linux, per `CLAUDE.md`'s folder layout (`%USERPROFILE%\ddnet-ai\data` on Windows, `DDNET_AI_DATA_DIR` overrides
/// both: see `ddai_os::dirs`), falling back to a relative `data` directory if there is no home (e.g. some minimal container/CI
/// environments) rather than failing outright — `--data-dir` overrides this either way.
fn default_data_dir() -> PathBuf {
    ddai_os::dirs::data_root_or_relative()
}

#[derive(Debug, Args)]
pub struct WebArgs {
    /// Address to listen on. MUST be loopback (127.0.0.1 or ::1) unless
    /// `--i-know-this-is-public` is also given: Caddy is meant to be the only public-facing hop
    /// (task 5.3), never this binary directly.
    #[arg(long, default_value = "127.0.0.1:7788")]
    pub(crate) listen: SocketAddr,
    /// Base data directory; secrets are written under `<data-dir>/secrets`. Defaults to
    /// `~/aiddnet/data`.
    #[arg(long)]
    pub(crate) data_dir: Option<PathBuf>,
    /// Trust `X-Forwarded-For` for the client IP used in rate limiting. Only honored when the
    /// TCP peer making the request is itself loopback (i.e. only a reverse proxy running on this
    /// same host, such as Caddy, could have set it) — never set this unless such a proxy is
    /// actually in front.
    #[arg(long)]
    pub(crate) trust_proxy: bool,
    /// Bypasses the non-loopback bind refusal. We will never use this: this server has no TLS
    /// of its own, and Caddy (task 5.3) is meant to be the only public-facing hop.
    #[arg(long)]
    pub(crate) i_know_this_is_public: bool,
    /// Marks session cookies `Secure` and uses the `__Host-` name prefix. Only correct once this
    /// server sits behind Caddy's HTTPS (task 5.3); a real browser refuses a `Secure` cookie sent
    /// over plain HTTP, so leave this unset for local/direct use.
    #[arg(long)]
    pub(crate) cookie_secure: bool,
    /// Task 5.2a: replay Oracle B `trace-b` files (task 1.5) as the live map view's data source —
    /// either a single `.trb` file, or a directory to loop through in sorted-filename order. The
    /// live bot itself doesn't exist yet (tasks 2.3/2.4/4.x); this is the one real `FrameSource`
    /// today. Omit to run the web UI with no live map (only the login/status screen from task
    /// 5.1).
    #[arg(long)]
    pub(crate) replay: Option<PathBuf>,
    /// Task 5.2a: a directory `crate::live::map_resolve` may read a real `.map` file's bytes
    /// from, by filename, when resolving a `--replay` trace's map (repeatable). Typically
    /// `~/aiddnet/data/ddnet-server/maps` and/or a subdirectory of `~/aiddnet/data/maps`. Only
    /// read from, never written to; a path derived from a trace's own (untrusted) metadata is
    /// never used directly — see `ddai_web::live::map_resolve`'s doc comment.
    #[arg(long = "maps-dir")]
    pub(crate) maps_dir: Vec<PathBuf>,
    /// Task 4.1: show the real game instead of replays — the live bot's Unix socket
    /// (`ddnet-ai play --brain ...` serves `<data-dir>/bot/live.sock`; `docs/formats.md` §21).
    /// Read-only. Conflicts with `--replay`. With no `--maps-dir` the bot's own map cache
    /// (`<data-dir>/maps/cache`) is searched.
    #[arg(long = "bot-socket", conflicts_with = "replay")]
    pub(crate) bot_socket: Option<PathBuf>,
    /// Task 5.7: the offline demo's bridge socket (`ddnet-ai fly watch --bridge <it>`, `docs/formats.md` §28): while the live bot
    /// (`--bot-socket`) is not there the site shows the fly playing an arena from this socket, with a badge saying it is not a
    /// real game, and switches to the bot the moment it is up. Needs `--bot-socket`, and must be another path (never the live
    /// socket). Commands never go to the demo: the bot panel is for the live bot only.
    #[arg(long = "demo-socket", requires = "bot_socket")]
    pub(crate) demo_socket: Option<PathBuf>,
    /// Task 5.6: the bot's control socket (`ddnet-ai play` serves `<data-dir>/bot/control.sock`, `docs/formats.md` §26),
    /// where the owner's commands from the site go. The web only connects to it. Default `<data-dir>/bot/control.sock`.
    #[arg(long = "control-socket")]
    pub(crate) control_socket: Option<PathBuf>,
    /// Task 5.6: the friend / war / ignore lists file the site edits. Default `<data-dir>/bot/relations.json`. Use the
    /// same file as the bot (`ddnet-ai play --relations`).
    #[arg(long = "relations")]
    pub(crate) relations: Option<PathBuf>,
    /// Task 5.8: the training runs the «Обучение» tab shows, read-only (`docs/formats.md` §29). Default `<data-dir>/runs`.
    #[arg(long = "runs-dir")]
    pub(crate) runs_dir: Option<PathBuf>,
    /// Task 5.10: the DDNet data directory (the `data/` of a DDNet 20.1 install or build: `game.png`, `skins/`, `mapres/`, ...)
    /// whose graphics the «Игра» tab draws with, served read-only at `/assets/...` (CC BY-SA 3.0, never copied into git, the page
    /// credits it). Default: `<data-dir>/../build/ddnet-20.1/build/data` when it exists. Without it tees are flat and external
    /// map images are missing.
    #[arg(long = "ddnet-data")]
    pub(crate) ddnet_data: Option<PathBuf>,
    /// Task 5.9 (D-089): the launcher's directory: the site writes `request.json` here (the status is read from
    /// `--launch-status-dir`); the root helper (`ddnet-ai launch apply`, run by a systemd path unit) consumes it. Default `<data-dir>/launch`.
    #[arg(long = "launch-dir")]
    pub(crate) launch_dir: Option<PathBuf>,
    /// Task 5.9: the root-owned directory the launcher helper writes `status.json` into (read-only for the site). Default `/run/ddnet-ai`.
    #[arg(long = "launch-status-dir")]
    pub(crate) launch_status_dir: Option<PathBuf>,
    /// Task 5.9: the owner's allow-list the launcher's server choice is built from (read-only). Default `<data-dir>/live-servers.toml`.
    #[arg(long = "live-servers")]
    pub(crate) live_servers: Option<PathBuf>,
    /// Task 5.9: the root-owned launcher config (read only to show the active fly bundle). Default `/etc/ddnet-ai/launch.toml`.
    #[arg(long = "launch-config")]
    pub(crate) launch_config: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct WebPasswdArgs {
    /// Also print the freshly generated plaintext password to stdout. Without this flag only the
    /// file paths that were written are printed — the password itself always goes to
    /// `web-password.txt` on disk, and is otherwise never logged.
    #[arg(long)]
    pub(crate) show: bool,
    /// Base data directory; secrets are written under `<data-dir>/secrets`. Defaults to
    /// `~/aiddnet/data`.
    #[arg(long)]
    pub(crate) data_dir: Option<PathBuf>,
}

/// Installs a `tracing` subscriber that writes to stdout *and* a daily-rotating file under
/// `<data_dir>/logs/web/`, so the audit log (`ddai_web::http::login`'s "login attempt" line —
/// acceptance criterion 3) actually reaches somewhere durable instead of being silently dropped.
///
/// Review finding F1 (blocker): before this, nothing in the binary ever installed a tracing
/// subscriber at all, so every `tracing::info!`/`warn!`/`error!` call in `ddai-web` — including
/// the audit log line itself — was a complete no-op. A reviewer's repro showed 0 "login attempt"
/// lines anywhere after 1.6M+ real login attempts.
///
/// Returns the file writer's `WorkerGuard`; the caller must keep it alive for as long as logging
/// should keep flushing (dropping it stops the background flush thread). `RUST_LOG` overrides the
/// default filter (`info`) in the usual `tracing-subscriber` `EnvFilter` syntax.
fn init_tracing(data_dir: &Path) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let stdout_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stdout);

    let log_dir = data_dir.join("logs").join("web");
    match std::fs::create_dir_all(&log_dir) {
        Ok(()) => {
            let file_appender = tracing_appender::rolling::daily(&log_dir, "web.log");
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            let file_layer = tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false);
            // `try_init` (not `init`): never panics if a subscriber is somehow already installed
            // (e.g. under a future test harness that sets one up itself) — logging is
            // best-effort, and failing to *start* the bot over it would be its own bug.
            let _ = tracing_subscriber::registry()
                .with(filter)
                .with(stdout_layer)
                .with(file_layer)
                .try_init();
            Some(guard)
        }
        Err(e) => {
            eprintln!(
                "warning: could not create log directory {} ({e}); logging to stdout only",
                log_dir.display()
            );
            let _ = tracing_subscriber::registry()
                .with(filter)
                .with(stdout_layer)
                .try_init();
            None
        }
    }
}

pub fn run_web(args: WebArgs) -> ExitCode {
    let data_dir = args.data_dir.unwrap_or_else(default_data_dir);
    // Kept alive for the rest of this function (which blocks until the server stops) so the
    // non-blocking file writer keeps flushing for as long as we're logging anything.
    let _tracing_guard = init_tracing(&data_dir);

    let mut config = ddai_web::WebConfig::new(args.listen, data_dir);
    config.trust_proxy = args.trust_proxy;
    config.i_know_this_is_public = args.i_know_this_is_public;
    config.cookie_secure = args.cookie_secure;
    config.replay_source = args.replay;
    config.demo_socket = args.demo_socket;
    config.map_search_dirs = args.maps_dir;
    if let Some(socket) = args.control_socket {
        config.control_socket = socket;
    }
    if let Some(path) = args.relations {
        config.relations_path = path;
    }
    if let Some(path) = args.runs_dir {
        config.runs_dir = path;
    }
    config.ddnet_data_dir = args.ddnet_data.or_else(|| {
        // The project layout keeps the DDNet build next to the data directory (`~/aiddnet/{data,build}`).
        let guess = config
            .data_dir
            .join("..")
            .join("build")
            .join("ddnet-20.1")
            .join("build")
            .join("data");
        guess.join("game.png").is_file().then_some(guess)
    });
    if let Some(path) = args.launch_dir {
        config.launch_dir = path;
    }
    if let Some(path) = args.launch_status_dir {
        config.status_dir = path;
    }
    if let Some(path) = args.live_servers {
        config.live_servers = path;
    }
    if let Some(path) = args.launch_config {
        config.launch_config = path;
    }
    if let Some(socket) = args.bot_socket {
        if config.map_search_dirs.is_empty() {
            config.map_search_dirs.push(config.data_dir.join("maps").join("cache"));
        }
        config.bot_socket = Some(socket);
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("failed to start the async runtime: {e}");
            return ExitCode::FAILURE;
        }
    };

    runtime.block_on(async move {
        let bound = match ddai_web::bind(config).await {
            Ok(bound) => bound,
            Err(e) => {
                eprintln!("failed to start the web server: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!("ddai-web listening on http://{} (loopback only)", bound.local_addr);
        match ddai_web::run(bound).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("web server stopped: {e}");
                ExitCode::FAILURE
            }
        }
    })
}

pub fn run_web_passwd(args: WebPasswdArgs) -> ExitCode {
    let data_dir = args.data_dir.unwrap_or_else(default_data_dir);
    let paths = ddai_web::secrets::SecretsPaths::new(&data_dir);
    let params = ddai_web::secrets::Argon2Params::default();
    match ddai_web::secrets::generate_and_store_password(&paths, params) {
        Ok(generated) => {
            println!("wrote {}", paths.auth_file().display());
            println!("wrote {}", paths.password_file().display());
            if args.show {
                println!("password: {}", generated.plaintext);
            } else {
                println!("(pass --show to also print the password here; it was written to the file above)");
            }
            // Review round 2, finding F3: the new password takes effect on the next login attempt
            // without a restart (it's read from disk on every POST /api/login), but a SESSION
            // created under the OLD password keeps working regardless — sessions aren't
            // re-checked against the current password hash on every request, only at login time.
            // If this password change is happening because an old session might be compromised
            // (not just routine rotation), only a restart actually invalidates it.
            println!(
                "note: existing logged-in sessions are NOT invalidated by a new password. If you \
                 suspect a session may be compromised, also run: sudo systemctl restart ddnet-ai-web"
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to generate the web password: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    /// A minimal raw HTTP POST, enough to drive a real login attempt through the real server —
    /// deliberately not a new HTTP-client dependency for one test (see `ddai-web`'s own test
    /// support module for the same reasoning at larger scale).
    fn post_login(addr: SocketAddr, password: &str) {
        let mut stream = TcpStream::connect(addr).expect("connect to the running server");
        let body = format!("{{\"password\":\"{password}\"}}");
        let request = format!(
            "POST /api/login HTTP/1.1\r\nHost: {addr}\r\nOrigin: http://{addr}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).expect("write request");
        let mut response = Vec::new();
        stream.read_to_end(&mut response).expect("read response");
    }

    /// Regression test for review finding F1 (blocker): the audit log requirement (acceptance
    /// criterion 3) was silently a no-op because nothing ever installed a tracing subscriber
    /// before starting the server. This drives real login attempts through the real HTTP+session
    /// stack (not just a direct call to the tracing macro) so it would fail if
    /// `ddai_web::http::login::login` ever stopped emitting the audit line, and checks the
    /// on-disk log file `init_tracing` sets up — not just an in-memory capture — since "reaches a
    /// durable log file" is the actual requirement.
    ///
    /// Also covers the log-dir-creation-fails fallback (`init_tracing` should still return, with
    /// no file guard, rather than panicking) as a second act of the *same* test rather than a
    /// separate `#[test]`: `tracing_subscriber`'s global subscriber can only ever be installed
    /// once per process, so whichever of two independent tests called `init_tracing` (and
    /// therefore `try_init`) first would silently "win" the slot — nondeterministic under
    /// `cargo test`'s default parallel execution. Running the real, log-emitting scenario first
    /// (the only one that actually depends on its subscriber winning) and the fallback check
    /// second (which only asserts on `init_tracing`'s return value, so it doesn't care whether
    /// its own `try_init` call actually took effect) makes this deterministic.
    #[test]
    fn run_web_audit_log_reaches_the_log_file_and_never_contains_the_password() {
        let tempdir = tempfile::tempdir().expect("tempdir");

        let guard = init_tracing(tempdir.path());
        assert!(
            guard.is_some(),
            "log directory creation should succeed in a fresh tempdir"
        );

        let secret_password = "hunter2-definitely-not-logged";
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let paths = ddai_web::secrets::SecretsPaths::new(tempdir.path());
            let params = ddai_web::secrets::Argon2Params {
                m_cost_kib: 8 * 1024,
                t_cost: 1,
                p_cost: 1,
            };
            // Store a *known* password (rather than using the randomly generated one) so we can
            // assert on its exact text never appearing in the log below.
            let hash = ddai_web::secrets::hash_password(secret_password, params).expect("hash");
            ddai_web::secrets::save_password_auth(&paths, &ddai_web::secrets::PasswordAuth { hash_phc: hash, params })
                .expect("save password auth");

            let config = ddai_web::WebConfig::new("127.0.0.1:0".parse().unwrap(), tempdir.path().to_path_buf());
            let bound = ddai_web::bind(config).await.expect("bind");
            let addr = bound.local_addr;
            let server = tokio::spawn(ddai_web::run(bound));

            post_login(addr, secret_password); // success = true
            post_login(addr, "definitely the wrong password"); // success = false

            server.abort();
        });

        // The non-blocking file writer flushes on its own interval; dropping the guard flushes
        // immediately and stops the background thread.
        std::thread::sleep(std::time::Duration::from_millis(150));
        drop(guard);
        std::thread::sleep(std::time::Duration::from_millis(50));

        let log_dir = tempdir.path().join("logs").join("web");
        let mut contents = String::new();
        for entry in std::fs::read_dir(&log_dir).expect("read log dir") {
            let path = entry.expect("dir entry").path();
            contents.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
        }

        let attempt_lines: Vec<&str> = contents.lines().filter(|line| line.contains("login attempt")).collect();
        assert_eq!(attempt_lines.len(), 2, "expected exactly 2 audit lines in:\n{contents}");
        assert!(
            attempt_lines.iter().any(|line| line.contains("success=true")),
            "missing a successful-login audit line in:\n{contents}"
        );
        assert!(
            attempt_lines.iter().any(|line| line.contains("success=false")),
            "missing a failed-login audit line in:\n{contents}"
        );
        assert!(
            !contents.contains(secret_password),
            "the password must never appear in the log file"
        );

        // Second act (see doc comment): the fallback when the log directory can't be created at
        // all. A *different* tempdir, so it can't interfere with the log file already asserted on
        // above.
        let other_tempdir = tempfile::tempdir().expect("tempdir");
        std::fs::write(other_tempdir.path().join("logs"), b"not a directory").expect("write blocker file");
        let fallback_guard = init_tracing(other_tempdir.path());
        assert!(
            fallback_guard.is_none(),
            "should report no file guard when the log directory can't be created"
        );
    }
}
