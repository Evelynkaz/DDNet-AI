//! A client of the bot's control socket (`docs/formats.md` §26): sends one typed [`ControlCommand`] and returns the bot's
//! [`ControlReply`]. One connection per request; every step is bounded in time and the reply in size, so a stuck or
//! hostile peer on that socket (it is a `0600` file of the same user, but still) cannot hang a request or fill memory.
//!
//! The web has no way to ask for anything outside [`ControlCommand`]: that enum is the whole vocabulary, it has no chat
//! and no connect/allowlist/`ready` verb, and this module only ever *connects* to the socket path from the config.

use std::path::{Path, PathBuf};
use std::time::Duration;

use ddai_botctl::proto::{ControlCommand, ControlReply, ControlRequest, MAX_REPLY_BYTES, VERSION};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::Semaphore;
use tokio::time::timeout;

/// Connecting to a local socket is instant; a longer wait means nobody serves it.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// The bot answers within 5 s (its own limit); a little more here.
pub const REPLY_TIMEOUT: Duration = Duration::from_secs(7);
/// Requests in flight at once from the web (the bot serves 4 connections).
pub const MAX_IN_FLIGHT: usize = 2;

/// Why no answer from the bot came back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SendError {
    /// Nobody is listening on the control socket (the bot is not running, or runs without control).
    #[error("the bot's control socket is not available")]
    Unavailable,
    /// The bot did not answer in time.
    #[error("the bot did not answer in time")]
    Timeout,
    /// What came back is not a reply of this protocol version (too long, not JSON, wrong version, closed early).
    #[error("the bot's reply is not understood")]
    Protocol,
}

/// The control-socket client.
pub struct ControlClient {
    path: PathBuf,
    slots: Semaphore,
    connect_timeout: Duration,
    reply_timeout: Duration,
}

impl ControlClient {
    pub fn new(path: PathBuf) -> ControlClient {
        ControlClient::with_timeouts(path, CONNECT_TIMEOUT, REPLY_TIMEOUT)
    }

    pub fn with_timeouts(path: PathBuf, connect_timeout: Duration, reply_timeout: Duration) -> ControlClient {
        ControlClient {
            path,
            slots: Semaphore::new(MAX_IN_FLIGHT),
            connect_timeout,
            reply_timeout,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the socket file exists right now (a cheap hint for the status panel; sending is the real test).
    pub async fn socket_present(&self) -> bool {
        tokio::fs::symlink_metadata(&self.path).await.is_ok()
    }

    /// Sends `cmd` on behalf of the web session tagged `session_tag` (`crate::auth::cookie::audit_tag`).
    pub async fn send(&self, session_tag: &str, cmd: ControlCommand) -> Result<ControlReply, SendError> {
        let _permit = self.slots.acquire().await.map_err(|_| SendError::Unavailable)?;
        let stream = match timeout(self.connect_timeout, UnixStream::connect(&self.path)).await {
            Ok(Ok(s)) => s,
            Ok(Err(_)) => return Err(SendError::Unavailable),
            Err(_) => return Err(SendError::Timeout),
        };
        let (read_half, mut write_half) = stream.into_split();
        let mut line = serde_json::to_vec(&ControlRequest::new(session_tag, cmd)).map_err(|_| SendError::Protocol)?;
        line.push(b'\n');
        match timeout(self.connect_timeout, write_half.write_all(&line)).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => return Err(SendError::Unavailable),
            Err(_) => return Err(SendError::Timeout),
        }
        // Never read more than the protocol allows, whatever the peer sends.
        let mut reader = BufReader::new(read_half.take(MAX_REPLY_BYTES as u64 + 1));
        let mut reply = Vec::with_capacity(256);
        match timeout(self.reply_timeout, reader.read_until(b'\n', &mut reply)).await {
            Ok(Ok(0)) => return Err(SendError::Protocol),
            Ok(Ok(_)) => {}
            Ok(Err(_)) => return Err(SendError::Protocol),
            Err(_) => return Err(SendError::Timeout),
        }
        if reply.last() != Some(&b'\n') || reply.len() > MAX_REPLY_BYTES {
            return Err(SendError::Protocol);
        }
        reply.pop();
        let reply: ControlReply = serde_json::from_slice(&reply).map_err(|_| SendError::Protocol)?;
        if reply.v != VERSION {
            return Err(SendError::Protocol);
        }
        Ok(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_botctl::proto::{ModeArg, ReplyCode};
    use tokio::net::UnixListener;

    fn sock() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control.sock");
        (dir, path)
    }

    /// A scripted bot: reads one request line, runs `respond` on it and writes whatever it returns.
    fn serve(path: &Path, respond: impl Fn(String) -> Vec<u8> + Send + 'static) -> tokio::task::JoinHandle<()> {
        let listener = UnixListener::bind(path).unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let (r, mut w) = stream.into_split();
                let mut line = String::new();
                if BufReader::new(r).read_line(&mut line).await.unwrap_or(0) == 0 {
                    continue;
                }
                let out = respond(line);
                let _ = w.write_all(&out).await;
            }
        })
    }

    #[tokio::test]
    async fn a_command_goes_out_as_the_documented_line_and_the_reply_comes_back() {
        let (_d, path) = sock();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let seen2 = seen.clone();
        let _server = serve(&path, move |line| {
            *seen2.lock().unwrap() = line;
            b"{\"v\":1,\"ok\":true,\"text\":\"mode: hold\"}\n".to_vec()
        });
        let client = ControlClient::new(path);
        let reply = client
            .send("00ff00ff00ff00ff", ControlCommand::Mode { mode: ModeArg::Hold })
            .await
            .unwrap();
        assert!(reply.ok && reply.text == "mode: hold" && reply.code.is_none());
        assert_eq!(
            seen.lock().unwrap().as_str(),
            "{\"v\":1,\"session\":\"00ff00ff00ff00ff\",\"cmd\":{\"type\":\"mode\",\"mode\":\"hold\"}}\n"
        );
    }

    #[tokio::test]
    async fn a_refusal_code_comes_back_as_a_reply_not_an_error() {
        let (_d, path) = sock();
        let _server = serve(&path, |_| {
            b"{\"v\":1,\"ok\":false,\"text\":\"slow\",\"code\":\"rate_limited\"}\n".to_vec()
        });
        let reply = ControlClient::new(path)
            .send("ab", ControlCommand::Go {})
            .await
            .unwrap();
        assert_eq!((reply.ok, reply.code), (false, Some(ReplyCode::RateLimited)));
    }

    #[tokio::test]
    async fn no_socket_means_unavailable() {
        let (_d, path) = sock();
        let client = ControlClient::new(path.clone());
        assert!(!client.socket_present().await);
        assert_eq!(
            client.send("ab", ControlCommand::Go {}).await,
            Err(SendError::Unavailable)
        );
        // A regular file in its place is no socket either.
        std::fs::write(&path, b"x").unwrap();
        assert_eq!(
            client.send("ab", ControlCommand::Go {}).await,
            Err(SendError::Unavailable)
        );
    }

    #[tokio::test]
    async fn hostile_or_broken_replies_are_a_protocol_error_never_a_panic_or_a_hang() {
        let (_d, path) = sock();
        let client = ControlClient::new(path.clone());
        let bad: Vec<Vec<u8>> = vec![
            b"not json\n".to_vec(),
            b"{\"v\":2,\"ok\":true,\"text\":\"\"}\n".to_vec(),
            b"{\"v\":1,\"ok\":true,\"text\":\"\",\"extra\":1}\n".to_vec(),
            b"{\"v\":1,\"ok\":true,\"text\":\"no newline\"}".to_vec(),
            Vec::new(),
            vec![b'x'; MAX_REPLY_BYTES * 4], // endless junk without a newline
            {
                let mut v = vec![b'x'; MAX_REPLY_BYTES + 10];
                v.push(b'\n');
                v
            },
        ];
        for (i, reply) in bad.into_iter().enumerate() {
            let server = serve(&path, move |_| reply.clone());
            let r = client.send("ab", ControlCommand::Go {}).await;
            assert_eq!(r, Err(SendError::Protocol), "case {i}");
            server.abort();
            let _ = std::fs::remove_file(&path);
        }
    }

    #[tokio::test]
    async fn a_bot_that_never_answers_times_out() {
        let (_d, path) = sock();
        let listener = UnixListener::bind(&path).unwrap();
        let held = tokio::spawn(async move {
            let (s, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(30)).await;
            drop(s);
        });
        let client = ControlClient::with_timeouts(path, Duration::from_secs(1), Duration::from_millis(150));
        let started = std::time::Instant::now();
        assert_eq!(client.send("ab", ControlCommand::Go {}).await, Err(SendError::Timeout));
        assert!(started.elapsed() < Duration::from_secs(20));
        held.abort();
    }

    #[tokio::test]
    async fn no_more_than_two_requests_are_in_flight_at_once() {
        let (_d, path) = sock();
        let listener = UnixListener::bind(&path).unwrap();
        let active = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (a2, p2) = (active.clone(), peak.clone());
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let (a, p) = (a2.clone(), p2.clone());
                tokio::spawn(async move {
                    let now = a.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    p.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                    let (r, mut w) = stream.into_split();
                    let mut line = String::new();
                    let _ = BufReader::new(r).read_line(&mut line).await;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    let _ = w.write_all(b"{\"v\":1,\"ok\":true,\"text\":\"\"}\n").await;
                    a.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                });
            }
        });
        let client = std::sync::Arc::new(ControlClient::new(path));
        let tasks: Vec<_> = (0..6)
            .map(|_| {
                let c = client.clone();
                tokio::spawn(async move { c.send("ab", ControlCommand::Go {}).await })
            })
            .collect();
        for t in tasks {
            assert!(t.await.unwrap().unwrap().ok);
        }
        assert!(peak.load(std::sync::atomic::Ordering::SeqCst) <= MAX_IN_FLIGHT);
    }
}
