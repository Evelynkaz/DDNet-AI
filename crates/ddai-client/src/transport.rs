//! The datagram path between the driver and the game server (task 2.6).
//!
//! Until this task the driver used a connected `UdpSocket` directly. [`Transport`] is the minimal seam
//! that lets a SOCKS5 UDP association ([`crate::socks5::Socks5UdpTransport`]) take the socket's place;
//! [`DirectUdp`] is the old behaviour, moved here unchanged (one socket bound once, `connect`ed to each
//! attempt's target, stale datagrams drained when an attempt starts, `recv` blocking for one poll
//! interval).

use crate::socks5::Socks5Error;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

/// Receive buffer size used by [`DirectUdp::begin_attempt`]'s drain — comfortably above
/// `ddai_net::packet::MAX_PACKET_SIZE` (1400).
const DRAIN_BUF_SIZE: usize = 2048;

/// Why a transport could not be made ready for a connection attempt.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// The local socket could not be pointed at the target (the direct path's old `GaveUp(LocalError)`).
    #[error("failed to connect the socket to {target}: {source}")]
    Connect { target: SocketAddr, source: io::Error },
    /// The SOCKS5 proxy refused or failed (see [`Socks5Error::is_fatal`] for which of those stop retrying).
    #[error(transparent)]
    Proxy(#[from] Socks5Error),
}

impl TransportError {
    /// A fatal error ends the whole [`crate::Client`] (no retry, no backoff): the direct path's socket
    /// failure as before, and a proxy answer that retrying cannot change (wrong password, reply 0x07 "UDP
    /// not supported", ...). Anything else is an ordinary lost connection.
    pub fn is_fatal(&self) -> bool {
        match self {
            TransportError::Connect { .. } => true,
            TransportError::Proxy(e) => e.is_fatal(),
        }
    }
}

/// What the driver needs from the datagram path. One implementation per way of reaching a server.
pub trait Transport {
    /// Called before every connection attempt (the first one, reconnects, redirects) with that attempt's
    /// target. Makes the transport ready to carry datagrams to `target` and discards anything left over
    /// from an earlier attempt.
    fn begin_attempt(&mut self, target: SocketAddr) -> Result<(), TransportError>;

    /// Called when an attempt ended in a *loss* (timeout, socket error, a peer close that is retried), before
    /// the next `begin_attempt`: a transport with connection-like state it cannot vouch for any more (a SOCKS5
    /// association) starts over. Not called for a server-requested reconnect or a redirect, which keep the
    /// same local port on the direct path and the same association on the proxied one. Default: nothing.
    fn reset_after_loss(&mut self) {}

    /// Sends one datagram to the current target. Best-effort like UDP itself.
    fn send(&mut self, datagram: &[u8]) -> io::Result<()>;

    /// Receives one datagram from the current target into `buf`, blocking for at most the transport's poll
    /// interval. `WouldBlock`/`TimedOut` mean "nothing now" (a datagram the transport drops as foreign or
    /// malformed is also reported that way). Any other error ends the connection attempt; ICMP-style
    /// `ConnectionRefused`/`ConnectionReset` are tolerated by the caller.
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

/// The plain UDP path: one socket for the whole [`crate::Client::connect`] call (D-050: reconnects keep the
/// local port, like the real client), `connect`ed to the current target.
#[derive(Debug)]
pub struct DirectUdp {
    socket: UdpSocket,
}

impl DirectUdp {
    /// Binds `0.0.0.0:0` with a read timeout of `poll`.
    pub fn bind(poll: Duration) -> Result<Self, String> {
        let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| format!("failed to bind a local socket: {e}"))?;
        socket
            .set_read_timeout(Some(poll))
            .map_err(|e| format!("failed to configure the socket: {e}"))?;
        Ok(DirectUdp { socket })
    }

    pub fn local_port(&self) -> Option<u16> {
        self.socket.local_addr().ok().map(|a| a.port())
    }

    /// Wraps an existing socket (tests).
    #[cfg(test)]
    pub(crate) fn from_socket(socket: UdpSocket) -> Self {
        DirectUdp { socket }
    }

    /// Discards anything still queued on the (reused) socket from the previous connection, so a stale
    /// `CLOSE`/`CONNECTACCEPT` from the old connection can never be mistaken for a reply to the new
    /// `CONNECT` (a `Connecting` connection has no token to check yet). Beyond the real client, which
    /// does not do this; strictly safer.
    fn drain_stale(&self) {
        if self.socket.set_nonblocking(true).is_err() {
            return;
        }
        let mut buf = [0u8; DRAIN_BUF_SIZE];
        let mut drained = 0u32;
        while self.socket.recv(&mut buf).is_ok() {
            drained += 1;
        }
        let _ = self.socket.set_nonblocking(false);
        if drained > 0 {
            tracing::info!(
                drained,
                "driver: discarded stale datagrams from the previous connection"
            );
        }
    }
}

impl Transport for DirectUdp {
    fn begin_attempt(&mut self, target: SocketAddr) -> Result<(), TransportError> {
        // Re-associate the one long-lived socket with this attempt's target (a no-op for a plain
        // reconnect; a new port for a redirect).
        self.socket
            .connect(target)
            .map_err(|source| TransportError::Connect { target, source })?;
        self.drain_stale();
        Ok(())
    }

    fn send(&mut self, datagram: &[u8]) -> io::Result<()> {
        self.socket.send(datagram).map(|_| ())
    }

    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.socket.recv(buf)
    }
}
