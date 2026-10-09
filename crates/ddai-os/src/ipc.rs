//! Unix-domain sockets, where the operating system has them.
//!
//! On Unix these are `std`'s own types. On Windows `std` has none (and a local socket there would need an authentication story that
//! the file-permission trick of the VPS deployment does not give), so the same names stand for types nothing can construct: `bind` and
//! `connect` fail with [`std::io::ErrorKind::Unsupported`], and so does every other method. Code written against this module compiles
//! everywhere, and on Windows the bot's local control channel and the live bridge to the web unit are simply "not available" (task 5.5b
//! decides how a Windows bot talks to its web page). Check [`SUPPORTED`] to skip the attempt.

/// Whether this platform has Unix-domain sockets.
pub const SUPPORTED: bool = cfg!(unix);

#[cfg(unix)]
pub use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};

#[cfg(not(unix))]
pub use unsupported::{SocketAddr, UnixListener, UnixStream};

#[cfg(not(unix))]
mod unsupported {
    use std::io::{self, Read, Write};
    use std::net::Shutdown;
    use std::path::Path;
    use std::time::Duration;

    fn unsupported<T>() -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Unix-domain sockets are not available on this platform (the bot's control channel and live bridge are Linux features for now)",
        ))
    }

    /// The address of a peer; never produced here.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SocketAddr(());

    /// A listening socket; cannot be made on this platform (the private field keeps anyone from building one), and every method says so.
    #[derive(Debug)]
    pub struct UnixListener(());

    /// A connected socket; cannot be made on this platform, and every method says so.
    #[derive(Debug)]
    pub struct UnixStream(());

    impl UnixListener {
        pub fn bind<P: AsRef<Path>>(_path: P) -> io::Result<UnixListener> {
            unsupported()
        }

        pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
            unsupported()
        }

        pub fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> {
            unsupported()
        }
    }

    impl UnixStream {
        pub fn connect<P: AsRef<Path>>(_path: P) -> io::Result<UnixStream> {
            unsupported()
        }

        pub fn try_clone(&self) -> io::Result<UnixStream> {
            unsupported()
        }

        pub fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> {
            unsupported()
        }

        pub fn set_read_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
            unsupported()
        }

        pub fn set_write_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
            unsupported()
        }

        pub fn shutdown(&self, _how: Shutdown) -> io::Result<()> {
            unsupported()
        }
    }

    impl Read for UnixStream {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            unsupported()
        }
    }

    impl Read for &UnixStream {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            unsupported()
        }
    }

    impl Write for UnixStream {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            unsupported()
        }

        fn flush(&mut self) -> io::Result<()> {
            unsupported()
        }
    }

    impl Write for &UnixStream {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            unsupported()
        }

        fn flush(&mut self) -> io::Result<()> {
            unsupported()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn unix_sockets_are_the_std_ones() {
        assert!(std::hint::black_box(SUPPORTED));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s");
        let listener = UnixListener::bind(&path).unwrap();
        let client = UnixStream::connect(&path).unwrap();
        let (_server, _addr) = listener.accept().unwrap();
        drop(client);
    }

    #[cfg(not(unix))]
    #[test]
    fn elsewhere_nothing_can_be_made() {
        assert!(!std::hint::black_box(SUPPORTED));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s");
        assert_eq!(
            UnixListener::bind(&path).unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
        assert_eq!(
            UnixStream::connect(&path).unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
    }
}
