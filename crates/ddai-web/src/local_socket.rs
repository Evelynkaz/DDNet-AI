//! The tokio Unix-domain stream the web unit uses to reach the bot (the live bridge and the control socket), where the operating system
//! has one.
//!
//! On Unix these are tokio's own types. On Windows there is no Unix-domain socket in `std`/tokio, so the same names stand for types that
//! nothing can construct: [`UnixStream::connect`] fails with [`std::io::ErrorKind::Unsupported`] and the page shows "the bot is not
//! running", exactly as when no socket file exists. How a Windows bot talks to its web page is task 5.5b's to decide.

#[cfg(unix)]
pub use tokio::net::UnixStream;
#[cfg(unix)]
pub use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

#[cfg(not(unix))]
pub use unsupported::{OwnedReadHalf, OwnedWriteHalf, UnixStream};

#[cfg(not(unix))]
mod unsupported {
    use std::io;
    use std::path::Path;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

    fn unsupported() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "Unix-domain sockets are not available on this platform (the bot's live bridge and control socket are Linux features for now)",
        )
    }

    /// A connected stream; cannot be made on this platform (the private field keeps anyone from building one).
    #[derive(Debug)]
    pub struct UnixStream(());
    /// Its read half.
    #[derive(Debug)]
    pub struct OwnedReadHalf(());
    /// Its write half.
    #[derive(Debug)]
    pub struct OwnedWriteHalf(());

    impl UnixStream {
        pub async fn connect<P: AsRef<Path>>(_path: P) -> io::Result<UnixStream> {
            Err(unsupported())
        }

        pub fn into_split(self) -> (OwnedReadHalf, OwnedWriteHalf) {
            (OwnedReadHalf(()), OwnedWriteHalf(()))
        }
    }

    impl AsyncRead for UnixStream {
        fn poll_read(self: Pin<&mut Self>, _cx: &mut Context<'_>, _buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Err(unsupported()))
        }
    }

    impl AsyncWrite for UnixStream {
        fn poll_write(self: Pin<&mut Self>, _cx: &mut Context<'_>, _buf: &[u8]) -> Poll<io::Result<usize>> {
            Poll::Ready(Err(unsupported()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Err(unsupported()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Err(unsupported()))
        }
    }

    impl AsyncRead for OwnedReadHalf {
        fn poll_read(self: Pin<&mut Self>, _cx: &mut Context<'_>, _buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Err(unsupported()))
        }
    }

    impl AsyncWrite for OwnedWriteHalf {
        fn poll_write(self: Pin<&mut Self>, _cx: &mut Context<'_>, _buf: &[u8]) -> Poll<io::Result<usize>> {
            Poll::Ready(Err(unsupported()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Err(unsupported()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Err(unsupported()))
        }
    }
}
