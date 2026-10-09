//! Task 4.9 (D-094, review F6): with the owner chat switched off, `ControlServer::start_with_owner_chat(.., false)` burns the process's
//! one `OwnerChannel`: a later `claim()` gets `None`. Its own test binary (= its own process): nothing else may have claimed first.

// The control server listens on a Unix-domain socket; on Windows there is none (`ddai_os::ipc`, D-127), so there is nothing to claim the channel.
#![cfg(unix)]

use std::sync::Arc;

use ddai_bot::command::CommandBus;
use ddai_bot::control::{AuditSink, ControlServer, MemoryAudit};
use ddai_net::owner_chat::OwnerChannel;

#[test]
fn switching_the_chat_off_burns_the_channel() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bot").join("control.sock");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let (sender, _inbox) = CommandBus::open();
    let _server = ControlServer::start_with_owner_chat(
        &path,
        sender,
        Arc::new(MemoryAudit::default()) as Arc<dyn AuditSink>,
        false,
    )
    .expect("the control socket");
    assert!(
        OwnerChannel::claim().is_none(),
        "the channel was burnt at startup: nobody can claim it now"
    );
}
