//! Task 4.9 (D-094): with the owner chat on, `ControlServer::start` claims the process's one `OwnerChannel` (the dispatcher holds it,
//! so nobody else can get it). Its own test binary (= its own process).

// The control server listens on a Unix-domain socket; on Windows there is none (`ddai_os::ipc`, D-127), so there is nothing to claim the channel.
#![cfg(unix)]

use std::sync::Arc;

use ddai_bot::command::CommandBus;
use ddai_bot::control::{AuditSink, ControlServer, MemoryAudit};
use ddai_net::owner_chat::OwnerChannel;

#[test]
fn starting_the_control_server_claims_the_channel() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bot").join("control.sock");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let (sender, _inbox) = CommandBus::open();
    let _server =
        ControlServer::start(&path, sender, Arc::new(MemoryAudit::default()) as Arc<dyn AuditSink>).expect("socket");
    assert!(OwnerChannel::claim().is_none(), "the dispatcher holds the one channel");
}
