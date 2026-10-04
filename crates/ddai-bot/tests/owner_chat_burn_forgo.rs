//! Task 4.9 (D-094, review F6): `--no-control` calls `forgo_owner_chat()`, which burns the process's one `OwnerChannel`.
//! Its own test binary (= its own process): nothing else may have claimed first.

use ddai_net::owner_chat::OwnerChannel;

#[test]
fn forgoing_the_owner_chat_burns_the_channel() {
    ddai_bot::control::forgo_owner_chat();
    assert!(OwnerChannel::claim().is_none());
    ddai_bot::control::forgo_owner_chat(); // idempotent
    assert!(OwnerChannel::claim().is_none());
}
