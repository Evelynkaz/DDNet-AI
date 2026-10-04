//! Review round 2, findings F1/F10: an automated test proving the D-027/D-038 live-servers safety
//! switch (`crate::live_servers::check`) actually refuses a real, non-loopback address when
//! exercised through the real, public [`Client::connect`] entry point — the same one `ddnet-ai
//! play`/`record` actually call — not just unit-tested in isolation
//! (`live_servers::tests::an_empty_list_refuses_every_non_loopback_address` and friends, which
//! call `check` directly). Round 2's reviewer confirmed the check itself runs before every
//! bind/connect (tested independently in a netns probe) but noted this crate had no *automated*
//! (CI-runnable, no root/netns needed) test proving it end to end.
//!
//! Uses a TEST-NET-3 address (`203.0.113.0/24`, RFC 5737) — reserved for documentation/testing,
//! never assigned to a real host on the real internet — so this test can never accidentally reach,
//! or hang waiting on, anything real on the network regardless of the machine it runs on. The
//! `live_servers` list passed in is an explicit, empty `LiveServers::default()` (never the real
//! `~/aiddnet/data/live-servers.toml`), so this test is fully hermetic: it can never pass or fail
//! because of what the owner's own file happens to contain.

use ddai_client::live_servers::LiveServers;
use ddai_client::{Client, ClientConfig, ClientEvent, GaveUpCategory};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

#[test]
fn connect_to_an_unlisted_test_net_address_is_refused_before_ever_touching_the_network() {
    let test_net_target: SocketAddr = "203.0.113.7:8303".parse().unwrap();
    let config = ClientConfig {
        live_servers: LiveServers::default(), // explicit empty list — refuses every non-loopback address.
        ..ClientConfig::default()
    };
    let mut client = Client::connect(test_net_target, config);

    // A real attempt to reach an unreachable/non-existent address would not itself block (UDP's
    // `connect`/`send` never wait for reachability), so a bounded deadline here is not "waiting out
    // a timeout" — it is simply generous enough that a flaky/slow CI runner cannot turn a genuine,
    // near-instant refusal into a false failure, while still being short enough that if the driver
    // ever *did* start a real join sequence first (the regression this test guards against), this
    // test would reliably notice by timing out at the end rather than hanging forever.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut gave_up: Option<(String, GaveUpCategory)> = None;
    let mut saw_anything_else = Vec::new();
    while Instant::now() < deadline && gave_up.is_none() {
        match client.recv_event(Duration::from_millis(100)) {
            Some(ClientEvent::GaveUp { reason, category }) => gave_up = Some((reason, category)),
            Some(other) => saw_anything_else.push(format!("{other:?}")),
            None => {}
        }
    }

    let (reason, category) = gave_up.unwrap_or_else(|| {
        panic!(
            "expected a ClientEvent::GaveUp within 5s for an unlisted TEST-NET address — if this \
             times out instead, the safety switch did not refuse before attempting to connect. \
             Other events seen meanwhile: {saw_anything_else:?}"
        )
    });
    assert_eq!(
        category,
        GaveUpCategory::LocalError,
        "a live-servers refusal must be categorized as a local error, not a protocol/peer one — got reason: {reason}"
    );
    assert!(
        reason.contains("live-servers"),
        "GaveUp reason should name the live-servers safety switch, got: {reason:?}"
    );
    // The one and only event before GaveUp must never be a real join-sequence event (proves the
    // check ran *before* any socket/session activity, not merely that some later step also failed).
    assert!(
        saw_anything_else.is_empty(),
        "no event should have been observed before the refusal, got: {saw_anything_else:?}"
    );

    client.disconnect();
    client.join();
}

/// Same address, but this time it *is* listed — under a different nick than requested — proving
/// the switch distinguishes "not listed at all" from "listed, wrong nick" (D-027/D-038: one nick
/// pinned per server, not "any nick on an allowed address") even through the real `Client::connect`
/// path, not just `live_servers::check` in isolation.
#[test]
fn connect_to_a_test_net_address_listed_under_a_different_nick_is_also_refused() {
    let test_net_target: SocketAddr = "203.0.113.8:8303".parse().unwrap();
    let config = ClientConfig {
        name: "Impostor".to_string(),
        live_servers: LiveServers {
            servers: vec![ddai_client::live_servers::LiveServerEntry {
                address: test_net_target.to_string(),
                nick: "Muha".to_string(),
                purpose: "test".to_string(),
                ready: true,
                proxy: None,
            }],
        },
        ..ClientConfig::default()
    };
    let mut client = Client::connect(test_net_target, config);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut gave_up: Option<(String, GaveUpCategory)> = None;
    while Instant::now() < deadline && gave_up.is_none() {
        if let Some(ClientEvent::GaveUp { reason, category }) = client.recv_event(Duration::from_millis(100)) {
            gave_up = Some((reason, category));
        }
    }
    let (reason, category) =
        gave_up.expect("expected a ClientEvent::GaveUp within 5s for a wrong-nick TEST-NET address");
    assert_eq!(category, GaveUpCategory::LocalError);
    assert!(
        reason.contains("live-servers"),
        "GaveUp reason should name the live-servers safety switch, got: {reason:?}"
    );

    client.disconnect();
    client.join();
}

/// Review F1 (task 4.3): a listed entry that the owner has not marked `ready = true` is refused by the client's
/// own gate too — with the right nick, at a documentation-range (TEST-NET-3) address, so nothing real is
/// ever contacted: the driver gives up before opening a socket.
#[test]
fn connect_to_a_listed_but_not_ready_test_net_address_is_refused_with_the_right_nick() {
    let target: SocketAddr = "203.0.113.9:8303".parse().unwrap();
    let config = ClientConfig {
        name: "Muha".to_string(),
        live_servers: LiveServers {
            servers: vec![ddai_client::live_servers::LiveServerEntry {
                address: target.to_string(),
                nick: "Muha".to_string(),
                purpose: "test".to_string(),
                ready: false,
                proxy: None,
            }],
        },
        ..ClientConfig::default()
    };
    let mut client = Client::connect(target, config);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut gave_up: Option<(String, GaveUpCategory)> = None;
    while Instant::now() < deadline && gave_up.is_none() {
        if let Some(ClientEvent::GaveUp { reason, category }) = client.recv_event(Duration::from_millis(100)) {
            gave_up = Some((reason, category));
        }
    }
    let (reason, category) = gave_up.expect("the gate must refuse");
    assert!(reason.contains("ready = true"), "{reason}");
    assert!(matches!(category, GaveUpCategory::LocalError), "{category:?}");
    client.disconnect();
    client.join();
}
