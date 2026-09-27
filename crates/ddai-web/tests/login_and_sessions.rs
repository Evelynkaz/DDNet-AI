//! Login/logout/session integration tests over real HTTP against a real server bound to an
//! ephemeral loopback port (acceptance criteria 2, 3, 4, 8).

mod support;

use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use support::{Req, TestServer, extract_set_cookie_pair, send};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn login_with_correct_password_succeeds_and_sets_cookie() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(response.status, 200, "{response:?}");
    let body = response.json();
    assert_eq!(body["ok"], true);
    assert!(!body["csrf_token"].as_str().unwrap_or("").is_empty());
    assert!(extract_set_cookie_pair(&response, server.cookie_name()).is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn login_with_wrong_password_fails_with_generic_401() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": "definitely not it" })),
    );
    assert_eq!(response.status, 401);
    assert_eq!(response.json()["error"], "invalid_credentials");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn login_accepts_form_encoded_body() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .form_body(&[("password", server.password.as_str())]),
    );
    assert_eq!(response.status, 200, "{response:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn login_with_unrecognized_content_type_is_bad_request() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .body("text/plain", server.password.clone().into_bytes()),
    );
    assert_eq!(response.status, 400);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_origin_login_post_is_rejected() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", "http://evil.example")
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(response.status, 403);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_origin_on_login_post_is_rejected() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/login").json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(response.status, 403);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sec_fetch_site_same_origin_is_accepted_without_origin_header() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Sec-Fetch-Site", "same-origin")
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(response.status, 200, "{response:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn me_reports_unauthenticated_without_a_cookie() {
    let server = TestServer::start().await;
    let response = send(server.addr, Req::new("GET", "/api/me"));
    assert_eq!(response.status, 200);
    assert_eq!(response.json()["authenticated"], false);
    assert!(response.json()["csrf_token"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn me_reports_authenticated_with_a_valid_cookie() {
    let server = TestServer::start().await;
    let (cookie, csrf) = server.login();
    let response = send(server.addr, Req::new("GET", "/api/me").cookie(&cookie));
    assert_eq!(response.json()["authenticated"], true);
    assert_eq!(response.json()["csrf_token"], csrf);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logout_without_a_session_cookie_is_unauthenticated() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("POST", "/api/logout").header("Origin", &server.origin()),
    );
    assert_eq!(response.status, 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logout_without_csrf_header_is_forbidden() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let response = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .header("Origin", &server.origin())
            .cookie(&cookie),
    );
    assert_eq!(response.status, 403);
    assert_eq!(response.json()["error"], "missing_csrf");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logout_with_wrong_csrf_header_is_forbidden() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let response = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", "not-the-real-token")
            .cookie(&cookie),
    );
    assert_eq!(response.status, 403);
    assert_eq!(response.json()["error"], "bad_csrf");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_origin_logout_is_forbidden_even_with_valid_session_and_csrf() {
    let server = TestServer::start().await;
    let (cookie, csrf) = server.login();
    let response = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .header("Origin", "http://evil.example")
            .header("X-CSRF-Token", &csrf)
            .cookie(&cookie),
    );
    assert_eq!(response.status, 403);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logout_with_correct_csrf_invalidates_the_session() {
    let server = TestServer::start().await;
    let (cookie, csrf) = server.login();

    let logout = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &csrf)
            .cookie(&cookie),
    );
    assert_eq!(logout.status, 200);
    assert_eq!(logout.json()["ok"], true);

    // The old cookie must now be rejected everywhere.
    let me_after = send(server.addr, Req::new("GET", "/api/me").cookie(&cookie));
    assert_eq!(me_after.json()["authenticated"], false);

    let logout_again = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &csrf)
            .cookie(&cookie),
    );
    assert_eq!(logout_again.status, 401, "logging out twice must not succeed twice");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cookie_has_expected_flags_when_insecure_by_default() {
    let server = TestServer::start().await; // cookie_secure = false: plain-HTTP test config
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    let set_cookie = response
        .headers_named("set-cookie")
        .into_iter()
        .find(|v| v.starts_with(server.cookie_name()))
        .unwrap_or_else(|| panic!("no {} cookie in {response:?}", server.cookie_name()));
    assert!(set_cookie.contains("HttpOnly"), "{set_cookie}");
    assert!(set_cookie.contains("SameSite=Strict"), "{set_cookie}");
    assert!(set_cookie.contains("Path=/"), "{set_cookie}");
    assert!(
        !set_cookie.contains("Secure"),
        "insecure config must not set Secure: {set_cookie}"
    );
    assert_eq!(
        server.cookie_name(),
        "ddai_session",
        "insecure config uses the non-__Host- name"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cookie_gets_secure_and_host_prefix_when_configured_secure() {
    let server = TestServer::start_with(|c| c.cookie_secure = true).await;
    assert_eq!(server.cookie_name(), "__Host-session");
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    let set_cookie = response
        .headers_named("set-cookie")
        .into_iter()
        .find(|v| v.starts_with("__Host-session"))
        .expect("__Host-session cookie");
    assert!(set_cookie.contains("Secure"), "{set_cookie}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_expires_after_the_idle_timeout() {
    let server = TestServer::start_with(|c| {
        c.idle_timeout = Duration::from_millis(50);
    })
    .await;
    let (cookie, _csrf) = server.login();
    tokio::time::sleep(Duration::from_millis(150)).await;
    let response = send(server.addr, Req::new("GET", "/api/me").cookie(&cookie));
    assert_eq!(response.json()["authenticated"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn activity_refreshes_the_idle_timeout() {
    let server = TestServer::start_with(|c| {
        c.idle_timeout = Duration::from_millis(150);
    })
    .await;
    let (cookie, _csrf) = server.login();
    for _ in 0..3 {
        tokio::time::sleep(Duration::from_millis(70)).await;
        let response = send(server.addr, Req::new("GET", "/api/me").cookie(&cookie));
        assert_eq!(
            response.json()["authenticated"],
            true,
            "activity should have refreshed the session"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_expires_after_the_absolute_timeout_even_with_activity() {
    let server = TestServer::start_with(|c| {
        c.idle_timeout = Duration::from_secs(3600);
        c.absolute_timeout = Duration::from_millis(100);
    })
    .await;
    let (cookie, _csrf) = server.login();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let response = send(server.addr, Req::new("GET", "/api/me").cookie(&cookie));
    assert_eq!(response.json()["authenticated"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tampered_cookie_value_is_rejected() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let (name, value) = cookie.split_once('=').expect("name=value");
    let mut tampered_value: Vec<u8> = value.bytes().collect();
    let last = tampered_value.len() - 1;
    tampered_value[last] ^= 0x01; // flip a bit near the end of the signature
    let tampered = format!("{name}={}", String::from_utf8(tampered_value).unwrap());
    let response = send(server.addr, Req::new("GET", "/api/me").cookie(&tampered));
    assert_eq!(response.json()["authenticated"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_cookie_value_is_rejected() {
    let server = TestServer::start().await;
    let response = send(
        server.addr,
        Req::new("GET", "/api/me").cookie(&format!("{}=nonsense", server.cookie_name())),
    );
    assert_eq!(response.json()["authenticated"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rate_limit_locks_out_after_too_many_login_attempts() {
    let server = TestServer::start_with(|c| {
        c.login_rate_limit.per_ip_limit = 3;
    })
    .await;
    for _ in 0..3 {
        let response = send(
            server.addr,
            Req::new("POST", "/api/login")
                .header("Origin", &server.origin())
                .json_body(&serde_json::json!({ "password": "wrong" })),
        );
        assert_eq!(response.status, 401);
    }
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": "wrong" })),
    );
    assert_eq!(response.status, 429);
    assert!(response.header("retry-after").is_some());

    // Even the *correct* password is locked out during the lockout window.
    let response = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(response.status, 429);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_locked_out_ip_cannot_lock_out_the_owner_from_a_different_ip() {
    // Direct regression test for review round 2's finding F8 (scratchpad/single_ip_lockout.py):
    // round 1's F3 fix (global-before-per-IP check ordering) had the side effect that *every*
    // repeat attempt from an already-locked-out IP still drained a global slot on its way to
    // being per-IP-rejected — so a single attacker hammering one IP could exhaust the *global*
    // budget alone (reviewer's repro: ~31 req/min from one host) and lock the real owner out even
    // though they're logging in from an entirely different IP. Uses the real default rate-limit
    // config (`per_ip_limit=5`, `global_limit=30`), not a shrunk test config, specifically so this
    // matches the reviewer's real-world numbers rather than an easier toy scenario.
    let server = TestServer::start_with(|c| {
        c.trust_proxy = true;
    })
    .await;

    // 40 wrong-password attempts from one attacker IP — comfortably more than enough to both
    // trigger that IP's own per-IP lockout (5) and, pre-fix, drain the entire global budget (30)
    // besides.
    for _ in 0..40 {
        send(
            server.addr,
            Req::new("POST", "/api/login")
                .header("Origin", &server.origin())
                .header("X-Forwarded-For", "198.51.100.7")
                .json_body(&serde_json::json!({ "password": "wrong" })),
        );
    }

    // The real owner, logging in correctly from a *different* IP with no trusted-device cookie,
    // must not be caught by the attacker's lockout.
    let owner_login = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .header("X-Forwarded-For", "203.0.113.42")
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(
        owner_login.status, 200,
        "the owner from a different IP must still be able to log in: {owner_login:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rate_limit_is_scoped_per_ip_via_trusted_proxy_header() {
    let server = TestServer::start_with(|c| {
        c.login_rate_limit.per_ip_limit = 2;
        c.trust_proxy = true;
    })
    .await;
    for _ in 0..2 {
        let response = send(
            server.addr,
            Req::new("POST", "/api/login")
                .header("Origin", &server.origin())
                .header("X-Forwarded-For", "10.1.1.1")
                .json_body(&serde_json::json!({ "password": "wrong" })),
        );
        assert_eq!(response.status, 401);
    }
    let locked = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .header("X-Forwarded-For", "10.1.1.1")
            .json_body(&serde_json::json!({ "password": "wrong" })),
    );
    assert_eq!(locked.status, 429, "10.1.1.1 should be locked out");

    let other_ip = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .header("X-Forwarded-For", "10.1.1.2")
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(
        other_ip.status, 200,
        "a different forwarded IP must not share the lockout"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_loopback_bind_is_refused_without_the_override_flag() {
    let tempdir = tempfile::tempdir().unwrap();
    let config = ddai_web::WebConfig::new(
        SocketAddr::new(std::net::Ipv4Addr::new(203, 0, 113, 5).into(), 0),
        tempdir.path().to_path_buf(),
    );
    match ddai_web::bind(config).await {
        Err(ddai_web::BindError::Config(_)) => {}
        Err(other) => panic!("expected a Config bind error, got: {other}"),
        Ok(_) => panic!("non-loopback bind should have been refused"),
    }
}

// -------------------------------------------------------------------------------------------
// Trusted-device global-limit bypass (review round 1, finding F7)
// -------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn trusted_device_bypasses_a_saturated_global_limit() {
    let server = TestServer::start_with(|c| {
        c.login_rate_limit.global_limit = 3;
        c.login_rate_limit.per_ip_limit = 1000; // keep the per-IP limiter out of the way
    })
    .await;

    // Login #1 (consumes global slot 1/3) earns a trusted-device cookie.
    let (_cookie, device_cookie, _csrf) = server.login_with_device();

    // Two more plain attempts consume the rest of the global budget (slots 2/3, 3/3).
    for _ in 0..2 {
        send(
            server.addr,
            Req::new("POST", "/api/login")
                .header("Origin", &server.origin())
                .json_body(&serde_json::json!({ "password": "wrong" })),
        );
    }

    // Confirm the global limit is now actually saturated for a plain request.
    let blocked = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(blocked.status, 429, "global limit should be exhausted: {blocked:?}");

    // The trusted device from login #1 can still log in.
    let bypass = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .cookie(&device_cookie)
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(
        bypass.status, 200,
        "a trusted device should bypass the saturated global limit: {bypass:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forged_device_cookie_does_not_bypass_the_global_limit() {
    let server = TestServer::start_with(|c| {
        c.login_rate_limit.global_limit = 2;
        c.login_rate_limit.per_ip_limit = 1000;
    })
    .await;

    for _ in 0..2 {
        send(
            server.addr,
            Req::new("POST", "/api/login")
                .header("Origin", &server.origin())
                .json_body(&serde_json::json!({ "password": "wrong" })),
        );
    }

    let forged_name = server.device_cookie_name();
    let forged = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .cookie(&format!("{forged_name}=not-a-real-signed-value"))
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(
        forged.status, 429,
        "a forged/undecodable device cookie must not bypass the global limit: {forged:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_device_cookie_does_not_bypass_the_global_limit() {
    let server = TestServer::start_with(|c| {
        c.trusted_device_ttl = Duration::from_millis(50);
        c.login_rate_limit.global_limit = 2;
        c.login_rate_limit.per_ip_limit = 1000;
    })
    .await;

    // Login #1 (global slot 1/2) earns a device cookie, which we then let expire.
    let (_cookie, device_cookie, _csrf) = server.login_with_device();
    tokio::time::sleep(Duration::from_millis(150)).await;

    // One more plain attempt (global slot 2/2) saturates the budget.
    send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": "wrong" })),
    );

    let expired = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .cookie(&device_cookie)
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(
        expired.status, 429,
        "an expired device cookie must not bypass the global limit: {expired:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn device_trust_survives_logout_but_not_a_password_change() {
    let server = TestServer::start_with(|c| {
        c.login_rate_limit.global_limit = 2;
        c.login_rate_limit.per_ip_limit = 1000;
    })
    .await;

    let (cookie, device_cookie, csrf) = server.login_with_device();
    // Ordinary logout must NOT revoke the device's trust.
    let logout = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &csrf)
            .cookie(&cookie),
    );
    assert_eq!(logout.status, 200);

    // Saturate the global limit (slot 1/2 was login #1 above; one more plain attempt is 2/2).
    send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .json_body(&serde_json::json!({ "password": "wrong" })),
    );
    let still_trusted = send(
        server.addr,
        Req::new("POST", "/api/login")
            .header("Origin", &server.origin())
            .cookie(&device_cookie)
            .json_body(&serde_json::json!({ "password": server.password })),
    );
    assert_eq!(
        still_trusted.status, 200,
        "logout must not revoke device trust: {still_trusted:?}"
    );
}

// -------------------------------------------------------------------------------------------
// Argon2 verification must not block unrelated requests (review round 1, finding F4)
// -------------------------------------------------------------------------------------------

// Exactly 1 async worker thread, deliberately: with more workers, whether the buggy (pre-fix,
// inline) version of this actually starves `GET /api/me` becomes a matter of scheduling luck
// (whichever worker frees up first might happen to pick up the cheap GET before the remaining
// slow logins). With a single worker, if argon2 verification ever again runs inline instead of
// on `spawn_blocking`'s separate pool, *every* other async task — including this GET — is
// deterministically stuck behind it. Client HTTP calls below use `spawn_blocking` themselves, so
// they run on tokio's blocking pool and don't compete for this one worker at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concurrent_slow_logins_do_not_starve_an_unrelated_request() {
    // Production-realistic (slow, ~0.2s) argon2 params, unlike every other test in this suite —
    // this is specifically about proving that cost doesn't block the async runtime's worker
    // threads anymore (it's moved to `spawn_blocking`, bounded by a semaphore).
    let server = TestServer::start_with_argon2(ddai_web::secrets::Argon2Params::default(), |_| {}).await;

    let mut login_tasks = Vec::new();
    for _ in 0..20 {
        let addr = server.addr;
        let origin = server.origin();
        login_tasks.push(tokio::task::spawn_blocking(move || {
            send(
                addr,
                Req::new("POST", "/api/login")
                    .header("Origin", &origin)
                    .json_body(&serde_json::json!({ "password": "wrong" })),
            )
        }));
    }

    // The "wait a bit, then measure" logic runs entirely on its own blocking-pool OS thread —
    // deliberately *not* `tokio::time::sleep` on the test's own (single-worker) runtime, which
    // would itself be stuck behind the same congested worker we're trying to measure around,
    // making any `Instant::now()` taken after it resuming already too late to see the delay.
    let me_addr = server.addr;
    let me_task = tokio::task::spawn_blocking(move || {
        std::thread::sleep(Duration::from_millis(20)); // let the logins actually start
        let started = std::time::Instant::now();
        let response = send(me_addr, Req::new("GET", "/api/me"));
        (started.elapsed(), response)
    });
    let (elapsed, me_response) = me_task.await.expect("GET /api/me task panicked");

    assert_eq!(me_response.json()["authenticated"], false);
    assert!(
        elapsed < Duration::from_millis(400),
        "an unrelated GET /api/me took {elapsed:?} while 20 slow logins were in flight — \
         looks like argon2 is blocking the (single, in this test) tokio worker thread again"
    );

    for task in login_tasks {
        task.await.expect("login task panicked");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loopback_bind_on_an_ephemeral_port_succeeds() {
    let tempdir = tempfile::tempdir().unwrap();
    let config = ddai_web::WebConfig::new(
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        tempdir.path().to_path_buf(),
    );
    let bound = ddai_web::bind(config).await.expect("loopback bind should succeed");
    assert_ne!(
        bound.local_addr.port(),
        0,
        "the OS should have assigned a real ephemeral port"
    );
}
