(function () {
  "use strict";

  // -----------------------------------------------------------------------------------------
  // Task 5.1: login / status (unchanged in spirit; extended with the WS reconnect-with-backoff
  // acceptance criterion 3 asks for, and to dispatch task 5.2a's new message types to GameView).
  // -----------------------------------------------------------------------------------------

  var loginView = document.getElementById("login-view");
  var statusView = document.getElementById("status-view");
  var loginForm = document.getElementById("login-form");
  var loginError = document.getElementById("login-error");
  var passwordInput = document.getElementById("password");
  var logoutButton = document.getElementById("logout-button");
  var connDot = document.getElementById("conn-dot");
  var wsStateEl = document.getElementById("ws-state");
  var botStateEl = document.getElementById("bot-state");
  var uptimeEl = document.getElementById("uptime");
  var tabbar = document.getElementById("tabbar");
  var tabStatusButton = document.getElementById("tab-status");
  var tabGameButton = document.getElementById("tab-game");
  var gameViewEl = document.getElementById("game-view");
  var tabBotButton = document.getElementById("tab-bot");
  var botViewEl = document.getElementById("bot-view");
  var tabFlyButton = document.getElementById("tab-fly");
  var flyViewEl = document.getElementById("fly-view");
  var tabTrainButton = document.getElementById("tab-train");
  var trainViewEl = document.getElementById("train-view");

  var csrfToken = null;
  var socket = null;
  var reconnectAttempts = 0;
  var reconnectTimer = null;
  var explicitlyClosed = true;
  var desiredLiveHz = 25;
  var currentTab = "status";

  function showLogin() {
    loginView.hidden = false;
    statusView.hidden = true;
    tabbar.hidden = true;
    gameViewEl.hidden = true;
    botViewEl.hidden = true;
    flyViewEl.hidden = true;
    trainViewEl.hidden = true;
    BotPanel.onHidden();
    FlyPanel.onHidden();
    TrainPanel.onHidden();
    setConnected(false);
  }

  function showTab(name) {
    currentTab = name;
    loginView.hidden = true;
    statusView.hidden = name !== "status";
    gameViewEl.hidden = name !== "game";
    botViewEl.hidden = name !== "bot";
    flyViewEl.hidden = name !== "fly";
    trainViewEl.hidden = name !== "train";
    tabTrainButton.classList.toggle("active", name === "train");
    tabFlyButton.classList.toggle("active", name === "fly");
    tabStatusButton.classList.toggle("active", name === "status");
    tabGameButton.classList.toggle("active", name === "game");
    tabBotButton.classList.toggle("active", name === "bot");
    if (name === "game") {
      GameView.onShown();
    }
    if (name === "bot") {
      BotPanel.onShown();
    } else {
      BotPanel.onHidden();
    }
    if (name === "fly") {
      FlyPanel.onShown();
    } else {
      FlyPanel.onHidden();
    }
    if (name === "train") {
      TrainPanel.onShown();
    } else {
      TrainPanel.onHidden();
    }
  }

  function showAuthenticated() {
    tabbar.hidden = false;
    showTab(
      currentTab === "game" || currentTab === "bot" || currentTab === "fly" || currentTab === "train" ? currentTab : "status",
    );
  }

  function setConnected(on) {
    connDot.classList.toggle("dot-on", on);
    connDot.classList.toggle("dot-off", !on);
    wsStateEl.textContent = on ? "подключено" : "не подключено";
    if (!on) {
      SourceBadge.hide();
    }
    GameView.onConnectionChanged(on);
    FlyPanel.onConnectionChanged(on);
  }

  function pad2(value) {
    return String(value).padStart(2, "0");
  }

  function formatUptime(seconds) {
    var total = Math.max(0, Math.floor(seconds));
    var h = Math.floor(total / 3600);
    var m = Math.floor((total % 3600) / 60);
    var s = total % 60;
    return pad2(h) + ":" + pad2(m) + ":" + pad2(s);
  }

  function sendJson(message) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }

  function sendSub(hz) {
    desiredLiveHz = hz;
    sendJson({ type: "sub", live: hz });
  }

  function sendReplay(action, extra) {
    var message = { type: "replay", action: action };
    if (extra) {
      for (var key in extra) {
        if (Object.prototype.hasOwnProperty.call(extra, key)) {
          message[key] = extra[key];
        }
      }
    }
    sendJson(message);
  }

  function scheduleReconnect() {
    if (explicitlyClosed) {
      return;
    }
    // Exponential backoff, capped at 10s (acceptance criterion 3: "reconnects automatically with
    // backoff and resubscribes").
    var delay = Math.min(10000, 500 * Math.pow(2, reconnectAttempts));
    reconnectAttempts += 1;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connectWs();
    }, delay);
  }

  function connectWs() {
    if (socket) {
      return;
    }
    explicitlyClosed = false;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(proto + "//" + location.host + "/ws");
    // Binary `live` frames arrive as `ArrayBuffer` (not `Blob`) so decoding never needs an extra
    // async round trip on the hot 25 Hz path.
    socket.binaryType = "arraybuffer";

    socket.addEventListener("open", function () {
      reconnectAttempts = 0;
    });
    socket.addEventListener("message", function (event) {
      if (event.data instanceof ArrayBuffer) {
        // Binary messages: `DFLY` (the fly's frame, task 7.4) or the live map's `DWLF`.
        var head = new Uint8Array(event.data, 0, Math.min(4, event.data.byteLength));
        if (head.length === 4 && head[0] === 0x44 && head[1] === 0x46) {
          FlyPanel.onFrame(event.data);
        } else {
          GameView.onLiveFrame(event.data);
        }
        return;
      }
      var msg;
      try {
        msg = JSON.parse(event.data);
      } catch (parseError) {
        return;
      }
      switch (msg.type) {
        case "hello":
          // Connection state is confirmed once the app-level `hello` message arrives, which
          // proves both the WebSocket handshake and our own protocol worked, not merely `onopen`.
          setConnected(true);
          if (desiredLiveHz !== 25) {
            sendSub(desiredLiveHz); // resubscribe at the user's chosen rate after a reconnect
          }
          break;
        case "status":
          botStateEl.textContent = msg.bot_state;
          uptimeEl.textContent = formatUptime(msg.uptime_s);
          break;
        case "map":
          GameView.onMap(msg);
          break;
        case "players":
          GameView.onPlayers(msg.list);
          break;
        case "events":
          GameView.onEvents(msg);
          break;
        case "replay_status":
          GameView.onReplayStatus(msg);
          break;
        case "bot":
          GameView.onBotStatus(msg.status);
          break;
        case "fly_meta":
          FlyPanel.onMeta(msg.meta);
          break;
        case "source":
          onSource(msg);
          break;
        case "live_error":
          GameView.onLiveError(msg.message);
          break;
        default:
          break; // forward-compatible: ignore anything this build doesn't know about
      }
    });
    socket.addEventListener("close", function () {
      setConnected(false);
      socket = null;
      // Review round 1, finding F5: a WS close after the session itself ended server-side
      // (logout on another tab, idle/absolute timeout, a password rotation) used to retry forever
      // at up to a 10s backoff while still showing the authenticated UI — the upgrade would just
      // keep getting a 401, and the user never saw the login form again without a manual reload.
      // `/api/me` is the same cheap, already-existing check `refresh()` uses at page load; only
      // reconnect if it says the session is still actually valid (a real transient network blip),
      // otherwise show the login form and stop retrying outright.
      if (explicitlyClosed) {
        return; // our own `disconnectWs()` call (e.g. the logout button) — nothing more to do
      }
      fetch("/api/me", { credentials: "same-origin" })
        .then(function (response) {
          return response.json();
        })
        .then(function (data) {
          if (data.authenticated) {
            scheduleReconnect();
          } else {
            csrfToken = null;
            explicitlyClosed = true; // stop any already-pending reconnect timer too
            if (reconnectTimer) {
              clearTimeout(reconnectTimer);
              reconnectTimer = null;
            }
            showLogin();
          }
        })
        .catch(function () {
          // `/api/me` itself is unreachable (the more likely "genuine network blip" case `/api/me`
          // was meant to rule OUT) — fall back to the old retry-with-backoff behavior rather than
          // giving up on a merely-offline connection.
          scheduleReconnect();
        });
    });
    socket.addEventListener("error", function () {
      setConnected(false);
    });
  }

  function disconnectWs() {
    explicitlyClosed = true;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (socket) {
      socket.close();
      socket = null;
    }
    setConnected(false);
  }

  function refresh() {
    fetch("/api/me", { credentials: "same-origin" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (data.authenticated) {
          csrfToken = data.csrf_token;
          showAuthenticated();
          connectWs();
        } else {
          csrfToken = null;
          showLogin();
        }
      })
      .catch(function () {
        showLogin();
      });
  }

  loginForm.addEventListener("submit", function (event) {
    event.preventDefault();
    loginError.hidden = true;
    fetch("/api/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: passwordInput.value }),
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("login failed");
        }
        return response.json();
      })
      .then(function (data) {
        csrfToken = data.csrf_token;
        passwordInput.value = "";
        showAuthenticated();
        connectWs();
      })
      .catch(function () {
        loginError.textContent = "Неверный пароль или сервер недоступен.";
        loginError.hidden = false;
      });
  });

  logoutButton.addEventListener("click", function () {
    fetch("/api/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-CSRF-Token": csrfToken || "" },
    }).finally(function () {
      disconnectWs();
      csrfToken = null;
      showLogin();
    });
  });

  tabStatusButton.addEventListener("click", function () {
    showTab("status");
  });
  tabGameButton.addEventListener("click", function () {
    showTab("game");
  });
  tabBotButton.addEventListener("click", function () {
    showTab("bot");
  });
  tabFlyButton.addEventListener("click", function () {
    showTab("fly");
  });
  tabTrainButton.addEventListener("click", function () {
    showTab("train");
  });
  FlyPanel.attach(sendJson);

  // -----------------------------------------------------------------------------------------
  // Task 5.7: what the site shows. The live bot has priority; while it is not running the offline demo (the fly playing an
  // arena, not a real game) stands in, and the server says which one is on show (`{"type":"source","kind":...}`). The badge
  // is on the «Игра», «Бот» and «Муха» tabs; every string is set with `textContent`.
  // -----------------------------------------------------------------------------------------

  var SourceBadge = (function () {
    var TITLES = {
      live: "Живой бот",
      demo: "Показ: муха на арене (не настоящая игра)",
      none: "Нет источника: бот не запущен",
    };
    var badges = ["game-source", "bot-source", "fly-source"]
      .map(function (id) {
        return document.getElementById(id);
      })
      .filter(Boolean);
    var kind = null;
    var info = null;
    var mapName = null;

    function detail() {
      var parts = [];
      if (kind === "live" || kind === "demo") {
        if (mapName) {
          parts.push("карта: " + mapName);
        }
      }
      if (kind === "demo" && info) {
        if (info.arena) {
          parts.push("арена: " + info.arena);
        }
        if (info.bundle) {
          parts.push("веса: " + info.bundle);
        }
      }
      return parts.join(" · ");
    }

    function render() {
      badges.forEach(function (el) {
        if (!kind || !TITLES[kind]) {
          el.hidden = true;
          return;
        }
        el.hidden = false;
        el.className = "source-badge source-" + kind;
        el.querySelector(".source-title").textContent = TITLES[kind];
        el.querySelector(".source-detail").textContent = detail();
      });
    }

    return {
      set: function (newKind, newInfo) {
        kind = newKind;
        info = newInfo && typeof newInfo === "object" ? newInfo : null;
        render();
      },
      setMap: function (name) {
        mapName = name || null;
        render();
      },
      // The connection to the site dropped: nothing is known about what is shown until it is back.
      hide: function () {
        kind = null;
        render();
      },
    };
  })();

  // What the server last said is on show (`live` / `demo` / `none`), kept across a reconnect so that a change that happened
  // while the connection was down still clears the previous source's drawing.
  var currentSource = null;

  function onSource(msg) {
    var kind = typeof msg.kind === "string" ? msg.kind : null;
    if (kind !== "live" && kind !== "demo" && kind !== "none") {
      return;
    }
    if (currentSource !== null && currentSource !== kind) {
      // Another source: what was drawn of the previous one (map, players, frames) is not this one's.
      GameView.reset();
    }
    currentSource = kind;
    GameView.setDemo(kind === "demo");
    SourceBadge.set(kind, msg.info);
    BotPanel.setDemo(kind === "demo");
  }

  // -----------------------------------------------------------------------------------------
  // Task 5.2a: the live map view.
  // -----------------------------------------------------------------------------------------

  var GameView = (function () {
    // Must match `crate::live::scene::Kind` (docs/formats.md) and `app.css`'s `--kind-*`
    // variables — index = the raw kind byte `/api/map/<sha256>` and the map scene both use.
    var KIND_COLORS = [
      [11, 14, 19], // 0 air
      [107, 114, 128], // 1 solid
      [139, 111, 71], // 2 nohook
      [220, 38, 38], // 3 death
      [30, 58, 138], // 4 deep freeze
      [56, 189, 248], // 5 freeze
      [13, 148, 136], // 6 deep unfreeze
      [134, 239, 172], // 7 unfreeze
      [168, 85, 247], // 8 tele in
      [217, 70, 239], // 9 tele out
      [139, 92, 246], // 10 tele checkpoint
      [249, 115, 22], // 11 speedup
      [234, 179, 8], // 12 switch
      [236, 72, 153], // 13 tune zone
      [146, 64, 14], // 14 stopper
      [74, 222, 128], // 15 spawn
    ];
    var KIND_NAMES = [
      "воздух",
      "стена",
      "без крюка",
      "смерть",
      "глуб. заморозка",
      "заморозка",
      "разморозка (глуб.)",
      "разморозка",
      "телепорт (вход)",
      "телепорт (выход)",
      "телепорт (чекпоинт)",
      "ускоритель",
      "переключатель",
      "зона тюнинга",
      "стопор",
      "спавн",
    ];
    var TILE_UNITS = 32; // game units per tile (DDNet convention)
    var CHUNK_TILES = 64;
    var MAX_DPR = 2; // matches orig-web's own DPR cap (docs/research/orig-web.md §2.4)

    var canvas = document.getElementById("game-canvas");
    var ctx = canvas.getContext("2d");
    var hudTick = document.getElementById("hud-tick");
    var hudFps = document.getElementById("hud-fps");
    var hudFrameRate = document.getElementById("hud-frame-rate");
    var hudMapName = document.getElementById("hud-map-name");
    var legendEl = document.getElementById("game-legend");
    var playerListEl = document.getElementById("player-list");
    var replayBar = document.getElementById("replay-bar");
    var replaySeek = document.getElementById("replay-seek");
    var replaySpeed = document.getElementById("replay-speed");
    var btnFollow = document.getElementById("btn-follow");
    var btnFit = document.getElementById("btn-fit");
    var btnNames = document.getElementById("btn-names");
    var btnEcon = document.getElementById("btn-econ");

    var mapMeta = null; // {sha256, name, w, h}
    var scene = null; // {width, height, kinds: Uint8Array}
    var chunks = null; // [{x0,y0,w,h,canvas}] in tile units
    var players = {}; // id -> {id, name, team}
    var prevFrame = null; // {tick, characters, wallTime}
    var latestFrame = null;
    var lastEventLog = []; // small ring of recent events, for a future log view / debugging
    var followId = null;
    // Task 5.7: while the offline demo is on show, the camera starts on the fly (the arena's halls are small next to the whole
    // map) instead of on the whole map; once per scene, and a drag or the fit button takes it over like any follow.
    var demoFraming = false;
    var pendingDemoFrame = false;
    var namesOn = true;
    // True when a scene just finished loading while `#game-view` was still `hidden` — see
    // `loadScene`'s completion handler and `onShown` below.
    var pendingFit = false;
    var camera = { x: 0, y: 0, scale: 1 }; // x/y in game units (map center), scale = device px per game unit
    var renderFrameTimes = []; // rAF frame durations, ms — exposed for Playwright (acceptance criterion 4)
    var receivedFrameTimestamps = []; // wall-clock ms of each received `live` frame, for the "кадры/с" HUD stat
    var lastRafTime = null;
    var replaySeekDragging = false;

    // Exposed for Playwright's CPU-throttled render-loop measurement (acceptance criterion 4:
    // "report the frame time p50/p95 of the render loop via requestAnimationFrame timestamps
    // exposed to the test").
    window.__ddaiRenderTimes = renderFrameTimes;

    function clamp(v, lo, hi) {
      return Math.min(hi, Math.max(lo, v));
    }

    // ------------------------------------------------------------------------------------- //
    // Binary `live` frame decoding (docs/formats.md's new section; `crate::live::frame`).
    // ------------------------------------------------------------------------------------- //

    function decodeLiveFrame(buffer) {
      var view = new DataView(buffer);
      if (
        buffer.byteLength < 12 ||
        view.getUint8(0) !== 0x44 /* D */ ||
        view.getUint8(1) !== 0x57 /* W */ ||
        view.getUint8(2) !== 0x4c /* L */ ||
        view.getUint8(3) !== 0x46 /* F */
      ) {
        return null;
      }
      var version = view.getUint8(4);
      if (version !== 1) {
        return null;
      }
      var tick = view.getUint32(6, true);
      var charCount = view.getUint16(10, true);
      var characters = [];
      var offset = 12;
      var RECORD_BYTES = 26;
      for (var i = 0; i < charCount; i++) {
        if (offset + RECORD_BYTES > buffer.byteLength) {
          break; // truncated — draw what we could decode rather than throwing
        }
        var flags = view.getUint8(offset + 1);
        var hookedRaw = view.getInt8(offset + 24);
        characters.push({
          id: view.getUint8(offset),
          alive: (flags & 0x01) !== 0,
          frozen: (flags & 0x02) !== 0,
          deepFrozen: (flags & 0x04) !== 0,
          liveFrozen: (flags & 0x08) !== 0,
          hookVisible: (flags & 0x10) !== 0,
          team: view.getUint8(offset + 2),
          weapon: view.getUint8(offset + 3),
          x: view.getInt32(offset + 4, true),
          y: view.getInt32(offset + 8, true),
          aimX: view.getInt16(offset + 12, true),
          aimY: view.getInt16(offset + 14, true),
          hookX: view.getInt32(offset + 16, true),
          hookY: view.getInt32(offset + 20, true),
          hookedId: hookedRaw < 0 ? null : hookedRaw,
        });
        offset += RECORD_BYTES;
      }
      return { tick: tick, characters: characters };
    }

    // ------------------------------------------------------------------------------------- //
    // Map scene: fetch, inflate (native `DecompressionStream`, no vendored inflate library),
    // classify into chunked offscreen canvases (acceptance criterion 3).
    // ------------------------------------------------------------------------------------- //

    // Review round 1, finding F3: the previous version applied whatever `/api/map/<sha256>`
    // response arrived last, with no check against which map is CURRENT by the time it actually
    // arrives — a slow response for a map the user has since navigated away from (e.g. pressing
    // ⏭ while an earlier fetch is still in flight) would still overwrite the (already correct)
    // scene for the NEW map once it finally landed. `sceneRequestSha256` records which sha256 the
    // most recently STARTED request is for; a response is only applied if it's still the most
    // recent one by the time it resolves. `sceneAbortController` additionally cancels the
    // previous in-flight request outright (so it doesn't keep using bandwidth/CPU for a map
    // nobody wants anymore), rather than relying on the stale-check alone.
    var sceneRequestSha256 = null;
    var sceneAbortController = null;

    function loadScene(sha256Hex) {
      if (sceneAbortController) {
        sceneAbortController.abort();
      }
      sceneRequestSha256 = sha256Hex;
      var abortController = new AbortController();
      sceneAbortController = abortController;
      fetchSceneWithRetry(sha256Hex, abortController.signal, 0)
        .then(function (built) {
          if (sha256Hex !== sceneRequestSha256) {
            return; // superseded by a newer `onMap` before this fetch finished — drop it
          }
          scene = built;
          pendingDemoFrame = demoFraming;
          buildChunks();
          renderLegend();
          // Fitting the camera needs the canvas's REAL on-screen size
          // (`canvas.getBoundingClientRect()`), which is all-zero while `#game-view` is still
          // `hidden` (the map often loads before the user ever opens the "Игра" tab, since the WS
          // connection — and therefore the `map` message — doesn't wait for that tab). Fit now if
          // the tab already happens to be open; otherwise `onShown()` does it the first time it
          // actually becomes visible (`pendingFit` below).
          if (!gameViewEl.hidden) {
            fitToMap();
          } else {
            pendingFit = true;
          }
        })
        .catch(function (error) {
          if (error.name === "AbortError" || sha256Hex !== sceneRequestSha256) {
            return; // expected: superseded by a newer map, not a real failure
          }
          onLiveError("не удалось загрузить карту: " + error.message);
        });
    }

    var SCENE_FETCH_MAX_RETRIES = 5;
    var SCENE_FETCH_RETRY_DELAY_MS = 500;

    /// Fetches and decodes one map scene, retrying a 404 a few times (the replay source resolves
    /// and caches a map asynchronously — a 404 right after a `map` message arrives most likely
    /// means "not cached yet", not "this sha256 will never exist", see `crate::http::map`) before
    /// giving up and surfacing it as a real error.
    function fetchSceneWithRetry(sha256Hex, signal, attempt) {
      return fetch("/api/map/" + sha256Hex, { credentials: "same-origin", signal: signal })
        .then(function (response) {
          if (response.status === 404 && attempt < SCENE_FETCH_MAX_RETRIES) {
            return new Promise(function (resolve, reject) {
              setTimeout(function () {
                fetchSceneWithRetry(sha256Hex, signal, attempt + 1).then(resolve, reject);
              }, SCENE_FETCH_RETRY_DELAY_MS);
            });
          }
          if (!response.ok) {
            throw new Error("map fetch failed: " + response.status);
          }
          return response.arrayBuffer().then(decodeSceneBuffer);
        });
    }

    function decodeSceneBuffer(buffer) {
      var view = new DataView(buffer);
      var width = view.getUint32(0, true);
      var height = view.getUint32(4, true);
      var deflatedLen = view.getUint32(8, true);
      var deflated = new Uint8Array(buffer, 12, deflatedLen);
      if (typeof DecompressionStream === "undefined") {
        throw new Error("DecompressionStream unsupported");
      }
      var stream = new Blob([deflated]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
      return new Response(stream)
        .arrayBuffer()
        .then(function (inflated) {
          return { width: width, height: height, kinds: new Uint8Array(inflated) };
        });
    }

    function buildChunks() {
      chunks = [];
      if (!scene) {
        return;
      }
      var chunksX = Math.ceil(scene.width / CHUNK_TILES);
      var chunksY = Math.ceil(scene.height / CHUNK_TILES);
      for (var cy = 0; cy < chunksY; cy++) {
        for (var cx = 0; cx < chunksX; cx++) {
          var x0 = cx * CHUNK_TILES;
          var y0 = cy * CHUNK_TILES;
          var w = Math.min(CHUNK_TILES, scene.width - x0);
          var h = Math.min(CHUNK_TILES, scene.height - y0);
          var chunkCanvas = document.createElement("canvas");
          chunkCanvas.width = w;
          chunkCanvas.height = h;
          var chunkCtx = chunkCanvas.getContext("2d");
          var imageData = chunkCtx.createImageData(w, h);
          for (var ty = 0; ty < h; ty++) {
            for (var tx = 0; tx < w; tx++) {
              var kind = scene.kinds[(y0 + ty) * scene.width + (x0 + tx)];
              var color = KIND_COLORS[kind] || KIND_COLORS[0];
              var pixelIndex = (ty * w + tx) * 4;
              imageData.data[pixelIndex] = color[0];
              imageData.data[pixelIndex + 1] = color[1];
              imageData.data[pixelIndex + 2] = color[2];
              imageData.data[pixelIndex + 3] = 255;
            }
          }
          chunkCtx.putImageData(imageData, 0, 0);
          chunks.push({ x0: x0, y0: y0, w: w, h: h, canvas: chunkCanvas });
        }
      }
    }

    function renderLegend() {
      legendEl.innerHTML = "";
      // Only the kinds actually present on this map, so the legend stays short.
      var present = new Set();
      if (scene) {
        for (var i = 0; i < scene.kinds.length; i++) {
          present.add(scene.kinds[i]);
          if (present.size === KIND_COLORS.length) {
            break;
          }
        }
      }
      present.forEach(function (kind) {
        if (kind === 0) {
          return; // "air" is the background; not worth a legend entry
        }
        var color = KIND_COLORS[kind] || KIND_COLORS[0];
        var item = document.createElement("span");
        var swatch = document.createElement("i");
        swatch.style.background = "rgb(" + color[0] + "," + color[1] + "," + color[2] + ")";
        item.appendChild(swatch);
        item.appendChild(document.createTextNode(KIND_NAMES[kind] || "?"));
        legendEl.appendChild(item);
      });
    }

    // ------------------------------------------------------------------------------------- //
    // Camera.
    // ------------------------------------------------------------------------------------- //

    function fitToMap() {
      if (!scene) {
        return;
      }
      followId = null;
      btnFollow.classList.remove("active");
      var mapWidthUnits = scene.width * TILE_UNITS;
      var mapHeightUnits = scene.height * TILE_UNITS;
      var rect = canvas.getBoundingClientRect();
      var scaleX = (rect.width || 300) / mapWidthUnits;
      var scaleY = (rect.height || 300) / mapHeightUnits;
      camera.scale = Math.min(scaleX, scaleY) * 0.95;
      camera.x = mapWidthUnits / 2;
      camera.y = mapHeightUnits / 2;
    }

    function zoomAt(factor, screenX, screenY) {
      var rect = canvas.getBoundingClientRect();
      var beforeX = camera.x + (screenX - rect.width / 2) / camera.scale;
      var beforeY = camera.y + (screenY - rect.height / 2) / camera.scale;
      camera.scale = clamp(camera.scale * factor, 0.02, 20);
      camera.x = beforeX - (screenX - rect.width / 2) / camera.scale;
      camera.y = beforeY - (screenY - rect.height / 2) / camera.scale;
    }

    function setFollow(id) {
      followId = id;
      btnFollow.classList.toggle("active", id !== null);
    }

    // ------------------------------------------------------------------------------------- //
    // Input: drag-pan, wheel-zoom (desktop), pinch-zoom + double-tap-zoom (phone) — acceptance
    // criterion 3: "Pinch-zoom and double-tap on phones, wheel and drag on desktop."
    // ------------------------------------------------------------------------------------- //

    var pointers = {}; // pointerId -> {x, y}
    var dragLast = null;
    var pinchStartDistance = null;
    var pinchStartScale = null;
    var lastTapTime = 0;
    var lastTapPos = null;
    var pointerDownPos = null;

    function pointerDistance() {
      var ids = Object.keys(pointers);
      if (ids.length < 2) {
        return null;
      }
      var a = pointers[ids[0]];
      var b = pointers[ids[1]];
      return Math.hypot(a.x - b.x, a.y - b.y);
    }

    function pointerMidpoint() {
      var ids = Object.keys(pointers);
      var a = pointers[ids[0]];
      var b = pointers[ids[1]];
      return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    }

    canvas.addEventListener("pointerdown", function (event) {
      canvas.setPointerCapture(event.pointerId);
      var rect = canvas.getBoundingClientRect();
      pointers[event.pointerId] = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      pointerDownPos = { x: event.clientX, y: event.clientY, time: performance.now() };
      if (Object.keys(pointers).length === 1) {
        dragLast = pointers[event.pointerId];
      } else {
        dragLast = null;
        pinchStartDistance = pointerDistance();
        pinchStartScale = camera.scale;
      }
    });

    canvas.addEventListener("pointermove", function (event) {
      if (!pointers[event.pointerId]) {
        return;
      }
      var rect = canvas.getBoundingClientRect();
      var pos = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      pointers[event.pointerId] = pos;
      var count = Object.keys(pointers).length;
      if (count === 1 && dragLast) {
        var dx = pos.x - dragLast.x;
        var dy = pos.y - dragLast.y;
        camera.x -= dx / camera.scale;
        camera.y -= dy / camera.scale;
        setFollow(null); // dragging disengages follow (acceptance criterion 3)
        dragLast = pos;
      } else if (count === 2 && pinchStartDistance) {
        // Recentring the zoom on the pinch midpoint (`pointerMidpoint()`) would be a further
        // nicety; scaling in place around the current camera center is simpler and still
        // satisfies "pinch-zoom" (acceptance criterion 3) correctly, just without that polish.
        var distance = pointerDistance();
        if (distance) {
          camera.scale = clamp(pinchStartScale * (distance / pinchStartDistance), 0.02, 20);
        }
      }
    });

    function endPointer(event) {
      delete pointers[event.pointerId];
      if (Object.keys(pointers).length < 2) {
        pinchStartDistance = null;
      }
      if (Object.keys(pointers).length === 0 && pointerDownPos) {
        var dx = event.clientX - pointerDownPos.x;
        var dy = event.clientY - pointerDownPos.y;
        var elapsed = performance.now() - pointerDownPos.time;
        if (Math.hypot(dx, dy) < 10 && elapsed < 400) {
          handleTap(event.clientX, event.clientY);
        }
      }
      dragLast = null;
    }
    canvas.addEventListener("pointerup", endPointer);
    canvas.addEventListener("pointercancel", endPointer);

    canvas.addEventListener(
      "wheel",
      function (event) {
        event.preventDefault();
        var rect = canvas.getBoundingClientRect();
        var factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
        zoomAt(factor, event.clientX - rect.left, event.clientY - rect.top);
      },
      { passive: false }
    );

    function handleTap(clientX, clientY) {
      var now = performance.now();
      var isDoubleTap =
        now - lastTapTime < 350 && lastTapPos && Math.hypot(clientX - lastTapPos.x, clientY - lastTapPos.y) < 30;
      lastTapTime = now;
      lastTapPos = { x: clientX, y: clientY };
      var rect = canvas.getBoundingClientRect();
      var localX = clientX - rect.left;
      var localY = clientY - rect.top;
      if (isDoubleTap) {
        zoomAt(2, localX, localY);
        lastTapTime = 0; // don't chain a third tap into another double-tap
        return;
      }
      var hit = hitTestCharacter(localX, localY, rect);
      if (hit !== null) {
        setFollow(hit);
      }
    }

    function hitTestCharacter(localX, localY, rect) {
      if (!latestFrame) {
        return null;
      }
      var best = null;
      var bestDist = 24; // px hit radius
      for (var i = 0; i < latestFrame.characters.length; i++) {
        var c = latestFrame.characters[i];
        var screen = worldToScreen(c.x, c.y, rect);
        var d = Math.hypot(screen.x - localX, screen.y - localY);
        if (d < bestDist) {
          bestDist = d;
          best = c.id;
        }
      }
      return best;
    }

    function worldToScreen(worldX, worldY, rect) {
      return {
        x: rect.width / 2 + (worldX - camera.x) * camera.scale,
        y: rect.height / 2 + (worldY - camera.y) * camera.scale,
      };
    }

    btnFit.addEventListener("click", function () {
      fitToMap();
    });
    btnFollow.addEventListener("click", function () {
      if (followId !== null) {
        setFollow(null);
      } else if (latestFrame && latestFrame.characters.length > 0) {
        setFollow(latestFrame.characters[0].id);
      }
    });
    btnNames.addEventListener("click", function () {
      namesOn = !namesOn;
      btnNames.classList.toggle("active", namesOn);
    });

    // Review round 1, finding F9: acceptance criterion 1 asks for a 10 Hz "эконом" mode for
    // mobile networks (the server already supports any rate via `sub{live: hz}`, see
    // `docs/formats.md` §15.4) but this task's first version never exposed a way to actually
    // request it from the UI. `localStorage` read/write is wrapped in `try/catch` per this
    // codebase's own convention elsewhere (a private browsing mode or blocked site data can throw
    // or silently no-op — never let a missing/broken storage API break the toggle itself, just its
    // persistence across reloads).
    var ECON_STORAGE_KEY = "ddai.econ";
    var econOn = false;
    try {
      econOn = window.localStorage.getItem(ECON_STORAGE_KEY) === "1";
    } catch (storageError) {
      econOn = false;
    }

    function applyEconMode(on) {
      econOn = on;
      btnEcon.classList.toggle("active", econOn);
      sendSub(econOn ? 10 : 25);
      try {
        window.localStorage.setItem(ECON_STORAGE_KEY, econOn ? "1" : "0");
      } catch (storageError) {
        // Best-effort persistence only — see this block's own comment above.
      }
    }

    btnEcon.addEventListener("click", function () {
      applyEconMode(!econOn);
    });
    // Applied once at load (not just on click) so a saved preference actually takes effect
    // immediately, including sending the initial `sub` — and so `onConnectionChanged`'s own
    // resubscribe-after-reconnect logic (`desiredLiveHz !== 25`, in the outer IIFE) sees the
    // right value from the very first connection, not only after the button is first clicked.
    if (econOn) {
      applyEconMode(true);
    }

    // ------------------------------------------------------------------------------------- //
    // Replay controls (acceptance criterion 2/3).
    // ------------------------------------------------------------------------------------- //

    document.getElementById("replay-play").addEventListener("click", function () {
      sendReplay("play");
    });
    document.getElementById("replay-pause").addEventListener("click", function () {
      sendReplay("pause");
    });
    document.getElementById("replay-next").addEventListener("click", function () {
      sendReplay("next");
    });
    replaySpeed.addEventListener("change", function () {
      sendReplay("speed", { value: parseFloat(replaySpeed.value) });
    });
    replaySeek.addEventListener("pointerdown", function () {
      replaySeekDragging = true;
    });
    replaySeek.addEventListener("change", function () {
      sendReplay("seek", { tick: parseInt(replaySeek.value, 10) });
      replaySeekDragging = false;
    });

    // ------------------------------------------------------------------------------------- //
    // Render loop.
    // ------------------------------------------------------------------------------------- //

    function resizeCanvasIfNeeded() {
      var dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
      var rect = canvas.getBoundingClientRect();
      var targetW = Math.max(1, Math.round(rect.width * dpr));
      var targetH = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
    }

    function interpolatedCharacters(now) {
      if (!latestFrame) {
        return [];
      }
      if (!prevFrame) {
        return latestFrame.characters;
      }
      var span = clamp(latestFrame.wallTime - prevFrame.wallTime, 20, 500);
      var t = clamp((now - latestFrame.wallTime) / span, 0, 1);
      var prevById = {};
      for (var i = 0; i < prevFrame.characters.length; i++) {
        prevById[prevFrame.characters[i].id] = prevFrame.characters[i];
      }
      return latestFrame.characters.map(function (c) {
        var p = prevById[c.id];
        if (!p) {
          return c;
        }
        var out = Object.assign({}, c);
        out.x = p.x + (c.x - p.x) * t;
        out.y = p.y + (c.y - p.y) * t;
        return out;
      });
    }

    function teeColor(id) {
      var hue = (id * 53) % 360; // deterministic, distinct-enough hues per id
      return "hsl(" + hue + ", 70%, 55%)";
    }

    var TEAM_RING_COLORS = ["#8a93a3", "#f97316", "#22d3ee", "#a3e635", "#e879f9", "#facc15", "#60a5fa", "#fb7185"];

    var DEMO_TILES_ACROSS = 36; // the arena's two halls are about 26 tiles wide

    // Puts the camera on the fly (slot 0 of the demo) at a zoom where the hall fills the view, and follows it.
    function frameDemo() {
      if (!pendingDemoFrame || !scene || !latestFrame || gameViewEl.hidden) {
        return;
      }
      var fly = latestFrame.characters.filter(function (c) {
        return c.id === 0;
      })[0];
      var rect = canvas.getBoundingClientRect();
      if (!fly || !rect.width || !rect.height) {
        return;
      }
      pendingDemoFrame = false;
      camera.scale = clamp(Math.min(rect.width, rect.height) / (DEMO_TILES_ACROSS * TILE_UNITS), 0.02, 20);
      camera.x = fly.x;
      camera.y = fly.y;
      setFollow(0);
    }

    function draw(now, dtMs) {
      frameDemo();
      resizeCanvasIfNeeded();
      var dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var rect = { width: canvas.width / dpr, height: canvas.height / dpr };
      ctx.clearRect(0, 0, rect.width, rect.height);

      if (followId !== null && latestFrame) {
        var target = latestFrame.characters.filter(function (c) {
          return c.id === followId;
        })[0];
        if (target) {
          // `dtMs` is the real gap since the PREVIOUS frame, passed in by `loop()` — computing it
          // from `lastRafTime` here instead would always read 0 (`loop()` already updates
          // `lastRafTime = now` before calling this function), which silently made `k` always 0
          // and follow mode never actually move the camera at all (caught by
          // tools/e2e/live-map.spec.ts's "following a player keeps the camera on them" test).
          var k = 1 - Math.exp(-(dtMs / 1000) * 14);
          camera.x += (target.x - camera.x) * k;
          camera.y += (target.y - camera.y) * k;
        }
      }

      if (chunks) {
        ctx.imageSmoothingEnabled = false;
        for (var i = 0; i < chunks.length; i++) {
          var chunk = chunks[i];
          var worldX0 = chunk.x0 * TILE_UNITS;
          var worldY0 = chunk.y0 * TILE_UNITS;
          var worldW = chunk.w * TILE_UNITS;
          var worldH = chunk.h * TILE_UNITS;
          var screenX = rect.width / 2 + (worldX0 - camera.x) * camera.scale;
          var screenY = rect.height / 2 + (worldY0 - camera.y) * camera.scale;
          var screenW = worldW * camera.scale;
          var screenH = worldH * camera.scale;
          if (screenX + screenW < 0 || screenY + screenH < 0 || screenX > rect.width || screenY > rect.height) {
            continue; // outside the viewport — skip (this is the render-time half of "chunked")
          }
          ctx.drawImage(chunk.canvas, 0, 0, chunk.w, chunk.h, screenX, screenY, screenW, screenH);
        }
      }

      var chars = interpolatedCharacters(now);
      for (var j = 0; j < chars.length; j++) {
        drawCharacter(chars[j], rect);
      }

      hudTick.textContent = latestFrame ? String(latestFrame.tick) : "—";
    }

    function drawCharacter(c, rect) {
      var screen = worldToScreen(c.x, c.y, rect);
      var radius = Math.max(3, 14 * camera.scale * TILE_UNITS * 0.02);

      if (c.hookVisible) {
        var hookScreen = worldToScreen(c.hookX, c.hookY, rect);
        ctx.strokeStyle = "#d4a373";
        ctx.lineWidth = Math.max(1, radius * 0.25);
        ctx.beginPath();
        ctx.moveTo(screen.x, screen.y);
        ctx.lineTo(hookScreen.x, hookScreen.y);
        ctx.stroke();
      }

      // Aim direction (acceptance criterion 3: "aim").
      var aimLen = Math.hypot(c.aimX, c.aimY) || 1;
      var aimDirX = c.aimX / aimLen;
      var aimDirY = c.aimY / aimLen;
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = Math.max(1, radius * 0.2);
      ctx.beginPath();
      ctx.moveTo(screen.x, screen.y);
      ctx.lineTo(screen.x + aimDirX * radius * 1.8, screen.y + aimDirY * radius * 1.8);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = c.alive ? teeColor(c.id) : "rgba(120,120,120,0.4)";
      ctx.fill();
      ctx.lineWidth = Math.max(1, radius * 0.18);
      ctx.strokeStyle = TEAM_RING_COLORS[c.team % TEAM_RING_COLORS.length];
      ctx.stroke();

      // Frozen/deep-frozen marker: a clear ring in the freeze palette color (acceptance
      // criterion 3: "a clear frozen/deep-frozen marker").
      if (c.deepFrozen) {
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 3, 0, Math.PI * 2);
        ctx.strokeStyle = "#1e3a8a";
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (c.frozen) {
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 3, 0, Math.PI * 2);
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      if (c.id === followId) {
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 6, 0, Math.PI * 2);
        ctx.strokeStyle = "#4f8cff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      if (namesOn) {
        var label = players[c.id] ? players[c.id].name : "#" + c.id;
        ctx.font = "11px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = "rgba(232,235,240,0.9)";
        ctx.fillText(label, screen.x, screen.y - radius - 6);
      }
    }

    function loop(now) {
      // Computed BEFORE `lastRafTime` is updated below — `draw()` needs the gap since the
      // PREVIOUS frame (for follow-camera smoothing), not since itself.
      var dtMs = lastRafTime !== null ? now - lastRafTime : 16;
      if (lastRafTime !== null) {
        renderFrameTimes.push(dtMs);
        if (renderFrameTimes.length > 600) {
          renderFrameTimes.shift();
        }
        var recentAvg =
          renderFrameTimes.slice(-30).reduce(function (a, b) {
            return a + b;
          }, 0) / Math.min(30, renderFrameTimes.length);
        hudFps.textContent = recentAvg > 0 ? Math.round(1000 / recentAvg) : "—";
      }
      lastRafTime = now;
      draw(now, dtMs);
      window.requestAnimationFrame(loop);
    }
    window.requestAnimationFrame(loop);

    // Recompute the "received frames per second" HUD stat, and refresh the player list's
    // alive/frozen indicators, twice a second — NOT on every single received `live` frame (up to
    // 50/s): rebuilding the list's DOM that often tore it down and rebuilt it out from under any
    // click a user (or a Playwright test) was in the middle of, well before the click could
    // land. The list's actual roster (`onPlayers`) and follow highlight (`setFollow`) still
    // re-render immediately, since those are rare, deliberate changes, not a per-tick refresh.
    setInterval(function () {
      var cutoff = performance.now() - 1000;
      receivedFrameTimestamps = receivedFrameTimestamps.filter(function (t) {
        return t >= cutoff;
      });
      hudFrameRate.textContent = String(receivedFrameTimestamps.length);
      renderPlayerList();
    }, 500);

    window.addEventListener("resize", function () {
      if (!followId) {
        // A resize while free-panning keeps the same center; only re-fit if nothing is being
        // followed and the view hasn't been manually adjusted — simplest reasonable default is
        // to leave the camera alone otherwise, so a phone rotation doesn't yank a manually
        // composed shot back to "fit".
      }
    });

    // ------------------------------------------------------------------------------------- //
    // Player list.
    // ------------------------------------------------------------------------------------- //

    function renderPlayerList() {
      playerListEl.innerHTML = "";
      var ids = Object.keys(players);
      ids.sort(function (a, b) {
        return Number(a) - Number(b);
      });
      var byId = {};
      if (latestFrame) {
        latestFrame.characters.forEach(function (c) {
          byId[c.id] = c;
        });
      }
      ids.forEach(function (idStr) {
        var id = Number(idStr);
        var player = players[id];
        var character = byId[id];
        var li = document.createElement("li");
        li.className = "player-row" + (id === followId ? " selected" : "");
        var swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = teeColor(id);
        var name = document.createElement("span");
        name.className = "name";
        name.textContent = player.name;
        li.appendChild(swatch);
        li.appendChild(name);
        if (character) {
          if (!character.alive) {
            var dead = document.createElement("span");
            dead.className = "frozen-mark";
            dead.textContent = "мёртв";
            li.appendChild(dead);
          } else if (character.deepFrozen) {
            var deep = document.createElement("span");
            deep.className = "frozen-mark deep";
            deep.textContent = "❄❄";
            li.appendChild(deep);
          } else if (character.frozen) {
            var frozenMark = document.createElement("span");
            frozenMark.className = "frozen-mark";
            frozenMark.textContent = "❄";
            li.appendChild(frozenMark);
          }
        }
        li.addEventListener("click", function () {
          setFollow(id);
        });
        playerListEl.appendChild(li);
      });
    }

    // ------------------------------------------------------------------------------------- //
    // Public API — called from the WS message dispatch above.
    // ------------------------------------------------------------------------------------- //

    function onMap(msg) {
      var changed = !mapMeta || mapMeta.sha256 !== msg.sha256;
      mapMeta = msg;
      hudMapName.textContent = msg.name;
      SourceBadge.setMap(msg.name);
      if (changed) {
        scene = null;
        chunks = null;
        loadScene(msg.sha256);
      }
    }

    // Task 5.7: another source is on show; nothing of the previous one's game stays (its map, players and frames).
    function reset() {
      if (sceneAbortController) {
        sceneAbortController.abort();
      }
      sceneRequestSha256 = null;
      mapMeta = null;
      scene = null;
      chunks = null;
      pendingFit = false;
      players = {};
      prevFrame = null;
      latestFrame = null;
      followId = null;
      lastEventLog = [];
      receivedFrameTimestamps = [];
      hudMapName.textContent = "—";
      document.getElementById("hud-bot").hidden = true;
      replayBar.hidden = true;
      SourceBadge.setMap(null);
      renderLegend();
      renderPlayerList();
    }

    // Task 5.7: the offline demo is (not) on show: whether the camera frames the fly when a scene is drawn.
    function setDemo(on) {
      demoFraming = !!on;
      pendingDemoFrame = demoFraming && !!scene;
    }

    function onPlayers(list) {
      players = {};
      list.forEach(function (p) {
        players[p.id] = p;
      });
      renderPlayerList();
    }

    function onEvents(msg) {
      lastEventLog.push(msg);
      if (lastEventLog.length > 200) {
        lastEventLog.shift();
      }
    }

    // Task 4.1: the live bot's status line in the HUD (target, mode, brain, p99 of the decision).
    // Every value is written through `textContent` — the status comes from the bot process, but it
    // is still only ever displayed as text.
    function onBotStatus(st) {
      var box = document.getElementById("hud-bot");
      var text = document.getElementById("hud-bot-text");
      if (!box || !text || !st || typeof st !== "object") {
        return;
      }
      box.hidden = false;
      var target = st.target >= 0 ? "цель " + st.target : "без цели";
      var state = !st.alive ? "мёртв" : st.frozen ? "во фризе" : "жив";
      text.textContent =
        st.mode + " · " + st.brain + " · " + target + " · " + state +
        " · блоки " + st.blocks + "/" + st.blocked_by +
        " · p99 " + (st.decide_p99_us / 1000).toFixed(1) + " мс";
    }

    function onReplayStatus(msg) {
      replayBar.hidden = false;
      if (!replaySeekDragging) {
        replaySeek.max = String(Math.max(1, msg.tick_count));
        replaySeek.value = String(msg.tick);
      }
      if (document.activeElement !== replaySpeed) {
        replaySpeed.value = String(msg.speed);
      }
      document.getElementById("replay-play").classList.toggle("active", msg.playing);
      document.getElementById("replay-pause").classList.toggle("active", !msg.playing);
    }

    function onLiveError(message) {
      // eslint-disable-next-line no-console
      console.error("live map error:", message);
      hudMapName.textContent = "ошибка: " + message;
    }

    function onLiveFrame(buffer) {
      var decoded = decodeLiveFrame(buffer);
      if (!decoded) {
        return;
      }
      var now = performance.now();
      receivedFrameTimestamps.push(now);
      prevFrame = latestFrame;
      latestFrame = { tick: decoded.tick, characters: decoded.characters, wallTime: now };
      // Player list alive/frozen indicators refresh on the slower 500ms timer below, not here —
      // see that timer's own comment for why.
    }

    function onConnectionChanged(connected) {
      if (!connected) {
        hudMapName.textContent = "нет соединения";
      }
    }

    function onShown() {
      resizeCanvasIfNeeded();
      if (scene && !chunks) {
        buildChunks();
      }
      if (pendingFit) {
        pendingFit = false;
        fitToMap();
      }
    }

    // Test-only hook (tools/e2e/live-map.spec.ts): read-only camera/frame snapshot and a way to
    // change the live subscription rate without a dedicated UI control for it — neither is part
    // of the WS protocol itself (`docs/formats.md` §15), just a way for Playwright to observe/
    // drive this module without reaching into its closed-over internals directly.
    window.__ddaiDebug = {
      getState: function () {
        return {
          followId: followId,
          camera: { x: camera.x, y: camera.y, scale: camera.scale },
          latestFrame: latestFrame ? { tick: latestFrame.tick, characters: latestFrame.characters } : null,
          mapMeta: mapMeta,
          // The actually-rendered scene's own dimensions (as opposed to `mapMeta`, which is set
          // synchronously from the `map` WS message alone) — lets a test tell "the HUD says map X"
          // apart from "the canvas is actually drawing map X's geometry" (review round 1, finding
          // F3: these two could disagree while a stale `/api/map/<sha256>` fetch was still in
          // flight for the PREVIOUS map).
          scene: scene ? { width: scene.width, height: scene.height } : null,
        };
      },
      setLiveHz: function (hz) {
        sendSub(hz);
      },
      screenPositionOf: function (id) {
        if (!latestFrame) {
          return null;
        }
        var character = latestFrame.characters.filter(function (c) {
          return c.id === id;
        })[0];
        if (!character) {
          return null;
        }
        var rect = canvas.getBoundingClientRect();
        return worldToScreen(character.x, character.y, rect);
      },
    };

    return {
      reset: reset,
      setDemo: setDemo,
      onMap: onMap,
      onPlayers: onPlayers,
      onEvents: onEvents,
      onReplayStatus: onReplayStatus,
      onBotStatus: onBotStatus,
      onLiveError: onLiveError,
      onLiveFrame: onLiveFrame,
      onConnectionChanged: onConnectionChanged,
      onShown: onShown,
    };
  })();

  // -----------------------------------------------------------------------------------------
  // Task 5.6: bot control (owner only). Status panel, commands, friends/war/ignore editor.
  // Every request carries the session cookie and the CSRF token; a 401 means the session ended
  // (back to the login form). Everything dynamic goes through `textContent`, never `innerHTML`,
  // so a nickname (or a bot reply) can never be interpreted as markup. Nothing is written to the
  // console, to storage or to the URL: names stay on this page.
  // -----------------------------------------------------------------------------------------

  var BotPanel = (function () {
    var STATUS_POLL_MS = 2000;
    var TICKS_PER_SECOND = 50; // DDNet server ticks (SERVER_TICK_SPEED); the kill cooldown is 500 ticks = 10 s
    var timer = null;
    var shown = false;
    var lastStatus = null;
    var busy = false;
    // Task 5.7: the site shows the offline demo, not a bot. The status is then "not running"; and when no bot's control socket
    // is there either, the commands are off (the server refuses them as well: `demo_only`). Nothing is ever sent to the demo.
    // A bot that runs without a bridge keeps its control socket, so it stays commandable under the demo's badge.
    var demoShown = false;
    var controlUp = false;

    var KIND_LABELS = {
      friend: "Друзья",
      war: "Война",
      ignore: "Игнор",
      clanfriend: "Клан-друзья",
      clanwar: "Клан-война",
    };
    var KIND_ORDER = ["friend", "war", "ignore", "clanfriend", "clanwar"];
    var MODE_LABELS = { fight: "бой", passive: "пассивный", hold: "стоит", goto: "идёт в точку" };
    var APPLIED_TEXT = {
      applied: "применено к работающему боту (бот перечитал тот же файл)",
      mismatch: "ВНИМАНИЕ: бот перечитал другой файл списков, не тот, что правит сайт (проверьте --relations у бота и сайта): бот может не щадить ваших друзей",
      unverified: "бот ответил, но сверить, что он прочитал тот же файл, не удалось",
      unchanged: "без изменений",
      unavailable: "бот не запущен: список сохранён и вступит в силу при запуске",
      rate_limited: "сохранено, но бот пока не принял (слишком частые команды): нажмите «Перечитать списки в боте»",
      busy: "сохранено, но бот занят: нажмите «Перечитать списки в боте»",
      timeout: "сохранено, но бот не ответил вовремя: нажмите «Перечитать списки в боте»",
      bot_refused: "сохранено, но бот не принял перечитывание списков",
      error: "сохранено, но ответ бота не понят",
    };
    var ERROR_TEXT = {
      unauthenticated: "Сессия закончилась.",
      cross_origin: "Запрос отклонён проверкой источника.",
      missing_csrf: "Нет токена защиты: обновите страницу.",
      bad_csrf: "Токен защиты не подошёл: обновите страницу.",
      bot_unavailable: "Бот не запущен (нет сокета управления).",
      demo_only: "Бот не запущен: сейчас на сайте показ, команды отключены.",
      bot_timeout: "Бот не ответил вовремя.",
      bot_protocol: "Ответ бота не понят.",
      bad_request: "Некорректный запрос.",
      invalid: "Недопустимое значение.",
      invalid_name: "Недопустимое имя.",
      list_full: "Список заполнен.",
      relations_unreadable: "Файл списков не читается (повреждён). Исправьте его вручную: сайт его не перезапишет.",
      relations_write_failed: "Не удалось записать файл списков.",
      json_required: "Нужен JSON.",
    };
    var NAME_DETAIL = {
      empty: "имя пустое после нормализации",
      too_long: "имя длиннее 64 байт",
      control: "в имени управляющие символы",
    };

    function el(id) {
      return document.getElementById(id);
    }

    function setText(id, text) {
      el(id).textContent = text;
    }

    function show(id, text, ok) {
      var node = el(id);
      node.textContent = text;
      node.classList.toggle("ok", ok === true);
      node.classList.toggle("bad", ok === false);
    }

    // Same normalisation as the bot's `fold_name` (trim, collapse whitespace, lower-case, drop one leading "(<digits>)"),
    // for the preview only: the server's answer is what is stored and what the bot matches on.
    function previewFold(name) {
      var folded = name.trim().split(/\s+/).join(" ").toLowerCase();
      // Every leading "(<digits>)", like the bot (the fold must be idempotent: the file is folded again on load).
      var stripped = folded.replace(/^\(\d+\)\s*/, "");
      while (stripped !== folded) {
        folded = stripped;
        stripped = folded.replace(/^\(\d+\)\s*/, "");
      }
      return folded.trim();
    }

    function api(method, path, body) {
      var headers = {};
      var options = { method: method, credentials: "same-origin", headers: headers };
      if (method !== "GET") {
        headers["Content-Type"] = "application/json";
        headers["X-CSRF-Token"] = csrfToken || "";
        options.body = JSON.stringify(body || {});
      }
      return fetch(path, options).then(function (response) {
        return response.json().catch(function () { return {}; }).then(function (data) {
          if (response.status === 401) {
            csrfToken = null;
            disconnectWs();
            showLogin();
          }
          return { status: response.status, data: data };
        });
      });
    }

    function errorText(res) {
      var d = res.data || {};
      if (d.error === "invalid_name" && NAME_DETAIL[d.detail]) {
        return "Недопустимое имя: " + NAME_DETAIL[d.detail] + ".";
      }
      if (d.error === "relations_write_failed" && d.detail === "read_only") {
        return "Файл списков для веб-юнита только для чтения: каталог data/bot создан заново или перенесён после его запуска. Перезапустите юнит: sudo systemctl restart ddnet-ai-web.";
      }
      if (d.code === "rate_limited" || res.status === 429) {
        return "Слишком много команд подряд: подождите секунду.";
      }
      if (d.text) {
        return d.text;
      }
      return ERROR_TEXT[d.error] || "Ошибка " + res.status + ".";
    }

    // ---- status panel ---------------------------------------------------------------------

    function fmtUs(us) {
      if (typeof us !== "number") {
        return "—";
      }
      return us >= 1000 ? (us / 1000).toFixed(2) + " мс" : us + " мкс";
    }

    var lastInfo = null;

    function renderStatus(info) {
      lastInfo = info;
      var dot = el("bot-conn-dot");
      var s = info && info.live ? info.status : null;
      lastStatus = s;
      var cooldownEl = el("kill-cooldown");
      if (!s) {
        dot.classList.remove("dot-on");
        dot.classList.add("dot-off");
        var why = !info
          ? "нет связи с сервером"
          : !info.bridge
            ? "мост к боту не подключён (сайт запущен без --bot-socket)"
            : demoShown && controlUp
              ? "живого статуса нет (моста бота не видно), на сайте показ; сокет управления на месте, команды идут боту"
              : demoShown
                ? "бот не запущен: сейчас на сайте показ (муха на арене), настоящей игры нет"
                : "бот не запущен (нет живого статуса)";
        setText("bot-conn-text", why);
        ["bs-server", "bs-map", "bs-mode", "bs-brain", "bs-target", "bs-wb", "bs-blocks", "bs-deaths", "bs-clips", "bs-latency", "bs-latency2", "bs-identity", "bs-tick"].forEach(function (id) {
          setText(id, "—");
        });
        cooldownEl.textContent = "—";
        updateButtons();
        return;
      }
      dot.classList.toggle("dot-on", !!s.connected);
      dot.classList.toggle("dot-off", !s.connected);
      setText("bot-conn-text", s.connected ? "В игре" : "Бот запущен, но не в игре (подключается или ждёт)");
      setText("bs-server", s.server || "—");
      setText("bs-map", s.map || "—");
      setText("bs-mode", MODE_LABELS[s.mode] || s.mode || "—");
      setText("bs-brain", s.brain || "—");
      setText("bs-target", s.target_tag || "нет");
      setText("bs-wb", (s.wb || "—") + (s.goto ? " · идёт: " + s.goto : ""));
      setText("bs-blocks", (s.blocks | 0) + " / " + (s.blocked_by | 0));
      setText("bs-deaths", (s.deaths | 0) + " / " + (s.self_kills | 0));
      setText("bs-clips", String(s.clips_saved | 0));
      setText("bs-latency", fmtUs(s.decide_p50_us) + " / " + fmtUs(s.decide_p99_us));
      setText("bs-latency2", fmtUs(s.brain_p99_us) + " / " + fmtUs(s.overhead_p99_us));
      setText("bs-identity", [s.name, s.clan, s.skin].filter(Boolean).join(" · ") || "—");
      setText("bs-tick", String(s.tick | 0));
      var cd = s.kill_cooldown_ticks | 0;
      cooldownEl.textContent = cd > 0 ? "доступно через " + Math.ceil(cd / TICKS_PER_SECOND) + " с" : "готово";
      updateButtons();
    }

    function updateButtons() {
      var s = lastStatus;
      var mode = s ? s.mode : null;
      // The bot's line is "WB: <auto|left|right|off>, ..." (docs/formats.md §23).
      var wbMatch = s && s.wb ? /^WB: (auto|left|right|off)\b/.exec(s.wb) : null;
      var wbMode = wbMatch ? wbMatch[1] : null;
      document.querySelectorAll('[data-cmd="mode"]').forEach(function (b) {
        b.classList.toggle("current", mode === b.getAttribute("data-mode"));
      });
      document.querySelectorAll('[data-cmd="wb"]').forEach(function (b) {
        b.classList.toggle("current", wbMode === b.getAttribute("data-mode"));
      });
      var brain = el("cmd-brain");
      if (s && s.brain && document.activeElement !== brain) {
        // The bot reports a descriptive name ("hybrid-none-4ms", "planner-normal-5ms", "hybrid:fly-..."): the select holds the kind.
        brain.value = String(s.brain).split(/[-:]/)[0];
      }
      el("cmd-kill").disabled = locked() || busy || !s || (s.kill_cooldown_ticks | 0) > 0;
    }

    // No bot to command: the demo is on show and no control socket is there.
    function locked() {
      return demoShown && !controlUp;
    }

    // The commands are on only while there is a bot to take them (and while no command is in flight).
    function applyDemoLock() {
      el("cmd-demo-note").hidden = !locked();
      document.querySelectorAll("#bot-cmd-card button, #bot-cmd-card input, #bot-cmd-card select").forEach(function (c) {
        c.disabled = locked() || busy;
      });
      // "Перечитать списки в боте" is a command to the bot; the lists editor itself is a file and stays.
      el("rel-reload").disabled = locked() || busy;
      updateButtons();
    }

    function setControl(up) {
      up = !!up;
      if (up === controlUp) {
        return;
      }
      controlUp = up;
      applyDemoLock();
    }

    function setDemo(on) {
      on = !!on;
      if (on === demoShown) {
        return;
      }
      demoShown = on;
      applyDemoLock();
      if (shown && !lastStatus) {
        renderStatus(lastInfo);
      }
    }

    function pollStatus() {
      api("GET", "/api/bot/status")
        .then(function (res) {
          if (res.status === 200 && res.data && typeof res.data.source === "string") {
            setDemo(res.data.source === "demo"); // the same answer as the WebSocket's, for when that is down
            setControl(res.data.control_socket);
          }
          if (shown) {
            renderStatus(res.status === 200 ? res.data : null);
          }
        })
        .catch(function () {
          if (shown) {
            renderStatus(null);
          }
        });
    }

    // ---- commands -------------------------------------------------------------------------

    function setBusy(on) {
      busy = on;
      document.querySelectorAll("#bot-view button").forEach(function (b) {
        // The «Запуск» card (launch.js) manages its own buttons.
        if (b.id !== "cmd-kill" && !b.closest("#launch-mount")) {
          b.disabled = on;
        }
      });
      applyDemoLock();
    }

    function sendCommand(body, label) {
      if (locked()) {
        show("cmd-result", label + ": бот не запущен, на сайте показ — команды отключены.", false);
        return Promise.resolve();
      }
      setBusy(true);
      show("cmd-result", label + "…", null);
      return api("POST", "/api/bot/command", body)
        .then(function (res) {
          var d = res.data || {};
          if (res.status === 200 && d.ok) {
            show("cmd-result", label + ": " + (d.text || "готово"), true);
          } else {
            show("cmd-result", label + ": " + errorText(res), false);
          }
          pollStatus();
        })
        .catch(function () {
          show("cmd-result", label + ": нет связи с сайтом.", false);
        })
        .then(function () {
          setBusy(false);
        });
    }

    function onCommandClick(event) {
      var target = event.target.closest("[data-cmd]");
      if (!target) {
        return;
      }
      var cmd = target.getAttribute("data-cmd");
      var mode = target.getAttribute("data-mode");
      var label = target.textContent.trim();
      if (cmd === "mode" || cmd === "wb") {
        sendCommand({ type: cmd, mode: mode }, label);
      } else {
        sendCommand({ type: cmd }, label);
      }
    }

    function bindCommands() {
      el("bot-view").addEventListener("click", onCommandClick);
      el("cmd-brain-apply").addEventListener("click", function () {
        sendCommand({ type: "brain", brain: el("cmd-brain").value }, "Мозг " + el("cmd-brain").value);
      });
      el("cmd-kill").addEventListener("click", function () {
        if (window.confirm("Убить бота и возродить? Следующее убийство — не раньше чем через 10 с.")) {
          sendCommand({ type: "kill" }, "Убить");
        }
      });
      el("cmd-clip").addEventListener("click", function () {
        sendCommand({ type: "clip", note: el("cmd-clip-note").value.trim() }, "Клип").then(function () {
          el("cmd-clip-note").value = "";
        });
      });
      el("cmd-goto").addEventListener("click", function () {
        var x = parseInt(el("cmd-goto-x").value, 10);
        var y = parseInt(el("cmd-goto-y").value, 10);
        if (isNaN(x) || isNaN(y)) {
          show("cmd-result", "Идти: введите целые x и y (в тайлах).", false);
          return;
        }
        sendCommand({ type: "goto", x: x, y: y }, "Идти в " + x + ", " + y);
      });
      el("cmd-spec").addEventListener("click", function () {
        if (window.confirm("Отправить бота в наблюдатели? Вернуть можно кнопкой «В игру».")) {
          sendCommand({ type: "spec" }, "В наблюдатели");
        }
      });
    }

    // ---- relations editor -----------------------------------------------------------------

    function renderLists(lists) {
      var root = el("rel-lists");
      while (root.firstChild) {
        root.removeChild(root.firstChild);
      }
      KIND_ORDER.forEach(function (kind) {
        var group = document.createElement("div");
        group.className = "rel-group";
        var title = document.createElement("h3");
        var names = (lists && lists[kind]) || [];
        title.textContent = KIND_LABELS[kind] + " (" + names.length + ")";
        group.appendChild(title);
        if (names.length === 0) {
          var empty = document.createElement("span");
          empty.className = "rel-empty";
          empty.textContent = "пусто";
          group.appendChild(empty);
        } else {
          var ul = document.createElement("ul");
          ul.className = "rel-entries";
          names.forEach(function (name) {
            var li = document.createElement("li");
            var label = document.createElement("span");
            label.textContent = name;
            var remove = document.createElement("button");
            remove.type = "button";
            remove.className = "alt";
            remove.textContent = "×";
            remove.setAttribute("aria-label", "Убрать из списка «" + KIND_LABELS[kind] + "»: " + name);
            remove.addEventListener("click", function () {
              editRelations("remove", kind, name);
            });
            li.appendChild(label);
            li.appendChild(remove);
            ul.appendChild(li);
          });
          group.appendChild(ul);
        }
        root.appendChild(group);
      });
    }

    function loadLists() {
      return api("GET", "/api/bot/relations").then(function (res) {
        if (res.status === 200) {
          renderLists(res.data.lists);
        } else {
          show("rel-result", errorText(res), false);
        }
      });
    }

    function editRelations(op, kind, name) {
      show("rel-result", "…", null);
      return api("POST", "/api/bot/relations", { op: op, kind: kind, name: name })
        .then(function (res) {
          var d = res.data || {};
          if (res.status !== 200 || !d.ok) {
            show("rel-result", errorText(res), false);
            return;
          }
          renderLists(d.lists);
          var what = op === "add" ? "Добавлено" : d.changed ? "Убрано" : "Такого имени в списке не было";
          var text = what + " («" + d.normalised + "», список «" + KIND_LABELS[d.kind] + "»)";
          if (d.moved_from && d.moved_from.length) {
            text += "; снято из: " + d.moved_from.map(function (k) { return KIND_LABELS[k]; }).join(", ");
          }
          text += " — " + (APPLIED_TEXT[d.applied] || d.applied) + ".";
          if (d.applied_text && d.applied !== "unchanged") {
            text += " Ответ бота: " + d.applied_text + ".";
          }
          show("rel-result", text, d.applied === "applied" || d.applied === "unchanged");
          if (op === "add") {
            el("rel-name").value = "";
            updatePreview();
          }
        })
        .catch(function () {
          show("rel-result", "Нет связи с сайтом.", false);
        });
    }

    function updatePreview() {
      var raw = el("rel-name").value;
      var folded = previewFold(raw);
      setText("rel-preview", raw.trim() === "" ? "" : folded === "" ? "после нормализации имя пустое" : "будет сохранено как: «" + folded + "»");
    }

    function bindRelations() {
      el("rel-name").addEventListener("input", updatePreview);
      el("rel-form").addEventListener("submit", function (event) {
        event.preventDefault();
        var name = el("rel-name").value;
        if (previewFold(name) === "") {
          show("rel-result", "Введите непустое имя.", false);
          return;
        }
        editRelations("add", el("rel-kind").value, name);
      });
      el("rel-reload").addEventListener("click", function () {
        sendCommand({ type: "reload_relations" }, "Перечитать списки").then(function () {
          show("rel-result", el("cmd-result").textContent, el("cmd-result").classList.contains("ok"));
        });
      });
    }

    function onShown() {
      shown = true;
      LaunchCard.onShown();
      pollStatus();
      loadLists();
      if (!timer) {
        timer = setInterval(pollStatus, STATUS_POLL_MS);
      }
    }

    function onHidden() {
      shown = false;
      LaunchCard.onHidden();
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    LaunchCard.mount(el("launch-mount"), api);
    bindCommands();
    bindRelations();
    return { onShown: onShown, onHidden: onHidden, setDemo: setDemo };
  })();

  refresh();
})();
