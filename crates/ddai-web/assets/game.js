/* The «Игра» tab (task 5.10): the real DDNet map and tees in a WebGL canvas, the camera, the owner's bot highlighted,
   the scoreboard and HUD, and a read-only chat. The WebSocket messages arrive from app.js; this file owns the tab.

   Everything that came from the game (names, clans, chat text, the map's own names) is only ever written with
   `textContent` or drawn on a canvas, never into markup (a test checks that these files build no markup from text). This file never
   sends chat and has no input for it: the owner's line is the separate say.js (D-094), mounted under the chat panel (task 5.11).

   Portions derived from DDNet (zlib license) and from Wranked1/DDNet-AI (GPL-3.0) src/bot/webView.ts: the name plates,
   the freeze bars, the scoreboard metrics and the HUD layout. This is an altered version, not the original software. */
(function (root) {
  "use strict";

  var host = {
    send: function () {},
    goTab: function () {},
  };

  var $ = function (id) {
    return document.getElementById(id);
  };
  var el = {
    view: $("game-view"),
    stage: $("stage"),
    canvas: $("game-canvas"),
    overlay: $("game-overlay"),
    msg: $("stage-msg"),
    topInfo: $("stage-info"),
    feed: $("game-feed"),
    chatFloat: $("chat-float"),
    board: $("board"),
    boardTitle: $("board-title"),
    boardRows: $("board-rows"),
    follow: $("sel-follow"),
    free: $("btn-free"),
    fit: $("btn-fit"),
    zoom: $("rng-zoom"),
    entities: $("btn-entities"),
    names: $("btn-names"),
    boardBtn: $("btn-board"),
    econ: $("btn-econ"),
    replay: $("replay-bar"),
    replayPlay: $("replay-play"),
    replayPause: $("replay-pause"),
    replayNext: $("replay-next"),
    replaySeek: $("replay-seek"),
    replaySpeed: $("replay-speed"),
    playerList: $("player-list"),
    playerCount: $("player-count"),
    botDot: $("gb-dot"),
    botState: $("gb-state"),
    botGrid: $("gb-grid"),
    toFly: $("gb-fly"),
    toBot: $("gb-bot"),
    chatLog: $("chat-log"),
    chatCount: $("chat-count"),
    chatEmpty: $("chat-empty"),
  };

  var TILE = 32;
  // The most the view may be zoomed out: BlmapChill (1244 x 667 tiles) needs about 27, ChillBlock5 (943 x 1075) about 33.
  var MAX_ZOOM = 50;
  // The last frame's name plates as {l, r, t, b, fs} boxes (the e2e test checks that none overlaps).
  var lastPlates = [];
  var TICK_MS = 20; // the server runs 50 ticks a second
  var MAX_DPR = 2;
  var MAX_PIXELS = 4.6e6; // drawing buffer cap (about 1080p at dpr 2 is 8.3 M: scaled down)
  var SOLID = { 1: true, 3: true }; // game layer: solid and no-hook walls
  var FREEZE_TILES = { 9: true, 12: true, 144: true }; // freeze, deep freeze, live freeze
  var FREEZE_MS = 3000; // the default freeze time: the bar's span (the real remaining time is not on the wire)
  var BOT_COLOR = "#ffd23f";
  var FONT = '"DejaVu Sans DDNet","DejaVu Sans","Segoe UI",system-ui,sans-serif';

  // ---------------------------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------------------------

  var renderer = null;
  var kit = null;
  var rendererFailed = false;
  var overlayCtx = null;

  var mapMeta = null; // {sha256, name, w, h}
  var mapGen = 0; // bumped on every new map: a stale fetch is dropped
  var sceneInfo = null; // {width, height} of the loaded scene's game layer
  var parsedScene = null; // parsed, waiting for the renderer
  var gameTiles = null; // game layer {w, h, tiles}
  var mapStatus = "none"; // none | loading | ready | fallback | failed
  var assetsOk = null; // null unknown, true once the sheets came, false when the data dir is missing
  var players = {}; // id -> {id, name, team, clan, skin, cc, cb, cf, country, score, ping}
  var botStatus = null;
  var ownId = -1;
  var sourceKind = null; // live | demo | none
  var frames = []; // [{tick, at, chars: {id: char}, list: [char]}]
  var clock = null; // tick clock offset (ms)
  var frameTimes = []; // arrival times for the frames/s readout
  var chat = []; // lines, newest last
  var chatDirty = true;
  var feedItems = []; // {text, kind, at}
  var freezeSince = {}; // id -> ms when the tee was last seen frozen starting
  var wasFrozen = {};
  var flakes = [];
  var flakeClock = 0;

  var view = {
    follow: "bot", // "bot" | id (number) | null (free camera)
    cam: { x: 0, y: 0 },
    camSet: false,
    zoom: 1,
    names: true,
    entities: 0, // 0 map, 1 map and entities, 2 entities
    board: false,
    econ: false,
    fitPending: false,
    fixedScale: false, // tests: no adaptive resolution
  };
  var renderScale = 1; // share of the screen's pixels the map is drawn at (adaptive, see draw())
  var fpsAcc = 0;
  var fpsN = 0;
  var fps = 0;
  var fpsAt = 0;
  var lastDraw = 0;
  var lastStats = null; // what the last frame drew: tile layers, quad layers' quads, draw calls, sprite vertices
  var renderTimes = [];
  root.__ddaiRenderTimes = renderTimes;
  var envEpoch = performance.now();

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  function make(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  function safeStorage(op, key, value) {
    try {
      if (op === "get") return root.localStorage.getItem(key);
      root.localStorage.setItem(key, value);
    } catch (e) {
      /* private mode or blocked storage: the tab works without it */
    }
    return null;
  }

  // ---------------------------------------------------------------------------------------------
  // Binary `live` frames (docs/formats.md §15.3)
  // ---------------------------------------------------------------------------------------------

  function decodeFrame(buffer) {
    var dv = new DataView(buffer);
    if (buffer.byteLength < 12 || dv.getUint32(0, true) !== 0x464c5744 /* "DWLF" */ || dv.getUint8(4) !== 1) {
      return null;
    }
    var tick = dv.getUint32(6, true);
    var count = dv.getUint16(10, true);
    var list = [];
    var byId = {};
    var off = 12;
    for (var i = 0; i < count; i++) {
      if (off + 26 > buffer.byteLength) break;
      var flags = dv.getUint8(off + 1);
      var hooked = dv.getInt8(off + 24);
      var c = {
        id: dv.getUint8(off),
        alive: (flags & 1) !== 0,
        frozen: (flags & 2) !== 0,
        deep: (flags & 4) !== 0,
        live: (flags & 8) !== 0,
        hookOut: (flags & 16) !== 0,
        team: dv.getUint8(off + 2),
        weapon: dv.getUint8(off + 3),
        x: dv.getInt32(off + 4, true),
        y: dv.getInt32(off + 8, true),
        aimX: dv.getInt16(off + 12, true),
        aimY: dv.getInt16(off + 14, true),
        hookX: dv.getInt32(off + 16, true),
        hookY: dv.getInt32(off + 20, true),
        hooked: hooked < 0 ? -1 : hooked,
        vx: 0,
        vy: 0,
      };
      list.push(c);
      byId[c.id] = c;
      off += 26;
    }
    return { tick: tick, chars: byId, list: list };
  }

  function onLiveFrame(buffer) {
    var f = decodeFrame(buffer);
    if (!f) {
      return;
    }
    var now = performance.now();
    var last = frames[frames.length - 1];
    if (last && f.tick < last.tick) {
      frames = []; // the clock went back (a seek, another game): start over
      clock = null;
    }
    f.at = now;
    // Velocity per tick from the previous frame (the wire carries none).
    var prev = frames[frames.length - 1];
    if (prev && f.tick > prev.tick) {
      var dt = f.tick - prev.tick;
      f.list.forEach(function (c) {
        var p = prev.chars[c.id];
        if (p && Math.abs(c.x - p.x) < 400 && Math.abs(c.y - p.y) < 400) {
          c.vx = (c.x - p.x) / dt;
          c.vy = (c.y - p.y) / dt;
        }
      });
    }
    if (last && f.tick === last.tick) {
      frames[frames.length - 1] = f;
    } else {
      frames.push(f);
    }
    if (frames.length > 12) frames.shift();
    clock = tickClock(clock, f.tick, now);
    frameTimes.push(now);
    // Freeze starts, for the bars and the feed.
    f.list.forEach(function (c) {
      var was = !!wasFrozen[c.id];
      if (c.frozen && !was) freezeSince[c.id] = now;
      wasFrozen[c.id] = c.frozen;
    });
  }

  /** The tick clock (ms offset between the server's ticks and this page's time): jumps are taken at once, drift slowly. */
  function tickClock(prev, tick, nowMs) {
    var o = tick * TICK_MS - nowMs;
    if (prev === null || Math.abs(o - prev) > 400) return o;
    return o > prev ? o : prev + (o - prev) * 0.05;
  }

  /** The characters at render time: positions between the two frames around (now - 100 ms), velocity and hook from the later one. */
  function currentChars(now) {
    var n = frames.length;
    if (n === 0) return null;
    var newest = frames[n - 1];
    if (n === 1 || clock === null) {
      return { tick: newest.tick, list: newest.list, byId: newest.chars };
    }
    var rt = (now + clock - 100) / TICK_MS;
    var a = frames[0];
    var b = frames[0];
    for (var i = 1; i < n; i++) {
      if (frames[i].tick >= rt) {
        a = frames[i - 1];
        b = frames[i];
        break;
      }
      a = frames[i];
      b = frames[i];
    }
    var span = b.tick - a.tick;
    var k = span > 0 ? clamp((rt - a.tick) / span, 0, 1) : 1;
    var out = [];
    var byId = {};
    b.list.forEach(function (c) {
      var o = a.chars[c.id];
      var m = c;
      if (o && Math.hypot(c.x - o.x, c.y - o.y) < 300) {
        m = Object.assign({}, c);
        m.x = o.x + (c.x - o.x) * k;
        m.y = o.y + (c.y - o.y) * k;
        // The hook is only blended while it was out in both frames (a retracted hook's position is meaningless).
        if (o.hookOut && c.hookOut) {
          m.hookX = o.hookX + (c.hookX - o.hookX) * k;
          m.hookY = o.hookY + (c.hookY - o.hookY) * k;
        }
        m.aimX = o.aimX + (c.aimX - o.aimX) * k;
        m.aimY = o.aimY + (c.aimY - o.aimY) * k;
      }
      out.push(m);
      byId[m.id] = m;
    });
    return { tick: a.tick + span * k, list: out, byId: byId };
  }

  // ---------------------------------------------------------------------------------------------
  // The renderer, the assets and the map
  // ---------------------------------------------------------------------------------------------

  function ensureRenderer() {
    if (renderer || rendererFailed) {
      return renderer;
    }
    try {
      renderer = root.DDMap.createRenderer(el.canvas);
    } catch (e) {
      renderer = null;
    }
    if (!renderer) {
      rendererFailed = true;
      showMessage("Не удалось запустить WebGL2: карту нарисовать нечем. Обновите браузер.", true);
      return null;
    }
    kit = root.DDTee.createTeeKit(renderer);
    kit.startLoading();
    overlayCtx = el.overlay.getContext("2d");
    loadEntityImages();
    if (parsedScene) {
      applyScene(parsedScene);
    }
    return renderer;
  }

  function loadEntityImages() {
    Promise.all([root.DDTee.loadPixels("/assets/editor/entities_clear/ddnet.png"), root.DDTee.loadPixels("/assets/editor/speed_arrow_array.png")]).then(function (r) {
      assetsOk = !!(r[0] || r[1]);
      if (renderer) {
        renderer.setEntityImages(r[0] && r[0].data, r[0] && r[0].w, r[0] && r[0].h, r[1] && r[1].data, r[1] && r[1].w, r[1] && r[1].h);
      }
    });
  }

  function showMessage(text, bad) {
    if (!text) {
      el.msg.hidden = true;
      return;
    }
    el.msg.hidden = false;
    el.msg.textContent = text;
    el.msg.classList.toggle("bad", !!bad);
  }

  function applyScene(parsed) {
    if (!renderer) {
      parsedScene = parsed;
      return;
    }
    parsedScene = null;
    var s = renderer.setScene(parsed);
    var g = s.gameLayer;
    gameTiles = g ? { w: g.w, h: g.h, tiles: g.tiles } : null;
    sceneInfo = g ? { width: g.w, height: g.h } : { width: mapMeta ? mapMeta.w : 0, height: mapMeta ? mapMeta.h : 0 };
    if (!view.camSet) {
      view.fitPending = view.follow === null;
    }
  }

  function fetchWithRetry(url, tries, gen) {
    return fetch(url, { credentials: "same-origin" }).then(function (r) {
      if (gen !== mapGen) {
        throw new Error("stale");
      }
      if (r.status === 404 && tries > 0) {
        // The source resolves a map in the background: a 404 right after the `map` message means "not yet".
        return new Promise(function (resolve) {
          setTimeout(resolve, 500);
        }).then(function () {
          return fetchWithRetry(url, tries - 1, gen);
        });
      }
      if (!r.ok) {
        throw new Error("http " + r.status);
      }
      return r.arrayBuffer();
    });
  }

  function loadMap(meta) {
    var gen = ++mapGen;
    mapStatus = "loading";
    sceneInfo = null;
    gameTiles = null;
    parsedScene = null;
    if (renderer) renderer.dispose();
    showMessage("Загрузка карты «" + meta.name + "»…");
    fetchWithRetry("/api/map/" + meta.sha256 + "/scene", 6, gen)
      .then(function (buf) {
        return root.DDMap.inflateRaw(new Uint8Array(buf));
      })
      .then(function (inflated) {
        if (gen !== mapGen) throw new Error("stale");
        var parsed = root.DDMap.parseScene(inflated);
        ensureRenderer();
        applyScene(parsed);
        mapStatus = "ready";
        showMessage("");
        loadImages(parsed.meta, gen);
        if (view.follow === null || !view.camSet) view.fitPending = view.follow === null;
      })
      .catch(function (err) {
        if (err && err.message === "stale") return;
        // No real layers (a synthetic map, or the scene could not be built): the coarse kinds grid still shows the geometry.
        loadFallback(meta, gen);
      });
  }

  function loadImages(meta, gen) {
    (meta.images || []).forEach(function (im, i) {
      var done = function (px, w, h) {
        if (gen !== mapGen || !renderer) return;
        renderer.setImage(i, px, w, h);
      };
      if (im.x) {
        if (!im.n) {
          done(null);
          return;
        }
        root.DDTee.loadPixels("/assets/mapres/" + encodeURIComponent(im.n) + ".png").then(function (px) {
          done(px ? px.data : null, px && px.w, px && px.h);
        });
      } else if (im.d) {
        fetchWithRetry("/api/map/" + mapMeta.sha256 + "/image/" + i, 3, gen)
          .then(function (buf) {
            return root.DDMap.inflateRaw(new Uint8Array(buf));
          })
          .then(function (raw) {
            var dv = new DataView(raw);
            var w = dv.getUint32(0, true);
            var h = dv.getUint32(4, true);
            if (w * h * 4 + 8 !== raw.byteLength) throw new Error("bad image");
            done(new Uint8ClampedArray(raw, 8, w * h * 4), w, h);
          })
          .catch(function () {
            done(null);
          });
      } else {
        done(null);
      }
    });
  }

  // The coarse "kind" grid (docs/formats.md §15.2) drawn as flat tile layers, for a map whose real layers are not available.
  var KIND_COLORS = [
    null,
    [107, 114, 128, 255], // solid
    [139, 111, 71, 255], // no-hook
    [220, 38, 38, 255], // death
    [30, 58, 138, 255], // deep freeze
    [56, 189, 248, 255], // freeze
    [13, 148, 136, 255],
    [134, 239, 172, 255], // unfreeze
    [168, 85, 247, 255],
    [217, 70, 239, 255],
    [139, 92, 246, 255],
    [249, 115, 22, 255],
    [234, 179, 8, 255],
    [236, 72, 153, 255],
    [146, 64, 14, 255],
    [74, 222, 128, 255],
  ];

  function loadFallback(meta, gen) {
    fetchWithRetry("/api/map/" + meta.sha256, 2, gen)
      .then(function (buf) {
        var dv = new DataView(buf);
        var w = dv.getUint32(0, true);
        var h = dv.getUint32(4, true);
        var len = dv.getUint32(8, true);
        return root.DDMap.inflateRaw(new Uint8Array(buf, 12, len)).then(function (inflated) {
          return { w: w, h: h, kinds: new Uint8Array(inflated) };
        });
      })
      .then(function (k) {
        if (gen !== mapGen) throw new Error("stale");
        var count = k.w * k.h;
        var blobParts = [];
        var layers = [];
        var offset = 0;
        function addLayer(role, color, pick) {
          var t = new Uint8Array(count * 2);
          var any = false;
          for (var i = 0; i < count; i++) {
            if (pick(k.kinds[i])) {
              t[i * 2] = 1;
              any = true;
            }
          }
          if (!any && role !== "game") return;
          layers.push({ k: "t", r: role, w: k.w, h: k.h, d: 0, c: color, ce: -1, co: 0, i: -1, o: offset, a: -1 });
          blobParts.push(t);
          offset += t.length;
        }
        KIND_COLORS.forEach(function (c, kind) {
          if (c) addLayer("visual", c, function (v) { return v === kind; });
        });
        addLayer("game", [255, 255, 255, 255], function (v) { return v === 1 || v === 2; });
        var blob = new Uint8Array(offset);
        var at = 0;
        blobParts.forEach(function (p) {
          blob.set(p, at);
          at += p.length;
        });
        var meta2 = { v: 1, game: { w: k.w, h: k.h }, images: [], env: [], groups: [{ ox: 0, oy: 0, px: 100, py: 100, clip: null, layers: layers }] };
        ensureRenderer();
        applyScene({ meta: meta2, blob: blob, blobI32: new Int32Array(blob.buffer, 0, blob.length >> 2) });
        mapStatus = "fallback";
        showMessage("Слои карты недоступны: показана только геометрия.");
      })
      .catch(function (err) {
        if (err && err.message === "stale") return;
        mapStatus = "failed";
        showMessage("Не удалось загрузить карту.", true);
      });
  }

  // ---------------------------------------------------------------------------------------------
  // Camera
  // ---------------------------------------------------------------------------------------------

  function stageSize() {
    var r = el.stage.getBoundingClientRect();
    return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
  }

  function baseViewWidth(aspect) {
    return root.DDMap.groupView(0, 0, 1, aspect, 100, 100, 0, 0)[2];
  }

  function fitZoom() {
    var s = stageSize();
    var aspect = s.w / s.h;
    var base = root.DDMap.groupView(0, 0, 1, aspect, 100, 100, 0, 0);
    var mw = (sceneInfo ? sceneInfo.width : 100) * TILE;
    var mh = (sceneInfo ? sceneInfo.height : 100) * TILE;
    return clamp(Math.max(mw / base[2], mh / base[3]) * 1.02, 0.1, MAX_ZOOM);
  }

  function setZoom(z, keepWorld, sx, sy) {
    var s = stageSize();
    var before = keepWorld ? screenToWorld(sx, sy, s) : null;
    view.zoom = clamp(z, 0.12, MAX_ZOOM);
    if (before) {
      var after = screenToWorld(sx, sy, s);
      view.cam.x += before.x - after.x;
      view.cam.y += before.y - after.y;
    }
    el.zoom.value = String(Math.round(Math.log(view.zoom) * 100));
  }

  function plainView(s) {
    return root.DDMap.groupView(view.cam.x, view.cam.y, view.zoom, s.w / s.h, 100, 100, 0, 0);
  }

  function screenToWorld(sx, sy, s) {
    var v = plainView(s);
    return { x: v[0] + (sx / s.w) * v[2], y: v[1] + (sy / s.h) * v[3] };
  }

  function followTarget(cur) {
    if (!cur) return null;
    if (view.follow === "bot") {
      return ownId >= 0 ? cur.byId[ownId] || null : null;
    }
    if (typeof view.follow === "number") {
      return cur.byId[view.follow] || null;
    }
    return null;
  }

  function setFollow(f) {
    view.follow = f;
    view.camSet = false;
    syncControls();
    renderPlayers();
  }

  // ---------------------------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------------------------

  function solidAt(x, y) {
    if (!gameTiles) return false;
    var tx = Math.floor(x / TILE);
    var ty = Math.floor(y / TILE);
    if (tx < 0 || ty < 0 || tx >= gameTiles.w || ty >= gameTiles.h) return true;
    return !!SOLID[gameTiles.tiles[(ty * gameTiles.w + tx) * 2]];
  }

  function freezeTileAt(x, y) {
    if (!gameTiles) return false;
    var tx = Math.floor(x / TILE);
    var ty = Math.floor(y / TILE);
    if (tx < 0 || ty < 0 || tx >= gameTiles.w || ty >= gameTiles.h) return false;
    return !!FREEZE_TILES[gameTiles.tiles[(ty * gameTiles.w + tx) * 2]];
  }

  function playerInfo(id) {
    return players[id] || { skin: "default", cc: false, cb: 0, cf: 0 };
  }

  function drawTees(cur, now, dt) {
    if (!cur) return;
    var order = cur.list.slice();
    // The followed tee is drawn on top of the others, the owner's bot last of all.
    order.sort(function (a, b) {
      return (a.id === ownId ? 1 : 0) - (b.id === ownId ? 1 : 0);
    });
    var bars = [];
    order.forEach(function (c) {
      if (c.hookOut) {
        var hx = c.hookX;
        var hy = c.hookY;
        if (c.hooked >= 0 && cur.byId[c.hooked]) {
          hx = cur.byId[c.hooked].x;
          hy = cur.byId[c.hooked].y;
        }
        var info0 = playerInfo(c.id);
        var ninja0 = c.weapon === 5 || c.frozen;
        kit.renderHook(kit.skin(info0, ninja0), c.x, c.y, hx, hy, 1);
      }
    });
    order.forEach(function (c) {
      var info = playerInfo(c.id);
      var ninja = c.weapon === 5 || c.frozen;
      var tex = kit.skin(info, ninja);
      var len = Math.hypot(c.aimX, c.aimY);
      var dx = len > 0.5 ? c.aimX / len : c.vx >= 0 ? 1 : -1;
      var dy = len > 0.5 ? c.aimY / len : 0;
      var inAir = !solidAt(c.x, c.y + 16);
      var dir = c.vx > 0.4 ? 1 : c.vx < -0.4 ? -1 : 0;
      var anim = root.DDTee.teeAnim(c.vx, inAir, dir, c.x, 5, c.frozen ? -1 : c.weapon);
      if (tex) {
        if (!c.frozen) kit.renderWeapon(tex, c.weapon, c.x, c.y, anim, dx, dy, 5);
        kit.renderTee(tex, anim, c.frozen ? 1 : 0, dx, dy, c.x, c.y, 1, false);
      }
      if (c.frozen && !c.deep && !freezeTileAt(c.x, c.y)) {
        var left = 1 - (now - (freezeSince[c.id] || now)) / FREEZE_MS;
        if (left > 0) bars.push({ x: c.x, y: c.y, p: clamp(left, 0, 1) });
      }
    });
    // Freeze particles: a few snowflakes rise off every frozen tee.
    flakeClock += dt;
    if (flakeClock >= 0.2) {
      flakeClock %= 0.2;
      cur.list.forEach(function (c) {
        if (!c.frozen) return;
        flakes.push({ x: c.x + (Math.random() - 0.5) * 32, y: c.y - 4 + (Math.random() - 0.5) * 32, vx: 0, vy: 0, g: Math.random() * 250, rot: Math.random() * Math.PI * 2, life: 0, size: (0.5 + Math.random()) * 16 });
      });
      if (flakes.length > 400) flakes.splice(0, flakes.length - 400);
    }
    flakes = flakes.filter(function (p) {
      return p.life < 1.5;
    });
    var steps = 1;
    flakes.forEach(function (p) {
      root.DDTee.flakeStep(p, dt, steps);
      var a = Math.min(1, p.life / 1.5);
      kit.renderSnowflake(p.x, p.y, p.size * (1 - 0.5 * a), p.rot, Math.max(0, 1 - a));
    });
    bars.forEach(function (b) {
      kit.renderFreezeBar(b.x, b.y, b.p);
    });
  }

  // A tee whose skin is not there (yet, or no DDNet data directory): flat shapes on the overlay, so it is never invisible.
  function drawPlainTees(ctx, cur, s, plain) {
    if (!cur) return;
    var k = s.w / plain[2];
    cur.list.forEach(function (c) {
      var ninja = c.weapon === 5 || c.frozen;
      if (kit && kit.skin(playerInfo(c.id), ninja)) return;
      var x = (c.x - plain[0]) * k;
      var y = (c.y - plain[1]) * k;
      var r = 28 * k;
      ctx.fillStyle = "rgba(0,0,0,.55)";
      ctx.beginPath();
      ctx.ellipse(x - 7 * k, y + 12 * k, 13 * k, 7 * k, 0, 0, 6.2832);
      ctx.ellipse(x + 7 * k, y + 12 * k, 13 * k, 7 * k, 0, 0, 6.2832);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y - 4 * k, r, 0, 6.2832);
      ctx.fillStyle = c.frozen ? "#4d8fc9" : c.id === ownId ? BOT_COLOR : "#c9ced8";
      ctx.fill();
      ctx.lineWidth = Math.max(1, 3 * k);
      ctx.strokeStyle = "rgba(0,0,0,.6)";
      ctx.stroke();
    });
  }

  function roundRect(ctx, x, y, w, h, r) {
    var rr = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function displayName(c) {
    var p = players[c.id];
    return p && p.name ? p.name : "#" + c.id;
  }

  /**
   * Name plates at a constant screen size. Each plate is a stack of lines (name, then the clan of the followed tee, then the
   * owner's "БОТ" badge); plates that would overlap are pushed up one after another, the owner's and the followed tee's first, so
   * they stay put and the others make way. With a tee followed, the others are drawn faint.
   */
  function drawPlates(ctx, plates, anyFollowed) {
    if (plates.length === 0) {
      lastPlates = [];
      return;
    }
    plates.sort(function (a, b) {
      var pa = (a.own ? 0 : 1) + (a.followed ? 0 : 2);
      var pb = (b.own ? 0 : 1) + (b.followed ? 0 : 2);
      return pa - pb || a.dist - b.dist;
    });
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.lineJoin = "round";
    var placed = [];
    plates.forEach(function (pl) {
      var lines = [{ text: pl.label, size: pl.fs, color: pl.own ? BOT_COLOR : "#fff", font: pl.fs.toFixed(1) + "px " + FONT, pill: false }];
      if (pl.clan && pl.followed) {
        lines.push({ text: pl.clan, size: pl.fs * 0.85, color: "rgba(255,255,255,.88)", font: (pl.fs * 0.85).toFixed(1) + "px " + FONT, pill: false });
      }
      if (pl.own) {
        lines.push({ text: "БОТ", size: pl.fs * 0.8, color: "#1b1503", font: "bold " + (pl.fs * 0.62).toFixed(1) + "px " + FONT, pill: true });
      }
      var width = 0;
      var height = 0;
      lines.forEach(function (ln) {
        ctx.font = ln.font;
        ln.w = ctx.measureText(ln.text).width + (ln.pill ? pl.fs * 0.7 : 0);
        ln.h = ln.size * 1.05;
        width = Math.max(width, ln.w);
        height += ln.h;
      });
      var box = { l: pl.x - width / 2 - 2, r: pl.x + width / 2 + 2, t: 0, b: 0 };
      var bottom = pl.y;
      for (var tries = 0; tries < 8; tries++) {
        box.b = bottom + 2;
        box.t = bottom - height;
        var hit = null;
        for (var i = 0; i < placed.length; i++) {
          var q = placed[i];
          if (box.l < q.r && box.r > q.l && box.t < q.b && box.b > q.t) {
            hit = q;
            break;
          }
        }
        if (!hit) break;
        bottom = hit.t - 3;
      }
      placed.push({ l: box.l, r: box.r, t: box.t, b: box.b, fs: pl.fs });
      ctx.globalAlpha = pl.own || pl.followed || !anyFollowed ? 1 : 0.5;
      var y = bottom;
      lines.forEach(function (ln) {
        ctx.font = ln.font;
        if (ln.pill) {
          var th = ln.size * 0.95;
          ctx.fillStyle = BOT_COLOR;
          roundRect(ctx, pl.x - ln.w / 2, y - th * 0.78, ln.w, th, th / 2);
          ctx.fill();
          ctx.fillStyle = ln.color;
          ctx.fillText(ln.text, pl.x, y - th * 0.06);
        } else {
          ctx.lineWidth = 3;
          ctx.strokeStyle = "rgba(0,0,0,.55)";
          ctx.strokeText(ln.text, pl.x, y);
          ctx.fillStyle = ln.color;
          ctx.fillText(ln.text, pl.x, y);
        }
        y -= ln.h;
      });
    });
    ctx.globalAlpha = 1;
    lastPlates = placed;
  }

  /** Name plates, the owner's highlight and the HUD, on the 2D overlay (world units mapped with the plain view). */
  function drawOverlay(ctx, cur, s, plain, dpr, now) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, s.w, s.h);
    var k = s.w / plain[2];
    if (cur) {
      drawPlainTees(ctx, cur, s, plain);
      var own = ownId >= 0 ? cur.byId[ownId] : null;
      var followedId = view.follow === "bot" ? (ownId >= 0 ? ownId : null) : view.follow;
      var plates = [];
      // The owner's bot: a pulsing ring, a pointer and a label in its own colour.
      cur.list.forEach(function (c) {
        var x = (c.x - plain[0]) * k;
        var y = (c.y - plain[1]) * k;
        if (x < -300 || x > s.w + 300 || y < -200 || y > s.h + 200) return;
        var isOwn = c.id === ownId;
        var tiny = 64 * k < 10;
        if (isOwn) {
          var pulse = 0.5 + 0.5 * Math.sin(now / 280);
          var r = (40 + 4 * pulse) * k;
          ctx.lineWidth = Math.max(2, 3.5 * k);
          ctx.strokeStyle = BOT_COLOR;
          ctx.globalAlpha = 0.55 + 0.4 * pulse;
          ctx.beginPath();
          ctx.arc(x, y - 4 * k, Math.max(r, 9), 0, 6.2832);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        if (tiny) {
          ctx.beginPath();
          ctx.arc(x, y, isOwn ? 6 : 4, 0, 6.2832);
          ctx.fillStyle = isOwn ? BOT_COLOR : c.frozen ? "#4d8fc9" : "#fff";
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = "rgba(0,0,0,.75)";
          ctx.stroke();
          return;
        }
        if (!view.names || k < 0.25) return;
        // The plate's bottom is 30 units over the tee (cl_nameplates_offset), but its size is constant on the screen (a plate
        // that grows with the zoom, as DDNet draws it, buries the page when several tees are close): at most 16 px.
        plates.push({
          x: x,
          y: y - clamp(33 * k, 12, 120),
          label: displayName(c),
          clan: players[c.id] && players[c.id].clan ? players[c.id].clan : "",
          fs: clamp(28 * k, 9, 16),
          own: isOwn,
          followed: c.id === followedId,
          dist: Math.hypot(c.x - plain[0] - plain[2] / 2, c.y - plain[1] - plain[3] / 2),
        });
      });
      drawPlates(ctx, plates, followedId !== null);
      // An arrow at the edge of the screen when the owner is out of sight.
      if (own) {
        var ox = (own.x - plain[0]) * k;
        var oy = (own.y - plain[1]) * k;
        if (ox < 0 || ox > s.w || oy < 0 || oy > s.h) {
          var cx = s.w / 2;
          var cy = s.h / 2;
          var ang = Math.atan2(oy - cy, ox - cx);
          var edge = Math.min(Math.abs((s.w / 2 - 30) / Math.cos(ang || 1e-6)), Math.abs((s.h / 2 - 30) / Math.sin(ang || 1e-6)));
          var ax = cx + Math.cos(ang) * edge;
          var ay = cy + Math.sin(ang) * edge;
          ctx.save();
          ctx.translate(ax, ay);
          ctx.rotate(ang);
          ctx.fillStyle = BOT_COLOR;
          ctx.strokeStyle = "rgba(0,0,0,.7)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(14, 0);
          ctx.lineTo(-10, -11);
          ctx.lineTo(-5, 0);
          ctx.lineTo(-10, 11);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          ctx.restore();
          ctx.font = "bold 11px " + FONT;
          ctx.textAlign = "center";
          ctx.fillStyle = BOT_COLOR;
          ctx.fillText("БОТ", ax - Math.cos(ang) * 26, ay - Math.sin(ang) * 26 + 4);
        }
      }
      drawEntityNumbers(ctx, s, plain, k);
      drawHud(ctx, cur, s);
    }
    ctx.textAlign = "left";
  }

  // Tele, switch and speedup numbers over the entity tiles (the client's "text entities"), only when zoomed in enough.
  function drawEntityNumbers(ctx, s, plain, k) {
    if (view.entities === 0 || !renderer || !renderer.scene() || 32 * k < 22) return;
    var scene = renderer.scene();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = clamp(32 * k * 0.34, 9, 26).toFixed(0) + "px " + FONT;
    scene.groups.forEach(function (g) {
      g.layers.forEach(function (l) {
        if (l.kind !== "t" || !l.aux) return;
        var gv = root.DDMap.groupView(view.cam.x, view.cam.y, view.zoom, s.w / s.h, g.px, g.py, g.ox, g.oy);
        var kk = s.w / gv[2];
        var x0 = Math.max(0, Math.floor(gv[0] / TILE));
        var y0 = Math.max(0, Math.floor(gv[1] / TILE));
        var x1 = Math.min(l.w - 1, Math.ceil((gv[0] + gv[2]) / TILE));
        var y1 = Math.min(l.h - 1, Math.ceil((gv[1] + gv[3]) / TILE));
        for (var y = y0; y <= y1; y++) {
          for (var x = x0; x <= x1; x++) {
            var i = y * l.w + x;
            var type = l.tiles[i * 2];
            if (!type) continue;
            var px = (x * TILE + 16 - gv[0]) * kk;
            var py = (y * TILE + 16 - gv[1]) * kk;
            var txt = null;
            if (l.role === "tele") txt = type !== 31 && type !== 63 ? String(l.aux[i]) : null;
            else if (l.role === "switch") txt = l.aux[i * 2] ? String(l.aux[i * 2]) : null;
            else if (l.role === "speedup") txt = l.aux[i * 4] ? String(l.aux[i * 4]) : null;
            if (!txt || txt === "0") continue;
            ctx.lineWidth = 3;
            ctx.strokeStyle = "rgba(0,0,0,.7)";
            ctx.strokeText(txt, px, py);
            ctx.fillStyle = "#fff";
            ctx.fillText(txt, px, py);
          }
        }
      });
    });
    ctx.textBaseline = "alphabetic";
  }

  function drawHud(ctx, cur, s) {
    // The followed tee's weapon, top left, like the client's HUD.
    var game = kit && kit.sheet("game");
    var who = followTarget(cur) || (ownId >= 0 ? cur.byId[ownId] : null) || cur.list[0];
    if (who && game && game.px && s.w >= 560) {
      var u = Math.max(0.8, s.h / 300);
      var wp = clamp(who.weapon, 0, 5);
      var spec = root.DDTee.SPRITES.weapons[wp];
      var r = root.DDTee.rect(spec.body, root.DDTee.SPRITES.game);
      var gw = game.px.w;
      var gh = game.px.h;
      var sc = [spec.body[2] / Math.hypot(spec.body[2], spec.body[3]), spec.body[3] / Math.hypot(spec.body[2], spec.body[3])];
      var ww = spec.size * sc[0] * 0.25 * u;
      var hh = spec.size * sc[1] * 0.25 * u;
      if (!hudCanvas) {
        hudCanvas = document.createElement("canvas");
        hudCanvas.width = gw;
        hudCanvas.height = gh;
        hudCanvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(game.px.data), gw, gh), 0, 0);
      }
      ctx.save();
      ctx.translate(26 * u, 56 * u);
      ctx.rotate((Math.PI * 7) / 4);
      ctx.globalAlpha = 0.95;
      ctx.drawImage(hudCanvas, r[0] * gw, r[1] * gh, (r[2] - r[0]) * gw, (r[3] - r[1]) * gh, -ww / 2, -hh / 2, ww, hh);
      ctx.restore();
      ctx.globalAlpha = 1;
    }
    ctx.font = "12px " + FONT;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    var self = ownId >= 0 ? players[ownId] : null;
    var line = fps + " FPS" + (self && self.ping > 0 ? "  ·  пинг " + self.ping : "");
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,.6)";
    ctx.strokeText(line, s.w - 10, 8);
    ctx.fillStyle = "#fff";
    ctx.fillText(line, s.w - 10, 8);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
  }
  var hudCanvas = null;

  function resizeCanvases(s, dpr, glDpr) {
    var w = Math.max(1, Math.round(s.w * dpr));
    var h = Math.max(1, Math.round(s.h * dpr));
    if (renderer) renderer.resize(Math.max(1, Math.round(s.w * glDpr)), Math.max(1, Math.round(s.h * glDpr)));
    if (el.overlay.width !== w || el.overlay.height !== h) {
      el.overlay.width = w;
      el.overlay.height = h;
    }
  }

  function draw(now) {
    var dt = lastDraw ? Math.min(0.1, (now - lastDraw) / 1000) : 0.016;
    // The camera's own step may be longer: after a slow frame it catches up instead of trailing a moving tee.
    var camDt = lastDraw ? Math.min(1, (now - lastDraw) / 1000) : 0.016;
    lastDraw = now;
    fpsAcc += dt;
    fpsN++;
    if (now - fpsAt > 500) {
      fps = Math.round(fpsN / Math.max(0.001, fpsAcc));
      // Adaptive resolution: a screen that cannot keep 25 frames a second is drawn at a lower resolution (the map's canvas only;
      // the name plates and HUD stay sharp) and goes back up when there is room again.
      if (!view.fixedScale) {
        if (fps < 22 && renderScale > 0.4) renderScale = Math.max(0.4, renderScale * 0.8);
        else if (fps > 50 && renderScale < 1) renderScale = Math.min(1, renderScale * 1.15);
      }
      fpsAcc = 0;
      fpsN = 0;
      fpsAt = now;
      updateTopInfo();
    }
    var s = stageSize();
    var dpr = Math.min(MAX_DPR, root.devicePixelRatio || 1);
    if (s.w * s.h * dpr * dpr > MAX_PIXELS) dpr = Math.sqrt(MAX_PIXELS / (s.w * s.h));
    var glDpr = dpr * renderScale;
    if (!renderer) {
      return;
    }
    resizeCanvases(s, dpr, glDpr);

    var cur = currentChars(now);
    // The camera: the followed tee, smoothed; a new target snaps.
    if (view.fitPending && sceneInfo) {
      view.zoom = fitZoom();
      view.cam.x = (sceneInfo.width * TILE) / 2;
      view.cam.y = (sceneInfo.height * TILE) / 2;
      view.fitPending = false;
      view.camSet = true;
      syncZoomControl();
    }
    var t = followTarget(cur);
    if (t) {
      if (!view.camSet) {
        view.cam.x = t.x;
        view.cam.y = t.y;
        view.camSet = true;
      }
      var kf = 1 - Math.exp(-camDt * 14);
      if (Math.hypot(t.x - view.cam.x, t.y - view.cam.y) > 900) {
        view.cam.x = t.x;
        view.cam.y = t.y;
      }
      view.cam.x += (t.x - view.cam.x) * kf;
      view.cam.y += (t.y - view.cam.y) * kf;
    } else if (!view.camSet && sceneInfo && view.follow !== null) {
      // Following someone who is not in the game yet: the middle of the map until they appear.
      view.cam.x = (sceneInfo.width * TILE) / 2;
      view.cam.y = (sceneInfo.height * TILE) / 2;
    }
    var timeMs = cur ? cur.tick * TICK_MS : now - envEpoch;
    lastStats = renderer.frame({
      camX: view.cam.x,
      camY: view.cam.y,
      zoom: view.zoom,
      aspect: s.w / s.h,
      timeMs: timeMs,
      entities: view.entities,
      onWorld: function () {
        drawTees(cur, now, dt);
      },
    });
    drawOverlay(overlayCtx, cur, s, plainView(s), dpr, now);
  }

  function loop(now) {
    root.requestAnimationFrame(loop);
    if (el.view.hidden || document.hidden) {
      lastDraw = 0;
      return;
    }
    var t0 = performance.now();
    try {
      draw(now);
    } catch (e) {
      // A drawing error must not stop the loop; it is shown once.
      if (!loop.failed) {
        loop.failed = true;
        showMessage("Ошибка отрисовки: " + (e && e.message ? e.message : e), true);
      }
    }
    renderTimes.push(performance.now() - t0);
    if (renderTimes.length > 600) renderTimes.shift();
  }

  function updateTopInfo() {
    var cutoff = performance.now() - 1000;
    frameTimes = frameTimes.filter(function (x) {
      return x >= cutoff;
    });
    var parts = [];
    if (mapMeta) parts.push(mapMeta.name);
    if (assetsOk === false) parts.push("без графики DDNet");
    var last = frames[frames.length - 1];
    if (last) parts.push("тик " + last.tick);
    parts.push(frameTimes.length + " кадр/с");
    el.topInfo.textContent = parts.join(" · ");
  }

  // ---------------------------------------------------------------------------------------------
  // Panels: players, scoreboard, bot card, feed, chat
  // ---------------------------------------------------------------------------------------------

  function sortedPlayers() {
    return Object.keys(players)
      .map(function (k) {
        return players[k];
      })
      .sort(function (a, b) {
        return b.score - a.score || a.id - b.id;
      });
  }

  // Tee icons (canvases) are painted once the skin is there; the ones that came before it are retried twice a second.
  var pendingIcons = [];

  function paintIcon(entry) {
    if (!kit) return false;
    return kit.drawIcon(entry.canvas, entry.p, entry.size);
  }

  function teeIconCell(p, size) {
    var c = make("canvas", "tee-icon");
    c.width = size * 2;
    c.height = size * 2;
    var entry = { canvas: c, p: p, size: size * 2 };
    if (!paintIcon(entry)) pendingIcons.push(entry);
    return c;
  }

  setInterval(function () {
    if (pendingIcons.length === 0) return;
    pendingIcons = pendingIcons.filter(function (entry) {
      if (!entry.canvas.isConnected) return false;
      return !paintIcon(entry);
    });
  }, 500);

  var playersKey = "";
  function renderPlayers() {
    var list = sortedPlayers();
    var key = list
      .map(function (p) {
        return [p.id, p.name, p.clan, p.skin, p.cc, p.cb, p.cf, p.score, p.ping, p.team, view.follow === p.id || (view.follow === "bot" && p.id === ownId)].join("\u0001");
      })
      .join("\u0002") + "|" + ownId + "|" + (kit ? kit.skinState("default") : "");
    if (key === playersKey) return;
    playersKey = key;
    el.playerList.textContent = "";
    pendingIcons = [];
    el.playerCount.textContent = list.length ? String(list.length) : "";
    if (list.length === 0) {
      el.playerList.appendChild(make("li", "empty", "Игроков нет."));
    }
    list.forEach(function (p) {
      var li = make("li", "player-row");
      var isOwn = p.id === ownId;
      var picked = view.follow === p.id || (view.follow === "bot" && isOwn);
      if (picked) li.classList.add("selected");
      if (isOwn) li.classList.add("own");
      li.appendChild(teeIconCell(p, 18));
      var main = make("span", "pr-main");
      var nm = make("span", "pr-name", p.name || "#" + p.id);
      main.appendChild(nm);
      if (isOwn) main.appendChild(make("span", "pr-badge", "БОТ"));
      if (p.clan) main.appendChild(make("span", "pr-clan", p.clan));
      li.appendChild(main);
      var frozen = frames.length ? frames[frames.length - 1].chars[p.id] : null;
      if (frozen && frozen.frozen) li.appendChild(make("span", "pr-frozen", frozen.deep ? "❄❄" : "❄"));
      li.appendChild(make("span", "pr-score", p.score));
      li.appendChild(make("span", "pr-ping", p.ping > 0 ? p.ping : ""));
      li.addEventListener("click", function () {
        setFollow(isOwn ? "bot" : p.id);
      });
      el.playerList.appendChild(li);
    });
    renderFollowSelect(list);
    renderBoard();
  }

  function renderFollowSelect(list) {
    var current = view.follow === null ? "free" : view.follow === "bot" ? "bot" : String(view.follow);
    el.follow.textContent = "";
    var opt = make("option", "", "Бот (мой)");
    opt.value = "bot";
    el.follow.appendChild(opt);
    list.forEach(function (p) {
      if (p.id === ownId) return;
      var o = make("option", "", p.name || "#" + p.id);
      o.value = String(p.id);
      el.follow.appendChild(o);
    });
    var free = make("option", "", "Свободная камера");
    free.value = "free";
    el.follow.appendChild(free);
    el.follow.value = current;
    if (el.follow.value !== current) el.follow.value = "bot";
  }

  function renderBoard() {
    el.boardRows.textContent = "";
    var list = sortedPlayers();
    el.boardTitle.textContent = mapMeta ? mapMeta.name : "";
    list.forEach(function (p) {
      var tr = make("tr", p.id === ownId ? "own" : "");
      var tdScore = make("td", "num", p.score);
      var tdName = make("td", "nm");
      tdName.appendChild(teeIconCell(p, 14));
      tdName.appendChild(make("span", "", p.name || "#" + p.id));
      if (p.id === ownId) tdName.appendChild(make("span", "pr-badge", "БОТ"));
      tr.appendChild(tdScore);
      tr.appendChild(tdName);
      tr.appendChild(make("td", "clan", p.clan || ""));
      tr.appendChild(make("td", "num ping", p.ping > 0 ? p.ping : ""));
      el.boardRows.appendChild(tr);
    });
  }

  function setBoard(on) {
    view.board = on;
    el.board.hidden = !on;
    el.boardBtn.classList.toggle("active", on);
    el.boardBtn.setAttribute("aria-pressed", on ? "true" : "false");
    if (on) renderBoard();
  }

  function renderBotCard() {
    var st = botStatus;
    var fam = el.botGrid;
    fam.textContent = "";
    if (!st || typeof st !== "object") {
      el.botState.textContent = sourceKind === "demo" ? "показ (не настоящая игра)" : "бот не запущен";
      el.botDot.className = "dot dot-off";
      return;
    }
    var inGame = st.connected !== false;
    el.botState.textContent = !inGame ? "не в игре" : !st.alive ? "мёртв" : st.frozen ? "заморожен" : "в игре";
    el.botDot.className = "dot " + (inGame ? "dot-on" : "dot-off");
    var rows = [
      ["Режим", st.mode],
      ["Мозг", st.brain],
      ["Цель", st.target >= 0 ? (st.target_tag || "#" + st.target) : "нет"],
      ["Блоки", st.blocks + " / " + st.blocked_by],
      ["Смерти", st.deaths !== undefined ? st.deaths : "—"],
      ["Решение p99", (st.decide_p99_us / 1000).toFixed(1) + " мс"],
    ];
    rows.forEach(function (r) {
      var cell = make("div", "gb-cell");
      cell.appendChild(make("span", "k", r[0]));
      cell.appendChild(make("span", "v", r[1]));
      fam.appendChild(cell);
    });
  }

  function addFeed(text, kind) {
    var now = performance.now();
    if (feedItems.some(function (f) { return f.text === text && now - f.at < 2500; })) return;
    feedItems.push({ text: text, kind: kind, at: now });
    if (feedItems.length > 6) feedItems.shift();
    renderFeed();
  }

  function renderFeed() {
    var now = performance.now();
    feedItems = feedItems.filter(function (f) {
      return now - f.at < 9000;
    });
    el.feed.textContent = "";
    feedItems.forEach(function (f) {
      var li = make("li", "feed-" + f.kind, f.text);
      el.feed.appendChild(li);
    });
  }
  setInterval(function () {
    if (feedItems.length) renderFeed();
    pruneChatFloat();
  }, 1000);

  function nameOfId(id) {
    var p = players[id];
    return p && p.name ? p.name : "#" + id;
  }

  // ---- chat ----

  var CHAT_LABEL = { all: "", team: "(команда) ", whisper_to: "→ ", whisper_from: "← ", system: "" };

  function chatLine(line, floating) {
    var row = make("div", "chat-line chat-" + line.kind);
    if (line.kind === "system") {
      row.appendChild(make("span", "cn", "*** "));
    } else {
      var lab = CHAT_LABEL[line.kind];
      row.appendChild(make("span", "cn", lab + (line.name || (line.id !== null ? nameOfId(line.id) : "")) + ": "));
    }
    row.appendChild(make("span", "ct", line.text));
    if (floating) row.dataset.at = String(line.arrived || performance.now());
    return row;
  }

  function renderChat() {
    if (!chatDirty) return;
    chatDirty = false;
    var atEnd = el.chatLog.scrollTop + el.chatLog.clientHeight >= el.chatLog.scrollHeight - 24;
    el.chatLog.textContent = "";
    chat.forEach(function (line) {
      el.chatLog.appendChild(chatLine(line, false));
    });
    el.chatEmpty.hidden = chat.length > 0;
    el.chatCount.textContent = chat.length ? String(chat.length) : "";
    if (atEnd) el.chatLog.scrollTop = el.chatLog.scrollHeight;
  }

  function pruneChatFloat() {
    var now = performance.now();
    Array.prototype.slice.call(el.chatFloat.children).forEach(function (row) {
      var at = Number(row.dataset.at || 0);
      if (now - at > 14000) row.remove();
      else if (now - at > 12000) row.classList.add("fade");
    });
  }

  function onChat(line) {
    line.arrived = performance.now();
    chat.push(line);
    if (chat.length > 200) chat.shift();
    chatDirty = true;
    el.chatFloat.appendChild(chatLine(line, true));
    while (el.chatFloat.children.length > 6) el.chatFloat.firstChild.remove();
    renderChat();
  }

  function onChatHistory(lines) {
    chat = lines.slice(-200).map(function (l) {
      l.arrived = 0;
      return l;
    });
    chatDirty = true;
    renderChat();
  }

  // ---------------------------------------------------------------------------------------------
  // Controls and input
  // ---------------------------------------------------------------------------------------------

  function syncZoomControl() {
    el.zoom.value = String(Math.round(Math.log(view.zoom) * 100));
  }

  function syncControls() {
    el.free.classList.toggle("active", view.follow === null);
    el.free.setAttribute("aria-pressed", view.follow === null ? "true" : "false");
    var cur = view.follow === null ? "free" : view.follow === "bot" ? "bot" : String(view.follow);
    if (el.follow.value !== cur) el.follow.value = cur;
  }

  el.follow.addEventListener("change", function () {
    var v = el.follow.value;
    setFollow(v === "free" ? null : v === "bot" ? "bot" : Number(v));
  });
  el.free.addEventListener("click", function () {
    setFollow(view.follow === null ? "bot" : null);
  });
  el.fit.addEventListener("click", function () {
    view.follow = null;
    view.fitPending = true;
    syncControls();
    renderPlayers();
  });
  el.zoom.addEventListener("input", function () {
    setZoom(Math.exp(Number(el.zoom.value) / 100), false);
  });
  el.entities.addEventListener("click", function () {
    view.entities = (view.entities + 1) % 3;
    el.entities.textContent = ["Вид: карта", "Вид: карта + сущности", "Вид: сущности"][view.entities];
    el.entities.classList.toggle("active", view.entities !== 0);
  });
  el.names.addEventListener("click", function () {
    view.names = !view.names;
    el.names.classList.toggle("active", view.names);
    el.names.setAttribute("aria-pressed", view.names ? "true" : "false");
  });
  el.boardBtn.addEventListener("click", function () {
    setBoard(!view.board);
  });
  el.econ.addEventListener("click", function () {
    view.econ = !view.econ;
    el.econ.classList.toggle("active", view.econ);
    el.econ.setAttribute("aria-pressed", view.econ ? "true" : "false");
    host.send({ type: "sub", live: view.econ ? 10 : 25 });
    safeStorage("set", "ddai.econ", view.econ ? "1" : "0");
  });
  el.toFly.addEventListener("click", function () {
    host.goTab("fly");
  });
  el.toBot.addEventListener("click", function () {
    host.goTab("bot");
  });

  // Hold Tab: the scoreboard, like the client.
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Tab" || el.view.hidden) return;
    var tag = (document.activeElement && document.activeElement.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    e.preventDefault();
    if (!view.board) setBoard(true);
    view.tabHeld = true;
  });
  document.addEventListener("keyup", function (e) {
    if (e.key === "Tab" && view.tabHeld) {
      view.tabHeld = false;
      setBoard(false);
    }
  });

  // Pan, wheel zoom, pinch and tap-to-follow on the stage.
  var pointers = {};
  var pinch = null;
  var downAt = null;
  el.stage.addEventListener("pointerdown", function (e) {
    if (e.target !== el.overlay && e.target !== el.canvas) return;
    el.stage.setPointerCapture(e.pointerId);
    var r = el.stage.getBoundingClientRect();
    pointers[e.pointerId] = { x: e.clientX - r.left, y: e.clientY - r.top, px: e.clientX - r.left, py: e.clientY - r.top };
    downAt = { x: e.clientX, y: e.clientY, t: performance.now() };
    var ids = Object.keys(pointers);
    if (ids.length === 2) {
      var a = pointers[ids[0]];
      var b = pointers[ids[1]];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), zoom: view.zoom };
    }
  });
  el.stage.addEventListener("pointermove", function (e) {
    var p = pointers[e.pointerId];
    if (!p) return;
    var r = el.stage.getBoundingClientRect();
    var x = e.clientX - r.left;
    var y = e.clientY - r.top;
    var ids = Object.keys(pointers);
    if (ids.length === 1) {
      var s = stageSize();
      var v = plainView(s);
      var dxw = ((x - p.x) / s.w) * v[2];
      var dyw = ((y - p.y) / s.h) * v[3];
      if (Math.hypot(x - p.px, y - p.py) > 4 || view.follow === null) {
        if (view.follow !== null) setFollow(null);
        view.cam.x -= dxw;
        view.cam.y -= dyw;
        view.camSet = true;
      }
    } else if (ids.length === 2 && pinch) {
      p.x = x;
      p.y = y;
      var a = pointers[ids[0]];
      var b = pointers[ids[1]];
      var d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch.d > 0) setZoom(pinch.zoom * (pinch.d / Math.max(1, d)), false);
    }
    p.x = x;
    p.y = y;
  });
  function endPointer(e) {
    if (!pointers[e.pointerId]) return;
    delete pointers[e.pointerId];
    if (Object.keys(pointers).length < 2) pinch = null;
    if (Object.keys(pointers).length === 0 && downAt) {
      var moved = Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y);
      if (moved < 8 && performance.now() - downAt.t < 450) tapAt(e.clientX, e.clientY);
      downAt = null;
    }
  }
  el.stage.addEventListener("pointerup", endPointer);
  el.stage.addEventListener("pointercancel", endPointer);
  el.stage.addEventListener(
    "wheel",
    function (e) {
      e.preventDefault();
      var r = el.stage.getBoundingClientRect();
      var f = e.deltaY < 0 ? 0.87 : 1 / 0.87;
      if (view.follow === null) setZoom(view.zoom * f, true, e.clientX - r.left, e.clientY - r.top);
      else setZoom(view.zoom * f, false);
    },
    { passive: false }
  );

  function tapAt(cx, cy) {
    var cur = currentChars(performance.now());
    if (!cur) return;
    var r = el.stage.getBoundingClientRect();
    var s = stageSize();
    var plain = plainView(s);
    var k = s.w / plain[2];
    var best = null;
    var bestD = Math.max(26, 34 * k);
    cur.list.forEach(function (c) {
      var d = Math.hypot((c.x - plain[0]) * k - (cx - r.left), (c.y - plain[1]) * k - (cy - r.top));
      if (d < bestD) {
        bestD = d;
        best = c.id;
      }
    });
    if (best !== null) setFollow(best === ownId ? "bot" : best);
  }

  // Replay controls (a replay source only).
  el.replayPlay.addEventListener("click", function () {
    host.send({ type: "replay", action: "play" });
  });
  el.replayPause.addEventListener("click", function () {
    host.send({ type: "replay", action: "pause" });
  });
  el.replayNext.addEventListener("click", function () {
    host.send({ type: "replay", action: "next" });
  });
  el.replaySpeed.addEventListener("change", function () {
    host.send({ type: "replay", action: "speed", value: parseFloat(el.replaySpeed.value) });
  });
  var seekDrag = false;
  el.replaySeek.addEventListener("pointerdown", function () {
    seekDrag = true;
  });
  el.replaySeek.addEventListener("change", function () {
    host.send({ type: "replay", action: "seek", tick: parseInt(el.replaySeek.value, 10) });
    seekDrag = false;
  });

  if (safeStorage("get", "ddai.econ") === "1") {
    view.econ = true;
    el.econ.classList.add("active");
    el.econ.setAttribute("aria-pressed", "true");
  }

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(function () {
      if (!el.view.hidden) lastDraw = 0;
    }).observe(el.stage);
  }

  // ---------------------------------------------------------------------------------------------
  // The messages from the server (called by app.js)
  // ---------------------------------------------------------------------------------------------

  function onMap(msg) {
    var changed = !mapMeta || mapMeta.sha256 !== msg.sha256;
    mapMeta = msg;
    if (changed) {
      frames = [];
      clock = null;
      view.camSet = false;
      view.fitPending = view.follow === null;
      loadMap(msg);
    }
    updateTopInfo();
  }

  function onPlayers(list) {
    var next = {};
    var nowMs = performance.now();
    list.forEach(function (p) {
      var np = {
        id: p.id,
        name: p.name,
        team: p.team,
        clan: p.clan || "",
        skin: p.skin || "default",
        cc: !!p.cc,
        cb: p.cb | 0,
        cf: p.cf | 0,
        country: p.country | 0,
        score: p.score | 0,
        ping: p.ping | 0,
        lookAt: nowMs,
      };
      // A server-side rainbow changes a colour every snapshot, and each new colour is a CPU re-tint plus a texture upload:
      // a player's custom colours may change at most about twice a second here (the bot sends the look once a second anyway).
      var prev = players[p.id];
      if (prev && prev.skin === np.skin && (prev.cc !== np.cc || prev.cb !== np.cb || prev.cf !== np.cf)) {
        if (nowMs - prev.lookAt < 500) {
          np.cc = prev.cc;
          np.cb = prev.cb;
          np.cf = prev.cf;
          np.lookAt = prev.lookAt;
        }
      } else if (prev && prev.skin === np.skin) {
        np.lookAt = prev.lookAt;
      }
      next[p.id] = np;
    });
    players = next;
    renderPlayers();
  }

  function onEvents(msg) {
    (msg.events || []).forEach(function (e) {
      var name = nameOfId(e.id);
      if (e.kind === "freeze") addFeed(name + " заморожен", "freeze");
      else if (e.kind === "unfreeze") addFeed(name + " разморожен", "thaw");
      else if (e.kind === "death") addFeed(name + " погиб", "death");
    });
  }

  function onBotStatus(st) {
    botStatus = st && typeof st === "object" ? st : null;
    if (botStatus && typeof botStatus.own === "number" && botStatus.own >= 0 && sourceKind !== "demo") {
      if (ownId !== botStatus.own) {
        ownId = botStatus.own;
        playersKey = "";
        renderPlayers();
      }
    }
    renderBotCard();
  }

  function onReplayStatus(msg) {
    el.replay.hidden = false;
    if (!seekDrag) {
      el.replaySeek.max = String(Math.max(1, msg.tick_count));
      el.replaySeek.value = String(msg.tick);
    }
    if (document.activeElement !== el.replaySpeed) el.replaySpeed.value = String(msg.speed);
    el.replayPlay.classList.toggle("active", !!msg.playing);
    el.replayPause.classList.toggle("active", !msg.playing);
  }

  function onLiveError(message) {
    showMessage("Ошибка источника: " + message, true);
  }

  function setDemo(on) {
    sourceKind = on ? "demo" : sourceKind === "demo" ? "live" : sourceKind;
    if (on) {
      // In the demo the fly is slot 0 (the arena's only bot).
      ownId = 0;
      playersKey = "";
      renderPlayers();
    }
    renderBotCard();
  }

  function setSource(kind) {
    sourceKind = kind;
    renderBotCard();
  }

  function reset() {
    mapGen++;
    mapMeta = null;
    mapStatus = "none";
    sceneInfo = null;
    gameTiles = null;
    parsedScene = null;
    if (renderer) renderer.dispose();
    players = {};
    botStatus = null;
    ownId = -1;
    frames = [];
    clock = null;
    chat = [];
    chatDirty = true;
    feedItems = [];
    freezeSince = {};
    wasFrozen = {};
    flakes = [];
    view.camSet = false;
    el.replay.hidden = true;
    el.chatFloat.textContent = "";
    playersKey = "";
    showMessage("");
    renderPlayers();
    renderChat();
    renderBotCard();
    renderFeed();
    el.topInfo.textContent = "";
  }

  function onConnectionChanged(connected) {
    if (connected && view.econ) {
      host.send({ type: "sub", live: 10 }); // the economy rate again after a reconnect
    }
    if (!connected) {
      showMessage("Нет соединения с сайтом…", true);
    } else if (mapStatus === "ready" || mapStatus === "fallback") {
      showMessage(mapStatus === "fallback" ? "Слои карты недоступны: показана только геометрия." : "");
    } else if (mapStatus === "none") {
      showMessage("");
    }
  }

  function onShown() {
    ensureRenderer();
    lastDraw = 0;
    renderPlayers();
    renderChat();
    renderBotCard();
    if (!mapMeta) {
      showMessage("Ждём карту от бота…");
    }
    if (el.view.hidden === false && !fontLoaded) loadFont();
  }

  var fontLoaded = false;
  function loadFont() {
    fontLoaded = true;
    if (!root.FontFace) return;
    var f = new root.FontFace("DejaVu Sans DDNet", "url(/assets/fonts/DejaVuSans.ttf)");
    f.load()
      .then(function () {
        document.fonts.add(f);
      })
      .catch(function () {
        /* the system font stack is used */
      });
  }

  root.requestAnimationFrame(loop);

  // Test hooks (tools/e2e): a read-only look at the camera, the frames and the loaded scene, and a way to change the rate.
  root.__ddaiDebug = {
    getState: function () {
      var last = frames[frames.length - 1];
      var s = renderer && renderer.scene();
      return {
        followId: view.follow === "bot" ? ownId : view.follow,
        follow: view.follow,
        ownId: ownId,
        camera: { x: view.cam.x, y: view.cam.y, scale: view.zoom },
        latestFrame: last ? { tick: last.tick, characters: last.list } : null,
        mapMeta: mapMeta,
        scene: sceneInfo,
        sceneImages: s ? s.images.map(function (i) { return i.state; }) : [],
        mapStatus: mapStatus,
        assetsOk: assetsOk,
        chatLines: chat.length,
        entities: view.entities,
        plates: lastPlates,
      };
    },
    setLiveHz: function (hz) {
      host.send({ type: "sub", live: hz });
    },
    // Draws one frame now and reads a grid of pixels back (the drawing buffer is only readable in the task that drew it).
    snapshotStats: function () {
      if (!renderer) return null;
      draw(performance.now());
      return renderer.sampleStats();
    },
    // Test hook: draw the map at full resolution and never change it.
    fixScale: function () {
      view.fixedScale = true;
      renderScale = 1;
    },
    renderScale: function () {
      return renderScale;
    },
    renderMs: function () {
      var a = renderTimes.slice(-120).sort(function (x, y) {
        return x - y;
      });
      return a.length ? { p50: a[Math.floor(a.length * 0.5)], p95: a[Math.floor(a.length * 0.95)], n: a.length, frame: lastStats } : null;
    },
    screenPositionOf: function (id) {
      var cur = currentChars(performance.now());
      var c = cur && cur.byId[id];
      if (!c) return null;
      var s = stageSize();
      var plain = plainView(s);
      var k = s.w / plain[2];
      return { x: (c.x - plain[0]) * k, y: (c.y - plain[1]) * k };
    },
  };

  root.GameView = {
    attach: function (h) {
      host.send = h.send;
      host.goTab = h.goTab;
    },
    reset: reset,
    setDemo: setDemo,
    setSource: setSource,
    onMap: onMap,
    onPlayers: onPlayers,
    onEvents: onEvents,
    onReplayStatus: onReplayStatus,
    onBotStatus: onBotStatus,
    onLiveError: onLiveError,
    onLiveFrame: onLiveFrame,
    onChat: onChat,
    onChatHistory: onChatHistory,
    onConnectionChanged: onConnectionChanged,
    onShown: onShown,
  };
})(window);
