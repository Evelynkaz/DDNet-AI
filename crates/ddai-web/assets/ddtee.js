/* Tees, weapons, hooks and the small effects of the «Игра» tab (task 5.10): real DDNet tee rendering from the stock
   skin textures served from the DDNet data directory.

   The sprite grids, animation key frames and tee drawing follow DDNet's datasrc/content.py, game/client/animstate.cpp,
   render.cpp (RenderTee6, RenderHand), components/players.cpp and components/freezebars.cpp, through the port in
   Wranked1/DDNet-AI (GPL-3.0) src/bot/webDraw.ts and webView.ts; skin recolouring follows CSkins::LoadSkin
   (ColorHSLA, the grey-scale normalisation of the body, the feet tint).
   Portions derived from DDNet (https://github.com/ddnet/ddnet), zlib license.
   Copyright (C) 2007-2014 Magnus Auvinen (Teeworlds); Copyright (C) DDRace and DDNet contributors.
   This is an altered version, not the original software. */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------------------------------------
  // Sprite grids (datasrc/content.py): [x, y, w, h] in cells of a sheet with the given grid.
  // ---------------------------------------------------------------------------------------------

  var SPRITES = {
    game: { gx: 32, gy: 16 },
    hookChain: [2, 0, 1, 1],
    hookHead: [3, 0, 2, 1],
    // body cell, drawn size and offset of each weapon: hammer, gun, shotgun, grenade, laser, ninja
    weapons: [
      { body: [2, 1, 4, 3], size: 96, ox: 4, oy: -20 },
      { body: [2, 4, 4, 2], size: 64, ox: 32, oy: 4 },
      { body: [2, 6, 8, 2], size: 96, ox: 24, oy: -2 },
      { body: [2, 8, 7, 2], size: 96, ox: 24, oy: -2 },
      { body: [2, 12, 7, 3], size: 92, ox: 24, oy: -2 },
      { body: [2, 10, 8, 2], size: 96, ox: 0, oy: 0 },
    ],
    tee: { gx: 8, gy: 4 },
    body: [0, 0, 3, 3],
    bodyOutline: [3, 0, 3, 3],
    foot: [6, 1, 2, 1],
    footOutline: [6, 2, 2, 1],
    hand: [6, 0, 1, 1],
    handOutline: [7, 0, 1, 1],
    // eyes: normal, angry, pain, happy, empty ("blink"), surprise
    eyes: [[2, 3, 1, 1], [3, 3, 1, 1], [4, 3, 1, 1], [5, 3, 1, 1], [6, 3, 1, 1], [7, 3, 1, 1]],
    emoticons: { gx: 4, gy: 4 },
    snowflake: [0, 0, 2, 2],
    hud: { gx: 16, gy: 16 },
    freezeBarFullLeft: [0, 2, 1, 1],
    freezeBarFull: [1, 2, 1, 1],
    freezeBarEmpty: [2, 2, 1, 1],
    freezeBarEmptyRight: [3, 2, 1, 1],
  };

  // ---------------------------------------------------------------------------------------------
  // Animations: key frames [time, x, y, angle] per part (datasrc/content.py, animstate.cpp)
  // ---------------------------------------------------------------------------------------------

  var NONE = [];
  var ANIMS = {
    base: { body: [[0, 0, -4, 0]], back: [[0, 0, 10, 0]], front: [[0, 0, 10, 0]], attach: NONE },
    idle: { body: NONE, back: [[0, -7, 0, 0]], front: [[0, 7, 0, 0]], attach: NONE },
    inair: { body: NONE, back: [[0, -3, 0, -0.1]], front: [[0, 3, 0, -0.1]], attach: NONE },
    walk: {
      body: [[0, 0, 0, 0], [0.2, 0, -1, 0], [0.4, 0, 0, 0], [0.6, 0, 0, 0], [0.8, 0, -1, 0], [1, 0, 0, 0]],
      back: [[0, 8, 0, 0], [0.2, -8, 0, 0], [0.4, -10, -4, 0.2], [0.6, -8, -8, 0.3], [0.8, 4, -4, -0.2], [1, 8, 0, 0]],
      front: [[0, -10, -4, 0.2], [0.2, -8, -8, 0.3], [0.4, 4, -4, -0.2], [0.6, 8, 0, 0], [0.8, 8, 0, 0], [1, -10, -4, 0.2]],
      attach: NONE,
    },
    runLeft: {
      body: [[0, 0, -1, 0], [0.2, 0, 0, 0], [0.4, 0, -1, 0], [0.6, 0, 0, 0], [0.8, 0, 0, 0], [1, 0, -1, 0]],
      back: [[0, 18, -8, -0.27], [0.2, 6, 0, 0], [0.4, -7, 0, 0], [0.6, -13, -4.5, 0.05], [0.8, 0, -8, -0.2], [1, 18, -8, -0.27]],
      front: [[0, -11, -2.5, 0.05], [0.2, -14, -5, 0.1], [0.4, 11, -8, -0.3], [0.6, 18, -8, -0.27], [0.8, 3, 0, 0], [1, -11, -2.5, 0.05]],
      attach: NONE,
    },
    runRight: {
      body: [[0, 0, -1, 0], [0.2, 0, 0, 0], [0.4, 0, 0, 0], [0.6, 0, -1, 0], [0.8, 0, 0, 0], [1, 0, -1, 0]],
      back: [[0, -18, -8, 0.27], [0.2, 0, -8, 0.2], [0.4, 13, -4.5, -0.05], [0.6, 7, 0, 0], [0.8, -6, 0, 0], [1, -18, -8, 0.27]],
      front: [[0, 11, -2.5, -0.05], [0.2, -3, 0, 0], [0.4, -18, -8, 0.27], [0.6, -11, -8, 0.3], [0.8, 14, -5, -0.1], [1, 11, -2.5, -0.05]],
      attach: NONE,
    },
    hammer: { body: NONE, back: NONE, front: NONE, attach: [[0, 0, 0, -0.1], [0.3, 0, 0, 0.25], [0.4, 0, 0, 0.3], [0.5, 0, 0, 0.25], [1, 0, 0, -0.1]] },
    ninja: { body: NONE, back: NONE, front: NONE, attach: [[0, 0, 0, -0.25], [0.1, 0, 0, -0.05], [0.15, 0, 0, 0.35], [0.42, 0, 0, 0.4], [0.5, 0, 0, 0.35], [1, 0, 0, -0.25]] },
  };

  function animSeq(frames, t) {
    if (frames.length === 0) return [0, 0, 0];
    if (frames.length === 1) return [frames[0][1], frames[0][2], frames[0][3]];
    for (var i = 1; i < frames.length; i++) {
      var a = frames[i - 1];
      var b = frames[i];
      if (a[0] <= t && b[0] >= t) {
        var k = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
        return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k, a[3] + (b[3] - a[3]) * k];
      }
    }
    return [0, 0, 0];
  }

  function animState(parts) {
    var out = { body: [0, 0, 0], back: [0, 0, 0], front: [0, 0, 0], attach: [0, 0, 0] };
    var keys = ["body", "back", "front", "attach"];
    parts.forEach(function (p) {
      keys.forEach(function (key) {
        var v = animSeq(p.anim[key], p.t);
        out[key][0] += v[0];
        out[key][1] += v[1];
        out[key][2] += v[2];
      });
    });
    return out;
  }

  /** The pose of a tee (components/players.cpp: RenderPlayer): vx in units per tick, dir the pressed direction. */
  function teeAnim(vx, inAir, dir, x, attackSec, weapon) {
    var parts = [{ anim: ANIMS.base, t: 0 }];
    var stationary = Math.abs(vx) <= 1 / 256;
    var running = Math.abs(vx) >= 5000 / 256;
    var wantOther = (dir === -1 && vx > 0) || (dir === 1 && vx < 0);
    var walk = (x % 100) / 100;
    if (walk < 0) walk += 1;
    var run = (x % 200) / 200;
    if (run < 0) run += 1;
    if (inAir) parts.push({ anim: ANIMS.inair, t: 0 });
    else if (stationary) parts.push({ anim: ANIMS.idle, t: 0 });
    else if (!wantOther) {
      if (running) parts.push({ anim: vx < 0 ? ANIMS.runLeft : ANIMS.runRight, t: run });
      else parts.push({ anim: ANIMS.walk, t: walk });
    }
    if (weapon === 0) parts.push({ anim: ANIMS.hammer, t: Math.min(1, Math.max(0, attackSec * 5)) });
    if (weapon === 5) parts.push({ anim: ANIMS.ninja, t: Math.min(1, Math.max(0, attackSec * 2)) });
    return animState(parts);
  }

  // ---------------------------------------------------------------------------------------------
  // Colours (CSkins::LoadSkin, ColorHSLA): DDNet's packed HSL (hue, saturation, lightness bytes) and the skin tint
  // ---------------------------------------------------------------------------------------------

  function ddnetColor(packed, darkest) {
    if (darkest === undefined) darkest = 0.5;
    var h = ((packed >>> 16) & 0xff) / 255;
    var s = ((packed >>> 8) & 0xff) / 255;
    var l = (packed & 0xff) / 255;
    l = darkest + l * (1 - darkest);
    var h1 = h * 6;
    var c = (1 - Math.abs(2 * l - 1)) * s;
    var x = c * (1 - Math.abs((h1 % 2) - 1));
    var r = 0;
    var g = 0;
    var b = 0;
    switch (Math.trunc(h1)) {
      case 0: r = c; g = x; break;
      case 1: r = x; g = c; break;
      case 2: g = c; b = x; break;
      case 3: g = x; b = c; break;
      case 4: r = x; b = c; break;
      default: r = c; b = x; break;
    }
    var m = l - c / 2;
    return [r + m, g + m, b + m];
  }

  /** Grey-scales the body area of a skin and normalises its most common grey to 192, as the client does before tinting. */
  function skinColorable(px, w, h) {
    var i;
    for (i = 0; i < w * h; i++) {
      var o = i * 4;
      var luma = Math.trunc(0.2126 * px[o] + 0.7152 * px[o + 1] + 0.0722 * px[o + 2]);
      px[o] = luma;
      px[o + 1] = luma;
      px[o + 2] = luma;
    }
    var bw = Math.trunc((w * 3) / 8);
    var bh = Math.trunc((h * 3) / 4);
    var freq = new Array(256).fill(0);
    for (var y = 0; y < bh; y++) {
      for (var x = 0; x < bw; x++) {
        var q = (y * w + x) * 4;
        if (px[q + 3] > 128) freq[px[q]]++;
      }
    }
    var org = 1;
    for (i = 1; i < 256; i++) if (freq[org] < freq[i]) org = i;
    var neu = 192;
    for (var y2 = 0; y2 < bh; y2++) {
      for (var x2 = 0; x2 < bw; x2++) {
        var p = (y2 * w + x2) * 4;
        var v = px[p];
        if (v <= org) v = Math.trunc((v / org) * neu);
        else v = Math.trunc(((v - org) / (255 - org)) * (255 - neu) + neu);
        px[p] = v;
        px[p + 1] = v;
        px[p + 2] = v;
      }
    }
  }

  function skinTint(src, w, h, body, feet) {
    var out = new Uint8ClampedArray(src.length);
    var fx0 = Math.trunc((w * 6) / 8);
    var fy0 = Math.trunc(h / 4);
    var fy1 = Math.trunc((h * 3) / 4);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var o = (y * w + x) * 4;
        var isFeet = x >= fx0 && y >= fy0 && y < fy1;
        var c = isFeet ? feet : body;
        out[o] = src[o] * c[0];
        out[o + 1] = src[o + 1] * c[1];
        out[o + 2] = src[o + 2] * c[2];
        out[o + 3] = src[o + 3];
      }
    }
    return out;
  }

  function rect(cell, grid) {
    var cw = 1 / grid.gx;
    var ch = 1 / grid.gy;
    return [cell[0] * cw, cell[1] * ch, (cell[0] + cell[2]) * cw, (cell[1] + cell[3]) * ch];
  }

  function spriteScale(cell) {
    var f = Math.hypot(cell[2], cell[3]);
    return [cell[2] / f, cell[3] / f];
  }

  /** The freeze bar's pieces (components/freezebars.cpp) for a progress of 0..1: [{s, u0, u1, x, w}] over 64 units. */
  function freezeBarPieces(progress) {
    var p = Math.max(0, Math.min(1, progress));
    var end = 16;
    var mid = 64 - 2 * end;
    var endProg = end * 0.5;
    var endRest = end * 0.5;
    var progW = 64 - 2 * endProg;
    var endProp = endProg / progW;
    var midProp = mid / progW;
    var out = [];
    var begin = p <= endProp ? p / endProp : 1;
    out.push({ s: "fullLeft", u0: 0, u1: 0.5 + 0.5 * begin, x: 0, w: endRest + endProg * begin });
    if (begin < 1) out.push({ s: "emptyRight", u0: 0.5 - 0.5 * begin, u1: 0, x: endRest + endProg * begin, w: endProg * (1 - begin) });
    var midP = p <= endProp + midProp ? (p <= endProp ? 0 : (p - endProp) / midProp) : 1;
    var fullW = mid * midP;
    var emptyW = mid - fullW;
    if (fullW > 0) out.push({ s: "full", u0: 0, u1: fullW <= end ? fullW / end : 1, x: end, w: fullW });
    if (emptyW > 0) out.push({ s: "empty", u0: emptyW <= end ? emptyW / end : 1, u1: 0, x: end + fullW, w: emptyW });
    var endP = p <= endProp + midProp ? 0 : (p - endProp - midProp) / endProp;
    if (endP > 0) out.push({ s: "fullLeft", u0: 1, u1: 1 - 0.5 * endP, x: end + mid, w: endProg * endP });
    out.push({ s: "emptyRight", u0: 0.5 - 0.5 * (1 - endP), u1: 1, x: end + mid + endProg * endP, w: endProg * (1 - endP) + endRest });
    return out;
  }

  /** Snowflake particles step (components/effects.cpp: the freeze effect), friction applied in fixed 0.9 steps. */
  function flakeStep(p, dt, frictionSteps) {
    p.vy += p.g * dt;
    for (var i = 0; i < frictionSteps; i++) {
      p.vx *= 0.9;
      p.vy *= 0.9;
    }
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.rot += Math.PI * dt;
    p.life += dt;
  }

  // ---------------------------------------------------------------------------------------------
  // Assets: sheets and skins, loaded through the page's own origin (`/assets/...`, served from the DDNet data dir)
  // ---------------------------------------------------------------------------------------------

  /** Loads a PNG (any same-origin URL) as {w, h, data: Uint8ClampedArray RGBA}; resolves null if it is not there. */
  function loadPixels(url) {
    return fetch(url, { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) {
          return null;
        }
        return r.blob().then(function (blob) {
          return createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" }).then(function (bmp) {
            var c = document.createElement("canvas");
            c.width = bmp.width;
            c.height = bmp.height;
            var ctx = c.getContext("2d", { willReadFrequently: true });
            ctx.drawImage(bmp, 0, 0);
            var d = ctx.getImageData(0, 0, c.width, c.height);
            if (bmp.close) bmp.close();
            return { w: c.width, h: c.height, data: d.data };
          });
        });
      })
      .catch(function () {
        return null;
      });
  }

  /** A skin name that may be asked of the server (the server checks again): plain stem, at most 24 bytes. */
  function skinNameOk(name) {
    return typeof name === "string" && /^[A-Za-z0-9_][A-Za-z0-9_ .+-]{0,23}$/.test(name) && name.indexOf("..") < 0;
  }

  function createTeeKit(renderer) {
    var sheets = {}; // name -> {tex, w, h} once loaded, "bad" once failed
    var skinPixels = {}; // name -> {w, h, data (original), gray (colorable copy), state}
    var skinTex = {}; // key -> {tex, w, h, canvas}
    var skinOrder = [];

    function sheet(name, url, mipmaps) {
      if (sheets[name] !== undefined) {
        return sheets[name];
      }
      sheets[name] = null;
      loadPixels(url).then(function (px) {
        sheets[name] = px ? Object.assign(renderer.texture2D(px.data, px.w, px.h, { mipmaps: mipmaps !== false }), { px: px }) : "bad";
      });
      return null;
    }

    function fetchSkin(name) {
      var s = skinPixels[name];
      if (s) {
        return s;
      }
      s = { state: "loading", w: 0, h: 0, data: null, gray: null };
      skinPixels[name] = s;
      if (!skinNameOk(name)) {
        s.state = "bad";
        return s;
      }
      loadPixels("/assets/skins/" + encodeURIComponent(name) + ".png").then(function (px) {
        if (px) {
          s.w = px.w;
          s.h = px.h;
          s.data = px.data;
          s.state = "ready";
        } else {
          s.state = "bad";
        }
      });
      return s;
    }

    function texFor(name, cc, cb, cf) {
      var s = fetchSkin(name);
      if (s.state !== "ready") {
        return null;
      }
      var key = name + "|" + (cc ? cb + "/" + cf : "o");
      var hit = skinTex[key];
      if (hit) {
        return hit;
      }
      var data = s.data;
      if (cc) {
        if (!s.gray) {
          s.gray = new Uint8ClampedArray(s.data);
          skinColorable(s.gray, s.w, s.h);
        }
        data = skinTint(s.gray, s.w, s.h, ddnetColor(cb), ddnetColor(cf));
      }
      var t = renderer.texture2D(data, s.w, s.h, { mipmaps: true });
      t.px = { w: s.w, h: s.h, data: data };
      skinTex[key] = t;
      skinOrder.push(key);
      if (skinOrder.length > 120) {
        var old = skinOrder.shift();
        if (old !== "default|o" && old !== "x_ninja|o" && skinTex[old]) {
          renderer.gl.deleteTexture(skinTex[old].tex);
          delete skinTex[old];
        }
      }
      return t;
    }

    /** The texture of a player's skin: their own when it exists locally, "default" otherwise; null while loading. */
    function skin(info, ninja) {
      var name = ninja ? "x_ninja" : info.skin || "default";
      var cc = !ninja && !!info.cc;
      var t = texFor(name, cc, info.cb | 0, info.cf | 0);
      if (!t) {
        var st = skinPixels[name];
        if (st && st.state === "bad" && name !== "default") {
          return texFor("default", cc, info.cb | 0, info.cf | 0);
        }
        return null;
      }
      return t;
    }

    function startLoading() {
      sheet("game", "/assets/game.png");
      sheet("hud", "/assets/hud.png");
      sheet("emoticons", "/assets/emoticons.png");
      sheet("extras", "/assets/extras.png");
      sheet("arrow", "/assets/arrow.png");
      fetchSkin("default");
      fetchSkin("x_ninja");
    }

    function get(name) {
      var s = sheets[name];
      return s && s !== "bad" ? s : null;
    }

    // ---- drawing (all in world units; the renderer's sprite batch does the rest) ----

    function rects(t) {
      var g = SPRITES.tee;
      return {
        body: rect(SPRITES.body, g),
        bodyO: rect(SPRITES.bodyOutline, g),
        foot: rect(SPRITES.foot, g),
        footO: rect(SPRITES.footOutline, g),
        hand: rect(SPRITES.hand, g),
        handO: rect(SPRITES.handOutline, g),
        eyes: SPRITES.eyes.map(function (e) {
          return rect(e, g);
        }),
      };
    }
    var TEE_RECTS = rects();

    function spr(tex, r, x, y, w, h, rot, cr, cg, cb, ca) {
      renderer.sprite(tex, r[0], r[1], r[2], r[3], x, y, w, h, rot, cr, cg, cb, ca);
    }

    /** RenderTee6: the feet and body, then the eyes. `feetDim` darkens the feet when there is no air jump left. */
    function renderTee(tex, anim, emote, dx, dy, x, y, alpha, feetDim) {
      var R = TEE_RECTS;
      var bx = x + anim.body[0];
      var by = y + anim.body[1];
      var fd = feetDim ? 0.5 : 1;
      for (var pass = 0; pass < 2; pass++) {
        var outline = pass === 0;
        for (var filling = 0; filling < 2; filling++) {
          if (filling === 1) {
            spr(tex, outline ? R.bodyO : R.body, bx, by, 64, 64, anim.body[2] * Math.PI * 2, 1, 1, 1, alpha);
            if (!outline) {
              var eye = emote === 1 ? 2 : emote === 2 ? 3 : emote === 3 ? 5 : emote === 4 ? 1 : 0;
              var es = 64 * 0.4;
              var eh = emote === 5 ? 64 * 0.15 : es;
              var sep = (0.075 - 0.01 * Math.abs(dx)) * 64;
              var ox = dx * 0.125 * 64;
              var oy = (-0.05 + dy * 0.1) * 64;
              spr(tex, R.eyes[eye], bx - sep + ox, by + oy, es, eh, 0, 1, 1, 1, alpha);
              spr(tex, R.eyes[eye], bx + sep + ox, by + oy, -es, eh, 0, 1, 1, 1, alpha);
            }
          }
          var foot = filling ? anim.front : anim.back;
          spr(tex, outline ? R.footO : R.foot, x + foot[0], y + foot[1], 64, 32, foot[2] * Math.PI * 2, fd, fd, fd, alpha);
        }
      }
    }

    function renderHand(tex, cx, cy, dx, dy, angleOff, postX, postY, alpha) {
      var R = TEE_RECTS;
      var ny = -dy;
      var nx = dx;
      if (dx < 0) {
        ny = -ny;
        nx = -nx;
      }
      var hx = cx + dx + dx * postX + ny * postY;
      var hy = cy + dy + dy * postX + nx * postY;
      var a = Math.atan2(dy, dx);
      var ang = dx < 0 ? a - angleOff : a + angleOff;
      spr(tex, R.handO, hx, hy, 20, 20, ang, 1, 1, 1, alpha);
      spr(tex, R.hand, hx, hy, 20, 20, ang, 1, 1, 1, alpha);
    }

    /** Weapon in the hand of a tee (components/players.cpp), with the hand of the heavier ones. */
    function renderWeapon(tex, wp, x, y, anim, dx, dy, attackSec) {
      var game = get("game");
      if (!game) return;
      wp = Math.max(0, Math.min(5, wp));
      var spec = SPRITES.weapons[wp];
      var r = rect(spec.body, SPRITES.game);
      var sc = spriteScale(spec.body);
      var w = spec.size * sc[0];
      var h = spec.size * sc[1];
      var flip = dx < 0;
      var angle = Math.atan2(dy, dx);
      var att = anim.attach[2] * Math.PI * 2;
      if (wp === 0 || wp === 5) {
        var px = x + anim.attach[0];
        var py = y + anim.attach[1] + spec.oy;
        if (flip) px -= spec.ox;
        var rot = flip ? -Math.PI / 2 - att : -Math.PI / 2 + att;
        // A flipped weapon is mirrored top-bottom in its own frame (the client's scale(1, -1) after the rotation).
        spr(game, r, px, py, w, flip ? -h : h, rot, 1, 1, 1, 1);
        return;
      }
      var ticks = attackSec * 50;
      var recoil = ticks / 5 < 1 ? Math.sin((ticks / 5) * Math.PI) : 0;
      var wx = x + dx * spec.ox - dx * recoil * 10;
      var wy = y + dy * spec.ox - dy * recoil * 10 + spec.oy;
      spr(game, r, wx, wy, w, flip ? -h : h, att + angle, 1, 1, 1, 1);
      if (tex) {
        if (wp === 1) renderHand(tex, wx, wy, dx, dy, (-3 * Math.PI) / 4, -15, 4, 1);
        else if (wp === 2) renderHand(tex, wx, wy, dx, dy, -Math.PI / 2, -5, 4, 1);
        else if (wp === 3) renderHand(tex, wx, wy, dx, dy, -Math.PI / 2, -4, 7, 1);
      }
    }

    /** The hook: the chain every 24 units, the head, and the hand on the hooking tee. */
    function renderHook(tex, x, y, hx, hy, alpha) {
      var d = Math.hypot(x - hx, y - hy);
      if (d < 1) return;
      var dx = (x - hx) / d;
      var dy = (y - hy) / d;
      var rot = Math.atan2(dy, dx) + Math.PI;
      var game = get("game");
      if (!game) return;
      var head = rect(SPRITES.hookHead, SPRITES.game);
      var link = rect(SPRITES.hookChain, SPRITES.game);
      for (var f = 24; f < d && f < 24 * 1024; f += 24) spr(game, link, hx + dx * f, hy + dy * f, 24, 16, rot, 1, 1, 1, alpha);
      spr(game, head, hx, hy, 24, 16, rot, 1, 1, 1, alpha);
      if (tex) renderHand(tex, x, y, -dx, -dy, -Math.PI / 2, 20, 0, alpha);
    }

    function renderFreezeBar(x0, y0, progress) {
      var hud = get("hud");
      if (!hud) return;
      var x = x0 - 32;
      var y = y0 + 32;
      var rects2 = {
        fullLeft: rect(SPRITES.freezeBarFullLeft, SPRITES.hud),
        full: rect(SPRITES.freezeBarFull, SPRITES.hud),
        empty: rect(SPRITES.freezeBarEmpty, SPRITES.hud),
        emptyRight: rect(SPRITES.freezeBarEmptyRight, SPRITES.hud),
      };
      freezeBarPieces(progress).forEach(function (p) {
        if (p.w <= 0) return;
        var r = rects2[p.s];
        var cellW = r[2] - r[0];
        var a = r[0] + Math.min(p.u0, p.u1) * cellW;
        var b = r[0] + Math.max(p.u0, p.u1) * cellW;
        if (b - a <= 0) return;
        // a reversed piece (u0 > u1) is mirrored
        var rr = p.u0 <= p.u1 ? [a, r[1], b, r[3]] : [b, r[1], a, r[3]];
        renderer.sprite(hud, rr[0], rr[1], rr[2], rr[3], x + p.x + p.w / 2, y + 8, p.w, 16, 0, 1, 1, 1, 1);
      });
    }

    function renderEmote(index, x, y, sinceSec, alpha) {
      var em = get("emoticons");
      if (!em) return;
      var fromEnd = 2 - sinceSec;
      if (sinceSec < 0 || fromEnd <= 0) return;
      var a = (fromEnd < 0.2 ? fromEnd / 0.2 : 1) * alpha;
      var h = sinceSec < 0.1 ? sinceSec / 0.1 : 1;
      var wig = sinceSec < 0.2 ? sinceSec / 0.2 : 0;
      var r = rect([index % 4, Math.floor(index / 4) % 4, 1, 1], SPRITES.emoticons);
      spr(em, r, x, y - 23 - 32 * h, 64, 64 * h, (Math.PI / 6) * Math.sin(5 * wig), 1, 1, 1, a);
    }

    function renderSnowflake(x, y, size, rot, alpha) {
      var ex = get("extras");
      if (!ex) return;
      spr(ex, rect(SPRITES.snowflake, { gx: 16, gy: 16 }), x, y, size, size, rot, 1, 1, 1, alpha);
    }

    return {
      startLoading: startLoading,
      sheet: get,
      skin: skin,
      skinState: function (name) {
        var s = skinPixels[name];
        return s ? s.state : "none";
      },
      /** A 2D canvas icon of a tee (scoreboard, player list, kill feed): body, feet and eyes of the skin in a fixed pose. */
      drawIcon: function (canvas2d, info, size) {
        var t = skin(info, false);
        var ctx = canvas2d.getContext("2d");
        ctx.clearRect(0, 0, canvas2d.width, canvas2d.height);
        if (!t || !t.px) return false;
        var c = document.createElement("canvas");
        c.width = t.px.w;
        c.height = t.px.h;
        c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(t.px.data), t.px.w, t.px.h), 0, 0);
        var cw = t.px.w / 8;
        var ch = t.px.h / 4;
        var u = size / 80; // the tee is 64 units in an 80 unit cell
        var cx = canvas2d.width / 2;
        var cy = canvas2d.height / 2;
        function part(cell, x, y, w, h) {
          ctx.drawImage(c, cell[0] * cw, cell[1] * ch, cell[2] * cw, cell[3] * ch, cx + (x - w / 2) * u, cy + (y - h / 2) * u, w * u, h * u);
        }
        var feetY = 10;
        part(SPRITES.footOutline, -7, feetY, 64, 32);
        part(SPRITES.foot, -7, feetY, 64, 32);
        part(SPRITES.footOutline, 7, feetY, 64, 32);
        part(SPRITES.foot, 7, feetY, 64, 32);
        part(SPRITES.bodyOutline, 0, -4, 64, 64);
        part(SPRITES.body, 0, -4, 64, 64);
        var es = 64 * 0.4;
        part(SPRITES.eyes[0], -4.8 + 8, -4 - 3.2, es, es);
        part(SPRITES.eyes[0], 4.8 + 8, -4 - 3.2, es, es);
        return true;
      },
      renderTee: renderTee,
      renderWeapon: renderWeapon,
      renderHook: renderHook,
      renderFreezeBar: renderFreezeBar,
      renderEmote: renderEmote,
      renderSnowflake: renderSnowflake,
    };
  }

  root.DDTee = {
    SPRITES: SPRITES,
    ANIMS: ANIMS,
    teeAnim: teeAnim,
    ddnetColor: ddnetColor,
    skinColorable: skinColorable,
    skinTint: skinTint,
    freezeBarPieces: freezeBarPieces,
    flakeStep: flakeStep,
    rect: rect,
    loadPixels: loadPixels,
    skinNameOk: skinNameOk,
    createTeeKit: createTeeKit,
  };
})(window);
