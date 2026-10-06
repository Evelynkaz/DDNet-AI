// Task 7.4: the «Муха» tab — the connectome brain at work (docs/FLY.md §10, docs/formats.md §27).
//
// Data: after `{"type":"fly","hz":N}` the server sends one `fly_meta` JSON message (the layout) and then binary
// `DFLY` frames (one per decision, decimated). Everything here is read-only drawing: heat strips of the group
// activity over the last seconds, a polar plot of the eye, DN z-score bars, the decoded action, small time series.
// All text is set with `textContent` (never `innerHTML`); all drawing is on <canvas> (the CSP forbids inline styles).
(function () {
  "use strict";

  var HISTORY = 240; // frames kept (about 20 s at the bot's 12.5 Hz)
  var HEADER = 44;

  // Colours come from CSS tokens (app.css, `.fly-view`), read once per draw so the stylesheet owns the palette.
  var TOKENS = {};
  function readTokens() {
    var cs = getComputedStyle(document.getElementById("fly-view"));
    var names = [
      "panel", "bg", "text", "muted", "grid",
      "fam-vpn", "fam-an", "fam-central", "fam-dn",
      "ch-0", "ch-1", "ch-2", "ch-3", "ch-4", "ch-5", "ch-6",
      "div-neg", "div-pos", "div-mid",
      "ramp-0", "ramp-1", "ramp-2", "ramp-3", "ramp-4",
    ];
    names.forEach(function (n) {
      TOKENS[n] = cs.getPropertyValue("--fly-" + n).trim() || "#888888";
    });
  }

  var CHANNEL_LABELS = {
    opponent_position: "соперник",
    opponent_approach: "сближение (скорость)",
    opponent_hook: "хук соперника",
    other_players: "другие игроки",
    walls: "стены",
    freeze_death_tiles: "фриз / смерть",
    no_hook_tiles: "no-hook",
  };
  // Colour slot of each channel (fixed order, validated as a set; `--fly-ch-N`).
  var CHANNEL_SLOT = {
    opponent_position: 0,
    opponent_approach: 1,
    opponent_hook: 2,
    other_players: 3,
    no_hook_tiles: 4,
    walls: 5,
    freeze_death_tiles: 6,
  };
  var SCALAR_LABELS = {
    self_velocity_x: "vx",
    self_velocity_y: "vy",
    self_velocity_flow: "поток",
  };
  var HEAD_LABELS = {
    direction_left: "влево",
    direction_stop: "стоп",
    direction_right: "вправо",
    jump: "прыжок",
    hook: "хук",
    fire: "огонь",
    aim: "прицел",
  };
  var FAMILY_LABELS = { vpn: "глаз (VPN)", an: "тело (AN)", central: "центр", dn: "выход (DN)" };

  var shown = false;
  var connected = false;
  var send = null;
  var meta = null;
  var dirty = false;
  var rafId = 0;
  var desiredHz = 0;
  var frames = 0;
  var framesAt = 0;
  var fps = 0;

  // The newest decoded frame and the history ring (index = number of frames seen modulo HISTORY).
  var last = null;
  var hist = null;
  var count = 0;
  var enabled = {};

  function el(id) {
    return document.getElementById(id);
  }

  function fmt(x, digits) {
    if (!isFinite(x)) {
      return "—";
    }
    return x.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  // ---------------------------------------------------------------------------------------------
  // Decoding (the layout of docs/formats.md §27.1; ddai_fly::viz is the writer).
  // ---------------------------------------------------------------------------------------------

  function decode(buf) {
    if (!meta || buf.byteLength < HEADER) {
      return null;
    }
    var dv = new DataView(buf);
    if (dv.getUint8(0) !== 0x44 || dv.getUint8(1) !== 0x46 || dv.getUint8(2) !== 0x4c || dv.getUint8(3) !== 0x59 || dv.getUint8(4) !== 1) {
      return null;
    }
    var nGroups = dv.getUint16(34, true);
    var nDn = dv.getUint16(36, true);
    var rays = dv.getUint16(38, true);
    var bins = dv.getUint8(40);
    var nCh = dv.getUint8(41);
    var nSc = dv.getUint8(42);
    var want = HEADER + nGroups + nDn + nCh * rays * bins + nSc;
    if (buf.byteLength !== want || nGroups !== meta.groups.length || nDn !== meta.dn.length || rays !== meta.rays || bins !== meta.bins || nCh !== meta.channels.length) {
      return null; // not the layout this page was told about: wait for the next fly_meta
    }
    var flags = dv.getUint8(5);
    var f = {
      flags: flags,
      direction: dv.getInt8(6),
      tick: dv.getUint32(8, true),
      seq: dv.getUint32(12, true),
      latencyUs: dv.getUint16(16, true),
      aim: dv.getInt16(18, true) / meta.aim_scale,
      logits: [],
      chosenTotal: dv.getUint32(26, true),
      decisionsTotal: dv.getUint32(30, true),
      chosenValid: (flags & 1) !== 0,
      chosen: (flags & 2) !== 0,
      jump: (flags & 4) !== 0,
      hook: (flags & 8) !== 0,
      fire: (flags & 16) !== 0,
      groups: new Float32Array(nGroups),
      dn: new Float32Array(nDn),
      eye: [],
      scalars: [],
    };
    var i;
    for (i = 0; i < 6; i++) {
      f.logits.push(dv.getInt8(20 + i) / meta.logit_scale);
    }
    var at = HEADER;
    for (i = 0; i < nGroups; i++) {
      f.groups[i] = (dv.getUint8(at++) / 255) * meta.rate_max;
    }
    for (i = 0; i < nDn; i++) {
      f.dn[i] = (dv.getInt8(at++) / 127) * meta.z_clip;
    }
    for (var c = 0; c < nCh; c++) {
      var grid = new Float32Array(rays * bins);
      for (i = 0; i < grid.length; i++) {
        grid[i] = dv.getUint8(at++) / 255;
      }
      f.eye.push(grid);
    }
    for (i = 0; i < nSc; i++) {
      f.scalars.push(dv.getInt8(at++) / 127);
    }
    return f;
  }

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function softmax3(l) {
    var m = Math.max(l[0], l[1], l[2]);
    var e = [Math.exp(l[0] - m), Math.exp(l[1] - m), Math.exp(l[2] - m)];
    var s = e[0] + e[1] + e[2];
    return [e[0] / s, e[1] / s, e[2] / s];
  }

  // ---------------------------------------------------------------------------------------------
  // History
  // ---------------------------------------------------------------------------------------------

  function resetHistory() {
    count = 0;
    last = null;
    hist = null;
    if (!meta) {
      return;
    }
    var g = meta.groups.length;
    hist = {
      groups: new Float32Array(g * HISTORY),
      family: [new Float32Array(HISTORY), new Float32Array(HISTORY), new Float32Array(HISTORY), new Float32Array(HISTORY)],
      jump: new Float32Array(HISTORY),
      hook: new Float32Array(HISTORY),
      fire: new Float32Array(HISTORY),
      steer: new Float32Array(HISTORY),
      latency: new Float32Array(HISTORY),
      share: new Float32Array(HISTORY),
    };
  }

  var FAMILY_INDEX = { vpn: 0, an: 1, central: 2, dn: 3 };

  function pushFrame(f) {
    var g = meta.groups.length;
    var slot = count % HISTORY;
    var sums = [0, 0, 0, 0];
    var ns = [0, 0, 0, 0];
    for (var i = 0; i < g; i++) {
      hist.groups[i * HISTORY + slot] = f.groups[i];
      var fi = FAMILY_INDEX[meta.groups[i].family];
      var n = meta.groups[i].neurons;
      sums[fi] += f.groups[i] * n;
      ns[fi] += n;
    }
    for (var k = 0; k < 4; k++) {
      hist.family[k][slot] = ns[k] > 0 ? sums[k] / ns[k] : 0;
    }
    hist.jump[slot] = sigmoid(f.logits[3]);
    hist.hook[slot] = sigmoid(f.logits[4]);
    hist.fire[slot] = sigmoid(f.logits[5]);
    var p = softmax3(f.logits);
    hist.steer[slot] = p[2] - p[0];
    hist.latency[slot] = f.latencyUs;
    hist.share[slot] = f.chosenValid && f.decisionsTotal > 0 ? f.chosenTotal / f.decisionsTotal : NaN;
    count += 1;
    last = f;
  }

  /** Value `back` frames before the newest, for a ring `arr` (NaN before the start). */
  function histAt(arr, j, nFrames) {
    // j in 0..nFrames-1, oldest first
    var have = Math.min(count, HISTORY);
    var offset = nFrames - have;
    if (j < offset) {
      return NaN;
    }
    var idx = (count - have + (j - offset)) % HISTORY;
    return arr[idx];
  }

  /** Seconds the whole history spans, from the measured frame rate (20 s before it is known). */
  function windowSeconds() {
    return fps > 0 ? Math.round(HISTORY / fps) : 20;
  }

  // ---------------------------------------------------------------------------------------------
  // Canvas helpers
  // ---------------------------------------------------------------------------------------------

  function setup(canvas, cssHeight) {
    var dpr = window.devicePixelRatio || 1;
    var w = Math.max(100, Math.floor(canvas.parentElement.clientWidth));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = cssHeight + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, cssHeight);
    return { ctx: ctx, w: w, h: cssHeight };
  }

  function hexToRgb(hex) {
    var h = hex.replace("#", "");
    if (h.length === 3) {
      h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    }
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }

  function mix(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }

  /** Sequential ramp: 0 recedes into the panel, 1 is the lightest step. */
  function rampColor(t) {
    var stops = [TOKENS["ramp-0"], TOKENS["ramp-1"], TOKENS["ramp-2"], TOKENS["ramp-3"], TOKENS["ramp-4"]].map(hexToRgb);
    t = Math.max(0, Math.min(1, t));
    var x = t * (stops.length - 1);
    var i = Math.min(stops.length - 2, Math.floor(x));
    return mix(stops[i], stops[i + 1], x - i);
  }

  /** Diverging: negative -> blue arm, positive -> red arm, 0 -> neutral. `t` in [-1, 1]. */
  function divColor(t) {
    var mid = hexToRgb(TOKENS["div-mid"]);
    t = Math.max(-1, Math.min(1, t));
    var arm = hexToRgb(t < 0 ? TOKENS["div-neg"] : TOKENS["div-pos"]);
    return mix(mid, arm, Math.abs(t));
  }

  function css(rgb) {
    return "rgb(" + Math.round(rgb[0]) + "," + Math.round(rgb[1]) + "," + Math.round(rgb[2]) + ")";
  }

  function font(ctx, px, weight) {
    ctx.font = (weight || 400) + " " + px + "px 'Inter Var', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  }

  // ---------------------------------------------------------------------------------------------
  // The eye: a polar plot, ray = direction (0 = right, counter-clockwise, up = against gravity), ring = distance.
  // ---------------------------------------------------------------------------------------------

  function drawEye() {
    var canvas = el("fly-eye");
    // (a hidden or just-shown view can report a zero width: never a canvas smaller than the plot's margins)
    var size = Math.max(120, Math.min(canvas.parentElement.clientWidth, 420));
    var s = setup(canvas, size);
    var ctx = s.ctx;
    if (!meta || !last) {
      return;
    }
    var cx = s.w / 2;
    var cy = s.h / 2;
    var R = Math.max(20, Math.min(s.w, s.h) / 2 - 22);
    var r0 = R * 0.14;
    var rays = meta.rays;
    var bins = meta.bins;
    var dr = (R - r0) / bins;
    var half = Math.PI / rays;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = TOKENS.panel;
    for (var d = 0; d < rays; d++) {
      var th = (2 * Math.PI * d) / rays;
      for (var b = 0; b < bins; b++) {
        // The strongest enabled channel of the cell gives the hue, its value the opacity.
        var best = -1;
        var bv = 0.03;
        for (var c = 0; c < meta.channels.length; c++) {
          if (!enabled[meta.channels[c]]) {
            continue;
          }
          var v = last.eye[c][d * bins + b];
          if (v > bv) {
            bv = v;
            best = c;
          }
        }
        var ri = r0 + b * dr;
        var ro = ri + dr;
        ctx.beginPath();
        // canvas y points down, the ring's "up" is -y: negate the angle.
        ctx.arc(cx, cy, ro, -(th - half), -(th + half), true);
        ctx.arc(cx, cy, ri, -(th + half), -(th - half), false);
        ctx.closePath();
        if (best < 0) {
          ctx.fillStyle = TOKENS.grid;
          ctx.globalAlpha = 0.5;
        } else {
          ctx.fillStyle = TOKENS["ch-" + CHANNEL_SLOT[meta.channels[best]]];
          ctx.globalAlpha = 0.18 + 0.82 * Math.min(1, bv);
        }
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.stroke();
      }
    }
    // Aim: a line from the tee to the edge, ring convention.
    ctx.strokeStyle = TOKENS.text;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(last.aim) * (R + 6), cy - Math.sin(last.aim) * (R + 6));
    ctx.stroke();
    ctx.fillStyle = TOKENS.text;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(last.aim) * (R + 6), cy - Math.sin(last.aim) * (R + 6), 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
    ctx.fill();
    // Compass labels.
    ctx.fillStyle = TOKENS.muted;
    font(ctx, 11);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("вверх", cx, 9);
    ctx.fillText("вниз", cx, s.h - 9);
    ctx.textAlign = "left";
    ctx.fillText("вправо", s.w - 44, cy - 12);
    ctx.textAlign = "right";
    ctx.fillText("влево", 40, cy - 12);
  }

  function buildChannelLegend() {
    var box = el("fly-eye-legend");
    while (box.firstChild) {
      box.removeChild(box.firstChild);
    }
    if (!meta) {
      return;
    }
    meta.channels.forEach(function (name) {
      if (enabled[name] === undefined) {
        enabled[name] = true;
      }
      var b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.setAttribute("aria-pressed", enabled[name] ? "true" : "false");
      var sw = document.createElement("span");
      sw.className = "swatch ch-" + CHANNEL_SLOT[name];
      var tx = document.createElement("span");
      tx.textContent = CHANNEL_LABELS[name] || name;
      b.appendChild(sw);
      b.appendChild(tx);
      b.addEventListener("click", function () {
        enabled[name] = !enabled[name];
        b.setAttribute("aria-pressed", enabled[name] ? "true" : "false");
        markDirty();
      });
      box.appendChild(b);
    });
  }

  // ---------------------------------------------------------------------------------------------
  // Group activity: one heat strip per group, time on the x axis (oldest left), the current rate on the right.
  // ---------------------------------------------------------------------------------------------

  var ROW_H = 14;
  var LABEL_W = 96;
  var VALUE_W = 38;
  var stripGeom = null;

  function drawGroups() {
    var canvas = el("fly-groups");
    var n = meta ? meta.groups.length : 0;
    var rowH = ROW_H;
    var s = setup(canvas, Math.max(60, n * rowH + 34));
    var ctx = s.ctx;
    if (!meta || !last) {
      return;
    }
    var stripW = Math.max(40, s.w - LABEL_W - VALUE_W - 6);
    var x0 = LABEL_W;
    stripGeom = { x0: x0, w: stripW, rowH: rowH, n: n };
    // Scale: the largest group rate seen in the window (at least 1), so a quiet fly is not painted bright.
    var vmax = 1;
    var shownFrames = Math.min(count, HISTORY);
    var gi, j;
    for (gi = 0; gi < n; gi++) {
      for (j = 0; j < shownFrames; j++) {
        var vv = hist.groups[gi * HISTORY + j];
        if (vv > vmax) {
          vmax = vv;
        }
      }
    }
    var off = document.createElement("canvas");
    off.width = HISTORY;
    off.height = n;
    var octx = off.getContext("2d");
    var img = octx.createImageData(HISTORY, n);
    var bg = hexToRgb(TOKENS.bg);
    for (gi = 0; gi < n; gi++) {
      for (j = 0; j < HISTORY; j++) {
        var v = histAt(hist.groups.subarray(gi * HISTORY, (gi + 1) * HISTORY), j, HISTORY);
        var rgb = isNaN(v) ? bg : rampColor(Math.sqrt(Math.min(1, v / vmax)));
        var p = (gi * HISTORY + j) * 4;
        img.data[p] = rgb[0];
        img.data[p + 1] = rgb[1];
        img.data[p + 2] = rgb[2];
        img.data[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, x0, 0, stripW, n * rowH);
    // Row separators and labels.
    ctx.strokeStyle = TOKENS.panel;
    ctx.lineWidth = 2;
    font(ctx, 11);
    ctx.textBaseline = "middle";
    for (gi = 0; gi < n; gi++) {
      var y = gi * rowH;
      var g = meta.groups[gi];
      if (gi > 0) {
        ctx.beginPath();
        ctx.moveTo(x0, y);
        ctx.lineTo(x0 + stripW, y);
        ctx.stroke();
      }
      // Family chip (colour = which part of the pipeline), the label itself in text ink.
      ctx.fillStyle = TOKENS["fam-" + g.family];
      ctx.fillRect(0, y + 2, 4, rowH - 4);
      ctx.fillStyle = TOKENS.muted;
      ctx.textAlign = "left";
      ctx.fillText(g.label, 10, y + rowH / 2 + 0.5, LABEL_W - 14);
      ctx.fillStyle = TOKENS.text;
      ctx.textAlign = "right";
      ctx.fillText(fmt(last.groups[gi], 1), s.w - 2, y + rowH / 2 + 0.5);
    }
    // Colour scale under the strips.
    var ly = n * rowH + 8;
    for (var k = 0; k < stripW; k++) {
      ctx.fillStyle = css(rampColor(k / (stripW - 1)));
      ctx.fillRect(x0 + k, ly, 1, 6);
    }
    ctx.fillStyle = TOKENS.muted;
    font(ctx, 10);
    ctx.textAlign = "left";
    ctx.fillText("0", x0, ly + 12);
    ctx.textAlign = "right";
    ctx.fillText("до " + fmt(vmax, 1) + " (√-шкала)", x0 + stripW, ly + 12);
  }

  function buildFamilyLegend() {
    var box = el("fly-fam-legend");
    while (box.firstChild) {
      box.removeChild(box.firstChild);
    }
    ["vpn", "an", "central", "dn"].forEach(function (f) {
      var s = document.createElement("span");
      s.className = "legend-item";
      var sw = document.createElement("span");
      sw.className = "swatch fam-" + f;
      var tx = document.createElement("span");
      tx.textContent = FAMILY_LABELS[f];
      s.appendChild(sw);
      s.appendChild(tx);
      box.appendChild(s);
    });
  }

  // ---------------------------------------------------------------------------------------------
  // DN outputs: the action heads (mean z-score of their DN) and every DN as one diverging strip.
  // ---------------------------------------------------------------------------------------------

  function drawDn() {
    var canvas = el("fly-dn");
    var heads = meta ? meta.heads : [];
    var rowH = 20;
    var s = setup(canvas, Math.max(60, heads.length * rowH + 34));
    var ctx = s.ctx;
    if (!meta || !last) {
      return;
    }
    var labelW = 70;
    var valueW = 38;
    var barX = labelW;
    var barW = Math.max(40, s.w - labelW - valueW - 6);
    var mid = barX + barW / 2;
    var zmax = 4; // the bars saturate at +-4 sigma; the number shows the true mean
    font(ctx, 12);
    ctx.textBaseline = "middle";
    for (var h = 0; h < heads.length; h++) {
      var slots = heads[h].dn;
      var sum = 0;
      for (var i = 0; i < slots.length; i++) {
        sum += last.dn[slots[i]];
      }
      var z = slots.length ? sum / slots.length : 0;
      var y = h * rowH;
      ctx.fillStyle = TOKENS.muted;
      ctx.textAlign = "left";
      ctx.fillText(HEAD_LABELS[heads[h].action] || heads[h].action, 0, y + rowH / 2);
      var t = Math.max(-1, Math.min(1, z / zmax));
      ctx.fillStyle = css(divColor(t >= 0 ? Math.max(0.35, t) : Math.min(-0.35, t)));
      var w = (barW / 2) * Math.abs(t);
      // Bars grow from the centre line and are anchored there (rounded only at the data end).
      if (t >= 0) {
        roundRectEnd(ctx, mid, y + 3, w, rowH - 6, true);
      } else {
        roundRectEnd(ctx, mid - w, y + 3, w, rowH - 6, false);
      }
      ctx.fillStyle = TOKENS.text;
      ctx.textAlign = "right";
      ctx.fillText(fmt(z, 1), s.w - 2, y + rowH / 2);
    }
    ctx.strokeStyle = TOKENS.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mid, 0);
    ctx.lineTo(mid, heads.length * rowH);
    ctx.stroke();
    // All DN as cells in output order.
    var cy = heads.length * rowH + 12;
    var nDn = meta.dn.length;
    var cw = (s.w - 2) / nDn;
    for (var k = 0; k < nDn; k++) {
      ctx.fillStyle = css(divColor(last.dn[k] / meta.z_clip));
      ctx.fillRect(1 + k * cw, cy, Math.max(1, cw - 0.5), 14);
    }
  }

  function roundRectEnd(ctx, x, y, w, h, roundRight) {
    if (w < 1) {
      ctx.fillRect(x, y, Math.max(w, 1), h);
      return;
    }
    var r = Math.min(4, w, h / 2);
    ctx.beginPath();
    if (roundRight) {
      ctx.moveTo(x, y);
      ctx.lineTo(x + w - r, y);
      ctx.arcTo(x + w, y, x + w, y + r, r);
      ctx.lineTo(x + w, y + h - r);
      ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
      ctx.lineTo(x, y + h);
    } else {
      ctx.moveTo(x + w, y);
      ctx.lineTo(x + r, y);
      ctx.arcTo(x, y, x, y + r, r);
      ctx.lineTo(x, y + h - r);
      ctx.arcTo(x, y + h, x + r, y + h, r);
      ctx.lineTo(x + w, y + h);
    }
    ctx.closePath();
    ctx.fill();
  }

  // ---------------------------------------------------------------------------------------------
  // Time series
  // ---------------------------------------------------------------------------------------------

  function drawSeries(canvasId, series, opts) {
    var canvas = el(canvasId);
    var s = setup(canvas, opts.height || 110);
    var ctx = s.ctx;
    if (!meta || !last) {
      return;
    }
    var padL = 34;
    var padR = 6;
    var padT = 6;
    var padB = 16;
    var w = s.w - padL - padR;
    var h = s.h - padT - padB;
    var lo = opts.min;
    var hi = opts.max;
    var j, k;
    if (hi === undefined || lo === undefined) {
      var mn = Infinity;
      var mx = -Infinity;
      series.forEach(function (ser) {
        for (j = 0; j < HISTORY; j++) {
          var v = histAt(ser.data, j, HISTORY);
          if (!isNaN(v)) {
            mn = Math.min(mn, v);
            mx = Math.max(mx, v);
          }
        }
      });
      if (!isFinite(mn)) {
        mn = 0;
        mx = 1;
      }
      lo = opts.min !== undefined ? opts.min : Math.min(0, mn);
      hi = opts.max !== undefined ? opts.max : Math.max(mx, lo + 1e-6) * 1.1;
    }
    // Recessive grid: top, middle, bottom.
    ctx.strokeStyle = TOKENS.grid;
    ctx.lineWidth = 1;
    font(ctx, 10);
    ctx.fillStyle = TOKENS.muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (k = 0; k <= 2; k++) {
      var gy = padT + (h * k) / 2;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + w, gy);
      ctx.stroke();
      ctx.fillText(fmt(hi - ((hi - lo) * k) / 2, opts.digits === undefined ? 1 : opts.digits), padL - 4, gy);
    }
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    series.forEach(function (ser) {
      ctx.strokeStyle = ser.color;
      ctx.beginPath();
      var pen = false;
      for (j = 0; j < HISTORY; j++) {
        var v = histAt(ser.data, j, HISTORY);
        if (isNaN(v)) {
          pen = false;
          continue;
        }
        var x = padL + (w * j) / (HISTORY - 1);
        var y = padT + h - (h * (v - lo)) / (hi - lo || 1);
        if (pen) {
          ctx.lineTo(x, y);
        } else {
          ctx.moveTo(x, y);
          pen = true;
        }
      }
      ctx.stroke();
    });
    ctx.fillStyle = TOKENS.muted;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("−" + windowSeconds() + " с", padL, s.h - 3);
    ctx.textAlign = "right";
    ctx.fillText("сейчас", padL + w, s.h - 3);
  }

  function buildLegend(boxId, items) {
    var box = el(boxId);
    while (box.firstChild) {
      box.removeChild(box.firstChild);
    }
    items.forEach(function (it) {
      var s = document.createElement("span");
      s.className = "legend-item";
      var sw = document.createElement("span");
      sw.className = "swatch line " + it.cls;
      var tx = document.createElement("span");
      tx.textContent = it.label;
      s.appendChild(sw);
      s.appendChild(tx);
      box.appendChild(s);
    });
  }

  function drawAllSeries() {
    drawSeries("fly-ts-fam", [
      { data: hist ? hist.family[0] : null, color: TOKENS["fam-vpn"] },
      { data: hist ? hist.family[1] : null, color: TOKENS["fam-an"] },
      { data: hist ? hist.family[2] : null, color: TOKENS["fam-central"] },
      { data: hist ? hist.family[3] : null, color: TOKENS["fam-dn"] },
    ], { min: 0, digits: 1 });
    drawSeries("fly-ts-act", [
      { data: hist ? hist.jump : null, color: TOKENS["ch-0"] },
      { data: hist ? hist.hook : null, color: TOKENS["ch-1"] },
      { data: hist ? hist.fire : null, color: TOKENS["ch-2"] },
    ], { min: 0, max: 1, digits: 1 });
    drawSeries("fly-ts-lat", [{ data: hist ? hist.latency : null, color: TOKENS["fam-an"] }], { min: 0, digits: 0 });
    if (last && last.chosenValid) {
      drawSeries("fly-ts-share", [{ data: hist.share, color: TOKENS["fam-dn"] }], { min: 0, max: 1, digits: 1 });
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Text parts: header, action, table.
  // ---------------------------------------------------------------------------------------------

  var DIR_TEXT = { "-1": "влево", "0": "стоп", "1": "вправо" };

  function setBar(id, p) {
    var node = el(id);
    node.value = Math.round(p * 100);
    node.setAttribute("aria-valuenow", String(Math.round(p * 100)));
  }

  function updateText() {
    if (!meta) {
      return;
    }
    el("fly-brain").textContent = meta.name + (meta.role === "proposer" ? " (предлагает точному поиску)" : "");
    var b = meta.bundle;
    el("fly-bundle").textContent = b ? b.name : "нет весов (начальные параметры)";
    el("fly-hash").textContent = b ? b.sha256.slice(0, 16) : "—";
    el("fly-hash").title = b ? b.sha256 : "";
    el("fly-rate").textContent = fps > 0 ? fmt(fps, 1) + " кадр/с" : "—";
    el("fly-window").textContent = String(windowSeconds());
    if (!last) {
      return;
    }
    el("fly-tick").textContent = String(last.tick);
    el("fly-latency").textContent = fmt(last.latencyUs, 0) + " мкс";
    var p = softmax3(last.logits);
    setBar("fly-p-left", p[0]);
    setBar("fly-p-stop", p[1]);
    setBar("fly-p-right", p[2]);
    el("fly-pv-left").textContent = fmt(p[0] * 100, 0) + "%";
    el("fly-pv-stop").textContent = fmt(p[1] * 100, 0) + "%";
    el("fly-pv-right").textContent = fmt(p[2] * 100, 0) + "%";
    el("fly-dir").textContent = DIR_TEXT[String(last.direction)] || "—";
    [["jump", last.jump, 3], ["hook", last.hook, 4], ["fire", last.fire, 5]].forEach(function (a) {
      var node = el("fly-act-" + a[0]);
      node.classList.toggle("on", a[1]);
      el("fly-act-" + a[0] + "-p").textContent = fmt(sigmoid(last.logits[a[2]]) * 100, 0) + "%";
      el("fly-act-" + a[0] + "-s").textContent = a[1] ? "да" : "нет";
    });
    var deg = (last.aim * 180) / Math.PI;
    el("fly-aim").textContent = fmt(deg, 0) + "°";
    var sc = [];
    meta.scalars.forEach(function (name, i) {
      sc.push((SCALAR_LABELS[name] || name) + " " + fmt(last.scalars[i], 2));
    });
    el("fly-scalars").textContent = sc.join(" · ");

    var prop = el("fly-proposer");
    prop.hidden = !last.chosenValid;
    el("fly-ts-share-box").hidden = !last.chosenValid;
    if (last.chosenValid) {
      var share = last.decisionsTotal > 0 ? (100 * last.chosenTotal) / last.decisionsTotal : 0;
      el("fly-prop-share").textContent = fmt(share, 0) + "% (" + last.chosenTotal + " из " + last.decisionsTotal + ")";
      el("fly-prop-now").textContent = last.chosen ? "да, сыграно предложение мухи" : "нет, точный поиск выбрал своё";
    }
  }

  var tableTimer = 0;
  function updateTable() {
    var body = el("fly-table-body");
    if (!meta || !last || !body || !el("fly-table").open) {
      return;
    }
    while (body.firstChild) {
      body.removeChild(body.firstChild);
    }
    meta.groups.forEach(function (g, i) {
      var tr = document.createElement("tr");
      [FAMILY_LABELS[g.family], g.label, String(g.neurons), fmt(last.groups[i], 2)].forEach(function (t, k) {
        var td = document.createElement("td");
        td.textContent = t;
        if (k >= 2) {
          td.className = "num";
        }
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------------------------------------
  // Status line and lifecycle
  // ---------------------------------------------------------------------------------------------

  function setState(kind, text) {
    var dot = el("fly-dot");
    dot.classList.toggle("dot-on", kind === "live");
    dot.classList.toggle("dot-off", kind !== "live");
    el("fly-state").textContent = text;
    var empty = kind !== "live" && !last;
    el("fly-empty").hidden = !empty;
    el("fly-content").hidden = empty;
  }

  function refreshState() {
    if (!connected) {
      setState("off", "нет связи с сервером…");
    } else if (!meta) {
      setState("wait", "потока нет: бот не запущен, его мозг не муха, либо ещё не дошли данные");
    } else if (!last) {
      setState("wait", "жду первый кадр… (с включённым вейблоком муха решает только в бою: кадры редки)");
    } else if (Date.now() - lastFrameAt > 3000) {
      setState("wait", "кадры перестали приходить (пауза между играми или бот остановился)");
    } else {
      setState("live", "муха работает");
    }
  }

  var lastFrameAt = 0;

  function markDirty() {
    dirty = true;
    if (!rafId && shown) {
      rafId = requestAnimationFrame(render);
    }
  }

  function render() {
    rafId = 0;
    if (!shown || !dirty) {
      return;
    }
    dirty = false;
    readTokens();
    refreshState();
    if (meta && last) {
      updateText();
      drawEye();
      drawGroups();
      drawDn();
      drawAllSeries();
    }
  }

  function subscribe() {
    if (send && connected && shown) {
      send({ type: "fly", hz: desiredHz });
    }
  }

  function chooseHz() {
    var sel = el("fly-hz");
    var v = sel ? parseFloat(sel.value) : NaN;
    if (isFinite(v) && v > 0) {
      return v;
    }
    return window.matchMedia && window.matchMedia("(max-width: 640px)").matches ? 6 : 12;
  }

  var FlyPanel = {
    attach: function (sender) {
      send = sender;
    },
    onShown: function () {
      shown = true;
      var sel = el("fly-hz");
      if (sel && !sel.dataset.touched) {
        sel.value = window.matchMedia && window.matchMedia("(max-width: 640px)").matches ? "6" : "12";
      }
      desiredHz = chooseHz();
      subscribe();
      framesAt = Date.now();
      frames = 0;
      markDirty();
      if (!tableTimer) {
        tableTimer = setInterval(function () {
          if (shown) {
            updateTable();
            refreshState();
          }
        }, 1000);
      }
    },
    onHidden: function () {
      if (!shown) {
        return;
      }
      shown = false;
      if (send && connected) {
        send({ type: "fly", hz: 0 });
      }
      if (tableTimer) {
        clearInterval(tableTimer);
        tableTimer = 0;
      }
    },
    onConnectionChanged: function (on) {
      connected = on;
      if (on) {
        subscribe(); // after a reconnect the server has forgotten us
      }
      markDirty();
    },
    onMeta: function (m) {
      var same = meta && m && JSON.stringify(meta.groups) === JSON.stringify(m.groups) && meta.frame_bytes === m.frame_bytes;
      meta = m && m.v === 1 ? m : null;
      if (!same) {
        resetHistory();
        buildChannelLegend();
        buildFamilyLegend();
        buildLegend("fly-fam-ts-legend", [
          { cls: "fam-vpn", label: FAMILY_LABELS.vpn },
          { cls: "fam-an", label: FAMILY_LABELS.an },
          { cls: "fam-central", label: FAMILY_LABELS.central },
          { cls: "fam-dn", label: FAMILY_LABELS.dn },
        ]);
        buildLegend("fly-act-ts-legend", [
          { cls: "ch-0", label: "прыжок" },
          { cls: "ch-1", label: "хук" },
          { cls: "ch-2", label: "огонь" },
        ]);
      }
      markDirty();
    },
    onFrame: function (buf) {
      var f = decode(buf);
      if (!f) {
        return;
      }
      pushFrame(f);
      lastFrameAt = Date.now();
      frames += 1;
      var now = Date.now();
      if (now - framesAt >= 2000) {
        fps = (frames * 1000) / (now - framesAt);
        frames = 0;
        framesAt = now;
      }
      markDirty();
    },
    /** For tests: the newest decoded frame. */
    _last: function () {
      return last;
    },
  };

  window.FlyPanel = FlyPanel;

  window.addEventListener("resize", function () {
    markDirty();
  });
  document.addEventListener("DOMContentLoaded", function () {
    var sel = el("fly-hz");
    if (sel) {
      sel.addEventListener("change", function () {
        sel.dataset.touched = "1";
        desiredHz = chooseHz();
        subscribe();
      });
    }
    var det = el("fly-table");
    if (det) {
      det.addEventListener("toggle", updateTable);
    }
    // Hover on the heat strips: the group and its value at that moment.
    var gc = el("fly-groups");
    var tip = el("fly-tip");
    if (gc && tip) {
      gc.addEventListener("pointermove", function (e) {
        if (!stripGeom || !meta || !last) {
          return;
        }
        var r = gc.getBoundingClientRect();
        var x = e.clientX - r.left;
        var y = e.clientY - r.top;
        var gi = Math.floor(y / stripGeom.rowH);
        if (gi < 0 || gi >= stripGeom.n) {
          tip.hidden = true;
          return;
        }
        var g = meta.groups[gi];
        var text = FAMILY_LABELS[g.family] + " · " + g.label + " · " + g.neurons + " нейр.: " + fmt(last.groups[gi], 2);
        if (x >= stripGeom.x0 && x <= stripGeom.x0 + stripGeom.w) {
          var j = Math.floor(((x - stripGeom.x0) / stripGeom.w) * HISTORY);
          var v = histAt(hist.groups.subarray(gi * HISTORY, (gi + 1) * HISTORY), j, HISTORY);
          if (!isNaN(v)) {
            text = FAMILY_LABELS[g.family] + " · " + g.label + ": " + fmt(v, 2) + " (" + fmt(((HISTORY - 1 - j) * windowSeconds()) / HISTORY, 1) + " с назад)";
          }
        }
        tip.textContent = text;
        tip.hidden = false;
        tip.style.left = Math.min(Math.max(0, x + 12), Math.max(0, r.width - 200)) + "px";
        tip.style.top = y + 16 + "px";
      });
      gc.addEventListener("pointerleave", function () {
        tip.hidden = true;
      });
    }
  });
})();
