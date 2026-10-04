/* The «Игра» tab's WebGL2 map renderer: the real DDNet map in the client's draw order (task 5.10).

   Layers, groups, parallax and clipping follow DDNet's game/map/map_renderer.cpp and render_layer.cpp; tile
   transforms, quad envelopes and the camera maths follow render_map.cpp and engine/graphics.cpp. A tile layer is drawn
   with ONE quad and a fragment shader that looks the tile up (index and flags) in a data texture and samples a 256-layer
   texture array of the tileset (mipmapped, no bleeding between tiles, like the client's own 2D-array path); a quads layer
   is one dynamic vertex buffer per layer. Everything the page draws in the world (tees, hooks, weapons, emoticons)
   goes through the same sprite batch, between the background and the foreground layers.

   Portions derived from DDNet (https://github.com/ddnet/ddnet), zlib license, and from the camera, parallax and
   envelope code of Wranked1/DDNet-AI (GPL-3.0) src/bot/webDraw.ts and webView.ts.
   Copyright (C) 2007-2014 Magnus Auvinen (Teeworlds); Copyright (C) DDRace and DDNet contributors.
   This is an altered version, not the original software. */
(function (root) {
  "use strict";

  var TILE = 32;

  // ---------------------------------------------------------------------------------------------
  // Camera maths (engine/graphics.cpp: CalcScreenParams / MapScreenToWorld)
  // ---------------------------------------------------------------------------------------------

  /** The world rectangle one group shows: [left, top, width, height]. */
  function groupView(cx, cy, zoom, aspect, px, py, ox, oy) {
    var amount = 1150 * 1000;
    var f = Math.sqrt(amount) / Math.sqrt(aspect);
    var w = f * aspect;
    var h = f;
    if (w > 1500) {
      w = 1500;
      h = w / aspect;
    }
    if (h > 1050) {
      h = 1050;
      w = h * aspect;
    }
    w *= zoom;
    h *= zoom;
    var pz = Math.min(100, Math.max(0, Math.max(px, py)));
    var scale = (pz * (zoom - 1) + 100) / 100 / zoom;
    w *= scale;
    h *= scale;
    return [ox + cx * (px / 100) - w / 2, oy + cy * (py / 100) - h / 2, w, h];
  }

  // ---------------------------------------------------------------------------------------------
  // Envelopes (render_map.cpp: RenderEvalEnvelope). `points` is the flat wire array
  // [time_ms, curve, v0, v1, v2, v3] * n with values in 22.10 fixed point.
  // ---------------------------------------------------------------------------------------------

  function envEval(env, timeMs, channels, out) {
    var p = env.p;
    var n = (p.length / 6) | 0;
    var ch = Math.min(channels, env.c, 4);
    if (n === 0 || ch <= 0) return out;
    if (n === 1) {
      for (var a = 0; a < ch; a++) out[a] = p[2 + a] / 1024;
      return out;
    }
    var maxT = p[(n - 1) * 6];
    var t = maxT > 0 ? timeMs % maxT : 0;
    if (t < 0) t += maxT;
    var found = -1;
    var lo = 0;
    var hi = n - 2;
    var ti = Math.trunc(t);
    while (lo <= hi) {
      var mid = lo + ((hi - lo) >> 1);
      var t0 = p[mid * 6];
      var t1 = p[(mid + 1) * 6];
      if (ti >= t0 && ti < t1) {
        found = mid;
        break;
      }
      if (ti < t0) hi = mid - 1;
      else lo = mid + 1;
    }
    if (found < 0) {
      for (var b = 0; b < ch; b++) out[b] = p[(n - 1) * 6 + 2 + b] / 1024;
      return out;
    }
    var span = p[(found + 1) * 6] - p[found * 6];
    if (span <= 0) {
      for (var d = 0; d < ch; d++) out[d] = p[found * 6 + 2 + d] / 1024;
      return out;
    }
    var k = (t - p[found * 6]) / span;
    switch (p[found * 6 + 1]) {
      case 0: // step
        k = 0;
        break;
      case 2: // slow
        k = k * k * k;
        break;
      case 3: // fast
        k = 1 - k;
        k = 1 - k * k * k;
        break;
      case 4: // smooth
        k = -2 * k * k * k + 3 * k * k;
        break;
      default: // linear; a bezier point is read as linear (its handles are not in the scene)
        break;
    }
    for (var e = 0; e < ch; e++) {
      var v0 = p[found * 6 + 2 + e] / 1024;
      var v1 = p[(found + 1) * 6 + 2 + e] / 1024;
      out[e] = v0 + (v1 - v0) * k;
    }
    return out;
  }

  // ---------------------------------------------------------------------------------------------
  // Scene decoding (docs/formats.md §35.1)
  // ---------------------------------------------------------------------------------------------

  /** Inflates a raw-DEFLATE body. */
  function inflateRaw(bytes) {
    if (typeof DecompressionStream === "undefined") {
      return Promise.reject(new Error("DecompressionStream unsupported"));
    }
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
    return new Response(stream).arrayBuffer();
  }

  /** Parses an inflated scene body into {meta, blob (Uint8Array), blobI32}. Throws on a malformed body. */
  function parseScene(buf) {
    var dv = new DataView(buf);
    if (buf.byteLength < 12 || dv.getUint32(0, true) !== 0x43535744 /* "DWSC" little-endian */) {
      throw new Error("not a scene");
    }
    if (dv.getUint8(4) !== 1) {
      throw new Error("unsupported scene version");
    }
    var jsonLen = dv.getUint32(8, true);
    if (12 + jsonLen > buf.byteLength || jsonLen % 4 !== 0) {
      throw new Error("bad scene header");
    }
    var meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, jsonLen)));
    var blobStart = 12 + jsonLen;
    var blobLen = (buf.byteLength - blobStart) & ~3;
    return {
      meta: meta,
      blob: new Uint8Array(buf, blobStart, blobLen),
      blobI32: new Int32Array(buf, blobStart, blobLen >> 2),
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Shaders
  // ---------------------------------------------------------------------------------------------

  var VS_WORLD =
    "#version 300 es\n" +
    "in vec2 aPos;\n" +
    "uniform vec4 uView;\n" +
    "out vec2 vWorld;\n" +
    "void main(){\n" +
    "  vWorld = aPos;\n" +
    "  vec2 n = (aPos - uView.xy) / uView.zw;\n" +
    "  gl_Position = vec4(n.x * 2.0 - 1.0, 1.0 - n.y * 2.0, 0.0, 1.0);\n" +
    "}\n";

  // One quad covers the part of the layer on screen; the tile is looked up per fragment.
  var FS_TILE =
    "#version 300 es\n" +
    "precision highp float;\n" +
    "precision highp int;\n" +
    "precision highp usampler2D;\n" +
    "precision highp sampler2DArray;\n" +
    "in vec2 vWorld;\n" +
    "uniform usampler2D uTiles;\n" +
    "uniform sampler2DArray uSet;\n" +
    "uniform ivec2 uSize;\n" +
    "uniform vec4 uColor;\n" +
    "uniform int uExtend;\n" + // bit0 left, bit1 right, bit2 top, bit3 bottom: the edge tile repeats outwards
    "uniform int uFlat;\n" + // 1: untextured tiles, solid colour
    "uniform int uBias;\n" + // subtracted from the tile index to get the sheet layer (the speedup arrows are shifted by one)
    "uniform vec4 uU[8];\n" + // per TableFlag: u of the corners TL, TR, BR, BL
    "uniform vec4 uV[8];\n" +
    "out vec4 oColor;\n" +
    "void main(){\n" +
    "  vec2 tc = vWorld / 32.0;\n" +
    "  ivec2 ti = ivec2(floor(tc));\n" +
    "  vec2 f = tc - vec2(ti);\n" +
    "  if (ti.x < 0) { if ((uExtend & 1) == 0) discard; ti.x = 0; }\n" +
    "  else if (ti.x >= uSize.x) { if ((uExtend & 2) == 0) discard; ti.x = uSize.x - 1; }\n" +
    "  if (ti.y < 0) { if ((uExtend & 4) == 0) discard; ti.y = 0; }\n" +
    "  else if (ti.y >= uSize.y) { if ((uExtend & 8) == 0) discard; ti.y = uSize.y - 1; }\n" +
    "  uvec2 t = texelFetch(uTiles, ti, 0).rg;\n" +
    "  if (t.r == 0u) discard;\n" +
    "  if (uFlat == 1) { oColor = uColor; return; }\n" +
    "  uint fl = t.g;\n" +
    "  int tf = int(fl & 3u) + int((fl & 8u) >> 1);\n" +
    "  vec4 cu = uU[tf];\n" +
    "  vec4 cv = uV[tf];\n" +
    "  float u = mix(mix(cu.x, cu.y, f.x), mix(cu.w, cu.z, f.x), f.y);\n" +
    "  float v = mix(mix(cv.x, cv.y, f.x), mix(cv.w, cv.z, f.x), f.y);\n" +
    "  vec4 c = textureGrad(uSet, vec3(u, v, float(int(t.r) - uBias)), dFdx(tc), dFdy(tc));\n" +
    "  oColor = c * uColor;\n" +
    "}\n";

  var VS_SPRITE =
    "#version 300 es\n" +
    "in vec2 aPos;\n" +
    "in vec2 aUV;\n" +
    "in vec4 aColor;\n" +
    "uniform vec4 uView;\n" +
    "out vec2 vUV;\n" +
    "out vec4 vColor;\n" +
    "void main(){\n" +
    "  vUV = aUV;\n" +
    "  vColor = aColor;\n" +
    "  vec2 n = (aPos - uView.xy) / uView.zw;\n" +
    "  gl_Position = vec4(n.x * 2.0 - 1.0, 1.0 - n.y * 2.0, 0.0, 1.0);\n" +
    "}\n";

  var FS_SPRITE =
    "#version 300 es\n" +
    "precision highp float;\n" +
    "in vec2 vUV;\n" +
    "in vec4 vColor;\n" +
    "uniform sampler2D uTex;\n" +
    "out vec4 oColor;\n" +
    "void main(){\n" +
    "  oColor = texture(uTex, vUV) * vColor;\n" +
    "}\n";

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error("shader: " + gl.getShaderInfoLog(s));
    }
    return s;
  }

  function program(gl, vs, fs) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error("program: " + gl.getProgramInfoLog(p));
    }
    return p;
  }

  // Tile texture coordinates per flag (render_layer.cpp: CalculateTexCoords). TableFlag = (flags & 3) + ((flags & 8) >> 1):
  // corners TL, TR, BR, BL start as X = 0 1 1 0, Y = 0 0 1 1; xflip rotates X by 2, yflip rotates Y by 2, rotate rotates both by 3.
  function rotateLeft(a, n) {
    return a.slice(n).concat(a.slice(0, n));
  }
  function tileTables() {
    var U = [];
    var V = [];
    for (var tf = 0; tf < 8; tf++) {
      var x = [0, 1, 1, 0];
      var y = [0, 0, 1, 1];
      if (tf & 1) x = rotateLeft(x, 2);
      if (tf & 2) y = rotateLeft(y, 2);
      if (tf & 4) {
        x = rotateLeft(x, 3);
        y = rotateLeft(y, 3);
      }
      U.push.apply(U, x);
      V.push.apply(V, y);
    }
    return { U: new Float32Array(U), V: new Float32Array(V) };
  }

  // ---------------------------------------------------------------------------------------------
  // Pixels: transparent pixels take the colour of a neighbour (the client's DilateImage), so a texture filter never
  // blends in black from the edge of a sprite.
  // ---------------------------------------------------------------------------------------------

  function dilate(rgba, w, h) {
    var n = w * h;
    var has = new Uint8Array(n);
    var holes = 0;
    for (var i = 0; i < n; i++) {
      if (rgba[i * 4 + 3] !== 0) has[i] = 1;
      else holes++;
    }
    if (holes === 0 || holes === n) {
      return rgba;
    }
    var out = new Uint8ClampedArray(rgba);
    for (var pass = 0; pass < 2; pass++) {
      var next = has.slice();
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          var p = y * w + x;
          if (has[p]) continue;
          var r = 0;
          var g = 0;
          var b = 0;
          var c = 0;
          if (x > 0 && has[p - 1]) { r += out[(p - 1) * 4]; g += out[(p - 1) * 4 + 1]; b += out[(p - 1) * 4 + 2]; c++; }
          if (x + 1 < w && has[p + 1]) { r += out[(p + 1) * 4]; g += out[(p + 1) * 4 + 1]; b += out[(p + 1) * 4 + 2]; c++; }
          if (y > 0 && has[p - w]) { r += out[(p - w) * 4]; g += out[(p - w) * 4 + 1]; b += out[(p - w) * 4 + 2]; c++; }
          if (y + 1 < h && has[p + w]) { r += out[(p + w) * 4]; g += out[(p + w) * 4 + 1]; b += out[(p + w) * 4 + 2]; c++; }
          if (c > 0) {
            out[p * 4] = (r / c) | 0;
            out[p * 4 + 1] = (g / c) | 0;
            out[p * 4 + 2] = (b / c) | 0;
            next[p] = 1;
          }
        }
      }
      has = next;
    }
    return out;
  }

  // ---------------------------------------------------------------------------------------------
  // The renderer
  // ---------------------------------------------------------------------------------------------

  function createRenderer(canvas) {
    var gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      powerPreference: "high-performance",
    });
    if (!gl) {
      return null;
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    var tileProg = program(gl, VS_WORLD, FS_TILE);
    var spriteProg = program(gl, VS_SPRITE, FS_SPRITE);
    var tileLoc = {
      view: gl.getUniformLocation(tileProg, "uView"),
      tiles: gl.getUniformLocation(tileProg, "uTiles"),
      set: gl.getUniformLocation(tileProg, "uSet"),
      size: gl.getUniformLocation(tileProg, "uSize"),
      color: gl.getUniformLocation(tileProg, "uColor"),
      extend: gl.getUniformLocation(tileProg, "uExtend"),
      flat: gl.getUniformLocation(tileProg, "uFlat"),
      bias: gl.getUniformLocation(tileProg, "uBias"),
      u: gl.getUniformLocation(tileProg, "uU"),
      v: gl.getUniformLocation(tileProg, "uV"),
      pos: gl.getAttribLocation(tileProg, "aPos"),
    };
    var spriteLoc = {
      view: gl.getUniformLocation(spriteProg, "uView"),
      tex: gl.getUniformLocation(spriteProg, "uTex"),
      pos: gl.getAttribLocation(spriteProg, "aPos"),
      uv: gl.getAttribLocation(spriteProg, "aUV"),
      color: gl.getAttribLocation(spriteProg, "aColor"),
    };
    var tables = tileTables();

    // A scratch buffer for the quad that covers a tile layer, and a vertex array for it.
    var rectBuf = gl.createBuffer();
    var rectVao = gl.createVertexArray();
    gl.bindVertexArray(rectVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, rectBuf);
    gl.enableVertexAttribArray(tileLoc.pos);
    gl.vertexAttribPointer(tileLoc.pos, 2, gl.FLOAT, false, 8, 0);
    var rectData = new Float32Array(12);

    // ---- sprite batch: 6 vertices per sprite, interleaved x y u v rgba8 (5 words) ----
    var MAX_SPRITES = 4096;
    var STRIDE = 20;
    var spriteBuf = gl.createBuffer();
    var spriteVao = gl.createVertexArray();
    var spriteWords = new ArrayBuffer(MAX_SPRITES * 6 * STRIDE);
    var spriteF = new Float32Array(spriteWords);
    var spriteU = new Uint32Array(spriteWords);
    gl.bindVertexArray(spriteVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, spriteBuf);
    gl.bufferData(gl.ARRAY_BUFFER, spriteWords.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(spriteLoc.pos);
    gl.vertexAttribPointer(spriteLoc.pos, 2, gl.FLOAT, false, STRIDE, 0);
    gl.enableVertexAttribArray(spriteLoc.uv);
    gl.vertexAttribPointer(spriteLoc.uv, 2, gl.FLOAT, false, STRIDE, 8);
    gl.enableVertexAttribArray(spriteLoc.color);
    gl.vertexAttribPointer(spriteLoc.color, 4, gl.UNSIGNED_BYTE, true, STRIDE, 16);
    gl.bindVertexArray(null);

    var drawCalls = 0; // draw calls issued since frame() started (a test hook reports it)
    var spriteVerts = 0;
    var spriteCount = 0; // vertices queued
    var spriteTex = null;
    var spriteView = [0, 0, 1, 1];

    function flushSprites() {
      if (spriteCount === 0) {
        return;
      }
      gl.useProgram(spriteProg);
      gl.uniform4f(spriteLoc.view, spriteView[0], spriteView[1], spriteView[2], spriteView[3]);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, spriteTex);
      gl.uniform1i(spriteLoc.tex, 0);
      gl.bindVertexArray(spriteVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, spriteBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, spriteF, 0, spriteCount * 5);
      gl.drawArrays(gl.TRIANGLES, 0, spriteCount);
      gl.bindVertexArray(null);
      drawCalls++;
      spriteVerts += spriteCount;
      spriteCount = 0;
    }

    function pack(r, g, b, a) {
      return (
        ((Math.max(0, Math.min(255, (a * 255 + 0.5) | 0)) << 24) |
          (Math.max(0, Math.min(255, (b * 255 + 0.5) | 0)) << 16) |
          (Math.max(0, Math.min(255, (g * 255 + 0.5) | 0)) << 8) |
          Math.max(0, Math.min(255, (r * 255 + 0.5) | 0))) >>>
        0
      );
    }

    function vert(i, x, y, u, v, c) {
      var o = i * 5;
      spriteF[o] = x;
      spriteF[o + 1] = y;
      spriteF[o + 2] = u;
      spriteF[o + 3] = v;
      spriteU[o + 4] = c;
    }

    /** One textured quad of any shape (corners TL, TR, BL, BR, as the file stores quads) with a colour per corner. */
    function freeform(tex, p, uv, c) {
      if (spriteTex !== tex.tex) {
        flushSprites();
        spriteTex = tex.tex;
      }
      if (spriteCount + 6 > MAX_SPRITES * 6) {
        flushSprites();
      }
      var n = spriteCount;
      // Two triangles on the TL-BR diagonal, like the client's QuadsDrawFreeform.
      vert(n, p[0], p[1], uv[0], uv[1], c[0]);
      vert(n + 1, p[2], p[3], uv[2], uv[3], c[1]);
      vert(n + 2, p[6], p[7], uv[6], uv[7], c[3]);
      vert(n + 3, p[0], p[1], uv[0], uv[1], c[0]);
      vert(n + 4, p[6], p[7], uv[6], uv[7], c[3]);
      vert(n + 5, p[4], p[5], uv[4], uv[5], c[2]);
      spriteCount += 6;
    }

    var fp = new Float32Array(8);
    var fuv = new Float32Array(8);
    var fc = [0, 0, 0, 0];

    /** A sprite: the texture region (u0, v0)-(u1, v1) centred at (x, y), |w| by h (a negative w mirrors it left-right, a negative h top-bottom), rotated by rot. */
    function sprite(tex, u0, v0, u1, v1, x, y, w, h, rot, r, g, b, a) {
      if (w < 0) {
        var t = u0;
        u0 = u1;
        u1 = t;
        w = -w;
      }
      if (h < 0) {
        var t2 = v0;
        v0 = v1;
        v1 = t2;
        h = -h;
      }
      var hw = w / 2;
      var hh = h / 2;
      var cos = Math.cos(rot);
      var sin = Math.sin(rot);
      // corners TL, TR, BL, BR relative to the centre, before rotation
      var cx = [-hw, hw, -hw, hw];
      var cy = [-hh, -hh, hh, hh];
      for (var k = 0; k < 4; k++) {
        fp[k * 2] = x + cx[k] * cos - cy[k] * sin;
        fp[k * 2 + 1] = y + cx[k] * sin + cy[k] * cos;
      }
      fuv[0] = u0;
      fuv[1] = v0;
      fuv[2] = u1;
      fuv[3] = v0;
      fuv[4] = u0;
      fuv[5] = v1;
      fuv[6] = u1;
      fuv[7] = v1;
      var col = pack(r, g, b, a);
      fc[0] = col;
      fc[1] = col;
      fc[2] = col;
      fc[3] = col;
      freeform(tex, fp, fuv, fc);
    }

    // ---- textures ----

    function texture2D(rgba, w, h, opts) {
      var t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      var mips = !!(opts && opts.mipmaps);
      if (mips) {
        gl.generateMipmap(gl.TEXTURE_2D);
      }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mips ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      var wrap = opts && opts.repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
      return { tex: t, w: w, h: h };
    }

    /** A 256-layer array texture from a 16x16 tileset image. */
    function tileArray(rgba, w, h) {
      var tw = (w / 16) | 0;
      var th = (h / 16) | 0;
      if (tw < 1 || th < 1) {
        return null;
      }
      var levels = Math.floor(Math.log2(Math.max(tw, th))) + 1;
      var t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, levels, gl.RGBA8, tw, th, 256);
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, w);
      gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, h); // with skipped rows the image height must be given, or the upload is refused
      for (var i = 0; i < 256; i++) {
        gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, (i % 16) * tw);
        gl.pixelStorei(gl.UNPACK_SKIP_ROWS, ((i / 16) | 0) * th);
        gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, i, tw, th, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      }
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
      gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0);
      gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
      gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
      gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return { tex: t };
    }

    function dataTexture(bytes, w, h) {
      var t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8UI, w, h, 0, gl.RG_INTEGER, gl.UNSIGNED_BYTE, bytes);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    }

    var white = texture2D(new Uint8Array([255, 255, 255, 255]), 1, 1, {});

    // ---- scene state ----
    var scene = null; // {meta, blob, blobI32, groups: [...], images: [...]}
    var entityArray = null;
    var arrowArray = null;
    var envScratch = [0, 0, 0, 1];
    var envScratch2 = [0, 0, 0, 1];
    var quadBuf = new Float32Array(0);

    function disposeScene() {
      if (!scene) {
        return;
      }
      scene.images.forEach(function (im) {
        if (im.array) gl.deleteTexture(im.array.tex);
        if (im.flat) gl.deleteTexture(im.flat.tex);
      });
      scene.groups.forEach(function (g) {
        g.layers.forEach(function (l) {
          if (l.dataTex) gl.deleteTexture(l.dataTex);
        });
      });
      scene = null;
    }

    /** Takes a parsed scene (see parseScene) and prepares its layers; images follow with setImage(). */
    function setScene(parsed) {
      disposeScene();
      var meta = parsed.meta;
      var s = {
        meta: meta,
        blob: parsed.blob,
        blobI32: parsed.blobI32,
        env: meta.env || [],
        images: (meta.images || []).map(function (im) {
          return { name: im.n, w: im.w, h: im.h, external: !!im.x, state: "wait", array: null, flat: null };
        }),
        groups: [],
        gameGroup: -1,
        gameLayer: null,
      };
      var passedGame = false;
      (meta.groups || []).forEach(function (g, gi) {
        var group = { ox: g.ox, oy: g.oy, px: g.px, py: g.py, clip: g.clip, layers: [] };
        (g.layers || []).forEach(function (l) {
          if (l.k === "t") {
            var count = l.w * l.h;
            var tiles = parsed.blob.subarray(l.o, l.o + count * 2);
            if (tiles.length < count * 2) {
              return;
            }
            var layer = {
              kind: "t",
              role: l.r,
              w: l.w,
              h: l.h,
              detail: !!l.d,
              color: l.c,
              colorEnv: l.ce,
              colorEnvOff: l.co,
              image: l.i,
              tiles: tiles,
              aux: l.a >= 0 ? parsed.blob.subarray(l.a, l.a + count * (l.r === "speedup" ? 4 : l.r === "switch" ? 2 : 1)) : null,
              dataTex: null,
              extend: 0,
              bg: !passedGame,
              entity: l.r !== "visual",
            };
            if (l.r === "game") {
              passedGame = true;
              s.gameGroup = gi;
              s.gameLayer = layer;
              layer.bg = true; // the game layer ends the background
            }
            // Which edges have a tile: only those repeat outwards (the client's border tiles).
            layer.extend = edgeMask(tiles, l.w, l.h);
            group.layers.push(layer);
          } else if (l.k === "q") {
            var n = l.n;
            var i32 = parsed.blobI32.subarray(l.o >> 2, (l.o >> 2) + n * 26);
            if (i32.length < n * 26) {
              return;
            }
            group.layers.push({
              kind: "q",
              detail: !!l.d,
              image: l.i,
              n: n,
              i32: i32,
              u8: parsed.blob.subarray(l.o, l.o + n * 104),
              bg: !passedGame,
              entity: false,
            });
          }
        });
        s.groups.push(group);
      });
      scene = s;
      return s;
    }

    function edgeMask(tiles, w, h) {
      var m = 0;
      for (var y = 0; y < h && !(m & 1); y++) if (tiles[y * w * 2] !== 0) m |= 1;
      for (var y2 = 0; y2 < h && !(m & 2); y2++) if (tiles[(y2 * w + w - 1) * 2] !== 0) m |= 2;
      for (var x = 0; x < w && !(m & 4); x++) if (tiles[x * 2] !== 0) m |= 4;
      for (var x2 = 0; x2 < w && !(m & 8); x2++) if (tiles[((h - 1) * w + x2) * 2] !== 0) m |= 8;
      return m;
    }

    /** Gives image `index` its pixels (RGBA, straight alpha) or marks it failed (`pixels` null). */
    function setImage(index, pixels, w, h) {
      if (!scene || !scene.images[index]) {
        return;
      }
      var im = scene.images[index];
      if (!pixels) {
        im.state = "bad";
        return;
      }
      var clean = dilate(pixels instanceof Uint8ClampedArray ? pixels : new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, pixels.length), w, h);
      im.w = w;
      im.h = h;
      im.pixels = clean;
      // The GPU copies the layers use are built now (a quad image is a plain texture, a tileset an array), and the CPU copy of
      // the pixels is dropped: on a texture-heavy map it is tens of MiB of JS heap for as long as the page lives.
      var uses = imageUses(index);
      if (uses.tiles) imageArray(im);
      if (uses.quads) imageFlat(im);
      im.pixels = null;
      im.state = "ready";
    }

    /** Which kinds of layer draw image `index`: a tileset (array texture) and/or a quad texture (plain texture). */
    function imageUses(index) {
      var uses = { tiles: false, quads: false };
      scene.groups.forEach(function (g) {
        g.layers.forEach(function (l) {
          if (l.image !== index) return;
          if (l.kind === "q") uses.quads = true;
          else uses.tiles = true;
        });
      });
      return uses;
    }

    function imageArray(im) {
      if (!im.array) {
        im.array = tileArray(im.pixels, im.w, im.h) || { tex: null };
      }
      return im.array;
    }

    function imageFlat(im) {
      if (!im.flat) {
        im.flat = texture2D(im.pixels, im.w, im.h, { mipmaps: true, repeat: true });
      }
      return im.flat;
    }

    function setEntityImages(entPixels, ew, eh, arrowPixels, aw, ah) {
      if (entityArray) gl.deleteTexture(entityArray.tex);
      if (arrowArray) gl.deleteTexture(arrowArray.tex);
      entityArray = entPixels ? tileArray(dilate(entPixels, ew, eh), ew, eh) : null;
      arrowArray = arrowPixels ? tileArray(dilate(arrowPixels, aw, ah), aw, ah) : null;
    }

    // ---- drawing ----

    function layerData(layer) {
      if (!layer.dataTex) {
        var bytes = layer.tiles;
        if (layer.role === "speedup") {
          // The arrow sheet is indexed by angle % 90, rotated by quadrant (render_layer.cpp: FillTmpTileSpeedup).
          bytes = new Uint8Array(layer.w * layer.h * 2);
          for (var i = 0; i < layer.w * layer.h; i++) {
            var type = layer.tiles[i * 2];
            var force = layer.aux[i * 4];
            var max = layer.aux[i * 4 + 1];
            var angle = (layer.aux[i * 4 + 2] | (layer.aux[i * 4 + 3] << 8)) << 16 >> 16;
            if (!((type === 28 && force !== 0) || (type === 29 && (force !== 0 || max !== 0)))) continue;
            var a = ((angle % 360) + 360) % 360;
            bytes[i * 2] = (a % 90) + 1; // 0 means "no tile" in the data texture, so the sheet index is shifted by one
            bytes[i * 2 + 1] = a >= 270 ? 1 | 2 | 8 : a >= 180 ? 1 | 2 : a >= 90 ? 8 : 0;
          }
        }
        layer.dataTex = dataTexture(bytes, layer.w, layer.h);
      }
      return layer.dataTex;
    }

    var view = [0, 0, 1, 1];

    function drawTileLayer(layer, groupView, tintR, tintG, tintB, alpha, set, flat, extendOverride) {
      var x0 = groupView[0];
      var y0 = groupView[1];
      var x1 = x0 + groupView[2];
      var y1 = y0 + groupView[3];
      var ext = extendOverride === undefined ? layer.extend : extendOverride;
      var lx0 = ext & 1 ? x0 : Math.max(x0, 0);
      var lx1 = ext & 2 ? x1 : Math.min(x1, layer.w * TILE);
      var ly0 = ext & 4 ? y0 : Math.max(y0, 0);
      var ly1 = ext & 8 ? y1 : Math.min(y1, layer.h * TILE);
      if (lx0 >= lx1 || ly0 >= ly1) {
        return;
      }
      flushSprites();
      gl.useProgram(tileProg);
      gl.uniform4f(tileLoc.view, groupView[0], groupView[1], groupView[2], groupView[3]);
      gl.uniform2i(tileLoc.size, layer.w, layer.h);
      gl.uniform4f(tileLoc.color, tintR, tintG, tintB, alpha);
      gl.uniform1i(tileLoc.extend, ext);
      gl.uniform1i(tileLoc.flat, flat ? 1 : 0);
      gl.uniform1i(tileLoc.bias, layer.role === "speedup" ? 1 : 0);
      gl.uniform4fv(tileLoc.u, tables.U);
      gl.uniform4fv(tileLoc.v, tables.V);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, layerData(layer));
      gl.uniform1i(tileLoc.tiles, 0);
      if (!flat) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, set.tex);
        gl.uniform1i(tileLoc.set, 1);
      } else {
        // The sampler must still be backed by something valid.
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, entityArray ? entityArray.tex : null);
        gl.uniform1i(tileLoc.set, 1);
      }
      // Corners: a limited rectangle, two triangles.
      rectData[0] = lx0; rectData[1] = ly0;
      rectData[2] = lx1; rectData[3] = ly0;
      rectData[4] = lx0; rectData[5] = ly1;
      rectData[6] = lx1; rectData[7] = ly0;
      rectData[8] = lx1; rectData[9] = ly1;
      rectData[10] = lx0; rectData[11] = ly1;
      gl.bindVertexArray(rectVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, rectBuf);
      gl.bufferData(gl.ARRAY_BUFFER, rectData, gl.STREAM_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
      drawCalls++;
    }

    function colorEnv(index, offset, timeMs, out) {
      var env = scene.env[index];
      out[0] = 1; out[1] = 1; out[2] = 1; out[3] = 1;
      if (!env || env.c <= 0) {
        return out;
      }
      envEval(env, timeMs + offset, 4, out);
      return out;
    }

    function drawQuadLayer(layer, gview, timeMs, alpha) {
      var im = layer.image >= 0 ? scene.images[layer.image] : null;
      var tex = white;
      if (im) {
        if (im.state !== "ready") {
          return;
        }
        tex = imageFlat(im);
      }
      var i32 = layer.i32;
      var u8 = layer.u8;
      var left = gview[0];
      var top = gview[1];
      var right = left + gview[2];
      var bottom = top + gview[3];
      var pts = fp;
      var vc = [0, 0, 0, 0];
      for (var q = 0; q < layer.n; q++) {
        var b = q * 26; // i32 index of the record
        var ob = q * 104; // byte offset of the record
        var ca = alpha;
        var er = 1;
        var eg = 1;
        var eb = 1;
        var ceIndex = i32[b + 24];
        if (ceIndex >= 0) {
          colorEnv(ceIndex, i32[b + 25], timeMs, envScratch);
          er = envScratch[0];
          eg = envScratch[1];
          eb = envScratch[2];
          ca *= envScratch[3];
        }
        if (ca <= 0) {
          continue;
        }
        var ox = 0;
        var oy = 0;
        var rot = 0;
        var peIndex = i32[b + 22];
        if (peIndex >= 0 && scene.env[peIndex]) {
          envScratch2[0] = 0; envScratch2[1] = 0; envScratch2[2] = 0; envScratch2[3] = 1;
          envEval(scene.env[peIndex], timeMs + i32[b + 23], 3, envScratch2);
          ox = envScratch2[0];
          oy = envScratch2[1];
          rot = (envScratch2[2] / 180) * Math.PI;
        }
        var cx = i32[b + 8] / 1024;
        var cy = i32[b + 9] / 1024;
        var cos = Math.cos(rot);
        var sin = Math.sin(rot);
        var minX = Infinity;
        var maxX = -Infinity;
        var minY = Infinity;
        var maxY = -Infinity;
        for (var k = 0; k < 4; k++) {
          var x = i32[b + k * 2] / 1024;
          var y = i32[b + k * 2 + 1] / 1024;
          if (rot !== 0) {
            var dx = x - cx;
            var dy = y - cy;
            x = cx + dx * cos - dy * sin;
            y = cy + dx * sin + dy * cos;
          }
          x += ox;
          y += oy;
          pts[k * 2] = x;
          pts[k * 2 + 1] = y;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
        if (maxX < left || minX > right || maxY < top || minY > bottom) {
          continue;
        }
        for (var c = 0; c < 4; c++) {
          vc[c] = pack((u8[ob + 40 + c * 4] / 255) * er, (u8[ob + 41 + c * 4] / 255) * eg, (u8[ob + 42 + c * 4] / 255) * eb, (u8[ob + 43 + c * 4] / 255) * ca);
        }
        for (var t = 0; t < 4; t++) {
          fuv[t * 2] = i32[b + 14 + t * 2] / 1024;
          fuv[t * 2 + 1] = i32[b + 15 + t * 2] / 1024;
        }
        freeform(tex, pts, fuv, vc);
      }
    }

    /**
     * Draws a frame. `s`: {camX, camY, zoom, aspect, timeMs, entities (0 map | 1 both | 2 entities only), onWorld}.
     * The world callback runs after the game layer's group and before the foreground, with the sprite helpers.
     */
    function frame(s) {
      var w = canvas.width;
      var h = canvas.height;
      gl.viewport(0, 0, w, h);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(0.0, 0.0, 0.0, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      var plain = groupView(s.camX, s.camY, s.zoom, s.aspect, 100, 100, 0, 0);
      var stats = { layers: 0, quads: 0, draws: 0, verts: 0 };
      drawCalls = 0;
      spriteVerts = 0;
      var overlay = s.entities || 0;
      var mapAlpha = overlay === 0 ? 1 : overlay === 1 ? 0.5 : 0;
      var worldDone = false;

      function world() {
        if (worldDone) {
          return;
        }
        worldDone = true;
        gl.disable(gl.SCISSOR_TEST);
        spriteView = plain;
        if (s.onWorld) s.onWorld(api, plain);
        flushSprites();
      }

      if (scene) {
        var passedGame = false;
        for (var gi = 0; gi < scene.groups.length; gi++) {
          var g = scene.groups[gi];
          var gview = groupView(s.camX, s.camY, s.zoom, s.aspect, g.px, g.py, g.ox, g.oy);
          flushSprites();
          spriteView = gview;
          var clipOn = false;
          if (g.clip) {
            // The clip is in the plain view's coordinates (render_layer.cpp: CRenderLayerGroup::DoRender).
            var cl = g.clip;
            var left = ((cl[0] - plain[0]) / plain[2]) * w;
            var right = ((cl[0] + cl[2] - plain[0]) / plain[2]) * w;
            var topPx = ((cl[1] - plain[1]) / plain[3]) * h;
            var bottomPx = ((cl[1] + cl[3] - plain[1]) / plain[3]) * h;
            if (right < 0 || left > w || bottomPx < 0 || topPx > h) {
              // off screen: nothing of this group shows, but the game layer may be in it
              var hasGame = g.layers.some(function (l) { return l.role === "game"; });
              if (!hasGame) continue;
            } else {
              gl.enable(gl.SCISSOR_TEST);
              var sx = Math.round(left);
              var sy = Math.round(h - bottomPx);
              gl.scissor(sx, sy, Math.max(0, Math.round(right) - sx), Math.max(0, Math.round(bottomPx) - Math.round(topPx)));
              clipOn = true;
            }
          }
          for (var li = 0; li < g.layers.length; li++) {
            var l = g.layers[li];
            if (!passedGame && l.role === "game") {
              // The game layer: the background is over after it is (optionally) drawn as entities.
              if (overlay > 0) drawEntityLayer(l, gview, plain, s);
              passedGame = true;
              if (clipOn) gl.disable(gl.SCISSOR_TEST);
              world();
              spriteView = gview;
              if (clipOn) gl.enable(gl.SCISSOR_TEST);
              continue;
            }
            if (l.entity) {
              if (overlay > 0) drawEntityLayer(l, gview, plain, s);
              continue;
            }
            if (overlay === 2) continue; // entities only
            if (l.kind === "t") {
              var im = l.image >= 0 ? scene.images[l.image] : null;
              if (im && im.state !== "ready") continue;
              var env = colorEnv(l.colorEnv, l.colorEnvOff, s.timeMs, envScratch);
              var alpha = (l.color[3] / 255) * env[3] * mapAlpha;
              if (alpha <= 0) continue;
              drawTileLayer(l, gview, (l.color[0] / 255) * env[0], (l.color[1] / 255) * env[1], (l.color[2] / 255) * env[2], alpha, im ? imageArray(im) : null, !im);
              stats.layers++;
            } else {
              drawQuadLayer(l, gview, s.timeMs, mapAlpha);
              flushSprites();
              stats.quads += l.n;
            }
          }
          if (clipOn) gl.disable(gl.SCISSOR_TEST);
        }
      }
      world(); // a map without a game layer (or no scene yet) still gets its tees
      flushSprites();
      stats.draws = drawCalls;
      stats.verts = spriteVerts;
      return stats;
    }

    // The entity overlay: the physics layers drawn with the entities tileset (the client's "entities" view).
    function drawEntityLayer(l, gview, plain, s) {
      if (l.role === "tune") return; // coloured by tune number in the client; not drawn here
      var set = entityArray;
      if (l.role === "speedup") set = arrowArray;
      if (!set || !set.tex) return;
      drawTileLayer(l, gview, 1, 1, 1, 1, set, false, 0);
    }

    /** Test hook: the share of non-black pixels and the number of distinct colours in a grid sampled from the drawing buffer
     *  (call right after frame(), in the same task, while the buffer is still readable). */
    function sampleStats() {
      var w = canvas.width;
      var h = canvas.height;
      var n = 28;
      // One read of the whole buffer (many single-pixel reads would each wait for the GPU), sampled in JS.
      var all = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, all);
      var seen = {};
      var nonBlack = 0;
      var total = 0;
      for (var iy = 0; iy < n; iy++) {
        for (var ix = 0; ix < n; ix++) {
          var o = (Math.floor(((iy + 0.5) / n) * h) * w + Math.floor(((ix + 0.5) / n) * w)) * 4;
          total++;
          if (all[o] + all[o + 1] + all[o + 2] > 12) nonBlack++;
          seen[(all[o] >> 3) | ((all[o + 1] >> 3) << 5) | ((all[o + 2] >> 3) << 10)] = 1;
        }
      }
      return { nonBlack: nonBlack / total, colors: Object.keys(seen).length };
    }

    var api = {
      gl: gl,
      sampleStats: sampleStats,
      groupView: groupView,
      parseScene: parseScene,
      setScene: setScene,
      setImage: setImage,
      setEntityImages: setEntityImages,
      texture2D: texture2D,
      sprite: sprite,
      freeform: freeform,
      pack: pack,
      flush: flushSprites,
      frame: frame,
      white: white,
      scene: function () {
        return scene;
      },
      dispose: disposeScene,
      resize: function (w, h) {
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w;
          canvas.height = h;
        }
      },
      info: function () {
        return gl.getParameter(gl.VERSION);
      },
    };
    return api;
  }

  root.DDMap = {
    createRenderer: createRenderer,
    groupView: groupView,
    envEval: envEval,
    inflateRaw: inflateRaw,
    parseScene: parseScene,
    dilate: dilate,
  };
})(window);
