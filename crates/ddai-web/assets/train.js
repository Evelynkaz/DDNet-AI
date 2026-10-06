// Task 5.8: the «Обучение» tab — the fly and control-model training runs, read-only (docs/formats.md §29).
//
// Data: `GET /api/train/runs` (experiments and runs) and `GET /api/train/run?exp=…&run=…` (one run: curves, DAgger rounds, arena
// results with Wilson intervals, checkpoints). Both are plain authenticated GETs; the page asks again every POLL_MS while the
// tab is open and a shown run is still running (the timer's requests carry `poll=1`: they do not keep the session alive).
// All text is set with `textContent` and all drawing is SVG built from DOM nodes (never `innerHTML`); colours come from the
// `--tr-*` tokens and classes in app.css (the CSP forbids inline styles).
(function () {
  "use strict";

  var POLL_MS = 10000;
  var MAX_PICK = 4;
  var SVG_NS = "http://www.w3.org/2000/svg";

  // ---------------------------------------------------------------------------------------
  // DOM and formatting helpers
  // ---------------------------------------------------------------------------------------

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) {
      n.className = cls;
    }
    if (text !== undefined && text !== null) {
      n.textContent = text;
    }
    return n;
  }

  function sv(tag, attrs, cls) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        n.setAttribute(k, String(attrs[k]));
      });
    }
    if (cls) {
      n.setAttribute("class", cls);
    }
    return n;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function isNum(x) {
    return typeof x === "number" && isFinite(x);
  }

  function num(x, digits) {
    return isNum(x) ? x.toFixed(digits === undefined ? 2 : digits).replace(".", ",") : "—";
  }

  function pct(x, digits) {
    return isNum(x) ? (x * 100).toFixed(digits === undefined ? 1 : digits).replace(".", ",") + "%" : "—";
  }

  function ciText(c) {
    if (!c) {
      return "—";
    }
    return pct(c.p) + " [" + (c.lo * 100).toFixed(1).replace(".", ",") + "–" + (c.hi * 100).toFixed(1).replace(".", ",") + "]";
  }

  function bytesText(n) {
    if (!isNum(n)) {
      return "—";
    }
    if (n < 1024) {
      return n + " Б";
    }
    if (n < 1024 * 1024) {
      return num(n / 1024, 1) + " КиБ";
    }
    return num(n / (1024 * 1024), 1) + " МиБ";
  }

  function ago(s) {
    if (!isNum(s)) {
      return "—";
    }
    if (s < 60) {
      return "только что";
    }
    if (s < 3600) {
      return Math.floor(s / 60) + " мин назад";
    }
    if (s < 86400) {
      return Math.floor(s / 3600) + " ч назад";
    }
    return Math.floor(s / 86400) + " дн. назад";
  }

  var STATE_TEXT = {
    running: "▶ идёт",
    done: "✓ готово",
    stalled: "⏸ не обновляется",
    unknown: "? нет статуса",
  };
  var STATE_HINT = {
    running: "Файлы запуска менялись недавно",
    done: "Обучение завершено",
    stalled: "Обучение не закончено, но файлы не менялись больше двух часов: остановлено или упало",
    unknown: "status.json не прочитан",
  };

  function keyOf(exp, run) {
    return exp + "/" + run;
  }

  function splitKey(key) {
    var i = key.indexOf("/");
    return { exp: key.slice(0, i), run: key.slice(i + 1) };
  }

  // ---------------------------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------------------------

  function apiOnce(path, poll) {
    var url = path + (poll ? (path.indexOf("?") >= 0 ? "&" : "?") + "poll=1" : "");
    return fetch(url, { credentials: "same-origin", cache: "no-store", headers: { Accept: "application/json" } }).then(function (r) {
      return r.json().then(
        function (data) {
          return { status: r.status, data: data };
        },
        function () {
          return { status: r.status, data: null };
        },
      );
    });
  }

  // `503 busy` means every disk-scan slot of the server stayed taken for its whole wait: ask once more after a pause.
  function api(path, poll) {
    return apiOnce(path, poll).then(function (res) {
      if (res.status !== 503) {
        return res;
      }
      return new Promise(function (resolve) {
        setTimeout(resolve, 700);
      }).then(function () {
        return apiOnce(path, poll);
      });
    });
  }

  // ---------------------------------------------------------------------------------------
  // Charts: SVG line charts with phase markers, a crosshair tooltip, a legend and a table view
  // ---------------------------------------------------------------------------------------

  var DASH_CLASS = ["", "d1", "d2", "d3"];

  function niceStep(raw) {
    var exp = Math.floor(Math.log(raw) / Math.LN10);
    var f = raw / Math.pow(10, exp);
    var nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
    return nf * Math.pow(10, exp);
  }

  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) {
      return [lo];
    }
    var step = niceStep((hi - lo) / count);
    var out = [];
    var t = Math.ceil(lo / step - 1e-9) * step;
    for (var guard = 0; t <= hi + step * 1e-9 && guard < 50; guard++) {
      out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
      t += step;
    }
    return out;
  }

  function tickText(v, step) {
    var digits = step >= 1 ? 0 : Math.min(4, Math.max(0, Math.ceil(-Math.log(step) / Math.LN10 - 1e-9)));
    return v.toFixed(digits).replace(".", ",");
  }

  function keySwatch(slot, dash) {
    var s = sv("svg", { width: 26, height: 8, viewBox: "0 0 26 8", "aria-hidden": "true" }, "tr-key");
    s.appendChild(sv("line", { x1: 1, y1: 4, x2: 25, y2: 4 }, "tr-line s" + slot + (dash ? " " + DASH_CLASS[dash] : "")));
    return s;
  }

  // series: [{label, short, slot, dash, points: [[x, y, extra?]], width?}]
  // opts: {title, aria, series, markers: [{x, label}], xFmt, yFmt, xName, clamp: [lo, hi], phaseAt(x)}
  // Returns {node, redraw}.
  function lineChart(opts) {
    var wrap = el("div", "tr-chart");
    if (opts.title) {
      wrap.appendChild(el("h3", null, opts.title));
    }
    var series = opts.series.filter(function (s) {
      return s.points && s.points.length > 0;
    });
    if (series.length === 0) {
      wrap.appendChild(el("p", "tr-empty", opts.empty || "Нет данных."));
      return { node: wrap, redraw: function () {} };
    }
    var plot = el("div", "tr-plot");
    var svg = sv("svg", { role: "img", "aria-label": opts.aria || opts.title || "График" }, "tr-svg");
    var tip = el("div", "tr-tip");
    tip.hidden = true;
    plot.appendChild(svg);
    plot.appendChild(tip);
    wrap.appendChild(plot);

    var legend = el("div", "tr-legend");
    if (series.length >= 2 || opts.alwaysLegend) {
      series.forEach(function (s) {
        var item = el("span", "tr-legend-item");
        item.appendChild(keySwatch(s.slot, s.dash));
        item.appendChild(el("span", null, s.label));
        legend.appendChild(item);
      });
      wrap.appendChild(legend);
    }

    var fmtX = opts.xFmt || function (x) { return String(Math.round(x)); };
    var fmtY = opts.yFmt || function (y) { return num(y, 3); };
    var scale = null; // set by draw(), used by the pointer handlers

    // Sorted union of the x values (for the table and for snapping the crosshair).
    var xsSet = {};
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        xsSet[p[0]] = true;
      });
    });
    var xs = Object.keys(xsSet).map(Number).sort(function (a, b) { return a - b; });

    function draw() {
      var width = Math.floor(plot.clientWidth);
      if (width < 120) {
        return; // hidden (display: none) or not laid out yet: drawn again when shown
      }
      var narrow = width < 480;
      var height = narrow ? 210 : 260;
      var labelled = series.length >= 2 && series.length <= 4;
      var ml = 46;
      var mr = labelled ? 34 : 12;
      var mt = 16;
      var mb = 24;
      var pw = width - ml - mr;
      var ph = height - mt - mb;

      var xlo = Infinity;
      var xhi = -Infinity;
      var ylo = Infinity;
      var yhi = -Infinity;
      series.forEach(function (s) {
        s.points.forEach(function (p) {
          xlo = Math.min(xlo, p[0]);
          xhi = Math.max(xhi, p[0]);
          ylo = Math.min(ylo, p[1]);
          yhi = Math.max(yhi, p[1]);
        });
      });
      if (xhi <= xlo) {
        xlo -= 1;
        xhi += 1;
      }
      if (yhi <= ylo) {
        var padFlat = Math.abs(yhi) > 0 ? Math.abs(yhi) * 0.1 : 0.5;
        ylo -= padFlat;
        yhi += padFlat;
      }
      var padY = (yhi - ylo) * 0.06;
      ylo -= padY;
      yhi += padY;
      if (opts.clamp) {
        ylo = Math.max(ylo, opts.clamp[0]);
        yhi = Math.min(yhi, opts.clamp[1]);
        if (yhi <= ylo) {
          yhi = ylo + 0.01;
        }
      }
      function X(x) {
        return ml + ((x - xlo) / (xhi - xlo)) * pw;
      }
      function Y(y) {
        return mt + (1 - (y - ylo) / (yhi - ylo)) * ph;
      }
      scale = { X: X, Y: Y, ml: ml, mt: mt, pw: pw, ph: ph, xlo: xlo, xhi: xhi, width: width, height: height };

      clear(svg);
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);
      svg.setAttribute("height", String(height));

      var yTicks = niceTicks(ylo, yhi, narrow ? 4 : 6);
      var yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1;
      yTicks.forEach(function (t) {
        svg.appendChild(sv("line", { x1: ml, x2: ml + pw, y1: Y(t), y2: Y(t) }, "tr-grid"));
        var label = sv("text", { x: ml - 6, y: Y(t) + 4, "text-anchor": "end" });
        label.textContent = (opts.yTick || function (v, st) { return tickText(v, st); })(t, yStep);
        svg.appendChild(label);
      });
      svg.appendChild(sv("line", { x1: ml, x2: ml + pw, y1: mt + ph, y2: mt + ph }, "tr-axis"));
      var xTicks = niceTicks(xlo, xhi, narrow ? 3 : 5);
      xTicks.forEach(function (t) {
        var label = sv("text", { x: X(t), y: height - 6, "text-anchor": "middle" });
        label.textContent = fmtX(t);
        svg.appendChild(label);
        svg.appendChild(sv("line", { x1: X(t), x2: X(t), y1: mt + ph, y2: mt + ph + 3 }, "tr-axis"));
      });

      (opts.markers || []).forEach(function (m) {
        if (m.x < xlo || m.x > xhi) {
          return;
        }
        svg.appendChild(sv("line", { x1: X(m.x), x2: X(m.x), y1: mt, y2: mt + ph }, "tr-mark"));
        var t = sv("text", { x: X(m.x) + 3, y: mt + 9 });
        t.textContent = m.label;
        svg.appendChild(t);
      });

      var showDots = series.every(function (s) {
        return s.points.length <= 40;
      });
      series.forEach(function (s) {
        var d = "";
        s.points.forEach(function (p, i) {
          d += (i === 0 ? "M" : "L") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1);
        });
        var line = sv("path", { d: d }, "tr-line s" + s.slot + (s.dash ? " " + DASH_CLASS[s.dash] : ""));
        if (s.width) {
          line.setAttribute("stroke-width", String(s.width));
        }
        svg.appendChild(line);
        if (showDots) {
          s.points.forEach(function (p) {
            svg.appendChild(sv("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3.5 }, "tr-dot s" + s.slot));
          });
        }
      });

      if (labelled) {
        // A label sits right after its line's last point (at the right edge for lines that run to the end); labels that
        // would overlap (close in x and in y) are pushed apart.
        var ends = series.map(function (s) {
          var last = s.points[s.points.length - 1];
          return { s: s, x: X(last[0]) + 5, y: Y(last[1]) };
        });
        ends.sort(function (a, b) { return a.y - b.y; });
        for (var i = 1; i < ends.length; i++) {
          for (var k = 0; k < i; k++) {
            if (Math.abs(ends[i].x - ends[k].x) < 28 && ends[i].y < ends[k].y + 12) {
              ends[i].y = ends[k].y + 12;
            }
          }
        }
        ends.forEach(function (e) {
          var t = sv("text", { x: e.x, y: Math.min(e.y + 4, mt + ph + 4) }, "tr-end");
          t.textContent = e.s.short || "";
          svg.appendChild(t);
        });
      }

      // The crosshair and the dots on the series live in one group, rebuilt on every pointer move.
      svg.appendChild(sv("g", null, "tr-hover"));
    }

    function nearestPoint(points, x) {
      var best = points[0];
      var bd = Math.abs(best[0] - x);
      for (var i = 1; i < points.length; i++) {
        var d = Math.abs(points[i][0] - x);
        if (d < bd) {
          best = points[i];
          bd = d;
        }
      }
      return best;
    }

    function hover(event) {
      if (!scale) {
        return;
      }
      var rect = svg.getBoundingClientRect();
      var px = event.clientX - rect.left;
      var dataX = scale.xlo + ((px - scale.ml) / scale.pw) * (scale.xhi - scale.xlo);
      var snapped = nearestPoint(xs.map(function (v) { return [v]; }), dataX)[0];
      var group = svg.querySelector(".tr-hover");
      if (!group) {
        return;
      }
      clear(group);
      group.appendChild(sv("line", { x1: scale.X(snapped), x2: scale.X(snapped), y1: scale.mt, y2: scale.mt + scale.ph }, "tr-cross"));
      clear(tip);
      var head = el("div", "tr-tip-head", opts.xName + " " + fmtX(snapped) + (opts.phaseAt ? opts.phaseAt(snapped) : ""));
      tip.appendChild(head);
      series.forEach(function (s) {
        var p = nearestPoint(s.points, snapped);
        group.appendChild(sv("circle", { cx: scale.X(p[0]), cy: scale.Y(p[1]), r: 4 }, "tr-dot s" + s.slot));
        var row = el("div", "tr-tip-row");
        var name = el("span", "tr-tip-name");
        name.appendChild(keySwatch(s.slot, s.dash));
        name.appendChild(el("span", null, s.label + (p[0] !== snapped ? " (" + opts.xName + " " + fmtX(p[0]) + ")" : "")));
        row.appendChild(name);
        row.appendChild(el("span", "tr-tip-val", fmtY(p[1]) + (p[2] ? " " + p[2] : "")));
        tip.appendChild(row);
      });
      tip.hidden = false;
      var tw = tip.offsetWidth;
      var left = scale.X(snapped) + 12;
      if (left + tw > scale.width) {
        left = scale.X(snapped) - tw - 12;
      }
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = Math.max(0, scale.mt) + "px";
    }

    function leave() {
      tip.hidden = true;
      var group = svg.querySelector(".tr-hover");
      if (group) {
        clear(group);
      }
    }

    svg.addEventListener("pointermove", hover);
    svg.addEventListener("pointerdown", hover);
    svg.addEventListener("pointerleave", leave);

    // The same numbers as a table (built on demand): every value a tooltip shows is reachable without hovering.
    var details = el("details", "tr-tbl");
    details.appendChild(el("summary", null, "Таблица значений"));
    var tableHost = el("div", "tr-table-wrap");
    details.appendChild(tableHost);
    details.addEventListener("toggle", function () {
      if (!details.open || tableHost.firstChild) {
        return;
      }
      var table = el("table", "tr-table");
      var thead = el("thead");
      var hr = el("tr");
      hr.appendChild(el("th", null, opts.xName));
      series.forEach(function (s) {
        hr.appendChild(el("th", null, s.label));
      });
      thead.appendChild(hr);
      table.appendChild(thead);
      var tbody = el("tbody");
      var shownXs = xs.slice(-300);
      var maps = series.map(function (s) {
        var m = {};
        s.points.forEach(function (p) {
          m[p[0]] = p;
        });
        return m;
      });
      shownXs.forEach(function (x) {
        var tr = el("tr");
        tr.appendChild(el("td", null, fmtX(x)));
        maps.forEach(function (m) {
          var p = m[x];
          tr.appendChild(el("td", null, p ? fmtY(p[1]) + (p[2] ? " " + p[2] : "") : ""));
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      tableHost.appendChild(table);
      if (xs.length > shownXs.length) {
        tableHost.appendChild(el("p", "tr-empty", "Показаны последние " + shownXs.length + " из " + xs.length + " строк."));
      }
    });
    wrap.appendChild(details);

    return { node: wrap, redraw: draw };
  }

  // A confidence interval as a whisker on a 0..max scale (the interval of the headline metric at a glance).
  function ciWhisker(c, max) {
    var w = 72;
    var h = 14;
    var svg = sv("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h, role: "img", "aria-label": "Доверительный интервал: " + ciText(c) }, "tr-ci-svg");
    function X(v) {
      return 4 + (Math.min(Math.max(v / max, 0), 1) * (w - 8));
    }
    svg.appendChild(sv("line", { x1: 4, x2: w - 4, y1: h / 2, y2: h / 2 }, "ci-track"));
    svg.appendChild(sv("line", { x1: X(c.lo), x2: X(c.hi), y1: h / 2, y2: h / 2 }, "ci-whisker"));
    svg.appendChild(sv("circle", { cx: X(c.p), cy: h / 2, r: 3.5 }, "ci-dot"));
    return svg;
  }

  // ---------------------------------------------------------------------------------------
  // Slots: the same entity has the same colour in every chart
  // ---------------------------------------------------------------------------------------

  var HEADS = [
    ["total", "итого", 0, "итого"],
    ["dir", "направление", 1, "напр."],
    ["jump", "прыжок", 2, "прыж."],
    ["hook", "хук", 3, "хук"],
    ["fire", "огонь", 4, "огонь"],
    ["aim", "прицел", 5, "прицел"],
  ];
  var SET_SLOT = { "dagger-val": 1, "teacher-val": 2, "human-val": 3, "human-val-tagged": 4 };
  var ARENA_SLOT = { "clb-left": 1, pit: 2, platform: 3, "clb-right": 4, "chillblock5-ruler": 5 };

  function slotFor(table, name, index) {
    return table[name] || 5 + (index % 3);
  }

  function uniqueSorted(values) {
    var seen = {};
    values.forEach(function (v) {
      seen[v] = true;
    });
    return Object.keys(seen).sort();
  }

  function phaseLabel(phases) {
    return function (x) {
      var name = "";
      (phases || []).forEach(function (p) {
        if (x >= p.start_step && x <= p.end_step) {
          name = p.name;
        }
      });
      return name ? " · " + name : "";
    };
  }

  function phaseMarkers(phases) {
    var out = [];
    (phases || []).forEach(function (p) {
      var m = /^dagger-(\d+)$/.exec(p.name);
      if (m) {
        out.push({ x: p.start_step, label: "D" + m[1] });
      }
    });
    return out;
  }

  function stepFmt(x) {
    return String(Math.round(x));
  }

  // ---------------------------------------------------------------------------------------
  // Chart builders (a run's metrics -> series)
  // ---------------------------------------------------------------------------------------

  function lossChart(metrics) {
    var series = HEADS.map(function (h) {
      return {
        label: h[1],
        short: h[3],
        slot: h[2],
        dash: 0,
        width: h[0] === "total" ? 2.5 : 1.8,
        points: metrics.train
          .filter(function (p) { return isNum(p[h[0]]); })
          .map(function (p) { return [p.step, p[h[0]]]; }),
      };
    });
    return lineChart({
      title: "Потери обучения",
      aria: "Потери обучения по шагам: итого и по головам",
      series: series,
      markers: phaseMarkers(metrics.phases),
      xName: "шаг",
      xFmt: stepFmt,
      yFmt: function (v) { return num(v, 3); },
      phaseAt: phaseLabel(metrics.phases),
      empty: "Записей обучения (train) пока нет.",
    });
  }

  function evalSeries(metrics, field) {
    var sets = uniqueSorted(metrics.eval.map(function (e) { return e.set; }));
    sets.sort(function (a, b) { return (SET_SLOT[a] || 9) - (SET_SLOT[b] || 9) || (a < b ? -1 : 1); });
    var out = [];
    sets.forEach(function (name, i) {
      var points = metrics.eval
        .filter(function (e) { return e.set === name && isNum(e[field]); })
        .map(function (e) { return [e.step, e[field]]; });
      if (points.length) {
        out.push({ label: name, short: "", slot: slotFor(SET_SLOT, name, i), dash: 0, points: points });
      }
    });
    return out.slice(0, 7);
  }

  function evalChart(metrics, field, title, aria) {
    return lineChart({
      title: title,
      aria: aria,
      series: evalSeries(metrics, field),
      markers: phaseMarkers(metrics.phases),
      xName: "шаг",
      xFmt: stepFmt,
      yFmt: function (v) { return num(v, 3); },
      clamp: [0, 1],
      alwaysLegend: true,
      phaseAt: phaseLabel(metrics.phases),
      empty: "Оценок на валидации пока нет.",
    });
  }

  function arenaSeries(arena, names) {
    var arenas = names || uniqueSorted(arena.map(function (a) { return a.arena; }));
    return arenas
      .map(function (name, i) {
        var points = arena
          .filter(function (a) { return a.arena === name && a.credited && isNum(a.step); })
          .map(function (a) { return [a.step, a.credited.p, "[" + (a.credited.lo * 100).toFixed(1).replace(".", ",") + "–" + (a.credited.hi * 100).toFixed(1).replace(".", ",") + "]"]; });
        return { label: name, short: "", slot: slotFor(ARENA_SLOT, name, i), dash: 0, points: points };
      })
      .filter(function (s) { return s.points.length; });
  }

  function arenaChart(metrics) {
    return lineChart({
      title: "Арена: доля игр, выигранных собственным засчитанным блоком (D-059)",
      aria: "Доля побед с засчитанным блоком по аренам после каждой фазы",
      series: arenaSeries(metrics.arena),
      markers: phaseMarkers(metrics.phases),
      xName: "шаг",
      xFmt: stepFmt,
      yFmt: pct,
      yTick: function (v) { return (v * 100).toFixed(0) + "%"; },
      clamp: [0, 1],
      alwaysLegend: true,
      phaseAt: phaseLabel(metrics.phases),
      empty: "Оценок в арене пока нет (они идут в конце каждой фазы).",
    });
  }

  function hookPlayChart(metrics) {
    var rows = metrics.hook_play;
    function pts(field) {
      return rows.filter(function (r) { return isNum(r[field]); }).map(function (r) { return [r.round, r[field]]; });
    }
    return lineChart({
      title: "Хук в игре: доля старта и отпускания, ученик и учитель (по раундам DAgger)",
      aria: "Доля старта и отпускания хука у ученика и учителя по раундам",
      series: [
        { label: "старт, ученик", short: "", slot: 1, dash: 0, points: pts("start_student") },
        { label: "старт, учитель", short: "", slot: 1, dash: 1, points: pts("start_teacher") },
        { label: "отпускание, ученик", short: "", slot: 2, dash: 0, points: pts("release_student") },
        { label: "отпускание, учитель", short: "", slot: 2, dash: 1, points: pts("release_teacher") },
      ],
      xName: "раунд",
      xFmt: function (x) { return String(Math.round(x)); },
      yFmt: pct,
      yTick: function (v) { return (v * 100).toFixed(0) + "%"; },
      clamp: [0, 1],
      alwaysLegend: true,
      empty: "Раундов DAgger с игрой хука пока нет.",
    });
  }

  // ---------------------------------------------------------------------------------------
  // Tables
  // ---------------------------------------------------------------------------------------

  function table(headers, rows, cls) {
    var wrap = el("div", "tr-table-wrap");
    wrap.tabIndex = 0; // a wide table scrolls sideways: the keyboard must be able to reach it
    var t = el("table", "tr-table" + (cls ? " " + cls : ""));
    var thead = el("thead");
    var hr = el("tr");
    headers.forEach(function (h) {
      hr.appendChild(el("th", null, h));
    });
    thead.appendChild(hr);
    t.appendChild(thead);
    var tbody = el("tbody");
    rows.forEach(function (cells) {
      var tr = el("tr");
      cells.forEach(function (c) {
        if (c && typeof c === "object" && c.nodeType === 1) {
          var td = el("td");
          td.appendChild(c);
          tr.appendChild(td);
        } else {
          tr.appendChild(el("td", null, c === null || c === undefined ? "—" : String(c)));
        }
      });
      tbody.appendChild(tr);
    });
    t.appendChild(tbody);
    wrap.appendChild(t);
    return wrap;
  }

  function tally(p) {
    if (!isNum(p.w) && !isNum(p.l)) {
      return "—";
    }
    return [p.w, p.l, p.d, p.t].map(function (v) { return isNum(v) ? v : "–"; }).join(" : ");
  }

  function ciCell(c, max) {
    var inner = el("div", "ci-inner");
    inner.appendChild(el("span", null, ciText(c)));
    if (c) {
      var holder = el("span", "tr-ci");
      holder.appendChild(ciWhisker(c, max));
      inner.appendChild(holder);
    }
    return inner;
  }

  function ciMax(points) {
    var m = 0.1;
    points.forEach(function (p) {
      if (p.credited && p.credited.hi > m) {
        m = p.credited.hi;
      }
    });
    return Math.min(1, Math.ceil(m * 10) / 10);
  }

  // Arena results: one row per arena / condition, the headline metric first (D-059) with its whisker.
  function arenaTable(points, firstHeader) {
    var max = ciMax(points);
    var rows = points.map(function (p) {
      return [
        p.arena,
        isNum(p.games) ? p.games : "—",
        tally(p),
        ciCell(p.credited, max),
        ciText(p.win_rate),
        num(p.blocks_per_min, 2),
        num(p.self_freezes_per_min, 2),
      ];
    });
    var wrap = table(
      [firstHeader, "Игр", "W : L : D : T", "Засчитанных побед, 95% ДИ Уилсона (шкала 0–" + Math.round(max * 100) + "%)", "W/(W+L+D), ДИ", "Блоки/мин", "Самозаморозки/мин"],
      rows,
      "tr-arena",
    );
    return wrap;
  }

  function roundsTable(metrics) {
    var byRound = {};
    metrics.collect.forEach(function (c) {
      (byRound[c.round] = byRound[c.round] || {}).collect = c;
    });
    metrics.hook_play.forEach(function (h) {
      (byRound[h.round] = byRound[h.round] || {}).hook = h;
    });
    var rounds = Object.keys(byRound).map(Number).sort(function (a, b) { return a - b; });
    if (!rounds.length) {
      return el("p", "tr-empty", "Раундов DAgger пока нет.");
    }
    function pair(a, b) {
      return pct(a, 0) + " / " + pct(b, 0);
    }
    var rows = rounds.map(function (r) {
      var c = byRound[r].collect;
      var h = byRound[r].hook;
      return [
        "D" + r,
        c ? num(c.beta, 2) : "—",
        c ? c.jobs : "—",
        c ? c.games : "—",
        c ? [c.w, c.l, c.d, c.t].join(" : ") : "—",
        c ? c.steps : "—",
        h ? pair(h.start_student, h.start_teacher) : "—",
        h ? pair(h.release_student, h.release_teacher) : "—",
      ];
    });
    return table(["Раунд", "β (доля учителя)", "Заданий", "Игр", "W : L : D : T (сбор)", "Решений", "Старт хука: ученик / учитель", "Отпускание: ученик / учитель"], rows);
  }

  // ---------------------------------------------------------------------------------------
  // The panel
  // ---------------------------------------------------------------------------------------

  var TrainPanel = (function () {
    var shown = false;
    var timer = null;
    var listing = null;
    var listSig = "";
    var openKey = null;
    var detail = null;
    var picked = [];
    var cmpOpen = false;
    var cmpKeys = [];
    var cmpData = {};
    var cmpEntity = { set: "dagger-val", arena: "" };
    var openExps = {};
    var charts = [];
    var cmpCharts = [];
    var sessionOver = false;
    var resizeTimer = null;

    var listEl = byId("train-list");
    var detailEl = byId("train-detail");
    var cmpEl = byId("train-compare");
    var stateEl = byId("train-state");
    var dotEl = byId("train-dot");
    var cmpBtn = byId("train-compare-btn");
    var cmpClear = byId("train-compare-clear");
    var pickNote = byId("train-pick-note");

    function setState(text, ok) {
      stateEl.textContent = text;
      dotEl.classList.toggle("dot-on", !!ok);
      dotEl.classList.toggle("dot-off", !ok);
    }

    function endSession() {
      sessionOver = true;
      stopTimer();
      setState("Сессия закончилась: обновите страницу и войдите снова.", false);
    }

    function failure(res) {
      if (res.status === 401) {
        endSession();
      } else if (res.status === 503) {
        setState("Сайт занят чтением файлов, повтор через несколько секунд.", false);
      } else {
        setState("Не удалось прочитать данные (код " + res.status + ").", false);
      }
    }

    // ----- list --------------------------------------------------------------------------

    function runsCount(l) {
      var total = 0;
      var running = 0;
      l.experiments.forEach(function (e) {
        e.runs.forEach(function (r) {
          total++;
          if (r.state === "running") {
            running++;
          }
        });
      });
      return { total: total, running: running };
    }

    function signature(l) {
      return JSON.stringify(
        l.experiments.map(function (e) {
          return [
            e.id,
            e.runs.map(function (r) {
              return [r.id, r.state, r.phase, r.step, r.phase_step, r.kind, isNum(r.age_s) ? Math.floor(r.age_s / 60) : null];
            }),
          ];
        }),
      );
    }

    function loadList(poll) {
      if (sessionOver) {
        return Promise.resolve();
      }
      return api("/api/train/runs", poll)
        .then(function (res) {
          if (res.status !== 200 || !res.data) {
            failure(res);
            return;
          }
          listing = res.data;
          var c = runsCount(listing);
          if (!listing.root_present) {
            setState("Каталог запусков не найден на сервере.", false);
          } else {
            var t = new Date();
            setState(
              "Запусков: " + c.total + ", идёт: " + c.running + ". Обновлено " + t.toLocaleTimeString("ru-RU") +
                (c.running ? " (каждые " + (listing.poll_secs || 10) + " с)" : ""),
              true,
            );
          }
          var sig = signature(listing);
          if (sig !== listSig) {
            listSig = sig;
            renderList();
          }
        })
        .catch(function () {
          setState("Нет связи с сайтом.", false);
        });
    }

    function renderList() {
      clear(listEl);
      if (!listing || !listing.experiments.length) {
        listEl.appendChild(el("p", "tr-empty", listing && !listing.root_present ? "Каталог запусков не найден." : "Запусков пока нет."));
        updatePickUi();
        return;
      }
      listing.experiments.forEach(function (exp, idx) {
        var running = exp.runs.filter(function (r) { return r.state === "running"; }).length;
        var details = el("details", "tr-exp");
        var open = openExps[exp.id];
        details.open = open === undefined ? idx === 0 || running > 0 : open;
        var summary = el("summary");
        summary.appendChild(el("span", null, exp.id));
        summary.appendChild(el("small", null, exp.runs.length + " запусков" + (running ? ", идёт: " + running : "") + (exp.runs_truncated ? " (список обрезан)" : "")));
        details.appendChild(summary);
        details.addEventListener("toggle", function () {
          openExps[exp.id] = details.open;
        });
        exp.runs.forEach(function (r) {
          details.appendChild(runRow(exp.id, r));
        });
        listEl.appendChild(details);
      });
      if (listing.experiments_truncated) {
        listEl.appendChild(el("p", "tr-note", "Экспериментов больше, чем показано."));
      }
      updatePickUi();
    }

    function runRow(expId, r) {
      var key = keyOf(expId, r.id);
      var row = el("div", "tr-run" + (key === openKey ? " current" : ""));
      var pickLabel = el("label", "tr-pick");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = picked.indexOf(key) >= 0;
      box.setAttribute("aria-label", "Сравнить: " + r.id);
      box.addEventListener("change", function () {
        togglePick(key, box);
      });
      pickLabel.appendChild(box);
      row.appendChild(pickLabel);

      var name = el("button", "tr-name", r.id);
      name.type = "button";
      name.setAttribute("aria-label", r.id);
      name.addEventListener("click", function () {
        openRun(expId, r.id);
      });
      if (r.kind) {
        name.appendChild(el("span", "tr-badge", r.kind));
      }
      row.appendChild(name);

      var st = el("span", "tr-state " + r.state, STATE_TEXT[r.state] || r.state);
      st.title = STATE_HINT[r.state] || "";
      row.appendChild(st);

      var meta = el("span", "tr-meta");
      var bits = [];
      if (r.phase && r.phase !== "done") {
        bits.push(r.phase + (isNum(r.phase_step) && isNum(r.phase_steps) ? " (" + r.phase_step + "/" + r.phase_steps + ")" : ""));
      }
      if (isNum(r.step)) {
        bits.push("шаг " + r.step + (isNum(r.planned_steps) && r.state !== "done" ? " из " + r.planned_steps : ""));
      }
      bits.push(ago(r.age_s));
      meta.appendChild(el("span", null, bits.join(" · ")));
      if (isNum(r.step) && isNum(r.planned_steps) && r.state !== "done") {
        var prog = document.createElement("progress");
        prog.max = r.planned_steps;
        prog.value = Math.min(r.step, r.planned_steps);
        prog.setAttribute("aria-label", "Прогресс обучения");
        meta.appendChild(prog);
      }
      row.appendChild(meta);
      return row;
    }

    function togglePick(key, box) {
      var i = picked.indexOf(key);
      if (box.checked && i < 0) {
        if (picked.length >= MAX_PICK) {
          box.checked = false;
          pickNote.textContent = "Не больше " + MAX_PICK + " запусков.";
          return;
        }
        picked.push(key);
      } else if (!box.checked && i >= 0) {
        picked.splice(i, 1);
      }
      updatePickUi();
    }

    function updatePickUi() {
      cmpBtn.textContent = "Сравнить" + (picked.length ? " (" + picked.length + ")" : "");
      cmpBtn.disabled = picked.length < 2 || picked.length > MAX_PICK;
      cmpClear.hidden = picked.length === 0;
      pickNote.textContent = picked.length === 1 ? "Отметьте ещё хотя бы один запуск." : picked.length === 0 ? "" : "Выбрано: " + picked.length + " (можно от 2 до " + MAX_PICK + ").";
    }

    // ----- one run -----------------------------------------------------------------------

    function openRun(exp, run) {
      openKey = keyOf(exp, run);
      detail = null;
      renderList();
      detailEl.hidden = false;
      clear(detailEl);
      detailEl.appendChild(el("p", "card hint", "Загрузка запуска…"));
      loadRun(false).then(function () {
        detailEl.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }

    function loadRun(poll) {
      if (!openKey || sessionOver) {
        return Promise.resolve();
      }
      var k = splitKey(openKey);
      var asked = openKey;
      return api("/api/train/run?exp=" + encodeURIComponent(k.exp) + "&run=" + encodeURIComponent(k.run), poll)
        .then(function (res) {
          if (asked !== openKey) {
            return;
          }
          if (res.status === 200 && res.data) {
            detail = res.data;
            renderDetail();
          } else if (res.status === 404) {
            clear(detailEl);
            detailEl.appendChild(el("p", "card hint", "Запуск не найден (возможно, его удалили)."));
          } else {
            failure(res);
          }
        })
        .catch(function () {
          setState("Нет связи с сайтом.", false);
        });
    }

    function kv(rows) {
      var dl = el("dl", "kv");
      rows.forEach(function (r) {
        dl.appendChild(el("dt", null, r[0]));
        dl.appendChild(el("dd", null, r[1]));
      });
      return dl;
    }

    function statusBlock(d) {
      var s = d.summary;
      var st = d.status || {};
      var rows = [
        ["Состояние", STATE_TEXT[s.state] || s.state],
        ["Фаза", s.phase || "—"],
        ["Шаг", isNum(s.step) ? s.step + (isNum(s.planned_steps) ? " из " + s.planned_steps : "") : "—"],
      ];
      if (isNum(s.phase_step) && isNum(s.phase_steps)) {
        rows.push(["Шаг в фазе", s.phase_step + " / " + s.phase_steps]);
      }
      if (isNum(s.loss)) {
        rows.push(["Потеря (последняя)", num(s.loss, 3)]);
      }
      if (isNum(st.elapsed_s)) {
        rows.push(["Время фазы", Math.round(st.elapsed_s) + " с"]);
      }
      rows.push(["Файлы менялись", ago(s.age_s)]);
      var c = d.config;
      if (c) {
        rows.push(["Модель", (c.kind || "—") + (isNum(c.hidden) && c.hidden > 0 ? ", скрытых " + c.hidden : "")]);
        rows.push(["Сид обучения", isNum(c.seed) ? c.seed : "—"]);
        rows.push(["Собственный хук", c.own_hook_mode || "—"]);
        rows.push(["Доля демок людей", isNum(c.human_fraction) ? pct(c.human_fraction, 0) : "—"]);
        rows.push([
          "План",
          isNum(c.bc_steps) ? "BC " + c.bc_steps + (isNum(c.rounds) && isNum(c.steps_per_round) ? " + " + c.rounds + " раундов × " + c.steps_per_round : "") : "—",
        ]);
        rows.push(["Оценка в арене", (isNum(c.eval_games) ? c.eval_games + " игр: " : "") + (c.eval_arenas.length ? c.eval_arenas.join(", ") : "—")]);
      }
      if (d.config_files && d.config_files.length) {
        rows.push(["Файлы конфигурации", d.config_files.join(", ")]);
      }
      return kv(rows);
    }

    function card(title, parts, cls) {
      var c = el("section", "card" + (cls ? " " + cls : ""));
      if (title) {
        c.appendChild(el("h2", null, title));
      }
      parts.forEach(function (p) {
        if (p) {
          c.appendChild(p);
        }
      });
      return c;
    }

    function addChart(list, parent, chart) {
      list.push(chart);
      parent.appendChild(chart.node);
    }

    function renderDetail() {
      var d = detail;
      charts = [];
      clear(detailEl);
      var s = d.summary;

      // Header
      var head = el("section", "card");
      var top = el("div", "tr-head");
      top.appendChild(el("h2", null, d.exp + " / " + d.run));
      var close = el("button", "alt", "Закрыть");
      close.type = "button";
      close.addEventListener("click", function () {
        openKey = null;
        detail = null;
        detailEl.hidden = true;
        clear(detailEl);
        renderList();
      });
      top.appendChild(close);
      head.appendChild(top);
      if (s.state === "running") {
        head.appendChild(el("p", "tr-live", "Идёт обучение: страница обновляется каждые " + (d.poll_secs || 10) + " с."));
      }
      head.appendChild(statusBlock(d));
      var m = d.metrics;
      if (m && m.tail_truncated) {
        head.appendChild(el("p", "tr-note", "Файл метрик большой (" + bytesText(m.file_len) + "): прочитан только конец, ранние записи не показаны."));
      }
      if (m && m.train_thinned) {
        head.appendChild(el("p", "tr-note", "Кривая потерь прорежена до " + m.train.length + " точек."));
      }
      if (m && m.skipped_lines) {
        head.appendChild(el("p", "tr-note", "Нечитаемых строк в метриках пропущено: " + m.skipped_lines + " (последняя строка могла быть записана не до конца)."));
      }
      if (!d.metrics) {
        head.appendChild(el("p", "tr-note", "metrics.jsonl не прочитан."));
      }
      detailEl.appendChild(head);

      if (m) {
        var curves = el("section", "card");
        curves.appendChild(el("h2", null, "Кривые"));
        curves.appendChild(
          el("p", "hint", "По горизонтали — шаг обучения; пунктирные вертикали D1…D5 — начало раундов DAgger. Наведите курсор или коснитесь графика, чтобы увидеть значения; под каждым графиком есть таблица."),
        );
        addChart(charts, curves, lossChart(m));
        addChart(charts, curves, evalChart(m, "dir_acc", "Направление: точность на валидации", "Точность направления на валидации по шагам"));
        addChart(charts, curves, evalChart(m, "hook_auroc", "Хук: AUROC на валидации", "AUROC хука на валидации по шагам"));
        addChart(charts, curves, hookPlayChart(m));
        addChart(charts, curves, arenaChart(m));
        detailEl.appendChild(curves);

        // DAgger rounds
        var dag = [roundsTable(m)];
        if (m.thresholds.length) {
          var t = m.thresholds[m.thresholds.length - 1];
          dag.push(el("p", "hint", "Пороги решений (последние, фаза " + (t.phase || "—") + "): прыжок " + num(t.jump, 3) + " · хук " + num(t.hook, 3) + " · огонь " + num(t.fire, 3) + (t.sets.length ? " (по " + t.sets.join(", ") + ")" : "")));
        }
        if (m.selection && m.selection.table.length) {
          var best = -Infinity;
          m.selection.table.forEach(function (r) { best = Math.max(best, r[1]); });
          dag.push(el("h3", null, "Отбор по средней доле засчитанных побед на аренах " + (m.selection.arenas.join(", ") || "—")));
          dag.push(
            table(
              ["Фаза", "Доля"],
              m.selection.table.map(function (r) {
                return [r[0], pct(r[1]) + (r[1] === best ? "  ← максимум" : "")];
              }),
            ),
          );
        }
        detailEl.appendChild(card("Раунды DAgger", dag));

        // Arena
        var arenaParts = [];
        var latest = latestArena(m.arena);
        if (latest.length) {
          arenaParts.push(el("p", "hint", "Последняя оценка в конце фазы " + (latest[0].phase || "—") + ", шаг " + (isNum(latest[0].step) ? latest[0].step : "—") + ". Главная метрика — доля игр, выигранных собственным засчитанным блоком (D-059); таймауты считаются против игрока."));
          arenaParts.push(arenaTable(latest, "Арена"));
        } else {
          arenaParts.push(el("p", "tr-empty", "Оценок в арене в метриках пока нет."));
        }
        (d.eval_summaries || []).forEach(function (sum) {
          var det = el("details", "tr-sub");
          var sm = el("summary", null, "Сводка арены «" + sum.source + "»: " + sum.conditions.length + " условий" + (sum.git_commit ? ", коммит " + sum.git_commit + (sum.git_dirty ? " (с правками)" : "") : "") + (isNum(sum.base_seed) ? ", сид " + sum.base_seed : ""));
          det.appendChild(sm);
          det.appendChild(arenaTable(sum.conditions, "Условие"));
          if (sum.conditions_truncated) {
            det.appendChild(el("p", "tr-note", "Условий больше, чем показано."));
          }
          arenaParts.push(det);
        });
        detailEl.appendChild(card("Арена", arenaParts));
      }

      // Checkpoints
      var ck = d.checkpoints || [];
      var ckParts = [
        el("p", "hint", "Только имя, размер и первые 16 знаков sha256 файла; сами веса сайт не отдаёт."),
        ck.length
          ? table(
              ["Каталог", "Файл", "Размер", "sha256 (16)", "Изменён"],
              ck.map(function (c) {
                return [c.group, c.name, bytesText(c.bytes), c.sha256 || "—", ago(c.age_s)];
              }),
            )
          : el("p", "tr-empty", "Чекпоинтов нет."),
      ];
      detailEl.appendChild(card("Чекпоинты", ckParts));
      redrawCharts();
    }

    function latestArena(points) {
      var best = {};
      points.forEach(function (p) {
        var cur = best[p.arena];
        if (!cur || (p.step || 0) >= (cur.step || 0)) {
          best[p.arena] = p;
        }
      });
      return Object.keys(best).sort().map(function (k) { return best[k]; });
    }

    // ----- comparison --------------------------------------------------------------------

    // The comparison runs are fetched one after another (never a burst of parallel requests). A run that could not be read
    // keeps its previous data when there is some; its colour and number follow its position among the picked runs, so a
    // missing run never recolours the others.
    function loadCompare(poll) {
      if (!cmpOpen || sessionOver) {
        return Promise.resolve();
      }
      var keys = cmpKeys.slice();
      var next = {};
      keys.forEach(function (key) {
        if (cmpData[key]) {
          next[key] = cmpData[key];
        }
      });
      var lost = false;
      var chain = Promise.resolve();
      keys.forEach(function (key) {
        chain = chain.then(function () {
          if (lost) {
            return null;
          }
          var k = splitKey(key);
          return api("/api/train/run?exp=" + encodeURIComponent(k.exp) + "&run=" + encodeURIComponent(k.run), poll).then(function (res) {
            if (res.status === 401) {
              lost = true;
            } else if (res.status === 200 && res.data) {
              next[key] = res.data;
            } else if (res.status === 404) {
              delete next[key];
            }
          });
        });
      });
      return chain
        .then(function () {
          if (lost) {
            endSession();
            return;
          }
          cmpData = next;
          renderCompare(keys);
        })
        .catch(function () {
          setState("Нет связи с сайтом.", false);
        });
    }

    function runColour(i) {
      return { slot: i + 1, dash: i % DASH_CLASS.length };
    }

    function select(options, value, onChange, label) {
      var wrap = el("label");
      wrap.appendChild(el("span", "hint", label));
      var sel = document.createElement("select");
      options.forEach(function (o) {
        var opt = el("option", null, o);
        opt.value = o;
        sel.appendChild(opt);
      });
      sel.value = value;
      sel.addEventListener("change", function () {
        onChange(sel.value);
      });
      wrap.appendChild(sel);
      return wrap;
    }

    function renderCompare(keys) {
      cmpCharts = [];
      clear(cmpEl);
      cmpEl.hidden = false;
      var runs = keys.filter(function (k) { return cmpData[k]; });
      // Colour, dash and number follow the position among the picked runs, whatever loaded.
      var pos = {};
      keys.forEach(function (k, i) {
        pos[k] = i;
      });

      var head = el("section", "card");
      var top = el("div", "tr-head");
      top.appendChild(el("h2", null, "Сравнение запусков (" + runs.length + ")"));
      var close = el("button", "alt", "Закрыть");
      close.type = "button";
      close.addEventListener("click", function () {
        cmpOpen = false;
        cmpEl.hidden = true;
        clear(cmpEl);
      });
      top.appendChild(close);
      head.appendChild(top);
      var legend = el("div", "tr-legend");
      runs.forEach(function (k) {
        var c = runColour(pos[k]);
        var item = el("span", "tr-legend-item");
        item.appendChild(keySwatch(c.slot, c.dash));
        var d = cmpData[k];
        item.appendChild(el("span", null, "#" + (pos[k] + 1) + " " + k + " — " + (STATE_TEXT[d.summary.state] || d.summary.state)));
        legend.appendChild(item);
      });
      head.appendChild(legend);
      if (runs.length < keys.length) {
        var missing = keys.filter(function (k) { return !cmpData[k]; }).map(function (k) { return "#" + (pos[k] + 1) + " " + k; });
        head.appendChild(el("p", "tr-note", "Не прочитано и не показано: " + missing.join(", ") + "."));
      }
      if (runs.some(function (k) { return cmpData[k].summary.state === "running"; })) {
        head.appendChild(el("p", "tr-live", "Идёт обучение: сравнение обновляется каждые 10 с."));
      }
      cmpEl.appendChild(head);

      var sets = uniqueSorted(
        [].concat.apply([], runs.map(function (k) { return ((cmpData[k].metrics || {}).eval || []).map(function (e) { return e.set; }); })),
      );
      var arenas = uniqueSorted(
        [].concat.apply([], runs.map(function (k) { return ((cmpData[k].metrics || {}).arena || []).map(function (a) { return a.arena; }); })),
      );
      if (sets.length && sets.indexOf(cmpEntity.set) < 0) {
        cmpEntity.set = sets.indexOf("dagger-val") >= 0 ? "dagger-val" : sets[0];
      }
      if (arenas.length && arenas.indexOf(cmpEntity.arena) < 0) {
        cmpEntity.arena = arenas.indexOf("clb-left") >= 0 ? "clb-left" : arenas[0];
      }

      var chartCard = el("section", "card");
      chartCard.appendChild(el("h2", null, "Кривые вместе"));
      var controls = el("div", "tr-controls");
      if (sets.length) {
        controls.appendChild(select(sets, cmpEntity.set, function (v) { cmpEntity.set = v; renderCompare(keys); }, "Набор валидации"));
      }
      if (arenas.length) {
        controls.appendChild(select(arenas, cmpEntity.arena, function (v) { cmpEntity.arena = v; renderCompare(keys); }, "Арена"));
      }
      chartCard.appendChild(controls);
      chartCard.appendChild(el("p", "hint", "Цвет и штрих — запуск (как в легенде выше). Раунды DAgger у запусков могут начинаться на разных шагах, поэтому маркеров здесь нет."));

      function overlay(title, aria, pointsOf, yFmt, clamp, yTick) {
        var series = runs.map(function (k) {
          var c = runColour(pos[k]);
          return { label: "#" + (pos[k] + 1) + " " + cmpData[k].run, short: "#" + (pos[k] + 1), slot: c.slot, dash: c.dash, points: pointsOf(cmpData[k].metrics || { train: [], eval: [], arena: [] }) };
        });
        var chart = lineChart({ title: title, aria: aria, series: series, xName: "шаг", xFmt: stepFmt, yFmt: yFmt, clamp: clamp, yTick: yTick, alwaysLegend: false, empty: "Нет данных у выбранных запусков." });
        cmpCharts.push(chart);
        chartCard.appendChild(chart.node);
      }
      overlay(
        "Потеря обучения (итого)",
        "Итоговая потеря выбранных запусков по шагам",
        function (m) { return m.train.filter(function (p) { return isNum(p.total); }).map(function (p) { return [p.step, p.total]; }); },
        function (v) { return num(v, 3); },
      );
      overlay(
        "Направление: точность на валидации (" + cmpEntity.set + ")",
        "Точность направления выбранных запусков по шагам",
        function (m) { return m.eval.filter(function (e) { return e.set === cmpEntity.set && isNum(e.dir_acc); }).map(function (e) { return [e.step, e.dir_acc]; }); },
        function (v) { return num(v, 3); },
        [0, 1],
      );
      overlay(
        "Хук: AUROC на валидации (" + cmpEntity.set + ")",
        "AUROC хука выбранных запусков по шагам",
        function (m) { return m.eval.filter(function (e) { return e.set === cmpEntity.set && isNum(e.hook_auroc); }).map(function (e) { return [e.step, e.hook_auroc]; }); },
        function (v) { return num(v, 3); },
        [0, 1],
      );
      overlay(
        "Арена " + cmpEntity.arena + ": доля побед с засчитанным блоком",
        "Доля побед с засчитанным блоком выбранных запусков по шагам",
        function (m) {
          return m.arena
            .filter(function (a) { return a.arena === cmpEntity.arena && a.credited && isNum(a.step); })
            .map(function (a) { return [a.step, a.credited.p, "[" + (a.credited.lo * 100).toFixed(1).replace(".", ",") + "–" + (a.credited.hi * 100).toFixed(1).replace(".", ",") + "]"]; });
        },
        pct,
        [0, 1],
        function (v) { return (v * 100).toFixed(0) + "%"; },
      );
      cmpEl.appendChild(chartCard);

      // Side by side: the last numbers of every run.
      var header = ["Запуск", "Модель", "Шаг", "Хук", "Сид", "Потеря", "Напр. (" + cmpEntity.set + ")", "AUROC хука"].concat(
        arenas.map(function (a) { return "Засчитанные: " + a; }),
      );
      var maxes = {};
      arenas.forEach(function (a) {
        var pts = [];
        runs.forEach(function (k) {
          var l = latestArena((cmpData[k].metrics || { arena: [] }).arena).filter(function (p) { return p.arena === a; });
          pts = pts.concat(l);
        });
        maxes[a] = ciMax(pts);
      });
      var rows = runs.map(function (k) {
        var d = cmpData[k];
        var m = d.metrics || { train: [], eval: [], arena: [] };
        var cfg = d.config || {};
        var lastEval = null;
        m.eval.forEach(function (e) {
          if (e.set === cmpEntity.set) {
            lastEval = e;
          }
        });
        var lastTrain = m.train.length ? m.train[m.train.length - 1] : null;
        var latest = {};
        latestArena(m.arena).forEach(function (p) { latest[p.arena] = p; });
        var row = [
          "#" + (pos[k] + 1) + " " + d.run,
          cfg.kind || "—",
          isNum(d.summary.step) ? d.summary.step : "—",
          cfg.own_hook_mode || "—",
          isNum(cfg.seed) ? cfg.seed : "—",
          lastTrain ? num(lastTrain.total, 3) : "—",
          lastEval ? num(lastEval.dir_acc, 3) : "—",
          lastEval ? num(lastEval.hook_auroc, 3) : "—",
        ];
        arenas.forEach(function (a) {
          row.push(latest[a] ? ciCell(latest[a].credited, maxes[a]) : "—");
        });
        return row;
      });
      var tableCard = el("section", "card");
      tableCard.appendChild(el("h2", null, "Последние значения"));
      tableCard.appendChild(el("p", "hint", "Потеря — последняя точка обучения; метрики валидации — последняя оценка выбранного набора; в аренах — доля игр с засчитанным блоком, 95% ДИ Уилсона, последняя оценка. Шкалы линеек у каждой арены общие для всех запусков."));
      tableCard.appendChild(table(header, rows));
      cmpEl.appendChild(tableCard);
      redrawCharts();
    }

    // ----- lifecycle ----------------------------------------------------------------------

    function redrawCharts() {
      charts.concat(cmpCharts).forEach(function (c) {
        c.redraw();
      });
    }

    function listedState(key) {
      var k = splitKey(key);
      var state = null;
      if (listing) {
        listing.experiments.forEach(function (e) {
          if (e.id === k.exp) {
            e.runs.forEach(function (r) {
              if (r.id === k.run) {
                state = r.state;
              }
            });
          }
        });
      }
      return state;
    }

    // One refresh at a time, in order: list, then the open run, then the comparison runs (each of those one by one), so the
    // page never asks the server for more than one thing at once. A refresh that is still running makes the next tick skip.
    var refreshing = false;

    function refreshAll(poll, force) {
      if (refreshing || sessionOver) {
        return Promise.resolve();
      }
      refreshing = true;
      return loadList(poll)
        .then(function () {
          if (!openKey) {
            return null;
          }
          var stale = !detail || detail.summary.state === "running" || listedState(openKey) !== detail.summary.state;
          return force || stale ? loadRun(poll) : null;
        })
        .then(function () {
          if (!cmpOpen) {
            return null;
          }
          var anyRunning = cmpKeys.some(function (k) {
            var d = cmpData[k];
            return !d || d.summary.state === "running" || listedState(k) === "running";
          });
          return force || anyRunning ? loadCompare(poll) : null;
        })
        .then(
          function () {
            refreshing = false;
          },
          function () {
            refreshing = false;
          },
        );
    }

    function tick() {
      if (!shown || sessionOver || document.hidden) {
        return;
      }
      refreshAll(true, false);
    }

    function stopTimer() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    function onShown() {
      shown = true;
      sessionOver = false;
      refreshAll(false, true);
      if (!timer) {
        timer = setInterval(tick, POLL_MS);
      }
      redrawCharts();
    }

    function onHidden() {
      shown = false;
      stopTimer();
    }

    cmpBtn.addEventListener("click", function () {
      if (picked.length < 2 || picked.length > MAX_PICK) {
        return;
      }
      cmpOpen = true;
      cmpKeys = picked.slice();
      cmpEl.hidden = false;
      clear(cmpEl);
      cmpEl.appendChild(el("p", "card hint", "Загрузка запусков…"));
      loadCompare(false).then(function () {
        cmpEl.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
    cmpClear.addEventListener("click", function () {
      picked = [];
      renderList();
    });
    window.addEventListener("resize", function () {
      if (resizeTimer) {
        clearTimeout(resizeTimer);
      }
      resizeTimer = setTimeout(function () {
        if (shown) {
          redrawCharts();
        }
      }, 120);
    });
    document.addEventListener("visibilitychange", function () {
      if (shown && !document.hidden) {
        tick();
      }
    });

    return { onShown: onShown, onHidden: onHidden };
  })();

  window.TrainPanel = TrainPanel;
})();
