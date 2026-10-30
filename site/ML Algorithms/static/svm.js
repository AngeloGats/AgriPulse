(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const VIEW = { w: 720, h: 460, left: 56, right: 20, top: 16, bottom: 48 };
  const PLOT = { w: VIEW.w - VIEW.left - VIEW.right, h: VIEW.h - VIEW.top - VIEW.bottom };
  const CLASS_COLORS = ["#34734f", "#d98c2b"]; // healthy, diseased
  const BANDS = {
    healthyStrong: [159, 203, 169],
    healthyLight: [226, 240, 229],
    diseasedLight: [251, 233, 210],
    diseasedStrong: [240, 185, 122],
  };

  const config = JSON.parse(document.getElementById("svm-config").textContent);
  const featureMeta = Object.fromEntries(config.features.map((f) => [f.key, f]));
  const $ = (id) => document.getElementById(id);

  const state = {
    kernel: "rbf",
    sample: null, // {x, y} in data units
    data: null,
    sequence: 0,
    timer: null,
    exampleIndex: 0,
    dragging: false,
  };

  function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const pct = (v) => `${(v * 100).toFixed(0)}%`;
  const fmt = (v) => String(Number(v.toPrecision(2)));
  const cValue = () => 10 ** Number($("svm-c").value);
  const gammaValue = () => 10 ** Number($("svm-gamma").value);

  function setStatus(message, isError) {
    const node = $("svm-status");
    node.textContent = message;
    node.classList.toggle("is-error", Boolean(isError));
  }

  function labelOf(key) {
    const f = featureMeta[key];
    return f.unit ? `${f.label} (${f.unit})` : f.label;
  }

  function pair() {
    return [$("svm-x").value, $("svm-y").value];
  }

  // ---- Request -----------------------------------------------------------

  async function refresh() {
    const [x, y] = pair();
    const body = { x, y, kernel: state.kernel, c: Number(cValue().toFixed(4)) };
    if (state.kernel === "rbf") body.gamma = Number(gammaValue().toFixed(4));
    if (state.sample) body.sample = { [x]: state.sample.x, [y]: state.sample.y };
    const ticket = ++state.sequence;
    try {
      const response = await fetch("/api/svm/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "The request failed");
      if (ticket !== state.sequence) return;
      state.data = data;
      setStatus("");
      render();
    } catch (error) {
      if (ticket === state.sequence) setStatus(error.message, true);
    }
  }

  function schedule(delay = 120) {
    clearTimeout(state.timer);
    state.timer = setTimeout(refresh, delay);
  }

  // ---- Drawing -------------------------------------------------------------

  function paintRegions(data) {
    const canvas = $("svm-canvas");
    const { width, height } = canvas;
    const ctx = canvas.getContext("2d");
    const G = data.grid_size;
    const grid = data.grid;
    const at = (gx, gy) => grid[gy * G + gx];
    const scores = new Float32Array(width * height);
    for (let j = 0; j < height; j += 1) {
      const fy = ((height - 1 - j) / (height - 1)) * (G - 1);
      const y0 = Math.min(Math.floor(fy), G - 2);
      const ty = fy - y0;
      for (let i = 0; i < width; i += 1) {
        const fx = (i / (width - 1)) * (G - 1);
        const x0 = Math.min(Math.floor(fx), G - 2);
        const tx = fx - x0;
        scores[j * width + i] =
          at(x0, y0) * (1 - tx) * (1 - ty) + at(x0 + 1, y0) * tx * (1 - ty) +
          at(x0, y0 + 1) * (1 - tx) * ty + at(x0 + 1, y0 + 1) * tx * ty;
      }
    }
    const image = ctx.createImageData(width, height);
    for (let j = 0; j < height; j += 1) {
      for (let i = 0; i < width; i += 1) {
        const s = scores[j * width + i];
        const edge =
          (i + 1 < width && s > 0 !== scores[j * width + i + 1] > 0) ||
          (j + 1 < height && s > 0 !== scores[(j + 1) * width + i] > 0);
        const color = edge
          ? [33, 54, 44]
          : s <= -1 ? BANDS.healthyStrong : s <= 0 ? BANDS.healthyLight : s < 1 ? BANDS.diseasedLight : BANDS.diseasedStrong;
        const o = (j * width + i) * 4;
        image.data[o] = color[0];
        image.data[o + 1] = color[1];
        image.data[o + 2] = color[2];
        image.data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }

  function toPixel(data) {
    const [[x0, x1], [y0, y1]] = data.bounds;
    return {
      x: (v) => VIEW.left + ((v - x0) / (x1 - x0)) * PLOT.w,
      y: (v) => VIEW.h - VIEW.bottom - ((v - y0) / (y1 - y0)) * PLOT.h,
    };
  }

  function drawChart(data) {
    const root = $("svm-chart");
    root.replaceChildren();
    const map = toPixel(data);
    const [[x0, x1], [y0, y1]] = data.bounds;

    for (let i = 0; i <= 4; i += 1) {
      const vx = x0 + ((x1 - x0) * i) / 4;
      const vy = y0 + ((y1 - y0) * i) / 4;
      root.append(
        svg("text", { class: "svm-tick", x: map.x(vx), y: VIEW.h - VIEW.bottom + 16, "text-anchor": "middle" }, fmt(vx)),
        svg("text", { class: "svm-tick", x: VIEW.left - 8, y: map.y(vy) + 4, "text-anchor": "end" }, fmt(vy)),
      );
    }
    root.append(
      svg("text", { class: "knn-axis-title", x: VIEW.left + PLOT.w / 2, y: VIEW.h - 10, "text-anchor": "middle" }, labelOf(data.x)),
      svg("text", { class: "knn-axis-title", x: 14, y: VIEW.top + PLOT.h / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${VIEW.top + PLOT.h / 2})` }, labelOf(data.y)),
    );

    const dots = svg("g");
    const showSv = $("svm-show-sv").checked;
    data.points.forEach(([x, y, label, isSv]) => {
      const cx = map.x(x).toFixed(1);
      const cy = map.y(y).toFixed(1);
      dots.append(svg("circle", { class: "svm-dot", cx, cy, r: 3.4, fill: CLASS_COLORS[label] }));
      if (showSv && isSv) dots.append(svg("circle", { class: "svm-sv", cx, cy, r: 6 }));
    });
    root.append(dots);

    if (state.sample) {
      const sx = map.x(state.sample.x);
      const sy = map.y(state.sample.y);
      root.append(
        svg("path", { d: `M${sx} ${sy - 11} L${sx + 9} ${sy} L${sx} ${sy + 11} L${sx - 9} ${sy} Z`, fill: "#21362c", stroke: "#fff", "stroke-width": 2 }),
        svg("text", { class: "knn-sample-label", x: sx + 13, y: sy - 10 }, "Your sample"),
      );
    }
  }

  function renderResult(data) {
    const sample = data.sample;
    const gauge = $("svm-gauge");
    gauge.hidden = !sample;
    $("svm-warning").hidden = true;
    if (!sample) {
      $("svm-top").textContent = "—";
      $("svm-context").textContent = "No sample yet";
      $("svm-intro").textContent = "Place a sample on the chart or enter its readings.";
      return;
    }
    $("svm-top").textContent = sample.prediction;
    $("svm-intro").textContent = "The model checked which side of the boundary your sample is on.";
    $("svm-context").textContent = sample.in_margin ? "Inside the margin: a close call" : "Clear of the margin";
    const clamped = Math.max(-2, Math.min(2, sample.score));
    $("svm-gauge-dot").style.left = `${((clamped + 2) / 4) * 100}%`;
    const distance = Math.abs(sample.score).toFixed(2);
    $("svm-gauge-note").textContent = sample.in_margin
      ? `Score ${sample.score.toFixed(2)}. It sits within one unit of the boundary, so a small change in readings could flip the answer.`
      : `Score ${sample.score.toFixed(2)}. It is ${distance} units from the boundary, beyond the margin, so the model is confident.`;
    if (sample.outside_training_range) {
      const warning = $("svm-warning");
      warning.hidden = false;
      warning.textContent = "This sample is outside the range of readings the model learned from, so the answer is less reliable.";
    }
  }

  function renderStats(data) {
    $("svm-stat-sv").textContent = `${data.support_vectors} of ${data.train_rows} (${pct(data.support_vectors / data.train_rows)})`;
    $("svm-stat-acc").textContent = pct(data.test_accuracy);
    const rec = config.recommended[`${data.x}+${data.y}`] || config.recommended[`${data.y}+${data.x}`];
    $("svm-stat-linear").textContent = rec ? pct(rec.linear.test_accuracy) : "—";
    $("svm-stat-rbf").textContent = rec ? pct(rec.rbf.test_accuracy) : "—";
    $("svm-stat-note").textContent =
      `Always guessing "healthy" would score ${pct(data.baseline)}. ` +
      `${data.inside_margin} training samples sit inside the margin, and ${data.support_vectors} support vectors hold the boundary in place.`;
  }

  function render() {
    const data = state.data;
    paintRegions(data);
    drawChart(data);
    renderResult(data);
    renderStats(data);
  }

  // ---- Controls ----------------------------------------------------------

  function updateSliderLabels() {
    $("svm-c-value").textContent = fmt(cValue());
    $("svm-gamma-value").textContent = fmt(gammaValue());
    $("svm-gamma-row").hidden = state.kernel !== "rbf";
  }

  function updateSampleInputs() {
    const [x, y] = pair();
    $("svm-sx-label").textContent = `Sample: ${labelOf(x)}`;
    $("svm-sy-label").textContent = `Sample: ${labelOf(y)}`;
    $("svm-sx").value = state.sample ? state.sample.x : "";
    $("svm-sy").value = state.sample ? state.sample.y : "";
  }

  function setSample(x, y) {
    const [kx, ky] = pair();
    const fx = featureMeta[kx];
    const fy = featureMeta[ky];
    state.sample = {
      x: Math.min(Math.max(Number(x.toFixed(2)), fx.min), fx.max),
      y: Math.min(Math.max(Number(y.toFixed(2)), fy.min), fy.max),
    };
    $("svm-sx").value = state.sample.x;
    $("svm-sy").value = state.sample.y;
  }

  function onAxisChange(changed) {
    const x = $("svm-x");
    const y = $("svm-y");
    if (x.value === y.value) {
      const other = changed === x ? y : x;
      const spare = config.features.find((f) => f.key !== changed.value);
      other.value = spare.key;
    }
    state.sample = null;
    updateSampleInputs();
    refresh();
  }

  function onSampleInput() {
    const x = $("svm-sx").value.trim();
    const y = $("svm-sy").value.trim();
    if (x === "" || y === "") return;
    if (!Number.isFinite(Number(x)) || !Number.isFinite(Number(y))) return;
    setSample(Number(x), Number(y));
    schedule(250);
  }

  function pointerToData(event) {
    if (!state.data) return null;
    const rect = $("svm-chart").getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * VIEW.w;
    const py = ((event.clientY - rect.top) / rect.height) * VIEW.h;
    const fx = Math.min(Math.max((px - VIEW.left) / PLOT.w, 0), 1);
    const fy = Math.min(Math.max((VIEW.h - VIEW.bottom - py) / PLOT.h, 0), 1);
    const [[x0, x1], [y0, y1]] = state.data.bounds;
    return [x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)];
  }

  function onPointer(event) {
    const point = pointerToData(event);
    if (!point) return;
    setSample(point[0], point[1]);
    schedule(40);
  }

  async function loadExample() {
    state.exampleIndex += 1;
    try {
      const response = await fetch(`/api/svm/example?n=${state.exampleIndex}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The request failed");
      const [x, y] = pair();
      setSample(data.inputs[x], data.inputs[y]);
      setStatus(`Example loaded. Its real condition is ${data.actual}.`);
      refresh();
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  function useRecommended() {
    const [x, y] = pair();
    const rec = config.recommended[`${x}+${y}`] || config.recommended[`${y}+${x}`];
    if (!rec) return;
    const best = rec[state.kernel];
    const clamp = (v, [lo, hi]) => Math.min(Math.max(v, lo), hi);
    $("svm-c").value = Math.log10(clamp(best.c, config.c_range)).toFixed(2);
    if (state.kernel === "rbf") $("svm-gamma").value = Math.log10(clamp(best.gamma, config.gamma_range)).toFixed(2);
    updateSliderLabels();
    refresh();
  }

  function setKernel(kernel) {
    state.kernel = kernel;
    document.querySelectorAll(".svm-seg").forEach((button) => {
      const active = button.dataset.kernel === kernel;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    updateSliderLabels();
    refresh();
  }

  $("svm-x").addEventListener("change", (e) => onAxisChange(e.target));
  $("svm-y").addEventListener("change", (e) => onAxisChange(e.target));
  document.querySelectorAll(".svm-seg").forEach((button) => button.addEventListener("click", () => setKernel(button.dataset.kernel)));
  ["svm-c", "svm-gamma"].forEach((id) => $(id).addEventListener("input", () => { updateSliderLabels(); schedule(); }));
  $("svm-show-sv").addEventListener("change", () => state.data && drawChart(state.data));
  $("svm-example").addEventListener("click", loadExample);
  $("svm-recommended").addEventListener("click", useRecommended);
  $("svm-sx").addEventListener("input", onSampleInput);
  $("svm-sy").addEventListener("input", onSampleInput);

  const wrap = $("svm-chart-wrap");
  wrap.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    wrap.setPointerCapture(event.pointerId);
    onPointer(event);
  });
  wrap.addEventListener("pointermove", (event) => state.dragging && onPointer(event));
  ["pointerup", "pointercancel"].forEach((name) => wrap.addEventListener(name, () => { state.dragging = false; }));

  $("svm-x").value = config.default_x;
  $("svm-y").value = config.default_y;
  updateSliderLabels();
  updateSampleInputs();
  refresh();
})();
