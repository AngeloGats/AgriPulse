(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = ["#34734f", "#d98c2b", "#3b7ea1", "#9b5fb0"];
  const RESULT_TITLES = {
    soil_type: "Predicted soil type",
    soil_fertility: "Predicted fertility",
    plant_condition: "Predicted crop condition",
  };
  const NOTES = {
    uniform: "Each neighbor counts as one vote.",
    distance: "Closer neighbors count for more in the vote.",
  };

  const $ = (id) => document.getElementById(id);
  const maxK = Number(document.body.dataset.maxK) || 25;

  const state = {
    dataset: null,
    chart: {}, // dataset -> chart payload
    result: null,
    k: 1,
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const percent = (value) => `${(value * 100).toFixed(value > 0 && value < 0.1 ? 1 : 0)}%`;

  function setStatus(message, isError) {
    const node = $("knn-status");
    node.textContent = message;
    node.classList.toggle("is-error", Boolean(isError));
  }

  async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The request failed");
    return data;
  }

  const form = () => $(`form-${state.dataset}`);

  function readInputs() {
    const inputs = {};
    for (const field of form().querySelectorAll("input")) {
      if (field.value.trim() === "") {
        const label = field.closest("label").firstElementChild.textContent.trim();
        throw new Error(`${label} is required`);
      }
      inputs[field.name] = Number(field.value);
    }
    return inputs;
  }

  // ---- Charts ------------------------------------------------------------

  let scale = null;

  function drawScatter() {
    const chart = state.chart[state.dataset];
    const root = $("knn-chart");
    root.replaceChildren();
    const width = 720;
    const height = 460;
    const pad = { left: 56, right: 20, top: 16, bottom: 48 };

    const xs = chart.points.map((p) => p[0]);
    const ys = chart.points.map((p) => p[1]);
    if (state.result) {
      xs.push(state.result.sample_point[0]);
      ys.push(state.result.sample_point[1]);
    }
    const spanX = Math.max(...xs) - Math.min(...xs) || 1;
    const spanY = Math.max(...ys) - Math.min(...ys) || 1;
    const minX = Math.min(...xs) - spanX * 0.04;
    const maxX = Math.max(...xs) + spanX * 0.04;
    const minY = Math.min(...ys) - spanY * 0.04;
    const maxY = Math.max(...ys) + spanY * 0.04;
    scale = {
      x: (v) => pad.left + ((v - minX) / (maxX - minX)) * (width - pad.left - pad.right),
      y: (v) => height - pad.bottom - ((v - minY) / (maxY - minY)) * (height - pad.top - pad.bottom),
    };

    const grid = svg("g");
    for (let i = 0; i <= 4; i += 1) {
      const gx = pad.left + (i / 4) * (width - pad.left - pad.right);
      const gy = pad.top + (i / 4) * (height - pad.top - pad.bottom);
      grid.append(
        svg("line", { class: "knn-grid", x1: gx, x2: gx, y1: pad.top, y2: height - pad.bottom }),
        svg("line", { class: "knn-grid", x1: pad.left, x2: width - pad.right, y1: gy, y2: gy }),
      );
    }
    grid.append(
      svg("line", { class: "knn-axis", x1: pad.left, x2: width - pad.right, y1: height - pad.bottom, y2: height - pad.bottom }),
      svg("line", { class: "knn-axis", x1: pad.left, x2: pad.left, y1: pad.top, y2: height - pad.bottom }),
    );
    const [v1, v2] = chart.explained_variance;
    const xTitle = svg("text", { class: "knn-axis-title", x: (pad.left + width - pad.right) / 2, y: height - 12, "text-anchor": "middle" }, `Summary axis 1 (${percent(v1)} of the variation)`);
    const yTitle = svg("text", { class: "knn-axis-title", x: 16, y: (pad.top + height - pad.bottom) / 2, "text-anchor": "middle", transform: `rotate(-90 16 ${(pad.top + height - pad.bottom) / 2})` }, `Summary axis 2 (${percent(v2)})`);
    root.append(grid, xTitle, yTitle);

    const dots = svg("g", { id: "knn-dots" });
    chart.points.forEach(([x, y, c]) => {
      dots.append(svg("circle", { class: "knn-dot", cx: scale.x(x).toFixed(1), cy: scale.y(y).toFixed(1), r: 3.2, fill: COLORS[c % COLORS.length] }));
    });
    root.append(dots, svg("g", { id: "knn-dyn" }));
    drawNeighbors();
  }

  function drawNeighbors() {
    const layer = $("knn-dyn");
    if (!layer || !state.result) return;
    layer.replaceChildren();
    const result = state.result;
    const [sx, sy] = [scale.x(result.sample_point[0]), scale.y(result.sample_point[1])];
    result.neighbors.slice(0, state.k).forEach((n) => {
      const nx = scale.x(n.x);
      const ny = scale.y(n.y);
      layer.append(
        svg("line", { class: "knn-link", x1: sx, y1: sy, x2: nx, y2: ny }),
        svg("circle", { class: "knn-ring", cx: nx, cy: ny, r: 7, stroke: COLORS[n.class_index % COLORS.length] }),
      );
    });
    layer.append(
      svg("path", { class: "knn-sample", d: `M${sx} ${sy - 11} L${sx + 9} ${sy} L${sx} ${sy + 11} L${sx - 9} ${sy} Z` }),
      svg("text", { class: "knn-sample-label", x: sx + 13, y: sy - 10 }, "Your sample"),
    );
  }

  function drawLegend() {
    const chart = state.chart[state.dataset];
    const list = $("knn-legend");
    list.replaceChildren();
    chart.classes.forEach((c, index) => {
      const item = element("li");
      const swatch = element("i", "knn-swatch");
      swatch.style.background = COLORS[index % COLORS.length];
      item.append(swatch, document.createTextNode(c.name));
      list.append(item);
    });
    const sample = element("li");
    const diamond = element("i", "knn-swatch is-diamond");
    diamond.style.background = "#21362c";
    sample.append(diamond, document.createTextNode("Your sample"));
    list.append(sample);
    $("knn-chart-note").textContent =
      `Map view of ${chart.points.length} known samples. The model compares every reading, ` +
      "so dots that look close here may be further apart in reality.";
  }

  function drawAccuracy() {
    const chart = state.chart[state.dataset];
    const root = $("knn-acc-chart");
    root.replaceChildren();
    const width = 720;
    const height = 300;
    const pad = { left: 56, right: 20, top: 16, bottom: 44 };
    const rows = chart.accuracy_by_k;
    const values = rows.flatMap((r) => [r.test_accuracy, r.cv_accuracy]);
    const lo = Math.max(0, Math.floor((Math.min(...values) - 0.01) * 50) / 50);
    const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.01) * 50) / 50);
    const x = (k) => pad.left + ((k - 1) / (chart.max_k - 1)) * (width - pad.left - pad.right);
    const y = (v) => height - pad.bottom - ((v - lo) / (hi - lo || 1)) * (height - pad.top - pad.bottom);

    for (let i = 0; i <= 4; i += 1) {
      const value = lo + ((hi - lo) * i) / 4;
      root.append(
        svg("line", { class: "knn-grid", x1: pad.left, x2: width - pad.right, y1: y(value), y2: y(value) }),
        svg("text", { class: "knn-tick", x: pad.left - 8, y: y(value) + 4, "text-anchor": "end" }, `${(value * 100).toFixed(0)}%`),
      );
    }
    for (let k = 1; k <= chart.max_k; k += k === 1 ? 4 : 5) {
      root.append(svg("text", { class: "knn-tick", x: x(k), y: height - pad.bottom + 16, "text-anchor": "middle" }, String(k)));
    }
    root.append(
      svg("line", { class: "knn-axis", x1: pad.left, x2: width - pad.right, y1: height - pad.bottom, y2: height - pad.bottom }),
      svg("text", { class: "knn-axis-title", x: (pad.left + width - pad.right) / 2, y: height - 6, "text-anchor": "middle" }, "K (neighbors voting)"),
    );
    const path = (key) => rows.map((r, i) => `${i ? "L" : "M"}${x(r.k).toFixed(1)} ${y(r[key]).toFixed(1)}`).join(" ");
    root.append(
      svg("path", { class: "knn-line-cv", d: path("cv_accuracy") }),
      svg("path", { class: "knn-line-test", d: path("test_accuracy") }),
    );
    const best = rows[chart.best_k - 1];
    root.append(svg("circle", { class: "knn-best-dot", cx: x(best.k), cy: y(best.cv_accuracy), r: 6 }));
    root.append(svg("line", { class: "knn-k-marker", id: "knn-acc-marker", x1: x(state.k), x2: x(state.k), y1: pad.top, y2: height - pad.bottom }));
    root._x = x;

    const legend = $("knn-acc-legend");
    legend.replaceChildren();
    [["Held-out test accuracy", "#34734f", false], ["Cross-validation accuracy (used to pick K)", "#e0a15a", false], [`Chosen K = ${chart.best_k}`, "#fff", true]].forEach(([text, color, ring]) => {
      const item = element("li");
      const swatch = element("i", ring ? "knn-swatch" : "knn-swatch is-line");
      swatch.style.background = color;
      if (ring) swatch.style.border = "2.5px solid #34734f";
      item.append(swatch, document.createTextNode(text));
      legend.append(item);
    });
    $("knn-acc-intro").textContent =
      `Very small K can follow noise; very large K blurs real differences. K = ${chart.best_k} gave the ` +
      `best cross-validation accuracy for this dataset. ${NOTES[chart.weights]}`;
  }

  function moveAccuracyMarker() {
    const marker = $("knn-acc-marker");
    const root = $("knn-acc-chart");
    if (!marker || !root._x) return;
    const px = root._x(state.k);
    marker.setAttribute("x1", px);
    marker.setAttribute("x2", px);
  }

  // ---- Results -----------------------------------------------------------

  function renderResult() {
    const result = state.result;
    if (!result) return;
    const entry = result.by_k[state.k - 1];
    $("knn-top").textContent = entry.prediction;
    $("knn-intro").textContent = "The neighbors have voted.";
    $("knn-context").textContent =
      `${state.k} nearest ${state.k === 1 ? "neighbor" : "neighbors"} voting` +
      (state.k === result.recommended_k ? " (the best K found in testing)" : "");

    const bars = $("knn-bars");
    bars.replaceChildren();
    const sorted = [...entry.votes].sort((a, b) => b.share - a.share);
    sorted.forEach((vote, index) => {
      const row = element("li", `nb-bar-row${index === 0 ? " is-top" : ""}`);
      const detail = `${vote.count} of ${state.k} · ${percent(vote.share)}`;
      row.append(element("b", "", vote.name), element("span", "", detail));
      const track = element("span", "nb-bar-track");
      const fill = element("span", "nb-bar-fill");
      fill.style.width = `${Math.max(vote.share * 100, vote.share > 0 ? 1.5 : 0)}%`;
      track.append(fill);
      row.append(track);
      bars.append(row);
    });

    const note = $("knn-k-note");
    const reference = result.recommended_prediction;
    note.hidden = false;
    if (entry.prediction === reference) {
      note.className = "knn-k-note";
      note.textContent = state.k === result.recommended_k
        ? "This is the recommended K. Try other values to see whether the answer holds."
        : `Same answer as the recommended K (${result.recommended_k}).`;
    } else {
      note.className = "knn-k-note is-changed";
      note.textContent =
        `The answer changed. At the recommended K (${result.recommended_k}) it is ${reference}. ` +
        "Your sample is close to a border between groups.";
    }

    const warning = $("knn-warning");
    warning.hidden = result.warnings.length === 0;
    warning.textContent = result.warnings.join(" ");

    drawNeighbors();
    moveAccuracyMarker();
    renderTable();
  }

  function renderTable() {
    const result = state.result;
    const features = Object.keys(result.neighbors[0].values);
    const labels = [...form().querySelectorAll(".ml-field > span")].map((s) => s.firstChild.textContent.trim());
    const head = $("knn-table-head");
    head.replaceChildren();
    const headRow = element("tr");
    ["#", "Group", "Distance", ...labels].forEach((text) => headRow.append(element("th", "", text)));
    head.append(headRow);
    const body = $("knn-table-body");
    body.replaceChildren();
    result.neighbors.forEach((n, index) => {
      const row = element("tr", index < state.k ? "is-in" : "is-out");
      [String(n.rank), n.label, n.distance.toFixed(2), ...features.map((f) => String(n.values[f]))].forEach((text) => row.append(element("td", "", text)));
      body.append(row);
    });
    $("knn-neighbors-card").hidden = false;
  }

  async function submit() {
    let inputs;
    try {
      inputs = readInputs();
    } catch (error) {
      setStatus(error.message, true);
      return;
    }
    const button = $("knn-submit");
    button.disabled = true;
    setStatus("Finding neighbors…");
    const requested = state.dataset;
    try {
      const data = await request(`/api/knn/${requested}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs, k: state.k }),
      });
      if (requested !== state.dataset) return;
      state.result = data;
      setStatus("");
      drawScatter();
      renderResult();
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  // ---- Dataset switching ---------------------------------------------------

  async function activate(name) {
    state.dataset = name;
    state.result = null;
    document.querySelectorAll(".knn-tabs .nb-tab").forEach((tab) => {
      const active = tab.dataset.dataset === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      if (active) {
        $("score-accuracy").textContent = `${(Number(tab.dataset.accuracy) * 100).toFixed(0)}%`;
        $("score-title").textContent = `${tab.dataset.title} accuracy`;
        $("score-baseline").textContent = `vs ${(Number(tab.dataset.baseline) * 100).toFixed(0)}% for always guessing the biggest group`;
        $("score-rows").textContent = `Measured on ${tab.dataset.testRows} unseen samples.`;
        state.k = Number(tab.dataset.bestK);
      }
    });
    document.querySelectorAll(".knn-form").forEach((f) => { f.hidden = f.dataset.dataset !== name; });
    $("form-title").textContent = form().dataset.question;
    $("knn-result-title").textContent = RESULT_TITLES[name];
    $("knn-k-slider").value = state.k;
    $("knn-k-value").textContent = state.k;
    $("knn-top").textContent = "—";
    $("knn-intro").textContent = "Enter a sample, then press Find neighbors.";
    $("knn-context").textContent = "No prediction yet";
    $("knn-bars").replaceChildren();
    $("knn-k-note").hidden = true;
    $("knn-warning").hidden = true;
    $("knn-neighbors-card").hidden = true;
    setStatus("");

    try {
      if (!state.chart[name]) state.chart[name] = await request(`/api/knn/${name}/chart`);
    } catch (error) {
      setStatus(error.message, true);
      return;
    }
    if (state.dataset !== name) return;
    drawScatter();
    drawLegend();
    drawAccuracy();
  }

  async function loadExample() {
    const name = state.dataset;
    state.exampleIndex = (state.exampleIndex || 0) + 1;
    try {
      const data = await request(`/api/knn/${name}/example?n=${state.exampleIndex}`);
      if (name !== state.dataset) return;
      for (const field of form().querySelectorAll("input")) field.value = data.inputs[field.name];
      setStatus(`Example loaded. Its real group is ${data.actual}. Press Find neighbors to see what the model says.`);
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  function resetValues() {
    for (const field of form().querySelectorAll("input")) field.value = field.dataset.default;
    setStatus("");
  }

  document.querySelectorAll(".knn-tabs .nb-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      if (tab.dataset.dataset !== state.dataset) activate(tab.dataset.dataset);
    });
  });
  $("knn-submit").addEventListener("click", submit);
  document.querySelectorAll(".knn-form").forEach((f) => {
    f.addEventListener("submit", (event) => event.preventDefault());
    f.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        submit();
      }
    });
  });
  $("knn-example").addEventListener("click", loadExample);
  $("knn-reset").addEventListener("click", resetValues);
  $("knn-k-slider").addEventListener("input", (event) => {
    state.k = Math.min(Math.max(Number(event.target.value), 1), maxK);
    $("knn-k-value").textContent = state.k;
    renderResult();
    moveAccuracyMarker();
  });

  activate(document.querySelector(".knn-tabs .nb-tab.is-active").dataset.dataset);
})();
