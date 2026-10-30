(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const MAX_BYTES = 5 * 1024 * 1024;
  const $ = (id) => document.getElementById(id);
  const state = { requestId: 0, randomIndex: 0, objectUrl: null };

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
    const node = $("ann-status");
    node.textContent = message;
    node.classList.toggle("is-error", Boolean(isError));
  }

  async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The request failed");
    return data;
  }

  function drawHeat(heat) {
    const layer = $("ann-heat");
    layer.replaceChildren();
    heat.forEach((row, r) =>
      row.forEach((value, c) =>
        layer.append(svg("rect", { x: c, y: r, width: 1, height: 1, fill: "#d93c3c", "fill-opacity": (value * 0.65).toFixed(2) }))
      )
    );
    layer.classList.toggle("is-off", !$("ann-heat-toggle").checked);
  }

  function showResult(result, imageUrl) {
    $("ann-intro").textContent = "The network has scored this leaf.";
    $("ann-leaf").hidden = false;
    $("ann-leaf-img").src = imageUrl;
    drawHeat(result.heatmap);

    const top = $("ann-top");
    top.textContent = result.name;
    $("ann-context").textContent =
      `${percent(result.confidence)} confident` + (result.uncertain ? " (a weak guess)" : "") +
      (result.actual_name ? (result.correct ? " · correct, it really is " : " · actually: ") + result.actual_name : "");

    const bars = $("ann-bars");
    bars.replaceChildren();
    result.scores.forEach((score, rank) => {
      const row = element("li", `nb-bar-row${rank === 0 ? " is-top" : ""}`);
      row.append(element("b", "", score.name), element("span", "", percent(score.probability)));
      const track = element("span", "nb-bar-track");
      const fill = element("span", "nb-bar-fill");
      fill.style.width = `${Math.max(score.probability * 100, score.probability > 0 ? 1.5 : 0)}%`;
      track.append(fill);
      row.append(track);
      bars.append(row);
    });
    $("ann-advice").textContent = result.advice;

    const warn = $("ann-warning");
    warn.hidden = !result.uncertain;
    warn.textContent = result.uncertain
      ? "The scores are spread out, so the network is unsure. The photo may not look like the leaves it trained on."
      : "";

    $("ann-net-view").src = result.network_view;
    $("ann-see").hidden = false;
  }

  function markActive(path) {
    document.querySelectorAll(".ann-thumb").forEach((b) => b.classList.toggle("is-active", b.dataset.sample === path));
  }

  async function classifySample(path) {
    const id = ++state.requestId;
    setStatus("Classifying…");
    try {
      const result = await request("/api/ann/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: path }),
      });
      if (id !== state.requestId) return;
      markActive(path);
      showResult(result, `/api/ann/sample-image/${path}`);
      setStatus("");
    } catch (error) {
      if (id === state.requestId) setStatus(error.message, true);
    }
  }

  async function classifyUpload(file) {
    if (!/^image\/(jpeg|png)$/.test(file.type)) return setStatus("Upload a JPEG or PNG photo of a leaf.", true);
    if (file.size > MAX_BYTES) return setStatus("That image is larger than 5 MB. Upload a smaller photo.", true);
    const id = ++state.requestId;
    setStatus("Classifying…");
    const body = new FormData();
    body.append("image", file);
    try {
      const result = await request("/api/ann/predict", { method: "POST", body });
      if (id !== state.requestId) return;
      if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = URL.createObjectURL(file);
      markActive(null);
      showResult(result, state.objectUrl);
      setStatus("");
    } catch (error) {
      if (id === state.requestId) setStatus(error.message, true);
    }
  }

  function lineChart(id, series, yMax, yFormat, zeroBased) {
    const chart = $(id);
    const w = 420, h = 240;
    const pad = { left: 44, right: 14, top: 14, bottom: 36 };
    const epochs = series[0].values.length;
    const x = (i) => pad.left + (i / (epochs - 1)) * (w - pad.left - pad.right);
    const lo = zeroBased ? 0 : Math.floor(Math.min(...series.flatMap((s) => s.values)) * 10) / 10;
    const y = (v) => h - pad.bottom - ((v - lo) / (yMax - lo)) * (h - pad.top - pad.bottom);
    chart.replaceChildren();
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((yMax - lo) * i) / 4;
      chart.append(
        svg("line", { class: "knn-grid", x1: pad.left, x2: w - pad.right, y1: y(v), y2: y(v) }),
        svg("text", { class: "knn-tick", x: pad.left - 6, y: y(v) + 4, "text-anchor": "end" }, yFormat(v))
      );
    }
    [1, Math.round(epochs / 2), epochs].forEach((e) =>
      chart.append(svg("text", { class: "knn-tick", x: x(e - 1), y: h - pad.bottom + 16, "text-anchor": "middle" }, e))
    );
    chart.append(svg("text", { class: "knn-tick", x: (pad.left + w - pad.right) / 2, y: h - 4, "text-anchor": "middle" }, "Epoch"));
    series.forEach((s) =>
      chart.append(svg("polyline", { fill: "none", stroke: s.color, "stroke-width": 2.4, points: s.values.map((v, i) => `${x(i)},${y(Math.min(v, yMax))}`).join(" ") }))
    );
  }

  function drawTraining(data) {
    const h = data.history;
    lineChart("ann-loss-chart", [{ values: h.map((r) => r.loss), color: "#b0506b" }], Math.ceil(Math.max(...h.map((r) => r.loss)) * 10) / 10, (v) => v.toFixed(1), true);
    lineChart("ann-acc-chart", [
      { values: h.map((r) => r.train_accuracy), color: "#3b7ea1" },
      { values: h.map((r) => r.val_accuracy), color: "#34734f" },
    ], 1, percent, false);
    const legend = $("ann-acc-legend");
    legend.replaceChildren();
    [["Training leaves", "#3b7ea1"], ["Validation leaves (held back)", "#34734f"]].forEach(([label, color]) => {
      const item = element("li");
      const swatch = element("i", "knn-swatch");
      swatch.style.background = color;
      item.append(swatch, document.createTextNode(label));
      legend.append(item);
    });

    const table = $("ann-confusion");
    table.replaceChildren();
    const head = element("tr");
    head.append(element("th", "", "Actual ↓ / Predicted →"));
    data.classes.forEach((c) => head.append(element("th", "", data.names[c])));
    table.append(head);
    data.confusion_matrix.forEach((row, i) => {
      const tr = element("tr");
      tr.append(element("th", "", data.names[data.classes[i]]));
      row.forEach((count, j) => tr.append(element("td", i === j ? "is-diagonal" : count ? "is-miss" : "", String(count))));
      table.append(tr);
    });
  }

  function bind() {
    $("ann-gallery").addEventListener("click", (event) => {
      const button = event.target.closest(".ann-thumb");
      if (button) classifySample(button.dataset.sample);
    });
    $("ann-random").addEventListener("click", async () => {
      try {
        const data = await request(`/api/ann/random-sample?n=${Math.floor(Math.random() * 900) + state.randomIndex++}`);
        classifySample(data.path);
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    const drop = $("ann-drop");
    $("ann-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) classifyUpload(file);
      event.target.value = "";
    });
    ["dragenter", "dragover"].forEach((name) =>
      drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("is-over"); })
    );
    ["dragleave", "drop"].forEach((name) =>
      drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("is-over"); })
    );
    drop.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (file) classifyUpload(file);
    });
    $("ann-heat-toggle").addEventListener("change", (event) =>
      $("ann-heat").classList.toggle("is-off", !event.target.checked)
    );
  }

  async function start() {
    bind();
    try {
      drawTraining(await request("/api/ann/training"));
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  start();
})();
