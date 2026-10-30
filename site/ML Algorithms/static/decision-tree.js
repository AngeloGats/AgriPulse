(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = { irrigate_now: "#3b7ea1", irrigate_soon: "#d98c2b", wait: "#34734f", hold: "#7a8a80" };
  const config = JSON.parse(document.getElementById("dt-config").textContent);
  const $ = (id) => document.getElementById(id);

  const state = { tree: null, result: null, requestId: 0, exampleIndex: 0 };

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

  const percent = (value) => `${(value * 100).toFixed(0)}%`;
  const depth = () => Number($("dt-depth-slider").value);

  function setStatus(message, isError) {
    const node = $("dt-status");
    node.textContent = message;
    node.classList.toggle("is-error", Boolean(isError));
  }

  async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The request failed");
    return data;
  }

  function readInputs() {
    const inputs = {};
    for (const field of $("dt-form").querySelectorAll("input, select")) {
      if (field.value.trim() === "") {
        const label = field.closest("label").firstElementChild.textContent.trim();
        throw new Error(`${label} is required`);
      }
      inputs[field.name] = Number(field.value);
    }
    return inputs;
  }

  function fillInputs(inputs) {
    for (const [key, value] of Object.entries(inputs)) {
      const field = $(`dt-${key}`);
      if (field) field.value = value;
    }
  }

  // Tree layout: x by level, y by leaf order so branches never cross.
  function layout(tree) {
    const nodes = new Map(tree.nodes.map((n) => [n.id, n]));
    const pos = new Map();
    const slot = 62;
    let leafIndex = 0;
    let maxLevel = 0;
    const place = (id, level) => {
      const node = nodes.get(id);
      maxLevel = Math.max(maxLevel, level);
      let y;
      if (node.left === null) {
        y = leafIndex++ * slot + 40;
      } else {
        const a = place(node.left, level + 1);
        const b = place(node.right, level + 1);
        y = (a + b) / 2;
      }
      pos.set(id, { x: level, y });
      return y;
    };
    place(0, 0);
    return { nodes, pos, maxLevel, height: leafIndex * slot + 20 };
  }

  function drawTree() {
    const tree = state.tree;
    const chart = $("dt-chart");
    const { nodes, pos, maxLevel, height } = layout(tree);
    const colW = 250;
    const boxW = { split: 200, leaf: 150 };
    const left = 12;
    const width = left + maxLevel * colW + boxW.leaf + 20;
    const onPath = new Set(state.result && state.result.depth === tree.depth ? state.result.path : []);
    const X = (level) => left + level * colW;

    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    chart.style.minWidth = `${Math.max(860, width * 0.8)}px`;
    chart.classList.toggle("has-path", onPath.size > 0);
    chart.replaceChildren();

    const edges = svg("g");
    const boxes = svg("g");
    tree.nodes.forEach((node) => {
      if (node.left === null) return;
      const from = pos.get(node.id);
      [["left", "Yes"], ["right", "No"]].forEach(([side, label]) => {
        const childId = node[side];
        const to = pos.get(childId);
        const x1 = X(from.x) + boxW.split;
        const x2 = X(to.x);
        const mid = (x1 + x2) / 2;
        const active = onPath.has(node.id) && onPath.has(childId);
        const cls = active ? " is-path" : "";
        edges.append(
          svg("path", { class: `dt-edge${cls}`, d: `M${x1},${from.y} C${mid},${from.y} ${mid},${to.y} ${x2},${to.y}` }),
          svg("text", { class: `dt-edge-label${cls}`, x: x1 + 8, y: to.y + (side === "left" ? -6 : 14) }, label)
        );
      });
    });

    tree.nodes.forEach((node) => {
      const p = pos.get(node.id);
      const group = svg("g", { class: `dt-node${onPath.has(node.id) ? " is-path" : ""}` });
      if (node.left === null) {
        const w = boxW.leaf;
        group.append(
          svg("rect", { class: "dt-leaf-box", x: X(p.x), y: p.y - 24, width: w, height: 48, rx: 10, fill: COLORS[node.prediction] || "#666" }),
          svg("text", { class: "dt-leaf-text", x: X(p.x) + w / 2, y: p.y - 3, "text-anchor": "middle" }, node.action),
          svg("text", { class: "dt-leaf-text dt-leaf-sub", x: X(p.x) + w / 2, y: p.y + 13, "text-anchor": "middle" },
            `${node.samples} records · ${percent(node.purity)} agree`)
        );
      } else {
        const w = boxW.split;
        group.append(
          svg("rect", { class: "dt-split", x: X(p.x), y: p.y - 20, width: w, height: 40, rx: 10 }),
          svg("text", { class: "dt-split-text", x: X(p.x) + w / 2, y: p.y + 4, "text-anchor": "middle" }, node.short)
        );
        group.append(svg("title", {}, node.question));
      }
      boxes.append(group);
    });
    chart.append(edges, boxes);
  }

  function drawLegend() {
    const list = $("dt-legend");
    list.replaceChildren();
    state.tree.classes.forEach((c) => {
      const item = element("li");
      const swatch = element("i", "knn-swatch");
      swatch.style.background = COLORS[c.key];
      item.append(swatch, document.createTextNode(c.name));
      list.append(item);
    });
    const t = state.tree;
    $("dt-tree-note").textContent =
      `Depth ${t.depth}: ${t.leaves} final recommendations. Accuracy on unseen readings: ${percent(t.test_accuracy)}. ` +
      (t.depth === t.recommended ? "This is the recommended depth." : `The recommended depth is ${t.recommended}.`);
  }

  function drawImportance() {
    const bars = $("dt-importance");
    bars.replaceChildren();
    state.tree.importance.forEach((item, index) => {
      const row = element("li", `nb-bar-row${index === 0 ? " is-top" : ""}`);
      row.append(element("b", "", item.label), element("span", "", item.value > 0 ? percent(item.value) : "not used"));
      const track = element("span", "nb-bar-track");
      const fill = element("span", "nb-bar-fill");
      fill.style.width = `${Math.max(item.value * 100, item.value > 0 ? 1.5 : 0)}%`;
      track.append(fill);
      row.append(track);
      bars.append(row);
    });
  }

  function drawAccuracy() {
    const chart = $("dt-acc-chart");
    const rows = state.tree.accuracy_by_depth;
    const w = 420, h = 240;
    const pad = { left: 44, right: 14, top: 14, bottom: 36 };
    const lo = Math.max(0, Math.floor((Math.min(...rows.map((r) => Math.min(r.test_accuracy, r.cv_accuracy))) - 0.05) * 10) / 10);
    const x = (d) => pad.left + ((d - rows[0].depth) / (rows.length - 1 || 1)) * (w - pad.left - pad.right);
    const y = (v) => h - pad.bottom - ((v - lo) / (1 - lo)) * (h - pad.top - pad.bottom);
    chart.replaceChildren();
    for (let v = Math.ceil(lo * 10) / 10; v <= 1.0001; v += 0.1) {
      chart.append(
        svg("line", { class: "knn-grid", x1: pad.left, x2: w - pad.right, y1: y(v), y2: y(v) }),
        svg("text", { class: "knn-tick", x: pad.left - 6, y: y(v) + 4, "text-anchor": "end" }, percent(v))
      );
    }
    rows.forEach((r) => chart.append(svg("text", { class: "knn-tick", x: x(r.depth), y: h - pad.bottom + 16, "text-anchor": "middle" }, r.depth)));
    chart.append(svg("text", { class: "knn-tick", x: (pad.left + w - pad.right) / 2, y: h - 4, "text-anchor": "middle" }, "Tree depth"));
    [["cv_accuracy", "#3b7ea1"], ["test_accuracy", "#34734f"]].forEach(([key, color]) => {
      chart.append(svg("polyline", { fill: "none", stroke: color, "stroke-width": 2.4, points: rows.map((r) => `${x(r.depth)},${y(r[key])}`).join(" ") }));
      rows.forEach((r) => chart.append(svg("circle", { cx: x(r.depth), cy: y(r[key]), r: r.depth === state.tree.depth ? 6 : 3.5, fill: color })));
    });
    const legend = $("dt-acc-legend");
    legend.replaceChildren();
    [["Cross-validation (training data)", "#3b7ea1"], ["Held-out test", "#34734f"]].forEach(([label, color]) => {
      const item = element("li");
      const swatch = element("i", "knn-swatch");
      swatch.style.background = color;
      item.append(swatch, document.createTextNode(label));
      legend.append(item);
    });
  }

  function showResult() {
    const r = state.result;
    if (!r) return;
    $("dt-intro").textContent = "Follow the highlighted path in the flowchart.";
    const top = $("dt-top");
    top.textContent = r.action;
    top.style.color = COLORS[r.prediction];
    $("dt-context").textContent = `Using a tree ${r.depth} questions deep` + (r.depth === r.recommended_depth ? " (recommended)" : "");
    const advice = $("dt-advice");
    advice.hidden = false;
    advice.textContent = r.advice;

    const list = $("dt-rule-list");
    list.replaceChildren();
    r.steps.forEach((s) => {
      const li = element("li");
      li.append(element("b", "", `${s.answer}: `), document.createTextNode(s.rule));
      list.append(li);
    });
    if (!r.steps.length) list.append(element("li", "", "No questions needed at this depth."));
    $("dt-rules").hidden = false;

    const leaf = $("dt-leaf");
    leaf.hidden = false;
    leaf.textContent =
      `${r.leaf_samples} similar past records ended here, and ${percent(r.leaf_purity)} of them needed "${r.action}".`;

    const warn = $("dt-warning");
    warn.hidden = !r.warnings.length;
    warn.textContent = r.warnings.join(" ");

    const note = $("dt-depth-note");
    const differing = r.by_depth.filter((d) => d.prediction !== r.prediction).map((d) => `depth ${d.depth}: ${d.action}`);
    note.hidden = false;
    note.className = differing.length ? "knn-k-note is-changed" : "knn-k-note";
    note.textContent = differing.length
      ? `The answer changes with tree depth (${differing.join(", ")}). Shallow trees are rougher.`
      : "Every tree depth gives this same answer.";
  }

  async function loadTree() {
    const id = ++state.requestId;
    try {
      const tree = await request(`/api/dt/tree?depth=${depth()}`);
      if (id !== state.requestId) return;
      state.tree = tree;
      drawTree();
      drawLegend();
      drawImportance();
      drawAccuracy();
    } catch (error) {
      if (id === state.requestId) setStatus(error.message, true);
    }
  }

  async function predict() {
    const id = ++state.requestId;
    try {
      const inputs = readInputs();
      setStatus("Working out the recommendation…");
      const result = await request("/api/dt/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs, depth: depth() }),
      });
      if (id !== state.requestId) return;
      state.result = result;
      if (!state.tree || state.tree.depth !== result.depth) {
        state.tree = await request(`/api/dt/tree?depth=${result.depth}`);
        drawImportance();
        drawAccuracy();
        drawLegend();
      }
      drawTree();
      showResult();
      setStatus("");
    } catch (error) {
      if (id === state.requestId) setStatus(error.message, true);
    }
  }

  function bind() {
    $("dt-form").addEventListener("submit", (event) => {
      event.preventDefault();
      predict();
    });
    $("dt-example").addEventListener("click", async () => {
      try {
        const data = await request(`/api/dt/example?n=${state.exampleIndex++}`);
        fillInputs(data.inputs);
        await predict();
        setStatus(`Example from the test set. The simulated field actually needed: ${data.actual}.`);
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    $("dt-reset").addEventListener("click", () => {
      for (const field of $("dt-form").querySelectorAll("input, select")) field.value = field.dataset.default;
      setStatus("");
    });
    $("dt-depth-slider").addEventListener("input", () => {
      $("dt-depth-value").textContent = depth();
      if (state.result) predict();
      else loadTree();
    });
  }

  bind();
  $("dt-depth-slider").value = config.recommended;
  loadTree();
})();
