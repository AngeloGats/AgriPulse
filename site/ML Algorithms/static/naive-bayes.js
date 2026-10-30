(() => {
  const MAX_BARS = 6;
  const MAX_EVIDENCE = 8;

  const $ = (id) => document.getElementById(id);

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function percent(value) {
    const pct = value * 100;
    if (pct > 0 && pct < 0.1) return "<0.1%";
    return `${pct.toFixed(pct >= 99.95 || pct < 10 ? 1 : 0)}%`;
  }

  function times(logRatio) {
    const ratio = Math.exp(Math.abs(logRatio));
    if (ratio >= 1000) return ">1000×";
    return `${ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(1)}×`;
  }

  function setStatus(node, message, isError) {
    node.textContent = message;
    node.classList.toggle("is-error", Boolean(isError));
  }

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The request failed");
    return data;
  }

  async function getJson(url) {
    const response = await fetch(url);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The request failed");
    return data;
  }

  function renderBars(list, predictions) {
    list.replaceChildren();
    predictions.slice(0, MAX_BARS).forEach((item, index) => {
      const row = element("li", `nb-bar-row${index === 0 ? " is-top" : ""}`);
      row.append(element("b", "", item.name), element("span", "", percent(item.probability)));
      const track = element("span", "nb-bar-track");
      const fill = element("span", "nb-bar-fill");
      fill.style.width = `${Math.max(item.probability * 100, 0.5)}%`;
      track.append(fill);
      row.append(track);
      list.append(row);
    });
  }

  function renderEvidence(prefix, result) {
    const section = $(`${prefix}-evidence`);
    const intro = $(`${prefix}-evidence-intro`);
    const list = $(`${prefix}-evidence-list`);
    list.replaceChildren();

    const priorNote =
      Math.abs(result.prior_log_ratio) < 0.05
        ? `Before any clues, ${result.top} and ${result.runner_up} were equally likely.`
        : result.prior_log_ratio > 0
          ? `Before any clues, ${result.top} was already ${times(result.prior_log_ratio)} as common as ${result.runner_up}.`
          : `Before any clues, ${result.runner_up} was ${times(result.prior_log_ratio)} as common as ${result.top}.`;
    intro.textContent =
      `${priorNote} Each clue below then shifts the odds between the top answer ` +
      `(${result.top}) and the runner-up (${result.runner_up}). Green favors ${result.top}; ` +
      `orange favors ${result.runner_up}.`;

    const items = result.evidence.slice(0, MAX_EVIDENCE);
    const largest = Math.max(...items.map((item) => Math.abs(item.log_ratio)), 1);
    items.forEach((item) => {
      const favorsTop = item.log_ratio >= 0;
      const row = element("li", "nb-evidence-item");
      const label = element("div");
      label.append(element("b", "", item.feature), element("small", "", item.value));

      const diverge = element("div", "nb-diverge");
      diverge.setAttribute("aria-hidden", "true");
      const bar = element("i", favorsTop ? "for" : "against");
      bar.style.width = `${Math.min(Math.abs(item.log_ratio) / largest, 1) * 50}%`;
      diverge.append(bar);

      const effect = element(
        "div",
        `nb-evidence-effect ${favorsTop ? "for" : "against"}`,
        `${times(item.log_ratio)} for ${favorsTop ? result.top : result.runner_up}`,
      );
      row.append(label, diverge, effect);
      list.append(row);
    });
    section.hidden = false;
  }

  function setupTabs() {
    const tabs = [
      { tab: $("tab-soybean"), panel: $("panel-soybean") },
      { tab: $("tab-crop"), panel: $("panel-crop") },
    ];
    tabs.forEach(({ tab }, index) => {
      tab.addEventListener("click", () => select(index));
      tab.addEventListener("keydown", (event) => {
        if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
          const next = (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
          select(next);
          tabs[next].tab.focus();
        }
      });
    });

    function select(active) {
      tabs.forEach(({ tab, panel }, index) => {
        const on = index === active;
        tab.classList.toggle("is-active", on);
        tab.setAttribute("aria-selected", String(on));
        tab.tabIndex = on ? 0 : -1;
        panel.hidden = !on;
      });
    }
    select(0);
  }

  function setupSoybean() {
    const form = $("soybean-form");
    const status = $("soybean-status");
    const selects = Array.from(form.querySelectorAll("select"));
    let exampleIndex = 0;

    function updateCount() {
      const chosen = selects.filter((select) => select.value !== "");
      selects.forEach((select) => select.classList.toggle("is-set", select.value !== ""));
      $("soybean-count").textContent = String(chosen.length);
    }

    function clearResult() {
      $("soybean-top").textContent = "—";
      $("soybean-context").textContent = "No diagnosis yet";
      $("soybean-intro").textContent = "Select symptoms, then press Diagnose.";
      $("soybean-bars").replaceChildren();
      $("soybean-evidence").hidden = true;
    }

    form.addEventListener("change", updateCount);

    $("soybean-clear").addEventListener("click", () => {
      selects.forEach((select) => (select.value = ""));
      updateCount();
      clearResult();
      setStatus(status, "");
    });

    $("soybean-example").addEventListener("click", async () => {
      setStatus(status, "Loading a held-out record…");
      try {
        const example = await getJson(`/api/nb/soybean/example?n=${exampleIndex++}`);
        selects.forEach((select) => {
          const code = example.symptoms[select.name];
          select.value = code === undefined ? "" : String(code);
        });
        updateCount();
        setStatus(status, `Loaded a real record. Its recorded disease is ${example.actual}. Press Diagnose to see what the model thinks.`);
      } catch (error) {
        setStatus(status, error.message, true);
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const symptoms = {};
      selects.forEach((select) => {
        symptoms[select.name] = select.value === "" ? null : Number(select.value);
      });
      const button = $("soybean-submit");
      button.disabled = true;
      setStatus(status, "Calculating…");
      try {
        const result = await postJson("/api/nb/soybean/predict", { symptoms });
        const top = result.predictions[0];
        $("soybean-top").textContent = top.name;
        $("soybean-context").textContent =
          `${percent(top.probability)} probability, from ${result.symptoms_used} of ${result.symptoms_total} clues`;
        $("soybean-intro").textContent =
          result.symptoms_used < 4
            ? "Only a few clues were given, so treat this as a rough guess."
            : "Ranked from most to least likely.";
        renderBars($("soybean-bars"), result.predictions);
        renderEvidence("soybean", result);
        setStatus(status, "");
      } catch (error) {
        setStatus(status, error.message, true);
      } finally {
        button.disabled = false;
      }
    });

    updateCount();
  }

  function setupCrop() {
    const form = $("crop-form");
    const status = $("crop-status");
    const inputs = Array.from(form.querySelectorAll("input"));
    let exampleIndex = 0;

    $("crop-example").addEventListener("click", async () => {
      setStatus(status, "Loading a held-out record…");
      try {
        const example = await getJson(`/api/nb/crop/example?n=${exampleIndex++}`);
        inputs.forEach((input) => (input.value = example.inputs[input.name]));
        setStatus(status, `Loaded a real record. Its recorded crop is ${example.actual}. Press Recommend crops to compare.`);
      } catch (error) {
        setStatus(status, error.message, true);
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const values = {};
      for (const input of inputs) {
        if (input.value.trim() === "" || !Number.isFinite(Number(input.value))) {
          setStatus(status, `Enter a number for ${input.closest("label").querySelector("span").textContent.trim()}.`, true);
          return;
        }
        values[input.name] = Number(input.value);
      }
      const button = $("crop-submit");
      button.disabled = true;
      setStatus(status, "Calculating…");
      try {
        const result = await postJson("/api/nb/crop/predict", { inputs: values });
        const top = result.predictions[0];
        $("crop-top").textContent = top.name;
        $("crop-context").textContent = `${percent(top.probability)} probability`;
        $("crop-intro").textContent = "Ranked from best to worst match.";
        renderBars($("crop-bars"), result.predictions);
        renderEvidence("crop", result);

        const warning = $("crop-warning");
        if (result.outside_training_range.length) {
          warning.textContent =
            `Outside the data the model learned from: ${result.outside_training_range.join(", ")}. ` +
            "Treat this result with extra caution.";
          warning.hidden = false;
        } else {
          warning.hidden = true;
        }
        setStatus(status, "");
      } catch (error) {
        setStatus(status, error.message, true);
      } finally {
        button.disabled = false;
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupTabs();
    setupSoybean();
    setupCrop();
  });
})();
