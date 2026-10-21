const form = document.getElementById("prediction-form");
const statusMessage = document.getElementById("form-status");
const predictButton = document.getElementById("predict-button");
const loadExampleButton = document.getElementById("load-example");
const resultValue = document.getElementById("result-value");
const resultContext = document.getElementById("result-context");
const resultWarning = document.getElementById("result-warning");
const resultIntro = document.querySelector(".ml-result-intro");
const regressionChart = document.getElementById("regression-chart");
const regressionChartDescription = regressionChart.querySelector("desc");
const chartCaption = document.getElementById("chart-caption");

const sensorFields = ["mean", "std", "min", "max", "observations"];
const svgNamespace = "http://www.w3.org/2000/svg";

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function renderRegressionChart(points, estimate = null) {
  const title = regressionChart.querySelector("title");
  regressionChart.replaceChildren(title, regressionChartDescription);

  const plot = { left: 76, top: 22, width: 650, height: 278 };
  const xMax = Math.ceil(Math.max(...points.map((point) => point.actual)) / 2) * 2;
  const meanActual =
    points.reduce((sum, point) => sum + point.actual, 0) / points.length;
  const meanPredicted =
    points.reduce((sum, point) => sum + point.predicted, 0) / points.length;
  const covariance = points.reduce(
    (sum, point) =>
      sum + (point.actual - meanActual) * (point.predicted - meanPredicted),
    0
  );
  const variance = points.reduce(
    (sum, point) => sum + (point.actual - meanActual) ** 2,
    0
  );
  const slope = covariance / variance;
  const intercept = meanPredicted - slope * meanActual;
  const fitStartY = intercept;
  const fitEndY = slope * xMax + intercept;
  const allPredictions = points.map((point) => point.predicted);
  if (estimate !== null) allPredictions.push(estimate);
  const yMin = Math.floor(
    Math.min(0, ...allPredictions, fitStartY, fitEndY) / 2
  ) * 2;
  const yMax = Math.ceil(
    Math.max(...allPredictions, fitStartY, fitEndY) / 2
  ) * 2 || 2;
  const xFor = (value) => plot.left + (value / xMax) * plot.width;
  const yFor = (value) =>
    plot.top + plot.height - ((value - yMin) / (yMax - yMin)) * plot.height;
  const tickCount = 5;

  for (let tick = 0; tick <= tickCount; tick += 1) {
    const xValue = (xMax * tick) / tickCount;
    const yValue = yMin + ((yMax - yMin) * tick) / tickCount;
    const x = xFor(xValue);
    const y = yFor(yValue);

    regressionChart.append(
      svgElement("line", {
        x1: plot.left,
        y1: y,
        x2: plot.left + plot.width,
        y2: y,
        class: "ml-chart-gridline",
      }),
      svgElement("line", {
        x1: x,
        y1: plot.top,
        x2: x,
        y2: plot.top + plot.height,
        class: "ml-chart-gridline vertical",
      })
    );

    const yTick = svgElement("text", {
      x: plot.left - 12,
      y: y + 4,
      "text-anchor": "end",
      class: "ml-chart-tick",
    });
    yTick.textContent = yValue.toFixed(0);
    regressionChart.append(yTick);

    const xTick = svgElement("text", {
      x,
      y: plot.top + plot.height + 21,
      "text-anchor": "middle",
      class: "ml-chart-tick",
    });
    xTick.textContent = xValue.toFixed(0);
    regressionChart.append(xTick);
  }

  regressionChart.append(
    svgElement("line", {
      x1: plot.left,
      y1: plot.top,
      x2: plot.left,
      y2: plot.top + plot.height,
      class: "ml-chart-axis",
    }),
    svgElement("line", {
      x1: plot.left,
      y1: plot.top + plot.height,
      x2: plot.left + plot.width,
      y2: plot.top + plot.height,
      class: "ml-chart-axis",
    })
  );

  regressionChart.append(
    svgElement("line", {
      x1: xFor(0),
      y1: yFor(fitStartY),
      x2: xFor(xMax),
      y2: yFor(fitEndY),
      class: "ml-chart-fit",
    })
  );

  for (const point of points) {
    const dot = svgElement("circle", {
      cx: xFor(point.actual),
      cy: yFor(point.predicted),
      r: 4,
      class: "ml-chart-point",
    });
    const tooltip = svgElement("title");
    tooltip.textContent =
      `Actual: ${point.actual.toFixed(2)} t/ha; ` +
      `predicted: ${point.predicted.toFixed(2)} t/ha`;
    dot.append(tooltip);
    regressionChart.append(dot);
  }

  if (estimate !== null) {
    const estimateY = yFor(estimate);
    const estimateGroup = svgElement("g", { class: "ml-chart-estimate" });
    const estimateLine = svgElement("line", {
      x1: plot.left,
      y1: estimateY,
      x2: plot.left + plot.width,
      y2: estimateY,
      class: "ml-chart-estimate-line",
    });
    const estimateMarker = svgElement("circle", {
      cx: plot.left + plot.width,
      cy: estimateY,
      r: 6,
      class: "ml-chart-estimate-marker",
    });
    const estimateTooltip = svgElement("title");
    estimateTooltip.textContent = `Your model estimate: ${estimate.toFixed(2)} t/ha`;
    estimateMarker.append(estimateTooltip);
    const estimateLabel = svgElement("text", {
      x: plot.left + plot.width - 9,
      y: estimateY - 9,
      "text-anchor": "end",
      class: "ml-chart-estimate-label",
    });
    estimateLabel.textContent = `Your estimate: ${estimate.toFixed(2)} t/ha`;
    estimateGroup.append(estimateLine, estimateMarker, estimateLabel);
    regressionChart.append(estimateGroup);
    regressionChartDescription.textContent =
      `Held-out actual and predicted yields, with a fitted trend line. ` +
      `Your latest model estimate is ${estimate.toFixed(2)} tonnes per hectare, ` +
      "shown as a horizontal reference.";
    chartCaption.textContent =
      `Your estimate: ${estimate.toFixed(2)} t/ha · ` +
      `${points.length} held-out fields for comparison.`;
  } else {
    regressionChartDescription.textContent =
      "Scatter plot of observed and predicted yields across held-out CYCleSS field records, with a fitted regression trend line. A new model estimate will be highlighted here after calculation.";
    chartCaption.textContent =
      `Submit your field inputs to highlight the new estimate against ${points.length} held-out fields.`;
  }

  const xLabel = svgElement("text", {
    x: plot.left + plot.width / 2,
    y: 359,
    "text-anchor": "middle",
    class: "ml-chart-axis-label",
  });
  xLabel.textContent = "Actual yield (t/ha)";
  regressionChart.append(xLabel);

  const yLabel = svgElement("text", {
    x: 20,
    y: plot.top + plot.height / 2,
    "text-anchor": "middle",
    transform: `rotate(-90 20 ${plot.top + plot.height / 2})`,
    class: "ml-chart-axis-label",
  });
  yLabel.textContent = "Predicted yield (t/ha)";
  regressionChart.append(yLabel);
}

const diagnosticPoints = JSON.parse(
  document.getElementById("regression-chart-data").textContent
);
renderRegressionChart(diagnosticPoints);

function payloadFromForm() {
  const payload = {
    crop: document.getElementById("crop").value,
    year: Number(document.getElementById("year").value),
  };

  for (const sensor of ["ratio", "vh", "vv"]) {
    payload[sensor] = {};
    for (const field of sensorFields) {
      const input = document.getElementById(`${sensor}_${field}`);
      payload[sensor][field] =
        input.disabled || input.value === "" ? null : Number(input.value);
    }
  }
  return payload;
}

function syncSensorInputs(sensor) {
  const countInput = document.getElementById(`${sensor}_observations`);
  const noObservations = countInput.value !== "" && Number(countInput.value) === 0;
  for (const field of ["mean", "std", "min", "max"]) {
    const input = document.getElementById(`${sensor}_${field}`);
    input.disabled = noObservations;
    input.required = !noObservations && field !== "std";
    input.placeholder = noObservations
      ? "Imputed from training"
      : field === "std"
        ? "Optional (imputed if blank)"
        : "Enter value";
    if (noObservations) input.value = "";
  }
}

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("is-error", isError);
}

function clearResult() {
  resultWarning.hidden = true;
  resultWarning.textContent = "";
}

for (const sensor of ["ratio", "vh", "vv"]) {
  const countInput = document.getElementById(`${sensor}_observations`);
  countInput.addEventListener("input", () => syncSensorInputs(sensor));
  syncSensorInputs(sensor);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  predictButton.disabled = true;
  predictButton.textContent = "Calculating…";
  clearResult();
  setStatus("Sending your inputs to the trained model…");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm()),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "The model could not process these inputs.");
    }

    resultValue.innerHTML = `${result.prediction_t_per_ha.toFixed(2)} <small>t/ha</small>`;
    resultContext.textContent = `${result.crop} · ${result.year}`;
    resultIntro.textContent = "Predicted mean field yield";
    setStatus("Prediction completed with the trained CYCleSS Linear Regression model.");
    renderRegressionChart(diagnosticPoints, result.prediction_t_per_ha);

    if (result.outside_training_range.length > 0) {
      resultWarning.hidden = false;
      resultWarning.textContent =
        "Some satellite inputs are outside the model's training ranges. Treat this estimate as an extrapolation.";
    }
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    predictButton.disabled = false;
    predictButton.innerHTML = 'Calculate yield estimate <span aria-hidden="true">→</span>';
  }
});

loadExampleButton.addEventListener("click", async () => {
  loadExampleButton.disabled = true;
  renderRegressionChart(diagnosticPoints);
  setStatus("Loading a sample field from the CYCleSS test split…");
  try {
    const response = await fetch("/api/example");
    const example = await response.json();
    if (!response.ok) {
      throw new Error(example.error || "Example inputs could not be loaded.");
    }

    document.getElementById("crop").value = example.crop;
    document.getElementById("year").value = String(example.year);
    for (const sensor of ["ratio", "vh", "vv"]) {
      for (const field of sensorFields) {
        document.getElementById(`${sensor}_${field}`).value = example[sensor][field];
      }
      syncSensorInputs(sensor);
    }
    setStatus("Example values loaded. Submit the form to calculate a prediction.");
    clearResult();
    resultValue.innerHTML = '— <small>t/ha</small>';
    resultContext.textContent = "No calculation yet";
    resultIntro.textContent = "Submit field inputs to calculate a prediction.";
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    loadExampleButton.disabled = false;
  }
});
