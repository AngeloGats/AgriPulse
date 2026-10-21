const models = {
  linear: {
    name: "Linear Regression",
    category: "REGRESSION",
    categoryClass: "regression-tag",
    icon: "↗",
    description: "See how historical rainfall and harvest data can reveal a trend in crop yield.",
    useCase: "Estimate crop yield",
    useDescription: "Learn how rainfall relates to harvest size, then use the trend to estimate a numeric yield.",
    outputLabel: "EXAMPLE OUTPUT",
    output: 'Yield estimate <span>5.8 t/ha</span>',
    outputNote: "An example numeric prediction",
    chips: ["Rainfall", "Temperature", "Soil quality"],
    note: "Regression predicts a continuous number, rather than a category.",
    legend: '<span><i class="legend-dot harvest-dot"></i> Sample harvests</span><span><i class="legend-line"></i> Fitted trend</span>',
    visualization: `
      <svg class="plot-svg" viewBox="0 0 540 190" role="img" aria-label="Illustrative scatter plot with an upward fitted trend between rainfall and crop yield">
        <line class="gridline" x1="58" y1="42" x2="510" y2="42"/><line class="gridline" x1="58" y1="82" x2="510" y2="82"/><line class="gridline" x1="58" y1="122" x2="510" y2="122"/>
        <line class="axis" x1="58" y1="155" x2="510" y2="155"/><line class="axis" x1="58" y1="25" x2="58" y2="155"/>
        <text x="250" y="180">Rainfall during growing season</text><text transform="translate(19 117) rotate(-90)">Crop yield</text>
        <text x="42" y="158">2</text><text x="42" y="125">4</text><text x="42" y="85">6</text><text x="42" y="45">8</text>
        <path d="M80 137 485 43" fill="none" stroke="#5e9367" stroke-width="2.5" stroke-linecap="round"/>
        <g fill="#82a987" stroke="#fff" stroke-width="2"><circle cx="108" cy="132" r="5"/><circle cx="150" cy="120" r="5"/><circle cx="183" cy="130" r="5"/><circle cx="218" cy="102" r="5"/><circle cx="252" cy="110" r="5"/><circle cx="286" cy="94" r="5"/><circle cx="330" cy="91" r="5"/><circle cx="361" cy="73" r="5"/><circle cx="406" cy="77" r="5"/><circle cx="448" cy="51" r="5"/></g>
      </svg>`,
  },
  bayes: {
    name: "Naive Bayes",
    category: "CLASSIFICATION",
    categoryClass: "classification-tag",
    icon: "◉",
    description: "Combine observed crop symptoms to estimate the likelihood of different conditions.",
    useCase: "Screen for crop disease",
    useDescription: "Compare symptom evidence to estimate which crop condition is most likely.",
    outputLabel: "ILLUSTRATIVE CLASS PROBABILITIES",
    output: 'Most likely <span>Leaf blight</span>',
    outputNote: "Example scores only—not a diagnosis",
    chips: ["Leaf spots", "Yellowing", "Humidity"],
    note: "Naive Bayes combines feature evidence while assuming features are conditionally independent.",
    legend: "",
    visualization: `
      <div class="probability-list" role="img" aria-label="Illustrative probability bars for three possible crop conditions">
        <div class="probability-row"><span>Leaf blight</span><div class="probability-track"><div class="probability-fill" style="width:76%"></div></div><strong>76%</strong></div>
        <div class="probability-row"><span>Healthy</span><div class="probability-track"><div class="probability-fill soft" style="width:49%"></div></div><strong>49%</strong></div>
        <div class="probability-row"><span>Rust disease</span><div class="probability-track"><div class="probability-fill alt" style="width:28%"></div></div><strong>28%</strong></div>
      </div>`,
  },
  knn: {
    name: "K-Nearest Neighbors",
    category: "CLASSIFICATION",
    categoryClass: "classification-tag",
    icon: "⌖",
    description: "Place a new field sample beside known examples and inspect its closest neighbors.",
    useCase: "Compare soil samples",
    useDescription: "Classify a new soil sample by looking at the classes of nearby known samples.",
    outputLabel: "ILLUSTRATIVE CLASS",
    output: 'Nearby samples suggest <span>Loamy soil</span>',
    outputNote: "The result depends on the selected K value",
    chips: ["Soil pH", "Moisture", "Nutrients"],
    note: "KNN uses the labels of the nearest examples; feature scales and the choice of K matter.",
    legend: '<span><i class="legend-dot harvest-dot"></i> Loamy samples</span><span><i class="legend-dot" style="background:#d5a36e"></i> Sandy samples</span><span><i class="legend-dot" style="background:#526f8b"></i> New sample</span>',
    visualization: `
      <svg class="plot-svg" viewBox="0 0 540 190" role="img" aria-label="Illustrative soil sample scatter plot with nearest neighbors highlighted">
        <line class="gridline" x1="58" y1="48" x2="510" y2="48"/><line class="gridline" x1="58" y1="92" x2="510" y2="92"/><line class="gridline" x1="58" y1="136" x2="510" y2="136"/>
        <line class="axis" x1="58" y1="155" x2="510" y2="155"/><line class="axis" x1="58" y1="25" x2="58" y2="155"/>
        <text x="237" y="180">Soil moisture</text><text transform="translate(19 121) rotate(-90)">Nutrient level</text>
        <g fill="#7ba782" stroke="#fff" stroke-width="2"><circle cx="130" cy="75" r="6"/><circle cx="178" cy="119" r="6"/><circle cx="213" cy="91" r="6"/><circle cx="248" cy="126" r="6"/><circle cx="295" cy="73" r="6"/></g>
        <g fill="#d7a36f" stroke="#fff" stroke-width="2"><circle cx="365" cy="111" r="6"/><circle cx="404" cy="64" r="6"/><circle cx="442" cy="126" r="6"/><circle cx="474" cy="88" r="6"/></g>
        <circle cx="276" cy="100" r="23" fill="none" stroke="#78a47c" stroke-width="1.5" stroke-dasharray="4 4"/>
        <circle cx="276" cy="100" r="7" fill="#526f8b" stroke="#fff" stroke-width="2"/>
        <text x="290" y="98">new sample</text>
      </svg>`,
  },
  svm: {
    name: "Support Vector Machine",
    category: "CLASSIFICATION",
    categoryClass: "classification-tag",
    icon: "⟋",
    description: "Explore a boundary that separates two groups of crop observations.",
    useCase: "Separate crop conditions",
    useDescription: "Learn how a boundary can distinguish healthy plants from examples with disease symptoms.",
    outputLabel: "ILLUSTRATIVE CLASS",
    output: 'Sample falls in <span>Healthy</span>',
    outputNote: "A simplified two-feature example",
    chips: ["Leaf color", "Spot count", "Plant vigor"],
    note: "An SVM finds a separating boundary with the widest possible margin between classes.",
    legend: '<span><i class="legend-dot harvest-dot"></i> Healthy examples</span><span><i class="legend-dot" style="background:#d5a36e"></i> Disease examples</span><span><i class="legend-line"></i> Decision boundary</span>',
    visualization: `
      <svg class="plot-svg" viewBox="0 0 540 190" role="img" aria-label="Illustrative SVM plot with classes, a separating boundary, and margin">
        <line class="gridline" x1="58" y1="48" x2="510" y2="48"/><line class="gridline" x1="58" y1="92" x2="510" y2="92"/><line class="gridline" x1="58" y1="136" x2="510" y2="136"/>
        <line class="axis" x1="58" y1="155" x2="510" y2="155"/><line class="axis" x1="58" y1="25" x2="58" y2="155"/>
        <text x="246" y="180">Leaf spot count</text><text transform="translate(19 121) rotate(-90)">Leaf discoloration</text>
        <path d="M122 134 421 47" fill="none" stroke="#7b9c7c" stroke-width="1.2" stroke-dasharray="5 4"/>
        <path d="M101 116 400 29M143 152 442 65" fill="none" stroke="#b5c8b4" stroke-width="1" stroke-dasharray="4 5"/>
        <g fill="#78a27d" stroke="#fff" stroke-width="2"><circle cx="116" cy="83" r="6"/><circle cx="168" cy="61" r="6"/><circle cx="211" cy="92" r="6"/><circle cx="256" cy="56" r="6"/></g>
        <g fill="#d4a16c" stroke="#fff" stroke-width="2"><circle cx="319" cy="115" r="6"/><circle cx="369" cy="93" r="6"/><circle cx="416" cy="127" r="6"/><circle cx="458" cy="102" r="6"/></g>
        <circle cx="256" cy="56" r="9" fill="none" stroke="#557d5b" stroke-width="1.5"/>
        <circle cx="319" cy="115" r="9" fill="none" stroke="#b78351" stroke-width="1.5"/>
      </svg>`,
  },
  tree: {
    name: "Decision Tree",
    category: "RULE-BASED",
    categoryClass: "rules-tag",
    icon: "⑂",
    description: "Follow a sequence of simple conditions to reach an irrigation recommendation.",
    useCase: "Guide an irrigation check",
    useDescription: "Turn soil moisture and recent rainfall into an easy-to-follow decision path.",
    outputLabel: "EXAMPLE DECISION",
    output: 'Soil is dry <span>Check irrigation</span>',
    outputNote: "A simplified rule path",
    chips: ["Soil moisture", "Recent rainfall", "Crop stage"],
    note: "A decision tree makes predictions by following feature-based questions from root to leaf.",
    legend: "",
    visualization: `
      <div class="tree-visual" role="img" aria-label="Illustrative decision tree for irrigation based on soil moisture and recent rainfall">
        <div class="tree-flow">
          <div class="tree-node root">Soil moisture below 30%?</div>
          <div class="tree-branches">
            <div class="tree-branch"><span class="branch-label">YES</span><div class="tree-node">Rain in last 24 hours?</div><div class="tree-node tree-leaf">Check irrigation</div></div>
            <div class="tree-branch"><span class="branch-label">NO</span><div class="tree-node tree-leaf">No immediate irrigation</div></div>
          </div>
        </div>
      </div>`,
  },
  neural: {
    name: "Artificial Neural Network",
    category: "NEURAL NETWORK",
    categoryClass: "neural-tag",
    icon: "✳",
    description: "See how connected layers can learn patterns in crop images and field conditions.",
    useCase: "Recognize leaf conditions",
    useDescription: "A neural network can learn image patterns that help distinguish possible leaf conditions.",
    outputLabel: "ILLUSTRATIVE CLASS",
    output: 'Image pattern suggests <span>Leaf condition</span>',
    outputNote: "An example model output",
    chips: ["Leaf image", "Color pattern", "Texture"],
    note: "Neural networks learn layered patterns from examples; reliable results need representative labeled data.",
    legend: '<span><i class="legend-dot" style="background:#789d7a"></i> Input features</span><span><i class="legend-dot" style="background:#dcebdc;border:1px solid #78a27b"></i> Hidden layer</span><span><i class="legend-dot" style="background:#ad7183"></i> Output</span>',
    visualization: `
      <svg class="neural-svg" viewBox="0 0 480 170" role="img" aria-label="Illustrative neural network with input, hidden, and output layers">
        <g>
          <line x1="76" y1="37" x2="220" y2="34"/><line x1="76" y1="37" x2="220" y2="84"/><line x1="76" y1="37" x2="220" y2="134"/>
          <line x1="76" y1="84" x2="220" y2="34"/><line x1="76" y1="84" x2="220" y2="84"/><line x1="76" y1="84" x2="220" y2="134"/>
          <line x1="76" y1="134" x2="220" y2="34"/><line x1="76" y1="134" x2="220" y2="84"/><line x1="76" y1="134" x2="220" y2="134"/>
          <line x1="220" y1="34" x2="374" y2="60"/><line x1="220" y1="34" x2="374" y2="110"/><line x1="220" y1="84" x2="374" y2="60"/><line x1="220" y1="84" x2="374" y2="110"/><line x1="220" y1="134" x2="374" y2="60"/><line x1="220" y1="134" x2="374" y2="110"/>
        </g>
        <g><circle cx="76" cy="37" r="9"/><circle cx="76" cy="84" r="9"/><circle cx="76" cy="134" r="9"/>
          <circle class="active-node" cx="220" cy="34" r="10"/><circle class="active-node" cx="220" cy="84" r="10"/><circle class="active-node" cx="220" cy="134" r="10"/>
          <circle cx="374" cy="60" r="9"/><circle cx="374" cy="110" r="9"/></g>
        <text x="49" y="160">INPUT</text><text x="190" y="160">HIDDEN LAYER</text><text x="352" y="160">OUTPUT</text>
      </svg>`,
  },
};

const cards = document.querySelectorAll(".algorithm-card[data-model], .algorithm-card-control[data-model]");
const modelDemoLinks = document.querySelectorAll("[data-model-demo]");
const demoCategory = document.getElementById("demoCategory");
const demoIcon = document.getElementById("demoIcon");
const demoName = document.getElementById("demoName");
const demoDescription = document.getElementById("demoDescription");
const visualization = document.getElementById("visualization");
const chartLegend = document.getElementById("chartLegend");
const useCaseTitle = document.getElementById("useCaseTitle");
const useCaseDescription = document.getElementById("useCaseDescription");
const outputLabel = document.getElementById("outputLabel");
const outputValue = document.getElementById("outputValue");
const outputNote = document.getElementById("outputNote");
const dataChips = document.getElementById("dataChips");
const learningNote = document.getElementById("learningNote");

if (
  ["127.0.0.1", "localhost"].includes(window.location.hostname) &&
  window.location.port === "5500"
) {
  modelDemoLinks.forEach((link) => {
    link.href = "http://127.0.0.1:5051/linear-regression";
  });
}

function selectModel(modelKey) {
  const model = models[modelKey];

  if (!model) {
    return;
  }

  cards.forEach((card) => {
    const isSelected = card.dataset.model === modelKey;
    const cardContainer = card.closest(".algorithm-card") || card;
    cardContainer.classList.toggle("selected", isSelected);
    card.setAttribute("aria-pressed", String(isSelected));
  });

  demoCategory.textContent = model.category;
  demoCategory.className = `category-tag ${model.categoryClass}`;
  demoIcon.textContent = model.icon;
  demoName.textContent = model.name;
  demoDescription.textContent = model.description;
  visualization.innerHTML = model.visualization;
  chartLegend.innerHTML = model.legend;
  chartLegend.hidden = !model.legend;
  useCaseTitle.textContent = model.useCase;
  useCaseDescription.textContent = model.useDescription;
  outputLabel.textContent = model.outputLabel;
  outputValue.innerHTML = model.output;
  outputNote.textContent = model.outputNote;
  dataChips.replaceChildren(
    ...model.chips.map((chip) => {
      const element = document.createElement("span");
      element.textContent = chip;
      return element;
    })
  );
  learningNote.textContent = model.note;
}

cards.forEach((card) => {
  const cardContainer = card.closest(".algorithm-card") || card;
  card.setAttribute("aria-pressed", String(cardContainer.classList.contains("selected")));
  card.addEventListener("click", () => selectModel(card.dataset.model));
});
