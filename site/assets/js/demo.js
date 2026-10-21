const fieldForm = document.getElementById("field-form");
const summaryNote = document.getElementById("summary-note");

const fieldSummaryTargets = {
  fieldName: "summary-field",
  crop: "summary-crop",
  rainfall: "summary-rainfall",
  moisture: "summary-moisture",
  humidity: "summary-humidity",
  nitrogen: "summary-nitrogen",
};

fieldForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const values = new FormData(fieldForm);
  const displayValues = {
    fieldName: values.get("fieldName").trim(),
    crop: values.get("crop"),
    rainfall: `${values.get("rainfall")} mm`,
    moisture: `${values.get("moisture")}%`,
    humidity: `${values.get("humidity")}%`,
    nitrogen: `${values.get("nitrogen")} mg/kg`,
  };

  Object.entries(fieldSummaryTargets).forEach(([field, targetId]) => {
    document.getElementById(targetId).textContent = displayValues[field];
  });

  document.getElementById("result-title").textContent = `${displayValues.fieldName} snapshot`;
  document.getElementById("result-description").textContent = `Sample measurements recorded for ${displayValues.crop}.`;
  summaryNote.textContent = "Inputs summarized successfully. No model interpreted them and no recommendation was generated.";
});
