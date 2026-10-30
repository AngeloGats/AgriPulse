const impactForm = document.getElementById("impact-form");
const savingsRange = document.getElementById("savings-range");
function updateSavingsEstimate() {
  const formData = new FormData(impactForm);
  const currency = formData.get("currency");
  const totalCosts = ["fertilizer", "protection", "irrigation"].reduce((total, key) => {
    const value = Number(formData.get(key));
    return total + (Number.isFinite(value) ? Math.max(value, 0) : 0);
  }, 0);

  const readPercent = (key, fallback) => {
    const raw = formData.get(key);
    if (raw === null || raw === "") return fallback;
    const value = Number(raw);
    return Number.isFinite(value) ? Math.min(Math.max(value, 0), 100) : fallback;
  };
  const lowPercent = readPercent("reductionLow", 25);
  const highPercent = readPercent("reductionHigh", 30);
  const lowEstimate = totalCosts * (Math.min(lowPercent, highPercent) / 100);
  const highEstimate = totalCosts * (Math.max(lowPercent, highPercent) / 100);
  const formatter = new Intl.NumberFormat("en-PH", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  });
  const currencySymbol = formatter
    .formatToParts(0)
    .find((part) => part.type === "currency").value;

  document.querySelectorAll("[data-currency-symbol]").forEach((element) => {
    element.textContent = currencySymbol;
  });
  savingsRange.textContent = `${formatter.format(lowEstimate)} – ${formatter.format(highEstimate)}`;
}

impactForm.addEventListener("input", updateSavingsEstimate);
impactForm.addEventListener("change", updateSavingsEstimate);
