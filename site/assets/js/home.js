const impactForm = document.getElementById("impact-form");
const savingsRange = document.getElementById("savings-range");
function updateSavingsEstimate() {
  const formData = new FormData(impactForm);
  const currency = formData.get("currency");
  const totalCosts = ["fertilizer", "protection", "irrigation"].reduce((total, key) => {
    const value = Number(formData.get(key));
    return total + (Number.isFinite(value) ? Math.max(value, 0) : 0);
  }, 0);

  const lowEstimate = totalCosts * 0.25;
  const highEstimate = totalCosts * 0.3;
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
