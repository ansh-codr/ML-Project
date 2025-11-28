const form = document.getElementById("patient-form");
const result = document.getElementById("result");

form.addEventListener("submit", async e => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());
    const payload = {
        age: Number(data.age),
        sex: data.sex === "female" ? 1 : 0,
        smoking_status: data.smoking_status === "former" ? 1 : data.smoking_status === "current" ? 2 : 0,
        pd_l1: Number(data.pd_l1),
        tmb: Number(data.tmb),
        kras_mutated: Number(data.kras_mutated),
        egfr_mutated: Number(data.egfr_mutated)
    };
    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            result.textContent = "request failed";
            return;
        }
        const body = await res.json();
        result.textContent = `probability ${body.probability.toFixed(3)} - ${body.recommendation}`;
    } catch (err) {
        result.textContent = "network error";
    }
});
