let activeTabUrl = "";

document.addEventListener("DOMContentLoaded", async () => {
    // Get the URL of the active browser tab
    let tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0] && tabs[0].url) {
        activeTabUrl = tabs[0].url;
        document.getElementById("currentUrl").innerText = activeTabUrl;
    } else {
        document.getElementById("currentUrl").innerText = "Unable to fetch active URL";
    }
});

async function scanCurrentTab() {
    if (!activeTabUrl) {
        alert("No URL available to scan.");
        return;
    }

    const loading = document.getElementById("loading");
    const resultCard = document.getElementById("resultCard");
    loading.style.display = "block";
    resultCard.style.display = "none";

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: activeTabUrl })
        });

        const data = await response.json();
        loading.style.display = "none";

        if (response.ok) {
            resultCard.className = `result-card ${data.risk_level}`;
            document.getElementById("resTitle").innerText = `Prediction: ${data.prediction}`;
            document.getElementById("resRisk").innerHTML = `Risk Level: <strong>${data.risk_level}</strong>`;

            let probsHTML = "";
            for (const [cls, prob] of Object.entries(data.probabilities)) {
                probsHTML += `<div class="metric"><span>${cls}</span><span>${prob}%</span></div>`;
            }
            document.getElementById("resProbs").innerHTML = probsHTML;
            resultCard.style.display = "block";
        } else {
            alert(data.error || "Prediction failed.");
        }
    } catch (err) {
        loading.style.display = "none";
        alert("Could not connect to Flask server. Ensure app.py is running on port 5000.");
    }
}