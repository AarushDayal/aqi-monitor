function getColor(aqi) {
    if (aqi <= 50) return "#00e400";
    if (aqi <= 100) return "#ffff00";
    if (aqi <= 200) return "#ff7e00";
    if (aqi <= 300) return "#ff0000";
    if (aqi <= 400) return "#8f3f97";
    return "#7e0023";
}

function getAdvice(aqi) {
    if (aqi <= 50) return "Air quality is good 👍";
    if (aqi <= 100) return "Safe for most people";
    if (aqi <= 200) return "Sensitive groups should limit outdoor activity";
    if (aqi <= 300) return "Wear masks outdoors";
    return "Stay indoors 🚨";
}

function safeSet(id, value) {
    const el = document.getElementById(id);

    if (!el) {
        console.error("Element missing:", id);
        return;
    }

    el.innerText = value;

    const color = getColor(value);
    el.style.color = color;
    el.style.textShadow = `0 0 15px ${color}`;
}

function fetchAQI() {
    console.log("Fetching AQI..."); // DEBUG

    fetch("/predict")
        .then(res => res.json())
        .then(data => {
            console.log("DATA:", data); // DEBUG

            if (data.status === "error") {
                alert(data.message);
                return;
            }

            safeSet("aqi_now", data.aqi_now);
            safeSet("aqi_8h", data.aqi_8h);
            safeSet("aqi_24h", data.aqi_24h);
            safeSet("aqi_7d", data.aqi_7d);

            const badge = document.getElementById("category");
            if (badge) {
                badge.innerText = data.category;
                badge.style.background = getColor(data.aqi_now);
            }

            const loc = document.getElementById("location");
            if (loc) {
                loc.innerText = data.location;
            }

            const pollutants = data.pollutants || {};

            safeSet("pm25", pollutants["pm25"]);
            safeSet("pm10", pollutants["pm10"]);
            safeSet("no2", pollutants["no2"]);
            safeSet("co", pollutants["co"]);
            safeSet("so2", pollutants["so2"]);

            const advice = document.getElementById("advice");
            if (advice) {
                advice.innerText = getAdvice(data.aqi_now);
            }
        })
        .catch(err => {
            console.error("FETCH ERROR:", err);
        });
}

/* 🔥 CRITICAL FIX */
window.onload = function () {
    console.log("Page loaded"); // DEBUG
    fetchAQI();
};

