let video = document.getElementById("video");
let startBtn = document.getElementById("start");
let stopBtn = document.getElementById("stop");
let status = document.getElementById("status");
let result = document.getElementById("result");
let logEl = document.getElementById("log");

let stream = null;
let intervalId = null;
let running = false;

// calibration and threshold
let threshold = 50.0; // percent
let calibrated = false;

// Chart config
const MAX_POINTS = 30;
let stressData = [];
let labels = [];
const ctx = document.getElementById("trendChart").getContext("2d");
const trendChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            label: 'Stress (%)',
            data: stressData,
            borderColor: '#007BFF',
            backgroundColor: 'rgba(0,123,255,0.2)',
            tension: 0.3,
            fill: true,
            pointRadius: 2
        }]
    },
    options: {
        responsive: true,
        animation: { duration: 200 },
        scales: { y: { min: 0, max: 100 }, x: { display: false } }
    }
});

// start camera
async function startCamera() {
    if (stream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Camera not supported in this browser.");
        status.innerText = "Status: camera unsupported";
        return;
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        await video.play();
        status.innerText = "Status: camera ready";
    } catch (e) {
        console.error(e);
        status.innerText = "Status: camera error";
        alert("Unable to access camera. Make sure you allow it in browser and use HTTPS.");
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
}

// Convert dataURL to blob
function dataURLtoBlob(dataurl) {
    let arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1];
    let bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new Blob([u8arr], { type: mime });
}

// Send frame to Flask and get confidence
async function sendFrameAndGetConfidence() {
    if (!stream) return null;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 360;
    const ctxC = canvas.getContext("2d");
    ctxC.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    const blob = dataURLtoBlob(dataUrl);
    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    try {
        const resp = await fetch("/predict_frame", { method: "POST", body: form });
        if (!resp.ok) {
            console.error("Server error", resp.status);
            return null;
        }
        const j = await resp.json();
        if (j.error) {
            console.error("API error", j.error);
            return null;
        }
        return j.confidence; // numeric percent
    } catch (e) {
        console.error("Network error", e);
        return null;
    }
}

function decideLabelFromConfidence(conf) {
    return (conf >= threshold) ? "Stressed" : "Relaxed";
}

async function sendFrameLoop() {
    const conf = await sendFrameAndGetConfidence();
    if (conf === null) {
        status.innerText = "Status: error";
        return;
    }

    const label = decideLabelFromConfidence(conf);

    result.innerHTML = `<span class="${label === 'Stressed' ? 'stressed' : 'relaxed'}">${label} — ${conf}%</span>`;
    status.innerText = `Status: running (threshold ${threshold.toFixed(1)}%)`;

    if (stressData.length >= MAX_POINTS) { stressData.shift(); labels.shift(); }
    stressData.push(conf);
    labels.push(new Date().toLocaleTimeString());
    trendChart.update();

    const logEntry = `[${new Date().toLocaleTimeString()}] ${label} (${conf}%)`;
    logEl.innerText = logEntry + "\n" + logEl.innerText;
}

// calibration: capture N frames, set threshold = avg + margin
async function calibrateBaseline(frames = 30, margin = 15.0) {
    if (!stream) {
        await startCamera();
        if (!stream) return;
    }
    status.innerText = "Calibrating... keep a neutral face";
    const vals = [];
    for (let i = 0; i < frames; i++) {
        const v = await sendFrameAndGetConfidence();
        if (v !== null) vals.push(v);
        await new Promise(r => setTimeout(r, 150));
    }
    if (vals.length === 0) {
        status.innerText = "Calibration failed";
        alert("Calibration failed. Make sure camera is working.");
        return;
    }
    const avg = vals.reduce((a,b) => a+b, 0) / vals.length;
    threshold = Math.min(95, avg + margin);
    calibrated = true;
    status.innerText = `Calibrated. baseline ${avg.toFixed(1)}% → threshold ${threshold.toFixed(1)}%`;
}

startBtn.onclick = async () => {
    await startCamera();
    running = true;
    intervalId = setInterval(() => { if (running) sendFrameLoop(); }, 600);
    status.innerText = "Status: running";
};

stopBtn.onclick = () => {
    running = false;
    clearInterval(intervalId);
    stopCamera();
    status.innerText = "Status: stopped";
};
