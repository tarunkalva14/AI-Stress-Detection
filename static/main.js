// static/main.js
let video = document.getElementById("video");
let startBtn = document.getElementById("start");
let stopBtn = document.getElementById("stop");
let status = document.getElementById("status");
let result = document.getElementById("result");
let logEl = document.getElementById("log");

let stream = null;
let intervalId = null;
let running = false;

// ------------------- Trend Chart -------------------
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
        animation: { duration: 300 },
        scales: {
            y: { min: 0, max: 100, title: { display: true, text: 'Stress %' } },
            x: { display: false }
        }
    }
});

// ------------------- Webcam Functions -------------------
async function startCamera() {
    if (stream) return;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.play();
    } catch (e) {
        status.innerText = "Status: camera error";
        console.error(e);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
}

// Convert dataURL to Blob
function dataURLtoBlob(dataurl) {
    let arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1];
    let bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new Blob([u8arr], { type: mime });
}

// ------------------- Frame Prediction -------------------
async function sendFrame() {
    if (!stream) return;
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 360;
    let ctxCanvas = canvas.getContext("2d");
    ctxCanvas.drawImage(video, 0, 0, canvas.width, canvas.height);

    let dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    let blob = dataURLtoBlob(dataUrl);

    let form = new FormData();
    form.append("image", blob, "frame.jpg");

    status.innerText = "Status: sending...";
    try {
        let resp = await fetch("/predict_frame", { method: "POST", body: form });
        if (!resp.ok) {
            status.innerText = "Status: error " + resp.status;
            return;
        }

        let j = await resp.json();
        if (j.error) {
            status.innerText = "Status: " + j.error;
            return;
        }

        let label = j.label;
        let conf = j.confidence.toFixed(1);

        // Update current prediction
        result.innerHTML = `<span class="${label === 'Stressed' ? 'stressed' : 'relaxed'}">${label} — ${conf}%</span>`;

        // Update trend chart
        if (stressData.length >= MAX_POINTS) {
            stressData.shift();
            labels.shift();
        }
        stressData.push(conf);
        labels.push(new Date().toLocaleTimeString());
        trendChart.update();

        // Update logs
        const logEntry = `[${new Date().toLocaleTimeString()}] ${label} (${conf}%)`;
        logEl.innerText = logEntry + "\n" + logEl.innerText;

        status.innerText = "Status: idle";

    } catch (e) {
        console.error(e);
        status.innerText = "Status: network error";
    }
}

// ------------------- Buttons -------------------
startBtn.onclick = async () => {
    await startCamera();
    running = true;
    intervalId = setInterval(() => { if (running) sendFrame(); }, 500);
    status.innerText = "Status: running";
}

stopBtn.onclick = () => {
    running = false;
    clearInterval(intervalId);
    stopCamera();
    status.innerText = "Status: stopped";
}
