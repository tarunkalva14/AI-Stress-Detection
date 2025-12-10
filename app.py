# app.py
import os
import time
import cv2
import boto3
import torch
import numpy as np
from collections import deque
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import models
import torch.nn as nn

# ---------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "stress_model.pth"
S3_BUCKET = "stress-detection-s3"   # replace with your bucket name or leave as-is
SMOOTH_WINDOW = 5

print("Device:", DEVICE)

# ---------- LOAD MODEL ----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------- HELPERS ----------
def preprocess_pil(pil_img):
    pil = pil_img.resize((64, 64)).convert("RGB")
    arr = np.array(pil).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.tensor(arr).unsqueeze(0).float().to(DEVICE)
    return tensor

smooth_buf = deque(maxlen=SMOOTH_WINDOW)

def predict_confidence_from_bgr(frame_bgr):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    t = preprocess_pil(pil)
    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    p_stress = float(probs[1])  # class 1 = stressed
    smooth_buf.append(p_stress)
    avg = float(sum(smooth_buf) / len(smooth_buf))
    return avg  # 0..1

# ---------- S3 ----------
s3 = boto3.client("s3")

def upload_frame_to_s3(frame_bgr):
    try:
        _, buf = cv2.imencode(".jpg", frame_bgr)
        key = f"predictions/{int(time.time())}.jpg"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.tobytes(), ContentType="image/jpeg")
        return key
    except Exception as e:
        # If upload fails, return None (upload is optional)
        print("S3 upload failed:", e)
        return None

# ---------- FLASK ----------
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    data = file.read()
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    conf = predict_confidence_from_bgr(frame)  # 0..1
    conf_pct = round(conf * 100, 1)

    s3_key = upload_frame_to_s3(frame)

    return jsonify({"confidence": conf_pct, "s3_key": s3_key})

@app.route("/predict", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    tmp = "uploaded_video.avi"
    f.save(tmp)
    cap = cv2.VideoCapture(tmp)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open uploaded video"}), 400
    probs = []
    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        probs.append(predict_confidence_from_bgr(frame))
    cap.release()
    if not probs:
        return jsonify({"error": "No frames"}), 400
    avg = float(sum(probs) / len(probs))
    conf_pct = round(avg * 100, 1)
    s3_key = None
    if last_frame is not None:
        s3_key = upload_frame_to_s3(last_frame)
    return jsonify({"confidence": conf_pct, "s3_key": s3_key})

if __name__ == "__main__":
    print("Starting Flask on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
