from flask import Flask, request, jsonify, render_template
import torch, os, cv2, numpy as np
from torchvision import models
from utils import extract_face_pil, preprocess_pil_to_tensor, append_log, pil_to_jpeg_bytes, upload_to_s3_bytes
from PIL import Image

app = Flask(__name__)

# ----------------- CONFIG -----------------
DEVICE = "cpu"
MODEL_PATH = "stress_model.pth"
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
# ------------------------------------------

# ----------------- RUN MODE -----------------
LOCAL_RUN = int(os.environ.get("LOCAL_RUN", "1"))
if LOCAL_RUN:
    cap = cv2.VideoCapture(0)
# ------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

def predict_image(img_pil):
    face, _ = extract_face_pil(img_pil)
    if face is None:
        return "NoFace", 0.0
    arr = preprocess_pil_to_tensor(face)
    tensor = torch.tensor(arr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        conf = float(probs[1]*100)
        label = "Stressed" if conf > 50 else "Relaxed"
        return label, conf

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    try:
        if LOCAL_RUN:
            ret, frame = cap.read()
            if not ret:
                return jsonify({"error":"Failed to capture frame"})
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            if "image" not in request.files:
                return jsonify({"error":"No image uploaded"})
            file = request.files["image"]
            pil_img = Image.open(file.stream).convert("RGB")

        label, conf = predict_image(pil_img)

        # ------------------- Logging -------------------
        log_data = {"label": label, "confidence": conf}
        append_log(log_data)  # local log
        # Upload log + frame to S3
        img_bytes = pil_to_jpeg_bytes(pil_img)
        s3_key = f"logs/{label}_{int(conf)}_{int(torch.randint(0,1000000,(1,)))}.jpg"
        upload_to_s3_bytes(img_bytes, s3_key)
        # -----------------------------------------------

        return jsonify({"label": label, "confidence": conf})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict_video():
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file uploaded"})
        file = request.files["file"]
        temp_path = "temp_video.avi"
        file.save(temp_path)
        cap_vid = cv2.VideoCapture(temp_path)
        stress_probs = []

        while True:
            ret, frame = cap_vid.read()
            if not ret:
                break
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _, conf = predict_image(pil_frame)
            stress_probs.append(conf/100)
            # Upload each frame to S3
            img_bytes = pil_to_jpeg_bytes(pil_frame)
            s3_key = f"video_frames/frame_{int(conf*100)}_{int(torch.randint(0,1000000,(1,)))}.jpg"
            upload_to_s3_bytes(img_bytes, s3_key)

        cap_vid.release()
        os.remove(temp_path)

        if not stress_probs:
            return jsonify({"error":"No frames found"})
        avg_prob = float(np.mean(stress_probs))
        final_label = "Stressed" if avg_prob > 0.5 else "Relaxed"
        append_log({"label": final_label, "confidence": avg_prob*100})

        return jsonify({"status": "success", "prediction": final_label, "stress_probability": avg_prob})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    if LOCAL_RUN:
        print("Running locally at http://127.0.0.1:5000/")
        app.run(host="127.0.0.1", port=5000, debug=True)
    else:
        print("Running on server/EC2, public access via browser")
        app.run(host="0.0.0.0", port=5000, debug=True)
