# demo_record.py
import cv2, time, torch, numpy as np
from collections import deque
from PIL import Image
from torchvision import models
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "stress_model.pth"

# load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

buf = deque(maxlen=5)

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img).resize((64,64))
    arr = np.array(pil).astype("float32") / 255.0
    arr = np.transpose(arr, (2,0,1))
    tensor = torch.tensor(arr).unsqueeze(0).to(DEVICE)
    return tensor

def predict_prob(frame):
    t = preprocess_frame(frame)
    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        p_stressed = float(probs[1])
        buf.append(p_stressed)
        avg = float(sum(buf)/len(buf))
    return avg

def record_demo(seconds=30, out_file="demo_output.avi"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_file, fourcc, 20.0, (640,480))
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        p = predict_prob(frame)
        label = "STRESSED" if p>0.5 else "RELAXED"
        color = (0,0,255) if label=="STRESSED" else (0,255,0)
        cv2.putText(frame, f"{label} ({p:.2f})", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        out.write(frame)
        cv2.imshow("Demo", frame)
        if time.time() - start > seconds: break
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Saved:", out_file)

if __name__ == "__main__":
    record_demo(30)
