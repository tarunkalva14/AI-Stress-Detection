import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

# ============================
# DEVICE
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# TRANSFORMS (same as training)
# ============================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ============================
# LOAD MODEL
# ============================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)   # 2 classes
model.load_state_dict(torch.load("stress_model.pth", map_location=device))
model.to(device)
model.eval()

# ============================
# PREDICT FUNCTION
# ============================
def predict(img_path):
    img_path = img_path.replace("\\", "/")   # Fix Windows path issue
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = "Relaxed 🙂" if pred.item() == 0 else "Stressed 😟"

    print("\n============================")
    print("Image:", img_path)
    print("Prediction :", label)
    print("Confidence :", f"{conf.item() * 100:.2f}%")
    print("============================")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict(sys.argv[1])
