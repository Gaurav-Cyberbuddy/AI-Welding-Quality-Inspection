from pathlib import Path
import sys

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "balanced_resnet18.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECK COMMAND-LINE ARGUMENT
# ============================================================

if len(sys.argv) != 2:

    print("Usage:")
    print("python scripts/predict_image.py <image_path>")

    sys.exit(1)


IMAGE_PATH = Path(sys.argv[1])


if not IMAGE_PATH.exists():

    print(f"ERROR: Image not found: {IMAGE_PATH}")

    sys.exit(1)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

])


# ============================================================
# LOAD MODEL
# ============================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(DEVICE)

model.eval()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(IMAGE_PATH).convert("RGB")

except Exception as error:

    print(f"ERROR: Could not open image: {error}")

    sys.exit(1)


original_size = image.size


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)


# ============================================================
# MODEL PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    predicted_label = outputs.argmax(
        dim=1
    ).item()


# ============================================================
# RESULTS
# ============================================================

ok_probability = probabilities[0, 0].item()

ko_probability = probabilities[0, 1].item()


CLASS_NAMES = {
    0: "OK",
    1: "KO",
}


predicted_class = CLASS_NAMES[predicted_label]


print("=" * 60)
print("WELDING IMAGE PREDICTION")
print("=" * 60)

print(f"Device: {DEVICE}")

print(f"Model: {MODEL_PATH.name}")

print(f"Image: {IMAGE_PATH}")

print(f"Original image size: {original_size}")


print("\nPREDICTION")

print("-" * 60)

print(f"Predicted class: {predicted_class}")

print(f"OK probability: {ok_probability * 100:.2f}%")

print(f"KO probability: {ko_probability * 100:.2f}%")


print("\n" + "=" * 60)

print("PREDICTION COMPLETED")

print("=" * 60)