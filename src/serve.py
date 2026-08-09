from io import BytesIO

import torch

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms

from src.model import get_model


CHECKPOINT_PATH = "./checkpoints/model.pt"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

app = FastAPI()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = get_model("resnet18", 10)
model_loaded = False
# classes = []

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


def load_model():
    global model_loaded
    # global classes

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # classes = CIFAR10_CLASSES

    model.to(device)
    model.eval()

    model_loaded = True


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    tensor = transform(image)\
        .unsqueeze(0)\
        .to(device)

    with torch.no_grad():
        outputs = model(tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )[0]

    result = {}

    for i, class_name in enumerate(CIFAR10_CLASSES):
        result[class_name] = float(probs[i])

    return result