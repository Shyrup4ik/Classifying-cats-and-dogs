from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


WEIGHTS_PATH = Path("/Users/yuri/Downloads/cat_dog_fixed.pt")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    model = models.resnet18(weights=None)
    model.conv1.stride = (1, 1)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, image: Image.Image) -> dict:
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    cat = probabilities[0].item()
    dog = probabilities[1].item()

    return {
        "label": "cat" if cat >= dog else "dog",
        "probabilities": {
            "cat": cat,
            "dog": dog,
        },
    }