"""Inference matching Run_ResNet18.ipynb, with no random augmentation."""

import hashlib
import io
import logging
import os
from pathlib import Path
import time
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    pass


class Classifier:
    def __init__(self, weights_path: Path):
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing model weights: {weights_path}")
        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))
        self.model = models.resnet18(weights=None)
        self.model.conv1.stride = (1, 1)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.weights_id = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:12]
        with torch.inference_mode():
            output = self.model(torch.zeros(1, 3, 32, 32))
            if not torch.isfinite(output).all():
                raise ValueError("Model weights produce non-finite predictions")
        logger.info("ResNet18 ready on CPU; weights=%s", self.weights_id)

    def predict(self, data: bytes) -> dict:
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format not in ALLOWED_FORMATS:
                        raise InvalidImage("Поддерживаются только JPG, PNG и WebP.")
                    if source.width * source.height > MAX_IMAGE_PIXELS:
                        raise InvalidImage("Фото слишком большое: максимум 20 мегапикселей.")
                    source.load()
                    photo = ImageOps.exif_transpose(source).convert("RGB")
                    width, height = photo.size
                    tensor = self.transform(photo).unsqueeze(0)
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise InvalidImage("Фото слишком большое: максимум 20 мегапикселей.") from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            if isinstance(exc, InvalidImage):
                raise
            raise InvalidImage("Не удалось прочитать фото. Выберите корректный JPG, PNG или WebP.") from exc

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Non-finite model prediction")
        cat, dog = probabilities.tolist()
        label = "cat" if cat >= dog else "dog"
        return {
            "label": label,
            "label_ru": "Кошка" if label == "cat" else "Собака",
            "confidence": max(cat, dog),
            "probabilities": {"cat": cat, "dog": dog},
            "image": {"width": width, "height": height},
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
            "model": "ResNet18",
            "weights_id": self.weights_id,
        }
