import io
import os
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import nn
from torchvision import models, transforms

from app import main, model


def upload(client, content, name="photo.png", content_type="image/png"):
    return client.post("/api/predict", files={"file": (name, content, content_type)})


def test_health_and_openapi(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ready"
    assert health["device"] == "cpu"
    assert len(health["weights_id"]) == 12
    assert "/api/predict" in client.get("/api/openapi.json").json()["paths"]


@pytest.mark.parametrize("format,mode", [("JPEG", "RGB"), ("PNG", "RGBA"), ("WEBP", "RGB"), ("PNG", "L")])
def test_supported_images_and_probabilities(client, image_bytes, format, mode):
    response = upload(client, image_bytes(format=format, mode=mode))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["label"] in ("cat", "dog")
    assert result["label_ru"] in ("Кошка", "Собака")
    assert result["image"] == {"width": 80, "height": 60}
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert result["confidence"] == max(result["probabilities"].values())
    assert result["inference_ms"] > 0


def test_deterministic_and_independent_of_filename(client, image_bytes):
    data = image_bytes()
    first = upload(client, data, "cat.png").json()
    second = upload(client, data, "dog.png").json()
    assert first["probabilities"] == second["probabilities"]


def test_matches_reference_notebook(client):
    """Compare production inference with the original notebook's architecture and preprocessing."""
    sample = Path("/samples/cat.jpg").read_bytes()
    reference = models.resnet18(weights=None)
    reference.conv1.stride = (1, 1)
    reference.maxpool = nn.Identity()
    reference.fc = nn.Linear(reference.fc.in_features, 2)
    reference.load_state_dict(torch.load(os.environ["MODEL_PATH"], map_location="cpu", weights_only=True))
    reference.eval()
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(io.BytesIO(sample)) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        expected = torch.softmax(reference(tensor), dim=1)[0].tolist()
    result = upload(client, sample).json()
    assert result["probabilities"]["cat"] == pytest.approx(expected[0], abs=1e-7)
    assert result["probabilities"]["dog"] == pytest.approx(expected[1], abs=1e-7)


@pytest.mark.parametrize("animal", ["cat", "dog"])
def test_real_sample(client, animal):
    response = upload(client, Path(f"/samples/{animal}.jpg").read_bytes())
    assert response.status_code == 200
    assert response.json()["label"] == animal


def test_rejects_missing_empty_and_corrupt_files(client):
    assert client.post("/api/predict").status_code == 422
    assert upload(client, b"").status_code == 400
    assert upload(client, b"not an image", "cat.jpg", "image/jpeg").status_code == 400


def test_rejects_gif_even_if_claimed_jpeg(client, image_bytes):
    assert upload(client, image_bytes(format="GIF"), "cat.jpg", "image/jpeg").status_code == 400


def test_rejects_large_upload(client, image_bytes, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    assert upload(client, image_bytes()).status_code == 413


def test_rejects_large_dimensions(client, image_bytes, monkeypatch):
    monkeypatch.setattr(model, "MAX_IMAGE_PIXELS", 100)
    assert upload(client, image_bytes()).status_code == 400


def test_exif_rotation(client, image_bytes):
    exif = Image.Exif()
    exif[274] = 6
    response = upload(client, image_bytes(format="JPEG", size=(80, 60), exif=exif))
    assert response.status_code == 200
    assert response.json()["image"] == {"width": 60, "height": 80}


def test_busy_model_returns_retryable_error(client, image_bytes):
    slots = client.app.state.inference_slots
    original = slots._value
    slots._value = 0
    try:
        response = upload(client, image_bytes())
        assert response.status_code == 429
    finally:
        slots._value = original
