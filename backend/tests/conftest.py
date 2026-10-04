import io

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def image_bytes():
    def make(format="PNG", mode="RGB", size=(80, 60), exif=None):
        image = Image.new(mode, size, color=100 if mode == "L" else (120, 90, 65))
        buffer = io.BytesIO()
        kwargs = {"exif": exif} if exif is not None else {}
        image.save(buffer, format=format, **kwargs)
        return buffer.getvalue()
    return make
