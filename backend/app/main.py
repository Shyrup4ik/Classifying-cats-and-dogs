import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .model import Classifier, InvalidImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights_path = Path(os.getenv("MODEL_PATH", "/models/cat_dog_fixed.pt")).expanduser()
    app.state.classifier = await run_in_threadpool(Classifier, weights_path)
    app.state.inference_slots = asyncio.Semaphore(2)
    yield


app = FastAPI(
    title="Лапа · Cat & Dog Classifier",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Добавьте фото в поле file."})


@app.get("/api/health")
async def health(request: Request):
    return {
        "status": "ready",
        "model": "ResNet18",
        "device": "cpu",
        "weights_id": request.app.state.classifier.weights_id,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_image_pixels": 20_000_000,
    }


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Файл слишком большой: максимум 10 МБ.")
    if not data:
        raise HTTPException(400, "Файл пустой. Выберите другое фото.")
    slots = request.app.state.inference_slots
    try:
        await asyncio.wait_for(slots.acquire(), timeout=5)
    except asyncio.TimeoutError:
        raise HTTPException(429, "Модель сейчас занята. Попробуйте через несколько секунд.")
    try:
        return await run_in_threadpool(request.app.state.classifier.predict, data)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(500, "Не удалось распознать фото. Попробуйте ещё раз.")
    finally:
        slots.release()
