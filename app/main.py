from fastapi import FastAPI, status
from contextlib import asynccontextmanager
from app.inference import load_model

app = FastAPI()

@app.get('/health', status_code=status.HTTP_200_OK)
def get_status():
    return {'status': 'ok'}


@asynccontextmanager
async def lifespan()