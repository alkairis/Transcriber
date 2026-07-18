from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from src.config import get_settings
from src.jobs import JobStore
from src.transcription import Transcriber

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    transcriber = Transcriber(settings)
    transcriber.load() 
    app.state.settings = settings
    app.state.transcriber = transcriber
    app.state.jobs = JobStore()
    yield
    
    
app = FastAPI(title="Audio Transcript", lifespan=lifespan)

@app.get("/")
async def health():
    return {"status": "ok"}