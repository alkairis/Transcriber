from contextlib import asynccontextmanager
import os

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from src.config import get_settings
from src.jobs import JobStore
from src.rag import prepare
from src.schemas import AskRequest, AskResponse, JobStatus, SummarizeRequest, SummarizeResponse, TranscriptResponse
from src.transcription import Transcriber
from src.embed import build_index
from src.summarize import summarize

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

def _process(job_id: str, path: str, state):
    settings, transcriber, jobs = state.settings, state.transcriber, state.jobs
    
    try:
        jobs.update(job_id, status="transcribing")
        segments, text, duration, language = transcriber.transcribe(path)
        jobs.update(
            job_id, segments=segments, text=text,
            duration_sec=duration,
            language=language,
            status="indexing" if (settings.enable_rag and segments) else "done"
        )
        
        if settings.enable_rag and segments:
            jobs.update(job_id, vectorstore=build_index(segments, settings), status="done")
    except Exception as ex:
        jobs.update(job_id, status="error", error=str(ex))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        
@app.post("/transcribe", response_model=JobStatus)
async def transcribe(request:Request, background: BackgroundTasks, file: UploadFile=File(...)):
    settings, transcriber, jobs = request.app.state.settings, request.app.state.transcriber, request.app.state.jobs
    job = jobs.create(filename=file.filename)
    path = os.path.join(settings.upload_dir, f"{job.job_id}_{file.filename}")
    
    max_bytes  = settings.max_upload_mb * 1024 * 1024
    size=0
    
    with open(path, "wb") as out:
        while chunk := await file.read(1024*1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                os.remove(path)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
            out.write(chunk)
            
    try:
        duration = transcriber.probe_duration(path)
    except Exception as ex:
        os.remove(path)
        raise HTTPException(400, f"Unreadable audio: {ex}")
    if duration > settings.max_audio_seconds:
        os.remove(path)
        raise HTTPException(
            413,f"Audio is {duration / 60:.1f} min; limit is {settings.max_audio_seconds / 60:.0f} min",
        )
        
    jobs.update(job.job_id, duration_sec=duration, status="queued")
    background.add_task(_process, job.job_id, path, request.app.state)
    return JobStatus(job_id=job.job_id, status="queued", duration_sec=duration, filename=file.filename)


@app.get("/jobs/{job_id}", response_model=TranscriptResponse)
async def get_job(job_id:str, request:Request):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    done = job.status== "done"
    return TranscriptResponse(
        job_id=job_id, status=job.status, error=job.error,
        duration_sec=job.duration_sec, language=job.language,
        filename=job.filename,
        text=job.text if done else None,
        segments=job.segments if done else None
    )
    
def _require_ready(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == "error":
        raise HTTPException(409, f"Job failed: {job.error}")
    if job.status != "done" or not job.text:
        raise HTTPException(409, f"Job not ready (status={job.status})")
    return job

@app.post("/summarize/{job_id}", response_model=SummarizeResponse)
def summarize_job(job_id: str, req: SummarizeRequest, request: Request):
    job = _require_ready(request, job_id)
    settings = request.app.state.settings
    summary, method = summarize(job.text, req.style, req.instructions, settings)
    return SummarizeResponse(job_id=job_id, summary=summary, method=method)
 
@app.post("/ask/{job_id}", response_model=AskResponse)
def ask_job(job_id:str, req:AskRequest, request:Request):
    job = _require_ready(request, job_id)
    settings = request.app.state.settings
    
    if req.k:
        settings = settings.model_copy(update={"retriever_top_k":req.k})
    
    chain, inputs, citations, mode = prepare(req.question, job, settings)
    answer = chain.invoke(inputs).content
    return AskResponse(job_id=job_id, answer=answer, mode=mode, citations=citations)

@app.post("/ask/{job_id}/stream")
async def ask_job_stream(job_id: str, req: AskRequest, request: Request):
    job = _require_ready(request, job_id)
    settings = request.app.state.settings
    chain, inputs, _citations, _mode = prepare(req.question, job, settings)

    async def token_stream():
        async for part in chain.astream(inputs):
            if part.content:
                yield f"data: {part.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
