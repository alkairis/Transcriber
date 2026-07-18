# Audio -> Insight

Upload audio (<= 40 min) -> faster-whisper transcript -> Ollama (Llama 3.1/3.2)
summarization and RAG-based Q&A. Local-first, no external API keys.

## Prerequisites

    # 1. Ollama running with the two models pulled
    ollama serve            
    ollama pull llama3.1
    ollama pull nomic-embed-text


## Run

    cp .env.example .env    
    uvicorn app.main:app --reload

First request downloads the faster-whisper model once; it is then held in memory.

## Flow

    ## API Reference

### POST /transcribe

Upload an audio file to start transcription.

Request:

    curl -F "file=@meeting.mp3" http://localhost:8000/transcribe

Response:

    {
      "job_id": "...",
      "status": "pending"
    }

### GET /jobs/{job_id}

Check job status and retrieve transcript metadata.

Request:

    curl http://localhost:8000/jobs/<job_id>

Response:

    {
      "job_id": "...",
      "status": "done",
      "transcript": "...",
      "segments": [ ... ]
    }

### POST /summarize/{job_id}

Summarize a completed transcript.

Request:

    curl -X POST http://localhost:8000/summarize/<job_id> \
      -H "Content-Type: application/json" \
      -d '{"style":"action_items"}'

Response:

    {
      "summary": "...",
      "style": "action_items"
    }

### POST /ask/{job_id}

Ask a question about transcript content.

Request:

    curl -X POST http://localhost:8000/ask/<job_id> \
      -H "Content-Type: application/json" \
      -d '{"question":"What did they decide about the deadline?"}'

Response:

    {
      "answer": "...",
      "citations": [ ... ]
    }

### POST /ask/{job_id}/stream

Stream an answer using Server-Sent Events.

Request:

    curl -N -X POST http://localhost:8000/ask/<job_id>/stream \
      -H "Content-Type: application/json" \
      -d '{"question":"Summarize the risks discussed."}'

## Design notes

- **40-min cap** is gated from container metadata (`probe_duration`) *before*
  any transcription runs — a 3-hour file is rejected in milliseconds.
- **Whisper model loads once** at startup (lifespan), never per request.
- **Blocking work stays off the event loop**: transcription runs in a
  background thread; summarize/ask are sync `def` (threadpool).
- **Chunking follows Whisper segment boundaries** and keeps timestamps as
  metadata, so RAG answers can cite [mm:ss].
- **`num_ctx=16384`** overrides Ollama's ~2k default so long transcripts
  aren't silently truncated.

## Scaling past v1

The in-memory `JobStore` + `BackgroundTasks` are single-process. For multiple
workers, retries, or persistence across restarts, move `_process` to an
arq/Celery worker backed by Redis and store jobs + FAISS indexes externally.