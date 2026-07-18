import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional
from .schemas import Segment

@dataclass
class Job:
    job_id: str
    status: str = "queued"
    filename: Optional[str] = None
    error: Optional[str] = None
    duration_sec: Optional[float] = None
    language: Optional[str] = None
    text: Optional[str] = None
    segments: list[Segment] = field(default_factory=list)
    vectorstore: object = None
    
class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        
    def create(self, filename:str):
        job = Job(job_id = uuid.uuid4().hex, filename=filename)
        with self._lock:
            self._jobs[job.job_id] = job
        return job
    
    def get(self, job_id:str):
        with self._lock:
            return self._jobs.get(job_id)
        
    def update(self, job_id:str, **kwargs):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)