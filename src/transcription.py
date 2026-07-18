import av
from faster_whisper import WhisperModel
from .schemas import Segment
from config import Settings


class Transcriber:
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model : WhisperModel | None = None
        
    def load(self):
        s = self._settings
        self._model = WhisperModel(
            s.whisper_model_size,
            device=s.whisper_device,
            compute_type=s.whisper_compute_type,
        )
        
    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            raise RuntimeError("Transcriber model is not loaded. Call load() first.")
        return self._model
    
    def probe_duration(self, path: str):
        with av.open(path) as container:
            if container.duration is not None:
                return float(container.duration) / av.time_base
            stream = next((st for st in container.streams if st.type == "audio"), None)
            
            if stream is not None and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
        raise ValueError("Could not determine duration of the audio file.")
    
    def transcribe(self, path: str):
        s = self._settings
        segments_iter, info = self.model.transcribe(
            path,
            beam_size=s.whisper_beam_size,
            vad_filter=s.vad_filter
        )
        
        segments: list[Segment] = []
        parts: list[str] = []
        
        for seg in segments_iter:
            clean = seg.text.strip()
            segments.append(Segment(start=round(seg.start, 2), end=round(seg.end, 2), text=clean))
            parts.append(clean)
        return segments, " ".join(parts), float(info.duration), info.language