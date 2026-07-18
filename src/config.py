from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    vad_filter: bool = True
    
    max_audio_seconds: int = 40*60
    max_upload_mb: int = 40
    
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 16384
    temperature: float = 0.2
    
    enable_rag: bool = True
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 150
    retriever_top_k: int = 4
    
    upload_dir: str = "/tmp/uploads"
    
@lru_cache
def get_settings() -> Settings:
    return Settings()