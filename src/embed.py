from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from .config import Settings
from .schemas import Segment

def build_index(segments: list[Segment], settings: Settings):
    docs = _segment_chunks(segments, settings)
    embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
    return FAISS.from_documents(docs, embeddings)

def _segment_chunks(segments: list[Segment], settings: Settings) -> list[Document]:
    docs: list[Document] = []
    buf: list[Segment] = []
    buf_len = 0
    for seg in segments:
        buf.append(seg)
        buf_len += len(seg.text)
        if buf_len >= settings.chunk_target_chars:
            docs.append(_flush(buf))
            buf, buf_len = _overlap_tail(buf, settings.chunk_overlap_chars)
    if buf:
        docs.append(_flush(buf))
    return docs

def _flush(buf: list[Segment]) -> Document:
    return Document(
        page_content=" ".join(s.text for s in buf),
        metadata={"start": buf[0].start, "end": buf[-1].end},
    )
 
def _overlap_tail(buf: list[Segment], overlap_chars: int) -> tuple[list[Segment], int]:
    tail: list[Segment] = []
    total = 0
    for seg in reversed(buf):
        tail.insert(0, seg)
        total += len(seg.text)
        if total >= overlap_chars:
            break
    return tail, total