import os
import threading
import uuid
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

APP_TITLE = "LLM Document Search & Analysis System"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "storage/faiss_index")

DEFAULT_PATTERNS = ["**/*.txt", "**/*.md", "**/*.pdf"]

app = FastAPI(title=APP_TITLE)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IngestRequest(BaseModel):
    source_dir: str = Field(..., description="Directory containing documents to ingest.")
    patterns: Optional[List[str]] = Field(
        default=None,
        description="Glob patterns to include. Defaults to txt, md, pdf.",
    )
    persist: bool = Field(default=True, description="Persist FAISS index to disk.")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)


class RagStore:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore: Optional[FAISS] = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    def _vectorstore_exists(self) -> bool:
        faiss_file = os.path.join(self.persist_path, "index.faiss")
        pkl_file = os.path.join(self.persist_path, "index.pkl")
        return os.path.exists(faiss_file) and os.path.exists(pkl_file)

    def load(self) -> None:
        if self._vectorstore_exists():
            self.vectorstore = FAISS.load_local(
                self.persist_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def _load_documents(self, source_dir: str, patterns: List[str]) -> List[Any]:
        documents: List[Any] = []
        for pattern in patterns:
            lower = pattern.lower()
            if ".pdf" in lower:
                loader_cls = PyPDFLoader
                loader_kwargs = {}
            else:
                loader_cls = TextLoader
                loader_kwargs = {"autodetect_encoding": True}
            loader = DirectoryLoader(
                source_dir,
                glob=pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
            )
            documents.extend(loader.load())
        return documents

    def ingest(
        self,
        source_dir: str,
        patterns: Optional[List[str]] = None,
        persist: bool = True,
        progress_cb=None,
    ) -> Dict[str, Any]:
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        use_patterns = patterns or DEFAULT_PATTERNS
        if progress_cb:
            progress_cb({"step": "ingest", "progress": 0, "message": "Scanning documents"})
        docs = self._load_documents(source_dir, use_patterns)
        if not docs:
            if progress_cb:
                progress_cb(
                    {
                        "step": "ingest",
                        "progress": 100,
                        "message": "No documents found.",
                    }
                )
            return {"ingested_chunks": 0, "documents": 0, "message": "No documents found."}

        if progress_cb:
            progress_cb(
                {
                    "step": "ingest",
                    "progress": 10,
                    "message": f"Loaded {len(docs)} documents",
                    "documents": len(docs),
                }
            )

        chunks = []
        total_docs = len(docs)
        for idx, doc in enumerate(docs, start=1):
            chunks.extend(self.splitter.split_documents([doc]))
            if progress_cb:
                progress_cb(
                    {
                        "step": "chunk",
                        "progress": 10 + int((idx / total_docs) * 25),
                        "message": f"Chunking {idx}/{total_docs} documents",
                    }
                )
        ingest_id = str(uuid.uuid4())
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["ingest_id"] = ingest_id

        if progress_cb:
            progress_cb(
                {
                    "step": "embed",
                    "progress": 35,
                    "message": f"Embedding {len(chunks)} chunks",
                    "chunks": len(chunks),
                }
            )

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))
        total_chunks = len(texts)
        embedded = 0

        for start in range(0, total_chunks, batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_metas = metadatas[start : start + batch_size]
            embeddings = self.embeddings.embed_documents(batch_texts)
            text_embeddings = list(zip(batch_texts, embeddings))

            if self.vectorstore is None and embedded == 0:
                self.vectorstore = FAISS.from_embeddings(
                    text_embeddings,
                    self.embeddings,
                    metadatas=batch_metas,
                )
            else:
                self.vectorstore.add_embeddings(text_embeddings, metadatas=batch_metas)

            embedded += len(batch_texts)
            if progress_cb:
                progress_cb(
                    {
                        "step": "embed",
                        "progress": 35 + int((embedded / total_chunks) * 55),
                        "message": f"Embedded {embedded}/{total_chunks} chunks",
                    }
                )

        if persist:
            os.makedirs(self.persist_path, exist_ok=True)
            self.vectorstore.save_local(self.persist_path)
            if progress_cb:
                progress_cb({"step": "index", "progress": 95, "message": "Persisted index"})

        if progress_cb:
            progress_cb({"step": "index", "progress": 100, "message": "Ingestion complete"})

        return {
            "ingested_chunks": len(chunks),
            "documents": len(docs),
            "ingest_id": ingest_id,
        }

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            raise ValueError("Vector store is empty. Ingest documents first.")
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        payload = []
        for doc, score in results:
            payload.append(
                {
                    "score": float(score),
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return payload

    def build_context(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        used_chars = 0
        for idx, item in enumerate(results, start=1):
            text = item["text"].strip()
            meta = item.get("metadata", {})
            source_label = meta.get("source", "unknown")
            chunk_id = meta.get("chunk_id", "?")
            header = f"[Source {idx}] {source_label} (chunk {chunk_id})"
            block = f"{header}\n{text}"
            if used_chars + len(block) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(block)
            used_chars += len(block)
            sources.append(
                {
                    "source": source_label,
                    "chunk_id": chunk_id,
                    "score": item.get("score"),
                }
            )
        return {"context": "\n\n---\n\n".join(context_parts), "sources": sources}

    def answer(self, query: str, top_k: int) -> Dict[str, Any]:
        results = self.search(query, top_k)
        built = self.build_context(results)
        context = built["context"]

        if not context:
            return {
                "answer": "I don't have enough information in the documents to answer that.",
                "sources": [],
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a document search assistant. Use only the provided context. "
                    "If the answer is not in the context, say you do not have enough information.",
                ),
                (
                    "user",
                    "Question: {question}\n\nContext:\n{context}",
                ),
            ]
        )

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)
        response = llm.invoke(prompt.format_messages(question=query, context=context))

        return {
            "answer": response.content,
            "sources": built["sources"],
        }


store = RagStore(VECTORSTORE_PATH)
store.load()

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _init_job(job_id: str, payload: Dict[str, Any]) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "step": "ingest",
            "progress": 0,
            "message": "Queued",
            "payload": payload,
            "result": None,
            "error": None,
        }


def _update_job(job_id: str, **kwargs: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/api")
def api_root() -> Dict[str, str]:
    return {"name": APP_TITLE, "status": "ready"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    if store.vectorstore is None:
        return {"indexed_vectors": 0, "persist_path": VECTORSTORE_PATH, "loaded": False}
    return {
        "indexed_vectors": int(store.vectorstore.index.ntotal),
        "persist_path": VECTORSTORE_PATH,
        "loaded": True,
    }


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    _init_job(
        job_id,
        {
            "source_dir": req.source_dir,
            "patterns": req.patterns,
            "persist": req.persist,
        },
    )

    def _progress_update(payload: Dict[str, Any]) -> None:
        _update_job(
            job_id,
            status="running",
            step=payload.get("step", "ingest"),
            progress=payload.get("progress", 0),
            message=payload.get("message"),
            documents=payload.get("documents"),
            chunks=payload.get("chunks"),
        )

    def _run_ingest() -> None:
        try:
            result = store.ingest(
                req.source_dir,
                req.patterns,
                req.persist,
                progress_cb=_progress_update,
            )
            _update_job(
                job_id,
                status="completed",
                step="index",
                progress=100,
                message="Ingestion complete",
                result=result,
            )
        except FileNotFoundError as exc:
            _update_job(job_id, status="error", step="ingest", error=str(exc), message=str(exc))
        except Exception as exc:
            _update_job(job_id, status="error", step="ingest", error=str(exc), message=str(exc))

    background_tasks.add_task(_run_ingest)
    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    try:
        results = store.search(req.query, req.top_k)
        return {"query": req.query, "results": results, "count": len(results)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/answer")
def answer(req: AnswerRequest) -> Dict[str, Any]:
    try:
        return store.answer(req.query, req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import webbrowser

    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    auto_open = os.getenv("AUTO_OPEN_BROWSER", "1") == "1"

    if auto_open:
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass

    uvicorn.run("main:app", host=host, port=port)
