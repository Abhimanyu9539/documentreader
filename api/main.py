from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from typing import Any, List, Optional
from src.document_ingestion.data_ingestion import (
    DocumentComparator, 
    DocHandler, 
    FAISSManager,
    ChatIngestor
)

from src.document_analyser.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparator as DocComparatorLLM
from src.document_chat.retrieval import ConversationalRAG


FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent


app = FastAPI(
    title="Multi-Document Chat API",
    description="API for ingesting, processing, and comparing for chat-based document analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static template
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request}) 

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is healthy"}


class FastAPIFileAdapter:
    """ Adapt FASTAPI UplodFile : .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def _read_pdf_via_handler(handler: DocHandler, file_path: str):
    """Helper function to read PDF using DocHandler."""
    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Read Failed : {str(e)}")

@app.post("/analyze")
async def analyze_documents(file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = _read_pdf_via_handler(dh, saved_path)
        analyser = DocumentAnalyzer()
        analysis_result = analyser.analyze_document(text)
        return JSONResponse(content=analysis_result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis Failed : {str(e)}")
    

@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), 
            FastAPIFileAdapter(actual)
        )

        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comparator = DocComparatorLLM()
        df = comparator.compare_documents(combined_text)
        print("##########", df)
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison Failed : {str(e)}")
    

@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        #log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        wrapped = [FastAPIFileAdapter(f) for f in files]
        # this is my main class for storing a data into VDB
        # created a object of ChatIngestor
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        #log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing Failed : {str(e)}")
    

@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        #log.info(f"Received chat query: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])
        #log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        #log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")