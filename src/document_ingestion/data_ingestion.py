
from __future__ import annotations
import os
import sys
import json
import uuid
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Iterable, List, Optional, Any, Dict



import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

from utils.file_io import save_uploaded_files, generate_session_id
from utils.document_ops import load_documents, concat_for_analysis

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}


## FAISS Manager to handle vector store operations 
class FAISSManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta : Dict[str, Any] = {"rows": {}}


        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        self.model_loader = model_loader or ModelLoader()
        self.embedder = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md:Dict[str, Any]) -> str:
        src = md.get("source")  or md.get("file_name")
        rid = md.get("row_id") 
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()  
        

    def _save_metadata(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def add_documents(self, docs : List[Document]):
        if self.vs is None:
            raise RuntimeError("call load_or_create() before adding documents")

        new_docs: List[Document] = []

        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue

            self._meta["rows"][key] = True
            new_docs.append(d)


        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_metadata()
        
        return len(new_docs)

    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.embedder,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        
        
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        
        self.vs = FAISS.from_texts(texts=texts, embedding=self.embedder, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs

class DocHandler:
    """"pdf SAVE + READ handler for page analysis"""
    def __init__(self, data_dir: Optional[str]= None, session_id: Optional[str]= None):
        self.logger = CustomLogger().get_logger(__name__)
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis")) 
        self.session_id = session_id or generate_session_id("session") 
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        self.logger.info("Doc Handler initialized.", session_id=self.session_id, session_path=self.session_path)
        

    def save_pdf(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid File Type. Only pdfs are allowed")
            
            save_path = os.path.join(self.session_path, filename)

            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())

            self.logger.info("PDF File saved successfully", file=filename, save_path=save_path, session_id = self.session_id)
            return save_path
        except Exception as e:
            self.logger.error("Failed to save pdf", error = str(e), session_id = self.session_id)
            raise DocumentPortalException(f"Failed to save pdf: {str(e)}", e ) from e
    
    
    def read_pdf(self, pdf_path:str) -> str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n=== Page {page_num + 1} ---\n {page.get_text()}")

                text = "\n".join(text_chunks)
                self.logger.info("PDF read successfully", pdf_path=pdf_path, session_id = self.session_id, pages=len(text_chunks))
                return text
        except Exception as e:
            self.logger.error("Failed to read pdf", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process the pdf:{pdf_path}", e ) from e
        
        

class DocumentComparator:
    def __init__(self):
        pass

    def save_uploaded_files(self):
        pass

    def read_pdf(self):
        pass

    def combine_documents(self):
        pass

    def clean_old_sessions(self):
        pass



class ChatIngestor:
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()
            self.logger = CustomLogger().get_logger(__name__)
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            self.logger.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            self.logger.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        self.logger.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            ## FAISS manager very very important class for the docchat
            fm = FAISSManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            self.logger.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            self.logger.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e
