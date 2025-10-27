import os
import sys
import json
import uuid
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Iterable, List, Optional, Any, Dict
from __future__ import annotations


import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

#from utils.file_io import session_id, save_uploaded_file
#from utils.document_ops import load_documents, concat_for_analysis, concact_for_comparison

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

    def load_or_create(self):
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings= self.embedder,
                allow_dangerous_deserialization=True
            )

            return self.vs


class DocHandler:
    """"pdf SAVE + READ handler for page analysis"""
    def __init__(self, data_dir: Optional[str]= None, session_id: Optional[str]= None):
        self.logger = CustomLogger.get_logger(__name__)
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis")) 
        self.session_id = session_id #or _session_id("session")
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
                    f.write(uploaded_file.get_buffer())

            self.logger.info("PDF File saved successfully", file=filename, save_path=save_path, session_id = self.session_id)
            return save_path
        except Exception as e:
            self.logger.error("Failed to save pdf", error = str(e), session_id = self.session_id)
            raise DocumentPortalException(f"Failed to save pdf: {str(e)}", e ) from e
    def load_pdf(self):
        pass

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
    def __init__(self):
        pass
    
    def _resolve_dir(self):
        pass

    def _split(self):
        pass

    def _build_retriever(self):
        pass