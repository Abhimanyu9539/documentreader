import os
import sys
import json
import uuid
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Iterable, List, Optional, Any
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

class FAISSManager:
    def  init__(self):
        pass

    def _exists(self):
        pass

    @staticmethod
    def _fingerprint():
        pass

    def _save_metadata(self):
        pass
    
    def add_documents(self):
        pass

    def load_or_create(self):
        pass


class DocHandler:
    def init__(self):
        pass

    def save_pdf(self):
        pass

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