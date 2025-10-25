import sys
from pathlib import Path
import uuid
from datetime import datetime, timezone
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader


class DocumentIngestor:
    SUPPORTED_FILE_TYPES = ['.pdf', '.docx', '.txt', '.md']

    def __init__(self, 
                temp_dir="../data/multi_document_chat", 
                faiss_dir : str = "faiss_index",
                session_id: str | None= None):
        try:
            self.logger = CustomLogger().get_logger(__name__)

            # Base Directories
            self.temp_dir = Path(temp_dir)
            self.faiss_dir = Path(faiss_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            # Session ID
            self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_temp_dir = self.temp_dir / self.session_id
            self.session_faiss_dir = self.faiss_dir / self.session_id
            self.session_temp_dir.mkdir(parents=True, exist_ok=True)
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()
            self.logger.info("DocumentIngestor initialized", 
                             session_id=self.session_id, 
                             temp_path=str(self.session_temp_dir), 
                             faiss_path=str(self.session_faiss_dir))

        except Exception as e:
            self.logger.error("Initialization error in Document ingestor", error = str(e))
            raise DocumentPortalException("Initialization error in Document ingestor", sys)

    def ingest_files(self, uploaded_files):
        try:
            documents = []

            for uploaded_file in uploaded_files:
                ext = Path(uploaded_file.name).suffix.lower()
                if ext not in self.SUPPORTED_FILE_TYPES:
                    self.logger.warning("Unsupported file type skipped", file=uploaded_file.name)
                    continue
                
                unique_filename = f"{uuid.uuid4().hex[:8]}{ext}"
                temp_path = self.session_temp_dir / unique_filename

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                self.logger.info("File saved for ingestion", file=str(temp_path))

                if ext == ".pdf":
                    loader = PyPDFLoader(str(temp_path))
                elif ext == ".docx":
                    loader = Docx2txtLoader(str(temp_path))
                elif ext == ".txt":
                    loader = TextLoader(str(temp_path), encoding="utf-8")
                else:
                    self.logger.warning("No loader available for file type", file=uploaded_file.name)
                    continue

                docs = loader.load()
                documents.extend(docs)

            if not documents:
                raise DocumentPortalException("No valid documents loaded for ingestion", sys)

            self.logger.info("All Documents Loaded", total_documents=len(documents), session_id=self.session_id)
            return self._create_retriever(documents)
                
        except Exception as e:
            self.logger.error("Error during file ingestion", error = str(e))
            raise DocumentPortalException("Error during file ingestion", sys)

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000, 
                chunk_overlap = 300
            )
            chunks = splitter.split_documents(documents=documents)
            self.logger.info("Document split into chunks", 
                             total_chunks = len(chunks), 
                             session_id = self.session_id)
            
            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

            # Save FAISS index under session folder 
            vectorstore.save_local(str(self.session_faiss_dir))
            self.logger.info("FAISS index created and saved", 
                             faiss_path=str(self.session_faiss_dir), 
                             session_id=self.session_id)
             
            retriever =  vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}
            )

            self.logger.info("Retriever created", session_id=self.session_id)
            return retriever
        
        except Exception as e:
            self.logger.error("Error creating retrieval", error = str(e))
            raise DocumentPortalException("Error creating retrieval", sys)