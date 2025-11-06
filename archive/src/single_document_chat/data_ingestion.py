import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from custom_logging import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from datetime import timezone, datetime

class SingleDocIngestor():
    def __init__(self, data_dir:str = "data/single_document_chat", faiss_dir:str = "faiss_index"):
        try:
            log = CustomLogger().get_logger()

            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            
            self.model_loader = ModelLoader()
            
            log.info("SingleDocIngestor initialized successfully", temp_path=str(self.data_dir), faiss_path=str(self.faiss_dir))

        except Exception as e:
            log.error("Error initializing SingleDocIngestor", error = str(e))
            raise DocumentPortalException("Initialization Error in SingleDocIngestor", sys)
    
    def ingest_files(self, uploaded_files):
        try:
            document_files  = []
            for uploaded_file in uploaded_files:
                unique_filename =  f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                temp_path = self.data_dir / unique_filename


                with temp_path.open("wb") as f:
                    f.write(uploaded_file.read())
                log.info("File saved for ingestion", file=str(temp_path))
                

                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                document_files.extend(docs)
            log.info("File loaded", file=str(temp_path), pages=len(docs))
            return self._create_retriever(document_files)

        except Exception as e:
            log.error("Error in ingest_files", error = str(e))
            raise DocumentPortalException("Error during file ingestion", sys)
        
    def _create_retriever(self, document_files):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            texts = text_splitter.split_documents(document_files)
            log.info("Documents split into chunks", chunks=len(texts))

            embeddings = self.model_loader.load_embeddings()

            vector_store_path = self.faiss_dir / "faiss_index"

            if vector_store_path.exists():
                vector_store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
                log.info("Existing FAISS index loaded", faiss_path=str(vector_store_path))
            else:
                vector_store = FAISS.from_documents(documents=texts, embedding=embeddings)
                vector_store.save_local(str(vector_store_path))
                log.info("New FAISS index created and saved", faiss_path=str(vector_store_path))

            return vector_store.as_retriever()

        except Exception as e:
            log.error("Retriever creation failed", error = str(e))
            raise DocumentPortalException("Error creating retriever", sys)
