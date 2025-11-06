import os
import fitz
import uuid
import sys
from datetime import datetime
from custom_logging import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

class DocumentHandler:
    """
    Handles PDF saving and reading operations
    Automatically logs all actions and support session based organization
    """
    def __init__(self, data_dir = None, session_id = None):
        try:
            
            self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
            self.session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)
            log.info("PDF Handler Initialized", session_id=self.session_id, session_path=self.session_path)
        except Exception as e:
            log.error(f"Error initializing document handler: {e}")
            raise DocumentPortalException(f"Error Initializing document handler : {e}")


    def save_pdf(self, uploaded_file):
        try:
            filename = os.path.basename(uploaded_file.name)

            if not filename.lower().endswith('.pdf'):
                raise DocumentPortalException("Uploaded file is not a PDF. Only PDF files are allowed.")
            
            save_path = os.path.join(self.session_path, filename)
            
            with open(save_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path

        except Exception as e:
            log.error(f"Error saving PDF: {e}", session_id=self.session_id)
            raise DocumentPortalException(f"Error saving PDF: {e}", sys)

    def read_pdf(self, pdf_path:str)->str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text_chunks.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
            text = "\n".join(text_chunks)

            log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error(f"Error reading PDF: {e}", session_id=self.session_id)
            raise DocumentPortalException(f"Error reading PDF: {e}", sys)

if __name__ == "__main__":
    from pathlib import Path   
    
    handler = DocumentHandler(session_id="test_session_001")

    pdf_path = "E:\\LLMOps\\documentreader\\data\\document_analysis\\1706.03762v7.pdf"   

    class DummyFile:
        def __init__(self, file_path):
            self._file_path = file_path
            self.name = Path(file_path).name


        def getbuffer(self):
            return open(self._file_path, "rb").read()
            
    dummy_file = DummyFile(pdf_path)

    try:
        saved_path = handler.save_pdf(dummy_file)
        print(f"PDF saved at: {saved_path}")
        
        content=handler.read_pdf(saved_path)
        print("PDF Content:")
        print(content[:500]) 
    except Exception as e:
        print(f"Error: {e}")