import sys
from pathlib import Path
import fitz
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class DocumentIngestion:
    """
    Handles saving, reading, and combining of PDFs for comparison with session-based versioning.
    """

    def __init__(self, base_dir: str = "data/document_compare"):
        self.logger = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)

    def delete_existing_file(self):
        """Deletes existing files in the upload directory."""
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.logger.info(f"Deleted existing file: {file}")
                self.logger.info("Directory cleaned successfully", keys=["directory", str(self.base_dir)])

        except Exception as e:
            self.logger.error(f"Error deleting existing file: {e}")
            raise DocumentPortalException(f"Error deleting existing file: {e}", sys)


    def save_uploaded_file(self, reference_file, actual_file):
        """Saves the uploaded PDF to the designated directory."""
        try:
            #self.delete_existing_file()
            self.logger.info("Deleted existing files successfully")

            ref_path = self.base_dir / "reference.pdf"
            act_path = self.base_dir / "actual.pdf"


            if not reference_file.name.endswith('.pdf') or not actual_file.name.endswith('.pdf'):
                raise ValueError("Only PDF files are supported.")

            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer()) 

            self.logger.info(f"Saved uploaded files: reference: {ref_path}, actual: {act_path}")
            return ref_path, act_path

        except Exception as e:
            self.logger.error(f"Error saving uploaded file: {e}")
            raise DocumentPortalException(f"Error saving uploaded file: {e}", sys)

    def read_pdf(self, pdf_path: Path) -> str:
        """Reads a PDF and extracts text from each page."""
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError("The PDF is encrypted and cannot be processed.")
                
                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()

                    if text.strip():  # Only add non-empty text
                        all_text.append(f"\n --- Page {page_num + 1} --- \n {text}")
                self.logger.info(f"Extracted text from {pdf_path.name}, total pages: {doc.page_count}") 
                return "\n".join(all_text)
                          
        except Exception as e:
            self.logger.error(f"Error reading PDF: {e}")
            raise DocumentPortalException(f"Error reading PDF: {e}", sys)


    def combine_documents(self)->str:
            try:
                content_dict = {}
                doc_parts = []

                for filename in sorted(self.base_dir.iterdir()):
                    if filename.is_file() and filename.suffix == ".pdf":
                        content_dict[filename.name] = self.read_pdf(filename)

                for filename, content in content_dict.items():
                    doc_parts.append(f"Document: {filename}\n{content}")

                combined_text = "\n\n".join(doc_parts)
                self.logger.info("Documents combined", count=len(doc_parts))
                return combined_text

            except Exception as e:
                self.logger.error(f"Error combining documents: {e}")
                raise DocumentPortalException("An error occurred while combining documents.", sys)