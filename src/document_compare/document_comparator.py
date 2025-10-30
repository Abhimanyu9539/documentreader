import sys
from dotenv import load_dotenv
import pandas as pd
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import ChangeFormat
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader

class DocumentComparator:
    """Compares two documents using LLMs and provides a detailed comparison."""

    def __init__(self):
        load_dotenv()
        self.logger = CustomLogger().get_logger(__name__)

        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.parser = JsonOutputParser(pydantic_object=ChangeFormat)
            self.fixing_parser = OutputFixingParser.from_llm(self.llm, self.parser)
            self.prompt = PROMPT_REGISTRY["document_comparison"]    
            self.chain = self.prompt | self.llm | self.parser

            self.logger.info("DocumentComparator initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing DocumentComparator: {e}")
            raise DocumentPortalException("Failed to initialize DocumentComparator", sys)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """Compares two documents and returns structured comparison results."""
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }

            self.logger.info("LLM powered document comparison started")
            response = self.chain.invoke(inputs)
            self.logger.info("Document comparison completed successfully")
            return self._format_response(response)
        
        except Exception as e:
            self.logger.error(f"Error comparing documents: {e}")
            raise DocumentPortalException(f"Error comparing documents: {e}", sys)
        

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:
        try:
            print("################", response_parsed)
            df = pd.DataFrame(response_parsed)
            
            self.logger.info("Response formatted into DataFrame successfully")
            return df
        except Exception as e:
            self.logger.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)
