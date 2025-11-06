import sys
from utils.model_loader import ModelLoader
from custom_logging import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from model.models import Metadata
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY



class DocumentAnalyzer:
    """Analyzes documents using LLMs and provides insights."""

    def __init__(self):
        

        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            # Parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(self.llm, self.parser) 
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            log.info("DocumentAnalyzer initialized successfully")


        except Exception as e:
            log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException(f"Error initializing DocumentAnalyzer: {e}", sys)

    def analyze_metadata(self):
        pass

    def analyze_document(self, document_text: str) -> dict:
        """Analyzes the document and returns structured metadata."""
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            log.info("LLM powered document analysis started")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document": document_text
            })
            log.info("Document analysis completed successfully", keys = list(response.keys()))
            
            return response
        
        except Exception as e:
            log.error(f"Error analyzing document: {e}")
            raise DocumentPortalException(f"Error analyzing document: {e}", sys)