import os
from utils.model_loader import ModelLoader
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser



class DocumentAnalyzer:
    def __init__(self):
        pass

    def analyze_metadata(self):
        pass

    def analyze_document(self):
        pass
