import sys
import os
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate

from utils.model_loader import ModelLoader
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

class ConversationalRAG:
    def __init__(self, session_id: str, retriever= None):
        try:
            self.logger = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            if retriever is None:
                raise ValueError("Retriever must be provided for ConversationalRAG initialization.")
            self.retriever = retriever
            self.chain = self._build_lcel_chain()
            self.logger.info("Conversational RAG initialized", session_id=self.session_id)

        except Exception as e:
            self.logger.error("Initialization error in Conversational RAG", error=str(e))
            raise DocumentPortalException("Initialization error in Conversational RAG", sys)
        
    def load_retriever_from_faiss(self, index_path: str) -> FAISS:
        """Load FAISS retriever from saved index."""
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index path does not exist: {index_path}")
            
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info("FAISS retriever loaded successfully", index_path=index_path)
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
                )
            self.logger.info("Retriever created from FAISS vectorstore", index_path=index_path)

            self._build_lcel_chain()
            return self.retriever
        except Exception as e:
            self.logger.error("Error loading FAISS retriever", error=str(e))
            raise DocumentPortalException("Error loading FAISS retriever", sys)

    def invoke(self):
        try:
            pass
        except Exception as e:
            self.logger.error("Error during RAG invocation", error=str(e))
            raise DocumentPortalException("Error during RAG invocation", sys)

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded.")
            self.logger.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            self.logger.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Error loading LLM", sys)
    
    @staticmethod
    def _format_document(docs):
        return "/n/n".join(d.page_content for d in docs)

    def _build_lcel_chain(self):
        try:
            
            question_rewriter = (
                {
                    "input": itemgetter("input"), 
                    "chat_history": itemgetter("chat_history")
                }
                |
                self.contextualize_prompt
                | 
                self.llm
                |
                StrOutputParser()
            )


            retrieve_docs = question_rewriter | self.retriever | self._format_document

            self.chain = (
                {
                    "context" : retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }
                |
                self.qa_prompt
                |
                self.llm
                |
                StrOutputParser()
            )
            self.logger.info("LCEL chain built successfully", session_id=self.session_id)
            return self.chain
        
        except Exception as e:
            self.logger.error("Error building LCEL chain", error=str(e))
            raise DocumentPortalException("Error building LCEL chain", sys)
