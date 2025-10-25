import os
import sys
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.model_loader import ModelLoader
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
import streamlit as st


class ConversationalRAG():
    def __init__(self, session_id: str, retriever) -> None:
        try:
            self.logger = CustomLogger().get_logger()
            self.model_loader = ModelLoader()
            self.session_id = session_id
            self.retriever = retriever
            self.llm = self.model_loader.load_llm()
            self.context_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, 
                self.retriever, 
                self.context_prompt
            )

            self.logger.info("ConversationalRAG initialized successfully", session_id = session_id)

            self.qa_chain = create_stuff_documents_chain(
                self.llm,
                self.qa_prompt
            )

            self.rag_chain = create_retrieval_chain(
                self.history_aware_retriever,
                self.qa_chain
            )

            self.logger.info("RAG Chain created successfully", session_id = session_id)

            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key= "input", 
                history_messages_key= "chat_history", 
                output_messages_key= "output"
            )

            self.logger.info("RunnableWithMessageHistory created successfully", session_id = session_id)

        except Exception as e:
            self.logger.error("Error initializing ConversationalRAG", error = str(e), session_id = session_id)
            raise DocumentPortalException("Initialization Error in ConversationalRAG", sys)


    def _load_llm(self):
        try:
            llm = self.model_loader.load_llm()
            self.logger.info("LLM loaded successfully", session_id = self.session_id)
            return llm
        except Exception as e:
            self.logger.error(f"Error loading LLM: {e}")
            raise DocumentPortalException("Error loading LLM", sys)
        

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            if "store" not in st.session_state:
                st.session_state.store = {}

            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
                self.logger.info("New chat session history created", session_id=session_id)

            return st.session_state.store[session_id]
        except Exception as e:
            self.logger.error("Failed to access session history", session_id=session_id, error=str(e))
            raise DocumentPortalException("Failed to retrieve session history", sys)


    def load_retriever_from_faiss(self, index_path: str):
        try:
            embeddings = self.model_loader.load_embeddings()

            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found at {index_path}")
            
            vector_store = FAISS.load_local(index_path, embeddings)
            self.retriever = vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
            self.logger.info("FAISS retriever loaded successfully", index_path=index_path)
            
            return self.retriever

        except Exception as e:
            self.logger.error("Error loading FAISS retriever", error = str(e))
            raise DocumentPortalException("Error loading FAISS retriever", sys)
        

    def invoke(self, user_input: str) -> str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            answer = response.get("answer", "No answer.")

            if not answer:
                self.logger.warning("Empty answer received", session_id=self.session_id)

            self.logger.info("Answer generated successfully", session_id=self.session_id, user_input=user_input, answer=answer[:111])

            return answer
        
        except Exception as e:
            self.logger.error("Error invoking ConversationalRAG", error = str(e))
            raise DocumentPortalException("Error invoking ConversationalRAG", sys)