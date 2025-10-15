import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from custom_logging.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

logger = CustomLogger().get_logger(__name__)

class ModelLoader:
    """Loads embedding models and LLMs based on config and environment."""
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        logger.info("Configuration loaded successfully.", config_keys = list(self.config.keys()))

    def _validate_env(self):
        """Validate necessary env variables"""
        required_keys = ["OPENAI_API_KEY"]
        self.api_keys = {key: os.getenv(key) for key in required_keys}

        missing = [k for k, v in self.api_keys.items() if not v]

        if missing:
            logger.error("Missing environment variables", missing_vars = missing)
            raise DocumentPortalException("Missing environmental variables", sys)
        
        logger.info("Environment variables validated", available_keys=[k for k in self.api_keys if self.api_keys[k]])

    def load_embeddings(self):
        """Load and return the embedding model"""
        try:
            logger.info("Loading embedding model ...")
            model_name = self.config['embedding_model']['model_name']
            return OpenAIEmbeddings(model=model_name)
        
        except Exception as e:
            logger.error("Error loading embedding model", erro =str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """Load and return LLM model"""
        llm_block = self.config['llm']
        logger.info("Loading LLM Model")

        # Choose the provider
        # Default groq
        provider_key = os.getenv("LLM_PROVIDER", "openai")

        if provider_key not in llm_block:
            logger.error("LLM Provider not found in config", provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key}' key not found in config.")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("emperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        logger.info("Loading LLM :", 
                    provider=provider,
                    model= model_name, 
                    temperature = temperature, 
                    max_tokens = max_tokens
                    )
        
        if provider == "google":
            llm=ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            return llm

        elif provider == "groq":
            llm=ChatGroq(
                model=model_name,
                api_key=self.api_keys["GROQ_API_KEY"],
                temperature=temperature,
            )
            return llm
            
        elif provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_keys["OPENAI_API_KEY"],
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            logger.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")
        

if __name__ == "__main__":
    loader = ModelLoader()
    
    # Test embedding model loading
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    
    # Test LLM loading based on YAML config
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    
    # Test the ModelLoader
    result=llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")