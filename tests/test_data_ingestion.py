# tests/test_data_ingestion.py

import pytest
import os
import sys
from unittest.mock import MagicMock

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.schema import Document
from src.document_ingestion.data_ingestion import ChatIngestor

# This fixture automatically mocks the ModelLoader for all tests in this file.
# This makes our tests true unit tests by isolating them from the actual model loading logic.
@pytest.fixture(autouse=True)
def mock_model_loader(monkeypatch):
    """Mocks the ModelLoader to prevent it from loading real models during tests."""
    # Create a mock object that simulates the ModelLoader
    mock = MagicMock()
    # Ensure the mock's load_embeddings method can be called and returns a dummy value
    mock.load_embeddings.return_value = None 
    
    # Use monkeypatch to replace the actual ModelLoader class with our mock
    # Whenever ChatIngestor tries to create a ModelLoader, it will get our mock instead.
    monkeypatch.setattr('src.document_ingestion.data_ingestion.ModelLoader', lambda: mock)

def test_chatingestor_initialization(tmp_path):
    """Tests that ChatIngestor initializes correctly and creates session directories."""
    # Arrange: Define temporary directories for testing
    temp_dir = tmp_path / "temp"
    faiss_dir = tmp_path / "faiss"
    
    # Act: Create an instance of the ChatIngestor
    ingestor = ChatIngestor(
        temp_base=str(temp_dir),
        faiss_base=str(faiss_dir),
        session_id="test_session_123"
    )
    
    # Assert: Check that the correct session-specific directories were created
    assert ingestor.session_id == "test_session_123"
    # Verify that the session directory exists within the base temporary directory
    assert os.path.isdir(temp_dir / "test_session_123")
    # Verify that the session directory exists within the base FAISS directory
    assert os.path.isdir(faiss_dir / "test_session_123")
    # Check that the ingestor's paths point to the correct session directories
    assert ingestor.temp_dir == temp_dir / "test_session_123"
    assert ingestor.faiss_dir == faiss_dir / "test_session_123"

def test_document_splitting_logic():
    """Tests the internal _split method to ensure it chunks documents correctly."""
    # Arrange
    # We can instantiate ChatIngestor with default params because ModelLoader is mocked
    ingestor = ChatIngestor() 
    
    # Create a long Document object that is guaranteed to need splitting
    long_text = "This is a single sentence. " * 300  # A long string to ensure chunking
    test_doc = Document(page_content=long_text, metadata={"source": "test_file.txt"})

    # Act: Call the _split method with a specific chunk size for a predictable outcome
    chunks = ingestor._split([test_doc], chunk_size=500, chunk_overlap=50)

    # Assert
    # 1. The document was split into more than one piece.
    assert len(chunks) > 1, "Document should have been split into multiple chunks."
    
    # 2. The first chunk's content is smaller than the original and correct.
    first_chunk = chunks[0]
    assert isinstance(first_chunk, Document)
    assert len(first_chunk.page_content) <= 500
    assert first_chunk.page_content.startswith("This is a single sentence.")
    
    # 3. The metadata from the original document was preserved in the chunk.
    assert first_chunk.metadata["source"] == "test_file.txt"
    
    # 4. The overlap logic is working.
    # The end of the first chunk should contain the start of the second chunk.
    start_of_second_chunk = chunks[1].page_content
    # Find the overlapping text by looking at the start of the second chunk
    overlapping_text = start_of_second_chunk[:50] 
    assert overlapping_text in first_chunk.page_content
