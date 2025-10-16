# ## Testing code for document comparison using local PDF files
# import io
# from pathlib import Path
# from src.document_compare.data_ingestion import DocumentIngestion
# from src.document_compare.document_comparator import DocumentComparator

# # ---- Setup: Load local PDF files as if they were "uploaded" ---- #
# def load_fake_uploaded_file(file_path: Path):
#     return io.BytesIO(file_path.read_bytes())  # simulate .getbuffer()

# # ---- Step 1: Save and combine PDFs ---- #
# def test_compare_documents():
#     ref_path = Path("E:/LLMOps/documentreader/data/document_compare/Long_Report_V1.pdf")
#     act_path = Path("E:/LLMOps/documentreader/data/document_compare/Long_Report_V2.pdf")

#     # Wrap them like Streamlit UploadedFile-style
#     class FakeUpload:
#         def __init__(self, file_path: Path):
#             self.name = file_path.name
#             self._buffer = file_path.read_bytes()

#         def getbuffer(self):
#             return self._buffer

#     # Instantiate
#     comparator = DocumentIngestion()
#     ref_upload = FakeUpload(ref_path)
#     act_upload = FakeUpload(act_path)

#     # Save files and combine
#     ref_file, act_file = comparator.save_uploaded_files(ref_upload, act_upload)
#     combined_text = comparator.combine_documents()
#     comparator.clean_old_sessions(keep_latest=3)

#     print("\n Combined Text Preview (First 1000 chars):\n")
#     print(combined_text[:1000])

#     # ---- Step 2: Run LLM comparison ---- #
#     llm_comparator = DocumentComparator()
#     df = llm_comparator.compare_documents(combined_text)
    
#     print("\n Comparison DataFrame:\n")
#     print(df)

# if __name__ == "__main__":
#     test_compare_documents()
    
    

## Code for Conversational RAG system
import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.single_document_chat.data_ingestion import SingleDocIngestor
from src.single_document_chat.retrieval import ConversationalRAG
from utils.model_loader import ModelLoader

FAISS_INDEX_PATH = Path("faiss_index/faiss_index")

def test_conversational_rag_on_pdf(pdf_path:str, question:str):
    try:
        model_loader = ModelLoader()
        
        if FAISS_INDEX_PATH.exists():
            print("Loading existing FAISS index...")
            embeddings = model_loader.load_embeddings()
            vectorstore = FAISS.load_local(folder_path=str(FAISS_INDEX_PATH), embeddings=embeddings,allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        else:
            # Step 2: Ingest document and create retriever
            print("FAISS index not found. Ingesting PDF and creating index...")
            with open(pdf_path, "rb") as f:
                uploaded_files = [f]
                ingestor = SingleDocIngestor()
                retriever = ingestor.ingest_files(uploaded_files)
                
        print("Running Conversational RAG...")
        session_id = "test_conversational_rag"
        rag = ConversationalRAG(retriever=retriever, session_id=session_id)
        response = rag.invoke(question)
        print(f"\nQuestion: {question}\nAnswer: {response}")
                    
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)
    
if __name__ == "__main__":
    # Example PDF path and question
    pdf_path = "/home/user/documentreader/data/single_document_chat/1706.03762v7.pdf"
    question = "What is the attention mechanism? can you explain it in simple terms?"

    if not Path(pdf_path).exists():
        print(f"PDF file does not exist at: {pdf_path}")
        sys.exit(1)
    
    # Run the test
    test_conversational_rag_on_pdf(pdf_path, question)
    