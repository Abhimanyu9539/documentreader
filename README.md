# DocumentReader - LLM-powered Document Processing

DocumentReader is a Python project that leverages Large Language Models (LLMs) to read, understand, and process text from various document formats including PDFs, Word documents, and plain text files. It integrates LLM embedding models for advanced semantic understanding, enabling sophisticated document processing, search, and AI-powered applications.

## Features

-   **Multi-format Document Processing:** Ingest and process text from PDF, Word (.docx), and plain text files.
-   **Single and Multi-Document Chat:** Engage in conversations with single or multiple documents, allowing you to ask questions and get context-aware answers.
-   **Document Comparison:** Compare two documents to identify similarities and differences.
-   **In-depth Document Analysis:** Analyze documents to extract key information and insights.
-   **Flexible LLM Integration:** Supports multiple LLM providers, including:
    -   OpenAI
    -   Groq
    -   Google Generative AI
-   **Structured and Centralized Logging:** Centralized logging that creates a single, timestamped log file for each run, making debugging and monitoring easier.
-   **Easy Configuration:** Simple configuration using a single `config.yaml` file for models, prompts, and other settings.

## Getting Started

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/DocumentReader.git
    cd DocumentReader
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Environment Variables:**
    Create a `.env` file in the root directory and add your API keys. At a minimum, you need the `OPENAI_API_KEY`.

    ```
    OPENAI_API_KEY="your-openai-api-key"
    # Add other keys if you plan to use other providers
    # GROQ_API_KEY="your-groq-api-key"
    # GOOGLE_API_KEY="your-google-api-key"
    ```

2.  **Configuration File:**
    The main configuration for the application is in `config/config.yaml`. You can customize the LLM provider, models, and other parameters in this file.

    By default, the `LLM_PROVIDER` is set to "openai". You can change this by setting the `LLM_PROVIDER` environment variable or by modifying the `config.yaml` file.

## Usage

To run the Streamlit user interface:

```bash
streamlit run streamlit_ui.py
```

This will launch a web-based interface where you can upload documents and interact with the various features of DocumentReader.

Alternatively, you can run the Flask application:

```bash
python app.py
```

## Project Structure

```
.
├── app.py                  # Main Flask application
├── streamlit_ui.py         # Streamlit user interface
├── requirements.txt        # Project dependencies
├── config/
│   └── config.yaml         # Configuration file
├── custom_logging/
│   └── custom_logger.py    # Custom logging implementation
├── exception/
│   └── custom_exception.py # Custom exception handling
├── src/                    # Source code
│   ├── document_analyser/
│   ├── document_compare/
│   ├── multi_document_chat/
│   └── single_document_chat/
├── static/                 # Static assets for the web interface
├── templates/              # HTML templates for the web interface
├── utils/                  # Utility functions
└── ...
```

## Logging

The application uses a custom logger that creates a structured log file for each run in the `logs/` directory. This makes it easy to trace the application's execution and debug any issues.

## Contributing

Contributions and improvements are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
