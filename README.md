# DocumentReader - LLM-powered Document Processing

DocumentReader is a Python project that leverages Large Language Models (LLMs) to read, understand, and process text from various document formats including PDFs, Word documents, and plain text files. It integrates LLM embedding models for advanced semantic understanding, enabling sophisticated document processing, search, and AI-powered applications.

## Features

- Support for multiple document formats: PDF, Word (.doc/.docx), and text files
- Integration with LLM embedding models for semantic text representation
- Configurable embedding model selection via YAML and environment variables
- Structured JSON logging powered by `structlog` for enhanced observability
- Modular design for easy extension and integration into AI/ML pipelines

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from utils.config_loader import load_config
from custom_logging.custom_logger import CustomLogger

logger = CustomLogger().get_logger(__file__)
config = load_config()

# Use LLM embedding models and document loaders as configured
```

## Configuration

- Use `config/config.yaml` for configuring embedding models, API keys, and other parameters
- Optionally set the `LLM_PROVIDER` environment variable to select your preferred LLM backend (default: "openai")

## Contributing

Contributions and improvements are welcome! Please fork the repository and submit pull requests.
