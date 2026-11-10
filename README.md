# MLensAI

A lightweight, document-grounded chatbot built with LangChain, Hugging Face embeddings, ChromaDB, and Google Gemini. The project reads a PDF document, splits it into searchable chunks, stores embeddings in a persistent vector database, and answers user questions using only the information available in the provided document.

## Overview

This project is designed to answer questions based strictly on a knowledge source rather than relying on general model knowledge. The system works as a Retrieval-Augmented Generation (RAG) pipeline:

1. A PDF is loaded and parsed.
2. The content is split into manageable text chunks.
3. Each chunk is embedded with a sentence-transformers model.
4. Relevant chunks are retrieved for a user query.
5. The LLM generates a response using only the retrieved context.

The project is best suited for local document Q&A and quick knowledge assistants over a single PDF.

## Features

- PDF ingestion using LangChain document loaders
- Text chunking with recursive splitting
- Semantic search using vector embeddings
- MMR-based retrieval for better contextual matching
- Persistent ChromaDB vector storage
- Prompt-based response grounding to reduce hallucination
- Command-line chat interface for quick interactions
- Strict document-only answer policy

## Tech Stack

- Python
- LangChain
- LangChain Community
- Hugging Face Embeddings
- ChromaDB
- Google Gemini via LangChain chat model
- PyPDF
- Python dotenv for environment configuration

## Project Structure

```text
ChatBot/
├── Data_processing.py        # Loads PDF, embeds text, and stores it in ChromaDB
├── main.py                   # Chat interface and Q&A loop
├── Docs.pdf                  # Source PDF used as the knowledge base
├── chromaDb/                 # Persistent vector database files
├── chroma.sqlite3            # SQLite database used by Chroma
├── .env                      # Local environment variables (not committed)
├── .gitignore                # Excludes local environment files
└── README.md                 # Project documentation
```

## How It Works

### 1. Data ingestion
`Data_processing.py` loads `Docs.pdf` using `PyPDFLoader`, splits the text using `RecursiveCharacterTextSplitter`, and stores the split documents into a persistent Chroma vector store.

### 2. Retrieval
`main.py` loads the vector store and creates a retriever with MMR search:

- `k = 4`
- `fetch_k = 10`
- `lambda_mult = 0.5`

This gives the model a set of the most relevant document chunks to answer a query.

### 3. Generation
The chat loop in `main.py`:

- takes the user question,
- retrieves context from Chroma,
- formats it into a prompt,
- sends the prompt to a Gemini model,
- prints the response.

The system prompt explicitly instructs the model to answer only using the retrieved document context and to return:

> "I am not able to answer this at the moment."

when the answer is not found in the source material.

## Setup

### Prerequisites

- Python 3.9+
- A working virtual environment
- A Google API key for Gemini access

### Install dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install python-dotenv langchain langchain-community langchain-core langchain-huggingface langchain-chroma langchain-google-genai sentence-transformers pypdf chromadb
```

If your environment already contains compatible package versions, you may only need to install the missing ones.

### Environment variables

Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Optional Hugging Face token variables may also be configured if required by your environment.

## Running the Project

### Step 1: Build the knowledge base
Run the document processing script to ingest and embed the PDF:

```bash
python Data_processing.py
```

This script creates/updates the vector database for the PDF content.

### Step 2: Start the chatbot
Run the chat application:

```bash
python main.py
```

Then enter a question related to the document content. Type `0` to exit the chat loop.

## Example Workflow

```bash
python Data_processing.py
python main.py
```

Example interaction:

```text
Type to Search: What is the main focus of this document?
Ai: ...answer based only on the retrieved PDF content...
```

## Important Notes

- The project is intentionally grounded in the provided PDF; it does not use external knowledge.
- It works best for one-document, domain-specific Q&A workflows.
- The script is currently a command-line tool and does not provide a web UI or API.
- If `Docs.pdf` changes, rerun `Data_processing.py` to refresh the vector index.

## Limitations

- Depends heavily on the quality of the source PDF.
- Retrieval quality depends on chunk size and embedding quality.
- Responses may be limited by the context window and retrieval relevance.
- The project is not optimized for large multi-document enterprise systems.

## Troubleshooting

### Chroma vector store not updating
Re-run `Data_processing.py` after changing the document or embedding settings.

### Model authentication errors
Ensure `GOOGLE_API_KEY` is configured correctly in `.env` and the model package is installed.

### Import errors
Check that your virtual environment contains the required LangChain and embedding libraries.

### Persistent database path issues
The ingestion script stores data in a configured persistent directory. If you move the project or run it from a different machine path, verify that the `persist_directory` matches the project layout.

## Future Improvements

- Add a web interface using Streamlit or FastAPI
- Support multiple PDF documents or directories
- Add document upload and automatic re-indexing
- Add chat history and session memory
- Improve evaluation and answer confidence scoring
- Add better source citation support in responses

## License

This project does not currently include a license file. Please confirm the intended licensing terms before using it in a production or shared environment.
