# RAG-Powered PDF Q&A Chatbot with GROQ & LLaMA3

This project is an interactive GenAI-powered Q&A chatbot that answers user queries based on the contents of a folder of research papers (PDFs). It combines LLMs, vector embeddings, and document retrieval using the LangChain framework, FAISS, and the GROQ API running LLaMA3.

## Features

- Upload and parse multiple PDFs from a folder
- Split documents into semantic chunks
- Embed text using Ollama Embeddings
- Store and retrieve using FAISS vector database
- Query using LLaMA3 via GROQ API
- Built-in UI with Streamlit
- Context-aware answers backed by source content

## Tech Stack

| Tool/Library              | Purpose                                |
|---------------------------|----------------------------------------|
| Streamlit                | Frontend for user interaction           |
| LangChain                | LLM chaining, vector store integration  |
| GROQ API                 | Fast LLM inference (LLaMA3 8B)          |
| OllamaEmbeddings         | Local embeddings for semantic search    |
| FAISS                    | Vector store for fast similarity search |
| PyPDFDirectoryLoader     | Load PDFs into LangChain format         |

