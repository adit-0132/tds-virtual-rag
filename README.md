
# TDS Insight Engine

**A Multimodal, Robust Retrieval-Augmented Generation System with Web Scraping, Semantic Search, and Dual LLM API Fallback**

---

## Overview

This project delivers a production-grade RAG (Retrieval-Augmented Generation) system for the IIT Madras "Tools for Data Science" (TDS) course, enabling users to ask questions and receive answers grounded in official course materials and forum discussions.  
It supports **text and image queries** (using OpenAI's CLIP for multimodal embeddings), leverages **Playwright for dynamic web scraping** of course and forum content, and ensures high availability by **automatically falling back from OpenRouter to AI Pipe** if one LLM provider is unavailable.

---

## Key Features

- **Smart Chunking:**  
  Course content is intelligently split at logical boundaries (paragraphs, lists, headings), not arbitrary lengths, for precise, context-rich retrieval.
- **Multimodal Search:**  
  Embeddings are generated for both text and images, allowing semantic search over course diagrams and screenshots.
- **Vector Database:**  
  All embeddings are indexed in ChromaDB for fast, accurate search across thousands of chunks.
- **Dynamic Web Scraping:**  
  Playwright ensures up-to-date content by scraping dynamic, JavaScript-heavy course pages and forums.
- **Dual LLM API Fallback:**  
  Answers are generated via OpenRouter, but the system automatically switches to [AI Pipe](https://aipipe.org/) if OpenRouter fails, delivering uninterrupted Q&A.
- **Production-Ready Deployment:**  
  The entire pipeline is containerized with Docker, version-pinned for reproducibility, and served via FastAPI for easy scaling and integration.

---

## Technologies Used

- **Backend:** FastAPI, Docker
- **Structured Data:** JSON, ChromaDB
- **Embeddings:** OpenAI CLIP
- **Web Scraping:** Playwright
- **LLM APIs:** OpenRouter, AI Pipe (fallback)
- **Vector Database:** ChromaDB

---

## How It Works

1. **Data Ingestion & Chunking:**  
   Course materials and forum posts are ingested and chunked for optimal retrieval.
2. **Multimodal Embedding:**  
   Text and images are encoded for semantic search.
3. **Vector Indexing:**  
   Embeddings are stored in ChromaDB.
4. **Query Processing:**  
   User questions (text or image) trigger retrieval of the most relevant chunks.
5. **Answer Generation:**  
   Retrieved context is passed to an LLM (OpenRouter or AI Pipe) for grounded, accurate answers.
6. **Link & Preview Generation:**  
   Responses include relevant course or forum links with descriptive previews.

---

## Usage

1. **Start the FastAPI server:**

docker-compose up --build \&\& docker exec -it <container> bash
python server.py


2. **Access the API at `http://localhost:7860/docs` to submit queries.**
3. **Submit text or image queries** (base64-encoded images supported).

---

## Example API Request


{
"question": "What are the main evaluation criteria for the TDS course?",
"image": "<base64-encoded-image>"
}

---

## Project Highlights

- **Precise, context-grounded answers** for TDS course and forum content.
- **Resilient to LLM API failures** with automatic fallback.
- **Up-to-date knowledge base** via dynamic web scraping.
- **Dockerized, reproducible deployment** for easy scaling.

---
## Extensible Across Topics and Domains

This pipeline is built for **easy adaptation to any topic, course, or forum**. The architecture is modular—seamlessly separating ingestion, embedding, vector storage, and LLM querying—so you can plug in new content sources with minimal setup.

**To build a domain-specific assistant:**

- **Add your data:**  
  Supply new course notes, documentation, or forum posts as JSON files. The existing ingestion and chunking logic remains unchanged.
- **Train on your corpus:**  
  The system generates multimodal embeddings from your data, indexes them in ChromaDB, and exposes a query API tailored to your content.
- **Customize prompts:**  
  Adjust LLM instructions for your topic to ensure focused, actionable responses.
- **Seamless deployment:**  
  Use Docker + FastAPI for easy scaling from local testing to production environments.

**Beyond education:**  
This framework powers internal knowledge bases, customer support Q&A, technical documentation chatbots, and community moderation bots—**anywhere authoritative, up-to-date answers are needed**.

**Get started:**  
Just add your data, deploy, and you have a production-ready RAG assistant for any domain!

**This system is ideal for educational platforms, knowledge base automation, and enterprise RAG applications requiring high availability, multimodal search, and robust data pipelines.**  
