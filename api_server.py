import os
import json
import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
import requests

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

# Load precomputed embeddings and metadata
embeddings = np.load("embeddings/combined_embeddings.npy")
with open("embeddings/combined_metadata.json") as f:
    metadata = json.load(f)

# Load CLIP model and processor for multimodal embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(text: Optional[str]=None, image: Optional[Image.Image]=None):
    if text:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
    elif image:
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
    else:
        raise ValueError("Either text or image must be provided.")
    emb = emb.cpu().numpy().flatten()
    emb = emb / np.linalg.norm(emb)
    return emb

def clean_metadata(meta):
    clean = {}
    for k, v in meta.items():
        if isinstance(v, list):
            clean[k] = ",".join(str(x) for x in v)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v)
        else:
            clean[k] = v
    return clean

client = chromadb.Client()

# Delete existing collection if dimensions mismatch
try:
    existing_col = client.get_collection("docs")
    if existing_col.metadata.get("dimension") != 512:
        client.delete_collection("docs")
except Exception as e:
    print(f"No existing collection or error: {str(e)}")

# Create new collection with proper configuration
collection = client.create_collection(
    name="docs",
    metadata={
        "hnsw:space": "cosine",
        "dimension": 512  # Explicitly set for CLIP ViT-B/32
    }
)

# Verify embedding dimensions match
assert embeddings.shape[1] == 512, f"Embedding dimension mismatch. Expected 512, got {embeddings.shape[1]}"

# Add documents with proper dimension validation
collection.add(
    ids=[str(i) for i in range(len(embeddings))],
    embeddings=embeddings.tolist(),
    documents=[m.get("text", "") for m in metadata],
    metadatas=[clean_metadata(m) for m in metadata]
)

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 string

'''def get_llm_answer(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY") or "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"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://huggingface.co/spaces/<your-username>/<your-space-name>",
        "X-Title": "TDS Virtual TA"
    }
    data = {
        "model": "meta-llama/llama-4-maverick:free",  # or any other free OpenRouter model
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]'''


openrouter_rate_limited = False

def get_llm_answer(prompt):
    global openrouter_rate_limited
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or "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"
    openrouter_headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        #"HTTP-Referer": "https://huggingface.co/spaces/<your-username>/<your-space-name>",
        "X-Title": "TDS Virtual TA"
    }
    openrouter_data = {
        "model": "meta-llama/llama-4-maverick:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=openrouter_headers,
            json=openrouter_data,
            timeout=60
        )
        if resp.status_code == 429:
            if not openrouter_rate_limited:
                print("OpenRouter rate limit hit, switching to AIPipe.")
                openrouter_rate_limited = True
            raise requests.exceptions.HTTPError("Rate limit hit", response=resp)
        resp.raise_for_status()
        if openrouter_rate_limited:
            print("OpenRouter is available again.")
            openrouter_rate_limited = False
        print("Response from OpenRouter.")
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if hasattr(e, "response") and e.response is not None and e.response.status_code == 429:
            aipipe_token = os.getenv("AIPIPE_TOKEN")
            if not aipipe_token:
                print("AIPipe token not set.")
                raise EnvironmentError("AIPIPE_TOKEN environment variable not set.")
            aipipe_headers = {
                "Authorization": f"Bearer {aipipe_token}",
                "Content-Type": "application/json"
            }
            aipipe_data = {
                "model": "openai/gpt-4.1-nano",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            aipipe_resp = requests.post(
                "https://aipipe.org/openrouter/v1/chat/completions",
                headers=aipipe_headers,
                json=aipipe_data,
                timeout=60
            )
            aipipe_resp.raise_for_status()
            print("Awaiting Response from AIPipe.")
            return aipipe_resp.json()["choices"][0]["message"]["content"]
        else:
            print(f"Non-429 HTTPError from OpenRouter: {e}")
            raise
    except Exception as e:
        print(f"General error: {e}")
        raise
  
# Helper to generate human-friendly link description using OpenRouter
def metadata_to_prompt(meta):
    return (
        f"Summarize the following forum post metadata in one sentence for a helpful link preview:\n"
        f"Title: {meta.get('topic_title', '')}\n"
        f"Author: {meta.get('author', '')}\n"
        f"Tags: {meta.get('tags', '')}\n"
        f"Date: {meta.get('created_at', '')}\n"
        f"Content type: {meta.get('content_type', '')}\n"
        f"Likes: {meta.get('like_count', 0)}, Replies: {meta.get('reply_count', 0)}"
    )

def generate_link_text(meta):
    prompt = metadata_to_prompt(meta)
    return get_llm_answer(prompt).strip()

@app.post("/api/")
async def answer_query(req: QueryRequest):
    # 1. Get CLIP embedding for question
    text_emb = get_clip_embedding(text=req.question)

    # 2. If image is provided, get CLIP embedding for image and average with text
    if req.image:
        image_bytes = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_emb = get_clip_embedding(image=image)
        query_emb = (text_emb + image_emb) / 2
        query_emb = query_emb / np.linalg.norm(query_emb)
    else:
        query_emb = text_emb

    # 3. Search ChromaDB for similar documents
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    # 4. Filter results by similarity threshold (cosine distance <= 0.3 ~ 70% similarity)
    filtered_results = [
        (doc, meta, dist)
        for doc, meta, dist in zip(results['documents'][0],
                                   results['metadatas'][0],
                                   results['distances'][0])
        if dist <= 0.3
    ]

    if not filtered_results:
        return {"answer": "I couldn't find relevant information in my knowledge base.",
                "links": []}

    # 5. Build context with relevance indicators
    context = "Relevant information from course materials:\n"
    for idx, (doc, meta, dist) in enumerate(filtered_results, 1):
        context += f"\n[Source {idx}] {doc}\n(Similarity: {1 - dist:.1%})\n"

    # 6. Enhanced prompt engineering
    prompt = f"""You're a Teaching Assistant for the Tools for Data Science (TDS) course. 
Answer the student's question using ONLY the context below. If unsure, say so.

Question: {req.question}

{context}

Guidelines:
1. Be specific about course structure, grading, or technical requirements
2. Cite sources using [Source X] notation
3. If asking about due dates/numbers, verify exact values from context
4. For technical questions, provide exact tool/version from context

Answer:"""

    # 7. Generate answer using OpenRouter
    answer = get_llm_answer(prompt)

    # 8. Prepare links for response, using OpenRouter to generate summary for each
    links = []
    for doc, meta, dist in filtered_results:
        desc = generate_link_text(meta)
        links.append({
            "url": meta.get("url", ""),
            "text": desc
        })

    return {"answer": answer, "links": links}
