import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch

# ---------- CONFIG ----------
TDS_COURSE_FILE = "tds_course_content.json"
DISCOURSE_FILE = "discourse_posts.json"
EMB_OUTPUT = "combined_embeddings.npy"
META_OUTPUT = "combined_metadata.json"
MIN_CHUNK_LEN = 30    # Minimum characters for a chunk
MAX_CHUNK_LEN = 350   # Maximum characters for a chunk (to stay under 77 CLIP tokens)

# ---------- TDS COURSE CHUNKING ----------
def chunk_course_sections(input_file: str) -> List[Dict]:
    """Chunk TDS course content at logical boundaries (headings, paragraphs, lists)."""
    with open(input_file, "r", encoding="utf-8") as f:
        courses = json.load(f)

    chunks = []
    for course in courses:
        course_title = course.get("course_title", "TDS")
        for section in course.get("sections", []):
            section_name = section.get("section_name", "")
            content = section.get("content", [])
            buffer = []
            for block in content:
                # Handle dict and string types
                if isinstance(block, dict):
                    if block.get("type") == "heading":
                        # Flush buffer as chunk
                        if buffer:
                            chunk_text = "\n".join(buffer).strip()
                            if MIN_CHUNK_LEN <= len(chunk_text) <= MAX_CHUNK_LEN:
                                chunks.append({
                                    "text": chunk_text,
                                    "course_title": course_title,
                                    "section_name": section_name,
                                    "source": "tds_course"
                                })
                            buffer = []
                        buffer.append(block.get("text", ""))
                    elif block.get("type") == "paragraph":
                        buffer.append(block.get("text", ""))
                    elif block.get("type") == "list":
                        items = block.get("items", [])
                        buffer.append("\n".join(f"- {item}" for item in items))
                elif isinstance(block, str):
                    buffer.append(block)
                else:
                    continue

                # If buffer exceeds max length, flush
                if buffer:
                    chunk_text = "\n".join(buffer).strip()
                    if len(chunk_text) > MAX_CHUNK_LEN:
                        chunks.append({
                            "text": chunk_text[:MAX_CHUNK_LEN],
                            "course_title": course_title,
                            "section_name": section_name,
                            "source": "tds_course"
                        })
                        buffer = [chunk_text[MAX_CHUNK_LEN:]]

            # Flush any remaining buffer
            if buffer:
                chunk_text = "\n".join(buffer).strip()
                if MIN_CHUNK_LEN <= len(chunk_text) <= MAX_CHUNK_LEN:
                    chunks.append({
                        "text": chunk_text,
                        "course_title": course_title,
                        "section_name": section_name,
                        "source": "tds_course"
                    })
    print(f"[TDS] Generated {len(chunks)} course chunks")
    return chunks

# ---------- DISCOURSE PROCESSING ----------
def process_discourse(input_file: str) -> List[Dict]:
    """Process forum posts into retrievable chunks."""
    with open(input_file, "r", encoding="utf-8") as f:
        posts = json.load(f)
    chunks = []
    for post in posts:
        text_parts = []
        if post.get("topic_title"):
            text_parts.append(f"Topic: {post['topic_title']}")
        if post.get("content"):
            text_parts.append(post["content"])
        text = "\n".join(text_parts).strip()
        if text:
            chunks.append({
                "text": text,
                "source": "discourse",
                "topic_id": post.get("topic_id"),
                "topic_title": post.get("topic_title"),
                "post_id": post.get("post_id"),
                "author": post.get("author"),
                "created_at": post.get("created_at"),
                "url": post.get("url", ""),
                "content_type": "forum_post"
            })
    print(f"[Discourse] Processed {len(chunks)} posts")
    return chunks

# ---------- EMBEDDING GENERATION ----------
def generate_embeddings(texts: List[str], model, processor, tokenizer, device: str) -> np.ndarray:
    """Generate CLIP embeddings, truncating to 77 tokens."""
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i+batch_size]
        # Truncate each text to 77 tokens
        truncated_batch = []
        for text in batch:
            tokens = tokenizer(text, truncation=True, max_length=77, return_tensors="pt")
            truncated_text = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            truncated_batch.append(truncated_text)
        inputs = processor(
            text=truncated_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        with torch.no_grad():
            batch_emb = model.get_text_features(**inputs)
        batch_emb = batch_emb.cpu().numpy()
        batch_emb = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    # 1. Load and chunk data
    print("Processing TDS course content...")
    tds_chunks = chunk_course_sections(TDS_COURSE_FILE)
    print("Processing Discourse posts...")
    discourse_chunks = process_discourse(DISCOURSE_FILE)
    all_chunks = tds_chunks + discourse_chunks
    all_texts = [chunk["text"] for chunk in all_chunks]
    print(f"Total chunks to embed: {len(all_texts)}")

    # 2. Initialize CLIP model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device.upper()}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 3. Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(all_texts, clip_model, clip_processor, clip_tokenizer, device)

    # 4. Save results
    print(f"Saving embeddings to {EMB_OUTPUT}")
    np.save(EMB_OUTPUT, embeddings.astype(np.float32))
    print(f"Saving metadata to {META_OUTPUT}")
    with open(META_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print("✅ Embedding generation complete!")
