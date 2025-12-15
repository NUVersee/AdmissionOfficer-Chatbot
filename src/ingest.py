"""Ingest JSON Q&A data, embed with Ollama, and persist to Chroma."""
import os
import json
from typing import List
from dotenv import load_dotenv

load_dotenv()

from src.ollama_client import embed
import argparse

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def ingest_json_data(json_path: str, dry_run: bool = False):
    """Ingest structured Q&A data from JSON with categories and IDs."""
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return
    
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_texts = []
    metadatas = []
    ids = []
    
    for entry in data:
        q_id = entry.get('id', 'unknown')
        category = entry.get('category', 'General')
        question = entry.get('question', '').strip()
        answer = entry.get('answer', '').strip()
        
        if not question and not answer:
            continue
        
        # Combine question and answer for embedding
        combined_text = f"Question: {question}\nAnswer: {answer}"
        
        all_texts.append(combined_text)
        metadatas.append({
            "source": "data.json",
            "qa_id": str(q_id),
            "category": category,
            "question": question,
            "answer": answer
        })
        ids.append(f"qa_{q_id}")
    
    if not all_texts:
        print("No Q&A data extracted from JSON.")
        return
    
    print(f"Prepared {len(all_texts)} Q&A pairs from {len(data)} entries.")
    if dry_run:
        print("Dry run enabled - skipping embeddings and DB persistence.")
        if all_texts:
            print("Sample metadata:", metadatas[0])
            print("Sample text (200 chars):", all_texts[0][:200])
        return
    
    print(f"Embedding {len(all_texts)} Q&A pairs using Ollama...")
    batch = 32
    embeddings = []
    for i in range(0, len(all_texts), batch):
        batch_texts = all_texts[i : i + batch]
        em = embed(batch_texts)
        embeddings.extend(em)
    
    dim = len(embeddings[0])
    print(f"Embedding dim: {dim}")
    
    if VECTOR_DB == "chroma":
        try:
            import chromadb
            try:
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            except Exception:
                try:
                    from chromadb.config import Settings
                    client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIR))
                except Exception:
                    client = chromadb.Client()
            
            # Delete existing collection and create new one
            try:
                client.delete_collection("qa_knowledge")
                print("Deleted existing collection 'qa_knowledge'")
            except Exception:
                pass
            
            col = client.create_collection("qa_knowledge")
            col.add(ids=ids, documents=all_texts, metadatas=metadatas, embeddings=embeddings)
            
            try:
                client.persist()
            except Exception:
                pass
            
            print(f"✅ Saved {len(ids)} Q&A pairs to Chroma at {CHROMA_PERSIST_DIR}")
        except Exception as e:
            print("Chroma save error:", e)


def main():
    parser = argparse.ArgumentParser(description="Ingest Q&A data from JSON to Chroma vector DB")
    parser.add_argument("--json", "-j", default="data.json", help="Path to JSON file with Q&A data (default: data.json)")
    parser.add_argument("--dry-run", action="store_true", help="Don't call Ollama / persist; just show counts")
    args = parser.parse_args()
    
    ingest_json_data(args.json, args.dry_run)


if __name__ == "__main__":
    main()
