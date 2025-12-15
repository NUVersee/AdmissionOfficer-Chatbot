"""Interactive query: embed user query, retrieve top-k, and ask Ollama LLM to answer using context."""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import List

load_dotenv()

from src.ollama_client import embed, generate

# Results storage settings
RESULTS_DIR = os.path.join(os.getcwd(), "rag_results_llama")
RESULTS_FILENAME = "llama_3.2billion.json"

def save_result(query: str, answer: str, sources: list):
    """Append a query/answer record to a JSON file under RESULTS_DIR."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    current_result = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.now().isoformat(),
    }

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
        data.append(current_result)
    else:
        data = [current_result]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅  Query and answer saved to {file_path}")

VECTOR_DB = os.getenv("VECTOR_DB", "chroma").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def retrieve_with_chroma(query_embedding: List[float], top_k: int = 4, category_filter: str = None):
    """Retrieve relevant Q&A pairs from Chroma with optional category filtering."""
    try:
        import chromadb

        # Prefer persistent client; fall back quietly
        client = None
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        except Exception:
            try:
                from chromadb.config import Settings
                client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIR))
            except Exception:
                try:
                    client = chromadb.EphemeralClient()
                except Exception:
                    return [], []

        # Get qa_knowledge collection (stores JSON Q&A data)
        col = None
        try:
            col = client.get_collection("qa_knowledge")
        except Exception:
            return [], []

        # Apply category filter if provided (e.g., only Fees, Admissions, etc.)
        if category_filter:
            results = col.query(
                query_embeddings=[query_embedding], 
                n_results=top_k,
                where={"category": category_filter}
            )
        else:
            results = col.query(
                query_embeddings=[query_embedding], 
                n_results=top_k
            )
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return docs, metas
    except Exception as e:
        # If there's an error, print it for debugging and return empty
        print(f"ChromaDB error: {e}")
        return [], []

def format_context(docs: List[str], metas: List[dict]) -> str:
    """Format retrieved documents with metadata for LLM context."""
    fragments = []
    for d, m in zip(docs, metas):
        # Avoid including IDs, source names, or chunk indices in the context to prevent echoing
        category = m.get("category")
        if category:
            fragments.append(f"Category: {category}\n{d}\n---\n")
        else:
            # For PDF-sourced content, include only the text
            fragments.append(f"{d}\n---\n")
    return "\n".join(fragments)

def detect_category(query: str) -> str:
    """Detect the category of a question based on keywords."""
    query_lower = query.lower()
    
    # Category keywords based on Categories.txt
    category_keywords = {
        "Admissions": ["apply", "admission", "accept", "requirements", "application", "enroll"],
        "Fees": ["fee", "tuition", "cost", "payment", "credit", "price", "pay", "refund"],
        "Academics": ["gpa", "grades", "scores", "grade", "cgpa", "dean"],
        "Academic Advising": ["advisor", "track", "course", "major", "register", "summer course"],
        "IT & Systems": ["portal", "moodle", "login", "system", "technical", "support"],
        "Emails": ["email", "gmail", "outlook", "mail", "inbox", "address", "contact email"],
    }
    
    # Count keyword matches for each category
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            category_scores[category] = score
    
    # Return category with highest score, or None if no matches
    if category_scores:
        return max(category_scores, key=category_scores.get)
    return None

def local_retrieve(query: str, top_k: int = 1, category_filter: str = None):
    """Keyword-overlap retrieval over JSON data on disk with optional category filtering."""
    from src.utils import chunk_text, list_pdfs, load_pdf_text

    cwd = os.getcwd()
    json_path = os.path.join(cwd, "data.json")
    all_chunks = []
    metadatas = []

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            items = []
            for entry in data:
                category = entry.get("category", "General")
                
                # Apply category filter if provided
                if category_filter and category != category_filter:
                    continue
                
                qa_id = entry.get("id", "")
                qtext = str(entry.get("question", "")).strip()
                atext = str(entry.get("answer", "")).strip()
                
                if qtext or atext:
                    combined = f"Question: {qtext}\nAnswer: {atext}"
                    items.append(combined)
                    # Store metadata with each item
                    metadatas.append({
                        "source": json_path,
                        "qa_id": str(qa_id),
                        "category": category,
                        "question": qtext,
                        "answer": atext
                    })
        except Exception as e:
            print(f"Error reading data.json: {e}")
            items = []
            metadatas = []
    else:
        items = []
        metadatas = []
    
    # Use items directly (already have metadata)
    if not items:
        return [], []

    # naive keyword overlap scoring
    q_tokens = set(w.lower() for w in query.split())
    scores = []
    for item in items:
        item_tokens = set(w.lower().strip('.,()\"') for w in item.split())
        scores.append(len(q_tokens & item_tokens))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = [items[i] for i, s in ranked[:top_k] if s > 0]
    top_meta = [metadatas[i] for i, s in ranked[:top_k] if s > 0]
    return top, top_meta

def run():
    from src.utils import ConversationMemory
    
    # Initialize conversation memory with window size of 10 (last 10 interactions)
    memory = ConversationMemory(window_size=10)
    
    print("RAG query with memory (type 'exit' to quit, 'clear' to reset conversation history)")
    print(f"Memory window size: {memory.window_size} past interactions\n")
    
    while True:
        q = input("Question> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "clear":
            memory.clear()
            print("✅ Conversation history cleared.\n")
            continue

        # Detect category from question
        detected_category = detect_category(q)
        if detected_category:
            print(f"🏷️  Detected category: {detected_category}")
        
        try:
            q_emb = embed([q])[0]
            # Try category-filtered retrieval first
            docs, metas = retrieve_with_chroma(q_emb, top_k=5, category_filter=detected_category)
            
            # If no results with category filter, try without filter
            if not docs and detected_category:
                print(f"No results in {detected_category}, searching all categories...")
                docs, metas = retrieve_with_chroma(q_emb, top_k=5)
            
            ctx = format_context(docs, metas)
        except Exception as e:
            print(f"\nVector DB or Ollama call failed ({e}); using local keyword-match fallback.\n")
            docs, metas = local_retrieve(q, top_k=3, category_filter=detected_category)
            
            if not docs and detected_category:
                print(f"No results in {detected_category}, searching all categories...")
                docs, metas = local_retrieve(q, top_k=3)
            
            if not docs:
                print("No local documents found to answer the query.")
                continue
            ctx = format_context(docs, metas)

        system = """
        You are an AI admissions officer at Nile University. Your role is to provide accurate, friendly, and helpful guidance to prospective students.
        Answer questions about programs, courses, application deadlines, admission requirements, scholarships, and campus life.
        Guide students step-by-step through the application process when needed.
        Clarify university policies politely and professionally.
        Adapt your tone to be friendly, encouraging, and approachable while maintaining professionalism.
        Ask follow-up questions to better understand the student’s needs before giving detailed advice.
        Avoid giving inaccurate or vague information; if you don't know the answer, direct the student to official resources.
        Use the conversation history to provide contextual and coherent responses that reference previous interactions when relevant.
        Do not include any internal metadata (IDs, file names, chunk indices, or bracketed tags) in your final answer.
        """ 
        prompt = f"Use the following context to answer the question.\n\nCONTEXT:\n{ctx}\n\nQUESTION:\n{q}\n\nAnswer concisely."

        out = generate(system, prompt, conversation_history=memory.get_history())
        print("\n--- ANSWER ---\n")
        print(out)
        print("\n---------------\n")
        
        # Add to conversation memory
        memory.add_interaction(q, str(out))
        print(f"[Memory: {len(memory)}/{memory.window_size} interactions stored]\n")

        # Save result with enhanced metadata
        sources = []
        for m in metas:
            if isinstance(m, dict):
                qa_id = m.get("qa_id")
                category = m.get("category")
                if qa_id:
                    sources.append(f"QA#{qa_id} ({category})")
                elif m.get("source"):
                    sources.append(m.get("source"))
        save_result(query=q, answer=str(out), sources=list(set(sources)))

if __name__ == "__main__":
    run()