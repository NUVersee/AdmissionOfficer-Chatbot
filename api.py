"""FastAPI REST API for RAG Admission Officer"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from datetime import datetime

# Import RAG functions
from src.query import (
    embed, 
    generate, 
    retrieve_with_chroma, 
    format_context, 
    detect_category,
    local_retrieve,
    save_result
)
from src.utils import ConversationMemory

# Initialize FastAPI app
app = FastAPI(
    title="RAG Admission Officer API",
    description="REST API for university admission chatbot with RAG",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global conversation memory storage (in production, use Redis or database)
conversation_sessions = {}

# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    category: Optional[str] = None
    use_memory: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    category: Optional[str] = None
    sources: List[str]
    timestamp: str
    memory_size: int

class ClearMemoryRequest(BaseModel):
    session_id: Optional[str] = "default"

class CategoryResponse(BaseModel):
    categories: List[str]
    detected: Optional[str] = None

# Helper function to get or create session memory
def get_session_memory(session_id: str) -> ConversationMemory:
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationMemory(window_size=10)
    return conversation_sessions[session_id]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "RAG Admission Officer API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask",
            "categories": "/categories",
            "detect_category": "/detect-category",
            "clear_memory": "/clear-memory",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    try:
        # Test Ollama connection
        test_embed = embed(["test"])
        ollama_status = "connected" if test_embed else "failed"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "active_sessions": len(conversation_sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main endpoint to ask questions to the RAG system
    
    - **question**: The user's question
    - **session_id**: Optional session ID for conversation memory (default: "default")
    - **category**: Optional category filter (Admissions, Fees, Academics, etc.)
    - **use_memory**: Whether to use conversation memory (default: True)
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    question = request.question.strip()
    session_id = request.session_id or "default"
    
    try:
        # Get session memory if enabled
        memory = get_session_memory(session_id) if request.use_memory else None
        
        # Auto-detect category if not provided
        detected_category = request.category or detect_category(question)
        
        # Get question embedding
        try:
            q_emb = embed([question])[0]
            docs, metas = retrieve_with_chroma(
                q_emb, 
                top_k=5, 
                category_filter=detected_category
            )
            
            # Fallback to all categories if no results
            if not docs and detected_category:
                docs, metas = retrieve_with_chroma(q_emb, top_k=5)
            
            ctx = format_context(docs, metas)
        except Exception as e:
            # Fallback to local keyword search
            docs, metas = local_retrieve(
                question, 
                top_k=3, 
                category_filter=detected_category
            )
            
            if not docs and detected_category:
                docs, metas = local_retrieve(question, top_k=3)
            
            if not docs:
                raise HTTPException(
                    status_code=404, 
                    detail="No relevant information found for this question"
                )
            
            ctx = format_context(docs, metas)
        
        # System prompt
        system = """
        You are an AI admissions officer at Nile University. Your role is to provide accurate, friendly, and helpful guidance to prospective students.
        Answer questions about programs, courses, application deadlines, admission requirements, scholarships, and campus life.
        Guide students step-by-step through the application process when needed.
        Clarify university policies politely and professionally.
        Adapt your tone to be friendly, encouraging, and approachable while maintaining professionalism.
        Ask follow-up questions to better understand the student's needs before giving detailed advice.
        Avoid giving inaccurate or vague information; if you don't know the answer, direct the student to official resources.
        Use the conversation history to provide contextual and coherent responses that reference previous interactions when relevant.
        Do not include any internal metadata (IDs, file names, chunk indices, or bracketed tags) in your final answer.
        """
        
        # Generate prompt with context
        prompt = f"Use the following context to answer the question.\n\nCONTEXT:\n{ctx}\n\nQUESTION:\n{question}\n\nAnswer concisely."
        
        # Get conversation history if memory is enabled
        history = memory.get_history() if memory else []
        
        # Generate answer
        answer = generate(system, prompt, conversation_history=history)
        
        # Add to memory if enabled
        if memory:
            memory.add_interaction(question, answer)
        
        # Extract sources
        sources = []
        for m in metas:
            if isinstance(m, dict):
                qa_id = m.get("qa_id")
                category = m.get("category")
                if qa_id:
                    sources.append(f"QA#{qa_id} ({category})")
                elif m.get("source"):
                    sources.append(m.get("source"))
        
        sources = list(set(sources))
        
        # Save result to file
        save_result(query=question, answer=answer, sources=sources)
        
        return QueryResponse(
            answer=answer,
            category=detected_category,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            memory_size=len(memory) if memory else 0
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/categories", response_model=CategoryResponse)
async def get_categories():
    """Get list of available question categories"""
    categories = [
        "Admissions",
        "Fees",
        "Academics",
        "Academic Advising",
        "IT & Systems",
        "Emails",
        "General"
    ]
    
    return CategoryResponse(categories=categories)

@app.post("/detect-category", response_model=CategoryResponse)
async def detect_question_category(request: QueryRequest):
    """Detect the category of a question"""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    detected = detect_category(request.question.strip())
    
    return CategoryResponse(
        categories=[
            "Admissions", "Fees", "Academics", 
            "Academic Advising", "IT & Systems", "Emails", "General"
        ],
        detected=detected
    )

@app.post("/clear-memory")
async def clear_conversation_memory(request: ClearMemoryRequest):
    """Clear conversation memory for a session"""
    session_id = request.session_id or "default"
    
    if session_id in conversation_sessions:
        conversation_sessions[session_id].clear()
        return {"status": "success", "message": f"Memory cleared for session: {session_id}"}
    
    return {"status": "success", "message": "No memory found for this session"}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session completely"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}
    
    return {"status": "success", "message": "Session not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active conversation sessions"""
    sessions = []
    for session_id, memory in conversation_sessions.items():
        sessions.append({
            "session_id": session_id,
            "interactions": len(memory),
            "max_size": memory.window_size
        })
    
    return {"active_sessions": len(sessions), "sessions": sessions}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Admission Officer API...")
    print("📖 API docs available at: http://localhost:8000/docs")
    print("🔗 React frontend should connect to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
