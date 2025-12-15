import os
from typing import List
from pypdf import PdfReader

def list_pdfs(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                out.append(os.path.join(root, f))
    return out


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            # some pages may fail extraction; skip
            continue
    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Simple chunker by characters. Returns chunks with overlap."""
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]


class ConversationMemory:
    """Window buffer memory to store last k conversation turns."""
    
    def __init__(self, window_size: int = 10):
        """Initialize with a specific window size (number of past interactions to remember).
        
        Args:
            window_size: Number of past question-answer pairs to keep in memory
        """
        self.window_size = window_size
        self.history = []
    
    def add_interaction(self, question: str, answer: str):
        """Add a question-answer pair to memory."""
        self.history.append({"question": question, "answer": answer})
        # Keep only the last window_size interactions
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
    
    def get_history(self) -> List[dict]:
        """Get the current conversation history."""
        return self.history
    
    def get_formatted_history(self) -> str:
        """Get formatted conversation history for context."""
        if not self.history:
            return ""
        
        formatted = ["Previous conversation:"]
        for i, interaction in enumerate(self.history, 1):
            formatted.append(f"\nTurn {i}:")
            formatted.append(f"Student: {interaction['question']}")
            formatted.append(f"Assistant: {interaction['answer']}")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Clear all conversation history."""
        self.history = []
    
    def __len__(self):
        """Return the number of interactions in memory."""
        return len(self.history)
