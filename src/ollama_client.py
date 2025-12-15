from typing import List, Optional
import os
import json
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

class OllamaClient:
    def __init__(self):
        self.use_pkg = False
        try:
            import ollama as _ollama

            self._ollama_pkg = _ollama
            self.use_pkg = True
        except Exception:
            self.use_pkg = False

        self.base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        model = model or EMBED_MODEL
        if self.use_pkg:
            # ollama package: call embeddings() with proper input
            try:
                embeddings = []
                for text in texts:
                    try:
                        res = self._ollama_pkg.embeddings(model=model, prompt=text)
                        # Handle EmbeddingsResponse object
                        if hasattr(res, 'embedding'):
                            embeddings.append(res.embedding)
                        elif isinstance(res, dict) and "embedding" in res:
                            embeddings.append(res["embedding"])
                        else:
                            print(f"Unexpected ollama.embeddings response: {type(res)} = {res}")
                            return []
                    except Exception as e:
                        print(f"Error calling ollama.embeddings: {e}")
                        return []
                return embeddings
            except Exception as e:
                print(f"Ollama package embedding error: {e}")
                # Fall through to HTTP fallback

        # HTTP fallback: /api/embeddings endpoint (Ollama API)
        url = f"{self.base_url}/api/embeddings"
        embeddings = []
        for text in texts:
            payload = {"model": model, "prompt": text}
            try:
                resp = requests.post(url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "embedding" in data:
                    embeddings.append(data["embedding"])
                else:
                    print(f"Unexpected API response: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
                    return []
            except Exception as e:
                print(f"Error embedding text via HTTP: {e}")
                return []
        return embeddings

    def generate(self, system: str, prompt: str, model: Optional[str] = None, max_tokens: int = 1024, conversation_history: Optional[List[dict]] = None) -> str:
        model = model or LLM_MODEL
        
        # Build messages list
        messages = [{"role": "system", "content": system}]
        
        # Add conversation history if provided
        if conversation_history:
            for interaction in conversation_history:
                messages.append({"role": "user", "content": interaction["question"]})
                messages.append({"role": "assistant", "content": interaction["answer"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        if self.use_pkg:
            resp = self._ollama_pkg.chat(model=model, messages=messages)
            # depending on package version
            return getattr(resp, "content", str(resp))

        # HTTP fallback: /api/chat endpoint
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # try to extract text
        if isinstance(data, dict):
            # Ollama /api/chat response format
            if "message" in data:
                return data["message"].get("content", "")
            if "choices" in data and data["choices"]:
                c = data["choices"][0]
                # openai-like
                return c.get("message", {}).get("content", c.get("text", ""))
            if "text" in data:
                return data["text"]
        return str(data)


_client = OllamaClient()


def embed(texts: List[str]) -> List[List[float]]:
    return _client.embed(texts)


def generate(system: str, prompt: str, conversation_history: Optional[List[dict]] = None) -> str:
    return _client.generate(system, prompt, conversation_history=conversation_history)
