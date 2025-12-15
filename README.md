# 🎓 RAG Admission Officer Chatbot

An intelligent chatbot system that answers student questions about university admissions, fees, academics, and more using **Retrieval-Augmented Generation (RAG)** powered by **Ollama**, **Chroma**, and conversational memory.

---

## ✨ Features

- **🤖 AI-Powered Responses** - Uses Ollama's Llama 3.2 model to generate contextual answers
- **🎯 Category-Based Filtering** - Routes questions to relevant Q&A categories (Admissions, Fees, Academics, etc.)
- **💾 Semantic Search** - Finds the most relevant answers using vector embeddings
- **🧠 Conversation Memory** - Remembers the last 10 interactions for contextual follow-ups
- **📚 Easy Q&A Management** - Store all questions and answers in a simple JSON file
- **⚡ Fast Retrieval** - Uses Chroma vector database with HNSW indexing for quick searches
- **🔒 Clean Output** - Automatically removes internal metadata from responses

---

## 📋 Prerequisites

### Required
- **Python 3.8+**
- **Ollama** (local LLM server) - [Download](https://ollama.ai)
- **Ollama Models**:
  - `all-minilm` (embedding model)
  - `llama3.2` (chat model)

### Installation
```bash
# Install Ollama and pull required models
ollama pull all-minilm
ollama pull llama3.2

# Start Ollama server (runs in background on port 11434)
ollama serve
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
# Create Python virtual environment
python -m venv .venv

# Activate venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy and edit configuration (optional)
copy .env.example .env
# Or use defaults (Ollama on localhost:11434)
```

### 3. Ingest Q&A Data
```bash
python -m src.ingest
```
This reads `data.json` and creates embeddings in `chroma_db/`

### 4. Run the Chatbot
```bash
python -m src.query
```

Then type your questions:
```
Question> What programs do you offer?
🏷️ Detected category: Admissions
--- ANSWER ---
[Bot responds with relevant information]
[Memory: 1/10 interactions stored]

Question> How much is the tuition?
🏷️ Detected category: Fees
--- ANSWER ---
[Bot responds with fee information, remembering previous context]
[Memory: 2/10 interactions stored]
```

**Commands:**
- Type your question → Get an answer
- Type `clear` → Reset conversation memory
- Type `exit` or `quit` → Close the chatbot

---

## 📁 Project Structure

```
RAG - Admission Officer/
├── src/                          # Core application code
│   ├── ingest.py                # Reads data.json, creates embeddings
│   ├── query.py                 # Interactive chatbot interface
│   ├── ollama_client.py         # Ollama API wrapper
│   └── utils.py                 # Helper functions & conversation memory
├── data.json                    # ⭐ Q&A dataset (questions, answers, categories)
├── Categories.txt               # Category definitions & keywords
├── chroma_db/                   # Vector database (auto-created)
├── rag_results_llama/           # Query logs (auto-created)
├── .env.example                 # Configuration template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔧 Configuration

Edit `.env` to customize (or use defaults):

```env
# Ollama Server
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434

# Model Names
EMBED_MODEL=all-minilm           # For text → vector embeddings
LLM_MODEL=llama3.2               # For generating answers

# Chroma Database Path
CHROMA_PERSIST_DIR=./chroma_db
```

---

## 📊 Q&A Data Format

`data.json` should be a JSON array with this structure:

```json
[
  {
    "id": 1,
    "category": "Admissions",
    "question": "What are the admission requirements?",
    "answer": "You need a high school diploma, a minimum GPA of 3.0..."
  },
  {
    "id": 2,
    "category": "Fees",
    "question": "What is the tuition cost?",
    "answer": "Annual tuition is $15,000..."
  },
  {
    "id": 3,
    "category": "Emails",
    "question": "What is the admissions email?",
    "answer": "You can reach us at admissions@university.edu"
  }
]
```

### Supported Categories
- **Admissions** - Application requirements, deadlines, acceptance
- **Fees** - Tuition, costs, payment plans
- **Academics** - GPA, grades, course information
- **Academic Advising** - Advisors, course selection, major planning
- **IT & Systems** - Moodle, portals, technical support
- **Emails** - Contact information, email addresses
- **General** - Any other questions

---

## 🔄 How It Works

### Ingestion Pipeline
```
data.json 
  ↓
[Extract Q&A pairs with categories]
  ↓
[Embed text using Ollama's all-minilm]
  ↓
[Store vectors + metadata in Chroma]
  ↓
chroma_db/ (ready for queries)
```

### Query Pipeline
```
User Question
  ↓
[Detect category from keywords]
  ↓
[Embed question using all-minilm]
  ↓
[Retrieve top 4 similar Q&A from Chroma]
  ↓ (apply category filter if detected)
[Format context + conversation history]
  ↓
[Send to Llama 3.2 with system prompt]
  ↓
[Clean output (remove metadata)]
  ↓
[Save to logs, update memory]
  ↓
Display Answer to User
```
---

**Last Updated:** December 2025  
**Version:** 1.0

