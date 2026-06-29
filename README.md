# RAG Study Companion

An AI-powered study assistant that lets you upload PDF study materials and ask natural language questions about them. Built with a production-grade RAG (Retrieval-Augmented Generation) pipeline featuring advanced retrieval strategies, conversational memory, pipeline orchestration, and a real-time chat interface.

---

## Features

- **PDF ingestion** — Upload any PDF and have it chunked, embedded, and stored automatically
- **Semantic search** — Questions are matched against your documents using vector similarity
- **Advanced retrieval** — Sentence-window and auto-merging retrieval strategies with a query router that picks the right one per question
- **Conversational memory** — Follow-up questions work naturally without repeating context
- **Pipeline orchestration** — Dagster-powered ingestion pipeline with a visual UI and run history
- **Custom evaluation** — LLM-as-judge scoring across faithfulness, answer relevancy, and context precision
- **Streaming chat UI** — Next.js frontend with real-time responses and PDF upload

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (HuggingFace) |
| Vector Store | pgvector (PostgreSQL) |
| RAG Framework | LangChain (LCEL) |
| Pipeline Orchestration | Dagster |
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js 14 + Tailwind CSS |
| Containerization | Docker |

---

## Project Structure

```
rag-study-companion/
├── ingest.py               # PDF loading, chunking, embedding, pgvector storage
├── retriever.py            # Basic semantic retrieval from pgvector
├── advanced_retriever.py   # Sentence-window + auto-merging retrieval + query router
├── chain.py                # LangChain LCEL chain: retriever + LLM + prompt
├── memory_chain.py         # Conversational memory with sliding window
├── pipeline.py             # Dagster pipeline with 3 observable assets
├── evaluate.py             # Custom RAG evaluation (faithfulness, relevancy, precision)
├── api.py                  # FastAPI backend with /ask and /upload endpoints
├── .env                    # Environment variables (not committed)
├── requirements.txt        # Python dependencies
└── rag-frontend/           # Next.js frontend
    └── app/
        └── page.tsx        # Chat UI with streaming and PDF upload
```

---

## Architecture

```
User Question
     │
     ▼
Query Router ──────────────────────────────────┐
     │                                         │
     ▼ (factual)                    ▼ (broad/summary)
Sentence-Window               Auto-Merging
Retriever                     Retriever
     │                             │
     └──────────┬──────────────────┘
                ▼
         pgvector DB
         (embeddings)
                │
                ▼
        Retrieved Chunks
                │
                ▼
     Conversational Memory
      (last 3 exchanges)
                │
                ▼
       GPT-3.5-turbo LLM
                │
                ▼
         Final Answer
                │
                ▼
     Next.js Chat Interface
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker Desktop
- OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-study-companion.git
cd rag-study-companion
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/studydb
```

### 4. Start the vector database

```bash
docker run -d \
  --name pgvector-db \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=studydb \
  -p 5432:5432 \
  ankane/pgvector
```

### 5. Ingest a PDF

Place any PDF in the project root and run:

```bash
python ingest.py
```

Or use the Dagster pipeline UI for a visual run:

```bash
dagster dev -f pipeline.py
# Open http://localhost:3000 and click Materialize all
```

### 6. Start the backend API

```bash
uvicorn api:app --reload --port 8000
```

### 7. Start the frontend

```bash
cd rag-frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## How to Use

1. Open `http://localhost:3000` in your browser
2. Type a question about your study materials and press Enter or click Send
3. To add new documents, click **Upload PDF** and select a file
4. Follow-up questions work naturally — the assistant remembers the last 3 exchanges

---

## Retrieval Strategies

### Sentence-Window Retrieval
Used for factual questions ("what is", "define", "explain"). Retrieves the matching chunk plus its neighboring chunks for richer context, reducing the chance of a truncated answer.

### Auto-Merging Retrieval
Used for broad questions ("summarize", "list all", "overview"). When multiple chunks from the same page are retrieved, they are merged into a single document to provide a more complete answer.

### Query Router
A keyword-based router automatically selects the best retrieval strategy based on the question type. Factual questions → sentence-window. Summary questions → auto-merging.

---

## Evaluation

Run the custom evaluation suite to score your RAG pipeline:

```bash
python evaluate.py
```

Outputs three metrics scored 0.0 to 1.0:

- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are the retrieved chunks relevant to the question?

---

## Restarting the Project

```bash
# 1. Start the database
docker start pgvector-db

# 2. Activate Python environment
source venv/bin/activate

# 3. Start the backend
uvicorn api:app --reload --port 8000

# 4. Start the frontend (new terminal)
cd rag-frontend && npm run dev
```

---

## Future Improvements

- Add user authentication
- Support multiple document collections per user
- Implement true streaming with Server-Sent Events
- Add re-ranking with a cross-encoder model
- Deploy to AWS/GCP with persistent volumes

---

## Author

Maninderjit Bhullar
Master of Computer Science — Arizona State University
