<div align="center">

# ✦ Airbnb AI Concierge

**A production-quality RAG chatbot that understands natural language travel intent,  
retrieves semantically matched listings, and responds through Gemini AI.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat-square)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

![Airbnb AI Concierge Demo](https://placehold.co/900x480/FF385C/white?text=Airbnb+AI+Concierge+Demo&font=montserrat)

*"Find a romantic villa in Tuscany with a pool, under $400 a night"*  
→ Semantic search matches vibe, Gemini crafts the perfect recommendation.

</div>

---

## What This Is

**Airbnb AI Concierge** is a full-stack Retrieval-Augmented Generation (RAG) application that replaces keyword search with conversational AI. Instead of filtering by checkbox, guests describe their dream stay in plain language — and the system understands what they *mean*, not just what they typed.

This project demonstrates three things that matter in modern AI product engineering:

| Capability | What it shows |
|---|---|
| **Semantic retrieval** | Using vector embeddings + cosine similarity to match *intent*, not keywords |
| **RAG orchestration** | Grounding an LLM in real structured data to prevent hallucination |
| **Full-stack AI integration** | A polished, mobile-responsive UI that feels like a real product |

> Built as a portfolio project to explore applied AI architecture patterns used at scale in search and recommendation systems.

---

## RAG Architecture

```
                        USER QUERY
                            │
                            │  "cozy cabin near a lake, fireplace, no WiFi needed"
                            ▼
               ┌────────────────────────┐
               │   React Frontend       │
               │   (Vite + Tailwind)    │
               └───────────┬────────────┘
                           │  POST /chat  { message }
                           ▼
               ┌────────────────────────┐
               │   FastAPI Backend      │
               └───────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  SentenceTransformer    │  all-MiniLM-L6-v2
              │  (Embed query → vec)    │  384-dim dense vector
              └────────────┬────────────┘
                           │  query vector
                           ▼
              ┌────────────────────────┐
              │       ChromaDB         │  cosine similarity search
              │  (50 listing vectors)  │──────────────────────────┐
              └────────────────────────┘                          │
                           │  top-3 listings                      │
                           │  (id, metadata, score)               │ (pre-embedded at startup)
                           ▼                                      │
              ┌────────────────────────┐               ┌──────────▼──────────┐
              │   Prompt Builder       │               │  listings.json       │
              │   (system + context)   │               │  vibe_descriptions   │
              └────────────┬───────────┘               └─────────────────────┘
                           │  structured prompt
                           ▼
              ┌────────────────────────┐
              │     Gemini 2.0 Flash   │  Google GenAI API
              │   (generate reply)     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   ChatResponse          │
              │   { reply, listings[] } │
              └────────────┬────────────┘
                           │
                           ▼
               ┌────────────────────────┐
               │   Chat UI + Listing    │
               │   Cards rendered       │
               └────────────┬────────────┘
```

### Why RAG and not fine-tuning?

| Approach | Trade-offs |
|---|---|
| **Fine-tuning** | Expensive, data-hungry, stale the moment inventory changes |
| **Prompt stuffing** | All 50 listings × token cost = slow and expensive at scale |
| **RAG (this project)** | Fast, cheap, inventory updates without retraining, answers grounded in real data |

RAG is the correct architecture for any search-over-inventory problem — including Airbnb's own personalization stack.

---

## Tech Stack

### Backend
| Package | Role |
|---|---|
| **FastAPI** | Async REST API framework with auto-generated OpenAPI docs |
| **Uvicorn** | ASGI server (production-ready with `--workers` flag) |
| **sentence-transformers** | Local embedding model — no OpenAI dependency, runs on CPU |
| **ChromaDB** | Lightweight vector database, in-memory for dev, disk-persist for prod |
| **google-genai** | Official Python SDK for the Gemini API |
| **Pydantic v2** | Request/response validation with zero boilerplate |

### Frontend
| Package | Role |
|---|---|
| **React 18** | UI framework with hooks-only architecture |
| **Vite** | Sub-second HMR dev server, optimized production bundles |
| **Tailwind CSS v3** | Utility-first styling with custom Airbnb color tokens |
| **Lucide React** | Consistent, accessible icon set |

### Embedding Model
`all-MiniLM-L6-v2` was chosen deliberately:
- **22M parameters** — fast on CPU, no GPU needed
- **384-dimensional** vectors — compact, low memory footprint
- **MTEB-benchmarked** for semantic similarity tasks
- Runs entirely **locally** — no API calls, no latency, no cost per embedding

---

## Project Structure

```
airbnb/
│
├── 🐍 Backend
│   ├── main.py              # FastAPI app — startup, routes, RAG pipeline
│   ├── requirements.txt     # Python dependencies
│   ├── listings.json        # 50 mock listings (data source)
│   └── .env                 # Secrets (git-ignored)
│
├── ⚛️  Frontend
│   ├── index.html           # Vite entry point
│   ├── vite.config.js
│   ├── tailwind.config.js   # Custom Airbnb color palette + animations
│   ├── postcss.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx         # React root
│       ├── App.jsx          # Chat UI, listing cards, quick chips
│       └── index.css        # Tailwind directives + custom component classes
│
└── 📄 Docs
    ├── README.md
    └── .env (just to show create it to use locally)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1 — Clone

```bash
git clone https://github.com/your-username/airbnb-ai-concierge.git
cd airbnb-ai-concierge
```

### 2 — Configure environment

```bash
# Create .env file, and
# Open .env and set your GEMINI_API_KEY
```

### 3 — Backend

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Load env vars and start the API server
# Windows (PowerShell)
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
    [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
  }
}

# macOS/Linux
export $(grep -v '^#' .env | xargs)

uvicorn main:app --reload --port 8000
```

The first run downloads the embedding model (~90 MB, one-time only).  
Swagger docs live at **http://localhost:8000/docs**.

### 4 — Frontend

```bash
# In a second terminal
npm install
npm run dev
```

Open **http://localhost:3000** → start chatting. 🎉

---

## API Reference

### `POST /chat`

```json
// Request
{ "message": "I want a beachfront villa in the Maldives with a butler" }

// Response
{
  "reply": "For an ultra-luxurious Maldivian escape...",
  "listings": [
    {
      "id": "abn-019",
      "name": "Maldives Overwater Glass-Floor Bungalow",
      "location": "North Malé Atoll, Maldives",
      "type": "villa",
      "price_per_night": 850,
      "rating": 4.99,
      "amenities": ["Glass Floor", "Private Deck", "Butler Service", ...],
      "vibe_description": "..."
    }
  ]
}
```
<!-- 
### `GET /listings`

Returns all 50 listings — useful for building a catalogue view or debugging retrieval.

### `GET /`

Health check. Returns `{"status": "ok"}`.

---

## Extending This Project

### 🗃️ Swap in Real Listings

Replace `listings.json` with live data from any source:

```python
# Example: fetch from a database instead of a JSON file
import psycopg2

def load_listings():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, location, type, price_per_night, rating, amenities, vibe_description FROM listings")
    return [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
```

For production-scale inventory (millions of listings), switch ChromaDB to a managed vector store:

| Scale | Recommended Store |
|---|---|
| < 100K listings | ChromaDB (disk-persist mode) |
| 100K – 10M | **Pinecone**, **Weaviate**, or **pgvector** |
| 10M+ | **Vertex AI Vector Search**, **Elasticsearch** with dense vectors |

### 🎙️ Add Voice Input

The Web Speech API requires ~10 lines of JavaScript. Add this to `App.jsx`:

```jsx
const startListening = () => {
  const recognition = new window.webkitSpeechRecognition()
  recognition.lang = 'en-US'
  recognition.onresult = (e) => setInput(e.results[0][0].transcript)
  recognition.start()
}
// Then bind a mic button to startListening()
```

For voice *output*, pipe `data.reply` through the Web Speech Synthesis API or use [ElevenLabs](https://elevenlabs.io) for studio-quality TTS.

### 🌍 Add Multi-language Support

Swap the system prompt language detection and pass `Accept-Language` headers:

```python
# In _build_system_prompt(), detect and respond in the user's language:
system_prompt = f"Respond in the same language the user writes in. {base_prompt}"
```

Gemini natively handles 50+ languages with no additional configuration.

### ☁️ Deploy to Cloud

#### Backend — Railway or Render (easiest)

```bash
# railway.toml
[build]
builder = "nixpacks"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
```

Set `GEMINI_API_KEY` in the platform's environment variables dashboard.

#### Backend — Google Cloud Run (production-grade)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
gcloud run deploy airbnb-api \
  --source . \
  --region us-central1 \
  --set-env-vars ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --allow-unauthenticated
```

#### Frontend — Vercel (one command)

```bash
npm install -g vercel
vercel --prod
```

Update `API_URL` in `src/App.jsx` to your deployed backend URL before building.

#### Persist ChromaDB across restarts

Switch from in-memory to disk-backed Chroma in `main.py`:

```python
# Replace:
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# With:
chroma_client = chromadb.PersistentClient(path="./chroma_db")
```

Then mount a persistent volume in your cloud platform to survive redeploys.

### 🔒 Add Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/chat")
async def chat(body: ChatRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # validate token against your auth provider (Clerk, Auth0, Supabase, etc.)
```

---

## Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| **Local embedding model** | No API cost per query; deterministic; no cold-start latency |
| **ChromaDB in-memory** | Zero infrastructure for development; swap path to Pinecone is one import change |
| **Top-3 retrieval** | Balances context richness vs. Claude token cost; tunable via `TOP_K` constant |
| **Cosine similarity** | Normalizes for vector magnitude; better than L2 for text embeddings |
| **`vibe_description` as embedding target** | Richer semantic signal than name/location alone; encodes feel, not just facts |
| **Pydantic v2 response models** | Type-safe contracts between backend and frontend; auto-validates Claude's context data |

---

## License

MIT © 2024 — free to use, fork, and build upon.

---

<div align="center">

Built with ❤️ as a demonstration of applied AI engineering.

*If you're from the Airbnb engineering team — let's talk.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF385C?style=flat-square&logo=airbnb&logoColor=white)](https://your-portfolio.dev)

</div> -->
