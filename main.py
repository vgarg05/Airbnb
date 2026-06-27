"""
Airbnb AI Chat Backend
======================
FastAPI server that:
  1. Loads 50 mock Airbnb listings from listings.json on startup
  2. Embeds vibe_descriptions with sentence-transformers (all-MiniLM-L6-v2)
  3. Stores embeddings in ChromaDB (in-memory)
  4. Exposes POST /chat — does semantic search then calls Gemini to generate a response
  5. Exposes GET  /chat/stream — same as above but streams tokens via SSE
"""

import json
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException, Query, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LISTINGS_PATH = Path(__file__).parent / "listings.json"
CHROMA_COLLECTION = "airbnb_listings"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"
# Fallback chain — tried in order on 503 (overloaded) or 429 (rate-limited).
# Each model has its own free-tier quota bucket, so trying the next one
# often succeeds even when the first is exhausted.
GEMINI_FALLBACK_MODELS = [
    # "gemini-2.0-flash",        # primary
    "gemini-2.5-flash-lite",   # lighter quota bucket
    # "gemini-1.5-flash",        # previous generation — separate quota
    # "gemini-1.5-flash-8b",     # smallest / most permissive free tier
]
TOP_K = 3  # number of listings to retrieve per query

# ---------------------------------------------------------------------------
# Global singletons (populated during startup)
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer | None = None
_chroma_collection: chromadb.Collection | None = None
_listings_by_id: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and once at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, embed listings, populate ChromaDB."""
    global _embed_model, _chroma_collection, _listings_by_id

    # 1. Load listings -------------------------------------------------------
    log.info("Loading listings from %s …", LISTINGS_PATH)
    with open(LISTINGS_PATH, "r", encoding="utf-8") as f:
        listings: list[dict[str, Any]] = json.load(f)
    _listings_by_id = {listing["id"]: listing for listing in listings}
    log.info("Loaded %d listings.", len(listings))

    # 2. Load embedding model ------------------------------------------------
    log.info("Loading sentence-transformer model '%s' …", EMBED_MODEL)
    _embed_model = SentenceTransformer(EMBED_MODEL)
    log.info("Embedding model loaded.")

    # 3. Set up ChromaDB (ephemeral / in-memory) ------------------------------
    log.info("Initialising ChromaDB …")
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    _chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Embed & store --------------------------------------------------------
    log.info("Embedding listings with rich metadata …")
    texts = [
        f"Name: {listing['name']}\n"
        f"Location: {listing['location']}\n"
        f"Type: {listing['type']}\n"
        f"Amenities: {', '.join(listing['amenities'])}\n"
        f"Vibe: {listing['vibe_description']}"
        for listing in listings
    ]
    ids = [listing["id"] for listing in listings]

    # Metadata stored alongside each vector (for display in the response)
    metadatas = [
        {
            "name": listing["name"],
            "location": listing["location"],
            "type": listing["type"],
            "price_per_night": listing["price_per_night"],
            "rating": listing["rating"],
            "amenities": ", ".join(listing["amenities"]),
            "vibe_description": listing["vibe_description"],
        }
        for listing in listings
    ]

    embeddings = _embed_model.encode(texts, show_progress_bar=True).tolist()

    _chroma_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    log.info("ChromaDB populated with %d vectors.", len(ids))

    yield  # ← application runs here

    # Shutdown (nothing to clean up for in-memory Chroma)
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Airbnb AI Chat API",
    description="Semantic search over Airbnb listings powered by ChromaDB + Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


class ListingSummary(BaseModel):
    id: str
    name: str
    location: str
    type: str
    price_per_night: float
    rating: float
    amenities: list[str]
    vibe_description: str


class ChatResponse(BaseModel):
    reply: str
    listings: list[ListingSummary]


# ---------------------------------------------------------------------------
# Helper — build the Gemini system prompt
# ---------------------------------------------------------------------------
def _build_system_prompt(retrieved: list[dict[str, Any]]) -> str:
    listings_text = ""
    for i, listing in enumerate(retrieved, start=1):
        amenities = listing["amenities"]
        if isinstance(amenities, str):
            amenities = amenities.split(", ")
        listings_text += (
            f"\n### Listing {i}\n"
            f"**Name:** {listing['name']}\n"
            f"**Location:** {listing['location']}\n"
            f"**Type:** {listing['type'].capitalize()}\n"
            f"**Price:** ${listing['price_per_night']} / night\n"
            f"**Rating:** {listing['rating']} ⭐\n"
            f"**Amenities:** {', '.join(amenities)}\n"
            f"**Vibe:** {listing['vibe_description']}\n"
        )

    return f"""You are a warm, knowledgeable Airbnb travel concierge. \
A guest has asked a question and you have retrieved the top matching listings \
from the property catalogue using semantic search.

Your job:
1. Answer the guest's question in a friendly, conversational tone.
2. Reference the retrieved listings naturally — highlight the most relevant \
details (location, vibe, price, standout amenities).
3. If multiple listings suit different aspects of the question, compare them briefly.
4. Keep your response concise but evocative — paint a picture, not a data sheet.
5. End with a helpful tip or follow-up suggestion.

Retrieved Listings:
{listings_text}"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """Health check — confirms the API is alive."""
    return {"status": "ok", "message": "Airbnb AI Chat API is running."}


@app.get("/listings", tags=["Listings"])
async def get_all_listings():
    """Return all listings (useful for debugging / frontend catalogue view)."""
    return list(_listings_by_id.values())


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(body: ChatRequest, x_user_api_key: str | None = Header(None)):
    """
    Accepts a natural-language query, retrieves the top 3 semantically similar
    listings from ChromaDB, then asks Claude to generate a helpful response.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if _embed_model is None or _chroma_collection is None:
        raise HTTPException(
            status_code=503, detail="Server not ready — embeddings not loaded yet."
        )

    # ── 1. Embed the user query ──────────────────────────────────────────────
    log.info("Embedding query: %r", body.message)
    query_embedding = _embed_model.encode([body.message]).tolist()[0]

    # ── 2. Semantic search in ChromaDB ───────────────────────────────────────
    log.info("Querying ChromaDB for top %d results …", TOP_K)
    results = _chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["metadatas", "documents", "distances"],
    )

    retrieved_metadatas: list[dict[str, Any]] = results["metadatas"][0]
    retrieved_ids: list[str] = results["ids"][0]

    log.info(
        "Retrieved IDs: %s (distances: %s)",
        retrieved_ids,
        [f"{d:.4f}" for d in results["distances"][0]],
    )

    # ── 3. Build full listing objects for the response ───────────────────────
    listings_for_response: list[ListingSummary] = []
    for listing_id, meta in zip(retrieved_ids, retrieved_metadatas):
        amenities_raw = meta.get("amenities", "")
        amenities_list = (
            amenities_raw.split(", ")
            if isinstance(amenities_raw, str)
            else amenities_raw
        )
        listings_for_response.append(
            ListingSummary(
                id=listing_id,
                name=meta["name"],
                location=meta["location"],
                type=meta["type"],
                price_per_night=float(meta["price_per_night"]),
                rating=float(meta["rating"]),
                amenities=amenities_list,
                vibe_description=meta["vibe_description"],
            )
        )

    # ── 4. Call Gemini (with model fallback + error handling) ─────────────────────
    api_key = x_user_api_key.strip() if (x_user_api_key and x_user_api_key.strip()) else os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key found. Please provide a valid Gemini API key in settings.",
        )

    gemini_client = genai.Client(api_key=api_key)
    system_prompt  = _build_system_prompt(retrieved_metadatas)

    reply_text: str | None = None
    last_error:  Exception | None = None

    for model_name in GEMINI_FALLBACK_MODELS:
        try:
            log.info("Calling Gemini model '%s' …", model_name)
            gemini_response = gemini_client.models.generate_content(
                model=model_name,
                contents=body.message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                    temperature=0.7,
                ),
            )
            reply_text = gemini_response.text
            log.info("Gemini '%s' responded (%d chars).", model_name, len(reply_text))
            break  # success — stop trying fallbacks

        except genai_errors.ServerError as exc:
            # 503 overloaded — try next model in the fallback chain
            log.warning("Gemini '%s' unavailable (503), trying next fallback …", model_name)
            last_error = exc
            continue

        except genai_errors.ClientError as exc:
            exc_str = str(exc)
            is_rate_limited = "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str

            if is_rate_limited:
                # 429 quota/rate-limit — each model has its own bucket, so
                # try the next one in the fallback chain before giving up.
                log.warning(
                    "Gemini '%s' quota exhausted (429), trying next fallback …", model_name
                )
                last_error = exc
                continue

            # Any other 4xx (401 bad key, 400 bad request, etc.) — abort immediately
            log.error("Gemini client error on '%s': %s", model_name, exc)
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Gemini API returned an error: {exc_str[:300]}"
                    if len(exc_str) > 300 else f"Gemini API returned an error: {exc_str}"
                ),
            ) from exc

    if reply_text is None:
        last_str = str(last_error)
        is_quota  = "RESOURCE_EXHAUSTED" in last_str or "429" in last_str
        log.error("All Gemini fallback models exhausted. is_quota=%s", is_quota)
        if is_quota:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Your Gemini free-tier quota is exhausted across all fallback models. "
                    "Options: (1) wait ~1 minute and retry, "
                    "(2) enable billing at https://console.cloud.google.com/billing, "
                    "(3) generate a new API key at https://aistudio.google.com/app/apikey."
                ),
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is currently unavailable across all fallback models. "
                "Please wait a moment and try again."
            ),
        )

    return ChatResponse(reply=reply_text, listings=listings_for_response)


# ---------------------------------------------------------------------------
# Streaming endpoint  GET /chat/stream?message=...
# ---------------------------------------------------------------------------
async def _stream_gemini(
    user_message: str,
    retrieved_metadatas: list[dict[str, Any]],
    listings_for_response: list["ListingSummary"],
    api_key: str,
) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted strings:
      - 'data: {"type":"token","value":"..."}\n\n'  for each streamed token
      - 'data: {"type":"listings","value":[...]}\n\n' once at the end
      - 'data: {"type":"done"}\n\n'                 to signal completion
    """
    gemini_client = genai.Client(api_key=api_key)
    system_prompt  = _build_system_prompt(retrieved_metadatas)

    streamed = False
    last_error: Exception | None = None

    for model_name in GEMINI_FALLBACK_MODELS:
        try:
            log.info("[stream] Calling Gemini model '%s' …", model_name)
            response_stream = gemini_client.models.generate_content_stream(
                model=model_name,
                contents=user_message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                    temperature=0.7,
                ),
            )

            for chunk in response_stream:
                token = chunk.text or ""
                if token:
                    payload = json.dumps({"type": "token", "value": token})
                    yield f"data: {payload}\n\n"

            streamed = True
            break  # success

        except (genai_errors.ServerError, genai_errors.ClientError) as exc:
            exc_str = str(exc)
            is_retryable = (
                "503" in exc_str or
                "unavailable" in exc_str.lower() or
                "429" in exc_str or
                "RESOURCE_EXHAUSTED" in exc_str
            )
            if is_retryable:
                log.warning("[stream] Gemini '%s' failed, trying next fallback …", model_name)
                last_error = exc
                continue
            # Non-retryable — send an error event and bail
            err_payload = json.dumps({"type": "error", "value": exc_str[:300]})
            yield f"data: {err_payload}\n\n"
            return

    if not streamed:
        err_msg = str(last_error) if last_error else "All Gemini models failed."
        err_payload = json.dumps({"type": "error", "value": err_msg[:300]})
        yield f"data: {err_payload}\n\n"
        return

    # Send the listings object as a final event
    listings_data = [l.model_dump() for l in listings_for_response]
    listings_payload = json.dumps({"type": "listings", "value": listings_data})
    yield f"data: {listings_payload}\n\n"

    # Signal stream end
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.get("/chat/stream", tags=["Chat"])
async def chat_stream(
    message: str = Query(..., min_length=1),
    x_user_api_key: str | None = Header(None),
):
    """
    SSE streaming endpoint.  The client sends the message as a query parameter
    and receives a stream of newline-delimited JSON events.
    """
    if _embed_model is None or _chroma_collection is None:
        raise HTTPException(
            status_code=503, detail="Server not ready — embeddings not loaded yet."
        )

    api_key = x_user_api_key.strip() if (x_user_api_key and x_user_api_key.strip()) else os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key found. Please provide a valid Gemini API key in settings.",
        )

    # Embed + retrieve (same as /chat)
    query_embedding = _embed_model.encode([message]).tolist()[0]
    results = _chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["metadatas", "documents", "distances"],
    )
    retrieved_metadatas: list[dict[str, Any]] = results["metadatas"][0]
    retrieved_ids: list[str] = results["ids"][0]

    listings_for_response: list[ListingSummary] = []
    for listing_id, meta in zip(retrieved_ids, retrieved_metadatas):
        amenities_raw = meta.get("amenities", "")
        amenities_list = (
            amenities_raw.split(", ") if isinstance(amenities_raw, str) else amenities_raw
        )
        listings_for_response.append(
            ListingSummary(
                id=listing_id,
                name=meta["name"],
                location=meta["location"],
                type=meta["type"],
                price_per_night=float(meta["price_per_night"]),
                rating=float(meta["rating"]),
                amenities=amenities_list,
                vibe_description=meta["vibe_description"],
            )
        )

    return StreamingResponse(
        _stream_gemini(message, retrieved_metadatas, listings_for_response, api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
