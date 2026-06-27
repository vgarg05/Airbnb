import json
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment or .env file")

LISTINGS_PATH = Path(__file__).parent / "listings.json"
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
CHROMA_COLLECTION = "airbnb_listings"
EMBED_MODEL = "gemini-embedding-2"

def build():
    # 1. Load listings
    log.info("Loading listings from %s...", LISTINGS_PATH)
    with open(LISTINGS_PATH, "r", encoding="utf-8") as f:
        listings = json.load(f)
    log.info("Loaded %d listings.", len(listings))

    # 2. Format documents to embed
    texts = [
        f"Name: {listing['name']}\n"
        f"Location: {listing['location']}\n"
        f"Type: {listing['type']}\n"
        f"Amenities: {', '.join(listing['amenities'])}\n"
        f"Vibe: {listing['vibe_description']}"
        for listing in listings
    ]
    ids = [listing["id"] for listing in listings]
    
    # Metadata for display
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

    # 3. Call Gemini to get embeddings in batch
    log.info("Generating embeddings using Gemini API (%s)...", EMBED_MODEL)
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    from google.genai import types
    import time
    embeddings = []
    chunk_size = 40
    for i in range(0, len(texts), chunk_size):
        if i > 0:
            log.info("Sleeping 65 seconds to avoid API rate limits...")
            time.sleep(65)
        chunk_texts = texts[i:i+chunk_size]
        log.info("Embedding chunk %d to %d...", i, i + len(chunk_texts))
        res = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in chunk_texts],
        )
        chunk_embeddings = [e.values for e in res.embeddings]
        embeddings.extend(chunk_embeddings)

    log.info("Generated %d embeddings.", len(embeddings))

    # 4. Save to persistent Chroma DB
    log.info("Initializing persistent ChromaDB at %s...", CHROMA_DB_PATH)
    
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH), settings=Settings(anonymized_telemetry=False))
    
    # Try deleting the collection to start fresh
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
        log.info("Deleted existing collection '%s' to rebuild.", CHROMA_COLLECTION)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    log.info("Adding listings to Chroma collection...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    log.info("ChromaDB persistent database successfully populated and saved!")

if __name__ == "__main__":
    build()
