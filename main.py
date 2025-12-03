"""
JARVIS Backend API - Railway Deployment
========================================
API backend pour le chatbot JARVIS avec acces a la memoire permanente.

Endpoints:
- POST /api/chat - Chat avec JARVIS (memoire + Ollama Cloud)
- GET /api/memory/search - Recherche semantique dans la memoire
- GET /api/conversations - Liste des conversations
- GET /api/conversations/{id} - Detail d'une conversation
- GET /api/health - Health check

Author: JARVIS System
"""

import os
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import psycopg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://api.ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# JARVIS System Prompt
JARVIS_SYSTEM_PROMPT = """Tu es JARVIS, un assistant IA personnel avance cree pour Chris.

## Tes caracteristiques:
- Tu reponds toujours en francais
- Tu es precis, concis et utile
- Tu as acces a la memoire permanente JARVIS (PostgreSQL, Neo4j, Qdrant)
- Tu peux analyser du code, des images, et aider sur tous types de projets
- Tu appelles Chris par son prenom, jamais "monsieur"

## Ton style:
- Reponses structurees avec markdown quand approprie
- Code formate dans des blocs ```
- Listes a puces pour les enumerations
- Direct et efficace, pas de bavardage inutile

## Contexte actuel:
- Date: {date}
- Systeme: JARVIS Backend (Railway Cloud)
- Backend: Ollama Cloud
"""


# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-oss:120b"
    images: Optional[List[str]] = None
    history: Optional[List[dict]] = None
    use_memory: bool = True


class ChatResponse(BaseModel):
    response: str
    model: str
    memory_context: Optional[str] = None


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 5


# Database connection
def get_db_connection():
    """Get PostgreSQL connection."""
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    return psycopg.connect(DATABASE_URL)


async def search_memory(query: str, limit: int = 5) -> List[dict]:
    """Search conversations in PostgreSQL for relevant context."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Search in conversation messages
                cur.execute("""
                    SELECT cm.content, cm.role, c.title, c.created_at
                    FROM conversation_messages cm
                    JOIN conversations c ON cm.conversation_id = c.id
                    WHERE cm.content ILIKE %s
                    ORDER BY c.created_at DESC
                    LIMIT %s
                """, (f"%{query}%", limit))

                results = []
                for row in cur.fetchall():
                    results.append({
                        "content": row[0][:500],  # Truncate
                        "role": row[1],
                        "title": row[2],
                        "date": row[3].isoformat() if row[3] else None
                    })
                return results
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        return []


async def get_memory_context(query: str) -> str:
    """Get relevant memory context for the chat."""
    memories = await search_memory(query, limit=3)
    if not memories:
        return ""

    context_parts = ["## Contexte de la memoire JARVIS:\n"]
    for mem in memories:
        context_parts.append(f"- [{mem['title']}] ({mem['role']}): {mem['content'][:200]}...")

    return "\n".join(context_parts)


async def call_ollama_cloud(
    message: str,
    model: str,
    system_prompt: str,
    history: List[dict] = None,
    images: List[str] = None
) -> str:
    """Call Ollama Cloud API."""

    messages = [{"role": "system", "content": system_prompt}]

    # Add history
    if history:
        for msg in history[-10:]:  # Last 10 messages
            if msg.get("role") != "system":
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

    # Add current message
    current_msg = {"role": "user", "content": message}
    if images:
        # Extract base64 from data URLs
        current_msg["images"] = [
            img.split(",")[1] if img.startswith("data:") else img
            for img in images
        ]
    messages.append(current_msg)

    # Call Ollama Cloud
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_CLOUD_HOST}/api/chat",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 4096
                    }
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                # Try generate endpoint as fallback
                response = await client.post(
                    f"{OLLAMA_CLOUD_HOST}/api/generate",
                    headers=headers,
                    json={
                        "model": model,
                        "prompt": message,
                        "system": system_prompt,
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    return response.json().get("response", "")

                raise HTTPException(status_code=500, detail=f"Ollama Cloud error: {response.text}")

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Ollama Cloud timeout")
        except Exception as e:
            logger.error(f"Ollama Cloud error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("JARVIS Backend starting...")
    logger.info(f"Database: {'configured' if DATABASE_URL else 'NOT configured'}")
    logger.info(f"Ollama Cloud: {OLLAMA_CLOUD_HOST}")
    yield
    logger.info("JARVIS Backend shutting down...")


app = FastAPI(
    title="JARVIS Backend API",
    description="Backend API pour le chatbot JARVIS avec memoire permanente",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "JARVIS Backend API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/api/chat", "/api/memory/search", "/api/conversations", "/api/health"]
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ollama_cloud": bool(OLLAMA_API_KEY),
            "postgresql": bool(DATABASE_URL),
            "neo4j": bool(NEO4J_URI),
            "qdrant": bool(QDRANT_URL)
        }
    }
    return status


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with JARVIS."""

    # Get memory context if enabled
    memory_context = ""
    if request.use_memory and DATABASE_URL:
        memory_context = await get_memory_context(request.message)

    # Build system prompt
    system_prompt = JARVIS_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"))
    if memory_context:
        system_prompt += f"\n\n{memory_context}"

    # Call Ollama Cloud
    response_text = await call_ollama_cloud(
        message=request.message,
        model=request.model,
        system_prompt=system_prompt,
        history=request.history,
        images=request.images
    )

    return ChatResponse(
        response=response_text,
        model=request.model,
        memory_context=memory_context if memory_context else None
    )


@app.post("/api/memory/search")
async def search_memory_endpoint(request: MemorySearchRequest):
    """Search in JARVIS memory."""
    results = await search_memory(request.query, request.limit)
    return {"results": results, "count": len(results)}


@app.get("/api/conversations")
async def get_conversations(limit: int = 20, offset: int = 0):
    """Get list of conversations."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, title, created_at, updated_at, message_count, status
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))

                conversations = []
                for row in cur.fetchall():
                    conversations.append({
                        "id": row[0],
                        "title": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "updated_at": row[3].isoformat() if row[3] else None,
                        "message_count": row[4],
                        "status": row[5]
                    })

                return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    """Get conversation with messages."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get conversation
                cur.execute("""
                    SELECT id, title, created_at, updated_at, message_count, status
                    FROM conversations WHERE id = %s
                """, (conversation_id,))

                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Conversation not found")

                conversation = {
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2].isoformat() if row[2] else None,
                    "updated_at": row[3].isoformat() if row[3] else None,
                    "message_count": row[4],
                    "status": row[5]
                }

                # Get messages
                cur.execute("""
                    SELECT role, content, created_at
                    FROM conversation_messages
                    WHERE conversation_id = %s
                    ORDER BY created_at ASC
                """, (conversation_id,))

                messages = []
                for msg_row in cur.fetchall():
                    messages.append({
                        "role": msg_row[0],
                        "content": msg_row[1],
                        "timestamp": msg_row[2].isoformat() if msg_row[2] else None
                    })

                conversation["messages"] = messages
                return conversation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
