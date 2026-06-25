# =========================
# IMPORTS
# =========================
import os
from enum import Enum
from typing import List

from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel


def is_small_talk(query: str) -> bool:
    """Detect casual greetings and small talk."""
    normalized = query.strip().lower()
    if not normalized:
        return False

    small_talk_keywords = [
        "hello",
        "hi",
        "hey",
        "how are you",
        "how's it going",
        "whats up",
        "good morning",
        "good afternoon",
        "good evening",
        "thank you",
        "thanks",
        "bye",
        "goodbye",
        "how are you doing",
        "what's up"
    ]

    return any(keyword in normalized for keyword in small_talk_keywords)


def is_general_fallback(query: str) -> bool:
    """Allow general conversational or factual queries when no KB docs are found."""
    normalized = query.strip().lower()
    if not normalized:
        return False

    if is_small_talk(normalized):
        return True

    product_keywords = [
        "kuja",
        "product",
        "stock",
        "order",
        "van",
        "distributor",
        "warehouse",
        "replenish",
        "replenishment",
        "dashboard",
        "alert",
        "login",
        "sign up",
        "signup",
        "payment",
        "invoice",
        "delivery",
        "customer",
        "return",
        "cancel",
        "route",
        "sales",
        "app",
        "platform"
    ]

    return not any(keyword in normalized for keyword in product_keywords)

# Pinecone
from pinecone import Pinecone

# LangChain
from langchain_pinecone import PineconeVectorStore


from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()
import asimov_config

# LangChain imports that depend on OpenAI env must come after Asimov config
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Kuja AI Support API"
)

# =========================
# PINECONE
# =========================
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# =========================
# EMBEDDINGS
# =========================
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# =========================
# LLM
# =========================
llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "openai/gpt-4o"),
    temperature=0
)

# =========================
# ENUMS
# =========================
class Platform(str, Enum):
    kujashop = "kujashop"
    kujaexpress = "kujaexpress"
    kujadrivers = "kujadrivers"
    kujaerp = "kujaerp"


class Role(str, Enum):
    driver = "driver"
    admin = "admin"
    distributor = "distributor"
    stockist = "stockist"
    bdr = "bdr"
    bulkbreaker = "bulkbreaker"
    pocs = "pocs"
    
    

# =========================
# REQUEST MODEL
# =========================
class ChatRequest(BaseModel):
    query: str
    platform: Platform
    role: Role

# =========================
# RESPONSE MODEL
# =========================
class ChatResponse(BaseModel):
    answer: str
    platform: str
    role: str
    sources: List[str]

# =========================
# CHAT ENDPOINT
# =========================
@app.post(
    "/v1/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    # =========================
    # RETRIEVER
    # =========================
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.6,
            "filter": {
                "platform": request.platform,
                "role": request.role
            }
        },
    )

    docs = retriever.invoke(
        request.query
    )

    # =========================
    # NO RESULTS
    # =========================
    if not docs:

        if is_general_fallback(request.query):
            fallback_prompt = (
                "You are a friendly support assistant. "
                "Answer greetings, casual conversation, and general factual questions naturally. "
                "If the user asks about KUJA product workflows or platform-specific actions and you have no knowledge base context, say you do not have that information. "
                "Keep the reply short and polite."
            )

            result = llm.invoke([
                SystemMessage(content=fallback_prompt),
                HumanMessage(content=request.query)
            ]).content

            return ChatResponse(
                answer=result,
                platform=request.platform,
                role=request.role,
                sources=[]
            )

        return ChatResponse(
            answer=(
                f"No relevant information found "
                f"for {request.platform} "
                f"({request.role})."
            ),
            platform=request.platform,
            role=request.role,
            sources=[]
        )

    # =========================
    # CONTEXT
    # =========================
    context = "\n\n".join([
        d.page_content
        for d in docs
    ])

    # =========================
    # SYSTEM PROMPT
    # =========================
    system_prompt = f"""
You are a support assistant for:

Platform: {request.platform}
Role: {request.role}

Use ONLY the provided context.

If the answer is not found in the context,
say you do not know.

Be clear, concise, and accurate.

CONTEXT:
{context}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request.query)
    ]

    # =========================
    # GENERATE RESPONSE
    # =========================
    result = llm.invoke(messages).content

    # =========================
    # SOURCES
    # =========================
    sources = list(set([
        d.metadata.get("source", "unknown")
        for d in docs
    ]))

    # =========================
    # RETURN RESPONSE
    # =========================
    return ChatResponse(
        answer=result,
        platform=request.platform,
        role=request.role,
        sources=sources
    )