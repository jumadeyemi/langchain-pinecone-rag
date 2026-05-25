# =========================
# IMPORTS
# =========================
import os
from enum import Enum
from typing import List

from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

# Pinecone
from pinecone import Pinecone

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

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
    model="text-embedding-3-large",
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
    model="gpt-4o",
    temperature=0
)

# =========================
# ENUMS
# =========================
class Platform(str, Enum):
    kujashop = "kujashop"
    kujaexpress = "kujaexpress"
    kujadrivers = "kujadrivers"


class Role(str, Enum):
    customer = "customer"
    bdr = "bdr"
    stockist = "stockist"
    driver = "driver"

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