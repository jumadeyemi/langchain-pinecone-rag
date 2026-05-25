
import os
from dotenv import load_dotenv


from fastapi import FastAPI
from pydantic import BaseModel

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# =========================
# INIT
# =========================
load_dotenv()

app = FastAPI(title="Kuja RAG Support API")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# =========================
# REQUEST SCHEMA
# =========================
class ChatRequest(BaseModel):
    query: str
    platform: str   # kujaexpress | kujashop | kujadrivers

# =========================
# RESPONSE SCHEMA
# =========================
class ChatResponse(BaseModel):
    answer: str
    platform: str
    sources: list[str]

# =========================
# ENDPOINT
# =========================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # -------------------------
    # RETRIEVER (platform filter)
    # -------------------------
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.6,
            "filter": {
                "platform": request.platform
            }
        },
    )

    docs = retriever.invoke(request.query)

    # -------------------------
    # HANDLE NO RESULTS
    # -------------------------
    if not docs:
        return ChatResponse(
            answer=f"No relevant information found for {request.platform}.",
            platform=request.platform,
            sources=[]
        )

    # -------------------------
    # BUILD CONTEXT
    # -------------------------
    context = "\n\n".join([d.page_content for d in docs])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are a support assistant for {request.platform}.

Use ONLY the context below to answer.

If the answer is not in the context, say you don't know.

Be clear and concise.

CONTEXT:
{context}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request.query)
    ]

    result = llm.invoke(messages).content

    # -------------------------
    # SOURCES
    # -------------------------
    sources = list(set([
        d.metadata.get("source", "unknown")
        for d in docs
    ]))

    return ChatResponse(
        answer=result,
        platform=request.platform,
        sources=sources
    )