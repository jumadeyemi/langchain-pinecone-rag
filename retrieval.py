# =========================
# IMPORTS
# =========================
import os
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

# =========================
# INITIALIZE PINECONE
# =========================
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# =========================
# EMBEDDINGS + VECTOR STORE
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
# PLATFORM SELECTION
# =========================
platform = "kujashop"

# =========================
# CREATE RETRIEVER
# =========================
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.6,
        "filter": {
            "platform": platform
        }
    },
)

# =========================
# QUERY
# =========================
query = "what does the distributor module show in admin page"

results = retriever.invoke(query)

# =========================
# DISPLAY RESULTS
# =========================
print("\nRESULTS:\n")

if not results:
    print("No relevant documents found.")

else:
    for i, res in enumerate(results, start=1):

        print(f"RESULT {i}")
        print(f"Platform: {res.metadata.get('platform')}")
        print(f"Source: {res.metadata.get('source')}")
        print(res.page_content[:500])
        print("\n" + "=" * 80 + "\n")