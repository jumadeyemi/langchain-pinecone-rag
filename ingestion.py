# =========================
# IMPORTS
# =========================
import os
import time
import glob
from pathlib import Path
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

# =========================
# INITIALIZE PINECONE
# =========================
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = os.environ.get("PINECONE_INDEX_NAME")

# =========================
# CREATE INDEX IF NOT EXISTS
# =========================
existing_indexes = [
    index_info["name"]
    for index_info in pc.list_indexes()
]

if index_name not in existing_indexes:

    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# =========================
# CONNECT TO INDEX
# =========================
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
# LOAD DOCUMENTS
# =========================
raw_documents = []

txt_paths = glob.glob(
    "data/**/*.txt",
    recursive=True
)

for path in txt_paths:

    path_obj = Path(path)

    # Example:
    # data/kujashop/customer/file.txt
    #
    # parts:
    # [data, kujashop, customer, file.txt]

    platform = path_obj.parts[1]
    role = path_obj.parts[2]
    source = path_obj.name

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    raw_documents.append(
        Document(
            page_content=text,
            metadata={
                "platform": platform,
                "role": role,
                "source": source
            }
        )
    )

print(f"Loaded {len(raw_documents)} raw documents.")

# =========================
# SPLIT DOCUMENTS
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(
    raw_documents
)

print(f"Created {len(documents)} chunks.")

# =========================
# GENERATE IDS
# =========================
ids = [
    f"doc_{i}"
    for i in range(len(documents))
]

# =========================
# UPSERT DOCUMENTS
# =========================
vector_store.add_documents(
    documents=documents,
    ids=ids
)

print("Documents successfully ingested into Pinecone.")