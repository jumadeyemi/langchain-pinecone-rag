# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

#documents (load .txt files from data/)
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() 

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# loading all .txt files from the data/ folder
txt_paths = sorted(glob.glob("data/*.txt"))
raw_documents = []
for p in txt_paths:
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    raw_documents.append(Document(page_content=text, metadata={"source": p}))

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
documents = text_splitter.split_documents(raw_documents)

# generate unique id's

i = 0
uuids = []

while i < len(documents):

    i += 1

    uuids.append(f"id{i}")

# add to database

vector_store.add_documents(documents=documents, ids=uuids)