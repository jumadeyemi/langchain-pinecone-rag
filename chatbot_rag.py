# =========================
# IMPORTS
# =========================
import os
import streamlit as st
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BEERTECH AI SUPPORT ASSISTANT",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 BEERTECH AI SUPPORT ASSISTANT")

# =========================
# PLATFORM SELECTOR
# =========================
platform = st.sidebar.selectbox(
    "Select Platform",
    [
        "kujaexpress",
        "kujashop",
        "kujadrivers"
    ]
)

# =========================
# INITIALIZE PINECONE
# =========================
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

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
# INITIALIZE CHAT HISTORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# DISPLAY CHAT HISTORY
# =========================
for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# =========================
# USER INPUT
# =========================
prompt = st.chat_input("Ask a question about the platform...")

# =========================
# HANDLE USER INPUT
# =========================
if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        HumanMessage(prompt)
    )

    # =========================
    # INITIALIZE LLM
    # =========================
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

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
    # RETRIEVE DOCUMENTS
    # =========================
    docs = retriever.invoke(prompt)

    # =========================
    # HANDLE NO RESULTS
    # =========================
    if not docs:

        result = (
            f"I could not find relevant information "
            f"for '{platform}' in the knowledge base."
        )

    else:

        # =========================
        # PREPARE CONTEXT
        # =========================
        docs_text = "\n\n".join(
            doc.page_content for doc in docs
        )

        # =========================
        # SYSTEM PROMPT
        # =========================
        system_prompt = f"""
        You are a support assistant for {platform}.

        Answer ONLY using the provided context.

        If the answer is not in the context,
        say you could not find the information
        in the knowledge base.

        Be concise, clear, and accurate.

        If applicable, provide step-by-step guidance.

        CONTEXT:
        {docs_text}
        """

        # =========================
        # LAST FEW MESSAGES ONLY
        # =========================
        recent_messages = st.session_state.messages[-6:]

        # =========================
        # CREATE MESSAGE LIST
        # =========================
        messages_for_llm = [
            SystemMessage(content=system_prompt),
            *recent_messages
        ]

        # =========================
        # GENERATE RESPONSE
        # =========================
        result = llm.invoke(
            messages_for_llm
        ).content

    # =========================
    # DISPLAY RESPONSE
    # =========================
    with st.chat_message("assistant"):

        st.markdown(result)

        # Show sources
        if docs:

            st.markdown("---")
            st.markdown("### Sources")

            shown_sources = set()

            for doc in docs:

                source = doc.metadata.get("source")

                if source not in shown_sources:

                    st.caption(f"📄 {source}")

                    shown_sources.add(source)

    # =========================
    # SAVE AI RESPONSE
    # =========================
    st.session_state.messages.append(
        AIMessage(result)
    )