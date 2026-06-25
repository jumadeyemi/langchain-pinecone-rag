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


from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
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
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BEERTECH AI SUPPORT ASSISTANT",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 BEERTECH AI SUPPORT ASSISTANT")

# =========================
# PLATFORM + ROLE MAPPING
# =========================
platform_roles = {
    "kujashop": [
        "bdr",
        "bulkbreaker",
        "pocs"
    ],
    "kujaexpress": [
        "stockist"
    ],
    "kujadrivers": [
        "driver"
    ],
    "kujaerp": [
        "admin",
        "distributor"
    ]
}

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Configuration")

# Platform selector
platform = st.sidebar.selectbox(
    "Select Platform",
    list(platform_roles.keys())
)

# Role selector
role = st.sidebar.selectbox(
    "Select Role",
    platform_roles[platform]
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
    model="openai/text-embedding-3-large",
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
# RESET CHAT IF CONTEXT CHANGES
# =========================
if "current_platform" not in st.session_state:
    st.session_state.current_platform = platform

if "current_role" not in st.session_state:
    st.session_state.current_role = role

# Clear chat if platform changes
if (
    st.session_state.current_platform != platform
    or
    st.session_state.current_role != role
):

    st.session_state.messages = []

    st.session_state.current_platform = platform
    st.session_state.current_role = role

# =========================
# DISPLAY ACTIVE CONTEXT
# =========================
st.sidebar.markdown("---")

st.sidebar.success(
    f"""
    Active Context:
    
    Platform: {platform}
    
    Role: {role}
    """
)

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
prompt = st.chat_input(
    f"Ask a question as a {role} on {platform}..."
)

# =========================
# HANDLE USER INPUT
# =========================
if prompt:

    # =========================
    # DISPLAY USER MESSAGE
    # =========================
    with st.chat_message("user"):
        st.markdown(prompt)

    # =========================
    # SAVE USER MESSAGE
    # =========================
    st.session_state.messages.append(
        HumanMessage(prompt)
    )

    # =========================
    # INITIALIZE LLM
    # =========================
    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o"),
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
                "platform": platform,
                "role": role
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
            f"for role '{role}' "
            f"on platform '{platform}'."
        )

    else:

        # =========================
        # PREPARE CONTEXT
        # =========================
        docs_text = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # =========================
        # SYSTEM PROMPT
        # =========================
        system_prompt = f"""
You are a support assistant for:

Platform: {platform}
Role: {role}

You MUST answer ONLY using the provided context.

Do NOT provide answers outside the context.

If the answer is not found in the context,
say you could not find the information
in the knowledge base.

Tailor your response specifically
for the role: {role}.

Be concise, accurate, and helpful.

If applicable, provide step-by-step guidance.

CONTEXT:
{docs_text}
"""

        # =========================
        # RECENT CHAT HISTORY
        # =========================
        recent_messages = (
            st.session_state.messages[-6:]
        )

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

        # =========================
        # DISPLAY SOURCES
        # =========================
        if docs:

            st.markdown("---")
            st.markdown("### Sources")

            shown_sources = set()

            for doc in docs:

                source = doc.metadata.get(
                    "source"
                )

                doc_platform = doc.metadata.get(
                    "platform"
                )

                doc_role = doc.metadata.get(
                    "role"
                )

                if source not in shown_sources:

                    st.caption(
                        f"📄 {source} "
                        f"({doc_platform} | {doc_role})"
                    )

                    shown_sources.add(source)

    # =========================
    # SAVE AI RESPONSE
    # =========================
    st.session_state.messages.append(
        AIMessage(result)
    )