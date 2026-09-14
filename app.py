import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "University Student & Academic Knowledge Assistant"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

FAISS_PATH = KNOWLEDGE_BASE_DIR / "index.faiss"

CHUNKS_PATH = KNOWLEDGE_BASE_DIR / "chunks.json"

METADATA_PATH = KNOWLEDGE_BASE_DIR / "metadata.json"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .source-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 0.8rem;
    }

    .source-title {
        font-weight: 600;
        font-size: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():

    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found: {CHUNKS_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.json not found: {METADATA_PATH}"
        )

    # Load FAISS index
    index = faiss.read_index(str(FAISS_PATH))

    # Load text chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return index, chunks, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(api_key=api_key)


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_documents(
    query,
    index,
    chunks,
    embedding_model,
    top_k=5
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append({
            "score": float(score),
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "department": chunk["department"],
            "document_type": chunk["document_type"],
            "year": chunk["year"]
        })

    return results


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}

Document: {result["source"]}
Department: {result["department"]}
Document Type: {result["document_type"]}
Year: {result["year"]}

Content:
{result["text"]}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context,
    client
):

    system_prompt = """
You are the University Student & Academic Knowledge Assistant.

Your job is to answer questions using ONLY the information
provided in the retrieved university knowledge base.

Rules:

1. Use the provided context as your primary source of truth.

2. Do not invent university policies, deadlines, fees,
   eligibility criteria or rules.

3. If the answer is not available in the provided context,
   clearly say that the information could not be found
   in the available university documents.

4. Give a clear and concise answer.

5. When possible, mention the document used to answer
   the question.

6. Do not claim that these documents are official University
   of Lahore policies. They are demonstration knowledge
   documents for this application.
"""

    user_prompt = f"""
Retrieved university knowledge:

{context}

Student question:

{question}

Answer the question using only the retrieved knowledge.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1000
    )

    return response.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 Knowledge Base")

    st.write(
        "University Student & Academic Knowledge"
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write("📄 6 documents")

    st.write("🔎 Multi document retrieval")

    st.write("🧠 Pre created embeddings")

    st.write("⚡ FAISS vector search")

    st.divider()

    st.subheader("Search Settings")

    top_k = st.slider(
        "Number of sources",
        min_value=2,
        max_value=8,
        value=5
    )

    st.divider()

    st.caption(
        "Embeddings are created offline. "
        "The application loads the existing FAISS index."
    )


# ============================================================
# MAIN INTERFACE
# ============================================================

st.markdown(
    '<div class="main-title">🎓 University Student & Academic Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about admissions, fees, examinations, scholarships, the academic calendar and student policies.</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    index, chunks, metadata = load_knowledge_base()

    embedding_model = load_embedding_model()

    groq_client = get_groq_client()

except Exception as error:

    st.error(
        f"Application setup error: {error}"
    )

    st.stop()


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Knowledge Chunks",
        len(chunks)
    )

with col2:
    st.metric(
        "Indexed Vectors",
        index.ntotal
    )

with col3:
    st.metric(
        "Vector Dimension",
        index.d
    )


st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "💬 Ask your question",
    placeholder=(
        "Example: What are the eligibility criteria "
        "for the scholarship, and when can I apply?"
    ),
    height=100
)


ask_button = st.button(
    "🔎 Ask Knowledge Assistant",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    with st.spinner(
        "Searching university knowledge..."
    ):

        results = retrieve_documents(
            question,
            index,
            chunks,
            embedding_model,
            top_k=top_k
        )

    if not results:

        st.warning(
            "No relevant information was found."
        )

        st.stop()

    context = build_context(results)

    with st.spinner(
        "Generating answer..."
    ):

        try:

            answer = generate_answer(
                question,
                context,
                groq_client
            )

        except Exception as error:

            st.error(
                f"LLM error: {error}"
            )

            st.stop()


    # ========================================================
    # ANSWER
    # ========================================================

    st.subheader("🤖 Answer")

    st.markdown(answer)


    # ========================================================
    # SOURCES
    # ========================================================

    st.divider()

    st.subheader("📚 Retrieved Sources")

    for number, result in enumerate(
        results,
        start=1
    ):

        with st.expander(
            f"{number}. {result['source']}  •  similarity: {result['score']:.3f}"
        ):

            st.write(
                f"**Department:** {result['department']}"
            )

            st.write(
                f"**Document Type:** {result['document_type']}"
            )

            st.write(
                f"**Year:** {result['year']}"
            )

            st.write(
                f"**Chunk:** {result['chunk_id']}"
            )

            st.markdown("**Retrieved text:**")

            st.write(
                result["text"]
            )
