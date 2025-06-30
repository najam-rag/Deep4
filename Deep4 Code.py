# ✅ Clause-Aware QA App with Real-Time Metadata Tagging
import streamlit as st
import tempfile
import hashlib
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever

# === CONFIG ===
st.set_page_config(page_title="📘 Upload AS3000 & Search Clauses", layout="wide")
st.title("📘 Advanced Clause-Aware Search (with Tags)")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo-1106"

# === GPT-based TAG Extractor ===
def gpt_extract_tags(text: str) -> List[str]:
    import openai
    openai.api_key = OPENAI_API_KEY

    prompt = f"""
You are a tagging assistant. Extract relevant tags from the following building code text to describe context. Return a list of short tags (1–3 words), like ["underground", "HD conduit", "clearance", "earthing"].

Text:
\"\"\"{text}\"\"\"
Return tags in a Python list:
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        tags_raw = response.choices[0].message.content.strip()
        tags = eval(tags_raw) if tags_raw.startswith("[") else []
        return tags
    except Exception:
        return []

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📄 Upload AS3000 PDF", type="pdf")
if not uploaded_file:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name
    file_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()

# === Build Vectorstore Once ===
if "vectorstore_hash" not in st.session_state or st.session_state.vectorstore_hash != file_hash:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    def enrich_metadata(doc: Document) -> Document:
        clause_match = re.search(r"\b(\d{1,2}(?:\.\d{1,2}){1,2})\b", doc.page_content)
        clause = clause_match.group(1) if clause_match else "Unknown"
        page = doc.metadata.get("page", "N/A")
        tags = gpt_extract_tags(doc.page_content[:500])
        doc.metadata.update({"clause": clause, "page": page, "tags": tags})
        return doc

    enriched_chunks = [enrich_metadata(d) for d in chunks]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    faiss_db = FAISS.from_documents(enriched_chunks, embeddings)
    dense_retriever = faiss_db.as_retriever()
    bm25 = BM25Retriever.from_documents(enriched_chunks)
    bm25.k = 3

    st.session_state.vectorstore = {
        "retriever": dense_retriever,
        "bm25": bm25,
        "chunks": enriched_chunks
    }
    st.session_state.vectorstore_hash = file_hash

# === RETRIEVERS ===
retriever = st.session_state.vectorstore["retriever"]
bm25 = st.session_state.vectorstore["bm25"]
enriched_chunks = st.session_state.vectorstore["chunks"]

# === USER QUERY ===
query = st.text_input("🔍 Ask your AS3000 question:")
if query:
    dense_docs = retriever.get_relevant_documents(query)
    sparse_docs = bm25.get_relevant_documents(query)
    combined_docs = dense_docs + sparse_docs

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=LLM_MODEL)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    result = qa_chain.combine_documents_chain.run(
        input_documents=combined_docs, question=query
    )

    st.success(result)

    # === Confidence Barometer ===
    def compute_confidence(docs):
        score = 10
        if len(docs) >= 2: score += 40
        if any("clause" in d.metadata and re.match(r"\d+\.\d+", d.metadata["clause"]) for d in docs): score += 30
        if all(len(d.page_content) > 300 for d in docs[:2]): score += 20
        return min(score, 100)

    confidence_score = compute_confidence(combined_docs)
    st.progress(confidence_score / 100)
    st.write(f"**Confidence: {confidence_score}%** — Based on clause match, document strength, and chunk quality.")

    st.subheader("📚 Top Clause Matches")
    for i, doc in enumerate(combined_docs[:3]):
        clause = doc.metadata.get("clause", "Unknown")
        page = doc.metadata.get("page", "N/A")
        tags = ", ".join(doc.metadata.get("tags", []))
        st.markdown(f"**Match {i+1}** — Clause `{clause}` | Page `{page}` | Tags: `{tags}`")
        st.code(doc.page_content[:500], language="text")
