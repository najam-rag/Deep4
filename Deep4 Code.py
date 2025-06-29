# ✅ Advanced Clause-Aware Code QA App (Optimized Vectorstore)
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
st.title("📘 Advanced Clause-Aware Search (PDF Only)")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo-1106"

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📄 Upload AS3000 PDF", type="pdf")
if not uploaded_file:
    st.stop()

# === HASH CHECK FOR ONE-TIME VECTORSTORE ===
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name
    file_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()

if "vectorstore_hash" not in st.session_state or st.session_state.vectorstore_hash != file_hash:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    def enrich_clause(doc: Document) -> Document:
        match = re.search(r"\b(\d{1,2}(?:\.\d{1,2}){1,2})\b", doc.page_content)
        doc.metadata["clause"] = match.group(1) if match else "Unknown"
        return doc

    enriched_chunks = [enrich_clause(d) for d in chunks]

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

    st.subheader("✅ Best Answer")
    st.success(result)

    st.subheader("📚 Top Clause Matches")
    for i, doc in enumerate(combined_docs[:3]):
        clause = doc.metadata.get("clause", "Unknown")
        page = doc.metadata.get("page", "N/A")
        st.markdown(f"**Match {i+1}** — Clause `{clause}` | Page `{page}`")
        st.code(doc.page_content[:500], language="text")

# === Confidence Barometer ===
def compute_confidence(docs):
    base_score = 0
    if len(docs) >= 2:
        base_score += 40
    if any("clause" in d.metadata and re.match(r"\d+\.\d+", d.metadata["clause"]) for d in docs):
        base_score += 30
    if all(len(d.page_content) > 300 for d in docs[:2]):
        base_score += 20
    return min(base_score + 10, 100)  # add 10 for base LLM confidence

confidence_score = compute_confidence(combined_docs)

st.subheader("📊 Confidence Barometer")
st.progress(confidence_score / 100)
st.write(f"**Confidence: {confidence_score}%** — Based on clause match, document strength, and chunk quality.")

