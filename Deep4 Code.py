# ✅ Advanced Clause-Aware Code QA App (Upload Only)
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

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name

# === LOAD & SPLIT PDF ===
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# === CHUNKING ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# === ENRICH METADATA WITH CLAUSE ===
def enrich_clause(doc: Document) -> Document:
    match = re.search(r"\b(\d{1,2}(?:\.\d{1,2}){1,2})\b", doc.page_content)
    doc.metadata["clause"] = match.group(1) if match else "Unknown"
    return doc

enriched_chunks = [enrich_clause(d) for d in chunks]

# === EMBEDDING + VECTORSTORE ===
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
faiss_db = FAISS.from_documents(enriched_chunks, embeddings)
dense_retriever = faiss_db.as_retriever()

# === SPARSE RETRIEVER (OPTIONAL HYBRID) ===
bm25 = BM25Retriever.from_documents(enriched_chunks)
bm25.k = 3

# === USER QUERY ===
st.title("📘 AS3000 Code Assistant")
query = st.text_input("💬 Ask your question:")

if query:
    docs = retriever.get_relevant_documents(query)
    result = qa.combine_documents_chain.run(
        input_documents=docs,
        question=query
    )

    st.subheader("🔍 Answer")
    st.success(result)

    st.subheader("📚 Source Snippets")
    for i, doc in enumerate(docs[:3]):
        clause = doc.metadata.get("clause", "N/A")
        page = doc.metadata.get("page", "N/A")
        preview = doc.page_content.strip().replace("\n", " ")[:400]
        with st.expander(f"Source {i+1} — Clause {clause}, Page {page}"):
            st.code(preview, language="text")

