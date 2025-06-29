# ✅ Advanced RAG File Upload + Semantic Search (Optimized Setup)
import streamlit as st
import tempfile
import hashlib
import re
import requests
import json
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
st.set_page_config(page_title="📂 Advanced Semantic Code Search")
st.title("📂 Upload & Search Your Code Document (Advanced Setup)")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo-1106"

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📄 Upload PDF Code Document", type="pdf")
if not uploaded_file:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name

# === PDF LOADER ===
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# === SEMANTIC CHUNKING ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# === ENRICH WITH CLAUSE METADATA ===
def enrich_metadata(doc: Document) -> Document:
    clause = re.search(r"\b(\d{1,2}(?:\.\d{1,2}){1,2})\b", doc.page_content)
    doc.metadata["clause"] = clause.group(1) if clause else "Unknown"
    return doc

enriched_chunks = [enrich_metadata(d) for d in chunks]

# === EMBEDDING + VECTORSTORE ===
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
faiss_db = FAISS.from_documents(enriched_chunks, embeddings)
dense_retriever = faiss_db.as_retriever()

# === OPTIONAL: Sparse Retriever for Hybrid (BM25) ===
bm25_retriever = BM25Retriever.from_documents(enriched_chunks)
bm25_retriever.k = 3

# === QUERY ===
st.title("📘 AS3000 Code Assistant")
query = st.text_input("🔎 Ask a code-related question:")
if query:
    dense_docs = dense_retriever.get_relevant_documents(query)
    sparse_docs = bm25_retriever.get_relevant_documents(query)
    combined_docs = dense_docs + sparse_docs

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=LLM_MODEL)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, retriever=dense_retriever, return_source_documents=True
    )

    result = qa_chain.combine_documents_chain.run(
        input_documents=combined_docs, question=query
    )

    st.subheader("✅ Answer")
    st.success(result)

    st.subheader("📚 Top Relevant Chunks")
    for i, doc in enumerate(combined_docs[:3]):
        clause = doc.metadata.get("clause", "N/A")
        page = doc.metadata.get("page", "N/A")
        st.markdown(f"**Chunk {i+1}** — Clause `{clause}` | Page `{page}`")
        st.code(doc.page_content[:500], language="text")
