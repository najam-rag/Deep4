# ✅ Streamlit App with Adobe PDF Services SDK for Metadata Extraction
import streamlit as st
import tempfile
import hashlib
import json
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === CONFIG ===
st.set_page_config(page_title="📘 Adobe Metadata Search App", layout="wide")
st.title("📘 Upload PDF (AS3000 etc.) — Powered by Adobe PDF SDK")

OPENAI_API_KEY = st.secrets["openai_api_key"]
ADOBE_CLIENT_ID = st.secrets["CLIENT_ID"]
ADOBE_CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
ADOBE_ORG_ID = st.secrets["ORGANIZRION_ID"]

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📄 Upload a Code PDF (e.g., AS3000)", type="pdf")
if not uploaded_file:
    st.stop()

# === HASH CHECK ===
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name
    file_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()

if "vectorstore_hash" not in st.session_state or st.session_state.vectorstore_hash != file_hash:
    # === ADOBE PDF SDK Text Extraction ===
    from adobe.pdfservices.operation.auth.credentials import Credentials
    from adobe.pdfservices.operation.execution_context import ExecutionContext
    from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
    from adobe.pdfservices.operation.io.file_ref import FileRef

    credentials = Credentials.service_principal_credentials_builder()\
        .with_client_id(ADOBE_CLIENT_ID)\
        .with_client_secret(ADOBE_CLIENT_SECRET)\
        .with_organization_id(ADOBE_ORG_ID)\
        .build()

    execution_context = ExecutionContext.create(credentials)
    extract_operation = ExtractPDFOperation.create_new()
    extract_operation.set_input(FileRef.create_from_local_file(pdf_path))

    result = extract_operation.execute(execution_context)
    extracted_path = os.path.join(tempfile.gettempdir(), "structured_data.json")
    result.save_as(extracted_path)

    # === Parse Adobe Output ===
    with open(extracted_path, "r", encoding="utf-8") as f:
        adobe_data = json.load(f)

    documents = []
    for elem in adobe_data.get("elements", []):
        text = elem.get("Text", "").strip()
        page = elem.get("Page", 0)
        if len(text) > 20:
            doc = Document(page_content=text, metadata={"page": page})
            documents.append(doc)

    # === Clause + Tag Enrichment ===
    def enrich(doc):
        import re
        match = re.search(r"\b(\d{1,2}(?:\.\d{1,2}){1,2})\b", doc.page_content)
        doc.metadata["clause"] = match.group(1) if match else "Unknown"
        if "underground" in doc.page_content.lower():
            doc.metadata.setdefault("tags", []).append("underground")
        return doc

    enriched_docs = [enrich(d) for d in documents]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    faiss_db = FAISS.from_documents(enriched_docs, embeddings)
    dense_retriever = faiss_db.as_retriever()
    bm25 = BM25Retriever.from_documents(enriched_docs)
    bm25.k = 3

    st.session_state.vectorstore = {
        "retriever": dense_retriever,
        "bm25": bm25,
        "chunks": enriched_docs
    }
    st.session_state.vectorstore_hash = file_hash

# === USER QUERY ===
query = st.text_input("🔍 Ask your code-related question:")
if query:
    retriever = st.session_state.vectorstore["retriever"]
    bm25 = st.session_state.vectorstore["bm25"]
    enriched_chunks = st.session_state.vectorstore["chunks"]

    dense_docs = retriever.get_relevant_documents(query)
    sparse_docs = bm25.get_relevant_documents(query)
    combined_docs = dense_docs + sparse_docs

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-1106")
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa_chain.combine_documents_chain.run(input_documents=combined_docs, question=query)

    st.success(result)

    def compute_confidence(docs):
        base = 0
        if len(docs) >= 2:
            base += 40
        if any("clause" in d.metadata and re.match(r"\d+\.\d+", d.metadata["clause"]) for d in docs):
            base += 30
        if all(len(d.page_content) > 300 for d in docs[:2]):
            base += 20
        return min(base + 10, 100)

    score = compute_confidence(combined_docs)
    st.progress(score / 100)
    st.write(f"**Confidence: {score}%** — Based on clause match, document strength, and chunk quality.")

    st.subheader("📚 Top Clause Matches")
    for i, doc in enumerate(combined_docs[:3]):
        clause = doc.metadata.get("clause", "Unknown")
        page = doc.metadata.get("page", "N/A")
        st.markdown(f"**Match {i+1}** — Clause `{clause}` | Page `{page}`")
        st.code(doc.page_content[:500], language="text")
