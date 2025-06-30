import streamlit as st
import tempfile
import hashlib
import re
import json
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === ADOBE PDF SERVICES SDK ===
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions, TableStructureType
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef

# === CONFIG ===
st.set_page_config(page_title="📘 Upload AS3000 & Search Clauses", layout="wide")
st.title("📘 Adobe-Powered AS3000 Search")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo-1106"

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📄 Upload AS3000 PDF", type="pdf")
if not uploaded_file:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name
    file_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()

if "vectorstore_hash" not in st.session_state or st.session_state.vectorstore_hash != file_hash:
    # === Adobe SDK: Extract JSON Content ===
    creds = Credentials.service_account_credentials_builder() \
        .from_file("pdfservices-api-credentials.json") \
        .build()
    context = ExecutionContext.create(creds)
    operation = ExtractPDFOperation.create_new()
    input_ref = FileRef.create_from_local_file(pdf_path)
    operation.set_input(input_ref)

    options = ExtractPDFOptions.builder() \
        .with_element_to_extract(["text"]) \
        .with_table_structure_format(TableStructureType.CSV) \
        .build()
    operation.set_options(options)

    result_path = tempfile.mktemp(suffix=".zip")
    result = operation.execute(context)
    result.save_as(result_path)

    import zipfile
    with zipfile.ZipFile(result_path, 'r') as zip_ref:
        zip_ref.extractall("output_adobe")

    with open("output_adobe/extractedData.json", "r", encoding="utf-8") as f:
        extracted_json = json.load(f)

    # === Convert Adobe JSON to LangChain Documents ===
    documents = []
    for element in extracted_json["elements"]:
        if element["Path"].endswith("Text"):
            content = element["Text"]
            page_number = element["Page"].get("PageNumber", "Unknown")
            match = re.search(r"\b(\d{1,2}(?:\.\d{1,2}){1,2})\b", content)
            clause = match.group(1) if match else "Unknown"
            doc = Document(
                page_content=content,
                metadata={"clause": clause, "page": page_number}
            )
            documents.append(doc)

    # === Split & Embed ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    faiss_db = FAISS.from_documents(chunks, embeddings)
    dense_retriever = faiss_db.as_retriever()
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 3

    st.session_state.vectorstore = {
        "retriever": dense_retriever,
        "bm25": bm25,
        "chunks": chunks
    }
    st.session_state.vectorstore_hash = file_hash

# === RETRIEVERS ===
retriever = st.session_state.vectorstore["retriever"]
bm25 = st.session_state.vectorstore["bm25"]
chunks = st.session_state.vectorstore["chunks"]

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

    def compute_confidence(docs):
        base_score = 0
        if len(docs) >= 2: base_score += 40
        if any("clause" in d.metadata and re.match(r"\d+\.\d+", d.metadata["clause"]) for d in docs): base_score += 30
        if all(len(d.page_content) > 300 for d in docs[:2]): base_score += 20
        return min(base_score + 10, 100)

    confidence_score = compute_confidence(combined_docs)
    st.progress(confidence_score / 100)
    st.write(f"**Confidence: {confidence_score}%**")

    st.subheader("📚 Top Clause Matches")
    for i, doc in enumerate(combined_docs[:3]):
        clause = doc.metadata.get("clause", "Unknown")
        page = doc.metadata.get("page", "N/A")
        st.markdown(f"**Match {i+1}** — Clause `{clause}` | Page `{page}`")
        st.code(doc.page_content[:500], language="text")
