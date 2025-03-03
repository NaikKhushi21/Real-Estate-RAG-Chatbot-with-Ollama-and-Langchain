import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from query_data import query_rag

# Directories for storing uploaded PDFs and vector store data.
DATA_PATH = "data"
CHROMA_PATH = "chroma"

os.makedirs(DATA_PATH, exist_ok=True)

st.title("Enhanced Real Estate Chatbot")

st.header("1. Upload Real Estate Documents")
uploaded_files = st.file_uploader(
    "Upload property data, regulatory documents, and financial reports (PDFs)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully!")

st.header("2. Process and Index Documents")
if st.button("Process and Index PDFs"):
    # Load documents from the data directory.
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    st.write(f"Loaded {len(documents)} documents.")

    # Split the documents into smaller chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)

    # Initialize the embedding function and vector store.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Add the chunks to the vector store.
    db.add_documents(chunks)
    st.success("Documents processed and indexed successfully!")

st.header("3. Ask a Real Estate Question")
user_question = st.text_input("Enter your real estate query:")

if st.button("Get Answer") and user_question:
    with st.spinner("Generating answer..."):
        answer = query_rag(user_question)
    st.subheader("Answer:")
    st.write(answer)