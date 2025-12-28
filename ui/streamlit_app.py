import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.loaders import load_documents
from app.text_splitter import split_documents
from app.vectorstore import build_vectorstore
from app.chains import make_qa_chain
from app.utils import format_sources

st.set_page_config(page_title="Construction RAG", layout="wide")
st.title("🏗 Construction Document AI")

with st.spinner("Loading documents..."):
    docs = load_documents()
    chunks = split_documents(docs)
    db = build_vectorstore(chunks)
    retriever = db.as_retriever(search_kwargs={"k":3})
    qa = make_qa_chain(retriever)

query = st.text_input("Ask your construction question")

if query:
    result = qa(query)

    st.markdown("### Retrieved Context")
    for d in result["context"]:
        st.code(d.page_content)

    st.markdown("### Final Answer")
    st.success(result["answer"])

    st.markdown("### Sources")
    for s in format_sources(result["context"]):
        st.write("•", s)
