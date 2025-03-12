from collections import defaultdict

import streamlit as st

from ui.constants import RELEVANT_DOCUMENTS
from utils.file_rendering import render_docx, render_pdf


def render_debug_section():
    """
    Render a debug section showing relevant documents grouped by file path.
    """
    if st.session_state[RELEVANT_DOCUMENTS]:
        st.markdown("#### Folgende Dokumente wurden als Kontext verwendet:")

        relevant_docs = defaultdict(list)
        for document in st.session_state[RELEVANT_DOCUMENTS]:
            doc_path = document.metadata["source"]
            doc_page = document.metadata["page_number"]
            # doc_relevance_score = document.score
            relevant_docs[doc_path].append(
                {
                    "page": doc_page,
                    # "relevance_score": doc_relevance_score,
                }
            )

        for file_path, docs in relevant_docs.items():
            with st.expander(f"File: {file_path}"):
                for doc in docs:
                    st.markdown(f"**Seite:** {doc['page']}")
                    # st.markdown(f"**Relevanz:** {doc['relevance_score']:.4f}")
                    render_page(file_path, doc["page"])
                    st.markdown("---")


def render_page(file_path: str, page_number: int):
    if not file_path:
        return
    if file_path.endswith(".pdf"):
        href = render_pdf(file_path, page_number)
    elif file_path.endswith((".docx")):
        href = render_docx(file_path, page_number)
    else:
        st.error("Unsupported file format. Only PDF and DOCX files are supported.")
        return
    st.markdown(href, unsafe_allow_html=True)
