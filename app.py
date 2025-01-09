import sys
from rich.traceback import install

from log_config import setup_logger
from collections import defaultdict
from config import load_config, get_config

import os

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv

from rag_pipeline import SHRAGPipeline
from utils import render_pdf, render_docx
import pandas as pd

load_dotenv()

install(show_locals=True)

logger = setup_logger()

def handle_exception(exc_type, exc_value, exc_traceback):
    """Handles unhandled exceptions, logs them using the logger."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

UI_RENDERED_MESSAGES = "ui_rendered_messages"
CONVERSATIONAL_PIPELINE = "conversational_pipeline"
RELEVANT_DOCUMENTS = "None"
TITLE_NAME = "Data Alchemy RAG Bot"

load_config()
config = get_config()
st.set_page_config(page_title=TITLE_NAME)
authenticator = stauth.Authenticate(**config.LOGIN_CONFIG)


def main():
    """
    Render the retrieval augmented generation (RAG) chatbot application.
    """
    authentication()
    config = load_config()
    initialize_session_state(config)
    setup_page()
    render_chat_history()
    manage_chat()
    render_feedback_section()
    render_debug_section()


def load_config():
    """
        Load the application configuration from a file or object.

    Returns:
        dict: Configuration dictionary containing title name,
              UI rendered messages, chat history, and conversational pipeline instance.
    """
    return {
        UI_RENDERED_MESSAGES: [],
        RELEVANT_DOCUMENTS: [],
        CONVERSATIONAL_PIPELINE: None,
    }


def setup_page():
    """
    Set Streamlit page configuration and title.
    """
    st.markdown("Developed with :heart: by Data Alchemy Team")
    st.title(TITLE_NAME)
    if st.session_state["authentication_status"]:
        authenticator.logout()
        st.subheader(f"Daten von: {', '.join(st.session_state['roles'])}")
        st.write(f'Hallo *{st.session_state["name"]}*')
        render_example_queries()
        if st.session_state[CONVERSATIONAL_PIPELINE] is None:
            st.session_state[CONVERSATIONAL_PIPELINE] = SHRAGPipeline(
                st.session_state["roles"]
            )
    elif st.session_state["authentication_status"] is False:
        st.error("Benutzername oder Passwort sind falsch")
    elif st.session_state["authentication_status"] is None:
        st.warning("Bitte Benutzername und Passwort eingeben")


def authentication():
    try:
        authenticator.login()
    except stauth.LoginError as e:
        st.error(e)


def initialize_session_state(config):
    """
        Initialize Streamlit session state variables using the provided configuration.

    Args:
        config (dict): Configuration dictionary.
    """
    for key, value in config.items():
        if key not in st.session_state:
            st.session_state[key] = value


def manage_chat():
    """
    Handle user interaction with the conversational AI and render
    the user query along with the AI response.
    """
    prompt = st.session_state.get('user_input') or st.chat_input("Wie kann ich Dir heute helfen?")
    if prompt is not None:
        st.session_state.user_input = None

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("assistant"):
            with st.spinner("Antwort wird generiert . . ."):
                *itterator, documents = st.session_state[CONVERSATIONAL_PIPELINE].stream_query(prompt)
                response = st.write_stream(itterator)
                st.session_state[RELEVANT_DOCUMENTS] = documents
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "assistant", "content": response}
        )


def render_feedback_section():
    """
    Render a feedback section for the user to provide feedback on the AI response.
    """
    if (
        st.session_state[UI_RENDERED_MESSAGES]
        and st.session_state[UI_RENDERED_MESSAGES][-1]["role"] == "assistant"
    ):
        with st.expander("Feedback geben", expanded=True):
            col1, col2 = st.columns([1, 3])

            with col1:
                is_helpful = st.radio(
                    "War die Antwort hilfreich?", ["Yes", "No"], key="feedback_helpful"
                )

            feedback_data = {
                "helpful": is_helpful,
                "reason": "",
                "feedback": "",
                "query": st.session_state[UI_RENDERED_MESSAGES][-2]["content"],
                "response": st.session_state[UI_RENDERED_MESSAGES][-1]["content"],
                "user": st.session_state["name"],
                "model": config["LLM"]["MODEL"],
                "retriever": config["RETRIEVER"],
                "doc_store": config["DOC_STORE"]
            }

            if is_helpful == "No":
                with col2:
                    feedback_data["reason"] = st.multiselect(
                        "Das ist schief gelaufen:",
                        [
                            "Kontext enthielt die korrekte Antwort, AI fand diese nicht",
                            "Kontext enthielt die korrekte Antwort nicht",
                            "Falsche Antwort",
                            "Antwort zu knapp / relevante Info weggelessen"
                            "Antwort zu lange",
                            "Anderes",
                        ],
                        key="feedback_reason",
                        placeholder="Grund auswählen"
                    )
                    feedback_data["feedback"] = st.text_area(
                        "Zusätzliches Feedback (optional):"
                    )

            if st.button("Feedback senden", key="feedback_submit"):
                save_feedback(feedback_data)
                st.success("Danke für deinFeedback!")

def render_example_queries():
    st.subheader("Beispiel Fragen:")
    example_queries = [
        "Was für Angebote zur Familienergänzenden Tagesbetreuung gibt es?",
        "Ich habe meine Stelle verloren, was muss ich tun?",
        "Werden Beiträge der Hilflosenentschädigung von den Ansprüchen auf andere Sozialbeiträge abgezogen?",
    ]
    cols = st.columns(3)
    for col, query in zip(cols, example_queries):
        with col:
            if st.button(query):
                st.session_state.user_input = query

def save_feedback(feedback_data):
    """
    Save feedback data to a CSV file.
    """
    feedback_file = "feedback.csv"

    feedback_data["response"] = " ".join(feedback_data["response"].split())
    feedback_data["response"] = feedback_data["response"].replace('\n',' ')
    feedback_data["feedback"] = " ".join(feedback_data["feedback"].split())
    feedback_data["feedback"] = feedback_data["feedback"].replace('\n',' ')

    if not os.path.exists(feedback_file):
        df = pd.DataFrame(columns=feedback_data.keys())
    else:
        df = pd.read_csv(feedback_file, encoding='utf-8')

    new_row = pd.DataFrame([feedback_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(feedback_file, index=False, encoding='utf-8')


def render_chat_history():
    """
    Display the chat message history stored in session state.
    """
    for message in st.session_state[UI_RENDERED_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


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
                    render_page(file_path, doc['page'])
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


if __name__ == "__main__":
    main()
