from collections import defaultdict

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv

from rag_pipeline import SHRAGPipeline
from utils import config_loader

load_dotenv()

UI_RENDERED_MESSAGES = "ui_rendered_messages"
CHAT_HISTORY = "chat_history"
CONVERSATIONAL_PIPELINE = "conversational_pipeline"
RELEVANT_DOCUMENTS = "None"
TITLE_NAME = "Data Alchemy RAG Bot"

login_config = config_loader("conf/local/users.yaml")
st.set_page_config(page_title=TITLE_NAME)
authenticator = stauth.Authenticate(**login_config)


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
        CHAT_HISTORY: [],
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
    if prompt := st.chat_input("Wie kann ich Dir heute helfen?"):
        # Render user message.
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "user", "content": prompt}
        )

        # Render AI assistant's response.
        with st.chat_message("assistant"):
            with st.spinner("Antwort wird generiert . . ."):
                response, documents = st.session_state[CONVERSATIONAL_PIPELINE].query(
                    prompt
                )
                st.session_state[RELEVANT_DOCUMENTS] = documents
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "assistant", "content": response}
        )


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
        st.markdown("#### Folgende Dokumente wurden als Kontext vewrwendet:")

        relevant_docs = defaultdict(list)
        for document in st.session_state[RELEVANT_DOCUMENTS]:
            doc_path = document.meta["file_path"]
            doc_page = document.meta["page_number"]
            # doc_relevance_score = document.score
            content = document.content
            relevant_docs[doc_path].append(
                {
                    "page": doc_page,
                    "content": content,
                    # "relevance_score": doc_relevance_score,
                }
            )

        for file_path, docs in relevant_docs.items():
            with st.expander(f"File: {file_path}"):
                for doc in docs:
                    st.markdown(f"**Seite:** {doc['page']}")
                    # st.markdown(f"**Relevanz:** {doc['relevance_score']:.4f}")
                    st.markdown(f"**Inhalt:** {doc['content']}")
                    st.markdown("---")


if __name__ == "__main__":
    main()
