import sys
import uuid

import streamlit as st
import streamlit_authenticator as stauth
import structlog
from components.chat import manage_chat, render_chat_history, render_example_queries
from components.feedback import render_feedback_section
from dotenv import load_dotenv
from rich.traceback import install

from core.rag_pipeline import SHRAGPipeline
from ui.constants import (
    CONVERSATIONAL_PIPELINE,
    RELEVANT_DOCUMENTS,
    TITLE_NAME,
    UI_RENDERED_MESSAGES,
)
from utils.config import get_config, load_config

load_dotenv()
install(show_locals=True)
logger = structlog.stdlib.get_logger()


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handles unhandled exceptions, logs them using the logger."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception

# Load initial configuration
load_config()
config = get_config()
st.set_page_config(
    page_title=TITLE_NAME,
    page_icon=":speech_balloon:",
    initial_sidebar_state="auto",
    menu_items=None,
)
authenticator = stauth.Authenticate(**config.LOGIN_CONFIG)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if UI_RENDERED_MESSAGES not in st.session_state:
        st.session_state[UI_RENDERED_MESSAGES] = []
    if RELEVANT_DOCUMENTS not in st.session_state:
        st.session_state[RELEVANT_DOCUMENTS] = []
    if CONVERSATIONAL_PIPELINE not in st.session_state:
        st.session_state[CONVERSATIONAL_PIPELINE] = None
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = uuid.uuid4()


def main():
    """Render the retrieval augmented generation (RAG) chatbot application."""
    authentication()
    initialize_session_state()
    setup_page()

    # Only render chat components if user is authenticated
    if st.session_state.get("authentication_status"):
        render_chat_history()
        manage_chat()
        render_feedback_section()
    # render_debug_section()


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
            st.session_state[CONVERSATIONAL_PIPELINE] = SHRAGPipeline()
    elif st.session_state["authentication_status"] is False:
        st.error("Benutzername oder Passwort sind falsch")
    elif st.session_state["authentication_status"] is None:
        st.warning("Bitte Benutzername und Passwort eingeben")


def authentication():
    try:
        authenticator.login()
    except stauth.LoginError as e:
        st.error(e)


if __name__ == "__main__":
    main()
