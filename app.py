import streamlit as st
from dotenv import load_dotenv

from document_storrage import create_inmemory_document_store
from rag_pipeline import SHRAGPipeline

load_dotenv()

TITLE_NAME = 'Sozialhilfe RAG Chat Bot'
UI_RENDERED_MESSAGES = 'ui_rendered_messages'
CHAT_HISTORY = 'chat_history'
CONVERSATIONAL_PIPELINE = 'conversational_pipeline'


def main():
    """
        Render the retrieval augmented generation (RAG) chatbot application.
    """
    config = load_config()
    initialize_session_state(config)
    setup_page()
    render_chat_history()
    manage_chat()


def load_config():
    """
        Load the application configuration from a file or object.

    Returns:
        dict: Configuration dictionary containing title name,
              UI rendered messages, chat history, and conversational pipeline instance.
    """
    document_store = create_inmemory_document_store()
    return {
        TITLE_NAME: 'Sozialhilfe RAG Chat Bot',
        UI_RENDERED_MESSAGES: [],
        CHAT_HISTORY: [],
        CONVERSATIONAL_PIPELINE: SHRAGPipeline(document_store)
    }


def setup_page():
    """
        Set Streamlit page configuration and title.
    """
    st.set_page_config(page_title=st.session_state[TITLE_NAME])
    st.title(st.session_state[TITLE_NAME])


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
    if prompt := st.chat_input('What can we help you with?'):
        # Render user message.
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append({'role': 'user', 'content': prompt})

        # Render AI assistant's response.
        with st.chat_message('assistant'):
            with st.spinner('Generating response . . .'):
                response = st.session_state[CONVERSATIONAL_PIPELINE].query(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append({'role': 'assistant', 'content': response})


def render_chat_history():
    """
        Display the chat message history stored in session state.
    """
    for message in st.session_state[UI_RENDERED_MESSAGES]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


if __name__ == '__main__':
    main()