import streamlit as st
from dotenv import load_dotenv

from document_storrage import create_inmemory_document_store
from rag_pipeline import SHRAGPipeline
from collections import defaultdict

load_dotenv()

TITLE_NAME = 'Sozialhilfe RAG Chat Bot'
UI_RENDERED_MESSAGES = 'ui_rendered_messages'
CHAT_HISTORY = 'chat_history'
CONVERSATIONAL_PIPELINE = 'conversational_pipeline'
RELEVANT_DOCUMENTS = 'None'


def main():
    """
        Render the retrieval augmented generation (RAG) chatbot application.
    """
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
    document_store = create_inmemory_document_store()
    return {
        TITLE_NAME: 'Sozialhilfe RAG Chat Bot',
        UI_RENDERED_MESSAGES: [],
        CHAT_HISTORY: [],
        CONVERSATIONAL_PIPELINE: SHRAGPipeline(document_store),
        RELEVANT_DOCUMENTS: []
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
                response, documents = st.session_state[CONVERSATIONAL_PIPELINE].query(prompt)
                st.session_state[RELEVANT_DOCUMENTS] = documents
        st.session_state[UI_RENDERED_MESSAGES].append({'role': 'assistant', 'content': response})


def render_chat_history():
    """
        Display the chat message history stored in session state.
    """
    for message in st.session_state[UI_RENDERED_MESSAGES]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def render_debug_section():
    """
    Render a debug section showing relevant documents grouped by file path.
    """
    if st.session_state[RELEVANT_DOCUMENTS]:
        st.markdown("#### Debug Section: Relevant Documents")
        
        relevant_docs = defaultdict(list)
        for document in st.session_state[RELEVANT_DOCUMENTS]:
            doc_path = document.meta['file_path']
            doc_page = document.meta['page_number']
            doc_relevance_score = document.score
            content = document.content
            relevant_docs[doc_path].append({
                'page': doc_page,
                'content': content,
                'relevance_score': doc_relevance_score
            })
        
        for file_path, docs in relevant_docs.items():
            with st.expander(f"File: {file_path}"):
                for doc in docs:
                    st.markdown(f"**Page:** {doc['page']}")
                    st.markdown(f"**Relevance Score:** {doc['relevance_score']:.4f}")
                    st.markdown(f"**Content:** {doc['content']}")
                    st.markdown("---")


if __name__ == '__main__':
    main()