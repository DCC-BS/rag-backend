from collections import defaultdict
import os

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv

from rag_pipeline import SHRAGPipeline
from utils import config_loader
import pandas as pd

load_dotenv()

UI_RENDERED_MESSAGES = "ui_rendered_messages"
CHAT_HISTORY = "chat_history"
CONVERSATIONAL_PIPELINE = "conversational_pipeline"
RELEVANT_DOCUMENTS = "None"
TITLE_NAME = "Data Alchemy RAG Bot"

login_config = config_loader("conf/local/users.yaml")
config = config_loader("conf/conf.yaml")
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
        st.subheader(f"Sources: {', '.join(st.session_state['roles'])}")
        st.write(f'Welcome *{st.session_state["name"]}*')
        st.session_state[CONVERSATIONAL_PIPELINE] = SHRAGPipeline(
            st.session_state["roles"]
        )
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")


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
    if prompt := st.chat_input("What can we help you with?"):
        # Render user message.
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "user", "content": prompt}
        )

        # Render AI assistant's response.
        with st.chat_message("assistant"):
            with st.spinner("Generating response . . ."):
                response, documents = st.session_state[CONVERSATIONAL_PIPELINE].query(
                    prompt
                )
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
        with st.expander("Provide Feedback", expanded=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                is_helpful = st.radio("Was this helpful?", ["Yes", "No"], key="feedback_helpful")
            
            feedback_data = {
                "helpful": is_helpful,
                "reason": "",
                "feedback": "",
                "query": st.session_state[UI_RENDERED_MESSAGES][-2]["content"],
                "response": st.session_state[UI_RENDERED_MESSAGES][-1]["content"],
                "user": st.session_state["name"],
                "model": config["LLM"]["MODEL"],
                "retriever": config["RETRIEVER"],
            }
            
            if is_helpful == "No":
                with col2:
                    feedback_data["reason"] = st.selectbox(
                        "Please select the reason why it was not helpful:",
                        [
                            "Too long",
                            "Incorrect answer",
                            "Context did not contain the answer",
                            "Context had the answer but AI did not find it",
                            "Other",
                        ],
                        key="feedback_reason",
                    )
                    feedback_data["feedback"] = st.text_area("Additional feedback (optional):")
            
            if st.button("Submit Feedback", key="feedback_submit"):
                save_feedback(feedback_data)
                st.success("Thank you for your feedback!")

def save_feedback(feedback_data):
    """
    Save feedback data to a CSV file.
    """
    feedback_file = "feedback.csv"
    
    if not os.path.exists(feedback_file):
        df = pd.DataFrame(columns=feedback_data.keys())
    else:
        df = pd.read_csv(feedback_file)
    
    new_row = pd.DataFrame([feedback_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(feedback_file, index=False)



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
        st.markdown("#### Debug Section: Relevant Documents")

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
                    st.markdown(f"**Page:** {doc['page']}")
                    # st.markdown(f"**Relevance Score:** {doc['relevance_score']:.4f}")
                    st.markdown(f"**Content:** {doc['content']}")
                    st.markdown("---")


if __name__ == "__main__":
    main()
