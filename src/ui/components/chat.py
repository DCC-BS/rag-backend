import streamlit as st

from ui.constants import (
    CONVERSATIONAL_PIPELINE,
    RELEVANT_DOCUMENTS,
    UI_RENDERED_MESSAGES,
)
from utils.config import get_config


def manage_chat():
    """
    Handle user interaction with the conversational AI and render
    the user query along with the AI response.
    """
    config = get_config()
    thread_id = st.session_state["user_id"]

    prompt = st.session_state.get("user_input") or st.chat_input(config.CHAT.DEFAULT_PROMPT)
    if prompt is not None:
        st.session_state.user_input = None
        # Add a flag to indicate that chat has started
        st.session_state["chat_started"] = True

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("Suche relevante Dokumente...", expanded=True) as status:
                full_response = ""

                def response_generator():
                    nonlocal full_response
                    for chunk in st.session_state[CONVERSATIONAL_PIPELINE].stream_query(
                        prompt, thread_id
                    ):
                        if isinstance(chunk, tuple):
                            graph_state, graph_output = chunk
                            status.update(label=graph_state, state="running")
                            if isinstance(graph_output, list):
                                st.session_state[RELEVANT_DOCUMENTS] = graph_output
                                yield f"{len(graph_output)} Dokumente gefunden \n\n"
                            else:
                                yield graph_output + "\n"
                        else:
                            # This is the text chunk
                            full_response += str(chunk)
                            yield str(chunk)
                    status.update(label="Fertig!", state="complete")

                st.write_stream(response_generator())

        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "assistant", "content": full_response}
        )


def render_example_queries():
    # Only show example queries if chat hasn't started
    if not st.session_state.get("chat_started", False):
        config = get_config()
        st.subheader("Beispiel Fragen:")
        example_queries = config.CHAT.EXAMPLE_QUERIES
        cols = st.columns(len(example_queries))
        for col, query in zip(cols, example_queries, strict=True):
            with col:
                if st.button(query):
                    st.session_state.user_input = query


def render_chat_history():
    """
    Display the chat message history stored in session state.
    """
    for message in st.session_state[UI_RENDERED_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
