import streamlit as st

from ui.constants import (
    CONVERSATIONAL_PIPELINE,
    RELEVANT_DOCUMENTS,
    UI_RENDERED_MESSAGES,
)
from utils.config import get_config


def process_citations(text, relevant_docs):
    import re

    def replacement(match):
        filename = match.group(1).strip()
        for doc in relevant_docs:
            if doc.metadata.get("filename", "").lower() == filename.lower():
                tooltip_content = doc.page_content
                # Optionally truncate long document content
                if len(tooltip_content) > 200:
                    tooltip_content = tooltip_content[:200] + "..."
                # Escape double quotes
                tooltip_content = tooltip_content.replace('"', "&quot;")
                return f'<span title="{tooltip_content}" style="border-bottom: 1px dotted #000; cursor: help;">[{filename}]</span>'
        return match.group(0)

    return re.sub(r"\[([^\]]+)\]", replacement, text)


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
        st.session_state["chat_started"] = True

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("Suche relevante Dokumente...", expanded=True) as status:
                full_response = ""
                flow_updates = []

                def response_generator():
                    nonlocal full_response, flow_updates
                    for chunk in st.session_state[CONVERSATIONAL_PIPELINE].stream_query(
                        prompt, thread_id
                    ):
                        if isinstance(chunk, tuple):
                            graph_state, graph_output = chunk
                            flow_updates.append((graph_state, graph_output))
                            status.update(label=graph_state, state="running")
                            if (
                                isinstance(graph_output, list)
                                and graph_state == "Relevante Dokumente gefunden"
                            ):
                                st.session_state[RELEVANT_DOCUMENTS] = graph_output
                                yield f"{len(graph_output)} Dokumente gefunden \n\n"
                            else:
                                yield graph_output + "\n"
                        else:
                            full_response += str(chunk)
                            yield str(chunk)
                    status.update(label="Fertig!", state="complete")

                st.write_stream(response_generator())

            st.session_state[UI_RENDERED_MESSAGES].append(
                {"role": "assistant", "content": full_response, "flow_updates": flow_updates}
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
            if message["role"] == "assistant":
                # Retrieve relevant documents from session state
                relevant_docs = st.session_state.get(RELEVANT_DOCUMENTS, [])
                processed_text = process_citations(message["content"], relevant_docs)
                st.markdown(processed_text, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

            if message.get("flow_updates"):
                with st.expander("Arbeitsablauf", expanded=False):
                    st.write("Details der einzelnen Arbeitsschritte im Arbeitsablauf:")
                    for node_label, node_output in message["flow_updates"]:
                        if node_label.startswith("Update generate_answer"):
                            continue
                        if isinstance(node_output, list):
                            st.markdown(f"**{node_label}:** Gefundene Dokumente:")

                            for doc in node_output:
                                filename = doc.metadata.get("filename", "unknown")
                                st.markdown(f"- {filename}")
                        else:
                            st.markdown(f"**{node_label}:** {node_output}")
