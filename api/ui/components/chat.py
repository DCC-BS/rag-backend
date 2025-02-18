import streamlit as st

from ui.constants import (
    CONVERSATIONAL_PIPELINE,
    RELEVANT_DOCUMENTS,
    UI_RENDERED_MESSAGES,
)
from utils.config import get_config
from utils.stream_response import StreamResponseType


def process_citations(text, relevant_docs):
    import re

    def replacement(match):
        filename = match.group(1).strip()
        for doc in relevant_docs:
            if doc.metadata.get("filename", "").lower() == filename.lower():
                tooltip_content = doc.page_content
                if len(tooltip_content) > 200:
                    tooltip_content = tooltip_content[:200] + "..."
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

    # Handle resume from interrupt if needed
    if "interrupt_state" in st.session_state:
        interrupt_type = st.session_state["interrupt_state"]["type"]
        if interrupt_type == "needs_information":
            st.info("Bitte geben Sie weitere Informationen ein:")
            additional_info = st.text_area("Zusätzliche Informationen")
            if st.button("Informationen senden"):
                with st.chat_message("assistant"):
                    with st.status(
                        "Verarbeite zusätzliche Informationen...", expanded=True
                    ) as status:
                        process_resume_stream(
                            thread_id=thread_id,
                            status=status,
                            action="provide_information",
                            data=additional_info,
                        )
                del st.session_state["interrupt_state"]
                st.rerun()
        elif interrupt_type == "needs_approval":
            st.info("Bitte überprüfen Sie die umformulierte Frage:")
            rewritten_query = st.session_state["interrupt_state"]["rewritten_query"]
            st.text(rewritten_query)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Akzeptieren"):
                    with st.chat_message("assistant"):
                        with st.status(
                            "Verarbeite Anfrage...", expanded=True
                        ) as status:
                            process_resume_stream(
                                thread_id=thread_id,
                                status=status,
                                action="accept",
                            )
                    del st.session_state["interrupt_state"]
                    st.rerun()
            with col2:
                if st.button("Ändern"):
                    st.session_state["modifying_query"] = True
                    st.session_state["original_query"] = rewritten_query

        if st.session_state.get("modifying_query"):
            modified_query = st.text_area(
                "Bearbeiten Sie die Frage:", value=st.session_state["original_query"]
            )
            if st.button("Geänderte Frage senden"):
                with st.chat_message("assistant"):
                    with st.status(
                        "Verarbeite geänderte Frage...", expanded=True
                    ) as status:
                        process_resume_stream(
                            thread_id=thread_id,
                            status=status,
                            action="modify",
                            data=modified_query,
                        )
                del st.session_state["interrupt_state"]
                del st.session_state["modifying_query"]
                del st.session_state["original_query"]
                st.rerun()
        return

    prompt = st.session_state.get("user_input") or st.chat_input(
        config.CHAT.DEFAULT_PROMPT
    )
    if prompt is not None:
        st.session_state.user_input = None
        st.session_state["chat_started"] = True

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("assistant"):
            with st.status("Suche relevante Dokumente...", expanded=True) as status:
                process_stream(prompt, thread_id, status)


def process_stream(prompt: str, thread_id: str, status):
    """Process the stream of responses from the RAG pipeline"""
    full_response = ""
    flow_updates = []

    for chunk in st.session_state[CONVERSATIONAL_PIPELINE].stream_query(
        prompt, "Sozialhilfe", thread_id
    ):
        if chunk.type == StreamResponseType.STATUS:
            flow_updates.append((chunk.message, chunk.decision))
            status.update(label=chunk.message, state="running")
        elif chunk.type == StreamResponseType.DOCUMENTS:
            flow_updates.append((chunk.message, chunk.documents))
            status.update(label=chunk.message, state="running")
            st.session_state[RELEVANT_DOCUMENTS] = chunk.documents
        elif chunk.type == StreamResponseType.ANSWER:
            full_response += chunk.answer
            if "placeholder" not in st.session_state:
                st.session_state.placeholder = st.empty()
            st.session_state.placeholder.markdown(full_response)
        elif chunk.type == StreamResponseType.INTERRUPT:
            flow_updates.append(
                (
                    "human_in_the_loop",
                    (chunk.metadata["question"], chunk.metadata["rewritten_query"]),
                )
            )
            status.update(label="Feedback benötigt", state="complete")
            st.info(chunk.metadata["question"])
            st.session_state["interrupt_state"] = {
                "type": chunk.metadata["type"],
                "question": chunk.metadata["question"],
                "rewritten_query": chunk.metadata["rewritten_query"],
            }
            break

    status.update(
        label="Fertig!"
        if "interrupt_state" not in st.session_state
        else "Warte auf Feedback",
        state="complete",
    )

    st.session_state[UI_RENDERED_MESSAGES].append(
        {
            "role": "assistant",
            "content": full_response,
            "flow_updates": flow_updates,
        }
    )


def process_resume_stream(thread_id: str, status, action: str, data: str | None = None):
    """Process the stream of responses after resuming from an interrupt"""
    full_response = ""
    flow_updates = []

    for chunk in st.session_state[CONVERSATIONAL_PIPELINE].resume_query(
        thread_id=thread_id, action=action, data=data
    ):
        if chunk.type == StreamResponseType.STATUS:
            flow_updates.append((chunk.message, ""))
            status.update(label=chunk.message, state="running")
        elif chunk.type == StreamResponseType.DOCUMENTS:
            flow_updates.append((chunk.message, chunk.documents))
            status.update(label=chunk.message, state="running")
            st.session_state[RELEVANT_DOCUMENTS] = chunk.documents
        elif chunk.type == StreamResponseType.ANSWER:
            full_response += chunk.answer
            if "placeholder" not in st.session_state:
                st.session_state.placeholder = st.empty()
            st.session_state.placeholder.markdown(full_response)
        elif chunk.type == StreamResponseType.INTERRUPT:
            flow_updates.append(
                (
                    "human_in_the_loop",
                    (chunk.metadata["question"], chunk.metadata["rewritten_query"]),
                )
            )
            status.update(label="Feedback benötigt", state="complete")
            st.info(chunk.metadata["question"])
            st.session_state["interrupt_state"] = {
                "type": chunk.metadata["type"],
                "question": chunk.metadata["question"],
                "rewritten_query": chunk.metadata["rewritten_query"],
            }
            break

    status.update(
        label="Fertig!"
        if "interrupt_state" not in st.session_state
        else "Warte auf Feedback",
        state="complete",
    )

    st.session_state[UI_RENDERED_MESSAGES].append(
        {
            "role": "assistant",
            "content": full_response.replace("ß", "ss"),
            "flow_updates": flow_updates,
        }
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
                message_content = message["content"].replace("ß", "ss")
                processed_text = process_citations(message_content, relevant_docs)
                st.markdown(processed_text, unsafe_allow_html=True)
                if st.session_state.get("human_in_the_loop", False):
                    st.markdown(
                        f"**Frage konnte nicht umformuliert werden. Feedback benötigt:** {st.session_state.get('question_to_user')}; Vorschlag der AI: {st.session_state.get('ai_proposed_query')}"
                    )
                    st.session_state.pop("human_in_the_loop")
                    feedback = st.text_area("Feedback")
                    if st.button("Feedback absenden"):
                        st.session_state.user_feedback = feedback
                        st.rerun()
            else:
                st.markdown(message["content"])

            if message.get("flow_updates"):
                with st.expander("Arbeitsablauf", expanded=False):
                    st.write("Details der einzelnen Arbeitsschritte im Arbeitsablauf:")
                    for node_label, node_output in message["flow_updates"]:
                        if node_label.startswith("Update generate_answer"):
                            continue
                        if node_label == "human_in_the_loop":
                            question_to_user, rewritten_query = node_output
                            st.markdown(
                                f"**Frage konnte nicht umformuliert werden. Feedback benötigt:** {question_to_user}; Vorschlag der AI: {rewritten_query}"
                            )
                            st.session_state.user_input = rewritten_query
                            continue
                        if isinstance(node_output, list):
                            st.markdown(f"**{node_label}:** Gefundene Dokumente:")

                            for doc in node_output:
                                filename = doc.metadata.get("filename", "unknown")
                                st.markdown(f"- {filename}")
                        elif isinstance(node_output, str):
                            st.markdown(f"**{node_label}:** {node_output}")
                        else:
                            st.markdown(f"**{node_label}**")
