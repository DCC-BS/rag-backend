import streamlit as st

from ui.constants import (
    CONVERSATIONAL_PIPELINE,
    RELEVANT_DOCUMENTS,
    UI_RENDERED_MESSAGES,
)


def manage_chat():
    """
    Handle user interaction with the conversational AI and render
    the user query along with the AI response.
    """
    prompt = st.session_state.get("user_input") or st.chat_input(
        "Wie kann ich Dir heute helfen?"
    )
    if prompt is not None:
        st.session_state.user_input = None

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("assistant"):
            # Create a status container for documents
            with st.status("Suche relevante Dokumente...", expanded=True) as status:
                full_response = ""
                
                def response_generator():
                    nonlocal full_response
                    stream = st.session_state[CONVERSATIONAL_PIPELINE].stream_query(prompt)
                    for chunk in stream:
                        if isinstance(chunk, list):  # This is the documents
                            st.session_state[RELEVANT_DOCUMENTS] = chunk
                            status.update(label="Antwort generieren...", state="running")
                            yield f"{len(chunk)} Dokumente gefunden \n\n"
                        else:
                            # This is the text chunk
                            full_response += str(chunk )
                            yield str(chunk)
                    status.update(label="Fertig!", state="complete")
                st.write_stream(response_generator())
                
        st.session_state[UI_RENDERED_MESSAGES].append(
            {"role": "assistant", "content": full_response}
        )


def render_example_queries():
    st.subheader("Beispiel Fragen:")
    example_queries = [
        "Was für Angebote zur Familienergänzenden Tagesbetreuung gibt es?",
        "Ich habe meine Stelle verloren, was muss ich tun?",
        "Werden Beiträge der Hilflosenentschädigung von den Ansprüchen auf andere Sozialbeiträge abgezogen?",
    ]
    cols = st.columns(3)
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
