import os

import pandas as pd
import streamlit as st

from ui.constants import UI_RENDERED_MESSAGES
from utils.config import get_config


def render_feedback_section():
    """
    Render compact feedback buttons (thumbs up/down) for the AI response.
    Aligned to the right of the chat answer.
    """
    config = get_config()

    if (
        st.session_state[UI_RENDERED_MESSAGES]
        and st.session_state[UI_RENDERED_MESSAGES][-1]["role"] == "assistant"
    ):
        # Create a container with right-aligned content
        with st.container():
            col1, col2, col3 = st.columns([6, 0.5, 0.5])

            with col2:
                if st.button("👍", key="thumbs_up"):
                    feedback_data = {
                        "helpful": "Yes",
                        "reason": "",
                        "feedback": "",
                        "query": st.session_state[UI_RENDERED_MESSAGES][-2]["content"],
                        "response": st.session_state[UI_RENDERED_MESSAGES][-1]["content"],
                        "user": st.session_state["name"],
                        "model": config.LLM.MODEL,
                        "retriever": config.RETRIEVER,
                        "doc_store": config.DOC_STORE,
                    }
                    save_feedback(feedback_data)
                    st.success("Danke für dein Feedback!")

            with col3:
                if st.button("👎", key="thumbs_down"):
                    st.session_state["show_feedback_form"] = True
                    st.session_state["feedback_submitted"] = False

        if st.session_state.get("show_feedback_form", False) and not st.session_state.get(
            "feedback_submitted", False
        ):
            with st.container():
                with st.form("feedback_form"):
                    st.write("### Detailliertes Feedback")
                    reason = st.multiselect(
                        "Das ist schief gelaufen:",
                        [
                            "Kontext enthielt die korrekte Antwort, AI fand diese nicht",
                            "Kontext enthielt die korrekte Antwort nicht",
                            "Falsche Antwort",
                            "Antwort zu knapp / relevante Info weggelessen",
                            "Antwort zu lange",
                            "Anderes",
                        ],
                        key="feedback_reason",
                        placeholder="Grund auswählen",
                    )

                    additional_feedback = st.text_area("Zusätzliches Feedback (optional):")
                    submitted = st.form_submit_button("Feedback senden")

                    if submitted:
                        feedback_data = {
                            "helpful": "No",
                            "reason": reason,
                            "feedback": additional_feedback,
                            "query": st.session_state[UI_RENDERED_MESSAGES][-2]["content"],
                            "response": st.session_state[UI_RENDERED_MESSAGES][-1]["content"],
                            "user": st.session_state["name"],
                            "model": config.LLM.MODEL,
                            "retriever": config.RETRIEVER,
                            "doc_store": config.DOC_STORE,
                        }
                        save_feedback(feedback_data)
                        st.session_state["feedback_submitted"] = True
                        st.session_state["show_feedback_form"] = False
                        st.success("Danke für dein Feedback!")
                        st.rerun()


def save_feedback(feedback_data):
    """
    Save feedback data to a CSV file.
    """
    feedback_file = "feedback.csv"

    feedback_data["response"] = " ".join(feedback_data["response"].split())
    feedback_data["response"] = feedback_data["response"].replace("\n", " ")
    feedback_data["feedback"] = " ".join(feedback_data["feedback"].split())
    feedback_data["feedback"] = feedback_data["feedback"].replace("\n", " ")

    if not os.path.exists(feedback_file):
        df = pd.DataFrame(columns=feedback_data.keys())
    else:
        df = pd.read_csv(feedback_file, encoding="utf-8")

    new_row = pd.DataFrame([feedback_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(feedback_file, index=False, encoding="utf-8")
