import os

import pandas as pd
import streamlit as st

from ui.constants import UI_RENDERED_MESSAGES
from utils.config import get_config


def render_feedback_section():
    """
    Render a feedback section for the user to provide feedback on the AI response.
    """
    config = get_config()  # Get config here

    if (
        st.session_state[UI_RENDERED_MESSAGES]
        and st.session_state[UI_RENDERED_MESSAGES][-1]["role"] == "assistant"
    ):
        with st.expander("Feedback geben", expanded=True):
            col1, col2 = st.columns([1, 3])

            with col1:
                is_helpful = st.radio(
                    "War die Antwort hilfreich?", ["Yes", "No"], key="feedback_helpful"
                )

            feedback_data = {
                "helpful": is_helpful,
                "reason": "",
                "feedback": "",
                "query": st.session_state[UI_RENDERED_MESSAGES][-2]["content"],
                "response": st.session_state[UI_RENDERED_MESSAGES][-1]["content"],
                "user": st.session_state["name"],
                "model": config.LLM.MODEL,
                "retriever": config.RETRIEVER,
                "doc_store": config.DOC_STORE,
            }

            if is_helpful == "No":
                with col2:
                    feedback_data["reason"] = st.multiselect(
                        "Das ist schief gelaufen:",
                        [
                            "Kontext enthielt die korrekte Antwort, AI fand diese nicht",
                            "Kontext enthielt die korrekte Antwort nicht",
                            "Falsche Antwort",
                            "Antwort zu knapp / relevante Info weggelessen"
                            "Antwort zu lange",
                            "Anderes",
                        ],
                        key="feedback_reason",
                        placeholder="Grund auswählen",
                    )
                    feedback_data["feedback"] = st.text_area(
                        "Zusätzliches Feedback (optional):"
                    )

            if st.button("Feedback senden", key="feedback_submit"):
                save_feedback(feedback_data)
                st.success("Danke für deinFeedback!")


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
