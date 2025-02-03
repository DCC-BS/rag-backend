#!/bin/bash
# For Linux/Mac
export PYTHONPATH=$PYTHONPATH:$(pwd)
uv run streamlit run ui/app.py