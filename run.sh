#!/bin/bash
# For Linux/Mac
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uv run streamlit run src/ui/app.py 