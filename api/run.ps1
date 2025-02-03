# For Windows
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"
uv run streamlit run ui/app.py 