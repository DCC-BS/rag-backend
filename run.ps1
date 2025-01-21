# For Windows
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)/src"
uv run streamlit run src/ui/app.py 