# For Windows
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)/src"
uv run main.py