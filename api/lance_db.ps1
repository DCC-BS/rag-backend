# For Windows
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"
uv run setup_lancedb.py