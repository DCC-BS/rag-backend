import shutil
import subprocess
import sys
from pathlib import Path


def run_app() -> None:
    """Run the Streamlit application.

    Raises:
        FileNotFoundError: If streamlit executable is not found
        ValueError: If app.py is not found or is not a file
    """
    streamlit_path = shutil.which("streamlit")
    if not streamlit_path:
        raise FileNotFoundError("Streamlit executable not found. Please ensure streamlit is installed.")

    app_path = Path(__file__).parent / "app.py"
    if not app_path.is_file():
        raise ValueError(f"Application file not found at {app_path}")
    try:
        _ = subprocess.run(
            [streamlit_path, "run", str(app_path.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        print(f"Error running streamlit: {err.stderr}", file=sys.stderr)
        raise


if __name__ == "__main__":
    run_app()
