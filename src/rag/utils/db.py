import os


def get_db_url():
    """Construct database URL from environment variables."""
    user = os.getenv("POSTGRES_USER", "myuser")
    password = os.getenv("POSTGRES_PASSWORD", "mypassword")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "mydatabase")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
