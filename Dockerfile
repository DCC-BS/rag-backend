FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install PostgreSQL client for pg_isready command
RUN apt-get update && apt-get install -y postgresql-client git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

RUN uv sync --frozen --no-install-project

COPY . /app

RUN uv sync --frozen

COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD uv run uvicorn src.rag.app:app --port ${BACKEND_PORT} --host ${BACKEND_HOST}
