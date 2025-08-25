FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install PostgreSQL client for pg_isready command
RUN apt-get update && apt-get install -y postgresql-client git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

RUN uv sync --frozen --no-install-project

COPY . /app

RUN uv sync --frozen

COPY ./entrypoint.sh /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uv", "run", "fastapi", "src/rag/app.py", "--port", "${BACKEND_PORT}", "--host", "${BACKEND_HOST}"]
