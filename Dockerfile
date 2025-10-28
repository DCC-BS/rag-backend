FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

# Install PostgreSQL client for pg_isready command
RUN apt-get update && apt-get install -y postgresql-client git build-essential && rm -rf /var/lib/apt/lists/*

# Create non-root user/group for rootless runtime
ARG APP_USER=app
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd -g ${APP_GID} ${APP_USER} \
    && useradd -m -u ${APP_UID} -g ${APP_GID} -s /usr/sbin/nologin ${APP_USER}
ENV HOME=/home/${APP_USER}

WORKDIR /app

COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

RUN uv sync --frozen --no-install-project

COPY . /app

RUN uv sync --frozen

COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Ensure app files are owned by non-root user
RUN chown -R ${APP_UID}:${APP_GID} /app ${HOME}

# Drop privileges for runtime
USER ${APP_UID}:${APP_GID}

ENTRYPOINT ["/app/entrypoint.sh"]

CMD uv run uvicorn src.rag.app:app --port ${BACKEND_PORT} --host ${BACKEND_HOST}
