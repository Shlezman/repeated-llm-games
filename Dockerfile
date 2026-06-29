# syntax=docker/dockerfile:1
# App image built with uv. Two-stage dependency caching, non-root runtime.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MPLBACKEND=Agg \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Resolve + install dependencies first (cached layer; project not yet copied).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
        --extra openai --extra anthropic --extra postgres

# 2) Copy source + data, then install the project itself.
COPY src ./src
COPY config ./config
COPY data ./data
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra openai --extra anthropic --extra postgres

# Non-root runtime user; results dir is a mount point owned by that user.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/results \
 && chown -R app:app /app
USER app

ENTRYPOINT ["python", "-m", "llmgames.cli"]
CMD ["run", "--config", "config/runs/llm_gw_multi.yaml"]
