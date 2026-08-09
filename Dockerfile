# syntax=docker/dockerfile:1

# ---------- Stage 1: builder ----------
# Installs dependencies into an isolated prefix so the runtime stage
# doesn't need build tools (gcc, headers) at all, keeping the final
# image smaller and reducing attack surface.
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Unpinned to avoid breakage when the base debian image updates.
# archives typically only retain the latest build-essential version for
# a given release, so this exact pin will need bumping if the base
# image's underlying release changes and the old version ages out —
# check with `apt-cache policy build-essential` inside the builder
# stage if this build ever fails on this line.

COPY requirements.txt .
# CPU-only torch first — see requirements.txt comment for why this is
# separate from the rest of the install. Deliberately kept as two RUN
# layers rather than merged (hadolint DL3059 flags this as
# consolidatable) so the torch layer stays cached across rebuilds where
# only requirements.txt changes — torch changes far less often than the
# rest of the dependency list.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

# Run as a non-root user — standard hardening for a container that
# accepts external HTTP traffic.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY artifacts/model/ ./artifacts/model/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_SOURCE=local \
    MODEL_LOCAL_PATH=/app/artifacts/model \
    PYTHONPATH=/app

USER 1000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else sys.exit(1)"]

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
