# docker/Dockerfile
#
# Multi-stage build. The builder stage installs Python dependencies
# into an isolated virtualenv; the final stage copies ONLY that
# virtualenv plus the application source — none of the build
# toolchain that produced it. Smaller image, smaller attack surface.

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# build-essential is defensive insurance, not a known hard requirement:
# fastapi/uvicorn/pydantic/redis-py all ship prebuilt wheels, and so
# does faster-whisper's ctranslate2 dependency on standard x86_64
# targets. Kept here anyway because it's cheap in a stage that never
# ships, and the failure mode of NOT having it — a cryptic compiler
# error mid-build on some unexpected platform — is worse than the
# extra seconds this costs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# An isolated virtualenv, not a global pip install — this is what
# makes "copy /opt/venv into the final stage" a clean, self-contained
# operation, with no risk of dragging along anything else that got
# installed into the base image's system Python.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# NOTE: assumes a pyproject.toml at the project root, per the
# structure agreed at the start of this build — not yet written in
# this conversation. The one dangling dependency in this file.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system dependencies — NOT build-time ones, so these belong
# in this stage, not the builder:
#   ffmpeg      - faster-whisper's decode path shells out to ffmpeg
#                 when given a file-like object. See the note in
#                 workers/transcribe.py on why we deliberately hand it
#                 a fresh BytesIO rather than a pre-decoded array.
#   libsndfile1 - the shared library soundfile's Python bindings load
#                 at import time, used for our own cheap silence-check
#                 decode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. --create-home matters specifically here: faster-
# whisper/huggingface_hub caches downloaded model weights under
# $HF_HOME (set below, inside this home directory) — without a real
# home dir, that cache falls back to somewhere unpredictable, or fails.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Explicit, stable cache location for downloaded model weights. Set
# BEFORE switching to appuser and before the app's first run, so it's
# consistent regardless of what faster-whisper's own default would
# otherwise resolve to. docker-compose.yml mounts a named volume at
# exactly this path — see the comment there for why that matters.
ENV HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/

USER appuser

EXPOSE 8000

# No dedicated /healthz route exists yet in api/ — this checks only
# that uvicorn is accepting HTTP connections and routing requests (a
# 404 still proves the server is alive; connection-refused means it
# isn't). That's a real but weak signal. A proper health endpoint —
# one that also confirms pool.get_pool() has live workers and reports
# Redis reachability — is a natural near-term addition that should
# replace this check, not just supplement it.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/', timeout=2)" || exit 1

# Deliberately NOT --workers N. This app manages its own internal
# CPU-bound concurrency via a ProcessPoolExecutor (workers/pool.py) —
# a single event-loop process is enough to keep that pool fed, since
# the event loop itself only ever does cheap I/O-bound work (network
# framing, cache lookups) and hands anything CPU-heavy off to the
# pool. Running multiple uvicorn workers here would spin up a SEPARATE
# ProcessPoolExecutor (and a separate LocalCache) per uvicorn worker —
# multiplying total model memory usage for no throughput benefit,
# since the actual bottleneck (CPU-bound inference) is already
# parallelized one layer down. Scale this service by running more
# container replicas behind a load balancer, not by adding uvicorn
# workers within one.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]