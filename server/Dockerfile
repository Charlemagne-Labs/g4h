# g4h live-demo container.
#
# Stages:
#   1. base: Python 3.10 (Ubuntu 22.04 system) + g4h source + g4h `[train]`
#      deps (torch, transformers, peft, bitsandbytes) so the trained adapter
#      can be loaded.
#   2. playwright: install Chromium + system deps used by `server/fetch.py`.
#
# Expects the trained artifact tarball to be placed at `runs/gemma4-e4b-cls/`
# inside the build context (or mounted as a volume at runtime).
#
# The image is GPU-aware via the nvidia/cuda runtime — for AWS EC2 g5
# instances with the NVIDIA Container Toolkit installed, run with
# `docker run --gpus all ...` and the model will use CUDA. On a CPU-only
# host the same image works but inference is ~20-30 s per request.
#
# Why no Python 3.11 install: we previously installed 3.11 alongside the
# system 3.10 and used update-alternatives to make `python` point to 3.11.
# But pip from `python3-pip` is bound to 3.10, so packages landed in 3.10's
# site-packages while `python -m playwright` (now 3.11) couldn't find them.
# Sticking to the system Python avoids the interpreter mismatch entirely;
# our pyproject.toml only requires >=3.10.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    G4H_ARTIFACT_DIR=/app/runs/gemma4-e4b-cls

# Python + the system libs Playwright's Chromium needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git curl ca-certificates \
      libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
      libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
      libcairo2 libasound2 libdrm2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install the g4h package + extras first so the layer caches across server changes.
# Use `python -m pip` (not bare `pip`) so the installer and the runtime
# interpreter are guaranteed to be the same.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --upgrade pip wheel \
 && python -m pip install -e ".[train]"

# Server-only deps (FastAPI + Playwright) + Chromium binary.
COPY server/requirements.txt ./server/requirements.txt
RUN python -m pip install -r ./server/requirements.txt \
 && python -m playwright install --with-deps chromium

# Server source.
COPY server/ ./server/

# Artifact: mount at runtime via `-v $(pwd)/runs:/app/runs` for fast iteration,
# OR ADD it into the image for a self-contained deployable.
# COPY runs/gemma4-e4b-cls/ /app/runs/gemma4-e4b-cls/

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
