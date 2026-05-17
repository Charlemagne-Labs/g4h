# g4h live-demo container.
#
# Stages:
#   1. base: Python 3.11 + g4h source + g4h `[train]` deps (torch, transformers,
#      peft, bitsandbytes) so the trained adapter can be loaded.
#   2. playwright: install Chromium + system deps used by `server/fetch.py`.
#
# Expects the trained artifact tarball to be placed at `runs/gemma4-e4b-cls/`
# inside the build context (or mounted as a volume at runtime).
#
# The image is GPU-aware via the nvidia/cuda runtime — for AWS EC2 g5
# instances with the NVIDIA Container Toolkit installed, run with
# `docker run --gpus all ...` and the model will use CUDA. On a CPU-only
# host the same image works but inference is ~20-30 s per request.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    G4H_ARTIFACT_DIR=/app/runs/gemma4-e4b-cls

# Python + the system libs Playwright's Chromium needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip git curl ca-certificates \
      libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
      libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
      libcairo2 libasound2 libdrm2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install the g4h package + extras first so the layer caches across server changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip wheel \
 && pip install -e ".[train]"

# Server-only deps (FastAPI + Playwright).
COPY server/requirements.txt ./server/requirements.txt
RUN pip install -r ./server/requirements.txt \
 && python -m playwright install --with-deps chromium

# Server source.
COPY server/ ./server/

# Artifact: mount at runtime via `-v $(pwd)/runs:/app/runs` for fast iteration,
# OR ADD it into the image for a self-contained deployable.
# COPY runs/gemma4-e4b-cls/ /app/runs/gemma4-e4b-cls/

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
