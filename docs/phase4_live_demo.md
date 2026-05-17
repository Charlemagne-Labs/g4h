# Phase 4 — Live demo runbook

The live demo is a FastAPI server that:
1. Takes a URL from the browser.
2. Runs URL-string-only feature extraction (`src/extract.py`).
3. Optionally runs a targeted DOM fetch via Playwright (`server/fetch.py`) to enrich the indicator list.
4. Feeds the combined indicator string to the trained Gemma 4 classifier.
5. Returns a verdict (`allow` / `warn` / `block`) + per-label scores + the indicators that were used.

Tracks issue: [`gateguard-suite#73`](https://github.com/Charlemagne-Labs/gateguard-suite/issues/73), Phase 4.

---

## Goal

Ship a public URL where a hackathon judge can paste a URL and get a live classification within a few seconds — no notebook, no terminal, no clone-and-install dance. Demonstrates the trained model on inputs that weren't in the training set, with the extractor pipeline visible in the response.

---

## Layout

```
server/
├── __init__.py
├── app.py              FastAPI server with /predict endpoint
├── fetch.py            Playwright-based targeted DOM fetch
├── requirements.txt    FastAPI + uvicorn + playwright
├── Dockerfile          GPU-aware (nvidia/cuda base) container
└── static/
    ├── index.html      Single-page UI — "Charley · Gemma 4 E4B demo"
    ├── style.css       Design system tokens + components
    └── main.js         Form submit + result rendering
```

The model + extractor are loaded once at server startup via FastAPI's lifespan.

---

## Local dev (Mac)

Pre-req: you've completed Phase 3, the artifact is at `runs/gemma4-e4b-cls/`.

```bash
cd ~/CharlemagneLabs/g4h
source .venv/bin/activate

# Install the server deps + Playwright Chromium.
pip install -r server/requirements.txt
python -m playwright install chromium

# Run the server (it loads the model in-process on first start).
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open <http://localhost:8000>. Paste a URL. Try with and without the **FETCH DOM** toggle.

Expected timing on M4 Max:
- URL-only path: <2 s total, mostly model inference
- With DOM fetch: 3–12 s depending on the target page

If the model fails to load, the `/health` endpoint will return `model_loaded: false` and `/predict` returns 503. Check `G4H_ARTIFACT_DIR` (defaults to `runs/gemma4-e4b-cls`).

---

## Local Docker build

The Dockerfile is GPU-aware (`nvidia/cuda:12.1.1-runtime-ubuntu22.04` base) so the same image runs on CPU laptops or GPU instances.

```bash
# From the repo root:
docker build -t g4h-demo -f server/Dockerfile .

# Run with the artifact mounted (fastest iteration):
docker run --rm -p 8000:8000 \
    -v "$(pwd)/runs:/app/runs:ro" \
    g4h-demo

# On a GPU host (NVIDIA Container Toolkit installed):
docker run --rm -p 8000:8000 --gpus all \
    -v "$(pwd)/runs:/app/runs:ro" \
    g4h-demo
```

For a self-contained image (artifact baked in), uncomment the `COPY runs/...` line in the Dockerfile before building. The image becomes ~5 GB larger but you can `docker push` it directly to ECR without a separate artifact volume.

---

## Deploy on AWS

Three production-ish paths, ordered by setup cost. For a hackathon demo, **path A (EC2 g5.xlarge)** is the right tradeoff.

### Path A — EC2 g5.xlarge (recommended)

A10G GPU, 24 GB VRAM, $1.006/hr on-demand. The model fits with headroom; latency is ~1–2 s per inference.

**One-time setup:**

1. **Pick a region with g5 capacity** — `us-east-1`, `us-west-2`, `eu-west-1` are the most reliable. Check **EC2 → Instance types → Filter by g5.xlarge** for availability.

2. **Launch an EC2 instance:**
   - AMI: **Deep Learning AMI (Ubuntu 22.04) — NVIDIA driver pre-installed**. Saves you a 45-minute driver install.
   - Instance type: `g5.xlarge`
   - Storage: 100 GB gp3 (the Docker image + base + Chromium adds up)
   - Security group: **inbound TCP 8000 from 0.0.0.0/0** (or restrict to your IP if you want auth-via-obscurity). **Inbound TCP 22** for SSH.
   - Key pair: pick one you have a `.pem` for.

3. **SSH in and install Docker + NVIDIA container toolkit:**
   ```bash
   ssh -i ~/.ssh/yourkey.pem ubuntu@<EC2-PUBLIC-DNS>

   # Docker is usually pre-installed on the Deep Learning AMI. If not:
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

4. **Get the code + artifact onto the box:**
   ```bash
   git clone https://github.com/Charlemagne-Labs/g4h.git
   cd g4h
   # Copy your trained tarball from your laptop to the EC2 instance:
   #   scp -i ~/.ssh/yourkey.pem ~/Downloads/gemma4-e4b-cls.tar.gz ubuntu@<EC2-PUBLIC-DNS>:~/g4h/runs/
   mkdir -p runs && cd runs && tar -xzf gemma4-e4b-cls.tar.gz && cd ..
   ```

5. **Build and run:**
   ```bash
   docker build -t g4h-demo -f server/Dockerfile .
   docker run -d --name g4h --restart unless-stopped \
       --gpus all -p 8000:8000 \
       -v "$(pwd)/runs:/app/runs:ro" \
       g4h-demo
   docker logs -f g4h  # wait for "model ready" line
   ```

6. **Test:**
   ```bash
   curl http://<EC2-PUBLIC-DNS>:8000/health
   # {"status":"ok","model_loaded":true}
   ```

7. **Open the UI**: <http://EC2-PUBLIC-DNS:8000>. Done.

**Cost**: g5.xlarge is $1.006/hr. Run it during judging, stop it overnight (`aws ec2 stop-instances --instance-ids i-...`). Storage costs $0.10/GB-month. A weekend of demo time runs under $50.

### Path B — AWS App Runner (no GPU)

App Runner gives you a managed container endpoint with HTTPS for free. **No GPU** — CPU inference of Gemma 4 takes 20–30 s per request, which is borderline unusable for a live demo. Skip unless you really don't want to manage EC2.

### Path C — ECS Fargate with GPU

Newer Fargate-with-GPU support exists in some regions. Cheaper for spiky usage than a 24/7 EC2 but more setup (task definitions, ALB, IAM roles). Worth it for a production deployment, overkill for a hackathon.

---

## HTTPS for the demo

Free Tier-friendly options:

- **Cloudflare Tunnel** (recommended) — no inbound port forwarding, free TLS, ~5-min setup:
  ```bash
  curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i cloudflared.deb
  cloudflared tunnel --url http://localhost:8000  # gives you a https://<random>.trycloudflare.com URL
  ```
- **AWS Application Load Balancer + ACM cert** — proper production setup, ~15 min, $16/mo.
- **Just leave it on HTTP** for a hackathon demo. The URL classifier doesn't process secrets.

---

## Failure-mode reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Server starts but `/health` returns `model_loaded: false` | `G4H_ARTIFACT_DIR` doesn't exist or is empty | Mount the artifact volume; check `docker logs g4h` for the path it tried |
| `/predict` times out at exactly 10 s with `fetch:error:{"reason":"TimeoutError"}` | Target site is slow / hanging / blocked | Reduce expectations; the URL-only path still produced a verdict |
| Playwright errors with `Executable doesn't exist` | Chromium not installed in the container | The Dockerfile runs `playwright install` — rebuild without cache: `docker build --no-cache ...` |
| OOM during startup on g5.xlarge | Some other process holding GPU memory | `docker rm -f g4h && nvidia-smi` to confirm; then re-run |
| Target site returns a captcha or blocks Playwright | Anti-bot detection caught our fingerprint | Best-effort already; bumping the Chrome version string in `server/fetch.py:_USER_AGENT` sometimes helps. Some sites just refuse headless browsers full stop. |
| `verdict-tag` shows `allow` for a known-phishing URL | URL-only extractor saw nothing alarming (subtle attack relying on page content) | Try the FETCH DOM toggle — header + form-action signals usually flip it |

---

## Sign-off checklist

Phase 4 is done when all of these are checked:

- [ ] `uvicorn server.app:app` runs locally on Mac and `/predict` works end-to-end
- [ ] `docker build` + `docker run` produces a working container locally
- [ ] EC2 g5.xlarge is provisioned and the container is running
- [ ] Public URL is reachable (with or without HTTPS)
- [ ] At least 3 live URLs classified end-to-end with sensible verdicts:
  - one obvious-phishing (IP hostname, no HTTPS) → `block`
  - one trusted brand (github.com, google.com) → `allow`
  - one borderline / brand-impersonation lookalike → `warn` or `block`
- [ ] Costs reviewed and the instance is set to stop outside of demo hours
- [ ] Public URL is added to the hackathon submission

---

## What's intentionally NOT in this phase

- **Per-request authentication / rate limiting.** This is a hackathon demo, not a production service. If you leave the URL public after judging, add an API key.
- **Caching.** Identical URLs re-run the full pipeline. For low traffic during demos, fine.
- **Quantization for CPU-only inference.** The model could be ONNX-exported or merged + 8-bit quantized to run reasonably on CPU. Out of scope; revisit if cost becomes a concern.
- **Database / history.** Predictions are not persisted. If you want a "recent classifications" feature, add a small SQLite store under `server/` — but skip for the demo.
