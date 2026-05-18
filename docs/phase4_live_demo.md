# Phase 4 — Live demo runbook

The live demo is a FastAPI server that:
1. Takes a URL from the browser.
2. Runs URL-string-only feature extraction (`src/extract.py`).
3. Optionally runs a targeted DOM fetch via Playwright (`server/fetch.py`) to enrich the indicator list.
4. Feeds the combined indicator string to the trained Gemma 4 classifier.
5. Returns a verdict (`allow` / `warn` / `block`) + per-label scores + per-phase timing + the indicators that were used.

Tracks issue: [`gateguard-suite#73`](https://github.com/Charlemagne-Labs/gateguard-suite/issues/73), Phase 4.

---

## Goal

A public URL where a hackathon judge can paste any URL and get a live classification within a few seconds. Demonstrates the trained model on inputs that weren't in the training set, with the extractor pipeline and per-phase latency visible in the response.

---

## Layout

```
server/
├── __init__.py
├── app.py              FastAPI server with /predict endpoint
├── fetch.py            Playwright-based targeted DOM fetch
├── requirements.txt    FastAPI + uvicorn + playwright + model deps
├── Dockerfile          GPU-aware (nvidia/cuda base) container
└── static/
    ├── index.html      Single-page UI — "Charley · Gemma 4 E4B demo"
    ├── style.css       Design system tokens + components
    └── main.js         Form submit + result rendering + timing strip
```

The model + extractor are loaded once at server startup via FastAPI's lifespan.

---

## Local dev (Mac)

Pre-req: Phase 3 complete, artifact at `runs/gemma4-e4b-cls/`.

```bash
cd ~/CharlemagneLabs/g4h
source .venv/bin/activate

# Server-side deps + Playwright Chromium
pip install -r server/requirements.txt
python -m playwright install chromium

# Run the server. Use the explicit venv binary if `uvicorn` from PATH ends up
# being a different Python (a common foot-gun).
./.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. Paste a URL. Try with and without the **FETCH DOM** toggle.

Expected timing on M4 Max:
- URL-only path: <2 s end-to-end (model inference is ~1-1.5 s of that)
- With DOM fetch: 3-12 s depending on the target page

If the model fails to load, `/health` returns `model_loaded: false` and `/predict` returns 503. Check `G4H_ARTIFACT_DIR` (defaults to `runs/gemma4-e4b-cls`).

### Local Docker (optional, mostly skip)

The Dockerfile uses the `nvidia/cuda:12.1.1-runtime-ubuntu22.04` base, which is x86_64 only. On Apple Silicon, Docker Desktop runs the build under QEMU emulation: **expect 20-40 min for the build vs 8-12 min on EC2**, and CPU-only inference at ~20-30 s per request. The bare `uvicorn` path above is faster on Mac.

If you really want to verify the image locally before pushing to EC2:

```bash
docker build -t g4h-demo -f server/Dockerfile .   # 20-40 min on M4 (QEMU)
docker run --rm -p 8000:8000 -v "$(pwd)/runs:/app/runs:ro" g4h-demo
```

But the recommended path is: skip local Docker, build directly on EC2 (faster, no emulation).

---

## Recommended path — Docker on AWS EC2 g5.xlarge

**~30 min total. ~$1/hr while running. Auto-restarts on crash. Survives EC2 reboot.**

### 0. Pre-flight

New AWS accounts have a default G-instance vCPU quota of 0. Check **Service Quotas → EC2 → Running On-Demand G and VT instances**. Request 4 vCPUs (g5.xlarge needs that many). Trial-account approvals are usually instant.

### 1. Launch the instance

AWS Console → EC2 → **Launch instance**:

| Field | Value |
|---|---|
| Name | `g4h-demo` |
| AMI | **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04)** (e.g. `ami-082ecb0714b440c33` in us-east-1, x86). Has CUDA + NVIDIA drivers + Docker + NVIDIA Container Toolkit pre-installed. **Skip** the "Base", "ARM64", and "Neuron" variants — those don't match what the Dockerfile expects. |
| Instance type | **g5.xlarge** ($1.006/hr — A10G, 24 GB VRAM, 16 GB RAM) |
| Key pair | New (`g4h-demo-key.pem`) or existing |
| Security group | Create new (`g4h-demo-sg`): SSH (22) from **My IP**, Custom TCP (8000) from **0.0.0.0/0** (or My IP) |
| Storage | 100 GB gp3, delete on termination ✓ |

Wait ~2 min for "2/2 checks passed". Copy the **Public IPv4 DNS** (`ec2-XX-XX-XX-XX.compute-1.amazonaws.com`). Call it `$EC2` from here on.

### 2. SSH + sanity check

On your laptop:

```bash
chmod 400 ~/Downloads/g4h-demo-key.pem
EC2=ec2-XX-XX-XX-XX.compute-1.amazonaws.com
ssh -i ~/Downloads/g4h-demo-key.pem ubuntu@$EC2
```

Inside EC2:

```bash
docker --version       # 24.x+ pre-installed
nvidia-smi             # should show NVIDIA A10G, 23028MiB
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
# ↑ confirms Docker can see the GPU. Pulls a tiny image (~80 MB), takes ~20s.
```

### 3. Upload the model

In a **second laptop terminal** (keep the SSH session open):

```bash
cd ~/CharlemagneLabs/g4h/runs
[ -f gemma4-e4b-cls.tar.gz ] || tar -czf gemma4-e4b-cls.tar.gz gemma4-e4b-cls/
scp -i ~/Downloads/g4h-demo-key.pem \
    gemma4-e4b-cls.tar.gz \
    ubuntu@$EC2:~/
```

Back inside EC2:

```bash
cd ~
git clone https://github.com/Charlemagne-Labs/g4h.git
cd g4h
mkdir -p runs && cd runs
tar -xzf ~/gemma4-e4b-cls.tar.gz
cd ..
ls runs/gemma4-e4b-cls/   # base_lm/, classifier_head.pt, etc.
```

### 4. Build the image

Inside EC2:

```bash
docker build -t g4h-demo -f server/Dockerfile .
```

8-12 min. The slow phases: pip-installing torch/transformers/bnb (~5 min), Playwright + Chromium install (~3-5 min). Look for `Successfully tagged g4h-demo:latest`.

### 5. Run

Inside EC2:

```bash
docker run -d --name g4h \
    --restart unless-stopped \
    --gpus all \
    -p 8000:8000 \
    -v "$(pwd)/runs:/app/runs:ro" \
    g4h-demo
```

What each flag does:
- `-d` — detached (background)
- `--name g4h` — stable container name for `docker logs g4h` etc.
- `--restart unless-stopped` — **auto-restart on process crash, on Docker restart, on EC2 reboot.** Won't restart only if you explicitly `docker stop g4h`.
- `--gpus all` — pass the A10G into the container
- `-p 8000:8000` — expose the port
- `-v "$(pwd)/runs:/app/runs:ro"` — mount the model directory read-only at the path the server expects

Tail logs until ready:

```bash
docker logs -f g4h
# wait for:
#   g4h.server | loading model from /app/runs/gemma4-e4b-cls
#   Loading weights: 100%|...
#   g4h.server | model ready (labels={0: 'allow', 1: 'warn', 2: 'block'}, max_length=256)
#   INFO:     Application startup complete.
```

`Ctrl-C` to detach (just stops log tail, container keeps running).

### 6. Test

Inside EC2:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

From your laptop:

```bash
curl http://$EC2:8000/health
open http://$EC2:8000    # macOS — or just paste the URL in a browser
```

Try a few URLs. Watch the timing strip in the result panel — model inference should be ~80-150 ms on A10G.

### 7. (Optional) Free HTTPS via Cloudflare Tunnel

For sharing with judges you usually want a `https://...trycloudflare.com` link instead of raw HTTP-on-IP.

Inside EC2:

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
sudo apt install -y tmux

tmux new -s tunnel
cloudflared tunnel --url http://localhost:8000
# Prints: https://something-random-words.trycloudflare.com — your public URL.
# Ctrl-B then D to detach the tmux session (tunnel keeps running).
# To reattach: tmux attach -t tunnel
```

The tunnel runs until the EC2 instance stops or you Ctrl-C it. After an EC2 stop/start, re-run the tmux + cloudflared commands.

---

## Day-2 operations

### Watch what's happening

```bash
docker logs -f g4h               # live log stream
docker logs --tail 100 g4h       # last 100 lines
docker stats g4h                 # live CPU / memory / GPU / network
docker ps                        # is it running?
docker ps -a                     # 'Up X' = running, 'Exited' = died
```

### Restart the container (without rebuilding)

```bash
docker restart g4h               # ~10s downtime
```

### Pull updated code, rebuild, swap

```bash
cd ~/g4h
git pull
docker build -t g4h-demo -f server/Dockerfile .
docker stop g4h && docker rm g4h
docker run -d --name g4h --restart unless-stopped --gpus all \
    -p 8000:8000 -v "$(pwd)/runs:/app/runs:ro" g4h-demo
docker logs -f g4h    # wait for "model ready"
```

### Swap in a re-trained model

```bash
# On your laptop:
scp -i ~/Downloads/g4h-demo-key.pem \
    new-model.tar.gz ubuntu@$EC2:~/

# On EC2:
cd ~/g4h/runs
rm -rf gemma4-e4b-cls
tar -xzf ~/new-model.tar.gz
docker restart g4h    # picks up new model on lifespan reload
```

### Stop overnight to save money

```
AWS console → EC2 → select instance → Instance state → Stop instance
(don't pick Terminate — that destroys the instance and storage)
```

Stopped instance bills only storage (~$10/month for 100 GB EBS gp3). When you Start it back up:
- **Public DNS changes** — note the new one
- `docker start g4h` (container is preserved across stop/start)
- Re-run the Cloudflare tunnel command (the URL also changes)

---

## What auto-restart will and won't handle

`--restart unless-stopped` covers:

- Server crashes (Python exception, OOM kill) → restart immediately
- Docker daemon restart → container starts when daemon does
- EC2 instance reboot → docker starts on boot, container starts with it

Caveats it does NOT cover:

- **Broken startup state** (model dir gone, GPU driver dead): container will crash → restart → crash → restart, forever. Check `docker ps -a` for a climbing restart count, and `docker logs --tail 200 g4h` for the actual error.
- **Server hang** (uvicorn deadlock, but process still running): Docker can't detect this. Add a `HEALTHCHECK` directive to the Dockerfile if you want this caught. (Skipped for hackathon scope.)
- **EC2 instance terminated**: terminal — nothing survives. Don't click Terminate when you mean Stop.

---

## Failure-mode quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `Permission denied (publickey)` on SSH | Wrong .pem perms | `chmod 400 ~/Downloads/g4h-demo-key.pem` |
| `docker: Cannot connect to the Docker daemon` after fresh install | User not in docker group | `sudo usermod -aG docker ubuntu` then log out + back in |
| Connection refused on port 8000 from laptop | Security group missing port 8000 rule | Fix in AWS console |
| Build OOM during pip install | Accidentally picked smaller instance (t2/t3) | Confirm `g5.xlarge`; it has 16 GB RAM |
| `nvidia-smi` works but `docker run --gpus all` fails | NVIDIA Container Toolkit missing | `sudo apt install -y nvidia-container-toolkit && sudo systemctl restart docker` |
| Container exits immediately, `docker logs` shows artifact path error | Volume not mounted right | Check `-v "$(pwd)/runs:/app/runs:ro"` — the `runs/gemma4-e4b-cls/` dir must exist on the host at the path you pass |
| `/predict` errors with `model not loaded` | Model failed to load at startup | `docker logs g4h` to see the actual exception; usually wrong path or missing files |
| Playwright fetch times out at 10s on every URL | Target site blocking the bot fingerprint | Best-effort already; some sites refuse headless browsers full stop |
| Cloudflare tunnel disconnected | Tunnel process killed (SSH session ended without tmux) | Re-run inside `tmux new -s tunnel` so it survives logouts |

---

## Cost reference

| Item | Cost | Notes |
|---|---|---|
| g5.xlarge running | $1.006/hr | Stop when not in use |
| g5.xlarge stopped, 100 GB EBS gp3 attached | ~$8/month | Storage only |
| Data transfer out | $0.09/GB | First 100 GB/month free |
| **Estimate for a weekend of judging** | **~$25-50** | Conservative — running ~24 hours of compute, idle the rest |

If you're on the AWS Free Trial $300 credit, you've got plenty of headroom.

---

## Sign-off checklist

Phase 4 is done when:

- [ ] `docker build` produces a working image on EC2
- [ ] `docker run` with `--restart unless-stopped --gpus all` starts the container; `docker logs g4h` shows "model ready"
- [ ] `curl http://$EC2:8000/health` returns `model_loaded: true`
- [ ] Browser at `http://$EC2:8000` loads the Charley demo UI
- [ ] At least 3 live URLs classified end-to-end with sensible verdicts:
  - one obvious phishing (IP hostname, no HTTPS) → block
  - one trusted brand (github.com, google.com) → allow
  - one borderline / brand-impersonation lookalike → warn or block
- [ ] (Optional) Cloudflare Tunnel running, public HTTPS URL works
- [ ] Stop schedule planned (or accept the ~$24/day burn)
- [ ] Demo URL added to the hackathon submission

---

## Alternative deployment paths (not recommended for this hackathon)

These exist for completeness but require more setup than they're worth for a 2-3 day demo:

- **AWS App Runner** — managed container with free HTTPS, but **no GPU**. CPU inference of Gemma 4 takes 20-30 s/request, borderline unusable for live demos.
- **ECS Fargate with GPU** — pay-per-use, cheaper for sporadic traffic, but task-definition + ALB + IAM setup eats 1-2 hours. Worth it for production, overkill here.
- **Bare uvicorn on EC2 under systemd** — works fine, but you trade Docker's auto-restart and reproducibility for one less abstraction. Use Docker.

---

## What's intentionally NOT in this phase

- **Per-request authentication / rate limiting.** Hackathon demo, not a production service. If you leave the URL public after judging, add an API key (FastAPI's `APIKeyHeader` is ~5 lines).
- **Caching.** Identical URLs re-run the full pipeline. Fine for low-volume demos.
- **Quantization for CPU-only inference.** The model could be merged + ONNX-quantized to run reasonably on CPU. Out of scope; revisit if cost is an issue.
- **Prediction history / database.** Not persisted. Add a small SQLite store under `server/` if you want "recent classifications" — but skip for the demo.
- **HEALTHCHECK in Dockerfile.** Docker's `--restart` catches crashes but not hangs. Add `HEALTHCHECK CMD curl -f http://localhost:8000/health` if you want hang-detection.
