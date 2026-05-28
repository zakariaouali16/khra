# Deploying the Phase 5 Router to GCP

Your router calls Ollama and makes **two LLM calls per request** (gate + route).
So "host on GCP with llama3" really means: run **Ollama + llama3 on a GPU**,
wrap the router in the included `app.py`, and serve `index.html` to query it.

Put these files in one folder before you build:

```
app.py  index.html  start.sh  Dockerfile  requirements.txt
llm_router_phase5_1.py  taxonomy_phase5.json
```

Run it locally first to confirm everything works:

```bash
ollama serve &            # in one terminal
ollama pull llama3
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
# open http://localhost:8080
```

---

## Pick a hosting path

llama3 (8B) needs a GPU to feel interactive. The NVIDIA **L4** (24 GB) is the
right-sized, cheapest data-center GPU for an 8B model and is what GCP offers on
both Cloud Run and G2 VMs. The decision is really about traffic pattern:

| Path | Best for | Roughly | Cold start |
|---|---|---|---|
| **A. Cloud Run + L4 GPU** | demos, bursty/low traffic | ~$0.67/hr **only while serving**, scales to zero | yes (model load, ~20–60s) |
| **B. GCE G2 VM + L4** | always-on, no cold starts | ~$0.70–1/hr 24/7 (~$500–700/mo) | none |
| **C. CPU + hosted Llama API** | cheapest, no GPU to manage | CPU Cloud Run + per-token API | none |

**Recommendation:** start with **A (Cloud Run + GPU)**. Scale-to-zero means a
demo costs almost nothing when idle, and it's the closest thing to your local
setup. Move to **B** only if cold starts annoy you or traffic is steady.

---

## Path A — Cloud Run with an L4 GPU (recommended)

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# One-time: a registry to hold the image.
gcloud artifacts repositories create router-repo \
  --repository-format=docker --location=us-central1

# Build the image (bakes llama3 in → large, slow build → raise the build timeout).
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/router-repo/router:latest \
  --timeout=1800s

# Deploy with a GPU. L4 is available in us-central1, us-east4,
# europe-west1, europe-west4, asia-south1, asia-southeast1.
gcloud run deploy router \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/router-repo/router:latest \
  --region us-central1 \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 4 --memory 16Gi \
  --no-cpu-throttling \
  --concurrency 4 \
  --max-instances 1 \
  --timeout 600 \
  --port 8080 \
  --allow-unauthenticated
```

`--no-gpu-zonal-redundancy` is the cheaper option and is what the first-time
auto quota grant covers (your project gets 3 L4 GPUs per region on first deploy).
`--no-cpu-throttling` keeps the model resident between requests. `--concurrency 4`
stops a single GPU from being swamped by parallel generations.

The command prints a `*.run.app` URL — that's your live site. The page calls
`/route` on the same origin, so no extra config needed.

Notes:
- The image is large (~6 GB) because the model is baked in. That's deliberate —
  it makes cold starts load weights from local disk instead of re-downloading.
- `--timeout 120` gives the first request room for a cold start + two LLM calls.
- GPU billing is per-second while an instance is alive; it stops when it scales
  to zero. There is **no GPU free tier**.

---

## Path B — GCE G2 VM (always-on)

```bash
gcloud compute instances create router-vm \
  --zone us-central1-a \
  --machine-type g2-standard-4 \
  --accelerator type=nvidia-l4,count=1 \
  --maintenance-policy TERMINATE \
  --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
  --boot-disk-size 60GB \
  --tags router

# Open the API port
gcloud compute firewall-rules create allow-router \
  --allow tcp:8080 --target-tags router --source-ranges 0.0.0.0/0
```

Then SSH in and set it up:

```bash
# Install NVIDIA driver (Ubuntu): follow GCP's GPU driver install, then:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3

# Copy your files up (gcloud compute scp ...), then:
pip install -r requirements.txt
nohup uvicorn app:app --host 0.0.0.0 --port 8080 &
```

Visit `http://EXTERNAL_IP:8080`. For production, put it behind a domain + HTTPS
(an HTTPS load balancer, or Caddy/Nginx on the box) instead of raw `:8080`.

---

## Path C — No GPU: CPU app + a hosted Llama endpoint

If you'd rather not run a GPU at all, host `app.py` on plain CPU Cloud Run and
point the router at a hosted Llama 3 endpoint (Vertex AI Model Garden, or a
provider like Groq/Together/Fireworks). The one change required: those use the
OpenAI-style `/chat/completions` schema, not Ollama's `/api/generate`, so you'd
adapt the two `requests.post(...)` calls in `llm_router_phase5_1.py`
(`check_gate` and `route_request`). Cheapest to operate at low volume and no
cold-start model load — at the cost of editing the router and depending on an
external API.

---

## Things worth doing before this is "production"

- **Lock down access.** `--allow-unauthenticated` + `allow_origins=["*"]` is fine
  for a demo. For real use, restrict CORS in `app.py` to your domain and add auth
  or a rate limit — the endpoint runs an LLM, so it's abusable.
- **Add a request timeout.** The router's `requests.post(...)` has no timeout; a
  stuck Ollama call will hang the worker. Add `timeout=30` to both calls.
- **Latency expectation.** Two sequential llama3 calls on an L4 land around
  2–5s per request. The page shows a spinner; that's why.
- **Pin the model.** `llama3` in Ollama tracks 8B; pin a digest/tag if you need
  reproducibility across rebuilds.
