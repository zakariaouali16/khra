# Single container: Ollama runtime + gemma4 (baked in) + the FastAPI router.
# Baking the model into the image means Cloud Run cold starts load weights from
# the local disk instead of re-downloading ~4.7 GB every time an instance spins up.
FROM ollama/ollama:0.20.0

# The ollama image is Ubuntu-based; add Python + curl.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip curl procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Bake the model into the image at build time.
# (Start the server briefly, pull, then stop it within the same layer.)
RUN ollama serve & \
    until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done && \
    ollama pull gemma4 && \
    pkill -f "ollama serve"

# App files. These must sit in the build context alongside this Dockerfile:
#   app.py  index.html  start.sh  llm_router_phase5_1.py  taxonomy_phase5.json
COPY llm_router_phase5_1.py taxonomy_phase5.json app.py index.html start.sh ./
RUN chmod +x start.sh

ENV OLLAMA_URL=http://localhost:11434/api/generate \
    MODEL_NAME=gemma4 \
    TAXONOMY_PATH=/app/taxonomy_phase5.json \
    PORT=8080 \
    OLLAMA_FLASH_ATTENTION=0

# The base image's entrypoint is `ollama`; clear it so our script runs.
ENTRYPOINT []
CMD ["./start.sh"]
