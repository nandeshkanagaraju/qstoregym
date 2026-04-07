FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose the API port
EXPOSE 7860

# Health check — the /health endpoint responds within 5s if the API is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default: run the REST API.
# Override at runtime:
#   docker run q-store-gym python train.py --curriculum
#   docker run q-store-gym python inference.py --benchmark
#   docker run q-store-gym python retrain.py --dry-run
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
