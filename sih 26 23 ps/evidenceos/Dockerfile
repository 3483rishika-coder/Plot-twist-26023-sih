FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    EVIDENCEOS_DATA_DIR=/data HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install "psycopg[binary]" pgvector

COPY evidenceos ./evidenceos
COPY web ./web
COPY scripts ./scripts
COPY data/manifest.json ./data/manifest.json

# pre-download OCR + embedding models into the image so first requests are fast and the container works offline
RUN python -c "try:\n    from rapidocr import RapidOCR\nexcept ImportError:\n    from rapidocr_onnxruntime import RapidOCR\nRapidOCR()" && \
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/api/v1/roles || exit 1
CMD ["uvicorn", "evidenceos.api:app", "--host", "0.0.0.0", "--port", "8000"]
