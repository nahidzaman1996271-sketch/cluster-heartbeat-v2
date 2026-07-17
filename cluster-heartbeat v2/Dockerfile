# Cluster Heartbeat — all-in-one image (API + trained pipeline)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Data + model are produced inside the image for the demo; mount
# ./checkpoints and ./data/synthetic to reuse host-side artifacts.
RUN python scripts/generate_synthetic_data.py \
 && python scripts/train.py --set train.epochs=30

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
