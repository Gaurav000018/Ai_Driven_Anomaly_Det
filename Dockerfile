FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/

# Generate data and fit the model at build time so the image is self-contained
# and the API can answer immediately rather than 503-ing until someone trains it.
RUN python scripts/generate_data.py --lots 12 --parts 400 \
 && python scripts/train_fusion.py

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
