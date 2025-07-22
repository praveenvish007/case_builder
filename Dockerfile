FROM python:3.11.9-slim

# Install poppler and tesseract-ocr
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "case_builder3:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120"]