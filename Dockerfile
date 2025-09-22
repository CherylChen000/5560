FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Match the notebook exactly: use en_core_web_lg
RUN python -m spacy download en_core_web_lg

COPY app /app/app

EXPOSE 8000

ENV SPACY_MODEL=en_core_web_lg

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
