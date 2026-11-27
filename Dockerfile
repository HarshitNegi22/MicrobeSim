FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/microbesim \
    MICROBESIM_STATE_DIR=/var/lib/microbesim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin microbesim \
    && mkdir -p /var/lib/microbesim \
    && chown microbesim:microbesim /var/lib/microbesim

COPY requirements.txt .
COPY requirements-tested.txt .
RUN pip install --no-cache-dir -r requirements.txt -c requirements-tested.txt

COPY . .

# Train every model the app can show now, inside this exact environment, so no visitor ever waits for
# training. The store is tied to the pinned library versions above; rebuild the image to refresh it.
RUN python scripts/pretrain_models.py --jobs 2 \
    && chmod -R a+rX data/pretrained_models \
    && python scripts/pretrain_models.py --check

VOLUME ["/var/lib/microbesim"]
USER microbesim

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
