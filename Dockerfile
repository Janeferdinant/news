# Local run only: builds the React front end, then serves it and the API from one container.
# The models are NOT baked in; docker-compose.yml mounts them read-only.
FROM node:22-slim AS web
WORKDIR /web
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /srv
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
COPY fakenews/ fakenews/
COPY app/server.py app/services.py app/
COPY --from=web /web/dist app/frontend/dist
ENV NEWS_CLASSICAL_DIR=/artifacts/classical NEWS_DISTILBERT_DIR=/artifacts/distilbert
EXPOSE 5000
CMD ["gunicorn", "--chdir", "app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "server:create_app()"]
