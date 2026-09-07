# Runs the research dashboard anywhere Docker runs.
#
# The image ships the rolling price cache that is committed to the repository,
# so the app serves every page without calling Yahoo on startup — verified by
# a cold start with the network blocked.

FROM python:3.11-slim

# Keep the image quiet and deterministic.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, configuration, and the committed price cache.
COPY src/ ./src/
COPY pages/ ./pages/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY app.py cli.py config.yaml conftest.py ./
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

# Streamlit's own health endpoint, so an orchestrator can tell live from ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/healthz')"

# address=0.0.0.0 is required for the port to be reachable from outside the
# container; Streamlit otherwise binds to localhost only.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
