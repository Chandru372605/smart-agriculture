# AgroSense — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t agrosense .
# Run:    docker run -p 5000:5000 agrosense
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies (Pillow needs libjpeg / zlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend/    ./backend/
COPY frontend/   ./frontend/
COPY ml_models/  ./ml_models/
COPY run.py      .

# Pre-create SQLite DB directory
RUN mkdir -p /app/backend

# Expose Flask port
EXPOSE 5000

# Use gunicorn for production; fall back to flask dev server for simplicity
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "5000"]
