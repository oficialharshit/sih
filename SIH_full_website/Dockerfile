# Use lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system audio dependencies (ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all application code and models
COPY backend /app/backend
COPY main.py /app/main.py

# Expose server port
EXPOSE 8000

# Start FastAPI Uvicorn server
CMD ["uvicorn", "main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
