# Use slim Python image (CPU only, lightweight)
FROM python:3.9-slim

# Set environment variables to reduce Python buffering issues
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache

# Create working directory
WORKDIR /app

# Copy code into the image
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -e .[training]

# Default run command
CMD ["python", "pathoMozhi/train/train.py"]