# Base image with CPU-only PyTorch
FROM pytorch/pytorch:2.1.0-cpu

# Create a non-root user and group
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy project files and set ownership
COPY . /app
RUN chown -R appuser:appgroup /app

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .[training]

# Set environment variable for HuggingFace cache
ENV TRANSFORMERS_CACHE=/app/cache

# Set default command
CMD ["python", "pathoMozhi/train/train.py"]