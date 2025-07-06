# Base image with CUDA and PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Create a non-root user and group
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy project files and give ownership to non-root user
COPY . /app
RUN chown -R appuser:appgroup /app

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .[training]

# (Optional) For HuggingFace transformers cache
ENV TRANSFORMERS_CACHE=/app/cache

# Switch to non-root user
USER appuser

# Set default command (override during inference)
CMD ["python", "pathoMozhi/train/train.py"]