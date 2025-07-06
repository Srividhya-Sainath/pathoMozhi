# Use the official PyTorch runtime image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Create a non-root user with a home directory and shell
RUN groupadd -r appgroup && useradd -m -r -g appgroup -s /bin/bash appuser

# Create standard directories and fix permissions
RUN mkdir -p /app /input /output /opt && \
    chown -R appuser:appgroup /app /input /output /opt

# Set working directory
WORKDIR /app

# Copy source code and install dependencies
COPY . /app
RUN chown -R appuser:appgroup /app

# Install system packages
RUN apt-get update && apt-get install -y \
    git wget curl libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .[training]

# Set HuggingFace cache dir (optional)
ENV TRANSFORMERS_CACHE=/app/cache

# Switch to non-root user
USER appuser

# Default command (adjust if needed)
CMD ["python", "pathoMozhi/train/train.py", "pathoMozhi/eval/evalOutput.py"]