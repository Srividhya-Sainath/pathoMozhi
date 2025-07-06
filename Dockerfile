# Base image with CUDA and PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .[training]

# (Optional) For HuggingFace transformers cache
ENV TRANSFORMERS_CACHE=/app/cache

# Set default command (override during inference)
CMD ["python", "pathoMozhi/train/train.py"]