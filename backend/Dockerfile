# Backend Dockerfile for CT600 Vision Inspection System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_fastapi.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_fastapi.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads processed results static templates models calibration

# Expose port
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app_fastapi.py"]

