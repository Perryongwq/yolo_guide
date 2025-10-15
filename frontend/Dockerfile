# Frontend Dockerfile for CT600 Vision Inspection System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (create a requirements file for frontend)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5001

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]

