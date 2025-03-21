# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/.files && \
    chmod -R 777 /app/.files

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860
ENV PYTHONPATH=/app

# Create a non-root user
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]