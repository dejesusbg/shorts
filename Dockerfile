FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu-core \
    # 🆕 CRITICAL FIX: Install ImageMagick 🆕
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# 🆕 CRITICAL FIX: Tell MoviePy the path to the 'convert' binary 🆕
ENV IMAGEMAGICK_BINARY /usr/bin/convert

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY video_generator.py .

# Create output and temp directories
RUN mkdir -p /app/output /app/temp /app/input

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Volume mounts for input/output
VOLUME ["/app/input", "/app/output"]

# Default command
ENTRYPOINT ["python", "video_generator.py"]
CMD ["--help"]
