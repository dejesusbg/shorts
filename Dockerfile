FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu-core \
    imagemagick \
    ghostscript \  
    gcc \ 
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick security policy during build to prevent @* security errors in MoviePy
RUN policy_file=$(find /etc/ImageMagick* -name policy.xml | head -n 1) && \
    if [ -n "$policy_file" ]; then \
        sed -i 's/<policy domain="path" rights="none" pattern="@\*"/<policy domain="path" rights="read|write" pattern="@\*"/g' $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="PDF" />' >> $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="EPT" />' >> $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="URL" />' >> $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="HTTPS" />' >> $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="MVG" />' >> $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="MSL" />' >> $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="TEXT" />' >> $policy_file && \
        echo '<policy domain="coder" rights="read|write" pattern="LABEL" />' >> $policy_file && \
        echo '<policy domain="path" rights="read|write" pattern="@*" />' >> $policy_file; \
    else \
        echo "Warning: ImageMagick policy.xml not found"; \
    fi && \
    chmod 777 /tmp  # Ensure temp dir writable for MoviePy

ENV IMAGEMAGICK_BINARY=/usr/bin/convert

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

# Volume mounts for input/output (declared for docs; actual mount at runtime)
VOLUME ["/app/input", "/app/output"]

# Default command
ENTRYPOINT ["python", "video_generator.py"]
CMD ["--help"]