FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies: OCR + PDF tools + curl for health check
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (your dependency manager)
RUN pip install --no-cache-dir uv

# Copy dependency metadata + README needed for building the package
COPY pyproject.toml uv.lock README.md ./

# Increase timeout for huge wheels (torch + CUDA etc.)
ENV UV_HTTP_TIMEOUT=600


# Install ONLY base dependencies (no dev, no semantic group)
RUN uv sync --frozen --no-dev --no-group semantic

# Make the .venv the default Python for the container
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Now copy the actual application code
COPY . .

# Streamlit settings
EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check: hit the root URL of the Streamlit app
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/ || exit 1

# Start the Streamlit app (note: app.py lives in src/ats)
CMD ["streamlit", "run", "src/ats/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
