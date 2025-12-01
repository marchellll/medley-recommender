# Multi-stage build for medley recommender
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY . .

# Install dependencies
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder (includes .venv with all dependencies)
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Set environment variables
# uv creates .venv in the project directory
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose ports (defaults, can be overridden via environment variables)
EXPOSE ${API_PORT:-9876} ${STREAMLIT_PORT:-9877}

# Default command (can be overridden in docker-compose)
# Uses environment variables with defaults
CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-9876} & streamlit run ui/app.py --server.port ${STREAMLIT_PORT:-9877} --server.address ${API_HOST:-0.0.0.0} --server.headless=true"]


