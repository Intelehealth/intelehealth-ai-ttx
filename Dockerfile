# Multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./
#COPY uv.lock* ./

# Sync dependencies (creates .venv by default)
RUN uv sync  --no-dev

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

RUN mkdir -p ops

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port for FastAPI
EXPOSE 5051


# Optional: For production with multiple workers
 CMD ["python", "-m", "uvicorn", "ttx_server:app", "--host", "0.0.0.0", "--port", "5051", "--workers", "4"]
