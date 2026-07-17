FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instructor-provided resources
COPY src/ src/
COPY config/ config/
COPY data/ data/

# Public-facing metadata
COPY README.md .