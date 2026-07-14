# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project folders into container
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY results/ results/
COPY plots/ plots/
COPY docs/ docs/
COPY README.md .

# Default command left unset for flexibility