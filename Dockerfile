# Use a slim Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for xgboost, catboost, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your Kedro project into the container
# This copies the whole assignment-1 folder to /app/assignment-1
COPY ./assignment-1 ./assignment-1

# Set working directory to the Kedro project root
WORKDIR /app/assignment-1

# Ensure data folders exist (in case they are not in git)
RUN mkdir -p data/01_raw \
    data/02_intermediate \
    data/03_primary \
    data/05_model_input \
    data/06_models \
    data/07_model_output \
    data/08_reporting \
    data/09_bmarket_db

# Make sure Python can see your src/ package
ENV PYTHONPATH=/app/assignment-1/src:$PYTHONPATH

# (Optional) Expose port for Kedro-Viz if you ever use it
EXPOSE 4141

# Default command: run the Kedro pipeline
CMD ["kedro", "run"]