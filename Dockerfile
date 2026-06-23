FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Install specific version of PyTorch for CPU to keep image size small
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn pydantic

# Copy source code
COPY . .

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
