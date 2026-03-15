# HuggingFace Spaces deployment
# Create a new Space at huggingface.co/spaces, select "Docker"
# Hardware: CPU Basic (free) for dashboard, T4 GPU for live inference
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Python deps (full stack including inference)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch transformers datasets accelerate peft anthropic openai

# App code
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]
