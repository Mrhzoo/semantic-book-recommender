FROM python:3.11-slim

WORKDIR /app

# Copy files
COPY . .

# Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Create data folder
RUN mkdir -p /app/chroma_data

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "gradio-dashboard.py"]