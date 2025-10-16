# Base image dengan Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (untuk caching layer)
COPY requirements.txt requirements.txt

# Upgrade pip dan install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Buat direktori yang dibutuhkan
RUN mkdir -p dataset/preprocess models

# Copy dataset dan model files
COPY dataset/preprocess/product_info.csv dataset/preprocess/
COPY models/product_combined_vectors_finetuned.npy models/
COPY models/tfidf_vectorizer.pkl models/
COPY models/fasttext_haircare_gensim.vec models/

# Copy aplikasi utama
COPY app.py .

# Expose port Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Streamlit config via environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]