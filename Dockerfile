# Use a slim Python 3.11 base image for speed and minimal image size
FROM python:3.11-slim

# Set environment variables for Python and port
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies (build-essential, libgomp1 for FAISS, and sqlite3 CLI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 to match Hugging Face Spaces requirements
RUN useradd -m -u 1000 user

# Set working directory to the app directory inside user's home
WORKDIR $HOME/app

# Copy requirements file first to optimize docker build caching
COPY --chown=user:user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container (respecting .dockerignore)
COPY --chown=user:user . .

# Set read/write permissions on the SQLite database file
RUN chmod 666 clouddash.db || true

# Switch to the non-root user
USER user

# Expose the default port for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI server using uvicorn on port 7860
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
