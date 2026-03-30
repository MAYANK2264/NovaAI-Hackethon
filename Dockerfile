# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# Using 3.11 to match environment dependencies
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first for layer caching
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files into the image
COPY --chown=user . /app

# Hugging face standard port
EXPOSE 7860

# Point to our server.py FastAPI app initialization and run Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
