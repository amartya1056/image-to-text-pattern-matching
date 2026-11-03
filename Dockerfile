# Container for the Streamlit web UI (used for the Hugging Face Space).
# Includes Tesseract and the optional EasyOCR engine (CPU-only PyTorch).
#
#   docker build -t ocrmatch .
#   docker run -p 7860:7860 ocrmatch      ->  http://localhost:7860

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements-easyocr.txt ./
# CPU build of PyTorch first (the default wheel bundles CUDA and is several GB)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-easyocr.txt

# Download the EasyOCR models at build time so the first request is not slow
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
