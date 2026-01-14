# 1. BASE IMAGE
FROM python:3.9-slim

# 2. WORK DIRECTORY
WORKDIR /app

# 3. INSTALL DEPENDENCIES
# Upgrade pip first to avoid installation errors
RUN pip install --upgrade pip

COPY requirements.txt .

# Install dependencies using the CPU-only index for PyTorch
# Added --default-timeout=1000 to prevent timeouts on slow connections
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 4. COPY CODE
COPY 30_sentimentapi.py .

# 5. OPEN PORT (Hugging Face default)
EXPOSE 7860

# 6. START COMMAND
# We change the port to 7860
CMD ["uvicorn", "30_sentimentapi:app", "--host", "0.0.0.0", "--port", "7860"]