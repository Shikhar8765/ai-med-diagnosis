FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r 0_setup/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
