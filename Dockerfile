# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# If you have requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r /app/requirements.txt

# Copy your single file
COPY app.py /app/app.py

# Render uses $PORT; locally you can map 8000
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
