FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "http_server:app", "--host", "127.0.0.1", "--port", "80"]