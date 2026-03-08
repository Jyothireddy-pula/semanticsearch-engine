FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV ARTIFACTS_DIR=/app/artifacts
ENV AUTO_BOOTSTRAP=false
ENV DATASET_ROOT=/app/data/20_newsgroups

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
