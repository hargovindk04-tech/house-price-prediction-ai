FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Optional Kaggle credentials for downloading competition data at build time.
# Accept competition rules at https://www.kaggle.com/c/house-prices-advanced-regression-techniques
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME} \
    KAGGLE_KEY=${KAGGLE_KEY}

RUN python -m ml.train

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
