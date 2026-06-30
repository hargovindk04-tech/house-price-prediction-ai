# House Price Prediction AI

Production-ready Machine Learning web application using FastAPI, Scikit-learn, Joblib, Docker, and cloud deployment. Includes REST APIs, Swagger docs, health monitoring, and responsive UI.

## Training data

The model trains from a CSV in `ml/data/`. Supported formats:

### House Price Prediction Dataset (recommended if you already have it)

Place your file at `ml/data/house_prices.csv`.

| API feature      | CSV column   | Notes                                      |
|------------------|--------------|--------------------------------------------|
| `area_sqft`      | `Area`       |                                            |
| `bedrooms`       | `Bedrooms`   |                                            |
| `bathrooms`      | `Bathrooms`  |                                            |
| `age`            | `YearBuilt`  | Computed as `max(YearBuilt) - YearBuilt`   |
| `location_score` | `Location`   | Rural=4, Suburban=6, Urban=7.5, Downtown=9 |
| target           | `Price`      |                                            |

When using the web UI or API, pick a `location_score` that matches the area type above.

### Kaggle Ames House Prices (alternative)

Place `train.csv` at `ml/data/train.csv`, or download via Kaggle:

| API feature      | Kaggle column(s)                     |
|------------------|--------------------------------------|
| `area_sqft`      | `GrLivArea`                          |
| `bedrooms`       | `BedroomAbvGr`                       |
| `bathrooms`      | Full/half baths (including basement) |
| `age`            | `YrSold - YearBuilt`                 |
| `location_score` | `OverallQual` (1–10)                 |
| target           | `SalePrice`                          |

Set a custom file path with `TRAINING_DATA_PATH`.

### Download Kaggle data (optional)

1. Join the [House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) and accept the rules.
2. Authenticate with Kaggle (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`).
3. Run:

```bash
pip install -r requirements.txt
python -m ml.download_data
```

### Option 2: Manual upload

Place your CSV at `ml/data/house_prices.csv` (House Price Prediction Dataset) or `ml/data/train.csv` (Kaggle Ames).

### Train the model

```bash
python -m ml.train
```

The saved model is written to `ml/artifacts/house_price_model.joblib`.

## Docker

Build with local data already in `ml/data/train.csv`, or pass Kaggle credentials so the image downloads data during build:

```bash
docker build \
  --build-arg KAGGLE_USERNAME=your_username \
  --build-arg KAGGLE_KEY=your_api_key \
  -t house-price-ai .
```

Run:

```bash
docker run -p 8000:8000 house-price-ai
```
