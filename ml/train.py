"""Train and save the house price prediction model."""



from pathlib import Path



import joblib

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



from ml.data_loader import FEATURE_NAMES, load_training_data



ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "house_price_model.joblib"





def train_and_save() -> Path:

    X, y = load_training_data()

    print(f"Training on {len(X)} samples with features: {', '.join(FEATURE_NAMES)}")



    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=42

    )



    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)



    score = model.score(X_test, y_test)

    print(f"Model R² score on test set: {score:.4f}")



    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")



    return MODEL_PATH





if __name__ == "__main__":

    train_and_save()

