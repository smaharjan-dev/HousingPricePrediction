
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Project directory
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


# --------------------------------------------------
# 2. Dataset configurations
# --------------------------------------------------

DATASETS = {
    "housing": {
        "file": "HousingPrices.csv",
        "target": "price",
        "features": [
            "area", "bedrooms", "bathrooms", "stories",
            "parking", "mainroad", "guestroom", "basement",
            "hotwaterheating", "airconditioning", "prefarea",
            "furnishingstatus"
        ]
    },

    "california": {
        "file": "CaliforniaHousingPrices.csv",
        "target": "median_house_value",
        "features": [
            "longitude", "latitude", "housing_median_age",
            "total_rooms", "total_bedrooms", "population",
            "households", "median_income", "ocean_proximity"
        ]
    },

    "usarealestate": {
        "file": "USARealEstate.csv",
        "target": "price",
        "features": [
            "bed", "bath", "acre_lot", "house_size",
            "status", "state"
        ]
    }
}


# --------------------------------------------------
# 3. Preprocessing function
# --------------------------------------------------

def prepare_dataset(dataset_name, sample_size=5000):

    config = DATASETS[dataset_name]

    # Load dataset
    df = pd.read_csv(DATA_DIR / config["file"])

    target = config["target"]
    features = config["features"]

    # Remove rows with missing or invalid target values
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]

    # Sample large datasets
    if len(df) > sample_size:
        df = df.sample(
            n=sample_size,
            random_state=4330
        )

    # Separate predictors and target
    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=4330
    )

    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using training parameters
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train.to_numpy(),
        y_test.to_numpy()
    )


# --------------------------------------------------
# 4. Test the preprocessing function
# --------------------------------------------------

if __name__ == "__main__":

    for name in DATASETS:

        print(f"\nTesting dataset: {name}")

        X_train, X_test, y_train, y_test = prepare_dataset(name)

        print("Training predictors:", X_train.shape)
        print("Testing predictors:", X_test.shape)
        print("Training targets:", y_train.shape)
        print("Testing targets:", y_test.shape)
