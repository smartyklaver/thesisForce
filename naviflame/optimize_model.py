from typing import Tuple, Dict, Any, Optional
import logging
import pickle

import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform

from keras.models import load_model, Model

from naviflame.utils import MyMagnWarping, MyScaling

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def optimize_mlp(
    feature_extractor_path: str,
    recorded_data: np.ndarray,
    recorded_labels: np.ndarray,
    mlp_model_path: str,
    scaler_path: str,
    layer_name: str = "dense_8",
    test_size: float = 0.2,
    n_iter_search: int = 20,
    cv: int = 3,
    random_state: int = 42,
    max_iter: int = 5000,
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fine-tunes an MLP regressor using features from a pretrained Keras feature extractor.

    This function:
    - Loads a Keras model and extracts features from `layer_name`.
    - Scales features and target values.
    - Runs a `RandomizedSearchCV` over sensible MLP hyperparameters.
    - Saves the best MLP and both scalers to disk.

    Returns the best model and a dict with metrics and best_params.

    Example:
        best_model, info = fine_tune_mlp(...)
        # load scalers:
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)

    """
    model = load_model(feature_extractor_path, custom_objects={"MyMagnWarping": MyMagnWarping, "MyScaling": MyScaling})

    X = np.asarray(recorded_data)
    y = np.asarray(recorded_labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Feature extraction
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features_train = feature_extractor.predict(X_train)
    features_val = feature_extractor.predict(X_val)

    # Feature scaling
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(features_train)
    X_val_scaled = X_scaler.transform(features_val)

    # Target scaling
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Pipeline + MLP
    base_mlp = MLPRegressor(random_state=random_state, max_iter=max_iter)
    pipe = Pipeline([("mlp", base_mlp)])

    param_dist = {
        "mlp__hidden_layer_sizes": [(32,), (32, 16), (64, 32), (64, 32, 16), (128, 64, 32)],
        "mlp__solver": ["adam", "lbfgs"],
        "mlp__alpha": [0.0001, 0.001, 0.01, 0.1],
        "mlp__learning_rate_init": [0.0001, 0.001, 0.01, 0.05, 0.07, 0.1],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    try:
        search.fit(X_train_scaled, y_train_scaled)
        best_model = search.best_estimator_.named_steps["mlp"]
        best_params = search.best_params_
        logger.info(f"RandomizedSearchCV best params: {best_params}")
    except Exception as e:
        logger.warning(f"RandomizedSearchCV failed ({e}), falling back to default MLP fit.")
        best_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=max_iter, random_state=random_state)
        best_model.fit(X_train_scaled, y_train_scaled)
        best_params = {"fallback": True}

    # Validate and inverse-transform predictions
    y_pred_scaled = best_model.predict(X_val_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    metrics = {"mse": float(mse), "mae": float(mae), "r2": float(r2)}
    logger.info(f"Validation MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Save model and scalers
    with open(mlp_model_path, "wb") as f:
        pickle.dump(best_model, f)
        logger.info(f"Saved MLP regressor to {mlp_model_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump({"X_scaler": X_scaler, "y_scaler": y_scaler}, f)
        logger.info(f"Saved scalers to {scaler_path}")

    info = {"metrics": metrics, "best_params": best_params}
    return best_model, info


def load_model_and_scalers(mlp_model_path: str, scaler_path: str):
    """Helper to load the saved MLP and scalers dict."""
    with open(mlp_model_path, "rb") as f:
        mlp = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    return mlp, scalers
