from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle
import logging
import numpy as np
import sys
from keras.models import load_model

from naviflame.utils import MyMagnWarping, MyScaling
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import time


def fine_tune_model(
    feature_extractor_path,
    recorded_data, 
    recorded_labels, 
    mlp_model_path,
    scaler_path,
):
    """
    Fine-tunes the MLP model using the recorded data.
    
    Args:
        model (keras.models.Model): Feature extractor model.
        recorded_data (list): List of recorded data.
        recorded_labels (list): List of recorded labels.
        mlp_model_path (str): Path to save the MLP model.
        scaler_path (str): Path to save the scaler.
        
        Returns:
            tuple: (MLP model, Scaler, Validation accuracy)        
    """
    model = load_model(feature_extractor_path, custom_objects={"MyMagnWarping": MyMagnWarping, "MyScaling": MyScaling})

    # Split data into training and validation
    y_labels = np.array(recorded_labels)
    max_label = np.max(y_labels)
    if max_label > 0:
        y_labels = y_labels.astype(np.float32) / max_label
        logging.info(f"Labels normalized to [0, 1] range by dividing by {max_label}.")
    else:
        y_labels = y_labels.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(np.array(recorded_data), y_labels, test_size=0.2, random_state=42)

    # Feature extraction
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer("dense_8").output)
    features_train = feature_extractor.predict(X_train)
    features_val = feature_extractor.predict(X_val)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_train)
    X_val_scaled = scaler.transform(features_val)
    logging.info("Data preprocessed and scaled.")

    # Save the scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        logging.info("Scaler saved.")

    # MLP regressor training for continuous force prediction
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=5000,
        random_state=42,
        activation='tanh',
        solver='lbfgs',
        alpha=0.01,
        learning_rate='constant',
    )

    # Fit regressor (expects continuous target values)
    mlp.fit(X_train_scaled, y_train)

    # Validation predictions and regression metrics
    y_pred = mlp.predict(X_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    logging.info(f"MLP validation MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Save the MLP regressor
    with open(mlp_model_path, "wb") as f:
        pickle.dump(mlp, f)
        logging.info("MLP regressor saved.")

    # Return scaler and validation metrics
    metrics = {"mse": float(mse), "mae": float(mae), "r2": float(r2)}
    return scaler, metrics
