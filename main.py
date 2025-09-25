from config import Config
from data.loader import load_data
from models.ModelsTrainer import train_sklearn, train_tensorflow, train_pytorch
from models.evaluator import evaluate_model
import torch

from sklearn.preprocessing import StandardScaler

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    logging.info("Starting ML pipeline...")

    # Load data
    logging.info("Loading dataset...")
    X_train, X_test, y_train, y_test = load_data(Config.DATA_PATH, Config.features, Config.target, Config.TEST_SIZE, Config.RANDOM_SEED)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    X_train_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

    # Train model
    logging.info("Training model...")
    model, acc = train_random_forest(
        X, y,
        test_size=Config.TEST_SIZE,
        n_estimators=Config.N_ESTIMATORS,
        random_state=Config.RANDOM_SEED
    )
    logging.info(f"Training completed with accuracy: {acc:.2f}")

    # Evaluate model
    logging.info("Evaluating model...")
    metrics = evaluate_model(model, X, y)
    logging.info(f"Evaluation metrics: {metrics}")

    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()