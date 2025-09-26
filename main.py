# Setup for Google Colab directories. Comment this out in other environtment.
import os
os.chdir('/content/drive/MyDrive/Colab Notebooks/MLexercise_CarDimensions')
print(os.getcwd())
###############################################################

from config import Config
from data.loader import load_data
from models.ModelsTrainer import ModelTrainer as Trainer
from models.evaluator import ModelEvaluator
import torch
from sklearn.preprocessing import StandardScaler
from helpers.utils import plot_scatter_comparison

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
    trainer = Trainer()
    sklearn_model = trainer.train_sklearn(X_train, y_train)
    tensorflow_model = trainer.train_tensorflow(X_scaled, y_scaled, epoch_models=Config.TENSORFLOW_EPOCHS)
    pytorch_model = trainer.train_pytorch(X_train_t, y_train_t, epochs=Config.PYTORCH_EPOCHS)

    # Evaluate model
    logging.info("Evaluating model...")
    evaluator = ModelEvaluator(y_test)

    sk_results = evaluator.evaluate(sklearn_model, X_test, model_name="RandomForest")
    print(sk_results.head())
    tf_results = evaluator.evaluate(tensorflow_model, X_scaled, model_name="TensorFlow", scaler=scaler_y)
    print(tf_results.head())
    X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    pt_results = evaluator.evaluate(pytorch_model, X_test_t, model_name="PyTorch", scaler=scaler_y)
    print(pt_results.head())

    # Plot results
    plot_scatter_comparison(
        dfs=[pt_results, tf_results, sk_results],
        labels=["PyTorch", "TF Model", "sklearn"],
        x_col="y_true", y_col="y_pred",
        xlabel="Actual",
        ylabel="Predicted",
        title="Model Comparison",
        xrange=[0, 4500],
        yrange=[0, 4500],
        # xrange=None,
        # yrange=None,
        drawline=True
    )

    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()