# Models' class
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ModelTrainer:
    def __init__(self):
        self.loss_history = []
        self.weight_history = []
        self.predictions_history = []

    def reset_history(self):
        """Clear stored histories before a new training run."""
        self.loss_history.clear()
        self.weight_history.clear()
        self.predictions_history.clear()

    def train_sklearn(self, X_train, y_train):
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_tensorflow(self, X_train, y_train, epochs=500):# Model layers and training
        model = Sequential([
            Dense(256, activation='relu',
                input_shape=(X_train.shape[1],)
                ),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)  # output layer for regression
        ])

        model.compile(optimizer='adam', loss='mse') # Default learning_rate=0.001

        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            # validation_split=0.2,  # 20% of training data used for validation
            callbacks=[early_stop],
            verbose=1)
        return model

    def train_pytorch(self, X_train, y_train, epochs=100):
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            for xb, yb in loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # === Logging ===
                self.loss_history.append(loss.item())
                self.weight_history.append([p.detach().cpu().clone() for p in model.parameters()])
                self.predictions_history.append(pred.detach().cpu().numpy())

                # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        return model
