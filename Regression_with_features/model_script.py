'Author: Prasad Maharana'


# ============================
# Imports
# ============================
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import helper_utils # Optional plotting / utilities (kept external by design)

SEED = 41
torch.manual_seed(SEED)

# ============================
# Feature Engineering
# ============================
def rush_hour_feature(hours_tensor: torch.Tensor,
                      weekends_tensor: torch.Tensor) -> torch.Tensor:
    """
    Identify rush‑hour periods.

    Rush hour definition:
    - Morning: 8–10
    - Evening: 17–19
    - Excludes weekends
    """
    morning_rush = (hours_tensor >= 8) & (hours_tensor <= 10)
    evening_rush = (hours_tensor >= 17) & (hours_tensor <= 19)

    rush_hour_mask = (~weekends_tensor.bool()) & (morning_rush | evening_rush)
    return rush_hour_mask.float()


# ============================
# Data Preparation
# ============================
def prepare_data(df: pd.DataFrame):
    """
    Convert raw dataframe into model‑ready tensors.

    Expected dataframe columns:
    - distance_miles
    - time_of_day_hours
    - is_weekend
    - delivery_time_minutes (target)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    features : torch.Tensor
    targets : torch.Tensor
    metadata : dict
        Useful intermediate values for debugging/plots
    """

    distance = torch.tensor(df["distance_miles"].values, dtype=torch.float32)
    hours = torch.tensor(df["time_of_day_hours"].values, dtype=torch.float32)
    weekends = torch.tensor(df["is_weekend"].values, dtype=torch.float32)
    targets = torch.tensor(df["delivery_time_minutes"].values,
                           dtype=torch.float32).unsqueeze(1)

    rush_hour = rush_hour_feature(hours, weekends)

    # Feature matrix: [distance, hour, weekend, rush_hour]
    features = torch.stack([
        distance,
        hours,
        weekends,
        rush_hour
    ], dim=1)

    metadata = {
        "rush_hour": rush_hour,
        "hours": hours,
        "weekends": weekends
    }

    return features, targets, metadata


# ============================
# Model Definition
# ============================
def init_model():
    """
    Initialize neural network, optimizer and loss function.

    Returns
    -------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    loss_function : nn.Module
    """

    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    return model, optimizer, loss_function


# ============================
# Training Loop
# ============================
def train_model(features: torch.Tensor,
                targets: torch.Tensor,
                epochs: int = 10_000,
                verbose: bool = True):
    """
    Train the regression model.

    Parameters
    ----------
    features : torch.Tensor
    targets : torch.Tensor
    epochs : int
    verbose : bool

    Returns
    -------
    model : nn.Module
    loss_history : list[float]
    final_outputs : torch.Tensor
    """

    model, optimizer, loss_function = init_model()
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and epoch % 5000 == 0:
            print(f"Epoch {epoch:6d} | Loss: {loss.item():.4f}")

    return model, loss_history, outputs


# ============================
# Inference Helper
# ============================
def predict_delivery_time(model: nn.Module,
                          distance_miles: float,
                          time_of_day_hours: float,
                          is_weekend: int) -> float:
    """
    Predict delivery time for a single input.
    """
    input_tensor = torch.tensor([
        distance_miles,
        time_of_day_hours,
        is_weekend,
        rush_hour_feature(
            torch.tensor([time_of_day_hours]),
            torch.tensor([is_weekend])
        ).item()
    ], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor)

    return prediction.item()


# ============================
# Script Entry Point
# ============================
if __name__ == "__main__":
    # -------- Load data --------
    DATA_PATH = "data_with_features.csv"  # update if needed
    data_df = pd.read_csv(DATA_PATH)

    print("Data preview:\n", data_df.head())
    print("Data shape:", data_df.shape)

    # -------- Prepare tensors --------
    features, targets, meta = prepare_data(data_df)

    # -------- Train model --------
    model, loss_hist, outputs = train_model(
        features,
        targets,
        epochs=70_000,
        verbose=True
    )

    # -------- Visual diagnostics --------
    helper_utils.plot_final_data(features, targets)
    helper_utils.plot_model_predictions(outputs, targets)

    # -------- Example prediction --------
    example_pred = predict_delivery_time(
        model,
        distance_miles=6,
        time_of_day_hours=18,
        is_weekend=0
    )

    print(f"\nExample prediction (minutes): {example_pred:.2f}")
