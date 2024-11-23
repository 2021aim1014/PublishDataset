import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def write_to_csv(model_name, num_epochs, train_losses, val_losses, train_metrics, val_metrics, total_params, trainable_params, non_trainable_params):
    # Save results to CSV
    results = {
        "Epoch": list(range(1, num_epochs + 1)),
        "Train Loss": train_losses,
        "Train MSE": train_metrics["MSE"],
        "Train MAE": train_metrics["MAE"],
        "Train R2": train_metrics["R2"],
    }
    
    # Add parameter counts as additional rows (repeated across all rows for clarity)
    results["Total Params"] = [total_params] * num_epochs
    results["Trainable Params"] = [trainable_params] * num_epochs
    results["Non-trainable Params"] = [non_trainable_params] * num_epochs

    results["Val Loss"] = val_losses * num_epochs
    results["Val MSE"] = val_metrics["MSE"] * num_epochs
    results["Val MAE"] = val_metrics["MAE"] * num_epochs
    results["Val R2"] = val_metrics["R2"] * num_epochs

    df = pd.DataFrame(results)
    df.to_csv(f"results/{model_name}.csv", index=False)
    print("Results saved to 'training_results_with_params.csv'")

def calculate_model_parameters(model):
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    return total_params, trainable_params, non_trainable_params


def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    total_params, trainable_params, non_trainable_params = calculate_model_parameters(model)

    # Early stopping parameters
    patience = 5
    best_train_loss = float('inf')
    epochs_no_improve = 0

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_metrics = {"MSE": [], "MAE": [], "R2": []}
    val_metrics = {"MSE": [], "MAE": [], "R2": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and predictions
            running_loss += loss.item()
            y_true_train.append(labels)
            y_pred_train.append(outputs)
            break

        # Normalize training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Calculate training metrics
        y_true_train = torch.cat(y_true_train, dim=0)
        y_pred_train = torch.cat(y_pred_train, dim=0)
        mse, mae, r2 = calculate_metrics(y_true_train, y_pred_train)
        train_metrics["MSE"].append(mse)
        train_metrics["MAE"].append(mae)
        train_metrics["R2"].append(r2)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Early stopping logic
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            epochs_no_improve = 0
            # Optionally save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")

    # Validation phase
    model.eval()
    val_loss = 0.0
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Accumulate validation loss and predictions
            val_loss += loss.item()
            y_true_val.append(labels)
            y_pred_val.append(outputs)
            break

    # Normalize validation loss
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Calculate validation metrics
    y_true_val = torch.cat(y_true_val, dim=0)
    y_pred_val = torch.cat(y_pred_val, dim=0)
    mse, mae, r2 = calculate_metrics(y_true_val, y_pred_val)
    val_metrics["MSE"].append(mse)
    val_metrics["MAE"].append(mae)
    val_metrics["R2"].append(r2)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    write_to_csv(model_name, num_epochs, train_losses, val_losses, train_metrics, val_metrics, total_params, trainable_params, non_trainable_params)
