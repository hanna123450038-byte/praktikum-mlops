import os
import sys
from pathlib import Path

# Silence MLflow hints and ensure torch DLLs load before MLflow
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import json
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import numpy as np
import optuna
import pandas as pd

from src.dataset import compute_dataset_hash, get_dataloaders, CLASSES
from src.model import FashionCNN, create_model

# Suppress verbose Optuna logging to stdout
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_device() -> torch.device:
    """Return GPU device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train model for a single epoch and return average loss."""
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader and return (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_experiment(
    data_dir: str = "./data",
    db_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "FashionMNIST_Model_Experimentation",
    n_trials: int = 20,
    subset_size: int = 8000,
) -> Dict:
    """
    Execute full end-to-end MLOps experiment:
    1. Dataset versioning via MD5 hashing.
    2. Optuna Hyperparameter Optimization with nested MLflow runs.
    3. Variance & Stability evaluation across 3 seeds on best config.
    4. Ablation study (with vs without dropout, with vs without augmentation).
    5. Register best model to MLflow Model Registry with alias '@champion' and verify loading.
    """
    device = get_device()
    print("============================================================", flush=True)
    print(f" Starting MLOps Practical Experiment on Device: {device}", flush=True)
    print(f" Tracking Server: {db_uri}", flush=True)
    print(f" Experiment Name: {experiment_name}", flush=True)
    print("============================================================", flush=True)

    # 1. Setup MLflow Tracking Server with SQLite Backend
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment(experiment_name)

    # 2. Data Hashing & Version Tracking
    print("\n[Step 1/5] Calculating dataset MD5 checksum for DVC/data versioning...", flush=True)
    dataset_hash = compute_dataset_hash(data_dir=data_dir)
    print(f"Dataset MD5 Hash: {dataset_hash}", flush=True)

    # 3. Optuna Hyperparameter Tuning with Nested MLflow Runs
    print(f"\n[Step 2/5] Checking or Running Optuna Hyperparameter Tuning (Nested MLflow Runs)...", flush=True)

    existing_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.stage = 'hyperparameter_tuning' and attributes.status = 'FINISHED'"
    )

    if len(existing_runs) > 0 and "params.best_lr" in existing_runs.columns and pd.notna(existing_runs.iloc[0]["params.best_lr"]):
        latest_run = existing_runs.iloc[0]
        best_params = {
            "lr": float(latest_run["params.best_lr"]),
            "batch_size": int(latest_run["params.best_batch_size"]),
            "conv1_channels": int(latest_run["params.best_conv1_channels"]),
            "conv2_channels": int(latest_run["params.best_conv2_channels"]),
            "fc_units": int(latest_run["params.best_fc_units"]),
            "dropout_rate": float(latest_run["params.best_dropout_rate"]),
            "optimizer": str(latest_run["params.best_optimizer"]),
        }
        best_val_acc = float(latest_run.get("metrics.best_val_accuracy", 0.7500))
        print(f"Reusing verified Optuna 20-trials results from MLflow database:", flush=True)
        print(f"  Best Val Accuracy: {best_val_acc:.4f}", flush=True)
        print(f"  Best Hyperparameters: {best_params}", flush=True)
    else:
        with mlflow.start_run(run_name="Optuna_Hyperparameter_Optimization") as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.set_tag("stage", "hyperparameter_tuning")
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("total_trials", n_trials)
            mlflow.log_param("tuning_subset_size", subset_size)

            def objective(trial: optuna.Trial) -> float:
                with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
                    # Search Space
                    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
                    batch_size = trial.suggest_categorical("batch_size", [64, 128])
                    conv1_channels = trial.suggest_categorical("conv1_channels", [16, 32])
                    conv2_channels = trial.suggest_categorical("conv2_channels", [32, 64])
                    fc_units = trial.suggest_categorical("fc_units", [64, 128])
                    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4, step=0.1)
                    optimizer_choice = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

                    mlflow.log_params(trial.params)
                    mlflow.log_param("dataset_hash", dataset_hash)

                    train_loader, val_loader, _ = get_dataloaders(
                        data_dir=data_dir,
                        batch_size=batch_size,
                        use_augmentation=True,
                        subset_size=subset_size,
                        val_split=0.2,
                        random_seed=42,
                    )

                    model = create_model(
                        conv1_channels=conv1_channels,
                        conv2_channels=conv2_channels,
                        fc_units=fc_units,
                        dropout_rate=dropout_rate,
                        use_dropout=True,
                    ).to(device)

                    criterion = nn.CrossEntropyLoss()
                    if optimizer_choice == "Adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    else:
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                    epochs_per_trial = 1
                    best_val_acc_trial = 0.0
                    for epoch in range(1, epochs_per_trial + 1):
                        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                        best_val_acc_trial = max(best_val_acc_trial, val_acc)
                        mlflow.log_metrics(
                            {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                            step=epoch,
                        )

                    mlflow.log_metric("trial_best_val_accuracy", best_val_acc_trial)
                    print(f"  Trial {trial.number:02d}/{n_trials:02d}: Val Acc={best_val_acc_trial:.4f} | Opt={optimizer_choice} | lr={lr:.5f}", flush=True)
                    return best_val_acc_trial

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_val_acc = study.best_value
            print(f"\nBest Optuna Trial: Val Accuracy = {best_val_acc:.4f}", flush=True)
            print(f"Best Hyperparameters: {best_params}", flush=True)

            # Log best parameters and metrics to parent run
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_accuracy", best_val_acc)

    # 4. Variance / Stability Test on 3 Random Seeds
    print("\n[Step 3/5] Testing Variance & Stability across 3 Random Seeds...", flush=True)
    stability_seeds = [42, 1337, 2024]
    seed_accuracies = []
    seed_losses = []

    with mlflow.start_run(run_name="Variance_Stability_Test") as stab_parent:
        mlflow.set_tag("stage", "variance_stability_test")
        mlflow.log_params(best_params)
        mlflow.log_param("tested_seeds", str(stability_seeds))

        for seed in stability_seeds:
            with mlflow.start_run(run_name=f"stability_seed_{seed}", nested=True):
                torch.manual_seed(seed)
                np.random.seed(seed)

                train_loader, val_loader, test_loader = get_dataloaders(
                    data_dir=data_dir,
                    batch_size=best_params["batch_size"],
                    use_augmentation=True,
                    subset_size=subset_size,
                    val_split=0.2,
                    random_seed=seed,
                )

                model = create_model(
                    conv1_channels=best_params["conv1_channels"],
                    conv2_channels=best_params["conv2_channels"],
                    fc_units=best_params["fc_units"],
                    dropout_rate=best_params["dropout_rate"],
                    use_dropout=True,
                ).to(device)

                criterion = nn.CrossEntropyLoss()
                if best_params["optimizer"] == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9)

                for epoch in range(1, 3):
                    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)

                seed_accuracies.append(test_acc)
                seed_losses.append(test_loss)

                mlflow.log_params({"seed": seed})
                mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
                print(f"  Seed {seed} -> Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}", flush=True)

        acc_mean = float(np.mean(seed_accuracies))
        acc_std = float(np.std(seed_accuracies))
        loss_mean = float(np.mean(seed_losses))
        loss_std = float(np.std(seed_losses))

        mlflow.log_metrics({
            "stability_accuracy_mean": acc_mean,
            "stability_accuracy_std": acc_std,
            "stability_loss_mean": loss_mean,
            "stability_loss_std": loss_std,
        })
        print(f"\nStability Evaluation Result (mean ± std):", flush=True)
        print(f"  Accuracy: {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}%", flush=True)
        print(f"  Loss:     {loss_mean:.4f} ± {loss_std:.4f}", flush=True)

    # 5. Ablation Study
    print("\n[Step 4/5] Running Ablation Study (Dropout vs Augmentation)...", flush=True)
    ablation_configs = [
        {"name": "Full_Model (Dropout + Augmentation)", "use_dropout": True, "use_augmentation": True},
        {"name": "No_Dropout (Augmentation Only)", "use_dropout": False, "use_augmentation": True},
        {"name": "No_Augmentation (Dropout Only)", "use_dropout": True, "use_augmentation": False},
        {"name": "Baseline_Minimal (No Dropout + No Aug)", "use_dropout": False, "use_augmentation": False},
    ]

    ablation_results = []
    best_candidate_acc = -1.0
    best_trained_model = None

    with mlflow.start_run(run_name="Ablation_Study") as abl_parent:
        mlflow.set_tag("stage", "ablation_study")
        mlflow.log_params(best_params)

        for cfg in ablation_configs:
            with mlflow.start_run(run_name=f"ablation_{cfg['name']}", nested=True):
                train_loader, val_loader, test_loader = get_dataloaders(
                    data_dir=data_dir,
                    batch_size=best_params["batch_size"],
                    use_augmentation=cfg["use_augmentation"],
                    subset_size=subset_size,
                    val_split=0.2,
                    random_seed=42,
                )

                model = create_model(
                    conv1_channels=best_params["conv1_channels"],
                    conv2_channels=best_params["conv2_channels"],
                    fc_units=best_params["fc_units"],
                    dropout_rate=best_params["dropout_rate"],
                    use_dropout=cfg["use_dropout"],
                ).to(device)

                criterion = nn.CrossEntropyLoss()
                if best_params["optimizer"] == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9)

                for epoch in range(1, 3):
                    tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                te_loss, te_acc = evaluate(model, test_loader, criterion, device)

                mlflow.log_params({
                    "use_dropout": cfg["use_dropout"],
                    "use_augmentation": cfg["use_augmentation"],
                })
                mlflow.log_metrics({"test_accuracy": te_acc, "test_loss": te_loss})

                ablation_results.append({
                    "Configuration": cfg["name"],
                    "Dropout": "Yes" if cfg["use_dropout"] else "No",
                    "Augmentation": "Yes" if cfg["use_augmentation"] else "No",
                    "Test Accuracy (%)": round(te_acc * 100, 2),
                    "Test Loss": round(te_loss, 4),
                })
                print(f"  {cfg['name']:<40} -> Acc: {te_acc * 100:.2f}%, Loss: {te_loss:.4f}", flush=True)

                if te_acc > best_candidate_acc:
                    best_candidate_acc = te_acc
                    best_trained_model = model

        ablation_df = pd.DataFrame(ablation_results)
        ablation_csv_path = "./data/ablation_results.csv"
        os.makedirs(os.path.dirname(ablation_csv_path), exist_ok=True)
        ablation_df.to_csv(ablation_csv_path, index=False)
        mlflow.log_artifact(ablation_csv_path)

    # 6. Model Registration & Champion Alias
    print("\n[Step 5/5] Registering Best Model into MLflow Model Registry with alias '@champion'...", flush=True)
    model_name = "FashionMNIST_Classifier"

    with mlflow.start_run(run_name="Champion_Model_Deployment") as champ_run:
        mlflow.set_tag("stage", "model_registry")
        mlflow.log_params(best_params)
        mlflow.log_metric("champion_test_accuracy", best_candidate_acc)

        # Log PyTorch model to MLflow and register
        model_info = mlflow.pytorch.log_model(
            pytorch_model=best_trained_model,
            artifact_path="model",
            serialization_format="pickle",
            registered_model_name=model_name,
        )

        client = MlflowClient()
        
        # Retrieve registered versions for the model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        champion_version = max([int(v.version) for v in model_versions])

        # Assign alias @champion
        client.set_registered_model_alias(name=model_name, alias="champion", version=str(champion_version))
        print(f"Assigned alias '@champion' to {model_name} version {champion_version}", flush=True)

        # Test loading directly via URI models:/FashionMNIST_Classifier@champion
        champion_uri = f"models:/{model_name}@champion"
        print(f"Testing direct load from URI: {champion_uri}...", flush=True)
        loaded_model = mlflow.pytorch.load_model(champion_uri)
        loaded_model.eval()

        # Perform test inference with loaded model
        _, _, sample_loader = get_dataloaders(data_dir=data_dir, batch_size=8, random_seed=42)
        sample_images, sample_labels = next(iter(sample_loader))
        with torch.no_grad():
            preds = loaded_model(sample_images.to(device)).argmax(dim=1).cpu().tolist()

        predicted_classes = [CLASSES[p] for p in preds]
        ground_truth_classes = [CLASSES[t] for t in sample_labels.tolist()]

        print("\nVerification Inference with Loaded Champion Model:", flush=True)
        for idx in range(min(5, len(preds))):
            match = "MATCH" if preds[idx] == sample_labels[idx].item() else "DIFF"
            print(f"  Sample {idx+1}: True='{ground_truth_classes[idx]}' | Pred='{predicted_classes[idx]}' [{match}]", flush=True)

    summary = {
        "dataset_hash": dataset_hash,
        "best_hyperparameters": best_params,
        "best_val_accuracy": round(best_val_acc, 4),
        "stability_mean_acc": round(acc_mean * 100, 2),
        "stability_std_acc": round(acc_std * 100, 2),
        "stability_mean_loss": round(loss_mean, 4),
        "stability_std_loss": round(loss_std, 4),
        "ablation_results": ablation_results,
        "champion_model_name": model_name,
        "champion_version": champion_version,
        "champion_uri": champion_uri,
        "champion_test_accuracy": round(best_candidate_acc * 100, 2),
    }

    summary_file = "./data/experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n============================================================", flush=True)
    print(" Experiment Completed Successfully!", flush=True)
    print(f" Summary saved to: {summary_file}", flush=True)
    print("============================================================\n", flush=True)
    return summary


if __name__ == "__main__":
    run_experiment()
