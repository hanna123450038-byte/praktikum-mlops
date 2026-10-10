import json
from pathlib import Path


def build_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True,
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Cell 1: Header Markdown
    add_md("""# Praktikum MLOps: Model Experimentation & Tracking
**Nama:** Hanna Gresia Sinaga  
**NIM:** 123450038  
**Topik:** Experiment Tracking, Hyperparameter Tuning (Optuna), Variance Stability Testing, Ablation Study, & MLflow Model Registry (`@champion` Alias)  
**Dataset:** Fashion-MNIST  

Notebook ini dirancang untuk dapat dijalankan secara langsung (end-to-end) baik secara lokal maupun di Google Colab.""")

    # Cell 2: Dependencies Installation
    add_code("""# Install dependencies required for MLOps pipeline
!pip install -q torch torchvision mlflow optuna dvc scikit-learn pandas numpy reportlab""")

    # Cell 3: Setup MLflow Tracking Server
    add_md("""## 1. Konfigurasi MLflow Tracking Server (SQLite Backend)
Menginisialisasi tracking server MLflow dengan backend database SQLite (`sqlite:///mlflow.db`) agar artefak dan metadata eksperimen tersimpan secara persisten.""")

    add_code("""import os
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import optuna

# Atur tracking server dengan backend SQLite
DB_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "FashionMNIST_Model_Experimentation"

mlflow.set_tracking_uri(DB_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment Name: {EXPERIMENT_NAME}")""")

    # Cell 4: Dataset & Hashing
    add_md("""## 2. Dataset Fashion-MNIST & MD5 Checksum untuk Data Versioning
Fungsi hash MD5 digunakan untuk membuat sidik jari (*fingerprint*) versi data guna menjamin integritas dan reproduktifitas data pada pipeline MLOps.""")

    add_code("""CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def download_fashion_mnist(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    datasets.FashionMNIST(root=data_dir, train=True, download=True)
    datasets.FashionMNIST(root=data_dir, train=False, download=True)

def compute_file_md5(file_path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dataset_hash(data_dir="./data") -> str:
    path = Path(data_dir)
    download_fashion_mnist(data_dir)
    all_files = sorted([f for f in path.rglob("*") if f.is_file()])
    master_hasher = hashlib.md5()
    for file_path in all_files:
        rel_path = file_path.relative_to(path).as_posix()
        file_hash = compute_file_md5(file_path)
        master_hasher.update(rel_path.encode("utf-8"))
        master_hasher.update(file_hash.encode("utf-8"))
    return master_hasher.hexdigest()

def get_dataloaders(
    data_dir="./data",
    batch_size=64,
    use_augmentation=False,
    subset_size=12000,
    val_split=0.2,
    random_seed=42
):
    norm = transforms.Normalize((0.2860,), (0.3530,))
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(28, padding=2, padding_mode="edge"),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), norm])
    
    test_transform = transforms.Compose([transforms.ToTensor(), norm])
    
    full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_transform)
    
    if subset_size is not None and subset_size < len(full_train):
        gen = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(len(full_train), generator=gen)[:subset_size]
        base_ds = Subset(full_train, indices)
    else:
        base_ds = full_train
        
    val_size = int(len(base_ds) * val_split)
    train_size = len(base_ds) - val_size
    gen = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset = random_split(base_ds, [train_size, val_size], generator=gen)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Hitung MD5 Hash Dataset
data_hash = compute_dataset_hash("./data")
print(f"Fashion-MNIST MD5 Checksum Hash: {data_hash}")""")

    # Cell 5: Model Definition
    add_md("""## 3. Arsitektur Model CNN dengan Opsi Dropout
Model Convolutional Neural Network dirancang fleksibel dengan parameter dropout yang dapat diaktifkan/dinonaktifkan untuk mendukung *Ablation Study*.""")

    add_code("""class FashionCNN(nn.Module):
    def __init__(
        self,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        fc_units: int = 128,
        dropout_rate: float = 0.25,
        use_dropout: bool = True,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate if use_dropout else 0.0
        self.use_dropout = use_dropout
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.pool2 = nn.MaxPool2d(2, 2)

        flatten_dim = conv2_channels * 7 * 7
        self.fc1 = nn.Linear(flatten_dim, fc_units)
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.use_dropout and self.dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(fc_units, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device yang digunakan: {device}")""")

    # Cell 6: Training & Evaluation Functions
    add_md("""## 4. Fungsi Training & Evaluasi""")

    add_code("""def train_epoch(model, loader, criterion, optimizer, device):
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

def evaluate(model, loader, criterion, device):
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
    return running_loss / total, correct / total""")

    # Cell 7: Optuna Hyperparameter Tuning
    add_md("""## 5. Hyperparameter Tuning dengan Optuna (20 Trials Nested MLflow Runs)
Setiap trial Optuna dicatat sebagai nested run di dalam parent run `Optuna_Hyperparameter_Optimization`.""")

    add_code("""optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Memulai 20 Trial Optuna Hyperparameter Optimization dengan Nested MLflow Runs...")

with mlflow.start_run(run_name="Optuna_Hyperparameter_Optimization") as parent_run:
    mlflow.set_tag("stage", "hyperparameter_tuning")
    mlflow.log_param("dataset_hash", data_hash)
    mlflow.log_param("total_trials", 20)

    def objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128])
            conv1_channels = trial.suggest_categorical("conv1_channels", [16, 32])
            conv2_channels = trial.suggest_categorical("conv2_channels", [32, 64])
            fc_units = trial.suggest_categorical("fc_units", [64, 128])
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4, step=0.1)
            optimizer_choice = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

            mlflow.log_params(trial.params)
            mlflow.log_param("dataset_hash", data_hash)

            train_loader, val_loader, _ = get_dataloaders(
                data_dir="./data",
                batch_size=batch_size,
                use_augmentation=True,
                subset_size=12000,
                val_split=0.2,
                random_seed=42
            )

            model = FashionCNN(
                conv1_channels=conv1_channels,
                conv2_channels=conv2_channels,
                fc_units=fc_units,
                dropout_rate=dropout_rate,
                use_dropout=True
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            if optimizer_choice == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            best_val = 0.0
            for epoch in range(1, 3):
                tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                best_val = max(best_val, val_acc)
                mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)

            mlflow.log_metric("trial_best_val_accuracy", best_val)
            return best_val

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    best_val_acc = study.best_value
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_val_accuracy", best_val_acc)

print(f"Optimal Val Accuracy: {best_val_acc:.4f}")
print("Konfigurasi Hyperparameter Terbaik:")
for k, v in best_params.items():
    print(f"  {k}: {v}")""")

    # Cell 8: Stability & Variance Test
    add_md("""## 6. Uji Variansi dan Stabilitas (3 Random Seed)
Menguji konfigurasi terbaik pada 3 seed berbeda (`[42, 1337, 2024]`) dan melaporkan agregasi `mean ± std`.""")

    add_code("""stability_seeds = [42, 1337, 2024]
seed_accuracies = []
seed_losses = []

print("Menjalankan Uji Stabilitas pada 3 Random Seeds...")

with mlflow.start_run(run_name="Variance_Stability_Test") as stab_parent:
    mlflow.set_tag("stage", "variance_stability_test")
    mlflow.log_params(best_params)
    mlflow.log_param("tested_seeds", str(stability_seeds))

    for seed in stability_seeds:
        with mlflow.start_run(run_name=f"stability_seed_{seed}", nested=True):
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_loader, val_loader, test_loader = get_dataloaders(
                data_dir="./data",
                batch_size=best_params["batch_size"],
                use_augmentation=True,
                subset_size=12000,
                val_split=0.2,
                random_seed=seed
            )

            model = FashionCNN(
                conv1_channels=best_params["conv1_channels"],
                conv2_channels=best_params["conv2_channels"],
                fc_units=best_params["fc_units"],
                dropout_rate=best_params["dropout_rate"],
                use_dropout=True
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            if best_params["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9)

            for epoch in range(1, 3):
                train_epoch(model, train_loader, criterion, optimizer, device)
            te_loss, te_acc = evaluate(model, test_loader, criterion, device)

            seed_accuracies.append(te_acc)
            seed_losses.append(te_loss)

            mlflow.log_params({"seed": seed})
            mlflow.log_metrics({"test_loss": te_loss, "test_accuracy": te_acc})
            print(f"Seed {seed} -> Test Accuracy: {te_acc:.4f}, Test Loss: {te_loss:.4f}")

    acc_mean = float(np.mean(seed_accuracies))
    acc_std = float(np.std(seed_accuracies))
    loss_mean = float(np.mean(seed_losses))
    loss_std = float(np.std(seed_losses))

    mlflow.log_metrics({
        "stability_accuracy_mean": acc_mean,
        "stability_accuracy_std": acc_std,
        "stability_loss_mean": loss_mean,
        "stability_loss_std": loss_std
    })

print(f"\\nHasil Uji Stabilitas (Mean ± Std):")
print(f"  Test Accuracy: {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}%")
print(f"  Test Loss:     {loss_mean:.4f} ± {loss_std:.4f}")""")

    # Cell 9: Ablation Study
    add_md("""## 7. Ablation Study (Dropout vs Data Augmentation)
Menganalisis kontribusi individual komponen regulasi Dropout dan Augmentasi data.""")

    add_code("""ablation_configs = [
    {"name": "Full_Model (Dropout + Augmentation)", "use_dropout": True, "use_augmentation": True},
    {"name": "No_Dropout (Augmentation Only)", "use_dropout": False, "use_augmentation": True},
    {"name": "No_Augmentation (Dropout Only)", "use_dropout": True, "use_augmentation": False},
    {"name": "Baseline_Minimal (No Dropout + No Aug)", "use_dropout": False, "use_augmentation": False},
]

ablation_results = []
best_candidate_acc = -1.0
champion_trained_model = None

with mlflow.start_run(run_name="Ablation_Study") as abl_parent:
    mlflow.set_tag("stage", "ablation_study")
    mlflow.log_params(best_params)

    for cfg in ablation_configs:
        with mlflow.start_run(run_name=f"ablation_{cfg['name']}", nested=True):
            train_loader, val_loader, test_loader = get_dataloaders(
                data_dir="./data",
                batch_size=best_params["batch_size"],
                use_augmentation=cfg["use_augmentation"],
                subset_size=12000,
                val_split=0.2,
                random_seed=42
            )

            model = FashionCNN(
                conv1_channels=best_params["conv1_channels"],
                conv2_channels=best_params["conv2_channels"],
                fc_units=best_params["fc_units"],
                dropout_rate=best_params["dropout_rate"],
                use_dropout=cfg["use_dropout"]
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            if best_params["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9)

            for epoch in range(1, 3):
                train_epoch(model, train_loader, criterion, optimizer, device)
            te_loss, te_acc = evaluate(model, test_loader, criterion, device)

            mlflow.log_params({"use_dropout": cfg["use_dropout"], "use_augmentation": cfg["use_augmentation"]})
            mlflow.log_metrics({"test_accuracy": te_acc, "test_loss": te_loss})

            ablation_results.append({
                "Configuration": cfg["name"],
                "Dropout": "Yes" if cfg["use_dropout"] else "No",
                "Augmentation": "Yes" if cfg["use_augmentation"] else "No",
                "Test Accuracy (%)": round(te_acc * 100, 2),
                "Test Loss": round(te_loss, 4),
            })

            if te_acc > best_candidate_acc:
                best_candidate_acc = te_acc
                champion_trained_model = model

df_ablation = pd.DataFrame(ablation_results)
df_ablation.to_csv("./data/ablation_results.csv", index=False)
mlflow.log_artifact("./data/ablation_results.csv")

print("\\nTabel Hasil Ablation Study:")
display(df_ablation)""")

    # Cell 10: Model Registry & Alias Champion
    add_md("""## 8. MLflow Model Registry & Deployment Alias `@champion`
Mendaftarkan model terbaik ke Model Registry, menetapkan alias `@champion`, dan memverifikasi loading model langsung dari URI `models:/...`.""")

    add_code("""model_name = "FashionMNIST_Classifier"

with mlflow.start_run(run_name="Champion_Model_Deployment") as champ_run:
    mlflow.set_tag("stage", "model_registry")
    mlflow.log_params(best_params)
    mlflow.log_metric("champion_test_accuracy", best_candidate_acc)

    # Log model ke MLflow
    model_info = mlflow.pytorch.log_model(
        pytorch_model=champion_trained_model,
        artifact_path="model",
        registered_model_name=model_name
    )

    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    champion_version = max([int(v.version) for v in model_versions])

    # Tetapkan alias @champion
    client.set_registered_model_alias(name=model_name, alias="champion", version=str(champion_version))
    print(f"Model {model_name} versi {champion_version} ditetapkan dengan alias '@champion'!")

    # Test load model dari URI models:/FashionMNIST_Classifier@champion
    champion_uri = f"models:/{model_name}@champion"
    print(f"Loading model dari URI: {champion_uri}...")
    loaded_champion = mlflow.pytorch.load_model(champion_uri)
    loaded_champion.eval()

    # Verifikasi inferensi
    _, _, test_loader = get_dataloaders(data_dir="./data", batch_size=5, random_seed=42)
    sample_images, sample_labels = next(iter(test_loader))
    with torch.no_grad():
        preds = loaded_champion(sample_images.to(device)).argmax(dim=1).cpu().tolist()

    print("\\nVerifikasi Inferensi Menggunakan Model Champion:")
    for idx in range(len(preds)):
        t_label = CLASSES[sample_labels[idx].item()]
        p_label = CLASSES[preds[idx]]
        status = "MATCH" if preds[idx] == sample_labels[idx].item() else "DIFF"
        print(f"Sample {idx+1}: True={t_label:<12} | Pred={p_label:<12} [{status}]")""")

    # Cell 11: Export JSON Summary
    add_md("""## 9. Simpan Ringkasan Hasil Eksperimen""")

    add_code("""summary = {
    "dataset_hash": data_hash,
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
    "champion_test_accuracy": round(best_candidate_acc * 100, 2)
}

with open("./data/experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Eksperimen Selesai! Ringkasan berhasil disimpan ke ./data/experiment_summary.json")""")

    out_file = Path("notebook_experiment.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Jupyter Notebook successfully written to: {out_file.resolve()}")


if __name__ == "__main__":
    build_notebook()
