"""
FedX-HAD: Explainable Federated Hybrid Anomaly Detection for IDS

Colab use:
1. Upload these files to the Colab working directory, or mount Drive:
   - UNSW_NB15_training-set.csv
   - UNSW_NB15_testing-set.csv
2. Run this file/cells.

If the UNSW-NB15 files are not found, the code uses a synthetic IDS-like
dataset so the full pipeline still executes for demonstration and debugging.
Use real UNSW-NB15 results in the final thesis/report.
"""

import os
import random
import warnings
import json
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is missing. In Colab, run: !pip install torch"
    ) from exc

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "A required ML package is missing. In Colab, run: "
        "!pip install pandas numpy scikit-learn matplotlib seaborn"
    ) from exc


@dataclass
class Config:
    project_name: str = "FedX-HAD"
    seed: int = 42
    n_clients: int = 5
    federated_rounds: int = 8
    local_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    window_size: int = 5
    lambda_cdaw: float = 0.15
    adaptive_cdaw_beta: float = 0.35
    ambiguity_low: float = 0.45
    ambiguity_high: float = 0.55
    max_train_windows: int = 30000
    max_test_windows: int = 30000
    use_only_normal_for_ae: bool = True
    enable_poisoned_client_stress_test: bool = True
    poisoned_client_id: int = 1
    poison_fraction: float = 0.35
    trust_validation_limit: int = 5000
    validation_fraction: float = 0.20
    trust_update_penalty_strength: float = 1.0
    output_dir: str = "fedx_had_outputs"


CFG = Config()
os.makedirs(CFG.output_dir, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on {DEVICE}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TA-FedX-CPS detector with optional poisoned-client stress testing"
    )
    poison_group = parser.add_mutually_exclusive_group()
    poison_group.add_argument(
        "--poison",
        action="store_true",
        help="Enable controlled label poisoning on one federated client.",
    )
    poison_group.add_argument(
        "--no-poison",
        action="store_true",
        help="Disable client poisoning and run the clean baseline.",
    )
    parser.add_argument(
        "--poison-client",
        type=int,
        default=CFG.poisoned_client_id,
        help="1-based client id to poison when poisoning is enabled.",
    )
    parser.add_argument(
        "--poison-fraction",
        type=float,
        default=CFG.poison_fraction,
        help="Fraction of the selected client's labels to flip.",
    )
    parser.add_argument(
        "--output-dir",
        default=CFG.output_dir,
        help="Folder where metrics, figures, and exported event JSON are saved.",
    )
    return parser.parse_args()


def configure_from_args(args):
    if args.poison:
        CFG.enable_poisoned_client_stress_test = True
    if args.no_poison:
        CFG.enable_poisoned_client_stress_test = False
    CFG.poisoned_client_id = args.poison_client
    CFG.poison_fraction = float(np.clip(args.poison_fraction, 0.0, 1.0))
    CFG.output_dir = args.output_dir
    os.makedirs(CFG.output_dir, exist_ok=True)


def find_file(filename: str):
    search_dirs = [
        ".",
        "/content",
        "/content/drive/MyDrive",
        "/content/drive/MyDrive/datasets",
        "/content/drive/MyDrive/UNSW_NB15",
    ]
    for base in search_dirs:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    return None


def make_synthetic_ids_data(n_train=12000, n_test=5000):
    rng = np.random.default_rng(CFG.seed)

    def make_part(n, attack_rate):
        y = rng.binomial(1, attack_rate, size=n)
        base = rng.normal(0, 1, size=(n, 14))
        drift = y.reshape(-1, 1) * rng.normal(1.4, 0.7, size=(n, 14))
        x = base + drift
        df = pd.DataFrame(x, columns=[f"num_{i}" for i in range(x.shape[1])])
        df["proto"] = np.where(y == 1, rng.choice(["tcp", "udp"], n), rng.choice(["tcp", "udp", "icmp"], n))
        df["service"] = np.where(y == 1, rng.choice(["http", "dns", "ftp"], n), rng.choice(["http", "dns", "-"], n))
        df["state"] = np.where(y == 1, rng.choice(["FIN", "INT", "CON"], n), rng.choice(["FIN", "CON"], n))
        df["label"] = y
        df["attack_cat"] = np.where(y == 1, rng.choice(["Generic", "Exploits", "Fuzzers"], n), "Normal")
        return df

    return make_part(n_train, 0.25), make_part(n_test, 0.30), "synthetic-demo"


def load_dataset():
    train_path = find_file("UNSW_NB15_training-set.csv")
    test_path = find_file("UNSW_NB15_testing-set.csv")
    if train_path and test_path:
        print(f"[INFO] Loaded UNSW-NB15 training file: {train_path}")
        print(f"[INFO] Loaded UNSW-NB15 testing file:  {test_path}")
        return pd.read_csv(train_path), pd.read_csv(test_path), "unsw-nb15"

    print("[WARN] UNSW-NB15 CSV files not found. Using synthetic demo data.")
    print("[WARN] Final report results must be generated with the real dataset.")
    return make_synthetic_ids_data()


def clean_columns(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Dataset must contain a binary 'label' column.")

    drop_cols = ["id"]
    for col in drop_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    attack_train = train_df["attack_cat"].astype(str).to_numpy() if "attack_cat" in train_df.columns else None
    attack_test = test_df["attack_cat"].astype(str).to_numpy() if "attack_cat" in test_df.columns else None

    group_col = None
    for candidate in ["service", "proto", "state"]:
        if candidate in train_df.columns:
            group_col = candidate
            break
    group_train = train_df[group_col].astype(str).to_numpy() if group_col else np.array(["all"] * len(train_df))
    group_test = test_df[group_col].astype(str).to_numpy() if group_col else np.array(["all"] * len(test_df))

    feature_train = train_df.drop(columns=["label", "attack_cat"], errors="ignore")
    feature_test = test_df.drop(columns=["label", "attack_cat"], errors="ignore")
    feature_test = feature_test.reindex(columns=feature_train.columns, fill_value=0)
    return feature_train, feature_test, y_train, y_test, group_train, group_test, attack_train, attack_test


def preprocess_features(x_train_df, x_val_df, x_test_df):
    cat_cols = x_train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_train_df.columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", MinMaxScaler(), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ("scale", MinMaxScaler()),
            ]),
            cat_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    x_train = preprocessor.fit_transform(x_train_df).astype(np.float32)
    x_val = preprocessor.transform(x_val_df).astype(np.float32)
    x_test = preprocessor.transform(x_test_df).astype(np.float32)

    feature_names = []
    feature_names.extend(num_cols)
    feature_names.extend(cat_cols)
    return x_train, x_val, x_test, feature_names


def create_windows(x, y, groups, window_size):
    if window_size <= 1:
        return x, y, groups
    xs, ys, gs = [], [], []
    for i in range(len(x) - window_size + 1):
        xs.append(x[i : i + window_size].reshape(-1))
        ys.append(y[i + window_size - 1])
        gs.append(groups[i + window_size - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=int), np.asarray(gs)


def limit_samples(x, y, groups, max_rows):
    if len(x) <= max_rows:
        return x, y, groups
    rng = np.random.default_rng(CFG.seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    idx.sort()
    return x[idx], y[idx], groups[idx]


def ordered_train_validation_split(x_df, y, groups, val_fraction):
    """Split without shuffling so sliding windows preserve row order.

    UNSW-NB15 rows are flow records, not guaranteed time-series samples, but
    creating windows after a shuffled split is definitely invalid because it
    stitches unrelated rows into fake temporal context. This split keeps row
    order intact for both train and validation windows.
    """
    split_at = int(len(x_df) * (1.0 - val_fraction))
    split_at = max(CFG.window_size, min(split_at, len(x_df) - CFG.window_size))
    x_fit_df = x_df.iloc[:split_at].reset_index(drop=True)
    x_val_df = x_df.iloc[split_at:].reset_index(drop=True)
    y_fit = y[:split_at]
    y_val = y[split_at:]
    group_fit = groups[:split_at]
    group_val = groups[split_at:]
    return x_fit_df, x_val_df, y_fit, y_val, group_fit, group_val


def split_clients_non_iid(x, y, groups, n_clients):
    client_indices = [[] for _ in range(n_clients)]
    unique_groups = pd.Series(groups).value_counts().index.tolist()
    for pos, group in enumerate(unique_groups):
        target_client = pos % n_clients
        group_idx = np.where(groups == group)[0].tolist()
        client_indices[target_client].extend(group_idx)

    empty_clients = [i for i, idx in enumerate(client_indices) if len(idx) == 0]
    if empty_clients:
        fallback = np.array_split(np.arange(len(x)), n_clients)
        client_indices = [part.tolist() for part in fallback]

    clients = []
    for idx in client_indices:
        idx = np.asarray(idx, dtype=int)
        clients.append((x[idx], y[idx]))
    return clients


def add_poisoned_client_for_stress_test(clients):
    if not CFG.enable_poisoned_client_stress_test:
        return clients

    poisoned = []
    target = max(0, min(CFG.poisoned_client_id - 1, len(clients) - 1))
    rng = np.random.default_rng(CFG.seed)

    for idx, (x_client, y_client) in enumerate(clients):
        y_new = y_client.copy()
        if idx == target and len(y_new) > 0:
            n_flip = max(1, int(len(y_new) * CFG.poison_fraction))
            flip_idx = rng.choice(len(y_new), size=n_flip, replace=False)
            y_new[flip_idx] = 1 - y_new[flip_idx]
            print(
                f"[STRESS] Client {idx + 1} label-poisoned: "
                f"{n_flip}/{len(y_new)} labels flipped"
            )
        poisoned.append((x_client, y_new))
    return poisoned


class DeepAutoencoder(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class FederatedIDSClassifier(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


def aggregate_state_dicts(state_dicts, method="trimmed_mean", weights=None):
    result = {}
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-9)

    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].detach().cpu() for sd in state_dicts], dim=0)
        if not torch.is_floating_point(stacked):
            best_idx = int(np.argmax(weights)) if weights is not None else 0
            result[key] = stacked[best_idx]
            continue

        if method == "trust_weighted" and weights is not None:
            w = torch.tensor(weights, dtype=stacked.dtype).view([-1] + [1] * (stacked.ndim - 1))
            result[key] = torch.sum(stacked * w, dim=0)
        elif method == "trimmed_mean" and stacked.shape[0] >= 3:
            sorted_values, _ = torch.sort(stacked, dim=0)
            result[key] = sorted_values[1:-1].mean(dim=0)
        else:
            result[key] = stacked.mean(dim=0)
    return result


def train_one_client(model, x_client, y_client):
    if CFG.use_only_normal_for_ae and np.any(y_client == 0):
        x_fit = x_client[y_client == 0]
    else:
        x_fit = x_client

    if len(x_fit) == 0:
        x_fit = x_client

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_fit, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(x_tensor, batch_size=CFG.batch_size, shuffle=True)

    for _ in range(CFG.local_epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def train_one_classifier_client(model, x_client, y_client):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
    positives = max(float(np.sum(y_client == 1)), 1.0)
    negatives = max(float(np.sum(y_client == 0)), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_tensor = torch.tensor(x_client, dtype=torch.float32)
    y_tensor = torch.tensor(y_client, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)

    for _ in range(CFG.local_epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def classifier_validation_metrics(model, x_val, y_val):
    if x_val is None or y_val is None:
        return {"auc": 0.5, "f1": 0.0, "performance": 0.5}

    if len(x_val) > CFG.trust_validation_limit:
        rng = np.random.default_rng(CFG.seed)
        idx = rng.choice(len(x_val), size=CFG.trust_validation_limit, replace=False)
        x_eval = x_val[idx]
        y_eval = y_val[idx]
    else:
        x_eval = x_val
        y_eval = y_val

    y_eval = np.asarray(y_eval).astype(int).ravel()
    scores = sanitize_scores(classifier_scores(model, x_eval))
    if len(np.unique(y_eval)) < 2:
        return {"auc": 0.5, "f1": 0.0, "performance": 0.5}

    try:
        auc = roc_auc_score(y_eval, scores)
    except ValueError:
        auc = 0.5
    if not np.isfinite(auc):
        auc = 0.5

    tau = best_threshold_from_validation(y_eval, scores)
    preds = (scores >= tau).astype(int)
    try:
        f1 = f1_score(y_eval, preds, zero_division=0)
    except ValueError:
        f1 = 0.0
    if not np.isfinite(f1):
        f1 = 0.0
    performance = 0.70 * auc + 0.30 * f1
    return {
        "auc": float(auc),
        "f1": float(f1),
        "performance": float(np.clip(performance, 0.05, 1.0)),
    }


def state_update_distance(local_state, global_state):
    total = 0.0
    for key in global_state.keys():
        if not torch.is_floating_point(global_state[key]):
            continue
        delta = local_state[key].detach().cpu() - global_state[key].detach().cpu()
        total += float(torch.sum(delta * delta).item())
    return float(np.sqrt(total))


def compute_trust_scores(performance_metrics, update_distances):
    distances = np.nan_to_num(np.asarray(update_distances, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    median_distance = float(np.median(distances) + 1e-9)
    trusts = []
    rows = []

    for idx, metrics in enumerate(performance_metrics):
        perf = float(np.nan_to_num(metrics["performance"], nan=0.5, posinf=1.0, neginf=0.05))
        auc = float(np.nan_to_num(metrics["auc"], nan=0.5, posinf=1.0, neginf=0.0))
        f1 = float(np.nan_to_num(metrics["f1"], nan=0.0, posinf=1.0, neginf=0.0))
        distance_ratio = float(distances[idx] / median_distance)
        excess_ratio = max(0.0, distance_ratio - 1.0)
        update_penalty = 1.0 / (1.0 + CFG.trust_update_penalty_strength * excess_ratio)
        trust = float(np.clip(perf * update_penalty, 0.05, 1.0))
        trusts.append(trust)
        rows.append({
            "client": idx + 1,
            "auc": auc,
            "f1": f1,
            "performance": perf,
            "update_distance": float(distances[idx]),
            "distance_ratio": distance_ratio,
            "update_penalty": update_penalty,
            "trust": trust,
        })

    return np.asarray(trusts, dtype=np.float32), rows


def train_federated_autoencoder(clients, n_features, aggregation="trimmed_mean"):
    global_model = DeepAutoencoder(n_features).to(DEVICE)
    global_state = global_model.state_dict()

    for rnd in range(CFG.federated_rounds):
        updates = []
        for x_client, y_client in clients:
            local_model = DeepAutoencoder(n_features).to(DEVICE)
            local_model.load_state_dict(global_state)
            updates.append(train_one_client(local_model, x_client, y_client))
        global_state = aggregate_state_dicts(updates, method=aggregation)
        global_model.load_state_dict(global_state)
        print(f"[TRAIN] {aggregation} round {rnd + 1}/{CFG.federated_rounds} complete")

    return global_model.eval()


def train_federated_classifier(clients, n_features, aggregation="trimmed_mean", x_val=None, y_val=None):
    global_model = FederatedIDSClassifier(n_features).to(DEVICE)
    global_state = global_model.state_dict()
    trust_history = []

    for rnd in range(CFG.federated_rounds):
        updates = []
        performance_metrics = []
        update_distances = []
        for x_client, y_client in clients:
            local_model = FederatedIDSClassifier(n_features).to(DEVICE)
            local_model.load_state_dict(global_state)
            local_state = train_one_classifier_client(local_model, x_client, y_client)
            updates.append(local_state)
            if aggregation == "trust_weighted":
                performance_metrics.append(classifier_validation_metrics(local_model, x_val, y_val))
                update_distances.append(state_update_distance(local_state, global_state))

        trust_weights = None
        if aggregation == "trust_weighted":
            trust_weights, trust_rows = compute_trust_scores(performance_metrics, update_distances)
            for row in trust_rows:
                row["round"] = rnd + 1
                trust_history.append(row)
            trust_text = ", ".join([
                f"C{row['client']}:AUC={row['auc']:.3f},F1={row['f1']:.3f},"
                f"dev={row['distance_ratio']:.2f}x,trust={row['trust']:.3f}"
                for row in trust_rows
            ])
            print(f"[TRUST] Round {rnd + 1} -> {trust_text}")

        global_state = aggregate_state_dicts(updates, method=aggregation, weights=trust_weights)
        global_model.load_state_dict(global_state)
        print(f"[TRAIN] federated_classifier_{aggregation} {rnd + 1}/{CFG.federated_rounds} complete")

    if aggregation == "trust_weighted" and trust_history:
        trust_df = pd.DataFrame(trust_history)
        trust_path = os.path.join(CFG.output_dir, "trust_history.csv")
        trust_df.to_csv(trust_path, index=False)
        print(f"[TRUST] Saved client trust history to {trust_path}")

    return global_model.eval()


def train_central_autoencoder(x_train, y_train, n_features):
    return train_federated_autoencoder([(x_train, y_train)], n_features, aggregation="fedavg")


def reconstruction_scores(model, x):
    model.eval()
    scores = []
    loader = torch.utils.data.DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            batch_scores = torch.mean((batch - recon) ** 2, dim=1)
            scores.extend(batch_scores.detach().cpu().numpy())
    return np.asarray(scores)


def classifier_scores(model, x):
    model.eval()
    scores = []
    loader = torch.utils.data.DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            probs = torch.sigmoid(model(batch))
            scores.extend(probs.detach().cpu().numpy())
    return sanitize_scores(scores)


def sanitize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return scores
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)


def normalize_by_reference(train_scores, scores):
    lo, hi = np.min(train_scores), np.max(train_scores)
    return np.clip((scores - lo) / (hi - lo + 1e-9), 0, 1)


def best_threshold_from_validation(y_val, val_scores):
    y_val = np.asarray(y_val).astype(int).ravel()
    val_scores = sanitize_scores(val_scores)
    if len(np.unique(y_val)) < 2 or len(np.unique(val_scores)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    if thresholds.size == 0:
        return 0.5

    f1_scores = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    finite = np.isfinite(f1_scores) & np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    tau = thresholds[finite][int(np.argmax(f1_scores[finite]))]
    return float(tau) if np.isfinite(tau) else 0.5


def evaluate_scores(name, y_val, val_scores, y_test, test_scores):
    y_val = np.asarray(y_val).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()
    val_scores = sanitize_scores(val_scores)
    test_scores = sanitize_scores(test_scores)
    tau = best_threshold_from_validation(y_val, val_scores)
    y_pred = (test_scores >= tau).astype(int)
    return {
        "Method": name,
        "Threshold": tau,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, test_scores),
        "PR_AUC": average_precision_score(y_test, test_scores),
    }, y_pred


def orient_scores_with_validation(name, y_val, val_scores, test_scores):
    """Ensure larger score means more attack-like.

    Some unsupervised detectors naturally produce larger values for normal
    samples after preprocessing/reconstruction. Validation labels are used only
    to choose the score direction; the test set remains untouched.
    """
    auc = roc_auc_score(y_val, val_scores)
    if auc < 0.5:
        print(f"[INFO] Flipping {name} score direction because validation AUC={auc:.3f}")
        return 1.0 - val_scores, 1.0 - test_scores
    return val_scores, test_scores


def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.output_dir, filename), dpi=160)
    plt.show()


def plot_roc_curves(y_test, score_map):
    plt.figure(figsize=(7, 5))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.output_dir, "roc_comparison.png"), dpi=160)
    plt.show()


def plot_score_distribution(y_test, scores, threshold):
    plt.figure(figsize=(8, 4.8))
    sns.kdeplot(scores[y_test == 0], label="Normal", fill=True)
    sns.kdeplot(scores[y_test == 1], label="Attack", fill=True)
    plt.axvline(threshold, color="green", linestyle="--", label=f"Validation threshold={threshold:.3f}")
    plt.axvspan(CFG.ambiguity_low, CFG.ambiguity_high, color="gray", alpha=0.18, label="Ambiguity zone")
    plt.title("Adaptive CDAW Score Distribution")
    plt.xlabel("Anomaly score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.output_dir, "cdaw_score_distribution.png"), dpi=160)
    plt.show()


def run_optional_shap(model, x_train, x_test, y_test, feature_names, proposed_scores):
    try:
        import shap
    except ModuleNotFoundError:
        print("[WARN] SHAP not installed. In Colab run: !pip install shap")
        return

    ambiguous = np.where((proposed_scores >= CFG.ambiguity_low) & (proposed_scores <= CFG.ambiguity_high))[0]
    if len(ambiguous) == 0:
        print("[INFO] No sample found inside ambiguity zone for SHAP.")
        return

    idx = int(ambiguous[0])
    background = shap.sample(x_train, min(50, len(x_train)), random_state=CFG.seed)

    def model_score(batch):
        return reconstruction_scores(model, batch)

    print(f"[XAI] Explaining ambiguous test sample index {idx}, true label={y_test[idx]}")
    explainer = shap.KernelExplainer(model_score, background)
    shap_values = explainer.shap_values(x_test[idx : idx + 1], nsamples=100)

    expanded_names = []
    for t in range(CFG.window_size):
        for name in feature_names:
            expanded_names.append(f"{name}_t-{CFG.window_size - 1 - t}")
    if len(expanded_names) != x_test.shape[1]:
        expanded_names = [f"feature_{i}" for i in range(x_test.shape[1])]

    shap.summary_plot(shap_values, x_test[idx : idx + 1], feature_names=expanded_names, show=False)
    plt.title("SHAP Explanation for Ambiguous Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.output_dir, "shap_ambiguous_sample.png"), dpi=160)
    plt.show()


def expanded_feature_names(feature_names):
    names = []
    for t in range(CFG.window_size):
        suffix = CFG.window_size - 1 - t
        for name in feature_names:
            names.append(f"{name}_t-{suffix}")
    return names


def export_incident_event_for_agent(y_test, predictions, trust_scores, cdaw_scores, feature_names):
    """Export one detector output as an event for the Fed-LLM response agent."""
    proposed_name = "Proposed TA-FedX-IDS"
    if proposed_name not in predictions:
        return

    y_pred = predictions[proposed_name]
    candidate_idx = np.where((y_pred == 1) | (cdaw_scores >= CFG.ambiguity_high))[0]
    idx = int(candidate_idx[0]) if len(candidate_idx) else 0

    full_feature_names = expanded_feature_names(feature_names)
    top_features = full_feature_names[:6] if len(full_feature_names) == len(feature_names) * CFG.window_size else feature_names[:6]
    confidence = float(trust_scores[idx])
    cdaw = float(cdaw_scores[idx])
    prediction = "cyber_attack" if int(y_pred[idx]) == 1 else "normal_or_fault_like"

    event = {
        "asset": "CPS Asset / Industrial Control Segment",
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "adaptive_cdaw_score": round(cdaw, 4),
        "client_trust": None,
        "top_shap_features": list(map(str, top_features)),
        "notes": (
            "Generated automatically from TA-FedX-CPS detector output. "
            "Use this JSON as input to src/fed_llm_cps_agent.py."
        ),
        "true_label_available_for_experiment": int(y_test[idx]),
        "sample_index": idx,
    }

    trust_path = os.path.join(CFG.output_dir, "trust_history.csv")
    if os.path.exists(trust_path):
        try:
            trust_df = pd.read_csv(trust_path)
            last_round = trust_df["round"].max()
            event["client_trust"] = round(float(trust_df[trust_df["round"] == last_round]["trust"].mean()), 4)
        except Exception:
            event["client_trust"] = None

    if event["client_trust"] is None:
        event["client_trust"] = 0.75

    out_path = os.path.join(CFG.output_dir, "incident_event_for_agent.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)
    print(f"[AGENT] Exported detector event for Fed-LLM agent: {out_path}")


def main(args=None):
    if args is None:
        args = parse_args()
    configure_from_args(args)

    stress_mode = "ON" if CFG.enable_poisoned_client_stress_test else "OFF"
    print(
        f"[EXPERIMENT] Poisoned-client stress test: {stress_mode} "
        f"(client={CFG.poisoned_client_id}, fraction={CFG.poison_fraction:.2f})"
    )
    print(f"[EXPERIMENT] Output directory: {CFG.output_dir}")

    train_df, test_df, dataset_name = load_dataset()
    print(f"[INFO] Dataset mode: {dataset_name}")
    print(f"[INFO] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    x_train_df, x_test_df, y_train_raw, y_test_raw, group_train_raw, group_test_raw, _, _ = clean_columns(train_df, test_df)

    x_fit_df, x_val_df, y_fit_raw, y_val_raw, group_fit_raw, group_val_raw = ordered_train_validation_split(
        x_train_df,
        y_train_raw,
        group_train_raw,
        CFG.validation_fraction,
    )

    x_fit, x_val, x_test, feature_names = preprocess_features(x_fit_df, x_val_df, x_test_df)

    x_fit_w, y_fit_w, group_fit_w = create_windows(x_fit, y_fit_raw, group_fit_raw, CFG.window_size)
    x_val_w, y_val_w, group_val_w = create_windows(x_val, y_val_raw, group_val_raw, CFG.window_size)
    x_test_w, y_test_w, group_test_w = create_windows(x_test, y_test_raw, group_test_raw, CFG.window_size)

    x_fit_w, y_fit_w, group_fit_w = limit_samples(x_fit_w, y_fit_w, group_fit_w, CFG.max_train_windows)
    x_test_w, y_test_w, group_test_w = limit_samples(x_test_w, y_test_w, group_test_w, CFG.max_test_windows)

    print("[INFO] Validation split preserves row order before windowing; no shuffled fake temporal windows.")
    print(f"[INFO] Windowed train: {x_fit_w.shape}, val: {x_val_w.shape}, test: {x_test_w.shape}")
    print(
        f"[INFO] Train attack ratio: {y_fit_w.mean():.3f}, "
        f"Val attack ratio: {y_val_w.mean():.3f}, Test attack ratio: {y_test_w.mean():.3f}"
    )

    clients = split_clients_non_iid(x_fit_w, y_fit_w, group_fit_w, CFG.n_clients)
    for i, (_, cy) in enumerate(clients, start=1):
        print(f"[CLIENT {i}] samples={len(cy)}, attack_ratio={cy.mean():.3f}")
    classifier_clients = add_poisoned_client_for_stress_test(clients)

    n_features = x_fit_w.shape[1]

    print("\n[MODEL] Training centralized AE baseline")
    central_ae = train_central_autoencoder(x_fit_w, y_fit_w, n_features)

    print("\n[MODEL] Training federated AE with FedAvg")
    fedavg_ae = train_federated_autoencoder(clients, n_features, aggregation="fedavg")

    print("\n[MODEL] Training proposed federated AE with robust trimmed mean")
    robust_ae = train_federated_autoencoder(clients, n_features, aggregation="trimmed_mean")

    print("\n[MODEL] Training supervised federated IDS classifier with FedAvg")
    fedavg_classifier = train_federated_classifier(classifier_clients, n_features, aggregation="fedavg")

    print("\n[MODEL] Training supervised federated IDS classifier with robust trimmed mean")
    robust_classifier = train_federated_classifier(classifier_clients, n_features, aggregation="trimmed_mean")

    print("\n[MODEL] Training proposed trust-aware adaptive FedX-IDS classifier")
    trust_classifier = train_federated_classifier(
        classifier_clients,
        n_features,
        aggregation="trust_weighted",
        x_val=x_val_w,
        y_val=y_val_w,
    )

    train_central_scores = reconstruction_scores(central_ae, x_fit_w)
    train_fedavg_scores = reconstruction_scores(fedavg_ae, x_fit_w)
    train_robust_scores = reconstruction_scores(robust_ae, x_fit_w)

    val_central = normalize_by_reference(train_central_scores, reconstruction_scores(central_ae, x_val_w))
    test_central = normalize_by_reference(train_central_scores, reconstruction_scores(central_ae, x_test_w))

    val_fedavg = normalize_by_reference(train_fedavg_scores, reconstruction_scores(fedavg_ae, x_val_w))
    test_fedavg = normalize_by_reference(train_fedavg_scores, reconstruction_scores(fedavg_ae, x_test_w))

    val_robust = normalize_by_reference(train_robust_scores, reconstruction_scores(robust_ae, x_val_w))
    test_robust = normalize_by_reference(train_robust_scores, reconstruction_scores(robust_ae, x_test_w))

    print("\n[MODEL] Training Isolation Forest baseline")
    if_train = x_fit_w[y_fit_w == 0] if np.any(y_fit_w == 0) else x_fit_w
    iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=CFG.seed, n_jobs=-1)
    iforest.fit(if_train)
    train_if_raw = -iforest.decision_function(if_train)
    val_if = normalize_by_reference(train_if_raw, -iforest.decision_function(x_val_w))
    test_if = normalize_by_reference(train_if_raw, -iforest.decision_function(x_test_w))

    print("\n[MODEL] Training supervised Random Forest reference baseline")
    rf = RandomForestClassifier(n_estimators=200, random_state=CFG.seed, n_jobs=-1, class_weight="balanced")
    rf.fit(x_fit_w, y_fit_w)
    val_rf = rf.predict_proba(x_val_w)[:, 1]
    test_rf = rf.predict_proba(x_test_w)[:, 1]

    val_fedavg_cls = classifier_scores(fedavg_classifier, x_val_w)
    test_fedavg_cls = classifier_scores(fedavg_classifier, x_test_w)

    val_robust_cls = classifier_scores(robust_classifier, x_val_w)
    test_robust_cls = classifier_scores(robust_classifier, x_test_w)

    val_trust_cls = classifier_scores(trust_classifier, x_val_w)
    test_trust_cls = classifier_scores(trust_classifier, x_test_w)

    val_disagreement = np.abs(val_robust - val_if)
    test_disagreement = np.abs(test_robust - test_if)
    val_adaptive_lambda = CFG.lambda_cdaw + CFG.adaptive_cdaw_beta * val_disagreement
    test_adaptive_lambda = CFG.lambda_cdaw + CFG.adaptive_cdaw_beta * test_disagreement
    val_cdaw = (val_robust + val_if) / 2.0 + val_adaptive_lambda * val_disagreement
    test_cdaw = (test_robust + test_if) / 2.0 + test_adaptive_lambda * test_disagreement
    val_cdaw = np.clip(val_cdaw, 0, 1)
    test_cdaw = np.clip(test_cdaw, 0, 1)

    val_central, test_central = orient_scores_with_validation("Central AE", y_val_w, val_central, test_central)
    val_fedavg, test_fedavg = orient_scores_with_validation("FedAvg AE", y_val_w, val_fedavg, test_fedavg)
    val_robust, test_robust = orient_scores_with_validation("Robust FedAE", y_val_w, val_robust, test_robust)
    val_if, test_if = orient_scores_with_validation("Isolation Forest", y_val_w, val_if, test_if)
    val_cdaw, test_cdaw = orient_scores_with_validation("Proposed CDAW", y_val_w, val_cdaw, test_cdaw)

    evaluations = []
    predictions = {}
    score_map = {
        "Central AE": test_central,
        "FedAvg AE": test_fedavg,
        "Robust FedAE": test_robust,
        "Isolation Forest": test_if,
        "Random Forest": test_rf,
        "FedAvg IDS": test_fedavg_cls,
        "Robust IDS": test_robust_cls,
        "Proposed TA-FedX-IDS": test_trust_cls,
        "Adaptive CDAW": test_cdaw,
    }
    val_map = {
        "Central AE": val_central,
        "FedAvg AE": val_fedavg,
        "Robust FedAE": val_robust,
        "Isolation Forest": val_if,
        "Random Forest": val_rf,
        "FedAvg IDS": val_fedavg_cls,
        "Robust IDS": val_robust_cls,
        "Proposed TA-FedX-IDS": val_trust_cls,
        "Adaptive CDAW": val_cdaw,
    }

    for name in score_map:
        row, pred = evaluate_scores(name, y_val_w, val_map[name], y_test_w, score_map[name])
        evaluations.append(row)
        predictions[name] = pred
        val_unique = len(np.unique(sanitize_scores(val_map[name])))
        print(f"[THRESHOLD] {name}: tau={row['Threshold']:.4f}, val_unique_scores={val_unique}")

    results = pd.DataFrame(evaluations)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    results[metric_cols] = results[metric_cols].round(4)
    results["Threshold"] = results["Threshold"].round(4)
    print("\n================ FINAL METRICS ================")
    print(results.to_string(index=False))
    print("===============================================")
    results.to_csv(os.path.join(CFG.output_dir, "metrics.csv"), index=False)

    ablation = results[results["Method"].isin([
        "FedAvg AE",
        "Robust FedAE",
        "Isolation Forest",
        "Adaptive CDAW",
        "FedAvg IDS",
        "Robust IDS",
        "Proposed TA-FedX-IDS",
    ])].copy()
    ablation.to_csv(os.path.join(CFG.output_dir, "ablation_summary.csv"), index=False)

    proposed_threshold = float(results.loc[results["Method"] == "Adaptive CDAW", "Threshold"].iloc[0])
    plot_confusion(y_test_w, predictions["Adaptive CDAW"], "Adaptive CDAW Confusion Matrix", "confusion_matrix_adaptive_cdaw.png")
    plot_confusion(
        y_test_w,
        predictions["Proposed TA-FedX-IDS"],
        "Proposed TA-FedX-IDS Confusion Matrix",
        "confusion_matrix_ta_fedx_ids.png",
    )
    plot_roc_curves(y_test_w, score_map)
    plot_score_distribution(y_test_w, test_cdaw, proposed_threshold)

    run_optional_shap(robust_ae, x_fit_w, x_test_w, y_test_w, feature_names, test_cdaw)
    export_incident_event_for_agent(y_test_w, predictions, test_trust_cls, test_cdaw, feature_names)

    print(f"\n[DONE] Outputs saved in: {CFG.output_dir}")
    print("[NOTE] For thesis results, use real UNSW-NB15 CSVs, not the synthetic demo mode.")


if __name__ == "__main__":
    main()
