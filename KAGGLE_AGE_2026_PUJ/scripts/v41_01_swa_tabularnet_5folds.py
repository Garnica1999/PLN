"""V41 · 01 — SWA-SE-TabularResNet (Stochastic Weight Averaging + SE + Pérdida Ordinal en GPU).

Incorpora promediado estocástico de pesos (SWA), compuertas Squeeze-and-Excitation
y pérdida de distancia ordinal esperada (lambda=0.2) sobre matriz_279 en GPU ROCm,
con validación cruzada canónica 5-fold StratifiedGroupKFold agrupada por dup_clusters.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import matriz_279

from src import config as cfg
from src.verificacion_numerica import autotest_numerico

RESULTADO = cfg.EXPERIMENTS / "v41_01_swa_tabularnet_5folds.json"
OOF_FILE = cfg.FEATURES / "v41_oof_swa_tabularnet.npy"
TEST_PROB_FILE = cfg.FEATURES / "v41_test_swa_tabularnet.npy"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"


class SETabularResNet(nn.Module):
    """Red residual profunda para datos tabulares con compuertas Squeeze-and-Excitation."""

    def __init__(self, in_features: int = 279, hidden: int = 256, num_classes: int = 6, drop: float = 0.2, reduction: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.act0 = nn.SiLU()

        # Bloque Residual 1 con Squeeze-and-Excitation
        self.fc1 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.se1_fc1 = nn.Linear(hidden, hidden // reduction)
        self.se1_act = nn.SiLU()
        self.se1_fc2 = nn.Linear(hidden // reduction, hidden)
        self.se1_sig = nn.Sigmoid()

        # Bloque Residual 2 con Squeeze-and-Excitation
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(drop)
        self.se2_fc1 = nn.Linear(hidden, hidden // reduction)
        self.se2_act = nn.SiLU()
        self.se2_fc2 = nn.Linear(hidden // reduction, hidden)
        self.se2_sig = nn.Sigmoid()

        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.act0(self.bn0(self.in_proj(x)))

        # Bloque 1
        res1 = self.drop1(self.act1(self.bn1(self.fc1(h0))))
        se1 = self.se1_sig(self.se1_fc2(self.se1_act(self.se1_fc1(res1))))
        h1 = h0 + res1 * se1

        # Bloque 2
        res2 = self.drop2(self.act2(self.bn2(self.fc2(h1))))
        se2 = self.se2_sig(self.se2_fc2(self.se2_act(self.se2_fc1(res2))))
        h2 = h1 + res2 * se2

        return self.head(h2)


class OrdinalDistanceLoss(nn.Module):
    """Entropía Cruzada combinada con regularización de distancia ordinal esperada."""

    def __init__(self, num_classes: int = 6, lambda_dist: float = 0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.lambda_dist = lambda_dist
        c = torch.arange(num_classes).float()
        self.register_buffer("dist_matrix", torch.abs(c.unsqueeze(0) - c.unsqueeze(1)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dists = self.dist_matrix[targets]
        loss_dist = torch.sum(probs * dists, dim=1).mean()
        return loss_ce + self.lambda_dist * loss_dist


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V41] Iniciando SWA-SE-TabularResNet Ordinal 5-fold CV en GPU ROCm...", flush=True)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V41] Dispositivo de ejecución: {dispositivo}", flush=True)

    X = matriz_279()
    y_true = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "y_true.npy")
    fold_idx = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "fold_idx.npy")
    v30_prob = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "v30_prob.npy")

    w_plugin = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
    pred_v30 = np.argmax(v30_prob * w_plugin, axis=1)
    f1_v30 = f1_score(y_true, pred_v30, average="macro")
    print(f"[V30] Macro-F1 Campeón base con plug-in: {f1_v30:.7f}", flush=True)

    X_test_raw = np.load(MATRIZ_TEST).astype(np.float32)
    assert X_test_raw.shape == (112892, 279), f"Forma inesperada de test: {X_test_raw.shape}"

    oof_probs = np.zeros((len(y_true), 6), dtype=np.float32)
    test_probs = np.zeros((len(X_test_raw), 6), dtype=np.float32)
    fold_scores = []
    v30_fold_scores = []

    epochs = 16
    swa_start = 8
    batch_size = 256
    lr = 1e-3
    weight_decay = 1e-4

    for fold in range(5):
        t0 = time.time()
        tr_mask = (fold_idx != fold)
        val_mask = (fold_idx == fold)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_mask])
        X_val = scaler.transform(X[val_mask])
        X_te_scaled = scaler.transform(X_test_raw)

        y_tr = y_true[tr_mask]
        y_val = y_true[val_mask]

        torch.manual_seed(cfg.SEED + fold * 100)
        np.random.seed(cfg.SEED + fold * 100)

        base_model = SETabularResNet(in_features=279, hidden=256, num_classes=6, drop=0.2, reduction=4).to(dispositivo)
        swa_model = AveragedModel(base_model)
        criterion = OrdinalDistanceLoss(num_classes=6, lambda_dist=0.2).to(dispositivo)
        optimizer = optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)

        ds_train = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            base_model.train()
            for bx, by in loader_train:
                bx, by = bx.to(dispositivo), by.to(dispositivo)
                optimizer.zero_grad()
                out = base_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            if ep >= swa_start:
                swa_model.update_parameters(base_model)

        # Actualizar BatchNorm para el modelo SWA
        update_bn(loader_train, swa_model, device=dispositivo)

        swa_model.eval()
        with torch.no_grad():
            val_t = torch.from_numpy(X_val).float().to(dispositivo)
            val_out = swa_model(val_t)
            val_p = torch.softmax(val_out, dim=1).cpu().numpy()
            oof_probs[val_mask] = val_p

            te_t = torch.from_numpy(X_te_scaled).float().to(dispositivo)
            te_out = swa_model(te_t)
            te_p = torch.softmax(te_out, dim=1).cpu().numpy()
            test_probs += te_p / 5.0

        s_v30 = f1_score(y_val, pred_v30[val_mask], average="macro")
        s_swa = f1_score(y_val, np.argmax(val_p * w_plugin, axis=1), average="macro")
        v30_fold_scores.append(float(s_v30))
        fold_scores.append(float(s_swa))
        dt = time.time() - t0
        print(f"  [Fold {fold}] V30={s_v30:.6f} | SWA-SE-TabNet Standalone={s_swa:.6f} | Tiempo={dt:.1f}s", flush=True)

    np.save(OOF_FILE, oof_probs)
    np.save(TEST_PROB_FILE, test_probs)
    print(f"[V41] OOF guardado en {OOF_FILE} y test en {TEST_PROB_FILE}", flush=True)

    pred_swa_standalone = np.argmax(oof_probs * w_plugin, axis=1)
    f1_swa_standalone = f1_score(y_true, pred_swa_standalone, average="macro")
    print(f"[V41] SWA-SE-TabularResNet Standalone Macro-F1: {f1_swa_standalone:.7f}", flush=True)

    duracion = time.time() - inicio
    resultado = {
        "fase": "V41-01",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(duracion, 2),
        "f1_v30_baseline": float(f1_v30),
        "f1_swa_standalone": float(f1_swa_standalone),
        "fold_scores_standalone": fold_scores,
        "v30_fold_scores": v30_fold_scores,
        "oof_shape": list(oof_probs.shape),
        "test_shape": list(test_probs.shape),
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)

    print(f"[V41] Experimento registrado en {RESULTADO} ({duracion:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
