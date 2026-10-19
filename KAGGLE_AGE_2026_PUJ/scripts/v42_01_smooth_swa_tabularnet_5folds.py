"""V42 · 01 — Smooth-SWA-SE-TabularResNet (Cosine Annealing + Label Smoothing 0.10 + Pérdida Ordinal + SWA en GPU).

Combina cuatro innovaciones complementarias sobre matriz_279 en GPU ROCm:
1. Compuertas de atención condicional Squeeze-and-Excitation (SE, reduction=4).
2. Regularización de distancia ordinal esperada (lambda=0.2).
3. Suavizado de etiquetas (label smoothing=0.10) para evitar colapso de margen en fronteras adyacentes.
4. Stochastic Weight Averaging (SWA, 10 checkpoints) con enfriamiento CosineAnnealingLR (1e-3 -> 1e-4) y update_bn.

Validación cruzada canónica 5-fold StratifiedGroupKFold agrupada por dup_clusters.
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

RESULTADO = cfg.EXPERIMENTS / "v42_01_smooth_swa_tabularnet_5folds.json"
OOF_FILE = cfg.FEATURES / "v42_oof_smooth_swa_tabularnet.npy"
TEST_PROB_FILE = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
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


class SmoothOrdinalDistanceLoss(nn.Module):
    """Entropía Cruzada con Suavizado de Etiquetas combinada con regularización de distancia ordinal."""

    def __init__(self, num_classes: int = 6, lambda_dist: float = 0.2, smoothing: float = 0.10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
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
    print("[V42] Iniciando entrenamiento 5-fold de Smooth-SWA-SE-TabularResNet...", flush=True)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V42] Dispositivo detectado: {dispositivo}", flush=True)

    X_train_raw = matriz_279()
    X_test_raw = np.load(MATRIZ_TEST).astype(np.float32)

    y_true = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "y_true.npy")
    fold_idx = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "fold_idx.npy")
    v30_probs = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "v30_prob.npy")

    n_samples, n_features = X_train_raw.shape
    n_test = X_test_raw.shape[0]
    n_classes = len(cfg.CLASSES)

    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    pesos_plug_in = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
    pred_v30_plugin = np.argmax(v30_probs * pesos_plug_in, axis=1)
    f1_v30_baseline = float(f1_score(y_true, pred_v30_plugin, average="macro"))
    print(f"[V42] Macro-F1 V30 (Baseline): {f1_v30_baseline:.7f}", flush=True)

    # Hiperparámetros optimizados V42
    batch_size = 256
    epochs = 16
    swa_start = 6
    lr = 1e-3
    lr_min = 1e-4
    weight_decay = 1e-4
    smoothing = 0.10
    lambda_dist = 0.20

    fold_scores_standalone = []
    v30_fold_scores = []

    for fold in range(5):
        t0 = time.time()
        tr_mask = (fold_idx != fold)
        val_mask = (fold_idx == fold)

        # Preprocesamiento estrictamente in-fold (zero leakage)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw[tr_mask])
        X_val = scaler.transform(X_train_raw[val_mask])
        X_te_scaled = scaler.transform(X_test_raw)

        y_tr = y_true[tr_mask]
        y_val = y_true[val_mask]

        torch.manual_seed(cfg.SEED + fold * 100)
        np.random.seed(cfg.SEED + fold * 100)

        base_model = SETabularResNet(in_features=n_features, hidden=256, num_classes=n_classes, drop=0.2, reduction=4).to(dispositivo)
        swa_model = AveragedModel(base_model)
        criterion = SmoothOrdinalDistanceLoss(num_classes=n_classes, lambda_dist=lambda_dist, smoothing=smoothing).to(dispositivo)
        optimizer = optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

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

            scheduler.step()

            # Acumular pesos en la cuenca plana
            if ep >= swa_start:
                swa_model.update_parameters(base_model)

        # Actualizar BatchNorm honestamente in-fold para el modelo SWA
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

        pred_fold_standalone = np.argmax(val_p * pesos_plug_in, axis=1)
        score_fold_standalone = float(f1_score(y_val, pred_fold_standalone, average="macro"))
        fold_scores_standalone.append(score_fold_standalone)

        score_v30_fold = float(f1_score(y_val, pred_v30_plugin[val_mask], average="macro"))
        v30_fold_scores.append(score_v30_fold)

        print(
            f"  Fold {fold}: Standalone SWA={score_fold_standalone:.6f} | V30={score_v30_fold:.6f} | "
            f"Tiempo: {time.time() - t0:.1f}s",
            flush=True,
        )

    # Evaluación Global OOF Standalone
    pred_global_standalone = np.argmax(oof_probs * pesos_plug_in, axis=1)
    f1_global_standalone = float(f1_score(y_true, pred_global_standalone, average="macro"))

    print("\n" + "=" * 80)
    print(f"[V42] Macro-F1 OOF Standalone Smooth-SWA-SE-TabNet : {f1_global_standalone:.7f}")
    print(f"[V42] Macro-F1 V30 Baseline                       : {f1_v30_baseline:.7f}")
    print(f"[V42] Delta Standalone vs V30                     : {f1_global_standalone - f1_v30_baseline:+.7f}")
    print("=" * 80 + "\n", flush=True)

    np.save(OOF_FILE, oof_probs)
    np.save(TEST_PROB_FILE, test_probs)
    print(f"[V42] Guardado OOF en {OOF_FILE} y Test Probs en {TEST_PROB_FILE}", flush=True)

    duracion = round(time.time() - inicio, 2)
    meta = {
        "fase": "V42-01",
        "fecha": "2026-10-16",
        "duracion_seg": duracion,
        "f1_v30_baseline": f1_v30_baseline,
        "f1_smooth_swa_standalone": f1_global_standalone,
        "fold_scores_standalone": fold_scores_standalone,
        "v30_fold_scores": v30_fold_scores,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs": epochs,
            "swa_start": swa_start,
            "lr": lr,
            "lr_min": lr_min,
            "smoothing": smoothing,
            "lambda_dist": lambda_dist,
            "weight_decay": weight_decay,
        },
        "oof_shape": list(oof_probs.shape),
        "test_shape": list(test_probs.shape),
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    RESULTADO.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[V42] Metadatos guardados en {RESULTADO}. Tiempo total: {duracion}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
