"""V47 · 01 — Weighted-Cramer SWA-SE-TabularResNet con Recorte Robusto de Outliers [-6, +6] en GPU ROCm.

Integra tres mecanismos teóricos de optimización estadística y transporte ordinal:
1. Normalización Robusta con Recorte de Outliers (Winsorización en Z in-fold [-6.0, +6.0]):
   Neutraliza los 109 predictores de matriz_279 que presentan |z| > 10 (con picos de hasta 120.94 sigma),
   eliminando la distorsión de varianza en BatchNorm1d y picos de gradiente, preservando intactos
   los ceros discretos (ausencia de mención/hashtag) y la monotonicidad del 99.9% de los datos.
2. Distancia de Cramér-von Mises Ponderada Asimétricamente (Transporte Óptimo L2 en 1D):
   L_WCramer(p, y) = sum_{k=0}^{K-2} gamma_k ( F_p(k) - I(y <= k) )^2
   con gamma = [1.2, 1.5, 1.3, 1.1, 1.0], concentrando el gradiente suave y continuo en las
   fronteras críticas más confusas (13-17 <-> 18-24, 18-24 <-> 25-34, 25-34 <-> 35-49).
3. Regularización Entrópica (Label Smoothing 0.10) + Cosine Annealing + SWA (Stochastic Weight Averaging):
   Promedia 10 checkpoints en la cuenca de mínimos planos, asegurando máxima robustez frente
   al 30% de datos retenidos en el leaderboard privado (defensa ante shakeup).

Validación cruzada canónica 5-fold StratifiedGroupKFold agrupada por dup_clusters (cero fuga).
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

RESULTADO = cfg.EXPERIMENTS / "v47_01_weighted_cramer_clipped_tabularnet_5folds.json"
OOF_FILE = cfg.FEATURES / "v47_oof_weighted_cramer_clipped_tabularnet.npy"
TEST_PROB_FILE = cfg.FEATURES / "v47_test_weighted_cramer_clipped_tabularnet.npy"
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


class WeightedCramerLoss(nn.Module):
    """Entropía Cruzada con Suavizado 0.10 combinada con Distancia de Cramér (CDF L2) ponderada en fronteras críticas."""

    def __init__(self, num_classes: int = 6, lambda_cramer: float = 0.50, smoothing: float = 0.10, gamma: list[float] | None = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes
        self.lambda_cramer = lambda_cramer
        if gamma is None:
            gamma = [1.2, 1.5, 1.3, 1.1, 1.0]
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        # CDF predicha acumulada F_p(k)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]
        # CDF escalón objetivo F_y(k) = I(y <= k)
        k_indices = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        cdf_target = (targets.unsqueeze(1) <= k_indices).float()
        # Pérdida de Cramér cuadrática (L2) ponderada por frontera
        diff_sq = (cdf_pred - cdf_target) ** 2
        weighted_sq = diff_sq * self.gamma.unsqueeze(0)
        loss_cramer = torch.mean(torch.sum(weighted_sq, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V47] Iniciando entrenamiento 5-fold de Weighted-Cramer SWA-SE-TabularResNet con Recorte Robusto...", flush=True)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V47] Dispositivo detectado: {dispositivo}", flush=True)

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

    # Hiperparámetros óptimos V47
    batch_size = 256
    epochs = 16
    swa_start = 6
    lr = 1e-3
    lr_min = 1e-4
    weight_decay = 1e-4
    smoothing = 0.10
    lambda_cramer = 0.50
    gamma = [1.2, 1.5, 1.3, 1.1, 1.0]
    clip_val = 6.0

    fold_scores_standalone = []
    v30_fold_scores = []

    for fold in range(5):
        t0 = time.time()
        tr_mask = (fold_idx != fold)
        val_mask = (fold_idx == fold)

        # Preprocesamiento estrictamente in-fold con recorte robusto [-6.0, +6.0]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw[tr_mask])
        X_val = scaler.transform(X_train_raw[val_mask])
        X_te = scaler.transform(X_test_raw)

        X_tr = np.clip(X_tr, -clip_val, clip_val)
        X_val = np.clip(X_val, -clip_val, clip_val)
        X_te_scaled = np.clip(X_te, -clip_val, clip_val)

        y_tr = y_true[tr_mask]
        y_val = y_true[val_mask]

        torch.manual_seed(cfg.SEED + fold * 100)
        np.random.seed(cfg.SEED + fold * 100)

        base_model = SETabularResNet(in_features=n_features, hidden=256, num_classes=n_classes, drop=0.2, reduction=4).to(dispositivo)
        swa_model = AveragedModel(base_model)
        criterion = WeightedCramerLoss(num_classes=n_classes, lambda_cramer=lambda_cramer, smoothing=smoothing, gamma=gamma).to(dispositivo)
        optimizer = optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

        ds_train = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            base_model.train()
            for bx, by in loader_train:
                bx, by = bx.to(dispositivo), by.to(dispositivo)
                optimizer.zero_grad()
                logits = base_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if ep >= swa_start:
                swa_model.update_parameters(base_model)

        # Actualizar BatchNorm del modelo SWA con estadísticas de train
        loader_eval_tr = DataLoader(ds_train, batch_size=512, shuffle=False)
        update_bn(loader_eval_tr, swa_model, device=dispositivo)

        swa_model.eval()

        # Inferencia OOF para validación
        ds_val = TensorDataset(torch.from_numpy(X_val).float())
        loader_val = DataLoader(ds_val, batch_size=512, shuffle=False)
        val_preds_list = []
        with torch.no_grad():
            for (bx,) in loader_val:
                bx = bx.to(dispositivo)
                probs = torch.softmax(swa_model(bx), dim=1)
                val_preds_list.append(probs.cpu().numpy())
        fold_oof_probs = np.vstack(val_preds_list)
        oof_probs[val_mask] = fold_oof_probs

        # Inferencia Test acumulada
        ds_test = TensorDataset(torch.from_numpy(X_te_scaled).float())
        loader_test = DataLoader(ds_test, batch_size=512, shuffle=False)
        test_preds_list = []
        with torch.no_grad():
            for (bx,) in loader_test:
                bx = bx.to(dispositivo)
                probs = torch.softmax(swa_model(bx), dim=1)
                test_preds_list.append(probs.cpu().numpy())
        fold_test_probs = np.vstack(test_preds_list)
        test_probs += fold_test_probs / 5.0

        # Evaluación fold
        f1_fold_alone = f1_score(y_val, np.argmax(fold_oof_probs, axis=1), average="macro")
        f1_fold_v30 = f1_score(y_val, np.argmax(v30_probs[val_mask], axis=1), average="macro")
        fold_scores_standalone.append(float(f1_fold_alone))
        v30_fold_scores.append(float(f1_fold_v30))

        print(f"  [Fold {fold}] Standalone Macro-F1: {f1_fold_alone:.6f} | V30 Baseline: {f1_fold_v30:.6f} ({time.time() - t0:.1f}s)", flush=True)

    # Evaluación global standalone
    f1_global_alone = f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro")
    f1_v30_baseline = f1_score(y_true, np.argmax(v30_probs, axis=1), average="macro")

    print("\n[V47] Resultados Globales Standalone:")
    print(f"  - Macro-F1 Global Standalone: {f1_global_alone:.7f}")
    print(f"  - Macro-F1 V30 Baseline:     {f1_v30_baseline:.7f}")
    print(f"  - Delta Standalone vs V30:    {f1_global_alone - f1_v30_baseline:+.7f}")
    print(f"  - Media Folds:                {np.mean(fold_scores_standalone):.6f} +/- {np.std(fold_scores_standalone):.6f}")

    # Guardar OOF y probabilidades de Test
    np.save(OOF_FILE, oof_probs)
    np.save(TEST_PROB_FILE, test_probs)
    print(f"\n[V47] OOF guardado en: {OOF_FILE}")
    print(f"[V47] Test probabilities guardadas en: {TEST_PROB_FILE}")

    # Guardar métricas JSON
    payload = {
        "fase": "V47-01",
        "fecha": "2026-10-16",
        "duracion_seg": round(time.time() - inicio, 2),
        "f1_v30_baseline": float(f1_v30_baseline),
        "f1_weighted_cramer_clipped_standalone": float(f1_global_alone),
        "fold_scores_standalone": fold_scores_standalone,
        "v30_fold_scores": v30_fold_scores,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs": epochs,
            "swa_start": swa_start,
            "lr": lr,
            "lr_min": lr_min,
            "smoothing": smoothing,
            "lambda_cramer": lambda_cramer,
            "gamma_boundaries": gamma,
            "outlier_clip": clip_val,
            "weight_decay": weight_decay,
        },
        "oof_shape": list(oof_probs.shape),
        "test_shape": list(test_probs.shape),
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[V47] Resultados guardados en: {RESULTADO}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
