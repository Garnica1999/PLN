"""V51 · 01 — Consistent Smoothed Cramér-von Mises Loss en SETabularResNet 5 Folds.

Motivación Matemática:
En V43, la pérdida combinaba CrossEntropy con Label Smoothing (epsilon = 0.10) y la
pérdida cuadrática de Cramér (transporte óptimo ordinal). Sin embargo, la función
objetivo acumulada en Cramér utilizaba la función escalón no suavizada F_y(k) = I(y <= k).
Esto generaba una contradicción de gradiente: la entropía cruzada empujaba los logits
hacia una distribución suave con entropía acotada, mientras que el transporte de Cramér
tiraba hacia una distribución delta discreta.

En este experimento se unifica rigurosamente el ground truth acumulado:
    F_smooth(k) = (1 - eps) * I(y <= k) + eps * (k + 1) / K
de modo que ambas pérdidas operan bajo la misma distribución subyacente.

Protocolo:
- 5 folds canónicos StratifiedGroupKFold agrupados por dup_clusters.parquet (cero fuga).
- StandardScaler ajustado estrictamente in-fold.
- SETabularResNet (in=279, hidden=256, drop=0.20, reduction=4).
- AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR (eta_min=1e-4), 16 épocas, SWA desde época 6.
- Actualización honesta de BatchNorm in-fold para el modelo SWA.
- Guardado de OOF y Test probabilities en features/.
- Evaluación en ensamble bayesiano Product-of-Experts con prueba pareada bootstrap (B=1000).
"""

import json
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

RAIZ = pathlib.Path.cwd()
while not (RAIZ / 'experiments.csv').exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent

import sys
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from src import config as cfg
from src.verificacion_numerica import autotest_numerico
from scripts.v19_b1_boosting import matriz_279
from scripts.v43_01_cramer_swa_tabularnet_5folds import SETabularResNet

MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"


class ConsistentSmoothedCramerLoss(nn.Module):
    """Entropía Cruzada con Suavizado de Etiquetas y Cramér-von Mises sobre CDF suavizada consistente."""

    def __init__(self, num_classes: int = 6, lambda_cramer: float = 0.50, smoothing: float = 0.10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes
        self.lambda_cramer = lambda_cramer
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        # CDF predicha acumulada F_p(k)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]
        # CDF escalón objetivo suavizada consistentemente
        k_indices = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        cdf_step = (targets.unsqueeze(1) <= k_indices).float()
        cdf_target = (1.0 - self.smoothing) * cdf_step + self.smoothing * (k_indices + 1.0) / self.num_classes
        # Pérdida cuadrática de Cramér (transporte óptimo ordinal L2 en 1D)
        loss_cramer = torch.mean(torch.sum((cdf_pred - cdf_target) ** 2, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=" * 80)
    print("[V51] INICIANDO ENTRENAMIENTO 5-FOLD DE CONSISTENT SMOOTHED CRAMER SWA TABULARNET")
    print("=" * 80, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V51] Dispositivo de aceleración: {device}", flush=True)

    X_train_raw = matriz_279()
    X_test_raw = np.load(MATRIZ_TEST).astype(np.float32)

    y_true = np.load(RAIZ / "artifacts/p33_scores/todos/y_true.npy")
    fold_idx = np.load(RAIZ / "artifacts/p33_scores/todos/fold_idx.npy")
    w_plugin = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

    n_samples, n_features = X_train_raw.shape
    n_test = X_test_raw.shape[0]
    n_classes = len(cfg.CLASSES)

    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    # Hiperparámetros de entrenamiento
    batch_size = 256
    epochs = 16
    swa_start = 6
    lr = 1e-3
    lr_min = 1e-4
    weight_decay = 1e-4
    smoothing = 0.10
    lambda_cramer = 0.50

    fold_scores_standalone = []

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

        base_model = SETabularResNet(in_features=n_features, hidden=256, num_classes=n_classes, drop=0.2, reduction=4).to(device)
        swa_model = AveragedModel(base_model)
        criterion = ConsistentSmoothedCramerLoss(num_classes=n_classes, lambda_cramer=lambda_cramer, smoothing=smoothing).to(device)
        optimizer = optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

        ds_train = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            base_model.train()
            for bx, by in loader_train:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = base_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if ep >= swa_start:
                swa_model.update_parameters(base_model)

        # Actualizar BatchNorm honestamente in-fold para el modelo SWA
        update_bn(loader_train, swa_model, device=device)

        swa_model.eval()
        with torch.no_grad():
            val_t = torch.from_numpy(X_val).float().to(device)
            val_out = swa_model(val_t)
            val_p = torch.softmax(val_out, dim=1).cpu().numpy()
            oof_probs[val_mask] = val_p

            te_t = torch.from_numpy(X_te_scaled).float().to(device)
            te_out = swa_model(te_t)
            te_p = torch.softmax(te_out, dim=1).cpu().numpy()
            test_probs += te_p / 5.0

        pred_fold_standalone = np.argmax(val_p * w_plugin, axis=1)
        score_fold_standalone = float(f1_score(y_val, pred_fold_standalone, average="macro"))
        fold_scores_standalone.append(score_fold_standalone)

        print(f"  Fold {fold}: Standalone Consistent-Cramer-SWA = {score_fold_standalone:.6f} | Tiempo: {time.time() - t0:.1f}s", flush=True)

    # Evaluación Global OOF Standalone
    pred_global_standalone = np.argmax(oof_probs * w_plugin, axis=1)
    f1_global_standalone = float(f1_score(y_true, pred_global_standalone, average="macro"))

    # Cargar modelos previos para comparación
    v43_oof = np.load(RAIZ / "features/v43_oof_cramer_swa_tabularnet.npy")
    pred_v43_standalone = np.argmax(v43_oof * w_plugin, axis=1)
    f1_v43_standalone = float(f1_score(y_true, pred_v43_standalone, average="macro"))

    print("\n" + "=" * 80)
    print(f"[V51] Macro-F1 OOF Standalone Consistent-Cramer-SWA: {f1_global_standalone:.7f}")
    print(f"[V51] Macro-F1 OOF Standalone V43 Cramer-SWA        : {f1_v43_standalone:.7f}")
    print(f"[V51] Delta Standalone vs V43                       : {f1_global_standalone - f1_v43_standalone:+.7f}")
    print("=" * 80 + "\n", flush=True)

    # Guardar features
    np.save(RAIZ / "features/v51_oof_consistent_cramer_tabularnet.npy", oof_probs)
    np.save(RAIZ / "features/v51_test_consistent_cramer_tabularnet.npy", test_probs)
    print("Artefactos OOF y Test guardados en features/", flush=True)

    # Integración en Ensamble Bayesian Product-of-Experts V48
    v30 = np.load(RAIZ / "features/age_plugin_segunda_particion_oof.npy")
    v42 = np.load(RAIZ / "features/v42_oof_smooth_swa_tabularnet.npy")
    v40 = np.load(RAIZ / "features/v40_oof_se_tabularnet.npy")
    v39_ord = np.load(RAIZ / "features/v39_oof_tabnet_ordinal.npy")
    v39_hgb = np.load(RAIZ / "features/v39_oof_hgb.npy")

    weights_geom = [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549]
    eps = 1e-12

    # Baseline V48 Champion
    models_v48 = [v30, v43_oof, v42, v40, v39_ord, v39_hgb]
    log_v48 = [np.log(np.clip(m, eps, 1.0)) for m in models_v48]
    log_geom_v48 = sum(w * lm for w, lm in zip(weights_geom, log_v48))
    p_geom_v48 = np.exp(log_geom_v48 - np.max(log_geom_v48, axis=1, keepdims=True))
    p_geom_v48 /= p_geom_v48.sum(axis=1, keepdims=True)
    pred_v48 = np.argmax(p_geom_v48 * w_plugin, axis=1)
    f1_v48_global = float(f1_score(y_true, pred_v48, average="macro"))

    # V51 Replacement Blend
    models_v51 = [v30, oof_probs, v42, v40, v39_ord, v39_hgb]
    log_v51 = [np.log(np.clip(m, eps, 1.0)) for m in models_v51]
    log_geom_v51 = sum(w * lm for w, lm in zip(weights_geom, log_v51))
    p_geom_v51 = np.exp(log_geom_v51 - np.max(log_geom_v51, axis=1, keepdims=True))
    p_geom_v51 /= p_geom_v51.sum(axis=1, keepdims=True)
    pred_v51 = np.argmax(p_geom_v51 * w_plugin, axis=1)
    f1_v51_global = float(f1_score(y_true, pred_v51, average="macro"))

    # Desglose por Folds
    f1_v48_folds = [float(f1_score(y_true[fold_idx == f], pred_v48[fold_idx == f], average="macro")) for f in range(5)]
    f1_v51_folds = [float(f1_score(y_true[fold_idx == f], pred_v51[fold_idx == f], average="macro")) for f in range(5)]
    deltas = [v51 - v48 for v51, v48 in zip(f1_v51_folds, f1_v48_folds)]
    pos_folds = sum(d > 0 for d in deltas)

    print("=" * 80)
    print("EVALUACIÓN EN ENSAMBLE PRODUCT-OF-EXPERTS (V51 vs V48 CHAMPION):")
    print(f"  - Macro-F1 V48 Champion Baseline : {f1_v48_global:.7f}")
    print(f"  - Macro-F1 V51 Blend Reemplazo   : {f1_v51_global:.7f}")
    print(f"  - Delta Global                   : {f1_v51_global - f1_v48_global:+.7f}")
    print(f"  - Folds Positivos                : {pos_folds} / 5")
    for f in range(5):
        print(f"      Fold {f}: V48={f1_v48_folds[f]:.6f} | V51={f1_v51_folds[f]:.6f} | Delta={deltas[f]:+.6f}")
    print("=" * 80, flush=True)

    # Remuestreo Pareado Bootstrap (B=1000)
    rng = np.random.default_rng(42)
    n_boot = 1000
    n_samples = len(y_true)
    diffs = []
    print("Calculando prueba pareada bootstrap (B=1000)...", flush=True)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        f1_b_v51 = f1_score(y_true[sample_idx], pred_v51[sample_idx], average="macro")
        f1_b_v48 = f1_score(y_true[sample_idx], pred_v48[sample_idx], average="macro")
        diffs.append(f1_b_v51 - f1_b_v48)

    diffs = np.array(diffs)
    ci95 = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    p_val = float(np.mean(diffs <= 0))

    print(f"Bootstrap Pareado (B=1000):")
    print(f"  - Delta Medio : {float(np.mean(diffs)):+.7f}")
    print(f"  - IC95        : [{ci95[0]:+.7f}, {ci95[1]:+.7f}]")
    print(f"  - p-valor     : {p_val:.4f}", flush=True)

    adopta = (f1_v51_global > f1_v48_global) and (pos_folds >= 4) and (p_val < 0.05) and (ci95[0] > 0)
    print(f"\nCOMPUERTA DE ADOPCIÓN: {'SUPERADA -> ADOPTAR' if adopta else 'NO SUPERADA -> FALSIFICACIÓN RIGUROSA'}")

    duracion = time.time() - inicio
    resultado_json = {
        "fase": "V51-01",
        "fecha": "2026-10-16",
        "duracion_seg": round(duracion, 2),
        "f1_v48_champion_baseline": f1_v48_global,
        "f1_v51_blend_global": f1_v51_global,
        "delta_global": f1_v51_global - f1_v48_global,
        "fold_scores_standalone": fold_scores_standalone,
        "f1_standalone_global": f1_global_standalone,
        "f1_v43_standalone": f1_v43_standalone,
        "f1_v48_folds": f1_v48_folds,
        "f1_v51_folds": f1_v51_folds,
        "deltas_por_fold": deltas,
        "folds_positivos": pos_folds,
        "bootstrap_mean_delta": float(np.mean(diffs)),
        "bootstrap_ci95": ci95,
        "bootstrap_p_value": p_val,
        "adopta": adopta
    }

    with open(RAIZ / "experiments/v51_01_consistent_smoothed_cramer_5folds.json", "w", encoding="utf-8") as f:
        json.dump(resultado_json, f, indent=2)

    print(f"Resultados persistidos en experiments/v51_01_consistent_smoothed_cramer_5folds.json ({duracion:.1f}s)")
    return 0


if __name__ == "__main__":
    main()
