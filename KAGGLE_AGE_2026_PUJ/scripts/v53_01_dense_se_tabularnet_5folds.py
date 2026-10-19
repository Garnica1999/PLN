"""V53 · 01 — Dense Squeeze-and-Excitation Tabular Network (Dense-SE-TabularNet) con SWA y Cramér Loss en GPU ROCm.

Motivación Matemática y Arquitectónica:
En las redes residuales previas (SETabularResNet en V40-V51), las características latentes se combinan
aditivamente (h_{l+1} = h_l + res_l * se_l), forzando que las representaciones intermedias compartan
el mismo espacio dimensional y diluyendo la información de los 279 predictores originales a lo largo
de las capas profundas.

Dense-SE-TabularNet introduce un sesgo inductivo basado en DenseNet (Huang et al., CVPR 2017) adaptado
a datos tabulares heterogéneos:
1. Concatenación Multi-Escala de Características Latentes:
   Cada bloque denso genera un nuevo conjunto de 128 características abstractas que se concatenan
   con todas las representaciones anteriores (192 -> 320 -> 448 -> 576). Esto permite que las capas
   superiores y la cabeza de clasificación inspeccionen simultáneamente los rasgos léxicos/estructurales
   básicos y las interacciones de alto orden.
2. Compuertas de Atención por Canal Squeeze-and-Excitation (SE):
   En cada bloque de crecimiento denso, una compuerta SE modula la relevancia de los nuevos canales
   generados antes de la concatenación.
3. Pérdida Cuadrática de Cramér-von Mises sobre CDF Suavizada Consistente:
   L = L_{CE}(smoothing=0.10) + 0.50 * L_{Cramér}(F_p, F_smooth).
4. Winsorización in-fold [-6.0, +6.0] en Z-score para blindaje ante outliers.
5. SWA (Stochastic Weight Averaging) desde época 6 con actualización honesta in-fold de BatchNorm.

Protocolo:
- Validación canónica 5-fold StratifiedGroupKFold agrupada por dup_clusters.parquet (cero fuga).
- Evaluación standalone con regla congelada plug-in w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128].
- Guardado de OOF y Test probabilities en features/.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, TensorDataset

RAIZ = pathlib.Path.cwd()
while not (RAIZ / "experiments.csv").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent

sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from scripts.v19_b1_boosting import matriz_279
from src import config as cfg
from src.verificacion_numerica import autotest_numerico

RESULTADO = cfg.EXPERIMENTS / "v53_01_dense_se_tabularnet_5folds.json"
OOF_FILE = cfg.FEATURES / "v53_oof_dense_se_tabularnet.npy"
TEST_PROB_FILE = cfg.FEATURES / "v53_test_dense_se_tabularnet.npy"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])


class SEChannelBlock(nn.Module):
    """Módulo de atención por canal Squeeze-and-Excitation para bloques densos tabulares."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sig(self.fc2(self.act(self.fc1(x))))
        return x * w


class TabularDenseNet(nn.Module):
    """Red densa profunda para datos tabulares con concatenación multi-escala y compuertas SE."""

    def __init__(
        self,
        in_features: int = 279,
        init_dim: int = 192,
        growth_rate: int = 128,
        num_classes: int = 6,
        drop: float = 0.20,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features, init_dim)
        self.bn0 = nn.BatchNorm1d(init_dim)
        self.act0 = nn.SiLU()

        # Bloque Denso 1: init_dim (192) -> growth_rate (128)
        self.bn1 = nn.BatchNorm1d(init_dim)
        self.fc1 = nn.Linear(init_dim, growth_rate)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.se1 = SEChannelBlock(growth_rate, reduction=4)

        # Bloque Denso 2: (192 + 128 = 320) -> growth_rate (128)
        in_dim2 = init_dim + growth_rate
        self.bn2 = nn.BatchNorm1d(in_dim2)
        self.fc2 = nn.Linear(in_dim2, growth_rate)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(drop)
        self.se2 = SEChannelBlock(growth_rate, reduction=4)

        # Bloque Denso 3: (320 + 128 = 448) -> growth_rate (128)
        in_dim3 = in_dim2 + growth_rate
        self.bn3 = nn.BatchNorm1d(in_dim3)
        self.fc3 = nn.Linear(in_dim3, growth_rate)
        self.act3 = nn.SiLU()
        self.drop3 = nn.Dropout(drop)
        self.se3 = SEChannelBlock(growth_rate, reduction=4)

        # Capa de Transición / Agregación Global: (448 + 128 = 576) -> 256
        total_dense_dim = in_dim3 + growth_rate
        self.bn_trans = nn.BatchNorm1d(total_dense_dim)
        self.fc_trans = nn.Linear(total_dense_dim, 256)
        self.act_trans = nn.SiLU()
        self.drop_trans = nn.Dropout(0.10)

        # Clasificador Lineal
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.act0(self.bn0(self.in_proj(x)))

        z1 = self.se1(self.drop1(self.act1(self.fc1(self.bn1(h0)))))
        h1 = torch.cat([h0, z1], dim=1)

        z2 = self.se2(self.drop2(self.act2(self.fc2(self.bn2(h1)))))
        h2 = torch.cat([h1, z2], dim=1)

        z3 = self.se3(self.drop3(self.act3(self.fc3(self.bn3(h2)))))
        h3 = torch.cat([h2, z3], dim=1)

        ht = self.drop_trans(self.act_trans(self.fc_trans(self.bn_trans(h3))))
        return self.head(ht)


class ConsistentSmoothedCramerLoss(nn.Module):
    """Pérdida híbrida de Entropía Cruzada con Label Smoothing y Cramér-von Mises consistente."""

    def __init__(self, num_classes: int = 6, lambda_cramer: float = 0.50, smoothing: float = 0.10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes
        self.lambda_cramer = lambda_cramer
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]
        k_indices = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        cdf_step = (targets.unsqueeze(1) <= k_indices).float()
        cdf_target = (1.0 - self.smoothing) * cdf_step + self.smoothing * (k_indices + 1.0) / self.num_classes
        loss_cramer = torch.mean(torch.sum((cdf_pred - cdf_target) ** 2, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=" * 80)
    print("[V53] ENTRENAMIENTO 5-FOLD DE TABULAR DENSENET (DENSE-SE-TABULARNET)")
    print("=" * 80, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V53] Dispositivo de aceleración: {device}", flush=True)

    X_train_raw = matriz_279()
    X_test_raw = np.load(MATRIZ_TEST).astype(np.float32)

    y_true = np.load(RAIZ / "artifacts/p33_scores/todos/y_true.npy")
    fold_idx = np.load(RAIZ / "artifacts/p33_scores/todos/fold_idx.npy")

    n_samples, n_features = X_train_raw.shape
    n_test = X_test_raw.shape[0]
    n_classes = len(cfg.CLASSES)

    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    # Hiperparámetros
    batch_size = 256
    epochs = 16
    swa_start = 6
    lr = 1e-3
    lr_min = 1e-4
    weight_decay = 1e-4
    smoothing = 0.10
    lambda_cramer = 0.50
    outlier_clip = 6.0

    fold_scores_raw = []
    fold_scores_plugin = []

    for fold in range(5):
        t0 = time.time()
        tr_mask = fold_idx != fold
        val_mask = fold_idx == fold

        # Preprocesamiento estrictamente in-fold (zero leakage)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw[tr_mask])
        X_val = scaler.transform(X_train_raw[val_mask])
        X_te_scaled = scaler.transform(X_test_raw)

        # Winsorización in-fold para blindar ante outliers extremos
        X_tr = np.clip(X_tr, -outlier_clip, outlier_clip)
        X_val = np.clip(X_val, -outlier_clip, outlier_clip)
        X_te_scaled = np.clip(X_te_scaled, -outlier_clip, outlier_clip)

        y_tr = y_true[tr_mask]
        y_val = y_true[val_mask]

        torch.manual_seed(cfg.SEED + fold * 100)
        np.random.seed(cfg.SEED + fold * 100)

        base_model = TabularDenseNet(
            in_features=n_features, init_dim=192, growth_rate=128, num_classes=n_classes, drop=0.20
        ).to(device)
        swa_model = AveragedModel(base_model)
        criterion = ConsistentSmoothedCramerLoss(
            num_classes=n_classes, lambda_cramer=lambda_cramer, smoothing=smoothing
        ).to(device)
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

        pred_f_raw = np.argmax(val_p, axis=1)
        pred_f_plug = np.argmax(val_p * PESOS_PLUG_IN, axis=1)

        sc_raw = float(f1_score(y_val, pred_f_raw, average="macro"))
        sc_plug = float(f1_score(y_val, pred_f_plug, average="macro"))

        fold_scores_raw.append(sc_raw)
        fold_scores_plugin.append(sc_plug)

        dur_fold = time.time() - t0
        print(f"  Fold {fold}: Raw F1={sc_raw:.6f} | Plug-in F1={sc_plug:.6f} ({dur_fold:.1f}s)", flush=True)

    # Métricas Globales OOF
    pred_global_raw = np.argmax(oof_probs, axis=1)
    pred_global_plug = np.argmax(oof_probs * PESOS_PLUG_IN, axis=1)

    f1_global_raw = float(f1_score(y_true, pred_global_raw, average="macro"))
    f1_global_plug = float(f1_score(y_true, pred_global_plug, average="macro"))

    duracion_total = time.time() - inicio
    print("=" * 80)
    print(f"[V53] Standalone OOF Macro-F1 Raw     : {f1_global_raw:.7f}")
    print(f"[V53] Standalone OOF Macro-F1 Plug-in : {f1_global_plug:.7f}")
    print(f"[V53] Tiempo total de entrenamiento   : {duracion_total:.1f}s")
    print("=" * 80, flush=True)

    # Persistir artefactos
    np.save(OOF_FILE, oof_probs)
    np.save(TEST_PROB_FILE, test_probs)
    print(f"[V53] OOF guardado en: {OOF_FILE}")
    print(f"[V53] Test guardado en: {TEST_PROB_FILE}")

    resultado = {
        "fase": "V53-01",
        "modelo": "TabularDenseNet_SWA_Cramer",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(duracion_total, 2),
        "f1_global_raw": f1_global_raw,
        "f1_global_plugin": f1_global_plug,
        "fold_scores_raw": fold_scores_raw,
        "fold_scores_plugin": fold_scores_plugin,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs": epochs,
            "swa_start": swa_start,
            "lr": lr,
            "lr_min": lr_min,
            "weight_decay": weight_decay,
            "smoothing": smoothing,
            "lambda_cramer": lambda_cramer,
            "outlier_clip": outlier_clip,
            "init_dim": 192,
            "growth_rate": 128,
            "drop": 0.20,
        },
        "oof_shape": list(oof_probs.shape),
        "test_shape": list(test_probs.shape),
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w") as f:
        json.dump(resultado, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
