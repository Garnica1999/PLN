"""V46 · 01 — RankGauss-Cramer-SWA-SE-TabularResNet (Transformación Normal Cuantílica RankGauss + Distancia de Cramér-von Mises + Suavizado de Etiquetas 0.10 + SWA en GPU).

Motivación Estadística y Matemática:
El análisis de distribución de las 279 features en `matriz_279` revela una asimetría extrema:
124 de las 279 features tienen sesgo |skewness| > 3, 26 features |skewness| > 10, y 71 features
tienen más del 50% de ceros. El preprocesamiento tradicional con StandardScaler preserva las colas
pesadas y valores atípicos que generan picos de gradiente en las proyecciones lineales densas.
La transformación RankGauss (QuantileTransformer con output_distribution='normal') mapea monótonamente
el rango de cada feature sobre una gaussiana estándar N(0, 1), acotando outliers en [-3.5, +3.5]
y obligando a las compuertas de atención condicional Squeeze-and-Excitation (SE) a operar sobre
un colector isotrópico.

Validación canónica 5-fold StratifiedGroupKFold agrupada por dup_clusters (cero fuga in-fold).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import matriz_279

from src import config as cfg
from src.verificacion_numerica import autotest_numerico

RESULTADO = cfg.EXPERIMENTS / "v46_01_rankgauss_cramer_swa_tabularnet_5folds.json"
OOF_FILE = cfg.FEATURES / "v46_oof_rankgauss_cramer_swa_tabularnet.npy"
TEST_PROB_FILE = cfg.FEATURES / "v46_test_rankgauss_cramer_swa_tabularnet.npy"
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


class CramerDistanceLoss(nn.Module):
    """Entropía Cruzada con Suavizado de Etiquetas combinada con regularización de distancia de Cramér (CDF)."""

    def __init__(self, num_classes: int = 6, lambda_cramer: float = 0.50, smoothing: float = 0.10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes
        self.lambda_cramer = lambda_cramer

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        # CDF predicha acumulada F_p(k)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]
        # CDF escalón objetivo F_y(k) = I(y <= k)
        k_indices = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        cdf_target = (targets.unsqueeze(1) <= k_indices).float()
        # Pérdida cuadrática de Cramér (Wasserstein-2 en 1D)
        loss_cramer = torch.mean(torch.sum((cdf_pred - cdf_target) ** 2, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V46] Iniciando entrenamiento 5-fold de RankGauss-Cramer-SWA-SE-TabularResNet...", flush=True)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V46] Dispositivo detectado: {dispositivo}", flush=True)

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
    print(f"[V46] Macro-F1 V30 (Baseline): {f1_v30_baseline:.7f}", flush=True)

    # Hiperparámetros
    batch_size = 256
    epochs = 16
    swa_start = 6
    lr = 1e-3
    lr_min = 1e-4
    weight_decay = 1e-4
    smoothing = 0.10
    lambda_cramer = 0.50

    fold_scores_standalone = []
    v30_fold_scores = []

    for fold in range(5):
        t0 = time.time()
        tr_mask = (fold_idx != fold)
        val_mask = (fold_idx == fold)

        # Preprocesamiento estrictamente in-fold con RankGauss (QuantileTransformer)
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=2000, random_state=cfg.SEED + fold, subsample=100000)
        X_tr = qt.fit_transform(X_train_raw[tr_mask])
        X_val = qt.transform(X_train_raw[val_mask])
        X_te_scaled = qt.transform(X_test_raw)

        y_tr = y_true[tr_mask]
        y_val = y_true[val_mask]

        torch.manual_seed(cfg.SEED + fold * 100)
        np.random.seed(cfg.SEED + fold * 100)

        base_model = SETabularResNet(in_features=n_features, hidden=256, num_classes=n_classes, drop=0.2, reduction=4).to(dispositivo)
        swa_model = AveragedModel(base_model)
        criterion = CramerDistanceLoss(num_classes=n_classes, lambda_cramer=lambda_cramer, smoothing=smoothing).to(dispositivo)
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
            f"  Fold {fold}: Standalone RankGauss-Cramer={score_fold_standalone:.6f} | V30={score_v30_fold:.6f} | "
            f"Tiempo: {time.time() - t0:.1f}s",
            flush=True,
        )

    # Evaluación Global OOF Standalone
    pred_global_standalone = np.argmax(oof_probs * pesos_plug_in, axis=1)
    f1_global_standalone = float(f1_score(y_true, pred_global_standalone, average="macro"))

    print("\n" + "=" * 80)
    print(f"[V46] Macro-F1 OOF Standalone RankGauss-Cramer-SE-TabNet : {f1_global_standalone:.7f}")
    print(f"[V46] Macro-F1 V30 Baseline                               : {f1_v30_baseline:.7f}")
    print(f"[V46] Delta Standalone vs V30                             : {f1_global_standalone - f1_v30_baseline:+.7f}")
    print("=" * 80 + "\n", flush=True)

    np.save(OOF_FILE, oof_probs)
    np.save(TEST_PROB_FILE, test_probs)
    print(f"[V46] Guardado OOF en {OOF_FILE} y Test Probs en {TEST_PROB_FILE}", flush=True)

    duracion = round(time.time() - inicio, 2)
    meta = {
        "fase": "V46-01",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": duracion,
        "f1_v30_baseline": f1_v30_baseline,
        "f1_rankgauss_cramer_standalone": f1_global_standalone,
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
            "weight_decay": weight_decay,
            "transformer": "QuantileTransformer(normal, 2000 quantiles)",
        },
        "oof_shape": list(oof_probs.shape),
        "test_shape": list(test_probs.shape),
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    RESULTADO.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[V46] Metadatos guardados en {RESULTADO}. Tiempo total: {duracion}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
