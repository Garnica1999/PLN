"""V39 · 01 — Ensamble Trimodal (V30 + TabularResNet Ordinal + HistGradientBoosting) sobre matriz_279.

Combina tres familias complementarias:
1. V30 (árboles LightGBM refinados por Optuna + blend cuatro cuartos)
2. TabularResNet con función de pérdida ordinal (penalización L1 de distancia esperada en clases de edad)
3. HistGradientBoostingClassifier (árboles con binning histograma regularizado)

Validación honesta 5-fold StratifiedGroupKFold agrupada por dup_clusters.parquet.
Genera predicciones OOF y predicciones de test si se supera el gate.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import matriz_279

from src import config as cfg
from src.verificacion_numerica import autotest_numerico

RESULTADO = cfg.EXPERIMENTS / "v39_01_trio_ensemble_5folds.json"
OOF_TAB_ORD = cfg.FEATURES / "v39_oof_tabnet_ordinal.npy"
OOF_HGB = cfg.FEATURES / "v39_oof_hgb.npy"
TEST_TAB_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"


class TabularResNet(nn.Module):
    def __init__(self, in_features: int, hidden: int = 256, num_classes: int = 6, drop: float = 0.2):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.act0 = nn.SiLU()

        self.fc1 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(drop)

        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.act0(self.bn0(self.in_proj(x)))
        h1 = h0 + self.drop1(self.act1(self.bn1(self.fc1(h0))))
        h2 = h1 + self.drop2(self.act2(self.bn2(self.fc2(h1))))
        return self.head(h2)


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int = 6, lambda_dist: float = 0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_dist = lambda_dist
        classes = torch.arange(num_classes, dtype=torch.float32)
        dist_mat = torch.abs(classes.unsqueeze(0) - classes.unsqueeze(1))
        self.register_buffer("dist_mat", dist_mat)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        batch_dists = self.dist_mat[targets]
        dist_loss = torch.sum(probs * batch_dists, dim=1).mean()
        return ce_loss + self.lambda_dist * dist_loss


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V39] Cómputo en: {device}", flush=True)

    # 1. Cargar datos
    car = ROOT / "artifacts" / "p33_scores" / "todos"
    y_true = np.load(car / "y_true.npy")
    fold_idx = np.load(car / "fold_idx.npy")
    v30_prob = np.load(car / "v30_prob.npy")
    w_plugin = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

    X = matriz_279()
    N, D = X.shape
    print(f"[V39] Matriz 279 cargada: {N:,} filas x {D} columnas", flush=True)

    oof_tab = np.zeros((N, 6), dtype=np.float32)
    oof_hgb = np.zeros((N, 6), dtype=np.float32)

    epochs = 14
    batch_size = 256

    print("\n=== [1/3] Entrenamiento 5-fold de TabularResNet Ordinal y HistGradientBoosting ===", flush=True)
    for fold in range(5):
        t0 = time.time()
        train_m = fold_idx != fold
        val_m = fold_idx == fold

        # 1. HistGradientBoosting
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=31, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42 + fold
        )
        clf.fit(X[train_m], y_true[train_m])
        hgb_prob = clf.predict_proba(X[val_m])
        oof_hgb[val_m] = hgb_prob

        # 2. TabularResNet Ordinal
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_m])
        X_va = scaler.transform(X[val_m])
        y_tr = y_true[train_m]
        y_va = y_true[val_m]

        tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        va_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=512, shuffle=False)

        torch.manual_seed(42 + fold)
        net = TabularResNet(D, hidden=256, drop=0.2).to(device)
        criterion = OrdinalCrossEntropyLoss(6, lambda_dist=0.2).to(device)
        optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for _ in range(epochs):
            net.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(net(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
        val_preds = []
        with torch.no_grad():
            for bx, _ in va_loader:
                bx = bx.to(device)
                val_preds.append(torch.softmax(net(bx), dim=1).cpu().numpy())
        tab_prob = np.concatenate(val_preds, axis=0)
        oof_tab[val_m] = tab_prob

        # Evaluaciones individuales
        s_v30 = f1_score(y_va, np.argmax(v30_prob[val_m] * w_plugin, axis=1), average="macro")
        s_hgb = f1_score(y_va, np.argmax(hgb_prob * w_plugin, axis=1), average="macro")
        s_tab = f1_score(y_va, np.argmax(tab_prob * w_plugin, axis=1), average="macro")

        # Blend trío Fold
        p_trio_val = 0.82 * v30_prob[val_m] + 0.12 * tab_prob + 0.06 * hgb_prob
        s_trio = f1_score(y_va, np.argmax(p_trio_val * w_plugin, axis=1), average="macro")

        dt = time.time() - t0
        print(f"  Fold {fold}: TabNet={s_tab:.5f} | HGB={s_hgb:.5f} | V30={s_v30:.5f} | "
              f"Trio={s_trio:.5f} (Delta: {s_trio - s_v30:+.5f}) [{dt:.1f}s]", flush=True)

    np.save(OOF_TAB_ORD, oof_tab)
    np.save(OOF_HGB, oof_hgb)

    # 2. Evaluación Global y Grid Search de Pesos
    print("\n=== [2/3] Evaluación Global OOF y Optimización de Pesos del Trío ===", flush=True)
    macro_v30 = float(f1_score(y_true, np.argmax(v30_prob * w_plugin, axis=1), average="macro"))
    macro_v38 = float(f1_score(y_true, np.argmax((0.88 * v30_prob + 0.12 * np.load(cfg.FEATURES / 'v38_oof_tabularnet.npy')) * w_plugin, axis=1), average="macro"))

    best_score = macro_v38
    best_a, best_b = 0.12, 0.0

    for a in [0.08, 0.10, 0.12, 0.14, 0.16]:
        for b in [0.02, 0.04, 0.06, 0.08]:
            c = 1.0 - a - b
            if c <= 0:
                continue
            p_blend = c * v30_prob + a * oof_tab + b * oof_hgb
            score = float(f1_score(y_true, np.argmax(p_blend * w_plugin, axis=1), average="macro"))
            if score > best_score:
                best_score = score
                best_a, best_b = a, b

    best_c = round(1.0 - best_a - best_b, 3)
    p_trio_best = best_c * v30_prob + best_a * oof_tab + best_b * oof_hgb
    delta_vs_v38 = best_score - macro_v38
    delta_vs_v30 = best_score - macro_v30

    print(f"  V30 Base Champion:   {macro_v30:.7f}")
    print(f"  V38 Champion Actual: {macro_v38:.7f}")
    print(f"  V39 Trío Óptimo:     {best_score:.7f} (TabNet={best_a:.2f}, HGB={best_b:.2f}, V30={best_c:.2f})")
    print(f"  Delta vs V38:        {delta_vs_v38:+.7f}")
    print(f"  Delta vs V30:        {delta_vs_v30:+.7f}")

    # Bootstrap pareado contra V38
    rng = np.random.default_rng(42)
    pred_v38_all = np.argmax((0.88 * v30_prob + 0.12 * np.load(cfg.FEATURES / 'v38_oof_tabularnet.npy')) * w_plugin, axis=1)
    pred_v39_all = np.argmax(p_trio_best * w_plugin, axis=1)

    diffs = []
    n_boot = 500
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        f39 = f1_score(y_true[idx], pred_v39_all[idx], average="macro", zero_division=0)
        f38 = f1_score(y_true[idx], pred_v38_all[idx], average="macro", zero_division=0)
        diffs.append(float(f39 - f38))

    ic95 = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    prop_pos = float(np.mean(np.array(diffs) > 0))
    print(f"  Bootstrap IC95 vs V38: {ic95} | Prop. positiva: {prop_pos:.3f}", flush=True)

    res = {
        "fase": "V39 01 Trio Ensemble",
        "macro_v30": macro_v30,
        "macro_v38": macro_v38,
        "macro_v39": best_score,
        "delta_vs_v38": delta_vs_v38,
        "delta_vs_v30": delta_vs_v30,
        "pesos": {"v30": best_c, "tabnet_ordinal": best_a, "hgb": best_b},
        "ic95_vs_v38": ic95,
        "prop_pos_vs_v38": prop_pos,
        "duracion_seg": round(time.time() - inicio, 1)
    }

    # 3. Si pasa el gate, generar predicciones sobre test
    print("\n=== [3/3] Generación de Predicciones sobre Test ===", flush=True)
    if delta_vs_v38 > 0:
        print("[V39] Gate superado. Entrenando modelos finales sobre todo train...", flush=True)
        test_matriz = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r"))

        # HGB full
        clf_full = HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=31, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42
        )
        clf_full.fit(X, y_true)
        test_hgb = clf_full.predict_proba(test_matriz)
        np.save(TEST_HGB, test_hgb)
        print(f"  HGB test listo: {test_hgb.shape}", flush=True)

        # TabNet full
        scaler_full = StandardScaler()
        X_full = scaler_full.fit_transform(X)
        X_test_scaled = scaler_full.transform(test_matriz)

        full_ds = TensorDataset(torch.tensor(X_full, dtype=torch.float32), torch.tensor(y_true, dtype=torch.long))
        full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
        test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

        torch.manual_seed(42)
        net_full = TabularResNet(D, hidden=256, drop=0.2).to(device)
        criterion_full = OrdinalCrossEntropyLoss(6, lambda_dist=0.2).to(device)
        optimizer = optim.AdamW(net_full.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for _ in range(epochs):
            net_full.train()
            for bx, by in full_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion_full(net_full(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        net_full.eval()
        test_preds = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(device)
                test_preds.append(torch.softmax(net_full(bx), dim=1).cpu().numpy())
        test_tab_ord = np.concatenate(test_preds, axis=0)
        np.save(TEST_TAB_ORD, test_tab_ord)
        print(f"  TabNet Ordinal test listo: {test_tab_ord.shape}", flush=True)
        res["test_generado"] = True
    else:
        print("[V39] Gate no superado para test.", flush=True)
        res["test_generado"] = False

    RESULTADO.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
