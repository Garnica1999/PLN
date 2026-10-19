"""V38 · 01 — TabularNet (Deep ResNet Tabular en GPU) como miembro diverso sobre matriz_279.

Entrena un modelo de redes neuronales profundas con bloques residuales, BatchNorm y SiLU
sobre las 279 meta-features en GPU, con validación cruzada 5-fold agrupada canónica.
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import matriz_279

from src import config as cfg
from src.verificacion_numerica import autotest_numerico

RESULTADO = cfg.EXPERIMENTS / "v38_01_tabularnet_5folds.json"
OOF_FILE = cfg.FEATURES / "v38_oof_tabularnet.npy"
TEST_PROB_FILE = cfg.FEATURES / "v38_test_prob_tabularnet.npy"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"


class TabularResNet(nn.Module):
    def __init__(self, in_features: int, hidden: int = 256, num_classes: int = 6, drop: float = 0.2):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.act0 = nn.SiLU()

        # Block 1
        self.fc1 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(drop)

        # Block 2
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


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V38] Dispositivo de cómputo: {device}", flush=True)

    # 1. Cargar datos
    car = ROOT / "artifacts" / "p33_scores" / "todos"
    y_true = np.load(car / "y_true.npy")
    fold_idx = np.load(car / "fold_idx.npy")
    v30_prob = np.load(car / "v30_prob.npy")
    w_plugin = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

    X = matriz_279()
    N, D = X.shape
    print(f"[V38] Matriz 279 cargada: {N:,} filas x {D} columnas", flush=True)

    oof_tabnet = np.zeros((N, 6), dtype=np.float32)
    scores_tabnet = []
    scores_v30 = []
    scores_blend = []

    epochs = 14
    batch_size = 256

    for fold in range(5):
        t0 = time.time()
        train_m = fold_idx != fold
        val_m = fold_idx == fold

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
        criterion = nn.CrossEntropyLoss()
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
                out = torch.softmax(net(bx), dim=1)
                val_preds.append(out.cpu().numpy())
        val_prob = np.concatenate(val_preds, axis=0)
        oof_tabnet[val_m] = val_prob

        # Evaluaciones
        v30_val = v30_prob[val_m]
        s_v30 = f1_score(y_va, np.argmax(v30_val * w_plugin, axis=1), average="macro")
        s_tab = f1_score(y_va, np.argmax(val_prob * w_plugin, axis=1), average="macro")

        # Blend con peso 0.10
        blend_val = 0.90 * v30_val + 0.10 * val_prob
        s_blend = f1_score(y_va, np.argmax(blend_val * w_plugin, axis=1), average="macro")

        scores_v30.append(s_v30)
        scores_tabnet.append(s_tab)
        scores_blend.append(s_blend)

        dt = time.time() - t0
        print(f"  Fold {fold}: TabNet={s_tab:.5f} | V30+Plugin={s_v30:.5f} | Blend={s_blend:.5f} "
              f"(Delta: {s_blend - s_v30:+.5f}) [{dt:.1f}s]", flush=True)

    np.save(OOF_FILE, oof_tabnet)

    # Métricas Globales
    macro_v30 = float(f1_score(y_true, np.argmax(v30_prob * w_plugin, axis=1), average="macro"))
    macro_tab = float(f1_score(y_true, np.argmax(oof_tabnet * w_plugin, axis=1), average="macro"))

    # Buscar el mejor alpha global
    best_alpha = 0.0
    best_macro_blend = macro_v30
    for alpha in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]:
        blend_all = (1.0 - alpha) * v30_prob + alpha * oof_tabnet
        score = float(f1_score(y_true, np.argmax(blend_all * w_plugin, axis=1), average="macro"))
        print(f"  Alpha {alpha:.2f}: Macro-F1 = {score:.7f} (Delta: {score - macro_v30:+.7f})", flush=True)
        if score > best_macro_blend:
            best_macro_blend = score
            best_alpha = alpha

    # Bootstrap pareado del mejor blend contra V30+plugin
    rng = np.random.default_rng(42)
    pred_v30_all = np.argmax(v30_prob * w_plugin, axis=1)
    pred_blend_all = np.argmax(((1.0 - best_alpha) * v30_prob + best_alpha * oof_tabnet) * w_plugin, axis=1)

    diffs = []
    n_boot = 500
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        f_b = f1_score(y_true[idx], pred_blend_all[idx], average="macro", zero_division=0)
        f_v = f1_score(y_true[idx], pred_v30_all[idx], average="macro", zero_division=0)
        diffs.append(float(f_b - f_v))

    ic95 = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    prop_pos = float(np.mean(np.array(diffs) > 0))

    print(f"\n[V38] Mejor Alpha: {best_alpha:.2f}", flush=True)
    print(f"[V38] Macro-F1 Campeon Base: {macro_v30:.7f}", flush=True)
    print(f"[V38] Macro-F1 Blend V38:    {best_macro_blend:.7f} (Delta: {best_macro_blend - macro_v30:+.7f})", flush=True)
    print(f"[V38] Bootstrap IC95: {ic95} | Proporción positiva: {prop_pos:.3f}", flush=True)

    res = {
        "fase": "V38 01 TabularResNet 5-folds",
        "duracion_seg": round(time.time() - inicio, 1),
        "scores_v30": [round(x, 7) for x in scores_v30],
        "scores_tabnet": [round(x, 7) for x in scores_tabnet],
        "scores_blend": [round(x, 7) for x in scores_blend],
        "macro_v30": macro_v30,
        "macro_tabnet": macro_tab,
        "best_alpha": best_alpha,
        "best_macro_blend": best_macro_blend,
        "delta": round(best_macro_blend - macro_v30, 7),
        "ic95": ic95,
        "proporcion_positiva": prop_pos,
        "ic95_excluye_cero": bool(ic95[0] > 0),
    }

    # Si pasa el gate, ajustar sobre todo el train y predecir test
    if ic95[0] > 0 and (best_macro_blend - macro_v30) >= 0.0003:
        print("[V38] ¡GATE SUPERADO! Entrenando modelo final sobre todo el train...", flush=True)
        scaler_full = StandardScaler()
        X_full = scaler_full.fit_transform(X)
        test_matriz = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r"))
        X_test_scaled = scaler_full.transform(test_matriz)

        full_ds = TensorDataset(torch.tensor(X_full, dtype=torch.float32), torch.tensor(y_true, dtype=torch.long))
        full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
        test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

        torch.manual_seed(42)
        net_full = TabularResNet(D, hidden=256, drop=0.2).to(device)
        optimizer = optim.AdamW(net_full.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for _ in range(epochs):
            net_full.train()
            for bx, by in full_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(net_full(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        net_full.eval()
        test_preds = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(device)
                out = torch.softmax(net_full(bx), dim=1)
                test_preds.append(out.cpu().numpy())
        test_prob_tabnet = np.concatenate(test_preds, axis=0)
        np.save(TEST_PROB_FILE, test_prob_tabnet)
        print(f"[V38] Probabilidades de test guardadas en {TEST_PROB_FILE.name}: {test_prob_tabnet.shape}", flush=True)
        res["test_prob_generado"] = True
    else:
        print("[V38] Gate no superado para generación de test.", flush=True)
        res["test_prob_generado"] = False

    RESULTADO.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
