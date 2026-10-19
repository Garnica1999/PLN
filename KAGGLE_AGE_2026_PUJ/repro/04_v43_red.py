"""Spike 04 - Reproducibilidad del componente neuronal V43 (Cramer-SWA-SE-TabularResNet).

Objetivo: reconstruir las probabilidades OOF y TEST de V43 con el MISMO procedimiento que
scripts/v43_01_cramer_swa_tabularnet_5folds.py y COMPARARLAS contra los .npy guardados
(no los sobrescribe). Esto aisla la reproducibilidad del entrenamiento neuronal entre
entornos (torch 2.14+ROCm local -> torch 2.8+CUDA servidor).

Entradas (del repo): meta_features_train_v11.npy + oof_ADV07_nbsvm_word.npy (matriz 279),
meta_features_test_v12_fresca.npy (test), artifacts/p33_scores/todos/{y_true,fold_idx}.npy,
features/v43_oof_cramer_swa_tabularnet.npy y features/v43_test_cramer_swa_tabularnet.npy
(referencias a comparar).

NO escribe artefactos del proyecto: solo imprime el informe. ASCII puro.

Uso:
    python repro/04_v43_red.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, TensorDataset

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = RAIZ / "features"
CARRIL = RAIZ / "artifacts" / "p33_scores" / "todos"
SEED = 42
PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])


class SETabularResNet(nn.Module):
    """Red residual tabular con compuertas Squeeze-and-Excitation (identica al script V43)."""

    def __init__(self, in_features: int = 279, hidden: int = 256, num_classes: int = 6,
                 drop: float = 0.2, reduction: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.act0 = nn.SiLU()

        self.fc1 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.se1_fc1 = nn.Linear(hidden, hidden // reduction)
        self.se1_act = nn.SiLU()
        self.se1_fc2 = nn.Linear(hidden // reduction, hidden)
        self.se1_sig = nn.Sigmoid()

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

        res1 = self.drop1(self.act1(self.bn1(self.fc1(h0))))
        se1 = self.se1_sig(self.se1_fc2(self.se1_act(self.se1_fc1(res1))))
        h1 = h0 + res1 * se1

        res2 = self.drop2(self.act2(self.bn2(self.fc2(h1))))
        se2 = self.se2_sig(self.se2_fc2(self.se2_act(self.se2_fc1(res2))))
        h2 = h1 + res2 * se2

        return self.head(h2)


class CramerDistanceLoss(nn.Module):
    """Entropia cruzada con suavizado 0.10 + distancia de Cramer (CDF) con peso lambda."""

    def __init__(self, num_classes: int = 6, lambda_cramer: float = 0.50, smoothing: float = 0.10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes
        self.lambda_cramer = lambda_cramer

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]
        k_indices = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        cdf_target = (targets.unsqueeze(1) <= k_indices).float()
        loss_cramer = torch.mean(torch.sum((cdf_pred - cdf_target) ** 2, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer


def informe_diferencias(nombre: str, reconstruido: np.ndarray, referencia: np.ndarray) -> None:
    """Imprime el diagnostico de diferencias entre dos matrices de probabilidades."""
    dif = np.abs(reconstruido.astype(np.float64) - referencia.astype(np.float64))
    acuerdo = float((np.argmax(reconstruido, axis=1) == np.argmax(referencia, axis=1)).mean())
    acuerdo_plug = float(
        (np.argmax(reconstruido * PESOS_PLUG_IN, axis=1)
         == np.argmax(referencia * PESOS_PLUG_IN, axis=1)).mean()
    )
    print(f"[spike04] {nombre}: max|d|={dif.max():.3e} media|d|={dif.mean():.3e} "
          f"p999|d|={np.percentile(dif, 99.9):.3e} | argmax igual={acuerdo:.8f} | "
          f"argmax plug-in igual={acuerdo_plug:.8f}")


def main() -> int:
    """Reentrena V43 completo y compara contra las referencias guardadas."""
    t_inicio = time.time()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[spike04] dispositivo: {dispositivo} | torch {torch.__version__}")

    base = np.load(FEATURES / "meta_features_train_v11.npy")
    extra = np.load(FEATURES / "oof_ADV07_nbsvm_word.npy")
    X_train_raw = np.concatenate([base.astype(np.float32), extra.astype(np.float32)], axis=1)
    assert X_train_raw.shape[1] == 279, X_train_raw.shape
    X_test_raw = np.load(FEATURES / "meta_features_test_v12_fresca.npy").astype(np.float32)

    y_true = np.load(CARRIL / "y_true.npy")
    fold_idx = np.load(CARRIL / "fold_idx.npy")

    n_samples, n_features = X_train_raw.shape
    n_test = X_test_raw.shape[0]
    n_classes = 6
    print(f"[spike04] matriz train {X_train_raw.shape} | test {X_test_raw.shape}")

    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    batch_size, epochs, swa_start = 256, 16, 6
    lr, lr_min, weight_decay = 1e-3, 1e-4, 1e-4
    smoothing, lambda_cramer = 0.10, 0.50

    puntajes_fold = []
    for fold in range(5):
        t0 = time.time()
        tr_mask = (fold_idx != fold)
        val_mask = (fold_idx == fold)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw[tr_mask])
        X_val = scaler.transform(X_train_raw[val_mask])
        X_te_scaled = scaler.transform(X_test_raw)

        y_tr = y_true[tr_mask]
        y_val = y_true[val_mask]

        torch.manual_seed(SEED + fold * 100)
        np.random.seed(SEED + fold * 100)

        base_model = SETabularResNet(in_features=n_features, hidden=256, num_classes=n_classes,
                                     drop=0.2, reduction=4).to(dispositivo)
        swa_model = AveragedModel(base_model)
        criterion = CramerDistanceLoss(num_classes=n_classes, lambda_cramer=lambda_cramer,
                                       smoothing=smoothing).to(dispositivo)
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

        update_bn(loader_train, swa_model, device=dispositivo)

        swa_model.eval()
        with torch.no_grad():
            val_t = torch.from_numpy(X_val).float().to(dispositivo)
            val_p = torch.softmax(swa_model(val_t), dim=1).cpu().numpy()
            oof_probs[val_mask] = val_p

            te_t = torch.from_numpy(X_te_scaled).float().to(dispositivo)
            te_p = torch.softmax(swa_model(te_t), dim=1).cpu().numpy()
            test_probs += te_p / 5.0

        puntaje = float(f1_score(y_val, np.argmax(val_p * PESOS_PLUG_IN, axis=1), average="macro"))
        puntajes_fold.append(puntaje)
        print(f"[spike04] fold {fold}: standalone={puntaje:.6f} | {time.time() - t0:.1f}s", flush=True)

    f1_global = float(f1_score(y_true, np.argmax(oof_probs * PESOS_PLUG_IN, axis=1), average="macro"))
    print(f"[spike04] OOF standalone global reconstruido: {f1_global:.7f}")

    ref_oof = np.load(FEATURES / "v43_oof_cramer_swa_tabularnet.npy")
    ref_test = np.load(FEATURES / "v43_test_cramer_swa_tabularnet.npy")
    f1_ref = float(f1_score(y_true, np.argmax(ref_oof * PESOS_PLUG_IN, axis=1), average="macro"))
    print(f"[spike04] OOF standalone global referencia   : {f1_ref:.7f}")

    informe_diferencias("OOF ", oof_probs, ref_oof)
    informe_diferencias("TEST", test_probs, ref_test)

    meta_path = RAIZ / "experiments" / "v43_01_cramer_swa_tabularnet_5folds.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[spike04] folds documentados (JSON): {meta.get('fold_scores_standalone')}")
        print(f"[spike04] folds reconstruidos      : {[round(v, 6) for v in puntajes_fold]}")

    print(f"[spike04] tiempo total {time.time() - t_inicio:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
