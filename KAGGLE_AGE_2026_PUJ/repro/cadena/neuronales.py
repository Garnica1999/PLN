"""Componentes neuronales V38-V53 + HGB (porte fiel de scripts/v38_01..v53_01).

Arquitecturas, perdidas, semillas, SWA y estrategias de test EXACTAS de cada script original:
- V38/V39: TabularResNet, seeds 42+fold, 14 epocas, Cosine(T_max=14), sin SWA; test = modelo full-train.
- V40: SETabularResNet, OrdinalDistanceLoss, seeds 42+fold*100, 12 epocas, sin scheduler, sin SWA; test = media/5.
- V41: SETabularResNet, OrdinalDistanceLoss, 16 epocas, sin scheduler, SWA swa_start=8; test = media/5.
- V42: SETabularResNet, SmoothOrdinalDistanceLoss, 16 epocas, Cosine(T_max=16, eta_min=1e-4), SWA swa_start=6; test = media/5.
- V43: SETabularResNet, CramerDistanceLoss, 16 epocas, Cosine(T_max=16, eta_min=1e-4), SWA swa_start=6; test = media/5.
- V47: SETabularResNet, WeightedCramerLoss (gamma por frontera), recorte Z [-6,+6], seeds 42+fold*100,
  16 epocas, Cosine(T_max=16, eta_min=1e-4), SWA swa_start=6 con update_bn en loader 512 sin barajar; test = media/5.
- V51: SETabularResNet, ConsistentSmoothedCramerLoss, 16 epocas, Cosine(T_max=16, eta_min=1e-4), SWA swa_start=6; test = media/5.
- V53: TabularDenseNet (init 192, growth 128) con SEChannelBlock, ConsistentSmoothedCramerLoss, recorte Z [-6,+6],
  16 epocas, Cosine(T_max=16, eta_min=1e-4), SWA swa_start=6; test = media/5.
- V39-HGB: HistGradientBoostingClassifier por fold; test = modelo full-train.
ASCII puro. Sin efectos en import.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, TensorDataset

from cadena import comun

#: Configuracion exacta por componente (pack 04; los scripts originales mandan).
CONFIGS: dict[str, dict] = {
    "v38": {"modelo": "tabular", "loss": "ce", "epocas": 14, "batch": 256, "lr": 1e-3,
            "wd": 1e-4, "coseno": True, "eta_min": 0.0, "swa_start": None,
            "seed": "42+fold", "test": "full"},
    "v39_ordinal": {"modelo": "tabular", "loss": "ordinal_ce", "epocas": 14, "batch": 256,
                    "lr": 1e-3, "wd": 1e-4, "coseno": True, "eta_min": 0.0, "swa_start": None,
                    "seed": "42+fold", "test": "full"},
    "v40": {"modelo": "se", "loss": "ordinal_dist", "epocas": 12, "batch": 256, "lr": 1e-3,
            "wd": 1e-4, "coseno": False, "eta_min": 0.0, "swa_start": None,
            "seed": "42+fold*100", "test": "media"},
    "v41": {"modelo": "se", "loss": "ordinal_dist", "epocas": 16, "batch": 256, "lr": 1e-3,
            "wd": 1e-4, "coseno": False, "eta_min": 0.0, "swa_start": 8,
            "seed": "42+fold*100", "test": "media"},
    "v42": {"modelo": "se", "loss": "smooth_ordinal", "epocas": 16, "batch": 256, "lr": 1e-3,
            "wd": 1e-4, "coseno": True, "eta_min": 1e-4, "swa_start": 6,
            "seed": "42+fold*100", "test": "media"},
    "v43": {"modelo": "se", "loss": "cramer", "epocas": 16, "batch": 256, "lr": 1e-3,
            "wd": 1e-4, "coseno": True, "eta_min": 1e-4, "swa_start": 6,
            "seed": "42+fold*100", "test": "media"},
    "v47": {"modelo": "se", "loss": "weighted_cramer", "epocas": 16, "batch": 256, "lr": 1e-3,
            "wd": 1e-4, "coseno": True, "eta_min": 1e-4, "swa_start": 6,
            "seed": "42+fold*100", "test": "media", "clip": 6.0, "bn_batch": 512},
    "v51": {"modelo": "se", "loss": "consistent_cramer", "epocas": 16, "batch": 256, "lr": 1e-3,
            "wd": 1e-4, "coseno": True, "eta_min": 1e-4, "swa_start": 6,
            "seed": "42+fold*100", "test": "media"},
    "v53": {"modelo": "dense_se", "loss": "consistent_cramer", "epocas": 16, "batch": 256,
            "lr": 1e-3, "wd": 1e-4, "coseno": True, "eta_min": 1e-4, "swa_start": 6,
            "seed": "42+fold*100", "test": "media", "clip": 6.0},
}


class TabularResNet(nn.Module):
    """Red residual tabular (scripts/v38_01_tabularnet_5folds.py:44-71)."""

    def __init__(self, in_features: int = 279, hidden: int = 256, num_classes: int = 6,
                 drop: float = 0.2):
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


class SETabularResNet(nn.Module):
    """Red residual con compuertas Squeeze-and-Excitation (scripts/v40_01..v43_01)."""

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


class SEChannelBlock(nn.Module):
    """Compuerta Squeeze-and-Excitation por canal (scripts/v53_01_dense_se_tabularnet_5folds.py:65-77)."""

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
    """Red densa tabular con concatenacion multi-escala y compuertas SE (scripts/v53_01:80-142)."""

    def __init__(self, in_features: int = 279, init_dim: int = 192, growth_rate: int = 128,
                 num_classes: int = 6, drop: float = 0.20):
        super().__init__()
        self.in_proj = nn.Linear(in_features, init_dim)
        self.bn0 = nn.BatchNorm1d(init_dim)
        self.act0 = nn.SiLU()
        self.bn1 = nn.BatchNorm1d(init_dim)
        self.fc1 = nn.Linear(init_dim, growth_rate)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.se1 = SEChannelBlock(growth_rate, reduction=4)
        in_dim2 = init_dim + growth_rate
        self.bn2 = nn.BatchNorm1d(in_dim2)
        self.fc2 = nn.Linear(in_dim2, growth_rate)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(drop)
        self.se2 = SEChannelBlock(growth_rate, reduction=4)
        in_dim3 = in_dim2 + growth_rate
        self.bn3 = nn.BatchNorm1d(in_dim3)
        self.fc3 = nn.Linear(in_dim3, growth_rate)
        self.act3 = nn.SiLU()
        self.drop3 = nn.Dropout(drop)
        self.se3 = SEChannelBlock(growth_rate, reduction=4)
        total_dense_dim = in_dim3 + growth_rate
        self.bn_trans = nn.BatchNorm1d(total_dense_dim)
        self.fc_trans = nn.Linear(total_dense_dim, 256)
        self.act_trans = nn.SiLU()
        self.drop_trans = nn.Dropout(0.10)
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


class OrdinalCrossEntropyLoss(nn.Module):
    """CE + distancia ordinal esperada (scripts/v39_01: OrdinalCrossEntropyLoss)."""

    def __init__(self, num_classes: int = 6, lambda_dist: float = 0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_dist = lambda_dist
        clases = torch.arange(num_classes, dtype=torch.float32)
        dist_mat = torch.abs(clases.unsqueeze(0) - clases.unsqueeze(1))
        self.register_buffer("dist_mat", dist_mat)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        batch_dists = self.dist_mat[targets]
        dist_loss = torch.sum(probs * batch_dists, dim=1).mean()
        return loss_ce + self.lambda_dist * dist_loss


class OrdinalDistanceLoss(nn.Module):
    """CE + distancia ordinal esperada (scripts/v40_01/v41_01)."""

    def __init__(self, num_classes: int = 6, lambda_dist: float = 0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_dist = lambda_dist
        c = torch.arange(num_classes).float()
        self.register_buffer("dist_matrix", torch.abs(c.unsqueeze(0) - c.unsqueeze(1)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dists = self.dist_matrix[targets]
        loss_dist = torch.sum(probs * dists, dim=1).mean()
        return loss_ce + self.lambda_dist * loss_dist


class SmoothOrdinalDistanceLoss(nn.Module):
    """CE con suavizado 0.10 + distancia ordinal (scripts/v42_01)."""

    def __init__(self, num_classes: int = 6, lambda_dist: float = 0.2, smoothing: float = 0.10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.lambda_dist = lambda_dist
        c = torch.arange(num_classes).float()
        self.register_buffer("dist_matrix", torch.abs(c.unsqueeze(0) - c.unsqueeze(1)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dists = self.dist_matrix[targets]
        loss_dist = torch.sum(probs * dists, dim=1).mean()
        return loss_ce + self.lambda_dist * loss_dist


class CramerDistanceLoss(nn.Module):
    """CE (suavizado 0.10) + Cramer-von Mises sobre la CDF (scripts/v43_01)."""

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


class WeightedCramerLoss(nn.Module):
    """CE (suavizado 0.10) + Cramer ponderada por frontera (scripts/v47_01:96-120)."""

    def __init__(self, num_classes: int = 6, lambda_cramer: float = 0.50, smoothing: float = 0.10,
                 gamma: list[float] | None = None):
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
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]
        k_indices = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        cdf_target = (targets.unsqueeze(1) <= k_indices).float()
        diff_sq = (cdf_pred - cdf_target) ** 2
        weighted_sq = diff_sq * self.gamma.unsqueeze(0)
        loss_cramer = torch.mean(torch.sum(weighted_sq, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer


class ConsistentSmoothedCramerLoss(nn.Module):
    """CE (suavizado 0.10) + Cramer sobre CDF objetivo suavizada (scripts/v51_01:53-74; v53_01:145-163)."""

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
        cdf_target = ((1.0 - self.smoothing) * cdf_step
                      + self.smoothing * (k_indices + 1.0) / self.num_classes)
        loss_cramer = torch.mean(torch.sum((cdf_pred - cdf_target) ** 2, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer


def _construir_modelo(nombre: str, n_features: int, n_clases: int) -> nn.Module:
    """Crea la arquitectura exacta del componente."""
    tipo = CONFIGS[nombre]["modelo"]
    if tipo == "tabular":
        return TabularResNet(in_features=n_features, hidden=256, num_classes=n_clases,
                             drop=0.2)
    if tipo == "dense_se":
        return TabularDenseNet(in_features=n_features, init_dim=192, growth_rate=128,
                               num_classes=n_clases, drop=0.20)
    return SETabularResNet(in_features=n_features, hidden=256, num_classes=n_clases,
                           drop=0.2, reduction=4)


def _construir_loss(nombre: str, n_clases: int) -> nn.Module:
    """Crea la perdida exacta del componente."""
    tipo = CONFIGS[nombre]["loss"]
    if tipo == "ce":
        return nn.CrossEntropyLoss()
    if tipo == "ordinal_ce":
        return OrdinalCrossEntropyLoss(num_classes=n_clases, lambda_dist=0.2)
    if tipo == "ordinal_dist":
        return OrdinalDistanceLoss(num_classes=n_clases, lambda_dist=0.2)
    if tipo == "smooth_ordinal":
        return SmoothOrdinalDistanceLoss(num_classes=n_clases, lambda_dist=0.2, smoothing=0.10)
    if tipo == "cramer":
        return CramerDistanceLoss(num_classes=n_clases, lambda_cramer=0.50, smoothing=0.10)
    if tipo == "weighted_cramer":
        return WeightedCramerLoss(num_classes=n_clases, lambda_cramer=0.50, smoothing=0.10,
                                  gamma=[1.2, 1.5, 1.3, 1.1, 1.0])
    if tipo == "consistent_cramer":
        return ConsistentSmoothedCramerLoss(num_classes=n_clases, lambda_cramer=0.50,
                                            smoothing=0.10)
    raise ValueError(f"loss desconocida: {tipo}")


def _semilla(nombre: str, fold: int) -> int:
    """Semilla exacta por fold de cada componente."""
    return comun.SEED + fold if CONFIGS[nombre]["seed"] == "42+fold" else comun.SEED + fold * 100


def _entrenar_una(dispositivo, cfg: dict, semilla: int, X_tr: np.ndarray, y_tr: np.ndarray,
                   n_features: int, n_clases: int, nombre: str) -> nn.Module:
    """Entrena un modelo con la receta exacta del componente (con o sin SWA)."""
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    base_model = _construir_modelo(nombre, n_features, n_clases).to(dispositivo)
    criterion = _construir_loss(nombre, n_clases).to(dispositivo)
    optimizer = optim.AdamW(base_model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = None
    if cfg["coseno"]:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epocas"], eta_min=cfg["eta_min"])
    ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True)
    swa_model = AveragedModel(base_model) if cfg["swa_start"] is not None else None
    for ep in range(cfg["epocas"]):
        base_model.train()
        for bx, by in loader:
            bx, by = bx.to(dispositivo), by.to(dispositivo)
            optimizer.zero_grad()
            out = base_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if swa_model is not None and ep >= cfg["swa_start"]:
            swa_model.update_parameters(base_model)
    if swa_model is not None:
        if cfg.get("bn_batch") is None:
            update_bn(loader, swa_model, device=dispositivo)
        else:
            loader_bn = DataLoader(ds, batch_size=cfg["bn_batch"], shuffle=False)
            update_bn(loader_bn, swa_model, device=dispositivo)
        modelo = swa_model
    else:
        modelo = base_model
    modelo.eval()
    return modelo


def _predecir(modelo: nn.Module, X: np.ndarray, dispositivo) -> np.ndarray:
    """Softmax en eval sobre una matriz (sin gradientes)."""
    with torch.no_grad():
        tensor = torch.from_numpy(X).float().to(dispositivo)
        return torch.softmax(modelo(tensor), dim=1).cpu().numpy()


def entrenar_componente(nombre: str, X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray,
                        fold_idx: np.ndarray, dispositivo: str | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Reentrena un componente V38-V53 y devuelve (oof, test) con las formas originales."""
    cfg = CONFIGS[nombre]
    if dispositivo is None:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    n, n_features = X_train.shape
    n_test = X_test.shape[0]
    n_clases = len(comun.CLASSES)
    oof = np.zeros((n, n_clases), dtype=np.float32)
    test = np.zeros((n_test, n_clases), dtype=np.float32)
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr])
        X_val = scaler.transform(X_train[va])
        X_te = scaler.transform(X_test)
        if cfg.get("clip") is not None:
            X_tr = np.clip(X_tr, -cfg["clip"], cfg["clip"])
            X_val = np.clip(X_val, -cfg["clip"], cfg["clip"])
            X_te = np.clip(X_te, -cfg["clip"], cfg["clip"])
        modelo = _entrenar_una(dispositivo, cfg, _semilla(nombre, fold), X_tr, y[tr],
                               n_features, n_clases, nombre)
        oof[va] = _predecir(modelo, X_val, dispositivo)
        if cfg["test"] == "media":
            test += _predecir(modelo, X_te, dispositivo) / 5.0
    if cfg["test"] == "full":
        scaler = StandardScaler()
        X_full = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        modelo = _entrenar_una(dispositivo, cfg, comun.SEED, X_full, y, n_features, n_clases,
                               nombre)
        test = _predecir(modelo, X_te, dispositivo)
    return oof, test


def hgb_componente(X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray,
                   fold_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V39-HGB: HistGradientBoosting por fold (OOF) y full-train (test)."""
    n = len(y)
    oof = np.zeros((n, len(comun.CLASSES)), dtype=np.float64)
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=31, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0, random_state=comun.SEED + fold)
        clf.fit(X_train[tr], y[tr])
        oof[va] = clf.predict_proba(X_train[va])
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=31, learning_rate=0.05,
        min_samples_leaf=50, l2_regularization=1.0, random_state=comun.SEED)
    clf.fit(X_train, y)
    test = clf.predict_proba(X_test)
    return oof, test
