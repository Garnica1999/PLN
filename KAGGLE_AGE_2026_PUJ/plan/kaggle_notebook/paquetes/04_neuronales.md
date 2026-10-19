# 04 — Paquete de porte: componentes neuronales V38–V43 (+ contexto V44/V46/V47)

**Alcance:** re-ejecutar en el notebook Kaggle los miembros neuronales del campeón V48 (V43, V42, V40, V39-ordinal) y el HGB de V39. V44/V46/V47 entran como contexto (fracasos/ablaciones).
**Fuentes:** `scripts/v38_01..v47_01_*.py`, `experiments/v*_01_*.json`, `artifacts/p33_scores/todos/manifest.json`, `scripts/v48_01_build_submission.py`, `scripts/v51_01_consistent_smoothed_cramer_5folds.py`. Todo el código citado es ASCII, tal cual aparece en el repo.

## 0. Denominadores comunes

- Datos de entrada (todos los scripts):

```python
car = ROOT / "artifacts" / "p33_scores" / "todos"
y_true = np.load(car / "y_true.npy")          # (169336,) int64, clases 0..5
fold_idx = np.load(car / "fold_idx.npy")      # (169336,) int64, valores 0..4
v30_prob = np.load(car / "v30_prob.npy")      # (169336, 6) referencia
w_plugin = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
```

- `fold_idx` viene del manifest de `artifacts/p33_scores/todos`: *"StratifiedGroupKFold, n_splits=5, n_repeats=1, scheme=group, grupos=dup_clusters, semilla 42. **Los mismos folds que V20 y V30**"*. Conteos por fold: 33868/33868/33867/33866/33867. Uso idéntico en los 9 scripts:

```python
train_m = fold_idx != fold      # o tr_mask = (fold_idx != fold)
val_m = fold_idx == fold        # o val_mask = (fold_idx == fold)
```

- Features de train — `matriz_279()` de `scripts/v19_b1_boosting.py`:

```python
base = np.load(cfg.FEATURES / "meta_features_train_v11.npy")   # (169336, 273) float64
extra = np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")     # (169336, 6) float32
matriz = np.concatenate([base.astype(np.float32), extra.astype(np.float32)], axis=1)
assert matriz.shape[1] == 279
```

- Features de test: `cfg.FEATURES / "meta_features_test_v12_fresca.npy"` (112892, 279) float64; V40/V41/V42/V43/V44/V46/V47 hacen `np.load(MATRIZ_TEST).astype(np.float32)` y V40/V41 además `assert X_test_raw.shape == (112892, 279)`.
- `cfg.SEED = 42` (`src/config.py`). Cada script llama `autotest_numerico()` al inicio (verificación numérica previa).
- Evaluación de folds siempre con la regla plug-in: `np.argmax(prob * w_plugin, axis=1)` + `f1_score(..., average="macro")` (excepción de reporte: V47 evalúa con `np.argmax(prob, axis=1)` sin plug-in).

## 1. Tabla resumen

| Comp. | Arquitectura | Loss | Seeds (fold / full) | Épocas·batch·lr | SWA | Salidas (forma) | Runtime |
|---|---|---|---|---|---|---|---|
| V38 | `TabularResNet` 279→256, BN+SiLU, 2 bloques residuales, drop 0.2, head 6 | `nn.CrossEntropyLoss()` | `torch.manual_seed(42 + fold)` / `torch.manual_seed(42)` | 14 · 256 · AdamW 1e-3 wd 1e-4, CosineAnnealingLR(T_max=14) | No | `v38_oof_tabularnet.npy` (169336,6); `v38_test_prob_tabularnet.npy` (112892,6) | 97.1 s |
| V39 red | `TabularResNet` idéntica | `OrdinalCrossEntropyLoss(6, lambda_dist=0.2)` | `torch.manual_seed(42 + fold)` / `torch.manual_seed(42)` | 14 · 256 · AdamW 1e-3 wd 1e-4, CosineAnnealingLR(T_max=14) | No | `v39_oof_tabnet_ordinal.npy` (169336,6); `v39_test_tabnet_ordinal.npy` (112892,6) | 192.3 s (todo V39) |
| V39 HGB | `HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, lr=0.05, min_samples_leaf=50, l2=1.0)` | (implícita del sklearn) | `random_state=42 + fold` / `random_state=42` | 200 iter · — · 0.05 | No | `v39_oof_hgb.npy` (169336,6); `v39_test_hgb.npy` (112892,6) float64 | — |
| V40 | `SETabularResNet`: +Squeeze-and-Excitation (reduction=4) en ambos bloques | `OrdinalDistanceLoss(6, lambda_dist=0.2)` | `torch.manual_seed(cfg.SEED + fold*100)` + `np.random.seed(idem)` | 12 · 256 · AdamW 1e-3 wd 1e-4, sin scheduler | No | `v40_oof_se_tabularnet.npy`; `v40_test_se_tabularnet.npy` (112892,6) | 122.6 s |
| V41 | `SETabularResNet` (igual que V40) | `OrdinalDistanceLoss(6, lambda_dist=0.2)` | `cfg.SEED + fold*100` (torch y numpy) | 16 · 256 · AdamW 1e-3 wd 1e-4, sin scheduler | `AveragedModel`, `swa_start=8` (8 ckpts) | `v41_oof_swa_tabularnet.npy`; `v41_test_swa_tabularnet.npy` | 177.17 s |
| V42 | `SETabularResNet` (SE) | `SmoothOrdinalDistanceLoss(6, lambda_dist=0.2, smoothing=0.10)` | `cfg.SEED + fold*100` | 16 · 256 · AdamW 1e-3 wd 1e-4, CosineAnnealingLR(T_max=16, eta_min=1e-4) | `AveragedModel`, `swa_start=6` (10 ckpts) + `update_bn` | `v42_oof_smooth_swa_tabularnet.npy`; `v42_test_smooth_swa_tabularnet.npy` | 190.84 s |
| V43 | `SETabularResNet` (SE) | `CramerDistanceLoss(6, lambda_cramer=0.50, smoothing=0.10)` | `cfg.SEED + fold*100` | 16 · 256 · AdamW 1e-3 wd 1e-4, CosineAnnealingLR(T_max=16, eta_min=1e-4) | `AveragedModel`, `swa_start=6` (10 ckpts) + `update_bn` | `v43_oof_cramer_swa_tabularnet.npy`; `v43_test_cramer_swa_tabularnet.npy` | 200.69 s |
| V44 (ctx) | `SETabularResNet` (SE) | `BoundaryEMDLoss(6, lambda_loss=0.50, smoothing=0.10, gamma=[1.0,1.4,1.4,1.1,1.0])` | `cfg.SEED + fold*100` | 16 · 256 · AdamW 1e-3 wd 1e-4, Cosine(T_max=16, eta_min=1e-4) | `swa_start=6` (10 ckpts) + `update_bn` | `v44_oof_boundary_emd_swa_tabularnet.npy`; `v44_test_boundary_emd_swa_tabularnet.npy` | 209.47 s |
| V46 (ctx) | `SETabularResNet` (SE) | `CramerDistanceLoss(6, 0.50, 0.10)` (igual que V43) | `cfg.SEED + fold*100`; RankGauss `random_state=cfg.SEED + fold` | 16 · 256 · AdamW 1e-3 wd 1e-4, Cosine(T_max=16, eta_min=1e-4) | `swa_start=6` (10 ckpts) + `update_bn` | `v46_oof_rankgauss_cramer_swa_tabularnet.npy`; `v46_test_rankgauss_cramer_swa_tabularnet.npy` | 196.01 s |
| V47 (ctx) | `SETabularResNet` (SE) | `WeightedCramerLoss(6, lambda_cramer=0.50, smoothing=0.10, gamma=[1.2,1.5,1.3,1.1,1.0])` | `cfg.SEED + fold*100` | 16 · 256 · AdamW 1e-3 wd 1e-4, Cosine(T_max=16, eta_min=1e-4) | `swa_start=6` (10 ckpts) + `update_bn` (loader 512, shuffle=False) | `v47_oof_weighted_cramer_clipped_tabularnet.npy`; `v47_test_weighted_cramer_clipped_tabularnet.npy` | 152.62 s |

Todos los .npy de OOF/test son `(169336, 6)` y `(112892, 6)`, `float32` (excepto `v39_test_hgb.npy` float64).

## 2. Detalle por componente

### V38 — `scripts/v38_01_tabularnet_5folds.py`
Arquitectura exacta:

```python
class TabularResNet(nn.Module):
    def __init__(self, in_features: int, hidden: int = 256, num_classes: int = 6, drop: float = 0.2):
        self.in_proj = nn.Linear(in_features, hidden); self.bn0 = nn.BatchNorm1d(hidden); self.act0 = nn.SiLU()
        self.fc1 = nn.Linear(hidden, hidden); self.bn1 = nn.BatchNorm1d(hidden); self.act1 = nn.SiLU(); self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, hidden); self.bn2 = nn.BatchNorm1d(hidden); self.act2 = nn.SiLU(); self.drop2 = nn.Dropout(drop)
        self.head = nn.Linear(hidden, num_classes)
    def forward(self, x):
        h0 = self.act0(self.bn0(self.in_proj(x)))
        h1 = h0 + self.drop1(self.act1(self.bn1(self.fc1(h0))))
        h2 = h1 + self.drop2(self.act2(self.bn2(self.fc2(h1))))
        return self.head(h2)
```

Preprocesamiento in-fold: `StandardScaler` ajustado SOLO con `X[train_m]`, aplicado a `X[val_m]`. Entrenamiento: `epochs = 14`, `batch_size = 256`, `optim.AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR(optimizer, T_max=epochs)`, seed por fold `torch.manual_seed(42 + fold)`. OOF: softmax del modelo de fold.
**Test: NO promedia folds**; si el gate pasa (`ic95[0] > 0 and (best_macro_blend - macro_v30) >= 0.0003`), entrena un modelo final con todo el train (`torch.manual_seed(42)`, mismo scaler reajustado sobre todo X) y guarda `v38_test_prob_tabularnet.npy`. Grid de alpha: `[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]`; resultado: alpha=0.12, delta=0.00068, gate superado.

### V39 — `scripts/v39_01_trio_ensemble_5folds.py`
Red idéntica a V38 pero loss ordinal:

```python
class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int = 6, lambda_dist: float = 0.2):
        self.ce = nn.CrossEntropyLoss(); self.lambda_dist = lambda_dist
        classes = torch.arange(num_classes, dtype=torch.float32)
        dist_mat = torch.abs(classes.unsqueeze(0) - classes.unsqueeze(1))
        self.register_buffer("dist_mat", dist_mat)
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        batch_dists = self.dist_mat[targets]
        dist_loss = torch.sum(probs * batch_dists, dim=1).mean()
        return ce_loss + self.lambda_dist * dist_loss
```

HGB in-fold (config exacta):

```python
clf = HistGradientBoostingClassifier(
    max_iter=200, max_leaf_nodes=31, learning_rate=0.05,
    min_samples_leaf=50, l2_regularization=1.0, random_state=42 + fold)
clf.fit(X[train_m], y_true[train_m]); hgb_prob = clf.predict_proba(X[val_m])
```

Red: `epochs = 14`, batch 256, AdamW 1e-3/1e-4, CosineAnnealingLR(T_max=14), `torch.manual_seed(42 + fold)`. Pesos del trío: grid `a in [0.08,0.10,0.12,0.14,0.16] x b in [0.02,0.04,0.06,0.08]`, `c = 1.0 - a - b`; óptimo final `{"v30": 0.84, "tabnet_ordinal": 0.12, "hgb": 0.04}`, `delta_vs_v38 = +0.000144 > 0` → gate pasa. Test full: HGB con `random_state=42` sobre todo X → `v39_test_hgb.npy`; red con `torch.manual_seed(42)`, scaler reajustado y loss ordinal → `v39_test_tabnet_ordinal.npy` (tampoco promedia folds).

### V40 — `scripts/v40_01_se_tabularnet_5folds.py`
`SETabularResNet`: `in_proj`/`bn0`/`SiLU`, y por cada bloque `fc`/`bn`/`SiLU`/`drop` + SE:
`se_fc1 = nn.Linear(hidden, hidden // reduction)` (256→64), `se_act = SiLU`, `se_fc2 = Linear(64, 256)`, `se_sig = Sigmoid`;
`res = drop(act(bn(fc(h))))`, `se = sig(fc2(act(fc1(res))))`, `h_next = h + res * se`. `hidden=256, drop=0.2, reduction=4`.
Loss:

```python
class OrdinalDistanceLoss(nn.Module):
    def __init__(self, num_classes=6, lambda_dist=0.2):
        self.ce = nn.CrossEntropyLoss(); self.lambda_dist = lambda_dist
        c = torch.arange(num_classes).float()
        self.register_buffer("dist_matrix", torch.abs(c.unsqueeze(0) - c.unsqueeze(1)))
    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dists = self.dist_matrix[targets]
        loss_dist = torch.sum(probs * dists, dim=1).mean()
        return loss_ce + self.lambda_dist * loss_dist
```

In-fold: `StandardScaler` fit en `X[tr_mask]`; aplicado a val y a `X_te_scaled`. Entrenamiento: 12 épocas, batch 256, AdamW 1e-3/1e-4, sin scheduler, seed `cfg.SEED + fold*100` (torch y numpy). Test: inferencia por fold `test_probs += te_p / 5.0` (media de 5 modelos, sin update_bn). Gate de adopción en V48 lo decide el blend, no este script.

### V41 — `scripts/v41_01_swa_tabularnet_5folds.py`
Igual que V40 pero SWA: `epochs = 16`, `swa_start = 8`, `batch_size = 256`, AdamW 1e-3/1e-4, sin scheduler.

```python
base_model = SETabularResNet(in_features=279, hidden=256, num_classes=6, drop=0.2, reduction=4).to(dispositivo)
swa_model = AveragedModel(base_model)
...
for ep in range(epochs):
    ...entrenar...
    if ep >= swa_start:
        swa_model.update_parameters(base_model)
update_bn(loader_train, swa_model, device=dispositivo)   # loader_train: batch 256, shuffle=True
```

8 actualizaciones SWA (épocas 8..15), promediado uniforme por defecto. Inferencia con `swa_model.eval()`; test `+= te_p / 5.0`.

### V42 — `scripts/v42_01_smooth_swa_tabularnet_5folds.py`
`SETabularResNet` + `SmoothOrdinalDistanceLoss` (smoothing 0.10 + distancia ordinal 0.2):

```python
self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)   # smoothing = 0.10
...
return loss_ce + self.lambda_dist * loss_dist
```

Hiperparámetros (documentados en el JSON): `batch_size=256, epochs=16, swa_start=6, lr=1e-3, lr_min=1e-4, smoothing=0.10, lambda_dist=0.20, weight_decay=1e-4`; scheduler `CosineAnnealingLR(T_max=epochs, eta_min=lr_min)` aplicado cada época. 10 checkpoints SWA (épocas 6..15) + `update_bn(loader_train, ...)`. Seed `cfg.SEED + fold*100`. Test media de folds `+= te_p / 5.0`. OOF standalone global: 0.613276.

### V43 — `scripts/v43_01_cramer_swa_tabularnet_5folds.py`
`SETabularResNet` + pérdida de Cramér (W2 1D sobre CDF) con smoothing 0.10:

```python
class CramerDistanceLoss(nn.Module):
    def __init__(self, num_classes=6, lambda_cramer=0.50, smoothing=0.10):
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.lambda_cramer = lambda_cramer
    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]
        k_indices = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0)
        cdf_target = (targets.unsqueeze(1) <= k_indices).float()
        loss_cramer = torch.mean(torch.sum((cdf_pred - cdf_target) ** 2, dim=1))
        return loss_ce + self.lambda_cramer * loss_cramer
```

Entrenamiento idéntico a V42 (16/256/1e-3/1e-4, Cosine eta_min 1e-4, swa_start=6, 10 ckpts, `update_bn`, seed `cfg.SEED + fold*100`, test `+= te_p / 5.0`). OOF standalone global: 0.612840. **Miembro del V48 con peso 0.0832.**

### V44 — `scripts/v44_01_boundary_emd_swa_tabularnet_5folds.py` (contexto)
`BoundaryEMDLoss`: CE(label_smoothing=0.10) + `lambda_loss=0.50` * EMD L1 ponderada por fronteras `gamma = [1.0, 1.4, 1.4, 1.1, 1.0]` normalizada por su media (`gamma / gamma.mean()`):

```python
diff_abs = torch.abs(cdf_pred - cdf_target)
loss_ot = torch.mean(torch.sum(diff_abs * self.gamma.unsqueeze(0), dim=1))
return loss_ce + self.lambda_loss * loss_ot
```

Resto igual a V42/V43 (16/256/1e-3/1e-4, Cosine, swa_start=6, update_bn, seed fold*100, media/5). OOF 0.612903.

### V46 — `scripts/v46_01_rankgauss_cramer_swa_tabularnet_5folds.py` (contexto)
Mismo modelo y `CramerDistanceLoss(0.50, 0.10)` que V43, pero preprocesamiento RankGauss in-fold:

```python
qt = QuantileTransformer(output_distribution="normal", n_quantiles=2000, random_state=cfg.SEED + fold, subsample=100000)
X_tr = qt.fit_transform(X_train_raw[tr_mask])
X_val = qt.transform(X_train_raw[val_mask])
X_te_scaled = qt.transform(X_test_raw)
```

OJO: `random_state=cfg.SEED + fold` (42+fold), no `42 + fold*100`. OOF 0.605551 (el peor de la familia; fracaso).

### V47 — `scripts/v47_01_weighted_cramer_clipped_swa_tabularnet_5folds.py` (contexto)
`WeightedCramerLoss` (gamma SIN normalizar `[1.2, 1.5, 1.3, 1.1, 1.0]`) + recorte robusto post-escalado:

```python
X_tr = scaler.fit_transform(X_train_raw[tr_mask]); X_val = scaler.transform(X_train_raw[val_mask]); X_te = scaler.transform(X_test_raw)
X_tr = np.clip(X_tr, -clip_val, clip_val); X_val = np.clip(X_val, -clip_val, clip_val); X_te_scaled = np.clip(X_te, -clip_val, clip_val)
```

`clip_val = 6.0`. SWA igual (swa_start=6, 10 ckpts) pero `update_bn` con loader NUEVO `DataLoader(ds_train, batch_size=512, shuffle=False)`; inferencia por DataLoader batch 512; test `+= fold_test_probs / 5.0`. Evaluación de folds con `argmax` crudo (sin w_plugin). OOF 0.611082.

## 3. Integración en el campeón V48 (para el notebook)

`scripts/v48_01_build_submission.py` consume EXACTAMENTE estos 6 ficheros de test y aplica Product-of-Experts geométrico:

```python
TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_V43 = cfg.FEATURES / "v43_test_cramer_swa_tabularnet.npy"
TEST_V42 = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
TEST_V40 = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"

WEIGHTS_GEOM = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])   # normalizado por suma
eps = 1e-12
log_models = [np.log(np.clip(m, eps, 1.0)) for m in models]
log_geom = sum(w * lm for w, lm in zip(WEIGHTS_GEOM, log_models))
p_geom = np.exp(log_geom - np.max(log_geom, axis=1, keepdims=True)); p_geom /= p_geom.sum(axis=1, keepdims=True)
pred_indices = np.argmax(p_geom * PESOS_PLUG_IN, axis=1)
```

Orden de pesos: V30=0.5628, V43=0.0832, V42=0.1724, V40=0.0754, V39-ordinal=0.0513, V39-HGB=0.0549. Nota: V38, V41, V44, V46, V47 NO entran en el V48 (V38 fue reemplazado por el trío V39). El notebook debe reproducir los 4 componentes neuronales + HGB y el blend V48 (SHA256 de submission: `bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62`).

## 4. Checklist de fidelidad para el porte

1. `matriz_279` = `meta_features_train_v11.npy` (273) + `oof_ADV07_nbsvm_word.npy` (6), ambos cast a float32.
2. `fold_idx` canónico (StratifiedGroupKFold, dup_clusters, seed 42) — NO regenerar con otro esquema.
3. Seeds: V38/V39 usan `42 + fold` (torch, y `42` en el modelo full); V40–V47 usan `cfg.SEED + fold*100` (torch y numpy); V46 RankGauss usa `42 + fold`. No homogeneizar.
4. V38/V39 guardan test de UN modelo full-train; V40–V47 promedian 5 modelos de fold (`suma/5`).
5. SWA: `AveragedModel` default; V41 8 ckpts (swa_start=8), V42–V47 10 ckpts (swa_start=6); `update_bn` con loader de train (V41–V46 shuffle=True; V47 batch 512 shuffle=False).
6. Escalado SIEMPRE in-fold (`fit` solo en máscara de train del fold); test se transforma con el scaler del fold.
7. Evaluación/mezcla final con plug-in `w_plugin = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`.
8. dtypes de salida float32 salvo `v39_test_hgb.npy` (float64, sklearn); formas (169336,6) y (112892,6).
9. Rutas runtime documentado (segundos): 97.1 / 192.3 / 122.6 / 177.17 / 190.84 / 200.69 / 209.47 / 196.01 / 152.62 (V38→V47).
10. Los 4 modelos V48 neuronales + HGB suman ~706 s documentados (sin contar V30 ni el ensamble).
