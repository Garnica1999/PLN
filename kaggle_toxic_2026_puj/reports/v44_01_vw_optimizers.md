# V44-01: Vowpal Wabbit con FTRL, Coin Betting y PiSTOL

Generado por `experiments/v44_01_vw_optimizers.py` (V44-01 (Vowpal Wabbit: FTRL, Coin Betting y PiSTOL sobre el TF-IDF de V03)).

**Estado `OOF_EVALUATED`. API: vowpalwabbit.sklearn.VWClassifier (API scikit-learn oficial). VW 9.11.8.**

## 1. Configuracion

- {'loss_function': 'logistic', 'link': 'logistic', 'passes': 10, 'random_seed': 42, 'power_t': 0.5, 'learning_rate': 0.5}
- Tres optimizadores: FTRL (`ftrl=True`), Coin Betting (`coin_betting=True`), PiSTOL (`pistol=True`).
- Misma representacion TF-IDF de V03, ajustada dentro de cada training fold.

## 2. Resultados

| estrategia | corrected_cv | Δ vs V03 | train | gap | rescates | danios | net | correlacion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FTRL | 0.597204 | -0.180007 | 0.6218 | +0.0246 | 355 | 1857 | -1502 | 0.5028 |
| Coin Betting | 0.629918 | -0.147294 | 0.6967 | +0.0668 | 335 | 1631 | -1296 | 0.5581 |
| PiSTOL | 0.636881 | -0.140331 | 0.7198 | +0.0829 | 332 | 1597 | -1265 | 0.5664 |

**Mejor: PiSTOL con 0.636881.**

**DECISION: SPARSE_ONLINE_OPTIMIZATION = CLOSED**

## 3. Lo que este script no hace

- No envia submission.
- No hace grid de hiperparametros.
- No modifica V03.

Coste: 30.0 s, memoria 273 MB, 3 modelos, 0 submissions.

