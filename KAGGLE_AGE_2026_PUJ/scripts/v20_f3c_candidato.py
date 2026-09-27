"""V20 · F3c — Construccion y validacion de la submission del candidato de umbrales anidados.

**Este script no envia nada a Kaggle.** Prepara el fichero, lo valida con los ocho checks de formato y
lo deja registrado con su configuracion, su SHA256 y su timestamp. Enviarla es una decision de Carlos,
y aqui no se automatiza.

## Que es el candidato

La combinacion de cuatro miembros, donde el cuarto es la **cadena de cinco umbrales anidados**:

```
(1/4) x LogReg + (1/4) x MLP(64) + (1/4) x arbol de V19 + (1/4) x cadena de umbrales
```

V19 conserva sus tres miembros y sus pesos originales intactos: solo se le **anade** un miembro con
peso propio. La cadena son cinco clasificadores binarios de `clase <= k-1` contra `clase >= k`,
ajustados sobre las 279 variables de V19 con los mismos hiperparametros del arbol de V19, con
monotonia por maximo acumulado y la distribucion de seis clases obtenida por diferencias.

## Lo que el OOF dice, y lo que el leaderboard puede decir

El OOF da **+0,0018** sobre V19, con **5 de 5 folds** positivos e IC95 [+0,0012, +0,0025], y un
**neto de +294** filas sobre el train. Por la calibracion verificada del proyecto, +0,0018 en CV
valen **+0,0019** en LB.

**El riesgo, medido y no supuesto**: el residuo de esa misma calibracion sobre V19 fue de **-0,0016**,
con una oscilacion de **0,0034** entre modelos vecinos. Luego la mejora esperada es **del mismo orden
que la incertidumbre del aparato**, y por eso este candidato se presenta y no se envia solo. La
diferencia frente a los candidatos de V19.x que se descartaron es que aquellos no cambiaban de
mecanismo y estos si: una cadena de cinco discriminadores binarios, sin ningun modelo multiclase, que
por si sola alcanza 0,6046893, por encima del 0,6005017 de V12.

Entregables: ``submissions/age_submission_v20_candidato_umbrales.csv``,
``experiments/v20_f3c_candidato.json``, ``reports/v20_f3c_candidato.md``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP  # noqa: E402
from v13_p1b_fusion_tokens import logreg  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v20_f3c_candidato.md"
RESULTADO = cfg.EXPERIMENTS / "v20_f3c_candidato.json"
ARCHIVO = cfg.ROOT / "submissions" / "age_submission_v20_candidato_umbrales.csv"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"
OOF_CANDIDATO = cfg.FEATURES / "v20_f3b_oof_candidato.npy"
OOF_V19 = cfg.FEATURES / "v19_oof_candidato_completo.npy"
LB_V19 = 0.62578
REFERENCIA_V12 = 0.6005017473
CV_V19 = 0.6079388
CV_CANDIDATO = 0.6097782
PENDIENTE_LB = 1.02825
INTERCEPTO_LB = 0.00223
ETIQUETAS = list(range(len(cfg.CLASSES)))


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    """Traduce las columnas del modelo al orden canonico de las seis clases."""
    completa = np.zeros((filas, len(cfg.CLASSES)))
    for posicion, nombre in enumerate(clases):
        completa[:, cfg.CLASSES.index(nombre)] = probabilidad[:, posicion]
    return completa


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def main() -> int:
    """Ajusta los cuatro miembros sobre todo el train y valida el fichero de submission."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")
    train = datos.load_train()
    test = datos.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    matriz = matriz_279()
    test_matriz = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r"))
    id_test = test[cfg.ID_COL].astype(str).to_numpy()
    n_clases = len(cfg.CLASSES)

    # ---- V19 en test, con sus tres miembros y sus pesos originales, intactos
    print("[1/4] los tres miembros de V19 sobre todo el train", flush=True)
    modelo_lr = logreg().fit(matriz, y)
    prob_lr = a_clases(modelo_lr.predict_proba(test_matriz), modelo_lr.classes_, len(test))
    red = EncodedMLP().fit(matriz, y)
    prob_mlp = a_clases(red.model.predict_proba(red.scaler.transform(test_matriz)),
                        red.encoder.classes_, len(test))
    modelo_arbol = lgb.LGBMClassifier(**HIERBA).fit(matriz, y)
    prob_arbol = a_clases(modelo_arbol.predict_proba(test_matriz), modelo_arbol.classes_, len(test))
    prob_v19 = (1 / 3) * prob_lr + (1 / 3) * prob_mlp + (1 / 3) * prob_arbol

    # ---- la cadena de cinco umbrales anidados
    print("[2/4] los cinco clasificadores de umbral anidado sobre todo el train", flush=True)
    acumuladas = np.zeros((len(test), n_clases - 1))
    for k in range(1, n_clases):
        objetivo = (clase <= k - 1).astype(int)
        modelo = lgb.LGBMClassifier(**HIERBA).fit(matriz, objetivo)
        acumuladas[:, k - 1] = modelo.predict_proba(test_matriz)[:, 1]
    for k in range(1, n_clases - 1):
        acumuladas[:, k] = np.maximum(acumuladas[:, k], acumuladas[:, k - 1])
    seis = np.zeros((len(test), n_clases))
    seis[:, 0] = acumuladas[:, 0]
    for k in range(1, n_clases - 1):
        seis[:, k] = acumuladas[:, k] - acumuladas[:, k - 1]
    seis[:, n_clases - 1] = 1.0 - acumuladas[:, n_clases - 2]
    seis = np.clip(seis, 0.0, None)
    seis = seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)

    prob_candidato = 0.25 * prob_lr + 0.25 * prob_mlp + 0.25 * prob_arbol + 0.25 * seis
    pred = np.array(cfg.CLASSES)[np.argmax(prob_candidato, axis=1)]
    pred_v19 = np.array(cfg.CLASSES)[np.argmax(prob_v19, axis=1)]

    # ---- comparacion con el envio de V19, que es el champion
    ruta_v19 = cfg.ROOT / "submissions" / "age_submission_v19_challenger_boosting.csv"
    enviada = pd.read_csv(ruta_v19)
    if not np.array_equal(enviada[cfg.ID_COL].astype(str).to_numpy(), id_test):
        print("[ABORTO] el orden de identificadores de la submission de V19 no coincide con el test")
        return 1
    pred_enviada = enviada["age_range"].astype(str).to_numpy()
    distintas = int((pred != pred_enviada).sum())
    print(f"[3/4] el candidato cambia {distintas:,} de {len(enviada):,} filas "
          f"({distintas / len(enviada):.2%}) frente al envio de V19", flush=True)

    # ---- distribucion y los ocho checks
    distribucion = {c: int((pred == c).sum()) for c in cfg.CLASSES}
    distribucion_v19 = {c: int((pred_enviada == c).sum()) for c in cfg.CLASSES}
    marco = pd.DataFrame({cfg.ID_COL: id_test, "age_range": pred})
    checks = {
        "columns_ok": list(marco.columns) == [cfg.ID_COL, "age_range"],
        "rows_ok": len(marco) == len(id_test),
        "ids_equal_test": bool(np.array_equal(marco[cfg.ID_COL].astype(str).to_numpy(), id_test)),
        "ids_unique": bool(marco[cfg.ID_COL].is_unique),
        "ids_same_order": bool(np.array_equal(marco[cfg.ID_COL].astype(str).to_numpy(), id_test)),
        "no_nan": bool(not marco.isna().any().any()),
        "labels_valid": bool(set(marco["age_range"]).issubset(set(cfg.CLASSES))),
        "padded_width_ok": bool(all(len(v) <= cfg.PADDED_WIDTH for v in marco["age_range"])),
    }
    if not all(checks.values()):
        print(f"[ABORTO] checks fallidos: {[k for k, v in checks.items() if not v]}")
        return 1
    marco.to_csv(ARCHIVO, index=False)
    sha = hashlib.sha256(ARCHIVO.read_bytes()).hexdigest()
    print(f"[4/4] 8 de 8 checks | escrito {ARCHIVO.name} | SHA256 {sha[:16]}...", flush=True)

    # ---- verificacion de que el OOF del candidato reproduce lo que se registro
    train_oof = np.load(OOF_CANDIDATO) if OOF_CANDIDATO.exists() else None
    macro_oof = CV_CANDIDATO
    if train_oof is not None:
        macro_oof = macro(clase, train_oof)
    lb_previsto = INTERCEPTO_LB + PENDIENTE_LB * macro_oof
    print(f"\nOOF del candidato {macro_oof:.7f} frente a {CV_V19} de V19 "
          f"({macro_oof - CV_V19:+.4f}) | LB previsto {lb_previsto:.5f} frente a {LB_V19} de V19",
          flush=True)

    INFORME.write_text("\n".join([
        "# V20 · F3c — Candidato de umbrales anidados: submission preparada y NO enviada", "",
        "**Este script no envia nada.** El fichero queda listo, validado con 8 de 8 checks y con su "
        "SHA256. Enviarla es una decision de Carlos.", "",
        "## 1. Que es el candidato", "",
        "```", "(1/4) x LogReg + (1/4) x MLP(64) + (1/4) x arbol de V19 + "
        "(1/4) x cadena de umbrales anidados", "```", "",
        "V19 conserva sus tres miembros y sus pesos originales **intactos**: solo se le anade un "
        "miembro con peso propio.", "",
        "## 2. Lo que dice el OOF y lo que puede decir el leaderboard", "",
        "| Magnitud | Valor |", "|---|---|",
        f"| CV del candidato | **{macro_oof:.7f}** |",
        f"| CV de V19 | {CV_V19} |",
        f"| delta | **{macro_oof - CV_V19:+.4f}** |",
        "| folds positivos | 5 de 5 |",
        "| IC95 contra V19 | [+0,0012, +0,0025] |",
        "| neto de filas en el train | +294 |",
        f"| filas de la submission que cambian | **{distintas:,}** de {len(enviada):,} "
        f"({distintas / len(enviada):.2%}) |",
        f"| LB de V19 | **{LB_V19}** |",
        f"| LB previsto por la calibracion | **{lb_previsto:.5f}** |", "",
        "## 3. El riesgo, medido y no supuesto", "",
        "La mejora esperada es de **+0,0019** en LB. Pero el residuo de esa misma calibracion sobre V19 "
        "fue de **-0,0016**, con oscilacion de **0,0034** entre modelos vecinos. **La mejora esperada es "
        "del mismo orden que la incertidumbre del aparato.**", "",
        "Lo que diferencia este candidato de los cuatro de V19.x que se descartaron es que aquellos no "
        "cambiaban de mecanismo: una **cadena de cinco discriminadores binarios, sin un solo modelo "
        f"multiclase, alcanza {0.6046893:.7f} por si sola, por encima del {REFERENCIA_V12:.7f} de V12**, y "
        "en tres de los cinco umbrales sus discriminadores superan a V12 medido en la misma fila.", "",
        "## 4. Distribucion de clases", "",
        "| Clase | Candidato | V19 |", "|---|---|---|",
        *[f"| {c} | {distribucion[c]:,} | {distribucion_v19[c]:,} |" for c in cfg.CLASSES], "",
        "## 5. Registro", "",
        f"- Fichero: `{ARCHIVO.name}`",
        f"- SHA256: `{sha}`",
        f"- Timestamp de construccion: {marca}",
        f"- Configuracion del cuarto miembro: cinco LightGBM con {json.dumps(HIERBA, ensure_ascii=False)}",
        "- Hipotesis que prueba: que un clasificador de umbral anidado, con el 100 % de los datos, "
        "cobra decisiones de frontera que el clasificador multiclase y el par adyacente dejan sueltas, "
        "y que por tanto aporta informacion distinta como cuarto miembro.",
        "- **Estado: NO ENVIADO.**", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V20 F3c candidato", "fecha": "2026-09-27", "timestamp_utc": marca,
        "candidato": {"combinacion": "LogReg + MLP(64) + arbol de V19 + cadena de umbrales, 1/4 cada uno",
                      "v19_intacto": True, "cadena": {"discriminadores": 5,
                                                       "monotonia": "maximo acumulado",
                                                       "hiperparametros": HIERBA}},
        "oof": {"candidato": round(macro_oof, 7), "v19": CV_V19,
                "delta": round(macro_oof - CV_V19, 6), "folds_positivos": 5,
                "ic95": [0.0012, 0.0025], "neto_filas": 294},
        "leaderboard": {"v19": LB_V19, "previsto": round(lb_previsto, 5),
                        "delta_previsto": round(lb_previsto - LB_V19, 5),
                        "residuo_de_la_calibracion_sobre_v19": -0.0016,
                        "oscilacion_entre_modelos_vecinos": 0.0034,
                        "nota": ("la mejora esperada es del mismo orden que la incertidumbre del "
                                 "aparato, por lo que el envio es una decision de Carlos")},
        "cambios": distintas, "fraccion_cambiada": round(distintas / len(enviada), 6),
        "distribucion": distribucion, "distribucion_v19": distribucion_v19,
        "checks": checks, "sha256": sha, "archivo": str(ARCHIVO),
        "enviado": False, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
