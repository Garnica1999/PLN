"""P88-00: auditoria de novedad de LightGBM sobre la matriz TF-IDF de V03 (directiva 2026-10-09).

La directiva propone arboles potenciados (LightGBM) directamente sobre la matriz
dispersa word+char de V03, con tres clasificadores binarios y umbrales congelados,
y ordena auditar precedentes antes de ejecutar. Si existe un modelo equivalente
con esa representacion y ese mecanismo, la propuesta es DUPLICATE y no se ejecuta.

Veredicto: DUPLICATE. El precedente metodologicamente equivalente es P61-01
(`experiments/p61_03_lightgbm.py`, 2026-09-28): LightGBM de tres clasificadores
binarios sobre el TF-IDF exacto de V03 (27.431 rasgos, CSR), pliegues canonicos y
umbrales [0.44, 0.51, 0.46]. Brazos medidos: por defecto -0.0302397191 y con
Optuna anidado -0.0258463023; 2/5 pliegues; neto -28; gate no superado; familia
cerrada por duplicacion (D-143).

0 entrenamientos, 0 submissions, 0 push. No se modifica ningun artefacto historico.

Genera:
    reports/toxic_lgbm_novelty_audit.md
    reports/toxic_lgbm_novelty_audit.csv
    fila P88-00 en experiments.csv (una sola vez; append de una linea con LF)

Uso:
    /home/carlos/venv/bin/python experiments/p88_00_auditoria_novedad_lgbm.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]

RUTA_P61_SCRIPT: Final[Path] = RAIZ / "experiments" / "p61_03_lightgbm.py"
RUTA_P61_JSON: Final[Path] = RAIZ / "reports" / "p61_lightgbm_capacidad.json"
RUTA_P61_MD: Final[Path] = RAIZ / "reports" / "p61_lightgbm_capacidad.md"
RUTA_V8_13: Final[Path] = RAIZ / "experiments" / "v8_13_clasificadores_alternativos.py"
RUTA_DECISIONS: Final[Path] = RAIZ / "DECISIONS.md"
RUTA_NOTEBOOK: Final[Path] = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_REGISTRO: Final[Path] = RAIZ / "experiments.csv"
RUTA_OOF: Final[Path] = RAIZ / "features" / "oof_v03_final.npz"
RUTA_SUBMISSION: Final[Path] = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
RUTA_MD: Final[Path] = RAIZ / "reports" / "toxic_lgbm_novelty_audit.md"
RUTA_CSV: Final[Path] = RAIZ / "reports" / "toxic_lgbm_novelty_audit.csv"

HASH_OOF: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_SUBMISSION: Final[str] = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
V03_CCV: Final[float] = 0.7772114525165721
FECHA: Final[str] = "2026-10-09"
COLUMNAS_REGISTRO: Final[list[str]] = [
    "experiment_id", "competition", "question", "hypothesis", "preprocessing",
    "feature_groups", "representation", "model", "hyperparameters", "validation",
    "f1", "macro_f1", "weighted_f1", "precision", "recall", "training_time",
    "dimension", "result", "interpretation", "decision", "date", "script", "notes",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P88-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_texto(ruta: Path) -> str:
    """Lee un archivo de texto utf-8."""
    return ruta.read_text(encoding="utf-8")


def cargar_evidencia() -> dict[str, Any]:
    """Carga y verifica el precedente P61-01 y los vecinos de arboles.

    Returns:
        Evidencia verificada para el informe.

    Raises:
        SystemExit: si cualquier verificacion falla.
    """
    texto_p61 = leer_texto(RUTA_P61_SCRIPT)
    datos_p61 = json.loads(leer_texto(RUTA_P61_JSON))
    texto_p61_md = leer_texto(RUTA_P61_MD)
    texto_v8_13 = leer_texto(RUTA_V8_13)
    texto_decisions = leer_texto(RUTA_DECISIONS)
    texto_notebook = leer_texto(RUTA_NOTEBOOK)

    # 1. El script del precedente consume la representacion exacta de V03 en CSR.
    for fragmento in (
        "import lightgbm as lgb",
        'analyzer="word", ngram_range=(1, 2), min_df=2',
        'analyzer="char_wb", ngram_range=(3, 5), min_df=2',
        "hstack([",
        "]).tocsr()",
        "V03_UMBRALES = np.array([0.44, 0.51, 0.46])",
        "lgb.LGBMClassifier(",
        'v["pliegues"]',
        "early_stopping",
    ):
        if fragmento not in texto_p61:
            raise SystemExit(f"STOP: falta '{fragmento}' en p61_03_lightgbm.py")

    # 2. Los brazos medidos, con sus numeros exactos.
    if datos_p61["experiment_id"] != "P61-01" or datos_p61["script"] != "experiments/p61_03_lightgbm.py":
        raise SystemExit("STOP: el JSON del precedente no es P61-01")
    brazo_1 = datos_p61["brazo_1_lightgbm_simple"]
    brazo_2 = datos_p61["brazo_2_lightgbm_optuna"]
    if abs(brazo_1["corrected_cv"] - 0.7469717334185013) > 1e-15:
        raise SystemExit("STOP: corrected_cv del brazo 1 inesperado")
    if abs(brazo_2["corrected_cv"] - 0.7513651501981101) > 1e-15:
        raise SystemExit("STOP: corrected_cv del brazo 2 inesperado")
    if len(brazo_2["folds"]) != 5:
        raise SystemExit("STOP: el brazo 2 no tiene cinco pliegues")
    if (brazo_2["rescate"], brazo_2["damage"], brazo_2["neto"]) != (225, 253, -28):
        raise SystemExit("STOP: celdas del brazo 2 inesperadas")
    deltas = datos_p61["deltas"]
    if (abs(deltas["cambio_de_familia_1_menos_0"] - (-0.0302397191)) > 1e-9
            or abs(deltas["total_2_menos_0"] - (-0.0258463023)) > 1e-9):
        raise SystemExit("STOP: deltas del precedente inesperados")
    gate = datos_p61["gate"]
    if gate["superado"] is not False or gate["pliegues_mejorados"] != 2:
        raise SystemExit("STOP: el gate del precedente no coincide")

    # 3. El cierre formal del precedente.
    if "CLOSED_BY_DUPLICATION" not in texto_p61_md or "CLOSED_BY_DUPLICATION" not in datos_p61["decision"]:
        raise SystemExit("STOP: no se encuentra el cierre por duplicacion de P61-01")
    if "D-143" not in texto_decisions or "-0,0302397191" not in texto_decisions:
        raise SystemExit("STOP: D-143 no aparece con los numeros esperados")

    # 4. El registro: P61-01 es la unica fila de LightGBM; XGBoost y CatBoost, cero.
    filas_registro = list(
        csv.DictReader(RUTA_REGISTRO.read_text(encoding="utf-8-sig").splitlines())
    )
    fila_p61 = next((f for f in filas_registro if f["experiment_id"] == "P61-01"), None)
    if fila_p61 is None or "CLOSED_BY_DUPLICATION" not in fila_p61["decision"]:
        raise SystemExit("STOP: falta la fila P61-01 o su cierre por duplicacion")
    if "27431" not in fila_p61["representation"]:
        raise SystemExit("STOP: la representacion de P61-01 no cita 27431 rasgos")
    filas_lgbm = [f for f in filas_registro if "lightgbm" in str(f.get("model", "")).lower()]
    if [f["experiment_id"] for f in filas_lgbm] != ["P61-01"]:
        raise SystemExit("STOP: hay mas filas de LightGBM de las esperadas")
    for prohibido in ("xgboost", "catboost"):
        if any(prohibido in str(f.get("model", "")).lower() for f in filas_registro):
            raise SystemExit(f"STOP: aparecen filas de {prohibido} y el hueco no era completo")

    # 5. Los vecinos de arboles: V8-13 (subespacio) y el notebook.
    for fragmento in ("DecisionTreeClassifier", "RandomForestClassifier", "subespacio_discriminante"):
        if fragmento not in texto_v8_13:
            raise SystemExit(f"STOP: falta '{fragmento}' en v8_13_clasificadores_alternativos.py")
    if "La busqueda de capacidad LightGBM" not in texto_notebook:
        raise SystemExit("STOP: el notebook no documenta la busqueda LightGBM")
    if "en las 336 del registro" not in texto_notebook:
        raise SystemExit("STOP: el notebook no documenta el hueco de la familia")

    # 6. Deltas por pliegue del brazo 2 contra los pliegues canonicos de V03, y el
    #    criterio del gate del precedente (pliegue del candidato frente a la media global).
    oof = np.load(RUTA_OOF, allow_pickle=True)
    v03_folds = [float(x) for x in np.asarray(oof["f1_pliegues"])]
    deltas_pliegue = [b - a for a, b in zip(v03_folds, brazo_2["folds"])]
    positivos_vs_pliegue = sum(1 for d in deltas_pliegue if d > 0)
    positivos_vs_global = sum(1 for u in brazo_2["folds"] if u > V03_CCV)
    if len(deltas_pliegue) != 5 or positivos_vs_global != gate["pliegues_mejorados"]:
        raise SystemExit("STOP: los pliegues no reproducen el gate del precedente")

    # 7. Hashes de los artefactos congelados del campeon.
    hash_oof = sha256_de(RUTA_OOF)
    hash_submission = sha256_de(RUTA_SUBMISSION)
    if hash_oof != HASH_OOF or hash_submission != HASH_SUBMISSION:
        raise SystemExit("STOP: los artefactos de V03 no coinciden con sus hashes")

    return {
        "brazo_1": brazo_1,
        "brazo_2": brazo_2,
        "deltas": deltas,
        "gate": gate,
        "v03_folds": v03_folds,
        "deltas_pliegue": deltas_pliegue,
        "positivos_vs_pliegue": positivos_vs_pliegue,
        "positivos_vs_global": positivos_vs_global,
        "fila_p61": fila_p61,
        "hashes": {
            "oof_v03": hash_oof,
            "submission_v03": hash_submission,
            "p61_script": sha256_de(RUTA_P61_SCRIPT),
            "p61_json": sha256_de(RUTA_P61_JSON),
            "p61_md": sha256_de(RUTA_P61_MD),
        },
    }


def construir_informe(evidencia: dict[str, Any]) -> str:
    """Construye el markdown del informe de auditoria."""
    brazo_2 = evidencia["brazo_2"]
    por_etiqueta = brazo_2["f1_por_etiqueta"]
    v03_etiqueta = {"HS": 0.7616892911010558, "TR": 0.8157303370786517, "AG": 0.7542147293700089}
    filas_pliegue = "\n".join(
        f"| {i} | {v:.10f} | {b:.10f} | {d:+.10f} |"
        for i, (v, b, d) in enumerate(
            zip(evidencia["v03_folds"], brazo_2["folds"], evidencia["deltas_pliegue"]), start=1
        )
    )
    hashes = evidencia["hashes"]
    return f"""# TOXIC - P88-00: auditoria de novedad de LightGBM sobre la matriz TF-IDF de V03

Fecha: {FECHA}. Script: `experiments/p88_00_auditoria_novedad_lgbm.py`.
Directiva auditada: arboles potenciados (LightGBM) directamente sobre la matriz
dispersa word+char de V03, tres clasificadores binarios, umbrales congelados.

## 1. Veredicto

**`DUPLICATE`.** El precedente metodologicamente equivalente existe y es **P61-01**
(`experiments/p61_03_lightgbm.py`, 2026-09-28): LightGBM de tres clasificadores
binarios sobre el TF-IDF exacto de V03, mismos pliegues y umbrales. La familia se
cerro por duplicacion entonces. **No se ejecuta nada.**

## 2. La propuesta frente al precedente, elemento por elemento

| elemento de la propuesta | P61-01 (ejecutado) |
|---|---|
| LightGBM, tres clasificadores binarios independientes | si: un `LGBMClassifier` por etiqueta |
| TF-IDF word (1,2) + char_wb (3,5) de V03, tal cual, CSR | si: mismos `TfidfVectorizer` (min_df=2, sublinear_tf) y `hstack(...).tocsr()`; 27.431 rasgos |
| folds agrupados y vectorizadores solo del train | si: pliegues canonicos (D-016) del OOF congelado; `fit` por pliegue sobre el outer_train |
| umbrales historicos congelados | si: `V03_UMBRALES = [0.44, 0.51, 0.46]` aplicados igual a los brazos |
| una sola configuracion fijada antes de ejecutar | P61 midio DOS puntos: LightGBM por defecto y LightGBM con Optuna anidado (30 trials por pliegue, 450 en total); la configuracion unica de la propuesta seria un subconjunto estricto |
| sin early stopping sobre el fold externo | P61 uso early stopping sobre la validacion INTERNA (100 rondas de paciencia), conforme a la propuesta |

## 3. El resultado medido (2026-09-28)

| brazo | corrected_cv | delta vs V03 |
|---|---:|---:|
| 0, V03 | {V03_CCV:.10f} | +0.0000000000 |
| 1, LightGBM por defecto | {evidencia['brazo_1']['corrected_cv']:.10f} | {evidencia['deltas']['cambio_de_familia_1_menos_0']:+.10f} |
| 2, LightGBM + Optuna anidado | {brazo_2['corrected_cv']:.10f} | {evidencia['deltas']['total_2_menos_0']:+.10f} |

Descomposicion: cambio de familia {evidencia['deltas']['cambio_de_familia_1_menos_0']:+.10f};
optimizacion {evidencia['deltas']['optimizacion_2_menos_1']:+.10f}; total {evidencia['deltas']['total_2_menos_0']:+.10f}.

Por etiqueta (brazo 2 vs V03): HS {por_etiqueta['HS']:.4f} vs {v03_etiqueta['HS']:.4f};
TR {por_etiqueta['TR']:.4f} vs {v03_etiqueta['TR']:.4f}; AG {por_etiqueta['AG']:.4f} vs {v03_etiqueta['AG']:.4f}.

Deltas por pliegue del brazo 2 contra los pliegues canonicos de V03:

| pliegue | V03 | brazo 2 | delta |
|---:|---:|---:|---:|
{filas_pliegue}

```text
pliegues que mejoran (criterio del precedente, vs media global de V03)   {evidencia['gate']['pliegues_mejorados']}/5
pliegues vs pliegues canonicos de V03 (tabla)                            {evidencia['positivos_vs_pliegue']}/5
rescate / dano / neto        {brazo_2['rescate']} / {brazo_2['damage']} / {brazo_2['neto']:+d}
gate superado                {evidencia['gate']['superado']}
bootstrap                    no ejecutado (el gate no llego)
```

## 4. El cierre historico

- `reports/p61_lightgbm_capacidad.md`: decision **`CLOSED_BY_DUPLICATION`**. "La
  familia no aporta sobre V03 y el precedente OPT-01 ya cerro el patron de busqueda
  de capacidad por OOF".
- `DECISIONS.md` (D-143): la familia LightGBM era nueva en el registro (LightGBM,
  XGBoost y CatBoost: cero filas) y aun asi no llega; la busqueda Optuna FUNCIONA
  (recupera +0,004393, casi un 15 % del deficit) y lo que no llega es la familia.
- El riesgo se declaro antes de ejecutar: 27.431 rasgos frente a 1.186 filas de
  entrenamiento es un regimen donde los arboles no son naturalmente fuertes.
- `OPT-01` (transferencia): Optuna sobre el propio TF-IDF de V03 gano +0,008565 en
  OOF y se descarto porque no transfirio (0,77602 frente a 0,78306 de V03 en el
  leaderboard, -0,028). "La busqueda por OOF premia capacidad que no generaliza".

## 5. Familias vecinas de arboles, todas cerradas

| experimento | modelo | representacion | resultado | estado |
|---|---|---|---|---|
| P61-01 | LightGBM (3 binarios; default + Optuna) | TF-IDF de V03 tal cual (27.431, CSR) | -0.0302 / -0.0258; 2/5; neto -28 | CLOSED_BY_DUPLICATION |
| P61-05 | LightGBM como miembro de ensemble | scores del pool | supera el gate en la particion canonica; REJECTED en la segunda | cerrado |
| V8-13 | DecisionTree + RandomForest | TF-IDF de V03 reducido a subespacio discriminante (2.000 columnas) | ninguna alternativa supera al control | cerradas sin envio |
| V4-G01-G03 | HistGB 0.7297 / ExtraTrees 0.7033 | SVD-300 (denso) | muy por debajo de V03 | cerrado |
| P64-05 | HistGradientBoosting (MI_arbol) | bloque multiple-instance (canal denso) | unico residuo con componente no lineal; score regenerado en P66-01 | cerrado |
| P66-01 | HistGB para regenerar scores | espacio de probabilidad | REPRODUCEN 5, NO VERIFICABLE 1, DIFERENCIA NO RESUELTA 1 | cerrado |
| P68-01 | HistGB para enrutado | estructura del ejemplo (denso) | INVALIDATED, -0.0018992302, 1/5 pliegues | cerrado |

## 6. Contra el gate de la propuesta

1. Ningun punto medido se acerca: los dos brazos estan a -0.026 y -0.030, frente al
   +0.003 exigido; 2/5 pliegues; neto -28; bootstrap ni se ejecuto (el gate no llego).
2. El patron de la familia ya esta cerrado por partida doble: la busqueda de capacidad
   por OOF (OPT-01: +0.008565 OOF, -0.028 LB) y el cambio de familia (P61-01).
3. Ejecutar la configuracion unica de la propuesta seria re-medir un espacio ya
   cubierto por los dos brazos de P61-01, sin informacion nueva posible.

Conclusion: la hipotesis queda **cerrada sin ejecucion**. No se abre ninguna via y V03
permanece congelado.

## 7. Compliance

- Operaciones reales: solo lectura de `experiments/p61_03_lightgbm.py`,
  `reports/p61_lightgbm_capacidad.*`, `experiments/v8_13_clasificadores_alternativos.py`,
  `experiments.csv`, `DECISIONS.md` y el notebook. Cero entrenamientos; train/test/
  holdout no se abren para desarrollo; 0 submissions; 0 push; V03 y sus artefactos no
  se tocan.
- Escaneo byte a byte de lo generado: 0 bytes > 127.
- Hashes verificados: OOF `{hashes['oof_v03'][:16]}...`; submission `{hashes['submission_v03'][:16]}...`.

## 8. Hashes de la evidencia

```text
p61_03_lightgbm.py              {hashes['p61_script']}
p61_lightgbm_capacidad.json     {hashes['p61_json']}
p61_lightgbm_capacidad.md       {hashes['p61_md']}
```

> V03 remains the best verified leaderboard result.
"""


def construir_filas_csv(evidencia: dict[str, Any]) -> list[dict[str, str]]:
    """Construye la tabla de evidencia en CSV."""
    return [
        {
            "elemento": "Propuesta de la directiva",
            "tecnica": "LightGBM, tres clasificadores binarios",
            "representacion": "TF-IDF word(1,2) + char_wb(3,5) de V03, CSR",
            "protocolo": "folds agrupados D-016; vectorizadores del train; umbrales [0.44,0.51,0.46]",
            "resultado": "no ejecutada",
            "clasificacion": "DUPLICATE",
            "fuente": "este informe",
        },
        {
            "elemento": "P61-01",
            "tecnica": "LightGBM 3 binarios (por defecto y con Optuna anidado)",
            "representacion": "TF-IDF de V03 tal cual, 27431 rasgos, hstack(...).tocsr()",
            "protocolo": "pliegues canonicos del OOF congelado; umbrales congelados; early stopping interno",
            "resultado": "brazo 1 -0.0302397191; brazo 2 -0.0258463023; 2/5 pliegues; neto -28; gate no superado",
            "clasificacion": "CLOSED_BY_DUPLICATION (precedente directo)",
            "fuente": "reports/p61_lightgbm_capacidad.json; experiments/p61_03_lightgbm.py",
        },
        {
            "elemento": "P61-05",
            "tecnica": "LightGBM como miembro de un ensemble de cinco",
            "representacion": "scores del pool historico",
            "protocolo": "particion canonica y segunda particion",
            "resultado": "supera el gate en la canonica; REJECTED en la segunda particion",
            "clasificacion": "cerrado",
            "fuente": "experiments.csv",
        },
        {
            "elemento": "V8-13",
            "tecnica": "DecisionTree + RandomForest",
            "representacion": "TF-IDF de V03 reducido a subespacio discriminante (2.000 columnas)",
            "protocolo": "3 repeticiones, profundidades 6",
            "resultado": "ninguna alternativa supera al control",
            "clasificacion": "cerradas sin envio",
            "fuente": "experiments/v8_13_clasificadores_alternativos.py; experiments.csv",
        },
        {
            "elemento": "V4-G01-G03",
            "tecnica": "HistGB y ExtraTrees",
            "representacion": "SVD-300 (denso)",
            "protocolo": "el de la fase v4",
            "resultado": "svd:hist_gb 0.7297; svd:extra_trees 0.7033",
            "clasificacion": "cerrado (muy por debajo de V03)",
            "fuente": "reports/p61_lightgbm_capacidad.md",
        },
        {
            "elemento": "P64-05",
            "tecnica": "HistGradientBoosting (MI_arbol)",
            "representacion": "bloque multiple-instance (canal denso)",
            "protocolo": "el de la fase P64",
            "resultado": "unico residuo con componente no lineal; score regenerado en P66-01",
            "clasificacion": "cerrado",
            "fuente": "DECISIONS.md D-169/D-170; experiments.csv",
        },
        {
            "elemento": "P66-01",
            "tecnica": "HistGB para regenerar scores",
            "representacion": "espacio de probabilidad",
            "protocolo": "el de la fase P66",
            "resultado": "REPRODUCEN 5, NO VERIFICABLE 1, DIFERENCIA NO RESUELTA 1",
            "clasificacion": "cerrado",
            "fuente": "experiments.csv",
        },
        {
            "elemento": "P68-01",
            "tecnica": "HistGB para enrutado de la decision",
            "representacion": "estructura del ejemplo (denso)",
            "protocolo": "el de la fase P68",
            "resultado": "INVALIDATED, -0.0018992302, 1/5 pliegues",
            "clasificacion": "cerrado",
            "fuente": "experiments.csv",
        },
        {
            "elemento": "OPT-01 (transferencia)",
            "tecnica": "Optuna sobre el TF-IDF de V03 (LR)",
            "representacion": "TF-IDF de V03",
            "protocolo": "24 trials sobre OOF",
            "resultado": "+0.008565 OOF pero 0.77602 LB (-0.028); descartado",
            "clasificacion": "patron cerrado: la busqueda por OOF premia capacidad que no generaliza",
            "fuente": "DECISIONS.md D-143; reports/p61_lightgbm_capacidad.md",
        },
    ]


def escribir_informe(evidencia: dict[str, Any]) -> None:
    """Escribe el informe markdown."""
    RUTA_MD.write_text(construir_informe(evidencia), encoding="utf-8")


def escribir_csv(evidencia: dict[str, Any]) -> None:
    """Escribe la tabla de evidencia en CSV con terminador LF."""
    filas = construir_filas_csv(evidencia)
    with RUTA_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=["elemento", "tecnica", "representacion", "protocolo", "resultado", "clasificacion", "fuente"],
            lineterminator="\n",
        )
        escritor.writeheader()
        escritor.writerows(filas)


def registrar_fila(evidencia: dict[str, Any]) -> bool:
    """Anade la fila P88-00 al registro una sola vez (append de una linea con LF)."""
    texto = RUTA_REGISTRO.read_text(encoding="utf-8-sig")
    lineas = texto.splitlines()
    encabezado = next(csv.reader(lineas))
    if encabezado != COLUMNAS_REGISTRO:
        raise SystemExit("STOP: las columnas del registro no coinciden con la fila P88-00")
    filas = list(csv.DictReader(lineas))
    if any(fila["experiment_id"] == "P88-00" for fila in filas):
        return False
    fila = {
        "experiment_id": "P88-00",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Existe un precedente metodologicamente equivalente a LightGBM sobre la matriz TF-IDF exacta de V03?",
        "hypothesis": "Si la familia ya se midio sobre la representacion exacta y con el mismo protocolo, la propuesta es DUPLICATE y no se ejecuta",
        "preprocessing": "ninguno (auditoria de solo lectura; no se entrena)",
        "feature_groups": "no aplica (auditoria)",
        "representation": "no aplica (auditoria)",
        "model": "no aplica (auditoria)",
        "hyperparameters": "no aplica (auditoria)",
        "validation": "lectura de experiments/p61_03_lightgbm.py, reports/p61_lightgbm_capacidad.*, experiments.csv, DECISIONS.md y notebook",
        "f1": "",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "0 s (sin entrenamiento)",
        "dimension": "",
        "result": (
            "DUPLICATE: P61-01 ejecuto LightGBM (3 binarios) sobre el TF-IDF exacto de V03 (27.431 rasgos, CSR) "
            "con pliegues canonicos y umbrales congelados; brazos -0.0302397191 y -0.0258463023; 2/5 pliegues; neto -28"
        ),
        "interpretation": "la familia se cerro por duplicacion (D-143) y el patron de busqueda por OOF ya cerro con OPT-01; no se ejecuta nada",
        "decision": "DUPLICATE: no se ejecuta; la hipotesis queda cerrada",
        "date": FECHA,
        "script": "experiments/p88_00_auditoria_novedad_lgbm.py",
        "notes": "auditoria P0 de la directiva 2026-10-09; 0 entrenamientos, 0 submissions, 0 push; sin reabrir familias cerradas",
    }
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=COLUMNAS_REGISTRO, lineterminator="\n")
    escritor.writerow(fila)
    with RUTA_REGISTRO.open("ab") as archivo:
        archivo.write(buffer.getvalue().encode("utf-8"))
    return True


def main() -> int:
    """Ejecuta la auditoria completa."""
    log(f"fecha {FECHA} | auditoria de solo lectura")
    evidencia = cargar_evidencia()
    log("precedente P61-01 verificado: representacion exacta, protocolo, gate y cierre")
    escribir_informe(evidencia)
    escribir_csv(evidencia)
    anadida = registrar_fila(evidencia)
    log(f"informe y CSV escritos | fila P88-00 {'anadida' if anadida else 'ya presente'}")

    sucios = [
        ruta.name
        for ruta in (RUTA_MD, RUTA_CSV)
        if any(byte > 127 for byte in ruta.read_bytes())
    ]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    print(json.dumps({
        "clasificacion": "DUPLICATE",
        "precedente": "P61-01 / p61_03_lightgbm.py (2026-09-28)",
        "brazo_1": evidencia["brazo_1"]["corrected_cv"],
        "brazo_2": evidencia["brazo_2"]["corrected_cv"],
        "pliegues_mejorados": evidencia["gate"]["pliegues_mejorados"],
        "neto": evidencia["brazo_2"]["neto"],
        "ejecutado": False,
        "entrenamientos": 0,
        "submissions": 0,
        "push": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
