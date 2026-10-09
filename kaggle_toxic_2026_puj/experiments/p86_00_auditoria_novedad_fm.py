"""P86-00: auditoria de novedad de la Factorization Machine dispersa (directiva 2026-10-08).

La directiva propone una FM dispersa con interacciones de bajo rango entre
caracteristicas lexicas, sobre la representacion word+char de V03, y ordena
auditar precedentes antes de ejecutar: si existe un precedente metodologicamente
equivalente, la propuesta es DUPLICATE y no se ejecuta.

Veredicto: DUPLICATE. V25-01/V25-02 (2026-09-27) ejecutaron exactamente esa
propuesta: FM desde cero sobre la representacion exacta de V03, mismo protocolo
(pliegues D-016, umbrales [0.44, 0.51, 0.46]), control que reproduce V03 bit a
bit, k en {1, 4, 8, 16} con rango agotado en 4, barrido de lambda_v con optimo
exacto en 1, mejor +0.005487 con neto de celdas -40 y dLB esperado -0.0079, y
cierre formal (CLOSE V25; CERRAR EL PROYECTO).

0 entrenamientos, 0 datos de desarrollo abiertos, 0 submissions, 0 push.

Genera:
    reports/toxic_fm_novelty_audit.md
    reports/toxic_fm_novelty_audit.csv
    fila P86-00 en experiments.csv (una sola vez; append de una linea con LF)

Uso:
    /home/carlos/venv/bin/python experiments/p86_00_auditoria_novedad_fm.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]

RUTA_V25_01_SCRIPT: Final[Path] = RAIZ / "experiments" / "v25_01_factorization_machine.py"
RUTA_V25_02_SCRIPT: Final[Path] = RAIZ / "experiments" / "v25_02_fm_paralela_lambda.py"
RUTA_V25_01_JSON: Final[Path] = RAIZ / "reports" / "v25_01_factorization_machine.json"
RUTA_V25_02_JSON: Final[Path] = RAIZ / "reports" / "v25_02_fm_paralela_lambda.json"
RUTA_V25_04: Final[Path] = RAIZ / "reports" / "v25_04_cierre.md"
RUTA_V25_05: Final[Path] = RAIZ / "reports" / "v25_05_dictamen_final.md"
RUTA_P59_05: Final[Path] = RAIZ / "experiments" / "p59_05_nmf_gap_analysis.py"
RUTA_DECISIONS: Final[Path] = RAIZ / "DECISIONS.md"
RUTA_RUNLOG: Final[Path] = RAIZ / "RUNLOG.md"
RUTA_NOTEBOOK: Final[Path] = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_PROJECT_STATE: Final[Path] = RAIZ / "PROJECT_STATE.md"
RUTA_REGISTRO: Final[Path] = RAIZ / "experiments.csv"
RUTA_OOF: Final[Path] = RAIZ / "features" / "oof_v03_final.npz"
RUTA_SUBMISSION: Final[Path] = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
RUTA_MD: Final[Path] = RAIZ / "reports" / "toxic_fm_novelty_audit.md"
RUTA_CSV: Final[Path] = RAIZ / "reports" / "toxic_fm_novelty_audit.csv"

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
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P86-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_texto(ruta: Path) -> str:
    """Lee un archivo de texto utf-8.

    Args:
        ruta: ruta del archivo.

    Returns:
        Contenido completo.
    """
    return ruta.read_text(encoding="utf-8")


def cargar_evidencia() -> dict[str, Any]:
    """Carga el precedente V25 y verifica la equivalencia con la propuesta.

    Returns:
        Evidencia verificada para el informe.

    Raises:
        SystemExit: si cualquier verificacion falla.
    """
    texto_v25_01 = leer_texto(RUTA_V25_01_SCRIPT)
    texto_v25_02 = leer_texto(RUTA_V25_02_SCRIPT)
    datos_v25_01 = json.loads(leer_texto(RUTA_V25_01_JSON))
    datos_v25_02 = json.loads(leer_texto(RUTA_V25_02_JSON))
    texto_v25_04 = leer_texto(RUTA_V25_04)
    texto_v25_05 = leer_texto(RUTA_V25_05)
    texto_p59_05 = leer_texto(RUTA_P59_05)
    texto_decisions = leer_texto(RUTA_DECISIONS)
    texto_runlog = leer_texto(RUTA_RUNLOG)
    texto_notebook = leer_texto(RUTA_NOTEBOOK)
    texto_project_state = leer_texto(RUTA_PROJECT_STATE)

    # 1. La representacion de V25-01 es la representacion exacta de V03.
    if 'PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}' not in texto_v25_01:
        raise SystemExit("STOP: PARAMETROS_WORD de V25-01 no coinciden con los de V03")
    if 'PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}' not in texto_v25_01:
        raise SystemExit("STOP: PARAMETROS_CHAR de V25-01 no coinciden con los de V03")
    if "UMBRALES_V03 = np.array([0.44, 0.51, 0.46])" not in texto_v25_01:
        raise SystemExit("STOP: los umbrales de V25-01 no son los congelados de V03")
    if "pliegues_corregidos" not in texto_v25_01:
        raise SystemExit("STOP: V25-01 no usa los pliegues corregidos D-016")
    if "nnz" not in texto_v25_01:
        raise SystemExit("STOP: V25-01 no declara la complejidad O(nnz * k)")

    # 2. El control de V25-02 reproduce V03 exactamente.
    control = datos_v25_02["controles"]["control_reproduce_v03"]
    if control["corrected_cv"] != V03_CCV or control["diferencia"] != 0.0:
        raise SystemExit("STOP: el control de V25-02 no reproduce V03")
    if control["predicciones_identicas"] is not True:
        raise SystemExit("STOP: el control de V25-02 no produce predicciones identicas")

    # 3. El rango efectivo se agota en k=4 (k=8 y k=16 identicos).
    if "k=8 y k=16 fueron identicos a k=4" not in datos_v25_02["decisiones_de_diseno"]["k"]:
        raise SystemExit("STOP: no se confirma el agotamiento del rango en k=4")

    # 4. La mejor entrada del barrido, con sus numeros exactos.
    mejor = max(datos_v25_02["resultados"], key=lambda fila: fila["corrected_cv"])
    if (mejor["k"], mejor["lambda_v"]) != (4, 1.0):
        raise SystemExit("STOP: el mejor candidato no es k=4, lambda_v=1")
    if abs(mejor["corrected_cv"] - 0.7826986101342172) > 1e-15:
        raise SystemExit("STOP: corrected_cv del mejor candidato inesperado")
    if abs(mejor["delta_vs_v03"] - 0.0054871576176450665) > 1e-15:
        raise SystemExit("STOP: delta del mejor candidato inesperado")
    if (mejor["rescates"], mejor["danos"], mejor["neto"]) != (119, 159, -40):
        raise SystemExit("STOP: celdas del mejor candidato inesperadas")
    if abs(mejor["d_lb_esperado"] - (-0.007856743815200488)) > 1e-15:
        raise SystemExit("STOP: dLB esperado del mejor candidato inesperado")
    if mejor["ajustes_sin_converger"] != 0:
        raise SystemExit("STOP: el mejor candidato registra ajustes sin converger")

    # 5. Cierre formal del precedente.
    if "NO CANDIDATE JUSTIFIES SUBMISSION" not in texto_v25_04 or "CLOSE V25" not in texto_v25_04:
        raise SystemExit("STOP: no se encuentra el cierre formal de V25")
    if "CERRAR EL PROYECTO" not in texto_v25_05:
        raise SystemExit("STOP: no se encuentra el dictamen final de V25")

    # 6. La familia esta cerrada en los ejes y en el notebook.
    if "V25, V27, V26, V43" not in texto_decisions:
        raise SystemExit("STOP: el eje de geometria de representacion no esta cerrado en DECISIONS")
    if "FACTORIZATION MACHINE" not in texto_notebook or "REJECTED (V25)" not in texto_notebook:
        raise SystemExit("STOP: el notebook no registra el rechazo de la FM")
    if "low-rank factorisation of second-order interactions on V03's" not in texto_p59_05:
        raise SystemExit("STOP: la auditoria P59-05 no caracteriza V25 como se espera")

    # 7. Filas del registro.
    filas_registro = list(
        csv.DictReader(RUTA_REGISTRO.read_text(encoding="utf-8-sig").splitlines())
    )
    fila_v25_01 = next((f for f in filas_registro if f["experiment_id"] == "V25-01"), None)
    fila_v25_02 = next((f for f in filas_registro if f["experiment_id"] == "V25-02"), None)
    if fila_v25_01 is None or "cerrar V25" not in fila_v25_01["decision"]:
        raise SystemExit("STOP: falta la fila V25-01 o su decision no cierra la fase")
    if fila_v25_02 is None or "cierre de V25" not in fila_v25_02["decision"]:
        raise SystemExit("STOP: falta la fila V25-02 o su decision no confirma el cierre")

    # 8. PROJECT_STATE.md no menciona la tecnica (la busqueda fue completa).
    for patron in ("V25", "Factorization", "factoriz"):
        if patron in texto_project_state:
            raise SystemExit(f"STOP: PROJECT_STATE.md menciona {patron} y la busqueda no es completa")

    # 9. Hashes de los artefactos congelados del campeon.
    hash_oof = sha256_de(RUTA_OOF)
    hash_submission = sha256_de(RUTA_SUBMISSION)
    if hash_oof != HASH_OOF:
        raise SystemExit("STOP: el OOF de V03 no coincide con su hash congelado")
    if hash_submission != HASH_SUBMISSION:
        raise SystemExit("STOP: la submission de V03 no coincide con su hash congelado")

    filas_k4 = [f for f in datos_v25_02["resultados"] if f["k"] == 4]
    return {
        "datos_v25_01": datos_v25_01,
        "datos_v25_02": datos_v25_02,
        "mejor": mejor,
        "filas_k4": filas_k4,
        "fila_v25_01": fila_v25_01,
        "fila_v25_02": fila_v25_02,
        "runlog_sin_v25": "V25" not in texto_runlog,
        "hashes": {
            "oof_v03": hash_oof,
            "submission_v03": hash_submission,
            "v25_01_script": sha256_de(RUTA_V25_01_SCRIPT),
            "v25_02_script": sha256_de(RUTA_V25_02_SCRIPT),
            "v25_02_json": sha256_de(RUTA_V25_02_JSON),
        },
    }


def fila_tabla_k4(fila: dict[str, Any]) -> str:
    """Formatea una entrada del barrido de V25-02 como fila de tabla.

    Args:
        fila: entrada de resultados del JSON de V25-02.

    Returns:
        Fila markdown.
    """
    return (
        f"| {fila['k']} | {fila['lambda_v']} | {fila['corrected_cv']:.10f} | "
        f"{fila['delta_vs_v03']:+.9f} | {fila['f1_train']:.4f} | {fila['gap_train_oof']:+.4f} | "
        f"{fila['celdas']} | {fila['rescates']} | {fila['danos']} | {fila['neto']:+d} |"
    )


def construir_informe(evidencia: dict[str, Any]) -> str:
    """Construye el markdown del informe de auditoria.

    Args:
        evidencia: evidencia verificada.

    Returns:
        Texto del informe.
    """
    mejor = evidencia["mejor"]
    por_etiqueta = mejor["por_etiqueta"]
    filas_k4 = "\n".join(fila_tabla_k4(fila) for fila in evidencia["filas_k4"])
    hashes = evidencia["hashes"]
    nota_runlog = (
        "El RUNLOG no menciona V25 (la fase se documento en DECISIONS, informes y notebook); "
        "esta auditoria lo registra ahora."
        if evidencia["runlog_sin_v25"]
        else "El RUNLOG menciona V25."
    )
    return f"""# TOXIC - P86-00: auditoria de novedad de la Factorization Machine dispersa

Fecha: {FECHA}. Script: `experiments/p86_00_auditoria_novedad_fm.py`.
Directiva auditada: FM dispersa con interacciones de bajo rango entre caracteristicas
lexicas, sobre la representacion word+char de V03, con el protocolo de V03.

## 1. Veredicto

**`DUPLICATE`.** El precedente metodologicamente equivalente existe y es
**V25-01 / V25-02 (2026-09-27)**: la Factorization Machine fue implementada desde cero
sobre la representacion exacta de V03, con el mismo protocolo y un control que
reproduce V03 bit a bit. La familia quedo cerrada entonces. **No se ejecuta nada.**

## 2. La propuesta frente al precedente, elemento por elemento

| elemento de la propuesta | V25-01 / V25-02 (ejecutado el 2026-09-27) |
|---|---|
| FM dispersa con interacciones de bajo rango | FM desde cero; interaccion de segundo orden de bajo rango; `O(nnz * k)`; 208.907.020 pares no materializados |
| representacion word (1,2) + char_wb (3,5) de V03 | `PARAMETROS_WORD` y `PARAMETROS_CHAR` identicos (min_df=2, sublinear_tf=True) |
| mismos folds agrupados y conjunto de desarrollo | StratifiedGroupKFold(42) sobre grupos near-duplicate (D-016); mismo dev; holdout intacto |
| mismos umbrales congelados y corrected_cv | `UMBRALES_V03 = [0.44, 0.51, 0.46]`; el control reproduce `{V03_CCV}` (diferencia 0.0; predicciones identicas) |
| control lineal equivalente sin interacciones | rama `interacciones=False` = la ruta exacta de V03 (liblinear, C=2.0, balanced, tol=1e-4) |
| una sola dimension latente, fijada con justificacion previa | k en {{1, 4, 8, 16}}: k=8 y k=16 identicos a k=4 (rango efectivo agotado en 4) |
| sin busqueda de regularizacion ni de semillas | barrido de `lambda_v` de 0.25 a 1e6: optimo exacto en 1, sin margen; 0 ajustes sin converger |

## 3. El resultado medido (no queda nada nuevo que medir)

Mejor candidato del barrido: k=4, `lambda_v`=1.

```text
corrected_cv          {mejor['corrected_cv']:.16f}
delta vs V03          {mejor['delta_vs_v03']:+.16f}
train F1              {mejor['f1_train']:.16f}  (gap train-OOF {mejor['gap_train_oof']:+.4f})
celdas cambiadas      {mejor['celdas']}
rescates / danos      {mejor['rescates']} / {mejor['danos']}
neto de celdas        {mejor['neto']:+d}
dLB esperado          {mejor['d_lb_esperado']:.16f}
ajustes sin converger {mejor['ajustes_sin_converger']}
```

Por etiqueta (F1 y celdas): HS {por_etiqueta['HS']['f1']:.6f} ({por_etiqueta['HS']['rescates']} rescates / {por_etiqueta['HS']['danos']} danos),
TR {por_etiqueta['TR']['f1']:.6f} ({por_etiqueta['TR']['rescates']} / {por_etiqueta['TR']['danos']}),
AG {por_etiqueta['AG']['f1']:.6f} ({por_etiqueta['AG']['rescates']} / {por_etiqueta['AG']['danos']}).
Los tres F1 suben y los tres netos son negativos: es el perfil que la propia directiva
advierte ("no interpretar una mayor capacidad como evidencia de transferencia"); el
entrenamiento sube (train F1 {mejor['f1_train']:.4f}) y el neto empeora.

Barrido de k=4 (V25-02):

| k | lambda_v | corrected_cv | dOOF | train F1 | gap | celdas | rescates | danos | neto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{filas_k4}

Con `lambda_v >= 16` el modelo colapsa a V03 exacto (dOOF +0.000000 a diez decimales).

## 4. El cierre historico (por que no se reabre)

- `reports/v25_04_cierre.md`: `NO CANDIDATE JUSTIFIES SUBMISSION - CLOSE V25`. El
  sobreajuste no se puede corregir, el rango se agota en 4 y el modo conjunto memoriza
  (train F1 1.0000, OOF -0.0284).
- `reports/v25_05_dictamen_final.md`: `CERRAR EL PROYECTO`. El gate del proyecto quedo
  corregido a `dOOF > +0.0376` **y** neto de celdas positivo en la misma configuracion;
  la FM es el caso que lo motiva (cambia 278 celdas, pasa el filtro de 200 celdas y aun
  asi tiene neto -40 con dOOF positivo).
- `DECISIONS.md` (D-190): el eje "geometria de representacion" (V25, V27, V26, V43)
  queda cerrado.
- Notebook: "FACTORIZATION MACHINE REJECTED (V25)" y el cierre de las cuatro familias
  no lineales (V23 RBF-SVC, V25 FM, V26 Tsetlin, V27 spectrum).
- `experiments/p59_05_nmf_gap_analysis.py` (auditoria posterior e independiente)
  describe V25 como "a low-rank factorisation of second-order interactions on V03's
  exact representation" y lo cita como vecino funcional ya cerrado.
- Registro: V25-01 (f1 {evidencia['fila_v25_01']['f1']}, "no candidata, cerrar V25") y
  V25-02 (f1 {evidencia['fila_v25_02']['f1']}, "confirmar el cierre de V25 sin submission").
- {nota_runlog}

## 5. Familias vecinas de interacciones y no linealidad, todas cerradas

| experimento | mecanismo | resultado | estado |
|---|---|---|---|
| V23 (RBF-SVC) | no linealidad por kernel | corrected_cv 0.790409; LB 0.77938 (-0.00368) | REJECTED |
| V25-01/V25-02 | FM (interacciones de bajo rango) | +0.005487; neto -40; dLB esperado -0.0079 | CLOSE V25 |
| V26-01..03 | Tsetlin Machine | V26-02 invalidada; V26-03 0.583397 | REJECTED |
| V27-01 | kernel de espectro de caracteres | 0.762122 (-0.015089); neto -45 | CHALLENGER, sin submission |
| V43-01/V43-02 | PLS | -0.0137423671 (PLS correcto) | cerrado |
| V4-D01-D04 | interacciones como tokens | 0.7758; TRACK D NO_SIGNAL | cerrado |
| E2-10-ORT-* | forma x token | -0.0238 a -0.0286 (0/5 pliegues) | cerrado |
| V6-05 | skip-grams con hueco 1-2 | -0.0001 (IC95 incluye 0) | cerrado |
| P65 (D-162) | pares lejanos 3-5 y 6-10 | -0.0066 / -0.0074 | cerrado |

## 6. Contra el gate de la propuesta

El delta del precedente (+0.005487) supera nominalmente el +0.003 propuesto, pero:

1. el propio proyecto corrigio su gate tras medirlo: neto de celdas negativo (-40) y
   dLB esperado negativo (-0.0079) en la misma configuracion;
2. el modelo de transferencia del proyecto (V23: mejor local 0.790409, peor LB) y el
   analisis de V25-05 ("V03 fue afortunado") apuntan en la misma direccion;
3. re-ejecutar la configuracion unica (k=4, `lambda_v`=1) reproduciria el punto ya
   medido, sin informacion nueva.

Conclusion: la hipotesis queda **cerrada sin ejecucion**. No se abre ninguna via y V03
permanece congelado.

## 7. Compliance

- Operaciones reales: solo lectura de `experiments.csv`, `DECISIONS.md`, `RUNLOG.md`,
  `PROJECT_STATE.md`, `notebooks/10_final_pipeline.ipynb`, `experiments/v25_0*.py`,
  `experiments/p59_05_nmf_gap_analysis.py` y `reports/v25_0*`. Cero entrenamientos;
  train/test/holdout no se abren para desarrollo; 0 submissions; 0 push; V03 y sus
  artefactos no se tocan.
- `PROJECT_STATE.md` no menciona V25 ni la FM (verificado).
- Escaneo byte a byte de lo generado: 0 bytes > 127.
- Hashes verificados: OOF `{hashes['oof_v03'][:16]}...`; submission `{hashes['submission_v03'][:16]}...`.

## 8. Hashes de la evidencia

```text
v25_01_factorization_machine.py   {hashes['v25_01_script']}
v25_02_fm_paralela_lambda.py      {hashes['v25_02_script']}
v25_02_fm_paralela_lambda.json    {hashes['v25_02_json']}
```

> V03 remains the best verified leaderboard result.
"""


def construir_filas_csv(evidencia: dict[str, Any]) -> list[dict[str, str]]:
    """Construye las filas de la tabla de evidencia en CSV.

    Args:
        evidencia: evidencia verificada.

    Returns:
        Filas del CSV.
    """
    return [
        {
            "propuesta_o_precedente": "Propuesta de la directiva",
            "tecnica": "FM dispersa con interacciones de bajo rango entre caracteristicas lexicas",
            "representacion": "word(1,2) + char_wb(3,5) TF-IDF de V03",
            "protocolo": "pliegues agrupados D-016; umbrales [0.44,0.51,0.46]; control lineal",
            "resultado": "no ejecutada",
            "clasificacion": "DUPLICATE",
            "fuente": "este informe",
        },
        {
            "propuesta_o_precedente": "V25-01",
            "tecnica": "FM desde cero (interaccion de segundo orden de bajo rango, O(nnz*k))",
            "representacion": "word(1,2)+char_wb(3,5), min_df=2, sublinear_tf=True (la de V03)",
            "protocolo": "StratifiedGroupKFold(42) sobre grupos (D-016); umbrales [0.44,0.51,0.46]; control reproduce V03",
            "resultado": "k=8 y k=16 identicos a k=4; modo conjunto memoriza (train F1 1.0000, OOF -0.0284)",
            "clasificacion": "DUPLICATE (precedente directo)",
            "fuente": "reports/v25_01_factorization_machine.json; experiments/v25_01_factorization_machine.py",
        },
        {
            "propuesta_o_precedente": "V25-02",
            "tecnica": "FM con barrido de lambda_v y reparto paralelo",
            "representacion": "identica a V25-01",
            "protocolo": "identico a V25-01; control reproduce V03 bit a bit (0.0 de diferencia)",
            "resultado": "mejor k=4 lambda_v=1: 0.782699 (+0.005487); neto -40; dLB esperado -0.0079; 0 sin converger",
            "clasificacion": "DUPLICATE (precedente directo)",
            "fuente": "reports/v25_02_fm_paralela_lambda.json y .md",
        },
        {
            "propuesta_o_precedente": "V23 (RBF-SVC)",
            "tecnica": "no linealidad por kernel",
            "representacion": "V03 completa",
            "protocolo": "el de V03",
            "resultado": "corrected_cv 0.790409; LB 0.77938 (-0.00368)",
            "clasificacion": "CERRADO EN CONTRA",
            "fuente": "reports/v25_04_cierre.md; notebook",
        },
        {
            "propuesta_o_precedente": "V26-01..V26-03",
            "tecnica": "Tsetlin Machine",
            "representacion": "binaria",
            "protocolo": "el de V03",
            "resultado": "V26-02 invalidada; V26-03 0.583397",
            "clasificacion": "REJECTED",
            "fuente": "experiments.csv",
        },
        {
            "propuesta_o_precedente": "V27-01",
            "tecnica": "kernel de espectro de caracteres",
            "representacion": "V03 completa",
            "protocolo": "el de V03",
            "resultado": "0.762122 (-0.015089); neto -45",
            "clasificacion": "CHALLENGER, sin submission",
            "fuente": "experiments.csv",
        },
        {
            "propuesta_o_precedente": "V43-01/V43-02",
            "tecnica": "PLS (proyeccion supervisada de bajo rango)",
            "representacion": "V03 completa",
            "protocolo": "el de V03",
            "resultado": "-0.0137423671 (PLS correcto)",
            "clasificacion": "cerrado",
            "fuente": "experiments.csv; D-190",
        },
        {
            "propuesta_o_precedente": "V4-D01-D04",
            "tecnica": "interacciones como tokens (manuales)",
            "representacion": "explicita",
            "protocolo": "el de V03",
            "resultado": "0.7758; TRACK D NO_SIGNAL",
            "clasificacion": "cerrado",
            "fuente": "experiments.csv",
        },
        {
            "propuesta_o_precedente": "E2-10-ORT-*",
            "tecnica": "forma x token (manuales)",
            "representacion": "explicita",
            "protocolo": "el de V03",
            "resultado": "-0.0238 a -0.0286 (0/5 pliegues)",
            "clasificacion": "cerrado",
            "fuente": "experiments.csv",
        },
        {
            "propuesta_o_precedente": "V6-05",
            "tecnica": "skip-grams con hueco 1-2",
            "representacion": "explicita",
            "protocolo": "el de V03",
            "resultado": "-0.0001 (IC95 incluye 0)",
            "clasificacion": "cerrado",
            "fuente": "experiments.csv",
        },
        {
            "propuesta_o_precedente": "P65 (D-162)",
            "tecnica": "pares lejanos 3-5 y 6-10 tokens",
            "representacion": "explicita",
            "protocolo": "el de V03",
            "resultado": "-0.0066 / -0.0074",
            "clasificacion": "cerrado",
            "fuente": "DECISIONS.md D-162/D-163",
        },
    ]


def escribir_informe(evidencia: dict[str, Any]) -> None:
    """Escribe el informe markdown.

    Args:
        evidencia: evidencia verificada.
    """
    RUTA_MD.write_text(construir_informe(evidencia), encoding="utf-8")


def escribir_csv(evidencia: dict[str, Any]) -> None:
    """Escribe la tabla de evidencia en CSV con terminador LF.

    Args:
        evidencia: evidencia verificada.
    """
    filas = construir_filas_csv(evidencia)
    with RUTA_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=list(filas[0].keys()), lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(filas)


def registrar_fila(evidencia: dict[str, Any]) -> bool:
    """Anade la fila P86-00 al registro una sola vez (append de una linea con LF).

    Args:
        evidencia: evidencia verificada.

    Returns:
        True si se anadio la fila; False si ya existia.
    """
    texto = RUTA_REGISTRO.read_text(encoding="utf-8-sig")
    lineas = texto.splitlines()
    encabezado = next(csv.reader(lineas))
    if encabezado != COLUMNAS_REGISTRO:
        raise SystemExit("STOP: las columnas del registro no coinciden con la fila P86-00")
    filas = list(csv.DictReader(lineas))
    if any(fila["experiment_id"] == "P86-00" for fila in filas):
        return False
    mejor = evidencia["mejor"]
    fila = {
        "experiment_id": "P86-00",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Existe un precedente metodologicamente equivalente a una FM dispersa con interacciones de bajo rango sobre la representacion word+char de V03?",
        "hypothesis": "Si el mecanismo ya fue ejecutado sobre la representacion exacta y con el mismo protocolo, la propuesta es DUPLICATE y no se ejecuta",
        "preprocessing": "ninguno (auditoria de solo lectura; no se abren datos de desarrollo)",
        "feature_groups": "no aplica (auditoria)",
        "representation": "no aplica (auditoria)",
        "model": "no aplica (auditoria)",
        "hyperparameters": "no aplica (auditoria)",
        "validation": "lectura de reports/v25_*, experiments/v25_*, experiments.csv, DECISIONS.md, RUNLOG.md y notebook",
        "f1": "",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "0 s (sin entrenamiento)",
        "dimension": "",
        "result": (
            f"DUPLICATE: V25-01/V25-02 ejecutaron la FM sobre la representacion exacta de V03 "
            f"(control reproduce {V03_CCV}; k=8/16 identicos a k=4; lambda_v optimo en 1; "
            f"mejor {mejor['corrected_cv']:.6f} con neto {mejor['neto']:+d} y dLB esperado {mejor['d_lb_esperado']:+.4f})"
        ),
        "interpretation": "el precedente esta cerrado (CLOSE V25; CERRAR EL PROYECTO) y el eje de geometria de representacion quedo cerrado en D-190; no se ejecuta nada",
        "decision": "DUPLICATE: no se ejecuta; la hipotesis queda cerrada",
        "date": FECHA,
        "script": "experiments/p86_00_auditoria_novedad_fm.py",
        "notes": "auditoria P0 de la directiva 2026-10-08; 0 entrenamientos, 0 submissions, 0 push; sin reabrir familias cerradas",
    }
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=COLUMNAS_REGISTRO, lineterminator="\n")
    escritor.writerow(fila)
    with RUTA_REGISTRO.open("ab") as archivo:
        archivo.write(buffer.getvalue().encode("utf-8"))
    return True


def main() -> int:
    """Ejecuta la auditoria completa.

    Returns:
        Codigo de salida.
    """
    log(f"fecha {FECHA} | auditoria de solo lectura")
    evidencia = cargar_evidencia()
    log("precedente V25 verificado: representacion exacta, protocolo y control")

    escribir_informe(evidencia)
    escribir_csv(evidencia)
    anadida = registrar_fila(evidencia)
    log(f"informe y CSV escritos | fila P86-00 {'anadida' if anadida else 'ya presente'}")

    sucios = [
        ruta.name
        for ruta in (RUTA_MD, RUTA_CSV)
        if any(byte > 127 for byte in ruta.read_bytes())
    ]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    mejor = evidencia["mejor"]
    print(
        json.dumps(
            {
                "clasificacion": "DUPLICATE",
                "precedente": "V25-01 / V25-02 (2026-09-27)",
                "mejor_corrected_cv": mejor["corrected_cv"],
                "delta_vs_v03": mejor["delta_vs_v03"],
                "neto": mejor["neto"],
                "d_lb_esperado": mejor["d_lb_esperado"],
                "ejecutado": False,
                "entrenamientos": 0,
                "submissions": 0,
                "push": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
