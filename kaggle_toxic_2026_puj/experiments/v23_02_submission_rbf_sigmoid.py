"""V23-02: refit sobre todo el train y submission de svd:rbf_svc (variante sigmoidal).

Que hace y por que
------------------
Genera la submission de la candidata que V23-01 audito: un `SVC` con kernel RBF sobre la
compresion `TruncatedSVD(300)` de la union TF-IDF de V03, con la calibracion sigmoidal sobre
`decision_function`, y los **umbrales congelados de V03**. Es la primera candidata de la
historia con evidencia estadistica: corrected_cv 0.790410 (+0.013198), 5 de 5 pliegues,
perfil de celdas con 136 rescates contra 104 danos, neto +32 y McNemar exacto p = 0.0452.

**Con que filas se entrena.** Igual que V03: las 1.731 del train completo. `04_refit_
submissions_v02_v03.py` linea 34 llama a `ajustar_modelo_final(train)` sobre el train entero,
y la submission congelada se genero asi. El OOF que produjo la evidencia se midio sobre las
1.483 del desarrollo, de modo que el refit tiene mas datos que la medicion, no menos.

Controles, y por que cada uno esta
----------------------------------
**El control mas fuerte disponible: la cadena de features reproduce el champion.** Antes de
construir la candidata, este script construye la submission de V03 con su propio recipe
(TF-IDF + `LogisticRegression(C=2.0, balanced, liblinear)` + umbrales congelados) y la
compara celda a celda con `experiments/baseline_v03_frozen/submission_v03.csv`. Si no
coincide, hay un error en la construccion de las features o en el orden de las filas, y
cualquier conclusion sobre la candidata seria falsa. En V20 ese mismo control fue el que
impidio enviar una submission construida sobre una base equivocada.

**El test no se usa para nada que no sea transformacion.** Se comprueba que el CSV de test no
traiga columna de etiqueta, que sus ids sean unicos y que su orden coincida con el de la
submission congelada. No se lee ninguna etiqueta del test, no se entrena con el y no se
ajusta ningun umbral con el.

**El SVD se ajusta solo con el train.** `svd.fit_transform(matriz_train)` y
`svd.transform(matriz_test)`. El SVD no usa etiquetas, asi que no hay fuga, pero ajustarlo
sobre la union train+test si seria transduccion, que el reglamento prohibe.

**Los umbrales son los congelados de V03**, no unos nuevos. Reoptimizarlos seria otra
familia, ya cerrada, y ademas seria seleccion sobre el mismo conjunto que se evalua.

Uso
---
    /home/carlos/venv/bin/python experiments/v23_02_submission_rbf_sigmoid.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train

EXPERIMENTO = "V23-02 (refit sobre el train completo y submission de svd:rbf_svc sigmoidal)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v23_02_submission_rbf_sigmoid.json"
CSV_SALIDA = PROJECT_ROOT / "submissions" / "toxic_submission_v23_01_svd_rbf_sigmoid.csv"
CSV_CONTROL = PROJECT_ROOT / "resultados" / "v23_control_v03_reconstruido.csv"
CSV_DIFF = PROJECT_ROOT / "resultados" / "v23_diff_v03_v23.csv"
CONGELADA = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
N_SVD = 300
SEMILLA = 42
PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V23-02] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    """Calcula el sha256 de un fichero.

    Args:
        ruta: fichero a resumir.

    Returns:
        El resumen hexadecimal.
    """
    resumen = hashlib.sha256()
    with ruta.open("rb") as manejador:
        for bloque in iter(lambda: manejador.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def main() -> None:
    """Construye el control, la candidata y sus controles completos."""
    inicio = time.time()
    log("V23-02: refit y submission de svd:rbf_svc sigmoidal")

    log("1/7 integridad de las entradas")
    train = read_train()
    test = read_test()
    # `read_csv_robusto` deja la columna `id` como `object` porque el fichero esta en latin-1
    # y separado por `;`. Sin forzar el tipo, la comparacion de arrays con la submission
    # congelada falla aunque el contenido y el orden sean identicos, y la submission se
    # generaria con los identificadores como cadenas. Se convierte aqui, antes de cualquier
    # comparacion o escritura.
    test["id"] = test["id"].astype(np.int64)
    if any(c in test.columns for c in ETIQUETAS):
        raise SystemExit(
            "PARADA OBLIGATORIA: el fichero de test trae columna de etiqueta. No se debe usar"
        )
    if len(test) != 1154:
        raise SystemExit(f"PARADA OBLIGATORIA: el test tiene {len(test)} filas y se esperaban 1154")
    if test["id"].duplicated().any() or test["id"].isna().any() or test["text"].isna().any():
        raise SystemExit("PARADA OBLIGATORIA: hay identificadores duplicados o nulos en el test")
    if test["id"].dtype != np.int64:
        raise SystemExit(
            f"PARADA OBLIGATORIA: la columna id del test sigue en {test['id'].dtype} "
            f"en lugar de int64"
        )
    congelada = pd.read_csv(CONGELADA)
    if not np.array_equal(congelada["id"].to_numpy(), test["id"].to_numpy()):
        raise SystemExit(
            "PARADA OBLIGATORIA: los ids de la submission congelada no coinciden con los del test"
        )
    log(f"   train {train.shape} | test {test.shape} | congelada {congelada.shape}")
    log(f"   el test no trae columna de etiqueta: {not any(c in test.columns for c in ETIQUETAS)}")
    log(f"   sha256 de la submission congelada: {sha256(CONGELADA)}")

    log("2/7 matrices TF-IDF ajustadas SOLO con el train completo")
    textos_train = train["text"].astype(str)
    textos_test = test["text"].astype(str)
    palabra = TfidfVectorizer(**PARAMETROS_WORD)
    caracter = TfidfVectorizer(**PARAMETROS_CHAR)
    matriz_train = sparse.hstack(
        [palabra.fit_transform(textos_train), caracter.fit_transform(textos_train)]
    ).tocsr()
    matriz_test = sparse.hstack(
        [palabra.transform(textos_test), caracter.transform(textos_test)]
    ).tocsr()
    log(f"   train {matriz_train.shape} con densidad {matriz_train.nnz / (matriz_train.shape[0] * matriz_train.shape[1]):.6f}")
    log(f"   test  {matriz_test.shape}")

    log("3/7 CONTROL: reconstruir la submission de V03 con su propio recipe")
    proba_control = np.zeros((len(test), len(ETIQUETAS)))
    for j, etiqueta in enumerate(ETIQUETAS):
        modelo = LogisticRegression(
            C=2.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEMILLA
        )
        modelo.fit(matriz_train, train[etiqueta].to_numpy().astype(int))
        proba_control[:, j] = modelo.predict_proba(matriz_test)[:, 1]
    pred_control = (proba_control >= UMBRALES_V03[None, :]).astype(int)
    marco_control = pd.DataFrame({e: pred_control[:, j] for j, e in enumerate(ETIQUETAS)})
    marco_control.insert(0, "id", test["id"].to_numpy())
    marco_control.to_csv(CSV_CONTROL, index=False, lineterminator="\n")
    esperado = congelada[list(ETIQUETAS)].to_numpy()
    diferencias = int(np.count_nonzero(pred_control != esperado))
    log(
        f"   celdas que difieren de la submission congelada: {diferencias} de {esperado.size}"
    )
    log(f"   sha256 de la reconstruccion: {sha256(CSV_CONTROL)}")
    if diferencias:
        raise SystemExit(
            "PARADA OBLIGATORIA: la cadena de features no reproduce al champion. La "
            "candidata se construiria sobre una base distinta de la que se audito"
        )
    log("   **reproduccion exacta**: la cadena de construccion es la del champion")

    log(f"4/7 la candidata: TruncatedSVD({N_SVD}) ajustado con el train, y SVC RBF por etiqueta")
    svd = TruncatedSVD(n_components=N_SVD, random_state=SEMILLA)
    # El SVD se ajusta SOLO con el train. Ajustarlo con train+test seria transduccion.
    reducido_train = svd.fit_transform(matriz_train)
    reducido_test = svd.transform(matriz_test)
    # En scikit-learn el atributo lleva guion bajo final: `singular_values_`. Ademas se
    # informa de la energia capturada en componentes, que es la cifra comparable entre
    # pliegues y entre refits.
    energia = svd.singular_values_**2
    capturada = float(energia.sum() / energia.sum())
    log(
        f"   SVD: {reducido_train.shape} -> {reducido_test.shape} | componentes 300 | "
        f"primer valor singular {svd.singular_values_[0]:.4f} | energia capturada "
        f"{capturada:.6f}"
    )
    log("   soporte por etiqueta (numero de vectores de soporte):")
    proba_candidata = np.zeros((len(test), len(ETIQUETAS)))
    for j, etiqueta in enumerate(ETIQUETAS):
        modelo = SVC(
            C=2.0, class_weight="balanced", gamma="scale", probability=False, random_state=SEMILLA
        )
        modelo.fit(reducido_train, train[etiqueta].to_numpy().astype(int))
        decision = modelo.decision_function(reducido_test)
        proba_candidata[:, j] = 1.0 / (1.0 + np.exp(-decision))
        log(f"      {etiqueta}: {modelo.n_support_.sum()} vectores de soporte")

    log("5/7 prediccion y diff exacto contra V03, con los umbrales congelados")
    pred_candidata = (proba_candidata >= UMBRALES_V03[None, :]).astype(int)
    marco = pd.DataFrame({e: pred_candidata[:, j] for j, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(CSV_SALIDA, index=False, lineterminator="\n")
    cambios = pred_candidata != esperado
    filas_modificadas = int(np.count_nonzero(cambios.any(axis=1)))
    celdas_modificadas = int(np.count_nonzero(cambios))
    por_etiqueta = {ETIQUETAS[j]: int(np.count_nonzero(cambios[:, j])) for j in range(3)}
    tipos = {
        "0->1": int(np.count_nonzero(cambios & (esperado == 0) & (pred_candidata == 1))),
        "1->0": int(np.count_nonzero(cambios & (esperado == 1) & (pred_candidata == 0))),
    }
    log(f"   filas modificadas: {filas_modificadas} de {len(test)}")
    log(f"   celdas modificadas: {celdas_modificadas} | por etiqueta: {por_etiqueta}")
    log(f"   tipos de cambio: {tipos}")
    log(
        f"   tasa positiva V03 : "
        f"{[round(float(esperado[:, j].mean()), 4) for j in range(3)]}"
    )
    log(
        f"   tasa positiva V23 : "
        f"{[round(float(pred_candidata[:, j].mean()), 4) for j in range(3)]}"
    )
    log(
        "   desplazamiento por etiqueta: "
        + ", ".join(
            f"{ETIQUETAS[j]} {pred_candidata[:, j].mean() - esperado[:, j].mean():+.4f}"
            for j in range(3)
        )
    )

    log("6/7 diff celda a celda en disco")
    filas: list[dict[str, Any]] = []
    ids_test = test["id"].to_numpy()
    for i in np.flatnonzero(cambios.any(axis=1)):
        for j in range(3):
            if cambios[i, j]:
                filas.append(
                    {
                        "id": int(ids_test[i]),
                        "etiqueta": ETIQUETAS[j],
                        "score_v03": float(proba_control[i, j]),
                        "margen_rbf": float(
                            np.log(max(proba_candidata[i, j], 1e-12))
                            - np.log(max(1 - proba_candidata[i, j], 1e-12))
                        ),
                        "pred_v03": int(esperado[i, j]),
                        "pred_v23": int(pred_candidata[i, j]),
                    }
                )
    pd.DataFrame(filas).to_csv(CSV_DIFF, index=False, lineterminator="\n")
    log(f"   diff escrito con {len(filas)} filas")

    log("7/7 verificacion del fichero de submission")
    leido = pd.read_csv(CSV_SALIDA)
    comprobaciones = {
        "columnas": list(leido.columns) == ["id", "HS", "TR", "AG"],
        "filas_1154": len(leido) == 1154,
        "ids_en_el_orden_del_test": bool(np.array_equal(leido["id"].to_numpy(), test["id"].to_numpy())),
        "ids_unicos": not leido["id"].duplicated().any(),
        "valores_binarios": all(sorted(leido[c].unique().tolist()) == [0, 1] for c in ETIQUETAS),
        "sin_nulos": not leido.isna().any().any(),
    }
    for nombre, ok in comprobaciones.items():
        log(f"   {nombre}: {ok}")
    if not all(comprobaciones.values()):
        raise SystemExit("PARADA OBLIGATORIA: el fichero de submission no supera las comprobaciones")
    hash_salida = sha256(CSV_SALIDA)
    log(f"   sha256 de la submission: {hash_salida}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidata": "svd:rbf_svc con calibracion sigmoidal sobre decision_function",
        "evidencia_oof": {
            "corrected_cv": 0.790410,
            "delta_vs_v03": 0.013198,
            "pliegues_a_favor": "5/5",
            "celdas": 240,
            "rescates": 136,
            "danios": 104,
            "neto": 32,
            "mcnemar_p": 0.0452,
        },
        "entrenamiento": "las 1.731 filas del train completo, igual que V03",
        "control_de_cordura": {
            "celdas_distintas_de_la_congelada": diferencias,
            "sha256_congelada": sha256(CONGELADA),
            "sha256_reconstruccion": sha256(CSV_CONTROL),
        },
        "cambios_en_test": {
            "filas_modificadas": filas_modificadas,
            "celdas_modificadas": celdas_modificadas,
            "por_etiqueta": por_etiqueta,
            "tipos": tipos,
        },
        "comprobaciones_fichero": comprobaciones,
        "artefactos": {
            "submission": str(CSV_SALIDA.relative_to(PROJECT_ROOT)),
            "control": str(CSV_CONTROL.relative_to(PROJECT_ROOT)),
            "diff": str(CSV_DIFF.relative_to(PROJECT_ROOT)),
        },
        "sha256_submission": hash_salida,
        "gastos": {"submissions_enviadas": 0, "test_entrenado": False},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"   informe en {REPORTE_JSON.relative_to(PROJECT_ROOT)}")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: lista para enviar | control exacto {diferencias == 0} | cambios en test "
        f"{celdas_modificadas} celdas en {filas_modificadas} filas | submissions 0"
    )


if __name__ == "__main__":
    main()
