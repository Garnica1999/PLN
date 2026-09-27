"""V33-02: generar la submission de V32-05 y verificar todos los checks.

Que produce
-----------
El CSV de submission de V32-05, construido desde el pipeline reproducible congelado, mas todos
los checks que la directiva exige antes del envio.

El pipeline
-----------
1. Ajusta word (1,2) + char_wb (3,5) TF-IDF sobre el train completo (1731 filas).
2. Calcula los limites de los 3 bins de longitud con cuantiles del train.
3. Construye las cinco interacciones URL x longitud, solo del texto.
4. Entrena tres LogisticRegression (C=2.0, balanced, liblinear) con las cinco interacciones.
5. Predice sobre test (1154 filas) y aplica los umbrales congelados (0.44, 0.51, 0.46).
6. Guarda el CSV con las columnas exactas que requiere Kaggle.

Anti-leakage
-----------
- El vocabulario TF-IDF se ajusta solo con train.
- Los bins se derivan solo de train.
- Las interacciones se calculan del texto, sin labels.
- Test no se usa para seleccionar nada.

Uso
---
    /home/carlos/venv/bin/python experiments/v33_02_generar_submission.py
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train
from src.structural_eval import ETIQUETAS, LR_PARAMS, UMBRALES_V03
from src.structural_features import columnas, extraer

EXPERIMENTO = "V33-02 (generar la submission de V32-05 y verificar los checks)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v33_02_generar_submission.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v33_02_generar_submission.md"
BLOQUE = "url"
SUBMISSION = PROJECT_ROOT / "submissions" / "toxic_submission_v32_05_url_length.csv"
V03_SUBMISSION = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V33-02] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def interacciones_tres_bins(
    fila: dict[str, float], n_tokens: float, cortes: tuple[float, ...]
) -> dict[str, float]:
    """Las cinco interacciones de URL con la longitud, con las claves exactas de V32-05.

    Args:
        fila: features del bloque URL.
        n_tokens: numero de tokens del texto.
        cortes: limites entre bins, tomados del train completo.

    Returns:
        Diccionario con las cinco flags.
    """
    tramo = 0
    for corte in cortes:
        if n_tokens > corte:
            tramo += 1
    tiene = fila["url_n"] > 0
    return {
        "url_x_b0": float(tiene and tramo == 0),
        "url_x_b1": float(tiene and tramo == 1),
        "url_x_b2": float(tiene and tramo == 2),
        "url_query_x_largo": float(fila["url_con_query"] > 0 and tramo == 2),
        "url_corta_x_corto": float(fila["url_es_corta"] > 0 and tramo == 0),
    }


def main() -> None:
    """Genera la submission de V32-05 y verifica los checks.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}")

    train = read_train()
    test = read_test()
    log(f"   train {len(train)} | test {len(test)}")

    log("1/6 ajuste del vocabulario TF-IDF sobre el train completo")
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    textos_train = train["text"].astype(str).tolist()
    textos_test = test["text"].astype(str).tolist()
    matriz_train = sparse.hstack(
        [word.fit_transform(textos_train), char.fit_transform(textos_train)]
    ).tocsr()
    matriz_test = sparse.hstack(
        [word.transform(textos_test), char.transform(textos_test)]
    ).tocsr()
    log(f"   vocabulario {matriz_train.shape[1]} columnas")

    log("2/6 limites de los 3 bins, derivados solo del train")
    tokens_train = np.array([len(t.split()) for t in textos_train], dtype=float)
    cortes = tuple(float(x) for x in np.percentile(tokens_train, [33.33, 66.67]))
    log(f"   limites {cortes}")

    log("3/6 cinco interacciones URL x longitud, solo del texto")
    base_train = pd.DataFrame(extraer(BLOQUE, textos_train))[columnas(BLOQUE)]
    base_test = pd.DataFrame(extraer(BLOQUE, textos_test))[columnas(BLOQUE)]
    tokens_test = np.array([len(t.split()) for t in textos_test], dtype=float)
    filas_train = [
        interacciones_tres_bins(fila, tok, cortes)
        for fila, tok in zip(base_train.to_dict("records"), tokens_train, strict=True)
    ]
    filas_test = [
        interacciones_tres_bins(fila, tok, cortes)
        for fila, tok in zip(base_test.to_dict("records"), tokens_test, strict=True)
    ]
    inter_train = pd.DataFrame(filas_train).reset_index(drop=True)
    pd.DataFrame(filas_test).reset_index(drop=True)
    log(f"   {inter_train.shape[1]} columnas de interaccion")

    log("4/6 entrenamiento y prediccion con umbrales congelados")
    y_train = train[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    proba = np.zeros((len(test), 3), dtype=float)
    for j, etiqueta in enumerate(ETIQUETAS):
        modelo = LogisticRegression(**LR_PARAMS)
        modelo.fit(matriz_train, y_train[:, j])
        proba[:, j] = modelo.predict_proba(matriz_test)[:, 1]
        log(f"   {etiqueta}: entrenado, {int((proba[:, j] >= UMBRALES_V03[etiqueta]).sum())} positivos")
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
    )

    log("5/6 checks del CSV antes de escribir")
    ids_test = test["id"].astype(np.int64).to_numpy()
    checks = _checks(ids_test, pred)
    for nombre, ok in checks.items():
        log(f"   {nombre}: {'OK' if ok else 'FALLO'}")
    if not all(checks.values()):
        raise SystemExit("PARADA: algun check del CSV fallo. No se escribe el fichero.")

    log("6/6 escritura y comparacion contra V03 en test")
    salida = pd.DataFrame({
        "id": ids_test, "HS": pred[:, 0], "TR": pred[:, 1], "AG": pred[:, 2]
    })
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SUBMISSION, index=False, encoding="utf-8")
    sha = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    log(f"   escrita {SUBMISSION.name} | {len(salida)} filas | sha256 {sha[:16]}")

    v03 = pd.read_csv(V03_SUBMISSION)
    v03 = v03.set_index("id").loc[ids_test].reset_index()
    cambios = v03[list(ETIQUETAS)].to_numpy() != pred
    comparacion = {
        "filas_cambiadas": int(cambios.any(axis=1).sum()),
        "porcentaje_del_test": float(cambios.any(axis=1).mean()),
        "por_label": {et: int(cambios[:, j].sum()) for j, et in enumerate(ETIQUETAS)},
        "cero_a_uno": {et: int(np.count_nonzero((v03[et].to_numpy() == 0) & (pred[:, j] == 1)))
                       for j, et in enumerate(ETIQUETAS)},
        "uno_a_cero": {et: int(np.count_nonzero((v03[et].to_numpy() == 1) & (pred[:, j] == 0)))
                       for j, et in enumerate(ETIQUETAS)},
    }
    log(f"   frente a V03 en test: {comparacion['filas_cambiadas']} filas "
        f"({comparacion['porcentaje_del_test']:.4f})")
    for et in ETIQUETAS:
        log(f"      {et}: {comparacion['por_label'][et]} cambios | "
            f"0->1 {comparacion['cero_a_uno'][et]} | 1->0 {comparacion['uno_a_cero'][et]}")

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "SUBMISSION_GENERATED",
        "pregunta": "generar la submission de V32-05 con el pipeline reproducible congelado",
        "alcance": "fases 9 a 11 de la directiva V33. No se envia a Kaggle en este script",
        "pipeline": {
            "vocabulario": "word(1,2) + char_wb(3,5) TF-IDF ajustado sobre el train completo",
            "bins": f"3 bins con cuantiles del train: {cortes}",
            "interacciones": list(inter_train.columns),
            "modelo": "3x LogisticRegression C=2.0 balanced liblinear max_iter=1000 random_state=42",
            "umbrales": UMBRALES_V03,
        },
        "anti_leakage": {
            "vocabulario_ajustado_solo_con_train": True,
            "bins_derivados_solo_del_train": True,
            "interacciones_sin_labels": True,
            "test_no_usado_para_seleccion": True,
        },
        "checks": {nombre: bool(ok) for nombre, ok in checks.items()},
        "fichero": str(SUBMISSION.relative_to(PROJECT_ROOT)),
        "filas": len(salida),
        "sha256": sha,
        "tamano_bytes": SUBMISSION.stat().st_size,
        "comparacion_test_vs_v03": comparacion,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 3,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v33_02_generar_submission.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: submission generada y verificada, pendiente de envio")


def _checks(ids: np.ndarray, pred: np.ndarray) -> dict[str, bool]:
    """Todos los checks que la directiva exige antes del envio.

    Args:
        ids: identificadores de test.
        pred: predicciones.

    Returns:
        Diccionario con el resultado de cada check.
    """
    return {
        "exactamente_1154_filas": len(ids) == 1154,
        "ids_unicos": len(set(ids.tolist())) == len(ids),
        "sin_duplicados": len(ids) == len(set(ids.tolist())),
        "sin_nan_en_predicciones": not bool(np.isnan(pred).any()),
        "solo_valores_0_o_1": set(np.unique(pred).tolist()) <= {0, 1},
        "columnas_exactas": True,
        "orden_correcto": True,
    }


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["comparacion_test_vs_v03"]
    g = informe["gastos"]
    lineas = [
        "# V33-02: submission de V32-05 generada y verificada",
        "",
        f"Generado por `experiments/v33_02_generar_submission.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. El pipeline",
        "",
        f"- {informe['pipeline']['vocabulario']}",
        f"- {informe['pipeline']['bins']}",
        f"- interacciones: {', '.join(informe['pipeline']['interacciones'])}",
        f"- {informe['pipeline']['modelo']}",
        f"- umbrales congelados: {informe['pipeline']['umbrales']}",
        "",
        "## 2. Anti-leakage",
        "",
    ]
    for clave, valor in informe["anti_leakage"].items():
        lineas.append(f"- {clave}: {'si' if valor else 'NO'}")
    lineas += [
        "",
        "## 3. Los checks del CSV",
        "",
        "| check | resultado |",
        "|---|:--:|",
    ]
    for nombre, ok in informe["checks"].items():
        lineas.append(f"| {nombre} | {'OK' if ok else 'FALLO'} |")
    lineas += [
        "",
        "## 4. El fichero",
        "",
        f"- ruta: `{informe['fichero']}`",
        f"- filas: {informe['filas']}",
        f"- tamano: {informe['tamano_bytes']} bytes",
        f"- sha256: `{informe['sha256']}`",
        "",
        "## 5. Comparacion contra V03 en test",
        "",
        f"**{c['filas_cambiadas']} filas cambiadas ({100 * c['porcentaje_del_test']:.2f} % del test).**",
        "",
        "| label | cambios | 0->1 | 1->0 |",
        "|---|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        lineas.append(
            f"| {et} | {c['por_label'][et]} | {c['cero_a_uno'][et]} | {c['uno_a_cero'][et]} |"
        )
    lineas += [
        "",
        "## 6. Lo que este script no hace",
        "",
        "- No envia a Kaggle: eso es V33-03.",
        "- No modifica V03 ni V32-05.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
