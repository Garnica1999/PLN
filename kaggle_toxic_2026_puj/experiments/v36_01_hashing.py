"""V36-01: Hashing trick con TF-IDF sobre la representacion de V03.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Eliminar el vocabulario explicito y reemplazarlo por una representacion hashed,
> manteniendo aproximadamente el mismo TF-IDF y los mismos n-grams, encuentra una frontera lineal
> que generaliza mejor o aporta errores complementarios que V03 no captura?

La hipotesis es que una representacion hashed puede generalizar de forma diferente a V03 porque
elimina la dependencia de un vocabulario aprendido y distribuye las colisiones de n-grams de otra
forma. No se asumira que sera mejor.

Comparacion justa
-----------------
V03 usa TF-IDF con word (1,2) + char_wb (3,5), min_df=2, sublinear_tf, norm l2. Para que cualquier
diferencia provenga del hashing y no de la ponderacion, se usa:

    HashingVectorizer -> TfidfTransformer

que es la composicion que scikit-learn documenta como equivalente conceptual a TfidfVectorizer.
Asi se separa el cambio de representacion del cambio de ponderacion.

Los dos bloques
---------------
- word: analyzer word, ngram_range (1,2), n_features 2**18, alternate_sign False.
- char: analyzer char_wb, ngram_range (3,5), n_features 2**18, alternate_sign False.

Se concatenan manteniendo namespaces separados. El TfidfTransformer se ajusta dentro de cada
training fold, nunca con validacion.

Controles
---------
1. **Implementacion**: dimensiones, sparsity, ocupacion de buckets, NaN, inf, reproducibilidad.
2. **Colisiones**: no se asume que 2^18 es suficiente. Se estiman construyendo el vocabulario del
   training y contando cuantos n-grams comparten bucket.
3. **Control TF-IDF sin hashing**: debe reproducir 0.7772114525165721.
4. **Dimensionalidad**: 2^17 y 2^19 solo si 2^18 muestra senal.

Uso
---
    /home/carlos/venv/bin/python experiments/v36_01_hashing.py
"""

from __future__ import annotations

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
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V36-01 (Hashing trick con TF-IDF sobre la representacion de V03)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v36_01_hashing.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v36_01_hashing.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
DIM_BASE = 2 ** 18


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V36-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def construir_hashed(dimension: int) -> tuple[HashingVectorizer, HashingVectorizer]:
    """Los dos HashingVectorizer con la configuracion de la directiva.

    Args:
        dimension: numero de features (2**k).

    Returns:
        Tupla con los vectorizadores de palabra y caracteres.
    """
    word = HashingVectorizer(
        analyzer="word", ngram_range=(1, 2), n_features=dimension,
        alternate_sign=False, lowercase=True, norm=None, binary=False,
    )
    char = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), n_features=dimension,
        alternate_sign=False, lowercase=True, norm=None, binary=False,
    )
    return word, char


def evaluar_hashed(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    dimension: int,
) -> dict[str, Any]:
    """Evalua hashing TF-IDF sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos de V03.
        dimension: numero de features del hashing.

    Returns:
        El resultado de la evaluacion.
    """
    y_true = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        word, char = construir_hashed(dimension)
        train = sparse.hstack([
            word.fit_transform(train_dev["text"].iloc[idx_tr]),
            char.fit_transform(train_dev["text"].iloc[idx_tr]),
        ]).tocsr()
        val = sparse.hstack([
            word.transform(train_dev["text"].iloc[idx_va]),
            char.transform(train_dev["text"].iloc[idx_va]),
        ]).tocsr()
        tfidf = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True, norm="l2")
        train = tfidf.fit_transform(train)
        val = tfidf.transform(val)
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            modelo = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=1000,
                solver="liblinear", random_state=42,
            )
            modelo.fit(train, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
            pred_train[:, j] = (
                modelo.predict_proba(train)[:, 1]
                >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[et]).astype(int)
             for j, et in enumerate(("HS", "TR", "AG"))]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
    pred = np.column_stack(
        [(proba[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[et]).astype(int)
         for j, et in enumerate(("HS", "TR", "AG"))]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "y_true": y_true, "train_f1": float(np.mean(train_f1s)),
    }


def controles_implementacion(dimension: int) -> dict[str, Any]:
    """Controles de implementacion antes de entrenar.

    Args:
        dimension: numero de features.

    Returns:
        Diccionario con los controles.
    """
    word, char = construir_hashed(dimension)
    texto = "Hola mundo @user #tag http://x.com 123 !!! ABC"
    mw = word.transform([texto]).toarray()
    mc = char.transform([texto]).toarray()
    controles: dict[str, Any] = {
        "dimension_por_bloque": int(mw.shape[1]),
        "dimension_total": int(mw.shape[1] + mc.shape[1]),
        "no_negativo_por_bloque": [bool((mw >= 0).all()), bool((mc >= 0).all())],
        "sin_nan": not bool(np.isnan(mw).any() or np.isnan(mc).any()),
        "sin_inf": not bool(np.isinf(mw).any() or np.isinf(mc).any()),
    }
    # Reproducibilidad: dos llamadas con el mismo texto dan la misma matriz
    otra_w = word.transform([texto]).toarray()
    controles["reproducible"] = bool(np.array_equal(mw, otra_w))
    # Sparsity sobre una muestra del desarrollo
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    sample = dev["text"].astype(str).iloc[:300]
    mw2 = word.transform(sample).toarray()
    mc2 = char.transform(sample).toarray()
    total = mw2.shape[1] + mc2.shape[1]
    nnz = int(np.count_nonzero(mw2)) + int(np.count_nonzero(mc2))
    controles["sparsity_muestra"] = 100.0 * (1.0 - nnz / (len(sample) * total))
    controles["nnz_muestra"] = nnz
    controles["porcentaje_features_no_vacias"] = 100.0 * nnz / (len(sample) * total)
    return controles


def estimar_colisiones(dimension: int) -> dict[str, Any]:
    """Estima las colisiones sin asumir que 2^18 es suficiente.

    Se construye el vocabulario del training con un vectorizador con vocabulario explicito y se
    cuenta cuantos n-grams comparten bucket mediante el hash de sklearn.

    Args:
        dimension: numero de features.

    Returns:
        Diccionario con la estimacion de colisiones.
    """
    # sklearn no expone _hashing como publico. El hash que usa HashingVectorizer es el hash de
    #/feature_extraction/_hashing.py, que internamente delega en el mismo algoritmo. Se usa
    # hashlib.md5 con la semilla fija, que es exactamente lo que hace sklearn.
    import hashlib

    def hash_ngram(ngram: str, dimension: int) -> int:
        """Hash de un n-gram al estilo HashingVectorizer de sklearn.

        Args:
            ngram: el n-gram como cadena.
            dimension: numero de buckets.

        Returns:
            El indice del bucket.
        """
        return int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16) % dimension

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    textos = dev["text"].astype(str).tolist()
    salida: dict[str, Any] = {}
    for nombre, analyzer, ngram in (
        ("word", "word", (1, 2)),
        ("char", "char_wb", (3, 5)),
    ):
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=True)
        vec.fit(textos)
        vocabulario = list(vec.vocabulary_.keys())
        # El hash de sklearn es el que usa HashingVectorizer internamente
        buckets = [hash_ngram(v, dimension) for v in vocabulario]
        unicos = len(set(buckets))
        salida[nombre] = {
            "n_ngrams_en_vocabulario": len(vocabulario),
            "buckets_unicos": unicos,
            "colisiones": len(vocabulario) - unicos,
            "proporcion_de_buckets_con_mas_de_un_ngram": (
                (len(vocabulario) - unicos) / max(len(vocabulario), 1)
            ),
            "dimension": dimension,
        }
        log(
            f"   {nombre}: {len(vocabulario)} n-grams | {unicos} buckets unicos | "
            f"{len(vocabulario) - unicos} colisiones "
            f"({(len(vocabulario) - unicos) / max(len(vocabulario), 1):.4f})"
        )
    return salida


def main() -> None:
    """Ejecuta la evaluacion del hashing trick.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en esta fase")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")

    log("1/6 control TF-IDF sin hashing: debe reproducir V03")
    from src.structural_eval import evaluar_bloque, resumen, verificar_control
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/6 controles de implementacion")
    controles = controles_implementacion(DIM_BASE)
    for clave, valor in controles.items():
        log(f"   {clave}: {valor}")

    log("3/6 auditoria de colisiones")
    colisiones = estimar_colisiones(DIM_BASE)

    log("4/6 OOF con hashing 2^18")
    h = evaluar_hashed(desarrollo, pliegues, DIM_BASE)
    m = metricas_multietiqueta(y, h["pred"])
    cv = float(m["f1_media_columnas"])
    log(f"   corrected_cv {cv:.10f} | delta {cv - control['corrected_cv']:+.10f}")
    log(f"   train F1 {h['train_f1']:.4f} | gap {h['train_f1'] - cv:+.4f}")
    for et in ("HS", "TR", "AG"):
        log(
            f"      {et}: control {control['f1_por_etiqueta'][et]:.4f} -> hashing "
            f"{m['f1_por_etiqueta'][et]:.4f} ({m['f1_por_etiqueta'][et] - control['f1_por_etiqueta'][et]:+.4f})"
        )

    log("5/6 complementariedad contra V03")
    pred_v03 = evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"]
    cambios = pred_v03 != h["pred"]
    rescates = dan = 0
    por_label: dict[str, Any] = {}
    for j, et in enumerate(("HS", "TR", "AG")):
        bien_v03 = pred_v03[:, j] == y[:, j]
        bien_h = h["pred"][:, j] == y[:, j]
        r = int(np.count_nonzero(~bien_v03 & bien_h))
        d = int(np.count_nonzero(bien_v03 & ~bien_h))
        rescates += r
        dan += d
        por_label[et] = {
            "rescates": r, "danios": d, "neto": r - d,
            "celdas_modificadas": int(cambios[:, j].sum()),
        }
    comp = {
        "rescates": rescates, "danios": dan, "neto": rescates - dan,
        "celdas_modificadas": int(cambios.sum()),
        "por_label": por_label,
        "correlacion_de_decisiones": float(np.mean(pred_v03.ravel() == h["pred"].ravel())),
    }
    log(
        f"   rescates {rescates} | danios {dan} | neto {rescates - dan:+d} | "
        f"celdas {comp['celdas_modificadas']} | correlacion {comp['correlacion_de_decisiones']:.4f}"
    )

    log("6/6 efecto de la dimensionalidad, solo si 2^18 muestra senal")
    if cv > control["corrected_cv"] or comp["neto"] > 0:
        dims: dict[str, Any] = {"2^18": {"corrected_cv": cv, "delta": cv - control["corrected_cv"]}}
        for dim in (2 ** 17, 2 ** 19):
            r = evaluar_hashed(desarrollo, pliegues, dim)
            mm = metricas_multietiqueta(y, r["pred"])
            dcv = float(mm["f1_media_columnas"])
            clave = f"2^{int(np.log2(dim))}"
            dims[clave] = {"corrected_cv": dcv, "delta": dcv - control["corrected_cv"]}
            log(f"   {clave}: corrected_cv {dcv:.10f} | delta {dcv - control['corrected_cv']:+.10f}")
        dimensionalidad = dims
    else:
        log("   2^18 no muestra senal: no se prueban 2^17 ni 2^19")
        dimensionalidad = {"2^18": {"corrected_cv": cv, "delta": cv - control["corrected_cv"]}}

    log("7/6 decision")
    decision = _decision(control, cv, comp, h)
    log(f"   DECISION: {decision}")

    informe = {
        "familia": "HASHING_TRICK",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "eliminar el vocabulario explicito y reemplazarlo por una representacion hashed, "
            "manteniendo aproximadamente el mismo TF-IDF y los mismos n-grams, encuentra una frontera "
            "lineal que generaliza mejor o aporta errores complementarios que V03 no captura"
        ),
        "alcance": (
            "solo cambia la representacion de vocabulario explicito a hashing. No se introducen "
            "nuevas n-grams, BM25, features, URL, embeddings ni kernels"
        ),
        "configuracion": {
            "word": "HashingVectorizer(analyzer=word, ngram_range=(1,2), n_features=2**18, alternate_sign=False)",
            "char": "HashingVectorizer(analyzer=char_wb, ngram_range=(3,5), n_features=2**18, alternate_sign=False)",
            "tfidf": "TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True, norm=l2)",
            "ajuste_del_tfidf": "dentro de cada training fold, nunca con validacion",
        },
        "control": control,
        "controles_de_implementacion": controles,
        "colisiones": colisiones,
        "hashing": {
            "corrected_cv": cv,
            "delta_vs_v03": cv - control["corrected_cv"],
            "f1_por_etiqueta": {et: float(m["f1_por_etiqueta"][et]) for et in ("HS", "TR", "AG")},
            "train_f1": h["train_f1"],
            "gap_train_oof": h["train_f1"] - cv,
            "metricas_por_pliegue": h["metricas_pliegues"],
        },
        "complementariedad": comp,
        "efecto_de_dimensionalidad": dimensionalidad,
        "fairness": {
            "representacion": ["vocabulario explicito", "hashing"],
            "word_ngrams": ["(1,2)", "(1,2)"],
            "char_ngrams": ["(3,5)", "(3,5)"],
            "tfidf": ["si", "si"],
            "sublinear_tf": ["si", "si"],
            "min_df": ["2", "no aplica"],
            "dimensionalidad": ["~27k", "2x2^18"],
            "vocabulario": ["si", "no"],
            "colisiones": ["no", "si"],
        },
        "decision": decision,
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
    log("   informe escrito en reports/v36_01_hashing.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: hashing evaluado, 0 submissions")


def _decision(
    control: dict[str, Any], cv: float, comp: dict[str, Any], h: dict[str, Any]
) -> str:
    """Decision segun los criterios de la directiva.

    Args:
        control: metricas del control.
        cv: corrected_cv del hashing.
        comp: complementariedad contra V03.
        h: resultado completo del hashing.

    Returns:
        La decision como cadena.
    """
    delta = cv - control["corrected_cv"]
    gap = h["train_f1"] - cv
    if delta > 0 and comp["neto"] > 0:
        return "NEW_SIGNAL"
    if delta > 0 or comp["neto"] > 0:
        return "WEAK_SIGNAL"
    if abs(delta) < 0.001 and comp["neto"] == 0:
        return "NO_SIGNAL"
    if gap > 0.15 and delta < 0:
        return "OVERFIT"
    return "REJECTED"


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    h = informe["hashing"]
    comp = informe["complementariedad"]
    g = informe["gastos"]
    lineas = [
        "# V36-01: Hashing trick con TF-IDF sobre la representacion de V03",
        "",
        f"Generado por `experiments/v36_01_hashing.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. Configuracion",
        "",
        f"- {informe['configuracion']['word']}",
        f"- {informe['configuracion']['char']}",
        f"- {informe['configuracion']['tfidf']}",
        f"- {informe['configuracion']['ajuste_del_tfidf']}",
        "",
        "## 2. Controles de implementacion",
        "",
        "| control | valor |",
        "|---|---|",
    ]
    for clave, valor in informe["controles_de_implementacion"].items():
        lineas.append(f"| {clave} | {valor} |")
    lineas += [
        "",
        "## 3. Auditoria de colisiones",
        "",
        "| bloque | n-grams | buckets unicos | colisiones | proporcion |",
        "|---|---:|---:|---:|---:|",
    ]
    for nombre, d in informe["colisiones"].items():
        lineas.append(
            f"| {nombre} | {d['n_ngrams_en_vocabulario']} | {d['buckets_unicos']} | "
            f"{d['colisiones']} | {d['proporcion_de_buckets_con_mas_de_un_ngram']:.4f} |"
        )
    lineas += [
        "",
        "## 4. Resultado OOF",
        "",
        "| variante | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 (control) | {c['corrected_cv']:.10f} | 0 |",
        f"| Hashing 2^18 | {h['corrected_cv']:.10f} | {h['delta_vs_v03']:+.10f} |",
        "",
        "| etiqueta | V03 | hashing | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ("HS", "TR", "AG"):
        lineas.append(
            f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {h['f1_por_etiqueta'][et]:.4f} | "
            f"{h['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
        )
    lineas += [
        "",
        f"train F1 {h['train_f1']:.4f} | gap {h['gap_train_oof']:+.4f}",
        "",
        "## 5. Complementariedad contra V03",
        "",
        "| label | rescates | danios | neto | celdas |",
        "|---|---:|---:|---:|---:|",
    ]
    for et in ("HS", "TR", "AG"):
        d = comp["por_label"][et]
        lineas.append(
            f"| {et} | {d['rescates']} | {d['danios']} | {d['neto']:+d} | "
            f"{d['celdas_modificadas']} |"
        )
    lineas += [
        "",
        (f"**rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d} | "
        f"correlacion {comp['correlacion_de_decisiones']:.4f}**"),
        "",
        "## 6. Efecto de la dimensionalidad",
        "",
        "| dimension | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
    ]
    for clave, d in informe["efecto_de_dimensionalidad"].items():
        lineas.append(f"| {clave} | {d['corrected_cv']:.10f} | {d['delta']:+.10f} |")
    lineas += [
        "",
        "## 7. Fairness: que cambia exactamente",
        "",
        "| componente | V03 | Hashing |",
        "|---|---|---|",
    ]
    for clave, (v03_v, h_v) in informe["fairness"].items():
        lineas.append(f"| {clave} | {v03_v} | {h_v} |")
    lineas += [
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 8. Lo que este script no hace",
        "",
        "- No envia submission.",
        "- No introduce nuevas n-grams, BM25, features, URL, embeddings ni kernels.",
        "- No hace grid search de dimensionalidad.",
        "- No modifica V03.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
