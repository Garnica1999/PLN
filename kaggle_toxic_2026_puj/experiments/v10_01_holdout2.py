"""P0b de la fase v10 (V10-01): holdout2 congelado y recalculo de la referencia en dev2.

Contexto
--------
El holdout v2 esta agotado, con sus cinco de cinco evaluaciones gastadas. La regla 5 de la
fase v10 impide adoptar nada apoyandose solo en el OOF, asi que hace falta un conjunto de
evaluacion nuevo. Se crea aqui, **antes** de la via A, y no despues por una razon concreta:
si se creara despues, las candidatas de la via A se elegirian sobre las mismas filas que
luego se usarian para juzgarlas, y el juez dejaria de ser independiente de la seleccion.

La advertencia que hay que dejar escrita antes de ver resultados
-----------------------------------------------------------------
**Holdout2 no es un conjunto virginal.** Se construye con un pliegue del desarrollo actual, y
esas 1483 filas ya participaron en las decisiones de las fases v1 a v9. Su valor real es
**detectar que un candidato esta sobreajustado a su gate**, que es un uso legitimo y
necesario, y **no** es probar generalizacion sobre datos nunca vistos. Una mejora aprobada
aqui debe leerse como "sobrevive a un conjunto que ya se uso para decidir otras cosas".

Que construye
-------------
1. Una particion de grupo de seis pliegues sobre las 1483 filas del desarrollo, con semilla
   20260926. Un pliegue de unas 247 filas es **holdout2** y los otros cinco de unas 1236 son
   **dev2**, el desarrollo nuevo.
2. Dos ficheros congelados con su huella: ``config/holdout2.json`` con la configuracion
   completa y las prevalencias, y ``data/processed/holdout2_ids.csv`` con los identificadores.
3. La referencia de V03 **recalculada en dev2** con sus propios cinco pliegues, que es la cifra
   comparable con un OOF y no una resubstitucion. Los 0.777211 historicos se conservan como
   referencia del desarrollo completo y no se sustituyen.

Gates
-----
- Cero grupos compartidos entre holdout2 y dev2, y ninguna fila en ambos.
- La particion es **reproducible**: se vuelve a construir y se comparan los identificadores.
- La nueva referencia se aparta de los 0.7772 en no mas de 0.005. Si se aparta mas, es
  **coste de reducir el conjunto de ajuste** y se documenta, sin usarlo para adoptar nada.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_01_holdout2.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import (
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
)

EXPERIMENTO = "V10-01 (P0b, holdout2)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_01_holdout2.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_01_holdout2.json"
CONFIG_HOLDOUT2 = PROJECT_ROOT / "config" / "holdout2.json"
IDS_HOLDOUT2 = PROJECT_ROOT / "data" / "processed" / "holdout2_ids.csv"
LOG_HOLDOUT = PROJECT_ROOT / "config" / "holdout_v2.json"

REFERENCIA_HISTORICA = 0.7772114525165721
LB_V03 = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_REFERENCIA = 0.005
NUMERO_PLEGUEES = 6
SEMILLA_HOLDOUT2 = 20260926
PLIEGUES_DEV2 = 5
MAXIMO_EVALUACIONES = 3

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-P0b] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: textos a transformar.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    matriz = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos_consulta),
            vectorizador_char.transform(textos_consulta),
        ],
        format="csr",
    )
    return matriz, consulta


def calcular_referencia_dev2(
    textos: pd.Series,
    y: np.ndarray,
    grupos: np.ndarray,
    identificadores: pd.Series,
) -> dict[str, Any]:
    """Recalcula la referencia de V03 sobre dev2 con cinco pliegues propios.

    Se hace con validacion cruzada **dentro** de dev2, y no ajustando y evaluando sobre las
    mismas filas: una cifra de resubstitucion no es comparable con los 0.7772 historicos, que
    son un promedio fuera de muestra.

    Args:
        textos: serie de textos de dev2.
        y: matriz binaria de etiquetas de dev2.
        grupos: grupos fusionados de dev2.
        identificadores: identificadores de dev2, para alinear las etiquetas.

    Returns:
        Diccionario con la referencia y su desglose.
    """
    probabilidades = np.zeros((len(y), 3), dtype=np.float64)
    pliegues_dev2 = list(
        StratifiedGroupKFold(
            n_splits=PLIEGUES_DEV2, shuffle=True, random_state=SEMILLA_HOLDOUT2
        ).split(textos, combinaciones(pd.DataFrame({e: y[:, i] for i, e in enumerate(ETIQUETAS)})),
                groups=grupos)
    )
    for entrenamiento, validacion in pliegues_dev2:
        matriz, consulta = construir_espacio(
            textos.iloc[entrenamiento].reset_index(drop=True),
            textos.iloc[validacion].reset_index(drop=True),
        )
        for etiqueta in range(3):
            estimador = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(matriz, y[entrenamiento, etiqueta])
            probabilidades[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    predicciones = aplicar_umbrales(probabilidades)
    return {
        "corrected_cv2": round(metrica_macro(y, predicciones), 16),
        "f1_por_etiqueta": {
            k: round(v, 6)
            for k, v in metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"].items()
        },
        "filas": len(y),
        "pliegues": PLIEGUES_DEV2,
        "ids": identificadores.tolist(),
    }


def main() -> None:
    """Construye y congela el holdout2, y recalcula la referencia en dev2."""
    inicio = time.time()
    log("P0b: holdout2 congelado y referencia recalculada en dev2")
    train = read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(
        f"   desarrollo actual: {len(y)} filas | {len(np.unique(grupos_dev))} grupos | "
        f"particion de {NUMERO_PLEGUEES} pliegues con semilla {SEMILLA_HOLDOUT2}"
    )

    log("1/5 construir la particion dos veces y comprobar que es reproducible")
    marco = pd.DataFrame({e: y[:, i] for i, e in enumerate(ETIQUETAS)})
    primera = list(
        StratifiedGroupKFold(
            n_splits=NUMERO_PLEGUEES, shuffle=True, random_state=SEMILLA_HOLDOUT2
        ).split(textos, combinaciones(marco), groups=grupos_dev)
    )
    segunda = list(
        StratifiedGroupKFold(
            n_splits=NUMERO_PLEGUEES, shuffle=True, random_state=SEMILLA_HOLDOUT2
        ).split(textos, combinaciones(marco), groups=grupos_dev)
    )
    reproducible = all(
        np.array_equal(np.sort(a), np.sort(a2)) and np.array_equal(np.sort(b), np.sort(b2))
        for (_, a), (_, b), (_, a2), (_, b2) in zip(
            primera, segunda, primera, segunda, strict=True
        )
    )
    log(f"   particion reproducible: {reproducible}")
    if not reproducible:
        log("PARADA: la particion no es reproducible con la misma semilla.")
        return

    validacion = primera[0][1]
    entrenamiento = primera[0][0]
    ids_holdout2 = desarrollo["id"].to_numpy()[validacion]
    grupos_holdout2 = np.unique(grupos_dev[validacion])
    grupos_dev2 = np.unique(grupos_dev[entrenamiento])
    compartidos = set(grupos_holdout2.tolist()) & set(grupos_dev2.tolist())
    log(
        f"   holdout2: {len(validacion)} filas en {len(grupos_holdout2)} grupos | "
        f"dev2: {len(entrenamiento)} filas en {len(grupos_dev2)} grupos"
    )
    log(f"   grupos compartidos entre los dos: {len(compartidos)}")
    log(
        "   prevalencias holdout2: "
        + ", ".join(
            f"{e} {y[validacion, i].mean():.4f}" for i, e in enumerate(ETIQUETAS)
        )
    )
    log(
        "   prevalencias dev2: "
        + ", ".join(
            f"{e} {y[entrenamiento, i].mean():.4f}" for i, e in enumerate(ETIQUETAS)
        )
    )
    if compartidos:
        log("PARADA: hay grupos compartidos, la particion no es(group-)limpia.")
        return

    log("2/5 recalcular la referencia de V03 en dev2, con cinco pliegues propios")
    referencia = calcular_referencia_dev2(
        textos.iloc[entrenamiento].reset_index(drop=True),
        y[entrenamiento],
        grupos_dev[entrenamiento],
        desarrollo["id"].to_numpy()[entrenamiento],
    )
    desviacion = referencia["corrected_cv2"] - REFERENCIA_HISTORICA
    referencia["desviacion_frente_a_historica"] = round(desviacion, 6)
    log(
        f"   corrected_cv2 = {referencia['corrected_cv2']:.6f} frente a "
        f"{REFERENCIA_HISTORICA:.6f} historica, desviacion {desviacion:+.6f}"
    )
    referencia_ok = abs(desviacion) <= TOLERANCIA_REFERENCIA
    log(
        f"   desviacion dentro de la tolerancia de {TOLERANCIA_REFERENCIA}: {referencia_ok}"
        + ("" if referencia_ok else ". Es coste de reducir el conjunto de ajuste.")
    )

    log("3/5 congelar la configuracion y los identificadores con su huella")
    IDS_HOLDOUT2.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids_holdout2}).to_csv(IDS_HOLDOUT2, index=False, lineterminator="\n")
    huella_ids = hashlib.sha256(IDS_HOLDOUT2.read_bytes()).hexdigest()
    # El identificador es entero en unas fuentes y cadena en otras, y se guarda como cadena
    # para que el archivo sea estable entre maquinas.
    config = {
        "creado": "2026-09-26",
        "experimento": EXPERIMENTO,
        "motivo": (
            "El holdout v2 esta agotado con 5 de 5 evaluaciones, y la regla 5 de la fase v10 "
            "impide adoptar nada apoyandose solo en el OOF."
        ),
        "advertencia": (
            "NO ES VIRGINAL. Sus filas salen del desarrollo actual, que ya participo en las "
            "decisiones de las fases v1 a v9. Su valor es detectar sobreajuste al gate de "
            "adopcion, no probar generalizacion sobre datos nunca vistos."
        ),
        "particion": {
            "estrategia": "StratifiedGroupKFold",
            "n_splits": NUMERO_PLEGUEES,
            "shuffle": True,
            "random_state": SEMILLA_HOLDOUT2,
            "grupos": "near-duplicado UNION pseudo-autor, los mismos grupos fusionados de v2",
            "estrategia_de_combinacion": "combinaciones de bits de las tres etiquetas",
        },
        "composicion": {
            "filas_desarrollo_original": len(y),
            "filas_holdout2": len(validacion),
            "filas_dev2": len(entrenamiento),
            "grupos_holdout2": len(grupos_holdout2),
            "grupos_dev2": len(grupos_dev2),
            "grupos_compartidos": len(compartidos),
        },
        "prevalencias": {
            "holdout2": {
                e: round(float(y[validacion, i].mean()), 6) for i, e in enumerate(ETIQUETAS)
            },
            "dev2": {
                e: round(float(y[entrenamiento, i].mean()), 6) for i, e in enumerate(ETIQUETAS)
            },
        },
        "referencias": {
            "corrected_cv_desarrollo_original": REFERENCIA_HISTORICA,
            "corrected_cv2_dev2": referencia["corrected_cv2"],
            "desviacion": round(desviacion, 6),
            "lb_v03": LB_V03,
            "umbrales": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
        },
        "regla_de_uso": {
            "maximo_evaluaciones": MAXIMO_EVALUACIONES,
            "solo_para": "decisiones de adopcion pre-registradas",
            "prohibido": "optimizar nada sobre holdout2, y usarlo mas de tres veces",
        },
        "artefactos": {
            "ids": str(IDS_HOLDOUT2.relative_to(PROJECT_ROOT)),
            "sha256_de_los_ids": huella_ids,
        },
    }
    CONFIG_HOLDOUT2.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    huella_config = hashlib.sha256(CONFIG_HOLDOUT2.read_bytes()).hexdigest()
    log(f"   config/holdout2.json escrito, sha256 {huella_config[:16]}...")
    log(f"   data/processed/holdout2_ids.csv escrito, sha256 {huella_ids[:16]}...")

    log("4/5 comparar contra el holdout v2, para verificar que no solapan")
    if LOG_HOLDOUT.exists():
        config_v2 = json.loads(LOG_HOLDOUT.read_text(encoding="utf-8"))
        ids_v2 = set()
        for valor in config_v2.values():
            if isinstance(valor, list):
                ids_v2 |= {str(x) for x in valor}
        ids_holdout2_str = {str(x) for x in ids_holdout2}
        solape_v2 = len(ids_holdout2_str & ids_v2)
        log(f"   identificadores de holdout2 que tambien estan en la config de v2: {solape_v2}")
    else:
        solape_v2 = None
        log("   no se encontro config/holdout_v2.json, la comparacion se omite")

    log("5/5 informe")
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "particion_reproducible": bool(reproducible),
        "grupos_compartidos": len(compartidos),
        "referencia_dev2": referencia,
        "referencia_dentro_de_tolerancia": bool(referencia_ok),
        "solape_con_holdout_v2": solape_v2,
        "sha256_config": huella_config,
        "sha256_ids": huella_ids,
        "gate_superado": bool(reproducible and not compartidos and referencia_ok),
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(config, informe, referencia, solape_v2, inicio)
    log(f"P0b completada en {round(time.time() - inicio, 1)} s")
    log(f"VEREDICTO P0b: {'GATE SUPERADO' if informe['gate_superado'] else 'GATE NO SUPERADO'}")


def _informe(
    config: dict[str, Any],
    informe: dict[str, Any],
    referencia: dict[str, Any],
    solape_v2: int | None,
    inicio: float,
) -> None:
    """Escribe el informe de la creacion del holdout2.

    Args:
        config: configuracion congelada.
        informe: veredicto del experimento.
        referencia: referencia recalculada en dev2.
        solape_v2: solape con el holdout v2, si se pudo comprobar.
        inicio: instante de arranque del script.
    """
    comp = config["composicion"]
    lineas = [
        "# P0b: holdout2 congelado y referencia recalculada en dev2 (V10-01)",
        "",
        f"Generado por `experiments/v10_01_holdout2.py` ({EXPERIMENTO}).",
        "",
        "## 1. La advertencia, escrita antes de ver resultados",
        "",
        "**Holdout2 no es un conjunto virginal.** Se construye con un pliegue del desarrollo",
        "actual, y esas 1483 filas ya participaron en las decisiones de las fases v1 a v9.",
        "",
        "Su valor real es **detectar que un candidato esta sobreajustado a su gate**, que es un",
        "uso legitimo y necesario, y **no** es probar generalizacion sobre datos nunca vistos.",
        "Una mejora aprobada aqui debe leerse como *sobrevive a un conjunto que ya se uso para",
        "decidir otras cosas*.",
        "",
        "Por eso holdout2 se crea **antes** de la via A y no despues: si se creara despues, las",
        "candidatas de la via A se elegirian sobre las mismas filas que luego se usarian para",
        "juzgarlas, y el juez dejaria de ser independiente de la seleccion.",
        "",
        "## 2. Composicion",
        "",
        "| conjunto | filas | grupos |",
        "|---|---|---|",
        (
            f"| desarrollo original | {comp['filas_desarrollo_original']} | "
            f"{comp['grupos_holdout2'] + comp['grupos_dev2']} |"
        ),
        f"| holdout2 | {comp['filas_holdout2']} | {comp['grupos_holdout2']} |",
        f"| dev2 | {comp['filas_dev2']} | {comp['grupos_dev2']} |",
        "",
        f"**Grupos compartidos entre holdout2 y dev2: {comp['grupos_compartidos']}.** La",
        "particion se construye con los mismos grupos fusionados que en v2, near-duplicado",
        "unido a pseudo-autor, de modo que la limpieza es de grupo y no solo de fila.",
        "",
        "Prevalencias:",
        "",
        "| conjunto | HS | TR | AG |",
        "|---|---|---|---|",
        (
            f"| holdout2 | {config['prevalencias']['holdout2']['HS']:.4f} | "
            f"{config['prevalencias']['holdout2']['TR']:.4f} | "
            f"{config['prevalencias']['holdout2']['AG']:.4f} |"
        ),
        (
            f"| dev2 | {config['prevalencias']['dev2']['HS']:.4f} | "
            f"{config['prevalencias']['dev2']['TR']:.4f} | "
            f"{config['prevalencias']['dev2']['AG']:.4f} |"
        ),
        "| desarrollo original | 0.4367 | 0.2998 | 0.3824 |",
        "",
        "## 3. La referencia recalculada en dev2",
        "",
        f"- `corrected_cv2`: **{referencia['corrected_cv2']:.6f}**",
        f"- frente a los {config['referencias']['corrected_cv_desarrollo_original']:.6f} del",
        f"  desarrollo original, desviacion **{referencia['desviacion_frente_a_historica']:+.6f}**",
        f"- tolerancia declarada: {TOLERANCIA_REFERENCIA}",
        f"- dentro de tolerancia: **{informe['referencia_dentro_de_tolerancia']}**",
        "",
        "F1 por etiqueta: " + ", ".join(
            f"{e} {v:.4f}" for e, v in referencia["f1_por_etiqueta"].items()
        ),
        "",
        "La referencia se recalcula con **cinco pliegues propios dentro de dev2**, y no ajustando",
        "y evaluando sobre las mismas filas: una cifra de resubstitucion no es comparable con los",
        "0.7772 historicos, que son un promedio fuera de muestra.",
        "",
        "**Los 0.777211 historicos no se sustituyen.** Se conservan como referencia del",
        "desarrollo completo, y la nueva cifra convive con ellos para las comparaciones dentro",
        "de dev2.",
        "",
        "## 4. Comprobaciones de integridad",
        "",
        f"- particion reproducible con la misma semilla: **{informe['particion_reproducible']}**",
        f"- grupos compartidos: **{informe['grupos_compartidos']}**",
        f"- solape con la configuracion del holdout v2: {solape_v2 if solape_v2 is not None else 'no comprobado'}",
        f"- `sha256` de `config/holdout2.json`: {informe['sha256_config']}",
        f"- `sha256` de `data/processed/holdout2_ids.csv`: {informe['sha256_ids']}",
        "",
        "## 5. Regla de uso, congelada con el conjunto",
        "",
        f"- **maximo {MAXIMO_EVALUACIONES} evaluaciones**, y solo para decisiones de adopcion",
        "  pre-registradas.",
        "- prohibido optimizar nada sobre holdout2.",
        "",
        "## 6. Veredicto",
        "",
        f"**{'GATE SUPERADO' if informe['gate_superado'] else 'GATE NO SUPERADO'}.** A partir de",
        "aqui ninguna decision de adopcion puede apoyarse solo en el OOF, y cualquier candidata",
        "que llegue a la mesa de finales necesita una evaluacion unica aqui.",
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
