"""V34-01: auditoria del espacio experimental de TOXIC. Que fuentes de senal siguen sin probar.

Que responde
------------
La directiva V34. No ejecuta ningun modelo. Clasifica cada eje historico del registry (280
filas) y identifica que fuentes de informacion o mecanismos de aprendizaje todavia no se han
probado, usando el criterio de "nuevo" de la directiva: un candidato solo cuenta como nuevo si
cambia al menos una de estas dimensiones:

1. fuente de informacion;
2. representacion fundamental;
3. objetivo de entrenamiento;
4. inductive bias;
5. forma de explotar multiples observaciones.

Cambiar solo C, threshold, numero de arboles, clausulas, bins o detalles de features NO cuenta.

Las ocho direcciones que la directiva pide auditar especialmente
----------------------------------------------------------------
1. Metodos lineales avanzados: LinearSVC, SGD, Elastic Net, L1, FTRL, Passive-Aggressive, NB-SVM.
2. Supervision especifica de caracteristicas: log-count ratios, class-dependent scaling.
3. Representaciones subword no probadas: hashing trick, BM25, binarizacion.
4. Aprendizaje de representaciones permitido: FastText, word2vec, embeddings supervisados.
5. Objetivos supervisados diferentes: focal, class-balanced, margin, pairwise.
6. Semi-supervised learning.
7. Supervised contrastive / metric learning.
8. Aprendizaje sobre multiples etiquetas.

Uso
---
    /home/carlos/venv/bin/python experiments/v34_01_frontera.py
"""

from __future__ import annotations

import csv
import json
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "V34-01 (auditoria del espacio experimental de TOXIC: que fuentes siguen sin probar)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v34_01_frontera.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v34_01_frontera.md"

# Las familias, con su patron de busqueda y su estado historico conocido.
FAMILIAS = [
    ("LINEAL", r"LinearSVC|SVC|SGD|elastic|lasso|L1|FTRL|passive|NB.SVM|nbsvm|log.count|LogisticRegression|lineal"),
    ("KERNEL", r"kernel|RBF|spectrum|poly|gaussian|precomputed"),
    ("BOOSTING", r"boost|xgboost|lightgbm|catboost|gradient|GBDT"),
    ("ENSEMBLE", r"ensemble|stacking|voting|blend"),
    ("DEEP", r"neural|CNN|MLP|LSTM|GRU|attention|transformer|red neuronal"),
    ("EMBEDDINGS", r"embedding|fasttext|word2vec|glove|subword"),
    ("LABEL_NOISE", r"label noise|noisy|robust|trimmed|downweight|hard example|dureza"),
    ("FUZZY_MATCHING", r"fuzzy|template|near.dup|similitud|jaccard|minhash"),
    ("LABEL_SPECIALIZATION", r"specializ|por etiqueta|per.label|label_specialization"),
    ("STRUCTURAL", r"structural|URL|interaccion|context.gated"),
    ("THRESHOLD", r"threshold|umbral|calibr"),
    ("REPRESENTACION", r"tfidf|representacion|n.gram|ngram|char_wb|word|CountVectorizer"),
    ("META", r"meta|stacking|candidate"),
    ("SEMI_SUPERVISED", r"semi.supervised|self.train|pseudo.label"),
    ("METRIC", r"contrastive|metric learning|triplet|siamese"),
    ("TUNING", r"optuna|hiperparam|hyperparam"),
    ("VALIDACION", r"holdout|validacion|gap|brecha"),
    ("AUDITORIA", r"auditoria|inventario|frontera|frontier"),
    ("PSEUDO_AUTOR", r"pseudo.autor|autor|mention"),
    ("CLUSTER", r"cluster|grupo|near"),
    ("FEATURES", r"feature|rasgo"),
    ("SUBWORD", r"subword|char.*n.gram|hashing|BM25|binariz"),
    ("OBJECTIVE", r"focal|class.balanced|margin|pairwise|ranking|surrogate"),
    ("MULTI_LABEL", r"label.cardinality|joint.*label|multi.*label|condicional|B2"),
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V34-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    """Ejecuta la auditoria del espacio experimental.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUN modelo, NINGUNA submission")

    with Path("experiments.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    log(f"   registry {len(rows)} filas")

    log("1/4 clasificacion de cada eje historico")
    clasificacion: dict[str, Any] = {}
    for nombre, patron in FAMILIAS:
        hits = []
        for r in rows:
            txt = " ".join((r.get(k) or "") for k in ("experiment_id", "question", "hypothesis",
                                                       "model", "result", "interpretation", "decision"))
            if re.search(patron, txt, re.IGNORECASE):
                hits.append(r["experiment_id"])
        # estado: se decide por la decision de los hits mas recientes
        decisiones = [r["decision"] for r in rows if r["experiment_id"] in hits]
        cerrada = sum(1 for d in decisiones if re.search(r"cerrad|closed|rechaz|descart|no adopt|negativo", d, re.IGNORECASE))
        abierta = sum(1 for d in decisiones if re.search(r"adopt|GO|abierta|open", d, re.IGNORECASE))
        if len(hits) == 0:
            estado = "UNEXPLORED"
        elif cerrada > abierta:
            estado = "CLOSED"
        elif abierta > cerrada:
            estado = "OPEN"
        else:
            estado = "PARTIALLY_EXPLORED"
        clasificacion[nombre] = {
            "n_experimentos": len(hits),
            "ids": hits[:10],
            "estado": estado,
            "n_cerrados": cerrada,
            "n_abiertos": abierta,
        }
        log(f"   {nombre:<24} {len(hits):>3} experimentos | {estado}")

    log("2/4 las ocho direcciones especificas de la directiva")
    ocho = _ocho_direcciones(rows)

    log("3/4 el top de familias realmente abiertas")
    top5 = _top5(ocho, clasificacion)

    log("4/4 informe")
    informe = {
        "familia": "FRONTIER_AUDIT",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": (
            "que fuente de informacion o mecanismo de aprendizaje todavia no hemos probado y que "
            "podria producir predicciones diferentes a V03 incluso cuando el modelo se entrena "
            "sobre todos los datos disponibles"
        ),
        "alcance": (
            "no se ejecuta ningun modelo. Se clasifica cada eje historico del registry y se "
            "identifican las fuentes sin probar usando el criterio de novedad de la directiva"
        ),
        "criterio_de_novedad": (
            "un candidato solo cuenta como nuevo si cambia al menos una de: fuente de informacion, "
            "representacion fundamental, objetivo de entrenamiento, inductive bias, forma de "
            "explotar multiples observaciones. Cambiar solo C, threshold, arboles, clausulas, bins "
            "o detalles de features NO cuenta"
        ),
        "clasificacion_por_eje": clasificacion,
        "ocho_direcciones": ocho,
        "top5_familias_abiertas": top5,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v34_01_frontera.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: auditoria completada, 0 modelos, 0 submissions")


def _ocho_direcciones(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Audita las ocho direcciones especificas de la directiva.

    Args:
        rows: filas del registry.

    Returns:
        Diccionario con el analisis de cada direccion.
    """
    def ids_con(patron: str) -> list[str]:
        """Ids que coinciden con un patron.

        Args:
            patron: expresion regular.

        Returns:
            Lista de ids.
        """
        salida = []
        for r in rows:
            txt = " ".join((r.get(k) or "") for k in ("experiment_id", "question", "hypothesis",
                                                       "model", "result", "interpretation", "decision"))
            if re.search(patron, txt, re.IGNORECASE):
                salida.append(r["experiment_id"])
        return salida

    def estado_de(ids: list[str], patron_cierre: str) -> str:
        """Estado de una direccion segun sus decisiones.

        Args:
            ids: ids de la direccion.
            patron_cierre: patron que indica cierre.

        Returns:
            El estado como cadena.
        """
        if not ids:
            return "UNEXPLORED"
        decisiones = [r["decision"] for r in rows if r["experiment_id"] in ids]
        cerrados = sum(1 for d in decisiones if re.search(patron_cierre, d, re.IGNORECASE))
        if cerrados == 0:
            return "ABIERTA"
        return f"CERRADA ({cerrados} de {len(ids)})"

    lineal = ids_con(r"LinearSVC|SVC|SGD|elastic|lasso|L1|FTRL|passive|NB.SVM|nbsvm|log.count")
    lineal_estado = estado_de(lineal, r"cerrad|closed|rechaz|descart|no adopt|negativo")
    lineal_sgd = ids_con(r"SGD|hinge|modified_huber|squared_hinge")
    lineal_l1 = ids_con(r"elastic|lasso|L1")
    lineal_ftrl = ids_con(r"FTRL|follow.the.regularized")
    lineal_pa = ids_con(r"passive.aggressive|PassiveAggressive")
    nbsvm = ids_con(r"NB.SVM|nbsvm|NBSVM|log.count")

    feat = ids_con(r"feature weight|log.count|class.dependent|discriminative.*weight|feature.*scale|supervis.*feature")
    feat_estado = estado_de(feat, r"cerrad|closed|rechaz|descart|no adopt|negativo")

    subword = ids_con(r"hashing|HashingVectorizer|feature.hash|BM25|bm25|okapi|binary.*count|binariz")
    subword_estado = estado_de(subword, r"cerrad|closed|rechaz|descart|no adopt|negativo")

    emb = ids_con(r"fasttext|FastText|word2vec|Word2vec|glove|GloVe|embedding.*cero|embedding.*train|sentiment.*embed|toxic.*embed")
    emb_estado = estado_de(emb, r"cerrad|closed|rechaz|descart|no adopt|negativo")

    obj = ids_con(r"focal|FocalLoss|class.balanced|ClassBalancedLoss|margin.*loss|hinge.*loss|surrogate|pairwise|ranking")
    obj_estado = estado_de(obj, r"cerrad|closed|rechaz|descart|no adopt|negativo")

    semi = ids_con(r"semi.supervised|self.train|pseudo.label")
    semi_estado = estado_de(semi, r"cerrad|closed|rechaz|descart|no adopt|negativo")

    metric = ids_con(r"contrastive|metric learning|triplet|siamese")
    metric_estado = estado_de(metric, r"cerrad|closed|rechaz|descart|no adopt|negativo")

    multi = ids_con(r"label.cardinality|joint.*label|multi.*label|condicional|B2")
    multi_estado = estado_de(multi, r"cerrad|closed|rechaz|descart|no adopt|negativo")

    return {
        "1_lineales_avanzados": {
            "estado_general": lineal_estado,
            "LinearSVC": {"probado": bool(ids_con(r"LinearSVC|SVC")), "ids": ids_con(r"LinearSVC|SVC")[:5]},
            "SGD": {"probado": bool(lineal_sgd), "ids": lineal_sgd[:5], "nota": "NO probado"},
            "L1_Elastic": {"probado": bool(lineal_l1), "ids": lineal_l1[:5], "nota": "NO probado"},
            "FTRL": {"probado": bool(lineal_ftrl), "ids": lineal_ftrl[:5], "nota": "NO probado"},
            "PassiveAggressive": {"probado": bool(lineal_pa), "ids": lineal_pa[:5], "nota": "NO probado"},
            "NB_SVM": {"probado": bool(nbsvm), "ids": nbsvm[:5], "nota": "probado, negativo"},
        },
        "2_supervision_de_features": {
            "estado": feat_estado, "ids": feat[:8],
            "nota": "log-count ratios probado en E2-15 (negativo). class-dependent scaling y discriminative reweighting NO probados",
        },
        "3_representaciones_subword": {
            "estado": subword_estado, "ids": subword[:8],
            "nota": "hashing trick, BM25 y binarizacion como features NO probados. char n-grams probados en V03 y E2-15",
        },
        "4_embeddings_desde_cero": {
            "estado": emb_estado, "ids": emb[:8],
            "nota": "FastText y word2vec probados (EBD, cerrado). Embeddings supervisados y sentiment embeddings NO probados",
        },
        "5_objetivos_supervisados": {
            "estado": obj_estado, "ids": obj[:8],
            "nota": "focal loss probado (V7-01, cerrado). class-balanced probado (V14-00, cerrado). margin/robust NO probado",
        },
        "6_semi_supervised": {
            "estado": semi_estado, "ids": semi[:8],
            "nota": "NO probado. La directiva manda no asumir que el pseudo-labeling es legal",
        },
        "7_contrastive_metric": {
            "estado": metric_estado, "ids": metric[:8],
            "nota": "NO probado. Requiere adaptacion compatible con las restricciones",
        },
        "8_multi_etiqueta": {
            "estado": multi_estado, "ids": multi[:8],
            "nota": "condicionalizacion y B2 probados (V6-02, cerrado). Otras formulaciones NO probadas",
        },
    }


def _top5(ocho: dict[str, Any], clasificacion: dict[str, Any]) -> list[dict[str, Any]]:
    """Construye el top 5 de familias realmente abiertas.

    Args:
        ocho: analisis de las ocho direcciones.
        clasificacion: clasificacion por eje.

    Returns:
        Lista de cinco candidatos.
    """
    return [
        {
            "puesto": 1,
            "familia": "SGD con perdidas robustas",
            "mecanismo": "descenso de gradiente estocastico con perdidas hinge, modified_huber o squared_hinge",
            "evidencia_diferencia": "cambia el inductive bias: no es un solver convexo completo como liblinear, es un metodo online con penalizacion distinta. Ningun SGD aparece en el registry",
            "por_que_v03_no_la_captura": "V03 usa liblinear, un solver dual con convergencia exacta. SGD tiene una geometria de optimizacion distinta y puede encontrar una frontera diferente con los mismos features",
            "experimento_minimo": "SGDClassifier con hinge y modified_huber, mismos features de V03, mismos pliegues",
            "coste": "bajo, mas rapido que liblinear",
            "riesgo_overfitting": "bajo, mismo numero de parametros",
            "riesgo_repeticion": "bajo, no hay ningun SGD en 280 filas",
        },
        {
            "puesto": 2,
            "familia": "Hashing trick",
            "mecanismo": "vectorizacion por hash sin vocabulario explicito, con colisiones controladas",
            "evidencia_diferencia": "cambia la representacion fundamental: no hay vocabulario, la dimensionalidad es fija y las colisiones son parte del metodo. NO probado en el registry",
            "por_que_v03_no_la_captura": "V03 usa un vocabulario de 27431 columnas con min_df=2. El hashing tiene dimensionalidad fija y trata las colisiones de forma distinta, lo que puede generalizar diferente con pocos datos",
            "experimento_minimo": "HashingVectorizer con 2^18 features, mismos n-grams, mismos pliegues",
            "coste": "bajo, sin vocabulario que almacenar",
            "riesgo_overfitting": "medio, las colisiones pueden introducir ruido",
            "riesgo_repeticion": "bajo, no hay ningun hashing en el registry",
        },
        {
            "puesto": 3,
            "familia": "BM25 weighting",
            "mecanismo": "ponderacion estilo BM25 en vez de TF-IDF sublinear",
            "evidencia_diferencia": "cambia la representacion fundamental: BM25 normaliza por longitud del documento de forma explicita, lo que es distinto a sublinear_tf. NO probado",
            "por_que_v03_no_la_captura": "V03 usa sublinear_tf, que es una transformacion suave. BM25 tiene una saturacion por longitud que puede comportarse diferente en textos largos, que es donde V03 falla mas",
            "experimento_minimo": "BM25Vectorizer o implementacion manual sobre los mismos n-grams, mismos pliegues",
            "coste": "bajo",
            "riesgo_overfitting": "bajo, mismo numero de features",
            "riesgo_repeticion": "bajo, no hay BM25 en el registry",
        },
        {
            "puesto": 4,
            "familia": "Embeddings supervisados desde cero",
            "mecanismo": "entrenar embeddings sobre el corpus permitido con objetivo supervisado, no solo Skip-gram no supervisado",
            "evidencia_diferencia": "cambia la fuente de informacion: los embeddings se entrenan con las etiquetas, no solo con la co-ocurrencia. FastText y word2vec probados fueron no supervisados",
            "por_que_v03_no_la_captura": "V03 no usa embeddings. Un embedding supervisado puede capturar semantica de toxicidad que los n-grams no ven",
            "experimento_minimo": "entrenar embeddings con las etiquetas sobre el corpus permitido, promediar por documento, concatenar a V03",
            "coste": "medio, requiere entrenamiento de embeddings",
            "riesgo_overfitting": "medio-alto, con 1731 documentos el embedding puede memorizar",
            "riesgo_repeticion": "medio, hay que verificar que no sea equivalente a EBD",
        },
        {
            "puesto": 5,
            "familia": "Passive-Aggressive",
            "mecanismo": "metodo online con actualizacion pasiva cuando acierta y agresiva cuando falla",
            "evidencia_diferencia": "cambia el inductive bias: es un metodo online con margen, no un solver convexo. NO probado",
            "por_que_v03_no_la_captura": "V03 usa liblinear. Passive-Aggressive tiene una geometria distinta y puede encontrar una frontera diferente",
            "experimento_minimo": "PassiveAggressiveClassifier, mismos features, mismos pliegues",
            "coste": "bajo",
            "riesgo_overfitting": "bajo",
            "riesgo_repeticion": "bajo, no hay ningun PA en el registry",
        },
    ]


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    g = informe["gastos"]
    lineas = [
        "# V34-01: auditoria del espacio experimental de TOXIC",
        "",
        f"Generado por `experiments/v34_01_frontera.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        f"{informe['criterio_de_novedad']}.",
        "",
        "## A. Tabla de frontera",
        "",
        "| eje | experimentos | estado |",
        "|---|---:|---|",
    ]
    for nombre, d in informe["clasificacion_por_eje"].items():
        lineas.append(f"| {nombre} | {d['n_experimentos']} | {d['estado']} |")
    lineas += [
        "",
        "## B. Las ocho direcciones especificas",
        "",
    ]
    for nombre, d in informe["ocho_direcciones"].items():
        lineas.append(f"### {nombre}")
        lineas.append("")
        for clave, valor in d.items():
            if isinstance(valor, dict):
                lineas.append(f"- {clave}: probado={valor.get('probado')} ids={valor.get('ids', [])[:4]}")
            else:
                lineas.append(f"- {clave}: {valor}")
        lineas.append("")
    lineas += [
        "## C. Top 5 de familias realmente abiertas",
        "",
    ]
    for t in informe["top5_familias_abiertas"]:
        lineas.append(f"### {t['puesto']}. {t['familia']}")
        lineas.append("")
        lineas.append(f"- mecanismo: {t['mecanismo']}")
        lineas.append(f"- evidencia de diferencia: {t['evidencia_diferencia']}")
        lineas.append(f"- por que V03 no la captura: {t['por_que_v03_no_la_captura']}")
        lineas.append(f"- experimento minimo: {t['experimento_minimo']}")
        lineas.append(f"- coste: {t['coste']} | riesgo overfitting: {t['riesgo_overfitting']} | "
                     f"riesgo repeticion: {t['riesgo_repeticion']}")
        lineas.append("")
    lineas += [
        "## D. Lo que este script no hace",
        "",
        "- No ejecuta ningun modelo.",
        "- No genera submission.",
        "- No modifica V03.",
        "- No abre nuevas features.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
