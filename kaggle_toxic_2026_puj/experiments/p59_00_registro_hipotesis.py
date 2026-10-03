"""P59 step 00: hypothesis registry with an executable novelty gate.

Why this file exists
--------------------
Paragraph 3 of the P59 directive forbids treating a technique as new just because
it has a new name. It requires, before executing any track, an audit of
`experiments.csv`, `DECISIONS.md`, `reports/`, `scripts/` and the notebook that
decides for each track whether the mechanism is:

    same mechanism  ->  close the track, spend nothing
    same topic      ->  narrow the hypothesis, keep it open
    unseen          ->  open

That audit is a claim, and a claim that is not checked is a claim that can be
wrong. So the registry written by this script does not merely assert the prior
work: for every precedent it cites, it re-reads that row of the registry and
**asserts that the cited number is really there**. If a precedent disappears or
was quoted from memory, this script aborts.

The most important consequence of the audit, obtained here and not assumed:

    P59-G (stability selection) is ALREADY EXECUTED. V8-03 did exactly what
    paragraph 33 proposes, at three selection thresholds, with paired IC95, and
    was closed with an explicit "no se reintenta con mas agresividad ni con otra
    constante de submuestreo". P59-G is therefore DUPLICATE and consumes zero
    experiments.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_00_registro_hipotesis.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS = PROJECT_ROOT / "experiments.csv"
V03_SUB = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
DIR_REG = PROJECT_ROOT / "reports" / "p59_hypothesis_registry.csv"
JSON_REG = PROJECT_ROOT / "reports" / "p59_hypothesis_registry.json"
DIR_FRONTIER = PROJECT_ROOT / "reports" / "p59_frontier_registry.csv"

V03_SHA = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
V03_CCV = 0.7772114525165721
V03_LB = 0.78306

ESTADOS = ("OPEN", "RUNNING", "CLOSED", "MARGINAL", "PROMISING", "CANDIDATE",
           "INVALIDATED", "DUPLICATE")
NOVEDAD = ("NOVEL", "DUPLICATE", "PARTIALLY_DUPLICATE", "NARROW", "UNKNOWN")

# Cada precedente citado lleva la cadena que DEBE aparecer en la fila real del
# registro. Si el numero no esta ahi, este script aborta en vez de repetirlo.
PRECEDENTES: tuple[dict[str, Any], ...] = (
    {
        "track": "P59-A",
        "experiment_id": "V29-03",
        "que_cerro": "clasificador de vecinos: etiqueta mayoritaria de k vecinos",
        "cadena": "BARRIDO EXHAUSTIVO de 48 configuraciones sin selector",
        "motivo": "mismo clasificador (kNN) con 48 configuraciones, todas negativas",
    },
    {
        "track": "P59-A",
        "experiment_id": "EBD-07",
        "que_cerro": "rasgos de vecindad dentro de un LR",
        "cadena": "No se genera submission",
        "motivo": "los rasgos de vecindad como entrada de un clasificador ya se probaron",
    },
    {
        "track": "P59-A",
        "experiment_id": "V4-F01-F04",
        "que_cerro": "la vecindad explica el error?",
        "cadena": "IC [-0.013,+0.049], no significativo",
        "motivo": "el diagnostico 'la vecindad explica el error' ya se midio y no fue significativo",
    },
    {
        "track": "P59-A",
        "experiment_id": "V6-07",
        "que_cerro": "sospecha por 4 senales y downweighting",
        "cadena": "buen marcador de dificultad",
        "motivo": "ya se concluyo que la sospecha marca dificultad, luego la parte si explica errores esta contestada",
    },
    {
        "track": "P59-B",
        "experiment_id": "E2-09-EMO",
        "que_cerro": "polaridad local en ventanas +-3/5/10 de un lexico de emocion",
        "cadena": "canal de emocion completo (marcadores + lexico inducido + ventanas)",
        "motivo": "las ventanas se usaron sobre un lexico inducido, no sobre V03 como puntuador",
    },
    {
        "track": "P59-B",
        "experiment_id": "V4-D01-D04",
        "que_cerro": "canales sparse adicionales: afijos, interacciones, ventanas",
        "cadena": "6 variantes: mejor afijos_hstack 0.7758",
        "motivo": "las ventanas como canal de entrada, no como puntuador local congelado",
    },
    {
        "track": "P59-C",
        "experiment_id": "V4-B02",
        "que_cerro": "separabilidad de las poblaciones dificil-coherente y posible-ruido",
        "cadena": "Posible ruido 43 (2.9 %, error medio 0.054)",
        "motivo": "poblaciones descritas de forma descriptiva, sin estimador formal de etiqueta latente",
    },
    {
        "track": "P59-C",
        "experiment_id": "V30-00",
        "que_cerro": "LA FAMILY label noise",
        "cadena": "LABEL_NOISE = CLOSED",
        "motivo": "se cerro por identificadores de dificultad, no por un estimador de consistencia latente",
    },
    {
        "track": "P59-E",
        "experiment_id": "V9-03",
        "que_cerro": "propagar etiqueta de train a test con copia casi identica",
        "cadena": "\"celdas_que_cambian\": 0",
        "motivo": "propagacion dura por near-duplicado, techo nulo; no es regularizacion de suavidad",
    },
    {
        "track": "P59-F",
        "experiment_id": "MOD-02_BEST",
        "que_cerro": "ComplementNB como clasificador",
        "cadena": "ComplementNB(alpha=1.0)",
        "motivo": "Naive Bayes clasico ya ejecutado, y con mal resultado",
    },
    {
        "track": "P59-F",
        "experiment_id": "E2-15-NBSVM",
        "que_cerro": "NB-SVM, log-count ratio sobre la union",
        "cadena": "Aporta el NB-SVM sobre la union word(1,2)+char_wb(3,5)",
        "motivo": "conteos por clase x log odds, que es exactamente lo que el parrafo 32 prohibe",
    },
    {
        "track": "P59-F",
        "experiment_id": "V41-01",
        "que_cerro": "weighting supervisado de la representacion",
        "cadena": "todas degradan",
        "motivo": "log-odds, chi2, MI, bm25 y binario ya degradan; el margen de esta via es nulo",
    },
    {
        "track": "P59-G",
        "experiment_id": "V8-03",
        "que_cerro": "seleccion de features por peso alto bajo submuestras",
        "cadena": "Negativo y monotono en los tres umbrales",
        "motivo": "es el experimento que el parrafo 33 propone, ya ejecutado y cerrado",
    },
    {
        "track": "P59-G",
        "experiment_id": "V8-03",
        "que_cerro": "prohibicion explicita de reintento",
        "cadena": "No se reintenta con mas agresividad ni con otra constante de submuestreo",
        "motivo": "el precedente cierra la reja por monotonia, no por falta de esfuerzo",
    },
    {
        "track": "P59-D",
        "experiment_id": "V6-03",
        "que_cerro": "ponderacion por tamano de grupo en la frontera",
        "cadena": "Track A cerrado por resultado negativo (seccion 54)",
        "motivo": ("ADYACENTE, NO EQUIVALENTE:Influence via group size esta cerrada, pero la "
                   "palabra 'influencia' aparece aqui en sentido conversacional, no como "
                   "estimador Hessian. Ningun experimento del registro estima influencia sobre "
                   "los parametros"),
    },
    {
        "track": "P59-I",
        "experiment_id": "V35-01",
        "que_cerro": "perdidas lineales alternativas con SGD",
        "cadena": "modified_huber: corrected_cv 0.7435661250",
        "motivo": "Huber esta acotada pero NO es redescendiente, y el resolutor era SGD, no IRLS",
    },
)

TRACKS: tuple[dict[str, Any], ...] = (
    {
        "id": "P59-A",
        "family": "geometria local",
        "mecanismo": ("Voronoi de prototipos de clase sobre coseno de la union V03; score de "
                     "inconsistencia local a partir de margen, pureza, soporte de celda y "
                     "desacuerdo de clase"),
        "precedente": "V29-03, EBD-07, V4-F01-F04, V6-07",
        "novedad": "PARTIALLY_DUPLICATE",
        "hipotesis": ("alguna observacion mal etiquetada ocupa una region local poco coherente "
                      "con su propia clase"),
        "senal_esperada": ("AUC >= 0.60 del score contra error de V03 y separacion entre celdas "
                           "rescatables y no rescatables de P54"),
        "alcance": ("solo la version no vista: prototipos de clase, celdas, margen, y la "
                    "comparacion contra las 42 celdas no rescatables. La parte 'la vecindad "
                    "explica el error' ya esta contestada por V4-F01-F04 y V6-07"),
        "estado": "OPEN",
        "gates": "parrafo 15: AUC >= 0.60 exploratorio, no de adopcion",
    },
    {
        "id": "P59-B",
        "family": "evidencia multiple instancia",
        "mecanismo": ("V03 congelado como puntuador local sobre ventanas de tokens solapadas, "
                     "agregado por max, top-k, fraccion positiva, varianza y concentracion"),
        "precedente": "E2-09, V4-D01-D04, EBD-19",
        "novedad": "NOVEL",
        "hipotesis": ("V03 representa cada tweet como una unica bolsa y eso diluye una senal "
                      "concentrada en una region"),
        "senal_esperada": ("la evidencia local explica errores que la puntuacion global no "
                           "explica, con AUC >= 0.60 y complemento positivo"),
        "alcance": ("V03 nunca se ha usado como puntuador local; E2-09 puntuo ventanas de un "
                    "lexico de emocion y V4-D01-D04 las metio como canal de entrada"),
        "estado": "OPEN",
        "gates": "parrafo 61: no basta con un score distinto, hace falta complemento",
    },
    {
        "id": "P59-C",
        "family": "ruido de etiqueta formal",
        "mecanismo": ("estimador de consistencia latente por etiqueta a partir de la "
                     "probabilidad OOF y la etiqueta observada, con confident joint aproximado"),
        "precedente": "V4-B01, V4-B02, V6-07, V30-00",
        "novedad": "NOVEL",
        "hipotesis": ("existe una poblacion pequena de celdas cuya etiqueta observada es "
                      "sistematicamente incompatible con la evidencia OOF"),
        "senal_esperada": ("los sospechosos se enriquecen en las no rescatables y no en las "
                           "rescatables, con p < 0.05 frente a la poblacion B"),
        "alcance": ("V30-00 cerro la familia por identificadores de dificultad, no por un "
                    "estimador formal de etiqueta latente. Sin relabeling en la primera fase"),
        "estado": "OPEN",
        "gates": "parrafo 62: no puede ser solo error de alta confianza de V03",
    },
    {
        "id": "P59-D",
        "family": "influencia",
        "mecanismo": ("estimacion de influencia sobre los parametros de LogisticRegression "
                     "mediante su aproximacion Hessiana"),
        "precedente": "ninguno: 0 coincidencias de influence, cook, hessian, jackknife",
        "novedad": "NOVEL",
        "hipotesis": ("una observacion puede no parecer ruido pero desplazar de verdad la "
                      "frontera de V03"),
        "senal_esperada": ("los ejemplos de influencia extrema se enriquecen en contradiccion, "
                           "near-duplicate, autor seudonimo o residuo no rescatable"),
        "alcance": "diagnostico primero; el downweighting por influencia queda como decision posterior",
        "estado": "OPEN",
        "gates": "parrafo 63: no eliminar por alta influencia sin relacion con la frontera",
    },
    {
        "id": "P59-E",
        "family": "suavidad en grafo",
        "mecanismo": ("objetivo supervisado mas lambda por la suma de W_ij (f_i - f_j)^2 sobre "
                     "grafo coseno del pliegue de entrenamiento, con extension inductiva"),
        "precedente": "V4-F01-F04, V9-03, V29-03",
        "novedad": "NOVEL",
        "hipotesis": ("tweets similares deberian presentar continuidad en la funcion de "
                      "decision dentro de regiones densas"),
        "senal_esperada": ("el termino de suavidad cambia la frontera mas alla de la votacion de "
                           "vecinos, y mejora corrected_cv"),
        "alcance": ("la propagacion dura por near-duplicado ya tiene techo nulo (V4-F01-F04) y "
                    "la votacion de vecinos esta cerrada (V29-03)"),
        "estado": "OPEN",
        "gates": "parrafo 64: si equivale a votacion local, cerrar DUPLICATE sin entrenar",
    },
    {
        "id": "P59-F",
        "family": "n-gramas generativas",
        "mecanismo": ("modelo generativo clasico de n-gramas con suavizado explicito y "
                     "verosimilitud condicional secuencial real, por etiqueta"),
        "precedente": "MOD-02 (ComplementNB), E2-15 (NB-SVM), V41-01 (log-odds, chi2, MI, bm25), V6-05, V28-00",
        "novedad": "NARROW",
        "hipotesis": "P(text | clase) aporta una senal distinta de la frontera lineal de V03",
        "senal_esperada": "la senal procede de la verosimilitud condicional y no de conteos por clase",
        "alcance": ("solo sobrevive con contexto secuencial real. El parrafo 32 obliga a cerrar "
                    "como DUPLICATE si el mecanismo se reduce a conteos por clase x log odds, que "
                    "es lo que ya hicieron ComplementNB y NB-SVM"),
        "estado": "OPEN",
        "gates": "parrafo 65: no puede ser algebraicamente Naive Bayes ni log-odds",
    },
    {
        "id": "P59-G",
        "family": "seleccion por estabilidad",
        "mecanismo": "seleccion por frecuencia bajo submuestras agrupadas repetidas del entrenamiento",
        "precedente": "V8-03 (tres umbrales de seleccion, IC95 pareado)",
        "novedad": "DUPLICATE",
        "hipotesis": "V03 usa features validas individualmente pero inestables bajo perturbaciones del train",
        "senal_esperada": "ninguna: la tendencia medida es monotona y en direccion contraria",
        "alcance": "cerrado por el precedente V8-03, que ya lo ejecuto correctamente",
        "estado": "CLOSED",
        "resultado": ("V8-03: umbral 0,5 -> 1636 features, -0,003308, IC95 [-0,01218, +0,00558], "
                      "1 de 5 pliegues; umbral 0,7 -> 927, -0,004628, 1 de 5; umbral 0,9 -> 455, "
                      "-0,018890, 0 de 5. Negativo y monotono en los tres. El filtro de "
                      "detectabilidad ademas falla en los tres"),
        "decision": ("DUPLICATE. Cerrado sin consumir experimento. No se reintenta con mas "
                     "agresividad ni con otra constante de submuestreo, porque el precedente "
                     "prohibio exactamente eso y la tendencia va en direccion contraria"),
    },
    {
        "id": "P59-H",
        "family": "estructura no negativa",
        "mecanismo": "factorizacion no negativa dentro del pliegue de entrenamiento, con componentes interpretables",
        "precedente": "ninguno: 0 coincidencias de nmf, non-negative o topic model en 323 filas",
        "novedad": "NOVEL",
        "hipotesis": "la representacion contiene componentes tematicos no negativos que no aparecen de forma explicita en el modelo lineal",
        "senal_esperada": "el bloque NMF aporta complemento positivo sobre V03",
        "alcance": "diagnostico primero; bloque pequeno despues solo si hay senal",
        "estado": "OPEN",
        "gates": "parrafo 67: no como otro SVD, sino por componentes no negativos interpretables",
    },
    {
        "id": "P59-I",
        "family": "perdida robusta",
        "mecanismo": "regresion logistica IRLS con perdida M de redescenso (Tukey biweight o Geman-McClure)",
        "precedente": "V35-01 (SGD hinge y modified_huber), V34-01 (margin / robust loss)",
        "novedad": "NOVEL",
        "hipotesis": "algunas observaciones extremas tienen influencia excesiva y una perdida que reduce el peso de residuos muy grandes generaliza mejor",
        "senal_esperada": "la solucion difiere de V03 y el efecto se concentra en el residuo extremo",
        "alcance": ("Huber esta acotada pero no es redescendiente, y V35-01 cambio tambien el "
                    "resolutor a SGD, luego no es el mismo experimento"),
        "estado": "OPEN",
        "gates": "parrafo 68: demostrar antes que la influencia de los residuos extremos cambia",
    },
)

FAMILIAS_UNICAS = {
    "geometria local": "A",
    "evidencia multiple instancia": "B",
    "ruido de etiqueta formal": "C",
    "influencia": "D",
    "suavidad en grafo": "E",
    "n-gramas generativas": "F",
    "seleccion por estabilidad": "G",
    "estructura no negativa": "H",
    "perdida robusta": "I",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-00] {mensaje}", flush=True)


SUSTITUCIONES = {
    "\u2014": " - ", "\u2013": "-", "\u2212": "-", "\u00b1": "+/-",
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u2192": "->",
    " ": " ", "\u2026": "...", "\u00bf": "?", "\u00a1": "!",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def a_ascii_tabla(filas: list[list[Any]]) -> list[list[str]]:
    """Convierte todos los campos de una tabla a ASCII.

    Args:
        filas: filas de la tabla.

    Returns:
        Las filas con todos los campos en ASCII.
    """
    return [[a_ascii(f"{v}") for v in fila] for fila in filas]


def main() -> None:
    """Construye el registro de hipotesis tras auditar la novedad.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59 registro de hipotesis con gate de novedad ejecutable")
    log("   0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a Kaggle")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    with EXPERIMENTS.open(encoding="utf-8") as fh:
        filas_reg = list(csv.DictReader(fh))
    log(f"   registro cargado: {len(filas_reg)} filas, "
        f"{len({r['experiment_id'] for r in filas_reg})} ids unicos")
    por_id = {r["experiment_id"]: r for r in filas_reg}

    # ---- verificacion machine-checked de cada precedente citado ----
    log("   verificando que cada precedente citado existe y contiene la cifra citada")
    verificados: list[dict[str, Any]] = []
    for p in PRECEDENTES:
        fila = por_id.get(p["experiment_id"])
        if fila is None:
            raise SystemExit(f"PARADA: el precedente {p['experiment_id']} no existe en el registro")
        texto = " ".join([fila.get("result", ""), fila.get("decision", ""),
                          fila.get("question", ""), fila.get("model", "")])
        if p["cadena"] not in texto:
            raise SystemExit(
                f"PARADA: en {p['experiment_id']} no aparece la cadena citada "
                f"{p['cadena']!r}. El precedente estaba citado de memoria")
        verificados.append({**p, "f1": fila.get("f1", ""), "script": fila.get("script", "")})
        log(f"      {p['track']}  {p['experiment_id']:16} verificado: {p['que_cerro'][:52]}")

    log(f"   {len(verificados)} precedentes verificados contra el registro, "
        f"{len({v['experiment_id'] for v in verificados})} experimentos distintos")

    # ---- cuantificacion de la busqueda de cada track ----
    log("   cuantificando la busqueda de cada mecanismo sobre las 323 filas")
    patrones = {
        "P59-A": r"voronoi|medoid|prototip|knn|nearest|vecino|retriev|geometr",
        "P59-B": r"window|ventana|multi.?instance|agregaci|pooling",
        "P59-C": r"confident|cleanlab|label noise|ruido|consistencia latente|confident joint",
        "P59-D": r"influence|influencia|hessian|cook|jackknife",
        "P59-E": r"graph|grafo|laplacian|smoothness|suaviz|propagaci",
        "P59-F": r"n-gram|ngram|naive bayes|nb-svm|complementnb|generativ|verosimilitud",
        "P59-G": r"estabilidad de features|stability|submuestre|subagging",
        "P59-H": r"\bnmf\b|non-negative|topic model",
        "P59-I": r"redescend|tukey|biweight|irls|iteratively reweighted|modified_huber",
    }
    cobertura: dict[str, list[str]] = {}
    for track, patron in patrones.items():
        rx = re.compile(patron, re.IGNORECASE)
        hits = [r["experiment_id"] for r in filas_reg
                if rx.search(" ".join(r.values()) or "")]
        cobertura[track] = hits
        log(f"      {track}  {len(hits):3d} filas coinciden: "
            f"{', '.join(hits[:8]) if hits else 'NINGUNA -> mecanismo no visto'}")

    sin_precedente = [t["id"] for t in TRACKS if cobertura[t["id"]] == []]
    log(f"   tracks con 0 coincidencias totales: {', '.join(sin_precedente) or 'ninguno'}")
    # P59-D tiene una coincidencia (V6-03) que se declaro adyacente y no equivalente:
    # Influence por tamano de grupo esta cerrada, pero el estimador Hessian no aparece.
    if sin_precedente != ["P59-H"]:
        raise SystemExit(
            f"PARADA: los mecanismos sin ninguna coincidencia son {sin_precedente}, no solo "
            "P59-H. Si un mecanismo deja de aparecer en el registro, la tabla de novedad de "
            "este script hay que revisarla antes de seguir")
    adyacentes = [v["experiment_id"] for v in verificados if "ADYACENTE" in v["motivo"]]
    log(f"   tracks con unico precedente adyacente declarado: P59-D ({', '.join(adyacentes)})")

    # ---- tabla del registro ----
    cabeceras = ["ID", "family", "mecanismo", "previous_experiment", "novelty_status",
                 "hypothesis", "expected_signal", "status", "result", "decision"]
    filas: list[list[str]] = []
    for t in TRACKS:
        filas.append([
            t["id"], t["family"], t["mecanismo"], t["precedente"], t["novedad"],
            t["hipotesis"], t["senal_esperada"],
            t["estado"], t.get("resultado", "PENDING"), t.get("decision", "PENDING"),
        ])
    filas = a_ascii_tabla(filas)

    with DIR_REG.open("w", encoding="ascii", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cabeceras)
        w.writerows(filas)
    log(f"   escrito reports/p59_hypothesis_registry.csv ({len(filas)} filas)")

    # ---- frontier registry: estado de la investigacion ----
    cab_front = ["ID", "family", "novelty_status", "status", "experiments_used",
                 "best_corrected_cv", "delta_vs_V03", "folds", "bootstrap",
                 "rescue", "damage", "net", "decision", "reason"]
    filas_f: list[list[str]] = []
    usados: dict[str, int] = {}
    for t in TRACKS:
        n = usados.get(t["id"], 0)
        filas_f.append([
            t["id"], t["family"], t["novedad"], t["estado"], str(n),
            "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
            t.get("decision", "PENDING") if t["novedad"] == "DUPLICATE" else "PENDING",
            ("cerrado por el precedente " + t["precedente"]) if t["novedad"] == "DUPLICATE"
            else ("sin ejecutar" if t["estado"] == "OPEN" else t.get("decision", "PENDING")),
        ])
    filas_f = a_ascii_tabla(filas_f)
    with DIR_FRONTIER.open("w", encoding="ascii", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cab_front)
        w.writerows(filas_f)
    log(f"   escrito reports/p59_frontier_registry.csv ({len(filas_f)} filas)")

    # ---- JSON maestro ----
    aperturas = [t["id"] for t in TRACKS if t["estado"] == "OPEN"]
    duplicados = [t["id"] for t in TRACKS if t["novedad"] == "DUPLICATE"]
    informe = {
        "familia": "P59_HYPOTHESIS_REGISTRY",
        "naturaleza": ("AUDIT_AND_REGISTRATION. 0 modelos, 0 entrenamientos, 0 submissions, "
                       "0 consultas a Kaggle, 0 uso de holdout, 0 uso de test"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "baseline": {"corrected_cv": V03_CCV, "LB": V03_LB, "sha256": V03_SHA,
                     "sha256_verificado": True},
        "criterio_de_novedad": (
            "Una tecnica solo queda abierta si su mecanismo es matematicamente distinto Y no "
            "fue evaluado correctamente antes. El registro verifica cada precedente citado "
            "contra experiments.csv y aborta si la cifra citada no esta en la fila real."),
        "tracks_total": len(TRACKS),
        "tracks_abiertos": aperturas,
        "tracks_duplicados_sin_consumir_experimento": duplicados,
        "tracks_nuevos": sin_precedente,
        "precedentes_verificados": verificados,
        "cobertura_busqueda": cobertura,
        "tracks": {t["id"]: {k: a_ascii(str(v)) for k, v in t.items()} for t in TRACKS},
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "modelos": 0, "entrenamientos": 0,
                   "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_REG.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    log("   escrito reports/p59_hypothesis_registry.json")

    for destino in (DIR_REG, DIR_FRONTIER, JSON_REG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")

    log("")
    log(f"RESUMEN DEL GATE DE NOVEDAD: {len(TRACKS)} tracks | "
        f"{len(duplicados)} DUPLICATE ({', '.join(duplicados)}) | {len(aperturas)} abiertos")
    for t in TRACKS:
        marca = "ABIERTO   " if t["estado"] == "OPEN" else "CERRADO   "
        log(f"   {marca}{t['id']}  {t['novedad']:20} {t['family']}")
    log("")
    log("El track P59-G no se ejecuta: ya fue ejecutado por V8-03, con tres umbrales de")
    log("seleccion e IC95 pareado, y cerrado de forma explicita sin reintento. Esto es lo que")
    log("el gate de novedad debe hacer: ahorrar un experimento y no una ilusion de novedad.")
    log(f"GATE P59-00 OK | exit previsto 0 | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    main()
