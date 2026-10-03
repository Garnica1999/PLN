"""V13 · P19 — Árboles de conjunciones por función, sin etiquetado POS.

Ultima pista pre-registrada del plan. El gap audit de P0 la describes así: el material del curso
recomienda árboles sobre palabras funcionales, y la vía sintáctica que no exige etiquetado POS es
exactamente esta. P26 (n-gramas de POS) está cerrada desde D-026, pero **clasificar conjunciones por
función no necesita POS**: son listas cerradas, y por tanto entran por la regla R-1 (vocabulario
autorizado en el proyecto) y no por la vía de un lexicon externo.

La hipótesis, tal como está pre-registrada: *la función gramatical de las conjunciones varía con la
edad*, porque cambia la subordinación causal, la concesión y la comparación, y un árbol puede capturar
esas interacciones que un modelo lineal no ve. El modelo lineal ya se ha medido tres veces (P1, P17 y
P20, con 0,3997, 0,1918 y 0,1270), así que el árbol es la parte nueva de la prueba.

Y aplica la puerta de rescate antes de pagar nada caro, que es la lección de P17b convertida en
protocolo: primero el control *block-only* con el árbol y con los dos lineales de referencia, después
la medición del rescate sobre el OOF de V12, y solo si el neto lo justifica una evaluación con
control de permutado y gate.

Por qué este bloque tiene una oportunidad real y los anteriores no la tenían: P0b mostró que las
familias de textometria de V12 están saturadas, y que las de evidencia supervisada y NB-SVM son las
únicas con margen. Las conjunciones no son ni una cosa ni la otra: son un phenomenon gramatical, y su
recuento por función es invariante a la longitud de forma distinta a como lo es la textometría. El
cribado de redundancia contra las 279 variables de V12 dirá si eso es cierto o si son otra
reformulación de la longitud del texto.

Entregables: ``reports/v19_conjunciones.md``, ``reports/v19_conjunciones_cobertura.csv``,
``reports/v19_conjunciones_redundancia.csv``, ``reports/v19_conjunciones_control.csv``,
``features/v19_conjunciones.npy``, ``experiments/v19_p19_conjunciones.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import build_features  # noqa: E402
from v13_p0b_colinealidad import criba_redundancia  # noqa: E402
from v13_p1b_fusion_tokens import PARES, oof_una_cabeza  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src import lexicons as lx  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v19_conjunciones.md"
COBERTURA = cfg.REPORTS / "v19_conjunciones_cobertura.csv"
REDUNDANCIA = cfg.REPORTS / "v19_conjunciones_redundancia.csv"
CONTROL = cfg.REPORTS / "v19_conjunciones_control.csv"
MATRIZ = cfg.FEATURES / "v19_conjunciones.npy"
NOMBRES = cfg.FEATURES / "v19_conjunciones_nombres.json"
RESULTADO = cfg.EXPERIMENTS / "v19_p19_conjunciones.json"

#: Conjunciones clasificadas por función. Listas cerradas escritas en este proyecto (R-1): el
#: corpus es ASCII sin tildes, así que `solo`/`sólo` son la misma cadena y `mas`/`más` también.
CONJUNCIONES = {
    "causal": ["porque", "pues", "dado", "puesto", "visto"],
    "concesiva": ["aunque", "aun", "siquiera", "pese"],
    "condicional": ["si", "caso", "mientras", "siempre", "salvo"],
    "temporal": ["cuando", "entonces", "antes", "despues", "apenas", "mientras"],
    "comparativa": ["como", "cual", "cuanto", "tanto", "igual", "similar"],
    "consecutiva": ["luego", "asi", "modo", "consecuencia", "entonces"],
    "adversativa": ["pero", "ahora", "cambio", "embargo", "consigo"],
    "sustitutiva": ["que", "quien", "cuyo", "cual"],
    "disyuntiva": ["o", "ora", "ahora", "ya"],
}
SUELO_AZAR = 1.0 / len(cfg.CLASSES)
MINIMO_DOCUMENTOS = 200
RESCATE_MINIMO = 1000
PORCENTAJE_ERROR = 0.015
#: Profundidades del arbol probadas. Un arbol poco profundo es el que puede capturar interacciones
#: de conteos sin sobreajustar 169.336 filas con veinte variables.
PROFUNDIDADES = (4, 6, 8)


def cuenta_categoria(documentos: list[list[str]], formas: list[str]) -> np.ndarray:
    """Apariciones de las formas de una categoría en cada documento."""
    conjunto = set(formas)
    return np.array([sum(1 for token in tokens if token in conjunto) for tokens in documentos],
                    dtype=np.float32)


def nombres_de_features() -> list[str]:
    """Nombres de las características derivadas de los recuentos por función."""
    nombres: list[str] = []
    for funcion in CONJUNCIONES:
        nombres += [f"{funcion}_conteo", f"{funcion}_ratio", f"{funcion}_presencia"]
    nombres += ["n_conjunciones", "n_conjunciones_por_token", "diversidad_conjunciones",
                "proporcion_adversativa", "proporcion_concesiva", "proporcion_causal",
                "proporcion_temporal", "indice_subordinacion", "rango_perfil_conjunciones"]
    return nombres


def main() -> int:
    """Construye el bloque de conjunciones, lo criba, lo mide y decide con la puerta de rescate."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    documentos = [lx.tokeniza(str(texto)) for texto in train["text"]]
    longitudes = np.clip(np.array([len(tokens) for tokens in documentos], dtype=np.float32),
                         1.0, None)
    conteos = {funcion: cuenta_categoria(documentos, formas)
               for funcion, formas in CONJUNCIONES.items()}

    cobertura = []
    for funcion, formas in CONJUNCIONES.items():
        por_forma = {forma: sum(1 for tokens in documentos if forma in tokens) for forma in formas}
        cobertura.append({
            "funcion": funcion, "formas": " ".join(formas),
            "documentos_con_la_familia": int(sum(1 for tokens in documentos
                                                  if any(forma in tokens for forma in formas))),
            "apariciones_totales": int(conteos[funcion].sum()),
            "formas_sobre_el_minimo": sum(1 for presentes in por_forma.values()
                                          if presentes >= MINIMO_DOCUMENTOS),
            "formas_totales": len(formas),
            "detalle": " ".join(f"{forma}:{presentes:,}" for forma, presentes in por_forma.items()),
        })
    pd.DataFrame(cobertura).to_csv(COBERTURA, index=False)
    for fila in cobertura:
        print(f"[cobertura] {fila['funcion']:14s} {fila['documentos_con_la_familia']:7,} documentos | "
              f"{fila['apariciones_totales']:8,} apariciones | formas sobre el minimo "
              f"{fila['formas_sobre_el_minimo']}/{fila['formas_totales']}", flush=True)

    nombres = nombres_de_features()
    matriz = np.zeros((len(documentos), len(nombres)), dtype=np.float32)
    for indice, funcion in enumerate(CONJUNCIONES):
        matriz[:, 3 * indice] = conteos[funcion]
        matriz[:, 3 * indice + 1] = conteos[funcion] / longitudes * 100.0
        matriz[:, 3 * indice + 2] = (conteos[funcion] > 0).astype(np.float32)
    base = 3 * len(CONJUNCIONES)
    total = np.sum([conteos[funcion] for funcion in CONJUNCIONES], axis=0)
    matriz[:, base] = total
    matriz[:, base + 1] = total / longitudes * 100.0
    matriz[:, base + 2] = sum((conteos[funcion] > 0).astype(np.float32)
                              for funcion in CONJUNCIONES)
    matriz[:, base + 3] = conteos["adversativa"] / np.maximum(total, 1.0)
    matriz[:, base + 4] = conteos["concesiva"] / np.maximum(total, 1.0)
    matriz[:, base + 5] = conteos["causal"] / np.maximum(total, 1.0)
    matriz[:, base + 6] = conteos["temporal"] / np.maximum(total, 1.0)
    subordinacion = (conteos["condicional"] + conteos["temporal"] + conteos["causal"]
                     + conteos["concesiva"])
    matriz[:, base + 7] = subordinacion / np.maximum(total, 1.0)
    matriz[:, base + 8] = np.std([conteos[funcion] for funcion in CONJUNCIONES], axis=0)
    np.save(MATRIZ, matriz)
    NOMBRES.write_text(json.dumps(nombres, ensure_ascii=False), encoding="utf-8")
    print(f"[bloque] {matriz.shape} | columnas con varianza "
          f"{int((matriz.std(axis=0) > 0).sum())} | conjunciones por documento "
          f"{float(total.mean()):.2f}", flush=True)

    base_v12, _, _, _, _ = build_features()
    base_v12 = np.concatenate([base_v12, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")],
                              axis=1)
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=20000, replace=False)
    criba = criba_redundancia(matriz, nombres, base_v12,
                              [f"v12_{i:03d}" for i in range(base_v12.shape[1])], filas)
    criba.to_csv(REDUNDANCIA, index=False)
    print(f"[cribado] {int(criba['redundante'].sum())} de {len(criba)} con |rho| >= 0,80 contra V12 | "
          f"maxima {criba['correlacion_maxima_v12'].max():.4f} | mediana "
          f"{criba['correlacion_maxima_v12'].median():.4f}", flush=True)

    # ---------------------------------------------------------------- control block-only
    filas_control = []
    modelos = ([(f"arbol_profundidad_{profundidad}",
                 lambda p=profundidad: DecisionTreeClassifier(
                     max_depth=p, min_samples_leaf=200, criterion="gini",
                     random_state=cfg.SEED))
                for profundidad in PROFUNDIDADES]
               + [("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000,
                                                          random_state=cfg.SEED)),
                  ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))])
    for nombre, constructor in modelos:
        por_fold = []
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            estimador = constructor()
            estimador.fit(matriz[entrenamiento], clase[entrenamiento])
            por_fold.append(float(f1_score(clase[validacion],
                                           estimador.predict(matriz[validacion]),
                                           average="macro",
                                           labels=list(range(len(cfg.CLASSES))))))
        filas_control.append({"modelo": nombre, "macro_f1": round(float(np.mean(por_fold)), 4),
                              "std": round(float(np.std(por_fold)), 4),
                              "per_fold": [round(v, 4) for v in por_fold],
                              "suelo_azar": round(SUELO_AZAR, 4),
                              "supera_el_azar": bool(float(np.mean(por_fold)) > SUELO_AZAR)})
        print(f"[control] {nombre}: {np.mean(por_fold):.4f} ± {np.std(por_fold):.4f} "
              f"(suelo {SUELO_AZAR:.4f})", flush=True)
    pd.DataFrame(filas_control).to_csv(CONTROL, index=False)

    # ---------------------------------------------------------------- puerta de rescate
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    acierto_v12 = np.argmax(referencia, axis=1) == clase
    mejor = max(filas_control, key=lambda fila: fila["macro_f1"])
    if mejor["supera_el_azar"]:
        if mejor["modelo"].startswith("arbol"):
            constructor = modelos[[m[0] for m in modelos].index(mejor["modelo"])][1]
        else:
            constructor = dict((m[0], m[1]) for m in modelos)[mejor["modelo"]]
        probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            estimador = constructor().fit(matriz[entrenamiento], clase[entrenamiento])
            probabilidades[validacion] = estimador.predict_proba(matriz[validacion])
        acierto_bloque = np.argmax(probabilidades, axis=1) == clase
        rescate = int((~acierto_v12 & acierto_bloque).sum())
        dano = int((acierto_v12 & ~acierto_bloque).sum())
        por_frontera = {}
        for (clase_a, clase_b), etiqueta in PARES.items():
            mascara = np.isin(clase, (cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)))
            res = int((~acierto_v12 & acierto_bloque & mascara).sum())
            dan = int((acierto_v12 & ~acierto_bloque & mascara).sum())
            por_frontera[etiqueta] = {"rescate": res, "dano": dan, "neto": res - dan}
    else:
        rescate = dano = 0
        por_frontera = {etiqueta: {"rescate": 0, "dano": 0, "neto": 0} for etiqueta in PARES.values()}
    errores = int((~acierto_v12).sum())
    umbral = max(RESCATE_MINIMO, int(PORCENTAJE_ERROR * errores))
    pasa = ((rescate - dano) >= umbral
            and all(m["neto"] > 0 for m in por_frontera.values()))
    print(f"[puerta] mejor control {mejor['modelo']} {mejor['macro_f1']} | rescate {rescate:,} | "
          f"dano {dano:,} | neto {rescate - dano:+,} | umbral {umbral:,} | pasa {pasa}", flush=True)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V13 · P19 — Árboles de conjunciones por función", "",
        "## 1. Qué se construye y por qué", "",
        "El material del curso recomienda árboles sobre palabras funcionales. La vía sintáctica que "
        "**no** exige etiquetado POS (cerrado desde D-026) es clasificar conjunciones por función, y "
        "eso son listas cerradas escritas en el proyecto, que es lo que permite R-1.", "",
        "La hipótesis pre-registrada es que la función gramatical de las conjunciones varía con la "
        "edad. El modelo lineal ya se ha medido tres veces (P1 0,3997, P17 0,1918, P20 0,1270), así "
        "que **el árbol es la parte nueva de la prueba**: si hay algo que un lineal no ve, es una "
        "interacción entre recuentos por función.", "",
        "## 2. Cobertura por familia", "",
        "| Familia | Documentos | Apariciones | Formas sobre el mínimo |", "|---|---|---|---|",
    ]
    for fila in cobertura:
        lineas.append(f"| {fila['funcion']} | {fila['documentos_con_la_familia']:,} | "
                      f"{fila['apariciones_totales']:,} | {fila['formas_sobre_el_minimo']}/"
                      f"{fila['formas_totales']} |")
    lineas += ["", "## 3. Características", "",
               f"{len(nombres)} columnas: conteo, ratio por cien tokens y presencia para cada familia, "
               f"más nueve derivadas, entre ellas la **proporción de cada función** (adversativa, "
               f"concesiva, causal, temporal), el **índice de subordinación** y el **rango del perfil "
               f"de conjunciones** del documento.", "",
              "## 4. Cribado de redundancia contra las 279 variables de V12", "",
              f"- Variables redundantes: **{int(criba['redundante'].sum())}** de {len(criba)}",
              f"- Correlación máxima: **{criba['correlacion_maxima_v12'].max():.4f}** · mediana "
              f"{criba['correlacion_maxima_v12'].median():.4f}", "",
              "## 5. Control block-only", "",
              "| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |", "|---|---|---|---|---|"]
    for fila in filas_control:
        lineas.append(f"| {fila['modelo']} | {fila['macro_f1']} | {fila['std']} | "
                      f"{fila['suelo_azar']} | {'si' if fila['supera_el_azar'] else 'no'} |")
    lineas += ["", "## 6. Puerta de rescate", "",
               f"- Rescate: **{rescate:,}** · daño: **{dano:,}** · **neto {rescate - dano:+,}**",
               f"- Umbral: {umbral:,} filas", ""]
    for etiqueta, medidas in por_frontera.items():
        lineas.append(f"- {etiqueta}: rescate {medidas['rescate']:,} · daño {medidas['dano']:,} · "
                      f"neto {medidas['neto']:+,}")
    lineas += ["", "## 7. Decisión", ""]
    if not mejor["supera_el_azar"]:
        lineas += ["**Ningún modelo llega al suelo de azar.** El árbol no encuentra nada donde los "
                   "lineales tampoco encuentran, que es lo coherente con P20: en un corpus de 140 "
                   "caracteres, las conjunciones no llegan a la frecuencia necesaria para discriminar. "
                   "La pista se cierra sin gastar la evaluación del meta-modelo."]
    elif pasa:
        lineas += ["**La puerta se supera**: procede la fusión con V12, control de permutado, gate §24 "
                   "y doble métrica §36."]
    else:
        lineas += ["**La puerta no se supera**: el bloque no justifica pagar la evaluación del "
                   "meta-modelo. La pista se cierra con la evidencia del rescate, como P17 y P20."]
    lineas.append("")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P19 arboles de conjunciones", "fecha": "2026-09-26",
        "familias": CONJUNCIONES, "n_caracteristicas": len(nombres),
        "profundidades_probadas": list(PROFUNDIDADES),
        "cobertura": cobertura, "suelo_azar": round(SUELO_AZAR, 4),
        "redundantes_contra_v12": int(criba["redundante"].sum()),
        "correlacion_maxima_v12": float(criba["correlacion_maxima_v12"].max()),
        "control": filas_control,
        "mejor_control": mejor,
        "prechequeo": {"errores_v12": errores, "rescate": rescate, "dano": dano,
                       "neto": rescate - dano, "umbral": umbral, "pasa": pasa,
                       "por_frontera": por_frontera},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefactos: {INFORME.name}, {COBERTURA.name}, {REDUNDANCIA.name}, {CONTROL.name}, "
          f"{MATRIZ.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
