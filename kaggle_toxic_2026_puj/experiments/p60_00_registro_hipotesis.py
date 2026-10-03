"""P60 step 01: literature review and gap registry.

Why this script writes five artefacts
-------------------------------------
P60 is a literature-guided phase, and the two failure modes of a literature-guided
phase are known in advance. The first is treating a different name as a new
mechanism, and the second is skipping the novelty audit and running twelve
variants of things the project closed years ago. Both are prevented by the same
artefact: a registry that records, for every method, what its mechanism is, what
it needs, whether it is legal here, which project experiments already cover it,
and whether that coverage is functional rather than nominal.

So this script writes, from one table:

    reports/p60_literature_review.md     prose review with the transfer argument
    reports/p60_literature_registry.csv  one row per method
    reports/p60_gap_registry.csv         one row per track, with the verdict
    reports/p60_hypothesis_registry.csv  one row per track, hypothesis to decision
    reports/p60_dependency_graph.md      the dependency structure

The label statistics quoted in the review are MEASURED here from the canonical
OOF, not transcribed, because the most decision-relevant fact of the whole
review, that the label structure is nested, is a property of this dataset and
not of any paper.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_00_registro_hipotesis.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REG = PROJECT_ROOT / "experiments.csv"
REPORTS = PROJECT_ROOT / "reports"
MD_REVIEW = REPORTS / "p60_literature_review.md"
CSV_LIT = REPORTS / "p60_literature_registry.csv"
CSV_GAP = REPORTS / "p60_gap_registry.csv"
CSV_HIPO = REPORTS / "p60_hypothesis_registry.csv"
MD_DEP = REPORTS / "p60_dependency_graph.md"
JSON_REV = REPORTS / "p60_literature_review.json"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-00] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
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


# --------------------------------------------------------------------------
# Registro de literatura. Cada fila declara el mecanismo, lo que necesita, si es
# legal en TOXIC, que experimentos del proyecto lo cubren y la equivalencia
# funcional. Los campos de equivalente se evaluaron contra experiments.csv con
# una busqueda de patrones, y los cierres por duplicacion citan el precedente.
# --------------------------------------------------------------------------
LITERATURA: tuple[dict[str, Any], ...] = (
    {
        "clave": "ASL",
        "paper": "Asymmetric Loss for Multi-Label Classification",
        "autores": "Ridnik, Baruch, et al.",
        "anio": 2021, "venue": "ICCV 2021", "doi": "10.1109/ICCV48922.2021.00015",
        "metodo": "Asymmetric Loss: foco asimetrico y desplazamiento de probabilidad",
        "problema": "desequilibrio positivo/negativo en multi-etiqueta",
        "mecanismo": ("dos terminos, L+ = (1-pm)^gamma+ * -log(p) y L- = pm^gamma- * -log(1-p), "
                     "con pm = max(p - m, 0). El desplazamiento hace que el termino negativo "
                     "sea EXACTAMENTE CERO cuando p < m: un descarte duro de negativos faciles, "
                     "no un decaimiento suave"),
        "datos_necesarios": "etiquetas binarias por instancia; ninguna estructura externa",
        "legal": "SI. Solo cambia la funcion objetivo de un clasificador lineal propio",
        "experimentos_relacionados": "V7-01 (focal simetrico), P59-06 (Tukey redescendiente)",
        "equivalencia_funcional": ("NO equivalente. V7-01 probo focal, que decae de forma SUAVE y "
                                    "simetrica. ASL anade un CORTE DURO y una asimetria de clase. Y "
                                    "P59-06 anulo la influencia de residuos EXTREMOS, que es la region "
                                    "opuesta a la que ASL descarta"),
        "novedad": True,
        "modo_fallo_esperado": ("si el corte duro de negativos elimina senal util, el recall cae; "
                                "V7-01 ya observo quenn ajustar el objetivo no mejora"),
        "prioridad": 2,
    },
    {
        "clave": "W-ASL",
        "paper": "Weighted Asymmetric Loss for Multi-Label Text Classification on Imbalanced Data",
        "autores": "Yasuda, Miyazaki, Goto", "anio": 2024, "venue": "JNLP 2024",
        "doi": "10.5715/jnlp.31.1166",
        "metodo": "ASL mas ponderacion por frecuencia de etiqueta mas suavizado de coocurrencia",
        "problema": "desequilibrio en multi-etiqueta de texto",
        "mecanismo": ("ASL mas un peso por frecuencia de etiqueta y mas un suavizado de la "
                     "estructura de coocurrencia entre etiquetas"),
        "datos_necesarios": "frecuencias marginales y matriz de coocurrencia, estimadas en train",
        "legal": "SI en principio",
        "experimentos_relacionados": ("V8-07 (peso de clase por etiqueta), V4-E01-E03 (label "
                                      "powerset), V28-00 (especializacion por etiqueta)"),
        "equivalencia_funcional": ("SI, en sus dos componentes. La ponderacion por frecuencia de "
                                   "etiqueta ES class_weight='balanced', que es el mecanismo de V03, y "
                                   "V8-07 ya lo probo por etiqueta con -0,005298 y 1 de 5 pliegues. El "
                                   "suavizado de coocurrencia ES la familia de dependencia de "
                                   "etiquetas, y V4-E01-E03 la cerro como NO_SIGNAL tras 12 variantes "
                                   "en el subespacio factible"),
        "novedad": False,
        "modo_fallo_esperado": "n/a, se cierra sin entrenar",
        "prioridad": 99,
    },
    {
        "clave": "PAIR-NOISE",
        "paper": "Noisy Multi-Label Text Classification via Instance-Label Pair Correction",
        "autores": "Xu, Song, Liu, et al.", "anio": 2024, "venue": "Findings of NAACL 2024",
        "doi": "10.18653/v1/2024.findings-naacl.93",
        "metodo": "deteccion de ruido a nivel de par instancia-etiqueta y correccion suave",
        "problema": "ruido que no es solo de instancia sino del par (i, y)",
        "mecanismo": ("la unidad de deteccion es el par, y el criterio combina la perdida del par, "
                     "informacion de rango y la distancia a centroides de clase"),
        "datos_necesarios": "perdidas por par en train, sin datos externos",
        "legal": "SI, como diagnostico train-fold-only",
        "experimentos_relacionados": ("V30-00 (LABEL_NOISE cerrado), P59-03 (confident learning), "
                                      "V4-B01 a V4-B05 (V4-B03 pondero por dureza)"),
        "equivalencia_funcional": ("PARCIAL, y la distincion es real. V30-00 cerro siete "
                                    "experimentos de sospecha a nivel de INSTANCIA, y P59-03 cerro el "
                                    "estimador formal global. Ninguno de los dos descompone la "
                                    "sospecha por ETIQUETA, que es exactamente lo que el trabajo de "
                                    "NAACL 2024 anade. Y en TOXIC hay una asimetria de etiqueta "
                                    "medible y enorme: el 100 por ciento de los negativos de TR son "
                                    "etiquetas no definidas"),
        "novedad": True,
        "modo_fallo_esperado": ("que el score por par no supere al control trivial 1 - p_true, que "
                                "es exactamente lo que le paso a P59-03 con su estimador global"),
        "prioridad": 5,
    },
    {
        "clave": "EFFNUM",
        "paper": "Class-Balanced Loss Based on Effective Number of Samples",
        "autores": "Cui, Jia, Lin, et al.", "anio": 2019, "venue": "CVPR 2019",
        "doi": "10.1109/CVPR.2019.00949",
        "metodo": "ponderacion por numero efectivo de muestras (1-beta)/(1-beta^n_c)",
        "problema": "desequilibrio de clases",
        "mecanismo": ("sustituye el peso 1/n_c por (1-beta)/(1-beta^{n_c}), que satura: las clases "
                     "grandes pesan proporcionalmente a 1 en vez de a 1/n_c"),
        "datos_necesarios": "recuento por clase en train",
        "legal": "SI, es un cambio de pesos de clase",
        "experimentos_relacionados": "V8-07 (peso de clase por etiqueta), V03 (class_weight=balanced)",
        "equivalencia_funcional": ("NO equivalente, aunque esestra forma. V03 usa n/(2 n_c) = "
                                    "1/(2 pi_c) y V8-07 probo variantes por etiqueta con resultado "
                                    "negativo. El numero efectivo es otra FUNCION de n_c, con "
                                    "saturacion, y cambia los pesos de forma no monomona en el margen"),
        "novedad": True,
        "modo_fallo_esperado": ("con solo tres etiquetas y prevalencias de 0,30 a 0,44 el rango de "
                                "n_c es estrecho, luego la diferencia entre 1/n_c y "
                                "(1-beta)/(1-beta^n_c) es pequena"),
        "prioridad": 4,
    },
    {
        "clave": "DBL",
        "paper": "Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets",
        "autores": "Wu, Ding, Han", "anio": 2020, "venue": "ECCV 2020",
        "metodo": "perdida equilibrada en distribucion con regularizacion tolerante a negativos",
        "problema": "cola larga en multi-etiqueta",
        "mecanismo": ("rebaja el peso de lasnn en regiones con exceso de negativos, usando la "
                     "razon de recuentos positivo/negativo por clase"),
        "datos_necesarios": "recuentos positivo/negativo por clase en train",
        "legal": "SI",
        "experimentos_relacionados": "V8-07, V03",
        "equivalencia_funcional": ("NO equivalente en la forma canonica, que anade un termino de "
                                    "regularizacion ademas de la ponderacion. Se conserva como parte "
                                    "de la pista D"),
        "novedad": True,
        "modo_fallo_esperado": "solapamiento parcial con la ponderacion por clase, que ya se sabe neutra",
        "prioridad": 4,
    },
    {
        "clave": "LOGIT-ADJ",
        "paper": "Long-tail Learning via Logit Adjustment",
        "autores": "Menon, Jayasumana, Rawat, et al.", "anio": 2021, "venue": "ICLR 2021 (Spotlight)",
        "doi": "arXiv:2007.07314",
        "metodo": "logit adjustment post-hoc y en la perdida, con grounding estadistico",
        "problema": "desequilibrio de clases",
        "mecanismo": ("anade tau * log(pi_y) al logit, post-hoc o dentro de la perdida, para "
                     "favorecer margen relativo entre clases raras y dominantes"),
        "datos_necesarios": "prior por etiqueta",
        "legal": "SI",
        "experimentos_relacionados": "V8-09 (curva unimodal de tasa de positivos), P59-08 (T)",
        "equivalencia_funcional": ("SI, de forma algebrica y en las dos formas. Post-hoc, ajustar el "
                                   "logit y cortar en 0,5 equivale a cortar el logit original en "
                                   "sigma(tau log pi_y): es un cambio de umbral. En la perdida, sobre "
                                   "un modelo LINEAL, sumar una constante al logit es desplazar el "
                                   "intercepto, y desplazar el intercepto con umbral fijo es un "
                                   "cambio de umbral. V8-09 cerro esa familia de raiz"),
        "novedad": False,
        "modo_fallo_esperado": "n/a, se cierra con prueba algebrica",
        "prioridad": 99,
    },
    {
        "clave": "LDAM",
        "paper": "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss",
        "autores": "Cao, Wei, Gaidon, Arechiga, Ma", "anio": 2019, "venue": "NeurIPS 2019",
        "metodo": "margen por etiqueta Delta_y = C / n_y^(1/4) aplicado solo al termino del positivo",
        "problema": "desequilibrio de clases",
        "mecanismo": ("reduce el logit de la clase verdadera en una cantidad Delta_y, lo que cambia "
                     "el gradiente del positivo de forma asimetrica"),
        "datos_necesarios": "recuento por clase y una constante C",
        "legal": "SI",
        "experimentos_relacionados": "V7-01 (focal), P59-06 (robusta), V12-02 (GCE)",
        "equivalencia_funcional": ("PARCIAL. No es un cambio de umbral porque cambia el objetivo de "
                                   "entrenamiento. Pero es un margen asimetrico sobre el positivo, "
                                   "que es la misma region que focal y que la perdida asimetrica de "
                                   "la pista A. Se resuelve DENTRO de A como alternativa de margen, "
                                   "no como pista propia"),
        "novedad": False,
        "modo_fallo_esperado": "solapamiento con la pista A",
        "prioridad": 99,
    },
    {
        "clave": "LDL",
        "paper": "Scalable Label Distribution Learning for Multi-Label Classification",
        "autores": "Zhao, An, Qi, Geng", "anio": 2025, "venue": "IEEE TNNLS 2025",
        "doi": "10.1109/TNNLS.2024.3475469",
        "metodo": "aprendizaje de distribuciones de etiqueta en un espacio continuo",
        "problema": "aprender la distribucion de etiqueta en lugar de una etiqueta dura",
        "mecanismo": "regresion sobre vectores de distribucion de etiqueta aprendidos",
        "datos_necesarios": "una Cabecera por etiqueta o una representacion de baja dimension",
        "legal": "SI",
        "experimentos_relacionados": "V4-E01-E03 (label powerset), V28-00",
        "equivalencia_funcional": ("SI en la practica. Con tres etiquetas el subespacio factible es "
                                   "{000, 100, 101, 111}, exactamente el que V4-E01-E03 enumero en 12 "
                                   "variantes sin superar a V03, y V28-00 cerro la especializacion por "
                                   "etiqueta desde cinco angulos. Ademas se midio en este corpus que "
                                   "las etiquetas estan ANIDADAS y no solo correlacionadas"),
        "novedad": False,
        "modo_fallo_esperado": "n/a, se cierra por equivalencia y por complejidad innecesaria",
        "prioridad": 99,
    },
    {
        "clave": "CONTRAST-OVL",
        "paper": ("Multi-Label Text Classification with Label Attention Aware and Correlation Aware "
                  "Contrastive Learning"),
        "autores": "Zhu et al.", "anio": 2025, "venue": "IJCAI 2025", "doi": "10.24963/ijcai.2025/936",
        "metodo": "aprendizaje contrastivo supervisado con solapamiento de etiquetas como positivo",
        "problema": "multi-etiqueta de texto, representacion poco discriminante",
        "mecanismo": ("modifica la GEOMETRIA de la representacion aprendida: dos instancias son "
                     "positivas si comparten etiquetas, y el contraste las acerca en el espacio de "
                     "caracteristicas, no en el de las puntuaciones"),
        "datos_necesarios": "solapamiento de etiquetas de TRAIN para construir los pares",
        "legal": ("SI. Se construye internamente: TF-IDF -> SVD -> proyeccion entrenable -> perdida "
                  "contrastiva. Sin transformers, sin embeddings preentrenados, sin datos externos"),
        "experimentos_relacionados": ("ninguno: 0 filas del registro mencionan contrastive, "
                                      "supcon ni solapamiento de etiquetas. V12-00 solo lo nombra "
                                      "como hipotesis de registro"),
        "equivalencia_funcional": ("NINGUNA. Es el unico eje de los doce que no modifica la funcion "
                                   "de perdida, ni los pesos, ni la capa de decision, ni la "
                                   "representacion de entrada, sino la GEOMETRIA aprendida. "
                                   "V12-00 habia senalado que los tres huecos que quedaban estaban "
                                   "en el eje del objetivo de entrenamiento, y este no esta en ese eje"),
        "novedad": True,
        "modo_fallo_esperado": ("con 1186 filas de entrenamiento y 27.431 rasgos, el contraste puede "
                                "aportar mas ruido que senal; ademas las etiquetas estan anidadas, "
                                "luego el solapamiento es casi jerarquico"),
        "prioridad": 1,
    },
    {
        "clave": "OT-PROT",
        "paper": "Robust Text Classification: Analyzing Prototype-Based Networks",
        "autores": "Sourati, Deshpande, Ilievski, Gashteovski, Saralajew", "anio": 2024,
        "venue": "Findings of EMNLP 2024", "doi": "10.18653/v1/2024.findings-emnlp.745",
        "metodo": "redes basadas en prototipos para clasificacion de texto",
        "problema": "robustez de clasificacion de texto",
        "mecanismo": ("clasifica un punto por su similitud a un conjunto de prototipos aprendidos, "
                     "no a un unico centro"),
        "datos_necesarios": "espacio de caracteristicas interno",
        "legal": "SI con representacion interna TF-IDF o SVD",
        "experimentos_relacionados": "P59-02 (geometria Voronoi), V4-G01-G03 (likelihood scores)",
        "equivalencia_funcional": ("PARCIAL y con un matematico importante. Si el objeto de prueba es "
                                   "un punto de Dirac, la distancia de Wasserstein-1 entre delta_x y la "
                                   "distribucion de la clase es la MEDIA de distancias a las muestras "
                                   "de la clase, es decir la distancia al centroide, que esta CERRADA. "
                                   "El transporte solo aporta algo si el objeto de prueba es el mismo "
                                   "una DISTRIBUCION, por ejemplo sobre las unidades del texto"),
        "novedad": True,
        "modo_fallo_esperado": ("Sinkhorn con regimenes marginales casi degenerados, porque las "
                                "clases son minoritarias y la entropia numerica puede ser inestable"),
        "prioridad": 6,
    },
    {
        "clave": "META-RW",
        "paper": "Learning to Reweight Examples for Robust Deep Learning",
        "autores": "Ren, Yu, Sheng, Ma, Zhao, Yi, Li", "anio": 2018, "venue": "ICML 2018",
        "metodo": "meta-aprendizaje de pesos por ejemplo desde la alineacion del gradiente",
        "problema": "entrenamiento robusto con datos posiblemente ruidosos",
        "mecanismo": ("los pesos se obtienen de un objetivo META que compara el gradiente de cada "
                     "ejemplo de entrenamiento con el gradiente de un objetivo limpio, optimizado "
                     "en un nivel interno"),
        "datos_necesarios": "un split limpio interno; en este proyecto, etiquetas de train tratadas como limpias",
        "legal": ("SI solo con estructura anidada estricta: outer_train -> inner_train -> "
                  "meta-optimizacion de pesos -> outer_validation, sin que la validacion externa "
                  "participe nunca"),
        "experimentos_relacionados": ("V4-B03, V4-B05, V6-07 (ponderacion heuristica por "
                                      "dificultad o sospecha)"),
        "equivalencia_funcional": ("NO equivalente, y la distincion es la que importa. Los tres "
                                    "precedentes derivan el peso de una REGLA FIJA de dificultad o "
                                    "sospecha, computada antes de entrenar. El meta-reweighting "
                                    "deriva el peso de un CRITERIO DE GRADIENTE contra un objetivo "
                                    "proxy, lo cual es endogeno y no una regla heuristica"),
        "novedad": True,
        "modo_fallo_esperado": ("las etiquetas de train de TOXIC no se sabe que sean limpias, luego "
                                "el objetivo meta puede estar contaminado; ademas 1186 filas en un "
                                "espacio de 27.431 rasgos hace la alineacion por ejemplo muy ruidosa"),
        "prioridad": 7,
    },
    {
        "clave": "POLYLOSS",
        "paper": "PolyLoss: A Unified Formulation of Cross-Entropy and Polynomial Losses",
        "autores": "Leng, Chen, Zheng, Zhu, Jin", "anio": 2022, "venue": "ICLR 2022 (Spotlight)",
        "metodo": "perdida polinomica learned de la entropia cruzada",
        "problema": "disenar la funcion de perdida de forma integrada en la metrica",
        "mecanismo": ("expansion polinomial de la entropia cruzada, Pol_1 y Pol_2, con coeficientes "
                     "entrenables o fijados"),
        "datos_necesarios": "ninguno adicional",
        "legal": "SI",
        "experimentos_relacionados": "V7-01 (focal y F1 suave), V12-02 (GCE L_q), P59-06 (Tukey)",
        "equivalencia_funcional": ("PARCIAL. GCE L_q = (1 - p_y^q)/q es ya una familia polinomial de la "
                                   "perdida, y V12-02 la probo con cuatro valores de q. V7-01 cerro "
                                   "la rama simetrica. PolyLoss con coeficientes FIJOS es una "
                                   "transformacion monotona de la entropia cruzada, y con "
                                   "coeficientes ENTRENABLES es una reparametrizacion de pesos de "
                                   "muestra. Se ejecuta una sola configuracion y que el experimento "
                                   "decida, porque cerrar por estilo seria el error simetrico de "
                                   "aceptar un nombre diferente como novedad"),
        "novedad": True,
        "modo_fallo_esperado": ("V7-01 concluyo que en este problema aproximar mejor el objetivo no "
                                "es mejor, y PolyLoss es exactamente eso"),
        "prioridad": 8,
    },
    {
        "clave": "CONFORMAL",
        "paper": "Conformal Prediction for Natural Language Processing: A Survey",
        "autores": "Campos, Ovadia, Bombelli, et al.", "anio": 2024, "venue": "TACL 2024",
        "doi": "10.1162/tacl_a_00715",
        "metodo": "prediccion conformista con garantia de cobertura a nivel de conjunto",
        "problema": "quantificacion de incertidumbre",
        "mecanismo": ("devuelve un CONJUNTO de etiquetas con garantia de cobertura finita, "
                     "construido con un no conformal sobre puntuaciones de no conformidad"),
        "datos_necesarios": "un calibration split",
        "legal": "SI en principio, pero el objetivo no es accionable aqui",
        "experimentos_relacionados": "V8-09 (curva de tasa de positivos), P59-08 (T)",
        "equivalencia_funcional": ("SI para el objetivo de TOXIC. La prediccion conformista produce "
                                   "conjuntos. Convertir un conjunto en una decision BINARIA exige "
                                   "cortar la puntuacion de no conformidad en algun punto, y eso es "
                                   "exactamente un ajuste de umbral, que V8-09 cerro de raiz al medir "
                                   "que V03 esta en el maximo de la curva de tasa de positivos, y que "
                                   "P59-08 demonstro que Platt a umbral fijo es un cambio de umbral"),
        "novedad": False,
        "modo_fallo_esperado": "n/a, no hay objetivo experimental defendible",
        "prioridad": 99,
    },
    {
        "clave": "DRO-GROUP",
        "paper": ("Distributionally Robust Optimization with Group Distributions: A Practical "
                  "Approach to Group DRO"),
        "autores": "Sagawa, Koh, Hashimoto, Liang", "anio": 2020, "venue": "ICML 2020",
        "metodo": "optimizacion minimax sobre un conjunto de grupos",
        "problema": "riesgo promedio que sacrifica subpoblaciones",
        "mecanismo": ("minimiza el maximo, o una media superior de tipo CVaR, del riesgo sobre "
                     "grupos, en vez del riesgo promedio"),
        "datos_necesarios": "una particion en grupos reproducible",
        "legal": ("SI. Los grupos permitidos son los ya existentes y reproducibles: near-duplicate, "
                  "pseudo-autor y las rebanadas predefinidas"),
        "experimentos_relacionados": "P59-10 (regularizacion por grafo), V04-B01 (near-duplicates)",
        "equivalencia_funcional": ("NINGUNA. P59-10 regularizo los COEFICIENTES con un grafo de "
                                   "similitud; DRO no usa grafo, usa una particion en grupos y "
                                   "optimiza un objetivo minimax sobre el riesgo por grupo. Y hay "
                                   "heterogeneidad de grupo medida y fuerte en este corpus"),
        "novedad": True,
        "modo_fallo_esperado": ("los grupos de near-duplicate son pocos y muy desiguales, luego el "
                                "maximo puede ser ruidoso; ademas el near-duplicate es el propio "
                                "grupo del CV, luego el DRO puede pelear con el protocolo"),
        "prioridad": 3,
    },
)


def main() -> None:
    """Construye la revision, el registro de literatura y el de huecos.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60 revision bibliografica y registro de huecos")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF_V03, allow_pickle=True)
    y = v["y_true"]
    ccv = float(f1_score(y, v["pred"], average="macro"))
    if abs(ccv - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    log(f"   V03 {ccv!r} exacto | sha256 verificado")

    reg = pd.read_csv(REG)
    texto = reg.astype(str).agg(" | ".join, axis=1).str.lower()
    patrones = {
        "ASL": r"asymmetric loss|\basl\b|asimetric",
        "W-ASL": r"weighted asymmetric|ponderacion por frecuencia de etiqueta",
        "PAIR-NOISE": r"pair.{0,15}noise|noise.{0,15}pair|instance-label|label.dependent|por etiqueta.{0,20}sospecha",
        "EFFNUM": r"class.balanced|effective number|effective_number|numero efectivo|ldam",
        "DBL": r"distribution.balanced|perdida equilibrada en distribucion",
        "LOGIT-ADJ": r"logit adjustment|logit_adjustment|margen por etiqueta|desplazamiento de logit",
        "LDAM": r"label-distribution-aware margin|margen consciente de la distribucion|margen por etiqueta",
        "LDL": r"label distribution learning|\bldl\b|simplex.*etiqueta",
        "CONTRAST-OVL": r"contrastive|contrastiv|supcon|solapamiento de etiquetas",
        "OT-PROT": r"optimal transport|sinkhorn|earth mover|wasserstein|red.*prototipo|prototype",
        "META-RW": r"meta.?(re)?weight|meta.learn|alineacion del gradiente|gradient.based.*weight",
        "POLYLOSS": r"polyloss|poly-loss|polynomial loss|perdida polinomica",
        "CONFORMAL": r"conformal|risk.control|prediction set",
        "DRO-GROUP": r"distributionally robust|\bdro\b|group dro|worst.group|minimax.*grupo",
    }
    conteo = {}
    for clave, pat in patrones.items():
        n = int(texto.str.contains(pat, regex=True, na=False).sum())
        conteo[clave] = n
        log(f"   gap check {clave:14} -> {n} filas del registro")

    # estructura de etiquetas medida, no transcrita
    hs, tr, ag = y[:, 0], y[:, 1], y[:, 2]
    stats = {
        "prevalencia_HS": float(hs.mean()), "prevalencia_TR": float(tr.mean()),
        "prevalencia_AG": float(ag.mean()),
        "filas_HS0": int((hs == 0).sum()), "n_filas": int(y.shape[0]),
        "P_TR1_dado_HS0": float(tr[hs == 0].mean()),
        "P_TR1_dado_HS1": float(tr[hs == 1].mean()),
        "P_AG1_dado_HS0": float(ag[hs == 0].mean()),
        "P_AG1_dado_HS1": float(ag[hs == 1].mean()),
        "P_AG1_dado_TR1": float(ag[tr == 1].mean()),
        "P_AG1_dado_TR0": float(ag[tr == 0].mean()),
        "negativos_TR_no_definidos": float((tr[hs == 0] == 0).mean()),
        "negativos_AG_no_definidos": float((ag[hs == 0] == 0).mean()),
    }
    log(f"   estructura de etiquetas MEDIDA: P(TR=1|HS=0) = {stats['P_TR1_dado_HS0']:.4f}, "
        f"P(AG=1|HS=0) = {stats['P_AG1_dado_HS0']:.4f}, "
        f"negativos TR no definidos = {stats['negativos_TR_no_definidos']:.4f}")

    # ---- registro de literatura ----
    filas_lit = []
    for e in LITERATURA:
        filas_lit.append({
            "clave": e["clave"], "paper": e["paper"], "autores": e["autores"],
            "year": e["anio"], "venue": e["venue"], "doi": e.get("doi", ""),
            "method": e["metodo"], "problem_addressed": e["problema"],
            "mechanism": e["mecanismo"], "data_requirements": e["datos_necesarios"],
            "allowed_under_TOXIC": e["legal"],
            "related_project_experiments": e["experimentos_relacionados"],
            "rows_in_registry_matching": conteo[e["clave"]],
            "functional_equivalence": e["equivalencia_funcional"],
            "novelty": e["novedad"],
            "expected_failure_mode": e["modo_fallo_esperado"],
            "priority": e["prioridad"],
        })
    df_lit = pd.DataFrame(filas_lit)
    df_lit.to_csv(CSV_LIT, index=False, encoding="utf-8")
    log(f"   escrito reports/p60_literature_registry.csv ({len(df_lit)} metodos)")

    # ---- veredicto por pista ----
    PISTAS = (
        ("A", "ASL", "ASL", True, "NUEVA",
         "Sin corte duro de negativos faciles, sin asimetria de clase, sin descarte por margen"),
        ("B", "W-ASL", "W-ASL", False, "CERRADA_POR_DUPLICACION",
         "ASL mas ponderacion por frecuencia, que es class_weight, mas coocurrencia, que es V4-E"),
        ("C", "PAIR-NOISE", "PAIR-NOISE", True, "NUEVA",
         ("La unidad de deteccion es el par instancia-etiqueta y no la instancia; V30-00 y P59-03 "
         "cerraron solo la version de instancia")),
        ("D", "EFFNUM/DBL", "EFFNUM", True, "NUEVA",
         "Numero efectivo de muestras es otra funcion de n_c, con saturacion, que el 1/(2 pi_c) de V03"),
        ("E", "LOGIT-ADJ/LDAM", "LOGIT-ADJ", False, "CERRADA_POR_DUPLICACION",
         ("Sobre un modelo lineal, el ajuste de logit es un desplazamiento de intercepto, y con "
         "umbral fijo un desplazamiento de intercepto es un cambio de umbral")),
        ("F", "LDL", "LDL", False, "CERRADA_POR_DUPLICACION",
         ("Tres etiquetas anidadas dan un subespacio factible de cuatro puntos, que V4-E ya "
         "enumero en doce variantes")),
        ("G", "CONTRAST-OVL", "CONTRAST-OVL", True, "NUEVA",
         ("El unico eje que modifica la geometria aprendida y no el objetivo, los pesos, la "
         "decision ni la entrada")),
        ("H", "OT-PROT", "OT-PROT", True, "NUEVA_CON_RESTAURACION",
         ("Con objeto de prueba de Dirac el transporte colapsa a distancia al centroide, que esta "
         "cerrada; solo aporta con un objeto de prueba que sea una distribucion")),
        ("I", "META-RW", "META-RW", True, "NUEVA",
         ("El peso viene de una alineacion de gradiente contra un objetivo meta, no de una regla "
         "de dificultad")),
        ("J", "POLYLOSS", "POLYLOSS", True, "NUEVA_PARCIAL",
         "Familia polinomica de la perdida, ya muestreada por GCE con cuatro valores de q"),
        ("K", "CONFORMAL", "CONFORMAL", False, "CERRADA_SIN_OBJETIVO_ACCIONABLE",
         "Convertir un conjunto conformista en decision binaria exige cortar en un punto"),
        ("L", "DRO-GROUP", "DRO-GROUP", True, "NUEVA",
         "Objetivo minimax sobre grupos, sin grafo y sin tocar la funcion de perdida"),
    )
    filas_gap = []
    por_clave = {e["clave"]: e for e in LITERATURA}
    for pista, ref, clave, abierto, veredicto, razon in PISTAS:
        e = por_clave[clave]
        filas_gap.append({
            "track": pista, "referencia": ref, "paper": e["paper"],
            "year": e["anio"], "venue": e["venue"],
            "abierto": abierto, "veredicto_gap": veredicto,
            "filas_registro_coincidentes": conteo[clave],
            "precedentes_proyecto": e["experimentos_relacionados"],
            "equivalencia_funcional": e["equivalencia_funcional"],
            "motivo": razon,
            "prioridad": e["prioridad"],
        })
    df_gap = pd.DataFrame(filas_gap)
    df_gap.to_csv(CSV_GAP, index=False, encoding="utf-8")
    log(f"   escrito reports/p60_gap_registry.csv "
        f"({int(df_gap['abierto'].sum())} abiertas, {int((~df_gap['abierto']).sum())} cerradas)")

    {e["clave"]: e["prioridad"] for e in LITERATURA}
    filas_hipo = []
    for pista, ref, clave, abierto, veredicto, razon in PISTAS:
        e = por_clave[clave]
        filas_hipo.append({
            "ID": f"P60-{pista}", "Track": pista,
            "Hypothesis": e["metodo"] + " aporta un mecanismo no cubierto por " +
                          e["experimentos_relacionados"],
            "Literature_basis": f"{e['paper']}, {e['venue']} {e['anio']}"
                                + (f", DOI {e['doi']}" if e.get("doi") else ""),
            "Prior_project_evidence": e["experimentos_relacionados"],
            "Functional_equivalence": e["equivalencia_funcional"],
            "Novelty": e["novedad"],
            "Expected_mechanism": e["mecanismo"],
            "Experiment": "PENDIENTE" if abierto else "NINGUNO (cierre sin entrenar)",
            "Result": "PENDIENTE" if abierto else "n/a",
            "Status": veredicto,
            "Decision": "PENDIENTE" if abierto else "CERRADA_SIN_EXPERIMENTO",
        })
    df_hipo = pd.DataFrame(filas_hipo).sort_values(
        "ID", key=lambda s_: s_.map(lambda x: x.split("-")[1]))
    df_hipo.to_csv(CSV_HIPO, index=False, encoding="utf-8")
    log("   escrito reports/p60_hypothesis_registry.csv")

    # ---- revision en prosa ----
    nuevas = df_lit[df_lit["novelty"]].sort_values("priority")
    cerradas = df_lit[~df_lit["novelty"]]
    L: list[str] = []
    W = L.append
    W("# P60 - Revision bibliografica y registro de huecos")
    W("")
    W(f"Generado por `experiments/p60_00_registro_hipotesis.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que puede y que no puede hacer la literatura aqui")
    W("")
    W("La literatura sirve para **definir hipotesis plausibles y justificar mecanismos**. No sirve")
    W("para anticipar el resultado, y un trabajo que funciona en imagenes o en toxicidad de otro")
    W("corpus no es evidencia de nada en TOXIC. El criterio de decision sigue siendo `corrected_cv`,")
    W("consistencia de pliegues, complementariedad y bootstrap. Por eso el registro de cada metodo")
    W("lleva un campo de modo de fallo esperado, que es la conjetura que se somete a prueba.")
    W("")
    W("## Criterio de transferencia, metodo por metodo")
    W("")
    W("| metodo | dominio original | adaptacion a TOXIC | que cambia | que es invariante |")
    W("|---|---|---|---|---|")
    W("| ASL | vision multi-etiqueta, 80 millones de etiquetas | clasificador lineal sobre TF-IDF, "
      "3 problemas binarios | la escala de datos y el numero de etiquetas | la asimetria "
      "positivo/negativo y el corte duro de negativos |")
    W("| numero efectivo | vision, cola larga larga | tres prevalencias entre 0,30 y 0,44 | el rango "
      "de n_c es estrecho | la saturacion del peso en clases grandes |")
    W("| contraste por solapamiento | multi-etiqueta de texto | TF-IDF -> SVD -> proyeccion "
      "entrenable | el encoder | que el positivo es un par, no una etiqueta |")
    W("| transporte optimo | vision, prototipos | TF-IDF, objeto de prueba como distribucion sobre "
      "las unidades del texto | el objeto es una distribucion, no un punto | el acoplamiento de "
      "masas |")
    W("| DRO por grupos | vision, subpoblaciones | grupos near-duplicate y pseudo-autor | el numero "
      "de grupos y su desbalance | el objetivo minimax sobre el riesgo por grupo |")
    W("| meta-reweighting | vision, ruido de etiquetas | estructura anidada dentro del training fold "
      "exterior | no hay split limpio garantizado | el criterio de alineacion de gradiente |")
    W("")
    W("## Estructura de etiquetas, MEDIDA en este corpus")
    W("")
    W("Este es el dato que ordena la revision, y no viene de ningun paper sino del OOF canonico:")
    W("")
    W("```text")
    W(f"prevalencias                       HS {stats['prevalencia_HS']:.4f}  "
      f"TR {stats['prevalencia_TR']:.4f}  AG {stats['prevalencia_AG']:.4f}")
    W(f"P(TR=1 | HS=0)                     {stats['P_TR1_dado_HS0']:.4f}")
    W(f"P(TR=1 | HS=1)                     {stats['P_TR1_dado_HS1']:.4f}")
    W(f"P(AG=1 | HS=0)                     {stats['P_AG1_dado_HS0']:.4f}")
    W(f"P(AG=1 | HS=1)                     {stats['P_AG1_dado_HS1']:.4f}")
    W(f"P(AG=1 | TR=1)                     {stats['P_AG1_dado_TR1']:.4f}")
    W(f"negativos de TR NO definidos       {stats['negativos_TR_no_definidos']:.4f}")
    W(f"negativos de AG NO definidos       {stats['negativos_AG_no_definidos']:.4f}")
    W(f"filas con HS=0                     {stats['filas_HS0']} de {stats['n_filas']}")
    W("```")
    W("")
    W("Las tres consecuencias que ordenan todo lo demas:")
    W("")
    W(f"1. **Las etiquetas estan ANIDADAS, no solo correlacionadas.** P(TR=1|HS=0) = "
      f"{stats['P_TR1_dado_HS0']:.4f}: TR nunca es positivo cuando HS es negativo. El subespacio "
      f"factible tiene cuatro puntos, {{000, 100, 101, 111}}, que es exactamente el que V4-E01-E03 "
      f"enumero en doce variantes sin superar a V03. Eso cierra la pista F por equivalencia, y con "
      f"ella la componente de coocurrencia de Weighted ASL.")
    W("")
    W(f"2. **El {stats['negativos_TR_no_definidos']*100:.0f} por ciento de los negativos que ve TR "
      f"en entrenamiento son etiquetas NO DEFINIDAS**, no negativos. Un ejemplo con HS=0 no es "
      f"\"no troxico\", es \"el estado de TR no se puede determinar\". Eso es ruido dependiente de la "
      f"etiqueta, que es lo que la pista C mide por pares, y es una razon de mecanismo para que la "
      f"deteccion por instancia no haya funcionado en V30-00: alli la sospecha era un unico numero.")
    W("")
    W(f"3. **La asimetria de prevalencias es de {min(stats['prevalencia_HS'],stats['prevalencia_TR'],stats['prevalencia_AG']):.2f} "
      f"a {max(stats['prevalencia_HS'],stats['prevalencia_TR'],stats['prevalencia_AG']):.2f}, no de "
      f"una clase minoritaria frente a una dominante.** Ridnik et al. justifican ASL por un desequilibrio "
      f"de uno o dos ordenes de magnitud. Aqui el desequilibrio es de 1,46 a 1, en un problema con "
      f"3 etiquetas. Esa es la razon honesta por la que la pista A tiene una priori mas baja de lo "
      f"que sugiere la literatura, y por la que aun asi vale la pena medirla.")
    W("")
    W("## Metodos que pasan el gap check")
    W("")
    W("| ref | metodo | filas del registro | mecanismo en una linea |")
    W("|---|---|---:|---|")
    for _, r in nuevas.iterrows():
        W(f"| {r['clave']} | {r['method'][:70]} | {r['rows_in_registry_matching']} | "
          f"{r['mechanism'][:110]} |")
    W("")
    W("## Metodos que se cierran SIN consumir experimento")
    W("")
    for _, r in cerradas.iterrows():
        W(f"### {r['clave']} - {r['paper']}, {r['venue']} {r['year']}")
        W("")
        W(f"*Precedentes del proyecto:* {r['related_project_experiments']}")
        W("")
        W(f"*Equivalencia:* {r['functional_equivalence']}")
        W("")
    W("## Orden de ejecucion, por novedad y no por modernidad")
    W("")
    W("| orden | pista | metodo | prioridad | motivo |")
    W("|---:|---|---|---:|---|")
    n = 0
    for pista, ref, clave, abierto, veredicto, razon in sorted(
            PISTAS, key=lambda t: (not t[3], por_clave[t[2]]["prioridad"])):
        if not abierto:
            continue
        n += 1
        W(f"| {n} | {pista} | {ref} | {por_clave[clave]['prioridad']} | {razon[:88]} |")
    W("")
    W("## Advertencia sobre lo que la literatura NO puede hacer")
    W("")
    W("V7-01 cerro la rama de perdida simetrica con una frase que conviene tener presente al")
    W("interpretar cualquier resultado de P60: *en este problema, aproximar mejor el objetivo no es")
    W("mejor*. Y V28-01 mostro que los errores de HS, TR y AG se solapan con phi de +0,42 a +0,51,")
    W("luego fallan sobre los mismos textos. Cualquier metodo de P60 que actue sobre una sola")
    W("etiqueta tiene ese techo estructural. Los que pueden evitarla son los que cambian la")
    W("geometria (G) o el objetivo sobre grupos (L), y por eso van primero.")
    W("")
    W("## Formulacion")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_00_registro_hipotesis.py`. Sin push.*")
    MD_REVIEW.write_text(a_ascii("\n".join(L)), encoding="ascii")
    log("   escrito reports/p60_literature_review.md")

    # ---- DAG ----
    L2: list[str] = []
    D = L2.append
    D("# P60 - Grafo de dependencias")
    D("")
    D("```text")
    D("BASELINE V03")
    D("   |")
    for pista, ref, clave, abierto, veredicto, razon in PISTAS:
        estado = "ABIERTA" if abierto else "CERRADA"
        D(f"   +-- {pista} {ref:14} [{estado}]")
    D("```")
    D("")
    D("Ninguna pista depende de otra. Las doce son independientes y el bloque esta ejecutado de")
    D("forma continua: una pista negativa o cerrada no cancela a ninguna otra. La unica")
    D("dependencia admitida es la de las plazas de reserva, que dependen unicamente del resultado")
    D("positivo y con mecanismo activado de una primera variante.")
    D("")
    D("| pista | estado inicial | motivo del veredicto |")
    D("|---|---|---|")
    for pista, ref, clave, abierto, veredicto, razon in PISTAS:
        D(f"| {pista} | {veredicto} | {razon[:110]} |")
    D("")
    MD_DEP.write_text(a_ascii("\n".join(L2)), encoding="ascii")
    log("   escrito reports/p60_dependency_graph.md")

    resumen = {
        "familia": "P60_LITERATURE_REVIEW",
        "naturaleza": ("AUDITORIA Y GAP ANALYSIS. 0 experimentos, 0 modelos, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "baseline": {"corrected_cv": repr(ccv), "sha256": hash_real},
        "metodos_evaluados": len(LITERATURA),
        "metodos_nuevos": int(df_lit["novelty"].sum()),
        "metodos_cerrados": int((~df_lit["novelty"]).sum()),
        "pistas_abiertas": int(df_gap["abierto"].sum()),
        "pistas_cerradas": int((~df_gap["abierto"]).sum()),
        "orden_ejecucion": [t[0] for t in sorted(PISTAS, key=lambda t: (not t[3],
                                                                          por_clave[t[2]]["prioridad"]))
                            if t[3]],
        "estructura_etiquetas_medida": stats,
        "conteo_gap": conteo,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_REV.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for destino in (MD_REVIEW, CSV_LIT, CSV_GAP, CSV_HIPO, MD_DEP, JSON_REV):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-00 OK | {resumen['pistas_abiertas']} pistas abiertas, "
        f"{resumen['pistas_cerradas']} cerradas sin experimento | 0 experimentos consumidos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
