"""Revision bibliografica 2026: registro, duplicaciones y decision.

## Lo que entrega

Los tres registros que el brief pide (`age_literature_review_2026.md` y `.csv`,
`age_literature_methods_registry.csv`), la tabla de duplicaciones, la de rechazados, y la decision.

## La validacion que se hizo antes de escribir

El veredicto de la busqueda se **valido contra el repositorio**, no se acepto de la lectura:

| Comprobacion | Resultado |
|---|---|
| ¿AGE ejecuto soft targets o label smoothing? | **0 scripts, 0 menciones.** Cero |
| ¿AGE ejecuto EMD/Wasserstein? | 0 scripts |
| ¿AGE proyecto features fuera de direcciones de topico? | 0 scripts |
| ¿Cuantas perdidas custom tiene el proyecto? | **2**: margen (V18-P3) y destilacion (V18-P5) |
| ¿CORAL esta cerrado? | Si: V22-ORDINAL-COMPARTIDO, V23-01-CORAL |
| ¿El intercepto optimo es 0? | Si, medido en P33 |

Dos recuentos iniciales fueron **falsos positivos de mi propia busqueda** y se declaran:
`complement` matcheaba "complementario" en el texto en espanol (391 veces), y `sord` matcheaba
`tensordot` (15 veces). Ninguno de los dos era evidencia de que el mecanismo se hubiera probado.

**La conclusion de la validacion**: existen **huecos de ejecucion reales** (targets suaves,
EMD, proyeccion de topicos nunca se corrieron), y la pregunta pasa a ser si la literatura
justifica llenarlos. Es una pregunta distinta y se responde aparte.

## La distincion que decide

El brief prohibe aceptar una propuesta porque el paper reporte una mejora. Hay que mirar la
**metrica**. Y ahi esta el hallazgo central de esta revision:

```text
macro-F1 es una metrica NOMINAL: trata todos los errores igual.
La literatura ordinal optimiza metricas DISTANCIA-CONSCIENTES: MAE, QWK, RPS, 1-off.
```

Dos fuentes lo dicen de forma explicita y convergente:

- **Vargas et al., IEEE TKDE 2026** (12 datasets ordinales, 20 configuraciones): el etiquetado
  suave mejora la generalizacion ordinal, pero **la cross-entropy categorica rinde mejor en
  promedio en metricas nominales**.
- **Kasai et al., arXiv 2405.11775** (texto, comparativa explicita de perdidas ordinales):
  *"CE performs best in terms of nominal metrics; OLL performs best in terms of ordinal metrics;
  hay un trade-off: la mejora ordinal se paga con metricas nominales"*.

Eso convierte el hallazgo en una respuesta: **el hueco de ejecucion existe y la evidencia dice que
llenarlo empuja en la direccion contraria a la del concurso.**
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_revision_bibliografica.json"
REV_MD = cfg.REPORTS / "age_literature_review_2026.md"
REV_CSV = cfg.REPORTS / "age_literature_review_2026.csv"
REG_CSV = cfg.REPORTS / "age_literature_methods_registry.csv"
EXC_CSV = cfg.REPORTS / "age_literature_excluded_methods.csv"
GAP_CSV = cfg.REPORTS / "age_literature_gap_registry.csv"
FECHA = "2026-10-08"
CV_V30 = 0.6134471
LB_V30 = 0.63302
VARIABILIDAD = 0.0008876

#: Los cinco candidatos, con el vector (novedad, evidencia, compatibilidad, transferibilidad,
#: coste invertido) que el especialista puntuo de 1 a 5.
CANDIDATOS = [
    {"id": "C1", "nombre": "Perdidas distancia-conscientes sobre cabeza multinomial (SORD, "
                            "triangular/beta, DLDL, OLL/MLL)",
     "paper_evidencia": "Diaz & Marathe CVPR 2019 (SORD, Adience, +2,7 pp exactitud en 2 replicas "
                        "independientes); Vargas et al. Pattern Recognition 2022 / Information "
                        "Fusion 2023 / Information Sciences 2023 / ESWA 2024 (familias de kernel, "
                        "mejora de sensibilidad en clases extremas); Castagnos et al. COLING 2022 "
                        "(OLL); Kasai et al. arXiv 2405.11775 (texto, comparativa explicita)",
     "por_que_no_probado": "AGE nunca ejecuto un target suave: **0 scripts, 0 menciones**. Lo mas "
                           "cercano son 2 perdidas custom (margen en V18-P3 y destilacion en "
                           "V18-P5), ninguna de target distribuido. La auditoria cerro 'label "
                           "smoothing' por argumento, no por medicion",
     "diferencia_con_cerrado": (
         "**No es CORAL**: CORAL son K-1 salidas sigmoidales acumulativas con pesos compartidos. "
         "**No es pairwise**: pairwise entrena K(K-1)/2 binarios (invalidado en V28-03); aqui hay un "
         "solo modelo. **Si es un superconjunto de label smoothing uniforme**: con q_k = "
         "exp(-beta|k-y|^2) se recupera LS cuando beta tiende a 0. Algebraicamente, -suma q_k "
         "log p_k = CE(y) - suma_{k!=y} q_k log p_k, y su gradiente p - q = (p - e_y) + (e_y - q): "
         "CE one-hot mas un **sesgo fijo dependiente de la clase**, no una superficie de decision "
         "distinta. Es regularizacion del eje de etiquetas, no informacion nueva"),
     "mecanismo_esperado": "reduccion de confianza en fronteras adyacentes, con cambios de argmax "
                           "solo en la region de empate",
     "modo_de_fallo": "el argmax no cambia; el trade-off nominal/ordinal documentado en texto; "
                      "ganancia dentro del ruido de entrenamiento",
     "control": "tres brazos: CE one-hot, LS uniforme (el cierre de la auditoria), kernel con "
                "etiquetas permutadas (regularizacion pura). Gate: delta >= +0,0015 con 4/5 folds "
                "**y** delta > 8,9e-04 con signo consistente",
     "coste": "small", "riesgo": "bajo; bandera gris si el kernel se aprende de la matriz de "
                                 "confusion OOF",
     "recomendacion": "UNICO EXPERIMENTO DE FALSACION DEFENDIBLE, con expectativa honesta de que "
                      "NO cruza el gate. Su valor es dejar el cierre con evidencia propia en vez de "
                      "por analogia",
     "vector": "novedad 2 / evidencia 4 / compatibilidad 5 / transferibilidad 4 / coste 5"},
    {"id": "C2", "nombre": "Aprendizaje con ruido ordinal explicito (Ord-NLL, Garg & Manwani)",
     "paper_evidencia": "Garg & Manwani ACML 2020 (primer robusto para ordinal bajo ruido "
                        "class-conditional); Yuan et al. PRICAI 2023; Takezaki et al. IEEE Access "
                        "2026 (Ord-NLL, **reporta macro-F1** = 0,625 en ruido severo, imagenes "
                        "medicas)",
     "por_que_no_probado": "el eje nunca estuvo en el espacio de mecanismos; la auditoria cerro "
                           "hard mining y focal por analogia, no con experimentos de ruido",
     "diferencia_con_cerrado": (
         "no es class_weight (la correccion es condicional al modelo y a la etiqueta) ni focal "
         "(focal sube el peso de ejemplos dificiles; una perdida robusta a ruido **baja** la "
         "contribucion de desviaciones confiadas). Pero el propio Ord-NLL demuestra (Corollary 1) "
         "que **su minimizador esperado es un target suave**: es decir, colapsa a C1 en expectativa"),
     "mecanismo_esperado": "reduce el sobreajuste a etiquetas corruptas; potencial mejora en las "
                           "clases extremas",
     "modo_de_fallo": "si train y test comparten el proceso de anotacion, el optimo sobre etiquetas "
                      "ruidosas coincide con el limpio **salvo un desplazamiento de umbral por "
                      "clase** (Natarajan et al. NeurIPS 2013), y AGE ya midio que el intercepto "
                      "optimo es **0** (P33). El unico canal corregible por la teoria ya esta "
                      "cerrado por medicion",
     "control": "inyectar ruido ordinal sintetico **solo en el fold de entrenamiento** y evaluar "
                "en el fold limpio: es la unica forma de verificar el mecanismo. Controles: brazo "
                "de soft labels (equivalencia en expectativa) y brazo de umbral",
     "coste": "medium", "riesgo": "bajo si el ruido se genera solo sobre train",
     "recomendacion": "NO REABRIR. Es el eje no auditado, pero tiene tres cierres convergentes: "
                      "equivalencia en expectativa con C1, el argumento de mismo-proceso de ruido "
                      "con el umbral ya medido en 0, y la no identificabilidad sin datos limpios",
     "vector": "novedad 5 / evidencia 3 / compatibilidad 4 / transferibilidad 3 / coste 3"},
    {"id": "C3", "nombre": "Regularizacion de forma unimodal (CO2, ORCU-LBE, stick-breaking)",
     "paper_evidencia": "ORCU arXiv 2410.15658v4: calibracion SOTA (ECE/SCE/ACE), y en clasificacion "
                        "SORD supera a ORCU en Adience (59,10 frente a 57,50 de exactitud). "
                        "Stick-breaking Neurocomputing 2020; Beckham & Pal ICML 2017",
     "por_que_no_probado": "el post-procesado de calibracion se cerro (PH32/PH33) con el intercepto "
                           "en 0; la restriccion de **forma** no se probo explicitamente",
     "diferencia_con_cerrado": "la calibracion es una transformacion monotona de las "
                               "probabilidades; la unimodalidad es una restriccion funcional sobre "
                               "los **gaps de logits**. No es equivalente a soft labels. Pero el "
                               "efecto sobre el argmax es marginal salvo en empates, y toda la "
                               "evidencia de la que presumiria es de **calibracion**, no de F1",
     "mecanismo_esperado": "cambios de prediccion solo en muestras cercanas al empate; mejora de "
                           "confiabilidad, no de F1",
     "modo_de_fallo": "identico a C1 sin su parte util, mas un hiperparametro extra (escala de "
                      "barrera) que infla la superficie de tuning",
     "control": "aislar SoftCE frente a SoftCE+LBE; el control decisivo es el brazo SoftCE solo",
     "coste": "small-medium", "riesgo": "ninguno",
     "recomendacion": "MANTENER CERRADO. La evidencia propia del paper dice que la ganancia es de "
                      "calibracion y que no supera a su componente de soft labels en exactitud",
     "vector": "novedad 3 / evidencia 4 / compatibilidad 5 / transferibilidad 2 / coste 4"},
    {"id": "C4", "nombre": "Separacion estilo/contenido y eliminacion de atajos sin etiquetas de "
                            "entorno (DePro, INLP, NFL, topic-marginalization)",
     "paper_evidencia": "DePro Dou et al. COLING 2022 (RFF + reponderacion; **la exactitud "
                        "in-distribution BAJA**: MNLI 84,3 a 83,2; FEVER 85,4 a 84,5, y solo mejora "
                        "OOD); INLP Ravfogel et al. ACL 2020; NFL Chew et al. EACL Findings 2024; "
                        "Seroussi et al. CoNLL 2011",
     "por_que_no_probado": "la seleccion supervisada de features se cerro (V25) y el "
                           "cluster-conditioning se uso (10 scripts), pero nunca se hizo una "
                           "proyeccion al espacio nulo de pseudo-atributos tematicos",
     "diferencia_con_cerrado": "no es chi2/MI (que selecciona por relevancia con la etiqueta) ni "
                               "cluster-conditioning (que condiciona la prediccion al cluster). Es "
                               "una **proyeccion de subespacio**, y ese mecanismo concreto no se "
                               "ejecuto: 0 scripts",
     "mecanismo_esperado": "reduce la dependencia de palabras topicas",
     "modo_de_fallo": "el CV IID no se mueve o baja: **la ganancia de DePro es estrictamente OOD y "
                      "el CV de AGE es IID**. Y en edad los topicos **son** la senal principal "
                      "(escuela, trabajo, hijos, jubilacion): proyectarlos fuera puede borrar senal",
     "control": "proyeccion de direcciones aleatorias con la misma dimensionalidad; chi2 como "
                "referencia del cierre",
     "coste": "large", "riesgo": "el clustering debe entrenarse dentro del fold de train; usar el "
                                 "texto de test rozaria 'test para desarrollo'",
     "recomendacion": "NO REABRIR. La propia Tabla 1 de DePro muestra caida in-distribution, y la "
                      "medicion de AGE ya dice que el canal estilometrico integrado aporta "
                      "+0,0003: el priors es negativo en las dos direcciones",
     "vector": "novedad 3 / evidencia 2 / compatibilidad 3 / transferibilidad 2 / coste 2"},
    {"id": "C5", "nombre": "Geometria ordinal de clases: prototipos restringidos y covarianzas "
                            "condicionales (CPL, Mahalanobis/QDA)",
     "paper_evidencia": "CPL Wang et al. AAAI 2023 (proxies con restriccion de layout; Adience "
                        "61,6-62,1 de exactitud frente a 59,6 de SORD, con VGG-16 preentrenado en "
                        "ImageNet)",
     "por_que_no_probado": "se usaron prototipos, centroides y kNN, pero sin restricciones de "
                           "layout ni metrica aprendida. La capacidad del meta-modelo se cerro "
                           "(V29/V30)",
     "diferencia_con_cerrado": "los prototipos de AGE no imponian geometria global; CPL anade una "
                               "restriccion de layout y un objetivo de similitud unimodal. Es una "
                               "diferencia real pero incremental",
     "mecanismo_esperado": "mejoras en MAE y en errores lejanos; para macro-F1 necesita mover "
                           "fronteras adyacentes",
     "modo_de_fallo": "la ganancia de CPL esta atada a un **extractor de features preentrenado "
                      "(VGG-16)**; con TF-IDF/SVD y un MLP la restriccion de layout puede no "
                      "aportar nada, y estimar covarianzas en dimension alta y dispersa es mal "
                      "condicionado",
     "control": "clasificador de centroides sin restriccion en la misma representacion y folds; "
                "QDA regularizada sobre SVD para la parte de Mahalanobis",
     "coste": "medium", "riesgo": "ninguno",
     "recomendacion": "MANTENER CERRADO. Baja novedad (prototipos ya USED) y evidencia dependiente "
                      "de un extractor prohibido",
     "vector": "novedad 2 / evidencia 2 / compatibilidad 5 / transferibilidad 2 / coste 3"},
]

#: Duplicaciones: paper_method -> metodo existente en AGE -> relacion -> por que.
DUPLICACIONES = [
    ("SORD, soft labels (Diaz & Marathe CVPR 2019)", "label smoothing (cerrado por argumento en la "
     "auditoria); class_weight (V18-P1)", "Superconjunto de LS uniforme",
     "CE con target suave = CE one-hot + sesgo fijo por clase; con beta tendiendo a 0 colapsa a LS"),
    ("DLDL (Gao et al. IEEE TIP 2017); LDL (Geng TKDE 2016)", "soft labels (misma familia)",
     "Variante con marginalizacion sobre la edad continua", "Misma modificacion de target"),
    ("Kernels triangular/beta/generalizada (Vargas 2022-2024)", "soft labels (familia)",
     "Eleccion de kernel dentro de la misma familia", "El kernel cambia la forma, no el mecanismo"),
    ("LCM (Guo et al. AAAI 2021); LDL Dual-kNN (SDM 2025)", "kernel aprendido; kNN (USED)",
     "Kernel de similitud aprendido + kNN en inferencia", "El modulo LDL es target-side; kNN cerrado"),
    ("Stick-breaking (Neurocomputing 2020); CO2; VS-SL; ORCU-LBE",
     "calibracion y post-proceso (PH32/PH33)", "Restriccion de forma frente a calibracion",
     "ORCU no supera a SORD en exactitud; su ganancia declarada es de calibracion"),
    ("EMD2 (Hou et al. 2017)", "CORAL y variantes (CLOSED); pairwise (invalidado)",
     "**Equivalencia algebraica**: EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2",
     "Es la familia acumulativa all-threshold con perdida L2: la misma descomposicion que "
     "V22/V23/H10/D-058, con otra perdida"),
    ("OLL/MLL (Castagnos et al. COLING 2022; Kasai et al. 2024)",
     "No probado; lo mas cercano es el cierre de label smoothing",
     "Penalizacion complementaria distancia-consciente, NO descomposicion",
     "Es la variante **menos cubierta** por los cierres, y por eso es lo unico que sobrevive como "
     "candidato (C1)"),
    ("WKL (de la Torre et al. PRL 2018)", "ranking como objetivo (descartado); metrica macro-F1",
     "Surrogado de QWK", "QWK es distancia-consciente y macro-F1 es nominal: desajuste de metrica"),
    ("CORAL (Cao et al. PRL 2020), CORN, CLM, OBD, ECOC ordinal",
     "V22-ORDINAL-COMPARTIDO, V23-01-CORAL, H10, D-058", "Identidad o variante directa",
     "Ya auditado con control de atribucion"),
    ("Nested dichotomies (Frank & Hall ECML 2001)", "descomposicion pairwise (V28-03, -0,067553)",
     "Misma familia de descomposicion", "Cambia el esquema de particion, no el mecanismo"),
    ("Ord-NLL (Takezaki et al. IEEE Access 2026)", "soft labels (familia)",
     "**Equivalencia en expectativa demostrada en el propio paper** (Corollary 1)",
     "El minimizador esperado es un target suave exponencial"),
    ("Garg & Manwani ACML 2020; SVOR robusto (Neurocomputing 2024)",
     "umbral e intercepto (PH32/PH33, optimo 0)", "Para ruido simetrico el optimo limpio solo "
     "cambia el umbral por clase", "La parte corregible por teoria ya se midio y no aporta"),
    ("JTT (Liu et al. ICML 2021)", "hard example mining (auditoria)", "Reponderar ejemplos "
     "dificiles", "Mismo conjunto objetivo de muestras"),
    ("DFR (Kirichenko et al. ICLR 2023)", "class_weight (V18-P1); capacidad (V29/V30)",
     "Reentrenar la ultima capa con muestra balanceada",
     "Las clases de AGE ya estan balanceadas; es un refit"),
    ("DePro (Dou et al. COLING 2022)", "seleccion supervisada (V25); seleccion de muestras",
     "Decorrelacion y purificacion (familia robustez OOD)",
     "La Tabla 1 del paper muestra caida in-distribution; la ganancia es OOD"),
    ("INLP (Ravfogel et al. ACL 2020)", "seleccion supervisada (V25)",
     "Proyeccion de subespacio espurio", "Mecanismo distinto, sin evidencia en edad/texto"),
    ("NFL (Chew et al. EACL Findings 2024)", "no probado",
     "Deteccion de tokens espurios por vecindad", "Robustez a OOD; no ataca macro-F1 IID"),
    ("LDA/author-topic (Seroussi et al. CoNLL 2011; CL 2014)", "cluster conditioning (10 scripts)",
     "Prediccion marginalizando topicos", "Es condicionar por cluster: misma familia"),
    ("CPL (Wang et al. AAAI 2023)", "prototipos y centroides (USED); kNN (USED)",
     "Prototipos con layout restringido", "Diferencia incremental; ganancia atada a VGG preentrenado"),
    ("Label enhancement (Xu & Geng TKDE 2019)", "soft labels (familia)",
     "Construccion de distribuciones desde vecindad", "Target-side con kernel de kNN"),
    ("Optimizacion directa de macro-F1 (Eban et al. AISTATS 2017); reglas plug-in (Lipton ICML 2014)",
     "calibracion y umbral (PH32/PH33), intercepto 0", "Regla de decision sobre posteriores",
     "En multiclase balanceado la parte accionable es el umbral por clase, ya medido"),
]

#: Papers y metodos rechazados, con motivo.
RECHAZADOS = [
    ("EMD/Wasserstein ordinal (Hou et al. 2017; EMD en XGBoost)",
     "Cerrado por equivalencia algebraica con la familia acumulativa CORAL/all-threshold"),
    ("Ranking ordinal distinto de lo probado",
     "No aparece nada que no sea all-threshold, pairwise (V28-03), CORAL/CORN o WKL (surrogado de "
     "QWK, metrica distancia-consciente)"),
    ("ORCU como palanca de macro-F1", "Su contribucion medida es calibracion; no supera a su "
     "componente SoftCE en exactitud"),
    ("Correlacion de logits por priors (logit adjustment, Menon et al. 2021)",
     "Las clases de AGE estan balanceadas; no aplica"),
    ("Calibracion post-hoc, temperature scaling, Dirichlet",
     "Cerrado por PH32/PH33; el intercepto optimo medido es 0"),
    ("Filtrado de muestras ruidosas, co-teaching, sample selection",
     "Equivalente a hard example mining (auditoria) o a Ord-NLL (C2)"),
    ("Curriculo por dificultad, focal, OHEM",
     "Cerrado por auditoria: apuntan a la poblacion A, donde ninguna de 26 fuentes tiene senal"),
    ("Stacking, bagging, seed ensembles",
     "Cerrado: H04/H06 por equivalencia funcional; V28-02 en -0,000312"),
    ("Seleccion de features supervisada, incluida SHAP",
     "Cerrado por V25 (D-060, -0,000629)"),
    ("ECOC ordinal (Barbero-Gomez et al. 2023, dlordinal)", "Misma familia de descomposicion; "
     "pairwise invalidado en V28-03"),
    ("CORAL, CORN, CLM, stick-breaking", "Cerrado por D-058 y anexos, o familia de C3"),
    ("CPL y prototipos restringidos",
     "Baja novedad (prototipos USED) y evidencia dependiente de extractor preentrenado"),
    ("Mahalanobis y QDA por clase",
     "Sin evidencia en texto/edad; geometria mal condicionada en representacion dispersa"),
    ("DePro, INLP, NFL, JTT, DFR, IRM, Group-DRO",
     "Requieren grupos o entornos, o son variantes de reponderacion, o su ganancia es "
     "estrictamente OOD. El CV de AGE es IID"),
    ("Author-topic y marginalizacion por LDA", "Cluster-conditioning ya usado en 10 scripts"),
    ("Label enhancement (Xu & Geng 2019)", "Familia soft-labels con kernel de vecindad; sin "
     "evidencia de macro-F1 en texto"),
    ("Concept-based LDL (Springer 2022)", "Requiere conocimiento externo de conceptos: "
     "INCOMPATIBLE por reglamento"),
    ("Contrastive learning desde label distribution (Neurocomputing 2022)",
     "Perdidas contrastivas con redes; sin evidencia de macro-F1 en edad; familia de "
     "representacion cerrada"),
    ("Aumentacion, mixup, EDA en texto",
     "Sin evidencia ordinal-texto-macro-F1; los modelos de AGE son lineales y el dataset es grande"),
    ("Modelos secuenciales from-scratch (char-CNN, LSTM)",
     "Permitidos en principio, pero la literatura clasica de perfilado muestra que los n-gramas "
     "con lineales son competitivos, y no hay evidencia de que el orden aporte en edad"),
    ("Modelos y etiquetas de edad externos, FastText/word2vec preentrenados, LLM, entailment",
     "PROHIBIDO; su transferibilidad es nula sin el modelo preentrenado"),
    ("Metadatos de usuario",
     "PROHIBIDO. La literatura (PLoS ONE 2017, F1 0,739) muestra que ahi esta la mayor ganancia, "
     "y por eso la prohibicion es la que mas limita"),
    ("Arboles y forests ordinales (ORCA, ordinal forest, LightGBM ordinal)",
     "Familia de objetivos ordinales cerrada; LightGBM multinomial ya usado"),
    ("Conformal, abtencion, opcion de rechazo",
     "No mejora macro-F1 con un argmax forzado por fila"),
    ("Ruido de anotacion con multiples anotadores",
     "AGE no tiene multiples anotadores; queda solo como C2"),
]


def main() -> int:
    """Escribe los registros de la revision y la decision."""
    autotest_numerico()
    inicio = time.time()
    print("=== registro de candidatos ===", flush=True)
    for c in CANDIDATOS:
        print(f"  {c['id']} {c['recomendacion'][:64]}", flush=True)
    print("\n=== duplicaciones ===", flush=True)
    print(f"  {len(DUPLICACIONES)} pares paper-metodo -> metodo existente en AGE", flush=True)
    print("\n=== rechazados ===", flush=True)
    print(f"  {len(RECHAZADOS)} metodos o familias con motivo", flush=True)

    # El criterio de decision, explicito.
    nuevo = [c for c in CANDIDATOS if c["recomendacion"].startswith("UNICO")]
    print("\n=== decision ===", flush=True)
    print(f"  candidatos con recomendacion de ejecucion: {len(nuevo)} ({[c['id'] for c in nuevo]})",
          flush=True)
    print("  y ese candidato es una **falsacion**, no una apuesta: el especialista dice que su", flush=True)
    print("  expectativa honesta es que NO cruce el gate, y que su valor es cerrar con "
          "evidencia propia", flush=True)
    print("  en vez de por analogia", flush=True)
    conclusion = "NO_NEW_METHOD"
    print(f"\n=== {conclusion} ===", flush=True)

    # CSV principal
    filas = []
    for c in CANDIDATOS:
        filas.append({
            "paper_id": c["id"], "year": 2026, "title": c["nombre"],
            "task": "clasificacion ordinal / age estimation", "language": "multi / en",
            "method": c["nombre"], "mechanism": c["diferencia_con_cerrado"][:200],
            "compatibility": ("COMPATIBLE" if c["id"] in ("C1", "C2", "C3", "C5")
                              else "PARTIALLY_COMPATIBLE"),
            "transferability": c["vector"], "AGE_existing_precedent": c["por_que_no_probado"][:200],
            "novelty": ("GENUINELY_NEW" if c["id"] == "C2" else
                        "PARTIAL_PRECEDENT" if c["id"] == "C1" else "FUNCTIONAL_DUPLICATE"),
            "leakage_risk": c["riesgo"], "implementation_cost": c["coste"],
            "priority": c["id"] == "C1", "recommendation": c["recomendacion"],
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(REV_CSV, index=False)
    pd.DataFrame(CANDIDATOS).to_csv(REG_CSV, index=False)
    pd.DataFrame(RECHAZADOS, columns=["metodo", "reason_excluded"]).to_csv(EXC_CSV, index=False)
    pd.DataFrame(DUPLICACIONES, columns=["paper_method", "AGE_existing_method", "relationship",
                                         "reason"]).to_csv(GAP_CSV, index=False)

    _md(tabla, conclusion)
    RESULTADO.write_text(json.dumps({
        "fase": "revision bibliografica", "fecha": FECHA, "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                                 text=True, timeout=30).stdout.strip(),
        "baseline_v30": CV_V30, "leaderboard": LB_V30,
        "conclusion": conclusion,
        "candidatos": CANDIDATOS, "duplicaciones": len(DUPLICACIONES),
        "rechazados": len(RECHAZADOS),
        "validacion_contra_repositorio": {
            "soft_targets_o_label_smoothing_ejecutados": 0,
            "emd_wasserstein_ejecutados": 0,
            "proyeccion_de_topicos_ejecutada": 0,
            "perdidas_custom_en_todo_el_proyecto": 2,
            "cuales": ["v18_p3_margen.perdida_con_margen", "v18_p5_destilacion.perdida_destilacion"],
            "falsos_positivos_declarados": [
                "'complement' matcheaba 'complementario' en el texto en espanol (391 veces)",
                "'sord' matcheaba 'tensordot' (15 veces)"],
            "coral_cerrado": True, "intercepto_optimo": 0.0,
        },
        "hallazgo_central": (
            "macro-F1 es una metrica NOMINAL y la literatura ordinal optimiza metricas "
            "DISTANCIA-CONSCIENTES. Dos fuentes convergentes lo dicen de forma explicita: Vargas et "
            "al. IEEE TKDE 2026 (12 datasets, 20 configuraciones) y Kasai et al. arXiv 2405.11775 "
            "(texto). El hueco de ejecucion existe y la evidencia dice que llenarlo empuja en la "
            "direccion contraria a la del concurso"),
        "restricciones": {"entrenamiento": 0, "usa_test": False, "usa_holdout": False,
                          "submissions": 0, "usa_leaderboard_para_elegir": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _md(tabla: pd.DataFrame, conclusion: str) -> None:
    lineas = [
        "# AGE - revision bibliografica orientada a mecanismos (2026)", "",
        f"**Conclusion: `{conclusion}`**", "",
        f"**Baseline congelado:** V30 = {CV_V30} corrected_cv, leaderboard {LB_V30}.",
        f"**Variabilidad de entrenamiento medida:** {VARIABILIDAD}.", "",
        "## La pregunta", "",
        "> ¿Existe algun mecanismo publicado y metodologicamente defendible que pueda implementarse "
        "sobre AGE text-only, sin embeddings preentrenados, transformers, LLMs, metadata ni datos "
        "etiquetados externos, y que sea genuinamente diferente de las 39 familias ya auditadas?", "",
        "## Lo que se valido antes de escribir, y no se acepto de la lectura", "",
        "El veredicto se comprobo contra el repositorio. Tres comprobaciones cambiaron el analisis:",
        "",
        "| Comprobacion | Resultado |", "|---|---|",
        "| ¿AGE ejecuto soft targets o label smoothing? | **0 scripts, 0 menciones** |",
        "| ¿AGE ejecuto EMD/Wasserstein? | **0 scripts** |",
        "| ¿AGE proyecto features fuera de direcciones de topico? | **0 scripts** |",
        "| ¿Cuantas perdidas custom tiene el proyecto entero? | **2**: margen (V18-P3) y destilacion "
        "(V18-P5) |",
        "| ¿CORAL esta cerrado? | Si: V22-ORDINAL-COMPARTIDO, V23-01-CORAL |",
        "| ¿El intercepto optimo es 0? | Si, medido en P33 |", "",
        "**Dos recuentos iniciales fueron falsos positivos de mi propia busqueda** y se declaran: "
        "`complement` matcheaba \"complementario\" en el texto en espanol (391 veces) y `sord` "
        "matcheaba `tensordot` (15 veces). Ninguno era evidencia de que el mecanismo se hubiera "
        "probado.", "",
        "**La validacion deja un hecho incomodo y hay que decirlo**: existen **huecos de ejecucion "
        "reales**. El target suave, EMD y la proyeccion de topicos nunca se corrieron. La "
        "`label smoothing` que la auditoria dijo \"cerrada\" lo estaba **por argumento, no por "
        "medicion**. La pregunta pasa a ser si la literatura justifica llenar esos huecos.", "",
        "## El hallazgo central: desajuste de metrica", "",
        "```text",
        "macro-F1 es una metrica NOMINAL: trata todos los errores igual.",
        "La literatura ordinal optimiza metricas DISTANCIA-CONSCIENTES: MAE, QWK, RPS, 1-off.",
        "```", "",
        "Dos fuentes convergentes lo dicen de forma explicita:", "",
        "- **Vargas et al., IEEE TKDE 2026** (12 datasets ordinales, 20 configuraciones): el "
        "etiquetado suave mejora la generalizacion ordinal, pero **la cross-entropy categorica "
        "rinde mejor en promedio en metricas nominales**.",
        "- **Kasai et al., arXiv 2405.11775** (texto, comparativa explicita de perdidas ordinales): "
        "*\"CE performs best in terms of nominal metrics; OLL performs best in terms of ordinal "
        "metrics; hay un trade-off: la mejora ordinal se paga con metricas nominales\"*.", "",
        "**Eso convierte el hallazgo en respuesta**: el hueco existe y la evidencia dice que "
        "llenarlo empuja en la direccion contraria a la del concurso.", "",
        "## Los cinco candidatos", "",
        "| ID | Mecanismo | Vector (nov/evid/compat/transf/coste) | Recomendacion |",
        "|---|---|---|---|",
    ]
    for c in CANDIDATOS:
        lineas.append(f"| **{c['id']}** | {c['nombre'][:88]} | {c['vector']} | "
                      f"{c['recomendacion'][:96]} |")
    for c in CANDIDATOS:
        lineas += ["", f"### {c['id']} - {c['nombre']}", "",
                   f"**Evidencia publicada:** {c['paper_evidencia']}", "",
                   f"**Por que AGE no lo probo:** {c['por_que_no_probado']}", "",
                   f"**Diferencia con lo cerrado:** {c['diferencia_con_cerrado']}", "",
                   f"**Control propuesto:** {c['control']}", "",
                   f"**Mecanismo esperado:** {c['mecanismo_esperado']}", "",
                   f"**Modo de fallo esperado:** {c['modo_de_fallo']}", "",
                   f"**Coste:** {c['coste']} | **Riesgo:** {c['riesgo']}", "",
                   f"**Recomendacion:** {c['recomendacion']}", ""]
    lineas += ["## Tabla de duplicaciones", "",
               "| Metodo del paper | Metodo existente en AGE | Relacion | Por que |",
               "|---|---|---|---|",
               ]
    for p, a, r, w in DUPLICACIONES:
        lineas.append(f"| {p} | {a} | {r} | {w} |")
    lineas += ["", "La duplicacion mas contundente es **EMD2**, que la literatura suele presentar "
               "como un objetivo ordinal distinto y que es **algebraicamente** la familia "
               "acumulativa: `EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2`. Es CORAL con perdida "
               "L2, y esa familia se cerro en V22/V23/H10/D-058.", "",
               "Y la mas importante para la decision es **Ord-NLL**: su propio paper demuestra "
               "(Corollary 1) que su minimizador esperado es un target suave, asi que colapsa a C1 "
               "en expectativa. Eso cierra el eje de ruido ordinal por equivalencia, no por "
               "opinion.", "",
               "## Metodos rechazados", "",
               f"**{len(RECHAZADOS)}** metodos o familias con su motivo, en "
               f"`age_literature_excluded_methods.csv`. No se borran: un rechazo documentado evita "
               f"que alguien los vuelva a proponer en seis meses.", "",
               "## La decision, y su razonamiento", "",
               "```text", conclusion, "```", "",
               "**Por que `NO_NEW_METHOD`.** Ningun candidato cumple a la vez las tres condiciones "
               "del brief: compatible con el reglamento, no equivalente a algo cerrado, y con "
               "evidencia de mejora de macro-F1 en clasificacion de edad a partir de texto con "
               "modelos clasicos.", "",
               "- **C1** es el unico con un hueco de ejecucion real, y su cierre no es algebraico "
               "sino **empirico**: la literatura dice que el trade-off va en contra de la metrica "
               "nominal. Ademas es un superconjunto de label smoothing, que la auditoria ya habia "
               "descartado.",
               "- **C2** es el eje mas genuinamente no auditado, y colapsa a C1 en expectativa. Su "
               "parte corregible por teoria (el umbral por clase) ya se midio: es 0.",
               "- **C3**, **C4** y **C5** tienen modos de fallo que predicen delta por debajo del "
               "ruido, o mecanismos estructuralmente OOD cuando el CV de AGE es IID.", "",
               "**Expectativa honesta de C1, si se ejecutara.** El especialista es explicito: "
               "**no cruza el gate**. Su valor seria dejar el cierre con evidencia propia en lugar "
               "de por analogia. Eso es un experimento de **falsacion**, no una apuesta, y el brief "
               "no lo pide: pide decidir si existe una razon cientifica para reabrir, y la "
               "respuesta es **no**.", "",
               "## Lo que esta revision anade al proyecto", "",
               "1. Un argumento **algebraico** nuevo para el cierre de EMD/Wasserstein: es la "
               "familia acumulativa, no un objetivo distinto.",
               "2. La distincion entre **calibracion** y **forma unimodal**, con la evidencia de "
               "ORCU de que la segunda no mueve el argmax.",
               "3. La **equivalencia en expectativa** Ord-NLL equivalente a soft labels, "
               "demostrada en el propio paper.",
               "4. La advertencia documentada de que **las perdidas ordinales transfieren mal a "
               "metricas nominales**, con dos fuentes convergentes.",
               "5. El unico eje no auditado (ruido ordinal) con su contra-argumento de "
               "mismo-proceso y el umbral ya medido en 0.", "",
               "Todo esto es evidencia de **cierre**, no un plan de accion. Y deja el mapa "
               "completo: representacion, familias de modelo, objetivos, estructura de etiquetas, "
               "integracion en score, ruido, sesgo de topicos y geometria de clases.", "",
               "## Lo que esta revision NO dice", "",
               "- No dice que V30 sea optimo.",
               "- No dice que no exista ninguna mejora posible.",
               "- No dice que la literatura no tenga nada util: dice que lo que tiene **ya esta "
               "cubierto** o **empuja contra la metrica**.", "",
               "La conclusion es sobre el espacio de mecanismos auditado, y ahora esta auditado "
               "dos veces: contra el codigo del proyecto y contra la literatura 2020-2026.", "",
               ]
    REV_MD.write_text("\n".join(lineas).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
