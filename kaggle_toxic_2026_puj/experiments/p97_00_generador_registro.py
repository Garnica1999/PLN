#!/usr/bin/env python
"""
P97-00: genera el registro de metodos y la matriz de novedad (P97 seccion 3 y 5).

Escribe, con validacion estricta y ASCII puro:
  reports/p97_method_registry.csv
  reports/p97_novelty_matrix.csv

Sin entrenamiento; es documentacion de la investigacion externa (fuentes
primarias revisadas el 2026-10-13) y de la clasificacion de novedad contra el
codigo y los resultados del proyecto.

Uso:
    /home/carlos/venv/bin/python experiments/p97_00_generador_registro.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAL_REGISTRO = RAIZ / "reports" / "p97_method_registry.csv"
SAL_MATRIZ = RAIZ / "reports" / "p97_novelty_matrix.csv"

CAB_REGISTRO = ["work_id", "nombre", "autores", "anio", "publicacion", "doi_url",
                "area", "problema", "mecanismo", "datos_requeridos",
                "modelo_requerido", "compatibilidad_toxic", "limitaciones",
                "experimentos_historicos_relacionados", "diferencia_exacta",
                "clasificacion"]

CAB_MATRIZ = ["candidate_id", "mecanismo", "fuentes", "novedad_clase",
              "compatibilidad", "probabilidad_cualitativa", "posible_impacto",
              "coste_computacional", "riesgo_sobreajuste", "evidencia_necesaria",
              "criterio_descarte", "separa_efectos", "decision"]

# Clasificacion A-E segun P97 seccion 3.
REGISTRO = [
    ["W01", "nnPU (Positive-Unlabeled with Non-Negative Risk Estimator)",
     "Kiryo, Niu, du Plessis, Sugiyama", "2017", "NeurIPS 30",
     "arXiv:1703.00593", "3 (ruido/etiquetas)",
     "entrenar sin negativos fiables: el negativo observado puede ser una "
     "etiqueta no definida",
     "R = pi*Rp+ + max(0, Ru- - pi*Rp-); el termino U corrige el sesgo con la "
     "mezcla; recorte no negativo evita el sobreajuste del estimador insesgado",
     "P y U (aqui: TR=1 y filas HS=0 como mezcla)", "cualquier score (lineal "
     "vale); optimizacion directa del estimador",
     "total: usa solo etiquetas oficiales de train; HS=0 es U de facto",
     "requiere pi (aqui estimable del propio train); asume que U es mezcla "
     "representativa; con n pequeno el sesgo de pi pesa",
     "V14 (la perdida enmascarada resulto identica a H2-M: ignora U, no lo "
     "usa); M-P94-14 (self-training, prohibido); V12-02 (techo oracle +0.046)",
     "V14 declaro cerrada la via ENMASCARADA, no la formulacion que trata "
     "HS=0 como NO SUPERVISADO; nnPU usa esas filas como mezcla (senal "
     "distinta de cero) y no existe como mecanismo en el historial",
     "D"],
    ["W02", "PNU risk estimator (P, N y U con pesos)",
     "Sakai, du Plessis, Niu, Sugiyama", "2017", "ICML 70",
     "https://proceedings.mlr.press/v70/sakai17a/sakai17a.pdf",
     "3 (ruido/etiquetas)",
     "aprovechar tambien los negativos definidos y la mezcla U",
     "combinacion lineal de riesgos PN, PU y NU con pesos; varianza menor que "
     "el riesgo PN simple",
     "P, N y U (aqui: TR=1; TR=0 con HS=1; filas HS=0)", "lineal o flexible",
     "total: particion natural del train",
     "los pesos dependen de priors; sensibilidad a pi",
     "mismo historial que W01; nunca ejecutado", "nunca probado; es la "
     "variante que usa toda la estructura de anotacion disponible", "D"],
    ["W03", "DC-PU (estimador que balancea tasas de error P y N)",
     "Li y col.", "2025", "NeurIPS 2025 (poster)",
     "https://neurips.cc/virtual/2025/poster/119218", "3 (ruido/etiquetas)",
     "inestabilidad de nnPU en regimenes dificiles",
     "restriccion de igualdad entre tasas de error, resuelta con Lagrange "
     "aumentado",
     "P y U", "flexible", "total", "muy reciente; sin replicas externas",
     "no probado; variante de W01", "B: misma familia PU; preferible agotar "
     "W01 antes que una variante 2025", "B"],
    ["W04", "Group DRO (minimax sobre peor grupo)",
     "Sagawa, Koh, Hashimoto, Liang", "2020", "ICLR (arXiv 1911.08731)",
     "https://arxiv.org/abs/1911.08731", "6 (generalizacion/transferencia)",
     "los errores por grupo pueden no transferir cuando la mejora se concentra "
     "en grupos grandes",
     "minimax: q_g ~ q_g * exp(eta * perdida_g); cada ronda reajusta el "
     "modelo con pesos de grupo adversarios",
     "grupos definidos a priori (near-dup UNION pseudo-autor ya existen)",
     "lineal vale (pesos por fila)",
     "total: los grupos son una propiedad del train",
     "exige regularizacion fuerte en modelos flexibles; puede sacrificar la "
     "media; con grupos muy pequenos es varianza",
     "V6-03 (peso estatico por tamano de grupo, 1/sqrt(g)); V4-B "
     "(downweighting por coste); M-P94-12 lo clasifico ALREADY_TESTED por "
     "'equivalencia de re-ponderacion'",
     "Lo probado fueron reglas ESTATICAS de peso; el minimax iterativo "
     "(objetivo de peor grupo) nunca se ejecuto. Cambia la hipotesis causal: "
     "de 'corregir una fuente de ponderacion' a 'comprar robustez/transferencia'",
     "B"],
    ["W05", "Reglas de decision por subgrupo (multicalibracion / omniprediccion)",
     "Hebert-Johnson, Kim, Reingold, Rothblum (2018); Gopalan y col. (2022+)",
     "2018-2025", "ICML 2018; ITCS 2022; NeurIPS 2025",
     "https://proceedings.mlr.press/v80/hebert-johnson18a.html",
     "2 (decision) y 6 (transferencia)",
     "una probabilidad globalmente calibrada puede no serlo por subgrupos; la "
     "decision optima depende del subgrupo",
     "recalibrar f en subgrupos identificables de forma iterativa; con "
     "predictor multicalibrado, cualquier perdida se optimiza con una funcion "
     "de la prediccion (linea omnipredictor)",
     "predictor ya entrenado (V03) + atributos de subgrupo (longitud, "
     "mencion, hashtag)", "ninguno nuevo: solo capa de decision",
     "total: subgrupos construidos con el texto, nada externo",
     "la multicalibracion garantiza calibracion por subgrupo, no F1; el "
     "numero de subgrupos debe pre-registrarse para no sobreajustar",
     "TH-01 (umbral global por etiqueta, adoptado); E2-16/V7-03 (umbral "
     "anidado GLOBAL: 0.776613; sin mejora); P55/V56-01 (anidamiento "
     "completo, +0.000925); M-P94-04 cerro el lado de DECISION GLOBAL",
     "Ninguna variante historica condiciono el umbral a atributos del "
     "ejemplo. La regla global es el caso especial; la condicional es un "
     "objeto distinto (umbral como funcion de x) y es falsable por separado",
     "D"],
    ["W06", "Multicalibracion sin discretizacion por minimizacion de perdida",
     "autores varios", "2025", "NeurIPS 2025",
     "https://proceedings.neurips.cc/paper_files/paper/2025/hash/b0435b5fc7ca",
     "2 (decision)", "coste de la multicalibracion clasica",
     "remuestreo iterativo sin discretizar el espacio de salida",
     "predictor + calibracion", "ninguno", "total",
     "muy reciente; disenado para clasificacion binaria general",
     "no probado; refinamiento de W05", "B: variante de W05; preferible una "
     "implementacion simple y pre-registrada", "B"],
    ["W07", "Regla plug-in optima para F1 (umbral relativo a la densidad)",
     "Lipton, Elkan, Naryanaswamy", "2014", "ECML/PKDD (PMC4442797)",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/",
     "2 (decision)",
     "relacion exacta entre el umbral optimo y la razon de densidades de "
     "positivos",
     "el umbral Bayes-optimo para F1 depende de la densidad local p(s|t=1), "
     "no es constante entre regiones",
     "score probabilistico", "ninguno",
     "total", "exige probabilidades bien calibradas",
     "TH-01 (umbral global adoptado); V47/P60 (decision rules cerradas)",
     "Aporta la justificacion teorica de umbrales dependientes de la region; "
     "la version CONDICIONAL por segmento nunca se ejecuto", "B"],
    ["W08", "Reglas plug-in vs minimizacion estructurada en multietiqueta F-macro",
     "Dembczynski, Jachnik, Kotlowski, Waegeman, Hullermeier", "2013", "ICML",
     "https://www.semanticscholar.org/paper/b07b1248", "2 (decision)",
     "hallar la regla optima para F-macro en multietiqueta",
     "con clasificadores por etiqueta independientes, la regla Bayes para "
     "F-macro es umbral por etiqueta sobre posteriores",
     "posteriores por etiqueta", "ninguno",
     "total", "asume independencia condicional",
     "TH-01; E2-16; M-P94-04", "Confirma que el marco global ya probado es "
     "la subclase con umbral constante; no aporta mecanismo nuevo por si "
     "mismo", "A"],
    ["W09", "F-measure via clasificacion sensible a costes (reduccion)",
     "Narasimhan, Vaish, Agarwal", "2014", "NeurIPS 27",
     "https://proceedings.neurips.cc/paper_files/paper/2014/hash/5c0314ec",
     "2 (decision)", "optimizar F1 via problema de costes secuencial",
     "reduce F1 a serie (infinita) de problemas sensibles a costes",
     "scores", "lineal o mas", "total",
     "aprox por discretizacion", "A_cost_sensitive_downweight (cerrado); "
     "P60 (perdidas)", "misma familia de decision/costes ya medida",
     "A"],
    ["W10", "BOVA (thresholds por clase desde distribucion balanceada)",
     "equipo Skeepers (blog tecnico)", "2023", "blog (sin peer review)",
     "https://techblog.skeepers.io/how-to-boost-your-f1-score-for-multiclass",
     "2 (decision)", "macro-F1 multiclase con desbalanceo",
     "thresholds optimizados aprendiendo de la distribucion original y una "
     "balanceada", "logits de validacion", "clasificador", "parcial",
     "sin fuente primaria revisada; resultados de blog",
     "sin historial directo", "No se acepta como evidencia (sin peer "
     "review); su parte de decision = W05/W07", "C"],
    ["W11", "SigmoidF1 (sustituto suave de F1)",
     "Benedict y col.", "2021", "arXiv:2108.10566",
     "https://arxiv.org/abs/2108.10566", "1 (objetivos)",
     "F1 no diferenciable", "sustituto continuo sigmoide de F1",
     "etiquetas", "red", "total",
     "misma clase de sustitutos", "V7-01 (F1-suave literal: la peor); "
     "P59-10/P60 (perdidas)", "sustituto nuevo como funcion, misma familia "
     "de mecanismo (ya medida)", "A"],
    ["W12", "Label smoothing uniforme",
     "Szegedy y col. (2016); Muller, Kornblith, Hinton", "2019", "NeurIPS 32",
     "https://arxiv.org/abs/1906.02629", "1/3 (objetivos/ruido)",
     "exceso de confianza y memorizacion de ruido",
     "mezclar objetivo one-hot con uniforme; penaliza confianza excesiva",
     "etiquetas", "cualquiera (en LR es suavizar objetivos con pesos)",
     "total", "efecto tipicamente pequeno; con LR el suavizado uniforme "
     "empuja ambas clases hacia 0.5",
     "no probado en el proyecto (ninguna fila lo registra)",
     "nunca ejecutado; candidato C por cercania a la familia de pesos/"
     "objetivos ya medida", "C"],
    ["W13", "Label smoothing mitiga ruido de etiqueta (analisis)",
     "Lukasik y col.", "2020", "ICML 119",
     "https://proceedings.mlr.press/v119/lukasik20a/lukasik20a.pdf",
     "3 (ruido)", "analiza por que el suavizado ayuda con ruido",
     "la suavizacion reduce la memorizacion de ejemplos mal etiquetados",
     "etiquetas", "red o lineal", "total", "evidencia en vision, no texto",
     "no probado", "soporte de W12; mismo mecanismo", "C"],
    ["W14", "Aprendizaje del suavizado de etiquetas para texto",
     "Ren y col.", "2024", "PLoS ONE (PMC11057568)",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC11057568/",
     "1/3", "suavizado uniforme ignora diferencias semanticas",
     "aprender la matriz de suavizado desde datos", "etiquetas", "red",
     "total", "requiere entrenamiento conjunto; con LR limita",
     "no probado", "variante de W12; mismo nucleo", "C"],
    ["W15", "Matriz de transicion dependiente de la instancia (part-dependent)",
     "Xia y col.", "2020", "NeurIPS 33",
     "https://proceedings.nips.cc/paper_files/paper/2020/file/5607fe8879e4fd269",
     "3 (ruido)", "modelar el ruido de etiqueta como funcion de la instancia",
     "aproxima T(x) como combinacion de matrices de 'partes' aprendidas",
     "etiquetas + instancias", "red flexible; varias epocas",
     "parcial: requiere modelo flexible y supuestos de partes",
     "identificabilidad dudosa con texto corto y 1483 filas",
     "V28-03 (detector MARGINAL); hipotesis de ruido por instancia nunca "
     "entrenada",
     "exige aprendizaje profundo con memorizacion; con LR sparse no hay "
     "partes identificables; muy cercano a familia de ruido ya acotada",
     "C"],
    ["W16", "Matriz de transicion causal",
     "Li y col.", "2024", "arXiv:2412.13516", "https://arxiv.org/abs/2412.13516",
     "3 (ruido)", "separar causa del ruido", "modelo causal del proceso de "
     "corrupcion", "etiquetas", "red", "parcial", "muy reciente; supuestos "
     "fuertes", "no probado", "misma familia que W15; C", "C"],
    ["W17", "Transicion con regularizacion por variedad",
     "Cheng y col.", "2022", "CVPR", "https://openaccess.thecvf.com/content/"
     "CVPR2022/papers/Cheng_Instance-Dependent_Label-Noise_Learning.pdf",
     "3 (ruido)", "estabilidad de la estimacion de T(x)",
     "estimacion de la matriz de transicion con suavidad manifold",
     "etiquetas + caracteristicas", "red", "parcial", "idem W15",
     "no probado", "misma familia; C", "C"],
    ["W18", "Co-teaching (dos redes intercambian instancias de menor perdida)",
     "Han y col.", "2018", "NeurIPS 31",
     "https://papers.nips.cc/paper/8072-co-teaching",
     "3 (ruido)", "evitar memorizar etiquetas ruidosas",
     "dos redes se ensenan mutuamente solo las instancias de menor perdida "
     "del mini-batch", "etiquetas", "redes neuronales (depende del efecto "
     "memorizacion)", "parcial: pensado para DL; con LR no hay fase de "
     "memorizacion", "con LR sparse el 'small-loss trick' degenera",
     "D-057 (prohibido preentrenados; DL desde cero posible); V11 "
     "(DL desde cero por debajo de V03)", "requiere memorizacion profunda "
     "que el proyecto no puede simular con LR; C", "C"],
    ["W19", "DISC (seleccion dinamica por instancia con umbral de memorizacion)",
     "Li y col.", "2023", "CVPR",
     "https://liner.com/review/coteaching-robust-training", "3 (ruido)",
     "seleccion por fuerzas de memorizacion", "bloquea instancias por umbral "
     "dinamico", "etiquetas", "red", "parcial", "idem W18", "no probado",
     "idem W18; C", "C"],
    ["W20", "Metrica aprendida (NCA) para kNN",
     "Goldberger y col. (NCA); sklearn", "2004-2023",
     "JMLR / sklearn docs", "https://scikit-learn.org/stable/modules/neighbors.html",
     "4 (modelos locales)", "kNN sensible a la metrica en alta dimension",
     "aprende transformacion lineal maximizando el kNN LOO estocastico",
     "etiquetas + vectores densos (SVD in-fold)", "lineal + kNN",
     "total", "en 27k sparse el NCA necesario = SVD (familia cerrada como "
     "representacion); alto coste y geometria memorizadora ya medida",
     "P94 H-P94-03 (kNN nunca ejecutado, INSUFFICIENT_EVIDENCE); V26-01 "
     "(RBF memorizador); kNN simple ~+0.005 medido en su dia",
     "la metrica APRENDIDA no se probo, pero su sustrato (SVD) y su salida "
     "(vecindad) pertenecen a familias cerradas; C", "C"],
    ["W21", "Regularizacion por variedad (Laplacian SVM)",
     "Belkin, Niyogi, Sindhwani", "2006", "JMLR 7",
     "https://www.jmlr.org/papers/volume7/belkin06a/belkin06a.pdf",
     "4 (grafos)", "suavizar la funcion en el grafo de similitud",
     "penaliza variacion de f sobre el grafo de vecinos (Laplaciano)",
     "etiquetas + grafo", "kernel o lineal",
     "parcial: el grafo de near-duplicados esta cuarentenado (D-288); el "
     "grafo kNN por coseno seria nuevo",
     "sin U autorizado el grafo aporta solo regularizacion transductiva "
     "sobre train",
     "D-288 (cuarentena de near-dup); no probado",
     "depende de construir un grafo nuevo (kNN); su efecto es regularizar "
     "direcciones de vecindad = territorio de familias cerradas; C", "C"],
    ["W22", "Born-Again Networks (auto-destilacion con objetivos suaves)",
     "Furlanello, Lipton, Tschannen, Itti, Anandkumar", "2018", "ICML 80",
     "https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf",
     "8 (otras areas)", "los estudiantes iguales superan a sus maestros",
     "entrenar de nuevo con los logits suaves del maestro (dark knowledge)",
     "logits + etiquetas", "red; con LR una 'generacion 2' es viable",
     "total", "efecto reportado en redes; con LR el equivalente es "
     "reentrenamiento con objetivos suaves (familia de pesos/objetivos)",
     "no probado", "cercano a W12 y a la familia de objetivos; C", "C"],
    ["W23", "Cross-validation anidado para replicabilidad (NCV)",
     "Bates, Hastie, Tibshirani", "2021/2024", "JASA 119(546); arXiv NCV",
     "https://pubmed.ncbi.nlm.nih.gov/39308484/",
     "6 (generalizacion)", "CV estima el error de modelos de otros trains, "
     "no del modelo final; IC clasicos subcubren",
     "CV anidado con repeticiones; IC con varianza correcta",
     "metricas de CV existentes", "ninguno",
     "total", "no mejora un modelo: mejora la evaluacion",
     "D-070 (banda 0.017); V10 (replicas); gates P95",
     "no genera modelo; se registra como criterio de evaluacion (fase P97 "
     "seccion 6: robustez ante particiones)", "B"],
    ["W24", "Confident Learning (deteccion y limpieza de errores de etiqueta)",
     "Northcutt, Jiang, Chuang", "2021", "JAIR 70",
     "https://arxiv.org/abs/1911.00068", "3 (ruido)",
     "estimar errores de etiqueta por clase",
     "matriz de confianza OOF + poda de ejemplos sospechosos",
     "probabilidades OOF", "cualquiera",
     "parcial: limpiar/reetiquetar el train modifica etiquetas (prohibido "
     "aqui); el detector propio ya fue MARGINAL (V28-03)",
     "el reetiquetado esta fuera del marco; solo valdria como pesos, que "
     "es territorio cerrado",
     "V28-03 (MARGINAL 0.9730 vs 0.9741); P87/P93-R5 (no modificar "
     "etiquetas)", "lo compatible (re-ponderar) ya se midio; C", "C"],
    ["W25", "Correccion de pares (instancia, etiqueta) en MLTC ruidoso",
     "y col.", "2024", "Findings NAACL 2024",
     "https://aclanthology.org/2024.findings-naacl.93.pdf",
     "3 (ruido)", "etiquetas ruidosas en clasificacion multietiqueta de texto",
     "correccion por pares (i,l) usando efecto memorizacion; umbral dinamico",
     "texto + etiquetas", "transformers/BERT (el propio paper usa BERT)",
     "no: depende de modelos preentrenados, prohibidos en TOXIC",
     "n/a", "no probado", "incompatible por la restriccion de "
     "preentrenados; aporta contexto del campo; E", "E"],
]

MATRIZ = [
    ["C97-A", "Umbrales condicionados por segmento (decision como funcion de x)",
     "W05, W06, W07, W08", "D",
     "total: solo V03 + texto del ejemplo",
     "baja-media: la mejora media de re-optimizaciones fue +0.0009; la parte "
     "condicional no medida",
     "pequeno pero potencialmente real en F1 por region (V28-01: el error se "
     "concentra en textos largos/URL)",
     "CPU-min (sin entrenamiento: usa OOF guardado)",
     "medio: slices pre-registrados y anidamiento estricto; sin grid",
     "seleccion nested por pliegue + bootstrap por clusters + salvage/damage",
     "descartar si no supera al umbral global anidado en las mismas "
     "condiciones (aisla el efecto de condicionar)",
     "CV: comparacion directa; complementariedad: salvage/damage vs V03; "
     "robustez: mejora por pliegue; transferencia: esperada si el efecto es "
     "de region y no de composicion",
     "SELECCIONADA (H97-A)"],
    ["C97-B", "Group DRO minimax (peor grupo) con grupos del proyecto",
     "W04", "B (reapertura de M-P94-12 con cambio de hipotesis causal)",
     "total: grupos ya definidos en el protocolo",
     "baja-media: compra robustez, no media; puede sacrificar la media",
     "medio si estabiliza pliegues; la varianza entre pliegues es 0.0161",
     "CPU: 225 ajustes LR (minutos)",
     "medio-alto: el minimax puede sobreajustar grupos pequenos; "
     "regularizacion = pesos acotados",
     "5 pliegues canonicos con umbrales congelados + control uniforme en el "
     "mismo arnes + bootstrap por clusters",
     "descartar si el peor grupo no mejora y la media cae, o si no "
     "estabiliza pliegues",
     "CV: comparacion con control identico; complementariedad: salvage/"
     "damage; robustez: dispersion entre pliegues; transferencia: hipotesis "
     "declarada (transferibilidad de grupos)",
     "SELECCIONADA (H97-B)"],
    ["C97-C", "nnPU / PNU para las cabezas TR y AG (HS=0 como mezcla U)",
     "W01, W02, W03", "D",
     "total: solo train oficial",
     "baja-media: la semantica de anotacion es real (80.44 por ciento de "
     "negativos no definidos en TR), pero el gold de evaluacion usa 0 para "
     "esas filas",
     "podria mover TR/AG sin tocar HS; o empeorar por usar una densidad "
     "distinta a la evaluada",
     "CPU/GPU menor: optimizacion directa del estimador (minutos)",
     "alto: estimador nuevo con pi estimado; se pre-registra pi unico y "
     "sin grid",
     "control torch BCE identico en el mismo arnes + thresholds propios "
     "nested + bootstrap por clusters",
     "descartar si no supera el control torch en pooled o si el sesgo de pi "
     "domina (peor que control en >=4/5 pliegues)",
     "CV: contra control identico; complementariedad: salvage/damage vs V03; "
     "robustez: pliegues; transferencia: no evaluable (freno vigente)",
     "SELECCIONADA (H97-C)"],
    ["C97-D", "kNN con metrica aprendida (NCA) sobre SVD in-fold",
     "W20", "C",
     "parcial: SVD y vecindad en familias cerradas",
     "muy baja: kNN simple ~+0.005 con geometria memorizadora en contra",
     "bajo", "CPU-min",
     "alto: la metrica aprendida sobre 1186 filas sobreajusta",
     "igual que los experimentos v4/p84", "descartar salvo evidencia "
     "adyacente nueva", "no aplica (no seleccionada)", "NO SELECCIONADA"],
    ["C97-E", "Suavizado de objetivos por ruido estimado (flip-soft)",
     "W12, W13, W14, W24", "C (pesos/objetivos es familia cerrada; el "
     "suavizado no se probo)",
     "total", "muy baja: la familia de pesos tiene techo medido",
     "bajo", "CPU-min", "alto: estimacion de ruido sobre 1186 filas",
     "detector de ruido propio (V28-03 MARGINAL) + pesos", "descartar si el "
     "efecto no supera el ruido", "no aplica", "NO SELECCIONADA"],
    ["C97-F", "Regularizacion Laplaciana sobre grafo kNN nuevo",
     "W21", "C (grafo nuevo, pero nucleo = vecindad/direcciones)",
     "parcial: exige grafo kNN por coseno (nuevo)",
     "muy baja", "bajo", "CPU-min a horas (kernel)", "alto",
     "grafo + validacion de fuga", "descartar por cercania a familias "
     "cerradas y coste", "no aplica", "NO SELECCIONADA"],
    ["C97-G", "Auto-destilacion (Born-Again) con logits de la generacion 1",
     "W22", "C", "total", "muy baja", "bajo", "CPU-min", "medio",
     "logits OOF por pliegue", "descartar salvo hallazgo nuevo",
     "no aplica", "NO SELECCIONADA"],
    ["C97-H", "Seleccion estable de n-gramas (stability selection)",
     "criterio de evaluacion W23", "C", "total", "muy baja",
     "bajo", "CPU-min", "medio",
     "repeticiones de submuestreo", "descartar: apunta a familia de "
     "seleccion de features cerrada", "no aplica", "NO SELECCIONADA"],
]


def validar(nombre: str, cab: list[str], filas: list[list[str]],
            clasificaciones: set[str] | None = None,
            col_clasif: int | None = None) -> list[str]:
    """Valida estructura, campos y ASCII."""
    fallos: list[str] = []
    for f in filas:
        if len(f) != len(cab):
            fallos.append(f"{nombre} {f[0]}: {len(f)} campos, esperados {len(cab)}")
            continue
        if any(not str(v).strip() for v in f):
            fallos.append(f"{nombre} {f[0]}: campo vacio")
        if clasificaciones and col_clasif is not None:
            valor = f[col_clasif].split(" ")[0].strip()
            if valor not in clasificaciones:
                fallos.append(f"{nombre} {f[0]}: clasificacion invalida {f[col_clasif]}")
    if any(ord(c) > 127 for fila in filas for celda in fila for c in str(celda)):
        fallos.append(f"{nombre}: byte > 127")
    return fallos


def main() -> None:
    """Escribe y valida ambos CSV."""
    for ruta in (SAL_REGISTRO, SAL_MATRIZ):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    fallos = validar("registry", CAB_REGISTRO, REGISTRO,
                     {"A", "B", "C", "D", "E"}, col_clasif=len(CAB_REGISTRO) - 1)
    fallos += validar("novelty", CAB_MATRIZ, MATRIZ,
                      {"A", "B", "C", "D", "E"}, col_clasif=3)
    if fallos:
        for f in fallos:
            print("  -", f)
        sys.exit(1)
    for ruta, cab, filas in ((SAL_REGISTRO, CAB_REGISTRO, REGISTRO),
                             (SAL_MATRIZ, CAB_MATRIZ, MATRIZ)):
        with open(ruta, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(cab)
            w.writerows(filas)
        with open(ruta, encoding="utf-8", newline="") as fh:
            leidas = list(csv.reader(fh))
        if len(leidas) != len(filas) + 1:
            raise SystemExit(f"Relectura inconsistente: {ruta.name}")
    from collections import Counter
    clases = Counter(f[-1].split(" ")[0] for f in REGISTRO)
    print("P97-00 registro de metodos: ejecutado")
    print(f"  registry: {len(REGISTRO)} mecanismos | clases: {dict(clases)}")
    print(f"  novelty matrix: {len(MATRIZ)} candidatos | "
          f"seleccionados: {sum(1 for f in MATRIZ if f[-1].startswith('SELECCIONADA'))}")
    print("  ambos CSV escritos y validados")


if __name__ == "__main__":
    main()
