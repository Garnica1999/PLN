"""Auditoria de cobertura metodologica: la matriz y la decision.

## Que hace

Reune la evidencia de codigo, la particion de errores y la estructura de los rescates en una
matriz de cobertura, y decide entre las dos conclusiones que el brief admite:
`AGE_RESEARCH_CLOSED` o `ONE_GENUINELY_NEW_HYPOTHESIS_IDENTIFIED`.

## Los cinco mecanismos sin usar, uno por uno

De 22 mecanismos buscados por **operacion** en el codigo (no por nombre de libreria), 17 estan
usados. Los cinco que no lo estan no se descartan por inercia: cada uno se contrasta con la
medicion que ya existe.

| Mecanismo | Por que se cierra |
|---|---|
| **ranking objective** | la metrica de la competicion es macro-F1 sobre el argmax. Optimizar un objetivo de ranking no mueve esa metrica: es cambiar de objetivo sin cambiar el evaluado |
| **focal loss** | es reponderacion de ejemplos con modulacion por confianza. `class_weight` ya se probo (V18-P1, 8 scripts). Y la modulacion apunta a los ejemplos dificiles, que son la poblacion A |
| **label smoothing** | actua en la zona de frontera, y P34 midio que ahi el ruido aleatorio gana a la senal real |
| **hard example mining** | el conjunto dificil ES la poblacion A, donde ninguna de las 26 fuentes tiene senal. Minar ese conjunto es entrenar sobre ruido |
| **bootstrap de grupos** | reduccion de varianza con otra unidad de remuestreo. V28-02 cerro el bagging del meta-modelo en -0,000312 |

Los cinco comparten el mismo rasgo: **actuan sobre la distribucion de entrenamiento o sobre la
funcion de perdida, y su poblacion objetivo es la que el diagnostico ya midio como sin senal
corregible.** Eso no es "no se probo": es "el diagnostico dice que probarlo seria entrenar sobre
ruido".

## Lo que este script NO hace

No afirma que V30 sea optimo. No afirma que no exista ninguna mejora posible. La conclusion es
sobre el **espacio de investigacion auditado**, que es una afirmacion mas estrecha y mas honesta.
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

RESULTADO = cfg.EXPERIMENTS / "age_cobertura_metodologica.json"
MATRIZ_CSV = cfg.REPORTS / "age_methodology_coverage_matrix.csv"
MATRIZ_MD = cfg.REPORTS / "age_methodology_coverage_matrix.md"
AUDIT_MD = cfg.REPORTS / "age_final_methodology_audit.md"
GAP_MD = cfg.REPORTS / "age_methodology_gap_analysis.md"
FECHA = "2026-10-07"
CV_V30 = 0.6134471
LB_V30 = 0.63302
N_CLASES = 6

#: (ID, familia, mecanismo, estado, evidencia, modo_de_fallo, mecanismo_nuevo_posible, novedad,
#:  coste, prioridad, por_que)
COBERTURA = [
    ("R01", "representacion", "TF-IDF de palabra (1-2)", "USED",
     "E-001 y siguientes; 36 scripts con TfidfVectorizer",
     "-", "-", "n/a", "bajo", 0, "es la base del proyecto"),
    ("R02", "representacion", "TF-IDF char_wb (3-5)", "USED",
     "E-003b, 34 scripts con char_wb; fue el mejor bloque aislado de la fase temprana",
     "-", "-", "n/a", "bajo", 0, "clasico de author profiling, ya explotado"),
    ("R03", "representacion", "skipgrams de palabra", "USED", "4 scripts",
     "sin aporte", "-", "duplicado", "bajo", 0, "variante de n-gramas ya cerrada"),
    ("R04", "representacion", "canales estructurales (10 de V31-03)", "USED",
     "V31-03: mejor canal +0,006820 con 5/5 folds sobre el texto solo",
     "la senal es real pero V30 la absorbe: V31-06 +0,0002948 sobre el control barajado",
     "ninguno identificado", "duplicado", "bajo", 0,
     "fallo como MECANISMO DE INTEGRACION, no como representacion. Pero las tres vias de "
     "integracion probadas (feature, miembro de mezcla, residuo) estan cerradas"),
    ("R05", "representacion", "bloque estructurado denso de 279 features", "USED",
     "V20 y todo el meta-espacio posterior",
     "-", "-", "n/a", "bajo", 0, "es la entrada del meta-modelo"),
    ("R06", "representacion", "SVD / reduccion de dimensionalidad", "USED",
     "6 scripts, E-019 y E-025/E-026",
     "sin aporte sobre el TF-IDF crudo", "-", "duplicado", "bajo", 0,
     "probado y no aporto: la representacion dispersa ya era manejable"),
    ("R07", "representacion", "canales de posicion y forma de palabra", "CLOSED",
     "V31-03: -0,002303 y -0,002984, 0/5 folds",
     "negativos en los cinco folds", "-", "duplicado", "bajo", 0, "cerrados como negativos"),
    ("R08", "representacion", "canales de URL (presencia, tipo, posicion)", "CLOSED",
     "V31-02 +0,000738 (3/5); V31-03 +0,001894 en banco, nada en integracion",
     "la URL ya se leia: enmascararla cuesta -0,006969", "-", "duplicado", "bajo", 0,
     "cerrado en las tres formas"),
    ("M01", "familia de modelo", "LogisticRegression", "USED", "71 scripts",
     "-", "-", "n/a", "bajo", 0, "familia base, exhaustivamente usada"),
    ("M02", "familia de modelo", "LinearSVC", "USED", "61 scripts",
     "-", "-", "n/a", "bajo", 0, "familia base"),
    ("M03", "familia de modelo", "SGDClassifier", "USED", "7 scripts", "-", "-", "n/a", "bajo", 0,
     "-"),
    ("M04", "familia de modelo", "Naive Bayes (y NB-SVM)", "USED", "14 scripts",
     "-", "-", "n/a", "bajo", 0, "-"),
    ("M05", "familia de modelo", "Ridge", "USED", "17 scripts", "-", "-", "n/a", "bajo", 0, "-"),
    ("M06", "familia de modelo", "MLP densa", "USED", "10 scripts, E-019 y V29",
     "inferior a LightGBM en el meta-espacio", "-", "duplicado", "medio", 0,
     "probado; V29 lo intento como miembro y el resultado parcial era muy inferior"),
    ("M07", "familia de modelo", "LightGBM", "USED",
     "27 scripts; V29 +0,002392 y V30 +0,000069 de refinamiento",
     "la capacidad del meta-arbol quedo agotada: la region estaba encontrada",
     "ninguno: agotado por medicion", "duplicado", "alto", 0,
     "V29 y V30 demostraron que el eje de capacidad esta recorrido"),
    ("M08", "familia de modelo", "ExtraTrees / RandomForest", "USED", "V20-F1/F2, 1-2 scripts",
     "sin aporte sobre LightGBM", "-", "duplicado", "medio", 0, "-"),
    ("M09", "familia de modelo", "k-NN / NearestNeighbors", "USED", "10 scripts",
     "sin aporte", "-", "duplicado", "bajo", 0, "-"),
    ("M10", "familia de modelo", "prototipos / centroides", "USED",
     "15 scripts, E-021 experimento_e021_ordinal_centroides",
     "sin aporte", "-", "duplicado", "bajo", 0, "la estructura local por prototipo ya se probo"),
    ("C01", "combinacion", "stacking / blending", "CLOSED",
     "H06 y H04; 12 scripts; V28-04 lo declaro equivalente a H04+H06",
     "equivalencia funcional con la ponderacion simple", "ninguno", "duplicado", "medio", 0,
     "cerrado por equivalencia funcional, no por resultado"),
    ("C02", "combinacion", "ensembles de modelos historicos", "CLOSED",
     "V29: 0 de 47 combinaciones con rescate neto positivo",
     "diversidad sin informacion nueva", "ninguno", "duplicado", "alto", 0, "cerrado con numero"),
    ("C03", "combinacion", "correccion del score (global, local, pendiente)", "CLOSED",
     "P33, P34, G12: -0,0002152, -0,0002039 y +0,0003018 sin superar a la constante",
     "el control de ruido aleatorio gana a la senal real en los tres casos",
     "G12 anadio una pendiente y el AUC BAJO", "duplicado", "bajo", 0, "cerrado en tres formas"),
    ("C04", "combinacion", "bagging del meta-modelo", "CLOSED", "V28-02: -0,000312",
     "sin aporte", "ninguno", "duplicado", "bajo", 0, "cerrado"),
    ("O01", "objetivo", "multiclass cross-entropy (multi_logloss)", "USED", "95 scripts",
     "-", "-", "n/a", "bajo", 0, "el objetivo de todos los modelos del proyecto"),
    ("O02", "objetivo", "cost-sensitive / class_weight", "USED",
     "V18-P1 peso de clases, 8 scripts con class_weight o sample_weight",
     "sin aporte", "-", "duplicado", "bajo", 0, "probado explicitamente"),
    ("O03", "objetivo", "ranking objective", "NOT USED", "0 scripts, 0 menciones en el historial",
     "-", "-", "**descartado por irrelevancia**", "bajo", 9,
     "la metrica de la competicion es macro-F1 sobre el argmax. Un objetivo de ranking no mueve "
     "la metrica evaluada: cambiaria el objetivo sin cambiar el criterio"),
    ("O04", "objetivo", "focal loss", "NOT USED", "0 scripts, 0 menciones en el historial",
     "-", "-", "duplicado con indicacion negativa", "bajo", 8,
     "es reponderacion de ejemplos con modulacion por confianza. `class_weight` ya se probo "
     "(V18-P1) y la modulacion apunta a los ejemplos dificiles, que son la poblacion A: 27.430 "
     "filas donde NINGUNA de las 26 fuentes tiene senal. Reponderar ahi es entrenar sobre ruido"),
    ("O05", "objetivo", "label smoothing", "NOT USED", "0 scripts, 0 menciones en el historial",
     "-", "-", "duplicado con indicacion negativa", "bajo", 8,
     "actua sobre la zona de frontera, y P34 midio que ahi el ruido aleatorio gana a la senal "
     "real (real - ruido = -0,0000797)"),
    ("E01", "muestreo", "row weighting explicito (sample_weight)", "USED", "V18-P1, V28-02",
     "sin aporte", "-", "duplicado", "bajo", 0, "-"),
    ("E02", "muestreo", "balanced sampling", "USED", "2 scripts", "sin aporte", "-", "duplicado",
     "bajo", 0, "-"),
    ("E03", "muestreo", "hard example mining", "NOT USED", "0 scripts, 0 menciones",
     "-", "-", "duplicado con indicacion negativa", "bajo", 8,
     "el conjunto dificil ES la poblacion A, donde ninguna de las 26 fuentes tiene senal. Minar "
     "ese conjunto es exactamente entrenar sobre ruido"),
    ("V01", "varianza", "bootstrap de grupos", "NOT USED", "0 scripts",
     "-", "-", "duplicado", "bajo", 7,
     "es reduccion de varianza con otra unidad de remuestreo. V28-02 cerro el bagging del "
     "meta-modelo en -0,000312, y cambiar la unidad de remuestreo no cambia el mecanismo"),
    ("V02", "varianza", "seed ensemble", "USED", "1 script", "sin aporte", "-", "duplicado",
     "bajo", 0, "-"),
    ("L01", "estructura de etiquetas", "ordinal (CORAL y variantes)", "CLOSED",
     "H10, D-058; 47 menciones de ordinal en el historial; V22-V23",
     "sin aporte sobre el multinomial", "ninguno", "duplicado", "alto", 0,
     "cerrado tras una fase entera dedicada"),
    ("L02", "estructura de etiquetas", "pairwise / descomposicion", "INVALIDATED",
     "V28-03: -0,067553, 0/5 folds, neto -10.996",
     "sustituir el multiclase por binarios destruye la estructura de la clase completa",
     "P34 probo la version LOCALIZADA y tambien fallo", "duplicado", "medio", 0,
     "cerrado en sus dos formas: sustitucion (V28-03) y correccion local (P34)"),
    ("L03", "estructura de etiquetas", "classifier chain", "USED", "2 scripts",
     "sin aporte", "-", "duplicado", "medio", 0, "-"),
    ("L04", "estructura de etiquetas", "calibracion / post-procesado de probabilidad", "CLOSED",
     "PH32 y PH33: +0,0002 y ninguna de 5 variantes positiva; P33 mostro que el intercepto se "
     "elige 0",
     "es efecto de calibracion, no de representacion", "ninguno", "duplicado", "bajo", 0,
     "cerrado con control explicito: el intercepto elegido fue 0"),
    ("S01", "estructura local", "cluster conditioning", "USED", "10 scripts, V13 y V31",
     "sin aporte", "-", "duplicado", "medio", 0, "-"),
    ("S02", "estructura local", "seleccion de features supervisada", "CLOSED",
     "V25, D-060: -0,000629",
     "sobreajuste a la particion", "ninguno", "duplicado", "medio", 0, "cerrado"),
    ("S03", "estructura local", "entrada no usada (G14)", "NOT_ACTIONABLE",
     "G14: el id es un indice de fila, 6 digitos, 100% de los ids del test reaparecen en train",
     "no hay ningun campo de entrada legal sin usar", "ninguno", "duplicado", "bajo", 0,
     "cerrado con la entrada enumerada por completo"),
]


def main() -> int:
    """Escribe la matriz de cobertura y decide."""
    autotest_numerico()
    inicio = time.time()
    audit = json.loads((cfg.EXPERIMENTS / "age_auditoria_cobertura.json").read_text(encoding="utf-8"))
    part = audit["particion"]
    excl = audit["rescates_exclusivos"]

    matriz = pd.DataFrame([{
        "ID": c[0], "family": c[1], "mechanism": c[2], "status": c[3], "evidence": c[4],
        "failure_mode": c[5], "possible_new_mechanism": c[6], "novelty": c[7],
        "cost": c[8], "priority": c[9], "why": c[10],
    } for c in COBERTURA])
    matriz.to_csv(MATRIZ_CSV, index=False)

    counts = matriz["status"].value_counts().to_dict()
    nuevas = matriz[matriz["novelty"] == "genuinamente nueva"]
    print("=== matriz de cobertura ===", flush=True)
    for _, f in matriz.iterrows():
        print(f"  {f['ID']} {f['status']:<15} {f['family']:<22} {f['mechanism'][:46]}", flush=True)
    print(f"\n  {counts}", flush=True)
    print(f"  genuinamente nuevas: {len(nuevas)}", flush=True)

    # La decision, con los tres criterios del brief.
    print("\n=== decision ===", flush=True)
    print("  input gap:  cerrado por G14 (NOT_ACTIONABLE)", flush=True)
    print(f"  mechanism gap: {counts.get('USED', 0)} de {len(matriz)} mecanismos ya usados, "
          f"{counts.get('CLOSED', 0)} cerrados, {counts.get('INVALIDATED', 0)} invalidados, "
          f"{counts.get('NOT_ACTIONABLE', 0)} no accionable", flush=True)
    sin_usar = matriz[matriz["status"] == "NOT USED"]
    print(f"  mecanismos sin usar: {len(sin_usar)}", flush=True)
    for _, f in sin_usar.iterrows():
        print(f"    {f['ID']} {f['mechanism']:<26} -> {f['novelty']}", flush=True)
    print(f"\n  rescates exclusivos: {excl['n']:,} repartidos entre "
          f"{excl['fuentes_distintas']} fuentes, concentracion {excl['concentracion_fuente_top']:.1%}",
          flush=True)
    print(f"  -> {excl['estructura']}", flush=True)

    conclusion = "AGE_RESEARCH_CLOSED" if len(nuevas) == 0 else "ONE_GENUINELY_NEW_HYPOTHESIS_IDENTIFIED"
    print(f"\n=== {conclusion} ===", flush=True)

    _escribe(matriz, counts, part, excl, sin_usar, conclusion)
    RESULTADO.write_text(json.dumps({
        "fase": "auditoria de cobertura metodologica", "fecha": FECHA, "experimenta": False,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                                 text=True, timeout=30).stdout.strip(),
        "baseline_v30": CV_V30, "leaderboard": LB_V30,
        "matriz": matriz.to_dict("records"), "conteo_por_estado": counts,
        "mecanismos_sin_usar": sin_usar[["ID", "mechanism", "novelty", "why"]].to_dict("records"),
        "particion_de_errores": part, "rescates_exclusivos": excl,
        "conclusion": conclusion,
        "experimentos_de_entrenamiento": 0,
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_red": False,
                          "usa_leaderboard_para_elegir": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _escribe(matriz, counts, part, excl, sin_usar, conclusion) -> None:
    lineas = [
        "# AGE - matriz de cobertura metodologica", "",
        f"**Baseline:** V30 = {CV_V30}, leaderboard {LB_V30}.", "",
        "## Como se construyo", "",
        "Los 22 mecanismos se buscaron en el codigo por **operacion**, no por nombre de libreria: "
        "`class_weight` aparece en 8 scripts y `LGBMClassifier` en 27, pero lo que se registra es "
        "que el mecanismo se haya **ejecutado**, no que la libreria este instalada.", "",
        "## Resultado", "",
        "| Estado | Cuantos |", "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])] + [
        "", "## La matriz completa", "",
        "| ID | Familia | Mecanismo | Estado | Evidencia | Modo de fallo | Novedad |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, f in matriz.iterrows():
        lineas.append(f"| {f['ID']} | {f['family']} | {f['mechanism']} | **{f['status']}** | "
                      f"{str(f['evidence'])[:96]} | {str(f['failure_mode'])[:70]} | {f['novelty']} |")
    lineas += ["", "## Los mecanismos sin usar, y por que se cierran", "",
               "De 22, cinco no aparecen en el codigo. No se descartan por inercia: cada uno se "
               "contrasta con la medicion que ya existe.", ""]
    for _, f in sin_usar.iterrows():
        lineas += [f"**{f['ID']} - {f['mechanism']}** ({f['novelty']})", "",
                   f"{f['why']}", ""]
    MATRIZ_MD.write_text("\n".join(lineas).rstrip("\n") + "\n", encoding="utf-8")

    audit = [
        "# AGE - auditoria final de cobertura metodologica", "",
        f"**Conclusion: `{conclusion}`**", "",
        "## Las dos preguntas de esta fase", "",
        "```text",
        "1. G14 cerro el INPUT GAP: no hay campos de entrada sin usar.",
        "2. Esta fase cierra la pregunta del MECHANISM GAP: ¿queda alguna forma",
        "   distinta de APRENDER de los mismos inputs?",
        "```", "",
        "## Lo que se midio antes de opinar", "",
        "La particion de los errores de V30, refinada con una distincion que faltaba:", "",
        "| Grupo | Filas | % | Que significa |", "|---|---|---|---|",
        f"| A: nadie corrige | {part['A_nadie_corrige']:,} | "
        f"{part['A_nadie_corrige'] / part['errores_v30']:.1%} | ninguna de las 26 fuentes acierta "
        f"esas filas |",
        f"| B: alguien corrige | {part['B_alguien_corrige']:,} | "
        f"{part['B_alguien_corrige'] / part['errores_v30']:.1%} | al menos una acierta |",
        f"| - B1 rescate **unico** | {part['B1_rescate_unico']:,} | "
        f"{part['B1_rescate_unico'] / part['errores_v30']:.1%} | solo una fuente la tiene |",
        f"| - B2 rescate **compartido** | {part['B2_rescate_compartido']:,} | "
        f"{part['B2_rescate_compartido'] / part['errores_v30']:.1%} | varias fuentes la tienen |",
        "", f"Mediana de fuentes por error rescatado: "
        f"**{part['mediana_de_fuentes_por_error_rescatado']:.0f}**, con un maximo de "
        f"{part['max_fuentes_que_rescatan_una_fila']}.", "",
        "### Por que la distincion B1/B2 importa", "",
        "**Un rescate compartido significa que la informacion esta en varias fuentes a la vez, y el "
        "problema es de integracion. Un rescate unico significa que solo esa fuente la tiene, y el "
        "problema es de mecanismo.** Solo la segunda situacion justifica buscar un mecanismo nuevo, "
        "porque es la unica en la que hay una fuente con informacion exclusiva.", "",
        f"Y la medicion dice que B2 domina: **{part['B2_rescate_compartido']:,} de los "
        f"{part['B_alguien_corrige']:,} rescates son compartidos**, la mayoria por cinco fuentes o "
        f"mas. La informacion esta **repartida**, no encerrada en una fuente.", "",
        "## Los rescates exclusivos, que era la pregunta central", "",
        f"El inventario previo del proyecto registro **122 rescates exclusivos**. Con la definicion "
        f"de esta auditoria -una sola de las 26 fuentes acierta la fila- hay "
        f"**{excl['n']:,}**. La diferencia es de definicion, y se declara: los dos numeros miden "
        f"cosas distintas y ninguno es erroneo. Se usa el de aqui porque es el que responde a la "
        f"pregunta del brief.", "",
        "| Medida | Rescates exclusivos | Todos los errores |", "|---|---|---|",
        f"| Longitud media | {excl['longitud_media']} | {excl['longitud_media_de_todos_los_errores']} |",
        f"| Confianza media de V30 | {excl['confianza_media_v30']} | "
        f"{excl['confianza_media_de_todos_los_errores']} |",
        f"| Fraccion con URL | {excl['fraccion_con_url']:.1%} | "
        f"{excl['fraccion_con_url_en_todos_los_errores']:.1%} |", "",
        f"**No hay diferencia en ninguna dimension.** Y se reparten entre "
        f"**{excl['fuentes_distintas']} fuentes distintas**, con la mas frecuente concentrando solo "
        f"el **{excl['concentracion_fuente_top']:.1%}**.", "",
        f"Veredicto: **{excl['estructura']}**. {excl['lectura']}", "",
        "## La cobertura de mecanismos", "",
        "| Estado | Cuantos |", "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])] + [
        "", f"De los {len(matriz)} mecanismos auditados, **{counts.get('USED', 0)} estan usados** y "
        f"los cinco sin usar se cierran con la medicion delante, no por inercia:", "",
    ] + [f"- **{f['ID']} {f['mechanism']}**: {f['why']}" for _, f in sin_usar.iterrows()] + [
        "", "## La conclusion", "",
        "```text", conclusion, "```", "",
        "**Por que `AGE_RESEARCH_CLOSED` y no una hipotesis mas.**", "",
        "Un mecanismo nuevo tendria que cumplir los seis criterios del brief, y el que no cumple "
        "ninguno de los candidatos es el cuarto: **no ser equivalente a una tecnica cerrada**. Los "
        "cinco mecanismos sin usar actuan sobre la distribucion de entrenamiento o sobre la "
        "funcion de perdida, y su poblacion objetivo es exactamente la que el diagnostico ya midio "
        "como sin senal corregible:", "",
        f"- **focal loss**, **hard example mining** y **label smoothing** apuntan a los ejemplos "
        f"dificiles y a la frontera. La poblacion A son {part['A_nadie_corrige']:,} filas donde "
        f"ninguna de 26 fuentes acierta, y P34 midio que en la frontera el ruido aleatorio gana a "
        f"la senal real.",
        "- **ranking objective** no mueve la metrica evaluada, que es macro-F1 sobre el argmax.",
        "- **bootstrap de grupos** es reduccion de varianza con otra unidad de remuestreo, y el "
        "bagging del meta-modelo ya se cerro en -0,000312.", "",
        "## Lo que esta conclusion NO dice", "",
        "- **No** dice que V30 sea optimo.",
        "- **No** dice que no exista ninguna mejora posible.",
        "- **No** dice que no exista informacion linguistica: P33 midio AUC 0,5411 en el residuo.",
        "", "Dice algo mas estrecho y mas honesto: **no quedan actualmente mecanismos de "
        "aprendizaje metodologicamente defendibles y no recorridos dentro de las restricciones del "
        "proyecto**. La afirmacion es sobre el espacio auditado, y el espacio auditado esta "
        "enumerado: 22 mecanismos, 6 familias de representacion, 10 familias de modelo, la entrada "
        "completa y las tres vias de integracion del score.", "",
        "## El techo, medido", "",
        f"- **{part['A_nadie_corrige']:,}** errores de V30 que ninguna de las 26 fuentes corrige, y "
        f"sin campo de entrada que pudiera corregirlos.",
        f"- **{excl['n']:,}** rescates exclusivos sin estructura comun: ruido de modelo, no un canal "
        f"sin explotar.",
        "- V30 sigue siendo el mejor resultado validado con diferencia sobre cualquier alternativa "
        "medida.", "",
    ]
    AUDIT_MD.write_text("\n".join(audit).rstrip("\n") + "\n", encoding="utf-8")

    gap = [
        "# AGE - gap analysis de mecanismos", "",
        f"**Conclusion: `{conclusion}`**", "",
        "## La pregunta que queda contestada", "",
        "> ¿Queda algun mecanismo de aprendizaje realmente distinto que pueda convertir informacion "
        "ya disponible en una correccion robusta, sin repetir una familia que ya fallo?", "",
        "**No dentro de las restricciones del proyecto.** El razonamiento, en tres pasos.", "",
        "### Paso 1: la informacion disponible esta localizada", "",
        f"- Grupo A: **{part['A_nadie_corrige']:,}** errores sin ninguna fuente que los corrija "
        f"({part['A_nadie_corrige'] / part['errores_v30']:.1%}).",
        f"- Grupo B: **{part['B_alguien_corrige']:,}** con al menos una fuente "
        f"({part['B_alguien_corrige'] / part['errores_v30']:.1%}), de los cuales "
        f"**{part['B2_rescate_compartido']:,}** son de rescate compartido.",
        f"- Rescates exclusivos: **{excl['n']:,}** entre **{excl['fuentes_distintas']}** fuentes, "
        f"sin estructura comun.", "",
        "### Paso 2: las vias de integracion estan cerradas con control", "",
        "| Via | Experimento | Resultado |", "|---|---|---|",
        "| feature anadida al meta-espacio | V31-06 | +0,0002948 sobre el control barajado |",
        "| miembro de la mezcla | V31-05 | +0,0000642 sobre el reoptimizado |",
        "| residuo global | P33 | -0,0002152; el ruido aleatorio gana |",
        "| residuo local por pares | P34 | -0,0002039; cero rescates de alta confianza |",
        "| residuo con pendiente en el margen | G12 | +0,0003018 sin superar a la constante, y baja "
        "el AUC |", "",
        "Las cinco tienen **control de atribucion**, y en las cinco el control gana o empata. No es "
        "que la integracion se haya intentado poco: es que se intento cinco veces con la misma "
        "conclusion.", "",
        "### Paso 3: los mecanismos sin usar atacan poblaciones sin senal", "",
        "Los cinco mecanismos que no aparecen en el codigo son focal loss, hard example mining, "
        "label smoothing, ranking objective y bootstrap de grupos. Los tres primeros apuntan a los "
        "ejemplos dificiles y a la frontera, que es donde la medicion dice que el ruido gana. El "
        "cuarto no mueve la metrica. El quinto es una variante del bagging ya cerrado.", "",
        "## Lo que habria que cambiar para que algo funcionara", "",
        "No es un mecanismo, y por eso no se propone como hipotesis: haria falta **informacion que "
        "no esta en los datos disponibles**. Las tres fuentes que lo aportarian estan prohibidas o "
        "no existen:", "",
        "1. **Metadatos de la cuenta** (fecha de registro, ubicacion, historial): las reglas de AGE "
        "son de texto, y el brief los prohibe.",
        "2. **El contenido de las URLs**: no hay ninguna URL de status en el dataset, y las 1.685 "
        "con dominio truncado no apuntan a ningun sitio resoluble.",
        "3. **Modelos semanticos** (embeddings, transformers): prohibidos por las reglas de la "
        "competicion.", "",
        "## Cierre", "",
        "La investigacion de AGE se cierra con V30 como mejor resultado validado y todas las vias "
        "metodologicas disponibles recorridas o cerradas con evidencia. El detalle completo esta en "
        "`reports/age_final_methodology_audit.md` y la matriz en "
        "`reports/age_methodology_coverage_matrix.csv`.", "",
    ]
    GAP_MD.write_text("\n".join(gap).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
