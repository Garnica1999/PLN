#!/usr/bin/env python
"""
P94-01: registro de hipotesis y propuestas auditadas (P2/P3 de P94).

Escribe reports/p94_hypothesis_registry.csv con las propuestas auditadas
(una fila por mecanismo), su clasificacion P2 y, para las tres candidatas
preseleccionadas (P3), el juego completo de campos de hipotesis.

Uso:
    /home/carlos/venv/bin/python experiments/p94_01_registro_hipotesis.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "reports" / "p94_hypothesis_registry.csv"

CLASES_VALIDAS = {"GENUINELY_NEW", "ALREADY_TESTED", "DUPLICATE", "NOT_FEASIBLE",
                  "INSUFFICIENT_EVIDENCE"}
NO_APLICA = "no aplica (cerrado; ver classifications y reason)"

CABECERA = ["hypothesis_id", "mechanism", "proposal_source", "classification",
            "p3_shortlisted", "prior_evidence", "historical_experiments_compared",
            "novelty_argument", "expected_failure_mode", "minimum_experiment",
            "required_artifacts", "validation_protocol", "primary_metric",
            "secondary_metrics", "compute_cost_class", "risk_of_overfitting",
            "adoption_gate", "recommendation", "reason"]

PROTO = ("StratifiedGroupKFold con grupos near-duplicate UNION pseudo-author; "
         "metrica primaria = definicion canonica de corrected_cv; sin test, holdout, "
         "holdout2 ni leaderboard; particion de evaluacion nunca usada para inventar "
         "o ajustar la hipotesis")

FILAS = [
    ["H-P94-01", "perdida sustituta de F-beta / macro-F1 para modelos lineales",
     "R08 Eban 2017; R09 Lee 2021; R06 Dembczynski 2013", "ALREADY_TESTED", "true",
     "V7-01 probo la F1-suave (sustituto literal del objetivo) con control BCE exacto: "
     "-0.0033 vs control, 3/5 pliegues; focal -0.0024; la mas directa de las dos fue la PEOR",
     "V7-01; V12-01..V12-05; P60-01/02/04/06/07/08; P59-10; D-053; D-092; D-096",
     "La implementacion con restricciones (TFCO) es nueva como codigo, pero el mecanismo "
     "(descenso por gradiente contra un sustituto de la metrica) ya fue evaluado; P2 "
     "prohibe tratar una implementacion nueva como mecanismo nuevo",
     "Con 1186 filas de ajuste el descenso se acopla a la composicion de la muestra "
     "(exactamente el patron de OPT-01); ademas el freno de transferencia exige que una "
     "mejora local de hasta +0.02 no se lee como externa",
     NO_APLICA, NO_APLICA, PROTO,
     "corrected_cv (solo si algun dia se reabriera con un acto nuevo)",
     "F1 por etiqueta; gap train-val; pliegues positivos; neto de celdas", "CPU (min)",
     "alto: el patron ya medido sobreajusta la composicion del OOF",
     "dOOF > +0.0376 con neto positivo en la misma configuracion + freno de transferencia",
     "NO", "ALREADY_TESTED: la formulacion literal ya se midio y perdio; no supera el criterio de novedad P2"],
    ["H-P94-02", "marcado de alcance de negacion en la representacion sparse",
     "R12 Abderrouaf 2019", "ALREADY_TESTED", "true",
     "E2-08-NEG: patrones NEG_* de spaCy simbolico (sin vectores): 0.5267 sola; "
     "E2-08-NEG-TXT: -0.0115 vs V03 (0/5 pliegues); la linea sintaxis/cita/negacion "
     "se cerro sin mejora",
     "E2-03-POS; E2-04-MORPH/SYN3; E2-07-CITA; E2-08-NEG/NEG-TXT; V37-01 (BM25); P62-01..05",
     "El marcado de alcance por token (sufijo en tokens negados) seria una formulacion "
     "distinta de la misma familia (evidencia negada como features del LR sparse); "
     "la familia ya tiene medicion negativa y P2 prohibe el cambio de nombre/implementacion",
     "Misma familia de features que E2-08; el corpus es ASCII normalizado (D-010), lo que "
     "limita los patrones superficiales de negacion en espanol",
     NO_APLICA, NO_APLICA, PROTO,
     "corrected_cv (solo si se reabriera con un acto nuevo)",
     "F1 por etiqueta; pliegues positivos", "CPU (min)",
     "medio: familia ya medida; el riesgo real es cerrar en ruido",
     "dOOF > +0.0376 con neto positivo + freno de transferencia",
     "NO", "ALREADY_TESTED: la familia negacion/cita/sintaxis esta cerrada por medicion (E2-08)"],
    ["H-P94-03", "clasificador instance-based (kNN) sobre la representacion de V03",
     "Q5 (frontera del inventario; sin referencia especifica a favor)", "INSUFFICIENT_EVIDENCE",
     "true",
     "Nunca se ejecuto un kNN como clasificador. Evidencia adyacente: el RBF exacto "
     "memoriza (mediana del vecino 0.6974, V26-01); la regularizacion por vecindad quedo "
     "MARGINAL (P59-10); los prototipos/transporte se cerraron (P60-06); fuzzy matching "
     "es INFORMATIONAL_ONLY (V29)",
     "V26-01; V27-01; P59-10; P60-06; V29-03/V29-04; D-190 (eje geometria cerrado)",
     "Ninguna referencia revisada situa a kNN por encima de un lineal regularizado "
     "ajustado en texto corto sparse; la novedad seria 'celda sin medir' de un "
     "mecanismo adyacente a familias ya cerradas, no un mecanismo nuevo",
     "kNN sobre TF-IDF con near-duplicados produce fuga por grupos y memorizacion; "
     "esperado muy por debajo de V03 (referencia: RBF memorizador y su espectro en 0.7621)",
     "Si algun dia se autorizara: kNN (k=5..25, coseno, agrupado, vectorizador por "
     "pliegue) contra V03 con el protocolo canonico; coste CPU de minutos",
     "Corpus de desarrollo y particion canonica (ya existentes)",
     "StratifiedGroupKFold con grupos fusionados; sin recursos reservados",
     "corrected_cv", "F1 por etiqueta; pliegues positivos", "CPU (min)",
     "alto: la geometria de la representacion es memorizadora para vecindad",
     "dOOF > +0.0376 con neto positivo + freno de transferencia",
     "NO", "INSUFFICIENT_EVIDENCE sin evidencia a favor: se registra como celda no medida pero "
     "no supera el criterio de novedad; recomendable solo como diagnostico de bajo valor"],
    ["M-P94-04", "regla de decision plug-in / umbrales optimos para F1 macro",
     "R05 Lipton 2014; R10 Berger 2020; R06 Dembczynski 2013", "ALREADY_TESTED", "false",
     "TH-01 (umbrales 0.44/0.51/0.46) ya forma parte de V03, adoptada por contraste "
     "pareado en LB (+0.0031); las re-optimizaciones posteriores se cerraron (F05); los "
     "errores de V03 no son de decision (D-142)",
     "TH-01; S02; S03; V7-P0 (D-055); V10-04/V10-09; P55 (D-126/127); V56-01; P76 (D-259)",
     "Ninguna: es la capa ya adoptada y ya re-auditada", "no aplica",
     NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED: lado de decision ya medido, adoptado y re-cerrado"],
    ["M-P94-05", "F-measure via clasificacion sensible a costes",
     "R07 Parambath 2014", "ALREADY_TESTED", "false",
     "La re-ponderacion por costes es la familia mas medida del proyecto: class_weight "
     "en V03; downweighting V4-B; ruido V30; ROS cerrado por equivalencia (P84-03); "
     "candidato cost-sensitive PROMISING_UNCONFIRMED con LB 0.77522 y nunca adoptado",
     "V4-B01..B05; V30-00..02; P84-03; D-191..D-217; FIN-01-C-REF-44; P8-REFIT-C",
     "Ninguna: mismo mecanismo de re-ponderacion", "no aplica",
     NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED: mecanismo equivalente a familias ya cerradas (F14)"],
    ["M-P94-06", "NB-SVM (interpolacion NB x SVM) sobre sparse",
     "R02 Wang y Manning 2012", "ALREADY_TESTED", "false",
     "E2-15: NBSVM -0.0160 (2/5 pliegues); NBSVM-SVC -0.0343; NB-WORD -0.0234",
     "E2-15-NB-WORD/NB-CHAR/NBSVM/NBSVM-SVC",
     "Ninguna: mismo clasico ya medido", "no aplica",
     NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED: medido y por debajo de V03"],
    ["M-P94-07", "ponderacion supervisada de terminos (chi2/IG/TF-IGM)",
     "R03 Lan 2009; R04 Chen 2016", "ALREADY_TESTED", "false",
     "v4-C: log-odds/chi2/MI/BM25/binario degradan; V37-01 BM25 cerrado",
     "v4-C; V37-01; V41-01 (auditoria)",
     "Ninguna: otra formula de la misma ponderacion", "no aplica",
     NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED (F08)"],
    ["M-P94-08", "estructura de salida multietiqueta (powerset/RAkEL/cadenas)",
     "R17 Bogatinovski 2022; R06 Dembczynski 2013", "ALREADY_TESTED", "false",
     "Cadenas +0.002 (ruido, D-017); estructura conjunta CLOSED (V45-01/D-116); techo "
     "oraculo +0.046271 no capturable; especialistas = reentrenado bit a bit (D-165)",
     "E2-01; E2-02; E2-21; V6-01/V6-02; V12-02/V12-03; V13-01/V13-02; V14-01; V28-00; V45-01",
     "RAkEL seria un ensemble de powersets: ensemble cerrado + estructura cerrada",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED: estructura real pero no explotable, con techo medido (F13)"],
    ["M-P94-09", "aumento de datos textual (EDA y similares)",
     "R16 Wei y Zou 2019", "ALREADY_TESTED", "false",
     "V6-08: el aumento (espaciado/elongacion/puntuacion) empeora; V47: expansion y "
     "augmentation cerradas; D-118: no existe expansion legal",
     "V6-08; V47-00/V47-01; PRE-00..09; V41-01/V42-00",
     "Las operaciones de EDA que preservan etiqueta estan cubiertas por V6-08; la "
     "sustitucion por sinonimos exigiria recursos externos (prohibido)",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED (F15)"],
    ["M-P94-10", "aproximaciones de kernel (RFF/Nystrom) y kernel de espectro",
     "V26-01 (auditoria interna)", "ALREADY_TESTED", "false",
     "El RBF exacto memoriza (100% de vecinos > 0.60); RFF/Nystrom aproximan ese kernel; "
     "el espectro se midio como SVC (V27-01 0.7621) y quedo CHALLENGER sin candidatura",
     "V26-01; V27-01; V4-G01-G03; P8-REFIT-C",
     "Ninguna: aproximar un kernel ya medido no cambia el mecanismo",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED: reserva declarada del propio proyecto; eje cerrado (D-190)"],
    ["M-P94-11", "logistica bayesiana dispersa con hiper-prior (ARD)",
     "P83 (referencia interna; ARD clasico)", "ALREADY_TESTED", "false",
     "P83 ejecutado: M1 (variacional con hiper-prior Gamma) y M2 (Bayes multinomial); "
     "pregunta de calibracion NEGATIVA; complementariedad medida con recuentos; nunca "
     "se afirmo que supere a V03",
     "P83; E2-15-NB-WORD",
     "Ninguna: modelo bayesiano ya ejecutado en el proyecto",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED (F21)"],
    ["M-P94-12", "optimizacion robusta por grupos (DRO) para peor grupo",
     "Literatura de robustez distribuida; sin referencia seleccionada (mecanismo de re-ponderacion)",
     "ALREADY_TESTED", "false",
     "La ponderacion por grupo se midio y no se adopto (V6-A, D-046); el downweighting "
     "por costes/cercania esta cerrado (V4-B, D-027); DRO cambia la regla de pesos, no "
     "el mecanismo (re-ponderacion de filas)",
     "V6-03 (Track A); V4-B; D-046/D-027; P84-03",
     "Ninguna: mismo mecanismo, otra regla",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "ALREADY_TESTED por equivalencia de mecanismo (P2)"],
    ["M-P94-13", "bagging con particion de calibracion separada",
     "Reapertura condicionada de V6-06 (limitacion de protocolo declarada)", "DUPLICATE",
     "false",
     "V6-06: bagging -0.0105 (0/5) con la limitacion de umbrales congelados; D-067: la "
     "mejora local del promediado es un artefacto; P94 excluye replicar ensembles",
     "V6-06; V8-01/V8-06..08; ENS-01..04; D-067",
     "Ninguna: es un ensemble/bagging, familia cerrada y excluida por el encargo",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "DUPLICATE: ensamblado cerrado + prohibicion explicita de replicarlo"],
    ["M-P94-14", "semi-supervisado / self-training sobre texto no etiquetado",
     "Frontera historica (V34-01 la listo sin ejecutar)", "NOT_FEASIBLE", "false",
     "V14-01: la linea partially-labeled se cerro; no existe fuente de texto no "
     "etiquetado autorizada (test intocable; sin corpus externos; pseudo-labeling prohibido)",
     "V14-01; V34-01; D-118; reglas P94",
     "No evaluable bajo las restricciones vigentes", "no aplica",
     NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA,
     "NO", "NOT_FEASIBLE: sin datos no etiquetados autorizados; pseudo-labeling prohibido"],
    ["M-P94-15", "transformers preentrenados en espanol (ALBETO, BETO)",
     "R14 HOMO-MEX 2023; panorama SOTA de odio en espanol", "NOT_FEASIBLE", "false",
     "El estado del arte actual en odio espanol es transformer-based; en este proyecto "
     "la prohibicion de preentrenados se ratifico (V8-00/D-057) y el DL desde cero "
     "quedo por debajo de V03 (F12)",
     "V8-00; F12 (DL-01..P82)",
     "No evaluable: restriccion explicita de la competencia/proyecto",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "NOT_FEASIBLE: prohibido por reglas (sin transformers ni preentrenados)"],
    ["M-P94-16", "arboles de etiquetas probabilisticos (PLT) y variantes extreme",
     "Literatura de extreme multi-label (estructura de salida)", "NOT_FEASIBLE", "false",
     "Con 3 etiquetas el mecanismo degenera; la familia de estructura esta cerrada con "
     "techo medido (F13)",
     "E2-01; V45-01; D-090..D-104",
     "No aplicable al regimen de 3 etiquetas; mecanismo de estructura cerrado",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "NOT_FEASIBLE: degenerado con 3 etiquetas + familia cerrada"],
    ["M-P94-17", "reapertura del candidato cost-sensitive downweight",
     "F14 (estado interno)", "DUPLICATE", "false",
     "Candidato PROMISING_UNCONFIRMED/PENDING; sonda externa LB 0.77522; causa faltante "
     "identificada (seleccion del baseline, D-216); holdout C rechazado",
     "D-191..D-217; FIN-01-C-REF-44; P8-REFIT-C",
     "Ninguna: no es un mecanismo nuevo; es un candidato existente congelado",
     "no aplica", NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA, NO_APLICA,
     NO_APLICA, NO_APLICA,
     "NO", "DUPLICATE: candidato existente, no nuevo; reabrir exigiria un acto de gobernanza"],
]


def main() -> None:
    """Escribe y valida el registro de hipotesis."""
    fallos: list[str] = []
    for fila in FILAS:
        if len(fila) != len(CABECERA):
            fallos.append(f"{fila[0]}: {len(fila)} campos, esperados {len(CABECERA)}")
        if fila[3] not in CLASES_VALIDAS:
            fallos.append(f"{fila[0]}: clasificacion invalida {fila[3]}")
        if fila[4] not in ("true", "false"):
            fallos.append(f"{fila[0]}: p3_shortlisted invalido")
        if fila[17] not in ("SI", "NO"):
            fallos.append(f"{fila[0]}: recommendation invalida")
    ids = [f[0] for f in FILAS]
    if len(ids) != len(set(ids)):
        fallos.append("ids duplicados")
    cortas = [f[0] for f in FILAS if f[4] == "true"]
    if len(cortas) > 3:
        fallos.append("mas de tres candidatas P3")

    if not fallos:
        with open(SALIDA, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(CABECERA)
            w.writerows(FILAS)
        with open(SALIDA, encoding="utf-8", newline="") as fh:
            leidas = list(csv.reader(fh))
        if len(leidas) != len(FILAS) + 1:
            fallos.append("estructura CSV invalida tras escribir")

    por_clase: dict[str, int] = {}
    for f in FILAS:
        por_clase[f[3]] = por_clase.get(f[3], 0) + 1

    print("P94-01 registro de hipotesis: ejecutado")
    print(f"  propuestas auditadas: {len(FILAS)} | clasificaciones: {por_clase}")
    print(f"  candidatas P3 (shortlist): {cortas}")
    recom = [f[0] for f in FILAS if f[17] == "SI"]
    print(f"  recomendadas: {recom if recom else 'NINGUNA'}")
    if fallos:
        for f in fallos:
            print("  -", f)
        sys.exit(1)
    print("  reports/p94_hypothesis_registry.csv escrito y validado")


if __name__ == "__main__":
    main()
