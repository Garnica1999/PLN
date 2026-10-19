#!/usr/bin/env python
"""
P95-00: opciones de gobernanza (P2) y fuentes nuevas de evidencia (P4).

Escribe, con validacion estricta y ASCII puro:
  reports/p95_governance_options.csv
  reports/p95_new_evidence_options.csv

Sin entrenamiento, sin predicciones, sin accesos a particiones reservadas.

Uso:
    /home/carlos/venv/bin/python experiments/p95_00_opciones.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAL_GOV = RAIZ / "reports" / "p95_governance_options.csv"
SAL_EVI = RAIZ / "reports" / "p95_new_evidence_options.csv"

CLASES_GOV = {"currently_allowed", "allowed_with_explicit_approval", "ambiguous",
              "prohibited"}

CAB_GOV = ["option_id", "class", "current_status", "governance_constraint",
           "source_of_constraint", "what_authorization_is_needed",
           "what_methodology_it_would_unlock", "expected_information_gain",
           "likely_compute_cost", "risk_of_leakage", "risk_of_overfitting",
           "historical_overlap", "recommendation"]

FILAS_GOV = [
    ["G01_propagacion_near_duplicados", "allowed_with_explicit_approval",
     "QUARANTINED (D-288); techo medido nulo (V4-F04 y sonda v18: 0 celdas)",
     "Requiere autorizacion del organizador para propagar etiquetas entre near-duplicados; las reglas generales no lo mencionan",
     "D-288; reglas verbatim docs/competition_rules.md (silencio sobre este punto)",
     "Autorizacion del organizador (profesor del curso) canalizada por el propietario",
     "Limpieza/propagacion de etiquetas dentro de grupos near-dup de train y re-medicion del ruido por duplicacion",
     "MUY BAJO: el techo ya fue medido como nulo (0 celdas) en V4-F04 y la sonda v18",
     "CPU-min (analisis) mas coste humano de correspondencia",
     "MEDIO-ALTO si la propagacion rozara ejemplos con near-dup en test (10 textos de test tienen near-dup en train, FINAL_REPORT seccion 23)",
     "BAJO (no hay modelo nuevo)", "V4-F01-F04; sonda v18; D-288",
     "NO solicitar: valor medido ~0; conservar la puerta documental sin accion"],
    ["G02_correccion_de_las_14_etiquetas", "ambiguous",
     "Las 14 candidatas quedan PRESERVED_NOT_CONFIRMED (P93-R5); el train no se toca",
     "Las reglas de la competencia son silenciosas sobre corregir etiquetas de entrenamiento (D-288); el marco de integridad exige originales inmutables",
     "D-288; D-002; P93-R5; P94 (no usar las candidatas como justificacion)",
     "Decision explicita del propietario; ademas evidencia humana INDEPENDIENTE previa (P95-P3/N1)",
     "Copia versionada del train con las celdas corregidas y experimento anidado/grupal comparativo (V03 vs variante)",
     "BAJO para el score (14 celdas de ~5193; techo historico ~+0.009 bajo la banda de ruido 0.017); MEDIO documental",
     "CPU-min para el experimento; coste humano previo de revision ciega",
     "MEDIO: exigiria particion de evaluacion no usada para definir las correcciones y diseno agrupado",
     "ALTO si se evalua sobre las mismas celdas que definen la correccion (circularidad)",
     "P87-00 (LOW_EXPECTED_VALUE); P91-P93; P93-R5",
     "NO modificar el train; sin accion hasta evidencia humana independiente y decision de gobernanza"],
    ["G03_uso_de_holdout2", "allowed_with_explicit_approval",
     "FROZEN con 3 evaluaciones disponibles; 0 gastadas (V10-08)",
     "Presupuesto explicito de evaluaciones; solo para finalistas; prohibido en fases de investigacion como esta",
     "D-069/D-074/V10-08 (gobernanza interna); reglas verbatim (el test es intocable)",
     "Solo tendria sentido con un candidato futuro que pase el gate interno; hoy no existe",
     "Validacion final externa de un candidato futuro antes de una submission",
     "NULO hoy (no hay candidato al que aplicarlo)",
     "Minimo (prediccion sobre 247 filas) cuando exista candidato",
     "BAJO si se gasta solo como instrumento final",
     "ALTO si se usara para seleccion iterativa (gastar mas de una vez)",
     "V10-01; V33-01 (uso puntual en la fase de replica)",
     "CONSERVAR sin gastar; no es una fuente de mejora por si mismo"],
    ["G04_modelos_preentrenados_transformers", "prohibited",
     "Prohibido (regla adicional del profesor); ratificado internamente (D-057)",
     "No se pueden usar transformers ni modelos LLM (BERT, RoBERTa); todo desde cero",
     "docs/competition_rules.md, seccion Restriccion adicional (texto verbatim del organizador); D-057",
     "Cambio de regla del profesor/organizador: fuera del control del proyecto",
     "Toda la familia transformer, hoy estado del arte en odio en espanol (R14 de P94)",
     "ALTO si las reglas cambiaran (no ocurre en el marco vigente)",
     "GPU-horas", "N/A", "MEDIO (ajuste fino con 1731 filas)",
     "D-057; V8-00; ninguna ejecucion (prohibida)",
     "CLOSED bajo el marco vigente; no es una decision del proyecto"],
    ["G05_corpora_externos_no_etiquetados", "allowed_with_explicit_approval",
     "Permitidos con procedencia/licencia y autorizacion escrita (caso Twitter ejecutado); la familia de embeddings quedo cerrada por medicion y directiva",
     "Exigencia de auditoria de procedencia y de mantener variante autocontenida; directiva de no reabrir la familia de embeddings",
     "docs/p73_provenance_autorizacion_corpus.md; D-224..D-227; D-290; P85-01",
     "Nueva autorizacion escrita y auditoria de procedencia para cualquier corpus nuevo",
     "Reentrenar representaciones no supervisadas con otro dominio",
     "MUY BAJO: la palanca de corpus quedo agotada (D-033/034/041/043) y el SGNS corregido no supero al control general (P85-01)",
     "CPU-horas a GPU", "BAJO (corpus sin etiquetas, con auditoria de solapamiento)", "N/A",
     "E-CORP; EBD; P69-P85",
     "NO reabrir: familia cerrada con medicion mas directiva vigente"],
    ["G06_particion_de_calibracion_interna", "currently_allowed",
     "No existe; el cierre de V6-06 condiciono la reapertura del bagging a disponer de una particion de calibracion separada",
     "Ninguna externa: es metodologia interna; D-016 fija el protocolo canonico y P94 excluye replicar ensembles",
     "V6-06; D-016; P94 (prohibicion de replicar ensembles/bagging)",
     "Ninguna formal; decision tecnica del propietario",
     "Re-medicion limpia de bagging/ensembles con umbrales anidados",
     "BAJO: la familia esta cerrada y el encargo vigente excluye replicarla",
     "CPU-min", "BAJO con diseno anidado", "MEDIO (el patron de busqueda por OOF ya fallo una vez)",
     "V6-06; F03; D-067",
     "NO ejecutar; solo tendria sentido dentro de una reapertura con hipotesis genuina"],
    ["G07_sondas_de_leaderboard_futuras", "allowed_with_explicit_approval",
     "Solo bajo pre-registro y gasto minimo; el sondeo quedo cerrado en D-014",
     "Maximo 10 envios/dia y solo cuenta la mejor puntuacion; el proyecto se autoimpuso pre-registro y no sobreajustar",
     "reglas verbatim docs/competition_rules.md; D-012/D-013/D-014",
     "Autorizacion explicita del propietario por sonda; solo con candidato que pase el gate interno",
     "Observacion externa de transferencia de un candidato futuro",
     "NULO hoy (sin candidato); y una sonda no crea senal, solo la observa",
     "Minimo (1 envio; submission ya generada)", "N/A",
     "ALTO si se usara para seleccionar (sobreajuste al leaderboard publico)",
     "23 sondas historicas; v02/v03 pareado; R1/R2/OPT/S04-S08/FIN-08/09/10",
     "CONSERVAR la gobernanza tal cual; sin accion"],
]

CAB_EVI = ["option_id", "new_evidence", "how_it_differs_from_existing_evidence",
           "what_uncertainty_it_resolves", "required_authorization", "minimum_cost",
           "possible_negative_result", "decision_that_would_follow",
           "expected_value_for_score", "recommendation"]

FILAS_EVI = [
    ["N0_sintesis", "SINTESIS: ninguna fuente nueva de evidencia con valor esperado suficiente para mejorar el score",
     "n/a (registro del veredicto P4)",
     "n/a", "n/a", "0",
     "n/a",
     "PAUSE_MODELING / PRESERVE_EVIDENCE / WAIT_FOR_NEW_INFORMATION_OR_AUTHORIZATION",
     "NO_ACTIONABLE_NEW_EVIDENCE_SOURCE",
     "Registrar el veredicto; las tres opciones documentales N1-N3 quedan a disposicion del propietario sin accion automatica"],
    ["N1_revision_humana_ciega_independiente",
     "Adjudicacion ciega INDEPENDIENTE de las 14 candidatas (revisor que nunca vio A/B) mas una sonda de ambiguedad sobre una muestra aleatoria pequena",
     "Las decisiones C1/C2 de la ronda 2 vieron A/B (no independientes); solo un revisor ciego aporta evidencia independiente",
     "Si las 14 candidatas tienen apoyo independiente o son un artefacto de la visibilidad de A/B; primera tasa directa de ambiguedad no enriquecida",
     "Decision del propietario (trabajo humano interno; no modifica el train ni toca particiones reservadas)",
     "Bajo-medio: 14 celdas por 1-2 revisores ciegos mas 30-60 celdas aleatorias; horas humanas, 0 GPU",
     "El revisor ciego rechaza la mayoria de las 14: cierra la via F23 de forma definitiva",
     "Cerrar F23 (si rechazo) o disenar el experimento de correccion pre-registrado (si apoyo fuerte y consistente)",
     "CERCANO A CERO (14 celdas; techo bajo la banda de ruido)",
     "Documental, a disposicion del propietario; no necesaria para el score"],
    ["N2_auditoria_dirigida_de_negativos",
     "Defecto de implementacion DEMOSTRADO (condicion 3 de reapertura) en resultados cerrados de alto peso (p. ej. V7-01 F1-suave; V26/V27 kernel; V32-05 identidad; S08 meta)",
     "No repite experimentos: audita codigo y artefactos publicados (lectura y comprobaciones estaticas), el mecanismo que historicamente si encontro errores (P84-03, P81, D-079)",
     "Si alguna familia cerrada lo esta por un bug y no por el mecanismo",
     "Ninguna (solo lectura); el propietario puede solicitarla",
     "CPU-min; 0 GPU; sin re-ejecutar entrenamientos",
     "Ningun defecto encontrado: refuerza el cierre actual",
     "Reabrir SOLO la familia cuyo defecto se demuestre, con el protocolo de reapertura P6",
     "BAJO-MUY BAJO (P89 y P94 ya auditaron consistencia sin hallazgos)",
     "Opcional; no prioritaria"],
    ["N3_clarificacion_semantica_TR",
     "Pronunciamiento del organizador/profesor sobre la definicion operativa de TR usada en el gold (D-005 abierto desde el inicio)",
     "No es analisis de datos: es informacion del protocolo de anotacion, la unica fuente capaz de resolver la ambiguedad oficial vs HatEval",
     "Que semantica de TR se premia; interpretacion de la frontera de anotacion y de los errores de TR",
     "El propietario decide si contacta (P95-P2 prohibe contacto automatico)",
     "~0 (una consulta)",
     "El organizador no responde o remite a la definicion publicada: sin cambio",
     "Solo informativa/documental; no habilita entrenamiento por si misma",
     "CERCANO A CERO (el modelo ya usa la etiqueta tal cual)",
     "Opcional, si el propietario lo considera util para el cierre academico"],
]


def validar(nombre: str, cab: list[str], filas: list[list[str]],
            admitidas: set[str] | None = None) -> list[str]:
    """Valida estructura, campos no vacios y ASCII."""
    fallos: list[str] = []
    ids: list[str] = []
    for f in filas:
        if len(f) != len(cab):
            fallos.append(f"{nombre} {f[0]}: {len(f)} campos, esperados {len(cab)}")
            continue
        ids.append(f[0])
        if any(not str(v).strip() for v in f):
            fallos.append(f"{nombre} {f[0]}: campo vacio")
        if admitidas is not None and f[1] not in admitidas:
            fallos.append(f"{nombre} {f[0]}: clase invalida {f[1]}")
    if len(ids) != len(set(ids)):
        fallos.append(f"{nombre}: ids duplicados")
    if any(ord(c) > 127 for fila in filas for celda in fila for c in str(celda)):
        fallos.append(f"{nombre}: byte > 127")
    return fallos


def main() -> None:
    """Escribe y valida ambos CSV."""
    fallos = validar("governance_options", CAB_GOV, FILAS_GOV, CLASES_GOV)
    fallos += validar("new_evidence_options", CAB_EVI, FILAS_EVI)
    if fallos:
        for f in fallos:
            print("  -", f)
        sys.exit(1)
    for ruta, cab, filas in ((SAL_GOV, CAB_GOV, FILAS_GOV), (SAL_EVI, CAB_EVI, FILAS_EVI)):
        with open(ruta, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(cab)
            w.writerows(filas)
    # relectura
    for ruta, n in ((SAL_GOV, len(FILAS_GOV)), (SAL_EVI, len(FILAS_EVI))):
        with open(ruta, encoding="utf-8", newline="") as fh:
            leidas = list(csv.reader(fh))
        if len(leidas) != n + 1:
            fallos.append(f"{ruta.name}: relectura inconsistente")
    print("P95-00 opciones: ejecutado")
    print(f"  governance_options: {len(FILAS_GOV)} vias | clases: "
          + str({c: sum(1 for f in FILAS_GOV if f[1] == c) for c in CLASES_GOV}))
    print(f"  new_evidence_options: {len(FILAS_EVI)} filas (incluye sintesis N0)")
    if fallos:
        sys.exit(1)
    print("  ambos CSV escritos y validados")


if __name__ == "__main__":
    main()
