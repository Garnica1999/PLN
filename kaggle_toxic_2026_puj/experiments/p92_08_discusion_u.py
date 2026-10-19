#!/usr/bin/env python
"""
P92-U - Paquete final de discusion humana, registro de decisiones y gate de distribucion
(fase de preparacion; sin decisiones automaticas, sin entrenamiento).

Entradas (solo lectura):
  reports/p92t_audit_results.json
  artifacts/human_annotation_v1/calibration_responses/calibration_response_{A,B,C,D}.csv
  artifacts/human_annotation_v1/manifest.json

Salidas (ASCII puro; no sobrescriben el paquete P92-T ni informes historicos):
  reports/p92u_human_discussion_pack.md
  reports/p92_guideline_decision_record.md
  reports/p92u_readiness_gate.md

Reglas: el agente NO decide cuestiones normativas ni completa campos de decision humana;
la opcion conservadora se documenta como respaldo, no como regla aprobada; sin etiquetas
originales, sin informacion de modelos, sin motivos de seleccion en los documentos.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESP = RAIZ / "artifacts" / "human_annotation_v1" / "calibration_responses"
RUTA_JSON = RAIZ / "reports" / "p92t_audit_results.json"

ETIQUETAS = ("HS", "TR", "AG")
COLUMNAS_CLASE = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}

HASHES_ESPERADOS = {
    "artifacts/human_annotation_v1/calibration_responses/calibration_response_A.csv": "a20521989aa3fea9",
    "artifacts/human_annotation_v1/calibration_responses/calibration_response_B.csv": "41e50c1f1d48760e",
    "artifacts/human_annotation_v1/calibration_responses/calibration_response_C.csv": "8de76e21046d5091",
    "artifacts/human_annotation_v1/calibration_responses/calibration_response_D.csv": "763229f9a9caa176",
    "artifacts/human_annotation_v1/calibration_responses/calibration_response_A_normalized.csv": "7e7b3c805620e613",
    "artifacts/human_annotation_v1/calibration_responses/calibration_response_B_normalized.csv": "7cca95e20a1243c0",
    "artifacts/human_annotation_v1/annotator_A.csv": "7dfd5b898d3905f2",
    "artifacts/human_annotation_v1/annotator_B.csv": "956646fa479f0c65",
    "submissions/toxic_submission_v03_lr_union_thr.csv": "7d7478f271e6edfd",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
}

SEIS_HS = ["P92C-05", "P92C-09", "P92C-11", "P92C-13", "P92C-14", "P92C-15"]
SECUNDARIOS = [
    ("P92C-01", "TR (C: UTD) y AG (D: UTD) - contexto insuficiente", ["Q6"]),
    ("P92C-03", "HS (D: UTD) - misma duda normativa que las seis", ["Q1"]),
    ("P92C-04", "HS (D: UTD) - contexto insuficiente + duda normativa", ["Q1", "Q6"]),
    ("P92C-06", "AG (D: UTD) - generalizacion hostil sin insulto", ["Q3"]),
    ("P92C-07", "HS (C: UTD) y AG (D: UTD) - sarcasmo/cita sin contexto", ["Q4", "Q5", "Q6"]),
    ("P92C-02", "flag de fila de A (notas=1, motivo no especificado)", ["Q7"]),
]

# ------------------------------------------------------------- agenda Q1-Q8
PREGUNTAS = [
    {
        "id": "Q1",
        "titulo": "Distincion entre odio y ofensa sexista individual",
        "oficial": ("HS: \"indica si hay discurso de odio (1 = si, 0 = no)\" "
                    "(docs/competition_rules.md). No existe criterio operativo, lista de grupos "
                    "protegidos, ejemplos ni FAQ (verificado en fases previas)."),
        "evidencia": [
            "6/15 filas de la muestra con UTD coincidente de C y D en HS (P92C-05, 09, 11, 13, "
            "14, 15); todas son ofensas o insultos sexistas dirigidos a personas concretas.",
            "D uso UTD con la misma duda normativa en P92C-03 y P92C-04; en 04 la nota de C "
            "registra la misma duda (odio vs. ofensa) aunque respondio POSITIVE.",
            "Ninguna de las notas menciona pertenencia a un grupo: la duda es individuo "
            "(persona concreta) frente a la categoria oficial 'discurso de odio'.",
            "La ambiguedad es sistematica y localizada, no ruido: dos anotadores independientes "
            "convergen en las mismas filas y el mismo motivo.",
        ],
        "alternativas": [
            ("O1 (estricta)",
             "HS = POSITIVE solo cuando la hostilidad se dirige a alguien por su identidad o "
             "pertenencia a un grupo.",
             "Lectura restrictiva y comun de 'discurso de odio'; decisiones reproducibles cuando "
             "el objetivo es un grupo; reduce el riesgo de llamar 'odio' a cualquier ofensa.",
             "Puede sub-anotar agresiones identitarias a individuos (p. ej., una mujer insultada "
             "por ser mujer sin mencion de grupo); no hay respaldo documental explicito para "
             "restringir a grupos."),
            ("O2 (amplia)",
             "HS = POSITIVE cuando el texto degrada o expresa hostilidad hacia una persona o "
             "grupo, aunque el motivo sea individual.",
             "Trato uniforme de la hostilidad dirigida; menos UTD; decisiones mas simples.",
             "Equipara 'ofensa' y 'odio', que el texto oficial trata como conceptos distintos "
             "('discurso de odio' frente a 'agresivo'); puede sobre-extender la etiqueta."),
            ("O3 (literal + UTD)",
             "Mantener la definicion oficial sin criterio operativo y usar UNABLE_TO_DETERMINE "
             "cuando no permita decidir.",
             "No inventa norma; fiel al texto oficial; preserva la incertidumbre de forma "
             "trazable; es la unica opcion que no requiere una interpretacion no documentada.",
             "Mas celdas UTD en la ronda principal; no produce etiquetas binarias para estas "
             "filas; traslada la decision al analisis posterior."),
        ],
        "conservadora": "O3 (literal + UTD) como respaldo si la discusion no adopta una regla.",
        "decision": ("Los responsables humanos (anotadores y responsables del proyecto) eligen "
                     "O1, O2 u O3, o deciden elevar la pregunta (ver Q2)."),
        "estado": "REQUIRES_GOVERNANCE_CLARIFICATION",
        "ch": ["CH-03", "CH-07"],
    },
    {
        "id": "Q2",
        "titulo": "Solicitar aclaracion normativa al responsable de la competicion",
        "oficial": ("No aplica: no existe canal documentado de aclaraciones para anotadores; la "
                    "pagina de la competencia no publica criterios operativos."),
        "evidencia": [
            "P91 dejo la gobernanza en revision (GOVERNANCE_REVIEW_REQUIRED) por la misma "
            "ausencia de criterios operativos.",
            "La calibracion confirmo que la duda es material (6/15 filas HS + 2 mas) y "
            "sistematica entre anotadores independientes.",
        ],
        "alternativas": [
            ("S1 enviar consulta",
             "Enviar una consulta escrita breve al organizador (redaccion propuesta en la "
             "seccion 5 de este paquete).",
             "Unica via de resolver normativamente; deja constancia institucional; puede "
             "beneficiar a todos los equipos.",
             "Depende de los tiempos del organizador; la respuesta puede no llegar antes de la "
             "ronda principal; el envio lo hacen los responsables (el agente no contacta a "
             "nadie)."),
            ("S2 no enviar; adoptar O3",
             "No elevar la consulta y operar con la definicion literal + UTD.",
             "Autonomia total; sin dependencia externa.",
             "El vacio normativo queda como riesgo permanente del piloto; las decisiones no "
             "quedan respaldadas por el organizador."),
            ("S3 enviar y continuar en paralelo",
             "Enviar la consulta y comenzar la ronda principal con UTD, registrando que se "
             "anoto sin la aclaracion.",
             "No bloquea la ronda principal; si llega la aclaracion, se aplica hacia adelante.",
             "Convivencia de criterios si la respuesta llega tarde; requiere documentar el "
             "punto de corte (sin re-anotar lo ya hecho)."),
        ],
        "conservadora": ("Ninguna especifica: la no-respuesta no cambia por si sola la guia; "
                         "aplica la opcion conservadora de Q1/Q3."),
        "decision": ("Los responsables deciden si se envia la consulta, quien la firma y cuando; "
                     "la redaccion propuesta esta lista para revision."),
        "estado": "UNRESOLVED",
        "ch": ["CH-07"],
    },
    {
        "id": "Q3",
        "titulo": "Distinguir agresividad de vulgaridad",
        "oficial": ("AG: \"indica si el texto es agresivo (1 = si, 0 = no)\". Sin criterio "
                    "operativo ni ejemplos."),
        "evidencia": [
            "P92C-06: D UTD en AG - generalizacion hostil contra un grupo sin insulto ni amenaza "
            "explicita ('la agresividad es discutible'); A, B y C respondieron NEGATIVE.",
            "P92C-13: C UTD en AG - insulto leve con posible tono de broma; duda agresivo vs. "
            "vulgar; D respondio POSITIVE.",
            "P92C-07: D UTD en AG - sarcasmo o cita sin contexto; agresividad no inequivoca.",
            "AG fue la etiqueta con menor coincidencia exacta de los 4 anotadores (6/15).",
        ],
        "alternativas": [
            ("A1 (agresion dirigida)",
             "AG = POSITIVE si hay agresion dirigida (insulto, amenaza, orden hostil) hacia una "
             "persona o grupo; la vulgaridad no dirigida no basta.",
             "Regla operativa clara para la mayoria de casos; coincide con el uso observado (los "
             "casos AG POSITIVE implican destinatario); reduce UTD por vulgaridad.",
             "No cubre la generalizacion hostil sin insulto (caso 06); es una interpretacion, no "
             "una definicion oficial."),
            ("A2 (amplia)",
             "Incluir tambien generalizaciones hostiles hacia grupos aunque no haya insulto "
             "dirigido.",
             "Captura hostilidad grupal; menos UTD.",
             "Se aleja del sentido comun de 'agresivo'; riesgo de solapamiento con HS y TR."),
            ("A3 (literal + UTD)",
             "Sin criterio operativo; UTD cuando 'agresivo' no sea claro.",
             "Sin interpretacion no documentada; incertidumbre trazable.",
             "Mas UTD; decisiones menos consistentes entre anotadores."),
        ],
        "conservadora": "A3 (literal + UTD) como respaldo.",
        "decision": "Los responsables humanos eligen A1, A2 u A3 (o elevan la parte no resoluble via Q2).",
        "estado": "REQUIRES_GOVERNANCE_CLARIFICATION",
        "ch": ["CH-03"],
    },
    {
        "id": "Q4",
        "titulo": "Tono de broma, ironia y sarcasmo",
        "oficial": "Ninguna regla oficial (no hay instrucciones de casos limite).",
        "evidencia": [
            "P92C-13: 'podria ser broma entre conocidos' (nota de C) como razon de duda en HS y AG.",
            "P92C-07: sarcasmo o critica; 'De puta madre' interpretada como expresion idiomatica "
            "(nota de D).",
            "Ningun caso de la muestra produjo consenso de que la broma neutralice la hostilidad.",
        ],
        "alternativas": [
            ("T1 (contenido expresado)",
             "Evaluar el contenido hostil expresado; el tono retorico (broma, ironia, sarcasmo) "
             "no decide por si solo.",
             "Regla uniforme; evita que 'era broma' excuse cualquier hostilidad; coincide con el "
             "uso observado (los cuatro anotadores evaluaron contenido).",
             "El sarcasmo puede invertir el significado; requiere lectura cuidadosa."),
            ("T2 (tono neutraliza)",
             "Una broma entre conocidos neutraliza agresion u odio.",
             "Refleja intuiciones sobre contexto social.",
             "Imposible de verificar sin conocer la relacion entre las personas; abre la puerta "
             "a excusar hostilidad; sin respaldo documental."),
            ("T3 (UTD si el tono es determinante)",
             "Usar UTD cuando el tono sea lo que decide el caso y no sea verificable.",
             "Honesta con la incertidumbre; conservadora.",
             "Mas UTD; conviene combinarla con T1 para no trasladar toda la carga a UTD."),
        ],
        "conservadora": "T1 + T3 (contenido expresado; UTD cuando el tono sea determinante e inverificable).",
        "decision": "Los responsables humanos eligen la combinacion operativa y la registran.",
        "estado": "UNRESOLVED",
        "ch": ["CH-05", "CH-03"],
    },
    {
        "id": "Q5",
        "titulo": "Citas y discurso reproducido",
        "oficial": "Ninguna regla oficial.",
        "evidencia": [
            "P92C-07: critica o sarcasmo sobre descendientes de una persona descrita como "
            "'genocida'; sin el contexto enlazado la intencion no es verificable (notas de C y D).",
            "P92C-01: apodo referenciado ('nio polla') que no puede interpretarse sin el contexto "
            "original.",
            "No hubo en la muestra ningun caso con cita textual entre comillas; la ambiguedad "
            "observada es de referencias y reproduccion indirecta.",
        ],
        "alternativas": [
            ("C1 (contenido tal como se expresa)",
             "Evaluar el contenido del texto tal como aparece, incluida la cita, sin inferir la "
             "intencion del autor.",
             "Reproducible; no requiere informacion externa.",
             "Puede penalizar textos que citan para denunciar."),
            ("C2 (inferir intencion)",
             "Intentar distinguir denuncia de adhesion.",
             "Mas justo con textos de denuncia.",
             "Requiere adivinar; sin contexto es especulacion; no verificable."),
            ("C3 (C1 + UTD)",
             "Aplicar C1 y usar UTD cuando el texto no permita saber si hay adhesion, denuncia "
             "o burla.",
             "Combina reproducibilidad con honestidad; coincide con la practica observada.",
             "Mas UTD."),
        ],
        "conservadora": "C3 (C1 + UTD).",
        "decision": "Los responsables humanos eligen la regla operativa y la registran.",
        "estado": "UNRESOLVED",
        "ch": ["CH-05"],
    },
    {
        "id": "Q6",
        "titulo": "Cuando corresponde UNABLE_TO_DETERMINE",
        "oficial": ("No aplica: UTD no existe en el formato oficial de la competencia; es un "
                    "mecanismo del protocolo de anotacion, ya presente en la guia v1."),
        "evidencia": [
            "Contexto insuficiente: P92C-01 (TR y AG), P92C-04 (HS), P92C-07 (HS y AG) - blanco "
            "o referencia no identificables.",
            "Ambiguedad normativa: P92C-05, 09, 11, 13, 14, 15 (HS) - la definicion oficial no "
            "alcanza.",
            "Practica observada: C uso UTD en 9 celdas y D en 11 (columnas de clase); A y B no "
            "usaron UTD en clases (registraron duda como flag de fila).",
        ],
        "alternativas": [
            ("U1 (solo contexto)",
             "UTD solo cuando falta contexto (lectura restrictiva de v1).",
             "Acota el uso; mantiene el foco en el contexto.",
             "Obliga a decidir casos normativamente indecidibles (contradice 'no inventar "
             "reglas'); ya resulto insuficiente en la practica."),
            ("U2 (contexto + ambiguedad normativa)",
             "UTD cuando falte contexto o cuando la definicion oficial no permita decidir "
             "(practica de C y D en la calibracion).",
             "Coherente con la evidencia; evita certezas simuladas; preserva trazabilidad.",
             "Mas UTD; requiere que el analisis posterior trate la incertidumbre explicitamente."),
            ("U3 (forzar binario)",
             "No usar UTD; toda celda POSITIVE o NEGATIVE.",
             "Formato binario uniforme.",
             "Simula certeza inexistente; contradice el protocolo; indefendible ante auditoria; "
             "descartada por P91 y P92 (no inventar reglas)."),
        ],
        "conservadora": "U2 (U1 como subconjunto), documentada como respaldo.",
        "decision": "Los responsables humanos confirman el alcance de UTD y lo registran.",
        "estado": "UNRESOLVED",
        "ch": ["CH-02", "CH-04"],
    },
    {
        "id": "Q7",
        "titulo": "Duda general del tweet frente a incertidumbre por etiqueta",
        "oficial": "No aplica (mecanismo interno del protocolo).",
        "evidencia": [
            "A y B registraron la duda con un flag binario de fila en notas (A: P92C-02, 04; B: "
            "P92C-01, 05, 13); su significado no era derivable de los documentos y requirio una "
            "aclaracion externa; ademas, el flag no indica que etiqueta se duda.",
            "C y D usaron UTD por etiqueta + nota de motivo.",
            "4 de las 5 filas con flag coinciden con UTD de C o D; P92C-02 quedo con flag sin "
            "motivo y sin UTD asociado.",
        ],
        "alternativas": [
            ("R1 (UTD por etiqueta + nota)",
             "Practica de C y D: UTD en la etiqueta afectada con nota de motivo.",
             "Preciso (indica que etiqueta); consistente con la guia v1; trazable.",
             "Si la fila entera es dificil, requiere marcar varias celdas."),
            ("R2 (R1 + nota de fila)",
             "UTD por etiqueta + nota libre adicional cuando la fila entera sea dificil.",
             "Captura ambos niveles; elimina la ambiguedad del flag binario.",
             "Dos lugares para la misma informacion; requiere disciplina de uso."),
            ("R3 (mantener flag binario)",
             "Conservar el mecanismo 0/1 de fila.",
             "Formato compacto.",
             "Demostrado ambiguo (su significado no era derivable de los documentos); no indica "
             "etiqueta; descartado por la evidencia."),
        ],
        "conservadora": "R2 (R1 + nota de fila) como respaldo.",
        "decision": "Los responsables humanos confirman el mecanismo y lo registran.",
        "estado": "UNRESOLVED",
        "ch": ["CH-04"],
    },
    {
        "id": "Q8",
        "titulo": "Formato obligatorio de respuestas para la ronda principal",
        "oficial": ("No aplica (formato del protocolo: tokens en las tres columnas; la guia v1 y "
                    "el README ya lo exigian)."),
        "evidencia": [
            "A y B devolvieron 0/1 en las cuatro columnas (con BOM de Excel), pese a que el "
            "README y la guia exigian POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE; la "
            "interpretacion requirio una aclaracion explicita del responsable del proyecto "
            "(P92-T, ENCODING_CONFIRMED).",
            "C y D devolvieron tokens + notas de texto libre.",
            "La desviacion no invalido las respuestas (fueron interpretables), pero costo una "
            "ronda completa de verificacion.",
        ],
        "alternativas": [
            ("F1 (tokens exclusivos)",
             "POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE en las tres columnas; sin 0/1; guardar "
             "como CSV plano (sin reabrir en Excel si altera el formato).",
             "Esquema sin ambiguedad; ya exigido por v1; elimina el costo de verificacion.",
             "Requiere cuidado al editar el CSV; se mitiga con instrucciones de guardado."),
            ("F2 (tokens + aceptar 0/1)",
             "Aceptar ambos formatos.",
             "Tolerante con herramientas de edicion.",
             "La correspondencia 0/1 no es universalmente intuitiva (0 = NEGATIVE); ya exigio "
             "una aclaracion; descartada por la evidencia."),
        ],
        "conservadora": "F1 (ya vigente en v1); la confirmacion humana queda pendiente.",
        "decision": "Los responsables humanos confirman F1 para ambos paquetes y lo registran.",
        "estado": "UNRESOLVED",
        "ch": ["CH-01", "CH-02"],
    },
]

CHANGELOG_MAPA = [
    ("CH-01", "Formato de respuesta (tokens obligatorios; 0/1 invalido)", ["Q8"]),
    ("CH-02", "Nota de motivo obligatoria cuando se usa UTD", ["Q6", "Q8"]),
    ("CH-03", "Criterios operativos propuestos (odio/ofensa; agresivo/vulgar; tono)", ["Q1", "Q3", "Q4"]),
    ("CH-04", "Duda de fila: UTD por etiqueta + nota (sin flag binario aparte)", ["Q6", "Q7"]),
    ("CH-05", "Citas, sarcasmo, ironia y contexto", ["Q4", "Q5"]),
    ("CH-06", "Ejemplos estructurales anonimizados", ["(derivado de Q1 y Q3)"]),
    ("CH-07", "Preguntas normativas formalizadas (consulta al organizador)", ["Q1", "Q2", "Q3"]),
    ("CH-08", "Independencia HS / TR / AG reforzada", ["(sin pregunta directa; se mantiene)"]),
]


def ascii_texto(texto: str) -> str:
    """Translitera a ASCII (NFKD sin diacriticos; comillas angulares a rectas)."""
    reemplazos = {
        "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'", "\u2013": "-", "\u2014": "-", "\u2026": "...",
    }
    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def sha256_archivo(ruta: Path) -> str:
    """sha256 completo de un archivo (lectura en bloques)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_respuestas() -> tuple[dict, dict]:
    """Respuestas crudas por anotador y textos por review_id."""
    respuestas: dict[str, dict[str, dict]] = {}
    textos: dict[str, str] = {}
    for anotador in ("A", "B", "C", "D"):
        with open(DIR_RESP / f"calibration_response_{anotador}.csv",
                  encoding="utf-8-sig", newline="") as fh:
            filas = list(csv.DictReader(fh))
        respuestas[anotador] = {
            f["review_id"]: {
                "HS": f["HS_annotation"], "TR": f["TR_annotation"],
                "AG": f["AG_annotation"], "notas": f["uncertainty_notes"],
            }
            for f in filas
        }
        for f in filas:
            textos[f["review_id"]] = f["text"]
    return respuestas, textos


def token_de(valor: str, anotador: str) -> str:
    """Token interpretado (A/B con codificacion confirmada 0 = NEGATIVE, 1 = POSITIVE)."""
    if anotador in ("A", "B"):
        return {"0": "NEGATIVE", "1": "POSITIVE"}[valor]
    return valor


def cita(nota: str) -> str:
    """Nota humana entre comillas rectas, con comillas internas en simple."""
    return '"' + ascii_texto(nota).replace('"', "'") + '"'


def linea_respuestas(respuestas: dict, rid: str, anotador: str) -> str:
    """Linea de respuestas de un anotador para una fila (tokens + detalle de formato)."""
    fila = respuestas[anotador][rid]
    tokens = " / ".join(token_de(fila[e], anotador) for e in ETIQUETAS)
    detalle = ""
    if anotador in ("A", "B"):
        crudas = "/".join(fila[e] for e in ETIQUETAS)
        detalle = f" (respuestas originales {crudas}"
        if fila["notas"] == "1":
            detalle += "; marca de fila: si"
        detalle += ")"
    elif fila["notas"]:
        detalle = ""
    return f"- {anotador}: {tokens}{detalle}"


def auditar_integridad() -> dict:
    """Re-verifica hashes de archivos clave contra los valores congelados."""
    auditoria = {}
    for nombre, prefijo in HASHES_ESPERADOS.items():
        ruta = RAIZ / nombre
        if ruta.exists():
            h = sha256_archivo(ruta)
            auditoria[nombre] = {"sha256_16": h[:16], "ok": h.startswith(prefijo)}
        else:
            auditoria[nombre] = {"sha256_16": None, "ok": False}
    return auditoria


# --------------------------------------------------------- paquete de discusion
def escribir_pack(res: dict, respuestas: dict, textos: dict) -> None:
    """Paquete final y autocontenido de discusion humana."""
    a = res["acuerdo"]
    hs, tr, ag = a["por_etiqueta"]["HS"], a["por_etiqueta"]["TR"], a["por_etiqueta"]["AG"]
    lineas = []
    lineas.append("# P92-U - Paquete final de discusion humana (autocontenido)")
    lineas.append("")
    lineas.append("Para: los cuatro participantes de la calibracion (A, B, C y D) y los")
    lineas.append("responsables del proyecto. Fecha: 2026-10-12. Estado:")
    lineas.append("**HUMAN_DISCUSSION_PREPARED** (las decisiones se registran en")
    lineas.append("`reports/p92_guideline_decision_record.md`; esta fase no decide nada).")
    lineas.append("")
    lineas.append("Este documento es la version final y autocontenida del paquete de discusion;")
    lineas.append("no reemplaza ni sobrescribe el paquete de P92-T")
    lineas.append("(`reports/p92_calibration_discussion_pack.md`, historico).")
    lineas.append("")
    lineas.append("Reglas de la discusion:")
    lineas.append("")
    lineas.append("- No se muestran etiquetas originales, probabilidades OOF, predicciones de V03,")
    lineas.append("  umbrales ni motivos de seleccion.")
    lineas.append("- No se decide aqui quien tiene razon; no hay adjudicacion automatica.")
    lineas.append("- La mayoria simple no resuelve por si sola una cuestion normativa.")
    lineas.append("- Resultado esperado: un registro de decisiones humanas y cuestiones")
    lineas.append("  pendientes, con la correspondencia frente a los cambios CH-01 a CH-08.")
    lineas.append("- Las notas citadas se transliteran a ASCII; los originales con acentos")
    lineas.append("  permanecen en los archivos de respuesta de cada persona.")
    lineas.append("")
    lineas.append("## 1. Contexto y evidencia disponible (resumen)")
    lineas.append("")
    lineas.append("- Calibracion: 15 tweets anotados de forma independiente por 4 personas con la")
    lineas.append("  guia v1; la codificacion 0/1 de A y B quedo confirmada (0 = NEGATIVE,")
    lineas.append("  1 = POSITIVE; en notas, 1 = dificil de determinar para cualquier clase).")
    lineas.append("- Acuerdo exacto (de 15 filas): HS 12/15, TR 14/15, AG 11/15 en el par C-D;")
    lineas.append("  kappa binario C-D = 1.000 en las tres etiquetas; coincidencia exacta de los 4")
    lineas.append(f"  en las tres etiquetas: {a['combinacion_exacta_global']['num']}/15.")
    lineas.append("- UTD por etiqueta (columnas de clase): HS C=7 D=8; TR C=1 D=0; AG C=1 D=3.")
    lineas.append("  A y B no usaron UTD en clases; registraron su duda como marca de fila")
    lineas.append("  (A: P92C-02, 04; B: P92C-01, 05, 13).")
    lineas.append("- La ambiguedad normativa dominante (odio vs. ofensa sexista individual) se")
    lineas.append("  concentra en 6 filas HS con UTD coincidente de C y D.")
    lineas.append("- Limitacion muestral: todos los numeros son de esta muestra de 15 tweets, con")
    lineas.append("  denominadores visibles; no se extrapolan a la muestra principal ni al dataset.")
    lineas.append("")
    lineas.append("## 2. Agenda de decision (Q1 a Q8)")
    lineas.append("")
    lineas.append("Estados usados: `UNRESOLVED` = decision humana pendiente;")
    lineas.append("`REQUIRES_GOVERNANCE_CLARIFICATION` = la definicion oficial no alcanza y se")
    lineas.append("requiere aclaracion del responsable de la competicion (o una decision humana")
    lineas.append("explicita de operar con la opcion conservadora); `DECIDED` = no usado en esta")
    lineas.append("fase (se actualizara en el registro). Ninguna pregunta esta decidida: el agente")
    lineas.append("no toma decisiones en nombre de los participantes.")
    lineas.append("")
    for q in PREGUNTAS:
        lineas.append(f"### {q['id']}. {q['titulo']}")
        lineas.append("")
        lineas.append(f"**Definicion oficial disponible**: {ascii_texto(q['oficial'])}")
        lineas.append("")
        lineas.append("**Evidencia observada en la calibracion**:")
        for ev in q["evidencia"]:
            lineas.append(f"- {ascii_texto(ev)}")
        lineas.append("")
        lineas.append("**Alternativas propuestas** (ventajas / riesgos):")
        for nombre, descripcion, ventajas, riesgos in q["alternativas"]:
            lineas.append(f"- {ascii_texto(nombre)}: {ascii_texto(descripcion)}")
            lineas.append(f"  - Ventajas: {ascii_texto(ventajas)}")
            lineas.append(f"  - Riesgos: {ascii_texto(riesgos)}")
        lineas.append("")
        lineas.append(f"**Opcion conservadora documentada (NO aprobada)**: {ascii_texto(q['conservadora'])}")
        lineas.append("")
        lineas.append(f"**Decision requerida**: {ascii_texto(q['decision'])}")
        lineas.append("")
        lineas.append(f"**Estado**: `{q['estado']}` | Cambios relacionados: {', '.join(q['ch'])}")
        lineas.append("")
    lineas.append("## 3. Los seis casos HS con UTD coincidente de C y D (evidencia de Q1)")
    lineas.append("")
    lineas.append("En las seis filas, C y D registraron UNABLE_TO_DETERMINE en HS con el mismo")
    lineas.append("motivo: la guia no define cuando una ofensa sexista individual constituye")
    lineas.append("discurso de odio. Textos (verbatim) y respuestas de los cuatro:")
    lineas.append("")
    for rid in SEIS_HS:
        lineas.append(f"### {rid}")
        lineas.append("")
        lineas.append(f"Texto: \"{textos[rid]}\"")
        lineas.append("")
        for anotador in ("A", "B", "C", "D"):
            lineas.append(linea_respuestas(respuestas, rid, anotador))
        lineas.append(f"- Nota C: {cita(respuestas['C'][rid]['notas'])}")
        lineas.append(f"- Nota D: {cita(respuestas['D'][rid]['notas'])}")
        lineas.append("")
    lineas.append("Pregunta para estas seis filas (Q1): debe anotarse HS = POSITIVE cuando el")
    lineas.append("insulto sexista se dirige a una persona concreta? Cual opcion (O1, O2 u O3)")
    lineas.append("describe mejor la definicion oficial? Si la discusion concluye que la norma no")
    lineas.append("alcanza, la pregunta se eleva al organizador (Q2).")
    lineas.append("")
    lineas.append("## 4. Casos secundarios que precisan la guia")
    lineas.append("")
    for rid, donde, preguntas in SECUNDARIOS:
        lineas.append(f"### {rid} - {donde} (relacionado con {', '.join(preguntas)})")
        lineas.append("")
        lineas.append(f"Texto: \"{textos[rid]}\"")
        lineas.append("")
        for anotador in ("A", "B", "C", "D"):
            lineas.append(linea_respuestas(respuestas, rid, anotador))
        for anotador in ("C", "D"):
            nota = respuestas[anotador][rid]["notas"]
            if nota:
                lineas.append(f"- Nota {anotador}: {cita(nota)}")
        if respuestas["A"][rid]["notas"] == "1":
            lineas.append("- A: marco la fila como dificil para cualquier clase (motivo no especificado).")
        if respuestas["B"][rid]["notas"] == "1" and rid != "P92C-02":
            lineas.append("- B: marco la fila como dificil.")
        lineas.append("")
    lineas.append("## 5. Informacion normativa faltante y consulta propuesta (Q2)")
    lineas.append("")
    lineas.append("Falta, como minimo: (1) criterio para distinguir ofensa sexista individual de")
    lineas.append("discurso de odio; (2) criterio agresivo frente a vulgar y tratamiento de la")
    lineas.append("generalizacion hostil; (3) tratamiento de citas, sarcasmo y discurso reproducido;")
    lineas.append("(4) relacion formal (si existe) entre HS y TR, y entre HS y AG.")
    lineas.append("")
    lineas.append("Redaccion propuesta de la consulta (para revision de los responsables; NO")
    lineas.append("enviada por el agente):")
    lineas.append("")
    lineas.append("```text")
    lineas.append("Estimado/a responsable de la competencia Tweet Toxic 2026 PUJ:")
    lineas.append("")
    lineas.append("Durante un piloto de calibracion de anotacion (4 anotadores independientes, 15")
    lineas.append("textos) encontramos una ambiguedad sistematica en las definiciones oficiales.")
    lineas.append("Agradeceriamos una aclaracion sobre:")
    lineas.append("")
    lineas.append("1. HS: cuando un insulto o agresion sexista se dirige a una persona concreta por")
    lineas.append("   su identidad, sin referirse a un grupo, debe anotarse HS = 1, HS = 0, o no")
    lineas.append("   existe criterio?")
    lineas.append("2. AG: como debe interpretarse \"agresivo\" frente a \"vulgar\" (por ejemplo, un")
    lineas.append("   insulto leve o lenguaje soez sin agresion dirigida)? Las generalizaciones")
    lineas.append("   hostiles hacia un grupo sin insulto explicito cuentan como AG?")
    lineas.append("")
    lineas.append("Las dudas quedaron registradas sin forzar decisiones. Gracias.")
    lineas.append("```")
    lineas.append("")
    lineas.append("## 6. Como se registran las decisiones")
    lineas.append("")
    lineas.append("Cada pregunta tiene una entrada en `reports/p92_guideline_decision_record.md`")
    lineas.append("con los campos: question_id, issue, official_rule, observed_evidence,")
    lineas.append("options_considered, human_decision, rationale, remaining_uncertainty,")
    lineas.append("decision_owner, decision_date, guideline_change. Los campos de decision humana")
    lineas.append("estan vacios y los completan las personas responsables durante la discusion.")
    lineas.append("")
    lineas.append("## 7. Que NO hace este paquete")
    lineas.append("")
    lineas.append("- No decide ninguna etiqueta ni cuestion normativa por si mismo.")
    lineas.append("- No muestra etiquetas originales, probabilidades OOF, predicciones, umbrales ni")
    lineas.append("  motivos de seleccion de los casos.")
    lineas.append("- No contacta a los anotadores ni al organizador.")
    lineas.append("- No reescribe los resultados de la calibracion: siguen interpretados bajo la guia")
    lineas.append("  v1; cualquier guia posterior no se aplica retroactivamente sin documentarlo.")
    ruta = RAIZ / "reports" / "p92u_human_discussion_pack.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


# ------------------------------------------------------- registro de decisiones
def escribir_registro() -> None:
    """Plantilla editable del registro de decisiones (campos humanos vacios)."""
    lineas = []
    lineas.append("# P92-U - Registro de decisiones de la guia de anotacion (plantilla editable)")
    lineas.append("")
    lineas.append("Estado: **PLANTILLA**. Los campos de decision humana estan VACIOS a proposito.")
    lineas.append("No inventar consenso ni simular una reunion. Este registro lo completan las")
    lineas.append("personas responsables (anotadores y responsables del proyecto) durante la")
    lineas.append("discusion, usando `reports/p92u_human_discussion_pack.md` como material.")
    lineas.append("")
    lineas.append("Instrucciones de uso:")
    lineas.append("")
    lineas.append("1. Para cada pregunta (Q1 a Q8), completar `human_decision`, `rationale`,")
    lineas.append("   `decision_owner` y `decision_date`; actualizar `remaining_uncertainty` y")
    lineas.append("   `guideline_change` segun lo acordado.")
    lineas.append("2. Los campos `issue`, `official_rule`, `observed_evidence` y `options_considered`")
    lineas.append("   ya estan prellenados desde la evidencia de P92-T/P92-U; pueden ampliarse.")
    lineas.append("3. Ninguna decision se aplica a los paquetes principales hasta que la version de")
    lineas.append("   la guia resultante este congelada, con changelog y SHA256 (seccion final).")
    lineas.append("")
    lineas.append("## 1. Registro por pregunta")
    lineas.append("")
    for q in PREGUNTAS:
        lineas.append(f"### {q['id']} - {q['titulo']}")
        lineas.append("")
        lineas.append(f"- question_id: {q['id']}")
        lineas.append(f"- issue: {ascii_texto(q['titulo'])}")
        lineas.append(f"- official_rule: {ascii_texto(q['oficial'])}")
        lineas.append("- observed_evidence:")
        for ev in q["evidencia"]:
            lineas.append(f"  - {ascii_texto(ev)}")
        lineas.append("- options_considered:")
        for nombre, descripcion, _ventajas, _riesgos in q["alternativas"]:
            lineas.append(f"  - {ascii_texto(nombre)}: {ascii_texto(descripcion)}")
        lineas.append(f"- status: {q['estado']}")
        lineas.append("- human_decision: (vacio - completar en la discusion)")
        lineas.append("- rationale: (vacio - completar en la discusion)")
        lineas.append(f"- remaining_uncertainty: PENDIENTE de la decision; opcion conservadora "
                       f"documentada: {ascii_texto(q['conservadora'])}")
        lineas.append("- decision_owner: (vacio - completar en la discusion)")
        lineas.append("- decision_date: (vacio - completar en la discusion)")
        lineas.append(f"- guideline_change: {', '.join(q['ch'])} (confirmar o ajustar)")
        lineas.append("")
    lineas.append("## 2. Correspondencia decision -> changelog CH-01 a CH-08 de la propuesta v2")
    lineas.append("")
    lineas.append("| cambio | descripcion | preguntas relacionadas | decision humana asociada |")
    lineas.append("|---|---|---|---|")
    for ch, descripcion, preguntas in CHANGELOG_MAPA:
        lineas.append(f"| {ch} | {ascii_texto(descripcion)} | {', '.join(preguntas)} | (vacio) |")
    lineas.append("")
    lineas.append("## 3. Decision de inicio de la anotacion principal (gate)")
    lineas.append("")
    lineas.append("| campo | valor |")
    lineas.append("|---|---|")
    lineas.append("| decision_iniciar_anotacion_principal | (vacio: SI / NO / DIFERIDA) |")
    lineas.append("| version_de_guia_aplicable | (vacio: identificar version congelada) |")
    lineas.append("| sha256_guia_congelada | (vacio) |")
    lineas.append("| responsable_que_aprueba | (vacio) |")
    lineas.append("| fecha | (vacio) |")
    lineas.append("")
    lineas.append("## 4. Procedimiento posterior a las decisiones (resumen)")
    lineas.append("")
    lineas.append("1. Mantener v1 como artefacto historico (no se modifica).")
    lineas.append("2. Aplicar a una version nueva de la guia EXCLUSIVAMENTE las decisiones")
    lineas.append("   registradas aqui (la propuesta v2 no se congela por si sola).")
    lineas.append("3. Generar el changelog explicito de la version resultante y separar: reglas")
    lineas.append("   oficiales, reglas operativas aprobadas y preguntas todavia abiertas.")
    lineas.append("4. Calcular el SHA256 de la version resultante y registrarlo.")
    lineas.append("5. Si persiste una ambiguedad semantica material, mantenerla documentada y")
    lineas.append("   determinar si el piloto continua de forma exploratoria con UTD explicito.")
    lineas.append("6. Actualizar los paquetes principales a la version congelada (trazable: nuevo")
    lineas.append("   manifest con hashes) y re-verificar el gate antes de distribuir.")
    ruta = RAIZ / "reports" / "p92_guideline_decision_record.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


# ----------------------------------------------------------------------- gate
def escribir_gate(res: dict, auditoria: dict) -> None:
    """Gate de distribucion de la muestra principal (estado actual y condiciones)."""
    a = res["acuerdo"]
    lineas = []
    lineas.append("# P92-U - Gate de distribucion de la muestra principal")
    lineas.append("")
    lineas.append("Fecha: 2026-10-12. Estado: **NOT_READY (por diseno)**: la muestra principal de")
    lineas.append("250 tweets NO se distribuye en esta fase. Este documento define las condiciones")
    lineas.append("exactas que deben cumplirse antes de distribuirla y su estado actual.")
    lineas.append("")
    lineas.append("## 1. Las seis condiciones del gate")
    lineas.append("")
    lineas.append("| # | condicion | estado actual | detalle |")
    lineas.append("|---|---|---|---|")
    lineas.append("| 1 | Version aplicable de la guia identificada y congelada | PENDIENTE | v1 es historica; v2 esta propuesta (no congelada); la version aplicable se decide en el registro |")
    lineas.append("| 2 | Criterios procedimentales resueltos | PENDIENTE | Q4 a Q8 sin decision registrada (formato, UTD, citas, tono, duda de fila) |")
    lineas.append("| 3 | Ambiguedades normativas restantes documentadas | CUMPLIDA (mantener al dia) | documentadas en `p92_calibration_analysis.md`, propuesta v2 seccion 4 y registro Q1/Q3 |")
    lineas.append("| 4 | Ambos paquetes con la misma version de instrucciones | PENDIENTE | hoy ambos usan v1 (manifest); al congelar la guia nueva se actualizan ambos paquetes de forma trazable |")
    lineas.append("| 5 | Cegamiento e integridad de paquetes verificados | PARCIAL | verificado contra v1 en P92 (14/14); requiere re-verificacion contra la version congelada en el momento de distribuir |")
    lineas.append("| 6 | Decision de iniciar la anotacion principal registrada | PENDIENTE | plantilla en `p92_guideline_decision_record.md` seccion 3 |")
    lineas.append("")
    lineas.append("## 2. Estado actual de los paquetes (evidencia)")
    lineas.append("")
    lineas.append("- `annotator_A.csv` y `annotator_B.csv`: 250 tweets cada uno, anotaciones vacias,")
    lineas.append("  review_id opacos; hashes verificados (seccion 4). NO distribuidos.")
    lineas.append("- `manifest.json` del paquete: guideline_version v1, hashes de los cinco archivos,")
    lineas.append("  listado de cegamiento (ocultos: etiquetas, probabilidades, predicciones, folds,")
    lineas.append("  componente, motivos; visibles: review_id, text, columnas vacias, notas, version).")
    lineas.append("- La clave interna (`reports/p92_internal_sampling_key.csv`) permanece separada.")
    lineas.append("")
    lineas.append("## 3. Que falta exactamente para poder distribuir")
    lineas.append("")
    lineas.append("1. Completar el registro de decisiones (Q1 a Q8) y el gate (seccion 3 del registro).")
    lineas.append("2. Congelar la version aplicable de la guia con changelog y SHA256.")
    lineas.append("3. Actualizar los dos paquetes a la version congelada (misma version para ambos),")
    lineas.append("   con manifest nuevo y trazable; re-verificar cegamiento e integridad.")
    lineas.append("4. Registrar la decision de inicio y distribuir.")
    lineas.append("")
    lineas.append("## 4. Si persiste una ambiguedad semantica material")
    lineas.append("")
    lineas.append("Dos rutas posibles (a decidir y registrar por los responsables; no excluyentes):")
    lineas.append("")
    lineas.append("- Ruta A - piloto exploratorio con UTD: distribuir con la version congelada que")
    lineas.append("  conserve UNABLE_TO_DETERMINE explicito para las categorias cuya definicion no")
    lineas.append("  alcance. No se fuerzan etiquetas binarias para cumplir el formato original.")
    lineas.append("- Ruta B - esperar aclaracion normativa (Q2): mantener la muestra sin distribuir")
    lineas.append("  hasta recibir respuesta del organizador o decidir formalmente continuar sin ella.")
    lineas.append("")
    lineas.append("## 5. Auditoria de integridad (re-verificada en esta fase)")
    lineas.append("")
    lineas.append("| archivo | sha256 (16) | esperado | ok |")
    lineas.append("|---|---|---|---|")
    for nombre, info in auditoria.items():
        esperado = HASHES_ESPERADOS[nombre]
        lineas.append(f"| {nombre} | {info['sha256_16']} | {esperado} | {'OK' if info['ok'] else 'REVISAR'} |")
    lineas.append("")
    lineas.append("- Respuestas humanas originales: intactas (sin cambios).")
    lineas.append("- Normalizados A/B: presentes y trazables (hash registrado en P92-T y PROVENANCE).")
    lineas.append("- Muestra principal: no distribuida; anotaciones vacias.")
    lineas.append("- V03, OOF canonico, etiquetas originales, particion y submissions: intactos.")
    lineas.append("- Test y holdouts: no utilizados en esta fase (ni siquiera hasheados para test).")
    lineas.append("- Sin entrenamiento, sin predicciones, sin pseudo-labeling, sin clasificadores")
    lineas.append("  externos.")
    lineas.append("")
    lineas.append("## 6. Recordatorio de estado congelado")
    lineas.append("")
    lineas.append("```text")
    lineas.append("CALIBRATION = ANALYZED")
    lineas.append("GUIDELINE_V1 = HISTORICAL")
    lineas.append("GUIDELINE_V2 = PROPOSED_PENDING_DISCUSSION")
    lineas.append("MAIN_ANNOTATION = NOT_DISTRIBUTED")
    lineas.append("V03 = FROZEN")
    lineas.append("MODEL_RETRAINING = NOT_AUTHORIZED")
    lineas.append("SOURCE_LABEL_MODIFICATION = NOT_AUTHORIZED")
    lineas.append("```")
    lineas.append("")
    lineas.append("## 7. Registro de la decision de inicio (referencia)")
    lineas.append("")
    lineas.append("La decision de iniciar la anotacion principal se registra en")
    lineas.append("`reports/p92_guideline_decision_record.md` (seccion 3). Hasta entonces, el gate")
    lineas.append("permanece cerrado y ningun paquete se entrega.")
    ruta = RAIZ / "reports" / "p92u_readiness_gate.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


def main() -> None:
    with open(RUTA_JSON, encoding="utf-8") as fh:
        res = json.load(fh)
    respuestas, textos = cargar_respuestas()
    auditoria = auditar_integridad()
    print("P92-U discusion y gate: generando artefactos")
    escribir_pack(res, respuestas, textos)
    escribir_registro()
    escribir_gate(res, auditoria)
    fallos = [n for n, info in auditoria.items() if not info["ok"]]
    print(f"  auditoria de integridad: {len(auditoria) - len(fallos)}/{len(auditoria)} OK"
          + (f" | REVISAR: {fallos}" if fallos else ""))
    print("P92-U discusion y gate: OK")


if __name__ == "__main__":
    main()
