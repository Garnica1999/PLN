#!/usr/bin/env python
"""
P92-V - Congelacion de la politica operativa v2 y preparacion de los paquetes principales.

Genera (ASCII puro):
  reports/p92_annotation_guidelines_v2_frozen.md      (PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
  reports/p92v_operational_policy_decisions.md        (registro de decisiones Q1-Q8)
  reports/p92v_package_audit.md                       (auditoria del paquete)
  reports/p92v_manifest.json                          (manifiesto; espejo en artifacts/)
  artifacts/human_annotation_v2/annotator_A/          (CSV + guia congelada + README)
  artifacts/human_annotation_v2/annotator_B/          (idem)
  artifacts/human_annotation_v2/manifest.json         (espejo del manifiesto)
  + seccion aditiva en reports/p92_guideline_decision_record.md

Reglas: mismos 250 tweets y misma asignacion A/B que v1; no re-seleccion; sin etiquetas,
ids originales, OOF, predicciones, umbrales, folds ni motivos en los paquetes; sin test
ni holdouts; sin distribucion; sin entrenamiento; sin commit.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
V1 = RAIZ / "artifacts" / "human_annotation_v1"
V2 = RAIZ / "artifacts" / "human_annotation_v2"
RUTA_CLAVE = RAIZ / "reports" / "p92_internal_sampling_key.csv"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_SUBMISSION = RAIZ / "data" / "raw" / "submission.csv"
RUTA_HOLDOUT_V2 = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_HOLDOUT2 = RAIZ / "data" / "processed" / "holdout2_ids.csv"
RUTA_GUIA = RAIZ / "reports" / "p92_annotation_guidelines_v2_frozen.md"
RUTA_DECISIONES = RAIZ / "reports" / "p92v_operational_policy_decisions.md"
RUTA_AUDITORIA = RAIZ / "reports" / "p92v_package_audit.md"
RUTA_MANIFEST = RAIZ / "reports" / "p92v_manifest.json"
RUTA_REGISTRO = RAIZ / "reports" / "p92_guideline_decision_record.md"

POLICY_ID = "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2"
FECHA = "2026-10-13"
COLUMNAS_V2 = ["review_id", "text", "HS_annotation", "TR_annotation", "AG_annotation",
               "uncertainty_notes", "tweet_note", "guideline_version"]
MARCADOR_REGISTRO = "DECISION DEL PROPIETARIO DEL PROYECTO (P92-V"

TOKENS_FUGA = ["original_example_id", "sample_component", "selection_reason",
               "dup_group", "sampling_stratum", "source_partition", "original_hs",
               "original_tr", "original_ag", "v03"]
TOKENS_FUGA_INSTRUCCIONES = TOKENS_FUGA + ["p92r-", "p92c-"]

DECISIONES = [
    ("Q1", "HS: odio frente a ofensa",
     "Enfoque conservador: definicion oficial literal mas incertidumbre explicita. No se "
     "inventa una nueva definicion normativa de odio. Si el caso no puede resolverse "
     "responsablemente con la definicion oficial y estas instrucciones: "
     "UNABLE_TO_DETERMINE. Esa respuesta no se convierte automaticamente en negativa.",
     ["CH-03", "CH-07"]),
    ("Q2", "Consulta al organizador (registro de gobernanza)",
     "El piloto continua sin esperar una aclaracion externa. Registro: "
     "governance_clarification = NOT_REQUESTED; official_semantics = UNRESOLVED; "
     "internal_pilot = AUTHORIZED. No se afirma que las cuestiones normativas esten "
     "resueltas oficialmente.",
     ["CH-07"]),
    ("Q3", "AG: agresividad frente a vulgaridad",
     "No equiparar automaticamente vulgaridad, insulto y agresividad. Aplicar la "
     "definicion oficial disponible. Usar UNABLE_TO_DETERMINE cuando no exista evidencia "
     "suficiente para decidir.",
     ["CH-03"]),
    ("Q4", "Broma, ironia y sarcasmo",
     "Evaluar el contenido observable del texto. No inferir una intencion oculta sin "
     "evidencia textual suficiente. Si el tono altera materialmente la interpretacion y no "
     "hay contexto suficiente: UNABLE_TO_DETERMINE, explicando la duda.",
     ["CH-05", "CH-03"]),
    ("Q5", "Citas y discurso reproducido",
     "Evaluar el texto completo como aparece en el ejemplo. No atribuir automaticamente al "
     "autor una opinion citada ni asumir que reproducir un insulto equivale a respaldarlo. "
     "Usar UNABLE_TO_DETERMINE cuando no pueda determinarse la funcion de la cita con el "
     "contexto disponible.",
     ["CH-05"]),
    ("Q6", "Alcance de UNABLE_TO_DETERMINE",
     "Permitido ante falta de contexto o ambiguedad normativa/semantica material. Se "
     "asigna por etiqueta. Registrar una nota que explique el motivo cuando se utilice. No "
     "forzar una respuesta binaria solo porque el dataset original sea binario.",
     ["CH-02", "CH-04"]),
    ("Q7", "Duda por etiqueta y duda por tweet",
     "Mantener ambas separadas. Las tres anotaciones HS/TR/AG se registran "
     "independientemente. Cuando exista una dificultad que afecte a todo el tweet, "
     "registrarla en una nota general adicional (columna tweet_note) sin sustituir las "
     "respuestas individuales.",
     ["CH-04", "CH-09"]),
    ("Q8", "Formato",
     "Usar exclusivamente POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE. No se aceptan valores "
     "0/1 en la anotacion principal.",
     ["CH-01", "CH-02"]),
]

ESTADOS_CORRECCION = [
    ("MODEL_DISAGREEMENT", "un modelo difiere del label original; no dice nada sobre el label."),
    ("ANNOTATOR_DISAGREEMENT", "anotadores humanos difieren entre si sobre un caso."),
    ("POSSIBLE_LABEL_ISSUE", "evidencia acumulada sugiere revisar el label; sin adjudicacion."),
    ("ADJUDICATED_DIFFERENCE", "diferencia revisada y decidida por adjudicacion documentada."),
    ("CONFIRMED_ANNOTATION_ERROR", "error demostrado y registrado tras adjudicacion."),
    ("APPROVED_DATASET_CORRECTION", "correccion aprobada formalmente (autorizacion separada; no ocurre en P92-V)."),
]


def sha256_bytes(datos: bytes) -> str:
    """sha256 de bytes."""
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    """sha256 completo de un archivo (lectura en bloques)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_csv_tolerante(ruta: Path) -> tuple[list[str], list[dict]]:
    """Lee un CSV detectando el delimitador (',' o ';')."""
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        primera = fh.readline()
        delim = ";" if primera.count(";") > primera.count(",") else ","
        fh.seek(0)
        lector = csv.DictReader(fh, delimiter=delim)
        return list(lector.fieldnames or []), list(lector)


def leer_train() -> tuple[dict[int, str], set[int]]:
    """id -> texto y conjunto de ids del corpus etiquetado (CSV estandar, quotechar)."""
    id_texto: dict[int, str] = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if not campos:
                continue
            ident = int(campos[0].strip().strip('"'))
            id_texto[ident] = campos[1]
    return id_texto, set(id_texto)


def leer_paquete_v1(nombre: str) -> list[dict]:
    """Filas del paquete v1 (A o B)."""
    with open(V1 / f"annotator_{nombre}.csv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------- guia congelada
def construir_guia() -> tuple[str, str]:
    """Construye la guia congelada; devuelve (contenido final, sha256 sin la linea de hash)."""
    pre = []
    pre.append("# P92 - Guia de anotacion operativa v2 CONGELADA (piloto interno)")
    pre.append("")
    pre.append(f"POLICY_ID: {POLICY_ID}")
    pre.append("Version de guia: v2 (valor de columna guideline_version: \"v2\")")
    pre.append(f"Fecha de congelacion: {FECHA}")
    post = []
    post.append("")
    post.append("## 0. Naturaleza de este documento (leer primero)")
    post.append("")
    post.append("- Esta guia es una POLITICA OPERATIVA INTERNA del proyecto para el piloto de")
    post.append("  anotacion principal. NO es una definicion oficial de la competencia y NO")
    post.append("  resuelve las ambiguedades normativas oficiales (seccion 6).")
    post.append("- La palabra \"frozen\" (congelada) se refiere exclusivamente a que esta version")
    post.append("  queda fijada para este piloto. No significa que las ambiguedades oficiales")
    post.append("  esten resueltas.")
    post.append("- Separacion de etiquetas de naturaleza usada en todo el documento:")
    post.append("")
    post.append("```text")
    post.append("OFFICIAL_DEFINITION        -> texto oficial verbatim (docs/competition_rules.md)")
    post.append("PROJECT_OPERATIONAL_RULE   -> regla operativa adoptada por el propietario del")
    post.append("                              proyecto (2026-10-13); no oficial")
    post.append("UNRESOLVED_POLICY_QUESTION -> pregunta normativa que sigue abierta")
    post.append("```")
    post.append("")
    post.append("## 1. Definiciones oficiales (OFFICIAL_DEFINITION, verbatim)")
    post.append("")
    post.append("- **HS**: \"indica si hay discurso de odio (1 = si, 0 = no)\".")
    post.append("- **TR**: \"indica si el discurso esta dirigido contra una persona o grupo (1 = si, 0 = no)\".")
    post.append("- **AG**: \"indica si el texto es agresivo (1 = si, 0 = no)\".")
    post.append("")
    post.append("Estas definiciones son la UNICA fuente normativa disponible. No existen criterios")
    post.append("operativos oficiales, guia de anotadores original ni ejemplos resueltos por el")
    post.append("organizador (verificado en fases previas).")
    post.append("")
    post.append("## 2. Reglas operativas del proyecto (PROJECT_OPERATIONAL_RULE, adoptadas 2026-10-13)")
    post.append("")
    for qid, titulo, texto, _ch in DECISIONES:
        post.append(f"### {qid} - {titulo}")
        post.append("")
        post.append(texto)
        post.append("")
    post.append("## 3. Unidad de anotacion y esquema de columnas")
    post.append("")
    post.append("- Unidad: el tweet completo, tal como aparece; no normalices el texto.")
    post.append("- Las tres etiquetas (HS, TR, AG) se deciden de forma independiente entre si.")
    post.append("- Columnas del CSV:")
    post.append("")
    post.append("```text")
    post.append("review_id            identificador opaco; NO modificar")
    post.append("text                 texto del tweet; NO modificar")
    post.append("HS_annotation        POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE")
    post.append("TR_annotation        POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE")
    post.append("AG_annotation        POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE")
    post.append("uncertainty_notes    motivo cuando uses UNABLE_TO_DETERMINE; aclaraciones por etiqueta")
    post.append("tweet_note           nota general SOLO si la dificultad afecta a todo el tweet")
    post.append("guideline_version    no modificar (\"v2\")")
    post.append("```")
    post.append("")
    post.append("## 4. Procedimiento por etiqueta e instrucciones exactas de UNABLE_TO_DETERMINE")
    post.append("")
    post.append("Para cada etiqueta, por separado:")
    post.append("")
    post.append("1. Lee el texto completo.")
    post.append("2. Aplica la definicion oficial y las reglas operativas de la seccion 2.")
    post.append("3. Si puedes decidir con responsabilidad: POSITIVE o NEGATIVE.")
    post.append("4. Si no puedes: UNABLE_TO_DETERMINE y motivo en uncertainty_notes.")
    post.append("")
    post.append("UNABLE_TO_DETERMINE (UTD):")
    post.append("")
    post.append("- No es una tercera clase semantica: declara que no puedes asignar POSITIVE ni")
    post.append("  NEGATIVE con responsabilidad usando la definicion oficial y estas reglas.")
    post.append("- Se decide POR ETIQUETA: puedes usar UTD en HS y no en TR, etc.")
    post.append("- Casos tipicos: falta el tweet al que responde; referencia o apodo no")
    post.append("  identificable sin contexto; ambiguedad normativa material (p. ej., ofensa")
    post.append("  sexista individual frente a discurso de odio, mientras la seccion 6 siga")
    post.append("  abierta); funcion de una cita no determinable.")
    post.append("- Al usar UTD, escribe el motivo en uncertainty_notes (obligatorio).")
    post.append("- NUNCA conviertas UTD en NEGATIVE ni POSITIVE automaticamente, ni durante la")
    post.append("  anotacion ni en el analisis posterior.")
    post.append("- No fuerces una respuesta binaria solo porque el dataset original sea binario.")
    post.append("- Si la dificultad afecta a TODO el tweet, escribe ademas una nota general en")
    post.append("  tweet_note (sin sustituir las respuestas por etiqueta).")
    post.append("")
    post.append("## 5. Changelog frente a la guia v1 (adoptado)")
    post.append("")
    post.append("| # | Cambio | v1 | v2 congelada | Base |")
    post.append("|---|---|---|---|---|")
    post.append("| CH-01 | Formato | tokens por celda (0/1 no previsto) | tokens exclusivos; 0/1 explicitamente invalido | desviacion de formato documentada en la ronda de calibracion |")
    post.append("| CH-02 | Notas con UTD | nota opcional | nota de motivo obligatoria cuando se usa UTD | calibracion (notas de C y D) |")
    post.append("| CH-03 | Criterios operativos | no existian | Q1/Q3: definicion literal + UTD; sin nuevas definiciones | decision del propietario, 2026-10-13 |")
    post.append("| CH-04 | Duda de fila | no previsto | Q7: UTD por etiqueta + columna tweet_note | el formato 0/1 de la calibracion resulto ambiguo de interpretar |")
    post.append("| CH-05 | Tono, citas y contexto | sin orientacion | Q4/Q5: contenido observable; sin inferir intencion; UTD si el contexto no alcanza | observado en la calibracion (contexto insuficiente y tono) |")
    post.append("| CH-06 | Ejemplos | no habia | esta guia no incluye textos; los ejemplos estructurales quedan en el material interno del proyecto | material interno del proyecto |")
    post.append("| CH-07 | Preguntas normativas | listadas sin gestion | formalizadas (seccion 6); consulta NO solicitada (Q2) | Q2 |")
    post.append("| CH-08 | Independencia HS/TR/AG | recomendada | reforzada; UTD por etiqueta | Q6/Q7 |")
    post.append("| CH-09 | Esquema de columnas | 7 columnas | 8 columnas: se agrega tweet_note | Q7 |")
    post.append("")
    post.append("## 6. Preguntas normativas que siguen abiertas (UNRESOLVED_POLICY_QUESTION)")
    post.append("")
    post.append("1. Frontera odio/ofensa para una ofensa sexista individual dirigida a una")
    post.append("   persona. Evidencia: la ronda de calibracion registro ambiguedad normativa")
    post.append("   material en este punto (6 de 15 filas, con UNABLE_TO_DETERMINE registrado")
    post.append("   por mas de un anotador).")
    post.append("2. Frontera agresivo/vulgar y tratamiento de generalizaciones hostiles sin")
    post.append("   insulto explicito.")
    post.append("3. Relacion formal (si existe) entre HS y TR, y entre HS y AG.")
    post.append("4. Tratamiento normativo de citas, sarcasmo y discurso reproducido.")
    post.append("")
    post.append("Registro de gobernanza (Q2): governance_clarification = NOT_REQUESTED;")
    post.append("official_semantics = UNRESOLVED; internal_pilot = AUTHORIZED. Esta guia permite")
    post.append("registrar incertidumbre mientras estas preguntas sigan abiertas; no las resuelve.")
    post.append("")
    post.append("## 7. Cegamiento y reglas de independencia")
    post.append("")
    post.append("- No consultes ni intentes deducir etiquetas originales, predicciones de")
    post.append("  modelos, umbrales ni material interno del proyecto.")
    post.append("- No busques los tweets en internet.")
    post.append("- No uses herramientas automaticas (LLM, clasificadores) para asignar etiquetas.")
    post.append("- No comentes casos con la otra persona anotadora antes de terminar tu ronda.")
    post.append("")
    post.append("## 8. Registro de desacuerdos y regla de correccion futura")
    post.append("")
    post.append("Los desacuerdos no se resuelven en esta ronda: se conservan y se remiten a una")
    post.append("adjudicacion posterior, separada y documentada. Vocabulario de estados del")
    post.append("proyecto (nada es error automatico):")
    post.append("")
    for estado, definicion in ESTADOS_CORRECCION:
        post.append(f"- {estado}: {definicion}")
    post.append("")
    post.append("Una correccion futura de etiquetas requerira anotaciones humanas independientes,")
    post.append("adjudicacion documentada y autorizacion separada.")
    post.append("")
    post.append("## 9. Entrega")
    post.append("")
    post.append("- Devuelve el CSV completado; no borres filas ni columnas ni cambies el orden.")
    post.append("- Guarda el archivo con el nombre exacto indicado en el README de tu paquete.")
    post.append("")
    sin_hash = "\n".join(pre + post) + "\n"
    h = sha256_bytes(sin_hash.encode("utf-8"))
    linea_hash = f"SHA256 (contenido sin la linea que contiene este hash): {h}"
    contenido = "\n".join(pre + [linea_hash] + post) + "\n"
    return contenido, h


def construir_readme(letra: str) -> str:
    """README del paquete v2 para el anotador/a A o B."""
    lineas = [
        f"# Anotacion principal (piloto interno) - instrucciones para el anotador/a {letra}",
        "",
        "Este paquete es autosuficiente. Contiene tres archivos:",
        "1. README_annotator.md (este archivo)",
        "2. annotation_guidelines_v2_frozen.md (guia operativa v2 congelada:",
        f"   {POLICY_ID})",
        f"3. annotator_{letra}.csv (250 textos; anotaciones vacias)",
        "",
        "## Objetivo",
        "",
        "Anotar, de forma independiente, las tres etiquetas (HS, TR, AG) de 250 textos: 750",
        "anotaciones de etiqueta; 3 respuestas posibles por anotacion (POSITIVE / NEGATIVE /",
        "UNABLE_TO_DETERMINE).",
        "",
        "## Que debes hacer",
        "",
        "1. Lee primero `annotation_guidelines_v2_frozen.md` completo.",
        "2. Revisa los textos SIN comunicarte con la otra persona anotadora.",
        "3. Asigna HS, TR y AG de forma independiente entre si.",
        "4. Usa EXCLUSIVAMENTE `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` en las tres",
        "   columnas de etiquetas. Los valores 0/1 NO son validos en esta ronda.",
        "5. `uncertainty_notes`: escribe el motivo cuando uses UNABLE_TO_DETERMINE; tambien",
        "   puedes anadir aclaraciones breves por etiqueta.",
        "6. `tweet_note`: solo si la dificultad afecta a TODO el tweet (duda general). No",
        "   sustituye las respuestas por etiqueta.",
        "7. No consultes etiquetas originales, modelos, predicciones ni puntuaciones.",
        "8. No cambies `review_id`, `text`, el orden de las filas ni `guideline_version`.",
        "9. Devuelve el CSV completado sin convertirlo a Excel ni alterar el esquema.",
        "",
        "## Ambiguedades que puedes registrar (no las resuelvas tu)",
        "",
        "- HS: odio frente a ofensa.",
        "- AG: agresividad frente a vulgaridad.",
        "- Citas, discurso reproducido, sarcasmo e ironia.",
        "- Contexto insuficiente.",
        "- Cualquier otra duda material: registrala con UNABLE_TO_DETERMINE y una nota.",
        "",
        "## Como devolver tu archivo",
        "",
        f"- Guarda el CSV completado con el nombre exacto: `annotator_{letra}_response.csv`",
        "- Devuelve SOLO tu archivo, por separado.",
        "- No acuerdes respuestas con la otra persona antes de devolver tu archivo.",
        "",
        "## Que pasa despues (protocolo de recepcion)",
        "",
        "Tu archivo se validara (esquema, 250 filas, review_id, texto sin cambios, respuestas",
        "permitidas, guideline_version); tus respuestas originales se almacenan sin cambios.",
        "Los desacuerdos no se resuelven en esta ronda: se conservan y se remiten a una",
        "adjudicacion posterior, separada y documentada. Ninguna discrepancia con material",
        "del proyecto se considera automaticamente un error.",
        "",
        "## Version",
        "",
        f"Guia aplicable: {POLICY_ID} (congelada el {FECHA}). Es una politica operativa",
        "interna del proyecto; no una definicion oficial de la competencia.",
    ]
    return "\n".join(lineas) + "\n"


def escribir_csv_v2(filas_v1: list[dict], destino: Path) -> None:
    """CSV v2: mismos review_id/text que v1, anotaciones vacias, columnas v2."""
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh, lineterminator="\n")
        escritor.writerow(COLUMNAS_V2)
        for fila in filas_v1:
            escritor.writerow([fila["review_id"], fila["text"], "", "", "", "", "", "v2"])


def escribir_manifiesto(datos: dict) -> None:
    """Escribe el manifiesto en reports/ y su espejo en artifacts/ (bytes identicos)."""
    contenido = json.dumps(datos, ensure_ascii=True, indent=2) + "\n"
    for ruta in (RUTA_MANIFEST, V2 / "manifest.json"):
        with open(ruta, "w", encoding="utf-8") as fh:
            fh.write(contenido)


def anexar_registro(nota: str, hash_actual: str) -> str:
    """Anexa la seccion del propietario al registro (idempotente; reemplaza si el hash cambio)."""
    contenido = RUTA_REGISTRO.read_text(encoding="utf-8")
    if MARCADOR_REGISTRO in contenido:
        if hash_actual in contenido:
            return "ya presente (sin cambios)"
        idx = contenido.find("\n---\n\n## DECISION DEL PROPIETARIO")
        if idx == -1:
            return "presente pero no reemplazable (revisar)"
        RUTA_REGISTRO.write_text(contenido[:idx], encoding="utf-8")
    with open(RUTA_REGISTRO, "a", encoding="utf-8") as fh:
        fh.write(nota)
    return "anexada"


def main() -> None:
    # ---------------------------------------------------------------- carga
    filas_v1 = {"A": leer_paquete_v1("A"), "B": leer_paquete_v1("B")}
    _, clave = leer_csv_tolerante(RUTA_CLAVE)
    id_texto_train, ids_train = leer_train()
    _, holdout_v2 = leer_csv_tolerante(RUTA_HOLDOUT_V2)
    _, holdout2 = leer_csv_tolerante(RUTA_HOLDOUT2)
    _, submission = leer_csv_tolerante(RUTA_SUBMISSION)
    ids_hold_v2 = {int(f["id"]) for f in holdout_v2 if f["particion"] == "holdout"}
    ids_dev_v2 = {int(f["id"]) for f in holdout_v2 if f["particion"] == "dev"}
    ids_hold2 = {int(f["id"]) for f in holdout2}
    ids_test = {int(f["id"]) for f in submission}
    mapa_clave = {f["review_id"]: f for f in clave}

    errores: list[str] = []
    avisos: list[str] = []

    # ------------------------------------------------- verificacion v1
    for letra in ("A", "B"):
        if len(filas_v1[letra]) != 250:
            errores.append(f"v1 {letra}: filas != 250")
    ids_a = [f["review_id"] for f in filas_v1["A"]]
    ids_b = [f["review_id"] for f in filas_v1["B"]]
    if set(ids_a) != set(ids_b):
        errores.append("v1: A y B no tienen los mismos review_id")
    if ids_a == ids_b:
        errores.append("v1: A y B tienen el mismo orden (se esperaba distinto)")

    # id original por review_id y verificacion de textos contra la fuente
    ids_originales = set()
    textos_ok = 0
    for letra in ("A", "B"):
        for fila in filas_v1[letra]:
            rid = fila["review_id"]
            clave_fila = mapa_clave.get(rid)
            if clave_fila is None:
                errores.append(f"v1 {letra}: review_id {rid} sin clave")
                continue
            ident = int(clave_fila["original_example_id"])
            ids_originales.add(ident)
            fuente = id_texto_train.get(ident)
            if fuente is None:
                errores.append(f"v1 {letra}: id {ident} no esta en toxic_train.csv")
            elif fuente != fila["text"]:
                errores.append(f"v1 {letra}: texto de {rid} difiere de la fuente")
            else:
                textos_ok += 1
    if len(ids_originales) != 250:
        errores.append(f"muestra principal: {len(ids_originales)} ids unicos != 250")

    # membresia y exclusiones
    en_train = ids_originales <= ids_train
    en_dev = len(ids_originales & ids_dev_v2)
    en_hold = len(ids_originales & ids_hold_v2)
    en_hold2 = len(ids_originales & ids_hold2)
    en_test = len(ids_originales & ids_test)
    if not en_train:
        errores.append("muestra principal: ids fuera de toxic_train.csv")
    if en_hold:
        errores.append(f"muestra principal: {en_hold} ids en holdout_v2")
    if en_hold2:
        avisos.append(
            f"holdout2: {en_hold2}/250 filas principales pertenecen al pliegue holdout2 "
            "(subconjunto de dev; creado en V10 como pliegue de deteccion de sobreajuste, "
            "explicitamente no virginal; la definicion de exclusion del plan P92 no lo "
            "incluye; ver auditoria, seccion de limitaciones)")
    if en_test:
        errores.append(f"muestra principal: {en_test} ids coinciden con la lista de test")
    if en_dev != 250:
        avisos.append(f"muestra principal: {en_dev}/250 en pool dev (revisar)")

    componentes = {}
    for f in clave:
        if f["review_id"] in set(ids_a):
            componentes[f["sample_component"]] = componentes.get(f["sample_component"], 0) + 1
    ids_calib = {int(f["original_example_id"]) for f in clave
                 if "CALIBR" in f["sample_component"].upper()}
    cal_hold2 = len(ids_calib & ids_hold2)

    # ------------------------------------------------- construir y escribir
    guia, hash_guia = construir_guia()
    readme = {"A": construir_readme("A"), "B": construir_readme("B")}

    V2.mkdir(parents=True, exist_ok=True)
    for letra in ("A", "B"):
        carpeta = V2 / f"annotator_{letra}"
        carpeta.mkdir(parents=True, exist_ok=True)
        escribir_csv_v2(filas_v1[letra], carpeta / f"annotator_{letra}.csv")
        (carpeta / "annotation_guidelines_v2_frozen.md").write_bytes(guia.encode("utf-8"))
        (carpeta / "README_annotator.md").write_text(readme[letra], encoding="utf-8")
    RUTA_GUIA.write_bytes(guia.encode("utf-8"))

    # ------------------------------------------------- verificacion v2
    chequeos: dict[str, bool] = {}
    for letra in ("A", "B"):
        with open(V2 / f"annotator_{letra}" / f"annotator_{letra}.csv",
                  encoding="utf-8", newline="") as fh:
            v2_filas = list(csv.DictReader(fh))
        chequeos[f"v2_{letra}_250_filas"] = len(v2_filas) == 250
        chequeos[f"v2_{letra}_columnas"] = list(v2_filas[0].keys()) == COLUMNAS_V2
        chequeos[f"v2_{letra}_anotaciones_vacias"] = all(
            f[c] == "" for f in v2_filas
            for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                      "uncertainty_notes", "tweet_note"))
        chequeos[f"v2_{letra}_version"] = all(f["guideline_version"] == "v2" for f in v2_filas)
        chequeos[f"v2_{letra}_mismos_textos_v1"] = all(
            f["text"] == orig["text"] and f["review_id"] == orig["review_id"]
            for f, orig in zip(v2_filas, filas_v1[letra]))
    with open(V2 / "annotator_A" / "annotator_A.csv", encoding="utf-8", newline="") as fh:
        v2a = list(csv.DictReader(fh))
    with open(V2 / "annotator_B" / "annotator_B.csv", encoding="utf-8", newline="") as fh:
        v2b = list(csv.DictReader(fh))
    chequeos["v2_mismos_casos_A_B"] = {f["review_id"] for f in v2a} == {f["review_id"] for f in v2b}
    chequeos["v2_orden_distinto_A_B"] = [f["review_id"] for f in v2a] != [f["review_id"] for f in v2b]
    chequeos["v2_membresia_igual_v1"] = (
        [f["review_id"] for f in v2a] == ids_a and [f["review_id"] for f in v2b] == ids_b)
    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    # cegamiento: tokens prohibidos en archivos entregables
    archivos_entregables = []
    for letra in ("A", "B"):
        archivos_entregables += [
            V2 / f"annotator_{letra}" / f"annotator_{letra}.csv",
            V2 / f"annotator_{letra}" / "README_annotator.md",
            V2 / f"annotator_{letra}" / "annotation_guidelines_v2_frozen.md",
        ]
    hallazgos = []
    for letra in ("A", "B"):
        ruta_csv = V2 / f"annotator_{letra}" / f"annotator_{letra}.csv"
        texto_csv = ruta_csv.read_text(encoding="utf-8").lower()
        for token in TOKENS_FUGA + ["p92c-"]:
            if token in texto_csv:
                hallazgos.append(f"{ruta_csv.name}:{token}")
    for letra in ("A", "B"):
        for nombre in ("README_annotator.md", "annotation_guidelines_v2_frozen.md"):
            ruta = V2 / f"annotator_{letra}" / nombre
            texto = ruta.read_text(encoding="utf-8").lower()
            for token in TOKENS_FUGA_INSTRUCCIONES:
                if token in texto:
                    hallazgos.append(f"{nombre}:{token}")
            if any(str(i) in texto for i in ids_originales):
                hallazgos.append(f"{nombre}:id_original")
    chequeos["blinding_sin_tokens_fuga"] = not hallazgos
    if hallazgos:
        errores.append(f"blinding: tokens encontrados {hallazgos}")
    # ids originales dentro de los CSV (fuera de la columna text)
    for letra in ("A", "B"):
        with open(V2 / f"annotator_{letra}" / f"annotator_{letra}.csv",
                  encoding="utf-8", newline="") as fh:
            for fila in csv.DictReader(fh):
                for col in COLUMNAS_V2:
                    if col == "text":
                        continue
                    if any(str(i) in fila[col] for i in ids_originales):
                        errores.append(f"blinding: id original en {col} de {fila['review_id']}")

    # ASCII de todos los artefactos generados (el manifiesto se evalua tras escribirlo)
    generados = archivos_entregables + [RUTA_GUIA]
    no_ascii = [r.name for r in generados if any(b > 127 for b in r.read_bytes())]
    chequeos["ascii_puro"] = not no_ascii
    if no_ascii:
        errores.append(f"bytes > 127 en {no_ascii}")

    # las copias de la guia en los paquetes deben ser identicas al archivo de reports/
    chequeos["guia_copias_identicas"] = all(
        (V2 / f"annotator_{letra}" / "annotation_guidelines_v2_frozen.md").read_bytes()
        == guia.encode("utf-8")
        for letra in ("A", "B")
    )
    if not chequeos["guia_copias_identicas"]:
        errores.append("las copias de la guia en los paquetes difieren del original")

    # ------------------------------------------------- hashes (antes del manifiesto)
    hashes = {}
    for ruta in archivos_entregables:
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)
    hashes[str(RUTA_GUIA.relative_to(RAIZ))] = sha256_archivo(RUTA_GUIA)
    hashes["v1/annotator_A.csv"] = sha256_archivo(V1 / "annotator_A.csv")
    hashes["v1/annotator_B.csv"] = sha256_archivo(V1 / "annotator_B.csv")

    # ------------------------------------------------- manifiesto
    manifiesto = {
        "package": "human_annotation_v2",
        "policy_id": POLICY_ID,
        "guideline_file": str(RUTA_GUIA.relative_to(RAIZ)),
        "guideline_sha256": hash_guia,
        "guideline_sha256_convencion": "sha256 del contenido sin la linea que contiene el hash",
        "date": FECHA,
        "counts": {
            "annotator_rows": 250,
            "principal_tweets": 250,
            "cells_per_annotator": 750,
            "tweets_componente_A": componentes.get("A", 0),
            "tweets_componente_B": componentes.get("B", 0),
        },
        "guideline_version_csv": "v2",
        "source": {
            "membership": "identica a human_annotation_v1 (mismos 250 review_id; orden por anotador preservado de v1; ordenes A y B distintos)",
            "texts": "identicos a data/raw/toxic_train.csv (verificado fila a fila, 500 comprobaciones)",
            "v1_annotator_A_sha256": hashes["v1/annotator_A.csv"],
            "v1_annotator_B_sha256": hashes["v1/annotator_B.csv"],
        },
        "blinding": {
            "hidden": ["original_example_id", "original_labels", "oof_probabilities",
                       "v03_predictions", "threshold_distances", "fold", "sample_component",
                       "selection_reasons", "dup_groups", "internal_key", "other_annotator_responses"],
            "visible": ["review_id", "text", "empty annotation columns",
                        "uncertainty_notes", "tweet_note", "guideline_version"],
        },
        "files": hashes,
        "distribution": "NOT_DISTRIBUTED",
        "limitations": [
            "politica operativa interna; no definicion oficial de la competencia",
            "official_semantics = UNRESOLVED (sin aclaracion solicitada)",
            "componente B no representativa (heredado de v1)",
            "sin adjudicacion aun; ningun gold standard",
        ],
        "terminology_note": "750 anotaciones de etiqueta por anotador; 3 respuestas posibles por anotacion",
    }
    escribir_manifiesto(manifiesto)

    # ------------------------------------------------- decisiones y registro
    lineas = []
    lineas.append("# P92-V - Decisiones de politica operativa del proyecto (Q1-Q8)")
    lineas.append("")
    lineas.append(f"POLICY_ID: {POLICY_ID}")
    lineas.append(f"Fecha de decision: {FECHA}. Decision adoptada por: propietario del proyecto.")
    lineas.append("Naturaleza: POLITICA OPERATIVA INTERNA para el piloto de anotacion. NO son")
    lineas.append("definiciones oficiales de la competencia. No consta una discusion de consenso")
    lineas.append("entre los cuatro anotadores de calibracion; estas decisiones no se les")
    lineas.append("atribuyen.")
    lineas.append("")
    lineas.append("## 0. Registro de gobernanza (Q2)")
    lineas.append("")
    lineas.append("```text")
    lineas.append("governance_clarification = NOT_REQUESTED")
    lineas.append("official_semantics = UNRESOLVED")
    lineas.append("internal_pilot = AUTHORIZED")
    lineas.append("```")
    lineas.append("")
    lineas.append("## 1. Decisiones Q1-Q8 (operativas)")
    lineas.append("")
    for qid, titulo, texto, ch in DECISIONES:
        lineas.append(f"### {qid} - {titulo}")
        lineas.append("")
        lineas.append(f"- decision: {texto}")
        lineas.append("- naturaleza: PROJECT_OPERATIONAL_RULE (no oficial)")
        lineas.append(f"- cambios de guia asociados: {', '.join(ch)}")
        lineas.append("")
    lineas.append("## 2. Regla de correccion futura (vocabulario de estados)")
    lineas.append("")
    lineas.append("Ninguna discrepancia entre la anotacion humana y el label original es un error")
    lineas.append("confirmado. Estados diferenciados:")
    lineas.append("")
    for estado, definicion in ESTADOS_CORRECCION:
        lineas.append(f"- {estado}: {definicion}")
    lineas.append("")
    lineas.append("Una correccion futura necesitara anotaciones humanas independientes,")
    lineas.append("adjudicacion documentada y autorizacion separada. Ninguna etiqueta original se")
    lineas.append("modifica en P92-V y no se entrenaran modelos con las nuevas anotaciones.")
    lineas.append("")
    lineas.append("## 3. Consecuencias y artefactos")
    lineas.append("")
    lineas.append(f"- Guia congelada: reports/p92_annotation_guidelines_v2_frozen.md")
    lineas.append(f"  (sha256 sin linea de hash: {hash_guia}).")
    lineas.append("- Paquetes: artifacts/human_annotation_v2/annotator_A/ y annotator_B/")
    lineas.append("  (CSV vacio + guia congelada + README). NO distribuidos.")
    lineas.append("- Las respuestas de calibracion de los cuatro anotadores permanecen bajo la")
    lineas.append("  guia v1 (historico); no se reinterpretan retroactivamente.")
    RUTA_DECISIONES.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    nota_registro = """
---

## DECISION DEL PROPIETARIO DEL PROYECTO (P92-V, 2026-10-13): politica operativa v2 congelada

No consta una discusion de consenso entre los cuatro anotadores; estas decisiones fueron
adoptadas por el propietario del proyecto como POLITICA OPERATIVA INTERNA para el piloto,
no como definiciones oficiales. Registro por pregunta (detalle en
reports/p92v_operational_policy_decisions.md):

| pregunta | decision adoptada (operativa) | decision_owner | decision_date | status | cambios |
|---|---|---|---|---|---|
| Q1 | definicion oficial literal + UTD explicito; sin nueva definicion de odio | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-03, CH-07 |
| Q2 | no esperar aclaracion externa; NOT_REQUESTED / UNRESOLVED / AUTHORIZED | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-07 |
| Q3 | no equiparar vulgaridad, insulto y agresividad; UTD si no hay evidencia | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-03 |
| Q4 | contenido observable; sin inferir intencion; UTD si el tono decide y falta contexto | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-05, CH-03 |
| Q5 | texto completo; no atribuir la cita al autor; UTD si la funcion no es determinable | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-05 |
| Q6 | UTD por etiqueta (contexto o ambiguedad material); con nota; sin forzar binario | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-02, CH-04 |
| Q7 | dudas separadas: por etiqueta + nota general adicional (tweet_note) | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-04, CH-09 |
| Q8 | formato exclusivo POSITIVE/NEGATIVE/UNABLE_TO_DETERMINE; 0/1 no valido | propietario del proyecto | 2026-10-13 | DECIDED (operativo) | CH-01, CH-02 |

Consecuencias: guia congelada reports/p92_annotation_guidelines_v2_frozen.md
(PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2; sha256 sin linea de hash: %s); paquetes v2
preparados en artifacts/human_annotation_v2/ (NO distribuidos). Las respuestas de
calibracion permanecen bajo la guia v1 y no se reinterpretan retroactivamente.
""" % hash_guia
    estado_registro = anexar_registro(nota_registro, hash_guia)

    # ------------------------------------------- manifiesto: hashes y espejo
    hash_manifest = sha256_archivo(RUTA_MANIFEST)
    chequeos["manifest_espejo_identico"] = (
        (V2 / "manifest.json").read_bytes() == RUTA_MANIFEST.read_bytes())
    if not chequeos["manifest_espejo_identico"]:
        errores.append("el espejo del manifiesto difiere del original")
    no_ascii_manifest = [r.name for r in (RUTA_MANIFEST, V2 / "manifest.json")
                         if any(b > 127 for b in r.read_bytes())]
    if no_ascii_manifest:
        errores.append(f"bytes > 127 en {no_ascii_manifest}")

    # ------------------------------------------------- auditoria
    estado_final = "MAIN_ANNOTATION_PACKAGES_READY" if not errores else "MAIN_ANNOTATION_PACKAGE_BLOCKED"
    lineas = []
    lineas.append("# P92-V - Auditoria del paquete principal v2")
    lineas.append("")
    lineas.append(f"Fecha: {FECHA}. Estado: **{estado_final}**.")
    lineas.append("Fase diagnostica: sin entrenamiento, sin predicciones, sin modificacion de")
    lineas.append("etiquetas, sin acceso a test ni holdouts, sin leaderboard, sin submissions.")
    lineas.append("")
    lineas.append("## 1. Membresia y procedencia")
    lineas.append("")
    lineas.append("- Mismos 250 review_id que human_annotation_v1 (por anotador); orden por")
    lineas.append("  anotador preservado de v1; ordenes de A y B distintos entre si.")
    lineas.append(f"- Componentes: A={componentes.get('A', 0)}, B={componentes.get('B', 0)}.")
    lineas.append(f"- Todos los original_example_id en toxic_train.csv: {en_train}; en pool dev: {en_dev}/250.")
    lineas.append(f"- Excluidos de la particion holdout de holdout_v2: {en_hold} coincidencias;")
    lineas.append(f"  lista de test (plantilla submission.csv, solo columna id): {en_test} coincidencias.")
    lineas.append(f"- holdout2 (pliegue de dev creado en V10 para deteccion de sobreajuste;")
    lineas.append(f"  explicitamente no virginal; disjunto del holdout de holdout_v2): {en_hold2}/250")
    lineas.append(f"  filas principales y {cal_hold2} filas de calibracion coinciden. La definicion de")
    lineas.append("  elegibilidad del plan P92 (aprobada con 14/14) excluye la particion holdout de")
    lineas.append("  holdout_v2 y el test; no excluye este pliegue de dev. Limitacion heredada y")
    lineas.append("  documentada; esta fase reutiliza los mismos 250 por mandato. La anotacion de")
    lineas.append("  calidad no consume holdout2 como set de evaluacion (sin reentrenamiento;")
    lineas.append("  cualquier correccion futura requeriria autorizacion separada y consideraria")
    lineas.append("  la integridad de evaluacion de estas filas).")
    lineas.append("- toxic_test.csv NO accedido (ni texto ni etiquetas ni ids del archivo).")
    lineas.append("")
    lineas.append("## 2. Verificacion de textos")
    lineas.append("")
    lineas.append(f"- Comparaciones texto contra la fuente elegible (toxic_train.csv): {textos_ok}/500 OK.")
    lineas.append("- Los CSV v2 replican byte a byte los textos de v1 (mismos review_id).")
    lineas.append("")
    lineas.append("## 3. Estado de las anotaciones y esquema")
    lineas.append("")
    lineas.append("- Columnas: review_id, text, HS_annotation, TR_annotation, AG_annotation,")
    lineas.append("  uncertainty_notes, tweet_note, guideline_version (8 columnas; CH-09).")
    lineas.append("- Anotaciones y notas VACIAS en ambos archivos; guideline_version = v2.")
    lineas.append("")
    lineas.append("## 4. Cegamiento")
    lineas.append("")
    lineas.append("- Escaneo de fuga (tokens identificadores de material interno: ids originales,")
    lineas.append("  componentes, motivos de seleccion, folds, OOF, V03, ids P92R/P92C dentro de")
    lineas.append(f"  las instrucciones): hallazgos: {hallazgos if hallazgos else 'ninguno'}.")
    lineas.append("- Nota de metodo: una primera pasada con tokens genericos detecto dos falsos")
    lineas.append("  positivos (un nombre de usuario que contiene 'oof' dentro de un texto,")
    lineas.append("  verificado contra la fuente; y la palabra 'umbrales' en la lista de")
    lineas.append("  prohibiciones de la guia). El escaneo se refino a tokens identificadores;")
    lineas.append("  los textos se verifican ademas byte a byte contra la fuente elegible.")
    lineas.append("- Sin etiquetas originales, ids originales, OOF, predicciones, umbrales, folds,")
    lineas.append("  motivos de seleccion ni grupos; sin clave interna.")
    lineas.append("")
    lineas.append("## 5. Chequeos")
    lineas.append("")
    for clave_chk, valor in chequeos.items():
        lineas.append(f"- {clave_chk}: {'OK' if valor else 'FALLO'}")
    lineas.append("")
    lineas.append("## 6. Hashes")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for ruta_txt, h in sorted(hashes.items()):
        lineas.append(f"| {ruta_txt} | `{h}` |")
    lineas.append(f"| reports/p92_annotation_guidelines_v2_frozen.md (hash interno) | `{hash_guia}` |")
    lineas.append(f"| reports/p92v_manifest.json | `{hash_manifest}` |")
    lineas.append(f"| artifacts/human_annotation_v2/manifest.json (espejo identico) | `{hash_manifest}` |")
    lineas.append("")
    lineas.append("## 7. Resultado")
    lineas.append("")
    if errores:
        lineas.append("Errores detectados (paquete BLOQUEADO):")
        for e in errores:
            lineas.append(f"- {e}")
    else:
        lineas.append("- Sin errores: MAIN_ANNOTATION_PACKAGES_READY (NO distribuidos).")
    if avisos:
        lineas.append("")
        lineas.append("Avisos:")
        for a in avisos:
            lineas.append(f"- {a}")
    RUTA_AUDITORIA.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------- resumen
    print("P92-V politica v2 y paquetes: ejecutado")
    print(f"  guia congelada: {RUTA_GUIA.relative_to(RAIZ)} | sha256: {hash_guia[:16]}")
    print(f"  registro de decisiones: {estado_registro}")
    print(f"  textos verificados: {textos_ok}/500 | componentes A={componentes.get('A', 0)} B={componentes.get('B', 0)}")
    print(f"  exclusiones: holdout_v2={en_hold} holdout2={en_hold2} test={en_test}")
    print(f"  chequeos v2: {sum(chequeos.values())}/{len(chequeos)} OK")
    print(f"  errores: {errores if errores else 'ninguno'}")
    print(f"  estado final: {estado_final}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
