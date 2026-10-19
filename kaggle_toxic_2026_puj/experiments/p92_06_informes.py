#!/usr/bin/env python
"""
P92-T - Generacion de informes del analisis de calibracion (fase diagnostica).

Entradas (solo lectura):
  reports/p92t_audit_results.json
  artifacts/human_annotation_v1/calibration_responses/calibration_response_{A,B,C,D}.csv

Salidas (ASCII puro, no sobrescriben informes de recepcion):
  reports/p92_calibration_agreement.csv
  reports/p92_calibration_case_review.csv
  reports/p92_calibration_analysis.md
  reports/p92_annotation_guidelines_v2_proposal.md
  reports/p92_calibration_discussion_pack.md

Reglas: sin etiquetas originales del dataset, sin informacion de modelos, sin
adjudicacion automatica; las notas humanas se citan transliteradas a ASCII y los
originales con acentos permanecen en los archivos de respuesta.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESP = RAIZ / "artifacts" / "human_annotation_v1" / "calibration_responses"
RUTA_JSON = RAIZ / "reports" / "p92t_audit_results.json"

COLUMNAS_CLASE = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}
ETIQUETAS = ("HS", "TR", "AG")


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


def cargar_respuestas() -> tuple[dict[str, dict[str, dict]], dict[str, str]]:
    """Devuelve respuestas crudas por anotador y textos por review_id."""
    respuestas: dict[str, dict[str, dict]] = {}
    textos: dict[str, str] = {}
    for anotador in ("A", "B", "C", "D"):
        with open(
            DIR_RESP / f"calibration_response_{anotador}.csv", encoding="utf-8-sig", newline=""
        ) as fh:
            filas = list(csv.DictReader(fh))
        respuestas[anotador] = {
            f["review_id"]: {
                "HS": f["HS_annotation"],
                "TR": f["TR_annotation"],
                "AG": f["AG_annotation"],
                "notas": f["uncertainty_notes"],
            }
            for f in filas
        }
        for f in filas:
            textos[f["review_id"]] = f["text"]
    return respuestas, textos


def interpretar_ab(valor: str) -> str:
    """Codificacion confirmada: 0 = NEGATIVE, 1 = POSITIVE."""
    return {"0": "NEGATIVE", "1": "POSITIVE"}[valor]


# --------------------------------------------------------------------- casos
# Clasificacion estructurada de los casos con duda (revision humana de notas).
CASOS = [
    {
        "rid": "P92C-03", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY",
        "gap": ("v1 no define criterio operativo para decidir si una ofensa sexista individual "
                "constituye discurso de odio (limitacion 7.1 de v1, confirmada empiricamente)."),
        "regla": ("Discutir en la ronda humana: O1 estricta (HS solo por identidad/pertenencia), "
                  "O2 amplia (toda degradacion hostil dirigida), O3 literal (definicion oficial + "
                  "UTD cuando no decida). Posible consulta al organizador."),
        "confianza": "alta",
        "resumen": ("D: insulto sexista dirigido a una persona; la guia no aclara cuando una ofensa "
                    "sexista individual constituye discurso de odio; la alusion a la hoguera anade "
                    "hostilidad violenta. C respondio POSITIVE con nota descriptiva (sin duda); "
                    "A y B discreparon (NEGATIVE vs POSITIVE)."),
    },
    {
        "rid": "P92C-04", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY; INSUFFICIENT_CONTEXT",
        "gap": ("v1 no define criterio operativo odio vs. ofensa (7.1) ni regla para referencias no "
                "identificables sin contexto (7.4)."),
        "regla": ("Discutir O1/O2/O3 (odio vs. ofensa) y regla candidata de contexto: UTD + nota "
                  "cuando la referencia no es identificable. Posible consulta al organizador."),
        "confianza": "alta",
        "resumen": ("D: 'el gnomo' no puede identificarse sin contexto; hay hostilidad y referencia "
                    "explicita a la guillotina, pero no queda claro si constituye discurso de odio. "
                    "C: duda normativa odio vs. ofensa (el odio no es por pertenencia a grupo) y "
                    "respondio POSITIVE. A marco la fila como dificil para cualquier clase."),
    },
    {
        "rid": "P92C-05", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY",
        "gap": ("v1 no define criterio operativo para decidir si una ofensa sexista individual "
                "constituye discurso de odio (7.1)."),
        "regla": ("Discutir O1/O2/O3. Posible consulta al organizador."),
        "confianza": "alta",
        "resumen": ("C: insulto sexista directo; duda normativa odio vs. ofensa. D: insulto sexista "
                    "dirigido a una persona; la guia no distingue operativamente entre ofensa sexista "
                    "individual y discurso de odio."),
    },
    {
        "rid": "P92C-07", "etiqueta": "HS",
        "tipo": "INSUFFICIENT_CONTEXT",
        "gap": ("v1 no especifica como proceder cuando falta el contexto necesario para identificar "
                "el blanco (7.4)."),
        "regla": ("Regla candidata: UTD + nota cuando falta contexto para identificar el blanco; no "
                  "adivinar. Discutir en la ronda humana."),
        "confianza": "alta",
        "resumen": ("C: hostil hacia descendientes de alguien (falta contexto de quien es el "
                    "'genocida'); tono mas acusatorio que agresivo. D respondio NEGATIVE en HS y "
                    "marco UTD en AG (contexto). B marco la fila."),
    },
    {
        "rid": "P92C-09", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY",
        "gap": ("v1 no define criterio operativo para decidir si una ofensa sexista individual "
                "constituye discurso de odio (7.1)."),
        "regla": ("Discutir O1/O2/O3. Posible consulta al organizador."),
        "confianza": "alta",
        "resumen": ("C: insulto sexista a una persona; duda normativa odio vs. ofensa. D: la guia no "
                    "aclara cuando una ofensa sexista individual constituye discurso de odio."),
    },
    {
        "rid": "P92C-11", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY",
        "gap": ("v1 no define criterio operativo para decidir si una ofensa sexista individual "
                "constituye discurso de odio (7.1)."),
        "regla": ("Discutir O1/O2/O3. Posible consulta al organizador."),
        "confianza": "alta",
        "resumen": ("C: insulto sexista a una persona; duda normativa odio vs. ofensa. D: insulto "
                    "sexista dirigido a destinatarios identificados por menciones; la guia no "
                    "establece si este tipo de ofensa individual constituye discurso de odio."),
    },
    {
        "rid": "P92C-13", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY; INSULT_VS_AGGRESSION",
        "gap": ("v1 no define criterio operativo odio vs. ofensa (7.1) ni distingue agresivo de "
                "vulgar (7.2) ni orienta sobre tono de broma."),
        "regla": ("Discutir O1/O2/O3 y regla de tono: el registro de broma o coloquial no decide por "
                  "si solo; evaluar el contenido expresado."),
        "confianza": "media",
        "resumen": ("C: insulto leve ('un poco zorra'), podria ser broma entre conocidos; duda "
                    "agresivo vs. vulgar. D: insulto sexista dirigido a una persona; la guia no "
                    "distingue operativamente entre ofensa sexista individual y discurso de odio."),
    },
    {
        "rid": "P92C-14", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY",
        "gap": ("v1 no define criterio operativo para decidir si una ofensa sexista individual "
                "constituye discurso de odio (7.1)."),
        "regla": ("Discutir O1/O2/O3. Posible consulta al organizador."),
        "confianza": "alta",
        "resumen": ("C: insultos fuertes dirigidos a una persona; duda normativa odio vs. ofensa. D: "
                    "ataque personal con insultos sexistas y vulgares y ordenes hostiles; la guia no "
                    "ofrece un criterio operativo para decidir si constituye discurso de odio."),
    },
    {
        "rid": "P92C-15", "etiqueta": "HS",
        "tipo": "HATE_VS_OFFENSE_AMBIGUITY",
        "gap": ("v1 no define criterio operativo para decidir si una ofensa sexista individual "
                "constituye discurso de odio (7.1)."),
        "regla": ("Discutir O1/O2/O3. Posible consulta al organizador."),
        "confianza": "alta",
        "resumen": ("C: insulto sexista y humillacion ('por zorra'); duda normativa odio vs. ofensa. "
                    "D: la guia no aclara cuando una ofensa sexista individual constituye discurso "
                    "de odio."),
    },
    {
        "rid": "P92C-01", "etiqueta": "TR",
        "tipo": "INSUFFICIENT_CONTEXT",
        "gap": ("v1 no especifica el tratamiento de contexto insuficiente para determinar el blanco "
                "(7.4)."),
        "regla": ("Regla candidata: UTD + nota cuando el blanco no es identificable sin contexto. "
                  "Discutir en la ronda humana."),
        "confianza": "alta",
        "resumen": ("C: apodo 'nio polla' sin contexto; tono burlon ('estais tocaisimos'); no se ve a "
                    "quien apunta ni si hay agresion real. D: la referencia no es clara sin contexto; "
                    "'estais tocaisimos' se dirige al interlocutor, pero puede ser burla coloquial o "
                    "insulto agresivo. B marco la fila."),
    },
    {
        "rid": "P92C-01", "etiqueta": "AG",
        "tipo": "INSUFFICIENT_CONTEXT",
        "gap": ("v1 no especifica el tratamiento de contexto insuficiente (7.4)."),
        "regla": ("Regla candidata: UTD + nota cuando la agresion no es inequivoca sin contexto."),
        "confianza": "alta",
        "resumen": ("C (NEGATIVE): apodo sin contexto; no se ve si hay agresion real. D: burla "
                    "coloquial vs. insulto agresivo; sin contexto no es inequivoca."),
    },
    {
        "rid": "P92C-06", "etiqueta": "AG",
        "tipo": "INSULT_VS_AGGRESSION",
        "gap": ("v1 no distingue agresivo de vulgar (7.2) ni define si la generalizacion hostil sin "
                "insulto cuenta como AG."),
        "regla": ("Regla candidata: AG=POSITIVE si hay agresion dirigida (insulto, amenaza, orden "
                  "hostil) a alguien; la generalizacion hostil sin insulto queda como pregunta "
                  "abierta."),
        "confianza": "alta",
        "resumen": ("D: generalizacion negativa contra inmigrantes como amenaza laboral; agresividad "
                    "discutible porque no contiene insulto ni amenaza explicita. C (NEGATIVE): el "
                    "tono no es agresivo."),
    },
    {
        "rid": "P92C-07", "etiqueta": "AG",
        "tipo": "INSUFFICIENT_CONTEXT",
        "gap": ("v1 no orienta sobre sarcasmo/citas ni contexto insuficiente (7.4)."),
        "regla": ("Regla candidata: UTD + nota para sarcasmo o citas sin contexto suficiente."),
        "confianza": "alta",
        "resumen": ("D: critica o sarcasmo; sin el contexto enlazado la agresividad no es "
                    "inequivoca; 'de puta madre' parece expresion idiomatica."),
    },
    {
        "rid": "P92C-13", "etiqueta": "AG",
        "tipo": "INSULT_VS_AGGRESSION",
        "gap": ("v1 no distingue agresivo de vulgar (7.2)."),
        "regla": ("Regla candidata: AG=POSITIVE solo con agresion dirigida; la vulgaridad no "
                  "dirigida no basta; el tono de broma se evalua con el contenido."),
        "confianza": "media",
        "resumen": ("C: insulto leve, posible broma entre conocidos; duda agresivo vs. vulgar. D "
                    "respondio POSITIVE."),
    },
    {
        "rid": "P92C-02", "etiqueta": "ANY_FLAG",
        "tipo": "OTHER",
        "gap": ("v1 no preve un mecanismo de duda a nivel de fila."),
        "regla": ("Regla candidata: si la fila entera es dificil, marcar UTD en las etiquetas "
                  "afectadas + nota (en lugar de un flag binario aparte)."),
        "confianza": "baja",
        "resumen": ("A marco la fila como dificil para cualquier clase (notas=1); motivo no "
                    "especificado. Respuestas de la fila: A 0/0/0; B 0/1/0; C NEG/POS/POS; D "
                    "NEG/POS/POS. C explico su lectura sin expresar duda; D sin notas."),
    },
]


def respuesta_de(respuestas: dict, rid: str, etiqueta: str, anotador: str) -> str:
    """Respuesta en tokens para el anotador (A/B interpretados con la codificacion confirmada)."""
    valor = respuestas[anotador][rid][etiqueta]
    if anotador in ("A", "B"):
        return interpretar_ab(valor)
    return valor


# ------------------------------------------------------------------ escritores
def escribir_acuerdo(res: dict) -> None:
    """CSV de acuerdo por etiqueta y par, con numeradores y denominadores."""
    ruta = RAIZ / "reports" / "p92_calibration_agreement.csv"
    columnas = [
        "label", "fila_tipo", "par", "n",
        "nn", "np", "nu", "pn", "pp", "pu", "un", "up", "uu",
        "acuerdo_3cat_num", "acuerdo_3cat_den", "acuerdo_bin_num", "acuerdo_bin_den",
        "dice_pos_num", "dice_pos_den", "dice_neg_num", "dice_neg_den",
        "kappa_2cat_bin", "kappa_3cat_nom",
        "exacto_4cat_num", "exacto_4cat_den",
        "alpha_utd_categoria", "alpha_utd_faltante",
        "utd_A", "utd_B", "utd_C", "utd_D", "nota",
    ]
    filas = []
    for etiqueta in ETIQUETAS:
        bloque = res["acuerdo"]["por_etiqueta"][etiqueta]
        for par, s in bloque["pares"].items():
            c = s["celdas"]
            filas.append({
                "label": etiqueta, "fila_tipo": "par", "par": par, "n": s["total"],
                "nn": c["NN"], "np": c["NP"], "nu": c["NU"],
                "pn": c["PN"], "pp": c["PP"], "pu": c["PU"],
                "un": c["UN"], "up": c["UP"], "uu": c["UU"],
                "acuerdo_3cat_num": s["acuerdo_3cat"]["num"],
                "acuerdo_3cat_den": s["acuerdo_3cat"]["den"],
                "acuerdo_bin_num": s["acuerdo_binario"]["num"],
                "acuerdo_bin_den": s["acuerdo_binario"]["den"],
                "dice_pos_num": s["acuerdo_positivo_dice"]["num"],
                "dice_pos_den": s["acuerdo_positivo_dice"]["den"],
                "dice_neg_num": s["acuerdo_negativo_dice"]["num"],
                "dice_neg_den": s["acuerdo_negativo_dice"]["den"],
                "kappa_2cat_bin": "" if s["kappa_2cat_binario"] is None else f"{s['kappa_2cat_binario']:.4f}",
                "kappa_3cat_nom": "" if s["kappa_3cat_nominal"] is None else f"{s['kappa_3cat_nominal']:.4f}",
                "exacto_4cat_num": "", "exacto_4cat_den": "",
                "alpha_utd_categoria": "", "alpha_utd_faltante": "",
                "utd_A": "", "utd_B": "", "utd_C": "", "utd_D": "",
                "nota": ("celdas XY: X=primer anotador del par, Y=segundo; N=NEGATIVE, P=POSITIVE, "
                         "U=UNABLE_TO_DETERMINE; A/B interpretados (0/1 confirmado); UTD conservado "
                         "como categoria diferenciada en C y D"),
            })
        utd = bloque["utd_por_anotador"]
        filas.append({
            "label": etiqueta, "fila_tipo": "global_4", "par": "ALL4", "n": 15,
            "nn": "", "np": "", "nu": "", "pn": "", "pp": "", "pu": "", "un": "", "up": "", "uu": "",
            "acuerdo_3cat_num": "", "acuerdo_3cat_den": "",
            "acuerdo_bin_num": "", "acuerdo_bin_den": "",
            "dice_pos_num": "", "dice_pos_den": "", "dice_neg_num": "", "dice_neg_den": "",
            "kappa_2cat_bin": "", "kappa_3cat_nom": "",
            "exacto_4cat_num": bloque["exacto_4cat"]["num"],
            "exacto_4cat_den": bloque["exacto_4cat"]["den"],
            "alpha_utd_categoria": f"{bloque['alpha_utd_categoria']:.4f}",
            "alpha_utd_faltante": f"{bloque['alpha_utd_faltante']:.4f}",
            "utd_A": utd["A"], "utd_B": utd["B"], "utd_C": utd["C"], "utd_D": utd["D"],
            "nota": ("fila global: exacto_4cat = filas con las 4 respuestas identicas (UTD como "
                     "categoria); alpha de Krippendorff (nominal) con dos tratamientos de UTD"),
        })
    with open(ruta, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(filas)} filas)")


def escribir_casos(res: dict, respuestas: dict) -> None:
    """CSV de revision por caso (sin etiquetas originales ni predicciones)."""
    ruta = RAIZ / "reports" / "p92_calibration_case_review.csv"
    columnas = [
        "review_id", "label",
        "annotator_A_response", "annotator_B_response",
        "annotator_C_response", "annotator_D_response",
        "annotator_notes_summary", "ambiguity_type", "guideline_gap",
        "candidate_rule", "confidence_in_interpretation", "requires_human_discussion",
    ]
    filas = []
    for caso in CASOS:
        rid = caso["rid"]
        etiqueta = caso["etiqueta"]
        if etiqueta == "ANY_FLAG":
            respuestas_fila = {"A": "-", "B": "-", "C": "-", "D": "-"}
        else:
            respuestas_fila = {
                a: respuesta_de(respuestas, rid, etiqueta, a) for a in ("A", "B", "C", "D")
            }
        filas.append({
            "review_id": rid, "label": etiqueta,
            "annotator_A_response": respuestas_fila["A"],
            "annotator_B_response": respuestas_fila["B"],
            "annotator_C_response": respuestas_fila["C"],
            "annotator_D_response": respuestas_fila["D"],
            "annotator_notes_summary": ascii_texto(caso["resumen"]),
            "ambiguity_type": caso["tipo"],
            "guideline_gap": ascii_texto(caso["gap"]),
            "candidate_rule": ascii_texto(caso["regla"]),
            "confidence_in_interpretation": caso["confianza"],
            "requires_human_discussion": "si",
        })
    with open(ruta, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(filas)} filas)")


def escribir_analisis(res: dict, respuestas: dict) -> None:
    """Informe principal de analisis de la calibracion."""
    a = res["acuerdo"]
    integridad = res["integridad"]
    cod = res["codificacion_AB"]
    hs = a["por_etiqueta"]["HS"]
    tr = a["por_etiqueta"]["TR"]
    ag = a["por_etiqueta"]["AG"]
    hashes = res["archivos"]["respuestas"]

    lineas = []
    lineas.append("# P92-T - Analisis de la calibracion humana (4 anotadores, 15 tweets)")
    lineas.append("")
    lineas.append("Fecha: 2026-10-12. Fase diagnostica: sin entrenamiento, sin predicciones, sin")
    lineas.append("modificacion de etiquetas, sin acceso a test/holdout, sin leaderboard, sin")
    lineas.append("submissions. Las respuestas humanas originales no se modificaron.")
    lineas.append("")
    lineas.append("Estados:")
    lineas.append("")
    lineas.append("```text")
    lineas.append("CALIBRATION_ANALYZED")
    lineas.append("ENCODING_AB = ENCODING_CONFIRMED (aclaracion del responsable del proyecto)")
    lineas.append("GUIDELINE_V2_PROPOSED_PENDING_DISCUSSION")
    lineas.append("HUMAN_DISCUSSION_PENDING")
    lineas.append("MAIN_ANNOTATION_NOT_DISTRIBUTED")
    lineas.append("V03 = FROZEN | CANONICAL_OOF = FROZEN | TEST = UNTOUCHED | LEADERBOARD = UNUSED")
    lineas.append("NO NEW MODELING PHASE")
    lineas.append("```")
    lineas.append("")
    lineas.append("## 0. Capas de datos (no mezcladas)")
    lineas.append("")
    lineas.append("```text")
    lineas.append("raw response            -> bytes originales de los 4 archivos de respuesta")
    lineas.append("normalized response     -> derivados de A/B con la codificacion confirmada")
    lineas.append("annotation interpretation -> uso de las respuestas como tokens en el analisis")
    lineas.append("adjudicated decision    -> NO EXISTE todavia (etapa posterior y separada)")
    lineas.append("```")
    lineas.append("")
    lineas.append("## 1. Auditoria de las cuatro respuestas")
    lineas.append("")
    lineas.append("| anotador | archivo | sha256 (16) | filas | ids | texto | esquema |")
    lineas.append("|---|---|---|---|---|---|---|")
    for anotador in ("A", "B", "C", "D"):
        h = hashes[anotador]
        lineas.append(
            f"| {anotador} | `{h['path']}` | `{h['sha256_16']}` | 15 | P92C-01..15 | "
            f"integro (identico en los 4 archivos) | "
            f"{'0/1 en clases y notas (desviacion de formato)' if anotador in ('A', 'B') else 'tokens + notas libres'} |"
        )
    lineas.append("")
    lineas.append("- sha256 completo de las respuestas originales:")
    for anotador in ("A", "B", "C", "D"):
        lineas.append(f"  - {anotador}: `{hashes[anotador]['sha256']}`")
    lineas.append("- Dominios crudos observados: A y B = {0, 1} en las cuatro columnas; C y D = "
                   "{POSITIVE, NEGATIVE, UNABLE_TO_DETERMINE} en las columnas de clase y texto "
                   "libre en uncertainty_notes.")
    lineas.append("- Los archivos crudos no cambiaron respecto a lo recibido "
                   f"(raw_annotation_files_unchanged = {integridad['raw_annotation_files_unchanged']}).")
    lineas.append("- Las copias del handoff de A y B coinciden byte a byte con las respuestas "
                   "(editaron en el sitio): hash identico verificado.")
    lineas.append("- Los textos de las cuatro respuestas son identicos entre si para los 15 "
                   f"review_id (textos_identicos = {integridad['original_texts_unchanged']}; "
                   f"sha256 del bloque de textos: `{integridad['textos_sha256'][:16]}`).")
    lineas.append("- `guideline_version` = v1 en los cuatro archivos; la seleccion de calibracion "
                   f"no cambio (calibration_sample_unchanged = {integridad['calibration_sample_unchanged']}).")
    lineas.append("- Desviaciones de formato detectadas (documentadas, no corregidas en el origen):")
    lineas.append("  1. A y B usaron 0/1 en lugar de los tokens exigidos por el README y la guia v1.")
    lineas.append("  2. A y B conservaron BOM de Excel (los archivos C y D no lo tienen).")
    lineas.append("  3. A y B usaron la columna de notas con un valor 0/1; C y D usaron texto libre.")
    lineas.append("")
    lineas.append("## 2. Codificacion de A y B: resultado de la auditoria")
    lineas.append("")
    lineas.append(f"Resultado: **{cod['clasificacion']}**.")
    lineas.append("")
    lineas.append(f"Regla establecida: {ascii_texto(cod['regla'])}")
    lineas.append("")
    lineas.append(f"Fuente: {ascii_texto(cod['fuente'])}.")
    lineas.append("")
    lineas.append("Evidencia revisada:")
    for punto in cod["evidencia"]:
        lineas.append(f"- {ascii_texto(punto)}")
    lineas.append("")
    lineas.append(f"{ascii_texto(cod['no_uso_de_cd'])}")
    lineas.append("")
    lineas.append(f"Tratamiento en metricas: {ascii_texto(cod['tratamiento_en_metricas'])}")
    lineas.append("")
    lineas.append(f"Flags de fila registrados (notas=1): A = {cod['flags_fila']['A']}; "
                   f"B = {cod['flags_fila']['B']}.")
    lineas.append("")
    lineas.append("Observacion de consistencia (NO fundamento de la codificacion): 4 de las 5 filas")
    lineas.append("marcadas por A o B coinciden con filas donde C o D usaron UNABLE_TO_DETERMINE;")
    lineas.append("la unica fila marcada sin UTD de C/D es P92C-02. El contraste de los archivos")
    lineas.append("normalizados derivados contra la interpretacion coincide en todas las celdas")
    lineas.append(f"de clase (45 por anotador): A={cod['contraste_con_normalizados']['A']}, "
                   f"B={cod['contraste_con_normalizados']['B']}.")
    lineas.append("")
    lineas.append("## 3. Acuerdo entre anotadores (respuestas interpretadas)")
    lineas.append("")
    lineas.append("Denominador: 15 filas por etiqueta. UTD se conserva como categoria diferenciada")
    lineas.append("para C y D; A y B no emitieron UTD en las columnas de clase (solo flags de fila).")
    lineas.append("")
    lineas.append("### 3.1 Acuerdo exacto (3 categorias) y por par")
    lineas.append("")
    lineas.append("| etiqueta | AB | AC | AD | BC | BD | CD |")
    lineas.append("|---|---|---|---|---|---|---|")
    for etiqueta, bloque in (("HS", hs), ("TR", tr), ("AG", ag)):
        valores = " | ".join(
            f"{bloque['pares'][p]['acuerdo_3cat']['num']}/15" for p in ("AB", "AC", "AD", "BC", "BD", "CD")
        )
        lineas.append(f"| {etiqueta} | {valores} |")
    lineas.append("")
    lineas.append("### 3.2 Acuerdo solo sobre celdas binarias (UTD excluido) y acuerdo positivo/negativo")
    lineas.append("")
    lineas.append("| etiqueta | par | binarias | acuerdo bin | dice positivo | dice negativo |")
    lineas.append("|---|---|---|---|---|---|")
    for etiqueta, bloque in (("HS", hs), ("TR", tr), ("AG", ag)):
        for par in ("AB", "AC", "AD", "BC", "BD", "CD"):
            s = bloque["pares"][par]
            lineas.append(
                f"| {etiqueta} | {par} | {s['acuerdo_binario']['den']}/15 | "
                f"{s['acuerdo_binario']['num']}/{s['acuerdo_binario']['den']} | "
                f"{s['acuerdo_positivo_dice']['num']}/{s['acuerdo_positivo_dice']['den']} | "
                f"{s['acuerdo_negativo_dice']['num']}/{s['acuerdo_negativo_dice']['den']} |"
            )
    lineas.append("")
    lineas.append("### 3.3 Kappas y combinacion exacta de los 4")
    lineas.append("")
    lineas.append("| etiqueta | kappa CD (bin) | kappa CD (3cat) | exacto 4/15 | alpha UTD=categoria | alpha UTD=faltante |")
    lineas.append("|---|---|---|---|---|---|")
    for etiqueta, bloque in (("HS", hs), ("TR", tr), ("AG", ag)):
        cd = bloque["pares"]["CD"]
        k2 = "n/d" if cd["kappa_2cat_binario"] is None else f"{cd['kappa_2cat_binario']:.3f}"
        k3 = "n/d" if cd["kappa_3cat_nominal"] is None else f"{cd['kappa_3cat_nominal']:.3f}"
        lineas.append(
            f"| {etiqueta} | {k2} | {k3} | {bloque['exacto_4cat']['num']}/15 | "
            f"{bloque['alpha_utd_categoria']:.3f} | {bloque['alpha_utd_faltante']:.3f} |"
        )
    lineas.append("")
    lineas.append("Matrices completas por par (celdas 3x3) y kappas de los seis pares: "
                   "`reports/p92_calibration_agreement.csv`. Las alfas coinciden con las")
    lineas.append("registradas en P92-S2 (reconciliacion: HS TR AG = True/True/True en ambos tratamientos).")
    lineas.append("")
    lineas.append("Lectura descriptiva de esta muestra (15 filas; no extrapolable):")
    lineas.append("- El par C-D tiene el acuerdo mas alto en las tres etiquetas y kappa binario 1.000")
    lineas.append("  en HS, TR y AG: cuando ambos emiten respuesta binaria, nunca discrepan; sus")
    lineas.append("  diferencias son solo por el uso de UTD (que si coincide en HS: 6 filas).")
    lineas.append("- Los pares con D en TR dan kappas bajos o negativos: D respondio POSITIVE en las")
    lineas.append("  15 filas de TR (varianza cero en su margen), lo que hace inestable el kappa de")
    lineas.append("  esos pares; el acuerdo bruto sigue siendo alto (9-14 de 15).")
    lineas.append("- A y B, entre si, tienen acuerdo alto en TR (11/15) y AG (12/15) y menor en HS (10/15).")
    lineas.append(f"- Combinacion exacta de los 4 anotadores (las tres etiquetas identicas): "
                   f"{a['combinacion_exacta_global']['num']}/15.")
    lineas.append("")
    lineas.append("### 3.4 UTD por etiqueta y anotador")
    lineas.append("")
    lineas.append("| etiqueta | A | B | C | D |")
    lineas.append("|---|---|---|---|---|")
    for etiqueta, bloque in (("HS", hs), ("TR", tr), ("AG", ag)):
        u = bloque["utd_por_anotador"]
        lineas.append(f"| {etiqueta} | {u['A']} | {u['B']} | {u['C']} | {u['D']} |")
    lineas.append("")
    lineas.append("A y B no usaron UTD en las columnas de clase: registraron la duda con el flag de")
    lineas.append("fila (A: 02, 04; B: 01, 05, 13). Los flags NO se convierten en UTD (seccion 2).")
    lineas.append("")
    lineas.append("## 4. Las seis filas HS con UTD coincidente de C y D")
    lineas.append("")
    lineas.append("Verificadas en los archivos reales: P92C-05, P92C-09, P92C-11, P92C-13, P92C-14,")
    lineas.append("P92C-15. En las seis, C y D registraron UNABLE_TO_DETERMINE en HS con notas que")
    lineas.append("apuntan al mismo vacio normativo: la guia v1 no define cuando una ofensa sexista")
    lineas.append("individual dirigida a una persona constituye discurso de odio.")
    lineas.append("")
    lineas.append("| review_id | C | D | motivo agrupado |")
    lineas.append("|---|---|---|---|")
    lineas.append("| P92C-05 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |")
    lineas.append("| P92C-09 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |")
    lineas.append("| P92C-11 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY (blanco por menciones) |")
    lineas.append("| P92C-13 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY + INSULT_VS_AGGRESSION (tono) |")
    lineas.append("| P92C-14 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |")
    lineas.append("| P92C-15 | UTD | UTD | HATE_VS_OFFENSE_AMBIGUITY |")
    lineas.append("")
    lineas.append("Los seis casos comparten un mismo tipo de ambiguedad (odio vs. ofensa), con un")
    lineas.append("matiz adicional en P92C-13 (agresivo vs. vulgar / posible tono de broma). No se")
    lineas.append("fuerza una categoria unica para 13: sus dos notas sustentan dos tipos.")
    lineas.append("")
    lineas.append("Casos relacionados fuera de la interseccion:")
    lineas.append("- P92C-03 y P92C-04: D marco UTD en HS con la misma duda normativa (C respondio")
    lineas.append("  POSITIVE en ambas, con la duda anotada en 04).")
    lineas.append("- P92C-07: C marco UTD en HS por contexto insuficiente (blanco no identificable).")
    lineas.append("")
    lineas.append("La coincidencia C-D se reporta como hallazgo descriptivo de la muestra; NO se")
    lineas.append("interpreta como prueba de que las etiquetas originales esten equivocadas, y no se")
    lineas.append("usa como fundamento unico de ninguna regla.")
    lineas.append("")
    lineas.append("## 5. Otras ambiguedades identificadas (AG, TR, contexto, flag de fila)")
    lineas.append("")
    lineas.append("- AG: P92C-06 (generalizacion hostil sin insulto: agresividad discutible, D UTD),")
    lineas.append("  P92C-13 (agresivo vs. vulgar, C UTD), P92C-07 (sarcasmo/contexto, D UTD).")
    lineas.append("- TR: P92C-01 (blanco no identificable sin contexto, C UTD; D respondio POSITIVE).")
    lineas.append("- Contexto: P92C-01, P92C-04 y P92C-07 dependen de informacion externa ausente.")
    lineas.append("- Flag de fila: P92C-02 (A, sin motivo especificado) es la unica senal de duda sin")
    lineas.append("  UTD asociado de C o D; se clasifica como OTHER.")
    lineas.append("")
    lineas.append("Detalle por caso (respuestas de los 4, tipo de ambiguedad, vacio de guia, regla")
    lineas.append("candidata): `reports/p92_calibration_case_review.csv` (15 filas).")
    lineas.append("")
    lineas.append("## 6. Limitacion muestral (explicita)")
    lineas.append("")
    lineas.append("- La calibracion consta de 15 tweets. Todas las fracciones de este informe son")
    lineas.append("  resultados de ESTA muestra, con denominadores visibles (n/15, n/den parcial).")
    lineas.append("- No se estima con esta muestra la tasa global de error de anotacion del dataset.")
    lineas.append("- 6/15 filas de esta muestra tienen UTD coincidente de C y D en HS: es un hallazgo")
    lineas.append("  de calibracion sobre las instrucciones, no una tasa del dataset completo.")
    lineas.append("- Los coeficientes de acuerdo no se extrapolan a la muestra principal de 250 tweets.")
    lineas.append("")
    lineas.append("## 7. Guia v2 propuesta")
    lineas.append("")
    lineas.append("Ver `reports/p92_annotation_guidelines_v2_proposal.md` (10 secciones, changelog")
    lineas.append("completo v1 -> v2, criterios operativos separados en OFFICIAL_DEFINITION /")
    lineas.append("PROPOSED_OPERATIONAL_RULE / UNRESOLVED_POLICY_QUESTION).")
    lineas.append("")
    lineas.append("Decision de version: **V2_PROPOSED_PENDING_DISCUSSION**. La guia no se congela y")
    lineas.append("la muestra principal no se distribuye hasta: resolver lo procedimental, registrar")
    lineas.append("las decisiones humanas, documentar lo normativo abierto, aprobar la version")
    lineas.append("aplicable y registrar version y hash de la guia.")
    lineas.append("")
    lineas.append("## 8. Discusion humana pendiente")
    lineas.append("")
    lineas.append("Ver `reports/p92_calibration_discussion_pack.md`: resumen de desacuerdos, las seis")
    lineas.append("filas HS, casos adicionales, preguntas operativas, opciones alternativas y el")
    lineas.append("registro de decisiones. El resultado esperado es un registro de decisiones humanas")
    lineas.append("y cuestiones pendientes, no una etiqueta correcta inferida.")
    lineas.append("")
    lineas.append("## 9. Trazabilidad y auditoria de integridad")
    lineas.append("")
    lineas.append("```text")
    lineas.append(f"raw annotation files unchanged     = {integridad['raw_annotation_files_unchanged']}")
    lineas.append(f"original labels unchanged          = {integridad['original_labels_unchanged']}")
    lineas.append(f"original texts unchanged           = {integridad['original_texts_unchanged']}")
    lineas.append(f"calibration sample unchanged       = {integridad['calibration_sample_unchanged']}")
    lineas.append(f"main sample not distributed        = {integridad['main_sample_not_distributed']}")
    lineas.append(f"V03 unchanged                      = {integridad['v03_unchanged']}")
    lineas.append(f"canonical OOF unchanged            = {integridad['canonical_oof_unchanged']}")
    lineas.append(f"holdouts untouched                 = {integridad['holdouts_untouched']}")
    lineas.append(f"test not accessed                  = {integridad['test_not_accessed']}")
    lineas.append(f"leaderboard not used               = {integridad['leaderboard_not_used']}")
    lineas.append(f"no models trained                  = {integridad['no_models_trained']}")
    lineas.append(f"no predictions generated           = {integridad['no_predictions_generated']}")
    lineas.append(f"no new submissions                 = {integridad['no_new_submissions']}")
    lineas.append(f"no pseudo-labeling                 = {integridad['no_pseudo_labeling']}")
    lineas.append(f"no external classifier             = {integridad['no_external_classifier']}")
    lineas.append(f"no external labeled dataset        = {integridad['no_external_labeled_dataset']}")
    lineas.append("```")
    lineas.append("")
    lineas.append("Hashes canonicos verificados (prefijos): " + ", ".join(
        f"{k}={'OK' if v.get('ok') else 'REVISAR'}"
        for k, v in integridad["hashes_canonicos"].items()
    ))
    lineas.append("")
    lineas.append("## 10. Estado final y siguientes pasos")
    lineas.append("")
    lineas.append("1. La discusion humana de la calibracion queda abierta (paquete de discusion).")
    lineas.append("2. Tras la discusion: registrar decisiones, decidir v1 vs v2 y congelar la guia")
    lineas.append("   aplicable con version y hash ANTES de distribuir la muestra principal.")
    lineas.append("3. La muestra principal (250 tweets) permanece sin distribuir; los paquetes no se")
    lineas.append("   regeneran hasta que exista guia congelada (cualquier regeneracion sera trazable).")
    lineas.append("4. Sin fase de modelado nueva: V03 y el OOF canonico permanecen congelados.")

    ruta = RAIZ / "reports" / "p92_calibration_analysis.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


def escribir_propuesta_v2(res: dict) -> None:
    """Propuesta de guia v2 para discusion humana (no normativa)."""
    lineas = []
    lineas.append("# P92 - Guia de anotacion v2 (PROPUESTA para discusion humana)")
    lineas.append("")
    lineas.append("Fecha: 2026-10-12. Estado: **V2_PROPOSED_PENDING_DISCUSSION**. Esta guia NO esta")
    lineas.append("congelada y NO es aplicable todavia: es una propuesta para la discusion de los")
    lineas.append("cuatro participantes de la calibracion. La ronda principal no debe iniciarse hasta")
    lineas.append("cerrar la discusion, registrar las decisiones humanas y congelar la version")
    lineas.append("aplicable (seccion 5).")
    lineas.append("")
    lineas.append("Base: `reports/p92_annotation_guidelines_v1.md` + evidencia de la calibracion")
    lineas.append("(`reports/p92_calibration_analysis.md`).")
    lineas.append("")
    lineas.append("Regla de construccion de esta propuesta: no se inventa ninguna definicion")
    lineas.append("normativa. Cada enunciado se clasifica como:")
    lineas.append("")
    lineas.append("```text")
    lineas.append("OFFICIAL_DEFINITION          -> texto oficial verbatim (competition_rules.md)")
    lineas.append("PROPOSED_OPERATIONAL_RULE    -> criterio operativo propuesto, pendiente de decision")
    lineas.append("UNRESOLVED_POLICY_QUESTION   -> pregunta normativa sin respuesta documentada")
    lineas.append("```")
    lineas.append("")
    lineas.append("## 1. Definiciones oficiales (verbatim)")
    lineas.append("")
    lineas.append("OFFICIAL_DEFINITION (docs/competition_rules.md; transliteracion ASCII de acentos):")
    lineas.append("")
    lineas.append("- **HS**: \"indica si hay discurso de odio (1 = si, 0 = no)\".")
    lineas.append("- **TR**: \"indica si el discurso esta dirigido contra una persona o grupo (1 = si, 0 = no)\".")
    lineas.append("- **AG**: \"indica si el texto es agresivo (1 = si, 0 = no)\".")
    lineas.append("")
    lineas.append("Estas tres definiciones siguen siendo la UNICA fuente normativa disponible. No")
    lineas.append("existen criterios operativos oficiales, guia de anotadores original ni ejemplos")
    lineas.append("resueltos por el organizador (verificado en fases previas).")
    lineas.append("")
    lineas.append("## 2. Instrucciones operativas")
    lineas.append("")
    lineas.append("### 2.1 Distinguir una posible ofensa sexista de discurso de odio")
    lineas.append("")
    lineas.append("UNRESOLVED_POLICY_QUESTION: no existe hoy ningun criterio documental (oficial o")
    lineas.append("provisto por el curso) que permita decidir cuando una ofensa sexista individual")
    lineas.append("dirigida a una persona constituye discurso de odio. La calibracion lo confirmo de")
    lineas.append("forma empirica: dos anotadores independientes (C y D) registraron la misma duda en")
    lineas.append("las mismas 6 de 15 filas de la muestra.")
    lineas.append("")
    lineas.append("PROPOSED_OPERATIONAL_RULE (para discusion; NO normativa): se proponen tres lecturas")
    lineas.append("alternativas para que la discusion humana elija o descarte:")
    lineas.append("")
    lineas.append("- O1 (estricta): HS = POSITIVE solo cuando la hostilidad se dirige a alguien por su")
    lineas.append("  identidad o pertenencia a un grupo (mujeres, migrantes, personas trans, etc.).")
    lineas.append("  El insulto por motivos personales sin componente identitario no seria HS.")
    lineas.append("- O2 (amplia): HS = POSITIVE cuando el texto degrada, deshumaniza o expresa")
    lineas.append("  hostilidad hacia una persona o grupo, aunque el motivo sea individual.")
    lineas.append("- O3 (literal): mantener la definicion oficial sin criterio operativo y usar")
    lineas.append("  UNABLE_TO_DETERMINE cuando la definicion no permita decidir.")
    lineas.append("")
    lineas.append("### 2.2 Que hacer cuando el criterio no existe o no resuelve el caso")
    lineas.append("")
    lineas.append("PROPOSED_OPERATIONAL_RULE: registrar UNABLE_TO_DETERMINE en la etiqueta afectada y")
    lineas.append("explicar el motivo en `uncertainty_notes`. No forzar una respuesta binaria. La duda")
    lineas.append("no invalida la anotacion: es un dato valido y esperado mientras la pregunta")
    lineas.append("normativa siga abierta. La incertidumbre registrada alimenta la consulta al")
    lineas.append("organizador (seccion 4).")
    lineas.append("")
    lineas.append("### 2.3 Insulto, vulgaridad, agresividad y discurso de odio")
    lineas.append("")
    lineas.append("OFFICIAL_DEFINITION: solo la de AG (\"indica si el texto es agresivo\").")
    lineas.append("")
    lineas.append("No se asume que estos conceptos sean equivalentes. Orientacion propuesta:")
    lineas.append("")
    lineas.append("- Vulgaridad: uso de lenguaje soez sin agresion dirigida. No implica AG ni HS.")
    lineas.append("- Insulto: agresion verbal dirigida a alguien. Puede implicar AG; su relacion con")
    lineas.append("  HS depende de la pregunta abierta de la seccion 2.1.")
    lineas.append("- Agresividad (AG): PROPOSED_OPERATIONAL_RULE: AG = POSITIVE si hay agresion")
    lineas.append("  dirigida (insulto, amenaza, orden hostil) hacia una persona o grupo. Queda como")
    lineas.append("  UNRESOLVED_POLICY_QUESTION si una generalizacion hostil sin insulto ni amenaza")
    lineas.append("  explicita cuenta como AG (caso real de la calibracion: generalizacion contra")
    lineas.append("  inmigrantes presentada como amenaza laboral).")
    lineas.append("- Discurso de odio (HS): segun la definicion oficial; sin criterio operativo")
    lineas.append("  documentado (seccion 2.1).")
    lineas.append("")
    lineas.append("### 2.4 Citas, discurso reproducido, sarcasmo, ironia y falta de contexto")
    lineas.append("")
    lineas.append("OFFICIAL_DEFINITION: no existe ninguna regla oficial para estos casos.")
    lineas.append("")
    lineas.append("PROPOSED_OPERATIONAL_RULE (para discusion):")
    lineas.append("- Evaluar el contenido hostil expresado en el texto, no la forma retorica.")
    lineas.append("- Sarcasmo o ironia: si el texto expresa hostilidad hacia alguien, evaluar la")
    lineas.append("  etiqueta correspondiente con el mismo criterio que una expresion literal.")
    lineas.append("- Citas o discurso reproducido: considerar el contexto disponible en el propio")
    lineas.append("  texto; si el texto no permite saber si hay adhesion, denuncia o burla, registrar")
    lineas.append("  UNABLE_TO_DETERMINE y anotarlo.")
    lineas.append("- Expresiones idiomaticas (por ejemplo, formulas coloquiales de enfasis) no")
    lineas.append("  cuentan por si solas como agresion.")
    lineas.append("- Contexto insuficiente: si falta el tweet original o la referencia y no se puede")
    lineas.append("  decidir responsablemente, usar UNABLE_TO_DETERMINE + nota. No adivinar.")
    lineas.append("")
    lineas.append("### 2.5 Criterios para elegir UNABLE_TO_DETERMINE")
    lineas.append("")
    lineas.append("Formulacion heredada de la v1 (se mantiene): UTD no es una clase semantica.")
    lineas.append("Usarlo cuando:")
    lineas.append("")
    lineas.append("1. La definicion oficial no permite decidir responsablemente.")
    lineas.append("2. Falta el contexto necesario (blanco no identificable, referencia ininteligible).")
    lineas.append("3. Existen dos lecturas competidoras y ninguna es claramente aplicable (por ejemplo,")
    lineas.append("   odio vs. ofensa mientras la seccion 2.1 siga abierta).")
    lineas.append("")
    lineas.append("Reglas: se decide POR ETIQUETA (una etiqueta puede ser UTD y otra no); la nota de")
    lineas.append("motivo es obligatoria cuando se use UTD; UTD nunca se convierte automaticamente en")
    lineas.append("POSITIVE ni NEGATIVE; si la fila entera es dificil, marcar UTD en las etiquetas")
    lineas.append("afectadas y explicarlo en la nota (sin flags binarios aparte: el incidente A/B")
    lineas.append("mostro que un 0/1 adicional fuera de las columnas se vuelve ambiguo).")
    lineas.append("")
    lineas.append("### 2.6 Independencia de HS, TR y AG")
    lineas.append("")
    lineas.append("OFFICIAL_DEFINITION: tres definiciones independientes. PROPOSED_OPERATIONAL_RULE:")
    lineas.append("anotar cada etiqueta por separado, sin jerarquias. En particular, no asumir que")
    lineas.append("TR = 1 exige HS = 1 ni que AG = 1 exige HS = 1: la relacion observada en los datos")
    lineas.append("no esta documentada como regla formal. Cada etiqueta admite su propio UTD.")
    lineas.append("")
    lineas.append("### 2.7 Ejemplos estructurales anonimizados (derivados de la calibracion)")
    lineas.append("")
    lineas.append("Ejemplos de TIPO de ambiguedad, sin textos verbatim y sin respuestas oficiales:")
    lineas.append("")
    lineas.append("- E1: insulto sexista directo a una persona, sin referencia a un grupo.")
    lineas.append("  Pregunta abierta: odio vs. ofensa (seccion 2.1).")
    lineas.append("- E2: ataque personal con insultos sexistas o vulgares y ordenes hostiles a una")
    lineas.append("  persona. Pregunta abierta: odio vs. ofensa; AG probablemente POSITIVE.")
    lineas.append("- E3: insulto leve con posible tono de broma entre personas conocidas.")
    lineas.append("  Pregunta abierta: agresivo vs. vulgar; el tono no decide por si solo.")
    lineas.append("- E4: generalizacion hostil contra un grupo (por ejemplo, presentarlo como amenaza")
    lineas.append("  laboral) sin insulto ni amenaza explicita. Pregunta abierta: cuenta como AG.")
    lineas.append("- E5: respuesta con apodo o referencia no identificable sin el tweet original.")
    lineas.append("  Tratamiento propuesto: UTD + nota (contexto insuficiente).")
    lineas.append("- E6: sarcasmo o cita con expresiones idiomaticas. Tratamiento propuesto: evaluar")
    lineas.append("  el contenido expresado; UTD si el contexto no alcanza.")
    lineas.append("")
    lineas.append("## 3. Changelog v1 -> v2 (cada diferencia)")
    lineas.append("")
    lineas.append("| # | Cambio | v1 | v2 propuesta | Evidencia |")
    lineas.append("|---|---|---|---|---|")
    lineas.append("| CH-01 | Formato de respuesta | \"una de POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE por celda\" | igual, mas: los valores 0/1 quedan explicitamente invalidos; ejemplo documentado del incidente A/B | A y B devolvieron 0/1 (P92-S2, P92-T) |")
    lineas.append("| CH-02 | Nota de motivo con UTD | nota opcional | nota OBLIGATORIA cuando se usa UTD | 6 filas con UTD de C y D sustentadas en notas; las notas hicieron interpretable cada duda |")
    lineas.append("| CH-03 | Criterios operativos | no existian | nueva seccion 2 con criterios marcados PROPOSED_OPERATIONAL_RULE, separados de OFFICIAL_DEFINITION y UNRESOLVED_POLICY_QUESTION | calibracion: la duda se concentro en odio vs. ofensa (6 filas HS) y agresivo vs. vulgar |")
    lineas.append("| CH-04 | Duda a nivel de fila | no previsto | 2.5: marcar UTD en las etiquetas afectadas + nota (sin flag binario aparte) | flag 0/1 de A/B resulto ambiguo de interpretar |")
    lineas.append("| CH-05 | Citas, sarcasmo, contexto | listados como preguntas abiertas sin orientacion | 2.4 con orientacion operativa propuesta + UTD cuando el contexto no alcanza | casos 01 y 07 (contexto insuficiente) y 06/07 (sarcasmo/idiomatismo) |")
    lineas.append("| CH-06 | Ejemplos | no habia | 2.7 con ejemplos estructurales anonimizados | tipos de ambiguedad observados en la calibracion |")
    lineas.append("| CH-07 | Preguntas normativas | 4 limitaciones listadas sin formulacion para terceros | seccion 4: formuladas como preguntas para decision humana y eventual consulta al organizador | evidencia empirica de la calibracion |")
    lineas.append("| CH-08 | Independencia HS/TR/AG | recomendada | reforzada en 2.6, con UTD por etiqueta | ningun anotador uso jerarquias; se mantiene como recordatorio |")
    lineas.append("")
    lineas.append("Se conservan sin cambios: unidad de anotacion (tweet completo), cegamiento,")
    lineas.append("registro de desacuerdos, prohibicion de herramientas automaticas, esquema de")
    lineas.append("columnas y el principio de no inventar reglas.")
    lineas.append("")
    lineas.append("## 4. Limitaciones semanticas que continuan sin resolverse")
    lineas.append("")
    lineas.append("1. UNRESOLVED_POLICY_QUESTION (principal): cuando una ofensa sexista individual")
    lineas.append("   constituye discurso de odio. Evidencia: 6/15 filas de calibracion con UTD")
    lineas.append("   coincidente de C y D; 2 mas con UTD de D en la misma duda (03, 04).")
    lineas.append("2. UNRESOLVED_POLICY_QUESTION: agresivo vs. vulgar, y si la generalizacion hostil")
    lineas.append("   sin insulto cuenta como AG (casos 06 y 13 de la calibracion).")
    lineas.append("3. UNRESOLVED_POLICY_QUESTION: relacion formal entre HS y TR (y HS y AG).")
    lineas.append("4. UNRESOLVED_POLICY_QUESTION: tratamiento de citas, sarcasmo y discurso")
    lineas.append("   reproducido; la seccion 2.4 solo propone orientacion operativa.")
    lineas.append("")
    lineas.append("Pregunta propuesta para el organizador (redaccion para discusion): \"Cuando un")
    lineas.append("insulto o agresion sexista se dirige a una persona concreta por su identidad, sin")
    lineas.append("referirse a un grupo, debe anotarse HS = 1, HS = 0, o no existe criterio?\".")
    lineas.append("")
    lineas.append("## 5. Decision de version y condiciones de congelamiento")
    lineas.append("")
    lineas.append("Estado actual: V2_PROPOSED_PENDING_DISCUSSION. La guia solo puede considerarse")
    lineas.append("congelada para la muestra principal despues de:")
    lineas.append("")
    lineas.append("1. Resolver las cuestiones procedimentales (formato, UTD, notas: secciones 2.2,")
    lineas.append("   2.5 y CH-01/CH-02).")
    lineas.append("2. Registrar las decisiones humanas sobre los criterios operativos (2.1, 2.3, 2.4).")
    lineas.append("3. Documentar las cuestiones normativas que siguen abiertas (seccion 4).")
    lineas.append("4. Aprobar la version aplicable a la anotacion principal.")
    lineas.append("5. Registrar version y hash de la guia congelada.")
    lineas.append("")
    lineas.append("Si la definicion oficial no permite resolver una ambiguedad material, no se simula")
    lineas.append("certeza normativa: se documenta la limitacion. La ronda principal puede continuar")
    lineas.append("como referencia exploratoria con UTD registrado, o esperar una aclaracion adicional")
    lineas.append("del organizador; esa eleccion es parte de la discusion humana.")

    ruta = RAIZ / "reports" / "p92_annotation_guidelines_v2_proposal.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


def escribir_pack_discusion(res: dict, respuestas: dict, textos: dict) -> None:
    """Paquete de discusion humana (sin etiquetas originales ni informacion de modelos)."""
    lineas = []
    lineas.append("# P92 - Paquete de discusion de la ronda de calibracion")
    lineas.append("")
    lineas.append("Para: los cuatro participantes de la calibracion (A, B, C y D).")
    lineas.append("Fecha: 2026-10-12. Estado: HUMAN_DISCUSSION_PENDING.")
    lineas.append("")
    lineas.append("La primera ronda independiente ya cerro: cada persona anoto los mismos 15 textos")
    lineas.append("sin comunicarse con las demas. Esta discusion es la etapa posterior prevista en el")
    lineas.append("protocolo. Reglas de la discusion:")
    lineas.append("")
    lineas.append("- No se muestran etiquetas originales del dataset ni informacion de modelos.")
    lineas.append("- No se decide aqui quien tiene razon; no hay adjudicacion automatica.")
    lineas.append("- La mayoria simple no resuelve por si sola una cuestion normativa.")
    lineas.append("- Resultado esperado: registro de decisiones humanas y cuestiones pendientes.")
    lineas.append("- Las notas citadas se transliteran a ASCII; los originales con acentos estan en")
    lineas.append("  los archivos de respuesta de cada persona.")
    lineas.append("")
    lineas.append("## 1. Resumen de los desacuerdos observados")
    lineas.append("")
    a = res["acuerdo"]
    hs, tr, ag = a["por_etiqueta"]["HS"], a["por_etiqueta"]["TR"], a["por_etiqueta"]["AG"]
    lineas.append("Acuerdo exacto entre cada par (de 15 filas):")
    lineas.append("")
    lineas.append("| etiqueta | A-B | A-C | A-D | B-C | B-D | C-D |")
    lineas.append("|---|---|---|---|---|---|---|")
    for etiqueta, bloque in (("HS", hs), ("TR", tr), ("AG", ag)):
        valores = " | ".join(f"{bloque['pares'][p]['acuerdo_3cat']['num']}/15"
                             for p in ("AB", "AC", "AD", "BC", "BD", "CD"))
        lineas.append(f"| {etiqueta} | {valores} |")
    lineas.append("")
    lineas.append("- Las tres etiquetas juntas coinciden exactamente en los 4 anotadores en 1/15 filas.")
    lineas.append("- Uso de UNABLE_TO_DETERMINE: HS (C: 7/15, D: 8/15); TR (C: 1/15); AG (C: 1/15,")
    lineas.append("  D: 3/15). A y B registraron su duda como marca de fila en notas (A: 2 filas, B: 3).")
    lineas.append("- Donde el desacuerdo es binario puro (sin UTD), C y D coinciden en todas las")
    lineas.append("  celdas binarias de las tres etiquetas.")
    lineas.append("")
    lineas.append("## 2. Las seis filas HS con incertidumbre coincidente de C y D")
    lineas.append("")
    lineas.append("En las seis, C y D registraron UNABLE_TO_DETERMINE en HS con el mismo motivo: la")
    lineas.append("guia no define cuando una ofensa sexista individual dirigida a una persona")
    lineas.append("constituye discurso de odio. Textos (verbatim) y notas:")
    lineas.append("")
    seis = ["P92C-05", "P92C-09", "P92C-11", "P92C-13", "P92C-14", "P92C-15"]
    for rid in seis:
        lineas.append(f"### {rid}")
        lineas.append("")
        lineas.append(f"Texto: \"{textos[rid]}\"")
        lineas.append("")
        lineas.append(f"- C (HS = UNABLE_TO_DETERMINE): \"{ascii_texto(respuestas['C'][rid]['notas'])}\"")
        lineas.append(f"- D (HS = UNABLE_TO_DETERMINE): \"{ascii_texto(respuestas['D'][rid]['notas'])}\"")
        linea_a = ("- A: respuestas 0/0/0" if respuestas["A"][rid]["notas"] != "1"
                   else "- A: respuestas 0/0/0 y marca de fila dificil para cualquier clase")
        linea_b = ("- B: respuestas " + "/".join(respuestas["B"][rid][e] for e in ETIQUETAS)
                   if respuestas["B"][rid]["notas"] != "1"
                   else "- B: respuestas " + "/".join(respuestas["B"][rid][e] for e in ETIQUETAS)
                   + " y marca de fila")
        lineas.append(linea_a)
        lineas.append(linea_b)
        lineas.append("")
    lineas.append("Pregunta para estas seis filas: debe anotarse HS = POSITIVE cuando el insulto")
    lineas.append("sexista se dirige a una persona concreta? Cual de las opciones de la seccion 6")
    lineas.append("(o ninguna) describe mejor la definicion oficial? Si la discusion concluye que la")
    lineas.append("norma no alcanza, la pregunta se eleva al organizador.")
    lineas.append("")
    lineas.append("## 3. Casos adicionales (AG, TR, contexto y marca de fila)")
    lineas.append("")
    adicionales = [
        ("P92C-01", "TR (C: UTD) y AG (D: UTD)", "INSUFFICIENT_CONTEXT"),
        ("P92C-03", "HS (D: UTD; misma duda normativa que las seis)", "HATE_VS_OFFENSE"),
        ("P92C-04", "HS (D: UTD; contexto + duda normativa)", "HATE_VS_OFFENSE; INSUFFICIENT_CONTEXT"),
        ("P92C-06", "AG (D: UTD)", "INSULT_VS_AGGRESSION"),
        ("P92C-07", "HS (C: UTD) y AG (D: UTD)", "INSUFFICIENT_CONTEXT"),
        ("P92C-13", "AG (C: UTD)", "INSULT_VS_AGGRESSION"),
        ("P92C-02", "marca de fila de A (notas=1, motivo no especificado)", "OTHER"),
    ]
    for rid, donde, tipo in adicionales:
        lineas.append(f"### {rid} - {donde} ({tipo})")
        lineas.append("")
        lineas.append(f"Texto: \"{textos[rid]}\"")
        lineas.append("")
        for anotador in ("C", "D"):
            nota = ascii_texto(respuestas[anotador][rid]["notas"])
            if nota:
                lineas.append(f"- {anotador}: \"{nota}\"")
        if respuestas["A"][rid]["notas"] == "1":
            lineas.append("- A: marco la fila como dificil para cualquier clase.")
        if respuestas["B"][rid]["notas"] == "1":
            lineas.append("- B: marco la fila como dificil.")
        lineas.append("")
    lineas.append("## 4. Preguntas operativas por resolver en la discusion")
    lineas.append("")
    lineas.append("- Q1. Frontera odio vs. ofensa: se adopta O1 (estricta), O2 (amplia) u O3")
    lineas.append("  (literal + UTD)? Con que fundamento dentro de la definicion oficial?")
    lineas.append("- Q2. Se eleva la pregunta normativa al organizador? (redaccion propuesta en la")
    lineas.append("  seccion 7.)")
    lineas.append("- Q3. Frontera agresivo vs. vulgar: basta la agresion dirigida para AG = POSITIVE?")
    lineas.append("  Cuenta como AG una generalizacion hostil sin insulto?")
    lineas.append("- Q4. Tono de broma o registro coloquial: como afecta cada etiqueta?")
    lineas.append("- Q5. Citas, sarcasmo y discurso reproducido: que regla operativa se adopta?")
    lineas.append("- Q6. Contexto insuficiente: se confirma UTD + nota (no adivinar)?")
    lineas.append("- Q7. Formato de la ronda principal: se confirma el uso exclusivo de")
    lineas.append("  POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE (los 0/1 quedan invalidos)?")
    lineas.append("- Q8. Duda de fila: se adopta marcar UTD en las etiquetas afectadas + nota?")
    lineas.append("")
    lineas.append("## 5. Opciones alternativas de definicion (cuando hay mas de una lectura)")
    lineas.append("")
    lineas.append("- HS (odio vs. ofensa): O1 estricta / O2 amplia / O3 literal + UTD.")
    lineas.append("- AG (agresividad): A1 = agresion dirigida; A2 = incluir tambien generalizaciones")
    lineas.append("  hostiles sin insulto.")
    lineas.append("- TR: sin alternativas documentadas; se mantiene la definicion oficial.")
    lineas.append("")
    lineas.append("## 6. Informacion normativa que sigue faltando")
    lineas.append("")
    lineas.append("1. Criterio para distinguir ofensa sexista individual de discurso de odio.")
    lineas.append("2. Criterio para distinguir agresivo de vulgar y para la generalizacion hostil.")
    lineas.append("3. Tratamiento de citas, sarcasmo y discurso reproducido.")
    lineas.append("4. Relacion formal (si existe) entre HS y TR, y entre HS y AG.")
    lineas.append("")
    lineas.append("Pregunta propuesta para el organizador: \"Cuando un insulto o agresion sexista")
    lineas.append("se dirige a una persona concreta por su identidad, sin referirse a un grupo, debe")
    lineas.append("anotarse HS = 1, HS = 0, o no existe criterio?\".")
    lineas.append("")
    lineas.append("## 7. Registro de decisiones (plantilla para completar en la discusion)")
    lineas.append("")
    lineas.append("| pregunta | decision | fundamento | fecha | estado |")
    lineas.append("|---|---|---|---|---|")
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        lineas.append(f"| {q} | (pendiente) | (pendiente) | (pendiente) | pendiente |")
    lineas.append("")
    lineas.append("Cuestiones que queden sin decision se registran como pendientes y, si son")
    lineas.append("normativas, se elevan al organizador antes de congelar la guia.")
    lineas.append("")
    lineas.append("## 8. Que NO hace este paquete")
    lineas.append("")
    lineas.append("- No muestra etiquetas originales del dataset ni motivos de seleccion de los casos.")
    lineas.append("- No muestra predicciones, probabilidades, umbrales ni informacion de modelos.")
    lineas.append("- No decide automaticamente ninguna etiqueta ni senala a nadie como incorrecto.")
    lineas.append("- No contacta a los anotadores ni al organizador por si mismo.")

    ruta = RAIZ / "reports" / "p92_calibration_discussion_pack.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


def main() -> None:
    with open(RUTA_JSON, encoding="utf-8") as fh:
        res = json.load(fh)
    respuestas, textos = cargar_respuestas()
    print("P92-T informes: generando artefactos")
    escribir_acuerdo(res)
    escribir_casos(res, respuestas)
    escribir_analisis(res, respuestas)
    escribir_propuesta_v2(res)
    escribir_pack_discusion(res, respuestas, textos)
    print("P92-T informes: OK")


if __name__ == "__main__":
    main()
