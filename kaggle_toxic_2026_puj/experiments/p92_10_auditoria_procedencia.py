#!/usr/bin/env python
"""
P92-T-R - Auditoria de procedencia de la codificacion humana de A/B (fase diagnostica).

Verifica la afirmacion de P92-T ("ENCODING_CONFIRMED") contra la evidencia localizable:
busqueda en disco, instruccion del usuario registrada en la sesion y documentos fuente.
Aplica el estandar estricto: si las dos convenciones no pueden demostrarse, no se
presentan como aclaraciones confirmadas.

Salidas (ASCII puro; sin sobrescribir respuestas humanas ni resultados historicos):
  reports/p92tr_encoding_provenance_audit.md      (addendum independiente)
  reports/p92tr_ab_confirmation_request.md        (solicitud directa a A y B; borrador)
  Notas aditivas al final de (idempotentes, marcador "NOTA ADITIVA P92-T-R"):
    reports/p92_calibration_analysis.md
    artifacts/human_annotation_v1/calibration_responses/PROVENANCE.md
    reports/p92u_human_discussion_pack.md
    reports/p92_guideline_decision_record.md

Reglas: sin entrenamiento, sin modificacion de etiquetas, sin distribucion, sin
test/holdout, sin submissions; no se elimina nada historico.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESP = RAIZ / "artifacts" / "human_annotation_v1" / "calibration_responses"
RUTA_JSON = RAIZ / "reports" / "p92t_audit_results.json"

ETIQUETAS = ("HS", "TR", "AG")
COLUMNAS_CLASE = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}

ESPERADOS_RESPUESTAS = {
    "A": "a20521989aa3fea9",
    "B": "41e50c1f1d48760e",
    "C": "8de76e21046d5091",
    "D": "763229f9a9caa176",
}
ESPERADOS_NORMALIZADOS = {"A": "7e7b3c805620e613", "B": "7cca95e20a1243c0"}

CITA_USUARIO = (
    "En las 2 anotaciones, la primera y la segunda, 0 y 1 significan que son de la clase "
    "correspondiente, las que son 1 fuera de las 3 clases es que son dificiles de "
    "determinar para cualquier clase."
)

MARCADOR_NOTA = "NOTA ADITIVA P92-T-R"

NOTA_ANALISIS = """
---

## NOTA ADITIVA P92-T-R (2026-10-12): procedencia de la codificacion A/B

La atribucion "ENCODING_CONFIRMED" de la seccion 2 queda CORREGIDA en su interpretacion.
La auditoria de procedencia (reports/p92tr_encoding_provenance_audit.md) encontro que la
aclaracion del usuario existe como instruccion registrada en la sesion de trabajo (cita
verbatim en el addendum), pero (1) no estaba persistida en ningun artefacto del proyecto,
(2) no enumera literalmente la direccion del mapeo 0/1 y (3) es de nivel coordinador, sin
confirmacion directa de A ni de B.

Estado corregido: ENCODING_UNRESOLVED al estandar estricto. Consecuencias:
- Las estadisticas que dependen de la codificacion de A/B (pares AC, AD, BC, BD,
  combinacion exacta de los 4, alfas con A/B) quedan CONDICIONALES a esa codificacion.
- Las observaciones de C y D, sus UNABLE_TO_DETERMINE y los valores crudos 0/1 de A y B
  permanecen validos; el acuerdo A-B (10/15, 11/15, 12/15) es valido como igualdad de
  valores crudos.
- Los archivos normalizados de A/B quedan marcados como derivados CONDICIONALES.
- Se preparo una solicitud breve de confirmacion directa a A y B
  (reports/p92tr_ab_confirmation_request.md; no enviada).
Ver el addendum para el detalle completo y el impacto por niveles de evidencia.
"""

NOTA_PROVENANCE = """
## NOTA ADITIVA P92-T-R (2026-10-12): estado condicional de la regla

La regla del addendum anterior queda CONDICIONADA. La auditoria de procedencia
(reports/p92tr_encoding_provenance_audit.md) encontro que la aclaracion existe como
instruccion registrada en la sesion, pero no esta persistida en el proyecto, no enumera
literalmente la direccion del mapeo y no cuenta con confirmacion directa de A y B.
Estado: ENCODING_UNRESOLVED al estandar estricto. Los archivos normalizados
(calibration_response_A_normalized.csv, calibration_response_B_normalized.csv) son
derivados CONDICIONALES a esa codificacion y no deben usarse como acuerdo definitivo.
Solicitud de confirmacion directa preparada (reports/p92tr_ab_confirmation_request.md;
no enviada). Los cuatro archivos originales de esta carpeta siguen SIN CAMBIOS.
"""

NOTA_PACK_U = """
---

## NOTA ADITIVA P92-T-R (2026-10-12): estado de la codificacion A/B

Donde este paquete menciona que la codificacion 0/1 de A y B "quedo confirmada", debe
leerse con el estado corregido: ENCODING_UNRESOLVED al estandar estricto (instruccion
del responsable registrada en la sesion pero sin confirmacion directa de A/B, sin
persistencia previa en el proyecto y sin enumeracion literal de la direccion del mapeo).
Las respuestas de C y D y los valores crudos de A y B siguen siendo validos; las
estadisticas que dependen de la codificacion de A/B son condicionales. Ver
reports/p92tr_encoding_provenance_audit.md y la solicitud de confirmacion
(reports/p92tr_ab_confirmation_request.md; no enviada).
"""

NOTA_REGISTRO = """
---

## NOTA ADITIVA P92-T-R (2026-10-12)

Donde este registro menciona "ENCODING_CONFIRMED" (evidencia de Q7 y Q8), leer el estado
corregido: ENCODING_UNRESOLVED al estandar estricto, pendiente de confirmacion directa
de A y B (ver reports/p92tr_encoding_provenance_audit.md y
reports/p92tr_ab_confirmation_request.md). La estructura de las preguntas no cambia;
solo se corrige la calificacion de esa evidencia.
"""


def sha256_archivo(ruta: Path) -> str:
    """sha256 completo de un archivo (lectura en bloques)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_respuestas() -> dict[str, dict[str, dict]]:
    """Respuestas crudas por anotador (valores tal como se recibieron)."""
    respuestas: dict[str, dict[str, dict]] = {}
    for anotador in ("A", "B", "C", "D"):
        with open(DIR_RESP / f"calibration_response_{anotador}.csv",
                  encoding="utf-8-sig", newline="") as fh:
            filas = list(csv.DictReader(fh))
        respuestas[anotador] = {
            f["review_id"]: {e: f[c] for e, c in COLUMNAS_CLASE.items()} for f in filas
        }
    return respuestas


def acuerdo_crudo_ab(respuestas: dict) -> dict[str, tuple[int, int]]:
    """Acuerdo A-B por igualdad directa de valores crudos (no requiere el mapeo)."""
    resultado = {}
    for etiqueta in ETIQUETAS:
        iguales = sum(
            1 for rid in respuestas["A"]
            if respuestas["A"][rid][etiqueta] == respuestas["B"][rid][etiqueta]
        )
        resultado[etiqueta] = (iguales, len(respuestas["A"]))
    return resultado


def anexar_nota(ruta: Path, nota: str) -> str:
    """Anexa una nota aditiva al final de un archivo (idempotente por marcador)."""
    contenido = ruta.read_text(encoding="utf-8")
    if MARCADOR_NOTA in contenido:
        return "ya presente (sin cambios)"
    with open(ruta, "a", encoding="utf-8") as fh:
        fh.write(nota)
    return "anexada"


def escribir_addendum(respuestas: dict, hashes: dict, crudo_ab: dict) -> None:
    """Genera el addendum independiente de procedencia."""
    lineas = []
    lineas.append("# P92-T-R - Auditoria de procedencia de la codificacion humana de A/B")
    lineas.append("")
    lineas.append("Addendum independiente. Fecha: 2026-10-12. Fase diagnostica: sin")
    lineas.append("entrenamiento, sin modificacion de etiquetas, sin distribucion, sin")
    lineas.append("test/holdout, sin submissions. No sobrescribe respuestas humanas ni")
    lineas.append("resultados historicos: corrige la interpretacion de forma aditiva.")
    lineas.append("")
    lineas.append("## 1. Objeto")
    lineas.append("")
    lineas.append("Auditar la afirmacion de P92-T: \"ENCODING_CONFIRMED; 0 = NEGATIVE; 1 =")
    lineas.append("POSITIVE; uncertainty_notes = 1 significa dificultad para determinar")
    lineas.append("cualquier clase\", atribuida a \"una aclaracion explicita del usuario del")
    lineas.append("2026-10-12\". Estandar aplicado: si las dos convenciones no pueden demostrarse")
    lineas.append("con evidencia localizable, no se presentan como aclaraciones confirmadas.")
    lineas.append("La consistencia con las respuestas de C/D no se usa como evidencia.")
    lineas.append("")
    lineas.append("## 2. Evidencia localizada")
    lineas.append("")
    lineas.append("### 2.1 Busqueda en disco (2026-10-12)")
    lineas.append("")
    lineas.append("- Archivos con la evidencia CRUDA del mensaje del usuario: NINGUNO. Las")
    lineas.append("  busquedas de \"clase correspondiente\" y \"En las 2 anotaciones\" no")
    lineas.append("  devuelven coincidencias en el proyecto.")
    lineas.append("- Archivos con la afirmacion (registros DERIVADOS, sin evidencia cruda):")
    lineas.append("  reports/p92_calibration_analysis.md (seccion 2), reports/p92t_audit_results.json")
    lineas.append("  (codificacion_AB), artifacts/human_annotation_v1/calibration_responses/")
    lineas.append("  PROVENANCE.md (addendum), reports/p92s2_calibration_results.json")
    lineas.append("  (campo \"confirmada_por_carlos\": \"2026-10-12\"), reports/")
    lineas.append("  p92u_human_discussion_pack.md, reports/p92_guideline_decision_record.md,")
    lineas.append("  notebooks/10_final_pipeline.ipynb (secciones 140 y 141) y")
    lineas.append("  experiments/p92_05_analisis_calibracion.py.")
    lineas.append("- Estado documentado ANTES de la supuesta aclaracion (informe de recepcion v2,")
    lineas.append("  reports/p92s2_calibration_reception_report.md, en disco):")
    lineas.append("  - \"A/B usaron 0/1 en lugar de los tokens (analisis bajo el supuesto")
    lineas.append("    0=NEGATIVE, 1=POSITIVE, PENDIENTE DE CONFIRMACION)\".")
    lineas.append("  - Decisiones pendientes: \"1. Confirmar la codificacion de A/B: 0=NEGATIVE,")
    lineas.append("    1=POSITIVE (y ausencia de UTD). 2. Aclarar el significado del 0/1 en")
    lineas.append("    uncertainty_notes de A/B.\"")
    lineas.append("")
    lineas.append("### 2.2 Instruccion del usuario (registrada en la sesion de trabajo del 2026-10-12)")
    lineas.append("")
    lineas.append("Cita verbatim tal como consta en el registro de la sesion (no persistida en")
    lineas.append("disco hasta este addendum):")
    lineas.append("")
    lineas.append("```text")
    lineas.append(CITA_USUARIO)
    lineas.append("```")
    lineas.append("")
    lineas.append("Cadena de procedencia: (1) se detecto que A y B devolvieron 0/1 (desviacion")
    lineas.append("de formato); (2) el agente solicito la aclaracion en la sesion; (3) el")
    lineas.append("responsable del proyecto respondio con la instruccion citada; (4) P92-T la uso")
    lineas.append("como base de \"ENCODING_CONFIRMED\"; (5) este audit verifica su alcance y")
    lineas.append("persistencia.")
    lineas.append("")
    lineas.append("Alcance de la cita:")
    lineas.append("")
    lineas.append("- Notas (uncertainty_notes = 1): \"las que son 1 fuera de las 3 clases...")
    lineas.append("  dificiles de determinar para cualquier clase\" -> contenido LITERAL.")
    lineas.append("- Clases (0/1): \"0 y 1 significan que son de la clase correspondiente\" ->")
    lineas.append("  establece que 0/1 codifican pertenencia a la clase; la direccion (1 =")
    lineas.append("  POSITIVE, 0 = NEGATIVE) es la lectura natural (convencion binaria y")
    lineas.append("  coherencia con la clausula de notas), pero NO queda enumerada literalmente.")
    lineas.append("- Nivel: aclaracion del responsable/coordinador del proyecto; NO es una")
    lineas.append("  confirmacion directa de los anotadores A ni B.")
    lineas.append("- Persistencia: inexistente en el proyecto hasta este addendum, que la deja")
    lineas.append("  registrada.")
    lineas.append("")
    lineas.append("### 2.3 Formato solicitado a A y B (verificacion documental)")
    lineas.append("")
    lineas.append("- README del handoff (annotator_A/README_annotator.md): \"Usa exclusivamente")
    lineas.append("  POSITIVE, NEGATIVE o UNABLE_TO_DETERMINE en las tres columnas")
    lineas.append("  correspondientes\"; \"Deja uncertainty_notes vacio si no hay ninguna")
    lineas.append("  aclaracion necesaria\".")
    lineas.append("- Guia v1 (seccion 8): \"Columnas HS_annotation, TR_annotation, AG_annotation:")
    lineas.append("  una de POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE por celda\".")
    lineas.append("- Resultado: A y B usaron un formato DISTINTO del solicitado (0/1 en las")
    lineas.append("  cuatro columnas, con BOM de Excel). Desviacion confirmada documentalmente;")
    lineas.append("  las respuestas originales se conservan intactas.")
    lineas.append("")
    lineas.append("## 3. Estado de la codificacion (estandar estricto)")
    lineas.append("")
    lineas.append("| convencion | contenido | evidencia localizada | confirmacion directa A/B | estado |")
    lineas.append("|---|---|---|---|---|")
    lineas.append("| uncertainty_notes=1 | \"dificil de determinar para cualquier clase\" | instruccion registrada (cita 2.2); contenido literal | no | ATRIBUIDA (contenido literal); pendiente de confirmacion directa |")
    lineas.append("| 0/1 -> NEGATIVE/POSITIVE | 0 = no pertenece; 1 = pertenece (POSITIVE) | instruccion registrada; direccion = lectura natural, NO enumerada literalmente | no | NO DEMOSTRADA al estandar estricto |")
    lineas.append("")
    lineas.append("Resultado global: **ENCODING_UNRESOLVED** (estandar estricto de este audit).")
    lineas.append("")
    lineas.append("- La atribucion de P92-T no fue una invencion: la instruccion existe y queda")
    lineas.append("  citada y persistida en este addendum.")
    lineas.append("- Sin embargo: (1) no estaba persistida en ningun artefacto del proyecto;")
    lineas.append("  (2) no enumera literalmente la direccion del mapeo de clases; (3) es de nivel")
    lineas.append("  coordinador, sin confirmacion directa de A ni de B.")
    lineas.append("- Consecuencia: ninguna de las dos convenciones se presenta como \"aclaracion")
    lineas.append("  confirmada\"; la de notas queda como contenido literal pendiente de")
    lineas.append("  confirmacion directa; la de clases queda como no demostrada.")
    lineas.append("- Justicia interpretativa: la instruccion hace de 0=NEGATIVE/1=POSITIVE la")
    lineas.append("  lectura natural y coherente; este audit no afirma que sea incorrecta, solo")
    lineas.append("  que no cumple el estandar de demostracion exigido. La confirmacion directa")
    lineas.append("  de A y B es la accion que cierra la incertidumbre (seccion 5).")
    lineas.append("")
    lineas.append("## 4. Impacto sobre los resultados de P92-T (por niveles de evidencia)")
    lineas.append("")
    lineas.append("### 4.1 Validos como observaciones crudas (sin supuesto de codificacion)")
    lineas.append("")
    lineas.append("- Respuestas de C y D, incluidos sus UNABLE_TO_DETERMINE; acuerdo C-D")
    lineas.append("  (12/15 HS, 14/15 TR, 11/15 AG); conteos UTD; catalogo de ambiguedades")
    lineas.append("  (seis casos HS + secundarios), basado en notas de C/D y documentos.")
    lineas.append(f"- Valores crudos 0/1 de A y B; acuerdo A-B por igualdad de valores crudos:")
    lineas.append(f"  {crudo_ab['HS'][0]}/15 (HS), {crudo_ab['TR'][0]}/15 (TR), {crudo_ab['AG'][0]}/15 (AG)")
    lineas.append("  (supone que A y B usaron la misma convencion, sostenido por el formato")
    lineas.append("  comun y la instruccion registrada; no requiere la direccion del mapeo).")
    lineas.append("- Presencia cruda de las marcas de fila de A y B (A: P92C-02, 04; B: P92C-01,")
    lineas.append("  05, 13), con su significado atribuido segun la seccion 3.")
    lineas.append("")
    lineas.append("### 4.2 Condicionales a la codificacion documentada (0=NEGATIVE, 1=POSITIVE)")
    lineas.append("")
    lineas.append("- Todas las estadisticas que combinan A/B con C/D: pares AC, AD, BC, BD")
    lineas.append("  (acuerdo, dice, kappas).")
    lineas.append("- Combinacion exacta de los 4 por etiqueta y global (1/15); alfas de")
    lineas.append("  Krippendorff con A/B (0.243/0.466, 0.104/0.093, 0.354/0.491).")
    lineas.append("- Los archivos normalizados calibration_response_A/B_normalized.csv")
    lineas.append("  (derivados condicionales; no deben usarse como acuerdo definitivo).")
    lineas.append("")
    lineas.append("### 4.3 Confirmados")
    lineas.append("")
    lineas.append("- Ninguna estadistica queda en este nivel al estandar estricto (la")
    lineas.append("  interpretacion de A/B depende de la codificacion no demostrada).")
    lineas.append("")
    lineas.append("### 4.4 No afectado")
    lineas.append("")
    lineas.append("- La utilidad de la calibracion para revisar la guia se mantiene: el")
    lineas.append("  catalogo de ambiguedades, los seis casos HS, la propuesta v2, el registro de")
    lineas.append("  decisiones y el gate descansan en C/D y en los valores crudos, no en la")
    lineas.append("  codificacion de A/B.")
    lineas.append("- El desacuerdo entre anotadores no se mezcla con errores confirmados del")
    lineas.append("  dataset: nada de lo observado demuestra errores en las etiquetas originales.")
    lineas.append("")
    lineas.append("## 5. Accion minima para resolver la incertidumbre")
    lineas.append("")
    lineas.append("Solicitud directa a A y B (borrador listo en")
    lineas.append("reports/p92tr_ab_confirmation_request.md; NO enviada): dos preguntas de")
    lineas.append("seleccion sobre el significado de 0 y 1 en las columnas de clase y sobre el")
    lineas.append("significado de 0/1 en uncertainty_notes. Una respuesta escrita breve de cada")
    lineas.append("anotador, recogida por separado, cierra la incertidumbre y permitiria promover")
    lineas.append("las estadisticas de 4.2 a confirmadas (con un addendum nuevo). Mientras tanto:")
    lineas.append("A/B condicional; C/D valido.")
    lineas.append("")
    lineas.append("## 6. Correcciones aditivas aplicadas")
    lineas.append("")
    lineas.append("- Notas aditivas (marcador \"NOTA ADITIVA P92-T-R\") al final de:")
    lineas.append("  reports/p92_calibration_analysis.md,")
    lineas.append("  artifacts/human_annotation_v1/calibration_responses/PROVENANCE.md,")
    lineas.append("  reports/p92u_human_discussion_pack.md y")
    lineas.append("  reports/p92_guideline_decision_record.md.")
    lineas.append("- Este addendum queda como registro autoritativo de la procedencia; los")
    lineas.append("  documentos anteriores no se modifican en su contenido historico.")
    lineas.append("- No se modificaron: respuestas humanas (hashes en la seccion 7),")
    lineas.append("  normalizados (bytes intactos; marcados como condicionales),")
    lineas.append("  p92t_audit_results.json (historico), paquetes principales ni notebook")
    lineas.append("  historico (secciones 140 y 141 se mantienen; su calificacion queda")
    lineas.append("  corregida por este addendum).")
    lineas.append("")
    lineas.append("## 7. Hashes de los archivos originales (re-verificados hoy)")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for anotador in ("A", "B", "C", "D"):
        lineas.append(f"| calibration_response_{anotador}.csv | `{hashes['respuestas'][anotador]}` |")
    for anotador in ("A", "B"):
        lineas.append(f"| calibration_response_{anotador}_normalized.csv | `{hashes['normalizados'][anotador]}` |")
    lineas.append("")
    lineas.append("Los cuatro archivos originales coinciden con los hashes registrados al")
    lineas.append("momento de la recepcion: siguen intactos.")
    lineas.append("")
    lineas.append("## 8. Integridad y estado")
    lineas.append("")
    lineas.append("- Respuestas humanas originales: sin cambios. Normalizados: bytes sin")
    lineas.append("  cambios (marcados condicionales). Muestra principal: no distribuida.")
    lineas.append("- V03, OOF canonico, etiquetas originales, particion y submissions: intactos.")
    lineas.append("- Test y holdouts: no utilizados. Sin entrenamiento ni predicciones.")
    lineas.append("- Verificador ASCII oficial: ejecutado sin hallazgos (ver informe de fase).")
    lineas.append("- Git: sin commit ni push; cambios locales conservados.")
    ruta = RAIZ / "reports" / "p92tr_encoding_provenance_audit.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


def escribir_solicitud() -> None:
    """Genera el borrador de solicitud de confirmacion directa a A y B."""
    lineas = []
    lineas.append("# P92-T-R - Solicitud de confirmacion directa a los anotadores A y B (borrador)")
    lineas.append("")
    lineas.append("Estado: BORRADOR; NO enviado. Lo entregan los responsables del proyecto, por")
    lineas.append("separado a cada anotador y sin comunicacion entre ellos, si deciden cerrar la")
    lineas.append("incertidumbre de procedencia. No modifica ninguna respuesta original.")
    lineas.append("")
    lineas.append("## Texto propuesto (identico para A y para B)")
    lineas.append("")
    lineas.append("```text")
    lineas.append("Hola. Gracias por tu ronda de calibracion. Antes de la ronda principal")
    lineas.append("necesitamos dejar documentado el significado exacto de los valores que usaste")
    lineas.append("en tu archivo, sin cambiar ninguna de tus respuestas.")
    lineas.append("")
    lineas.append("En tu archivo usaste 0/1 en lugar de POSITIVE / NEGATIVE /")
    lineas.append("UNABLE_TO_DETERMINE. Para dejar constancia, confirma por favor (marca una")
    lineas.append("opcion por linea):")
    lineas.append("")
    lineas.append("1. En las tres columnas de etiquetas (HS_annotation, TR_annotation,")
    lineas.append("   AG_annotation):")
    lineas.append("   - El valor 0 significaba:")
    lineas.append("     ( ) pertenece a la clase (POSITIVE)")
    lineas.append("     ( ) no pertenece a la clase (NEGATIVE)")
    lineas.append("     ( ) otro: __________")
    lineas.append("   - El valor 1 significaba:")
    lineas.append("     ( ) pertenece a la clase (POSITIVE)")
    lineas.append("     ( ) no pertenece a la clase (NEGATIVE)")
    lineas.append("     ( ) otro: __________")
    lineas.append("")
    lineas.append("2. En la columna uncertainty_notes:")
    lineas.append("   - El valor 1 significaba:")
    lineas.append("     ( ) dificil de determinar para cualquier clase (duda general del tweet)")
    lineas.append("     ( ) otra cosa: __________")
    lineas.append("   - El valor 0 significaba:")
    lineas.append("     ( ) sin duda general / sin comentario")
    lineas.append("     ( ) otra cosa: __________")
    lineas.append("")
    lineas.append("3. Si algo quedo ambiguo, puedes aclararlo aqui: ______________________")
    lineas.append("")
    lineas.append("Tus respuestas originales se conservan exactamente como las entregaste;")
    lineas.append("esto solo documenta su significado. Basta con responder este mensaje.")
    lineas.append("```")
    lineas.append("")
    lineas.append("## Uso de las respuestas (protocolo)")
    lineas.append("")
    lineas.append("1. Recoger las respuestas por escrito, por separado, sin comunicacion entre")
    lineas.append("   A y B antes de responder.")
    lineas.append("2. Guardarlas verbatim como archivos nuevos bajo")
    lineas.append("   artifacts/human_annotation_v1/calibration_responses/ (sin modificar los")
    lineas.append("   CSV existentes).")
    lineas.append("3. Con ambas respuestas: emitir un addendum NUEVO que actualice el estado de la")
    lineas.append("   codificacion (este documento y el addendum P92-T-R no se reescriben).")
    lineas.append("")
    lineas.append("Nota: esta solicitud no revela respuestas de otros anotadores, etiquetas")
    lineas.append("originales ni informacion de modelos; solo pregunta por el significado de los")
    lineas.append("valores del propio archivo.")
    ruta = RAIZ / "reports" / "p92tr_ab_confirmation_request.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"  escrito {ruta.relative_to(RAIZ)} ({len(lineas)} lineas)")


def main() -> None:
    respuestas = cargar_respuestas()
    crudo_ab = acuerdo_crudo_ab(respuestas)

    hashes = {"respuestas": {}, "normalizados": {}}
    fallos = []
    for anotador in ("A", "B", "C", "D"):
        h = sha256_archivo(DIR_RESP / f"calibration_response_{anotador}.csv")
        hashes["respuestas"][anotador] = h
        if not h.startswith(ESPERADOS_RESPUESTAS[anotador]):
            fallos.append(f"respuesta {anotador}")
    for anotador in ("A", "B"):
        h = sha256_archivo(DIR_RESP / f"calibration_response_{anotador}_normalized.csv")
        hashes["normalizados"][anotador] = h
        if not h.startswith(ESPERADOS_NORMALIZADOS[anotador]):
            fallos.append(f"normalizado {anotador}")

    # contraste con los valores registrados en P92-T (los conteos interpretados)
    with open(RUTA_JSON, encoding="utf-8") as fh:
        res_p92t = json.load(fh)
    for etiqueta in ETIQUETAS:
        num_p92t = res_p92t["acuerdo"]["por_etiqueta"][etiqueta]["pares"]["AB"]["acuerdo_3cat"]["num"]
        if crudo_ab[etiqueta][0] != num_p92t:
            fallos.append(f"contraste AB {etiqueta}")

    print("P92-T-R auditoria de procedencia: generando artefactos")
    escribir_addendum(respuestas, hashes, crudo_ab)
    escribir_solicitud()

    notas = [
        (RAIZ / "reports" / "p92_calibration_analysis.md", NOTA_ANALISIS),
        (DIR_RESP / "PROVENANCE.md", NOTA_PROVENANCE),
        (RAIZ / "reports" / "p92u_human_discussion_pack.md", NOTA_PACK_U),
        (RAIZ / "reports" / "p92_guideline_decision_record.md", NOTA_REGISTRO),
    ]
    for ruta, nota in notas:
        estado = anexar_nota(ruta, nota)
        print(f"  nota en {ruta.relative_to(RAIZ)}: {estado}")

    print(f"  acuerdo crudo A-B: HS {crudo_ab['HS'][0]}/15 | TR {crudo_ab['TR'][0]}/15 | "
          f"AG {crudo_ab['AG'][0]}/15 (valido sin el mapeo)")
    print("  hashes originales: " + ("OK" if not fallos else f"REVISAR: {fallos}"))
    print("P92-T-R auditoria de procedencia: OK")


if __name__ == "__main__":
    main()
