"""Reconciliation of two status labels that ended up sharing one field name.

The ambiguity, stated plainly
-----------------------------
Two documents in this dossier carry a key called `research_status` and they do not agree:

    a_cost_sensitive_downweight_final_evidence.json   CANDIDATE_PENDING_GOVERNANCE
    toxic_candidate_cost_sensitive_manifest.json      EXPERIMENTAL_RESEARCH_CLOSED

Neither is wrong. They answer different questions, and the field name was doing two jobs:

    are we still searching for new mechanisms?     no  -> the research is closed
    is the candidate adopted?                      no  -> it is unconfirmed
    is the governance decision made?               no  -> it is pending

`CANDIDATE_PENDING_GOVERNANCE` was an earlier single label that collapsed the second and third
questions into one. It is not incorrect, it is underspecified, and leaving both in the record
without saying so is the kind of ambiguity that later gets read as a contradiction.

What this script does
---------------------
It does not touch the candidate, does not retrain anything, and does not rewrite the historical
documents. It states the three axes separately, marks the earlier collapsed label as superseded
while preserving what it meant, and leaves both files as they were.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_SHA, V03_SUB

REPORTS = PROJECT_ROOT / "reports"
MD_G = REPORTS / "toxic_candidate_governance_note.md"
MD_R = REPORTS / "toxic_candidate_status_reconciliation.md"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-09] {mensaje}", flush=True)


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
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def main() -> None:
    """Resuelve la ambiguedad de las dos etiquetas de estado.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-09 reconciliacion de las dos etiquetas de estado")
    log("   0 experimentos, 0 entrenamientos, 0 submissions")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    log(f"   V03 intacto | commit {commit}")

    # Los dos documentos, con lo que dicen y donde viven.
    a = json.loads((REPORTS / "a_cost_sensitive_downweight_final_evidence.json")
                   .read_text(encoding="utf-8"))
    b = json.loads((REPORTS / "toxic_candidate_cost_sensitive_manifest.json")
                   .read_text(encoding="utf-8"))
    log("")
    log("   LAS DOS ETIQUETAS, EN SU SITIO")
    log(f"      a_cost_sensitive_downweight_final_evidence.json   "
        f"{a['research_status']}")
    log(f"      toxic_candidate_cost_sensitive_manifest.json      {b['research_status']}")

    ejes = [
        {"eje": "research_status", "valor": "EXPERIMENTAL_RESEARCH_CLOSED",
         "pregunta": "se siguen buscando mecanismos nuevos?",
         "respuesta": "no",
         "desde": "commit bafd6c6"},
        {"eje": "candidate_status", "valor": "PROMISING_UNCONFIRMED",
         "pregunta": "el candidato esta adoptado?",
         "respuesta": "no",
         "desde": "commit bafd6c6"},
        {"eje": "governance_status", "valor": "PENDING",
         "pregunta": "la decision de gobernanza esta tomada?",
         "respuesta": "no",
         "desde": "commit bafd6c6"},
    ]
    log("")
    log("   LOS TRES EJES, SEPARADOS")
    for e in ejes:
        log(f"      {e['eje']:20} {e['valor']:30} {e['pregunta']}")

    resuelto = {
        "hay_ambiguedad_en_el_registro": True,
        "donde": ["reports/a_cost_sensitive_downweight_final_evidence.json",
                  "reports/toxic_candidate_cost_sensitive_manifest.json",
                  "DECISIONS.md, D-201 y D-205"],
        "que_la_causo": ("un unico campo llamado `research_status` haciendo dos trabajos: decir "
                         "si la busqueda sigue abierta, y decir si el candidato esta adoptado"),
        "etiqueta_anterior": "CANDIDATE_PENDING_GOVERNANCE",
        "estado_de_la_etiqueta_anterior": "SUPERSEDED, no incorrecta sino subespecificada",
        "lo_que_significaba": ("que el candidato no estaba adoptado y que la decision de "
                               "gobernanza estaba pendiente. Correcto, pero colapsaba dos ejes "
                               "en una palabra"),
        "resolucion": ejes,
        "documentos_historicos_modificados": False,
        "cifras_historicas_modificadas": False,
        "nota": ("los dos ficheros se dejan como estan. Reescribir un informe ya publicado para "
                 "corregir su vocabulario es exactamente lo que este proyecto no hace: se "
                 "preserva el original y se documenta la relacion"),
    }
    log("")
    log("   LA RESOLUCION")
    log("      etiqueta anterior: CANDIDATE_PENDING_GOVERNANCE -> SUPERSEDED")
    log("      significaba lo correcto, pero colapsaba dos ejes en una palabra")
    log("      los ficheros historicos NO se modifican")

    # ---------- el documento de reconciliacion ----------
    R: list[str] = []
    W = R.append
    W("# Reconciliacion de las dos etiquetas de estado")
    W("")
    W(f"Generado por `experiments/p69_09_reconciliar_estado.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La ambiguedad")
    W("")
    W("Dos documentos de este expediente llevan una clave llamada `research_status` y no")
    W("coinciden:")
    W("")
    W("| documento | valor |")
    W("|---|---|")
    W(f"| `a_cost_sensitive_downweight_final_evidence.json` | "
      f"`{a['research_status']}` |")
    W(f"| `toxic_candidate_cost_sensitive_manifest.json` | `{b['research_status']}` |")
    W("")
    W("Ninguna es incorrecta. Responden a preguntas distintas, y el nombre del campo estaba")
    W("haciendo dos trabajos a la vez.")
    W("")
    W("## Los tres ejes, separados")
    W("")
    W("| eje | valor | pregunta | respuesta |")
    W("|---|---|---|---|")
    for e in ejes:
        W(f"| `{e['eje']}` | `{e['valor']}` | {e['pregunta']} | {e['respuesta']} |")
    W("")
    W("## Que era `CANDIDATE_PENDING_GOVERNANCE`")
    W("")
    W("```text")
    W("estado      SUPERSEDED")
    W("no es       incorrecta, es subespecificada")
    W("significaba que el candidato no estaba adoptado y que la decision de")
    W("            gobernanza estaba pendiente")
    W("problema    colapsaba dos ejes en una sola palabra")
    W("```")
    W("")
    W("## Que NO se ha hecho")
    W("")
    W("```text")
    W("no se han modificado los dos ficheros historicos")
    W("no se ha cambiado ninguna cifra historica")
    W("no se ha reescrito ningun informe ya publicado")
    W("```")
    W("")
    W("Reescribir un informe publicado para corregir su vocabulario es exactamente lo que este")
    W("proyecto no hace. Se preserva el original y se documenta la relacion.")
    W("")
    W("## El estado, sin ambiguedad")
    W("")
    W("```text")
    W("research_status     EXPERIMENTAL_RESEARCH_CLOSED")
    W("candidate_status    PROMISING_UNCONFIRMED")
    W("governance_status   PENDING")
    W("")
    W("V03                          FROZEN")
    W("submission                   0")
    W("test_development_use         0")
    W("leaderboard_tuning           0")
    W("```")
    W("")
    W("A partir de aqui, cuando un documento de este proyecto diga `research_status`, se refiere")
    W("al primer eje y solo al primero.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_09_reconciliar_estado.py`. Sin push.*")
    MD_R.write_text(a_ascii("\n".join(R)), encoding="ascii")
    log("")
    log(f"   {MD_R.name} escrito")

    # ---------- la seccion equivalente en la nota de gobernanza ----------
    nota = MD_G.read_text(encoding="ascii")
    marca = "## El estado, sin ambiguedad"
    seccion = """## El estado, sin ambiguedad

Dos documentos del expediente usan la clave `research_status` con valores distintos, y no son
contradictorios: responden a preguntas distintas. Se separan aqui.

| eje | valor | pregunta |
|---|---|---|
| `research_status` | `EXPERIMENTAL_RESEARCH_CLOSED` | se siguen buscando mecanismos nuevos? |
| `candidate_status` | `PROMISING_UNCONFIRMED` | el candidato esta adoptado? |
| `governance_status` | `PENDING` | la decision de gobernanza esta tomada? |

`CANDIDATE_PENDING_GOVERNANCE` fue una etiqueta anterior que colapsaba los dos ultimos ejes en
una palabra. Queda `SUPERSEDED`: no incorrecta, sino subespecificada. Detalle completo en
`reports/toxic_candidate_status_reconciliation.md`.

"""
    if marca not in nota:
        i = nota.index("> V03 remains the best verified leaderboard result.")
        MD_G.write_text(nota[:i] + seccion + nota[i:], encoding="ascii")
        log(f"   seccion anadida a {MD_G.name}")
    else:
        log(f"   {MD_G.name} ya tenia la seccion")

    documentos = [MD_R, MD_G, REPORTS / "toxic_candidate_status_reconciliation.json"]
    (REPORTS / "toxic_candidate_status_reconciliation.json").write_text(
        json.dumps({"familia": "P69_RECONCILIACION_ESTADO", "experiment_id": "P69-09",
                    "fecha": inicio.isoformat(timespec="seconds"),
                    "naturaleza": "RECONCILIACION DOCUMENTAL. 0 experimentos",
                    "resuelto": resuelto, "ejes": ejes,
                    "documentos_historicos_modificados": False,
                    "formulacion": "V03 remains the best verified leaderboard result."},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sucios = [d.name for d in documentos if any(x > 127 for x in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log("P69-09 OK | ambiguedad resuelta en tres ejes separados")
    log("   research_status     EXPERIMENTAL_RESEARCH_CLOSED")
    log("   candidate_status    PROMISING_UNCONFIRMED")
    log("   governance_status   PENDING")
    log("   CANDIDATE_PENDING_GOVERNANCE -> SUPERSEDED")
    log("   ficheros historicos modificados: 0")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
