"""Verifica la cobertura experimento -> script y artefactos del proyecto AGE.

Uso:
    python verifica_experimentos.py [--raiz RUTA]

Localiza la raiz del proyecto por la presencia de ``experiments.csv`` (o usa ``--raiz``),
lee el registro completo (220 experimentos), los scripts (``scripts/``), los artefactos
JSON de resultados (``experiments/*.json``) y el inventario de la auditoria
(``experiments/auditoria_v1v37_inventario.json`` si esta presente).

Reporta:
  - experimentos ejecutados con script productor (propio o compartido);
  - experimentos sin script por diseno (registros documentales, pre-registros, enmiendas,
    bloqueados, auditorias, actas de decision y las falsaciones V45/V50/V52);
  - experimentos que requieren revision (ejecutados sin script identificable);
  - presencia de los JSON de resultados citados por cada experimento.

Codigo de salida 0 si no hay faltantes que requieran revision; 1 en caso contrario.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

#: Falsaciones documentadas sin script dedicado (asi consta en el inventario del proyecto).
DOCUMENTADAS_SIN_SCRIPT = {"v45", "v50", "v52"}
#: Tipos de registro que no implican ejecucion.
SIN_EJECUCION = {"DOCUMENTAL", "PREREGISTRO", "ENMIENDA", "BLOQUEADO"}


def localizar_raiz(argumento: str | None) -> Path:
    """Devuelve la raiz del proyecto (por experiments.csv o por --raiz)."""
    if argumento:
        raiz = Path(argumento).resolve()
        if not (raiz / "experiments.csv").exists():
            raise SystemExit(f"--raiz no contiene experiments.csv: {raiz}")
        return raiz
    actual = Path.cwd().resolve()
    for candidato in (actual, *actual.parents):
        if (candidato / "experiments.csv").exists():
            return candidato
    raise SystemExit("No se encontro experiments.csv; ejecutar desde la raiz del proyecto.")


def token_familia(ident: str) -> str | None:
    """Token de familia/numeracion usado para emparejar con nombres de scripts."""
    normalizado = ident.lower().replace("-", "").replace(" ", "")
    if normalizado.startswith("adv"):
        coincidencia = re.match(r"adv0?(\d)", normalizado)
        return f"advanced_0{coincidencia.group(1)}" if coincidencia else None
    for patron, plantilla in (
            (r"e(\d{3})", "e{0}"), (r"v(\d{2})", "v{0}_"), (r"ph(\d+)", "phase{0}"),
            (r"b([0-5])$", "baselines_b0_b5"), (r"v(\d)", "v{0}_")):
        coincidencia = re.match(patron, normalizado)
        if coincidencia:
            return plantilla.format(coincidencia.group(1))
    if normalizado.startswith("age"):
        return "age_"
    return None


def main() -> int:
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--raiz", default=None, help="raiz del proyecto (por defecto: sube hasta experiments.csv)")
    argumentos = analizador.parse_args()

    raiz = localizar_raiz(argumentos.raiz)
    filas = list(csv.DictReader(open(raiz / "experiments.csv", encoding="utf-8")))
    scripts = {ruta.name.lower(): ruta for ruta in (raiz / "scripts").rglob("*")
               if ruta.suffix in (".py", ".sh")}
    textos = {nombre: ruta.read_text(encoding="utf-8", errors="ignore")
              for nombre, ruta in scripts.items()}
    carpeta_json = raiz / "experiments"
    trabajos = {ruta.name for ruta in carpeta_json.glob("*.json")} if carpeta_json.exists() else set()

    tipos: dict[str, str] = {}
    inventario = carpeta_json / "auditoria_v1v37_inventario.json"
    if inventario.exists():
        datos = json.loads(inventario.read_text(encoding="utf-8"))
        tipos = {fila["experiment_id"]: fila.get("tipo_registro", "") for fila in datos.get("filas", [])}

    con_script, sin_ejecucion, documentadas, revision = [], [], [], []
    json_presente = json_ausente = 0
    faltantes_json = []

    for fila in filas:
        ident = fila["experiment_id"]
        tipo = tipos.get(ident, "")
        artefactos = [a.strip() for a in (fila.get("artefactos") or "").split(",") if a.strip()]
        citados = [a for a in artefactos if a.startswith("experiments/") and a.endswith(".json")]
        if citados:
            if all((raiz / a).exists() for a in citados):
                json_presente += 1
            else:
                json_ausente += 1
                faltantes_json.append(ident)

        if tipo in SIN_EJECUCION:
            sin_ejecucion.append((ident, tipo))
            continue

        # (1) script que menciona algun artefacto del experimento
        productor = None
        for artefacto in artefactos:
            nombre = artefacto.split("/")[-1]
            if not nombre:
                continue
            for script, texto in textos.items():
                if nombre in texto:
                    productor = script
                    break
            if productor:
                break
        # (2) emparejamiento por familia/numeracion
        if not productor:
            token = token_familia(ident)
            if token:
                candidatos = [s for s in scripts if token in s]
                productor = candidatos[0] if candidatos else None
        if productor:
            con_script.append((ident, productor))
        elif re.match(r"(aud|audit|g14)", ident.lower()):
            sin_ejecucion.append((ident, "AUDITORIA"))
        elif ident.upper().startswith("D-"):
            sin_ejecucion.append((ident, "DECISION"))
        elif any(ident.lower().startswith(v) for v in DOCUMENTADAS_SIN_SCRIPT):
            documentadas.append(ident)
        else:
            revision.append(ident)

    total = len(filas)
    print("=== VERIFICACION DE EXPERIMENTOS (AGE) ===")
    print(f"Raiz: {raiz}")
    print(f"Registro: {total} experimentos | scripts: {len(scripts)} | JSON de resultados: {len(trabajos)}")
    print()
    print(f"Con script productor (propio o compartido): {len(con_script)}/{total}")
    print(f"Sin script por diseno:")
    desglose = Counter(tipo for _, tipo in sin_ejecucion)
    for tipo, cuenta in sorted(desglose.items()):
        print(f"  - {tipo:14} {cuenta}")
    print(f"  - falsaciones documentadas sin script: {len(documentadas)} {sorted(documentadas)}")
    print(f"Requieren revision (ejecutados sin script): {len(revision)}")
    for ident in revision:
        print(f"  REVISAR: {ident}")
    print()
    print(f"JSON de resultados citados: presentes {json_presente} | ausentes {json_ausente}")
    if faltantes_json:
        for ident in faltantes_json:
            print(f"  JSON AUSENTE: {ident}")
    print("=== FIN DE LA VERIFICACION ===")
    return 0 if not revision and not faltantes_json else 1


if __name__ == "__main__":
    sys.exit(main())
