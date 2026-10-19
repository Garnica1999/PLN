#!/usr/bin/env python
"""
P96-00: auditoria forense del registro experiments.csv (P1/P2 de P96).

Solo lectura: no entrena, no predice, no toca particiones reservadas. Calcula:

- conteos reales (filas, ids unicos, duplicados, campos vacios);
- existencia y hash de los scripts referenciados;
- referencias a artefactos y su existencia;
- presencia de metricas, pliegues y provenance;
- configuraciones duplicadas (candidatas a runs/replicas);
- fase derivada del id y clasificacion de trazabilidad por fila.

Salida: reports/p96_experiment_integrity_matrix.csv (una fila por registro).

Uso:
    /home/carlos/venv/bin/python experiments/p96_00_registro_audit.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path


def ascii_seguro(texto: str) -> str:
    """Translitera a ASCII puro los textos verbatim copiados del registro."""
    normalizado = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in normalizado if not unicodedata.combining(c))
    return "".join(c if ord(c) < 128 else "?" for c in sin_tildes)

RAIZ = Path(__file__).resolve().parents[1]
RUTA_REGISTRO = RAIZ / "experiments.csv"
SALIDA = RAIZ / "reports" / "p96_experiment_integrity_matrix.csv"

PATRON_ARTEFACTO = re.compile(
    r"(features|models|submissions|reports|data|experiments)/[\w./+-]+\.(npz|joblib|csv|json|md)"
)
PATRON_FASE = re.compile(r"^([A-Z]+[0-9]*)")


def sha16(ruta: Path) -> str:
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def fase_de(experiment_id: str) -> str:
    """Deriva la fase a partir del prefijo del id."""
    coincidencia = PATRON_FASE.match(experiment_id)
    base = coincidencia.group(1) if coincidencia else experiment_id.split("-")[0]
    if base == "KAGGLE_TOXIC_2026_PUJ":
        return "sonda-inicial"
    if base.startswith("V") and base[1:].isdigit():
        return f"fase-v{base[1:]}"
    if base.startswith("P") and base[1:].isdigit():
        return f"fase-{base}"
    return base.lower()


def politica_umbral(texto: str) -> str:
    """Clasifica la politica de umbral declarada en el registro."""
    t = texto.lower()
    if "anidado" in t or "nested" in t:
        return "nested"
    if "congelad" in t:
        return "congelados-v03"
    if "0.5" in t or "umbral fijo" in t or "0,5" in t:
        return "fijo-0.5"
    if "umbral" in t or "thr" in t:
        return "mencionado-sin-detalle"
    return "no-declarado"


def main() -> None:
    """Ejecuta la auditoria del registro y escribe la matriz."""
    if SALIDA.exists():
        raise SystemExit(f"Ya existe {SALIDA}; no se sobrescribe (P96).")
    with open(RUTA_REGISTRO, encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.DictReader(fh))
    columnas = list(filas[0].keys())

    ids = [f["experiment_id"].strip() for f in filas]
    duplicados = [i for i, n in Counter(ids).items() if n > 1 and i]
    vacios = [j for j, i in enumerate(ids) if not i]
    faltantes = {c: sum(1 for f in filas if not f[c].strip()) for c in columnas}

    # scripts y su existencia/hash
    seguimiento = subprocess.run(
        ["git", "-C", str(RAIZ), "ls-files"], capture_output=True, text=True
    ).stdout.splitlines()
    rastreados = set(seguimiento)
    cache_sha: dict[str, tuple[bool, str]] = {}

    def estado_script(ruta_rel: str) -> tuple[bool, str, bool]:
        if not ruta_rel:
            return False, "", False
        if ruta_rel not in cache_sha:
            ruta = RAIZ / ruta_rel
            if ruta.exists():
                cache_sha[ruta_rel] = (True, sha16(ruta), True)
            else:
                cache_sha[ruta_rel] = (False, "", False)
        existe, sha, _ = cache_sha[ruta_rel]
        return existe, sha, ruta_rel in rastreados

    filas_salida = []
    sin_script = 0
    scripts_faltantes: list[str] = []
    configuraciones: Counter[tuple[str, str, str]] = Counter()
    for f in filas:
        eid = f["experiment_id"].strip()
        scripts = [s.strip() for s in re.split(r"[;\n]", f.get("script", "")) if s.strip()]
        existencias = [estado_script(s) for s in scripts]
        todos_existen = bool(scripts) and all(e[0] for e in existencias)
        algunos_faltan = [s for s, e in zip(scripts, existencias) if not e[0]]
        scripts_faltantes.extend(algunos_faltan)
        if not scripts:
            sin_script += 1
        sha_codigo = existencias[0][1] if existencias else ""
        en_git = existencias[0][2] if existencias else False

        texto = " ".join(f.get(c, "") for c in ("result", "notes", "decision", "validation"))
        artefactos = list(dict.fromkeys(
            m.group(0) for m in PATRON_ARTEFACTO.finditer(
                " ".join([f.get("notes", ""), f.get("result", ""), f.get("decision", "")]))))
        artefacto_existe = [a for a in artefactos if (RAIZ / a).exists()]

        f1 = f.get("f1", "").strip()
        tiene_metrica = bool(f1) or "f1_media_columnas" in f.get("result", "")
        tiene_pliegues = "pliegue" in texto.lower()
        tiene_provenance = todos_existen or bool(artefacto_existe)

        if existencias and existencias[0][0] and tiene_metrica:
            nivel = "A_codigo_y_metrica"
            if artefacto_existe:
                nivel = "A+_codigo_metrica_artefacto"
        elif tiene_metrica:
            nivel = "C_metrica_sin_codigo"
        elif scripts:
            nivel = "B_codigo_sin_metrica"
        else:
            nivel = "D_sin_codigo_sin_metrica"

        clave_config = (
            f.get("representation", "").strip().lower(),
            f.get("model", "").strip().lower(),
            f.get("hyperparameters", "").strip().lower(),
        )
        if clave_config[1] and "no aplica" not in clave_config[1]:
            configuraciones[clave_config] += 1

        filas_salida.append({
            "experiment_id": eid,
            "phase": fase_de(eid),
            "date": f.get("date", ""),
            "script": "; ".join(scripts),
            "script_exists": "si" if todos_existen else ("parcial" if any(e[0] for e in existencias) else "no"),
            "script_sha16": sha_codigo,
            "script_en_git": "si" if en_git else "no",
            "artifact_hints": "; ".join(artefactos[:3]),
            "artifact_exists": "si" if artefacto_existe else ("no" if artefactos else ""),
            "has_metric": "si" if tiene_metrica else "no",
            "has_folds": "si" if tiene_pliegues else "no",
            "threshold_policy": politica_umbral(texto),
            "validation": ascii_seguro(f.get("validation", "")[:80]),
            "trace_level": nivel,
            "decision": ascii_seguro(f.get("decision", "")[:120]),
        })

    dup_config = sum(1 for n in configuraciones.values() if n > 1)
    total = len(filas)
    con_script = sum(1 for r in filas_salida if r["script_exists"] != "no")
    con_artefacto = sum(1 for r in filas_salida if r["artifact_exists"] == "si")
    con_metrica = sum(1 for r in filas_salida if r["has_metric"] == "si")
    con_pliegues = sum(1 for r in filas_salida if r["has_folds"] == "si")
    niveles = Counter(r["trace_level"] for r in filas_salida)
    fases = Counter(r["phase"] for r in filas_salida)

    with open(SALIDA, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas_salida[0].keys()),
                                  lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_salida)

    print("P96-00 auditoria del registro: ejecutada")
    print(f"  total_rows: {total} | ids unicos: {len(set(ids) - {''})} | duplicados: {duplicados or 'ninguno'} | ids vacios: {len(vacios)}")
    print(f"  campos vacios por columna: { {c: n for c, n in faltantes.items() if n} }")
    print(f"  filas sin script: {sin_script} | scripts faltantes: {len(set(scripts_faltantes))}")
    if scripts_faltantes:
        for s in sorted(set(scripts_faltantes))[:10]:
            print(f"    falta: {s}")
    print(f"  con script existente: {con_script}/{total} | con artefacto existente: {con_artefacto} | con metrica: {con_metrica} | con pliegues: {con_pliegues}")
    print(f"  niveles de trazabilidad: {dict(niveles)}")
    print(f"  configuraciones duplicadas (mismo repr+modelo+hiper): {dup_config} grupos")
    ejemplos = [(k[1][:40], n) for k, n in configuraciones.most_common(5) if n > 1]
    print(f"  top grupos repetidos: {ejemplos}")
    print(f"  fases: {dict(fases)}")
    print(f"  salida: {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
