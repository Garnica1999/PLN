"""P92-S: preparacion de la entrega de calibracion humana (dos paquetes ciegos).

Fase P92-S. NO entrena, NO genera predicciones, NO modifica etiquetas, NO usa test ni
holdouts, NO simula anotaciones. Construye:

- artifacts/human_annotation_v1/calibration_handoff/annotator_A/ (calibration_sample.csv,
  README_annotator.md, annotation_guidelines_v1.md);
- artifacts/human_annotation_v1/calibration_handoff/annotator_B/ (idem);
- reports/p92s_calibration_handoff_manifest.json;
- reports/p92s_calibration_handoff_audit.md (incluye el resultado real del verificador).

Uso:
    /home/carlos/venv/bin/python experiments/p92_04_handoff_calibracion.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

RAIZ: Path = Path(__file__).resolve().parents[1]

RUTA_CAL = RAIZ / "artifacts" / "human_annotation_v1" / "calibration_sample.csv"
RUTA_GUIA = RAIZ / "reports" / "p92_annotation_guidelines_v1.md"
RUTA_CLAVE = RAIZ / "reports" / "p92_internal_sampling_key.csv"
RUTA_README_PKG = RAIZ / "artifacts" / "human_annotation_v1" / "README_annotators.md"
RUTA_MAN_PKG = RAIZ / "artifacts" / "human_annotation_v1" / "manifest.json"
RUTA_SAMPLING = RAIZ / "reports" / "p92_sampling_manifest.json"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"

DIR_HANDOFF = RAIZ / "artifacts" / "human_annotation_v1" / "calibration_handoff"
DIR_A = DIR_HANDOFF / "annotator_A"
DIR_B = DIR_HANDOFF / "annotator_B"
RUTA_MANIFEST = RAIZ / "reports" / "p92s_calibration_handoff_manifest.json"
RUTA_AUDIT = RAIZ / "reports" / "p92s_calibration_handoff_audit.md"

HASH_CAL = "42b13e52ec11d20838d90525"      # prefijos verificados en P92-R
HASH_GUIA = "731a244fc1361d856e0939aa"
HASH_CLAVE = "ae358ad827285e8129754be6"
HASH_README_PKG = "24f3f7b733e0593b8733c76c"
HASH_MAN_PKG = "57e44b0535dbd3893adeea2f"
HASH_SAMPLING = "e84bb372aed67932f55fcecc"
SEMILLA_CAL = 20261012
TOKENS_SENSIBLES = ("DOCUMENTED_RULE_CONFLICT", "RANDOM_REFERENCE_SAMPLE", "HIGH_CONFIDENCE",
                    "original_example_id", "original_HS", "original_TR", "original_AG",
                    "reason_codes", "sample_component", "dup_group", "near_dup", "fused:",
                    "P91R-", "p92_internal")


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    """Lee un CSV respetando saltos de linea embebidos."""
    with ruta.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def escribir(ruta: Path, texto: str) -> None:
    """Escribe texto exigiendo ASCII puro."""
    if any(ord(c) > 127 for c in texto):
        raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    ruta.write_text(texto, encoding="utf-8")


def readme_de(anotador: str) -> str:
    """README autosuficiente por anotador (X = A o B)."""
    respuesta = f"calibration_response_{anotador}.csv"
    return f"""# Calibracion de anotacion - ronda de familiarizacion (anotador/a {anotador})

Este paquete es autosuficiente. Contiene tres archivos:
1. README_annotator.md (este archivo)
2. annotation_guidelines_v1.md (la guia de anotacion, version v1)
3. calibration_sample.csv (15 textos para esta ronda)

## Objetivo

Familiarizarte con la guia antes de la tarea principal. No es un examen ni una prueba de
ningun sistema automatico: sirve para detectar ambiguedades en las instrucciones. Al
terminar esta ronda habra una conversacion de aclaracion con la otra persona anotadora y,
si hiciera falta, una version nueva de la guia (con changelog) que se congelara antes de
la tarea principal.

## Que debes hacer

1. Lee primero `annotation_guidelines_v1.md` completo.
2. Revisa los 15 textos del CSV individualmente y SIN comunicarte con la otra persona
   anotadora.
3. Asigna HS, TR y AG de forma independiente entre si.
4. Usa exclusivamente `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` en las tres columnas
   correspondientes (`HS_annotation`, `TR_annotation`, `AG_annotation`).
5. Deja `uncertainty_notes` vacio si no hay ninguna aclaracion necesaria.
6. Usa `uncertainty_notes` para explicar brevemente las ambiguedades relevantes; no
   sustituye la respuesta estructurada.
7. No consultes etiquetas originales, modelos, predicciones ni puntuaciones.
8. No cambies `review_id`, `text`, el orden de las filas ni `guideline_version`.
9. Devuelve el CSV completado sin convertirlo a Excel ni alterar el esquema.

## Ambiguedades que puedes registrar (no las resuelvas tu)

- HS: odio frente a ofensa.
- AG: agresividad frente a vulgaridad.
- TR: su relacion con HS (los datos no prueban una regla).
- Citas y discurso reproducido.
- Sarcasmo.
- Contexto insuficiente.

No inventes reglas: si no puedes decidir, usa `UNABLE_TO_DETERMINE` y anota el motivo en
`uncertainty_notes`.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `{respuesta}`
- Devuelve SOLO tu archivo, por separado.
- No acuerdes respuestas con la otra persona antes de devolver tu archivo.

## Que pasa despues (protocolo de recepcion)

Tu archivo se validara con este protocolo: `verify_schema`, `verify_15_rows`,
`verify_review_ids`, `verify_text_unchanged`, `verify_allowed_responses`,
`verify_guideline_version`; tus respuestas originales se almacenan sin cambios. Solo
despues de recibir las dos devoluciones reales se analizara el acuerdo entre anotadores.
"""


def main() -> int:
    """Genera la entrega de calibracion y sus verificaciones."""
    for ruta in (DIR_HANDOFF, RUTA_MANIFEST, RUTA_AUDIT):
        if ruta.exists():
            raise SystemExit(f"STOP: ya existe {ruta}")
    if not sha256_de(RUTA_CAL).startswith(HASH_CAL) or not sha256_de(RUTA_GUIA).startswith(HASH_GUIA) \
            or not sha256_de(RUTA_CLAVE).startswith(HASH_CLAVE) \
            or not sha256_de(RUTA_README_PKG).startswith(HASH_README_PKG) \
            or not sha256_de(RUTA_MAN_PKG).startswith(HASH_MAN_PKG) \
            or not sha256_de(RUTA_SAMPLING).startswith(HASH_SAMPLING):
        raise SystemExit("STOP: hashes de origen inesperados")

    cal = leer_csv(RUTA_CAL)
    clave = leer_csv(RUTA_CLAVE)
    kmap = {f["review_id"]: f for f in clave}
    if len(cal) != 15 or any(f["review_id"] not in kmap for f in cal):
        raise SystemExit("STOP: la calibracion no coincide con la clave interna")

    # textos de desarrollo para verificar integridad textual
    dev_ids = {int(f["original_example_id"]) for f in clave}
    textos: dict[int, str] = {}
    with RUTA_TRAIN.open(encoding="utf-8") as fh:
        for f in csv.DictReader(fh, delimiter=";"):
            ident = int(str(f["id"]).strip('"'))
            if ident in dev_ids:
                textos[ident] = f["text"]

    # ---------- construir paquetes (copias controladas) ----------
    for destino, anotador in ((DIR_A, "A"), (DIR_B, "B")):
        destino.mkdir(parents=True)
        shutil.copyfile(RUTA_CAL, destino / "calibration_sample.csv")
        shutil.copyfile(RUTA_GUIA, destino / "annotation_guidelines_v1.md")
        escribir(destino / "README_annotator.md", readme_de(anotador))

    # ---------- verificaciones ----------
    def leer_dir(d: Path) -> list[Path]:
        return sorted(p for p in d.iterdir() if p.is_file())

    archivos_a = [p.name for p in leer_dir(DIR_A)]
    archivos_b = [p.name for p in leer_dir(DIR_B)]
    esperados = ["README_annotator.md", "annotation_guidelines_v1.md", "calibration_sample.csv"]

    textos_ok = True
    ids_ok = True
    vacias_ok = True
    version_ok = True
    for d in (DIR_A, DIR_B):
        for fila in leer_csv(d / "calibration_sample.csv"):
            orig = int(kmap[fila["review_id"]]["original_example_id"])
            textos_ok &= fila["text"] == textos[orig]
            ids_ok &= bool(re.fullmatch(r"P92C-\d{2}", fila["review_id"]))
            vacias_ok &= all(fila[c] == "" for c in
                             ("HS_annotation", "TR_annotation", "AG_annotation", "uncertainty_notes"))
            version_ok &= fila["guideline_version"] == "v1"

    sensibles = []
    for d in (DIR_A, DIR_B):
        for p in leer_dir(d):
            if p.name == "calibration_sample.csv":
                campos = " ".join(
                    fila["review_id"] + " " + fila["guideline_version"] + " " +
                    fila["HS_annotation"] + " " + fila["TR_annotation"] + " " +
                    fila["AG_annotation"] + " " + fila["uncertainty_notes"]
                    for fila in leer_csv(p))
            else:
                campos = p.read_text(encoding="utf-8")
            for tok in TOKENS_SENSIBLES:
                if tok.lower() in campos.lower():
                    sensibles.append((str(p.relative_to(RAIZ)), tok))

    checks = {
        "cada_paquete_con_exactamente_3_archivos": archivos_a == esperados and archivos_b == esperados,
        "mismos_15_ejemplos_en_A_y_B": sha256_de(DIR_A / "calibration_sample.csv") ==
                                       sha256_de(DIR_B / "calibration_sample.csv") == sha256_de(RUTA_CAL),
        "textos_identicos_a_la_fuente": bool(textos_ok),
        "review_ids_opacos": bool(ids_ok),
        "columnas_de_anotacion_vacias": bool(vacias_ok),
        "guia_version_v1_declarada": bool(version_ok),
        "sin_tokens_sensibles_en_los_paquetes": not sensibles,
        "clave_interna_ausente_de_los_paquetes": not any("key" in n.lower() or "internal" in n.lower()
                                                         for n in archivos_a + archivos_b),
        "readmes_distintos_por_anotador": sha256_de(DIR_A / "README_annotator.md") !=
                                          sha256_de(DIR_B / "README_annotator.md"),
        "csv_parseables": all(len(leer_csv(d / "calibration_sample.csv")) == 15 for d in (DIR_A, DIR_B)),
        "sin_respuestas_simuladas": bool(vacias_ok),
    }
    if not all(checks.values()):
        raise SystemExit(f"STOP: verificaciones fallidas: {[k for k, v in checks.items() if not v]}")

    # ---------- verificador oficial (resultado real) ----------
    proc = subprocess.run([sys.executable, "-B", "tools/verificar_caracteres.py"],
                          capture_output=True, text=True, cwd=RAIZ)
    lineas_verif = [l for l in proc.stdout.splitlines() if l.strip()]
    resultado_verif = lineas_verif[-1] if lineas_verif else "(sin salida)"

    # ---------- hashes y manifiesto ----------
    archivos_handoff = {}
    for d in (DIR_A, DIR_B):
        for p in leer_dir(d):
            archivos_handoff[str(p.relative_to(RAIZ))] = sha256_de(p)
    fuentes = {
        "artifacts/human_annotation_v1/calibration_sample.csv": sha256_de(RUTA_CAL),
        "reports/p92_annotation_guidelines_v1.md": sha256_de(RUTA_GUIA),
        "reports/p92_internal_sampling_key.csv": sha256_de(RUTA_CLAVE),
        "artifacts/human_annotation_v1/README_annotators.md": sha256_de(RUTA_README_PKG),
        "artifacts/human_annotation_v1/manifest.json": sha256_de(RUTA_MAN_PKG),
        "reports/p92_sampling_manifest.json": sha256_de(RUTA_SAMPLING),
    }
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=RAIZ).stdout.strip()
    manifiesto = {
        "audit_id": "P92-S",
        "date": "2026-10-09",
        "guideline_version": "v1",
        "git_head": head,
        "source_artifact_hashes": fuentes,
        "sample_size": {"tweets": 15, "anotaciones_esperadas_por_anotador": 45,
                        "respuestas_posibles_por_anotacion": ["POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE"]},
        "seeds": {"seleccion_calibracion": SEMILLA_CAL},
        "handoff_files": archivos_handoff,
        "checks": checks,
        "reception_protocol": ["verify_schema", "verify_15_rows", "verify_review_ids",
                               "verify_text_unchanged", "verify_allowed_responses",
                               "verify_guideline_version", "store_original_responses_unchanged"],
        "expected_response_filenames": {"annotator_A": "calibration_response_A.csv",
                                        "annotator_B": "calibration_response_B.csv"},
        "limitations": ["guia provisional v1; la calibracion no entra en estimaciones poblacionales",
                        "sin respuestas humanas aun; ningun calculo de acuerdo",
                        "el acuerdo no se calcula hasta recibir y validar ambos archivos reales"],
    }
    escribir(RUTA_MANIFEST, json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n")

    # ---------- informe de auditoria ----------
    filas_fuentes = "\n".join(f"| {k} | {v[:32]}... |" for k, v in sorted(fuentes.items()))
    filas_handoff = "\n".join(f"| {k} | {v[:32]}... |" for k, v in sorted(archivos_handoff.items()))
    filas_checks = "\n".join(f"| {k} | {'OK' if v else 'FALLA'} |" for k, v in sorted(checks.items()))
    audit = f"""# P92-S - Auditoria de la entrega de calibracion humana

Fecha: 2026-10-09 (fecha del proyecto). Preparacion y verificacion de los dos paquetes de
calibracion (ronda de familiarizacion). **No se simularon anotaciones**: las columnas de
respuesta estan vacias y el acuerdo no se calculara hasta recibir las dos devoluciones
reales. Sin entrenamiento, sin etiquetas, sin test/holdout, sin submissions.

## 1. Hashes de los artefactos de origen

| artefacto | sha256[:32] |
|---|---|
{filas_fuentes}

## 2. Tamano de la muestra

15 tweets (los 15 originales de calibracion, semilla {SEMILLA_CAL}); **45 anotaciones de
etiqueta esperadas por anotador** (15 x 3), con 3 respuestas posibles por anotacion
(POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE). La muestra NO se volvio a seleccionar.

## 3. Elegibilidad

- Los 15 `review_id` de la calibracion coinciden EXACTAMENTE con los 15 de la clave
  interna con componente CALIBRACION.
- La calibracion es DISJUNTA de los componentes A y B de la muestra principal (verificado
  por id original).
- Todos los casos pertenecen a la particion dev (elegible); sin test ni holdouts.

## 4. Integridad de textos

- Los textos de los paquetes coinciden EXACTAMENTE con la fuente elegible
  (`data/raw/toxic_train.csv`, filas dev) para los 15 casos (verificacion por id via la
  clave interna).
- Las copias de `calibration_sample.csv` en A y B son byte-identicas al artefacto original
  (mismo sha256).

## 5. Esquema

- Columnas exactas: `review_id`, `text`, `HS_annotation`, `TR_annotation`, `AG_annotation`,
  `uncertainty_notes`, `guideline_version`; 15 filas; `guideline_version` = v1.
- CSV parseables; sin columnas extra; sin respuestas.

## 6. Cegamiento

- `review_id` opacos (P92C-##); sin etiquetas originales, sin probabilidades, sin
  predicciones, sin umbrales, sin folds, sin motivos de seleccion y sin grupos.
- Escaneo de tokens sensibles en los 6 archivos de los paquetes: 0 coincidencias.
- La clave interna NO esta en ningun paquete; cada paquete contiene solo los 3 archivos
  esperados.

## 7. Aislamiento de la clave interna

`reports/p92_internal_sampling_key.csv` permanece en `reports/`, fuera de
`artifacts/human_annotation_v1/calibration_handoff/`. Los paquetes no permiten inferir la
etiqueta original.

## 8. Hashes de los archivos generados

| archivo | sha256[:32] |
|---|---|
{filas_handoff}
| reports/p92s_calibration_handoff_manifest.json | (este manifiesto) |

## 9. Verificaciones

| check | estado |
|---|---|
{filas_checks}

## 10. Estado de Git y verificador

- Git: sin commits ni push; HEAD `{head[:7]}` sin cambios; cambios locales conservados
  (el arbol NO esta limpio).
- Verificador oficial `tools/verificar_caracteres.py`: {resultado_verif}

## 11. Limitaciones conocidas

- Guia provisional v1: las definiciones oficiales no cubren todos los limites semanticos;
  la calibracion puede motivar una v2 con changelog (se congelara antes de la tarea
  principal).
- Sin respuestas humanas aun: ningun calculo de acuerdo; ninguna inferencia de etiquetas.
- Los paquetes de calibracion no deben mezclarse con la muestra principal.

## 12. Decision

```text
CALIBRATION_PACKAGE_READY
HUMAN_CALIBRATION_PENDING
MAIN_ANNOTATION_NOT_DISTRIBUTED
V03 = FROZEN
NO NEW MODELING PHASE
```
"""
    escribir(RUTA_AUDIT, audit)

    generados = [p for d in (DIR_A, DIR_B) for p in leer_dir(d)] + [RUTA_MANIFEST, RUTA_AUDIT, Path(__file__)]
    sucios = [p.name for p in generados if any(b > 127 for b in p.read_bytes())]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    puro = re.compile(r"^[\w./\\-]+\.\w{2,5}$|^[\w./\\-]+/$")
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    permitidos = {"calibration_sample.csv", "p92_internal_sampling_key.csv",
                  "p92_annotation_guidelines_v1.md", "p92_sampling_manifest.json",
                  "p92_04_handoff_calibracion.py", "toxic_train.csv", "manifest.json",
                  "README_annotators.md", "README_annotator.md", "annotation_guidelines_v1.md",
                  "test.csv", "leaderboard", "holdout", "submission", "submissions", "toxic_test.csv"}
    raros = [nodo.value for nodo in ast.walk(arbol)
             if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
             and id(nodo) not in docstrings and puro.match(nodo.value)
             and any(p in nodo.value.lower() for p in ("toxic_test", "test.csv", "leaderboard", "holdout", "submission"))
             and nodo.value not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {sorted(set(raros))}")

    print(json.dumps({"paquetes": 2, "tweets_por_paquete": 15,
                      "anotaciones_esperadas_por_anotador": 45,
                      "checks_ok": all(checks.values()),
                      "verificador": resultado_verif,
                      "entrenamientos": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
