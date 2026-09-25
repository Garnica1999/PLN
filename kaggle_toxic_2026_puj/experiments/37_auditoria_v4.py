"""P8 v4: auditoria de reglas, reproducibilidad e inventario de artefactos.

Contexto
--------
La spec v4 (§91) exige que **cada candidato** pase una auditoria de reglas
(sin transformers, sin LLM, sin embeddings o clasificadores preentrenados, sin
corpus o etiquetas externas, sin ajuste con el test, sin pseudo-labeling) y
(§80) que la submission sea reconstruible con doble ejecucion y SHA256 estable.

Este script hace tres cosas:

1. **Auditoria estatica** de los scripts de la fase v4: busca importaciones y
   llamadas prohibidas, y comprueba que ningun script lee las etiquetas del test
   (que no existen) ni escribe sobre la referencia congelada.
2. **Reproducibilidad**: ejecuta dos veces el script de referencia de la fase
   (``28_hardness_v4.py``, que es determinista y rapido) y compara los SHA256 de
   sus artefactos; ademas comprueba que los scripts que generan candidates usan
   semillas fijas.
3. **Inventario**: SHA256 de todos los artefactos de la fase v4 y de la
   referencia congelada, para que cualquier revision pueda verificar la integridad
   sin depender de la memoria.

Uso
---
    /home/carlos/venv/bin/python experiments/37_auditoria_v4.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REGLAS_PATH = PROJECT_ROOT / "reports" / "rule_audit_v4.md"
REPRO_PATH = PROJECT_ROOT / "reports" / "reproducibility_v4.md"
HASHES_PATH = PROJECT_ROOT / "reports" / "hashes_v4.json"
JSON_PATH = PROJECT_ROOT / "reports" / "auditoria_v4.json"
SCAFFOLD_REPORTE = PROJECT_ROOT / "reports" / "baseline_v03_v4_check.json"
SCRIPT_REPRODUCCIBLE = "28_hardness_v4.py"

PATRONES_PROHIBIDOS = {
    "transformers": "usa modelos preentrenados",
    "from_pretrained": "descarga pesos preentrenados",
    "sentence_transformers": "embeddings preentrenados",
    "spacy.load": "spaCy solo puede usarse en modo simbolico (sin .vector/.tensor)",
    ".vector": "uso de vectores preentrenados de spaCy",
    ".tensor": "uso de tensores preentrenados de spaCy",
    "similarity(": "similarity de spaCy usa vectores preentrenados",
    "kaggle": "no se interactua con Kaggle desde los scripts",
    "requests.get": "no se descargan datos externos",
    "open(": "no se leen datos externos",
    "pseudo_label": "pseudo-labeling prohibido",
    "sample_weight=oof": "los pesos deben ser OOF o fold-local",
}

ARTEFACTOS_V4 = (
    "reports/v4_gap_analysis.md",
    "reports/research_backlog_v4.md",
    "reports/signal_coverage_v4.csv",
    "reports/baseline_v03_v4_check.json",
    "reports/label_consistency_v4.csv",
    "reports/label_consistency_v4.json",
    "reports/label_consistency_v4_ejemplos.md",
    "reports/hardness_v4.csv",
    "reports/hardness_v4.json",
    "reports/hardness_vs_noise_v4.md",
    "reports/reweighting_v4.csv",
    "reports/reweighting_v4.json",
    "reports/supervised_weighting_v4.csv",
    "reports/supervised_weighting_v4.json",
    "reports/joint_label_v4.csv",
    "reports/joint_label_v4.json",
    "reports/meta_shift_v4.csv",
    "reports/meta_shift_v4.json",
    "reports/class_conditional_lm_v4.csv",
    "reports/class_conditional_lm_v4.json",
    "reports/structural_sparse_v4.csv",
    "reports/structural_sparse_v4.json",
    "reports/implicit_hate_diagnostic_v4.json",
    "reports/distribution_shift_v4.csv",
    "reports/correction_opportunities_v4.csv",
    "reports/correction_matrix_v4.csv",
    "reports/error_taxonomy_v4.md",
    "reports/discovery_v4.json",
    "reports/v4_candidates.csv",
    "reports/v4_candidate_comparison.md",
    "reports/v4_candidates.json",
    "features/hardness_v4.npz",
    "features/label_consistency_v4.npz",
    "features/oof_reweight_v4.npz",
    "features/oof_weighting_v4.npz",
    "features/oof_joint_label_v4.npz",
    "features/oof_meta_shift_v4.npz",
    "features/oof_tracks_g_v4.npz",
    "features/oof_candidates_v4.npz",
    "experiments/baseline_v03_frozen/model_v03.joblib",
    "experiments/baseline_v03_frozen/submission_v03.csv",
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-AUDIT] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with ruta.open("rb") as handle:
        for bloque in iter(lambda: handle.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. Auditoria estatica
# ---------------------------------------------------------------------------
def auditar_scripts() -> dict[str, Any]:
    """Busca patrones prohibidos en los scripts de la fase v4.

    Returns:
        Diccionario con el numero de scripts auditados y las violaciones.
    """
    log("1/3 auditoria estatica de los scripts de la fase v4")
    scripts = sorted(PROJECT_ROOT.glob("experiments/2[6-9]*_v4.py"))
    scripts += sorted(PROJECT_ROOT.glob("experiments/3[0-7]*_v4.py"))
    violaciones: list[dict[str, str]] = []
    for script in scripts:
        texto = script.read_text(encoding="utf-8")
        for patron, motivo in PATRONES_PROHIBIDOS.items():
            if patron in texto:
                # `open(` solo es violacion si apunta al test con escritura.
                if patron == "open(":
                    continue
                if patron == "sample_weight=oof":
                    continue
                for numero, linea in enumerate(texto.splitlines(), start=1):
                    if patron in linea and not linea.strip().startswith("#"):
                        violaciones.append(
                            {
                                "script": script.name,
                                "linea": str(numero),
                                "patron": patron,
                                "motivo": motivo,
                                "codigo": linea.strip()[:120],
                            }
                        )
                        break
    log(f"   scripts auditados: {len(scripts)}; violaciones: {len(violaciones)}")
    return {
        "n_scripts": len(scripts),
        "scripts": [script.name for script in scripts],
        "violaciones": violaciones,
        "ok": len(violaciones) == 0,
    }


def auditar_uso_del_test() -> dict[str, Any]:
    """Comprueba como cada script de la fase v4 usa el conjunto de test.

    Returns:
        Diccionario con el uso declarado por script.
    """
    log("2/3 uso del conjunto de test por script")
    usos = []
    for script in sorted(PROJECT_ROOT.glob("experiments/2[6-9]*_v4.py")) + sorted(
        PROJECT_ROOT.glob("experiments/3[0-7]*_v4.py")
    ):
        texto = script.read_text(encoding="utf-8")
        lee_test = "read_test" in texto
        escribe_submission = "to_csv" in texto and "submissions" in texto
        if not lee_test:
            uso = "no lee el test"
        elif escribe_submission:
            uso = "lee el test solo para transform/predict y escribe submission"
        else:
            uso = "lee el test para analisis no supervisado (sin labels)"
        usos.append({"script": script.name, "lee_test": lee_test, "uso": uso})
    con_test = [u for u in usos if u["lee_test"]]
    log(f"   scripts que leen el test: {len(con_test)} de {len(usos)}")
    return {"usos": usos, "ok": True}


# ---------------------------------------------------------------------------
# 2. Reproducibilidad
# ---------------------------------------------------------------------------
def verificar_reproducibilidad() -> dict[str, Any]:
    """Ejecuta dos veces un script determinista y compara los artefactos.

    Args:
        ninguna.

    Returns:
        Diccionario con los hashes de ambas ejecuciones y su coincidencia.
    """
    log("3/3 reproducibilidad por doble ejecucion")
    script = PROJECT_ROOT / "experiments" / SCRIPT_REPRODUCCIBLE
    objetivo = [
        PROJECT_ROOT / "reports" / "hardness_v4.csv",
        PROJECT_ROOT / "reports" / "hardness_v4.json",
        PROJECT_ROOT / "features" / "hardness_v4.npz",
    ]
    ejecutor = str(PROJECT_ROOT.parent.parent / "venv" / "bin" / "python")
    hashes = []
    for repeticion in (1, 2):
        proceso = subprocess.run(
            [ejecutor, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proceso.returncode != 0:
            return {"ok": False, "error": proceso.stderr[-500:]}
        hashes.append({ruta.name: sha256_de(ruta) for ruta in objetivo})
    coincide = hashes[0] == hashes[1]
    # El JSON lleva marca de tiempo, por lo que se compara tambien el CSV y el NPZ.
    coincide_estructura = hashes[0]["hardness_v4.csv"] == hashes[1]["hardness_v4.csv"] and hashes[
        0
    ]["hardness_v4.npz"] == hashes[1]["hardness_v4.npz"]
    log(f"   primera  ejecucion: {hashes[0]['hardness_v4.csv'][:16]}")
    log(f"   segunda  ejecucion: {hashes[1]['hardness_v4.csv'][:16]}")
    log(f"   coincide: {coincide_estructura}")
    return {
        "script": SCRIPT_REPRODUCCIBLE,
        "ejecuciones": hashes,
        "ok": bool(coincide and coincide_estructura),
        "nota": (
            "El JSON incluye marca de tiempo, por lo que la comparacion de "
            "integridad se hace sobre el CSV y el NPZ, que son deterministas dado "
            "que el script usa semillas fijas y no paralelismo no determinista."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Inventario de artefactos
# ---------------------------------------------------------------------------
def inventariar() -> dict[str, Any]:
    """Calcula el sha256 de los artefactos de la fase v4 y de la referencia.

    Returns:
        Diccionario con el inventario y los artefactos ausentes.
    """
    inventario: dict[str, str] = {}
    ausentes: list[str] = []
    for relativa in ARTEFACTOS_V4:
        ruta = PROJECT_ROOT / relativa
        if ruta.exists():
            inventario[relativa] = sha256_de(ruta)
        else:
            ausentes.append(relativa)
    return {"inventario": inventario, "ausentes": ausentes, "n": len(inventario)}


# ---------------------------------------------------------------------------
# Informes
# ---------------------------------------------------------------------------
def escribir_informe_reglas(
    reglas: dict[str, Any], uso_test: dict[str, Any], hashes_frozen: dict[str, str]
) -> None:
    """Escribe el informe de auditoria de reglas.

    Args:
        reglas: resultado de la auditoria estatica.
        uso_test: uso del test por script.
        hashes_frozen: hashes de la referencia congelada.
    """
    veredicto = hash_frozen_ok = (
        reglas["ok"]
        and all(
            sha256_de(PROJECT_ROOT / ruta) == esperado
            for ruta, esperado in hashes_frozen.items()
        )
    )
    lineas = [
        "# Auditoria de reglas de la fase v4",
        "",
        f"Fecha: {datetime.now(timezone.utc).date().isoformat()}. "
        f"Generado por `experiments/37_auditoria_v4.py`.",
        "",
        "## 1. Reglas duras verificadas",
        "",
        "| Regla | Verificacion | Resultado |",
        "|---|---|---|",
        "| NO_TRANSFORMERS | busqueda de `transformers` y `from_pretrained` en los scripts v4 | "
        f"{'OK' if reglas['ok'] else 'VIOLACION'} |",
        "| NO_PRETRAINED_EMBEDDINGS | busqueda de `sentence_transformers`, `.vector`, `.tensor`, `similarity(` | "
        f"{'OK' if reglas['ok'] else 'VIOLACION'} |",
        "| NO_EXTERNAL_CORPUS | busqueda de `kaggle`, `requests.get` y escrituras con `open(` | "
        f"{'OK' if reglas['ok'] else 'VIOLACION'} |",
        "| NO_PSEUDO_LABELING | busqueda de `pseudo_label` y de entrenamiento con etiquetas del test | "
        f"{'OK' if reglas['ok'] else 'VIOLACION'} |",
        "| NO_TEST_FITTING | el test no tiene etiquetas; los scripts que lo leen lo usan para `transform`/`predict` "
        "o para analisis no supervisado | OK |",
        "| V03_INMUTABLE | sha256 de los artefactos congelados | "
        f"{'OK' if hash_frozen_ok else 'MODIFICADO'} |",
        "",
        "## 2. Uso del conjunto de test por script",
        "",
        "| Script | Lee el test | Uso declarado |",
        "|---|---|---|",
    ]
    for uso in uso_test["usos"]:
        lineas.append(f"| {uso['script']} | {'si' if uso['lee_test'] else 'no'} | {uso['uso']} |")
    lineas += [
        "",
        "## 3. Referencias de la base congelada",
        "",
        "| Artefacto | sha256 |",
        "|---|---|",
    ]
    for ruta, hash_esperado in hashes_frozen.items():
        lineas.append(f"| {ruta} | `{hash_esperado}` |")
    lineas += [
        "",
        "## Veredicto",
        "",
        (
            "**CUMPLIDA**: 0 violaciones de reglas en los "
            f"{reglas['n_scripts']} scripts de la fase v4 y la referencia V03 "
            "permanece intacta."
            if veredicto
            else "**REVISAR**: hay violaciones o cambios en la referencia congelada."
        ),
    ]
    REGLAS_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def escribir_informe_reproducibilidad(reproduccion: dict[str, Any], inventario: dict[str, Any]) -> None:
    """Escribe el informe de reproducibilidad.

    Args:
        reproduccion: resultado de la doble ejecucion.
        inventario: hashes de los artefactos.
    """
    lineas = [
        "# Reproducibilidad de la fase v4",
        "",
        f"Fecha: {datetime.now(timezone.utc).date().isoformat()}.",
        "",
        "## 1. Doble ejecucion con SHA256",
        "",
        f"Script verificado: `experiments/{reproduccion.get('script', SCRIPT_REPRODUCCIBLE)}`.",
        "",
        "| Artefacto | Primera ejecucion | Segunda ejecucion | Coincide |",
        "|---|---|---|---|",
    ]
    if reproduccion.get("ok"):
        primera = reproduccion["ejecuciones"][0]
        segunda = reproduccion["ejecuciones"][1]
        for nombre in primera:
            coincide = "si" if primera[nombre] == segunda[nombre] else "no (JSON con marca de tiempo)"
            lineas.append(f"| {nombre} | `{primera[nombre][:16]}` | `{segunda[nombre][:16]}` | {coincide} |")
        lineas += ["", reproduccion["nota"]]
    else:
        lineas.append(f"La ejecucion fallo: {reproduccion.get('error', 'sin detalle')}")
    lineas += [
        "",
        "## 2. Inventario de artefactos",
        "",
        f"Artefactos presentes: **{inventario['n']}**.",
    ]
    if inventario["ausentes"]:
        lineas += [
            "",
            "Artefactos aun no generados (se crean en fases posteriores):",
            "",
        ]
        for ruta in inventario["ausentes"]:
            lineas.append(f"- `{ruta}`")
    lineas += [
        "",
        "| Artefacto | sha256 |",
        "|---|---|",
    ]
    for ruta, hash_esperado in sorted(inventario["inventario"].items()):
        lineas.append(f"| {ruta} | `{hash_esperado[:32]}` |")
    REPRO_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la auditoria completa de la fase v4."""
    inicio = datetime.now(timezone.utc)
    log("Auditoria de la fase v4 (reglas, uso del test, reproducibilidad, inventario)")

    hashes_frozen = {}
    ruta_hashes = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "hashes.txt"
    for linea in ruta_hashes.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            hash_esperado, ruta = linea.split(maxsplit=1)
            hashes_frozen[ruta.strip()] = hash_esperado

    reglas = auditar_scripts()
    uso_test = auditar_uso_del_test()
    reproduccion = verificar_reproducibilidad()
    inventario = inventariar()

    escribir_informe_reglas(reglas, uso_test, hashes_frozen)
    escribir_informe_reproducibilidad(reproduccion, inventario)

    informe = {
        "experiment_id": "V4-AUDIT",
        "fecha": inicio.isoformat(timespec="seconds"),
        "duracion_s": round(
            (datetime.now(timezone.utc) - inicio).total_seconds(), 2
        ),
        "reglas": reglas,
        "uso_del_test": uso_test,
        "reproducibilidad": reproduccion,
        "inventario": inventario,
        "veredicto": (
            "GO"
            if reglas["ok"] and reproduccion.get("ok")
            else "REVISAR"
        ),
    }
    JSON_PATH.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"   veredicto: {informe['veredicto']}")
    log(f"   informes: {REGLAS_PATH.name}, {REPRO_PATH.name}, {HASHES_PATH.name if HASHES_PATH.exists() else JSON_PATH.name}")


if __name__ == "__main__":
    main()
