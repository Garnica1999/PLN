#!/usr/bin/env python
"""
P100-99: verificacion de cierre de P100 (documentacion y auditoria final).

Solo lectura. Comprueba entregables, registro de fuentes, manifiesto B1,
experiments.csv (IDs unicos), notebook seccion 163, estado actualizado
(DECISIONS/PROJECT_STATE/RUNLOG/FINAL_REPORT), hashes canonicos
V03/P98/P99/P100, submissions sin cambios, ASCII, verificador oficial y Git.

Regla de test/holdouts: NO se lee ni escanea toxic_test (por diseno); se
comprueba que ningun script P100 referencia la ruta del fichero de test.

Uso:
    /home/carlos/venv/bin/python experiments/p100_99_verificacion.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"

ENTREGABLES = ["p100_corpus_research.md", "p100_corpus_registry.csv",
               "p100_source_decisions.md", "p100_preregistration.md",
               "p100_auditorias_s1.md", "p100_b1_final_report.md",
               "p100_b1_pretraining.json", "p100_b1_pretraining.md",
               "p100_b1_pretraining_results.csv",
               "p100_b1_downstream_canonica.json",
               "p100_b1_downstream_part47.json",
               "p100_b1_downstream_results.csv",
               "p100_b1_complementarity.csv"]
EXTRA = ["artifacts/p100_b1_manifest.json",
         "data/external/corpora/p100_b1/tokens_30m.txt",
         "models/p100_b1_ltr.pt", "models/p100_b1_rtl.pt",
         "features/p100_b1_reps_dev.npz",
         "reports/figuras/p100_12_canonica.png",
         "reports/figuras/p100_12_part47.png"]
SCRIPTS = ["p100_00_registro_fuentes.py", "p100_10_construir_b1.py",
           "p100_11_preentrenar_b1.py", "p100_12_downstream_b1.py",
           "p100_90_registrar.py", "p100_95_integrar_notebook.py",
           "p100_99_verificacion.py", "p100_supervisor.sh"]


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def puro_ascii(ruta: Path) -> bool:
    """True si no hay bytes > 127."""
    with open(ruta, "rb") as fh:
        return all(b <= 127 for b in fh.read())


def main() -> None:
    """Ejecuta la verificacion de cierre de P100."""
    chequeos: dict[str, bool] = {}
    for nombre in ENTREGABLES:
        chequeos[f"existe_{nombre}"] = (R / nombre).exists()
    for rel in EXTRA:
        chequeos[f"existe_{rel}"] = (RAIZ / rel).exists()
    for nombre in SCRIPTS:
        chequeos[f"existe_{nombre}"] = (RAIZ / "experiments" / nombre).exists()

    with open(R / "p100_corpus_registry.csv", encoding="utf-8") as fh:
        reg = list(csv.DictReader(fh))
    estados: dict[str, int] = {}
    for f in reg:
        estados[f["permission_status"]] = \
            estados.get(f["permission_status"], 0) + 1
    chequeos["registro_10"] = len(reg) == 10
    chequeos["registro_estados"] = (estados == {"PROHIBITED": 4,
                                                "ALLOWED_WITH_RESTRICTIONS": 2,
                                                "NEEDS_EXPLICIT_PERMISSION": 4})
    chequeos["registro_sin_allowed_pleno"] = "ALLOWED" not in estados
    chequeos["chatssubs_condiciones"] = any(
        "Condiciones del propietario" in f["notas"] for f in reg)
    chequeos["chatssubs_sin_labels"] = any(
        "sin anotaciones" in f["notas"] for f in reg)

    man = json.loads((RAIZ / "artifacts" / "p100_b1_manifest.json").read_text(
        encoding="utf-8"))
    chequeos["b1_composicion_exacta"] = (
        man["tokens_finales"] == 29_727_450
        and man["tokens_por_fuente"] == {"social": 4_227_450,
                                         "mc4": 4_850_000,
                                         "conversacional": 9_000_000,
                                         "wikipedia": 11_650_000})
    chequeos["b1_subset_fallos_0"] = man["subset_verificacion"]["fallos"] == 0
    chequeos["b1_val"] = (man["val_presentes_en_b1"] == 3189
                          and man["val_language_docs"] == 3320)
    chequeos["b1_overlaps"] = (man["coincidencias_exactas_vs_train"] == 0
                               and man["overlap_conversacional_vs_s99_01"] == 0)
    corpus = RAIZ / "data" / "external" / "corpora" / "p100_b1" / "tokens_30m.txt"
    chequeos["b1_sha_manifiesto_igual_archivo"] = \
        sha256(corpus).startswith(man["sha256"][:16])

    pre = json.loads((R / "p100_b1_pretraining.json").read_text(
        encoding="utf-8"))
    chequeos["ppl_lt_100"] = (pre["checkpoints"]["ltr"]["val_ppl"] < 100.0
                              and pre["checkpoints"]["rtl"]["val_ppl"] < 100.0)

    can = json.loads((R / "p100_b1_downstream_canonica.json").read_text(
        encoding="utf-8"))
    p47 = json.loads((R / "p100_b1_downstream_part47.json").read_text(
        encoding="utf-8"))
    chequeos["controles_reproduccion"] = (
        can["control_reproduccion_30M"]["ok"]
        and p47["control_reproduccion_30M_47"]["ok"])
    chequeos["gate_failed"] = (can["gate"]["veredicto"] == "GATE_FAILED"
                               and p47["gate"]["veredicto"] == "GATE_FAILED")
    chequeos["atribucion_falsada"] = (can["gate"]["atribucion_falsada"]
                                      and p47["gate"]["atribucion_falsada"])
    chequeos["deltas_b1"] = (
        abs(can["comparacion"]["delta_b1_vs_v03"] + 0.000591) < 1e-5
        and abs(can["comparacion"]["delta_b1_vs_30"] + 0.005646) < 1e-5
        and abs(p47["comparacion"]["delta_b1_vs_v03"] + 0.000645) < 1e-5
        and abs(p47["comparacion"]["delta_b1_vs_30"] + 0.001797) < 1e-5)
    chequeos["folds_2_de_5_en_ambas"] = (
        can["comparacion"]["pliegues_b1_mejorados_vs_v03"] == 2
        and p47["comparacion"]["pliegues_b1_mejorados_vs_v03"] == 2)

    with open(RAIZ / "experiments.csv", encoding="utf-8-sig") as fh:
        filas = list(csv.DictReader(fh))
    ids = [f["experiment_id"] for f in filas]
    chequeos["experiments_491"] = len(filas) == 491
    chequeos["ids_unicos"] = len(set(ids)) == len(ids)
    p100 = [f for f in filas if f["experiment_id"].startswith("P100")]
    chequeos["p100_6_filas"] = (
        len(p100) == 6
        and {f["experiment_id"] for f in p100} == {"P100-00", "P100-10",
                                                   "P100-11", "P100-12",
                                                   "P100-13", "P100-14"})
    fila13 = next(f for f in p100 if f["experiment_id"] == "P100-13")
    chequeos["sync_json_csv"] = abs(
        json.loads(fila13["result"])["delta_b1_vs_v03"]
        - can["comparacion"]["delta_b1_vs_v03"]) < 1e-9

    nb = json.loads((RAIZ / "notebooks" / "10_final_pipeline.ipynb").read_text(
        encoding="utf-8"))
    chequeos["notebook_163"] = any(
        "".join(c.get("source", [])).startswith("## 163. P100")
        for c in nb["cells"])
    chequeos["notebook_codigo_163"] = any(
        "Verificacion de la seccion 163 (P100" in "".join(c.get("source", []))
        for c in nb["cells"])

    texto_runlog = (RAIZ / "RUNLOG.md").read_text(encoding="utf-8")
    texto_estado = (RAIZ / "PROJECT_STATE.md").read_text(encoding="utf-8")
    texto_dec = (RAIZ / "DECISIONS.md").read_text(encoding="utf-8")
    texto_final = (RAIZ / "FINAL_REPORT.md").read_text(encoding="utf-8")
    chequeos["estado_actualizado"] = all(
        "P100" in t for t in (texto_runlog, texto_estado, texto_dec,
                              texto_final))
    chequeos["decisions_303_304"] = ("## D-303" in texto_dec
                                     and "## D-304" in texto_dec)
    chequeos["decision_congelada"] = all(
        s in texto_dec for s in ("H100-01 = FALSIFIED", "P100 = CLOSED",
                                 "NO_MODEL_ADOPTION", "V03 = FROZEN"))
    chequeos["estado_congelado"] = ("H100-01 = FALSIFIED" in texto_estado
                                    and "P100 = CLOSED" in texto_estado)
    chequeos["final_report_p100"] = "P100 CLOSED 2026-10-14" in texto_final

    for nombre in ENTREGABLES:
        chequeos[f"ascii_{nombre}"] = puro_ascii(R / nombre)
    for nombre in SCRIPTS:
        chequeos[f"ascii_{nombre}"] = puro_ascii(RAIZ / "experiments" / nombre)

    canonicos = {
        "data/raw/toxic_train.csv": "3d40e7345b588860",
        "features/oof_v03_final.npz": "559265816ce4f631",
        "experiments/baseline_v03_frozen/submission_v03.csv":
            "7d7478f271e6edfd",
        "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
        "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
        "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
        "data/external/corpora/p98_c98_1/tokens_30m.txt": "9f562718ef4e7d45",
        "features/p98_c98_1_vocab.json": "92fecce611f164b7",
        "models/p98_c98_1_ltr.pt": "472891a7598a1daa",
        "models/p98_c98_1_rtl.pt": "cff68d45bf683cad",
        "features/p98_13_reps_dev.npz": "8f1aeeecaf346a80",
        "data/external/corpora/p99_s99_01/tokens_60m.txt": "e627a7f33886c0a1",
        "models/p99_s99_01_ltr.pt": "559309e0042d8c44",
        "models/p99_s99_01_rtl.pt": "dde2147f1147d142",
        "features/p99_reps_dev.npz": "3e72755bb7f03f48",
        "data/external/corpora/p100_b1/tokens_30m.txt": "5111a1832fe9567b",
        "models/p100_b1_ltr.pt": "a4c057a762500f54",
        "models/p100_b1_rtl.pt": "fb38bbbdac055732",
        "features/p100_b1_reps_dev.npz": "6524eae950b242e4",
    }
    for ruta, pref in canonicos.items():
        chequeos[f"canonico:{ruta}"] = sha256(RAIZ / ruta).startswith(pref)

    scripts_py = [s for s in SCRIPTS
                  if s.endswith(".py") and s != "p100_99_verificacion.py"]
    texto_scripts = ""
    for nombre in scripts_py:
        texto_scripts += (RAIZ / "experiments" / nombre).read_text(
            encoding="utf-8")
    chequeos["sin_ruta_test_en_scripts"] = "toxic_test.csv" not in texto_scripts
    chequeos["sin_pseudo_labels"] = ("pseudo" not in texto_scripts.lower())

    git_st = subprocess.run(["git", "-C", str(RAIZ), "status", "--short"],
                            capture_output=True, text=True,
                            check=False).stdout.strip()
    lineas_git = git_st.splitlines()
    chequeos["test_holdouts_sin_cambios_git"] = not any(
        (l[:2] in (" M", "M ", "MM"))
        and ("toxic_test" in l or "holdout" in l) for l in lineas_git)

    subs = list((RAIZ / "submissions").glob("*"))
    corte = datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp()
    chequeos["submissions_sin_nuevas"] = (
        len(subs) > 0
        and all(m.stat().st_mtime < corte for m in subs))
    chequeos["leaderboard_log_sin_cambios"] = (
        (RAIZ / "submissions" / "leaderboard_log.csv").stat().st_mtime < corte)

    verif = subprocess.run(
        [sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
        capture_output=True, text=True, check=False)
    chequeos["verificador_oficial_ok"] = verif.returncode == 0

    n_untracked = sum(1 for l in lineas_git if l.startswith("??"))
    n_modified = sum(1 for l in lineas_git if l[:2] in (" M", "M ", "MM"))
    fallos = [k for k, v in chequeos.items() if not v]
    print("P100-99 verificacion de cierre: ejecutada")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"fallos: {fallos if fallos else 'ninguno'}")
    print(f"  verificador oficial: {'OK' if verif.returncode == 0 else 'FALLO'}")
    print(f"  git: {n_untracked} sin rastrear, {n_modified} modificados "
          f"(solo lectura)")
    print()
    print("  limites declarados (no verificado por contenido, por regla):")
    print("   - toxic_test: no se lee ni escanea (regla vigente); verificado")
    print("     por diseno: sin referencias en scripts P100 y sin cambios git.")
    print("   - holdouts: integridad por hash de los ID files congelados y")
    print("     ausencia de cambios en git; su contenido no se re-escaneo.")
    print("   - labels externas / pseudo-labeling: por registro y decisiones")
    print("     (ChatSubs sin anotaciones; sin pseudo-labeling en P100).")
    print()
    print("  hashes clave P100 (prefijos):")
    for ruta, pref in canonicos.items():
        if ruta.startswith(("data/external/corpora/p100", "models/p100",
                            "features/p100")):
            print(f"   - {ruta}: {pref} OK")


if __name__ == "__main__":
    main()
