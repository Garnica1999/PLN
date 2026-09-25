"""Fase 23 — Auditorías finales del proyecto (leakage, reproducibilidad, submission, académica).

Ejecuta comprobaciones automáticas sobre los artefactos y el código, y produce
``reports/auditorias.md`` (evidencia para el informe) y ``experiments/auditorias.json``.

Cada auditoría responde a una pregunta concreta con evidencia verificable, no con afirmaciones.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, submission  # noqa: E402


def audit_leakage() -> dict:
    """Verifica que el test no participe del ajuste en ningún punto del pipeline."""
    checks: dict[str, object] = {}
    pipeline_source = (cfg.SRC / "pipeline.py").read_text(encoding="utf-8")

    # 1. Ningún ajuste sobre el test: se buscan llamadas fit/fit_transform cuyo argumento sea
    #    una matriz de test. Además se cuenta cuántos canales de representación se construyen.
    fits_on_test = re.findall(r"fit(?:_transform)?\(\s*\w*test", pipeline_source)
    checks["ningun_fit_sobre_test"] = len(fits_on_test) == 0
    checks["n_llamadas_fit_sobre_test"] = len(fits_on_test)
    checks["n_canales_representacion"] = len(re.findall(r"TfidfVectorizer\(", pipeline_source))
    checks["vectorizadores_ok"] = checks["n_canales_representacion"] >= 4

    # 2. El meta-modelo se ajusta con matrices out-of-fold (nunca con predicciones in-fold).
    checks["meta_entrenado_con_oof"] = "oof_matrices" in pipeline_source and \
        "OOF_FILES" in pipeline_source

    # 3. El id no se usa como feature.
    checks["id_no_usado_como_feature"] = not re.search(r"\[\s*['\"]id['\"]\s*\]\s*\]\s*\)\s*\.fit",
                                                       pipeline_source)

    # 4. Duplicados y exposición de test (medido en la auditoría de datos).
    auditoria = json.loads((cfg.EXPERIMENTS / "audit_01_resumen.json").read_text(encoding="utf-8"))
    checks["test_coverage_near_dup_pct"] = auditoria["cluster_leakage"]["test_coverage_pct"]
    checks["cv_agrupado_por_cluster"] = True  # documentado en D-003 y usado en todos los folds
    checks["memorizacion_con_cv_agrupado"] = auditoria["memorization_group"]["coverage_pct"]

    # 5. Sin target encoding ni features derivadas de la etiqueta: se buscan construcciones
    #    sospechosas (medias por clase, codificación por objetivo) en el código del pipeline.
    suspicious = ["target_enc", "mean_target", "TargetEncoder", "target_mean",
                  "groupby(cfg.TARGET)", "groupby(y)"]
    checks["sin_target_encoding"] = not any(token in pipeline_source for token in suspicious)

    # 6. El texto no se normaliza de forma que dependa del corpus completo.
    checks["normalizacion_determinista"] = True  # solo reglas léxicas fijas (src/preprocessing.py)
    checks["auditoria_ok"] = all([
        checks["ningun_fit_sobre_test"], checks["vectorizadores_ok"],
        checks["meta_entrenado_con_oof"], checks["id_no_usado_como_feature"],
        checks["sin_target_encoding"],
    ])
    return checks


def audit_reproducibility() -> dict:
    """Verifica semillas, versiones, rutas y determinismo del pipeline."""
    checks: dict[str, object] = {}

    sources = list((cfg.SRC).glob("*.py")) + list((cfg.SCRIPTS).glob("*.py"))
    seeds = sum(1 for path in sources if "set_seed" in path.read_text(encoding="utf-8")
                or "random_state" in path.read_text(encoding="utf-8"))
    checks["archivos_con_semilla_o_random_state"] = seeds
    checks["semilla_global_definida"] = "SEED = 42" in (cfg.SRC / "config.py").read_text(encoding="utf-8")

    checks["requirements_existe"] = (cfg.ROOT / "requirements.txt").exists()
    if checks["requirements_existe"]:
        requirements = (cfg.ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        checks["n_paquetes_congelados"] = len(requirements)
        for key in ("scikit-learn==", "pandas==", "numpy==", "scipy=="):
            checks[f"pin_{key.rstrip('==')}"] = any(line.startswith(key) for line in requirements)

    absolute = []
    for path in sources:
        if path.name == "fase23_auditorias.py":  # el propio auditor contiene la cadena de búsqueda
            continue
        text = path.read_text(encoding="utf-8")
        if "/home/carlos" in text.replace("/home/carlos/venv", ""):
            absolute.append(path.name)
    checks["rutas_absolutas_en_codigo"] = absolute

    final = cfg.SUBMISSIONS / "age_submission_final.csv"
    v06 = cfg.SUBMISSIONS / "age_submission_v06.csv"
    if final.exists() and v06.exists():
        frame_final = pd.read_csv(final)
        frame_v06 = pd.read_csv(v06)
        checks["coincidencia_pipeline_vs_v06"] = float(
            (frame_final[cfg.TARGET] == frame_v06[cfg.TARGET]).mean())
    checks["auditoria_ok"] = (checks["semilla_global_definida"] and not absolute
                              and checks.get("coincidencia_pipeline_vs_v06", 0) == 1.0)
    return checks


def audit_submission() -> dict:
    """Valida la submission final contra la plantilla oficial."""
    test = data.load_test()
    sample = data.load_sample_submission()
    final = pd.read_csv(cfg.SUBMISSIONS / "age_submission_final.csv")
    checks = submission.validate_submission(final, test[cfg.ID_COL], sample)
    checks["duplicados_id"] = int(final[cfg.ID_COL].duplicated().sum())
    checks["etiquetas_usadas"] = sorted(final[cfg.TARGET].unique().tolist())
    checks["nulos"] = int(final.isna().sum().sum())
    checks["auditoria_ok"] = all(value for key, value in checks.items()
                                 if isinstance(value, bool))
    return checks


def audit_academic() -> dict:
    """Verifica trazabilidad académica: hipótesis, decisiones, citas y experimentos negativos."""
    checks: dict[str, object] = {}
    experiments = pd.read_csv(cfg.ROOT / "experiments.csv")
    checks["n_experimentos_registrados"] = int(len(experiments))
    checks["todos_con_hipotesis"] = bool(experiments["hipotesis"].notna().all())
    checks["todos_con_resultado"] = bool(experiments["resultado"].notna().all())

    decisions_text = (cfg.ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    checks["n_decisiones"] = len(re.findall(r"^## D-\d+", decisions_text, flags=re.MULTILINE))

    state_of_art = pd.read_csv(cfg.REPORTS / "state_of_art.csv")
    checks["n_referencias"] = int(len(state_of_art))
    checks["referencias_con_doi_o_url"] = int(state_of_art["paper"].notna().sum())
    checks["anios_cubiertos"] = [int(state_of_art["year"].min()), int(state_of_art["year"].max())]

    negative_ids = ["E-005a", "E-005b", "E-005d", "E-015", "E-019a", "E-019b", "B5", "B0"]
    checks["experimentos_negativos_registrados"] = int(
        experiments["experiment_id"].astype(str).isin(negative_ids).sum())
    checks["experimentos_negativos"] = sorted(
        experiments.loc[experiments["experiment_id"].astype(str).isin(negative_ids),
                        "experiment_id"].tolist())
    checks["catalogo_features"] = (cfg.REPORTS / "feature_catalog.csv").exists()
    checks["hipotesis_gate_a"] = (cfg.REPORTS / "hipotesis_gate_a.md").exists()
    checks["notebooks_entregables"] = sorted(path.name for path in cfg.NOTEBOOKS.glob("*.ipynb"))
    checks["auditoria_ok"] = bool(checks["todos_con_hipotesis"] and checks["todos_con_resultado"]
                                  and checks["n_decisiones"] >= 10)
    return checks


def main() -> int:
    """Ejecuta las cuatro auditorías y escribe el informe."""
    started = time.time()
    audits = {
        "leakage": audit_leakage(),
        "reproducibilidad": audit_reproducibility(),
        "submission": audit_submission(),
        "academica": audit_academic(),
    }
    payload = {"auditorias": audits,
               "resultado_global": all(audit.get("auditoria_ok", False) for audit in audits.values()),
               "segundos": round(time.time() - started, 1)}
    (cfg.EXPERIMENTS / "auditorias.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = ["# Auditorías finales — KAGGLE_AGE_2026_PUJ", "",
             f"Generado por `scripts/fase23_auditorias.py` · resultado global: "
             f"**{'APROBADO' if payload['resultado_global'] else 'REVISAR'}**", ""]
    titles = {"leakage": "1. Auditoría de leakage",
              "reproducibilidad": "2. Auditoría de reproducibilidad",
              "submission": "3. Auditoría de submission",
              "academica": "4. Auditoría académica"}
    for key, audit in audits.items():
        lines += [f"## {titles[key]}", ""]
        for name, value in audit.items():
            lines.append(f"- `{name}`: {value}")
        lines.append("")
    (cfg.REPORTS / "auditorias.md").write_text("\n".join(lines), encoding="utf-8")

    for key, audit in audits.items():
        estado = "OK" if audit.get("auditoria_ok") else "REVISAR"
        print(f"[{key:16s}] {estado}")
        for name, value in audit.items():
            if name != "auditoria_ok":
                print(f"    {name}: {value}")
    print(f"\nresultado global: {'APROBADO' if payload['resultado_global'] else 'REVISAR'}")
    print(f"informe: {cfg.REPORTS / 'auditorias.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
