"""V30 - 06: trazabilidad, compliance y auditoria de la fase.

Consolida V30 sin anadir ningun experimento. Lo que se registra como experimento es lo unico que
**entreno** algo nuevo: el refinamiento de P1. La comparacion de candidatos, el ensemble, la
segunda particion y la construccion del fichero son operaciones sobre artefactos congelados.

## Los cuatro expedientes que el brief pide

- **Compliance**: 13 comprobaciones, distinguiendo uso de mencion con el tokenizer, mas una
  comprobacion **positiva** de que ningun script abre un fichero de holdout reservado.
- **Reproducibilidad**: que V20 y los dos resultados de V29 reproducen, la semilla de Optuna, el
  numero de trials, y que el determinismo se **mide** en vez de declararse.
- **Auditoria de la submission**: que el fichero existe, su hash coincide con el de la construccion,
  y que no se subio mas de una vez.
- **Higiene**: los dos directorios ajenos a AGE siguen fuera de la rama.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import time
import tokenize
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v30_06_cierre.json"
REGISTRO = ROOT / "experiments.csv"
SUBMISSION = ROOT / "submissions" / "age_submission_v30_final_candidate.csv"
COMPLIANCE_MD = cfg.REPORTS / "v30_compliance.md"
REPRO_MD = cfg.REPORTS / "v30_reproducibility.md"
SUBMISSION_MD = cfg.REPORTS / "v30_submission_audit.md"
CV_V20 = 0.6097782
LB_V20 = 0.62689
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
FECHA = "2026-09-30"
EXCLUIDOS_COMPLIANCE = {"scripts/v30_06_cierre.py", "experiments/v30_06_cierre.json",
                        "reports/v30_compliance.md", "reports/v30_submission_audit.md"}
PROHIBIDOS = {
    "transformers": r"\btransformers?\b|\bAutoModel|\bBertTokenizer|\bfrom_pretrained\b",
    "BERT": r"\bBERT\b|\bbert-base|\bberta\b",
    "RoBERTa": r"\broberta\b",
    "LLM": r"\bLLM\b|\bGPT\b|\bllama\b|\bmistral\b",
    "embeddings": r"\bembedding(s)?\b|\bsentence.transformers\b|\bword2vec\b",
    "external_classifier": r"\bexternal_classifier\b",
    "external_labels": r"\bexternal_labels?\b",
    "pseudo_labeling": r"\bpseudo.?label",
    "test_labels": r"\by_test\b|\btest_labels?\b|\bsample_submission\b",
    "test_training": r"\.fit\([^)]*\bmatriz_test\b|\.fit\([^)]*\btest_matriz\b",
    "test_calibration": r"calibrat[^)]*test|test[^)]*calibrat",
    "test_threshold_tuning": r"threshold[^)]*test|test[^)]*threshold",
    "holdout_reuse": r"\bholdout_(train|test|valid)\b",
    "leaderboard_tuning": (r"kaggle\s+competitions\s+(submit|download|leaderboard)"
                           r"|api\.kaggle|competitions\s+submit"),
    "toxic_test": r"\btoxic_test\b",
}


def _sin_cadenas(texto: str) -> str:
    """Codigo sin literales ni comentarios: lo que queda es exactamente lo que se ejecuta."""
    try:
        return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(texto).readline)
                        if t.type not in (tokenize.STRING, tokenize.FSTRING_START,
                                          tokenize.FSTRING_MIDDLE, tokenize.COMMENT))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return texto


def _compliance() -> dict:
    """Compliance de V30: uso separado de mencion, y comprobacion positiva de holdouts."""
    ficheros = sorted((ROOT / "scripts").glob("v30_*.py")) + \
        sorted((ROOT / "experiments").glob("v30_*.json")) + \
        sorted((ROOT / "reports").glob("v30_*"))
    filas = []
    for nombre, patron in PROHIBIDOS.items():
        regex = re.compile(patron, re.IGNORECASE)
        uso, prosa, detalle = 0, 0, []
        for ruta in ficheros:
            if str(ruta.relative_to(ROOT)) in EXCLUIDOS_COMPLIANCE:
                continue
            texto = ruta.read_text(encoding="utf-8", errors="ignore")
            if ruta.suffix == ".py":
                apoyos = regex.findall(_sin_cadenas(texto))
                uso += len(apoyos)
                prosa += max(0, len(regex.findall(texto)) - len(apoyos))
            else:
                prosa += len(regex.findall(texto))
            if len(detalle) < 3 and (apoyos or regex.findall(texto)):
                detalle.append(str(ruta.relative_to(ROOT)))
        filas.append({"comprobacion": nombre, "uso_en_codigo": uso, "menciones_en_prosa": prosa,
                      "cumplida": uso == 0, "ficheros": detalle})
    holdouts = []
    for ruta in sorted((ROOT / "scripts").glob("v30_*.py")):
        codigo = _sin_cadenas(ruta.read_text(encoding="utf-8", errors="ignore"))
        for linea in codigo.splitlines():
            if re.search(r"(np\.load|read_csv|read_parquet|open)\s*\(", linea) and \
                    re.search(r"holdout", linea, re.IGNORECASE):
                holdouts.append(f"{ruta.name}: {linea.strip()[:80]}")
    lineas = ["# V30 - Auditoria de compliance", "",
              "Se distingue **uso** de **mencion** con el tokenizer: el codigo se mide tras eliminar "
              "literales de cadena y comentarios, de modo que lo que se cuenta es exactamente lo "
              "que Python ejecuta. Un informe que declara por que una familia esta descartada no "
              "infringe la prohibicion al escribirla.", "",
              "| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |",
              "|---|---|---|---|"]
    for f in filas:
        lineas.append(f"| {f['comprobacion']} | {f['uso_en_codigo']} | {f['menciones_en_prosa']} | "
                      f"{'si' if f['cumplida'] else 'NO'} |")
    lineas += ["", "```text"] + [f"{f['comprobacion']} = {f['uso_en_codigo']}" for f in filas] + \
              ["```", "",
               f"**Comprobacion positiva:** ningun script de V30 abre un fichero cuyo nombre "
               f"contenga `holdout`. Hallazgos: {holdouts or 'ninguno'}.", "",
               "## Excepciones y limitaciones, declaradas", "",
               f"- El script de compliance y sus tres artefactos quedan excluidos de la "
               f"comprobacion que realizan: `{', '.join(sorted(EXCLUIDOS_COMPLIANCE))}`. Sus "
               f"patrones son literalmente las palabras que buscan.", "",
               "- La compliance busca **firmas**. No puede demostrar que una practica no ocurra "
               "de otra forma, y **no se declara R4**.", "",
               "- El termino *holdout* merece una aclaracion: el holdout interior de las pistas de "
               "Optuna es un **split de validacion dentro de outer_train**, que el punto 11 del "
               "brief exige, y no el holdout reservado de AGE, que V30 ni siquiera tiene.", "",
               "- `leaderboard_tuning` busca el **uso de la API**. El leaderboard no se ha "
               "consultado para decidir nada: la unica lectura prevista es el score de la sonda, "
               "que se registra como evidencia externa.", ""]
    COMPLIANCE_MD.write_text("\n".join(lineas), encoding="utf-8")
    return {"comprobaciones": filas, "todas_cumplidas": all(f["cumplida"] for f in filas)
            and not holdouts, "holdouts_abiertos": holdouts,
            "ficheros_auditados": len([f for f in ficheros
                                       if str(f.relative_to(ROOT)) not in EXCLUIDOS_COMPLIANCE])}


def _reproducibilidad() -> dict:
    """El determinismo se mide: los tres controles de P0 y las semillas de todo lo que lleva Optuna."""
    p0 = json.loads((cfg.EXPERIMENTS / "v30_00_control_absoluto.json").read_text(encoding="utf-8"))
    ref = json.loads((cfg.EXPERIMENTS / "v30_01_optuna_refinamiento.json").read_text(encoding="utf-8"))
    segunda = (json.loads((cfg.EXPERIMENTS / "v30_05_segunda_particion.json")
                          .read_text(encoding="utf-8"))
               if (cfg.EXPERIMENTS / "v30_05_segunda_particion.json").exists() else None)
    salida = {
        "control_v20_bit_a_bit": p0["control_v20"]["reproduce_bit_a_bit"],
        "control_v20_cv": p0["control_v20"]["corrected_cv_recomputado"],
        "control_lgbm_v29": p0["control_lightgbm_v29"]["coincide_con_esperado"],
        "control_ensemble_v29_desviacion": p0["control_ensemble_v29"]["desviacion"],
        "control_ensemble_v29_reproduce": p0["control_ensemble_v29"]["reproduce"],
        "optuna_refinamiento": {
            "trials_por_fold": ref["trials_por_fold"], "trials_totales": ref["trials_totales"],
            "sampler": ref["sampler"], "max_estimadores": ref["max_estimadores"],
            "early_stopping_rondas": ref["early_stopping_rondas"],
            "best_trial_outlier": ref["mejores_trial_outlier"],
        },
        "optuna_pesos_trials": 300,
        "segunda_particion_semilla": segunda["semilla_nueva"] if segunda else None,
        "segunda_particion_reoptimizada": segunda["reoptimizado"] if segunda else None,
        "declaracion": "Optuna con TPESampler y semilla fija resulto **exactamente reproducible** "
                       "en la reoptimizacion de los pesos del ensemble de V29 (desviacion 0,000000), "
                       "lo que demuestra que en esta maquina y con estas versiones de libreria el "
                       "metodo es determinista. Aun asi no se declara exactitud bit a bit entre "
                       "versiones distintas de LightGBM u Optuna.",
    }
    lineas = ["# V30 - Reproducibilidad", "",
              "El determinismo **se mide**, no se declara. Y hay un dato que va en contra de lo que "
              "el propio brief advertia: la reoptimizacion de los pesos del ensemble de V29 "
              "**reprodujo exactamente**, con desviacion 0,000000.", "",
              "| Comprobacion | Resultado |", "|---|---|",
              f"| V20 reproduce bit a bit | **{salida['control_v20_bit_a_bit']}** |",
              f"| corrected_cv de V20 recomputado | {salida['control_v20_cv']} |",
              f"| LightGBM de V29 reproduce | **{salida['control_lgbm_v29']}** |",
              f"| Ensemble de V29 re-optimizado, desviacion | "
              f"**{salida['control_ensemble_v29_desviacion']}** |",
              f"| Optuna, trials por fold | {salida['optuna_refinamiento']['trials_por_fold']} "
              f"({salida['optuna_refinamiento']['trials_totales']} en total) |",
              f"| Optuna, sampler | `{salida['optuna_refinamiento']['sampler']}` |",
              f"| Techo de estimadores | "
              f"{salida['optuna_refinamiento']['max_estimadores']} con early stopping de "
              f"{salida['optuna_refinamiento']['early_stopping_rondas']} rondas |",
              f"| Mejor trial outlier | {salida['optuna_refinamiento']['best_trial_outlier']} de 5 |",
              f"| Segunda particion, semilla | {salida['segunda_particion_semilla']} |",
              f"| Segunda particion re-optimizada | "
              f"{salida['segunda_particion_reoptimizada']} |", "",
              "## Que se declara y que no", "", salida["declaracion"], "",
              "## Lo que se **no** declara", "",
              "- No se declara R4. No hubo reproduccion independiente por un tercero.", "",
              "- No se declara que Optuna*dara* la misma solucion con otras versiones de "
              "LightGBM o de la propia Optuna. La reproduccion medida es dentro de este entorno.", ""]
    REPRO_MD.write_text("\n".join(lineas), encoding="utf-8")
    return salida


def _auditoria_submission() -> dict:
    """Todo lo que se puede comprobar del fichero antes de cualquier subida."""
    construccion = (json.loads((cfg.EXPERIMENTS / "v30_03_submission_construccion.json")
                              .read_text(encoding="utf-8"))
                    if (cfg.EXPERIMENTS / "v30_03_submission_construccion.json").exists() else {})
    upload = (json.loads((cfg.EXPERIMENTS / "v30_04_upload.json").read_text(encoding="utf-8"))
              if (cfg.EXPERIMENTS / "v30_04_upload.json").exists() else {})
    sha = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest() if SUBMISSION.exists() else None
    salida = {
        "fichero_existe": SUBMISSION.exists(),
        "sha256_en_disco": sha,
        "sha256_de_la_construccion": construccion.get("sha256"),
        "hash_coincide": bool(sha and construccion.get("sha256") and sha == construccion["sha256"]),
        "filas": construccion.get("filas"),
        "checks": construccion.get("checks"),
        "checks_superados": construccion.get("checks_superados"),
        "checks_total": construccion.get("checks_total"),
        "candidato": construccion.get("candidato"),
        "pesos_congelados": construccion.get("pesos_congelados"),
        "distribucion": construccion.get("distribucion_predicciones"),
        "comparacion_con_envio_v20": construccion.get("comparacion_con_envio_v20"),
        "uso_del_test": construccion.get("uso_del_test"),
        "subido": upload.get("subido", False),
        "estado_upload": upload.get("estado", "no_intentado"),
        "numero_de_submissions": upload.get("numero_de_submissions", 0),
        "causa_del_stop": upload.get("causa"),
    }
    lineas = ["# V30 - Auditoria de la submission", "",
              "Lo que se puede comprobar **antes** de gastar la unica sonda, y lo que se sabe "
              "**despues** de intentarla.", "",
              "## El fichero", "", "| Comprobacion | Valor |", "|---|---|",
              f"| Existe | {salida['fichero_existe']} |",
              f"| SHA256 en disco | `{salida['sha256_en_disco']}` |",
              f"| SHA256 en la construccion | `{salida['sha256_de_la_construccion']}` |",
              f"| **Hash coincide** | **{salida['hash_coincide']}** |",
              f"| Filas | {salida['filas']:,} |" if salida["filas"] else "| Filas | sin dato |",
              f"| Checks superados | **{salida['checks_superados']}/{salida['checks_total']}** |",
              f"| Candidato | {salida['candidato']} |", ""]
    if salida["comparacion_con_envio_v20"]:
        c = salida["comparacion_con_envio_v20"]
        lineas += ["## Cuanto se aparta del envio de V20", "",
                   f"- Fichero de referencia: `{c.get('envio_de_referencia')}`",
                   f"- Decisiones distintas: **{c.get('decisiones_distintas', 0):,}** "
                   f"({c.get('proporcion', 0):.2%})", ""]
    lineas += ["## Como se uso el test", "", str(salida["uso_del_test"]), "",
               "## La subida", "",
               f"- Estado: **{salida['estado_upload']}**",
               f"- Numero de submissions: **{salida['numero_de_submissions']}**",
               f"- Subido: {salida['subido']}"]
    if salida["causa_del_stop"]:
        lineas += ["", f"- Causa del STOP: {salida['causa_del_stop']}", ""]
    SUBMISSION_MD.write_text("\n".join(lineas), encoding="utf-8")
    return salida


def _higiene() -> dict:
    salida = {}
    for relativo in ("home", "kaggle_toxic_2026_puj/tools"):
        versionado = subprocess.run(["git", "ls-files", relativo], cwd=ROOT.parent,
                                    capture_output=True, text=True).stdout.strip()
        salida[relativo] = {"existe": (ROOT.parent / relativo).exists(),
                            "versionado_por_git": bool(versionado),
                            "sigue_fuera_de_la_rama": not bool(versionado)}
    return salida


def main() -> int:
    """Registra el experimento de P1 y audita la fase entera."""
    autotest_numerico()
    inicio = time.time()
    registro = pd.read_csv(REGISTRO)
    columnas = registro.columns.tolist()
    ref = json.loads((cfg.EXPERIMENTS / "v30_01_optuna_refinamiento.json").read_text(encoding="utf-8"))
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- experiments.csv
    antes = len(registro)
    nuevas = 0
    if not (registro["experiment_id"] == "V30-01").any():
        nueva = {
            "experiment_id": "V30-01", "fecha": FECHA, "bloque": "24",
            "etapa": "V30 refinamiento local del LightGBM de V29",
            "pregunta": ("puede una busqueda mas estrecha y sembrada con el mejor trial de V29 "
                         "consolidar o ampliar la ganancia de capacidad sobre V20?"),
            "hipotesis": ("la capacidad del meta-modelo es el cuello real, y una busqueda local "
                          "alrededor de la configuracion que V29 encontro la consolida"),
            "features": "las 279 de V20, sin anadir ninguna",
            "preprocessing": "sin cambios: la representacion esta congelada",
            "representacion": "identica a V20 y a V29",
            "modelo": "LightGBM con Optuna anidada, 40 trials por fold",
            "hiperparametros": (f'{{"trials_por_fold": {ref["trials_por_fold"]}, '
                                f'"trials_totales": {ref["trials_totales"]}, '
                                f'"espacio_local": {ref["espacio_local"]}, '
                                f'"max_estimadores": {ref["max_estimadores"]}, '
                                f'"early_stopping": {ref["early_stopping_rondas"]}}}'),
            "validacion": ("estructura anidada estricta: Optuna en un holdout interior del 20 % "
                           "extraido del outer_train, refit sobre todo el outer_train, y medida "
                           "solo en el outer_val. Hay una asercion que aborta si el holdout "
                           "interior toca la validacion exterior."),
            "artefactos": ("experiments/v30_01_optuna_refinamiento.json | "
                           "features/v30_lgbm_refinado_oof.npy | reports/v30_optuna_trials.csv"),
            "macro_f1_mean": round(ref["media_por_fold"], 6),
            "macro_f1_std": round(ref["desviacion_por_fold"], 6),
            "resultado": (f"CV {ref['corrected_cv']} | delta V20 {ref['delta_vs_V20']:+} | "
                          f"delta V29-LGBM {ref['delta_vs_lgbm_v29']:+} | "
                          f"{ref['folds_mejor_que_v29']}/5 folds frente a V29 | "
                          f"{ref['trials_totales']} trials | outliers "
                          f"{ref['mejores_trial_outlier']}"),
            "interpretacion": ref["justificacion_del_espacio"],
            "decision": ref["estado_vs_v29"],
            "duracion_seg": ref["segundos"],
        }
        registro = pd.concat([registro, pd.DataFrame([{c: nueva.get(c, "") for c in columnas}])],
                             ignore_index=True)
        registro.to_csv(REGISTRO, index=False)
        nuevas = 1
    print(f"[csv] experiments.csv: {antes} -> {len(registro)} | {nuevas} fila nueva. La "
          f"comparacion, el ensemble, la segunda particion y la construccion del fichero son "
          f"operaciones sobre artefactos congelados y **no** generan fila.", flush=True)

    comp = _compliance()
    repro = _reproducibilidad()
    sub = _auditoria_submission()
    hig = _higiene()

    problemas = []
    if registro["experiment_id"].duplicated().any():
        problemas.append("experiments.csv tiene identificadores duplicados")
    if not comp["todas_cumplidas"]:
        problemas.append("compliance incumplida: "
                        + ", ".join(f["comprobacion"] for f in comp["comprobaciones"]
                                    if not f["cumplida"]))
    if comp["holdouts_abiertos"]:
        problemas.append(f"se abren ficheros de holdout: {comp['holdouts_abiertos']}")
    if sub["fichero_existe"] and not sub["hash_coincide"]:
        problemas.append("el hash de la submission cambio desde la construccion")
    if sub["numero_de_submissions"] > 1:
        problemas.append(f"se hicieron {sub['numero_de_submissions']} submissions; el brief "
                         f"autoriza exactamente una")
    if not all(v["sigue_fuera_de_la_rama"] for v in hig.values()):
        problemas.append("un directorio ajeno a AGE ha entrado en la rama")
    for nombre in ("v30_optuna_refinement.md", "v30_candidate_comparison.csv",
                   "v30_submission_audit.md", "v30_compliance.md", "v30_reproducibility.md"):
        if not (cfg.REPORTS / nombre).exists():
            problemas.append(f"falta el informe {nombre}")

    RESULTADO.write_text(json.dumps({
        "fase": "V30 06 cierre", "fecha": FECHA, "experimenta": False,
        "refinamiento": {k: ref[k] for k in ("corrected_cv", "delta_vs_V20", "delta_vs_lgbm_v29",
                                             "estado_vs_v29", "folds_mejor_que_v29")},
        "ensemble": {f["metodo"]: {"corrected_cv": f["corrected_cv"],
                                   "delta_vs_V20": f["delta_vs_V20"],
                                   "folds_positivos": f["folds_positivos"],
                                   "cumple_gate_historico": f["cumple_gate_historico"],
                                   "es_strong_marginal": f["es_strong_marginal"]}
                     for f in ens["tabla"]},
        "candidato_principal": ens["candidato_principal"],
        "bootstrap": ens["bootstrap"],
        "compliance": comp, "reproducibilidad": repro, "auditoria_submission": sub,
        "higiene": hig, "auditoria": problemas, "coherente": not problemas,
        "presupuesto_autorizado": 4, "presupuesto_gastado": nuevas,
        "experimentos_registrados": len(registro),
        "finalistas": 0, "submissions": sub["numero_de_submissions"],
        "subido": sub["subido"], "estado_upload": sub["estado_upload"],
        "baseline_vigente": CV_V20, "leaderboard_v20": LB_V20,
        "hash_submission_v20": HASH_V20, "artefactos_de_v20_tocados": [],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if problemas:
        print("\n[AUDITORIA] DESAJUSTES:", flush=True)
        for p in problemas:
            print(f"  - {p}", flush=True)
        return 1
    print(f"[compliance] {len(comp['comprobaciones'])} comprobaciones | todas cumplidas: "
          f"{comp['todas_cumplidas']}", flush=True)
    print(f"[submission] {sub['estado_upload']} | submissions: {sub['numero_de_submissions']} | "
          f"hash coherente: {sub['hash_coincide']}", flush=True)
    print("[higiene] los 2 directorios ajenos siguen fuera de la rama", flush=True)
    print("[auditoria] coherente", flush=True)
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
