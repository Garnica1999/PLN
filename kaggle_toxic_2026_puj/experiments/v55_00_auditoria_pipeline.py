"""V55-00 (P55): auditoria estrategica del pipeline final de V03.

Que audita
----------
La pregunta es si existe un defecto o perdida de informacion **en el procedimiento**
de V03 que pueda explicar parte de la brecha OOF a leaderboard, y que no se haya
evaluado correctamente. No es una busqueda de modelos.

El hallazgo que dispara la fase
------------------------------
Leyendo el codigo real de ``experiments/04_refit_submissions_v02_v03.py``:

    los umbrales 0,44 / 0,51 / 0,46 se optimizaron sobre
    ``features/oof_mod03b.npz``, generado con grupos = near-duplicado (D-004)

y despues

    el ``corrected_cv = 0,7772114525`` se reporta sobre
    ``features/v2_oof_base_corrected.npz``, generado con grupos = near-duplicate
    UNION pseudo-autor (D-016)

Es decir **los dos numeros de la referencia de V03 vienen de protocolos de
agrupacion distintos**, y D-004 es el mas debil porque deja que un pseudo-autor
cruce pliegues. El cambio de protocolo ocurrio en D-016 y el ``corrected_cv`` se
recalculo con los umbrales **viejos**, sin reoptimizarlos bajo el protocolo nuevo.

Ademas ``features/oof_mod03b.npz`` **ya no existe en disco**, luego el artefacto
que produjo los umbrales adoptados no es auditable.

Como se mide, siniltrar
-----------------------
Los umbrales optimos sobre el mismo OOF que se puntua son **in-sample para el
umbral**, luego no son una mejora esperable. Por eso aqui se anida: los umbrales
se optimizan en cuatro pliegues y se aplican al quinto, con los pliegues
canonicos de TOXIC. Eso da una estimacion honesta y ademas la distribucion de
los umbrales, que es la pregunta de estabilidad.

Que NO hace
-----------
- No entrena el modelo final sobre todo el train.
- No genera ninguna submission.
- No consulta Kaggle.
- No usa holdout ni test.
- No modifica V03.

Uso
---
    /home/carlos/venv/bin/python experiments/v55_00_auditoria_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_EXTERNOS = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj/features")
OOF_D016 = FEATURES_EXTERNOS / "v2_oof_base_corrected.npz"
OOF_D004 = PROJECT_ROOT / "features" / "oof_mod03b.npz"
SUBMISSION_V03 = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
SHA_V03_ESPERADO = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
CORRECTED_CV_CANONICO = 0.7772114525165721
LB_V03 = 0.78306

ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5
REJILLA = np.round(np.arange(0.20, 0.801, 0.01), 2)
V03_UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

REPORTE_JSON = PROJECT_ROOT / "reports" / "p55_pipeline_audit.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p55_pipeline_audit.md"
CSV_AUDIT = PROJECT_ROOT / "reports" / "p55_pipeline_audit.csv"
CSV_TRANSFER = PROJECT_ROOT / "reports" / "p55_threshold_transfer.csv"
CSV_INTEGRIDAD = PROJECT_ROOT / "reports" / "p55_artifact_integrity.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V55-00] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    """Calcula el sha256 de un fichero.

    Args:
        ruta: fichero de entrada.

    Returns:
        El hash hexadecimal, o cadena vacia si no existe.
    """
    if not ruta.exists():
        return ""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def umbral_optimo(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Umbral que maximiza la F1 binaria de una etiqueta.

    Args:
        y: etiqueta verdadera.
        p: probabilidad de la clase positiva.

    Returns:
        Tupla ``(umbral, f1)``.
    """
    mejor, mejor_f1 = float(REJILLA[0]), -1.0
    for u in REJILLA:
        f = float(f1_score(y, (p >= u).astype(int), zero_division=0))
        if f > mejor_f1:
            mejor, mejor_f1 = float(u), f
    return mejor, mejor_f1


def macro_f1(y: np.ndarray, p: np.ndarray, umbrales: dict[str, float]) -> float:
    """Macro-F1 multietiqueta con un umbral por etiqueta.

    Args:
        y: matriz de etiquetas verdaderas.
        p: matriz de probabilidades.
        umbrales: umbral por etiqueta.

    Returns:
        La media de los tres F1 por etiqueta.
    """
    return float(np.mean([
        f1_score(y[:, j], (p[:, j] >= umbrales[ETIQUETAS[j]]).astype(int), zero_division=0)
        for j in range(3)
    ]))


def main() -> None:
    """Ejecuta la auditoria del pipeline de V03.

    Returns:
        None.
    """
    inicio = time.time()
    log("P55-00 auditoria estrategica del pipeline de V03. 0 modelos, 0 submissions")

    d = np.load(OOF_D016, allow_pickle=True)
    ids = d["ids"].astype(int)
    y = np.asarray(d["y_true"], dtype=np.int64)
    modelos = [str(x) for x in d["modelos"]]
    idx_m2 = modelos.index("M2_union_lr_c2")
    proba = d["proba"][:, idx_m2, :].astype(np.float64)
    grupos = d["grupos"].astype(int)
    # Los pliegues corregidos se recuperan del protocolo canonico, no del npz: el
    # npz guarda los grupos pero no la asignacion de pliegue.
    from src.data import read_train
    from src.splits import grupos_fusionados, pliegues_corregidos
    tr = read_train()
    tr["id"] = tr["id"].astype("int64")
    part = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    dev_ids = set(part.loc[part["particion"] == "dev", "id"].astype("int64"))
    dev = tr[tr["id"].isin(dev_ids)].reset_index(drop=True)
    if not np.array_equal(dev["id"].to_numpy(), ids):
        raise SystemExit("PARADA: el orden del dev no coincide con el OOF canonico")
    grupos_canon = grupos_fusionados(dev)
    asignacion = np.full(len(ids), -1, dtype=np.int64)
    for k, (_, val) in enumerate(pliegues_corregidos(dev, grupos_canon)):
        asignacion[val] = k
    log(f"   OOF D-016 cargado | pliegues canonicos {[int((asignacion == v).sum()) for v in range(N_FOLD)]}")
    log(f"   grupos near-dup del npz: {len(np.unique(grupos))} | grupos fusionados D-016: "
        f"{len(np.unique(grupos_canon))}")

    # ================================================ 1. INTEGRIDAD
    log("1/6 integridad de artefactos")
    filas_int = []
    esperado_v03 = sha256(SUBMISSION_V03)
    filas_int.append({
        "artefacto": "submissions/toxic_submission_v03_lr_union_thr.csv",
        "existe": SUBMISSION_V03.exists(), "sha256": esperado_v03,
        "coincide_con_el_esperado": bool(esperado_v03 == SHA_V03_ESPERADO),
        "nota": "campeon congelado",
    })
    filas_int.append({
        "artefacto": "features/oof_mod03b.npz (OOF D-004 que produjo los umbrales)",
        "existe": OOF_D004.exists(), "sha256": sha256(OOF_D004),
        "coincide_con_el_esperado": None,
        "nota": "ARTEFACTO AUSENTE: los umbrales adoptados no son auditables",
    })
    for nombre in ("features/v2_oof_base_corrected.npz",
                   "features/oof_v03_final.npz",
                   "features/v2_structured_features.parquet",
                   "data/raw/toxic_train.csv",
                   "data/processed/holdout_v2_ids.csv",
                   "data/processed/grupos_near_duplicados.csv",
                   "experiments/04_refit_submissions_v02_v03.py",
                   "experiments/14_oof_corregido_v2.py"):
        filas_int.append({
            "artefacto": nombre,
            "existe": (PROJECT_ROOT / nombre).exists() or (FEATURES_EXTERNOS / Path(nombre).name).exists(),
            "sha256": sha256(PROJECT_ROOT / nombre) or sha256(FEATURES_EXTERNOS / Path(nombre).name),
            "coincide_con_el_esperado": None, "nota": "",
        })
    idf = pd.DataFrame(filas_int)
    idf.to_csv(CSV_INTEGRIDAD, index=False, encoding="utf-8")
    for _, x in idf.iterrows():
        marca = "" if x["coincide_con_el_esperado"] is not False else "  <- NO COINCIDE"
        log(f"   {'OK ' if x['existe'] else 'AUSENTE'} {x['artefacto']:62}{marca}")

    # ================================================ 2. UMBRALES
    log("2/6 auditoria de umbrales: in-sample, anidados y estabilidad")
    f1_v03 = macro_f1(y, proba, V03_UMBRALES)
    log(f"   macro-F1 con umbrales V03 sobre D-016: {f1_v03:.16f}")
    log(f"   corrected_cv canonico                  : {CORRECTED_CV_CANONICO:.16f}")
    coincide = abs(f1_v03 - CORRECTED_CV_CANONICO) < 1e-12
    log(f"   reproduce el corrected_cv canonico: {coincide}")

    opt_in = {}
    for j, et in enumerate(ETIQUETAS):
        u, _ = umbral_optimo(y[:, j], proba[:, j])
        opt_in[et] = u
    f1_in = macro_f1(y, proba, opt_in)
    log("   -- umbrales optimos IN-SAMPLE sobre el mismo OOF que se puntua --")
    for et in ETIQUETAS:
        log(f"      {et}: D-016 {opt_in[et]:.2f} | V03 {V03_UMBRALES[et]:.2f} | "
            f"diferencia {V03_UMBRALES[et] - opt_in[et]:+.2f}")
    log(f"      macro-F1 in-sample: {f1_in:.10f} | delta {f1_in - f1_v03:+.6f}")
    log("      OJO: es in-sample para el umbral, luego NO es una mejora esperable")

    # Anidado: umbrales optimizados en 4 pliegues, aplicados al quinto.
    log("   -- umbrales ANIDADOS, la estimacion honesta --")
    pred_anidado = np.zeros_like(y)
    filas_transfer = []
    for v in range(N_FOLD):
        ent = asignacion != v
        val = asignacion == v
        umb = {}
        for j, et in enumerate(ETIQUETAS):
            u, _ = umbral_optimo(y[ent, j], proba[ent, j])
            umb[et] = u
        pred_anidado[val] = (proba[val] >= np.array([umb[et] for et in ETIQUETAS])[None, :]).astype(int)
        filas_transfer.append({
            "fold": v, "n_validacion": int(val.sum()),
            **{f"umbral_{et}": umb[et] for et in ETIQUETAS},
            **{f"umbral_V03_{et}": V03_UMBRALES[et] for et in ETIQUETAS},
        })
        log(f"      pliegue {v}: umbrales anidados "
            f"{umb['HS']:.2f}/{umb['TR']:.2f}/{umb['AG']:.2f} sobre {int(val.sum())} celdas")
    f1_anidado = float(np.mean([
        f1_score(y[:, j], pred_anidado[:, j], zero_division=0) for j in range(3)
    ]))
    log(f"      macro-F1 ANIDADO: {f1_anidado:.10f} | delta vs V03 {f1_anidado - f1_v03:+.6f}")

    for et in ETIQUETAS:
        vals = [f[f"umbral_{et}"] for f in filas_transfer]
        log(f"      {et}: umbrales anidados {vals} | mediana {np.median(vals):.2f} "
            f"| V03 {V03_UMBRALES[et]:.2f} | V03 esta "
            f"{'DENTRO' if min(vals) <= V03_UMBRALES[et] <= max(vals) else 'FUERA'} del rango")

    # Tasa positiva, que es la consecuencia operativa.
    log("   -- tasa positiva sobre el OOF --")
    tasa_v03 = [(proba[:, j] >= V03_UMBRALES[ETIQUETAS[j]]).mean() for j in range(3)]
    tasa_in = [(proba[:, j] >= opt_in[ETIQUETAS[j]]).mean() for j in range(3)]
    tasa_anid = [pred_anidado[:, j].mean() for j in range(3)]
    log(f"      con umbrales V03    : {[f'{v:.4f}' for v in tasa_v03]}")
    log(f"      con optimos in-sample: {[f'{v:.4f}' for v in tasa_in]}")
    log(f"      con anidados         : {[f'{v:.4f}' for v in tasa_anid]}")

    # ================================================ 3. TAMANO DE ENTRENAMIENTO
    log("3/6 trayectoria con el tamano de entrenamiento, sin seleccionar nada")
    filas_tam = []
    for frac in (0.80, 0.90, 1.00):
        pred = np.zeros_like(y)
        for v in range(N_FOLD):
            ent = asignacion != v
            val = asignacion == v
            # Subconjunto del training fold, agrupado, para imitar un train menor.
            rng = np.random.default_rng(1000 + v)
            grupos_unicos = np.unique(grupos_canon[ent])
            n_grupos = max(round(len(grupos_unicos) * frac), 1)
            elegido = set(rng.choice(grupos_unicos, size=n_grupos, replace=False).tolist())
            mask = ent & np.isin(grupos_canon, list(elegido))
            umb = {}
            for j, et in enumerate(ETIQUETAS):
                u, _ = umbral_optimo(y[mask, j], proba[mask, j])
                umb[et] = u
            pred[val] = (proba[val] >= np.array([umb[et] for et in ETIQUETAS])[None, :]).astype(int)
        f1 = float(np.mean([f1_score(y[:, j], pred[:, j], zero_division=0) for j in range(3)]))
        filas_tam.append({
            "fraccion_de_grupos": frac, "macro_f1": round(f1, 10),
            "umbral_HS": umb["HS"], "umbral_TR": umb["TR"], "umbral_AG": umb["AG"],
        })
        log(f"      {frac:.0%} de grupos del training fold: macro-F1 {f1:.10f} | "
            f"umbrales {umb['HS']:.2f}/{umb['TR']:.2f}/{umb['AG']:.2f}")
    base = filas_tam[-1]["macro_f1"]
    for f in filas_tam:
        f["delta_vs_100"] = round(f["macro_f1"] - base, 10)
    log("      la trayectoria describe el efecto del tamano, NO selecciona un modelo")

    # ================================================ 4. CONSISTENCIA
    log("4/6 consistencia de vectorizador, grupos y balance de pliegues")
    codigo_04 = (PROJECT_ROOT / "experiments" / "04_refit_submissions_v02_v03.py").read_text(
        encoding="utf-8")
    codigo_14 = (PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py").read_text(
        encoding="utf-8")
    # Se verifica el CODIGO y no el docstring: el vectorizador tiene que
    # ajustarse sobre los textos de entrenamiento y transformar los de validacion.
    vectorizador_dentro = ("fit_transform(textos_entrenamiento)" in codigo_14
                          and "construir_matrices(" in codigo_14)
    grupos_d004 = "grupos_near_duplicados.csv" in codigo_04
    grupos_d016 = "grupos_fusionados" in codigo_14
    log(f"   vectorizador ajustado DENTRO de cada pliegue: {vectorizador_dentro} "
        f"-> sin leakage de representacion")
    log(f"   el script de umbrales usa grupos D-004 (near-dup): {grupos_d004}")
    log(f"   el OOF corregido usa grupos D-016 (near-dup UNION pseudo-autor): {grupos_d016}")
    log(f"   numero de grupos: D-004 {len(np.unique(grupos))} vs D-016 "
        f"{len(np.unique(grupos_canon))}")

    filas_balance = []
    for v in range(N_FOLD):
        sel = asignacion == v
        filas_balance.append({
            "pliegue": v, "n_filas": int(sel.sum()),
            "n_grupos": len(np.unique(grupos_canon[sel])),
            "tasa_HS": round(float(y[sel, 0].mean()), 6),
            "tasa_TR": round(float(y[sel, 1].mean()), 6),
            "tasa_AG": round(float(y[sel, 2].mean()), 6),
            "macro_f1_de_V03": round(float(np.mean([
                f1_score(y[sel, j], (proba[sel, j] >= V03_UMBRALES[ETIQUETAS[j]]).astype(int),
                         zero_division=0) for j in range(3)])), 10),
        })
        log(f"   pliegue {v}: n {int(sel.sum()):4d} | HS {y[sel,0].mean():.4f} "
            f"TR {y[sel,1].mean():.4f} AG {y[sel,2].mean():.4f} | "
            f"macro-F1 {filas_balance[-1]['macro_f1_de_V03']:.6f}")
    bs = pd.DataFrame(filas_balance)
    rango = bs["macro_f1_de_V03"].max() - bs["macro_f1_de_V03"].min()
    log(f"   dispersion del macro-F1 entre pliegues: {rango:.6f}")
    log(f"   la media de pliegues es {bs['macro_f1_de_V03'].mean():.10f} frente al global "
        f"{CORRECTED_CV_CANONICO:.10f}")

    # ================================================ 5. BRECHA
    log("5/6 la brecha OOF a leaderboard")
    brecha = LB_V03 - CORRECTED_CV_CANONICO
    log(f"   OOF {CORRECTED_CV_CANONICO:.10f} | LB {LB_V03:.5f} | delta {brecha:+.8f}")
    log(f"   sigma de pliegues del corrected_cv: "
        f"{bs['macro_f1_de_V03'].std(ddof=1):.6f} | la brecha es "
        f"{brecha / max(bs['macro_f1_de_V03'].std(ddof=1), 1e-9):.2f} sigma")
    log("   lo que se SABE: el signo es positivo, luego el LB fue MAS alto que el OOF")
    log("   lo que se DESCONOCE: por que. Ninguna consulta a Kaggle es legal aqui")
    log("   lo que es PLAUSIBLE pero no probado: variacion de muestreo, y que el")
    log("   OFF subestima por umbrales heredados de un protocolo mas debil")

    # ================================================ 6. VEREDICTO
    delta_anidado = f1_anidado - f1_v03
    hay_discrepancia = abs(delta_anidado) > 0.0005 or any(
        not (min([f[f"umbral_{et}"] for f in filas_transfer]) <= V03_UMBRALES[et]
             <= max([f[f"umbral_{et}"] for f in filas_transfer])) for et in ETIQUETAS)
    # El veredicto distingue la EXISTENCIA de la discrepancia de su IMPACTO. Una
    # discrepancia real que ademas queda por debajo del margen de adopcion sigue
    # siendo una discrepancia que hay que registrar, no un motivo para enviar una
    # submission nueva.
    impacto_sobre_el_gate = delta_anidado >= 0.003
    if not hay_discrepancia:
        veredicto = "A"
    else:
        veredicto = "B"
    log(f"   hay discrepancia procedural: {hay_discrepancia}")
    log(f"   su impacto honesto alcanza el margen de 0,003: {impacto_sobre_el_gate}")
    if veredicto == "B" and not impacto_sobre_el_gate:
        log("   => se registra como BUG_FOUND, pero NO justifica una submission nueva")
        log("      porque la estimacion anidada queda por debajo del margen")

    filas_audit = [
        {"comprobacion": "corrected_cv reproducible desde el OOF D-016",
         "resultado": str(coincide), "detalle": f"{f1_v03:.16f}"},
        {"comprobacion": "submission V03 con el sha256 esperado",
         "resultado": str(esperado_v03 == SHA_V03_ESPERADO), "detalle": esperado_v03},
        {"comprobacion": "vectorizador ajustado dentro de cada pliegue",
         "resultado": str(vectorizador_dentro), "detalle": "sin leakage de representacion"},
        {"comprobacion": "umbrales del script de V03 usan grupos D-004",
         "resultado": str(grupos_d004), "detalle": "near-duplicado solamente"},
        {"comprobacion": "corrected_cv usa grupos D-016",
         "resultado": str(grupos_d016), "detalle": "near-dup UNION pseudo-autor"},
        {"comprobacion": "el OOF D-004 que produjo los umbrales sigue en disco",
         "resultado": str(OOF_D004.exists()), "detalle": "auditoria del umbral impedida"},
        {"comprobacion": "umbrales anidados dentro del rango de los de V03",
         "resultado": str(any(not (min([f[f'umbral_{et}'] for f in filas_transfer])
                                    <= V03_UMBRALES[et]
                                    <= max([f[f'umbral_{et}'] for f in filas_transfer]))
                                for et in ETIQUETAS)),
         "detalle": "; ".join(f"{et}: anidados "
                              f"{[f[f'umbral_{et}'] for f in filas_transfer]} vs V03 "
                              f"{V03_UMBRALES[et]}" for et in ETIQUETAS)},
        {"comprobacion": "delta de los umbrales anidados frente a V03",
         "resultado": f"{delta_anidado:+.6f}", "detalle": "estimacion honesta"},
        {"comprobacion": "tasa positiva con umbrales V03",
         "resultado": str([round(v, 4) for v in tasa_v03]), "detalle": "sobre el OOF"},
        {"comprobacion": "brecha OOF a LB", "resultado": f"{brecha:+.8f}",
         "detalle": f"{brecha / max(bs['macro_f1_de_V03'].std(ddof=1), 1e-9):.2f} sigma"},
    ]
    pd.DataFrame(filas_audit).to_csv(CSV_AUDIT, index=False, encoding="utf-8")
    pd.DataFrame(filas_transfer).to_csv(CSV_TRANSFER, index=False, encoding="utf-8")

    informe = {
        "familia": "P55_FINAL_PIPELINE_AUDIT",
        "experimento": "V55-00 (auditoria estrategica del pipeline final de V03)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pipeline_real": {
            "umbrales_adoptados_de": ("experiments/04_refit_submissions_v02_v03.py sobre "
                                      "features/oof_mod03b.npz con grupos D-004"),
            "corrected_cv_reportado_de": ("experiments/14_oof_corregido_v2.py sobre "
                                          "v2_oof_base_corrected.npz con grupos D-016"),
            "vectorizador_dentro_del_pliegue": bool(vectorizador_dentro),
            "modelo_final": ("refit sobre el 100 por ciento del train con los mismos "
                             "hiperparametros, y umbrales los mismos del OOF D-004"),
            "discrepancia": ("los umbrales y la metrica reportada provienen de protocolos "
                             "de agrupacion distintos, y D-004 es el mas debil"),
        },
        "umbrales": {
            "V03": V03_UMBRALES,
            "optimos_in_sample_D016": opt_in,
            "macro_f1_V03": round(f1_v03, 10),
            "macro_f1_in_sample": round(f1_in, 10),
            "macro_f1_anidado": round(f1_anidado, 10),
            "delta_in_sample": round(f1_in - f1_v03, 10),
            "delta_anidado": round(delta_anidado, 10),
            "anidados_por_pliegue": filas_transfer,
            "advertencia": ("el delta in-sample NO es una mejora esperable, porque el "
                            "umbral se eligio sobre el mismo OOF que se puntua; el "
                            "delta anidado si es una estimacion honesta"),
        },
        "tasa_positiva": {
            "con_umbrales_V03": [round(v, 6) for v in tasa_v03],
            "con_optimos_in_sample": [round(v, 6) for v in tasa_in],
            "con_anidados": [round(v, 6) for v in tasa_anid],
        },
        "tamano_de_entrenamiento": filas_tam,
        "balance_de_pliegues": filas_balance,
        "integridad_de_artefactos": filas_int,
        "brecha_oof_lb": {
            "oof": CORRECTED_CV_CANONICO, "lb": LB_V03, "delta": round(brecha, 8),
            "sigma_de_pliegues": round(float(bs["macro_f1_de_V03"].std(ddof=1)), 6),
            "delta_en_sigma": round(brecha / max(float(bs["macro_f1_de_V03"].std(ddof=1)),
                                                1e-9), 3),
            "sabido": "el LB fue mas alto que el OOF",
            "desconocido": "la causa, y ninguna consulta a Kaggle es legal en esta fase",
            "plausible_no_probado": ["variacion de muestreo entre dev y test",
                                     ("el OOF subestima por umbrales heredados de un "
                                     "protocolo mas debil")],
        },
        "P55_PROCEDURAL_GAP": veredicto,
        "hay_discrepancia": bool(hay_discrepancia),
        "impacto_alcanza_el_gate_de_0p003": bool(impacto_sobre_el_gate),
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe)
    log("   escrito reports/p55_*")
    log(f"VEREDICTO: P55_PROCEDURAL_GAP={veredicto} | delta_anidado={delta_anidado:+.6f} | "
        f"0 modelos, 0 submissions, {round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe de la auditoria del pipeline.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    pr = informe["pipeline_real"]
    u = informe["umbrales"]
    br = informe["brecha_oof_lb"]
    tp = informe["tasa_positiva"]
    lineas = [
        "# P55 - Auditoria estrategica del pipeline final de V03",
        "",
        (f"Generado por `experiments/v55_00_auditoria_pipeline.py` "
        f"({informe['experimento']})."),
        "",
        (f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle, "
        "0 uso de holdout o test. V03 no se modifica.**"),
        "",
        "## 1. El pipeline real, leido del codigo y no del docstring",
        "",
        "```text",
        "raw train (1731)",
        "  -> particion congelada dev 1483 / holdout 248",
        "  -> pliegues: DOS protocolos distintos, y ahi esta el problema",
        "",
        "A) 04_refit_submissions_v02_v03.py",
        "      grupos = near-duplicado (D-004), 5 pliegues",
        "      vectorizador y modelo DENTRO de cada pliegue",
        "      -> oof_mod03b.npz",
        "      -> umbrales 0,44 / 0,51 / 0,46 maximizando F1 sobre ESE OOF",
        "      -> refit sobre el 100 por ciento del train con ESOS umbrales",
        "      -> submission V03",
        "",
        "B) 14_oof_corregido_v2.py",
        "      grupos = near-duplicate UNION pseudo-autor (D-016), 5 pliegues",
        "      vectorizador y modelo DENTRO de cada pliegue",
        "      -> v2_oof_base_corrected.npz",
        "      -> corrected_cv = 0,7772114525 con LOS UMBRALES DE A)",
        "```",
        "",
        f"> {pr['discrepancia']}",
        "",
        f"- umbrales adoptados de: {pr['umbrales_adoptados_de']}",
        f"- corrected_cv reportado de: {pr['corrected_cv_reportado_de']}",
        (f"- vectorizador dentro del pliegue: **{pr['vectorizador_dentro_del_pliegue']}**, "
        "luego no hay leakage de representacion"),
        f"- modelo final: {pr['modelo_final']}",
        "",
        "## 2. Transferencia de umbrales: in-sample, anidados y estabilidad",
        "",
        "| metodo | HS | TR | AG | macro-F1 | delta vs V03 |",
        "|---|---:|---:|---:|---:|---:|",
        (f"| umbrales de V03 | {u['V03']['HS']:.2f} | {u['V03']['TR']:.2f} | "
        f"{u['V03']['AG']:.2f} | {u['macro_f1_V03']:.10f} | +0,0000000000 |"),
        (f"| optimos in-sample D-016 | {u['optimos_in_sample_D016']['HS']:.2f} | "
        f"{u['optimos_in_sample_D016']['TR']:.2f} | "
        f"{u['optimos_in_sample_D016']['AG']:.2f} | {u['macro_f1_in_sample']:.10f} | "
        f"{u['delta_in_sample']:+.10f} |"),
        (f"| **anidados (honesto)** | ver CSV | ver CSV | ver CSV | "
        f"**{u['macro_f1_anidado']:.10f}** | **{u['delta_anidado']:+.10f}** |"),
        "",
        f"> {u['advertencia']}",
        "",
        "### Los umbrales anidados por pliegue",
        "",
        "| pliegue | " + " | ".join(f"anidado {et}" for et in ETIQUETAS)
        + " | " + " | ".join(f"V03 {et}" for et in ETIQUETAS) + " |",
        "|---:|" + "---:|" * 6,
    ]
    for f in u["anidados_por_pliegue"]:
        lineas.append(f"| {f['fold']} | " + " | ".join(f"{f[f'umbral_{et}']:.2f}"
                                                       for et in ETIQUETAS) + " | "
                     + " | ".join(f"{f[f'umbral_V03_{et}']:.2f}" for et in ETIQUETAS) + " |")
    lineas += [
        "",
        "## 3. Tasa positiva, que es la consecuencia operativa",
        "",
        "| umbrales | HS | TR | AG |",
        "|---|---:|---:|---:|",
        (f"| de V03 | {tp['con_umbrales_V03'][0]:.4f} | {tp['con_umbrales_V03'][1]:.4f} | "
        f"{tp['con_umbrales_V03'][2]:.4f} |"),
        (f"| optimos in-sample | {tp['con_optimos_in_sample'][0]:.4f} | "
        f"{tp['con_optimos_in_sample'][1]:.4f} | {tp['con_optimos_in_sample'][2]:.4f} |"),
        (f"| anidados | {tp['con_anidados'][0]:.4f} | {tp['con_anidados'][1]:.4f} | "
        f"{tp['con_anidados'][2]:.4f} |"),
        "",
        "## 4. La brecha OOF a leaderboard",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| corrected_cv | {br['oof']:.10f} |",
        f"| LB | {br['lb']:.5f} |",
        f"| delta | **{br['delta']:+.8f}** |",
        f"| sigma de pliegues del OOF | {br['sigma_de_pliegues']:.6f} |",
        f"| delta en sigma | {br['delta_en_sigma']:.2f} |",
        "",
        f"- **lo que se sabe**: {br['sabido']}",
        f"- **lo que se desconoce**: {br['desconocido']}",
        "- **lo que es plausible y NO esta probado**: "
        + "; ".join(br["plausible_no_probado"]),
        "",
        "## 5. Decision",
        "",
        f"**`P55_PROCEDURAL_GAP = {informe['P55_PROCEDURAL_GAP']}`.**",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
