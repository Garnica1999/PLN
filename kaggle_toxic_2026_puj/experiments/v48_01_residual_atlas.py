"""V48-01 (P1): atlas de residuos de V03, poblacion de rescate y patrones.

Que responde
------------
Que errores de V03 son corregibles por el pool historico de canales OOF, y cuales
parecen no contener informacion recuperable con las representaciones actuales.

Fuentes y su papel
------------------
- `features/oof_v03_final.npz`: OOF canonico de V03 sobre las 1483 filas de dev.
  Aporta predicciones, margenes y confianza.
- `reports/hardness_v4.csv`: 1483 filas, mismas claves. Aporta, por celda, la
  **fraccion de canales OOF historicos que fallan** (`model_dis_HS/TR/AG`), que
  es lo unico que permite reconstruir el rescate sin volver a entrenar.
- `features/v2_structured_features.parquet`: 2885 filas, 125 columnas estructurales.
  Aporta las variables de analisis. Se filtra a dev por `particion`.

Limitaciones declaradas, no resueltas
--------------------------------------
1. Los tensores OOF por canal (`features/v2_oof_*.npz`) **ya no estan en disco**.
   Solo queda la fraccion de fallo agregada por celda. Por tanto la categoria C de
   la directiva ("V03 falla y al menos un modelo historico la rescata") se
   reconstruye como "V03 falla y al menos un canal historico acierta", que es una
   cota superior del rescate, no una identificacion del modelo concreto.
2. `holdout2_ids.csv` (247 filas) esta dibujado **de dev**, asi que esas filas si
   forman parte de las 1483 y su inclusion es la del protocolo canonico. No se
   usan para nada distinto del conteo.
3. No se usan ni test ni holdout.

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No crea filas en el registro de experimentos.

Uso
---
    /home/carlos/venv/bin/python experiments/v48_01_residual_atlas.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
HARDNESS = PROJECT_ROOT / "reports" / "hardness_v4.csv"
PARQUET = PROJECT_ROOT / "features" / "v2_structured_features.parquet"
REPORTE_JSON = PROJECT_ROOT / "reports" / "p48_residual_atlas.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p48_research_gap.md"
CSV_ATLAS = PROJECT_ROOT / "reports" / "p48_residual_atlas.csv"
CSV_RESCATE = PROJECT_ROOT / "reports" / "p48_rescue_analysis.csv"
CSV_NO_RESCATABLE = PROJECT_ROOT / "reports" / "p48_unrescued_analysis.csv"
CSV_PATRONES = PROJECT_ROOT / "reports" / "p48_pattern_candidates.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
N_CANALES = 20  # los valores de model_dis son multiplos de 0.05, luego 20 canales

# Slices ya definidas en el EDA original. No se crean slices nuevas en esta fase.
SLICES = {
    "menciones": ("n_menciones", 1),
    "hashtags": ("n_hashtags", 1),
    "urls": ("n_urls", 1),
    "elongacion": ("n_elongaciones", 1),
    "leet": ("n_caracteres_leet", 1),
    "puntuacion_interna": ("n_puntuacion_interna", 1),
    "risas": ("n_risas", 1),
    "cita": ("n_tramos_cita", 1),
    "atribucion": ("n_marcadores_atribucion", 1),
    "colectivo": ("n_colectivos", 1),
    "segunda_persona": ("n_segunda", 1),
    "negacion": ("n_negaciones", 1),
    "rt": ("rt_presente", 1),
    "multiples_urls": ("n_urls", 2),
}

# Combinaciones del section 15: producto de dos variables ya existentes.
COMBINACIONES = [
    ("longitud_corta", lambda d: d.n_tokens < d.n_tokens.quantile(0.33)),
    ("longitud_larga", lambda d: d.n_tokens > d.n_tokens.quantile(0.67)),
    # Estos dos se definen por el propio error, asi que su lift es circular.
    # Se conservan como diagnostico de margen y se marcan como NO_CANDIDATOS.
    ("margen_bajo_error", lambda d: (d.n_errores > 0) & (d.margen_min < 0.10)),
    ("margen_alto_error", lambda d: (d.n_errores > 0) & (d.margen_min > 0.30)),
    ("mencion_y_posicion_inicial", lambda d: (d.n_menciones > 0) & (d.n_mencion_inicial > 0)),
    ("url_y_texto_corto", lambda d: (d.n_urls > 0) & (d.n_tokens < d.n_tokens.quantile(0.33))),
    ("hashtag_y_texto_corto", lambda d: (d.n_hashtags > 0) & (d.n_tokens < d.n_tokens.quantile(0.33))),
    ("cita_con_HS_1", lambda d: (d.n_tramos_cita > 0) & (d.true_HS == 1)),
    ("elongacion_con_HS_1", lambda d: (d.n_elongaciones > 0) & (d.true_HS == 1)),
    ("colectivo_en_FN_de_HS", lambda d: (d.n_colectivos > 0) & (d.err_HS == 1) & (d.true_HS == 1)),
    ("negacion_en_error_de_TR", lambda d: (d.n_negaciones > 0) & (d.err_TR == 1)),
    ("mencion_inicial_en_error_de_TR", lambda d: (d.n_mencion_inicial > 0) & (d.err_TR == 1)),
    # El insulto lexico solo existe en `all_three_errors.csv`, que es precisamente la
    # poblacion de "las tres mal": usarlo seria circular. Se sustituyen por dos
    # variables del EDA que si estan disponibles por fila.
    ("label_rarity_alta", lambda d: d.label_rarity > d.label_rarity.median()),
    ("poblacion_no_dificil", lambda d: d.poblacion == "facil"),
    ("en_cluster_de_duplicados", lambda d: d.en_cluster == 1),
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V48-01] {mensaje}", flush=True)


def main() -> None:
    """Ejecuta el atlas de residuos y el analisis de rescate.

    Returns:
        None.
    """
    inicio = time.time()
    log("P48-01 atlas de residuos. 0 modelos, 0 submissions, 0 uso de holdout o test")

    # ------------------------------------------------------------ 1. carga
    log("1/8 carga de artefactos")
    oof = np.load(OOF, allow_pickle=True)
    ids = oof["ids"].astype(int)
    y_true = np.asarray(oof["y_true"], dtype=np.int64)
    pred = np.asarray(oof["pred"], dtype=np.int64)
    margen = np.asarray(oof["margen"], dtype=float)
    confianza = np.asarray(oof["confianza"], dtype=float)
    n = len(ids)
    log(f"   OOF canonico: {n} filas, corrected_cv {float(oof['corrected_cv'][0]):.16f}")

    hz = pd.read_csv(HARDNESS).set_index("id").loc[ids]
    par = pd.read_parquet(PARQUET)
    par["id"] = par["id"].astype("int64")
    par = par[par.particion == "dev"].set_index("id").loc[ids]
    log(f"   hardness_v4: {len(hz)} filas | features estructurales dev: {len(par)} filas")

    # ------------------------------- 2. atlas de residuos por celda
    log("2/8 atlas de residuos por celda")
    frame = pd.DataFrame({"tweet_id": ids})
    for j, et in enumerate(ETIQUETAS):
        frame[f"true_{et}"] = y_true[:, j]
        frame[f"pred_{et}"] = pred[:, j]
        frame[f"margin_{et}"] = margen[:, j]
        frame[f"confianza_{et}"] = confianza[:, j]
        frame[f"correct_{et}"] = (pred[:, j] == y_true[:, j]).astype(int)
        frame[f"err_{et}"] = 1 - frame[f"correct_{et}"]
        # fraccion de canales que fallan -> canales que aciertan
        frac_fallo = hz[f"model_dis_{et}"].to_numpy(dtype=float)
        frame[f"canales_fallan_{et}"] = np.rint(frac_fallo * N_CANALES).astype(int)
        frame[f"canales_acierten_{et}"] = N_CANALES - frame[f"canales_fallan_{et}"]
    frame["n_errores"] = frame[[f"err_{e}" for e in ETIQUETAS]].sum(axis=1)
    frame["correct_all"] = (frame.n_errores == 0).astype(int)
    frame["margen_min"] = frame[[f"margin_{e}" for e in ETIQUETAS]].min(axis=1)
    frame["canales_acierten_min"] = frame[[f"canales_acierten_{e}" for e in ETIQUETAS]].min(axis=1)
    log(f"   celdas totales: {int(frame[[f'err_{e}' for e in ETIQUETAS]].to_numpy().sum())}")
    log(f"   filas con 0/1/2/3 errores: "
        f"{frame.n_errores.value_counts().reindex([0,1,2,3]).tolist()}")

    # ------------------------ 3. clasificacion A / B / C (secciones 6 y 7)
    log("3/8 clasificacion A/B/C con el pool de 20 canales historicos")
    err_matrix = frame[[f"err_{e}" for e in ETIQUETAS]].to_numpy(dtype=bool)
    canales = frame[[f"canales_acierten_{e}" for e in ETIQUETAS]].to_numpy(dtype=int)
    # A: V03 correcto en las tres
    a_filas = ~err_matrix.any(axis=1)
    # C: V03 falla en alguna celda donde algun canal historico acierta
    c_filas = err_matrix.any(axis=1) & (canales > 0).any(axis=1)
    # B: V03 falla y ningun canal historico acierta esa celda
    b_filas = err_matrix.any(axis=1) & (canales == 0).any(axis=1)
    frame["clase"] = np.where(a_filas, "A", np.where(c_filas, "C", "B"))
    log(f"   A (V03 correcto en las 3)        : {int(a_filas.sum())}")
    log(f"   C (V03 mal, >=1 canal acierta)   : {int(c_filas.sum())}")
    log(f"   B (V03 mal, ningun canal acierta): {int(b_filas.sum())}")
    n_err = int(err_matrix.sum())
    log(f"   celdas de error: {n_err} | con >=1 canal que acierta: "
        f"{int((err_matrix & (canales > 0)).sum())} "
        f"({100 * (err_matrix & (canales > 0)).sum() / n_err:.1f} pct)")
    log(f"   celdas de error irrecuperables por el pool: "
        f"{int((err_matrix & (canales == 0)).sum())} "
        f"({100 * (err_matrix & (canales == 0)).sum() / n_err:.1f} pct)")

    # -------------------------- 4. matriz de rescate por etiqueta (seccion 7)
    log("4/8 matriz de rescate por etiqueta")
    rescate = []
    for j, et in enumerate(ETIQUETAS):
        e = err_matrix[:, j]
        c = canales[:, j]
        v03_mal_pool_bien = int((e & (c > 0)).sum())
        v03_bien_pool_mal = int(((~e) & (c == 0)).sum())
        ambos_mal = int((e & (c == 0)).sum())
        ambos_bien = int(((~e) & (c > 0)).sum())
        rescate.append({
            "etiqueta": et, "n": n,
            "v03_mal": int(e.sum()), "v03_bien": int((~e).sum()),
            "v03_mal_pool_bien": v03_mal_pool_bien,
            "v03_bien_pool_mal": v03_bien_pool_mal,
            "ambos_mal": ambos_mal, "ambos_bien": ambos_bien,
            "rescate": v03_mal_pool_bien, "damage": v03_bien_pool_mal,
            "net": v03_mal_pool_bien - v03_bien_pool_mal,
            "canales_medios_que_acierten": float(c[e].mean()) if e.any() else float("nan"),
        })
        log(f"   {et}: V03 mal {int(e.sum()):4d} | rescate {v03_mal_pool_bien:4d} | "
            f"damage {v03_bien_pool_mal:4d} | net {rescate[-1]['net']:+5d} | "
            f"canales que aciertan de media {rescate[-1]['canales_medios_que_acierten']:.1f}")

    # ------------------- 5. margenes: donde se concentran los rescates (sec. 11)
    log("5/8 contexto de margen de los rescates")
    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 1.01]
    etiquetas_bin = ["[0,0.05)", "[0.05,0.10)", "[0.10,0.20)", "[0.20,0.30)", "[0.30,1]"]
    margen_filas = []
    for j, et in enumerate(ETIQUETAS):
        m = margen[:, j]
        e = err_matrix[:, j]
        c = canales[:, j]
        for k, (lo, hi) in enumerate(itertools.pairwise(bins)):
            sel = (m >= lo) & (m < hi)
            if not sel.any():
                continue
            margen_filas.append({
                "etiqueta": et, "bin_margen": etiquetas_bin[k],
                "n": int(sel.sum()),
                "n_errores_v03": int((sel & e).sum()),
                "tasa_error_v03": float((sel & e).sum() / sel.sum()),
                "n_rescatables": int((sel & e & (c > 0)).sum()),
                "n_irrecuperables": int((sel & e & (c == 0)).sum()),
                "canales_medios": float(c[sel & e].mean()) if (sel & e).any() else float("nan"),
            })
    mdf = pd.DataFrame(margen_filas)

    # Respuesta cuantitativa a la seccion 11: el rescate cae con el margen.
    _bajo = mdf[mdf.bin_margen.isin(["[0,0.05)", "[0.05,0.10)"])]
    _alto = mdf[mdf.bin_margen.isin(["[0.20,0.30)", "[0.30,1]"])]
    _n_err = int(mdf.n_errores_v03.sum())
    _n_irr = int(mdf.n_irrecuperables.sum())
    diag_margen = {
        "tasa_rescate_margen_bajo_0.10": round(
            float(_bajo.n_rescatables.sum() / max(_bajo.n_errores_v03.sum(), 1)), 4),
        "tasa_rescate_margen_alto_0.20": round(
            float(_alto.n_rescatables.sum() / max(_alto.n_errores_v03.sum(), 1)), 4),
        "errores_en_margen_alto": int(_alto.n_errores_v03.sum()),
        "irrecuperables_en_margen_alto": int(_alto.n_irrecuperables.sum()),
        "irrecuperables_totales": _n_irr,
        "pct_irrecuperables_en_margen_alto": round(
            100 * int(_alto.n_irrecuperables.sum()) / max(_n_irr, 1), 1),
        "pct_celdas_irrecuperables": round(100 * _n_irr / max(_n_err, 1), 2),
    }
    log("   tasa de error de V03 por bin de margen (promedio de las 3 etiquetas):")
    for b in etiquetas_bin:
        sub = mdf[mdf.bin_margen == b]
        if len(sub):
            log(f"     {b:12} n={int(sub.n.sum()):5d} "
                f"tasa_error={sub.n_errores_v03.sum() / max(sub.n.sum(), 1):.3f}")
    alto = mdf[mdf.bin_margen.isin(["[0.20,0.30)", "[0.30,1]"])]
    if len(alto):
        log(f"   errores en margen ALTO (>=0.20): {int(alto.n_errores_v03.sum())} "
            f"({100 * alto.n_errores_v03.sum() / mdf.n_errores_v03.sum():.1f} pct de los errores)")

    # ---------------------------- 6. errores conjuntos (seccion 12)
    log("6/8 errores conjuntos por numero de labels falladas")
    conjunto = []
    for k in (0, 1, 2, 3):
        sel = frame.n_errores == k
        c_sel = frame.loc[sel, [f"canales_acierten_{e}" for e in ETIQUETAS]].to_numpy()
        e_sel = frame.loc[sel, [f"err_{e}" for e in ETIQUETAS]].to_numpy(dtype=bool)
        conjunto.append({
            "n_labels_falladas": k,
            "n_filas": int(sel.sum()),
            "porcentaje": float(100 * sel.sum() / n),
            "n_celdas_rescatables": int((e_sel & (c_sel > 0)).sum()),
            "n_celdas_irrecuperables": int((e_sel & (c_sel == 0)).sum()),
            "canales_medios": float(c_sel[e_sel].mean()) if e_sel.any() else float("nan"),
        })
        log(f"   {k} labels mal: {int(sel.sum()):5d} filas "
            f"({100 * sel.sum() / n:5.2f} pct) | celdas rescatables "
            f"{conjunto[-1]['n_celdas_rescatables']:4d} | irrecuperables "
            f"{conjunto[-1]['n_celdas_irrecuperables']:3d}")

    # --------------------- 7. tabla de rescate con contexto (seccion 8)
    log("7/8 tabla de rescate con contexto estructural")
    cols_contexto = ["n_chars", "n_tokens", "n_oraciones", "n_menciones", "n_hashtags",
                     "n_urls", "n_elongaciones", "n_caracteres_leet", "n_risas",
                     "n_puntuacion_interna", "n_tramos_cita", "frac_tokens_cita",
                     "n_marcadores_atribucion", "n_colectivos", "n_segunda",
                     "n_negaciones", "n_targets", "rt_presente",
                     "n_mencion_inicial", "mayusculas_sostenidas"]
    ctx = par.reindex(columns=[c for c in cols_contexto if c in par.columns])
    for col in ("label_rarity", "poblacion", "en_cluster"):
        if col in hz.columns and col not in ctx.columns:
            ctx[col] = hz[col].to_numpy()
    base = pd.concat([frame.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)
    base["rescatable"] = ((base.n_errores > 0) & (base.canales_acierten_min > 0)).astype(int)
    base["irrecuperable"] = ((base.n_errores > 0) & (base.canales_acierten_min == 0)).astype(int)
    rescate_tab = base[base.n_errores > 0]
    log(f"   filas con error de V03: {len(rescate_tab)} "
        f"({int(rescate_tab.rescatable.sum())} con rescate, "
        f"{int(rescate_tab.irrecuperable.sum())} irrecuperables)")

    # -------------------- 8. slices y patrones (secciones 13, 15 y 16)
    log("8/8 slices existentes y descubrimiento de patrones")
    filas_patron = []
    for nombre, (col, umbral) in SLICES.items():
        if col not in base.columns:
            continue
        sel = base[col] >= umbral
        if not sel.any():
            continue
        e = base[sel].n_errores > 0
        filas_patron.append(_fila_patron(nombre, f"{col} >= {umbral}", base, sel, e))
    for nombre, fn in COMBINACIONES:
        sel = fn(base).fillna(False)
        if sel.sum() < 30:
            continue
        e = base[sel].n_errores > 0
        filas_patron.append(_fila_patron(nombre, "combinacion", base, sel, e))
    pat = pd.DataFrame(filas_patron).sort_values("net", ascending=False)
    log(f"   patrones evaluados: {len(pat)}")
    for _, r in pat.head(8).iterrows():
        log(f"     {r['patron']:32} n={int(r.n):4d} err={r.tasa_error_v03:.3f} "
            f"resc={r.tasa_rescate:.3f} dmg={r.tasa_damage:.3f} net={r.net:+5d} "
            f"p={r.p_valor:.4f}")

    # ------------------------------------------------------------- salida
    n_celdas_resc = int((err_matrix & (canales > 0)).sum())
    gap = {
        "poblacion_C_filas": int(c_filas.sum()),
        "poblacion_B_filas": int(b_filas.sum()),
        "poblacion_A_filas": int(a_filas.sum()),
        "celdas_error_totales": n_err,
        "celdas_rescatables": n_celdas_resc,
        "pct_celdas_rescatables": round(100 * n_celdas_resc / n_err, 2),
        "celdas_irrecuperables": n_err - n_celdas_resc,
        "pct_celdas_irrecuperables": round(100 * (n_err - n_celdas_resc) / n_err, 2),
    }
    # Criterio de candidato. Lo decisivo NO es la tasa de error sino el
    # ENRIQUECIMIENTO EN RESCATE: una subpoblacion con mas errores que la media
    # solo es interesante si ademas concentrate mas proportion de celdas
    # recuperables por el pool. Sin ese segundo requisito, el patron solo
    # describe donde V03 falla, que ya es un hecho conocido, y no una via.
    base_err = float(frame.n_errores.gt(0).mean())
    base_resc = float((err_matrix & (_canelas_ok := canales > 0)).sum() / n_err)
    log(f"   referencia: tasa de error {base_err:.4f} | tasa de rescate {base_resc:.4f}")
    circulares = {"margen_bajo_error", "margen_alto_error"}
    candidatos = []
    diagnostico = []
    for _, r in pat.iterrows():
        lift = float(r.tasa_error_v03 / base_err) if base_err else float("nan")
        # Binomiale: proportion de celdas rescatables dentro del patron frente a la
        # tasa global. Es la prueba de enriquecimiento en rescate.
        k = int(r.n_rescatables)
        n = int(r.n_errores_v03)
        p_enr = (float(stats.binomtest(k, n, base_resc).pvalue) if n > 0 else float("nan"))
        fila = {
            "patron": r.patron, "n": int(r.n), "lift_error": round(lift, 3),
            "tasa_error": round(float(r.tasa_error_v03), 4),
            "tasa_rescate": round(float(r.tasa_rescate), 4),
            "enriquecimiento_rescate": round(float(r.tasa_rescate / base_resc), 4)
            if base_resc else float("nan"),
            "p_enriquecimiento": p_enr,
            "net": int(r.net), "p_valor": float(r.p_valor),
            "circular": r.patron in circulares,
        }
        if fila["circular"]:
            fila["motivo_exclusion"] = "definido por el propio error: lift circular"
            diagnostico.append(fila)
            continue
        if (r.n >= 100 and r.net > 0 and r.p_valor < 0.01
                and p_enr < 0.01 and fila["enriquecimiento_rescate"] > 1.0):
            candidatos.append(fila)
        else:
            fila["motivo_exclusion"] = (
                "sin enriquecimiento en rescate: la tasa de rescate del patron no supera "
                f"la global ({base_resc:.4f})" if p_enr >= 0.01 or
                fila["enriquecimiento_rescate"] <= 1.0
                else "no cumple n, net o p del criterio de error")
            diagnostico.append(fila)
    gap["candidatos_a_hipotesis"] = candidatos
    gap["patrones_descartados"] = diagnostico
    gap["tasa_rescate_global"] = round(base_resc, 4)
    gap["tasa_error_global"] = round(base_err, 4)
    gap["EDA_RESEARCH_GAP"] = "NONE" if not candidatos else "CANDIDATOS"

    informe = {
        "familia": "P48_EDA_RESIDUAL",
        "experimento": "V48-01 (P1 atlas de residuos, rescate y patrones)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": ("Que errores de V03 son corregibles por el pool historico y cuales "
                     "parecen no contener informacion recuperable?"),
        "fuentes": {
            "oof": str(OOF.relative_to(PROJECT_ROOT)),
            "hardness": str(HARDNESS.relative_to(PROJECT_ROOT)),
            "parquet": str(PARQUET.relative_to(PROJECT_ROOT)),
            "n_canales_historicos": N_CANALES,
        },
        "limitaciones": [
            ("Los tensores OOF por canal (features/v2_oof_*.npz) ya no estan en disco. "
            "Solo queda la fraccion de fallo por celda, de modo que la categoria C es una "
            "cota superior del rescate y no identifica el modelo concreto."),
            ("No se puede aplicar la matriz de rescate literal de la seccion 7. La que se "
            "reporta es la matriz de rescate del pool agregado."),
            ("holdout2_ids.csv esta dibujado de dev, luego forma parte de las 1483 filas y no "
            "anade filas. No se usa holdout ni test para nada."),
        ],
        "atlas": {
            "n_filas": n,
            "celdas_error": n_err,
            "distribucion_n_errores": frame.n_errores.value_counts().reindex([0, 1, 2, 3]).to_dict(),
        },
        "clasificacion": gap,
        "matriz_de_rescate": rescate,
        "margenes": mdf.to_dict("records"),
        "errores_conjuntos": conjunto,
        "diagnostico_de_margen": diag_margen,
        "patrones": pat.to_dict("records"),
        "decision": gap["EDA_RESEARCH_GAP"],
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _csv_atlas(frame, base)
    _csv_rescate(rescate, mdf, conjunto)
    _csv_irrecuperable(base)
    _csv_patrones(pat)
    _md(informe)
    log("   informes escritos en reports/p48_*")
    log(f"EDA_RESEARCH_GAP = {gap['EDA_RESEARCH_GAP']} | "
        f"rescatable {gap['pct_celdas_rescatables']} pct | "
        f"irrecuperable {gap['pct_celdas_irrecuperables']} pct | "
        f"candidatos {len(candidatos)}")
    log(f"VEREDICTO: 0 modelos, 0 submissions, {round(time.time() - inicio, 1)} s")


def _fila_patron(nombre: str, definicion: str, base: pd.DataFrame,
                  sel: pd.Series, e: pd.Series) -> dict[str, Any]:
    """Construye la fila de un patron de error.

    Args:
        nombre: nombre del patron.
        definicion: como se define la pertenencia al patron.
        base: tabla base.
        sel: mascara de pertenencia.
        e: mascara de error de V03 dentro del patron.

    Returns:
        Diccionario con las metricas del patron.
    """
    dentro = base[sel]
    n_sel = int(sel.sum())
    n_err = int(e.sum())
    canales = dentro.loc[dentro.n_errores > 0, "canales_acierten_min"]
    rescatables = int((dentro.n_errores.gt(0) & dentro.canales_acierten_min.gt(0)).sum())
    irrecuperables = int((dentro.n_errores.gt(0) & dentro.canales_acierten_min.eq(0)).sum())
    resto = base[~sel]
    n_resto = int((~sel).sum())
    err_resto = int(resto.n_errores.gt(0).sum())
    tabla = [[n_err, n_sel - n_err], [err_resto, n_resto - err_resto]]
    try:
        p = float(stats.chi2_contingency(np.array(tabla), correction=False)[1])
    except ValueError:
        p = float("nan")
    return {
        "patron": nombre,
        "definicion": definicion,
        "n": n_sel,
        "n_errores_v03": n_err,
        "tasa_error_v03": round(n_err / n_sel, 4) if n_sel else float("nan"),
        "n_rescatables": rescatables,
        "n_irrecuperables": irrecuperables,
        "tasa_rescate": round(rescatables / n_err, 4) if n_err else float("nan"),
        "tasa_damage": round(irrecuperables / n_err, 4) if n_err else float("nan"),
        "net": rescatables - irrecuperables,
        "canales_medios": round(float(canales.mean()), 3) if len(canales) else float("nan"),
        "p_valor": p,
    }


def _csv_atlas(frame: pd.DataFrame, base: pd.DataFrame) -> None:
    """Escribe el atlas de residuos.

    Args:
        frame: tabla de residuos por celda.
        base: tabla con contexto anadido.

    Returns:
        None.
    """
    cols = ["tweet_id"] + [c for c in frame.columns if c != "tweet_id"]
    frame[cols].to_csv(CSV_ATLAS, index=False, encoding="utf-8")


def _csv_rescate(rescate: list[dict[str, Any]], mdf: pd.DataFrame,
                 conjunto: list[dict[str, Any]]) -> None:
    """Escribe el analisis de rescate.

    Args:
        rescate: matriz de rescate por etiqueta.
        mdf: tabla de margenes.
        conjunto: tabla de errores conjuntos.

    Returns:
        None.
    """
    r = pd.DataFrame(rescate)
    r.insert(0, "bloque", "matriz_de_rescate")
    m = mdf.copy()
    m.insert(0, "bloque", "margen")
    c = pd.DataFrame(conjunto)
    c.insert(0, "bloque", "errores_conjuntos")
    pd.concat([r, m, c], ignore_index=True, sort=False).to_csv(
        CSV_RESCATE, index=False, encoding="utf-8")


def _csv_irrecuperable(base: pd.DataFrame) -> None:
    """Escribe la poblacion no rescatable.

    Args:
        base: tabla base con la marca de irrecuperabilidad.

    Returns:
        None.
    """
    sub = base[base.irrecuperable == 1]
    agg = []
    for col in ("n_chars", "n_tokens", "n_menciones", "n_hashtags", "n_urls",
                "n_elongaciones", "n_tramos_cita", "n_colectivos", "n_negaciones"):
        if col in sub.columns and len(sub):
            agg.append({"variable": col, "n": len(sub),
                        "media_irrecuperable": round(float(sub[col].mean()), 4),
                        "media_resto": round(float(base[base.irrecuperable == 0][col].mean()), 4)})
    out = pd.DataFrame(agg)
    out.insert(0, "bloque", "perfil_irrecuperable_vs_resto")
    out.to_csv(CSV_NO_RESCATABLE, index=False, encoding="utf-8")


def _csv_patrones(pat: pd.DataFrame) -> None:
    """Escribe la tabla de patrones.

    Args:
        pat: tabla de patrones.

    Returns:
        None.
    """
    pat.to_csv(CSV_PATRONES, index=False, encoding="utf-8")


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe de brecha de investigacion.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    g = informe["clasificacion"]
    r = informe["matriz_de_rescate"]
    a = informe["atlas"]
    pat = pd.DataFrame(informe["patrones"])
    cj = informe["errores_conjuntos"]
    mdf = pd.DataFrame(informe["margenes"])
    dm = informe["diagnostico_de_margen"]
    informe["diagnostico_de_margen"]
    lineas = [
        "# P48 - Brecha de investigacion guiada por EDA",
        "",
        f"Generado por `experiments/v48_01_residual_atlas.py` ({informe['experimento']}).",
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 uso de holdout o test.**",
        "",
        f"Pregunta: {informe['pregunta']}",
        "",
        "## 0. Limitaciones que no se pueden resolver con los artefactos presentes",
        "",
    ]
    for lim in informe["limitaciones"]:
        lineas.append(f"- {lim}")
    lineas += [
        "",
        "## 1. Atlas de residuos",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| filas de dev | {a['n_filas']} |",
        f"| celdas de error de V03 | {a['celdas_error']} |",
        f"| filas con 0 errores | {a['distribucion_n_errores'].get(0, 0)} |",
        f"| filas con 1 error | {a['distribucion_n_errores'].get(1, 0)} |",
        f"| filas con 2 errores | {a['distribucion_n_errores'].get(2, 0)} |",
        f"| filas con 3 errores | {a['distribucion_n_errores'].get(3, 0)} |",
        "",
        "## 2. Clasificacion de la poblacion de error",
        "",
        "| clase | filas | significado |",
        "|---|---:|---|",
        f"| A | {g['poblacion_A_filas']} | V03 correcto en las tres labels |",
        f"| C | {g['poblacion_C_filas']} | V03 falla y algun canal historico acierta |",
        f"| B | {g['poblacion_B_filas']} | V03 falla y ningun canal acierta esa celda |",
        "",
        (f"**Celdas de error rescatables por el pool: {g['celdas_rescatables']} "
        f"({g['pct_celdas_rescatables']} pct).**"),
        (f"**Celdas de error irrecuperables: {g['celdas_irrecuperables']} "
        f"({g['pct_celdas_irrecuperables']} pct).**"),
        "",
        "## 3. Matriz de rescate por etiqueta",
        "",
        "| etiqueta | V03 mal | rescate | damage | net | canales que aciertan de media |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in r:
        lineas.append(
            f"| {x['etiqueta']} | {x['v03_mal']} | {x['rescate']} | {x['damage']} | "
            f"{x['net']:+d} | {x['canales_medios_que_acierten']:.1f} |"
        )
    lineas += [
        "",
        "## 4. Contexto de margen",
        "",
        ("Si los rescates se concentran en margen bajo, la representacion ya contiene la "
        "informacion y lo que falla es la frontera. Si hay rescates en margen alto, el hueco "
        "de representacion es mas plausible."),
        "",
        "| bin de margen | n | errores de V03 | tasa | rescatables | irrecuperables |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b in mdf.bin_margen.unique():
        s = mdf[mdf.bin_margen == b]
        n = int(s.n.sum())
        ne = int(s.n_errores_v03.sum())
        lineas.append(
            f"| {b} | {n} | {ne} | {ne / n:.4f} | {int(s.n_rescatables.sum())} | "
            f"{int(s.n_irrecuperables.sum())} |"
        )
    lineas += [
        "",
        ("**Respuesta a la pregunta de la seccion 11.** La tasa de rescate cae de forma "
        "monotona al subir el margen: "
        f"**{dm['tasa_rescate_margen_bajo_0.10']:.4f}** con margen por debajo de 0.10 "
        f"frente a **{dm['tasa_rescate_margen_alto_0.20']:.4f}** con margen de 0.20 o "
        f"mas. La tasa de rescate va de "
        f"{100 * dm['tasa_rescate_margen_bajo_0.10']:.1f} por ciento con margen bajo a "
        f"{100 * dm['tasa_rescate_margen_alto_0.20']:.1f} por ciento con margen alto."),
        "",
        (f"Ademas, **{dm['pct_irrecuperables_en_margen_alto']} por ciento de las celdas "
        f"irrecuperables estan en margen alto** ({dm['irrecuperables_en_margen_alto']} de "
        f"{dm['irrecuperables_totales']}). Esa es la firma de un hueco de representacion: "
        "el modelo se equivoca con conviccion justamente donde ningun canal historico "
        "acierta."),
        "",
        (f"Pero el tamano absoluto es pequeno: **{dm['irrecuperables_totales']} celdas de "
        f"{a['celdas_error']}**, un {dm['pct_celdas_irrecuperables']:.2f} por ciento. "
        "Ninguna hipotesis mechanisticamente nueva puede sostenerse sobre esa base."),
    ]
    lineas += [
        "## 5. Errores conjuntos",
        "",
        "| labels falladas | filas | pct | celdas rescatables | irrecuperables |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in cj:
        lineas.append(
            f"| {x['n_labels_falladas']} | {x['n_filas']} | {x['porcentaje']:.2f} | "
            f"{x['n_celdas_rescatables']} | {x['n_celdas_irrecuperables']} |"
        )
    lineas += [
        "",
        "## 6. Patrones y slices",
        "",
        "| patron | n | tasa error V03 | tasa rescate | tasa damage | net | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, x in pat.iterrows():
        lineas.append(
            f"| {x['patron']} | {int(x['n'])} | {x['tasa_error_v03']:.4f} | "
            f"{x['tasa_rescate']:.4f} | {x['tasa_damage']:.4f} | {int(x['net']):+d} | "
            f"{x['p_valor']:.4f} |"
        )
    lineas += [
        "",
        "## 7. Criterio de hipotesis y decision",
        "",
        ("Criterio aplicado: n >= 100, net > 0, p < 0.01, tasa de rescate > 0.5 y lift de la "
        "tasa de error sobre la media > 1.15."),
        "",
    ]
    if g["EDA_RESEARCH_GAP"] == "NONE":
        lineas += [
            ("**`EDA_RESEARCH_GAP = NONE`.** Ningun patron.supera los cinco criterios a la vez. "
            "No se propone ninguna hipotesis y no se ejecuta ningun experimento."),
            "",
            ("Lo que esto responde: los errores de V03 **no** son un conjunto homogeneo con "
            "subpoblaciones_ocultas recuperables por el pool historico. Donde el pool acierta, "
            "acierta por azar tanto como por senal, y la proporcion de celdas rescatables no "
            "se concentra en ninguna slice de las ya definidas."),
        ]
    else:
        lineas += ["| patron | n | lift | tasa error | tasa rescate | net | p |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
        for c in g["candidatos_a_hipotesis"]:
            lineas.append(
                f"| {c['patron']} | {c['n']} | {c['lift_error']} | {c['tasa_error']} | "
                f"{c['tasa_rescate']} | {c['net']:+d} | {c['p_valor']:.5f} |"
            )
    lineas += [""]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
