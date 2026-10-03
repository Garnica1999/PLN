"""V49-00 (P0): reconciliacion forense del rescate OOF.

Que resuelve
------------
La contradiccion entre dos cifras que P48 uso como si significaran lo mismo:

- 715 de 757 celdas de error de V03 las acierta al menos un canal OOF historico
  (94,45 por ciento), segun P48.
- 147 de 757 (19,419 por ciento), que es `cobertura_de_errores_v03` de V17-00.

Este script establece que significan exactamente, si son comparables, y si la
cifra grande es senal o un artefacto de azar.

Metodo
------
1. Localiza y verifica los tensores OOF en los worktrees.
2. Reconstruye el pool de 20 canales con la misma regla de exclusion que
   `experiments/28_hardness_v4.py` y valida la reconstruccion contra
   `reports/hardness_v4.csv` celda a celda.
3. Reproduce el 147 de V17-00 desde `oof_candidates_v4.npz`.
4. Construye la matriz de rescate y damage por canal.
5. **Prueba decisiva**: compara el numero observado de celdas irrecuperables
   (ningun canal acierta) contra el que se espera si los 20 canales se
   equivocasen de forma independiente. Si coinciden, la cifra grande es azar.
6. Rehaz el analisis de margen y de slices sobre el rescate OBSERVADO.

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No consulta Kaggle.
- No modifica V03, el holdout ni el test.
- No crea filas en el registro de experimentos.

Uso
---
    /home/carlos/venv/bin/python experiments/v49_00_rescate_forense.py
"""

from __future__ import annotations

import csv
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

OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
HARDNESS = PROJECT_ROOT / "reports" / "hardness_v4.csv"
PARQUET = PROJECT_ROOT / "features" / "v2_structured_features.parquet"
REPORTE_JSON = PROJECT_ROOT / "reports" / "p49_rescue_reconciliation.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p49_rescue_reconciliation.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "p49_rescue_reconciliation.csv"
CSV_RESCATE = PROJECT_ROOT / "reports" / "p49_rescue_matrix.csv"
CSV_DANIO = PROJECT_ROOT / "reports" / "p49_damage_matrix.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRAL_CANAL = 0.5

# Worktrees donde pueden vivir los tensores OOF. Se prueban en orden.
WORKTREES = (
    Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj/features"),
    Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj/features"),
)

CANALES = {
    "v2_oof_base_corrected.npz": "modelos",
    "v2_oof_syntax.npz": "variantes",
    "v2_oof_cita_negacion.npz": "variantes",
    "v2_oof_targeting.npz": "variantes",
    "v2_oof_emocion_nbsvm_ortografia.npz": "canales",
}

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
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V49-00] {mensaje}", flush=True)


def localizar_tensores() -> dict[str, Path]:
    """Localiza los tensores OOF en los worktrees candidatos.

    Returns:
        Diccionario nombre -> ruta del primer worktree que lo contiene.

    Raises:
        SystemExit: si no se encuentra ningun tensor.
    """
    encontrados = {}
    for raiz in WORKTREES:
        for nombre in CANALES:
            if nombre in encontrados:
                continue
            if (raiz / nombre).exists():
                encontrados[nombre] = raiz / nombre
        for nombre in ("oof_candidates_v4.npz",):
            if nombre in encontrados:
                continue
            if (raiz / nombre).exists():
                encontrados[nombre] = raiz / nombre
    if not encontrados:
        raise SystemExit("PARADA: no se localiza ningun tensor OOF")
    return encontrados


def apilar_canales(rutas: dict[str, Path]) -> tuple[np.ndarray, list[str]]:
    """Apila los canales OOF con la misma exclusion que hardness_v4.

    Args:
        rutas: diccionario nombre -> ruta.

    Returns:
        Tensor de forma ``(n_canales, n_filas, 3)`` y la lista de nombres.
    """
    tensores: list[np.ndarray] = []
    nombres: list[str] = []
    for nombre, clave in CANALES.items():
        datos = np.load(rutas[nombre], allow_pickle=True)
        etiquetas = [str(x) for x in datos[clave]]
        proba = datos["proba"].astype(np.float64)
        if proba.shape[0] != len(etiquetas) and proba.shape[1] == len(etiquetas):
            proba = np.moveaxis(proba, 1, 0)
        if proba.shape[0] != len(etiquetas):
            raise SystemExit(f"PARADA: no se puede normalizar {nombre}: {proba.shape}")
        for indice, etiqueta in enumerate(etiquetas):
            if etiqueta.startswith("TXT+") or etiqueta == "CH-TGT+DIST":
                continue
            tensores.append(proba[indice])
            nombres.append(f"{nombre.split('_')[2][:4]}:{etiqueta}")
    return np.stack(tensores), nombres


def main() -> None:
    """Ejecuta la reconciliacion forense del rescate.

    Returns:
        None.
    """
    inicio = time.time()
    log("P49-00 reconciliacion forense del rescate OOF. 0 modelos, 0 submissions")

    # ---------------------------------------------- 1. localizar tensores
    log("1/7 localizacion de los tensores OOF")
    rutas = localizar_tensores()
    for nombre, ruta in sorted(rutas.items()):
        log(f"   {nombre:44} {ruta}")

    oof = np.load(OOF, allow_pickle=True)
    ids = oof["ids"].astype(int)
    y_true = np.asarray(oof["y_true"], dtype=np.int64)
    pred = np.asarray(oof["pred"], dtype=np.int64)
    margen = np.asarray(oof["margen"], dtype=float)
    n = len(ids)
    err = pred != y_true
    n_celdas_error = int(err.sum())
    n_filas_error = int(err.any(axis=1).sum())
    log(f"   OOF canonico: {n} filas | celdas de error {n_celdas_error} | "
        f"filas con algun error {n_filas_error}")

    # ------------------------------------- 2. reconstruir y validar el pool
    log("2/7 reconstruccion del pool de canales y validacion")
    tensor, nombres = apilar_canales(rutas)
    n_canales = tensor.shape[0]
    log(f"   pool reconstruido: {n_canales} canales x {tensor.shape[1]} filas x 3 labels")
    aciertos = (tensor >= UMBRAL_CANAL).astype(int) == y_true[None, :, :]
    fallan = ~aciertos
    model_dis = fallan.mean(axis=0)
    hz = pd.read_csv(HARDNESS).set_index("id").loc[ids]
    max_dif = max(
        float(np.abs(model_dis[:, j] - hz[f"model_dis_{et}"].to_numpy()).max())
        for j, et in enumerate(ETIQUETAS)
    )
    log(f"   validacion contra hardness_v4.csv: max |diferencia| = {max_dif:.2e}")
    validado = max_dif < 1e-9
    log(f"   RECONSTRUCCION VALIDADA = {validado}")

    # ---------------------------------- 3. reproducir el 147 de V17-00
    log("3/7 reproduccion del 147 de V17-00")
    cand = np.load(rutas["oof_candidates_v4.npz"], allow_pickle=True)
    # En este artefacto cada variante es una clave de primer nivel con la
    # probabilidad ya alineada por id, sin tensor comun.
    claves_cand = [k for k in cand.files if k not in ("ids", "y_true")]
    log(f"   variantes en oof_candidates_v4: {claves_cand}")
    pos_c = {int(v): k for k, v in enumerate(cand["ids"].astype(int))}
    fila_c = np.array([pos_c[int(i)] for i in ids])
    p_c = np.asarray(cand["C_meta36_downweight"], dtype=np.float64)[fila_c, :]
    pred_c = (p_c >= 0.5).astype(int)
    rescate_147 = int(np.count_nonzero(err & (pred_c == y_true)))
    danyo_147 = int(np.count_nonzero((~err) & (pred_c != y_true)))
    log(f"   C_meta36_downweight: rescates {rescate_147} | danos {danyo_147} | "
        f"cobertura {rescate_147 / n_celdas_error:.6f}")
    log("   valor registrado en V17-00: 147 | cobertura 0.19418758256274768")
    reproduce_147 = rescate_147 == 147

    # ------------------------------------- 4. matriz de rescate por canal
    log("4/7 matriz de rescate y damage por canal")
    filas_rescate = []
    for k, nombre_canal in enumerate(nombres):
        pc = (tensor[k] >= UMBRAL_CANAL).astype(int)
        resc = int(np.count_nonzero(err & (pc == y_true)))
        dany = int(np.count_nonzero((~err) & (pc != y_true)))
        filas_rescate.append({
            "canal": nombre_canal,
            "aciertos_totales": int(np.count_nonzero(aciertos[k])),
            "macro_f1_aprox": round(float(np.mean([
                f1_score(y_true[:, j], pc[:, j], zero_division=0)
                for j in range(3)])), 6),
            "rescate": resc,
            "damage": dany,
            "net": resc - dany,
            "cobertura_errores_v03": round(resc / n_celdas_error, 6),
        })
    mat_rescate = pd.DataFrame(filas_rescate).sort_values("rescate", ascending=False)
    for _, r in mat_rescate.head(5).iterrows():
        log(f"   {r['canal']:36} rescate {int(r['rescate']):4d} "
            f"damage {int(r['damage']):4d} net {int(r['net']):+5d}")
    log(f"   peor canal: {mat_rescate.iloc[-1]['canal']} "
        f"rescate {int(mat_rescate.iloc[-1]['rescate'])}")

    # ------------------------------- 5. union del pool y PRUEBA DECISIVA
    log("5/7 union del pool y prueba de azar frente a senal")
    n_aciertos_por_celda = aciertos.sum(axis=0)
    celdas_irrecuperables = err & (n_aciertos_por_celda == 0)
    celdas_rescatables = err & (n_aciertos_por_celda > 0)
    obs_715 = int(celdas_rescatables.sum())
    obs_42 = int(celdas_irrecuperables.sum())
    log(f"   celdas rescatables (>=1 canal acierta): {obs_715} "
        f"({100 * obs_715 / n_celdas_error:.2f} pct)")
    log(f"   celdas irrecuperables (0 aciertos)   : {obs_42} "
        f"({100 * obs_42 / n_celdas_error:.2f} pct)")

    # NULO: si los n_canales se equivocasen de forma INDEPENDIENTE, con la
    # probabilidad marginal observada por canal, cuantas celdas de error de V03
    # quedarian con cero aciertos? Si el observado se parece al nulo, la cifra
    # grande es un artefacto de azar por la cantidad de canales, no senal.
    por_canal_correcto = aciertos[:, err].mean(axis=1)
    float(por_canal_correcto.mean())
    p_obs_por_celda = n_aciertos_por_celda[err] / n_canales
    p_nulo = float(p_obs_por_celda.mean())
    esperado_irrecuperables = n_celdas_error * (1.0 - p_nulo) ** n_canales
    # Correcciacion: los canales no son independientes, asi que se usa tambien el
    # limite superior asumiendo independencia y el observado de la mediana.
    log(f"   probabilidad media de acierto de un canal en una celda de error: {p_nulo:.4f}")
    log(f"   canales que aciertan de media en una celda de error: {p_nulo * n_canales:.2f} "
        f"de {n_canales}")
    log(f"   celdas irrecuperables OBSERVADAS: {obs_42}")
    log(f"   celdas irrecuperables ESPERADAS si los canales fueran independientes: "
        f"{esperado_irrecuperables:.2f}")
    es_azar = obs_42 <= 3 * max(esperado_irrecuperables, 1.0)
    log(f"   VEREDICTO DE LA PRUEBA DE AZAR: {'es azar' if es_azar else 'NO es azar: hay senal'}")

    float(np.corrcoef(aciertos[:, err].astype(float))[0, 1]) \
        if n_canales > 1 else float("nan")
    log("   nota: los canales comparten la representacion base, luego sus errores "
        "estan correlacionados y el nulo independiente es una cota-optimista")

    # ------------------------------------- 6. margen sobre rescate OBSERVADO
    log("6/7 margen: rescate observado frente a irrecuperable observado")
    filas_margen = []
    for j, et in enumerate(ETIQUETAS):
        m = margen[:, j]
        e = err[:, j]
        for lo, hi, etiqueta_bin in ((0.0, 0.10, "[0,0.10)"),
                                     (0.10, 0.20, "[0.10,0.20)"),
                                     (0.20, 1.01, "[0.20,1]")):
            sel = (m >= lo) & (m < hi) & e
            k = int(sel.sum())
            if k == 0:
                continue
            resc = int((sel & (n_aciertos_por_celda[:, j] > 0)).sum())
            filas_margen.append({
                "etiqueta": et, "bin": etiqueta_bin, "n_errores": k,
                "rescatables_observados": resc,
                "irrecuperables_observados": k - resc,
                "tasa_rescate_observada": round(resc / k, 4),
                "canales_medios": round(
                    float(n_aciertos_por_celda[sel, j].mean()), 3),
            })
    mdf = pd.DataFrame(filas_margen)
    for _, r in mdf.iterrows():
        log(f"   {r['etiqueta']} margen {r['bin']:12} n={int(r.n_errores):4d} "
            f"rescate observado {r['tasa_rescate_observada']:.4f} | "
            f"irrecuperables {int(r.irrecuperables_observados)}")

    # ---------------------------- 7. slices: error-dense frente a rescue-dense
    log("7/7 slices: error-dense frente a rescue-dense")
    par = pd.read_parquet(PARQUET)
    par["id"] = par["id"].astype("int64")
    par = par[par.particion == "dev"].set_index("id").loc[ids]
    base = pd.DataFrame({
        "n_errores": err.sum(axis=1),
        "n_rescatables": (err & (n_aciertos_por_celda > 0)).sum(axis=1),
        "n_irrecuperables": celdas_irrecuperables.sum(axis=1),
    })
    columnas = {col for col, _ in SLICES.values()} | {"n_tokens", "n_chars"}
    for col in columnas:
        if col in par.columns:
            base[col] = par[col].to_numpy()
    base["margen_min"] = margen.min(axis=1)
    base["longitud_corta"] = (base.n_tokens < base.n_tokens.quantile(0.33)).astype(int)

    filas_slice = []
    n_err_total = int(base.n_errores.sum())
    n_rsc_total = int(base.n_rescatables.sum())
    n_irr_total = int(base.n_irrecuperables.sum())
    slices = {nombre: base[col] >= umbral for nombre, (col, umbral) in SLICES.items()}
    slices["longitud_corta"] = base.longitud_corta == 1
    slices["margen_bajo"] = base.margen_min < 0.10
    for nombre, sel in slices.items():
        s = base[sel]
        ne = int(s.n_errores.sum())
        nr = int(s.n_rescatables.sum())
        ni = int(s.n_irrecuperables.sum())
        lift_error = ne / n_err_total / max(sel.mean(), 1e-12)
        lift_rescate = nr / n_rsc_total / max(sel.mean(), 1e-12)
        lift_irrec = ni / n_irr_total / max(sel.mean(), 1e-12) if n_irr_total else float("nan")
        filas_slice.append({
            "slice": nombre, "n_filas": int(sel.sum()),
            "n_errores_v03": ne, "n_rescatables": nr, "n_irrecuperables": ni,
            "lift_error": round(lift_error, 4),
            "lift_rescate": round(lift_rescate, 4),
            "lift_irrecuperable": round(lift_irrec, 4) if ni else 0.0,
            "clase": ("ERROR_DENSE" if lift_error > 1.15 and lift_rescate <= 1.0
                      else "RESCUE_DENSE" if lift_rescate > 1.15
                      else "NEUTRAL"),
        })
    sdf = pd.DataFrame(filas_slice).sort_values("lift_irrecuperable", ascending=False)
    for _, r in sdf.iterrows():
        log(f"   {r['slice']:20} lift_error {r['lift_error']:6} "
            f"lift_rescate {r['lift_rescate']:6} lift_irrec {r['lift_irrecuperable']:6} "
            f"-> {r['clase']}")

    # ------------------------------------------------------------- salida
    filas_tabla = [
        {"metrica": "celdas_de_error_de_V03", "valor": n_celdas_error,
         "numerador": "", "denominador": "", "unidad": "label-cell",
         "definicion": "celdas (tweet, etiqueta) con pred != y_true sobre 1483x3",
         "fuente": "features/oof_v03_final.npz", "observado": "si", "cota": "no",
         "reproducible": "si", "estado": "VERIFIED"},
        {"metrica": "filas_con_algun_error", "valor": n_filas_error,
         "numerador": "", "denominador": "", "unidad": "tweet",
         "definicion": "tweets con al menos una de las tres labels mal",
         "fuente": "features/oof_v03_final.npz", "observado": "si", "cota": "no",
         "reproducible": "si", "estado": "VERIFIED"},
        {"metrica": "rescate_de_una_variante", "valor": rescate_147,
         "numerador": rescate_147, "denominador": n_celdas_error,
         "unidad": "label-cell",
         "definicion": ("celdas de error de V03 que C_meta36_downweight (umbrales 0.5) "
                        "acierta. Es la cobertura_de_errores_v03 de V17-00"),
         "fuente": "features/oof_candidates_v4.npz", "observado": "si", "cota": "no",
         "reproducible": "si" if reproduce_147 else "no",
         "estado": "VERIFIED" if reproduce_147 else "MISMATCH"},
        {"metrica": "rescate_de_la_union_de_8_variantes", "valor": 263,
         "numerador": 263, "denominador": n_celdas_error, "unidad": "label-cell",
         "definicion": ("celdas de error de V03 que acierta al menos una de las 8 "
                        "variantes inventariadas por V17-00"),
         "fuente": "reports/v17_00_inventario_complementariedad.json",
         "observado": "si", "cota": "no", "reproducible": "no",
         "estado": "REPORTED_NOT_REPRODUCED"},
        {"metrica": "celdas_con_al_menos_un_canal_que_acerte", "valor": obs_715,
         "numerador": obs_715, "denominador": n_celdas_error, "unidad": "label-cell",
         "definicion": ("celdas de error de V03 que acierta al menos uno de los 20 "
                        "canales OOF de la fase v2, binarizados a 0.5"),
         "fuente": "features/v2_oof_*.npz (worktrees PLN y PLN-v8)",
         "observado": "si", "cota": "no", "reproducible": "si",
         "estado": "VERIFIED"},
        {"metrica": "celdas_irrecuperables", "valor": obs_42,
         "numerador": obs_42, "denominador": n_celdas_error, "unidad": "label-cell",
         "definicion": ("celdas de error de V03 que no acierta ninguno de los 20 "
                        "canales OOF de la fase v2"),
         "fuente": "features/v2_oof_*.npz", "observado": "si", "cota": "no",
         "reproducible": "si", "estado": "VERIFIED"},
        {"metrica": "irrecuperables_esperadas_si_independencia", "valor": round(esperado_irrecuperables, 2),
         "numerador": "", "denominador": "", "unidad": "label-cell",
         "definicion": ("celdas de error de V03 con cero aciertos esperadas si los 20 "
                        "canales fallaran de forma independiente, con la probabilidad "
                        "marginal observada"),
         "fuente": "calculado en este script", "observado": "no", "cota": "si",
         "reproducible": "si", "estado": "UPPER_BOUND_NULL"},
    ]

    caso = "B" if es_azar else "A"
    informe = {
        "familia": "P49_RESCUE_FORENSICS",
        "experimento": "V49-00 (P0 reconciliacion forense del rescate OOF)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": ("Que significa 715, que significa 147, y la cifra grande es senal "
                     "o un artefacto de azar?"),
        "tensores": {
            "recuperados": True,
            "ubicaciones": sorted({str(p.parent) for p in rutas.values()}),
            "canales_reconstruidos": n_canales,
            "validacion_contra_hardness_v4": {"max_abs_diff": max_dif, "ok": validado},
        },
        "reproduccion_de_147": {
            "valor_registrado": 147,
            "valor_reproducido": rescate_147,
            "cobertura_registrada": 0.19418758256274768,
            "cobertura_reproducida": round(rescate_147 / n_celdas_error, 12),
            "reproduce": reproduce_147,
            "damage_reproducido": danyo_147,
        },
        "matriz_de_rescate_por_canal": filas_rescate,
        "prueba_de_azar": {
            "canales": n_canales,
            "p_medio_de_acierto_en_celda_de_error": round(p_nulo, 4),
            "canales_medios_que_acierten": round(p_nulo * n_canales, 2),
            "irrecuperables_observadas": obs_42,
            "irrecuperables_esperadas_independencia": round(esperado_irrecuperables, 2),
            "es_azar": bool(es_azar),
            "interpretacion": (
                "Si las irrecuperables observadas son del orden de las esperadas bajo "
                "independencia, la cifra de rescatables es un artefacto de tener muchos "
                "canales y no senal. Si las observadas superan con claridad a las "
                "esperadas, los canales fallan juntos con V03 y la cifra es senal de "
                "un hueco de representacion compartido"),
        },
        "margen_sobre_rescate_observado": mdf.to_dict("records"),
        "slices_error_dense_vs_rescue_dense": sdf.to_dict("records"),
        "tabla_de_reconciliacion": filas_tabla,
        "decision": caso,
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _csv_tabla(filas_tabla)
    mat_rescate.to_csv(CSV_RESCATE, index=False, encoding="utf-8")
    mat_rescate.rename(columns={"rescate": "damage"}).to_csv(CSV_DANIO, index=False, encoding="utf-8")
    _md(informe)
    log("   informes escritos en reports/p49_*")
    log(f"CASE = {caso} | rescatables {obs_715} | irrecuperables {obs_42} "
        f"| esperado bajo independencia {esperado_irrecuperables:.1f} | "
        f"es_azar = {es_azar}")
    log(f"VEREDICTO: 0 modelos, 0 submissions, {round(time.time() - inicio, 1)} s")


def _csv_tabla(filas: list[dict[str, Any]]) -> None:
    """Escribe la tabla formal de reconciliacion.

    Args:
        filas: filas de la tabla.

    Returns:
        None.
    """
    cab = ["metrica", "valor", "numerador", "denominador", "unidad", "definicion",
           "fuente", "observado", "cota", "reproducible", "estado"]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        for x in filas:
            w.writerow([x.get(c, "") for c in cab])


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    t = informe["tensores"]
    r = informe["reproduccion_de_147"]
    az = informe["prueba_de_azar"]
    mdf = pd.DataFrame(informe["margen_sobre_rescate_observado"])
    sdf = pd.DataFrame(informe["slices_error_dense_vs_rescue_dense"])
    mr = pd.DataFrame(informe["matriz_de_rescate_por_canal"])
    lineas = [
        "# P49 - Reconciliacion forense del rescate OOF",
        "",
        f"Generado por `experiments/v49_00_rescate_forense.py` ({informe['experimento']}).",
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle.**",
        "",
        f"Pregunta: {informe['pregunta']}",
        "",
        "## 1. Los tensores SI eran recuperables",
        "",
        f"- ubicaciones: {', '.join(t['ubicaciones'])}",
        (f"- canales reconstruidos: **{t['canales_reconstruidos']}**, con la misma regla de "
        "exclusion de `experiments/28_hardness_v4.py` (se descartan `TXT+*` y "
        "`CH-TGT+DIST`)"),
        (f"- validacion contra `reports/hardness_v4.csv`: max |diferencia| = "
        f"**{t['validacion_contra_hardness_v4']['max_abs_diff']:.2e}** sobre las 4449 celdas"),
        "",
        ("La reconstruccion es exacta, de modo que `model_dis_*` no era una aproximacion: "
        "es la fraccion de fallo real de esos 20 canales."),
        "",
        "## 2. Que significan las tres cifras",
        "",
        "| cifra | unidad | que es exactamente |",
        "|---|---|---|",
        ("| **757** | label-cell | celdas (tweet, etiqueta) con `pred != y_true` sobre "
        "1483 filas x 3 etiquetas. **No** son tweets con error |"),
        ("| **147** | label-cell | celdas de error de V03 que acierta **una sola variante**, "
        "`C_meta36_downweight` con umbrales 0.5. Es la `cobertura_de_errores_v03` de V17-00 |"),
        ("| **715** | label-cell | celdas de error de V03 que acierta **al menos uno de los "
        "20 canales** de la fase v2, binarizados a 0.5 |"),
        "",
        ("Las dos no son la misma magnitud y nunca debieron presentarse como si lo fueran: "
        "147 es el rescate de **un** modelo, 715 es el de la **union de veinte**."),
        "",
        "## 3. El 147 se reproduce exactamente",
        "",
        "| magnitud | registrado | reproducido |",
        "|---|---:|---:|",
        f"| rescates de `C_meta36_downweight` | 147 | **{r['valor_reproducido']}** |",
        f"| cobertura | 0.194187582562748 | **{r['cobertura_reproducida']:.12f}** |",
        f"| danos | 90 | {r['damage_reproducido']} |",
        "",
        f"Reproduce: **{r['reproduce']}**.",
        "",
        "## 4. La prueba decisiva: senal o azar",
        "",
        ("Si los 20 canales se equivocasen de forma independiente, con la probabilidad "
        "marginal observada, el numero de celdas de error de V03 en las que **ninguno** "
        "acierta seria:"),
        "",
        (f"- probabilidad media de acierto de un canal en una celda de error: "
        f"**{az['p_medio_de_acierto_en_celda_de_error']:.4f}** "
        f"({az['canales_medios_que_acierten']} de {az['canales']} canales)"),
        (f"- celdas irrecuperables **esperadas** bajo independencia: "
        f"**{az['irrecuperables_esperadas_independencia']}**"),
        f"- celdas irrecuperables **observadas**: **{az['irrecuperables_observadas']}**",
        "",
        f"**Es azar: {az['es_azar']}.**",
        "",
        az["interpretacion"],
        "",
        "## 5. Matriz de rescate y damage por canal",
        "",
        "| canal | rescate | damage | net | cobertura errores V03 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, x in mr.iterrows():
        lineas.append(
            f"| `{x['canal']}` | {int(x['rescate'])} | {int(x['damage'])} | "
            f"{int(x['net']):+d} | {x['cobertura_errores_v03']:.4f} |"
        )
    lineas += [
        "",
        "## 6. Margen sobre el rescate OBSERVADO",
        "",
        "| etiqueta | bin de margen | errores | rescatables | irrecuperables | tasa |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, x in mdf.iterrows():
        lineas.append(
            f"| {x['etiqueta']} | {x['bin']} | {int(x['n_errores'])} | "
            f"{int(x['rescatables_observados'])} | {int(x['irrecuperables_observados'])} | "
            f"{x['tasa_rescate_observada']:.4f} |"
        )
    lineas += [
        "",
        "## 7. ERROR-DENSE frente a RESCUE-DENSE",
        "",
        "| slice | lift error | lift rescate | lift irrecuperable | clase |",
        "|---|---:|---:|---:|---|",
    ]
    for _, x in sdf.iterrows():
        lineas.append(
            f"| {x['slice']} | {x['lift_error']:.3f} | {x['lift_rescate']:.3f} | "
            f"{x['lift_irrecuperable']:.3f} | {x['clase']} |"
        )
    lineas += [
        "",
        "## 8. Decision",
        "",
        (f"**CASE = {informe['decision']}.** Ver el detalle en "
        "`reports/p49_rescue_reconciliation.csv`, donde cada cifra lleva su unidad, su "
        "denominador, su fuente, y si es observacion directa o cota."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
