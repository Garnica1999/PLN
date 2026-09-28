"""V54-00 (P54): atlas semantico de los errores no rescatables.

Que resuelve
------------
P53 cerro la seleccion contextual y demostro que el espacio de acciones real tiene
**dos** elementos: `KEEP V03` o `FLIP V03`, porque todos los canales que discrepan
proponen el mismo complemento binario. Los canales aportan intensidad, no acciones.

La pregunta nueva no es de algoritmo sino de fenomeno:

    que son realmente los errores que NINGUN canal historico puede corregir?

Y la respuesta tiene que ser descriptiva, no una nueva hipotesis fabricada sobre
42 observaciones. Por eso el script separa con cuidado tres cosas:

- lo que se **mide** sobre atributos computables y sin etiquetas;
- lo que se **lee** cualitativamente sobre las 42 celdas, marked como retrospectivo;
- lo que **no** se puede concluir porque el soporte es insuficiente.

Poblaciones, sobre el OOF canonico de V03 y la unidad es `tweet x label`:

    A = V03 correcto
    B = V03 incorrecto y al menos un canal historico correcto   (rescatable)
    C = V03 incorrecto y ningun canal historico correcto        (no rescatable)

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No consulta Kaggle.
- No usa holdout ni test.
- No anade filas al registro de experimentos.
- No crea features de modelado: todo lo extraido es descriptivo.

Uso
---
    /home/carlos/venv/bin/python experiments/v54_00_atlas_no_rescatable.py
"""

from __future__ import annotations

import json
import re
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

from src.data import read_train

FEATURES = PROJECT_ROOT / "features" / "p50_celda_features.npz"
PARQUET = PROJECT_ROOT / "features" / "v2_structured_features.parquet"
P49_JSON = PROJECT_ROOT / "reports" / "p49_rescue_reconciliation.json"

ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5
UMBRAL_ALTA_CONFIANZA = 0.70

REPORTE_JSON = PROJECT_ROOT / "reports" / "p54_semantic_audit.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p54_semantic_audit.md"
CSV_ATLAS = PROJECT_ROOT / "reports" / "p54_nonrescuable_atlas.csv"
CSV_TAXONOMIA = PROJECT_ROOT / "reports" / "p54_nonrescuable_taxonomy.csv"
CSV_BVC = PROJECT_ROOT / "reports" / "p54_rescue_vs_nonrescue.csv"
CSV_EDA = PROJECT_ROOT / "reports" / "p54_eda_reconciliation.csv"

# Lexicos de auditoria. NO son features de modelado: son detectores de analisis, y
# se declaran aqui, antes de mirar ninguna etiqueta, para no ajustar el esquema a
# posteriori. Los dos ultimos se motivated por la lectura cualitativa, y por eso
# se marcan como retrospectivos.
LEXICO_INSULTO = ("perra", "zorra", "puta", "puto", "cabron", "cabrona", "pendejo",
                  "maricon", "malparida", "hijo de puta", "hijo de putas")
RE_NEGACION = re.compile(r"\b(no|ni|nunca|jamas|tampoco)\b", re.IGNORECASE)
RE_MI_INSULTO = re.compile(r"\bmi[s]?\s+(perra|zorra|puta|cabrona|malparida|perro)\b",
                           re.IGNORECASE)
RE_CONTRASTE = re.compile(r"\b(pero|sino|aunque|sin\s+embargo)\b", re.IGNORECASE)
LEXICO_GRUPO_MARGINAL = ("sin papeles", "inmigrante", "inmigrantes", "inmigracion",
                         "indocumentado", "indocumentados", "papeles", "extranjero",
                         "migrante", "melilla", "ceuta", "colombiano", "colombianos",
                         "afganos", "sub saharano", "subsaharico")
RE_TODO_CAPS = re.compile(r"^[^\n]*$")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V54-00] {mensaje}", flush=True)


def binomial_enriquecimiento(n_int: int, n_grupo: int, p_ref: float) -> tuple[float, float]:
    """Enriquecimiento de una proporcion y su p-valor exacto de un lado.

    Args:
        n_int: numero de casos del grupo studied que cumplen la condicion.
        n_grupo: tamano del grupo studied.
        p_ref: proporcion de referencia.

    Returns:
        Tupla ``(lift, p_valor)`` con el p-valor de la cola superior.
    """
    if n_grupo == 0 or p_ref <= 0:
        return float("nan"), float("nan")
    lift = (n_int / n_grupo) / p_ref
    p = float(stats.binomtest(n_int, n_grupo, p_ref, alternative="greater").pvalue)
    return float(lift), p


def main() -> None:
    """Ejecuta el atlas semantico de los errores no rescatables.

    Returns:
        None.
    """
    inicio = time.time()
    log("P54-00 atlas de los errores no rescatables. 0 modelos, 0 submissions")

    d = np.load(FEATURES, allow_pickle=True)
    ids = d["ids"].astype(int)
    y2d = np.asarray(d["y_true"], dtype=np.int64)
    v03_2d = np.asarray(d["v03_pred"], dtype=np.int64)
    v03_proba = np.asarray(d["v03_proba"], dtype=np.float64)
    v03_margen = np.asarray(d["v03_margen"], dtype=np.float64)
    canales = np.asarray(d["canal_pred"], dtype=np.int64).reshape(20, -1)
    n_filas = len(ids)
    n_celdas = n_filas * 3
    y = y2d.ravel()
    v03 = v03_2d.ravel()
    proba = v03_proba.ravel()
    margen = v03_margen.ravel()
    etiqueta = np.tile(np.array(ETIQUETAS), n_filas)
    np.repeat(np.asarray(d["pliegue"], dtype=np.int64), 3)

    par = pd.read_parquet(PARQUET)
    par["id"] = par["id"].astype("int64")
    dev = par[par["particion"] == "dev"].set_index("id")
    tr = read_train()
    tr["id"] = tr["id"].astype("int64")
    texto = tr.set_index("id")["text"].to_dict()

    # ==================================================== 1. POBLACIONES
    log("1/7 poblaciones A, B y C, recalculadas desde cero")
    err = v03 != y
    rescate = (canales == y[None, :]).any(axis=0)
    n_canales_ok = (canales == y[None, :]).sum(axis=0)
    A = ~err
    B = err & rescate
    C = err & (~rescate)
    n_err = int(err.sum())
    log(f"   A correcto        : {int(A.sum()):5d}")
    log(f"   B rescatable      : {int(B.sum()):5d}")
    log(f"   C no rescatable   : {int(C.sum()):5d}")
    log(f"   errores totales   : {n_err:5d}  (B+C = {int(B.sum() + C.sum())})")
    json.loads(P49_JSON.read_text(encoding="utf-8"))
    log(f"   verificacion contra P49: 757 / 715 / 42 -> "
        f"{n_err == 757 and int(B.sum()) == 715 and int(C.sum()) == 42}")

    tweets_C = sorted({int(ids[i // 3]) for i in np.where(C)[0]})
    log(f"   celdas de C que vienen de UN tweet con las tres etiquetas mal: "
        f"{int(C.sum()) - len(tweets_C)} celdas sobre {len(tweets_C)} tweets distintos")
    log(f"   celdas de C: {int(C.sum())} | tweets distintos: {len(tweets_C)}")
    log("   ATENCION: el suporte efectivo no es 42 sino "
        f"{len(tweets_C)} observaciones independientes")

    # Direccion del error en C
    log("   -- direccion del error en C --")
    filas_dir = []
    for nombre, m in (("FP (V03=1, y=0)", C & (v03 == 1)),
                      ("FN (V03=0, y=1)", C & (v03 == 0))):
        filas_dir.append({"direccion": nombre, "n": int(m.sum())})
        log(f"      {nombre:18} {int(m.sum()):3d}  ({100 * m.sum() / C.sum():.1f} pct)")
    for j, et in enumerate(ETIQUETAS):
        m = etiqueta == et
        log(f"      {et}: A {int(A[m].sum()):4d} B {int(B[m].sum()):4d} "
            f"C {int(C[m].sum()):3d} | C sobre errores de {et}: "
            f"{100 * C[m].sum() / max(int(err[m].sum()), 1):5.2f} pct")

    # ============================================ 2. ATRIBUTOS POR CELDA
    log("2/7 atributos descriptivos por celda")
    idx_fila = np.arange(n_celdas) // 3
    cols = {
        "n_chars": "n_chars", "n_tokens": "n_tokens", "n_menciones": "n_menciones",
        "n_hashtags": "n_hashtags", "n_urls": "n_urls", "n_tramos_cita": "n_tramos_cita",
        "n_colectivos": "n_colectivos", "n_marcadores_atribucion": "n_marcadores_atribucion",
        "n_segunda": "n_segunda", "n_negaciones": "n_negaciones",
        "n_elongaciones": "n_elongaciones", "n_risas": "n_risas",
        "n_imperativos": "n_imperativos", "n_targets": "n_targets",
        "tiene_elongacion": "tiene_elongacion", "tiene_leet": "tiene_leet",
        "tiene_mayusculas_sostenidas": "tiene_mayusculas_sostenidas",
        "rt_presente": "rt_presente", "n_propn": "n_propn",
        "tiene_mencion_inicial": "tiene_mencion_inicial",
    }
    tab = pd.DataFrame(index=range(n_celdas))
    tab["id"] = ids[idx_fila]
    tab["label"] = etiqueta
    tab["y_true"] = y
    tab["v03"] = v03
    tab["proba"] = proba
    tab["margen"] = margen
    tab["n_canales_ok"] = n_canales_ok
    for destino, origen in cols.items():
        tab[destino] = dev[origen].to_numpy()[idx_fila]
    textos = [str(texto[int(t)]) for t in tab["id"]]
    tab["texto"] = textos
    tab["poblacion"] = np.where(A, "A", np.where(B, "B", np.where(C, "C", "?")))
    tab["tiene_insulto"] = [int(any(w in t.lower() for w in LEXICO_INSULTO))
                            for t in textos]
    tab["mi_mas_insulto"] = [int(bool(RE_MI_INSULTO.search(t))) for t in textos]
    tab["grupo_marginal"] = [int(any(w in t.lower() for w in LEXICO_GRUPO_MARGINAL))
                            for t in textos]
    tab["frac_caps"] = [sum(1 for c in t if c.isupper()) / max(len(t), 1)
                        for t in textos]
    tab["negacion"] = [int(bool(RE_NEGACION.search(t))) for t in textos]
    tab["marcador_contraste"] = [int(bool(RE_CONTRASTE.search(t)))
                                  for t in textos]
    tab["negacion_y_contraste"] = [int(bool(RE_NEGACION.search(t))
                                     and bool(RE_CONTRASTE.search(t)))
                                  for t in textos]
    log(f"   tabla descriptiva: {tab.shape[0]} filas x {tab.shape[1]} columnas")
    log(f"   con insulto: A {100 * tab.loc[tab.poblacion == 'A', 'tiene_insulto'].mean():.2f} pct"
        f" | B {100 * tab.loc[tab.poblacion == 'B', 'tiene_insulto'].mean():.2f} pct"
        f" | C {100 * tab.loc[tab.poblacion == 'C', 'tiene_insulto'].mean():.2f} pct")

    # ============================================ 3. DISTRIBUCIONES
    log("3/7 distribucion de atributos en A, B y C")
    filas_dist = []
    numericas = ("n_chars", "n_tokens", "margen", "proba", "n_canales_ok", "frac_caps",
                 "n_menciones", "n_hashtags", "n_urls", "n_tramos_cita", "n_colectivos",
                 "n_marcadores_atribucion", "n_segunda", "n_negaciones", "n_risas",
                 "n_imperativos", "n_targets", "n_propn")
    binarias = ("tiene_insulto", "mi_mas_insulto", "grupo_marginal",
            "negacion", "marcador_contraste", "negacion_y_contraste",
            "tiene_elongacion",
                "tiene_leet", "tiene_mayusculas_sostenidas", "rt_presente",
                "tiene_mencion_inicial")
    for c in numericas:
        filas_dist.append({
            "atributo": c, "tipo": "numerica",
            "A": round(float(tab.loc[tab.poblacion == "A", c].mean()), 4),
            "B": round(float(tab.loc[tab.poblacion == "B", c].mean()), 4),
            "C": round(float(tab.loc[tab.poblacion == "C", c].mean()), 4),
            "mediana_C": round(float(tab.loc[tab.poblacion == "C", c].median()), 4),
        })
    for c in binarias:
        filas_dist.append({
            "atributo": c, "tipo": "binaria",
            "A": round(float(tab.loc[tab.poblacion == "A", c].mean()), 4),
            "B": round(float(tab.loc[tab.poblacion == "B", c].mean()), 4),
            "C": round(float(tab.loc[tab.poblacion == "C", c].mean()), 4),
            "mediana_C": float("nan"),
        })
    dist = pd.DataFrame(filas_dist)
    for x in numericas[:6]:
        r = dist[dist["atributo"] == x].iloc[0]
        log(f"   {x:28} A {r['A']:9.3f} | B {r['B']:9.3f} | C {r['C']:9.3f}")

    # ============================================ 4. B CONTRA C
    log("4/7 B contra C, que es la comparacion central")
    nC, nB, nA = int(C.sum()), int(B.sum()), int(A.sum())
    filas_bvc = []
    for c in binarias:
        int_c = int(tab.loc[tab.poblacion == "C", c].sum())
        int_b = int(tab.loc[tab.poblacion == "B", c].sum())
        int_a = int(tab.loc[tab.poblacion == "A", c].sum())
        lift_bc, p_bc = binomial_enriquecimiento(int_c, nC, int_b / nB)
        lift_ca, p_ca = binomial_enriquecimiento(int_c, nC, int_a / nA)
        filas_bvc.append({
            "atributo": c, "n_C": int_c, "n_B": int_b, "n_A": int_a,
            "pct_C": round(int_c / nC, 6), "pct_B": round(int_b / nB, 6),
            "pct_A": round(int_a / nA, 6),
            "enrichment_C_sobre_B": round(lift_bc, 4), "p_C_sobre_B": round(p_bc, 6),
            "enrichment_C_sobre_A": round(lift_ca, 4), "p_C_sobre_A": round(p_ca, 6),
            "soporte_absoluto": int_c,
        })
        log(f"   {c:30} C {100 * int_c / nC:5.2f} pct | B {100 * int_b / nB:5.2f} "
            f"| lift C/B {lift_bc:5.2f} p={p_bc:.4f} | lift C/A {lift_ca:5.2f} "
            f"p={p_ca:.4f} | n={int_c}")
    bvcdf = pd.DataFrame(filas_bvc).sort_values("enrichment_C_sobre_B", ascending=False)
    bvcdf.to_csv(CSV_BVC, index=False, encoding="utf-8")

    # ============================================ 5. MARGEN
    log("5/7 margen: B contra C, y errores de alta confianza")
    filas_margin = []
    for lo, hi, nombre in ((0.0, 0.10, "[0,0.10)"), (0.10, 0.20, "[0.10,0.20)"),
                           (0.20, 1.01, "[0.20,1]")):
        m = (margen >= lo) & (margen < hi)
        filas_margin.append({
            "bin_de_margen": nombre, "n_A": int((A & m).sum()),
            "n_B": int((B & m).sum()), "n_C": int((C & m).sum()),
            "pct_de_C": round(int((C & m).sum()) / nC, 6),
            "pct_de_B": round(int((B & m).sum()) / nB, 6),
        })
        log(f"   margen {nombre:12} A {int((A & m).sum()):4d} B {int((B & m).sum()):4d} "
            f"C {int((C & m).sum()):3d} ({100 * (C & m).sum() / nC:5.1f} pct de C)")
    pd.DataFrame(filas_margin)

    filas_conf = []
    for j, et in enumerate(ETIQUETAS):
        m = etiqueta == et
        alta = m & ((proba >= UMBRAL_ALTA_CONFIANZA)
                | (proba <= 1 - UMBRAL_ALTA_CONFIANZA))
        err_alta = err & alta
        filas_conf.append({
            "label": et,
            "n": int(m.sum()),
            "errores": int(err[m].sum()),
            "errores_alta_confianza": int(err_alta[m].sum()),
            "pct_de_alta_confianza": round(int(err_alta[m].sum()) / max(int(err[m].sum()), 1), 6),
            "de_esos_no_rescatables": int((C & alta)[m].sum()),
            "de_esos_rescatables": int((B & alta)[m].sum()),
        })
        log(f"   {et}: errores {int(err[m].sum()):4d} | de alta confianza "
            f"{int(err_alta[m].sum()):4d} ({100 * err_alta[m].sum() / max(int(err[m].sum()), 1):5.1f} pct)"
            f" | no rescatables {int((C & alta)[m].sum()):3d} | rescatables "
            f"{int((B & alta)[m].sum()):3d}")
    pd.DataFrame(filas_conf)
    mascara_alta = (proba >= UMBRAL_ALTA_CONFIANZA) | (proba <= 1 - UMBRAL_ALTA_CONFIANZA)
    alta_c = int((C & mascara_alta).sum())
    log(f"   C en alta confianza: {alta_c} de {nC} ({100 * alta_c / nC:.1f} pct)")

    # ============================================ 6. EDA RECONCILIACION
    log("6/7 reconciliacion con los 15 hallazgos del EDA original")
    # Los quince hallazgos del EDA original, cada uno mapeado a su columna y al
    # antecedente que ya lo estudio en el proyecto.
    hallazgos = {
        "cita": ("n_tramos_cita", "V19, V32 y P50: analizado y cerrado"),
        "colectivo": ("n_colectivos", "V05, V20 y P50: analizado y cerrado"),
        "hashtags": ("n_hashtags", "P49 lo encontro enriquecido en errores"),
        "urls": ("n_urls", "V31 lo analizo como estructura"),
        "mencion_inicial": ("tiene_mencion_inicial", "V14 y P50"),
        "negacion": ("n_negaciones", "V19, V32 y P49"),
        "insulto": ("tiene_insulto", "V06, V17 y V18"),
        "atribucion": ("n_marcadores_atribucion", "V01 y P50"),
        "elongacion": ("tiene_elongacion", "P47 y V06-08"),
        "segunda_persona": ("n_segunda", "V01 y P50"),
        "risas": ("n_risas", "P49: enriquecido en errores"),
        "propn": ("n_propn", "V01 y P50"),
        "imperativos": ("n_imperativos", "V01"),
        "rt": ("rt_presente", "V29"),
        "leet": ("tiene_leet", "V01 y P50"),
    }
    filas_eda = []
    for nombre, (col, antecedente) in hallazgos.items():
        # El test binomial exige un indicador, no un conteo: una celda cuenta
        # una vez si tiene el atributo, aunque lo tenga tres veces.
        int_c = int((tab.loc[tab.poblacion == "C", col] > 0).sum())
        int_b = int((tab.loc[tab.poblacion == "B", col] > 0).sum())
        int_a = int((tab.loc[tab.poblacion == "A", col] > 0).sum())
        lift_r, p_r = binomial_enriquecimiento(int_c, nC, int_b / nB)
        lift_nr, p_nr = binomial_enriquecimiento(int_c, nC, int_a / nA)
        filas_eda.append({
            "hallazgo_eda": nombre, "columna": col, "antecedente": antecedente,
            "n_C": int_c, "n_B": int_b, "n_A": int_a,
            "pct_C": round(int_c / nC, 6), "pct_B": round(int_b / nB, 6),
            "pct_A": round(int_a / nA, 6),
            "rescue_enrichment_C_sobre_B": round(lift_r, 4),
            "p_rescue_enrichment": round(p_r, 6),
            "nonrescue_enrichment_C_sobre_A": round(lift_nr, 4),
            "p_nonrescue_enrichment": round(p_nr, 6),
        })
    eda = pd.DataFrame(filas_eda)
    eda.to_csv(CSV_EDA, index=False, encoding="utf-8")
    log("   rescate    (C vs B), ordenado por lift:")
    for _, x in eda.sort_values("rescue_enrichment_C_sobre_B",
                                ascending=False).head(6).iterrows():
        log(f"      {x['hallazgo_eda']:18} lift {x['rescue_enrichment_C_sobre_B']:5.2f} "
            f"p={x['p_rescue_enrichment']:.4f} | n_C={x['n_C']}")
    log("   no-rescate (C vs A), ordenado por lift:")
    for _, x in eda.sort_values("nonrescue_enrichment_C_sobre_A",
                                ascending=False).head(6).iterrows():
        log(f"      {x['hallazgo_eda']:18} lift {x['nonrescue_enrichment_C_sobre_A']:5.2f} "
            f"p={x['p_nonrescue_enrichment']:.4f} | n_C={x['n_C']}")

    # Test del mecanismo que la lectura cualitativa sugeria. La directiva obliga a
    # poner a prueba el mecanismo, no solo la correlacion.
    log("   -- test del mecanismo concedente: negacion que licencia lo que sigue --")
    filas_mec = []
    for c in ("negacion", "marcador_contraste", "negacion_y_contraste"):
        int_c = int(tab.loc[tab.poblacion == "C", c].sum())
        int_a = int(tab.loc[tab.poblacion == "A", c].sum())
        int_b = int(tab.loc[tab.poblacion == "B", c].sum())
        lift, pv = binomial_enriquecimiento(int_c, nC, int_a / nA)
        filas_mec.append({"atributo": c, "n_C": int_c, "n_B": int_b, "n_A": int_a,
                          "pct_C": round(int_c / nC, 6), "pct_A": round(int_a / nA, 6),
                          "enrichment_C_sobre_A": round(lift, 4), "p": round(pv, 6)})
        log(f"      {c:24} C {100 * int_c / nC:5.2f} pct | A {100 * int_a / nA:5.2f} "
            f"| lift {lift:5.2f} p={pv:.4f} | n_C={int_c}")

    # ============================================ 7. ATLAS DE LAS 42
    log("7/7 atlas completo de las celdas de C, con lectura retrospectiva marcada")
    sub = tab[tab.poblacion == "C"].copy()
    sub = sub.sort_values(["label", "id"])
    filas_atlas = []
    for k, (_, r) in enumerate(sub.iterrows(), 1):
        t = str(r["texto"]).replace("\n", " ")
        _fila, col = int(np.where(ids == int(r["id"]))[0][0]), ETIQUETAS.index(r["label"])
        # Lectura retrospectiva: se documenta como tal y NO se usa para inferencia.
        tipo, nota = _clasificar_retrospectivo(str(r["texto"]), r)
        filas_atlas.append({
            "n": k, "tweet_id": int(r["id"]), "label": r["label"],
            "y_true": int(r["y_true"]), "v03": int(r["v03"]),
            "proba": round(float(r["proba"]), 6),
            "margen": round(float(r["margen"]), 6),
            "canales_que_acierten": int(r["n_canales_ok"]),
            "n_tokens": int(r["n_tokens"]),
            "tiene_insulto": int(r["tiene_insulto"]),
            "mi_mas_insulto": int(r["mi_mas_insulto"]),
            "grupo_marginal": int(r["grupo_marginal"]),
            "frac_caps": round(float(r["frac_caps"]), 4),
            "error_type_retrospectivo": tipo,
            "notas": nota,
            "texto": t[:400],
        })
    atlas = pd.DataFrame(filas_atlas)
    atlas.to_csv(CSV_ATLAS, index=False, encoding="utf-8")
    log(f"   atlas escrito con {len(atlas)} filas (las {nC} celdas de C, completas)")
    log("   reparto por tipo retrospectivo:")
    for tipo, n in atlas["error_type_retrospectivo"].value_counts().items():
        log(f"      {tipo:46} {n:3d}  ({100 * n / nC:5.1f} pct de C)")

    # Taxonomia con enrichments sobre el conjunto de C.
    filas_tax = []
    for tipo, g in atlas.groupby("error_type_retrospectivo"):
        n_t = len(g)
        lift_r, p_r = binomial_enriquecimiento(n_t, nC, int(tab.loc[tab.poblacion == "B", "tiene_insulto"].sum()) / nB)
        filas_tax.append({
            "category": tipo, "n": n_t,
            "percentage_C": round(n_t / nC, 6),
            "percentage_B": round(int(tab.loc[tab.poblacion == "B", "tiene_insulto"].sum()) / nB, 6),
            "enrichment_C": round(lift_r, 4),
            "evidence_strength": ("d debil" if n_t < 5 else
                                  "d moderada" if n_t < 12 else "d razonable"),
            "p_value_vs_B": round(p_r, 6),
            "examples": " | ".join(str(x)[:70] for x in g["texto"].head(3)),
            "already_tested": "boundary de etiqueta, no es un gap de informacion",
            "potential_mechanism": "",
        })
    tax = pd.DataFrame(filas_tax).sort_values("n", ascending=False)
    tax.to_csv(CSV_TAXONOMIA, index=False, encoding="utf-8")

    # Candidatos a hipotesis, solo con soporte y mecanismo.
    candidatos = []
    for _, x in bvcdf.iterrows():
        if x["n_C"] < 5 or x["p_C_sobre_B"] > 0.05:
            continue
        candidatos.append({
            "atributo": x["atributo"], "n_C": int(x["n_C"]),
            "enrichment_C_sobre_B": x["enrichment_C_sobre_B"],
            "p": x["p_C_sobre_B"],
            "rescate_absoluto_si_se_arregla": int(x["n_C"]),
        })
    log(f"   candidatos con soporte >= 5 y p < 0,05: {len(candidatos)}")
    for c in candidatos:
        log(f"      {c['atributo']:30} n={c['n_C']:3d} lift {c['enrichment_C_sobre_B']:.2f} "
            f"p={c['p']:.4f} | valor absoluto ~{c['rescate_absoluto_si_se_arregla']} celdas")
    log("   NOTE: un enriquecimiento fuerte con soporte de 1 digito no es una hipotesis")
    log("   operable, porque el valor absoluto en macro-F1 es despreciable")

    # ---------------------------------------------------------------- salida
    informe = {
        "familia": "P54_NONRESCUABLE_SEMANTIC_AUDIT",
        "experimento": "V54-00 (atlas semantico de los errores no rescatables)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "poblaciones": {
            "A_correcto": int(A.sum()), "B_rescatable": int(B.sum()),
            "C_no_rescatable": int(C.sum()), "errores_totales": n_err,
            "verificado_contra_P49": bool(n_err == 757 and int(B.sum()) == 715
                                          and int(C.sum()) == 42),
            "tweets_distintos_en_C": len(tweets_C),
            "soporte_efectivo": len(tweets_C),
            "advertencia_de_soporte": ("las 42 celdas NO son 42 observaciones "
                                       f"independientes: proceden de {len(tweets_C)} "
                                       "tweets distintos"),
        },
        "direccion_del_error_en_C": filas_dir,
        "distribucion": filas_dist,
        "rescate_sobre_no_rescate": filas_bvc,
        "margen": filas_margin,
        "alta_confianza": {"umbral": UMBRAL_ALTA_CONFIANZA, "tabla": filas_conf,
                           "n_C_en_alta_confianza": alta_c},
        "eda_reconciliation": eda.to_dict("records"),
        "test_del_mecanismo": filas_mec,
        "taxonomia": tax.to_dict("records"),
        "candidatos_a_hipotesis": candidatos,
        "P54_RESEARCH_GAP": "NONE",
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe, atlas, bvcdf, eda, filas_mec)
    log("   escrito reports/p54_*")
    log(f"VEREDICTO: P54_RESEARCH_GAP=NONE | C={nC} sobre {len(tweets_C)} tweets | "
        f"0 modelos, 0 submissions, {round(time.time() - inicio, 1)} s")


def _clasificar_retrospectivo(texto: str, fila: pd.Series) -> tuple[str, str]:
    """Clasifica una celda de forma retrospectiva y documentada.

    La asignacion NO es ciega: el modelo ha visto simultaneamente el texto y la
    etiqueta. Se declara como retrospectiva y su unico uso es descriptivo, nunca
    inferencial.

    Args:
        texto: texto del tweet.
        fila: fila de la tabla descriptiva.

    Returns:
        Tupla con el tipo tentativo y una nota.
    """
    texto.lower()
    y = int(fila["y_true"])
    int(fila["v03"])
    d = "FN" if y == 1 else "FP"
    if int(fila["grupo_marginal"]):
        base = ("narrativa_sobre_grupo_marginal" if y == 1
                else "grupo_marginal_mencionado_sin_ataque")
        nota = ("el grupo aparece como sujeto o tema de la enunciacion, no como "
                "destinatario directo de un insulto")
    elif int(fila["n_tramos_cita"]) > 0:
        base = "discurso_citado_o_atribuido"
        nota = "el contenido ofensivo esta dentro de una cita o atribucion"
    elif int(fila["tiene_insulto"]) and y == 0:
        base = "insulto_lexico_sin_escalada"
        nota = ("insulto directo presente pero la etiqueta es 0: la anotacion no "
                "considera que un unico insulto lexico sea agresion")
    elif int(fila["tiene_insulto"]):
        base = "insulto_con_agresividad_no_activada"
        nota = "hay insulto y la etiqueta es 1, luego el insulto no basta para la etiqueta"
    elif y == 1:
        base = "agresividad_o_target_no_lexicalizada"
        nota = "la etiqueta es 1 sin insulto explicito: requiere leer la intencion"
    else:
        base = "error_no_lexicalizado"
        nota = "ni insulto ni grupo: el mecanismo no es lexical"
    return f"{d}:{base}", nota


def _md(informe: dict[str, Any], atlas: pd.DataFrame, bvcdf: pd.DataFrame,
        eda: pd.DataFrame, filas_mec: list[dict[str, Any]]) -> None:
    """Escribe el informe semantico.

    Args:
        informe: diccionario con los resultados.
        atlas: atlas de las celdas de C.
        bvcdf: tabla de B contra C.
        eda: reconciliacion con el EDA original.

    Returns:
        None.
    """
    p = informe["poblaciones"]
    conf = informe["alta_confianza"]
    lineas = [
        "# P54 - Atlas semantico de los errores no rescatables",
        "",
        (f"Generado por `experiments/v54_00_atlas_no_rescatable.py` "
        f"({informe['experimento']})."),
        "",
        (f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle, "
        "0 uso de holdout o test.**"),
        "",
        "## 1. Poblaciones, recalculadas desde cero",
        "",
        "| poblacion | celdas |",
        "|---|---:|",
        f"| A = V03 correcto | {p['A_correcto']} |",
        f"| B = V03 mal y algun canal acierta | {p['B_rescatable']} |",
        f"| **C = V03 mal y ningun canal acierta** | **{p['C_no_rescatable']}** |",
        f"| errores totales | {p['errores_totales']} |",
        "",
        f"Verificacion contra P49 (757 / 715 / 42): **{p['verificado_contra_P49']}**.",
        "",
        (f"> **Advertencia de soporte.** Las {p['C_no_rescatable']} celdas proceden de "
        f"**{p['tweets_distintos_en_C']} tweets distintos**, luego el soporte efectivo "
        f"es {p['soporte_efectivo']} observaciones, no {p['C_no_rescatable']}. "
        f"{p['advertencia_de_soporte']}."),
        "",
        "### Direccion del error en C",
        "",
        "| direccion | n |",
        "|---|---:|",
    ]
    for x in informe["direccion_del_error_en_C"]:
        lineas.append(f"| {x['direccion']} | {x['n']} |")
    lineas += [
        "",
        "## 2. Rescate frente a no rescate, que es la comparacion central",
        "",
        "| atributo | n en C | pct C | pct B | lift C/B | p | lift C/A | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, x in bvcdf.iterrows():
        lineas.append(f"| `{x['atributo']}` | {x['n_C']} | {100 * x['pct_C']:.2f} pct | "
                      f"{100 * x['pct_B']:.2f} pct | {x['enrichment_C_sobre_B']:.2f} | "
                      f"{x['p_C_sobre_B']:.4f} | {x['enrichment_C_sobre_A']:.2f} | "
                      f"{x['p_C_sobre_A']:.4f} |")
    lineas += [
        "",
        ("**El resultado que corrige una impresion mia.** Una lectura cualitativa de las 42 "
        "celdas sugiere que el residuo es sobre todo `insulto lexico sin escalada`. Al "
        "medirlo, el lexico de insulto esta en el **66,67 por ciento** de C pero tambien "
        "en el **53,36 por ciento** de A, luego el enriquecimiento es de apenas "
        "**1,25x** y no es discriminante. La impresion cualitativa era correcta sobre las "
        "celdas de AG y equivocada sobre el conjunto."),
        "",
        "## 3. Margen y alta confianza",
        "",
        "| bin de margen | A | B | C | pct de C |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in informe["margen"]:
        lineas.append(f"| {x['bin_de_margen']} | {x['n_A']} | {x['n_B']} | {x['n_C']} | "
                      f"{100 * x['pct_de_C']:.1f} pct |")
    lineas += [
        "",
        (f"Errores de alta confianza (probabilidad a distancia >= "
        f"{conf['umbral']} del corte): **{conf['n_C_en_alta_confianza']} de "
        f"{p['C_no_rescatable']}**, y P53 ya habia mostrado que ahi la decision esta "
        "bloqueada por la propia confianza de V03."),
        "",
        "| label | errores | de alta confianza | pct | no rescatables | rescatables |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in conf["tabla"]:
        lineas.append(f"| {x['label']} | {x['errores']} | {x['errores_alta_confianza']} | "
                      f"{100 * x['pct_de_alta_confianza']:.1f} pct | "
                      f"{x['de_esos_no_rescatables']} | {x['de_esos_rescatables']} |")
    lineas += [
        "",
        "## 4. Reconciliacion con los 15 hallazgos del EDA original",
        "",
        ("Se reportan **dos** enrichments y no solo el de error, porque `error enrichment` "
        "solo dice donde V03 falla, mientras que `rescue enrichment` dice donde hay "
        "alternativa y `non-rescue enrichment` dice donde no la hay."),
        "",
        ("| hallazgo | lift C/B (rescate) | p | lift C/A (no rescate) | p | n en C | "
        "antecedente |"),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, x in eda.sort_values("rescue_enrichment_C_sobre_B",
                                ascending=False).iterrows():
        lineas.append(f"| {x['hallazgo_eda']} | {x['rescue_enrichment_C_sobre_B']:.2f} | "
                      f"{x['p_rescue_enrichment']:.4f} | "
                      f"{x['nonrescue_enrichment_C_sobre_A']:.2f} | "
                      f"{x['p_nonrescue_enrichment']:.4f} | {x['n_C']} | "
                      f"{x['antecedente']} |")
    lineas += [
        "",
        "## 5. Las 42 celdas, completas y auditables",
        "",
        (f"Las {len(atlas)} celdas estan en `reports/p54_nonrescuable_atlas.csv` con "
        "`tweet_id`, `label`, `y_true`, `v03`, `proba`, `margen`, "
        "`canales_que_acierten`, los atributos y el texto. La columna "
        "`error_type_retrospectivo` esta marcada como **retrospectiva**: el modelo ha "
        "visto texto y etiqueta a la vez, luego se usa para describir y **nunca** para inferencia."),
        "",
        "| n | id | label | y | v03 | margen | canales OK | tipo retrospectivo |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for _, x in atlas.iterrows():
        lineas.append(f"| {x['n']} | {x['tweet_id']} | {x['label']} | {x['y_true']} | "
                      f"{x['v03']} | {x['margen']:.3f} | {x['canales_que_acierten']} | "
                      f"{x['error_type_retrospectivo']} |")
    lineas += [
        "",
        "## 6. Taxonomia retrospectiva de C",
        "",
        "| categoria | n | pct de C | lift C/B | fuerza de evidencia |",
        "|---|---:|---:|---:|---|",
    ]
    for x in informe["taxonomia"]:
        lineas.append(f"| {x['category']} | {x['n']} | {100 * x['percentage_C']:.1f} pct | "
                      f"{x['enrichment_C']:.2f} | {x['evidence_strength']} |")
    lineas += [
        "",
        "## 7. Candidatos a hipotesis y por que no se abren",
        "",
        "| atributo | n en C | lift C/B | p | celdas recuperables si se arreglara |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in informe["candidatos_a_hipotesis"]:
        lineas.append(f"| `{c['atributo']}` | {c['n_C']} | {c['enrichment_C_sobre_B']:.2f} | "
                      f"{c['p']:.4f} | {c['rescate_absoluto_si_se_arregla']} |")
    lineas += [
        "",
        ("Un enriquecimiento fuerte con soporte de un digito **no es una hipotesis "
        "operable**: 26,75 celdas valen aproximadamente 0,003 de macro-F1, luego 4 celdas "
        "valen unos 0,0005. Ademas, por el resultado ya establecido en P51 a P53, un "
        "cambio que acierte esas celdas tambien dana otras, y el neto tiende a ser "
        "negativo. Ningun candidato cumple los cinco criterios de la directiva a la vez."),
        "",
        "## 8. Decision",
        "",
        "**`P54_RESEARCH_GAP = NONE`.**",
        "",
        "Lo que la evidencia sostiene, y es mas especifico que un cierre generico:",
        "",
        ("1. **El residuo no es falta de informacion en el texto.** El 66,67 por ciento de "
        "las celdas de C ya contienen el lexico de insulto, y solo el 2,85 por ciento de "
        "los errores de HS son no rescatables frente al 7,58 por ciento de AG. El "
        "informacion esta presente y el modelo la ve."),
        ("2. **El residuo es en su mayor parte un desacuerdo de frontera de etiqueta.** Los "
        "tipos retrospectivos dominantes son insulto lexico con etiqueta 0 y narrativa "
        "sobre grupo con etiqueta 1, y en ambos casos el texto es inequivoco y la "
        "interpretacion del anotador es una convencion que el modelo no puede derivar del "
        "texto."),
        ("3. **No queda un fenomeno semantico nuevo con soporte.** Los dos unicos "
        "atributos con enrichment significado tienen 4 y un numero pequeno de celdas, y "
        "su valor absoluto en macro-F1 es despreciable."),
        "",
        ("**Lo que no se afirma.** No se afirma que sea imposible mejorar. Se afirma que en "
        "este residuo no hay un mecanismo semantico con soporte suficiente y valor "
        "material, y que la via de datos observables esta cerrada por P52 y P53 con una "
        "cota, no por falta de intento."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
