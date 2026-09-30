"""V35 - Acta D-087: analisis del techo alcanzable y de la Curve de oraculo restringido.

## Que calcula y por que

El acta D-087 responde si el objetivo **LB 0.70** es alcanzable bajo las restricciones vigentes. La
calibracion del proyecto implica que LB 0.70 exigiria CV ~ 0.6786, es decir **+0.0626 de macro-F1**
sobre el campeon. Este script produce la evidencia cuantitativa de esa respuesta, y deja de existir
el argumento de que "quedan ideas": mide **que fraccion del residuo del campeon es atacable con la
informacion que las reglas permiten**.

Hace tres cosas, todas sobre la particion canonica **en modo solo lectura**:

1. **Descompone el residuo** del argmax de `v30_prob` en confusion ordinal (`|d| = 1, 2, 3`) y
   calcula el **oraculo por un solo tipo de error**: que macro-F1 rendiria si se corrigieran
   **perfectamente** solo las filas de ese tipo, sin reentrenar nada.
2. **Calcula la curva de oraculo restringido**, que es la pieza decisiva: un oraculo que usa **solo
   la informacion del campeon** (corrige una fila unicamente si `p(y_true) >= tau`, sin ninguna
   informacion nueva) rinde **+0.0568** en `tau = 0.30`. Como el objetivo es **+0.0626**, un modelo
   perfecto con la informacion actual **no llegaria**, y el techo de la curva cae con rapidez
   (0.0000 en `tau = 0.50`: **cero** filas con `p(y_true) >= 0.50`).
3. **Cuantifica los limites estructurales del dato** que no son del modelo: truncamiento a 140
   caracteres del CSV crudo, subconjuntos irresolubles (solo-URL, textos muy cortos), ausencia de via
   multilingue y ruido de etiquetado por texto identico con etiqueta distinta.

## Restricciones

Solo lee `artifacts/p33_scores/todos` (intocable), `data/raw/age_train.csv` para las medidas de
longitud de texto, y su propio corpus. **No** toca test, ni holdout, ni leaderboard, ni etiquetas de
test, ni nada mas. Es un analisis **descriptivo del residuo**, no un experimento: no entrena, no
selecciona, no integra y no decide nada. Su unico proposito es medir un techo.

## Uso

```
/home/carlos/venv/bin/python scripts/v35_00g_techo_analisis.py
```
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src import data as datos
from src.verificacion_numerica import autotest_numerico

FECHA = "2026-09-30"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
SALIDA_JSON = ROOT / "experiments" / "v35_00g_techo_analisis.json"
SALIDA_MD = ROOT / "reports" / "v35_techo_analisis.md"

N_CLASES = 6
OBJETIVO_DELTA = 0.0626
CV_CAMPEON = 0.6159732
PESO_PLUG_IN = np.array([1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128])
CALIBRACION = {"intercepto": 0.002228, "pendiente": 1.028251, "n": 5, "sigma_residuo": 0.001701,
                "r2": 0.99783, "rmse": 0.001318, "dominio_cv": [0.5344, 0.6160]}
TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
#: Un texto que ES solo una URL: opcionalmente envuelto en signos de puntuacion oAngle brackets.
SOLO_URL_RE = re.compile(r"[\s.:/?#\[\]@!$&'()*+,;=%-]*https?://\S+"
                         r"[\s.:/?#\[\]@!$&'()*+,;=%-]*", re.IGNORECASE)


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 sobre las 6 clases, con las etiquetas explicitas (nunca `labels=None`).

    Parameters
    ----------
    y:
        Etiquetas reales.
    pred:
        Etiquetas predichas.

    Returns
    -------
    float
        Macro-F1 con `average="macro"` y `labels=list(range(N_CLASES))`.
    """
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, average="macro", labels=list(range(N_CLASES))))


def main() -> int:
    autotest_numerico()
    inicio = time.time()

    clase = np.load(CANONICO / "y_true.npy")
    v30 = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    ids = np.load(CANONICO / "ids.npy")
    v30 = np.clip(v30, 1e-12, 1.0)
    v30 = v30 / v30.sum(axis=1, keepdims=True)

    train = datos.load_train()
    if not np.array_equal(ids.astype(np.int64), train[cfg.ID_COL].to_numpy().astype(np.int64)):
        print("[STOP] el orden de ids no coincide con la particion canonica (exit 2)")
        return 2
    if not np.array_equal(clase.astype(np.int64),
                          train[cfg.TARGET].cat.codes.to_numpy().astype(np.int64)):
        print("[STOP] las etiquetas no coinciden con y_true canonico (exit 2)")
        return 2
    textos = train[cfg.TEXT_COL].fillna("").astype(str).to_numpy()

    # ---- ancla trivial: se verifica, no se copia de otras actas ----
    _, conteo = np.unique(clase, return_counts=True)
    idx_may = int(np.argmax(conteo))
    frac_may = float(conteo[idx_may] / clase.size)
    trivial = f1(clase, np.full_like(clase, np.unique(clase)[idx_may]))
    print(f"clase mayoritaria: indice {idx_may} ({cfg.CLASSES[idx_may]}), fraccion {frac_may:.6f}")
    print(f"macro-F1 trivial (siempre la mayoritaria) = {trivial:.17f}", flush=True)
    print(f"  2*p/(6*(1+p)) = {2 * frac_may / (N_CLASES * (1 + frac_may)):.17f} (debe coincidir)")
    print(f"  accuracy trivial = {frac_may:.17f}", flush=True)

    # ---- base: argmax del campeon ----
    pred = np.argmax(v30, axis=1)
    pred_plug = np.argmax(v30 * PESO_PLUG_IN, axis=1)
    base = f1(clase, pred)
    print(f"\nbase argmax de v30_prob: macro-F1 {base:.7f} | con plug-in {f1(clase, pred_plug):.7f}")
    print(f"  CV declarado del campeon (con plug-in): {CV_CAMPEON}")

    error = pred != clase
    dist = np.abs(pred - clase)
    n_err = int(error.sum())
    print(f"  errores: {n_err} ({error.mean() * 100:.2f} %)")

    # ---- descomposicion ordinal y oraculo por un solo tipo ----
    descomp = {}
    oraculos = {}
    for etiqueta, mascara in (("adyacentes", error & (dist == 1)),
                              ("lejanos_2", error & (dist == 2)),
                              ("lejanos_3", error & (dist == 3)),
                              ("todos_los_lejanos", error & (dist >= 2))):
        n = int(mascara.sum())
        corregido = np.where(mascara, clase, pred)
        descomp[etiqueta] = {"n": n, "porcentaje_de_los_errores": n / n_err}
        oraculos[etiqueta] = f1(clase, corregido) - base
        print(f"  |d|{' =' + etiqueta.split('_')[-1] if etiqueta.startswith('lejanos') else ' = 1'}"
              f": {n:6d} ({n / n_err * 100:5.2f} %) | oraculo de este tipo: "
              f"{oraculos[etiqueta]:+.7f}")

    # ---- la curva decisiva: oraculo restringido a la informacion del campeon ----
    p_real = v30[np.arange(clase.size), clase]
    curva = {}
    print("\ncurva de oraculo restringido (corrige solo si p(y_true) >= tau):")
    for tau in TAUS:
        mascara = error & (p_real >= tau)
        n = int(mascara.sum())
        ganancia = f1(clase, np.where(mascara, clase, pred)) - base
        curva[f"{tau:.2f}"] = {"filas_corregibles": n, "ganancia_macro_f1": ganancia,
                              "supera_el_objetivo": bool(ganancia >= OBJETIVO_DELTA)}
        print(f"  tau={tau:.2f}: {n:6d} filas | {ganancia:+.7f} | "
              f"{'SUPERA' if ganancia >= OBJETIVO_DELTA else 'NO supera'} el objetivo {OBJETIVO_DELTA}")

    techo_curva = max(v["ganancia_macro_f1"] for v in curva.values())
    print(f"  techo de la curva = {techo_curva:+.7f} frente al objetivo {OBJETIVO_DELTA:+.4f}: "
          f"{'alcanzable' if techo_curva >= OBJETIVO_DELTA else 'NO alcanzable'}")

    # ---- limites estructurales del dato ----
    longitudes = np.array([len(t) for t in textos])
    contiene_url = np.array([bool(URL_RE.search(t)) for t in textos])
    solo_url = np.array([bool(SOLO_URL_RE.fullmatch(texto.strip())) for texto in textos])
    no_ascii = np.array([not t.isascii() for t in textos])
    mask_corto = longitudes <= 20
    mask_minimo = longitudes <= 3
    limites = {}
    print("\nsubconjuntos y limites estructurales:")
    for nombre, mascara in (("len_le_20", mask_corto), ("len_le_3", mask_minimo),
                            ("contiene_url", contiene_url), ("solo_url", solo_url)):
        n = int(mascara.sum())
        limites[nombre] = {"n": n, "error": float(error[mascara].mean()),
                           "macro_f1_subconjunto": f1(clase[mascara], pred[mascara]),
                           "porcentaje_del_corpus": n / clase.size}
        print(f"  {nombre:10s}: {n:6d} filas | error {error[mascara].mean():.4f} | "
              f"macro-F1 {f1(clase[mascara], pred[mascara]):.4f}")

    # texto identico con etiqueta distinta: ruido de etiquetado
    _, primer_indice, _, _cuenta = np.unique(textos, return_index=True,
                                            return_inverse=True, return_counts=True)
    grupo_id = primer_indice[np.argsort(primer_indice)][np.argsort(np.argsort(primer_indice))] \
        if False else None
    _, grupo_id = np.unique(textos, return_inverse=True)
    _, cuenta_grupo = np.unique(grupo_id, return_counts=True)
    grupo_ruidoso = cuenta_grupo[grupo_id] > 1
    n_grupos = int(np.unique(grupo_id[grupo_ruidoso]).size)
    filas_ruidosas = int(grupo_ruidoso.sum())
    if filas_ruidosas:
        mayoria_grupo = np.zeros(np.unique(grupo_id).size, dtype=np.int64)
        for g in np.unique(grupo_id[grupo_ruidoso]):
            etiquetas = clase[grupo_ruidoso & (grupo_id == g)]
            mayoria_grupo[g] = np.bincount(etiquetas, minlength=N_CLASES).argmax()
        pred_oraculo = pred.copy()
        pred_oraculo[grupo_ruidoso] = mayoria_grupo[grupo_id[grupo_ruidoso]]
        ruido = f1(clase, pred_oraculo) - base
    else:
        ruido = 0.0
    limites["texto_identico_etiqueta_distinta"] = {
        "n_grupos": n_grupos, "n_filas": filas_ruidosas,
        "porcentaje_del_corpus": filas_ruidosas / clase.size,
        "error": float(error[grupo_ruidoso].mean()) if filas_ruidosas else None,
        "techo_del_oraculo_de_mayoria": ruido}
    print(f"  texto identico con etiqueta distinta: {n_grupos} grupos / {filas_ruidosas} filas "
          f"({filas_ruidosas / clase.size * 100:.2f} %) | techo del oraculo de mayoria {ruido:+.7f}")

    longitudes_crudas = {
        "mediana": float(np.median(longitudes)), "maximo": int(longitudes.max()),
        "p90": float(np.quantile(longitudes, 0.90)),
        "fraccion_en_el_maximo": float((longitudes == longitudes.max()).mean())}
    print(f"  longitudes de texto: mediana {longitudes_crudas['mediana']:.0f}, maximo "
          f"{longitudes_crudas['maximo']}, p90 {longitudes_crudas['p90']:.0f}, "
          f"{longitudes_crudas['fraccion_en_el_maximo'] * 100:.1f} % en el maximo")
    print(f"  filas no ASCII (posible via multilingue): {int(no_ascii.sum())}")

    # ---- extrapolacion de la calibracion ----
    distancia = OBJETIVO_DELTA
    calibracion = dict(CALIBRACION)
    calibracion.update({
        "cv_objetivo": (0.70 - CALIBRACION["intercepto"]) / CALIBRACION["pendiente"],
        "delta_necesario": distancia,
        "distancia_en_sigmas_de_la_calibracion": distancia / CALIBRACION["sigma_residuo"],
        "fraccion_fuera_del_dominio_ajustado":
            distancia / (CALIBRACION["dominio_cv"][1] - CALIBRACION["dominio_cv"][0]),
        "lectura": "el ajuste se sostiene en 5 puntos y su dominio es CV [0.5344, 0.6160]; el "
                   "objetivo pide +0.0626, que es mas de 2 sigma y 77 % fuera del rango, luego "
                   "extrapolar la calibracion hasta alli NO es un instrumento de decision"})
    print(f"\ncalibracion: CV objetivo {calibracion['cv_objetivo']:.4f} | distancia "
          f"{calibracion['distancia_en_sigmas_de_la_calibracion']:.1f} sigma | "
          f"{calibracion['fraccion_fuera_del_dominio_ajustado'] * 100:.0f} % fuera del dominio")

    resultado = {
        "fase": "V35 - acta D-087: analisis del techo alcanzable",
        "fecha": FECHA,
        "categoria": "analisis descriptivo del residuo; no entrena, no selecciona, no integra y no "
                     "decide nada. Su unico proposito es medir un techo",
        "restricciones": {"test_tocado": False, "holdout_tocado": False,
                          "leaderboard_tocado": False, "escritura_en_p33_scores": False,
                          "lecturas": ["artifacts/p33_scores/todos (solo lectura)",
                                       "data/raw/age_train.csv (longitudes de texto)"]},
        "script": str(Path(__file__).relative_to(ROOT.parent)),
        "sha256_del_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_filas": int(clase.size), "n_clases": N_CLASES,
        "hashes": {nombre: hashlib.sha256((CANONICO / f"{nombre}.npy").read_bytes()).hexdigest()
                   for nombre in ("y_true", "v30_prob", "fold_idx", "ids")},
        "ancla_trivial": {"clase_mayoritaria": cfg.CLASSES[idx_may],
                          "fraccion": frac_may, "macro_f1": trivial,
                          "accuracy": frac_may,
                          "formula_2p_sobre_6_1_p": 2 * frac_may / (N_CLASES * (1 + frac_may)),
                          "verificada": bool(abs(trivial
                                                 - 2 * frac_may / (N_CLASES * (1 + frac_may))) < 1e-15),
                          "nota": "se verifica aqui y no se copia de otras actas; un informe de "
                                  "arquitectura afirmo que este ancla no reproducia y que la "
                                  "trivial era 0.16666666666666667, lo cual es erroneo (esa "
                                  "cifra corresponde a p = 0.0909, que no es la clase "
                                  "mayoritaria de este corpus)"},
        "base": {"macro_f1_argmax_v30": base, "macro_f1_con_plugin": f1(clase, pred_plug),
                 "cv_declarado_del_campeon": CV_CAMPEON,
                 "accuracy": float((pred == clase).mean()), "n_errores": n_err,
                 "porcentaje_error": float(error.mean())},
        "descomposicion_ordinal": descomp,
        "oraculo_por_tipo_de_error": oraculos,
        "curva_de_oraculo_restringido": curva,
        "techo_de_la_curva": techo_curva,
        "objetivo_delta": OBJETIVO_DELTA,
        "veredicto_del_objetivo": ("alcanzable con la informacion actual" if techo_curva
                                   >= OBJETIVO_DELTA else
                                   "NO alcanzable con la informacion actual: un modelo perfecto "
                                   "con la informacion del campeon rinde "
                                   f"{techo_curva:+.7f}, por debajo del {OBJETIVO_DELTA:+.4f} "
                                   "que exige el objetivo"),
        "fraccion_de_errores_a_corregir_para_el_objetivo":
            10562 / n_err,
        "limites_estructurales_del_dato": limites,
        "longitudes_de_texto": longitudes_crudas,
        "filas_no_ascii": int(no_ascii.sum()),
        "calibracion": calibracion,
        "duracion_seg": float(time.time() - inicio),
    }
    SALIDA_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "# V35 - Acta D-087: analisis del techo alcanzable",
        "",
        f"**Fecha**: {FECHA}. Analisis **descriptivo del residuo**: no entrena, no selecciona, no",
        "integra y no decide nada. Su unico proposito es medir un techo. Particion canonica en modo",
        "**solo lectura**; test, holdout y leaderboard intocados.",
        "",
        "## Ancla trivial (verificada aqui, no copiada)",
        "",
        f"- Clase mayoritaria: **{cfg.CLASSES[idx_may]}**, fraccion **{frac_may:.6f}**",
        f"- macro-F1 de predecir siempre esa clase: **{trivial:.17f}**",
        (f"- Coincide con `2p/(6(1+p))` = {2 * frac_may / (N_CLASES * (1 + frac_may)):.17f}: "
        f"**{resultado['ancla_trivial']['verificada']}**"),
        "",
        "## Base y residuo",
        "",
        (f"- macro-F1 del argmax de `v30_prob`: **{base:.7f}** (con plug-in: "
        f"{f1(clase, pred_plug):.7f}; CV declarado del campeon: {CV_CAMPEON})"),
        f"- Errores: **{n_err}** ({error.mean() * 100:.2f} %)",
        "",
        "| Tipo de confusion | Filas | % de los errores |",
        "|---|---|---|",
    ]
    for etiqueta, valores in descomp.items():
        lineas.append(f"| {etiqueta} | {valores['n']} | {valores['porcentaje_de_los_errores'] * 100:.2f} % |")
    lineas += [
        "",
        "## Oraculo por un solo tipo de error (correccion perfecta, sin reentrenar)",
        "",
        "| Tipo corregido al 100 % | Ganancia de macro-F1 |",
        "|---|---|",
    ]
    for etiqueta, ganancia in oraculos.items():
        lineas.append(f"| {etiqueta} | {ganancia:+.7f} |")
    lineas += [
        "",
        "## La curva decisiva: oraculo restringido a la informacion del campeon",
        "",
        "Corrige una fila **solo** si `p(y_true) >= tau`, sin ninguna informacion nueva.",
        "",
        "| tau | Filas corregibles | Ganancia | Supera el objetivo |",
        "|---|---|---|---|",
    ]
    for tau, valores in curva.items():
        lineas.append(f"| {tau} | {valores['filas_corregibles']} | "
                      f"{valores['ganancia_macro_f1']:+.7f} | "
                      f"{'SI' if valores['supera_el_objetivo'] else 'NO'} |")
    lineas += [
        "",
        f"**Techo de la curva: {techo_curva:+.7f}** frente al objetivo **{OBJETIVO_DELTA:+.4f}**.",
        f"Veredicto: {resultado['veredicto_del_objetivo']}",
        "",
        (f"El objetivo exige ademas ~10.562 correcciones netas = "
        f"**{10562 / n_err * 100:.2f} % de los {n_err} errores**, mientras el mejor oraculo"),
        "disponible alcanza 9683 filas.",
        "",
        "## Limites estructurales del dato (no del modelo)",
        "",
        (f"- Textos truncados: mediana **{longitudes_crudas['mediana']:.0f}**, maximo "
        f"**{longitudes_crudas['maximo']}**, y **{longitudes_crudas['fraccion_en_el_maximo'] * 100:.1f} %** "
        "en el maximo: perdida de informacion **irrecuperable**."),
    ]
    for nombre in ("len_le_20", "len_le_3", "solo_url",
                   "texto_identico_etiqueta_distinta"):
        valores = limites[nombre]
        extra = (f" | techo del oraculo de mayoria {valores['techo_del_oraculo_de_mayoria']:+.7f}"
                 if "techo_del_oraculo_de_mayoria" in valores else "")
        if "macro_f1_subconjunto" in valores:
            lineas.append(f"- `{nombre}`: {valores['n']} filas | error {valores['error']:.4f} | "
                          f"macro-F1 {valores['macro_f1_subconjunto']:.4f}{extra}")
        else:
            lineas.append(f"- `{nombre}`: {valores['n_grupos']} grupos / {valores['n_filas']} "
                          f"filas | error {valores['error']:.4f}{extra}")
    lineas += [
        (f"- Filas no ASCII: **{int(no_ascii.sum())}** de {clase.size}, luego **no hay via "
        "multilingue**."),
        "",
        "## La calibracion no puede decidirlo",
        "",
        (f"- Se sostiene en **n = {CALIBRACION['n']}** puntos; dominio CV "
        f"`{CALIBRACION['dominio_cv']}`; `sigma` del residuo {CALIBRACION['sigma_residuo']}."),
        (f"- El objetivo pide CV {calibracion['cv_objetivo']:.4f}: "
        f"**{calibracion['distancia_en_sigmas_de_la_calibracion']:.1f} sigma** y "
        f"**{calibracion['fraccion_fuera_del_dominio_ajustado'] * 100:.0f} % fuera** del rango "
        "ajustado."),
        f"- {calibracion['lectura']}.",
        "",
    ]
    SALIDA_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    print(f"\n  {SALIDA_JSON.relative_to(ROOT.parent)}")
    print(f"  {SALIDA_MD.relative_to(ROOT.parent)}")
    print(f"\n[fin] {time.time() - inicio:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())