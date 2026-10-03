"""FASE 40 (P7) — Análisis de error y complementariedad de los candidatos léxicos.

Aplica sobre el argmax OOF de cada candidato (guardado por P6) los análisis que exige el brief:

* **§19 fronteras**: errores de cruce en ``18-24|25-34`` y ``25-34|35-49``, los dos cuellos de
  botella centrales, comparados con V12.
* **§20 segmentos**: quintiles de longitud, con/sin menciones, con/sin hashtags, RT/no RT y los
  tres segmentos ya conocidos (``solo_url``, ``muy_corto``, ``sin_menciones``).
* **§21 complementariedad**: tabla de 4 celdas, rescate, daño, neto, acuerdo y desacuerdo. La
  pregunta es si el lexicón corrige errores reales de V12 o solo reescribe filas que V12 ya
  acertaba.

Usa el mismo módulo de análisis que la fase de capa de decisión (``src.decision_layer``) para que
las métricas sean comparables con las de aquella fase.

Salida: ``reports/age_lexicon_complementarity.md``, ``reports/age_lexicon_error_analysis.md``,
``experiments/phase40_analisis_error.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src import decision_layer as dl  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "phase40_analisis_error.json"
COMPLEMENTARIEDAD = cfg.REPORTS / "age_lexicon_complementarity.md"
ERRORES = cfg.REPORTS / "age_lexicon_error_analysis.md"
FRONTERAS_CENTRALES = ("18-24|25-34", "25-34|35-49")


def segmentos_extra(contexto: pd.DataFrame) -> dict:
    """Segmentos adicionales del §20: quintiles de longitud, menciones, hashtags y retuit."""
    extra: dict[str, np.ndarray] = {
        "solo_url": contexto[dl.GRUPOS_SOLO_URL].to_numpy() > 0,
        "muy_corto": contexto[dl.GRUPOS_MUY_CORTO].to_numpy() > 0,
        "sin_menciones": contexto[dl.GRUPOS_SIN_MENCIONES].to_numpy() > 0,
        "con_menciones": contexto["n_mention"].to_numpy() > 0,
        "con_hashtags": contexto["n_hashtag"].to_numpy() > 0,
        "sin_hashtags": contexto["n_hashtag"].to_numpy() == 0,
        "retuit": contexto["is_rt"].to_numpy() > 0,
        "no_retuit": contexto["is_rt"].to_numpy() == 0,
    }
    quintiles = pd.qcut(contexto["n_chars"].rank(method="first"), 5, labels=False)
    for indice in range(5):
        extra[f"longitud_q{indice + 1}"] = quintiles.to_numpy() == indice
    return extra


def main() -> int:
    """Calcula la complementariedad y el desglose de error de cada candidato léxico."""
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    reales = np.array([cfg.CLASSES.index(valor) for valor in y])
    contexto = datos.contexto_segmentos() if hasattr(datos, "contexto_segmentos") else None
    if contexto is None:
        denso = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
        contexto = pd.DataFrame({
            "n_chars": denso["n_chars"].to_numpy(float),
            "n_mention": denso["n_mention"].to_numpy(float),
            "n_hashtag": denso["n_hashtag"].to_numpy(float),
            "n_url": denso["n_url"].to_numpy(float),
            "is_rt": denso["is_rt"].to_numpy(float),
            dl.GRUPOS_SOLO_URL: ((denso["n_url"] >= 1) & (denso["n_words"] <= 3)).astype(float),
            dl.GRUPOS_MUY_CORTO: (denso["n_chars"] <= 15).astype(float),
            dl.GRUPOS_SIN_MENCIONES: (denso["n_mention"] == 0).astype(float),
        })

    base_indice = np.load(cfg.FEATURES / "oof_argmax_v12.npy")
    aciertos_base = base_indice == reales
    errores_base = int((~aciertos_base).sum())
    print(f"[V12] OOF argmax cargado | errores: {errores_base}", flush=True)

    candidatos = {}
    for ruta in sorted(cfg.FEATURES.glob("oof_argmax_v12_*.npy")):
        candidatos[ruta.stem.replace("oof_argmax_v12_", "")] = np.load(ruta)
    if not candidatos:
        raise SystemExit("no hay candidatos: ejecuta phase39 antes")

    informe = {"fecha": "2026-09-25", "v12": {"errores": errores_base}, "candidatos": {}}
    filas_complementaridad, filas_segmentos = [], []
    for nombre, predicciones in candidatos.items():
        aciertos = predicciones == reales
        rescate = int((aciertos & ~aciertos_base).sum())
        danio = int((aciertos_base & ~aciertos).sum())
        cambio = int((predicciones != base_indice).sum())
        filas_complementaridad.append({
            "candidato": nombre,
            "ambos_correctos": int((aciertos & aciertos_base).sum()),
            "solo_candidato": rescate, "solo_v12": danio, "ambos_mal": int((~aciertos & ~aciertos_base).sum()),
            "neto": rescate - danio,
            "acuerdo_pct": round(100.0 * float((predicciones == base_indice).mean()), 3),
            "desacuerdo_pct": round(100.0 - round(100.0 * float((predicciones == base_indice).mean()), 3), 3),
            "filas_modificadas": cambio, "filas_modificadas_pct": round(100.0 * cambio / len(y), 3),
            "tasa_rescate_del_error_pct": round(100.0 * rescate / max(1, errores_base), 3),
        })
        fronteras = {}
        for borde in range(len(cfg.CLASSES) - 1):
            izquierda, derecha = cfg.CLASSES[borde], cfg.CLASSES[borde + 1]
            clave = f"{izquierda}|{derecha}"
            cruza_base = int((((base_indice == borde) & (reales == borde + 1))
                              | ((base_indice == borde + 1) & (reales == borde))).sum())
            cruza_nuevo = int((((predicciones == borde) & (reales == borde + 1))
                               | ((predicciones == borde + 1) & (reales == borde))).sum())
            fronteras[clave] = {"errores_v12": cruza_base, "errores_candidato": cruza_nuevo,
                                "delta": cruza_nuevo - cruza_base}
        for nombre_segmento, mascara in segmentos_extra(contexto).items():
            error_base = 100.0 * float((~aciertos_base[mascara]).mean())
            error_nuevo = 100.0 * float((~aciertos[mascara]).mean())
            filas_segmentos.append({"candidato": nombre, "segmento": nombre_segmento,
                                    "filas": int(mascara.sum()),
                                    "error_v12_pct": round(error_base, 2),
                                    "error_candidato_pct": round(error_nuevo, 2),
                                    "delta_pp": round(error_nuevo - error_base, 2)})
        informe["candidatos"][nombre] = {
            "aciertos": int(aciertos.sum()),
            "complementariedad": filas_complementaridad[-1],
            "fronteras": fronteras,
            "fronteras_centrales_delta": {clave: fronteras[clave]["delta"]
                                         for clave in FRONTERAS_CENTRALES},
        }
        centrales = informe["candidatos"][nombre]["fronteras_centrales_delta"]
        print(f"[{nombre}] rescate {rescate} | daño {danio} | neto {rescate - danio:+d} | "
              f"modificadas {cambio} | fronteras centrales {centrales}", flush=True)

    pd.DataFrame(filas_complementaridad).to_csv(
        cfg.REPORTS / "age_lexicon_complementarity.csv", index=False)
    pd.DataFrame(filas_segmentos).to_csv(cfg.REPORTS / "age_lexicon_segments.csv", index=False)
    RESULTADO.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")

    mejor = max(filas_complementaridad, key=lambda fila: fila["neto"])
    lineas = ["# Complementariedad de los candidatos léxicos con V12 (P7)", "",
              f"Errores de V12 en el OOF: **{errores_base:,}** filas.", "",
              "| Candidato | Ambos correctos | Solo candidato (rescate) | Solo V12 (daño) | Neto | "
              "Acuerdo | Modificadas | Tasa de rescate |", "|---|---|---|---|---|---|---|---|"]
    for fila in filas_complementaridad:
        lineas.append(f"| {fila['candidato']} | {fila['ambos_correctos']:,} | {fila['solo_candidato']} "
                      f"| {fila['solo_v12']} | {fila['neto']:+d} | {fila['acuerdo_pct']:.3f}% | "
                      f"{fila['filas_modificadas']:,} ({fila['filas_modificadas_pct']:.3f}%) | "
                      f"{fila['tasa_rescate_del_error_pct']:.3f}% |")
    lineas += ["", f"**Mejor complementariedad**: `{mejor['candidato']}` con neto {mejor['neto']:+d} "
              f"(rescate {mejor['solo_candidato']} frente a daño {mejor['solo_v12']}).", "",
              "Interpretación (§21): un bloque léxico solo aporta si el **rescate supera al daño**. "
              "Un neto negativo significa que el lexicón destruye más aciertos de V12 de los que "
              "construye, por mucho que eleve su propio acierto.", ""]
    COMPLEMENTARIEDAD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    tabla_segmentos = pd.DataFrame(filas_segmentos)
    lineas = ["# Análisis de error por fronteras y segmentos (P7)", "",
              "## 1. Errores de cruce en las fronteras centrales (§19)", "",
              "| Candidato | 18-24|25-34 (V12 -> candidato) | 25-34|35-49 (V12 -> candidato) |",
              "|---|---|---|"]
    for nombre in candidatos:
        fronteras = informe["candidatos"][nombre]["fronteras"]
        izquierda = fronteras["18-24|25-34"]
        derecha = fronteras["25-34|35-49"]
        lineas.append(f"| {nombre} | {izquierda['errores_v12']:,} -> {izquierda['errores_candidato']:,} "
                      f"({izquierda['delta']:+d}) | {derecha['errores_v12']:,} -> "
                      f"{derecha['errores_candidato']:,} ({derecha['delta']:+d}) |")
    lineas += ["", "## 2. Error por segmento (§20)", "",
              "| Candidato | Segmento | Filas | Error V12 | Error candidato | Δ pp |",
              "|---|---|---|---|---|---|"]
    for _, fila in tabla_segmentos.iterrows():
        lineas.append(f"| {fila['candidato']} | {fila['segmento']} | {fila['filas']:,} | "
                      f"{fila['error_v12_pct']:.2f}% | {fila['error_candidato_pct']:.2f}% | "
                      f"{fila['delta_pp']:+.2f} |")
    ERRORES.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nartefactos: {COMPLEMENTARIEDAD.name}, {ERRORES.name}, "
          f"age_lexicon_complementarity.csv, age_lexicon_segments.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
