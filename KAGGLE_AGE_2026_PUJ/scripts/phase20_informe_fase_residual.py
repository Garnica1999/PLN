"""FASE 20 — Informe consolidado de la búsqueda de señal residual (cierre de fases A-J).

Reúne los artefactos de todas las fases de la búsqueda (auditoría, análisis de errores, consenso,
calibración, diversidad de subconjuntos, ordinal, estructura v2 y limpieza) en un único informe con
la tabla de variantes frente al criterio de adopción declarado de antemano
(Δ ≥ +0,003 de macro-F1 con ≥4/5 folds positivos y confirmación en el holdout congelado).

Uso::

    python scripts/phase20_informe_fase_residual.py

Salida: ``reports/fase_residual_informe.md`` y ``experiments/phase20_resumen.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

INFORME = cfg.REPORTS / "fase_residual_informe.md"
RESUMEN = cfg.EXPERIMENTS / "phase20_resumen.json"
CRITERIO = "Δ ≥ +0,003 y ≥4/5 folds positivos, con confirmación independiente en el holdout congelado"

FUENTES = [
    ("A · Auditoría del estado", "phase12_auditoria_estado.json"),
    ("B · Errores y desacuerdo", "phase13_error_desacuerdo.json"),
    ("C · Consenso y gating", "phase14_consenso_familias.json"),
    ("D · Calibración y logits", "phase15_calibracion_scores.json"),
    ("D5 · Diversidad de subconjuntos", "phase15b_diversidad_subsets.json"),
    ("G · Ordinal y greedy", "phase16_ordinal_greedy_g1g2.json"),
    ("E · Estructura v2 (bloques libres)", "phase17_estructura_v2_libres.json"),
    ("F · Limpieza selectiva", "phase18_limpieza_selectiva.json"),
    ("§13 · Vista estructural y extremos", "phase21_vista_estructural.json"),
    ("§13 · Refuerzo de la vista", "phase22_estructura_ampliada.json"),
    ("H · Holdout congelado de la vista", "phase19_holdout_VistaEstructura.json"),
]


def cargar(nombre: str) -> dict | None:
    """Carga un artefacto de la fase si existe."""
    ruta = cfg.EXPERIMENTS / nombre
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def main() -> int:
    """Genera el informe consolidado y el resumen en JSON."""
    lineas = ["# Búsqueda de señal residual — informe consolidado (fases A-J)", "",
              "Generado por `scripts/phase20_informe_fase_residual.py` a partir de los artefactos en "
              "`experiments/`.", "",
              f"**Criterio de adopción declarado de antemano**: {CRITERIO}.", "",
              "**Regla de integridad**: validación agrupada por clústeres de near-duplicados "
              "(`StratifiedGroupKFold`), comparación LOFO sobre las mismas particiones, holdout "
              "congelado intacto, ninguna decisión guiada por el leaderboard.", ""]
    resumen: dict = {"criterio": CRITERIO, "fases": {}}
    variantes: list[dict] = []

    for titulo, nombre in FUENTES:
        datos = cargar(nombre)
        if datos is None:
            lineas += [f"## {titulo}", "", "_No ejecutada o sin artefacto._", ""]
            continue
        resumen["fases"][titulo] = datos
        lineas += [f"## {titulo}", "", f"Artefacto: `experiments/{nombre}`", ""]
        for clave_lista, clave_nombre in (("variantes", "variante"), ("vistas", "vista")):
            if clave_lista not in datos:
                continue
            lineas += ["| Variante | macro-F1 | Δ vs v12 | folds + | n features |",
                       "|---|---|---|---|---|"] if clave_lista == "variantes" else []
            for fila in datos[clave_lista]:
                if "macro_f1" not in fila:
                    continue
                variantes.append({"fase": titulo, "variante": fila.get(clave_nombre), **fila})
                etiqueta = str(fila.get(clave_nombre, fila.get("variante", "—")))
                lineas.append(f"| {etiqueta} | {fila['macro_f1']:.4f} | "
                              f"{fila.get('delta_vs_v12', 0):+.4f} | "
                              f"{fila.get('folds_positivos', '—')}/5 | "
                              f"{fila.get('n_features', fila.get('n_features_nuevas', '—'))} |")
            lineas.append("")
        if "resultados" in datos and "baseline" not in datos:
            lineas += ["| Configuración | macro-F1 en holdout |", "|---|---|"]
            for nombre, valores in datos["resultados"].items():
                if isinstance(valores, dict) and "promedio 0,5/0,5" in valores:
                    variantes.append({"fase": titulo, "variante": nombre,
                                      "macro_f1": valores["promedio 0,5/0,5"]["macro_f1"],
                                      "delta_vs_v12": datos.get("delta_vs_base_misma_corrida", 0)
                                      if nombre != "base" else 0.0,
                                      "folds_positivos": 0, "n_features": valores.get("n_features")})
                lineas.append(f"| {nombre} | "
                              f"{valores.get('promedio 0,5/0,5', {}).get('macro_f1', '—')} |")
            lineas.append("")
        for clave in ("libres", "supervisado", "g1_ordinal", "g2_greedy"):
            if clave in datos and isinstance(datos[clave], dict):
                seccion = datos[clave]
                if "macro_f1" in seccion:
                    variantes.append({"fase": f"{titulo} · {clave}", "variante": clave, **seccion})
                    lineas += [f"- **{clave}**: macro-F1 {seccion['macro_f1']:.4f} "
                               f"({seccion.get('delta_vs_v12', 0):+.4f}, "
                               f"{seccion.get('folds_positivos', '—')}/5 folds, "
                               f"adoptado: {seccion.get('adoptado', False)})", ""]
        if "baseline" in datos:
            lineas += [f"Referencia: **{datos['baseline']['macro_f1']:.4f}** "
                       f"(por fold {datos['baseline']['per_fold']}).", ""]
            # Error estándar de las diferencias pareadas por fold (ruido de la evaluación).
            referencia = datos["baseline"]["per_fold"]
            for fila in datos.get("variantes", []):
                pareado = fila.get("per_fold")
                if not pareado or len(pareado) != len(referencia):
                    continue
                diferencias = np.asarray(pareado, dtype=float) - np.asarray(referencia, dtype=float)
                error_estandar = float(diferencias.std(ddof=1) / np.sqrt(len(diferencias)))
                resumen.setdefault("error_estandar_pareado", []).append(
                    {"fase": titulo, "variante": fila["variante"],
                     "delta_pareado_medio": round(float(diferencias.mean()), 4),
                     "error_estandar": round(error_estandar, 4),
                     "t_aproximado": round(float(diferencias.mean() / error_estandar), 2)
                     if error_estandar > 0 else 0.0})

    adoptadas = [fila for fila in variantes
                 if float(fila.get("delta_vs_v12", 0) or 0) >= 0.003
                 and int(fila.get("folds_positivos", 0) or 0) >= 4]
    lineas += ["## Veredicto", ""]
    if adoptadas:
        for fila in adoptadas:
            lineas.append(f"- **{fila['fase']} · {fila['variante']}**: "
                          f"{fila['macro_f1']:.4f} ({fila['delta_vs_v12']:+.4f}), "
                          f"{fila['folds_positivos']}/5 folds -> **pasa a holdout congelado**.")
    else:
        lineas += ["Ninguna variante alcanza el criterio. **La línea de señal residual se declara "
                   "saturada**: las fuentes evaluadas no producen una mejora suficientemente grande "
                   "y consistente respecto al modelo final v12 (CV 0,6004; holdout 0,6077; "
                   "leaderboard 0,62182).", "",
                   "El modelo final **no cambia** y no se genera submission nueva.", ""]
    lineas += ["", "## Líneas cerradas en esta fase", "",
               "| Línea | Evidencia |",
               "|---|---|",
               "| Consenso/desacuerdo por familia y gating contextual | D-028 |",
               "| Calibración por miembro y blending en logits | D-029 |",
               "| Diversidad de subconjuntos de features en el segundo nivel | D-029 |",
               "| Edad implícita/consistencia ordinal y selección greedy | D-030 |",
               "| Estructura v2 (retuit, hashtag, extremos) | D-030 |",
               "| Limpieza selectiva y representantes por clúster | D-031 |", ""]

    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESUMEN.write_text(json.dumps({"criterio": CRITERIO, "variantes": variantes,
                                   "adoptadas": adoptadas,
                                   "n_variantes_evaluadas": len(variantes)},
                                  indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"informe: {INFORME}")
    print(f"variantes evaluadas: {len(variantes)} | adoptadas: {len(adoptadas)}")
    for fila in variantes:
        etiqueta = str(fila.get("variante", fila.get("fase", "?")))
        print(f"  {fila['fase'][:34]:36s} {etiqueta[:42]:44s} "
              f"{fila.get('delta_vs_v12', 0):+.4f} {fila.get('folds_positivos', '—')}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
