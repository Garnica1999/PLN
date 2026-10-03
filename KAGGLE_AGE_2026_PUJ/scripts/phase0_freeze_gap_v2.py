"""FASE 0 — Congelar v09/v10 y auditar el espacio de features (gap analysis v2).

Entregables:
    - ``experiments/baseline_v09_frozen/`` con hashes, parámetros, folds y submission.
    - ``reports/advanced_gap_analysis_v2.md`` con la tabla de 21 fenómenos del brief §5:
      phenoménon · detected · represented · representation · supervised · model_channel ·
      error_analysis · remaining_gap · new_hypothesis.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

FROZEN = cfg.EXPERIMENTS / "baseline_v09_frozen"
FILES = ["submissions/age_submission_v09.csv", "submissions/age_submission_v10.csv",
         "experiments/submission_v09_resumen.json" if (cfg.EXPERIMENTS / "submission_v09_resumen.json").exists()
         else "experiments/submission_v06_resumen.json",
         "experiments/advanced_02_resultados.json", "experiments/advanced_03_holdout.json",
         "experiments/advanced_04_incertidumbre.json", "experiments/advanced_05_nbsvm_seleccion.json",
         "experiments/advanced_07_frontera.json", "experiments/e028_en_ensamble.json",
         "reports/tables/fase17_matriz_confusion.csv", "reports/tables/fase17_pares_confundidos.csv",
         "reports/submissions_registry.csv", "src/pipeline.py", "src/advanced_features.py",
         "src/cv.py", "src/config.py"]

#: Estado de cada fenómeno del brief §5, con el Δ macro-F1 medido en este proyecto.
PHENOMENA = [
    ("longitud", "sí", "sí", "n_chars + 26 de distribución", "no aplica", "todos los miembros",
     "domina el error (52,5% en Q1)", "ninguna", "¿queda señal tras descontar longitud?"),
    ("riqueza léxica", "sí", "sí", "TTR, CTTR, hapax, entropía, MATTR, Yule's K, Maas, Simpson",
     "no", "miembros + meta", "redundante con longitud", "baja", "residualización (L2)"),
    ("rareza léxica", "sí", "sí", "12 features (frecuencia de corpus in-fold)", "sí (corpus de train)",
     "meta", "aporta poco", "+0,0003 medido", "¿por tipo de tweet?"),
    ("menciones", "sí", "sí", "TF-IDF + 17 features supervisadas", "sí P(clase|mención)", "canal + meta",
     "la señal más fuerte (−0,125 al eliminarlas)", "baja", "¿margen específico por frontera?"),
    ("hashtags", "sí", "sí", "TF-IDF + 17 supervisadas", "sí", "canal + meta", "neutro", "−0,0001 medido",
     "¿representación sin # o n-gramas del hashtag?"),
    ("RT", "sí", "sí", "is_rt + 17 de fuente supervisada + pares", "sí P(clase|fuente)", "miembros + meta",
     "RT más frecuente en mayores", "+0,0007 medido", "¿RT puro vs RT con comentario?"),
    ("URLs", "sí", "sí", "conteo, densidad, posición, longitud", "no", "meta", "más URL en los errores",
     "medido", "sin dominio (todas t.co)"),
    ("dominios", "sí (EDA)", "no aplica", "—", "no", "—", "no hay dominios distintos: todo es t.co",
     "vacío por datos", "cerrada"),
    ("puntuación", "sí", "sí", "densidad por tipo, transiciones, entropía", "no", "meta",
     "quitarla cuesta −0,008", "medido", "¿transiciones de carácter (L4)?"),
    ("ortografía", "sí", "sí", "10 de forma de palabra, elongaciones, mayúsculas", "no", "meta",
     "informalidad en clases jóvenes", "medido", "¿distribución y no solo frecuencia?"),
    ("repetición", "sí", "sí", "runs, ratios de unigramas/bigramas/trigramas", "no", "meta",
     "jajaja característico de jóvenes", "medido", "cerrada salvo repetición de largo alcance"),
    ("entropía", "sí", "sí", "carácter, token, puntuación, entidad", "no", "meta", "aportó poco",
     "medido", "cerrada"),
    ("distribución", "sí", "sí", "percentiles, asimetría, curtosis, CV, IQR, MAD", "no", "meta",
     "colas distintas por clase", "+0,0006 (bloque)", "residualización (L2)"),
    ("posición", "sí", "sí", "20 features de posición normalizada", "no", "meta", "nunca explotada antes",
     "medido", "cerrada"),
    ("OOV", "sí", "sí", "ratio OOV y de novedad léxica", "sí (vocabulario de train)", "meta",
     "no explorado antes", "+0,0003 (bloque)", "cerrada"),
    ("ordinalidad", "sí", "sí", "acumulativo, pares adyacentes, jerarquía, léxico por clase",
     "parcial", "miembros + meta", "fronteras medidas (0,486-0,601)", "+0,0006 (pares)",
     "¿scores pairwise como features del meta?"),
    ("perfiles léxicos", "sí (EDA)", "sí", "log-odds por clase (10 features)", "sí", "meta",
     "términos distintivos muy interpretables", "+0,0001", "cerrada"),
    ("tipos de tweet", "parcial", "no", "—", "no", "—", "especialistas por longitud dieron 0",
     "**sin taxonomía explícita**", "**L5: tipología + F1 por tipo**"),
    ("incertidumbre", "sí", "sí", "51 features de confianza, margen, entropía, acuerdo", "no", "meta",
     "aportó poco", "+0,0002", "cerrada"),
    ("diversidad de modelos", "sí", "parcial", "tasas de desacuerdo puntuales", "no", "—",
     "ComplementNB y MLP aportan", "**sin matriz formal**", "**L6/§48: matriz de complementariedad**"),
    ("near duplicates", "sí", "sí (CV)", "clústeres Jaccard ≥ 0,8 para agrupar folds", "no", "—",
     "1,21% literal / 4,26% en clústeres", "usado en CV, no como feature",
     "retrieval bloqueado por reglas"),
    ("errores universales", "no", "no", "—", "no", "—", "nunca analizado",
     "**ausente**", "**L1: intersección de errores**"),
]


def sha256(path: Path) -> str:
    """SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> dict:
    """Congela v09/v10 y los artefactos que los reproducen."""
    FROZEN.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for relative in FILES:
        source = cfg.ROOT / relative
        if source.exists():
            shutil.copy2(source, FROZEN / source.name)
            hashes[relative] = sha256(source)
    manifest = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "baseline": "v09 (AGE_ADVANCED_V01) — LB 0,60807",
        "cv_lofo": 0.5897, "cv_std": 0.0043,
        "holdout_congelado": 0.5966, "holdout_baseline": 0.5914,
        "leaderboard": 0.60807,
        "variante_lista_sin_enviar": {"v10": {"cv_lofo": 0.5904, "delta": 0.0007}},
        "presupuesto_envios": {"usados": 9, "disponibles": 1},
        "arquitectura": {
            "miembros_base": 14,
            "meta_modelo": "StandardScaler + LogisticRegression (273 features del candidato v10)",
            "validacion": "StratifiedGroupKFold 5 folds agrupado por clúster de near-duplicados, "
                          "evaluación leave-one-fold-out",
        },
        "hashes_sha256": hashes,
    }
    (FROZEN / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return manifest


def gap_analysis_v2() -> str:
    """Tabla de 21 fenómenos con estado, representación y brecha restante."""
    lines = [
        "# Gap analysis v2 — espacio de features y señal ausente",
        "",
        "Generado por `scripts/phase0_freeze_gap_v2.py`. Cada fila indica si el fenómeno ya está "
        "detectado, representado, si tiene versión supervisada, en qué canal entra al modelo, qué "
        "dice el análisis de errores y **qué brecha queda**. La última columna es la hipótesis nueva.",
        "",
        "> **Propósito metodológico.** Este documento evita re-ejecutar líneas ya medidas. La columna "
        "`Δ medido` recoge el aporte real al ensamble cuando existe.",
        "",
        "| Fenómeno | Detectado | Representado | Representación actual | Supervisado | Canal | "
        "Análisis de errores | Brecha | Hipótesis nueva |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in PHENOMENA:
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Líneas con brecha confirmada (las únicas que aborda esta fase)",
        "",
        "1. **L1 — Errores universales y de intersección** (ausente): nunca se midió cuántos tweets "
        "tienen al menos un modelo base acertando mientras el ensamble falla. Es el diagnóstico que "
        "decide si el problema es de información o de ponderación.",
        "2. **L2 — Residualización por longitud** (ausente): la longitud domina el error y confunde a "
        "las features de riqueza; el residuo puede contener señal de edad ortogonal.",
        "3. **L3 — Separación contenido/estilo** (sin representación explícita): el contenido se "
        "comparte entre cohortes (RT de humor y deportes en 18-24 y 25-34, ya documentado); el "
        "estilo podría conservar señal demográfica.",
        "4. **L4 — Transiciones de caracteres** (ausente): patrones letra->dígito, letra->puntuación, "
        "espacio->puntuación que el TF-IDF no expresa como feature explícita.",
        "5. **L5 — Tipología de tweets y umbral de información** (sin taxonomía explícita): el error "
        "puede depender del tipo estructural, no solo de la longitud.",
        "6. **L6 — Matriz de complementariedad y ponderación condicional** (sin matriz formal): "
        "guiada por L1.",
        "7. **L7 — NB-SVM por canal** (extensión barata de algo ya validado: +0,0007 en char+mención).",
        "",
        "## Líneas cerradas con evidencia (no se repiten)",
        "",
        "| Línea | Δ medido | Evidencia |",
        "|---|---|---|",
        "| Mención supervisada | +0,0016 | ADV-02 (ya en v09) |",
        "| RT source supervisado | +0,0007 | ADV-02 |",
        "| Rareza léxica | +0,0003 | ADV-02 |",
        "| Léxico por clase | +0,0001 | ADV-02 |",
        "| Hashtag supervisado | −0,0001 | ADV-02 |",
        "| Distribución/formas/posición | +0,0006 | ADV-02 |",
        "| Incertidumbre e interacciones | +0,0002 | ADV-04 |",
        "| Selección de features | 0,0000 | ADV-05 |",
        "| NB-SVM como modelo | −0,0017 | ADV-05 (se conserva como miembro) |",
        "| Especialistas por longitud | +0,0000 | E-023/E-024 |",
        "| Frontera con compuerta dura | −0,0036 | ADV-07 |",
        "| Enrutamiento por margen (par-SVM) | 0,0000 | ADV-07 |",
        "| BM25 sin normalización · kNN sobre SVD · meta LightGBM · jerarquía gruesa | ≤ 0 | E-022/E-025/E-026/E-028b |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Ejecuta el congelamiento y el gap analysis v2."""
    manifest = freeze()
    print(f"[freeze] {FROZEN.name}: {len(manifest['hashes_sha256'])} archivos con hash")
    text = gap_analysis_v2()
    (cfg.REPORTS / "advanced_gap_analysis_v2.md").write_text(text, encoding="utf-8")
    print(f"[gap] reports/advanced_gap_analysis_v2.md ({len(text.splitlines())} líneas)")

    # Backlog de la fase
    backlog = "\n".join([
        "# Research backlog — fase avanzada v2",
        "",
        "| ID | Línea | Hipótesis | Coste | Estado |",
        "|---|---|---|---|---|",
        "| L1 | Intersección de errores | Algún modelo base acierta donde el ensamble falla | 15 min | en curso |",
        "| L5 | Tipología y umbral de información | El error depende del tipo estructural | 20 min | en curso |",
        "| L2 | Residualización por longitud | Hay señal de edad ortogonal a la longitud | 25 min | pendiente |",
        "| L4 | Transiciones de carácter | El estilo de transición aporta | 15 min | pendiente |",
        "| L3 | Contenido vs estilo | El estilo conserva señal que el contenido no | 40 min | pendiente |",
        "| L7 | NB-SVM por canal | Cada canal aporta una vista diversa | 25 min | pendiente |",
        "| L6 | Meta 2.0 y complementariedad | La ponderación por tipo recupera errores | 30 min | pendiente |",
        "",
        "Puerta de decisión tras L1/L5: si `one_model_right` > 5% -> priorizar L6; si < 2% -> L3/L2.",
    ])
    (cfg.REPORTS / "research_backlog_v2.md").write_text(backlog, encoding="utf-8")
    print("[backlog] reports/research_backlog_v2.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
