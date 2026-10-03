"""Integra la fase V44 (frontier audit) en el notebook consolidado.

Anade la seccion 39 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas
de codigo leen los JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v44_03_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORMES = {
    "vw": PROJECT_ROOT / "reports" / "v44_01_vw_optimizers.json",
    "pls": PROJECT_ROOT / "reports" / "v44_02_pls_correcto.json",
}

TITULO = """\
## 39. Fase v44: frontier audit - que queda sin probar (2026-09-27)

La pregunta de la fase: despues de cerrar modelos no lineales, kernels, boosting,
ensembles, label specialization, label noise y representaciones sin vocabulario,
¿existe alguna fuente de señal que todavia no se haya probado?

La respuesta se construye en dos pasos. Primero, una revisión exhaustiva del
registry de 310 filas. Segundo, una evaluacion empírica de los dos candidatos
que sobreviven a esa revisión.
"""

REGISTRY = """\
### 39.1 Revision del registry: 10 familias principales

| familia | experimentos | verdict |
|---|---:|---|
| Clasificadores lineales (LR, Ridge, PA) | 60+ | Probados exhaustivamente |
| Kernels (RBF, poly, spectrum) | 25+ | Cerrados - sin senal |
| Boosting (XGB, LGB, CatBoost) | 8 | Cerrados - sin senal |
| Ensembles (stacking, voting, blending) | 20+ | Cerrados - sin senal |
| Label specialization | 7 | Cerrado - sin senal |
| Label noise / robustez | 12 | Cerrado - sin senal |
| Embeddings no supervisados (FastText) | 25+ | Cerrados - sin senal |
| Embeddings supervisados (SIF, uSIF) | 8 | Cerrados - sin senal |
| Combinaciones hibridas | 15+ | Cerradas - sin senal |
| Meta-modelos / stacking | 12 | Cerrados - sin senal |

Ninguna de estas familias ha producido una mejora transferible sobre V03.
"""

CANDIDATES = """\
### 39.2 Los dos candidados que sobreviven

Tras la revision, solo dos mecanismos tienen una justificacion para ser probados:

**1. PLS Regression sobre la representacion TF-IDF de V03.**
PLS (Partial Least Squares) es un metodo de reduccion de dimensionalidad
supervisado. A diferencia de PCA (que maximiza la varianza de X), PLS encuentra
componentes que maximizan la covarianza entre X e Y. Esto lo hace particularmente
adecuado para problemas de clasificacion con alta dimensionalidad y muestreo
pequeño, porque las componentes estan orientadas hacia la tarea.

**2. Optimizacion de hiperparámetros del pipeline de V03.**
Aunque los modelos individuales han sido probados, la optimizacion conjunta de
los hiperparametros del pipeline completo (analyzer, ngram_range, min_df, C,
class_weight) con Optuna no ha sido explorada de forma sistematica.
"""

CODIGO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
print("PLAN: frontier audit")
print()
print("Las dos familias candidatas que sobreviven a la revision del registry:")
print("  1. PLS Regression sobre el TF-IDF de V03")
print("  2. Optimizacion de hiperparámetros del pipeline completo con Optuna")
print()
print("NOTA: Ambas familias son de capacidad acotada y no representan")
print("un cambio de paradigma. Son candidatas a exploracion, no a submission.")
print()
print("El proyecto se cierra despues de evaluar estas dos familias.")
print("V03 (0.78306 LB) permanece como champion congelado.")
print("Total submissions gastadas en esta sesion (V34-V44): 0")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown."""
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin outputs."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def main() -> None:
    """Anade la seccion 39 al notebook consolidado."""
    if not all(p.exists() for p in INFORMES.values()):
        raise SystemExit("PARADA: faltan informes previos")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
    if "frontier audit" in marcas:
        print("  la seccion 39 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [markdown(TITULO), markdown(REGISTRY), markdown(CANDIDATES), codigo(CODIGO)]
    )
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 39 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
