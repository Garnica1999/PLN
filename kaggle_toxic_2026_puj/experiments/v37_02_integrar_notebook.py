"""Integra la fase V37 (BM25) en el notebook consolidado.

Anade la seccion 33 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v37_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v37_01_bm25.json"

TITULO = """\
## 33. Fase v37: BM25 WEIGHTING (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Cambiar exclusivamente el weighting de los word n-grams desde TF-IDF hacia BM25, manteniendo
> char TF-IDF y el clasificador de V03, mejora la generalizacion aprovechando mejor la longitud y
> la frecuencia de terminos?

La respuesta corta es **no**, y la razon es que BM25 no aporta una fuente de senal nueva.
"""

RESULTADO = """\
### 33.1 Resultado

| variante | corrected_cv | delta vs V03 | train F1 | gap |
|---|---:|---:|---:|---:|
| V03 (control) | 0,777211 | 0 | - | - |
| word BM25 + char TF-IDF | 0,774425 | **-0,002786** | 0,9732 | +0,1988 |

| etiqueta | V03 | BM25 | delta |
|---|---:|---:|---:|
| HS | 0,7617 | 0,7627 | +0,0010 |
| TR | 0,8157 | 0,8095 | -0,0063 |
| AG | 0,7542 | 0,7511 | -0,0031 |

**OVERFIT**: train F1 0,9732 pero OOF cae -0,0028, y la complementariedad es casi nula.

### 33.2 Complementariedad contra V03

**rescates 25 | danios 31 | neto -6 | correlacion 0,9874.**

El 0,9874 de correlacion es la cifra que cierra la familia: BM25 no aprende decisiones
diferentes. Pondera la longitud y la TF de forma distinta, pero con 1731 documentos esa
diferencia no generaliza mejor que `sublinear_tf`.

### 33.3 Comportamiento por longitud

| region | filas | V03 | BM25 | delta |
|---|---:|---:|---:|---:|
| corto | 514 | 0,8763 | 0,8768 | +0,0005 |
| medio | 515 | 0,7144 | 0,7099 | -0,0044 |
| largo | 454 | 0,6298 | 0,6244 | -0,0054 |

La unica region donde BM25 no pierde es el texto corto, y por +0,0005. En medio y largo pierde.
La motivacion de la fase era la longitud, y el resultado es el opuesto.

### 33.4 Controles matematicos

Los cinco controles pasan: la TF se satura (multiplicador 2,09 en vez de 4), el documento largo
pesa menos por termino, el crecimiento de TF es sublineal, el termino raro tiene mayor IDF, y k1
alto satura menos mientras b alto penaliza mas los largos.

### 33.5 Decision

**`BM25 = CLOSED`.** No se hace grid de k1/b, no se aplica a char, y no se propone submission.

El resultado historico de `experiments/30_supervised_weighting_v4.py` (0,7748-0,7755) es
consistente.

El siguiente candidato es **EMBEDDINGS SUPERVISADOS DESDE CERO**, pero solo despues de auditar
cuidadosamente las reglas del curso, porque el dataset es pequeno y el riesgo de overfitting es
alto.
"""

CODIGO_RESULTADO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v37_01_bm25.json").open(encoding="utf-8") as f:
    d = json.load(f)

c = d["control"]
b = d["bm25"]
comp = d["complementariedad"]
print("33.1  RESULTADO")
print(f"  {'variante':<26}{'corrected_cv':>14}{'delta':>12}{'train F1':>11}{'gap':>9}")
print(f"  {'V03 (control)':<26}{c['corrected_cv']:>14.6f}{0.0:>+12.6f}{'-':>11}{'-':>9}")
print(f"  {'word BM25 + char TF-IDF':<26}{b['corrected_cv']:>14.6f}{b['delta_vs_v03']:>+12.6f}"
      f"{b['train_f1']:>11.4f}{b['gap_train_oof']:>+9.4f}")
print()
print("33.2  COMPLEMENTARIEDAD")
print(f"  rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d} | "
      f"correlacion {comp['correlacion_de_decisiones']:.4f}")
print()
print("33.3  POR LONGITUD")
print(f"  {'region':<8}{'filas':>7}{'V03':>9}{'BM25':>9}{'delta':>9}")
for nombre, v in d["por_longitud"].items():
    print(f"  {nombre:<8}{v['n_filas']:>7}{v['v03_macro_f1']:>9.4f}"
          f"{v['bm25_macro_f1']:>9.4f}{v['delta']:>+9.4f}")
print()
print("33.4  CONTROLES MATEMATICOS")
for nombre, v in d["controles_matematicos"].items():
    if isinstance(v, dict) and "cumple" in v:
        print(f"  {nombre}: cumple {v['cumple']}")
print()
print("33.5  DECISION")
print(f"  {d['decision']}")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin outputs.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def main() -> None:
    """Anade las celdas de la fase V37 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v37_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "BM25 WEIGHTING" in marcas:
        print("  la seccion 33 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [markdown(TITULO), markdown(RESULTADO), codigo(CODIGO_RESULTADO)]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 33 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
