"""Integra la fase V39 (auditoria algebraica) en el notebook consolidado.

Anade la seccion 35 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v39_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"

TITULO = """\
## 35. Fase v39: auditoria algebraica del mean-pooling (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Embedding entrenado desde cero + mean pooling + Linear(3) introduce realmente una nueva capacidad
> funcional con respecto a los modelos lineales ya probados?

La respuesta corta es **no**, y se demuestra algebraicamente sin gastar CPU.
"""

DEMOSTRACION = """\
### 35.1 La demostracion

Con `E en R^{Vxd}` (embedding de cada token), `W en R^{dx3}` (capa lineal) y `z = (1/n) sum_t E_t`
(mean pooling):

```
logit_y = W_y^T z + b_y = W_y^T (1/n sum_t E_t) + b_y = (1/n) sum_t (W_y^T E_t) + b_y
```

Definiendo `M = E W en R^{Vx3}` (matriz terminoxlabel), donde `M[t,y] = E_t^T W_y`:

```
logit_y = (1/n) sum_t M[t,y] + b_y
```

**Esto es exactamente un clasificador lineal sobre bag-of-words** con pesos `M[t,y]`.

### 35.2 El rango

`rank(M) <= min(d, 3)`. Con d=2: rango 2. Con d=32: rango 3.

**Aumentar d de 2 a 32 NO aporta capacidad funcional adicional**, porque M siempre tiene rango
<= 3 (solo hay 3 labels).

### 35.3 La unica diferencia

La **optimizacion** (SGD vs liblinear) y la **regularizacion implicita** del mean pooling. No es
una capacidad funcional nueva.

### 35.4 Decision

**`SUPERVISED_EMBEDDING_MEANPOOL = REDUNDANT`.** No se ejecuta.

**`SUPERVISED_EMBEDDINGS = OPEN`** - queda por evaluar PLS / supervised low-rank projection, que
si introduce una capacidad funcional diferente: la proyeccion se aprende maximizando la covarianza
entre X e Y, no minimizando una perdida.

### 35.5 PLS no existe en el proyecto

Ni en el codigo ni en el registry. Podria ser la siguiente familia si se confirma que no es
equivalente a SVD ni a terminoxetiqueta.
"""

CODIGO_DEMOSTRACION = """\
print("35.1  LA DEMOSTRACION")
print()
print("  Con E en R^{V x d}, W en R^{d x 3} y z = (1/n) sum_t E_t:")
print()
print("    logit_y = W_y^T z + b_y")
print("            = W_y^T (1/n sum_t E_t) + b_y")
print("            = (1/n) sum_t (W_y^T E_t) + b_y")
print()
print("  Definiendo M = E W en R^{V x 3}, donde M[t,y] = E_t^T W_y:")
print()
print("    logit_y = (1/n) sum_t M[t,y] + b_y")
print()
print("  Esto es EXACTAMENTE un clasificador lineal sobre bag-of-words con pesos M[t,y].")
print()
print("35.2  EL RANGO")
print()
print("  rank(M) <= min(d, 3). Con d=2: rango 2. Con d=32: rango 3.")
print("  Aumentar d de 2 a 32 NO aporta capacidad funcional adicional,")
print("  porque M siempre tiene rango <= 3 (solo hay 3 labels).")
print()
print("35.3  LA UNICA DIFERENCIA")
print("  La optimizacion (SGD vs liblinear) y la regularizacion implicita del mean pooling.")
print("  No es una capacidad funcional nueva.")
print()
print("35.4  DECISION")
print("  SUPERVISED_EMBEDDING_MEANPOOL = REDUNDANT. No se ejecuta.")
print("  SUPERVISED_EMBEDDINGS = OPEN. Queda PLS / supervised low-rank projection.")
print()
print("35.5  PLS NO EXISTE EN EL PROYECTO")
print("  Ni en el codigo ni en el registry. Podria ser la siguiente familia.")
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
    """Anade las celdas de la fase V39 al notebook consolidado.

    Returns:
        None.
    """
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "auditoria algebraica del mean-pooling" in marcas:
        print("  la seccion 35 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [markdown(TITULO), markdown(DEMOSTRACION), codigo(CODIGO_DEMOSTRACION)]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 35 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
