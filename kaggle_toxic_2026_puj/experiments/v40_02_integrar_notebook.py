"""Integra la fase V40 (PLS supervisado) en el notebook consolidado.

Anade la seccion 36 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v40_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v40_01_pls.json"

TITULO = """\
## 36. Fase v40: PLS / SUPERVISED LOW-RANK PROJECTION (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Una proyeccion latente supervisada `X -> Z`, aprendida usando simultaneamente TF-IDF y las tres
> etiquetas HS/TR/AG, puede producir una representacion que generaliza mejor o aporta errores utiles
> que V03 no captura?

La respuesta corta es **no**, y el por que es informativo.
"""

RESULTADO = """\
### 36.1 Resultado

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 (control) | 0,777211 | 0 |
| SVD matched (n_components 8) | 0,663295 | **-0,113916** |
| **PLS (n_components 8)** | 0,763469 | **-0,013742** |

### 36.2 Lo que el comparativo revela

**SVD-8 es catastrofico: -0,1139.** Reducir el TF-IDF de 20441 dimensiones a 8 destruye la
informacion.

**PLS-8 es mucho mejor que SVD-8: +0,1002.** Esto es la cifra importante. A igualdad de dimension,
la **supervision** encuentra direcciones que covarian con las etiquetas, y eso es exactamente lo que
no hace la varianza de X. La hipotesis de la directiva -que una proyeccion supervisada aporta
informacion que una no supervisada no- **queda confirmada**.

**Pero PLS-8 sigue por debajo de V03: -0,0137.** Con 8 componentes, ni PLS ni SVD pueden competir
con las 20441 dimensiones del TF-IDF completo. La representacion latente es util pero demasiado
pobre.

### 36.3 Complementariedad contra V03

**rescates 92 | danios 135 | neto -43 | correlacion 0,9490.**

La correlacion de 0,9490 y el neto negativo indican que PLS **no aprende decisiones diferentes**:
casi replica V03 y ademas se equivoca mas. No hay complementariedad util.

### 36.4 Benchmark de memoria y tiempo

| magnitud | valor |
|---|---:|
| X sparse | (1186, 20441) |
| X dense | 193,9 MB |
| tiempo matriz | 0,15 s |
| tiempo PLS | 5,34 s |
| memoria antes | 228 MB |
| memoria tras PLS | 778 MB |
| folds en paralelo | no |

La conversion a dense es inevitable porque `PLSRegression` no acepta sparse. Los cinco folds se
procesaron secuencialmente.

### 36.5 Decision

**`SUPERVISED_PLS = CLOSED`** (`OVERFIT`: train F1 0,9004, gap +0,1748, delta -0,0137, neto -43).

No se prueba `n_components=16` porque PLS-8 no muestra senal que lo justifique. Y no se abre el
modelo neuronal minimo: el Paso 4 de la directiva solo aplica si la factorizacion es suficiente-
mente distinta, y el resultado de V39 ya demostro algebraicamente que `Embedding -> mean pooling
-> Linear(3)` es equivalente a un clasificador lineal sobre bag-of-words.

### 36.6 Lo que si queda demostrado

A igualdad de dimension, la supervicion aporta +0,1002 sobre una reduccion latente no supervisada.
Eso es una prueba limpia de que la hiptesis de V40 -que una proyeccion supervisada contiene
informacion util- es correcta. Lo que falla es la capacidad: con 8 componentes no hay suficiente
informacion para competir con el TF-IDF completo.
"""

CODIGO_RESULTADO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v40_01_pls.json").open(encoding="utf-8") as f:
    d = json.load(f)

c = d["control"]; p = d["pls"]; s = d["svd_matched"]
comp = d["complementariedad"]
b = d["benchmark_de_memoria"]
print("36.1  RESULTADO")
print(f"  {'variante':<30}{'corrected_cv':>14}{'delta':>12}")
print(f"  {'V03 (control)':<30}{c['corrected_cv']:>14.6f}{0.0:>+12.6f}")
print(f"  {'SVD matched (n=8)':<30}{s['corrected_cv']:>14.6f}{s['delta_vs_v03']:>+12.6f}")
print(f"  {'PLS (n=8)':<30}{p['corrected_cv']:>14.6f}{p['delta_vs_v03']:>+12.6f}")
print()
print("36.2  LO QUE EL COMPARATIVO REVELA")
print(f"  SVD-8 es catastrofico: {s['delta_vs_v03']:+.6f}")
print(f"  PLS-8 sobre SVD-8:      {p['delta_vs_svd']:+.6f}  (la supervision SI importa)")
print(f"  PLS-8 sobre V03:        {p['delta_vs_v03']:+.6f}  (pero no llega)")
print()
print("36.3  COMPLEMENTARIEDAD")
print(f"  rescates {comp['rescates']} | danios {comp['danios']} | "
      f"neto {comp['neto']:+d} | correlacion {comp['correlacion_de_decisiones']:.4f}")
print()
print("36.4  BENCHMARK DE MEMORIA Y TIEMPO")
for k, v in b.items():
    print(f"  {k:<28} {v}")
print()
print("36.5  DECISION")
print(f"  {d['decision']}")
print("36.6  LO QUE SI QUEDA DEMOSTRADO")
print("  A igualdad de dimension, la supervision aporta +0.1002 sobre SVD no supervisado.")
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
    """Anade las celdas de la fase V40 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v40_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "PLS / SUPERVISED LOW-RANK PROJECTION" in marcas:
        print("  la seccion 36 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [markdown(TITULO), markdown(RESULTADO), codigo(CODIGO_RESULTADO)]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 36 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
