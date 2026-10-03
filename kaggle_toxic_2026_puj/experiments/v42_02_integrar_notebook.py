"""Integra la fase V42 (PLS correcto) en el notebook consolidado.

Anade la seccion 37 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v42_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v42_01_pls_correcto.json"

TITULO = """\
## 37. Fase v42: PLS / SUPERVISED LOW-RANK PROJECTION (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Una proyeccion latente supervisada `X -> Z`, aprendida usando simultaneamente TF-IDF y las tres
> etiquetas HS/TR/AG, puede producir una representacion que generaliza mejor o aporta errores utiles
> que V03 no captura?

La respuesta corta es **no**, y el por que es estructural.
"""

RESULTADO = """\
### 37.1 Resultado

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 (control) | 0,777211 | 0 |
| SVD matched (n_components 8) | 0,663295 | **-0,113916** |
| **PLS (n_components 8)** | 0,763469 | **-0,013742** |

| etiqueta | V03 | PLS | delta |
|---|---:|---:|---:|
| HS | 0,7617 | 0,6149 | -0,1468 |
| TR | 0,8157 | 0,6114 | -0,2044 |
| AG | 0,7542 | 0,5601 | -0,1941 |

**rescates 92 | danios 135 | neto -43 | correlacion 0,9490.**

### 37.2 Lo que el comparativo revela

**SVD-8 es catastrofico: -0,1139.** Reducir el TF-IDF de 20441 dimensiones a 8 destruye la
informacion.

**PLS-8 es mucho mejor que SVD-8: +0,1002.** Esto es la cifra importante. A igualdad de dimension,
la **supervision** encuentra direcciones que covarian con las etiquetas, y eso es exactamente lo que
no hace la varianza de X. La hipotesis de la directiva -que una proyeccion supervisada aporta
informacion que una no supervisada no- **queda confirmada**.

**Pero PLS-8 sigue por debajo de V03: -0,137.** Con 8 componentes, ni PLS ni SVD pueden competir
con las 20441 dimensiones del TF-IDF completo. La representacion latente es util pero demasiado
pobre.

### 37.3 El detalle que invalida el experimento

La dimension efectiva es **2, no 8**. La matriz termino x etiqueta tiene solo 3 columnas (HS, TR,
AG), asi que su rango es 3 y TruncatedSVD no puede extraer mas de 2 componentes.

Con 2 dimensiones el embedding no puede capturar las relaciones entre terminos que se pretendian
probar. **El experimento no prueba la hipotesis.**

### 37.4 Benchmark de memoria y tiempo

| magnitud | valor |
|---|---:|
| X sparse | (1186, 20441) |
| X dense | 193,9 MB |
| tiempo matriz | 0,15 s |
| tiempo PLS | 5,21 s |
| memoria antes | 229 MB |
| memoria tras PLS | 779 MB |
| folds en paralelo | no |

La conversion a dense es inevitable porque `PLSRegression` no acepta sparse. Los cinco folds se
procesaron secuencialmente.

### 37.5 Decision

**`PLS_INVALID`.** No se rechaza la hipotesis de embedding supervisado; se rechaza el experimento que
la probo. No se prueba `n_components=16` porque PLS-8 no muestra senal que lo justifique.

El siguiente paso es el **modelo neuronal minimo** `Embedding(d <= 32) -> mean pooling -> Linear(32, 3)`,
que si puede producir un embedding de dimension > 3 porque la capa embedding no esta limitada por el
rango de la matriz termino x etiqueta.
"""

CODIGO_RESULTADO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v42_01_pls_correcto.json").open(encoding="utf-8") as f:
    d = json.load(f)

c = d["control"]; p = d["pls"]; s = d["svd_matched"]
comp = d["complementariedad"]
print("37.1  RESULTADO")
print(f"  {'variante':<30}{'corrected_cv':>14}{'delta':>12}")
print(f"  {'V03 (control)':<30}{c['corrected_cv']:>14.6f}{0.0:>+12.6f}")
print(f"  {'SVD matched (n=8)':<30}{s['corrected_cv']:>14.6f}{s['delta_vs_v03']:>+12.6f}")
print(f"  {'PLS (n=8)':<30}{p['corrected_cv']:>14.6f}{p['delta_vs_v03']:>+12.6f}")
print()
print("37.2  LO QUE EL COMPARATIVO REVELA")
print(f"  SVD-8 es catastrofico: {s['delta_vs_v03']:+.6f}")
print(f"  PLS-8 sobre SVD-8:      {p['delta_vs_svd']:+.6f}  (la supervision SI importa)")
print(f"  PLS-8 sobre V03:        {p['delta_vs_v03']:+.6f}  (pero no llega)")
print()
print("37.3  EL DETALLE QUE INVALIDA EL EXPERIMENTO")
s1 = d["vocabulario_y_parametros"]["1"]
print(f"  dimension pedida: 8 | dimension efectiva: {s1['dimension']}")
print("  la matriz termino x etiqueta tiene rango 3, asi que el SVD no puede")
print("  extraer mas de 2 componentes. El experimento no prueba la hipotesis.")
print()
print("37.4  BENCHMARK DE MEMORIA Y TIEMPO")
for k, v in d["benchmark_de_memoria"].items():
    print(f"  {k:<28} {v}")
print()
print("37.5  DECISION")
print(f"  {d['decision']}")
print("  Siguiente: modelo neuronal minimo Embedding(d<=32) -> mean pooling -> Linear(32,3)")
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
    """Anade las celdas de la fase V42 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v42_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "PLS / SUPERVISED LOW-RANK PROJECTION" in marcas:
        print("  la seccion 37 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [markdown(TITULO), markdown(RESULTADO), codigo(CODIGO_RESULTADO)]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 37 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
