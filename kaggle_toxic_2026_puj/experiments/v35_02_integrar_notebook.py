"""Integra la fase V35 (SGD) en el notebook consolidado.

Anade la seccion 31 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v35_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v35_01_sgd.json"

TITULO = """\
## 31. Fase v35: SGD hinge y modified_huber (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Una perdida lineal distinta, particularmente Modified Huber, puede encontrar una solucion que
> generalice mejor que la regresion logistica de V03 sobre exactamente la misma representacion?

La respuesta corta es **no**, y la razon es el sobreajuste: SGD aprende el train perfectamente y
generaliza peor que liblinear.
"""

RESULTADO = """\
### 31.1 Resultado

| variante | corrected_cv | delta vs V03 | F1 HS | F1 TR | F1 AG | train F1 | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| V03 (control) | 0,777211 | 0 | 0,7617 | 0,8157 | 0,7542 | - | - |
| V35-01 hinge | 0,761035 | **-0,016177** | 0,7244 | 0,7939 | 0,7179 | **1,0000** | +0,2390 |
| V35-02 modified_huber | 0,743566 | **-0,033645** | - | - | - | **1,0000** | +0,2564 |

**Ambos negativos, ambos con train F1 1,0000.** El gap de +0,24 a +0,26 es sobreajuste, no falta
de capacidad.

### 31.2 Convergencia (Fase 4, obligatoria)

| variante | n_iter medio | max_iter alcanzado | warnings |
|---|---:|:--:|---:|
| hinge | 18,4 | no | 0 |
| modified_huber | 17,2 | no | 0 |

**Si convergen.** El bajo F1 no es falta de optimizacion: SGD encuentra una solucion que ajusta el
train al 100 % y generaliza peor que la regresion logistica.

Sensibilidad, sin seleccionar mirando el OOF:

| configuracion | corrected_cv |
|---|---:|
| max_iter 500 | 0,743566 |
| max_iter 5000 | 0,743566 |
| tol 1e-2 | 0,745873 |
| tol 1e-4 | 0,751675 |

`max_iter` no importa (500 = 5000). `tol` 1e-4 mejora hasta 0,751675 pero sigue muy por debajo de
V03. No se intenta rescatar con tuning.

### 31.3 Complementariedad contra V03

| variante | rescates | danios | net | celdas |
|---|---:|---:|---:|---:|
| hinge | 116 | 194 | **-78** | - |
| modified_huber | 139 | 241 | **-102** | - |

Ambos tienen rescates, pero los danios los superan con mucho. No hay complementariedad util.

### 31.4 La correccion conceptual, respetada

`SGDClassifier` NO es una nueva clase de funcion de decision respecto a V03. Sigue produciendo un
clasificador lineal. Lo que cambia es la **perdida** (hinge = control SVM, modified_huber = objetivo
distinto de la logistica) y el **procedimiento de optimizacion estocastico**. No se describe como
"nueva geometria" ni "nueva clase de funcion".

### 31.5 Decision

**`SGD = CLOSED`.** Resultado C de la directiva: hinge y modified_huber son negativos.

El siguiente candidato es **HASHING TRICK**, no otro tuning de SGD.
"""

CODIGO_RESULTADO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v35_01_sgd.json").open(encoding="utf-8") as f:
    d = json.load(f)

c = d["control"]
h = d["V35_01_hinge"]
m = d["V35_02_modified_huber"]
print("31.1  RESULTADO")
print(f"  {'variante':<24}{'corrected_cv':>14}{'delta':>12}{'train F1':>11}{'gap':>9}")
print(f"  {'V03 (control)':<24}{c['corrected_cv']:>14.6f}{0.0:>+12.6f}{'-':>11}{'-':>9}")
for nombre, v in (("V35-01 hinge", h), ("V35-02 modified_huber", m)):
    print(f"  {nombre:<24}{v['corrected_cv']:>14.6f}{v['delta_vs_v03']:>+12.6f}"
          f"{v['train_f1']:>11.4f}{v['gap_train_oof']:>+9.4f}")
print()
print("31.2  CONVERGENCIA")
for nombre, clave in (("hinge", "V35_01_hinge"), ("modified_huber", "V35_02_modified_huber")):
    cv = d[clave]["convergencia"]
    n_medio = sum(sum(v) for v in cv["n_iter"].values()) / sum(len(v) for v in cv["n_iter"].values())
    print(f"  {nombre:<24} n_iter medio {n_medio:.1f} | max_iter alcanzado "
          f"{any(cv['max_iter_alcanzado'].values())} | warnings {len(cv['warnings'])}")
print()
print("  sensibilidad (sin seleccionar mirando OOF):")
for k, v in d["sensibilidad"].items():
    print(f"    {k:<20} {v:.10f}")
print()
print("31.3  COMPLEMENTARIEDAD")
for nombre, clave in (("hinge", "hinge"), ("modified_huber", "modified_huber")):
    comp = d["complementariedad"][clave]
    print(f"  {nombre:<24} rescates {comp['rescates']} | danios {comp['danios']} | "
          f"neto {comp['neto']:+d}")
print()
print("31.5  DECISION")
print(f"  {d['decision']}")
print("  Siguiente candidato: HASHING TRICK")
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
    """Anade las celdas de la fase V35 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v35_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "SGD hinge y modified_huber" in marcas:
        print("  la seccion 31 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(RESULTADO),
            codigo(CODIGO_RESULTADO),
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 31 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
