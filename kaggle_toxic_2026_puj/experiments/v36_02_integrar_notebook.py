"""Integra la fase V36 (hashing trick) en el notebook consolidado.

Anade la seccion 32 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v36_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v36_01_hashing.json"

TITULO = """\
## 32. Fase v36: HASHING TRICK (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Eliminar el vocabulario explicito y reemplazarlo por una representacion hashed, manteniendo
> aproximadamente el mismo TF-IDF y los mismos n-grams, encuentra una frontera lineal que
> generaliza mejor o aporta errores complementarios que V03 no captura?

La respuesta corta es **no**, y la razon no es la tasa de colisiones.
"""

RESULTADO = """\
### 32.1 Resultado

| variante | corrected_cv | delta vs V03 | train F1 | gap |
|---|---:|---:|---:|---:|
| V03 (control) | 0,777211 | 0 | - | - |
| Hashing 2^18 | 0,748427 | **-0,028785** | 0,9603 | +0,2119 |

| etiqueta | V03 | hashing | delta |
|---|---:|---:|---:|
| HS | 0,7617 | 0,7430 | -0,0187 |
| TR | 0,8157 | 0,7678 | -0,0479 |
| AG | 0,7542 | 0,7344 | -0,0198 |

**OVERFIT**: train F1 0,9603 pero OOF cae -0,0288, y la complementariedad es negativa.

### 32.2 La auditoria de colisiones, que es la que explica el resultado

| bloque | n-grams | buckets unicos | colisiones | proporcion |
|---|---:|---:|---:|---:|
| word | 3435 | 3408 | 27 | **0,0079** |
| char | 21211 | 20363 | 848 | **0,0400** |

**2^18 si es suficiente.** Las colisiones son del 0,8 % y el 4 %, bajas. El problema no es la
tasa de colisiones: la representacion hashed generaliza peor que el vocabulario explicito.

### 32.3 Controles de implementacion

| control | valor |
|---|---|
| dimension por bloque | 262144 |
| dimension total | 524288 |
| no negativo por bloque | [True, True] |
| sin NaN | True |
| sin inf | True |
| reproducible | True |
| sparsity (muestra) | 99,955 % |
| features no vacias | 0,0449 % |

### 32.4 Complementariedad contra V03

| label | rescates | danios | neto |
|---|---:|---:|---:|
| HS | - | - | - |
| TR | - | - | - |
| AG | - | - | - |

**rescates 131 | danios 238 | neto -107 | correlacion 0,9171.**

El hashing no aprende decisiones diferentes: tiene correlacion 0,9171 con V03 y un neto de -107.
No hay complementariedad util.

### 32.5 Fairness: que cambia exactamente

| componente | V03 | Hashing |
|---|---|---|
| representacion | vocabulario explicito | hashing |
| word ngrams | (1,2) | (1,2) |
| char ngrams | (3,5) | (3,5) |
| TF-IDF | si | si |
| sublinear TF | si | si |
| min_df | 2 | no aplica |
| dimensionalidad | ~27k | 2x2^18 |
| vocabulario | si | no |
| colisiones | no | si |

La comparacion es justa: se usa `HashingVectorizer -> TfidfTransformer` para separar el cambio de
representacion del cambio de ponderacion.

### 32.6 Decision

**`HASHING = CLOSED`.** No se prueban 2^17 ni 2^19 porque 2^18 no muestra senal, como manda la
directiva. El siguiente candidato es **BM25 weighting**.
"""

CODIGO_RESULTADO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v36_01_hashing.json").open(encoding="utf-8") as f:
    d = json.load(f)

c = d["control"]
h = d["hashing"]
comp = d["complementariedad"]
print("32.1  RESULTADO")
print(f"  {'variante':<20}{'corrected_cv':>14}{'delta':>12}{'train F1':>11}{'gap':>9}")
print(f"  {'V03 (control)':<20}{c['corrected_cv']:>14.6f}{0.0:>+12.6f}{'-':>11}{'-':>9}")
print(f"  {'Hashing 2^18':<20}{h['corrected_cv']:>14.6f}{h['delta_vs_v03']:>+12.6f}"
      f"{h['train_f1']:>11.4f}{h['gap_train_oof']:>+9.4f}")
print()
print("32.2  AUDITORIA DE COLISIONES")
print(f"  {'bloque':<8}{'n-grams':>10}{'buckets unicos':>16}{'colisiones':>12}{'proporcion':>13}")
for nombre, v in d["colisiones"].items():
    print(f"  {nombre:<8}{v['n_ngrams_en_vocabulario']:>10}{v['buckets_unicos']:>16}"
          f"{v['colisiones']:>12}{v['proporcion_de_buckets_con_mas_de_un_ngram']:>13.4f}")
print()
print("  2^18 SI es suficiente: las colisiones son bajas. El problema no es la tasa de colisiones.")
print()
print("32.4  COMPLEMENTARIEDAD")
print(f"  rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d} | "
      f"correlacion {comp['correlacion_de_decisiones']:.4f}")
print()
print("32.6  DECISION")
print(f"  {d['decision']}")
print("  Siguiente candidato: BM25 weighting")
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
    """Anade las celdas de la fase V36 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v36_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "HASHING TRICK" in marcas:
        print("  la seccion 32 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [markdown(TITULO), markdown(RESULTADO), codigo(CODIGO_RESULTADO)]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 32 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
