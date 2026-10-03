"""Integra la fase V38 (embedding supervisado) en el notebook consolidado.

Anade la seccion 34 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v38_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v38_01_embedding_supervisado.json"

TITULO = """\
## 34. Fase v38: SUPERVISED EMBEDDINGS FROM SCRATCH (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Una representacion distribuida entrenada directamente con las etiquetas HS/TR/AG puede capturar
> relaciones utiles entre terminos que el TF-IDF explicito de V03 no captura, sin caer en
> memorization con solo ~1483 ejemplos?

La respuesta corta es **no**, y la razon es estructural: la matriz termino x etiqueta tiene rango 3,
asi que no puede producir un embedding de dimension 16.
"""

RESULTADO = """\
### 34.1 Resultado

| variante | corrected_cv | delta vs V03 | train F1 | gap |
|---|---:|---:|---:|---:|
| V03 (control) | 0,777211 | 0 | - | - |
| embedding supervisado | 0,595466 | **-0,181746** | 0,9143 | +0,3188 |

| etiqueta | V03 | embedding | delta |
|---|---:|---:|---:|
| HS | 0,7617 | 0,6149 | -0,1468 |
| TR | 0,8157 | 0,6114 | -0,2044 |
| AG | 0,7542 | 0,5601 | -0,1941 |

**rescates 321 | danios 1465 | neto -1144 | correlacion 0,5986.**

### 34.2 El detalle que invalida el experimento

La dimension efectiva es **2, no 16**. La matriz termino x etiqueta tiene solo 3 columnas (HS, TR,
AG), asi que su rango es 3 y TruncatedSVD no puede extraer mas de 2 componentes.

Con 2 dimensiones el embedding no puede capturar las relaciones entre terminos que se pretendian
probar. **El experimento no prueba la hipotesis.**

### 34.3 Analisis de memorization

| magnitud | valor |
|---|---:|
| terminos con df <= 5 | 6681 de 7101 (94,09 %) |
| correlacion proporcion rara y error | -0,0394 |
| train F1 | 0,9143 |
| OOF F1 | 0,595466 |
| gap | +0,3188 |

El 94 % del vocabulario son terminos raros, pero la correlacion entre proporcion de terminos raros y
error es **negativa** (-0,0394): el modelo no funciona mejor en terminos frecuentes. El gap de +0,3188
es sobreajuste.

### 34.4 Decision

**`SUPERVISED_EMBEDDINGS = CLOSED`** (factorizacion invalida por rango 3).

El siguiente paso es el **modelo neuronal minimo** `Embedding(d <= 32) -> mean pooling -> Linear(32, 3)`,
que si puede producir un embedding de dimension > 3 porque la capa embedding no esta limitada por el
rango de la matriz termino x etiqueta.

### 34.5 Lo que esta fase no hace

- No envia submission.
- No construye una red neuronal completa.
- No usa el test para nada.
- No modifica V03.
"""

CODIGO_RESULTADO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v38_01_embedding_supervisado.json").open(encoding="utf-8") as f:
    d = json.load(f)

c = d["control"]
e = d["embedding"]
comp = d["complementariedad"]
mem = d["memorization"]
print("34.1  RESULTADO")
print(f"  {'variante':<26}{'corrected_cv':>14}{'delta':>12}{'train F1':>11}{'gap':>9}")
print(f"  {'V03 (control)':<26}{c['corrected_cv']:>14.6f}{0.0:>+12.6f}{'-':>11}{'-':>9}")
print(f"  {'embedding supervisado':<26}{e['corrected_cv']:>14.6f}{e['delta_vs_v03']:>+12.6f}"
      f"{e['train_f1']:>11.4f}{e['gap_train_oof']:>+9.4f}")
print()
print("34.2  EL DETALLE QUE INVALIDA EL EXPERIMENTO")
s1 = d["vocabulario_y_parametros"]["1"]
print(f"  dimension pedida: 16 | dimension efectiva: {s1['dimension']}")
print(f"  la matriz termino x etiqueta tiene rango 3 (solo 3 etiquetas), asi que el SVD no puede")
print(f"  extraer mas de 2 componentes. Con 2 dimensiones el embedding no puede capturar las")
print(f"  relaciones entre terminos que se pretendian probar. El experimento no prueba la hipotesis.")
print()
print("34.3  ANALISIS DE MEMORIZATION")
print(f"  terminos con df <= 5: {mem['n_terminos_raros']} de 7101 ({100*mem['proporcion_de_terminos_raros']:.2f} %)")
print(f"  correlacion proporcion rara y error: {mem['correlacion_proporcion_rara_y_error']:+.4f}")
print(f"  train F1 {mem['train_f1']:.4f} | OOF F1 {mem['oof_f1']:.4f} | gap {mem['gap']:+.4f}")
print()
print("34.4  DECISION")
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
    """Anade las celdas de la fase V38 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v38_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "SUPERVISED EMBEDDINGS FROM SCRATCH" in marcas or "embedding supervisado" in marcas:
        print("  la seccion 34 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [markdown(TITULO), markdown(RESULTADO), codigo(CODIGO_RESULTADO)]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 34 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
