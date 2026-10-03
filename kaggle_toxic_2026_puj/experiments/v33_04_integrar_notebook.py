"""Integra la fase V33 (validacion y submission de V32-05) en el notebook consolidado.

Anade la seccion 29 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
los JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v33_04_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
NOMBRES = ("01_validacion", "02_generar_submission")

TITULO = """\
## 29. Fase v33: validacion y submission de V32-05 (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> La mejora OOF de V32-05 se transfirio a Kaggle?

La respuesta corta es **no**, y la razon es estructural: la submission es byte a byte identica a
V03.
"""

RESULTADO = """\
### 29.1 El resultado del leaderboard

| magnitud | valor |
|---|---|
| ID de submission | **56608788** |
| fichero | `submissions/toxic_submission_v32_05_url_length.csv` |
| sha256 | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` |
| score LB | **0,78306** |
| V03 LB | **0,78306** |
| **delta LB** | **0,00000** |

Y el detalle que lo explica todo: **el sha256 del CSV de V32-05 es identico al de V03**. La
submission no cambio ni una sola prediccion sobre las 1154 filas de test.

### 29.2 Por que la submission es identica a V03

El modelo entrenado con el train completo (1731 filas) aprende **coeficientes cero** para las
cinco interacciones. Son redundantes con el TF-IDF de V03 cuando se entrena con todos los datos.

En OOF parecian aportar porque con menos datos (1186 filas por pliegue) el modelo no podia aprender
la combinacion exacta. Con el train completo, el TF-IDF ya captura la informacion que aportarian las
interacciones, y el modelo las descarta.

Y un detalle adicional: `url_con_query` y `url_corta_x_corto` son **cero en todo el dataset**, asi
que de las cinco interacciones solo tres son no triviales.

### 29.3 La cadena completa de evidencia

| etapa | resultado | que indica |
|---|---|---|
| OOF (5 pliegues) | **+0,004510** | la interaccion parece aportar |
| holdout2 (248 filas) | **-0,000519**, neto +1 | la transferencia ya no estaba garantizada |
| test (1154 filas) | **0 filas cambiadas** | el modelo no usa las interacciones |
| leaderboard | **0,78306 = V03** | la mejora OOF no transfirio |

Holdout2 ya anticipo el resultado. La delta era ligeramente negativa con neto de solo +1 sobre 248
filas, y el test confirmo que el modelo entrenado con todos los datos descarta las interacciones.
"""

DOS_ERRORES = """\
### 29.4 Los dos errores de reproduccion que la Fase 1 detecto

La directiva exigia reproducir V32-05 exactamente, y la primera corrida **no** reprodujo: dio
+0,005443 en vez de +0,004510. Dos causas, ambas instructivas:

1. **Claves dinamicas en las interacciones.** Al generar los nombres de columna como `url_x_b0`,
   `url_x_b1` segun el tramo, cada fila tenia claves distintas y `pd.DataFrame` rellena las que
   faltan con **NaN**. sklearn rechazo la matriz. Corregido con claves fijas.

2. **V32-05 apila solo las interacciones, no las variables URL aditivas.** La primera reproduccion
   incluia tambien `base` (las 13 variables URL), lo que cambiaba el modelo y daba +0,0054. La
   reproduccion exige la misma matriz de features, no solo el mismo algoritmo.

Ambos errores se detectaron porque la directiva exigia la reproduccion exacta. Sin ese requisito,
la submission se habria generado sobre una configuracion distinta a la evaluada.
"""

CIERRE = """\
### 29.5 La respuesta a la pregunta de la fase

> La mejora OOF de V32-05 se transfirio a Kaggle?

**No.** `URL_LENGTH_INTERACTION = WEAK_TRANSFER`, con delta LB exactamente 0 porque la submission
es identica a V03.

Y la razon no es que la mejora OOF fuera ruido: es que **las interacciones son redundantes con el
TF-IDF cuando se entrena con el train completo**. En OOF parecian aportar porque el modelo tenia
menos datos; con todos los datos, el modelo las descarta.

Esto es coherente con lo que V32 ya establecio: el mecanismo no generaliza a mencion, a hashtag ni
a n-grams. Y ahora tampoco generaliza de OOF a train completo.

### 29.6 Estado final de la familia

| condicion de continuacion | se cumple |
|---|:--:|
| el efecto URL x longitud se reproduce | si |
| mantiene direccion positiva | si |
| no depende del fold 1/2/3 | si |
| mantiene rescates netos positivos | si |
| mention/hashtag con senal | no |
| generaliza a n-gram x longitud | no |
| **transfiere a Kaggle** | **NO** |

**La familia queda cerrada.** No se abre segunda linea de generalizacion, no se hace tuning de bins,
y no se propone otra submission. La evidencia acumulada es especifica: la interaccion entre la
estructura de URL y la longitud del tweet aporta en OOF pero no transfiere.

### 29.7 Lo que esta fase no hace

- No abre otras familias.
- No hace tuning de bins.
- No propone otra submission.
- No modifica V03 ni V32-05.
- No toca `N = 1186`.
"""

CODIGO_RESULTADO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent

def leer(nombre):
    with (RAIZ / "reports" / f"v33_{nombre}.json").open(encoding="utf-8") as f:
        return json.load(f)

d1 = leer("01_validacion")
d2 = leer("02_generar_submission")
r1 = d1["fase_1_reproducibilidad"]
ho = d1["fase_5_holdout2"]
print("29.1  EL RESULTADO DEL LEADERBOARD")
print(f"  ID de submission: 56608788")
print(f"  score LB: 0.78306 | V03 LB: 0.78306 | delta LB: 0.00000")
print(f"  sha256 del CSV: {d2['sha256'][:16]}... (identico al de V03)")
print()
print("29.3  LA CADENA COMPLETA DE EVIDENCIA")
print(f"  {'etapa':<22}{'resultado':>16}{'que indica':<40}")
print(f"  {'OOF 5 pliegues':<22}{r1['delta_vs_v03']:>+16.6f}{'parece aportar':<40}")
print(f"  {'holdout2 248':<22}{ho['delta']:>+16.6f}{'transferencia no garantizada':<40}")
print(f"  {'test 1154':<22}{'0 filas':>16}{'el modelo descarta las interacciones':<40}")
print(f"  {'leaderboard':<22}{'0.78306':>16}{'no transfiere':<40}")
print()
print("29.4  LOS DOS ERRORES DE REPRODUCCION")
print("  1. claves dinamicas -> pd.DataFrame rellena con NaN -> sklearn rechaza")
print("  2. V32-05 apila SOLO las interacciones, no las variables URL aditivas")
print("  ambos se detectaron porque la directiva exigia la reproduccion exacta.")
print()
print("29.5  RESPUESTA")
print("  URL_LENGTH_INTERACTION = WEAK_TRANSFER")
print("  delta LB exactamente 0 porque la submission es identica a V03.")
print("  La familia queda cerrada. No se abre segunda linea.")
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
    """Anade las celdas de la fase V33 al notebook consolidado.

    Returns:
        None.
    """
    faltan = [
        str(PROJECT_ROOT / "reports" / f"v33_{n}.json")
        for n in NOMBRES
        if not (PROJECT_ROOT / "reports" / f"v33_{n}.json").exists()
    ]
    if faltan:
        raise SystemExit(f"PARADA: faltan informes: {faltan}")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "validacion y submission de V32-05" in marcas:
        print("  la seccion 29 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(RESULTADO),
            markdown(DOS_ERRORES),
            markdown(CIERRE),
            codigo(CODIGO_RESULTADO),
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 29 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
