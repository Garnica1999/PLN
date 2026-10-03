"""Integra la familia STRUCTURAL_TEXT_FEATURES (V31) en el notebook consolidado.

Anade la seccion 27 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
los JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v31_05_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORMES = {
    "01": PROJECT_ROOT / "reports" / "v31_01_url_estructural.json",
    "02": PROJECT_ROOT / "reports" / "v31_02_mencion_hashtag.json",
    "03": PROJECT_ROOT / "reports" / "v31_03_enfasis_ortografico.json",
    "04": PROJECT_ROOT / "reports" / "v31_04_interacciones_url.json",
}

TITULO = """\
## 27. Fase v31: STRUCTURAL_TEXT_FEATURES (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Existe informacion estructural del tweet que V03 no puede representar adecuadamente mediante
> sus n-grams aditivos, y que pueda incorporarse con pocas features contextuales?

La respuesta corta es **si, pero solo en las interacciones, y solo en el bloque URL**. Las features
estructurales aditivas apenas mueven la metrica; las interacciones explicitas con la longitud la
multiplican por veinte.
"""

TABLA = """\
### 27.1 La tabla de la fase

| experimento | corrected_cv | delta | rescates | danios | net | pliegues + | estado |
|---|---:|---:|---:|---:|---:|---:|---|
| V03 (control) | 0,777211 | 0 | - | - | - | - | congelado |
| V31-01 URL aditivo | 0,777420 | +0,000209 | 35 | 29 | +6 | - | WEAK_SIGNAL |
| V31-02 mencion/hashtag | 0,776479 | -0,000732 | 72 | 79 | -7 | - | REJECTED |
| V31-03 enfasis | 0,774442 | -0,002769 | 49 | 61 | -12 | - | REJECTED |
| **V31-04 URL + interacciones** | **0,781378** | **+0,004167** | **70** | **53** | **+17** | **4/5** | **NEW_SIGNAL** |

El control reproduce V03 con exactitud en los cuatro experimentos: **0,7772114525**.
"""

EL_HALLAZGO = """\
### 27.2 El hallazgo: la senal esta en las interacciones, no en las features

El Control A del bloque URL da **+0,000209** con neto +6. El Control B, que anade cinco cruces
discretos con la longitud del texto, da **+0,004167** con neto +17. **El Control B aporta +0,003958
sobre el Control A, que es veinte veces mas.**

Las cinco interacciones son:

| interaccion | que condicion expresa |
|---|---|
| `url_x_corto` | hay URL y el texto tiene 12 tokens o menos |
| `url_x_medio` | hay URL y el texto tiene entre 13 y 22 tokens |
| `url_x_largo` | hay URL y el texto tiene mas de 22 tokens |
| `url_query_x_largo` | la URL tiene query y el texto es largo |
| `url_corta_x_corto` | la URL es corta (<= 20 caracteres) y el texto es corto |

Y el mecanismo es exactamente el que predijo V30-1: **el efecto de una URL segun la longitud del
texto es de signo contrario**. En texto corto, tener URL duplica el error de HS (0,2542 frente a
0,1338); en texto largo, la URL lo empeora en AG (0,2798 frente a 0,2190). Un modelo lineal con
n-grams no puede representar eso, porque no tiene forma de condicionar un n-grama a la longitud del
texto. Las cinco flags si.

**Estabilidad**: 4 de 5 pliegues positivos. El pliegue 4 es -0,0307, que es el pliegue mas debil de
V03 en general (0,7386 frente a 0,7960 del pliegue 1). La media es +0,0043 con sd 0,0177.

**Gap**: train F1 0,9711, gap +0,1897. V03 tiene train F1 0,9710 y gap +0,1938. **El gap no
empeora**: las interacciones no anaden sobreajuste, anaden condicionamiento.

**Por label**: AG +0,0081 (neto +12), HS +0,0017 (neto +3), TR +0,0027 (neto +2). La mejora se
concentra en AG, que es la etiqueta con mas errores (277, frente a 316 de HS y 164 de TR) y la que
V30-1 encontro con mayor efecto de la longitud (d = +0,337).
"""

LOS_PERDEDORES = """\
### 27.3 Los dos bloques que no funcionaron, y que leccion dejan

**V31-02 mencion/hashtag: -0,000732, neto -7.** Trece features de posicion y contexto, y ninguna
mejora. No se prueba su Control B ni se combina.

**V31-03 enfasis ortografico: -0,002769, neto -12.** Es el caso mas instructivo de la fase, porque
ilustra la regla central de la directiva: **una feature con fuerte asociacion marginal pero cero
rescates es informational only**. V30-1 encontro que `caracteres_repetidos` tiene el mayor efecto
despues de la longitud (d = +0,0994, 4/5 pliegues, HS 0,2982 frente a 0,1937). Y sin embargo el
bloque de enfasis **empeora** la metrica. Describe el error, no lo corrige.

La diferencia entre el bloque URL y el bloque enfasis es la interaccion. El enfasis no tiene una
interaccion con la longitud que cambie el signo de su efecto; la URL si.
"""

CIERRE = """\
### 27.4 Cierre de la fase

1. **`NEW_SIGNAL` en el bloque URL con interacciones.** +0,004167, neto +17, 4/5 pliegues, gap sin
   empeorar, mejora concentrada en AG.
2. **`WEAK_SIGNAL` en el bloque URL aditivo.** +0,000209, neto +6. No basta por si solo, pero es la
   base sobre la que construyen las interacciones.
3. **`REJECTED` en mencion/hashtag y en enfasis.** Ambos negativos. No se combinan ni se prueban
   sus interacciones.
4. **No se combinan bloques.** La directiva lo permite solo si ambos muestran senal, y solo URL la
   muestra.
5. **No se toca `N = 1186`.** Nada de esta fase dice nada sobre si es un techo.
6. **0 submissions, V03 intacto, mismo modelo lineal.**

**El bloque que merece segunda fase es el de URL con interacciones.** La pregunta abierta para esa
segunda fase es si el mismo mecondicionamiento (interacciones discretas con la longitud) aplicado a
mencion y hashtag, que fueron negativos en aditivo, podria comportarse como el de URL. La directiva no
lo exige, pero el mecanismo que funciono en URL es exactamente el tipo de cosa que V30-1 encontro
para la mencion: su efecto segun la longitud es de signo contrario.
"""

CODIGO_TABLA = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent

def leer(nombre):
    with (RAIZ / "reports" / f"v31_{nombre}.json").open(encoding="utf-8") as f:
        return json.load(f)

d1, d2, d3, d4 = (leer(n) for n in ("01_url_estructural", "02_mencion_hashtag",
                                  "03_enfasis_ortografico", "04_interacciones_url"))
base = d1["control"]["corrected_cv"]
print("27.1  LA TABLA DE LA FASE")
print(f"  {'experimento':<28}{'corrected_cv':>14}{'delta':>12}{'resc':>7}{'danios':>8}"
      f"{'net':>6}{'pliegues':>10}  estado")
filas = [
    ("V03 (control)", base, 0.0, None, None, None, "-", "congelado"),
    ("V31-01 URL aditivo", d1["challenger"]["corrected_cv"],
     d1["challenger"]["delta_oof"], d1["comparacion"]["rescates"],
     d1["comparacion"]["danios"], d1["comparacion"]["neto"], "-", "WEAK_SIGNAL"),
    ("V31-02 mencion/hashtag", d2["challenger"]["corrected_cv"], d2["delta_oof"],
     d2["comparacion"]["rescates"], d2["comparacion"]["danios"],
     d2["comparacion"]["neto"], "-", "REJECTED"),
    ("V31-03 enfasis", d3["challenger"]["corrected_cv"], d3["delta_oof"],
     d3["comparacion"]["rescates"], d3["comparacion"]["danios"],
     d3["comparacion"]["neto"], "-", "REJECTED"),
    ("V31-04 URL + interacciones", d4["control_B_con_interacciones"]["corrected_cv"],
     d4["delta_oof"], d4["comparacion"]["rescates"], d4["comparacion"]["danios"],
     d4["comparacion"]["neto"], "4/5", "NEW_SIGNAL"),
]
for nombre, cv, delta, r, d, n, pl, estado in filas:
    fmt = lambda v, p=6: "-" if v is None else f"{v:+.{p}f}"
    print(f"  {nombre:<28}{cv:>14.6f}{fmt(delta):>12}"
          f"{('-' if r is None else r):>7}{('-' if d is None else d):>8}"
          f"{('-' if n is None else f'{n:+d}'):>6}{pl:>10}  {estado}")
print()
print(f"  control reproduce V03 en los cuatro experimentos: {base:.10f}")
"""

CODIGO_HALLAZGO = """\
print("27.2  EL HALLAZGO: LA SENAL ESTA EN LAS INTERACCIONES")
print()
a = d4["control_A_solo_features"]
b = d4["control_B_con_interacciones"]
print(f"  Control A (solo features URL):  {a['corrected_cv']:.10f} | "
      f"delta {a['corrected_cv'] - base:+.10f}")
print(f"  Control B (+ interacciones):    {b['corrected_cv']:.10f} | "
      f"delta {b['corrected_cv'] - base:+.10f}")
print(f"  el Control B aporta sobre el A: {d4['delta_B_menos_A']:+.10f}")
print()
print("  las cinco interacciones, que son cruces discretos y no productos de variables continuas")
for f in d4["features_interaccion"]:
    print(f"    {f}")
print()
print("  ESTABILIDAD ENTRE PLIEGUES")
print(f"  {'pliegue':>8}{'F1 media':>11}{'delta vs V03':>14}")
for m in b["metricas_por_pliegue"]:
    print(f"  {m['pliegue']:>8}{m['f1_media_columnas']:>11.4f}"
          f"{m['f1_media_columnas'] - base:>+14.4f}")
deltas = [m["f1_media_columnas"] - base for m in b["metricas_por_pliegue"]]
print(f"  pliegues positivos: {sum(1 for x in deltas if x > 0)} de {len(deltas)}")
print()
print("  POR LABEL")
for et in ("HS", "TR", "AG"):
    c = d4["control"]["f1_por_etiqueta"][et]
    n = b["f1_por_etiqueta"][et]
    print(f"  {et}: {c:.4f} -> {n:.4f} ({n-c:+.4f})")
print()
print(f"  train F1 {b['train_f1']:.4f} | gap {b['gap_train_oof']:+.4f} (V03 tiene +0.1938)")
print()
print("  LECTURA: el salto no esta en las features aditivas sino en las interacciones. El efecto")
print("  de una URL segun la longitud es de signo contrario, y un modelo lineal con n-grams no")
print("  puede representarlo. Las cinco flags si. Y el gap no empeora.")
"""

CODIGO_PERDEDORES = """\
print("27.3  LOS DOS BLOQUES QUE NO FUNCIONARON")
print()
for nombre, d, etiqueta in (("V31-02 mencion/hashtag", d2, "REJECTED"),
                            ("V31-03 enfasis ortografico", d3, "REJECTED")):
    ch = d["challenger"]
    comp = d["comparacion"]
    print(f"  {nombre}")
    print(f"    corrected_cv {ch['corrected_cv']:.10f} | delta {d['delta_oof']:+.10f}")
    print(f"    rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d}")
    print(f"    train F1 {ch['train_f1']:.4f} | gap {ch['gap_train_oof']:+.4f}  -> {etiqueta}")
print()
print("  LECTURA: el bloque enfasis es el caso mas instructivo. V30-1 encontro que")
print("  caracteres_repetidos tiene el mayor efecto despues de la longitud (d +0.0994, 4/5")
print("  pliegues, HS 0.2982 vs 0.1937). Y sin embargo el bloque empeora la metrica. Describe el")
print("  error, no lo corrige. La diferencia con URL es la interaccion: el enfasis no tiene una")
print("  interaccion con la longitud que cambie el signo de su efecto; la URL si.")
"""

CODIGO_CIERRE = """\
print("27.4  CIERRE DE LA FASE")
print()
print("  NEW_SIGNAL       URL con interacciones: +0.004167, neto +17, 4/5 pliegues")
print("  WEAK_SIGNAL      URL aditivo: +0.000209, neto +6")
print("  REJECTED         mencion/hashtag: -0.000732, neto -7")
print("  REJECTED         enfasis ortografico: -0.002769, neto -12")
print()
print("  No se combinan bloques: la directiva lo permite solo si ambos muestran senal.")
print("  N = 1186 no se toca. 0 submissions, V03 intacto, mismo modelo lineal.")
print()
print("  El bloque que merece segunda fase es el de URL con interacciones. La pregunta abierta")
print("  es si el mismo condicionamiento aplicado a mencion y hashtag, negativos en aditivo,")
print("  podria comportarse como el de URL.")
g = d4["gastos"]
print()
print(f"  Coste total de la fase: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB.")
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
    """Anade las celdas de la familia V31 al notebook consolidado.

    Returns:
        None.
    """
    faltan = [str(ruta) for ruta in INFORMES.values() if not ruta.exists()]
    if faltan:
        raise SystemExit(f"PARADA: faltan informes: {faltan}")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "STRUCTURAL_TEXT_FEATURES" in marcas:
        print("  la seccion 27 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(TABLA),
            markdown(EL_HALLAZGO),
            markdown(LOS_PERDEDORES),
            markdown(CIERRE),
            codigo(CODIGO_TABLA),
            codigo(CODIGO_HALLAZGO),
            codigo(CODIGO_PERDEDORES),
            codigo(CODIGO_CIERRE),
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 27 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
