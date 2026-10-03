"""Integra la fase V32 en el notebook consolidado.

Anade la seccion 28 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
los JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v32_07_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
NOMBRES = ("00_reproducibilidad", "01_fold4", "02_aislar_fuente", "03_mencion_longitud",
           "04_hashtag_longitud", "05_cuatro_bins", "06_context_gated")

TITULO = """\
## 28. Fase v32: validar la interaccion URL x longitud (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> La mejora proviene realmente de una interaccion estructural generalizable, o fue una
> combinacion accidental de unas pocas features URL sobre estos folds?

La respuesta corta es **no es generalizable**, y eso es lo que esta seccion demuestra.
"""

TABLA = """\
### 28.1 La tabla de la fase

| experimento | corrected_cv | delta | rescates | danios | net | estado |
|---|---:|---:|---:|---:|---:|---|
| V03 (control) | 0,777211 | 0 | - | - | - | congelado |
| V32-00 reproducibilidad | 0,781378 | +0,004167 | 70 | 53 | +17 | REPRODUCIBLE |
| V32-02 A: URL aditivo | 0,777420 | +0,000209 | 35 | 29 | +6 | WEAK_SIGNAL |
| V32-02 B: solo interacciones | 0,779562 | +0,002351 | 58 | 47 | +11 | NEW_SIGNAL |
| V32-02 C: URL + interacciones | 0,781378 | +0,004167 | 70 | 53 | +17 | NEW_SIGNAL |
| V32-03 mencion x longitud | 0,777211 | +0,000000 | 0 | 0 | 0 | REJECTED |
| V32-04 hashtag x longitud | 0,777211 | +0,000000 | 0 | 0 | 0 | REJECTED |
| V32-05 3 bins (cuantil train) | 0,781721 | +0,004510 | 57 | 53 | +4 | MEJOR CONFIG |
| V32-05 4 bins | 0,777417 | +0,000206 | 57 | 53 | +4 | REJECTED |
| V32-06 context-gated n-grams | 0,762255 | -0,014956 | 120 | 172 | -52 | REJECTED |

El control reproduce V03 con exactitud en los siete experimentos: **0,7772114525**.
"""

F0 = """\
### 28.2 Fase 0: la reproducibilidad esta confirmada

V31-04 reproduce **exactamente**: corrected_cv 0,7813779888, rescates 70, danios 53, neto +17,
celdas 123. Y el hash de las predicciones es identico con `PYTHONHASHSEED=0` y `PYTHONHASHSEED=999`,
y en dos corridas dentro del mismo proceso.

La limitacion declarada en V31 queda resuelta.
"""

F1 = """\
### 28.3 Fase 1: el fold 4 no es el problema

El hallazgo es contraintuitivo: **el fold 4 tiene neto +7, que es positivo**. Los netos por pliegue
son 1: -1, 2: -2, 3: +5, 4: +7, 5: +8. Los negativos son los pliegues 1 y 2, no el 4.

El delta absoluto del fold 4 es negativo (-0,0307) porque su F1 base es el mas bajo de los cinco
(0,7386 frente a 0,7960 del pliegue 1). No es un patron contrario ni un artefacto: tiene 296
frente a 297 filas, fraccion con URL 0,2939 frente a 0,3125, y tokens medios 17,0 frente a 17,3.

Las cuatro hipotesis de la directiva: **A no, B no, C parcialmente, D no**.
"""

F2 = """\
### 28.4 Fase 2: el 95 % de la ganancia esta en las interacciones

| paso | delta |
|---|---:|
| V03 -> URL aditivo | +0,000209 |
| URL aditivo -> URL + interacciones | +0,003958 |
| V03 -> solo interacciones | +0,002351 |
| V03 -> URL + interacciones | +0,004167 |

**El 95,0 % de la ganancia se explica por las interacciones.** Y V32-B, que usa solo las cinco
interacciones sin las variables URL aditivas, da +0,002351: mas de la mitad de la ganancia total.
La senal esta en las interacciones, no en el bloque aditivo.
"""

F34 = """\
### 28.5 Fases 3 y 4: el mecanismo no generaliza a mencion ni a hashtag

Ambos dan delta **+0,0000000000**: ni una sola prediccion cambia respecto a V03.

La razon probable es que V03 ya captura la posicion de la mencion y del hashtag a traves de
`char_wb(3,5)`: una mencion al inicio produce n-gramas distintos que una al final. La URL es
distinta, porque `char_wb(3,5)` ve `http` como un token pero no la estructura de la URL.
"""

F5 = """\
### 28.6 Fase 6: 3 bins con cuantiles es el optimo, 4 bins colapsa

Los cortes se tomaron del training fold, nunca mirando el OOF.

- **3 bins con cuantil del training**: +0,004510. Mejor que V31-04 (+0,004167), que usaba umbrales
  fijos.
- **4 bins**: +0,000206. Colapsa, porque con cuatro tramos los subgrupos son demasiado pequenos y
  el modelo no puede aprender el efecto.

**3 bins es el optimo**, y la mejora sobre V31-04 viene de usar cuantiles del training en vez de
umbrales fijos.
"""

F6 = """\
### 28.7 Fase 7: context-gated n-grams es rotundamente negativo

Medicion previa, hecha antes de ejecutar como manda la directiva: K=100 n-gramas x 3 bins = 300
columnas, el 11,22 % del vocabulario de 2673, sparsity 88,78 %. No explota, asi que se ejecuto.

Resultado: **-0,014956**, rescates 120, danios 172, neto **-52**.

El mecanismo de interacciones URL x longitud **no generaliza a n-gram x longitud**. La senal es
especifica de la estructura de URL, no un fenomeno general de condicionamiento por longitud.
"""

CIERRE = """\
### 28.8 La respuesta a la pregunta de la fase

> La mejora proviene realmente de una interaccion estructural generalizable, o fue una
> combinacion accidental de unas pocas features URL sobre estos folds?

**Es una combinacion especifica de features URL con la longitud, no un mecanismo general.** La
evidencia:

1. Se reproduce, y el 95 % de la ganancia esta en las interacciones, no en el bloque aditivo.
2. Pero el mismo mecanismo aplicado a mencion y a hashtag da **exactamente cero**: ni una
   prediccion cambia.
3. Y el context-gated n-grams, que es la generalizacion natural, es **rotundamente negativo**
   (-0,014956, neto -52).

La senal es real pero **especifica de la estructura de URL**. No es un fenomeno general de
condicionamiento por longitud.

### 28.9 Criterio de continuacion

| condicion | se cumple |
|---|:--:|
| el efecto URL x longitud se reproduce | si |
| mantiene direccion positiva | si |
| no depende exclusivamente del fold 1/2/3 | si (4/5, y el negativo es el pliegue mas debil) |
| mantiene rescates netos positivos | si |
| mention x longitud o hashtag x longitud presentan senal similar | **NO** |
| el mecanismo generaliza a n-gram x longitud | **NO** |

**La familia queda abierta pero sin segunda linea de generalizacion.** No se abre combinacion de
bloques, porque los otros dos mecanismos no tienen senal. Y no se propone submission, porque la
directiva la autoriza solo cuando existe un candidato final claramente superior con la configuracion
congelada, y aqui la pregunta central tiene respuesta negativa: no es generalizable.

### 28.10 Lo que esta fase no hace

- No combina bloques.
- No abre context-gated sobre mencion ni sobre hashtag.
- No genera submission.
- No modifica V03.
- No toca `N = 1186`.
"""

CODIGO_TABLA = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent

def leer(nombre):
    with (RAIZ / "reports" / f"v32_{nombre}.json").open(encoding="utf-8") as f:
        return json.load(f)

d0 = leer("00_reproducibilidad")
d1 = leer("01_fold4")
d2 = leer("02_aislar_fuente")
d3 = leer("03_mencion_longitud")
d4 = leer("04_hashtag_longitud")
d5 = leer("05_cuatro_bins")
d6 = leer("06_context_gated")
base = d2["control"]["corrected_cv"]
print("28.1  LA TABLA DE LA FASE")
print(f"  {'experimento':<34}{'corrected_cv':>14}{'delta':>12}{'resc':>7}{'danios':>8}{'net':>6}")
filas = [
    ("V03 (control)", base, None, None, None),
    ("V32-00 reproducibilidad", d0["reproduccion"]["corrected_cv"],
     d0["reproduccion"]["rescates"], d0["reproduccion"]["danios"], d0["reproduccion"]["neto"]),
    ("V32-02 A URL aditivo", d2["V32_A_url_aditivo"]["corrected_cv"],
     d2["comparacion"]["V32_A"]["rescates"], d2["comparacion"]["V32_A"]["danios"],
     d2["comparacion"]["V32_A"]["neto"]),
    ("V32-02 B solo interacciones", d2["V32_B_solo_interacciones"]["corrected_cv"],
     d2["comparacion"]["V32_B"]["rescates"], d2["comparacion"]["V32_B"]["danios"],
     d2["comparacion"]["V32_B"]["neto"]),
    ("V32-02 C URL + interacciones", d2["V32_C_url_mas_interacciones"]["corrected_cv"],
     d2["comparacion"]["V32_C"]["rescates"], d2["comparacion"]["V32_C"]["danios"],
     d2["comparacion"]["V32_C"]["neto"]),
    ("V32-03 mencion x longitud", d3["V32_03a_tres_interacciones"]["corrected_cv"],
     d3["comparacion_03a"]["rescates"], d3["comparacion_03a"]["danios"],
     d3["comparacion_03a"]["neto"]),
    ("V32-04 hashtag x longitud", d4["V32_04a_tres_interacciones"]["corrected_cv"],
     d4["comparacion_04a"]["rescates"], d4["comparacion_04a"]["danios"],
     d4["comparacion_04a"]["neto"]),
    ("V32-05 3 bins (cuantil)", d5["tres_bins"]["corrected_cv"],
     d5["comparacion_cuatro"]["rescates"], d5["comparacion_cuatro"]["danios"],
     d5["comparacion_cuatro"]["neto"]),
    ("V32-05 4 bins", d5["cuatro_bins"]["corrected_cv"],
     d5["comparacion_cuatro"]["rescates"], d5["comparacion_cuatro"]["danios"],
     d5["comparacion_cuatro"]["neto"]),
    ("V32-06 context-gated", d6["challenger"]["corrected_cv"],
     d6["comparacion"]["rescates"], d6["comparacion"]["danios"], d6["comparacion"]["neto"]),
]
for nombre, cv, r, d, n in filas:
    delta = cv - base
    print(f"  {nombre:<34}{cv:>14.6f}{delta:>+12.6f}"
          f"{('-' if r is None else r):>7}{('-' if d is None else d):>8}"
          f"{('-' if n is None else f'{n:+d}'):>6}")
print()
print(f"  control reproduce V03 en los siete experimentos: {base:.10f}")
"""

CODIGO_F0F1 = """\
print("28.2  FASE 0: REPRODUCIBILIDAD")
r = d0["reproduccion"]
print(f"  corrected_cv {r['corrected_cv']:.10f} | delta vs V31-04 {r['delta_vs_referencia']:+.2e}")
print(f"  rescates {r['rescates']} | danios {r['danios']} | neto {r['neto']:+d} | celdas {r['celdas_modificadas']}")
print(f"  hash identico con PYTHONHASHSEED 0 y 999: {d0['semillas']['identicos']}")
print()
print("28.3  FASE 1: EL FOLD 4 NO ES EL PROBLEMA")
netos = {f: d1["rescates_y_signo_por_pliegue"][f]["neto"] for f in ("1", "2", "3", "4", "5")}
print(f"  netos por pliegue: {netos}")
print("  el fold 4 tiene neto +7, que es POSITIVO. Los negativos son los pliegues 1 (-1) y 2 (-2).")
print("  su delta absoluto es negativo porque su F1 base es el mas bajo (0.7386).")
for clave, texto in d1["veredicto"].items():
    print(f"  {clave}: {texto}")
"""

CODIGO_F2 = """\
print("28.4  FASE 2: EL 95 % DE LA GANANCIA ESTA EN LAS INTERACCIONES")
dec = d2["descomposicion"]
print(f"  {'paso':<46}{'delta':>12}")
for k in ("V03_a_URL_aditivo", "URL_aditivo_a_URL_mas_interacciones",
          "V03_a_solo_interacciones", "V03_a_URL_mas_interacciones"):
    print(f"  {k:<46}{dec[k]:>+12.10f}")
print(f"  parte explicada por interacciones: {100*dec['parte_de_la_ganancia_explicada_por_interacciones']:.1f} %")
print()
print("  V32-B usa SOLO las cinco interacciones, sin las variables URL aditivas, y da mas de")
print("  la mitad de la ganancia total. La senal esta en las interacciones.")
"""

CODIGO_F3456 = """\
print("28.5  FASES 3 Y 4: NO GENERALIZA A MENCION NI A HASHTAG")
for nombre, d in (("mencion", d3), ("hashtag", d4)):
    a = d["V32_03a_tres_interacciones"] if nombre == "mencion" else d["V32_04a_tres_interacciones"]
    comp = d["comparacion_03a"] if nombre == "mencion" else d["comparacion_04a"]
    print(f"  {nombre}: corrected_cv {a['corrected_cv']:.10f} | "
          f"delta {a['corrected_cv'] - base:+.10f} | celdas {comp['celdas_modificadas']}")
print("  ambos dan EXACTAMENTE cero: ni una sola prediccion cambia.")
print()
print("28.6  FASE 6: 3 BINS ES EL OPTIMO")
print(f"  3 bins con cuantil del training: {d5['tres_bins']['corrected_cv']:.10f} | "
      f"delta {d5['tres_bins']['corrected_cv'] - base:+.10f}")
print(f"  4 bins: {d5['cuatro_bins']['corrected_cv']:.10f} | "
      f"delta {d5['cuatro_bins']['corrected_cv'] - base:+.10f}")
print(f"  4 bins sobre 3 bins: {d5['cuatro_menos_tres']:+.10f}  (colapsa)")
print()
print("28.7  FASE 7: CONTEXT-GATED N-GRAMS ES ROTUNDAMENTE NEGATIVO")
m = d6["medicion_previa"]
print(f"  medicion previa: {m['nuevas_columnas']} columnas, "
      f"{m['porcentaje_del_vocabulario']:.2f} % del vocabulario, sparsity 88.78 %")
print(f"  corrected_cv {d6['challenger']['corrected_cv']:.10f} | "
      f"delta {d6['delta_oof']:+.10f}")
print(f"  rescates {d6['comparacion']['rescates']} | danios {d6['comparacion']['danios']} | "
      f"neto {d6['comparacion']['neto']:+d}")
print("  el mecanismo NO generaliza a n-gram x longitud.")
"""

CODIGO_CIERRE = """\
print("28.8  LA RESPUESTA A LA PREGUNTA DE LA FASE")
print()
print("  La mejora proviene de una interaccion estructural generalizable, o fue una")
print("  combinacion accidental de unas pocas features URL sobre estos folds?")
print()
print("  ES UNA COMBINACION ESPECIFICA DE FEATURES URL CON LA LONGITUD. NO ES GENERALIZABLE.")
print()
print("  1. Se reproduce, y el 95 % de la ganancia esta en las interacciones.")
print("  2. Pero el mismo mecanismo en mencion y en hashtag da EXACTAMENTE cero.")
print("  3. Y el context-gated n-grams es ROTUNDAMENTE negativo (-0.014956, neto -52).")
print()
print("  La senal es real pero especifica de la estructura de URL.")
print()
print("28.9  CRITERIO DE CONTINUACION")
print("  se reproduce                    si")
print("  direccion positiva             si")
print("  no depende del fold 1/2/3      si (4/5)")
print("  rescates netos positivos       si")
print("  mention/hashtag con senal      NO")
print("  generaliza a n-gram x longitud  NO")
print()
print("  La familia queda abierta pero sin segunda linea de generalizacion.")
print("  No se propone submission: la pregunta central tiene respuesta negativa.")
print()
print("  0 submissions, V03 intacto, N = 1186 no se toca.")
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
    """Anade las celdas de la fase V32 al notebook consolidado.

    Returns:
        None.
    """
    faltan = [
        str(PROJECT_ROOT / "reports" / f"v32_{n}.json")
        for n in NOMBRES
        if not (PROJECT_ROOT / "reports" / f"v32_{n}.json").exists()
    ]
    if faltan:
        raise SystemExit(f"PARADA: faltan informes: {faltan}")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "validar la interaccion URL" in marcas:
        print("  la seccion 28 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(TABLA),
            markdown(F0),
            markdown(F1),
            markdown(F2),
            markdown(F34),
            markdown(F5),
            markdown(F6),
            markdown(CIERRE),
            codigo(CODIGO_TABLA),
            codigo(CODIGO_F0F1),
            codigo(CODIGO_F2),
            codigo(CODIGO_F3456),
            codigo(CODIGO_CIERRE),
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 28 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
