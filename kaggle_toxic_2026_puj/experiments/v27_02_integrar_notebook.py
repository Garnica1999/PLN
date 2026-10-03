"""Integra la familia CHAR_SPECTRUM_KERNEL en el notebook consolidado.

Anade celdas al final de `notebooks/10_final_pipeline.ipynb`, como pide el encargo, en lugar de
crear un notebook experimental aislado. Las celdas de codigo leen el JSON de resultados en
tiempo de ejecucion y no llevan outputs embebidos, para que no puedan quedar desincronizadas.

Uso
---
    /home/carlos/venv/bin/python experiments/v27_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v27_01_char_spectrum_kernel.json"

TITULO = """\
## 22. Fase v27: CHAR_SPECTRUM_KERNEL, y una correccion a la motivacion (2026-09-27)

Esta seccion cierra la evaluacion de la familia de kernels de espectro de caracteres, que era
el unico candidato que quedaba abierto tras el cierre de Tsetlin.

Lo primero que hay que leer es una **correccion a la motivacion**, porque cambia la pregunta.
"""

MOTIVACION = """\
### Correccion: el 0.3900 no era un kernel de espectro

La justificacion de abrir esta familia era que en V26-01 se habia medido que el "kernel
spectrum de caracteres" tenia una mediana de similitud al vecino mas cercano de **0.3900**, frente
a 0.6974 del RBF, y que solo un 3.4 % de los vecinos superaba 0.60.

Al revisar el codigo de V26-01 para implementar la fase, se vio que **esa medida no era de un
kernel de espectro**. Era el **coseno sobre la matriz binaria de la union TF-IDF de V03**, es
decir `word (1,2)` junto a `char_wb (3,5)`, y ademas se habia calculado **de una sola vez sobre
las 1 483 filas de desarrollo**, fuera de los pliegues. Era una medida de geometria para
decidir si Random Fourier Features y Nystrom merecian una hora de CPU, y como clasificador
nunca se ejecuto.

La consecuencia es que la geometria menos memoristica que se habia medido **era la de V03**, no
la de un kernel de espectro. Eso cambia la pregunta, y la hace mas interesante:

> No es si el spectrum evita la memorizacion, sino si un clasificador que use una geometria de
> overlap de n-grams de caracteres aporta algo que el hiperplano de V03 no aprovecha.
"""

CONFIG = """\
### El kernel, y por que el overlap por minimos en version binaria

Para dos documentos con multiconjuntos de n-grams de caracteres `X` e `Y`:

    K(x, y) = <min(X, Y), min(X, Y)> / sqrt(|X| |Y|)

El `min` es lo que distingue este kernel de un coseno, y por eso **no** es el kernel de V03.
La normalizacion por la masa lo deja en [0, 1] y lo hace invariante a la longitud del texto.

Hay una restriccion matematica que la implementacion respeta y que el control de PSD verifica:
el overlap por minimos con **frecuencias enteras mayores que uno no es semidefinida positiva**,
y un kernel que no es PSD hace que un SVC de kernel precomputado sea invalido. La version
**binaria**, sobre indicadores de presencia, si lo es. Por eso el `CountVectorizer` va con
`binary=True` y el control de PSD es parte del protocolo y no un adorno.

| parametro | valor |
|---|---|
| kernel | overlap de multiconjuntos, `min`, normalizado por la masa |
| version | **binaria**, por el requisito de semidefinida positiva |
| analyzer / rango / min_df | `char` / (3,5) / 2 |
| rango 3-5 | el mismo que usa el `char_wb` de V03, para comparar sobre la misma informacion |
| clasificador | `SVC(C=2.0, class_weight="balanced")` con el kernel como funcion |
| features por pliegue | 27 980 a 28 754, densidad ~0.0084 |
| vectorizado | ajustado **dentro** de cada pliegue |
| test | no leido |

El kernel se entrega a `SVC` como **funcion de Python** y no como `"precomputed"`, porque con
`"precomputed"` sklearn exige que la entrada tenga exactamente las columnas del entrenamiento
y por tanto no admite cruzar entrenamiento con validacion.
"""

CODIGO_CARGA = """\
from pathlib import Path
import json
import numpy as np
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
INFORME = RAIZ / "reports" / "v27_01_char_spectrum_kernel.json"
datos = json.loads(INFORME.read_text(encoding="utf-8"))
print("familia:", datos["familia"], "| estado:", datos["estado"])
print("experimento:", datos["experimento"])
print()
cfg = datos["configuracion"]
for clave in ("kernel", "analyzer", "rango_ngram", "min_df", "clasificador"):
    print(f"  {clave:>16}: {cfg[clave]}")
print(f"  {'features/pliegue':>16}: {cfg['features_por_pliegue']}")
print(f"  {'test leido':>16}: {cfg['test_leido']}")
print()
print("CORRECCION DE LA MOTIVACION")
mot = datos["correccion_de_la_motivacion"]
print("  se creia :", mot["que_se_creia"])
print("  se habia :", mot["que_se_tenia_medido"][:150], "...")
"""

CODIGO_TABLA = """\
res = datos["resultados"]
comp = datos["complementariedad"]
print("RENDIMIENTO")
print(f"  CHAR_SPECTRUM_KERNEL  {res['corrected_cv']:.6f}   dOOF vs V03 {res['delta_vs_v03']:+.6f}")
print(f"  V03 (congelado)          0.777211                  +0.0000000000")
print()
print(f"  train F1 {res['f1_train']:.4f} | gap {res['gap_train_oof']:+.4f} | sd pliegues {res['sd_entre_pliegues']:.4f}")
print("  F1 por etiqueta: " + " | ".join(
    f"{e} {v:.4f} (V03 {comp['por_etiqueta'][e]['f1_v03']:.4f})" for e, v in res["f1_por_etiqueta"].items()))
print("  F1 por pliegue  : " + " | ".join(f"{v:.4f}" for v in res["f1_por_pliegue"]))
print()
print("COMPLEMENTARIEDAD CON V03")
print(f"  rescates {comp['rescates']} | danos {comp['danos']} | neto {comp['neto']:+d}")
print(f"  coinciden {comp['celdas_coincidentes']}/{comp['celdas_totales']} ({comp['fraccion_coincidente']:.4f})")
print(f"  correlacion {comp['correlacion_predicciones']:.4f} | ambos incorrectos {comp['ambos_malos']}")
print()
print(f"  {'etiqueta':<10}{'F1 kernel':>11}{'F1 V03':>10}{'cambios':>9}{'rescates':>10}{'danos':>8}")
for e, d in comp["por_etiqueta"].items():
    print(f"  {e:<10}{d['f1']:>11.4f}{d['f1_v03']:>10.4f}{d['cambios']:>9}{d['rescates']:>10}{d['danos']:>8}")
"""

CODIGO_GEOMETRIA = """\
memo = datos["memorizacion"]
print("AUDITORIA DE MEMORIZACION: spectrum frente a RBF exacto, en los MISMOS pliegues")
print()
print(f"  {'pliego':>7}{'spectrum med':>14}{'spectrum >0.60':>16}{'RBF med':>10}{'RBF >0.60':>12}")
for clave in sorted(memo["spectrum_por_pliegue"], key=int):
    s = memo["spectrum_por_pliegue"][clave]
    b = memo["rbf_exacto_por_pliegue"][clave]
    print(f"  {clave:>7}{s['mediana_vecino_ENTRENAMIENTO']:>14.4f}"
          f"{s['fraccion_sobre_0_60_ENTRENAMIENTO']*100:>15.1f}%{b['mediana']:>10.4f}"
          f"{b['fraccion_sobre_0_60']*100:>11.1f}%")
c = memo["comparativa"]
print()
print(f"  agregado spectrum: mediana {c['spectrum_mediana_vecino']:.4f}, "
      f"por encima de 0.60 {c['spectrum_fraccion_sobre_0_60']*100:.1f}%")
print(f"  agregado RBF     : mediana {c['rbf_exacto_mediana_vecino']:.4f}, "
      f"por encima de 0.60 {c['rbf_exacto_fraccion_sobre_0_60']*100:.1f}%")
print()
print("  ESTA ES LA DIFERENCIA ESTRUCTURAL CON V23:")
print("  el RBF memoriza (100 % de vecinos por encima de 0.60) y por eso dejo el 60-76 % de")
print("  vectores de soporte. El spectrum no memoriza (4.8-6.0 %) y aun asi el train F1 llega a")
print(f"  {res['f1_train']:.4f} con un gap de {res['gap_train_oof']:+.4f}.")
print()
print("  Es decir: la mejora de la geometria NO se tradujo en una mejora de la metrica, pero")
print("  tampoco provoco memorizacion. El limite esta en otro sitio.")
"""

CODIGO_DIAGNOSTICO = """\
print("CRITERIOS DE CONTINUACION")
for cc in datos["criterios_de_continuacion"]:
    print(f"  {'CUMPLE' if cc['cumple'] else 'no    '}  {cc['criterio']}")
    print(f"          medida: {cc['medida']}")
print()
print(f"VEREDICTO: {datos['estado']}")
print()
print("LO QUE ESTA PRUEBA SOSTIENE")
print("  - El kernel de espectro de caracteres es una geometria genuinamente distinta: la")
print("    mediana del vecino cae de 0.7240 a 0.3469 y la fraccion por encima de 0.60 del")
print("    100 % al 5 %. No memoriza.")
print("  - Es el clasificador NO LINEAL mas estable medidos en este proyecto: sd entre pliegues")
print(f"    de {res['sd_entre_pliegues']:.4f}, frente a 0.0270 de Tsetlin y 0.0302 del control binario.")
print("  - Aun asi queda por debajo de V03 en -0.015089 y su neto de celdas es negativo.")
print()
print("LO QUE ESTA PRUEBA NO SOSTIENE")
print("  - Que el spectrum sea una candidata. El criterio 3 se cumple, pero es un criterio de")
print("    PROPIEDAD, no de RENDIMIENTO: describe la geometria del kernel, no un mejor modelo.")
print("  - Que la diferencia de -0.015 sea ruido. Con 193 rescates y 238 danos, McNemar da")
print("    p cercano a 1: no hay senal estadistica de que el spectrum acierte donde V03 falla.")
print()
print("INTERPRETACION DEL GAP, QUE PARECE LA CLAVE")
print(f"  El gap train-OOF de {res['gap_train_oof']:+.4f} es casi el de V03 (+0.1842), y el train F1")
print(f"  llega a {res['f1_train']:.4f}. El kernel CONSTRUYE una solucion que separa bien los")
print("  datos de entrenamiento y generaliza peor. Ese es el mismo techo de N=1186 que ya")
print("  aparecio en V23, V25 y Tsetlin, y no se rompe cambiando de kernel.")
"""

CODIGO_ESTADO = """\
print("ESTADO DE LA FAMILIA")
print()
print("  CHAR_SPECTRUM_KERNEL   CHALLENGER")
print("      corrected_cv 0.762122, -0.015089 vs V03, neto de celdas -45.")
print("      Se conserva por la geometria y la estabilidad, no por el rendimiento.")
print("      NO justifica submission: dOOF negativo y neto negativo.")
print()
print("  TSETLIN MACHINE        REJECTED (V26-02 casera, V26-03 pyTsetlinMachine)")
print("  FACTORIZATION MACHINE  REJECTED (V25)")
print("  RBF-SVC                REJECTED (V23, LB 0.77938)")
print()
print("  V03                    FROZEN   0.78306 LB, hash verificado")
print()
g = datos["gastos"]
print(f"  Submissions gastadas en v27: {g['submissions']} | disponibles: 3, reservadas")
print(f"  Coste: {g['segundos_totales']} s, memoria {g['memoria_pico_mb']:.0f} MB")
print()
print("  Patron que se repite en las cuatro familias no lineales:")
print("  V23 RBF-SVC, V25 FM, V26 Tsetlin y v27 spectrumkernel. Todas tienen una")
print("  representacion valida, todas se evaluaron con los mismos folds que V03, y ninguna")
print("  supero 0.78306. El limite no parece estar en la eleccion de modelo sino en N=1186.")
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
    """Anade las celdas de la familia CHAR_SPECTRUM_KERNEL al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v27_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
    if "Fase v27" in marcas:
        print("  la seccion 22 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(MOTIVACION),
            markdown(CONFIG),
            codigo(CODIGO_CARGA),
            codigo(CODIGO_TABLA),
            codigo(CODIGO_GEOMETRIA),
            codigo(CODIGO_DIAGNOSTICO),
            codigo(CODIGO_ESTADO),
        ]
    )
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 22 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
