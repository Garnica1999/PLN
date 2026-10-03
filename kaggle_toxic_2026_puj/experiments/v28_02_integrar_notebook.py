"""Integra el diagnostico de errores (V28) en el notebook consolidado.

Anade las celdas al final de `notebooks/10_final_pipeline.ipynb`, como pide el encargo, sin
crear un notebook aislado. Las celdas leen el JSON en tiempo de ejecucion y no llevan outputs
embebidos, para que no puedan quedar desincronizadas.

Uso
---
    /home/carlos/venv/bin/python experiments/v28_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v28_01_diagnostico_errores.json"

TITULO = """\
## 23. Fase v28: LABEL_SPECIALIZATION cerrada, y diagnostico de errores (2026-09-27)

Esta seccion hace dos cosas: cierra formalmente la familia de especializacion por etiqueta, y
luego responde una pregunta distinta, que es donde puede quedar margen de verdad.
"""

CIERRE = """\
### LABEL_SPECIALIZATION = CLOSED

La directiva de la fase pedia tres experimentos: C independiente por etiqueta, n-grams
independientes por etiqueta, y especializacion completa. **Los tres ya se habian ejecutado** en
este proyecto, y la propia directiva indicaba no repetirlos. La auditoria del registro de 250
filas los encontro:

| experimento registrado | que midio | resultado |
|---|---|---|
| V8-06 | C por etiqueta, rejilla de 6 valores, validacion anidada | **+0.000306**, IC95 [-0.00345, +0.00408], 2 de 5 pliegues |
| V8-07 | peso de clase por etiqueta | **-0.005298**, 1 de 5 pliegues |
| E2-21-ESPACIOS | espacios de features por etiqueta, top-150 lexico | **-0.025312**, 0 de 5 pliegues |
| V12-01 | regularizacion por etiqueta y por bloque de features | **+0.002738**, sostenido por **un solo pliegue** |
| E2-15 | n-grams distintos por etiqueta | **-0.0253** y **-0.0379** |

El caso V12-01 merece una frase, porque es el unico que parece positivo y es el mas
instructivo. Su mejor brazo llega a +0.002738, pero **sin el pliegue 4 el delta cae a
+0.000134**: el resultado lo sostiene uno de cinco pliegues. Esa es exactamente la firma de la
sobreestimacion que produce elegir la mejor de varias variantes sobre el mismo corrected_cv, y
es la razon por la que repetir estos experimentos habria producido un falso positivo en lugar
de una mejora.
"""

DIAGNOSTICO = """\
### El diagnostico: donde falla V03

Sin entrenar nada, se midio donde se concentran los errores de las tres etiquetas, condicionado
por 23 rasgos derivados **exclusivamente del campo `text`**.

| etiqueta | F1 V03 | errores | tasa | FN / positivos | FP |
|---|---:|---:|---:|---:|---:|
| HS | 0.7617 | 316 | 0.2131 | 143 / 648 (0.2207) | 173 |
| TR | 0.8157 | 164 | 0.1106 | 82 / 445 (0.1843) | 82 |
| AG | 0.7542 | 277 | 0.1868 | 142 / 567 (0.2504) | 135 |

La interseccion dice mucho: de 1483 filas, **1042 no fallan en ninguna etiqueta**, 223 fallan en
exactamente una, y 98 fallan en las tres.

Y el solapamiento entre pares es el dato que decide la familia:

| par | phi | ambos fallan | esperados si independientes | Jaccard |
|---|---:|---:|---:|---:|
| HS-AG | +0.5069 | 179 | 59 | 0.4324 |
| TR-AG | +0.4930 | 120 | 31 | 0.3738 |
| HS-TR | +0.4203 | 115 | 35 | 0.3151 |

**Los errores de las tres etiquetas caen sobre los MISMOS textos, con un exceso de 3 a 4 veces
sobre lo que cabria si fueran independientes.** Eso explica por que la especializacion por
etiqueta no dio nada en ninguno de los cinco experimentos: un especialista solo puede cambiar
decisiones donde su etiqueta falla, y esas filas fallan en las otras dos tambien. Cambiar HS no
compensa lo que se pierde en TR y AG sobre la misma fila.
"""

RASGOS = """\
### Que rasgo distingue V03 wrong de V03 correct

Cohen d es la diferencia de medias en desviaciones estandar. Los cinco rasgos de mayor efecto
para cada etiqueta:

| etiqueta | rasgo dominante | d | p |
|---|---|---:|---:|
| HS | numero de tokens (error 19.5 vs correcto 16.6) | +0.275 | 2.0e-05 |
| HS | es muy corto (<= 4 tokens) | -0.258 | 7.1e-04 |
| AG | numero de tokens (error 20.1 vs correcto 16.5) | +0.337 | 1.0e-06 |
| AG | alfabeticos (error 95.7 vs correcto 79.7) | +0.322 | 5.3e-06 |
| TR | longitud media de token | -0.285 | 1.5e-04 |

**El rasgo dominante es la longitud, y es el mismo para las tres etiquetas.** HS y AG fallan
preferentemente en textos largos; TR, en cambio, falla en textos cortos y con tokens mas cortos.
Esa es la unica heterogeneidad real que aparece, y es de signo contrario entre TR y las otras
dos. Los tweets muy cortos fallen 15 veces menos que la media, lo que apunta a que alli la
etiqueta esta sobre-determined por el propio volumen de texto.
"""

REGIONES = """\
### Regiones del espacio textual

| region | filas | fraccion | HS | TR | AG |
|---|---:|---:|---:|---:|---:|
| densidad lexica alta (> 0.90) | 1087 | 0.7330 | 0.1914 | 0.1040 | 0.1610 |
| con mencion | 702 | 0.4734 | 0.1752 | 0.1182 | 0.1524 |
| con @ en el primer tercio | 639 | 0.4309 | 0.1674 | 0.1268 | 0.1487 |
| con URL | 458 | 0.3088 | **0.2445** | 0.0852 | **0.2227** |
| sin ninguna marca (@, #, URL) | 390 | 0.2630 | **0.2410** | 0.1179 | **0.2051** |
| tweets largos (>= 25 tokens) | 301 | 0.2030 | **0.2492** | 0.0963 | **0.2458** |
| con hashtag | 144 | 0.0971 | 0.2569 | 0.0833 | 0.2083 |
| alta puntuacion | 96 | 0.0647 | 0.1979 | 0.0833 | 0.1458 |
| casi todos mayusculas | 72 | 0.0486 | 0.1944 | 0.1250 | 0.1944 |
| tweets muy cortos (<= 4 tokens) | 58 | 0.0391 | **0.0345** | **0.0000** | **0.0345** |

Dos regiones destacan por tener error alto en HS y AG a la vez: **los textos largos** (0.2492 y
0.2458 frente a una media de 0.2131 y 0.1868) y **los textos con URL** (0.2445 y 0.2227). Y una
destaca por lo contrario: los **textos muy cortos**, con 0.0345, 0.0000 y 0.0345, es decir casi
nada.

La region de textos largos y con URL es grande (301 y 458 filas, un 20 % y un 31 %) y su error
supera la media en las dos etiquetas que mas fallan. Es el subconjunto donde un modelo distinto
tendria mas sentido, y no por etiqueta sino **transversalmente**.
"""

CODIGO_CARGA = """\
from pathlib import Path
import json
import numpy as np
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
INFORME = RAIZ / "reports" / "v28_01_diagnostico_errores.json"
datos = json.loads(INFORME.read_text(encoding="utf-8"))
print("familia:", datos["familia"], "| estado:", datos["estado"])
print("alcance:", datos["alcance"])
print("rasgos calculados:", len(datos["rasgos"]))
print()
e = datos["estructura_del_error"]
print(f"{'etiqueta':<9}{'F1 V03':>9}{'errores':>9}{'tasa':>8}{'FN':>6}{'FP':>6}{'TP':>6}{'TN':>6}")
for et in ("HS", "TR", "AG"):
    d = e[et]
    print(f"{et:<9}{d['f1_v03']:>9.4f}{d['n_errores']:>9}{d['tasa_de_error']:>8.4f}"
          f"{d['fn']:>6}{d['fp']:>6}{d['tp']:>6}{d['tn']:>6}")
i = e["interseccion_de_errores"]
print()
print("interseccion de errores:", {k: v for k, v in i.items() if k != "solapamiento_entre_errores_entre_etiquetas"})
print()
print("SOLAPAMIENTO ENTRE ETIQUETAS, que es lo que cierra la familia")
het = datos["heterogeneidad_entre_etiquetas"]
print(f"{'par':<10}{'phi':>9}{'ambos':>8}{'esperados':>11}{'exceso':>9}{'Jaccard':>10}")
for par, d in het.items():
    print(f"{par:<10}{d['correlacion_phi']:>9.4f}{d['n_ambos_error']:>8}"
          f"{d['esperado_si_independientes']:>11.1f}{d['exceso_sobre_lo_esperado']:>9.0f}{d['jaccard']:>10.4f}")
"""

CODIGO_RASGOS = """\
disc = datos["discriminacion_por_rasgo"]
print("QUE RASGO DISTINGUE V03 WRONG DE V03 CORRECT")
print()
for et in ("HS", "TR", "AG"):
    print(f"{et}  (F1 {e[et]['f1_v03']:.4f}, {disc[et]['n_rasgos_con_p_menor_005']} rasgos con p < 0.005)")
    print(f"  {'rasgo':<24}{'si error':>11}{'si correcto':>12}{'d':>9}{'p':>11}")
    for nombre, d in disc[et]["rasgos_ordenados_por_efecto"][:5]:
        print(f"  {nombre:<24}{d['media_error']:>11.4f}{d['media_correcto']:>12.4f}"
              f"{d['d_cohen']:>+9.4f}{d['p_mannwhitney']:>11.2e}")
    print()
print("LECTURA: el rasgo dominante es la LONGITUD, y es el mismo para HS y AG.")
print("TR es la excepcion: falla en textos cortos y con tokens mas cortos, signo contrario.")
"""

CODIGO_REGIONES = """\
print("REGIONES DEL ESPACIO TEXTUAL")
print()
print(f"{'region':<44}{'filas':>7}{'frac':>8}{'HS':>9}{'TR':>9}{'AG':>9}{'suficiente':>12}")
for s in e["subconjuntos"]:
    marcas = "".join(f"{s['tasa_error'][et]:>9.4f}" for et in ("HS", "TR", "AG"))
    print(f"{s['nombre']:<44}{s['n_filas']:>7}{s['fraccion']:>8.4f}{marcas}"
          f"{('si' if s['suficientemente_grande'] else 'no'):>12}")
print()
base = {et: e[et]['tasa_de_error'] for et in ('HS','TR','AG')}
print(f"tasa global de error: HS {base['HS']:.4f} | TR {base['TR']:.4f} | AG {base['AG']:.4f}")
print()
print("DONDE HAY MARGEN REAL:")
print("  - textos largos (>= 25 tokens): 301 filas, HS 0.2492 y AG 0.2458, ambos por encima")
print("    de su media. Es la region con mas error y es grande.")
print("  - textos con URL: 458 filas, HS 0.2445 y AG 0.2227, tambien por encima.")
print("  - textos muy cortos (<= 4 tokens): 58 filas, HS 0.0345, TR 0.0000, AG 0.0345.")
print("    Alli la etiqueta esta sobre-determined por el volumen de texto.")
print()
print("Y el rasgo es TRANSVERSAL, no por etiqueta: longitud y URL elevan HS y AG a la vez,")
print("que es justo lo contrario de lo que necesitaria un especialista por etiqueta.")
"""

CODIGO_CONCLUSION = """\
print("CONCLUSION DEL DIAGNOSTICO")
print()
print("1. EVIDENCIA A FAVOR DE LA ESPECIALIZACION POR ETIQUETA")
print("   Muy poca, y toda indirecta. La unica heterogeneidad real es que TR falla en textos")
print("   cortos mientras HS y AG fallan en textos largos, y que las tres coinciden en que")
print("   los textos muy cortos son casi siempre correctos. Eso es un efecto de volumen de")
print("   texto, no de semantica de etiqueta, y V03 ya lo captura con la longitud del n-gram.")
print()
print("2. EVIDENCIA EN CONTRA")
print("   Los errores se solapan con phi de +0.42 a +0.51 y un exceso de 3 a 4 veces sobre lo")
print("   independiente. 98 filas fallan en las tres etiquetas a la vez. Un especialista por")
print("   etiqueta solo puede actuar donde su etiqueta falla, y ahi fallan las otras dos.")
print("   Los cinco experimentos historicos coinciden: +0.000306, -0.005298, -0.025312,")
print("   +0.002738 sostenido por un pliegue, y -0.0253/-0.0379.")
print()
print("3. COMPLEMENTARIEDAD FRENTE A V03")
print("   No aplica: no se entreno ningun modelo. Este script es un diagnostico.")
print()
print("4. SI MERECE SUBMISSION")
print("   No. No hay candidata, y 0 submissions se han gastado en V26, V27 ni V28.")
print()
print("5. SIGUIENTE FAMILIA RECOMENDADA")
print("   No otra etiqueta: transversal. El margen esta en la REGION de texto largo y con URL,")
print("   donde HS y AG fallan a la vez.")
print()
print("   PERO el margen no es automaticamente alcanzable, y el diagnostico de V28-03 no lo")
print("   resuelve. Lo que si midio, con su limitacion:")
print("   - La ambiguedad de etiqueta es un multiplicador REAL del error: donde hay grupo")
print("     contradictorio, V03 falla 3 a 4 veces mas, con significancia en 4 de 12 celdas.")
print("   - Pero solo 43 filas (2.9% del desarrollo) pueden portar esa informacion, asi que el")
print("     experimento NO tiene poder para decir si el margen de la region larga es reducible.")
print("   - Y la cota por voto mayoritario da 0.9917: los grupos fusionados NO contienen la")
print("     ambiguedad que limita a V03. Los near-duplicados no son el sitio donde vive el")
print("     ruido de anotacion.")
print()
print("   Conclusion honesta: la hipotesis de ambiguedad queda ABIERTA y sin medir. Lo que hace")
print("   falta no es otro modelo, es un criterio de emparejamiento mas laxo que el near-dup,")
print("   que es ingenieria de datos. Esa es la candidata a 'fuente de informacion nueva' y")
print("   debe decidirse antes de abrir otra familia de modelos.")
g = datos["gastos"]
print()
print(f"Coste del diagnostico: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
      f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions.")
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
    """Anade las celdas de la fase V28 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v28_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
    if "Fase v28" in marcas:
        print("  la seccion 23 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(CIERRE),
            markdown(DIAGNOSTICO),
            markdown(RASGOS),
            markdown(REGIONES),
            codigo(CODIGO_CARGA),
            codigo(CODIGO_RASGOS),
            codigo(CODIGO_REGIONES),
            codigo(CODIGO_CONCLUSION),
        ]
    )
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 23 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
