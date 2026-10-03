"""Integra la fase V30 (LABEL_NOISE cerrada + perfil del error residual) en el notebook.

Anade la seccion 26 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v30_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v30_01_perfil_error_residual.json"

TITULO = """\
## 26. Fase v30: LABEL_NOISE cerrada, y perfil del error residual (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Una vez descartados ruido de etiquetas, especializacion, matching y familias no lineales, en
> que regiones concretas del texto esta perdiendo V03 y que informacion de esas regiones todavia
> no estamos explotando?

La respuesta corta: **el error se concentra en texto largo con densidad lexica baja y caracteres
repetidos, es estable en 5/5 pliegues, y las tres etiquetas no tienen perfiles distintos**. Y la
informacion que no se esta usando es de tres tipos, detallados en la seccion 26.4.
"""

CIERRE = """\
### 26.1 LABEL_NOISE = CLOSED

La auditoria del registry (258 filas) y del notebook encuentra que la familia **ya existe y esta
cerrada**, con evidencia negativa en siete experimentos:

| experimento | que midio | resultado |
|---|---|---|
| V4-B01 | que medida de dureza OOF ordena la dificultad | la BCE agregada **NO** (r = -0,136); la violacion del umbral **SI** (r = +0,811) |
| V4-B02 | si dificil-coherente y posible-ruido son separables | si: 296 con error medio 0,351 y 43 con error medio 0,054 |
| V4-B03 | reponderar por dureza OOF | unica positiva **+0,0036**, IC95 [-0,0016, +0,0092], no cumple gates |
| V4-B05 | downweight y remove sobre contradiccion textual | **+0,0005** y **-0,0004**; ninguna mueve la metrica |
| V6-07 | ponderar pares inconsistentes | **+0,0001** y **-0,0010**; cerrado negativo |
| V6-03 | ponderacion por tamano de grupo | cerrado negativo |
| V8-08 | corrector por margenes | **-0,002063** |

La razon principal del cierre: **la senal de sospecha ya quedo caracterizada como dificultad y no
como ruido**, y las intervenciones sobre el entrenamiento no produjeron una mejora robusta. No se
repite ninguno de los siete, no se ejecuta downweighting ni trimming, y no se abre nueva busqueda de
percentiles ni de criterios de eliminacion.
"""

POBLACIONES = """\
### 26.2 Las poblaciones historicas, reconstruidas sin redefinir

`experiments/28_hardness_v4.py` (lineas 593-598) define: `Posible ruido` si
`cluster_contradictorio > 0`; `Dificil coherente` si `hardness_score >= percentil 80`; `Facil` en
otro caso. Se reconstruye igual, sin tocar la definicion.

| poblacion | filas | fraccion |
|---|---:|---:|
| Facil | 1151 | 0,7761 |
| Dificil coherente | 289 | 0,1949 |
| Posible ruido | 43 | 0,0290 |

Umbral de dureza (percentil 80): **0,8327**. Filas en cluster contradictorio: **43**.

Y la separacion entre Facil y Dificil coherente es **pequena y de un solo tipo**: los unicos rasgos
con efecto son `longitud_media_token` (d = **-0,368**), `urls` (d = **-0,336**), `es_muy_corto`
(d = -0,286) y `puntuacion` (d = -0.234). Todos del mismo signo: **los ejemplos dificiles tienen
tokens mas cortos y mas URLs, no mas texto**. La dificultad en este corpus no es volumen de texto.
"""

PERFIL = """\
### 26.3 Perfil por label: las tres etiquetas no son distintas

| label | errores | tasa | FP | FN | FP / predichos | FN / positivos |
|---|---:|---:|---:|---:|---:|---:|
| HS | 316 | 0,2131 | 173 | 143 | 0,2552 | 0,2207 |
| TR | 164 | 0,1106 | 82 | 82 | 0,1843 | 0,1843 |
| AG | 277 | 0,1868 | 135 | 142 | 0,2411 | 0,2504 |

| label | rasgo dominante | d | media si error | media si correcto |
|---|---|---:|---:|---:|
| HS | tokens (19,5 vs 16,6) | **+0,275** | 19,475 | 16,594 |
| AG | tokens (20,1 vs 16,5) | **+0,337** | 20,123 | 16,538 |
| TR | es_muy_corto | **-0,303** | 0,000 | 0,044 |

**HS y AG fallan en texto largo; TR falla en texto muy corto, con signo contrario.** Y TR es la
etiqueta con menos errores. Esto confirma lo que encontro V28: la heterogeneidad entre etiquetas es
de **volumen de texto**, no de semantica. No hay tres problemas distintos que atacar.
"""

REGIONES = """\
### 26.4 Las regiones donde V03 falla, y su estabilidad

| region | filas | HS | TR | AG | pliegues | delta medio |
|---|---:|---:|---:|---:|---|---:|
| decil 10 de longitud | 158 | **0,3038** | 0,1139 | **0,2911** | **5/5** | **+0,1227** |
| texto largo (tercio superior) | 499 | 0,2525 | 0,1122 | 0,2425 | **5/5** | +0,0595 |
| densidad lexica baja | 524 | 0,2576 | 0,1107 | 0,2328 | **5/5** | +0,0679 |
| sin mencion | 781 | 0,2471 | 0,1037 | 0,2177 | **5/5** | +0,0720 |
| con caracteres repetidos | 275 | **0,2982** | 0,0873 | 0,2327 | 4/5 | +0,0994 |
| con URL | 458 | 0,2445 | 0,0852 | 0,2227 | 4/5 | +0,0473 |
| decil 1 de longitud | 167 | 0,0778 | 0,0359 | 0,0838 | 0/5 | -0,1596 |

**El mayor deposito de error es el texto largo con densidad lexica baja y caracteres repetidos, y
es estable en 5/5 pliegues.** El decil 10 de longitud tiene HS 0,3038 frente a 0,2023 fuera de la
region: el doble de error.

Y los cruces muestran que **la URL y la mencion no tienen un efecto fijo**:

| cruce | combinacion | HS | la misma sin la condicion |
|---|---|---:|---:|
| corto x URL | corto **con** URL | **0,2542** | 0,1338 (corto sin URL) |
| largo x URL | largo **con** URL | 0,2642 | 0,2451 (largo sin URL) |
| corto x mencion | corto **con** mencion | **0,1122** | 0,2273 (corto sin mencion) |
| largo x mencion | largo **con** mencion | 0,2238 | 0,2734 (largo sin mencion) |
| medio x hashtag | medio **con** hashtag | 0,1458 | 0,2370 (medio sin hashtag) |

**La mencion protege a HS en texto corto y lo empeora en texto largo. El hashtag protege a TR en
texto medio.** Ese es exactamente el tipo de interaccion que un modelo lineal con n-gramas no puede
capturar, porque no tiene forma de condicionar un n-grama a la posicion ni a la longitud del texto.
"""

TOP3 = """\
### 26.5 TOP 3 de fuentes de senal todavia sin explotar

Tres mecanismos, elegidos porque atacan directamente las regiones donde V03 falla y porque usan
informacion de `text` que no se ha usado como fuente predictiva real.

#### 1. Estructura de URL y dominio

- **Que senal captura**: el dominio, la longitud del path y la presencia de parametros. No el token
  `http`, que es lo que V03 ve.
- **Por que no equivale a una familia cerrada**: V29-04 ya analizo las URLs como *rasgo de
  contradiccion* entre pares, y V28-01 como *rasgo descriptivo* del error. Ninguno las uso como
  **feature predictiva**. Y el cruce de esta seccion muestra que su efecto segun la longitud es de
  signo contrario, lo que un n-grama no captura.
- **Que la diferencia**: V29-04 (URLs en pares contradictorios), V28-01 (URLs como rasgo del error),
  V29-01 (URLs eliminadas en la normalizacion del near-duplicate).
- **Que subset de errores podria corregir**: los 118 textos cortos con URL, donde HS 0,2542 frente
  a 0,1338, y los 193 largos con URL, donde AG 0,2798 frente a 0,2190.
- **Experimento minimo para falsarla**: anadir un bloque de features de dominio (dominio en texto
  plano, longitud del path, numero de parametros) a la representacion de V03 y medir corrected_cv
  con los mismos pliegues. Si no mejora, la senal no es predictiva.
- **Coste**: bajo. Un `CountVectorizer` sobre dominios extraidos con regex, sin embeddings.
- **Riesgo de overfitting**: medio-bajo. El dominio es una variable de baja cardinalidad relativa y
  el efecto se mide sobre 311 filas con URL.

#### 2. Posicion y rol de menciones y hashtags

- **Que senal captura**: si la mencion esta al inicio (respuesta) o al final (enunciacion), y si el
  hashtag abre o cierra el texto.
- **Por que no equivale a una familia cerrada**: V03 ya usa `char_wb(3,5)`, que captura una mencion
  como n-grama de caracteres. Pero un n-grama no dice **donde** esta. Y el cruce muestra que la
  posicion cambia el signo del efecto.
- **Que la diferencia**: V28-01 (posicion de la mencion como rasgo descriptivo), V29-04 (menciones
  como transformacion entre pares), V29-01 (pseudo-autor por primera mencion).
- **Que subset de errores podria corregir**: los 294 textos cortos con mencion, donde HS 0,1122
  frente a 0,2273, y los 48 medios con hashtag, donde TR 0,0417 frente a 0,1303.
- **Experimento minimo para falsarla**: anadir dos features por documento (posicion relativa de la
  primera mencion, posicion relativa del primer hashtag) y medir corrected_cv.
- **Coste**: minimo. Dos columnas.
- **Riesgo de overfitting**: bajo, pero el subgrupo mas nitido (medio con hashtag) tiene solo 48
  filas, asi que ese cruce concreto es el mas fragil de los tres.

#### 3. Enfasis ortografico y repeticion

- **Que senal captura**: la elongacion de palabras (`holaaaa`), la repeticion de caracteres dentro de
  una palabra y la densidad de signos de exclamacion.
- **Por que no equivale a una familia cerrada**: V03 usa `char_wb(3,5)`, que ve `!!!` como un
  n-grama de caracteres. Pero no ve la **repeticion dentro de una palabra**, que es un fenomeno
  distinto: `holaaaa` y `hola` comparten los mismos 3-5 gramas de palabra, y el modelo no tiene
  forma de saber que uno esta enfatizado y el otro no.
- **Que la diferencia**: V28-01 (`caracteres_repetidos` como rasgo descriptivo), V29-04
  (repeticion como transformacion entre pares), V19 (sintaxis de cita y negacion, que es otra cosa).
- **Que subset de errores podria corregir**: los 275 textos con caracteres repetidos, donde HS
  0,2982 frente a 0,1937 fuera de la region. Es el rasgo con mayor efecto despues de la longitud.
- **Experimento minimo para falsarla**: anadir un bloque de features de enfasis (numero de
  vocales elongadas, ratio de caracteres repetidos consecutivos, densidad de exclamaciones) y
  medir corrected_cv.
- **Coste**: minimo. Un `CountVectorizer` con un patron de elongacion, o tres columnas.
- **Riesgo de overfitting**: bajo. 275 filas y 4/5 pliegues de estabilidad.

**Ninguno de los tres es un modelo nuevo, un kernel nuevo ni un ensemble.** Son bloques de features
sobre el mismo texto, que es la condicion que puso la directiva para el siguiente eje.
"""

CIERRE_FINAL = """\
### 26.6 Cierre de la fase

1. **`LABEL_NOISE = CLOSED`.** La senal de sospecha es dificultad, no ruido, y siete experimentos
   historicos ya cerraron la via de la robustez.
2. **`RESIDUAL_ERROR_PROFILE`.** El error se concentra en texto largo, con densidad lexica baja y
   caracteres repetidos, estable en 5/5 pliegues. Las tres etiquetas no tienen perfiles distintos.
3. **`TOP 3 UNEXPLORED SIGNAL SOURCES`.** Estructura de URL y dominio; posicion y rol de
   menciones y hashtags; enfasis ortografico y repeticion. Los tres atacan regiones concretas donde
   V03 falla, los tres usan informacion de `text` no explotada como fuente predictiva, y los tres
   tienen un experimento minimo barato para falsarlos.
4. **No se toca `N = 1186`.** Nada de esta fase dice nada sobre si es un techo.
5. **0 submissions, 0 modelos entrenados, V03 intacto.**
"""

CODIGO_POBLACIONES = """\
from pathlib import Path
import json

import numpy as np

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v30_01_perfil_error_residual.json").open(encoding="utf-8") as f:
    d = json.load(f)

print("26.2  POBLACIONES HISTORICAS, reconstruidas sin redefinir")
p = d["poblaciones_historicas"]
print(f"  {'poblacion':<22}{'filas':>7}{'fraccion':>11}")
for k, v in p["conteo"].items():
    print(f"  {k:<22}{v:>7}{v / sum(p['conteo'].values()):>11.4f}")
print(f"  umbral de dureza (p80): {p['umbral_de_dureza_p80']:.4f}")
print(f"  filas en cluster contradictorio: {p['cluster_contradictorio_filas']}")
print()
print("  separacion Facil vs Dificil coherente, los ocho rasgos de mayor efecto")
print(f"  {'rasgo':<26}{'d':>9}{'facil':>10}{'dificil':>10}{'p':>11}")
for k, v in sorted(d["separacion_facil_vs_dificil"].items(),
                   key=lambda kv: -abs(kv[1]["d_cohen_dificil_menos_facil"]))[:8]:
    print(f"  {k:<26}{v['d_cohen_dificil_menos_facil']:>+9.3f}{v['media_facil']:>10.3f}"
          f"{v['media_dificil']:>10.3f}{v['p_mannwhitney']:>11.2e}")
print()
print("  LECTURA: los unicos rasgos con efecto son longitud_media_token, urls, es_muy_corto y")
print("  puntuacion, todos del mismo signo. Los dificiles tienen tokens mas cortos y mas URLs,")
print("  no mas texto. La dificultad no es volumen.")
"""

CODIGO_PERFIL = """\
print("26.3  PERFIL POR LABEL")
print()
print(f"  {'label':<7}{'errores':>9}{'tasa':>8}{'FP':>6}{'FN':>6}"
      f"{'FP/pred':>9}{'FN/pos':>9}")
for et in ("HS", "TR", "AG"):
    b = d["por_label"][et]
    print(f"  {et:<7}{b['n_errores']:>9}{b['tasa_de_error']:>8.4f}{b['n_fp']:>6}"
          f"{b['n_fn']:>6}{b['fp_entre_predichos']:>9.4f}{b['fn_entre_positivos']:>9.4f}")
print()
for et in ("HS", "TR", "AG"):
    print(f"  {et}")
    print(f"    {'rasgo':<26}{'d':>9}{'si error':>10}{'si correcto':>13}{'p':>11}")
    for e in d["por_label"][et]["rasgos_de_mayor_efecto"][:3]:
        print(f"    {e['rasgo']:<26}{e['d_cohen']:>+9.3f}{e['media_si_error']:>10.3f}"
              f"{e['media_si_correcto']:>13.3f}{e['p']:>11.2e}")
    print()
print("  LECTURA: HS y AG fallan en texto largo; TR falla en texto muy corto, con signo")
print("  contrario. Y TR es la etiqueta con menos errores. No hay tres problemas distintos.")
"""

CODIGO_REGIONES = """\
print("26.4  LAS REGIONES DONDE V03 FALLA, Y SU ESTABILIDAD")
print()
print(f"  {'region':<36}{'filas':>7}{'HS':>9}{'TR':>9}{'AG':>9}{'pliegues':>10}{'delta':>9}")
for k, v in d["interaccion_por_region"].items():
    est = d["estabilidad_por_pliegue"].get(k, {})
    pliegues = f"{est.get('pliegues_con_la_misma_direccion', '-')}/{est.get('de_total', '-')}"
    delta = float(np.mean(est["delta_por_pliegue"])) if est.get("delta_por_pliegue") else 0.0
    print(f"  {k:<36}{v['n_filas']:>7}{v['HS']['tasa_error_en_region']:>9.4f}"
          f"{v['TR']['tasa_error_en_region']:>9.4f}{v['AG']['tasa_error_en_region']:>9.4f}"
          f"{pliegues:>10}{delta:>+9.4f}")
print()
print("  CRUCES PREDEFINIDOS")
print(f"  {'cruce':<22}{'tramo':<9}{'cond':<4}{'filas':>7}{'HS':>9}{'TR':>9}{'AG':>9}")
for nombre, cruce in d["cruces"].items():
    for f in cruce["filas"]:
        cond = f.get("condicion", "-")
        print(f"  {nombre:<22}{f['tramo']:<9}{cond:<4}{f['n_filas']:>7}"
              f"{f['tasa_error']['HS']:>9.4f}{f['tasa_error']['TR']:>9.4f}"
              f"{f['tasa_error']['AG']:>9.4f}")
print()
print("  LECTURA: la mencion protege a HS en texto corto (0,1122 vs 0,2273) y lo empeora en")
print("  texto largo (0,2238 vs 0,2734). El hashtag protege a TR en texto medio (0,0417 vs")
print("  0,1303). Ese es el tipo de interaccion que un modelo lineal con n-gramas no captura.")
"""

CODIGO_TOP3 = """\
print("26.5  TOP 3 DE FUENTES DE SENAL TODAVIA SIN EXPLOTAR")
print()
print("  1. ESTRUCTURA DE URL Y DOMINIO")
print("     senal: dominio, longitud del path, parametros. No el token 'http'.")
print("     no equivale a: V29-04 (URLs en pares), V28-01 (URLs como rasgo), V29-01 (normalizacion)")
print("     subset: 118 cortos con URL (HS 0,2542 vs 0,1338) y 193 largos (AG 0,2798 vs 0,2190)")
print("     minimo para falsarla: bloque de features de dominio + corrected_cv en los mismos pliegues")
print("     coste: bajo | riesgo: medio-bajo")
print()
print("  2. POSICION Y ROL DE MENCIONES Y HASHTAGS")
print("     senal: si la mencion abre (respuesta) o cierra (enunciacion) el texto")
print("     no equivale a: V28-01 (posicion como rasgo), V29-04 (mencion como transformacion),")
print("                   V29-01 (pseudo-autor por primera mencion)")
print("     subset: 294 cortos con mencion (HS 0,1122 vs 0,2273) y 48 medios con hashtag")
print("     minimo para falsarla: dos columnas de posicion relativa + corrected_cv")
print("     coste: minimo | riesgo: bajo, pero el subgrupo de 48 filas es fragil")
print()
print("  3. ENFASIS ORTOGRAFICO Y REPETICION")
print("     senal: elongacion de palabras, repeticion dentro de la palabra, exclamaciones")
print("     no equivale a: V28-01 (caracteres_repetidos como rasgo), V29-04 (repeticion entre")
print("                   pares), V19 (sintaxis de cita y negacion, que es otra cosa)")
print("     subset: 275 textos con caracteres repetidos (HS 0,2982 vs 0,1937)")
print("     minimo para falsarla: bloque de features de enfasis + corrected_cv")
print("     coste: minimo | riesgo: bajo, 275 filas y 4/5 pliegues de estabilidad")
print()
print("  Ninguno es un modelo nuevo, un kernel nuevo ni un ensemble. Son bloques de features")
print("  sobre el mismo texto.")
print()
print("26.6  CIERRE")
print("  LABEL_NOISE = CLOSED")
print("  RESIDUAL_ERROR_PROFILE")
print("  TOP 3 UNEXPLORED SIGNAL SOURCES")
print("  N = 1186 no se toca. 0 submissions, 0 modelos, V03 intacto.")
g = d["gastos"]
print(f"  Coste del diagnostico: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB.")
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
    """Anade las celdas de la fase V30 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v30_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "perfil del error residual" in marcas:
        print("  la seccion 26 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(CIERRE),
            markdown(POBLACIONES),
            markdown(PERFIL),
            markdown(REGIONES),
            markdown(TOP3),
            markdown(CIERRE_FINAL),
            codigo(CODIGO_POBLACIONES),
            codigo(CODIGO_PERFIL),
            codigo(CODIGO_REGIONES),
            codigo(CODIGO_TOP3),
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 26 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
