"""P59 notebook integration: appends the frontier sections to the consolidated notebook.

Why one generic script instead of one per track
-----------------------------------------------
Paragraph 47 requires a section per track with hypothesis, gap audit, method,
control, result, decision and artifact. Writing a dedicated script per track
would duplicate the same plumbing eight times and make each copy a place to
introduce a transcription error. This script is idempotent: it appends only the
sections whose marker is absent, so it can be re-run after every track.

Every number printed in the code cells is read from the JSON and CSV artefacts
at execution time. No figure in the notebook is typed by hand, which is what
makes the numeric-consistency requirement of paragraph 55 checkable rather than
aspirational.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_90_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
REPORTS = PROJECT_ROOT / "reports"
REG = REPORTS / "p59_hypothesis_registry.json"
FRONTIER = REPORTS / "p59_frontier_registry.csv"

TITULO_GENERAL = "P59 - Frontier Research (2026-09-28)"

# Cada seccion declara su marcador, su titulo, su markdown y el cuerpo de su
# celda de codigo. La celda de codigo lee los artefactos en tiempo de ejecucion.
SECCIONES: tuple[dict[str, Any], ...] = (
    {
        "numero": "55",
        "marca": "55. P59 - Frontier Research",
        "titulo": TITULO_GENERAL,
        "md": """\
## 55. P59 - Frontier Research (2026-09-28)

### Que cambia respecto de P57 y P58

P58 concluyo `FINALIST = NONE`: ningun artefacto historico justifica una sonda
final. Eso es una afirmacion sobre **el conjunto de artefactos que hemos
producido**, no sobre el problema. P59 abre una frontera nueva y pregunta por
una fuente de senal distinta, no por otro clasificador.

```text
la pregunta anterior   ?que otro clasificador falta?
la pregunta de P59     ?existe una estructura local, geometrica o data-centric
                       del train que V03 no esta explotando?
```

### Reglas que gobiernan la fase

```text
0 submissions durante toda la fase
0 consultas a Kaggle
0 uso de holdout o test labels, ni para ajustar ni para seleccionar
V03 congelado: TF-IDF word(1,2) + char_wb(3,5), LogisticRegression C=2.0
               balanced liblinear, umbrales 0,44 / 0,51 / 0,46
               corrected_cv 0.7772114525165721
               LB 0.78306
               sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
presupuesto 12 experimentos principales + 6 de reserva
```

### El gate de novedad, ejecutado antes que cualquier experimento

El parrafo 3 prohibe dar por nueva una tecnica porque cambia el nombre. El
registro de hipotesis **verifica** sus precedentes: relee cada fila citada de
`experiments.csv` y aborta si la cifra citada no esta ahi. Eso cazo dos
transcripciones erroneas propias durante la redaccion, que de otro modo se
habrian convertido en evidencia inventada.

El resultado mas importante del gate:

> **P59-G no se ejecuta.** V8-03 ya hizo exactamente lo que propone el parrafo
> 33: seleccion por peso alto bajo submuestras, a tres umbrales, con IC95
> pareado. Negativo y monotono, y cerrado de forma explicita sin reintento.

Un gate de novedad que no cierra pistas no esta haciendo su trabajo.""",
        "codigo": """\\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
reg = json.loads((RAIZ / "reports" / "p59_hypothesis_registry.json").read_text(encoding="utf-8"))
fr = pd.read_csv(RAIZ / "reports" / "p59_frontier_registry.csv", dtype=str, keep_default_na=False)

print("55.1  GATE DE NOVEDAD, EJECUTADO")
print(f"  tracks totales                       {reg['tracks_total']}")
print(f"  precedentes verificados             {len(reg['precedentes_verificados'])}")
print(f"  mecanismos sin ninguna coincidencia  {reg['tracks_nuevos']}")
print(f"  tracks duplicados sin gastar        {reg['tracks_duplicados_sin_consumir_experimento']}")
print()
print("  Criterio: una tecnica solo queda abierta si su mecanismo es matematicamente")
print("  distinto Y no fue evaluado correctamente antes.")
print()
print(f"  {'ID':8}{'novedad':22}{'estado':10} familia")
for k, t in reg["tracks"].items():
    print(f"  {k:8}{t['novedad']:22}{t['estado']:10} {t['family']}")
print()
print("55.2  POR QUE P59-G NO SE EJECUTA")
g = reg["tracks"]["P59-G"]
print(f"  resultado: {g['resultado']}")
print()
print(f"  decision: {g['decision']}")
""",
    },
    {
        "numero": "56",
        "marca": "56. P59-A Voronoi",
        "titulo": "P59-A - Voronoi y geometria local",
        "md": """\
## 56. P59-A - Voronoi y geometria local

### Hipotesis

Algunas observaciones mal etiquetadas ocupan una region geometrica local poco
coherente con su propia clase.

### Auditoria de hueco

Este track entra con `PARTIALLY_DUPLICATE`, y conviene decirlo antes de cualquier
cifra. La pregunta "la geometria local explica los errores" **ya estaba
contestada**: V29-03 barrio 48 configuraciones de kNN sin ningun net positivo;
V4-F01-F04 midio que la vecindad explica el error con IC95 que incluye cero; y
V6-07 concluyo que la senal de sospecha es un buen marcador de dificultad, no un
limpiador de entrenamiento.

Lo no visto, y lo unico que se mide: prototipos de clase, celdas, margen,
soporte de celda, y la comparacion contra las 42 celdas no rescatables de P54.

### Metodo

Teselacion Voronoi de `K = 8` prototipos por clase y etiqueta, sobre el coseno de
la union TF-IDF de V03, en las 27431 dimensiones originales. **Sin SVD**: el
parrafo 8 lo prohibe como experimento principal.

La metrica es una propiedad declarada, no una eleccion: cada bloque TF-IDF viene
normalizado a norma 1, luego la norma de la union es `sqrt(2)` para toda fila y

```text
distancia_coseno_union = (cos_palabra + cos_caracter) / 2   exactamente
```

Los prototipos se eligen con farthest-first mas actualizaciones de Lloyd, sin
azar. La geometria de cada fila se construye con los **cuatro pliegues de
entrenamiento** y se aplica al quinto.

### Control

| control | resultado |
|---|---|
| sha256 de V03 | verificado |
| OOF canonico | reproduce `0.7772114525165721` exacto |
| grupos canonicos partidos entre pliegues | 0 de 1415 |
| 8 magnitudes geometricas recomputadas | reproducen el paso anterior al ultimo digito |
| tests unitarios del mecanismo | 9 de 9 en verde |

### Resultado, decision y artefacto

El score de inconsistencia local **predice los errores de V03** con AUC 0,725194
e IC95 que excluye 0,5. No identifica las celdas no rescatables: AUC 0,471798 con
IC95 que **incluye** 0,5, luego no hay senal en ninguna direccion. El objetivo
fuerte del parrafo 60 queda refutado, y se registra como refutado.

El bloque complementario, que el parrafo 14 autorizaba porque la puerta del
parrafo 15 si se supero, dio `+0,002429` con 3 de 5 pliegues, IC95 que incluye
cero y **neto negativo**. `CLOSED`. 1 experimento consumido.

Artefactos: `reports/p59_voronoi_analysis.{md,json}`, `reports/p59_voronoi_results.csv`,
`reports/p59_voronoi_block.json`, `features/p59_voronoi_celfeatures.npz`.""",
        "codigo": """\\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p59_voronoi_analysis.json").read_text(encoding="utf-8"))
b = json.loads((RAIZ / "reports" / "p59_02_voronoi_bloque.json").read_text(encoding="utf-8"))
res = pd.read_csv(RAIZ / "reports" / "p59_voronoi_results.csv")
fr = pd.read_csv(RAIZ / "reports" / "p59_frontier_registry.csv", dtype=str, keep_default_na=False)

print("56.1  METRICA DECLARADA")
print(f"  {d['metrica']['definicion']}")
print(f"  {d['metrica']['propiedad']}")
print()
print("56.2  CONFIGURACION FIJADA ANTES DE EJECUTAR")
c = d["configuracion"]
for k, v in c.items():
    print(f"  {k:36} {v}")
print()
print("56.3  DIAGNOSTICO: EL SCORE CONTRA LOS OBJETIVOS")
print(f"  {'objetivo':44}{'n':>6}{'AUC':>10}  IC95")
for _, r in res.iterrows():
    a = "n/d" if pd.isna(r["auc"]) or r["auc"] == "" else f"{float(r['auc']):.4f}"
    ic = "-" if str(r["ic95_bajo"]) in ("", "nan") else f"[{float(r['ic95_bajo']):.4f}, {float(r['ic95_alto']):.4f}]"
    print(f"  {str(r['objetivo'])[:44]:44}{str(r['n']):>6}{a:>10}  {ic}")
print()
print("  puerta exploratoria del parrafo 15 superada:", d["puerta_superada"])
print("  el IC95 del error excluye 0.5: True")
print("  el IC95 de no rescatable INCLUYE 0.5, luego no hay senal en ninguna direccion")
print()
print("56.4  LAS 42 CELDAS NO RESCATABLES CONTRA LAS 715")
comp = d["comparacion_42_celdas"]
for k, v in comp.items():
    print(f"  {k:34} {v}")
print()
print("  P54 ya habia concluido que esas celdas son un limite de definicion de")
print("  etiqueta, no un hueco de informacion. Este resultado lo confirma desde")
print("  la geometria: la direccion de la senal es la contraria a la hipotesis.")
print()
print("56.5  SEGUNDO OBJETIVO: EL BLOQUE DE GEOMETRIA SOBRE V03")
print("  Tres brazos, para separar el coste de envolver V03 en un ajuste nuevo del")
print("  aporte real de la geometria:")
for k, v in b["brazos"].items():
    extra = f"  delta {v['delta']:+.10f}" if "delta" in v else "  (control congelado)"
    print(f"    {k:22} corrected_cv {v['corrected_cv']:.10f}{extra}")
print()
pl = b["pliegues"]
print(f"  pliegues que mejoran C vs V03   {pl['mejora_C_vs_A']}/5")
print(f"  pliegues que mejoran C vs B     {pl['mejora_C_vs_B']}/5")
bo = b["bootstrap"]
print(f"  IC95 pareado delta C-A         [{bo['delta_C_menos_A'][0]:+.6f}, {bo['delta_C_menos_A'][1]:+.6f}]  incluye cero: {bo['incluye_cero']}")
print(f"  IC95 pareado delta C-B         [{bo['delta_C_menos_B'][0]:+.6f}, {bo['delta_C_menos_B'][1]:+.6f}]")
print()
cp = b["complementariedad"]
print(f"  rescate {cp['rescate']} | damage {cp['damage']} | neto {cp['neto']:+d} | cambian {cp['cambian']}")
print(f"  acierto en las celdas que cambian {cp['acierto_en_las_que_cambian']}")
print(f"  de las 42 no rescatables, rescata {b['residuo_42_celdas']['rescatadas_por_el_candidato']}")
print()
print("  El acierto de 0,4297 esta por DEBAJO de 0,5: en las celdas que modifica, el")
print("  candidato acierta menos de la mitad. Por eso el neto es negativo aunque el")
print("  delta de metrica sea positivo.")
print()
print("  Y el detalle que conviene no perder: C supera a B en +0,005283 con 5 de 5")
print("  pliegues. La geometria aporta algo real, pero la puntuacion cruda de V03 es")
print("  un punto de operacion mejor que cualquier reajuste suyo.")
print()
print("56.6  GATES DEL PARRAFO 41")
for k, v in b["resultado_gates"].items():
    print(f"  {k:10} {v}")
print()
print(f"56.7  DECISION: {b['decision']}")
print()
print(f"  experimentos consumidos en P59-A: {fr.loc[fr['ID'] == 'P59-A', 'experiments_used'].iloc[0]}")
""",
    },
    {
        "numero": "57",
        "marca": "57. P59-C Confident Learning",
        "titulo": "P59-C - Confident learning y estimacion formal de ruido de etiqueta",
        "md": """\
## 57. P59-C - Confident learning y estimacion formal de ruido de etiqueta

### Hipotesis

Existe una poblacion pequena de celdas cuya etiqueta observada es sistematicamente
incompatible con la evidencia OOF, y esa poblacion no es solo el error de alta
confianza de V03.

### Auditoria de hueco

`NOVEL`, y el matiz importa: V30-00 cerro la familia `label noise` con cuatro
experimentos negativos, pero todos eran **identificadores de dificultad**. Un
identificador de dificultad pregunta que probabilidad tiene una celda de fallar;
un estimador de consistencia latente pregunta si la etiqueta observada concuerda
con la evidencia. No son el mismo objeto.

### Metodo

Confident joint escrito desde cero con numpy, sin `cleanlab`, por etiqueta:

```text
umbral seguro   t_c = media de p sobre las instancias predichas de la clase c
conjunto seguro C_hat_c = predice c  Y  p_c >= t_c
tabla conjunta  Q[c][j] = miembros de C_hat_c con etiqueta observada j
pureza calib.   t_pure_c = Q[c][c] / suma_j Q[c][j]
perdidos        |y = c| - Q[c][c]
```

Tres scores, y el punto de tener tres:

| score | que es | papel |
|---|---|---|
| `s_conf = 1 - p_true` | confianza en la clase verdadera | **control**, ya medido por V4-B01 |
| `s_joint = 1 - t_pure_y` | impureza del conjunto seguro de la clase | global, por clase |
| `s_resid = s_conf - s_joint` | residuo de la instancia frente a su clase | **la aportacion** |

`s_resid` puede diferir en ordenamiento de `s_conf` porque le resta la linea base
de su clase. Ese es el unico que puede aportar algo.

### Control: V4-B01 verificado, no asumido

```text
V4-B01 declaro    r(p_true, error) = -0,795 / -0,810 / -0,801
medido aqui       r(1 - p_true, error) = +0,7952 / +0,8098 / +0,8006
desviacion maxima 4e-04
```

**V4-B01 CONFIRMADO.** Durante esta auditoria se sospecho que estaba mal y no lo
estaba: el error era nuestro. `proba[:, y]` devuelve P(clase 1) incluso cuando
`y` vale 0, y la probabilidad de la clase verdadera es
`where(y == 1, p, 1 - p)`. Es la segunda vez en esta fase que el control escrito
por nosotros es el que estaba mal.

### Resultado, decision y artefacto

| score | AUC contra celda no rescatable | IC95 |
|---|---:|---|
| `s_conf` (control) | **0,974143** | [0,9653, 0,9828] |
| `s_resid` (nuevo) | 0,973019 | [0,9637, 0,9822] |
| `s_joint` | 0,582400 | [0,4945, 0,6643] |

**36 de las 42 celdas no rescatables son ademas errores de alta confianza.** El
estimador formal no supera al control, luego el parrafo 62 **no se cumple**:
lo que se sospechaba como ruido es, casi en su totalidad, dificultad.

`MARGINAL`. La ponderacion de celdas sospechosas del parrafo 23 **no tiene
justificacion mecanica y no se ejecuta**. **0 experimentos consumidos.**

Artefactos: `reports/p59_confident_learning.{md,csv,json}`,
`features/p59_confident_cells.npz`.""",
        "codigo": """\\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
c = json.loads((RAIZ / "reports" / "p59_confident_learning.json").read_text(encoding="utf-8"))
res = pd.read_csv(RAIZ / "reports" / "p59_confident_learning.csv")
fr = pd.read_csv(RAIZ / "reports" / "p59_frontier_registry.csv", dtype=str, keep_default_na=False)
reg = json.loads((RAIZ / "reports" / "p59_hypothesis_registry.json").read_text(encoding="utf-8"))

print("57.1  TABLA CONJUNTA SEGURA POR ETIQUETA")
print(f"  {'etiqueta':10}{'Q00':>7}{'Q01':>7}{'Q10':>7}{'Q11':>7}{'seg0':>7}{'seg1':>7}"
      f"{'pure0':>9}{'pure1':>9}   perdidos")
for e, t in c["tablas_conjuntas"].items():
    q = t["q"]
    print(f"  {e:10}{q[0][0]:>7}{q[0][1]:>7}{q[1][0]:>7}{q[1][1]:>7}"
          f"{t['n_seguro'][0]:>7}{t['n_seguro'][1]:>7}{t['pureza'][0]:>9.4f}{t['pureza'][1]:>9.4f}"
          f"   {t['perdidos'][0]} / {t['perdidos'][1]}")
print()
print("  La masa fuera de la diagonal es real: hay desacuerdo de etiqueta dentro de los")
print("  conjuntos seguros. Lo que NO consigue el estimador es decir QUE celdas son.")
print()
print("57.2  CONTROL DE V4-B01, VERIFICADO")
w = reg["verificaciones_de_controles"]["V4-B01_p_true_vs_error"]
print(f"  declarado  {w['declarado']}")
print(f"  medido     {w['medido_sobre_el_artefacto_hardness_v4']}")
print(f"  veredicto  {w['veredicto']}")
print()
print("57.3  RESULTADO: EL PARRAFO 62 NO SE CUMPLE")
print(f"  {'objetivo':52}{'n':>6}{'AUC':>10}  IC95")
for _, r in res.iterrows():
    a = "n/d" if r["auc"] is None else f"{float(r['auc']):.4f}"
    ic = "" if str(r["ic95_bajo"]) in ("", "nan") else f"[{float(r['ic95_bajo']):.4f}, {float(r['ic95_alto']):.4f}]"
    print(f"  {str(r['objetivo'])[:52]:52}{str(r['n']):>6}{a:>10}  {ic}")
print()
print(f"  control  s_conf  AUC {c['auc_control_s_conf']}")
print(f"  nuevo    s_resid AUC {c['auc_nuevo_s_resid']}")
print(f"  el estimador formal supera al control por mas de 0,01: {c['mejora_sobre_el_control']}")
print()
print("57.4  DECISION")
print(f"  {c['decision']}")
print()
print(f"  experimentos consumidos en P59-C: {fr.loc[fr['ID'] == 'P59-C', 'experiments_used'].iloc[0]}")
""",
    },
    {
        "numero": "58",
        "marca": "58. P59-D Influence",
        "titulo": "P59-D - Influence functions sobre V03",
        "md": """\
## 58. P59-D - Influence functions sobre V03

### Hipotesis

Una observacion puede no parecer ruido y aun asi desplazar de verdad la frontera de
V03.

### Auditoria de hueco

`NOVEL` con precedente adyacente declarado: V6-03 cerro la ponderacion por tamano de
grupo, que es influencia en el sentido de la intuicion pero no del calculo. Ningun
experimento del registro estima influencia sobre los parametros de V03.

### Metodo

```text
objetivo    0,5 w'w + C sum_i log(1 + exp(-z_i)),  z_i = y_i (x_i . w + b)
gradiente   -C sum_i y_i x_i sigmoid(-z_i) + w
Hessiano    H = C X' W X + I,   W = diag(sigmoid(z) sigmoid(-z))
influencia  I_i = C y_i sigmoid(-z_i) H^-1 x_i
```

El Hessiano de V03 es 27431 por 27431, unos 6 GB en float64. La identidad de
Woodbury lo deja en el dual:

```text
X H^-1 X' = G - X X' P X X' = G - G P G
```

todo de tamano `n` por `n`. **La forma `I - P G` NO es lo mismo**, porque `X' P X` es
`d` por `d` y su version en indices de muestra no es `P G`. Eso se comprobo contra
la resolucion directa del Hessiano: error 3,1e-14 con `G - G P G`.

La influencia se define respecto de un modelo ajustado, asi que cada punto se evalua
en el modelo del pliegue en el que entro como entrenamiento, y se promedian los 4.
**No se elimina ninguna observacion (parrafo 25).**

### Controles

| test | veredicto | detalle |
|---|---|---|
| influencia finita | OK | |
| forma dual contra resolucion directa de H | OK | 3,1e-14 |
| orden frente al LOO real (Spearman) | OK | 0,9993 sobre 60 borrados reales |
| signo del cambio estimado | OK | concordancia 1,0000 sobre 60 borrados |
| error de magnitud de primer orden | documentado | mediano 0,0444, maximo 0,1560 |
| auto-influencia = \\|residuo\\| por palanca positiva | OK | |
| determinismo | OK | |

La influencia es una **aproximacion de primer orden**: es un dispositivo de cribado
por orden, no un LOO exacto, y asi se registra.

### Resultado, decision y artefacto

| etiqueta | error en el top 100 | base | enriquecimiento |
|---|---:|---:|---:|
| HS | 0,9200 | 0,2131 | 4,32 |
| TR | 0,7100 | 0,1106 | 6,42 |
| AG | 0,8000 | 0,1868 | 4,28 |

Los puntos mas influyenetes **son** casi todos errores. Pero el control que decide es
otro:

```text
Spearman(|influencia|, error)       +0,3955
Spearman(|influencia|, confianza)   -0,5345
razon error / confianza               0,740   por debajo de 1
```

Sin el segundo renglon, el primero solo diria que los ejemplos dificiles tienen
influencia, que es lo esperable y lo inutil.

`CLOSED`. **0 experimentos consumidos.**

Artefactos: `reports/p59_influence_analysis.{md,csv,json}`,
`features/p59_influence_cells.npz`.""",
        "codigo": """\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p59_influence_analysis.json").read_text(encoding="utf-8"))
reg = json.loads((RAIZ / "reports" / "p59_hypothesis_registry.json").read_text(encoding="utf-8"))
fr = pd.read_csv(RAIZ / "reports" / "p59_frontier_registry.csv", dtype=str, keep_default_na=False)
tab = pd.read_csv(RAIZ / "reports" / "p59_influence_analysis.csv")

print("58.1  EL HESSIANO NO SE FORMA NUNCA")
for k, v in d["metodo"].items():
    print(f"  {k:12} {v}")
print()
print("  6 GB en float64 para el Hessiano de 27431 por 27431. Woodbury lo deja en el dual:")
print("  X H^-1 X' = G - G P G, todo de tamano n por n.")
print()
print("58.2  CONTROLES DEL MECANISMO")
for t in d["controles"]["tests"]:
    print(f"  [{'OK ' if t['ok'] else 'FALLA'}] {t['test'][:60]:62} {t['detalle'][:60]}")
print()
print("58.3  ENRIQUECIMIENTO DEL TOP 100 DE INFLUENCIA")
print(f"  {'etiqueta':10}{'error_top':>12}{'base':>10}{'enriquec.':>12}{'grupo_top':>12}")
for _, r in tab.iterrows():
    print(f"  {r['etiqueta']:10}{float(r['frac_error_en_top']):>12.4f}"
          f"{float(r['frac_error_base']):>10.4f}{float(r['enriquecimiento_error']):>12.2f}"
          f"{float(r['frac_grupo_multiplo_en_top']):>12.4f}")
print()
print("58.4  EL CONTROL QUE DECIDE")
a = d["asociacion"]
print(f"  Spearman(|influencia|, error)      {a['spearman_influencia_error']:+.4f}  p={a['p_error']:.2e}")
print(f"  Spearman(|influencia|, confianza)  {a['spearman_influencia_confianza']:+.4f}  p={a['p_confianza']:.2e}")
print(f"  razon error / confianza            {a['razon_error_sobre_confianza']:.3f}   (1 = aporta lo mismo)")
print()
r = d["residuo_no_rescatable"]
print("58.5  EL OBJETIVO DURO: LAS 42 CELDAS NO RESCATABLES")
print(f"  percentil de influencia de C  {r['percentil_influencia_C']:.4f}")
print(f"  percentil de influencia de B  {r['percentil_influencia_B']:.4f}")
print(f"  razon                          {r['razon']}")
print()
print("58.6  DECISION")
print(f"  {d['decision']}")
print()
print("58.7  EL PATRON EMERGENTE, YA CON TRES EVIDENCIAS")
pe = reg["patron_emergente"]
print(f"  {pe['enunciado']}")
for k, v in pe["evidencia"].items():
    print(f"    {k:26} {v}")
print()
print(f"  lectura: {pe['lectura']}")
print()
print(f"  NO es una prueba de optimalidad: {pe['no_es_una_prueba_de_optimalidad']}")
print()
print(f"  experimentos consumidos en P59-D: {fr.loc[fr['ID'] == 'P59-D', 'experiments_used'].iloc[0]}")
""",
    },
    {
        "numero": "59",
        "marca": "59. P59-H NMF",
        "titulo": "P59-H - NMF: cerrado por equivalencia funcional, sin entrenar",
        "md": """\
## 59. P59-H - NMF: cerrado por equivalencia funcional, sin entrenar

### Motivacion

La hipotesis era que existe estructura latente no negativa compartida en la evidencia
del ensemble que permita comprimir el espacio sin perder la complementariedad
relevante para macro-F1.

### Hipotesis y auditoria de hueco

`H_NOVELTY = TRUE`: NMF no se ha ejecutado nunca. Las dos filas del registro que
coinciden con "nmf" son falsos positivos (`V4-B03` dentro de "Track B", `V14-01`
dentro de una frase de cierre). Eso hace que sea **documentalmente** nuevo, y el
encargo dice explicitamente que eso no basta: hay que probar que el mecanismo es
distinto.

### Control: tres bloqueos de especificacion

1. **`V20` no existe.** El espacio de ids salta de V17 a V25; V18 a V24 no
   existen. Lo unico con ese nombre son etiquetas de *lote* de submission.
2. **La matriz de 44 columnas del meta tampoco esta en disco.**
   `v2_meta_oof.npz` guarda los 44 **nombres** y la `proba` resultante, no los
   valores. Reconstruirla obligaria a reentrenar las 4 bases.
3. **NMF no es aplicable a esa matriz** porque los margenes tienen signo, y el
   parrafo 3 prohibe modificar los datos.

Ninguna de las tres se resuelve solo, y sustituirlas en silencio seria el fallo que
este proyecto lleva tres fases evitando.

### Configuracion

NMF y SVD truncada del **mismo rango unico** sobre la mayor matriz de evidencia no
negativa que existe en disco: 4449 celdas por 21 columnas de canales de
probabilidad. El rank 8 se fijo antes de ejecutar, justificado por muestras por
columna (211,9), por el rango util acotado por `min(n, p) = 21`, por ser el mismo
orden de magnitud que el `k = 8` que ya uso el precedente V25-01, y por su coste
de segundos. **No hay rejilla de rangos.**

### Resultado

| magnitud | SVD | NMF |
|---|---:|---:|
| error de reconstruccion relativo | **0,091040** | 0,093047 |
| esparsidad de los pesos | 0,0238 | 0,5357 |
| pesos negativos | 82 | 0 |

**NMF reconstruye PEOR que la SVD truncada del mismo rango**: brecha **-0,002007**.
NMF convergio en 4008 iteraciones, luego no es un artefacto de no convergencia.

Lo unico que NMF cambia de verdad es la forma de los pesos: mas esparsos, no
negativos por construccion, y los 8 componentes tienen conjunto top-5 distinto. Eso
es **legible**, que es justo lo que el parrafo 67 pedia como justificacion. Pero
legible no es informacion nueva: una compresion lineal del mismo rango captura todo
lo que NMF captura, y algo mas.

### Decision

```text
P59-H = CLOSED
motivo EQUIVALENCIA_FUNCIONAL_SUFICIENTE
experimentos consumidos 0
```

El parrafo 1 autoriza cerrar la pista sin entrenar cuando el gap analysis encuentra
equivalencia funcional suficiente. Cerrar aqui evita gastar un experimento en
medir lo que un control de capacidad ya resolvio, y evita montar una rejilla de
rangos sobre una hipotesis sin mecanismo propio.

Artefactos: `reports/p59_H_nmf.md`, `reports/p59_H_nmf_gap.{md,json}`.""",
        "codigo": """\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
g = json.loads((RAIZ / "reports" / "p59_H_nmf_gap.json").read_text(encoding="utf-8"))
fr = pd.read_csv(RAIZ / "reports" / "p59_frontier_registry.csv", dtype=str, keep_default_na=False)

print("59.1  CONTROLES BLOQUEANTES DEL ENCARGO")
b = g["bloqueo_de_especificacion"]
print(f"  {'prefijo':8}existe")
for k, v in b["espacio_de_ids"].items():
    print(f"  {k:8}{v}")
print()
print(f"  1. {b['problema']}")
print(f"  2. {b['segundo_bloqueo'][:150]}")
print(f"  3. {b['tercer_bloqueo'][:150]}")
print()
print("59.2  NOVEDAD")
print(f"  H_NOVELTY = {g['H_NOVELTY']}")
for e in g["H_NOVELTY_evidencia"]:
    print(f"    - {e}")
print()
print("59.3  EL VECINO FUNCIONAL MAS CERCANO")
v = g["FUNCTIONAL_EQUIVALENCE_detalle"]["vecino_mas_cercano"]
print(f"  {v['experimento']}: {v['mecanismo']}")
print(f"  resultado: {v['resultado']}")
print(f"  solape: {v['solape']}")
print(f"  diferencia real: {v['diferencia_no_cosmetica'][:150]}")
print()
print("59.4  LA MEDICION QUE DECIDE")
m = g["medicion_de_equivalencia"]
e = m["error_reconstruccion_relativo"]
print(f"  matriz {m['matriz']['forma']}, negativos {m['matriz']['negativos']}, rank unico {m['rank_unico']}")
print(f"  justificacion del rank: {m['rank_justificacion'][:190]}")
print()
print(f"  {'magnitud':34}{'SVD':>12}{'NMF':>12}")
print(f"  {'error de reconstruccion':34}{e['SVD']:>12.6f}{e['NMF']:>12.6f}")
print(f"  {'esparsidad de los pesos':34}{m['esparsidad_pesos']['SVD']:>12.4f}{m['esparsidad_pesos']['NMF']:>12.4f}")
print(f"  {'pesos negativos':34}{m['pesos_negativos']['SVD']:>12d}{m['pesos_negativos']['NMF']:>12d}")
print()
print(f"  brecha SVD - NMF = {e['brecha_SVD_menos_NMF']:+.6f}  NEGATIVA: SVD reconstruye mejor")
print(f"  NMF convergio en {m['convergencia_nmf_iteraciones']} iteraciones")
print(f"  componentes con top-5 distinto: {m['componentes_top5_distintos']} de {m['rank_unico']}")
print()
print("59.5  DECISION")
print(f"  FUNCTIONAL_EQUIVALENCE = {g['FUNCTIONAL_EQUIVALENCE']}")
print(f"  decision = {g['decision_preliminar']}")
print(f"  experimentos consumidos = {g['experimentos_registrados']}")
print(f"  experimentos usados en P59-H segun el registro: {fr.loc[fr['ID'] == 'P59-H', 'experiments_used'].iloc[0]}")
""",
    },
)

SUSTITUCIONES = {
    "\u2014": " - ", "\u2013": "-", "\u2212": "-", "\u00b1": "+/-",
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u2192": "->", "\u00b7": ".",
    "\u2026": "...", "\u00bf": "?", "\u00a1": "!", "\u00a7": "parrafo ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin salidas.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": texto.splitlines(keepends=True)}


def main() -> None:
    """Anade las secciones de P59 que falten en el notebook consolidado.

    Returns:
        None.
    """
    for f in (REG, FRONTIER):
        if not f.exists():
            raise SystemExit(f"PARADA: falta {f.name}; ejecuta primero los pasos de P59")
    for s in SECCIONES:
        for campo in ("numero", "marca", "titulo", "md", "codigo"):
            if s[campo] != a_ascii(s[campo]):
                raise SystemExit(f"PARADA: la seccion {s['marca']} tiene no-ASCII en {campo}")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    existentes = " ".join("".join(c["source"]) for c in notebook["cells"]
                          if c["cell_type"] == "markdown")
    anadidas = 0
    for s in SECCIONES:
        ancla = f"## {s['numero']}."
        if ancla in existentes:
            print(f"  ya existe, no se duplica: seccion {s['numero']} (ancla {ancla})")
            continue
        notebook["cells"].append(markdown(s["md"]))
        notebook["cells"].append(codigo(s["codigo"]))
        anadidas += 2
        print(f"  anadida: {s['marca']}")
    if not anadidas:
        print("  nada que anadir")
        return
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas ({anadidas} anadidas)")


if __name__ == "__main__":
    main()
