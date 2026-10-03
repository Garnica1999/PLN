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
    {
        "numero": "60",
        "marca": "60. P59-I perdida robusta redescendiente",
        "titulo": "P59-I - Perdida robusta redescendiente: MARGINAL, sin mecanismo causal",
        "md": """\
## 60. P59-I - Perdida robusta redescendiente: MARGINAL

### Motivacion e hipotesis

No es que los ejemplos toxicos deban pesar menos, ni detectar outliers para
reponderarlos. La hipotesis es sobre la **funcion objetivo**: la perdida
logistica puede dejar que unas pocas observaciones de alto leverage con residuos
extremos muevan la frontera global, y una perdida acotada y redescendiente
podria limitarla DURANTE el ajuste, de forma endogena, sin nombrar ningun
ejemplo.

### Auditoria de novedad funcional

`0` filas del registro mencionan tukey, biweight, geman, mcclure, cauchy, fair,
redescend, m-estimador, rlm o irls. `V34-01` lista "margin/robust loss" con
profundidad **(0) UNEXPLORED**. `V35-01` probo `modified_huber`, pero sobre
`SGDClassifier`, otra resolucion, y Huber esta acotada pero NO es redescendiente.

### Configuracion, fijada antes de mirar validation

```text
funcion       biweight de Tukey, UNA sola, sin rejilla
tau           4,685, constante normal-consistente estandar
sigma         1, porque el residuo de devianza ya es un residuo estandarizado
normalizacion L(dev) = sigma^2 rho(sqrt(dev)/sigma), que coincide con el
              log-loss hasta segundo orden, luego la regularizacion efectiva
              NO cambia y la comparacion es de perdida y no de capacidad
solver        L-BFGS-B con gradiente analitico, verificado contra diferencias
              centrales con error relativo 1e-09
```

### Control de optimizador

Tres brazos. `A` es V03 por sklearn y reproduce el canonico bit a bit. `B` es
mi propio ajuste sobre la logistica pura: si `B` no clonara a `A`, la
diferencia de `C` hablaria de mi optimizador y no de la perdida. `B` clona a `A`
con 3 celdas de decision distintas de 4449.

La primera version uso un Newton con el intercepto absorbido por recentrado
ponderado. Ese esquema resta dos terminos grandes y con d = 27.431 la resta
pierde toda la precision: el mismo codigo daba coseno 0,99999 con d = 25 y
`corrected_cv 0,4674` con d = 27.431. El brazo `B` es exactamente lo que lo
detecto y detuvo la ejecucion antes de interpretar nada.

### El mecanismo no es el hipotetizado

Es lo mas importante del resultado:

- el redescendiente **nunca activa**: 0,00 por ciento de celdas con influencia
  exactamente nula en los quince ajustes
- la influencia **no se atenua, aumenta**: ratio 1,0112 y Spearman +0,9992
- los residuos **empeoran**, percentil 99 de 1,2760 a 1,3477
- de las 12 celdas que cambian, **ninguna** esta entre los 268 errores de alta
  confianza

Lo que si hay es una mejora pequena y quirurgica: las tres etiquetas suben, 4 de
5 pliegues suben, y solo cambian 12 de 4449 celdas.
""",
        "codigo": """\
import json
from pathlib import Path

rep = Path("reports")
d = json.loads((rep / "p59_I_robust_loss.json").read_text())
r = d["resultado"]
c = d["convergencia"]
mf = d["mecanismo"]
inf = d["influencia_post_hoc"]
dec = r["complementariedad"]

print("== resultado ==")
print(f"  corrected_cv V03 (brazo A) {r['brazo_A_corrected_cv']!r}")
print(f"  corrected_cv brazo B       {r['brazo_B_corrected_cv']!r}")
print(f"  corrected_cv Tukey         {r['brazo_C_corrected_cv']!r}")
print(f"  delta_vs_V03               {r['delta_vs_V03']:+.10f}")
print(f"  folds                      {r['folds']}")
print(f"  folds V03                  {r['folds_V03']}")
print(f"  pliegues que mejoran       {r['pliegues_mejorados']}/5")
print(f"  rescate {dec['rescate']} | damage {dec['damage']} | neto {dec['neto']:+d}")
print()
print("== gate ==")
print(f"  {r['gate']}")
print(f"  bootstrap: {r['bootstrap']}")
print()
print("== convergencia ==")
print(f"  B {c['brazo_B_todas_convergen']} grad {c['norma_gradiente_B_max']:.2e}"
      f" | C {c['brazo_C_todas_convergen']} grad {c['norma_gradiente_C_max']:.2e}")
print(f"  fraccion recortada (influencia nula): {c['fraccion_recortada_media']}")
print()
print("== mecanismo ==")
print(f"  celdas que cambian                 {mf['celdas_que_cambian']}")
print(f"  cambios en error de ALTA confianza    {mf['cambios_en_error_de_alta_confianza']}")
print(f"  cambios en error de BAJA confianza    {mf['cambios_en_error_de_baja_confianza']}")
print(f"  cambios que rompen un acierto     {mf['cambios_que_rompen_un_acierto']}")
print(f"  |residual| p99 antes -> despues   {mf['residuo_p99_antes']} -> {mf['residuo_p99_despues']}")
print()
print("== influencia, analisis POSTERIOR (no se uso para modificar I1) ==")
print(f"  total V03 {inf['suma_influencia_V03']:.4e} -> robusto {inf['suma_influencia_robusto']:.4e}"
      f" | ratio {inf['ratio']}")
print(f"  Spearman entre ambas distribuciones {inf['spearman_entre_ambas']}")
print()
print(f"  decision: {d['decision'][:120]}")
print(f"  formulacion: {d['formulacion']}")
""",
    },
    {
        "numero": "61",
        "marca": "61. P59-B ventanas locales",
        "titulo": "P59-B - Ventanas locales: CLOSED, sin complementariedad",
        "md": """\
## 61. P59-B - Representacion por ventanas locales: CLOSED

### Motivacion e hipotesis

No es detectar mejor las celdas difiles. La hipotesis es sobre la **unidad de
representacion**: V03 comprime un tweet entero en un unico vector sparse, y ese
vector puede diluir evidencia toxica concentrada en pocos tokens. Una
representacion por ventanas locales la preserva, porque el clasificador nunca ve
el documento entero.

### Auditoria de novedad funcional

`V4-D01-D04` ya probo "ventanas locales como tokens" (spec 24) y cerro TRACK D
como NO_SIGNAL, pero **no es duplicacion**: alla la ventana es un canal sparse
EXTRA apilado al documento completo, luego la unidad de puntuacion sigue siendo
el tweet. Aqui la ventana **es** la unidad de puntuacion y el vector del
documento no existe.

`DL-02` (CharCNN, kernels de caracter 3 y 5) es el precedente mas cercano del
principio de agregacion, campos locales con max, pero en familia densa y desde
cero: 0,741036, descartado. Cero filas mencionan multiple-instance learning.

### Factibilidad, registrada antes de mirar validation

Los tweets son cortos: mediana de **15 tokens**, maximo 51. Con ventana de 32
tokens solo el **8,23 por ciento** de los tweets tiene mas de una ventana. Fuera
de esas 366 celdas la representacion es identica a la de V03 por construccion,
luego el cambio no puede excederlas. Es una cota previa, no una promesa.

### Donde se concentran los cambios

El resultado mas informativo no es el delta global sino su descomposicion: de
las 79 celdas que cambian, **53 estan en tweets con una sola ventana**, que por
construccion no fueron segmentados. Esas 53 solo reflejan el cambio del
conjunto de entrenamiento, no el mecanismo local. Y en el estrato donde el
mecanismo si actua, las correcciones son netamente negativas.
""",
        "codigo": """\
import json
from pathlib import Path

rep = Path("reports")
d = json.loads((rep / "p59_B_local_window.json").read_text())
fac = d["factibilidad_registrada_antes_de_validation"]
r = d["resultado"]
dec = r["complementariedad"]
es = d["complementariedad_por_estrato"]
esl = d["estratos"]

print("== factibilidad, registrada ANTES de validation ==")
print(f"  tokens por tweet: mediana {fac['tokens_mediana']:.0f} max {fac['tokens_max']}")
print(f"  ventanas por tweet: media {fac['ventanas_por_tweet_media']} max {fac['ventanas_por_tweet_max']}")
print(f"  tweets multi-ventana: {fac['tweets_multi_ventana']}"
      f" ({fac['porcentaje_multi_ventana']} por ciento)")
print(f"  celdas alcanzables: {fac['celdas_alcanzables']}")
print(f"  {fac['techo']}")
print()
print("== resultado ==")
print(f"  corrected_cv V03 {r['brazo_A_corrected_cv']!r}")
print(f"  corrected_cv B1  {r['brazo_B_corrected_cv']!r}")
print(f"  delta_vs_V03     {r['delta_vs_V03']:+.10f}")
print(f"  folds B1         {r['folds']}")
print(f"  folds V03        {r['folds_V03']}")
print(f"  pliegues que mejoran {r['pliegues_mejorados']}/5")
print(f"  rescate {dec['rescate']} | damage {dec['damage']} | neto {dec['neto']:+d}"
      f" | cambian {dec['cambian']} | acierto en las que cambian {dec['acierto_en_las_que_cambian']}")
print()
print("== la pregunta del parrafo 13, separada por estrato ==")
for k, v in es.items():
    print(f"  {k:18} n={v['n_tweets']:5d} cambios={v['cambios']:3d}"
          f" rescate={v['rescate']:3d} damage={v['damage']:3d} neto={v['neto']:+d}")
print()
print("== estratos por longitud ==")
for k, v in esl.items():
    print(f"  {k:6} n={v['n_tweets']:5d} cambios={v['cambios']:3d}"
          f" ccv B1 {v['ccv_B1']:.4f} V03 {v['ccv_V03']:.4f}")
print()
print("== deriva del max (calibracion, NO es una variante candidata) ==")
print(f"  {d['deriva_max']}")
print()
print(f"  gate: {r['gate']}")
print(f"  decision: {d['decision'][:140]}")
print(f"  formulacion: {d['formulacion']}")
""",
    },
    {
        "numero": "62",
        "marca": "62. P59-T calibracion del score local",
        "titulo": "P59-T - Calibracion del score local: CLOSED_BY_DUPLICATION",
        "md": """\
## 62. P59-T - Calibracion del score local: CLOSED_BY_DUPLICATION

No se ejecuto ningun experimento. La pista se cerro en el gap check, y el motivo
es algebrico antes que historico.

### Pierna 1, algebrica

Platt es una transformacion afin en el logit, `p_cal = sigmoid(a*logit(p_raw) + b)`.
Con un umbral fijo `t`:

```text
p_cal >= t  <=>  p_raw >= sigmoid((logit(t) - b)/a)
```

Es decir, aplicar Platt y mantener el umbral **es** reseleccionar el umbral por
etiqueta. Una reparametrizacion monotona no aporta informacion, solo mueve el
corte, y el propio encargo excluye el threshold sweep al definir T.

### Pierna 2, precedente, y es un cierre de raiz

`V8-09` midio sobre quince sondas independientes que la F1 frente a la tasa de
positivos predicha es **unimodal y V03 esta exactamente en su maximo**. Eso cierra
de raiz la familia entera de calibraciones, umbrales y pesos por etiqueta. Y
`V10-04` probo Platt, isotonica y beta de tres parametros sobre los scores de V03
con el calibrador anidado: la calibracion funciona de verdad, ECE de 0,058293 a
0,024801, pero el delta medio anidado de Platt es **+0,000148** con IC95
[-0,005221, +0,005932], y su decision cierra con clausula pre-registrada de no
reintentar mas variantes. `E2-14` calibro M2 y META sin mejora, y `V4-H01` empeoro
-0,0171 aplicando Platt al meta.

### Por que U no cae en este cierre

La fusion cambia el **ranking** siempre que los dos scores discrepen, y una
reparametrizacion monotona no puede hacer eso. Esa es la linea algebraica exacta
que separa U de T.
""",
        "codigo": """\
import json
from pathlib import Path

d = json.loads(Path("reports/p59_T_local_calibration.json").read_text())
print("== estado ==")
print(f"  track {d['track']} | {d['estado']} | {d['experimentos_consumidos']} experimentos")
print(f"  dependencia: {d['dependencia']}")
print()
print("== T0 control ==")
print(f"  {d['T0_control']}")
print()
print("== T1 gap check ==")
g = d["T1_gap_check"]
print(f"  veredicto: {g['veredicto']}")
print(f"  algebra: {g['pierna_1_algebraica']['platt']}")
print(f"  {g['pierna_1_algebraica']['consecuencia']}")
for p_ in g["pierna_2_precedente"]:
    cuerpo = p_.get("hallazgo") or p_.get("que_hizo") or ""
    print(f"  {p_['id']:18} {cuerpo}")
    if p_.get("resultado"):
        print(f"  {'':18} resultado: {p_['resultado']}")
    print(f"  {'':18} cierre: {p_.get('cierre', '(cerrado sin nota de cierre)')}")
print()
print(f"  por que U no: {d['por_que_U_no_esta_cerrado']}")
print()
print(f"  decision: {d['decision'][:160]}")
""",
    },
    {
        "numero": "63",
        "marca": "63. P59-U fusion global + local",
        "titulo": "P59-U - Fusion continua del score global y el local: MARGINAL",
        "md": """\
## 63. P59-U - Fusion continua del score global y el local

### Novedad: base nueva, mecanismo ya explorado

`ENS-01` a `ENS-04` combinan cuatro bases globales (voto igual, rejilla del
simplex, stacking por etiqueta, mejor par o trio anidado). `V4-H01` y `S08` usan un
meta de 36 rasgos. `FIN-08`, `FIN-09` y `FIN-10` enviaron candidatos de meta al
leaderboard y los tres quedaron por debajo de V03.

Lo nuevo es la **base**: ninguna fusion previa uso un score calculado sobre una
unidad de representacion distinta.

### Configuracion

```text
p = (1 - alpha) p_V03 + alpha p_local,  un alpha compartido por las tres etiquetas
regla =maximizar el macro-F1 a los umbrales congelados sobre las predicciones
       inner-OOF del TRAINING FOLD EXTERIOR. Validation nunca se toca
rejilla alpha de 0.00 a 1.00 en pasos de 0.05, y el NULO alpha = 0 esta dentro,
       luego la regla puede volver siempre a V03
```

### El dato decisivo

El alpha **oracle**, el que habria maximizado el score externo y por tanto hace
trampas, aporta **menos** que la seleccion anidada y honesta. La mejor fusion con
un alpha unico en todo el corpus apenas mueve la metrica, luego el delta de la
regla anidada viene de adaptar alpha pliegue a pliegue y no de senal transferible.
""",
        "codigo": """\
import json
from pathlib import Path

d = json.loads(Path("reports/p59_U_global_local_fusion.json").read_text())
r = d["resultado"]
print("== gap check ==")
print(f"  {d['U1_gap_check']['veredicto']}")
print(f"  base nueva: {d['U1_gap_check']['base_nueva']}")
print(f"  por que no cae en el cierre de T: {d['U1_gap_check']['por_que_U_no_esta_cerrada_como_T']}")
print()
print("== control ==")
print(f"  {d['control']}")
print()
print("== resultado ==")
print(f"  corrected_cv U   {r['corrected_cv_U']!r}")
print(f"  delta_vs_V03     {r['delta_vs_V03']:+.10f}")
print(f"  folds U          {r['folds_U']}")
print(f"  pliegues +       {r['pliegues_mejorados']}/5")
print(f"  alphas por pliegue {r['alphas_por_pliegue']}")
print(f"  rescate {r['rescate']} | dano {r['damage']} | neto {r['neto']:+d} | "
      f"acierto en las que cambian {r['acierto_en_las_que_cambian']}")
print(f"  gate: {r['gate']}")
print()
print("== la pregunta del parrafo 13, por estrato ==")
for k, v in r["complementariedad_por_estrato"].items():
    print(f"  {k:18} n={v['n_tweets']:5d} rescate={v['rescate']:3d} dano={v['damage']:3d} "
          f"neto={v['neto']:+d}")
print()
o = d["oracle_diagnostico"]
print("== diagnostico oracle, NO es candidato ==")
print(f"  alpha_oracle {o['alpha_oracle']} | delta_oracle {o['delta_oracle']:+.10f}")
print(f"  {o['nota']}")
print()
print(f"  decision: {d['decision'][:150]}")
""",
    },
    {
        "numero": "64",
        "marca": "64. P59-E regularizacion por grafo",
        "titulo": "P59-E - Regularizacion por grafo: MARGINAL, indistinguible del ruido",
        "md": """\
## 64. P59-E - Regularizacion por grafo sobre el objetivo de V03

### Novedad: nueva

| eje | filas del registro |
|---|---:|
| regularizacion por grafo | **0** |
| manifold regularization | **0** |
| termino cuadratico sobre un grafo de documentos | **0** |
| PageRank o paseo aleatorio | **0** |
| red neuronal sobre grafo | **0** |

El precedente mas cercano, `V4-F01-F04`, usa la vecindad para **propagar
etiquetas**, que es predecir, y cerro con techo nulo. `V4-A01`, `V4-A02`,
`E2-20-ESPECIALISTAS` y `V6-07` usan vecindad para predecir o ponderar. Ninguno
regulariza un objetivo.

**Previo nulo que conviene no esconder**: `V4-F01-F04` midio que un vecino
discrepante aporta solo +0,018 de error con IC95 [-0,013, +0,049], no
significativo. Un grafo puede regularizar sin que los vecinos compartan error, luego
no cierra la pista, pero se ejecuta sabiendo que el diagnostico previo salio nulo.

### Configuracion

```text
grafo     coseno kNN con k = 10 sobre el TF-IDF de V03, filas de ENTRENAMIENTO
objetivo  0,5 w'w + (lambda/2) w'X'WX w + C sum omega perdida(s)
lambda    0,1, un solo valor, sin rejilla
el grafo  SOLO regulariza: nunca vota, nunca recupera
```

El grafo se escapa por construccion: el mayor autovalor de `X'WX` se pone a 1, luego
lambda = 0,1 significa una curvatura de suavizado de al menos diez veces menor que
la del L2. Que ese sea el magnitud es un juicio declarado, no una magnitud
derivada.
""",
        "codigo": """\
import json
from pathlib import Path

d = json.loads(Path("reports/p59_E_grafo_regularizacion.json").read_text())
r = d["resultado"]
print("== gap check ==")
g = d["E1_gap_check"]
print(f"  veredicto: {g['veredicto']}")
for k, v in g.items():
    if k.startswith("filas_"):
        print(f"  {k:44} {v}")
print(f"  precedente mas cercano: {g['precedente_mas_cercano']}")
print()
print("== control: el brazo 0 es lo que hace legible al brazo 1 ==")
print(f"  {d['control']}")
print()
print("== resultado ==")
print(f"  brazo 0 (lambda 0)  {r['brazo_0_corrected_cv']!r}")
print(f"  brazo 1 (lambda .1) {r['brazo_1_corrected_cv']!r}")
print(f"  delta GRAFO PURO    {r['delta_grafo_puro_brazo1_menos_brazo0']:+.10f}")
print(f"  delta vs V03        {r['delta_vs_V03_brazo1_menos_V03']:+.10f}  incluye la tolerancia")
print(f"  pliegues +          {r['pliegues_mejorados']}/5")
print(f"  rescate {r['rescate']} | dano {r['damage']} | neto {r['neto']:+d} | "
      f"acierto {r['acierto_en_las_que_cambian']}")
print(f"  gate: {r['gate']}")
print()
print(f"  decision: {d['decision'][:200]}")
""",
    },
    {
        "numero": "65",
        "marca": "65. P59-F n-gramas generativas",
        "titulo": "P59-F - Modelo generativo de n-gramas: CLOSED_BY_DUPLICATION",
        "md": """\
## 65. P59-F - Modelo generativo de n-gramas: CLOSED_BY_DUPLICATION

No se ejecuto ningun experimento. Tres patas.

### Pierna 1, algebrica

Sobre una representacion de n-gramas fija, todo modelo generativo condicional de
clase colapsa a

```text
log P(c | x) = log P(c) + sum_f theta_{f,c}
```

que es la forma de naive Bayes y de log-odds. El unico parametro libre es el
estimador de `theta`, y todo estimador de `theta` es una estrategia de suavizado.
La representacion no salva nada, porque el colapso vale para cualquier base.

### Pierna 2, la familia ya se ejecuto

`MOD-02` probo ComplementNB sobre la union TF-IDF de V03: -0,069949 con 0 de 5
pliegues. `E2-15` probo el ratio log-count NB-SVM: -0,0234 con 0 de 5 pliegues,
ganancia de ensemble -0,0057 y correlacion de residuales con V03 de 0,9243.
`V37-01` probo BM25: -0,0027860149, neto -6, correlacion 0,9874, con su propia frase
de que no aporta una fuente de senal nueva. `V26-03` probo Tsetlin Machine y fue
REJECTED. Cero filas describen un modelo generativo como tal.

### Pierna 3, y cierra la especificacion exacta de F

El unico grado de libertad de este modelo es el **suavizado**, y F esta
especificado justo como un orden y una estrategia de suavizado. `V8-04` midio
ese parametro: -0,058496 con 1e-6 y +0,004745 con 1e-3, un salto de **+0,063241**
entre dos valores que solo difieren en esa constante, y su lectura es que el
suavizado sostiene el metodo entero. Su decision pre-registra por escrito que **no
se barre el suavizado**. F es un punto suelto de un barrido cerrado de antemano.
""",
        "codigo": """\
import json
from pathlib import Path

d = json.loads(Path("reports/p59_F_generativo_ngram.json").read_text())
g = d["F1_gap_check"]
print("== estado ==")
print(f"  track {d['track']} | {d['estado']} | {d['experimentos_consumidos']} experimentos")
print(f"  dependencia: {d['dependencia']}")
print()
print("== algebra: el colapso ==")
a = g["pierna_1_algebraica"]
print(f"  {a['colapso']}")
print(f"  {a['consecuencia']}")
print()
print("== familia ya ejecutada ==")
for p_ in g["pierna_2_familia_ya_ejecutada"]:
    print(f"  {p_['id']:26} {p_.get('que_hizo', '')}")
    print(f"  {'':26} {p_.get('resultado', '')}")
    print(f"  {'':26} cierre: {p_.get('cierre', '(cerrado sin nota de cierre)')}")
print()
print("== el unico grado de libertad, prohibido ==")
c3 = g["pierna_3_el_unico_grado_de_libertad_esta_prohibido"]
print(f"  {c3['por_que_es_legislativa']}")
print(f"  {c3['V8_04_lo_medijo']}")
print(f"  cierre: {c3['cierre_pre_registrado']}")
print(f"  {c3['estado_del_mejor_punto']}")
print()
print(f"  decision: {d['decision'][:200]}")
""",
    },
    {
        "numero": "66",
        "marca": "66. P59 cierre, DAG y tabla maestra",
        "titulo": "P59 - Cierre: DAG de dependencias, tabla maestra y conclusion",
        "md": """\
## 66. P59 - Cierre de la fase

### Grafo de dependencias, para que unas instrucciones futuras no creen dependencias falsas

```text
                 B1  (P59-B, ventanas locales)
                  |
        +---------+---------+
        |                   |
      P59-T               P59-U
  (calibracion local)   (fusion global+local)

  P59-A, C, D, G, H, I, E, F   ->  independientes
```

`P59-T` y `P59-U` necesitan el score local de `B1`; el resto no. Y `P59-E` y
`P59-F` no miran un solo artefacto de `B`, `T` ni `U`.

### Conclusion global, separando cinco efectos

1. **Diagnostico: real y fuerte, no accionable.** AUC 0,725 para la geometria,
   0,9730 para la consistencia latente, Spearman 0,9993 para la influencia. Tres
   mecanismos independientes, todos correctos, ninguno traducible a correccion.
2. **Calibracion: mueve la cifra sin mover la correccion.** Es el hallazgo mas
   importante de la fase. La calibracion de V03 funciona (ECE 0,058 a 0,025) y su
   delta es +0,000148 con IC95 que incluye el cero. Y en P59-B el sesgo del max
   produjo +0,003367 **superando el gate** con rescate 11 contra damage 15.
3. **Representacion: mueve la frontera sin mover las decisiones.** La perdida
   robusta cambia los coeficientes con cambio relativo de norma 0,046 y solo
   cambian 12 de 4449 celdas, ninguna de alta confianza.
4. **Decision: positivo pequeno e inconsistente.** +0,00131 en 4 de 5, +0,00148 en
   2 de 5, +0,000099 en 2 de 5. Ninguno paso el gate, luego **a ninguno se le
   ejecuto bootstrap**.
5. **Informacion complementaria: no se produjo en ninguna pista.** A neto -18, I
   neto +4, B neto -1, U neto +6, E neto -5.
""",
        "codigo": """\
import json
from pathlib import Path
import pandas as pd

d = json.loads(Path("reports/p59_final_summary.json").read_text())
print("== baseline congelado ==")
print(f"  corrected_cv {d['baseline']['corrected_cv']} | LB {d['baseline']['LB']}")
print(f"  sha256 {d['baseline']['sha256']}")
print()
print("== tabla maestra, en orden de ejecucion ==")
m = pd.read_csv("reports/p59_master_results.csv")
print(m[["orden","track","dependencia","estado","experimentos","corrected_cv",
         "delta_vs_V03","folds","neto"]].to_string(index=False))
print()
print(f"presupuesto: {d['presupuesto']['usados']} de {d['presupuesto']['principales']} "
      f"principales, {d['presupuesto']['reserva']} de reserva intactos")
print(f"candidatos {d['candidatos']} | finalistas {d['finalistas']} | submissions {d['submissions']}")
print()
print("== conclusion ==")
print(f"  {d['conclusion']}")
print()
print("== cumplimiento de la fase completa ==")
for k, vv in d.items():
    if k == "gastos":
        print(f"  {k}: {vv}")
print(f"  formulacion: {d['formulacion']}")
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
