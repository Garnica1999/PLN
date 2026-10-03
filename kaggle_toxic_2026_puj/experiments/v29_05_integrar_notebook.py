"""Integra la familia FUZZY_TEMPLATE_MATCHING (V29) en el notebook consolidado.

Anade la seccion 25 al final de `notebooks/10_final_pipeline.ipynb`, que recoge los cuatro
informes de la fase. Las celdas de codigo leen los JSON en tiempo de ejecucion y no llevan
outputs embebidos, para que no puedan quedar desincronizadas.

Uso
---
    /home/carlos/venv/bin/python experiments/v29_05_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORMES = {
    "v01": PROJECT_ROOT / "reports" / "v29_01_auditoria_near_duplicate.json",
    "v02": PROJECT_ROOT / "reports" / "v29_02_similitudes_laxas.json",
    "v03": PROJECT_ROOT / "reports" / "v29_03_vecinos_corrigen_errores.json",
    "v04": PROJECT_ROOT / "reports" / "v29_04_transformaciones_contradictorias.json",
}

TITULO = """\
## 25. Fase v29: FUZZY_TEMPLATE_MATCHING (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Existe una estructura de repeticion o plantilla en los tweets que contenga informacion de
> etiqueta que V03 esta dejando sobre la mesa?

La respuesta corta es **no**, y esta sostenida por cuatro mediciones. Esta seccion las expone
en orden, porque cada una descarta una forma distinta de que la respuesta hubiera sido si.
"""

REENCUADRE = """\
### 25.1 Lo primero: el 7,08 % no era near-duplicate

La auditoria del metodo historico (V29-01) reencuadra la premisa antes de medir nada.

| fuente | filas del desarrollo en grupo >= 2 | cobertura |
|---|---:|---:|
| near-duplicate puro (coseno char_wb 4-4 >= 0,85) | 6 | **0,0040** |
| pseudo-autor (primera mencion `@usuario`) | 99 | 0,0668 |
| grupos fusionados, los que usa V03 | 105 | 0,0708 |

**99 de las 105 filas entran solo por la heuristica de pseudo-autor.** El criterio con el que
V03 valida no es, en su mayor parte, "texto casi identico": es "mismo usuario". Y ese criterio
es el que produjo los 43 grupos contradictorios que midio V28-03.

El metodo historico, documentado: minusculas, URLs sustituidas por espacio, espacios colapsados;
`TfidfVectorizer(analyzer="char_wb", ngram_range=(4,4), min_df=1)`; coseno; **umbral de union
0,85**; componentes conexas por union-find; calculado sobre `train` + `test` concatenados. Los
otros dos umbrales del modulo (0,90 y 0,80) solo se usan para el resumen descriptivo, no para
agrupar. Se reprodujo el particionado y coincide con el CSV congelado.

La consecuencia practica: **el techo al que se puede aspirar relajando similitud es 0,40 %, no
7,08 %**.

Y el otro hallazgo de la auditoria: la **normalizacion pesa mas que el umbral**. Con la
normalizacion agresiva, la cobertura a 0,85 pasa de 0,0040 a 0,0303, y a 0,50 de 0,0668 a
0,1639.
"""

METODOS = """\
### 25.2 Cinco similitudes, todas sobre el texto

| clave | metodo |
|---|---|
| A | coseno TF-IDF char_wb 4-4, normalizacion historica (es el metodo de V03) |
| B | coseno TF-IDF char_wb 4-4, normalizacion agresiva |
| C | coseno TF-IDF char_wb 3-5, normalizacion agresiva |
| D | Jaccard de char 3-5 binarios, normalizacion agresiva |
| E | Jaccard de unigramas de palabra binarios, normalizacion agresiva |

Sin MinHash ni LSH: con 1731 filas hay 1 497 315 pares y la matriz densa ocupa 12 MB. Un indice
aproximado no aportaria nada y anadiria una fuente de error. Sin distancia de edicion: la
directiva la admite solo para candidatos prefiltrados, y un diagnostico descriptivo no la
necesita.

**Todos dan el veredicto B**: la probabilidad de compartir etiqueta mejora mucho sobre la linea
base, pero la cobertura no pasa del 2-5 %.

| metodo | mejor franja | probabilidad de misma etiqueta | linea base | mejora | cobertura maxima |
|---|---|---:|---:|---:|---:|
| A historico | 0,40-0,50 | 0,6716 | 0,5080 | +0,1636 | 0,0485 |
| B char 4 agresivo | 0,60-0,70 | 0,7591 | 0,5276 | +0,2465 | 0,0324 |
| C char 3-5 agresivo | 0,60-0,70 | 0,7368 | 0,5276 | +0,2245 | 0,0324 |
| **D Jaccard char 3-5** | 0,50-0,60 | **0,8405** | 0,5080 | **+0,3325** | 0,0191 |
| E Jaccard tokens | 0,40-0,50 | 0,7960 | 0,5080 | +0,2880 | 0,0214 |
"""

EL_VEREDICTO = """\
### 25.3 El veredicto: los vecinos puros no corrigen nada

Aqui esta el resultado que cierra la familia, y la Fase 5 de la directiva lo pedia explicito.

**Selector anidado** (la regla se elige dentro del training fold por CV interna de 4 pliegues,
asi que el OOF reportado no participa en su propia eleccion):

| mecanismo | cobertura | rescates | danos | net | delta OOF |
|---|---:|---:|---:|---:|---:|
| V03 | - | - | - | - | 0 |
| vecinos por similitud, anidado | 0,0011 | **0** | 5 | **-5** | **-0,001522** |

**Barrido exhaustivo**: 48 configuraciones con mayoria simple, sin elegir ninguna. No es lo
mismo que el selector, y es la medicion que de verdad responde a la pregunta.

- La **mejor** de las 48 tiene **net -8**.
- La **peor** tiene net **-234**.
- 45 de 48 producen al menos un rescate, pero la mejor es Jaccard char 3-5 con k=1 y umbral
  0,50: **1 rescate y 12 danos**, net -11.
- **Ninguna configuracion tiene net positivo.**

Y el detalle por franja de similitud es lo mas informativo:

| franja | celdas | rescates | danos |
|---|---:|---:|---:|
| 0,90-1,01 | 45 | **0** | **0** |
| 0,70-0,90 | 15 | 0 | 2 |
| 0,50-0,70 | 42 | 0 | 0 |

**En la franja de similitud por encima de 0,90 hay 45 celdas y ni un solo cambio de decision,
en ninguna direccion.** Donde el vecino es casi una copia del texto, V03 acierta todo lo que el
vecino acertaria. La repeticion de texto no aporta nada que V03 no tenga ya.

De los 441 filas del desarrollo donde V03 falla, solo **10 (2,27 %)** tienen un vecino por encima
del umbral, y el vecino trae la etiqueta correcta en 3 casos. En sentido contrario, hay **17
celdas donde V03 acertaba y el vecino lo contradice**.
"""

TRANSFORMACIONES = """\
### 25.4 Que transformacion produce las contradicciones

Parejas de texto casi identico con etiquetas distintas, y que las separa:

| umbral | pares | contradictorios | fraccion contradictoria |
|---:|---:|---:|---:|
| 0,90 | 39 | **0** | 0,000 |
| 0,85 | 39 | **0** | 0,000 |
| 0,75 | 55 | 4 | 0,073 |
| 0,70 | 62 | 7 | 0,113 |
| 0,60 | 185 | 27 | 0,146 |

**Por encima de 0,85 no hay ni una sola pareja contradictoria.** Las contradicciones solo
aparecen cuando la diferencia lexica ya es apreciable.

Y la Fase 11 pedia buscar que transformacion las genera: negacion, insulto anadido, mencion,
hashtag, URL, ortografia, truncamiento, puntuacion, repeticion, numerico. El resultado es que
**ninguna de esas categorias discrimina**. Lo unico que separa los pares contradictorios de los
concordantes es que su diferencia **no encaja en ninguna categoria**: razon de razones 1,96 con
p = 0,0155 a umbral 0,70.

Los ejemplos lo dicen sin ambiguedad:

- similitud 0,8182, etiquetas `[0,0,0] -> [1,1,1]`, y lo unico que anade el segundo texto es
  `mex`.
- similitud 0,7500, `[1,1,0] -> [1,1,1]`, y lo unico que anade el primero es `jeje`.
- similitud 0,7273, `[1,1,1] -> [0,0,0]`, y lo unico que cambia es `una` por `mi`.

Es decir: las contradicciones no son un fenomeno semantico tipificado, sino **cambio lexico
arbitrario**. Y en casi todos los casos cambian **las tres etiquetas a la vez**, lo que apunta a
que en este corpus las tres etiquetas se mueven juntas.
"""

CIERRE = """\
### 25.5 Conclusion de la fase

1. **LO QUE SOSTIENE.** Que existe estructura de repeticion: hay pares de texto casi identico
   cuya etiqueta se puede predecir con probabilidad 0,84 frente a 0,51 de la linea base. Eso es
   real y medido.
2. **LO QUE REFUTA.** Que esa estructura contenga informacion que V03 no tenga. En 48
   configuraciones, ninguna tiene net positivo; la mejor es -8. En la franja de similitud
   superior a 0,90 hay **cero** cambios de decision.
3. **LO QUE NO SOSTIENE.** Que la baja cobertura explique el resultado. Con la cobertura mas
   baja (0,0011) el resultado es negativo, y en la franja mas pura tambien. La conclusion no
   depende de la cobertura.
4. **LO QUE NO SOSTIENE NADA.** Nada de esta fase dice algo sobre `N = 1186`.
5. **LA CONCLUSION OPERATIVA.** `FUZZY_MATCHING = INFORMATIONAL_ONLY`, es decir el **resultado
   B** de la directiva. Cerrar la linea.

**No hay segunda fase de label propagation**, porque la condicion que la directiva pone para
abrirla es que los vecinos corrijan errores reales de V03 con dano bajo. El dano no es bajo: es
mayor que el rescate en las 48 configuraciones.

Y el hallazgo lateral mas util de la fase no es sobre V03, sino sobre el dataset: **por encima
de similitud 0,85 no existe ni una contradiccion de etiqueta.** Cualquier ruido de anotacion
que exista esta en pares cuya diferencia lexica es arbitraria, y en ese regimen el problema no
es de representacion sino de consistencia entre anotadores.
"""

CODIGO_AUDITORIA = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
# Carga de los cuatro informes de la fase v29, en tiempo de ejecucion.
import json as _json

def leer(nombre):
    # Devuelve el informe v29 correspondiente como diccionario.
    with (RAIZ / "reports" / f"v29_{nombre}.json").open(encoding="utf-8") as f:
        return _json.load(f)


a = leer("01_auditoria_near_duplicate")
b = leer("02_similitudes_laxas")
c = leer("03_vecinos_corrigen_errores")
d = leer("04_transformaciones_contradictorias")

print("25.1  DE DONDE SALE EL 7,08 %")
dc = a["desglose_de_cobertura"]
print(f"  {'fuente':<46}{'filas':>7}{'cobertura':>11}")
for k in ("near_duplicate_puro", "pseudo_autor", "grupos_fusionados_usados_por_v03"):
    print(f"  {k:<46}{dc[k]['filas_en_grupo_2_o_mas']:>7}{dc[k]['cobertura']:>11.4f}")
print(f"  de las 105, {dc['filas_que_entran_solo_por_pseudo_autor']} entran SOLO por pseudo-autor")
print()
print("  metodo historico:")
m = a["metodo_historico"]
for k in ("normalizacion", "no_normaliza", "representacion", "similitud", "umbral_de_union", "datos"):
    print(f"    {k:<18} {m[k]}")
print(f"  identidad del particionado frente al CSV congelado: "
      f"{a['controles']['identidad_del_particionado']}")
print()
print("  por que 0,40 %: umbral frente a normalizacion")
print(f"  {'umbral':>7}{'historica':>12}{'agresiva':>12}{'ganancia':>11}")
for f in a["comparacion_de_normalizacion"]:
    print(f"  {f['umbral']:>7.2f}{f['historica_cobertura_dev']:>12.4f}"
          f"{f['agresiva_cobertura_dev']:>12.4f}{f['ganancia_de_normalizacion']:>+11.4f}")
print()
print("  LECTURA: el techo al que se aspira relajando SIMILITUD es 0,40 %, no 7,08 %.")
print("  Y la normalizacion pesa mas que el umbral.")
"""

CODIGO_METODOS = """\
print("25.2  LAS CINCO SIMILITUDES")
print()
print("  veredictos por metodo")
print(f"  {'metodo':<34}{'veredicto':<36}")
for k, v in b["veredictos_por_metodo"].items():
    print(f"  {k:<34}{v:<36}")
print()
print("  mejor franja por metodo y label, y su mejora sobre la linea base")
print(f"  {'metodo':<30}{'label':>7}{'franja':>14}{'prob':>9}{'base':>9}{'mejora':>9}{'pares':>8}")
for h in b["mejor_franja_por_metodo_y_label"]:
    mfr = h["mejor_franja"]
    if mfr is None:
        continue
    print(f"  {h['metodo']:<30}{h['etiqueta']:>7}{mfr['franja']:>14}"
          f"{mfr['prob_misma_etiqueta']:>9.4f}{mfr['linea_base_esperada']:>9.4f}"
          f"{mfr['mejora_sobre_linea_base']:>+9.4f}{mfr['n_pares']:>8}")
print()
print("  LECTURA: la estructura de repeticion EXISTE (0,84 contra 0,51 de linea base),")
print("  pero ningun metodo pasa del 5 % de cobertura. Los cinco son resultado B.")
"""

CODIGO_VEREDICTO = """\
print("25.3  LOS VECINOS PUROS NO CORRIGEN NADA")
print()
m3 = c["metricas"]
print(f"  {'mecanismo':<34}{'cobertura':>11}{'rescates':>10}{'danios':>8}{'net':>6}{'delta OOF':>12}")
for f in c["tabla_de_mecanismos"]:
    fmt = lambda v, p=6: "-" if v is None else f"{v:.{p}f}"
    print(f"  {f['mecanismo']:<34}{fmt(f['cobertura'], 4):>11}{fmt(f['rescates'], 0):>10}"
          f"{fmt(f['danios'], 0):>8}{fmt(f['neto'], 0):>6}{fmt(f['delta_oof']):>12}")
print()
print("  BARRIDO EXHAUSTIVO, sin selector, todas las configuraciones")
ba = c["barrido_exhaustivo_sin_selector"]
print(f"  configuraciones: {len(ba)} | con al menos un rescate: {c['configuraciones_con_rescates']}")
positivas = [x for x in ba if x["neto"] > 0]
print(f"  configuraciones con net POSITIVO: {len(positivas)}")
print(f"  {'metodo':<32}{'k':>3}{'umbral':>8}{'cobertura':>11}{'resc':>7}{'danios':>8}{'net':>7}")
for x in ba[:8] + ([None] if len(ba) > 12 else []) + ba[-4:]:
    if x is None:
        print("  ...")
        continue
    print(f"  {x['metodo']:<32}{x['k']:>3}{x['umbral']:>8.2f}{x['cobertura']:>11.4f}"
          f"{x['rescates']:>7}{x['danios']:>8}{x['neto']:>+7}")
print()
print("  rescates y danos por franja de similitud del vecino")
print(f"  {'franja':<12}{'celdas':>8}{'rescates':>10}{'danios':>8}{'net':>7}")
for f in c["desglose_por_similitud"]:
    print(f"  {f['franja']:<12}{f['n_celdas']:>8}{f['rescates']:>10}{f['danios']:>8}{f['neto']:>+7}")
print()
print("  detalle de la Fase 5 sobre las filas donde V03 falla")
dd = c["detalle_de_vecinos"]
for k, v in dd.items():
    if k != "nota":
        print(f"    {k:<44} {v}")
print()
print("  LECTURA: en la franja de similitud 0,90-1,01 hay 45 celdas y CERO cambios de")
print("  decision. Donde el vecino es casi una copia, V03 ya acierta. La repeticion de texto")
print("  no aporta nada que V03 no tenga.")
"""

CODIGO_TRANSFORMACIONES = """\
print("25.4  QUE TRANSFORMACION PRODUCE LAS CONTRADICCIONES")
print()
print(f"  {'umbral':>7}{'pares':>9}{'contradictorios':>17}{'fraccion':>11}")
for p in d["pares_por_umbral"]:
    fr = p["fraccion_contradictoria"]
    print(f"  {p['umbral']:>7.2f}{p['n_pares']:>9}{p['n_contradictorios']:>17}"
          f"{(f'{fr:.4f}' if fr is not None else '-'):>11}")
print()
print("  que categoria discrimina, y cual es ruido (comparado con los pares CONCORDANTES)")
disc = [x for x in d["veredicto_por_categoria"] if x["veredicto"] in ("DISCRIMINANTE", "INVERTIDA")]
print(f"  {'umbral':>7}{'categoria':<20}{'razon':>8}{'p Fisher':>12}{'veredicto':<16}")
for x in disc:
    print(f"  {x['umbral']:>7}{x['categoria']:<20}{x['razon_de_razones']:>8.2f}"
          f"{x['p_fisher']:>12.2e}{x['veredicto']:<16}")
print()
print(f"  categorias discriminantes en todo el barrido: {d['n_categorias_discriminantes']}")
print(f"  y las dos son 'sin_clasificar': ninguna categoria SEMANTICA discrimina.")
print()
print("  ejemplos de pares contradictorios")
for e in d["ejemplos_contradictorios"][:6]:
    print(f"    sim {e['similitud']:.4f} | {e['labels_a']} -> {e['labels_b']} | "
          f"cat: {', '.join(e['categorias']) or 'sin clasificar'}")
    print(f"       solo A: {', '.join(e['tokens_solo_a']) or '(ninguno)'}")
    print(f"       solo B: {', '.join(e['tokens_solo_b']) or '(ninguno)'}")
print()
print("  LECTURA: las contradicciones no son un fenomeno semantico tipificado, sino cambio")
print("  lexico arbitrario. Y en casi todos los casos cambian LAS TRES etiquetas a la vez.")
"""

CODIGO_CIERRE = """\
print("25.5  CONCLUSION DE LA FASE")
print()
print(f"  {'lo que SOSTIENE':<20} la estructura de repeticion EXISTE: 0,8405 frente a 0,5080 de")
print(f"  {'':<20} linea base. Real y medido.")
print(f"  {'lo que REFUTA':<20} que contenga informacion que V03 no tenga: de 48 configuraciones,")
print(f"  {'':<20} NINGUNA tiene net positivo; la mejor es -8; en la franja 0,90+ hay cero")
print(f"  {'':<20} cambios de decision.")
print(f"  {'lo que NO SOSTIENE':<20} que la baja cobertura explique el resultado: la conclusion")
print(f"  {'':<20} es la misma con la cobertura mas alta y con la mas baja.")
print(f"  {'lo que NO DICE NADA':<20} nada de esta fase toca N = 1186.")
print()
print("  FUZZY_MATCHING = INFORMATIONAL_ONLY   (resultado B de la directiva)")
print("  NO se abre segunda fase de label propagation: la condicion era que los vecinos")
print("  corrigieran errores reales con dano bajo, y el dano es mayor que el rescate.")
print()
print("  El hallazgo lateral mas util es sobre el DATASET, no sobre V03: por encima de")
print("  similitud 0,85 no existe NI UNA contradiccion de etiqueta. El ruido de anotacion, si lo")
print("  hay, esta en pares con diferencia lexico arbitraria, y ahi el problema no es de")
print("  representacion sino de consistencia entre anotadores.")
print()
g = c["gastos"]
print(f"  Coste de la evaluacion OOF: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB,")
print(f"  {g['predictores_auxiliares']} predictor auxiliar, {g['submissions']} submissions.")
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
    """Anade las celdas de la familia V29 al notebook consolidado.

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
    if "FUZZY_TEMPLATE_MATCHING" in marcas:
        print("  la seccion 25 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(REENCUADRE),
            markdown(METODOS),
            markdown(EL_VEREDICTO),
            markdown(TRANSFORMACIONES),
            markdown(CIERRE),
            codigo(CODIGO_AUDITORIA),
            codigo(CODIGO_METODOS),
            codigo(CODIGO_VEREDICTO),
            codigo(CODIGO_TRANSFORMACIONES),
            codigo(CODIGO_CIERRE),
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 25 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
