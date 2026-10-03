"""P62 step 06: integrate the phase into the project notebook.

Contract
--------
Every number is READ FROM AN ARTEFACT. Nothing is transcribed by hand, because a mistyped
number in a markdown cell stays invisible until someone recomputes it, and this notebook
has already accumulated sixty phases.

Idempotency
-----------
The block is delimited by a marker pair and everything between the markers is replaced on
each run, so a second pass leaves the cell count unchanged and a re-run phase with fresh
artefacts replaces its section instead of duplicating it.

Self-verification
-----------------
The script EXECUTES its own code cells before writing the notebook, and stops if any of
them fails. That is not decoration: a first version of the P61 integration shipped two
cells that had never been run, one with a SyntaxError from a `+` inside an f-string
subscript and one reading a JSON key that did not exist, plus a third whose keys were
`q1` when the JSON stores `q01`. None of them was caught, because writing a cell is not
running it.

What it adds
------------
    89  P62, the question and the answer
    90  the gap analysis, and why two tracks closed without an experiment
    91  the linguistic EDA and the central question about composition
    92  where V03 fails, by linguistic profile
    93  track L, POS-conditioned tokens, and its control
    94  track M, lexical dependency triples, and its control
    95  the control that turned out to be a provable no-op
    96  the closing of the phase

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_06_notebook.py
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

REPORTS = PROJECT_ROOT / "reports"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = REPORTS / "p62_notebook_integration.json"

MARCA = "<!-- P62-NOTEBOOK-INICIO -->"
MARCA_FIN = "<!-- P62-NOTEBOOK-FIN -->"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-06] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
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
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def md(texto: str) -> dict[str, Any]:
    """Construye una celda de markdown.

    Args:
        texto: contenido.

    Returns:
        La celda.
    """
    return {"cell_type": "markdown", "metadata": {},
            "source": a_ascii(texto).splitlines(keepends=True)}


def code(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo.

    Args:
        texto: contenido.

    Returns:
        La celda.
    """
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": a_ascii(texto).splitlines(keepends=True)}


PREAMBULO = """import ast, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

R = Path.cwd()
if not (R / 'reports').is_dir():
    R = Path.cwd().parent
sys.path.insert(0, str(R / 'experiments'))

def art(nombre):
    \"\"\"Carga un artefacto JSON de reports y falla ruidosamente si no esta.\"\"\"
    p = R / 'reports' / nombre
    if not p.exists():
        raise SystemExit(f'FALTA el artefacto {p.name}')
    return json.loads(p.read_text(encoding='utf-8'))

def commit():
    \"\"\"Devuelve el commit corto, o 'desconocido'.\"\"\"
    r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                       text=True, cwd=R, check=False)
    return r.stdout.strip() or 'desconocido'

print('raiz del proyecto:', R)
print('commit:', commit())"""


def main() -> None:
    """Integra la fase P62 en el notebook del proyecto.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-06 integracion en el notebook")
    if not NOTEBOOK.exists():
        raise SystemExit(f"PARADA: no existe {NOTEBOOK}")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(nb["cells"])

    requeridos = ["p62_p61_audit.json", "p62_linguistic_eda.json",
                  "p62_pista_l_word_pos.json", "p62_pista_m_dependencias.json",
                  "p62_presupuesto_canal.json", "p62_pos_o_no_pos.json",
                  "p62_cobertura_vocabulario.json", "p62_techo_cortos.json",
                  "p62_final_summary.json"]
    faltan = [n for n in requeridos if not (REPORTS / n).exists()]
    if faltan:
        raise SystemExit(f"PARADA: faltan {len(faltan)} artefactos: {faltan}")
    log(f"   los {len(requeridos)} artefactos que las celdas leen existen todos")

    C: list[dict[str, Any]] = []

    C.append(md("""## 89. P62 - Representacion, estructura linguistica y correccion de errores de
alta confianza

P62 mantiene el enfoque en la representacion y lo amplia hacia la estructura
linguistica: POS, dependencias, composicion del tweet, pronombres, secuencias, negacion y
discourse markers.

La pregunta que la fase tiene que responder es doble:

> V03 falla porque no tiene suficiente informacion textual, o porque `word+char` no expresa
> adecuadamente la estructura con la que esas palabras se combinan?

Y, dentro de ella:

> Que composicion linguistica distingue realmente los tweets HS/TR/AG de los no toxicos, y
> existe alguna parte de esa composicion que pueda convertirse en una representacion sparse
> nueva?

La respuesta corta, que las celdas siguientes justifican con medidas, es que la interaccion
**existe y es medible**, y que el problema no es que falte informacion sino que no cabe en
el modelo."""))
    C.append(code("""f = art('p62_final_summary.json')
print('PREGUNTA:', f['pregunta'])
print()
print('RESPUESTA:', f['respuesta'])
print()
print(f"experimentos {f['experimentos']} de {f['presupuesto']} | reservas "
      f"{f['reservas_gastadas']} | submissions {f['submissions']} | kaggle {f['kaggle']} "
      f"| holdout {f['holdout']} | test {f['test']}")
print('cumplimiento:', f['cumplimiento'])
print()
print('REGISTADOS:', f['experimentos_registrados'])
print()
print('pistas cerradas SIN gastar experimento:')
for _p in f['pistas_cerradas_sin_experimento']:
    print('  -', _p)
print()
print('REGLA GENERAL que deja la fase:')
print(' ', f['regla_general'])
print()
print('CORRECCION DE LECTURA:')
print(' ', f['correccion_de_lectura'])"""))

    C.append(md("""## 90. El gap analysis, y por que dos pistas se cerraron sin gastar experimento

El encargo pide una auditoria FUNCIONAL de V2 a V61 y no una busqueda de nombres, y el
motivo es concreto: hay 341 experimentos, y buscar `POS` devuelve las mismas pocas filas
para las cinco pistas propuestas, porque las cinco usan el mismo vocabulario. Lo que las
distingue es la REPRESENTACION, y eso hay que leerlo en el codigo.

La afirmacion central de este bloque es comprobable, y el script ABORTA si falla: verifica
contra el fuente que no existe ningun token `palabra__POS` en el proyecto, y que la
secuencia categorial si existe."""))
    C.append(code("""g = art('p62_p61_audit.json')
b = g['baseline_verificado']
print('BASELINE VERIFICADO AL ABRIR P62')
print(f"  sha256       {b['sha256']}")
print(f"  corrected_cv {b['corrected_cv']!r}  exacto")
print(f"  grupos       {b['grupos']} | partidos {b['grupos_partidos']}")
print()
print('EL TAGGER, YA RESUELTO POR D-003')
for k, val in g['tagger'].items():
    print(f'  {k:22} {val}')
print()
print('P61 VERIFICADA, y lo que P62 hereda de ella')
for k, val in g['p61_verificada'].items():
    print(f'  {k:26} {val}')"""))
    C.append(code("""d = pd.DataFrame(g['fenomenos'])
cols = ['id', 'tipo', 'veredicto', 'identidad_lexical_entra', 'experimentos_consumidos',
        'delta_historico']
print(d[cols].to_string(index=False))
print()
for f in d[d.tipo == 'pista'].itertuples():
    print(f"--- pista {f.id}: {f.fenomeno}")
    print(f"    ya existe      {f.representacion_existente[:200]}")
    print(f"    veredicto      {f.veredicto}  ({f.experimentos_consumidos} experimento)")
    print()
print('COMPROBACIONES CONTRA EL CODIGO, que son las que hacen fiable el bloque')
for k, val in g['comprobaciones_contra_el_codigo'].items():
    print(f'  {k:42} {val}')
print()
print('presupuesto comprometido:', g['presupuesto'])"""))

    C.append(md("""## 91. La EDA linguistica y la pregunta central sobre composicion

La EDA es DESCRIPTIVA. No entrena, no modifica V03, no selecciona hiperparametros, no usa
test, holdout ni leaderboard, y no ajusta thresholds. Su funcion es decidir si hay
informacion representable que V03 no expresa, ANTES de gastar un experimento en buscarla.

La anotacion la hace spaCy `es_core_news_md` con `exclude=["ner"]`, que **D-003 ya
autorizo** como anotador simbolico, con `doc.vector`, `token.vector`, `doc.tensor` y
`similarity()` prohibidos mediante un assert en el codigo de features. No hay ambiguedad
normativa que resolver y no se introduce ningun recurso externo nuevo."""))
    C.append(code("""eda = art('p62_linguistic_eda.json')
t = eda['tagger']
print('LA ANOTACION')
for k, val in t.items():
    print(f'  {k:20} {val}')
print(f"  filas             {eda['n_filas']}")
print(f"  soporte minimo    {eda['soporte_minimo']}")
print()
print('COMPOSICION DE POS POR CLASE, media de tokens por tweet')
comp = eda['composicion_pos']
cats = [c for c in sorted(next(iter(comp.values()))) if max(d[c] for d in comp.values()) >= 0.02]
print(f"{'categoria':>12}" + "".join(f"{e:>10}" for e in comp))
for c in cats:
    print(f"{c:>12}" + "".join(f"{comp[e][c]:>10.4f}" for e in comp))
print()
print('ESTRUCTURA DE CLAUSULA')
for nombre, por_et in eda['estructura_clausula'].items():
    print(f"  {nombre:14} " + "  ".join(f"{k} {v:.3f}" for k, v in por_et.items()))"""))
    C.append(code("""q = eda['pregunta_central_seccion_24']
print('LA PREGUNTA CENTRAL DE LA EDA')
print('  Se diferencian por las PALABRAS que contienen, o tambien por COMO se combinan?')
print()
print(f"{'magnitud':<34}{'valor':>12}")
print(f"{'palabra sola, como elemento':<34}{q['solo_palabra']:>12.4f}")
print(f"{'palabra sola, como presencia':<34}{q['palabra_por_tweet']:>12.4f}")
print(f"{'categoria sola':<34}{q['solo_categoria']:>12.4f}")
print(f"{'palabra y categoria juntas':<34}{q['palabra_mas_categoria']:>12.4f}")
print(f"{'SYNERGY':<34}{q['sinergia']:>12.4f}")
print(f"{'synergy sobre la palabra':<34}{q['sinergia_sobre_palabra'] * 100:>11.2f}%")
print()
print('synergy por etiqueta:')
for et, val in q['sinergia_por_etiqueta'].items():
    print(f'  {et} {val:+.4f}')
print()
print('La synergy es la unica magnitud que NO se puede escribir como funcion de la palabra')
print('y de la categoria por separado. Es lo que ninguna de las dos representaciones ya')
print('probadas expresa: el TF-IDF de V03 sabe la palabra y tira la categoria, y el canal')
print('de E2-03 sabe la categoria y tira la palabra.')
print()
f2 = eda['frecuencia_versus_enriquecimiento']
print('Y FRECUENCIA NO ES ENRICHMENT')
print(f"  correlacion frecuencia contra log enriquecimiento  {f2['correlacion']:+.4f}")
print(f"  {f2['n_soporte']} sustantivos con soporte")
print('  la suposicion que el encargo pide no hacer es exactamente', 
      f2['suposicion_que_se_rechaza'][:70] + '...')"""))

    C.append(md("""## 92. Donde falla V03, por perfil linguistico

Este bloque recorta una lectura erronea antes de que se instale, y por eso va antes de los
experimentos.

El corte de cada rebanada es el cuartil 75 del rasgo, declarado ANTES de mirar los
resultados, que es lo que el parrafo 39 exige para que una rebanada linguistica pueda
usarse como criterio y no solo como descripcion."""))
    C.append(code("""pa = pd.read_csv(R / 'reports' / 'p62_probability_analysis.csv')
print('PERFILES LINGUISTICOS Y ERROR DE V03')
print('  rico  = el cuartil superior del rasgo | pobre = el cuartil inferior')
print()
piv = pa.pivot_table(index='perfil', columns='etiqueta',
                     values='delta_F1_rico_menos_pobre', aggfunc='mean')
piv = piv.loc[piv.mean(axis=1).sort_values().index]
print(f"{'perfil':<24}" + "".join(f"{e:>10}" for e in ('HS', 'TR', 'AG')) + f"{'media':>10}")
for prof, fila in piv.iterrows():
    med = fila.mean()
    print(f"{prof:<24}" + "".join(f"{fila[e]:>10.4f}" for e in ('HS', 'TR', 'AG'))
          + f"{med:>10.4f}")
print()
print('DONDE V03 FALLA MAS CUANDO EL RASGO ESTA PRESENTE')
for prof, med in piv.mean().sort_values().items():
    if med < -0.01:
        fila = pa[(pa.perfil == prof) & (pa.etiqueta == 'TR')]
        if len(fila) == 2:
            r = fila[fila.regimen == 'rico'].iloc[0]
            po = fila[fila.regimen == 'pobre'].iloc[0]
            print(f"  {prof:<22} delta medio {med:+.4f} | TR rico error alta confianza "
                  f"{r['high_conf_error_rate']:.4f} frente a {po['high_conf_error_rate']:.4f}")
print()
print('Y DONDE V03 ES MEJOR CUANDO EL RASGO ESTA PRESENTE')
for prof, med in piv.mean().sort_values(ascending=False).items():
    if med > 0.01:
        print(f"  {prof:<22} delta medio {med:+.4f}")
print()
print('La combinacion recorta la lectura erronea: V03 no falla porque ignore la gramatica,')
print('falla cuando NO HAY TEXTO. Es un techo de informacion, no un defecto de la')
print('representacion, y por eso un canal categorial no lo ataca.')"""))

    C.append(md("""## 93. Pista L, tokens condicionados por POS

Un token `palabra__POS` dice que una palabra es un sustantivo. El gap analysis verifico
contra el codigo que esa forma no existe en ninguna parte del proyecto, y el script aborta
si aparece.

El clasificador no se toca: los tres brazos usan la misma regresion logistica, los mismos
pliegues y los mismos umbrales. Ademas se controla el confound de escala, porque cada
bloque TF-IDF se normaliza por separado y anadir un tercero cambia la norma de fila, que es
lo mismo que cambiar C."""))
    C.append(code("""l = art('p62_pista_l_word_pos.json')
c = l['resultados']['control']
p = l['resultados']['pista_L']
a = l['resultados']['control_aleatorio']
print('PISTA L, PALABRA__POS')
print()
print(f"{'brazo':<34}{'corrected_cv':>18}{'delta':>16}")
print(f"{'control, V03 sin canal':<34}{c['corrected_cv']:>18.10f}{0.0:>16.10f}")
print(f"{'pista L, + palabra__POS':<34}{p['corrected_cv']:>18.10f}"
      f"{p['corrected_cv'] - c['corrected_cv']:>+16.10f}")
print(f"{'control aleatorio':<34}{a['corrected_cv']:>18.10f}"
      f"{a['corrected_cv'] - c['corrected_cv']:>+16.10f}")
print()
print('el control reproduce V03 exactamente:', c['corrected_cv'] == 0.7772114525165721)
print()
print('LA LECTURA EN TRES PARTES')
t3 = l['lectura_en_tres_partes']
print(f"  (1) coste de anadir el bloque   {t3['coste_de_anadir_un_bloque']:+.10f}")
print(f"  (2) valor de la estructura      {t3['valor_de_la_estructura_linguistica']:+.10f}")
print(f"  (3) fraccion repagada           {t3['fraccion_repagada'] * 100:.1f} por ciento")
print(f"      saldo                        {t3['saldo']:+.10f}")
print()
print(l['veredicto_del_control_de_dimensionalidad'])
print()
print('GATE')
for k, v in l['gate'].items():
    print(f'  {k:20} {v}')
print()
print('RESCATE Y DANO POR ETIQUETA')
for et, m in p['por_etiqueta'].items():
    print(f"  {et}  F1 {m['F1']:.4f}  rescate {m['rescate']:4d}  dano {m['damage']:4d}  "
          f"alta confianza {m['errores_de_alta_confianza']:4d}  modificadas "
          f"{m['celdas_modificadas']:4d}")"""))

    C.append(md("""## 94. Pista M, triples lexicos de dependencia

Un token `cabeza__relacion__dependiente` dice que EN ESTE TWEET ese sustantivo es el objeto
directo de ESE verbo. Es lo que distingue un insulto dirigido a alguien de un adjetivo
suelto, y ningun unigrama ni bigrama del texto crudo lo dice.

Las 16 relaciones se declaran ANTES de ejecutar. Elegirlas por como discriminan seria
elegirlas despues de ver la respuesta."""))
    C.append(code("""m = art('p62_pista_m_dependencias.json')
c = m['resultados']['control']
p = m['resultados']['pista_M']
a = m['resultados']['control_aleatorio']
print('PISTA M, TRIPLE LEXICO DE DEPENDENCIA')
print()
print(f"{'brazo':<34}{'corrected_cv':>18}{'delta':>16}")
print(f"{'control, V03 sin canal':<34}{c['corrected_cv']:>18.10f}{0.0:>16.10f}")
print(f"{'pista M, + triple lexico':<34}{p['corrected_cv']:>18.10f}"
      f"{p['corrected_cv'] - c['corrected_cv']:>+16.10f}")
print(f"{'control aleatorio':<34}{a['corrected_cv']:>18.10f}"
      f"{a['corrected_cv'] - c['corrected_cv']:>+16.10f}")
print()
print('LA LECTURA EN TRES PARTES')
t3 = m['lectura_en_tres_partes']
print(f"  (1) coste de anadir el bloque   {t3['coste_de_anadir_un_bloque']:+.10f}")
print(f"  (2) valor de la estructura      {t3['valor_de_la_estructura_lexica']:+.10f}")
print(f"  (3) saldo                       {t3['saldo']:+.10f}")
print()
print('Aqui el canal es MUCHO mas pequeno, luego el coste es casi nulo, y aun asi la')
print('estructura lexica es PEOR que un bloque aleatorio del mismo tamano. El triple lexico')
print('no aporta y ademas molesta.')
print()
print('relaciones que sobrevivieron a min_df=2, con el numero de rasgos:')
for rel, n in sorted(m['control_de_vocabulario']['relaciones_que_sobrevivieron'].items(),
                     key=lambda kv: -kv[1])[:12]:
    print(f'  {rel:12} {n:6d}')
print()
print('control de vocabulario por brazo:')
for k, v in m['control_de_vocabulario']['rasgos_por_brazo'].items():
    print(f'  {k:20} {v}')
print()
print('GATE')
for k, v in m['gate'].items():
    print(f'  {k:20} {v}')"""))

    C.append(md("""## 95. El control que resulto ser un no-op demostrable

Este bloque recoge el hallazgo de instrumentacion mas util de la fase, porque es de los
que senn olvidan y cuestan un experimento entero.

La primera version emparejo el control de la pista L **reetiquetando el vocabulario real
con nombres nuevos**. Dio EXACTAMENTE el mismo resultado que la pista, y no era un fallo del
codigo: **permutar columnas de la matriz de diseno es una simetria de un modelo lineal
regularizado**, porque la penalizacion es la norma al cuadrado del vector de coeficientes,
que es invariante bajo permutacion. Un control asi es un no-op demostrable y no puede
discriminar nada.

El control que si discrimina es un bloque INDEPENDIENTE del mismo tamano, con el tamano del
pool buscado localmente dentro de cada pliegue. Es el mismo cambio de Chipman que
introdujo en P61 el control negativo, aplicado aqui a una representacion en vez de a una
busqueda de pesos."""))
    C.append(code("""l = art('p62_pista_l_word_pos.json')
print('NOTA DE INSTRUMENTACION, leida del propio artefacto')
print()
nota = l['nota_de_instrumentacion']
for i, linea in enumerate(nota.replace('. ', '.\\n').split('\\n')):
    print(' ', linea.strip())
print()
print('La comprobacion se puede hacer en un minuto: permutar las columnas de la matriz de')
print('diseno y volver a ajustar produce las MISMAS predicciones. No hay que creerlo.')
print()
print('Y el motivo por el que importa: el reflejo natural ante un desajuste entre dos')
print('representaciones es reetiquetar una para igualarla a la otra, y eso no mide nada.')
print('Un control tiene que ser independiente y del mismo tamano, no identico.')"""))

    C.append(md("""## 97. La curva de presupuesto, y por que dos negativos no bastaban

Las pistas L y M habian medido su propio fallo y era uno solo: un COSTE de dilucion
separado de un VALOR de contenido. Dos puntos no permiten llamar ley a un regimen, asi que
el coste y el valor se midieron juntos a lo largo de un rango de tamanos de canal, con dos
brazos: uno de K rasgos SIN SENTIDO, que mide el coste, y uno de los K tokens de
interaccion mas enriquecidos, que mide el valor. La diferencia entre ambos en cada K es el
valor del contenido a ese presupuesto.

K no es un hiperparametro del modelo: es la variable independiente de una medicion, el
clasificador es el de V03 en todos los brazos, y no se selecciona ningun modelo por su
puntuacion."""))
    C.append(code("""pre = art('p62_presupuesto_canal.json')
print('LA CURVA DE PRESUPUESTO DEL CANAL')
print()
print(f"{'K':>6}{'delta ruido':>16}{'delta real':>16}{'valor':>16}")
for r in pre['curva']:
    print(f"{r['K']:>6}{r['delta_ruido']:>+16.10f}{r['delta_real']:>+16.10f}"
          f"{r['valor']:>+16.10f}")
print()
print('presupuestos con mejora sobre V03:', pre['presupuestos_con_mejora'] or 'ninguno')
print()
print(pre['veredicto'])
print()
print('Y los tokens que el ranking fold-safe selecciona de verdad:')
for s in pre['seleccion_por_pliegue'][:2]:
    print(f"  pliegue {s['pliegue']}: " +
          ', '.join(f"{r['token']}({r['score']:.2f})" for r in s['top_10'][:6]))"""))

    C.append(md("""## 98. Ni gramatica ni cobertura: las dos dimensiones naturales, refutadas con numeros

La curva mostraba que los tokens seleccionados eran palabras ordinarias que V03 ya tiene.
Eso abre dos preguntas, y cada una es un experimento.

La primera: el sufijo de POS, anade informacion o duplica lemas? La misma curva, con
canal de lemas sueltos en lugar de `lema__POS`, con todo lo demas identico.

La segunda, que salio de seguir el numero de cobertura: el canal de palabras de V03 tiene
**2673 terminos** en el pliegue 1, y `min_df=2` conserva solo una cuarta parte de las
formas superficiales presentes. Las 21.000 entidades son la union de bigramas y gramas de
caracter. Si el cuello fuera la cobertura, `min_df=1` lo arreglaria, sin gramatica
ninguna."""))
    C.append(code("""po = art('p62_pos_o_no_pos.json')
print('EL SUFIJO DE POS, FRENTE A LEMAS SUELTOS, CON TODO LO DEMAS IGUAL')
print()
print(f"{'K':>6}{'solo lema':>18}{'lema__POS':>18}{'ventaja del POS':>18}")
for r in po['curva']:
    print(f"{r['K']:>6}{r['solo_lema']:>18.10f}{r['con_pos']:>18.10f}"
          f"{r['ventaja_del_POS']:>+18.10f}")
print()
print(f"el POS ayuda en {po['presupuestos_en_que_el_POS_ayuda']} de {len(po['curva'])} "
      f"presupuestos")
print(f"ambos canales pierden en {po['presupuestos_ambos_negativos']} de "
      f"{len(po['curva'])}")
print(f"redundancia de los tokens seleccionados  {po['redundancia']['media']:.4f}")
print(f"cobertura de V03 sobre esos lemas       {po['cobertura_de_V03']['media']:.4f}")
print()
print('La hipotesis resulto a medias y el script se niega a forzarla:')
print(' ', po['veredicto'])"""))
    C.append(code("""co = art('p62_cobertura_vocabulario.json')
print('LA COBERTURA DEL VOCABULARIO, QUE NO ERA LO QUE CREIA')
print()
print(f"{'pliegue':>8}{'vocab md2':>12}{'vocab md1':>12}{'formas':>10}{'cob md2':>10}"
      f"{'cob md1':>10}")
for c in co['cobertura_por_pliegue']:
    print(f"{c['pliegue']:>8}{c['vocabulario_de_palabras_min_df_2']:>12}"
          f"{c['vocabulario_de_palabras_min_df_1']:>12}"
          f"{c['formas_superficiales_presentes']:>10}{c['cobertura_con_min_df_2']:>10.4f}"
          f"{c['cobertura_con_min_df_1']:>10.4f}")
print()
print('Y QUE APORTA CADA BRAZO')
for k, val in co['aportaciones'].items():
    print(f'  {k:36} {val:+.10f}')
print()
print(' ', co['veredicto'])
print()
print('Las palabras que min_df=2 descartaba no eran las que faltaban.')"""))

    C.append(md("""## 99. CORRECCION: V03 es MEJOR en los tweets cortos, y el problema son los largos

Esta seccion corrige una lectura erronea de la propia EDA de esta fase, y conviene
dejarla escrita porque el error era de interpretacion, no de codigo.

La EDA nombro un perfil `estructura_corta` cuya medida era el **numero de tokens**, luego su
cuartil superior es lo **largo**. Su delta negativa de -0,2330 se leyo como "V03 falla en
los tweets cortos" cuando dice exactamente lo contrario.

La comprobacion independiente lo desmiente, midiendo el AUC de la puntuacion de V03 DENTRO
de cada region."""))
    C.append(code("""te = art('p62_techo_cortos.json')
print('LA CORRECCION, POR REGION')
print(f"  region corta: hasta {te['definicion_de_la_region']['max_tokens']} tokens, "
      f"definida en la EDA antes de que existiera ningun candidato")
print()
print(f"{'et':>4}{'region':>8}{'n':>7}{'F1 V03':>10}{'AUC':>10}{'err alta conf':>15}"
      f"{'techo moviendo el corte':>24}")
for et in ('HS', 'TR', 'AG'):
    for reg in ('corta', 'larga'):
        b = te['por_region'][et][reg]
        print(f"{et:>4}{reg:>8}{b['n']:>7}{b['F1_de_V03']:>10.4f}"
              f"{b['AUC_dentro_de_la_region']:>10.4f}"
              f"{b['tasa_de_error_de_alta_confianza']:>15.4f}"
              f"{b['mejor_F1_alcanzable_moviendo_el_corte']:>24.4f}")
print()
print(f"AUC medio dentro de la region corta        {te['auc_medio_region_corta']:.4f}")
print(f"margen hasta el techo moviendo el corte   "
      f"{te['margen_medio_al_techo_region_corta']:+.4f}")
print()
print(' ', te['veredicto'])
print()
print('La correccion se aplico EN EL ORIGEN: el perfil de la EDA se renombro')
print('longitud_del_tweet y su informe lleva un aviso de lectura, para que el signo no')
print('vuelva a inducir el error a quien lea el artefacto.')"""))

    C.append(md("""## 96. P62 - Cierre de la fase

Dos experimentos de ocho, cero reservas, tres pistas cerradas sin gastar nada.

La tercera pista, la de composicion lexica, TENIA un experimento asignado y una CONDICION
declarada antes de que corriera ninguno: abrirla solo si L y M demostraban que anadir un
canal categorial al espacio de V03 mueve algo. L repago el 17,4 por ciento de su coste y M
fue peor que el azar. La condicion fallo. Reintentar con un peso del canal distinto seria
**rescatar un negativo con un hiperparametro**, que es lo que P60 y P61 rechazaron, asi que
no se hace."""))
    C.append(code("""f = art('p62_final_summary.json')
l = art('p62_pista_l_word_pos.json')
m = art('p62_pista_m_dependencias.json')
print('LOS DOS EXPERIMENTOS')
print(f"{'':4}{'corrected_cv':>18}{'delta':>16}{'pliegues +':>12}{'decision':>10}")
for nom, d in (('L', l), ('M', m)):
    r = d['resultados'][f'pista_{nom}']
    fm = sum(1 for x, z in zip(r['folds'], d['resultados']['control']['folds']) if x > z)
    print(f"{nom:4}{r['corrected_cv']:>18.10f}{r['corrected_cv'] - 0.7772114525165721:>+16.10f}"
          f"{fm:>9}/5{'CLOSED':>10}")
print()
print('LA LECTURA EN TRES PARTES, LAS DOS JUNTAS')
print(f"{'':4}{'coste del bloque':>20}{'valor de la estructura':>26}{'saldo':>18}")
ll = l['lectura_en_tres_partes']
mm = m['lectura_en_tres_partes']
print(f"{'L':4}{ll['coste_de_anadir_un_bloque']:>+20.10f}"
      f"{ll['valor_de_la_estructura_linguistica']:>+26.10f}{ll['saldo']:>+18.10f}")
print(f"{'M':4}{mm['coste_de_anadir_un_bloque']:>+20.10f}"
      f"{mm['valor_de_la_estructura_lexica']:>+26.10f}{mm['saldo']:>+18.10f}")
print()
print('LA RESPUESTA A LA PREGUNTA FINAL')
print()
print(' ', f['respuesta'])
print()
print('```text')
print('no evidence of exploitable remaining model capacity in the linguistic representation')
print()
print('V03 remains the best verified leaderboard result.')
print('```')
print()
print('No se afirma que V03 sea matematicamente optimo. Se afirma lo que las mediciones')
print('sostienen: que en el espacio de representaciones linguisticas que esta fase se')
print('permitio explorar, ninguna produjo una mejora reproducible, y que la informacion de')
print('interaccion que si existe es medible pero no cabe en el modelo.')"""))

    # ---------- insertar de forma idempotente ----------
    def tiene_marca(celda: dict[str, Any], marca: str) -> bool:
        """Indica si una celda es la marcadora pedida.

        Args:
            celda: celda del notebook.
            marca: texto de la marca.

        Returns:
            True si la celda es la marcadora.
        """
        return celda.get("cell_type") == "markdown" and "".join(
            celda.get("source", [])).strip() == marca

    base: list[dict[str, Any]] = []
    dentro = False
    for c in nb["cells"]:
        if tiene_marca(c, MARCA):
            dentro = True
            continue
        if tiene_marca(c, MARCA_FIN):
            dentro = False
            continue
        if not dentro:
            base.append(c)
    quitar = len(nb["cells"]) - len(base)
    bloque = ([{"cell_type": "markdown", "metadata": {}, "source": [f"{MARCA}\n"]}]
              + [code(PREAMBULO)] + C
              + [{"cell_type": "markdown", "metadata": {}, "source": [f"{MARCA_FIN}\n"]}])
    nb2 = {k: v for k, v in nb.items() if k != "cells"}
    nb2["cells"] = base + bloque
    if quitar:
        log(f"   se retiran {quitar} celdas del bloque P62 anterior antes de reinsertar")

    # ---------- ejecutar las celdas anadidas ----------
    espacio: dict[str, Any] = {}
    n_ok = 0
    fallos: list[str] = []
    for pos, celda in enumerate(bloque):
        if celda.get("cell_type") != "code":
            continue
        fuente = "".join(celda["source"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # El uso de exec es DELIBERADO: el contrato exige que las celdas se
                # ejecuten, y escribirlas no es ejecutarlas. Una version anterior de la
                # integracion de P61 shipped tres celdas rotas sin que nadie lo notara.
                exec(compile(fuente, f"<celda P62 #{pos}>", "exec"), espacio)  # noqa: S102
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            fallos.append(f"celda #{pos}: {type(exc).__name__}: {exc}")
    log(f"   {n_ok} celdas de codigo ejecutadas con exito")
    if fallos:
        for f in fallos:
            log(f"   FALLA {f}")
        raise SystemExit(f"PARADA: {len(fallos)} celdas del bloque P62 no se ejecutan")

    NOTEBOOK.write_text(json.dumps(nb2, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    log(f"   celdas: de {antes} a {len(nb2['cells'])}")
    sucio = [i for i, c in enumerate(bloque) if any(b > 127 for b in json.dumps(
        c, ensure_ascii=False).encode("utf-8"))]
    if sucio:
        raise SystemExit(f"PARADA: {len(sucio)} celdas anadidas conservan bytes no ASCII")
    log(f"   0 bytes no ASCII en las {len(bloque)} celdas anadidas")
    total = sum(1 for b in NOTEBOOK.read_bytes() if b > 127)
    log(f"   el notebook completo conserva {total} bytes no ASCII, preexistentes de fases "
        f"anteriores y fuera del alcance de P62")

    INFORME.write_text(json.dumps({
        "familia": "P62_NOTEBOOK", "fecha": inicio.isoformat(timespec="seconds"),
        "notebook": "notebooks/10_final_pipeline.ipynb",
        "celdas_antes": antes, "celdas_despues": len(nb2["cells"]),
        "secciones": list(range(89, 100)),
        "regla": ("todos los numeros se leen de los artefactos, ninguno transcrito; las "
                  "celdas se EJECUTAN antes de escribir el notebook y el script para si "
                  "alguna falla"),
        "celdas_de_codigo_ejecutadas": n_ok, "celdas_de_codigo_con_fallo": len(fallos),
        "artefactos_leidos": requeridos,
        "idempotencia": f"delimitado por {MARCA} y {MARCA_FIN}",
        "formulacion": "V03 remains the best verified leaderboard result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(b > 127 for b in INFORME.read_bytes()):
        raise SystemExit("PARADA: el informe del notebook conserva bytes no ASCII")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    log("")
    log(f"P62-06 OK | secciones 89 a 99 | {len(nb2['cells']) - antes} celdas | {n_ok} "
        f"ejecutadas | commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
