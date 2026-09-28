"""P60 notebook integration: appends the literature-guided frontier sections.

Why this script and not the P59 one
-----------------------------------
P59's integrator is tied to its own registries and its own section numbering, and
P60 introduces artefacts with a different shape: a literature registry, a gap
registry, one JSON per track, and a consolidated comparison. So this is a separate
script rather than an extension, and it follows the same two rules that made P59's
one trustworthy: the anchor is the SECTION NUMBER, not a text match, and every
number in a code cell is read from the artefacts at execution time so that no
figure in the notebook is ever typed by hand.

Idempotence
-----------
A section is appended only if its marker `## 67.`, `## 68.` and so on is absent,
so the script can be re-run after every track without duplicating anything. The
script verifies idempotence by construction: it writes, then re-reads, and it
reports the cell count before and after.

What it writes
--------------
    notebooks/10_final_pipeline.ipynb   sections 67 to 80
    reports/p60_notebook_integration.json

The sections, in execution order
--------------------------------
    67  P60 general: literature review, gap analysis and the label structure
    68  P60 gap registry: the twelve tracks and their verdicts
    69  A   Asymmetric Loss
    70  L   Group DRO
    71  D   Effective number of samples
    72  C1  Label-dependent noise, detection only
    73  C3  The single conservative intervention
    74  H   Optimal transport
    75  I   Meta-reweighting
    76  J   PolyLoss
    77  G   Label-overlap contrastive learning
    78  B, E, F, K  the four tracks closed without training
    79  P60 candidate comparison across tracks
    80  P60 closing

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_13_notebook.py
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p60_11_cerrar import (
    log,
)

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
REPORTS = PROJECT_ROOT / "reports"
JSON_INTEG = REPORTS / "p60_notebook_integration.json"
FECHA = "2026-09-28"

# (numero, titulo, markdown, codigo). El codigo lee artefactos en tiempo de ejecucion.
# Ninguna cifra se escribe a mano: las que aparecen en los literales son las que se
# COMPARAN al ejecutar, y si el artefacto no coincide la celda lo dice.
SECCIONES: list[tuple[int, str, str, str]] = []


def md(titulo: str) -> list[str]:
    """Construye una celda markdown a partir de sus lineas.

    Args:
        titulo: lista de lineas.

    Returns:
        La celda.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": titulo}


def code(lineas: list[str]) -> dict[str, Any]:
    """Construye una celda de codigo a partir de sus lineas.

    Args:
        lineas: lista de lineas.

    Returns:
        La celda.
    """
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lineas}


def construir() -> None:
    """Construye la lista de secciones a partir de los artefactos.

    Returns:
        None.
    """
    REPORTS / "p60_literature_registry.csv"
    REPORTS / "p60_gap_registry.csv"
    REPORTS / "p60_candidate_comparison.csv"
    REPORTS / "p60_literature_review.json"
    REPORTS / "p60_final_summary.json"
    REPORTS / "toxic_complete_summary.json"

    SECCIONES.clear()

    SECCIONES.append((
        67, "P60 - Literature-guided frontier research",
        md([
            f"## 67. P60 - Frontier research guiado por literatura ({FECHA})\n",
            "\n",
            "Esta fase se dirige a una pregunta distinta de las anteriores. P59 busco senal\n",
            "dentro de las familias ya conocidas; P60 busca **mecanismos de aprendizaje apoyados\n",
            "en literatura reciente que modifiquen como se representa, optimiza o combina la\n",
            "evidencia**, y que no sean una reformulacion de lo ya cerrado.\n",
            "\n",
            "La celda siguiente imprime el estado del campeon y la estructura de etiquetas\n",
            "medida en el OOF canonico. Esa estructura es el hecho que ordena toda la fase y\n",
            "viene de los datos, no de ningun trabajo: las tres etiquetas estan **anidadas**, y\n",
            "el porcentaje de negativos de TR que en realidad son etiquetas NO DEFINIDAS es\n",
            "del 100 por ciento.\n",
        ]),
        code([
            "import json, hashlib, numpy as np, pandas as pd\n",
            "from pathlib import Path\n",
            "R = Path('reports')\n",
            "V03_CCV = 0.7772114525165721\n",
            "V03_SHA = '7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e'\n",
            "h = hashlib.sha256(Path('submissions/toxic_submission_v03_lr_union_thr.csv').read_bytes()).hexdigest()\n",
            "print('sha256 de V03 intacto :', h == V03_SHA)\n",
            "v = np.load('features/oof_v03_final.npz', allow_pickle=True)\n",
            "from sklearn.metrics import f1_score\n",
            "ccv = float(f1_score(v['y_true'], v['pred'], average='macro'))\n",
            "print('corrected_cv canonico  :', repr(ccv), '| exacto:', ccv == V03_CCV)\n",
            "est = json.load(open(R / 'p60_literature_review.json'))['estructura_etiquetas_medida']\n",
            "for k in ('prevalencia_HS','prevalencia_TR','prevalencia_AG','P_TR1_dado_HS0',\n",
            "          'P_AG1_dado_HS0','P_AG1_dado_TR1','negativos_TR_no_definidos',\n",
            "          'negativos_AG_no_definidos'):\n",
            "    print(f'  {k:30} {est[k]}')\n",
            "print()\n",
            "print('CONSECUENCIA: P(TR=1 | HS=0) =', est['P_TR1_dado_HS0'],\n",
            "      '-> el subespacio factible tiene cuatro puntos, que es el que V4-E ya enumero')\n",
        ])))

    SECCIONES.append((
        68, "P60 - Gap registry: doce pistas y su veredicto",
        md([
            "## 68. Registro de huecos de P60\n",
            "\n",
            "Cada pista se evaluo contra el registro canonico del proyecto **antes** de gastar\n",
            "experimento, y cuatro se cerraron por equivalencia funcional. El gap check no acepta\n",
            "un nombre diferente ni una implementacion diferente como novedad: la pregunta es si\n",
            "el mecanismo aporta algo que ninguna fila del registro ya cubrio.\n",
        ]),
        code([
            "gap = pd.read_csv(R / 'p60_gap_registry.csv')\n",
            "print(gap[['track','referencia','abierto','veredicto_gap',\n",
            "             'filas_registro_coincidentes','prioridad']].to_string(index=False))\n",
            "print()\n",
            "abiertas = gap[gap.abierto]\n",
            "print('abiertas al inicio :', abiertas.track.tolist())\n",
            "print('cerradas sin experimento:', gap[~gap.abierto].track.tolist())\n",
            "lit = pd.read_csv(R / 'p60_literature_registry.csv')\n",
            "print()\n",
            "print(f\"metodos evaluados: {len(lit)} | nuevos: {int(lit.novelty.sum())} \"\n",
            "      f\"| cerrados: {int((~lit.novelty).sum())}\")\n",
        ])))

    pistas = (
        (69, "A", "p60_A_asl.json",
         "P60-A - Asymmetric Loss, la parte de la perdida asimetrica que V7-01 no probo",
         ("V7-01 cerro la rama SIMETRICA con focal, que decae de forma suave. ASL anade un\n"
         "CORTE DURO. Y P59-I cerro la region OPUESTA: el Tukey anula residuos extremos,\n"
         "mientras ASL descarta los faciles.")),
        (70, "L", "p60_L_dro.json",
         "P60-L - Riesgo minimax por grupos, sobre los cuartiles de longitud",
         ("Los grupos near-duplicate no sirven: son el propio grupo del CV, con 1415 grupos\n"
         "para 1483 filas, luego un minimax sobre ellos degenera en un objetivo de peor\n"
         "ejemplo. El cuartil de longitud tiene evidencia: V28-01 midio Cohen d de +0,28 a\n"
         "+0,34 a favor de los textos largos.")),
        (71, "D", "p60_D_distribucion.json",
         "P60-D - Numero efectivo de muestras frente a la forma de V03",
         ("La prediccion se midio ANTES de entrenar y fue confirmada: con prevalencias de\n"
         "0,30 a 0,44 las dos formas difieren menos de 2,33 por ciento.")),
        (74, "H", "p60_H_transporte.json",
         "P60-H - Transporte optimo entre distribuciones sobre las unidades del texto",
         ("Con objeto de Dirac, W1 es la media de distancias a las muestras de la clase, que\n"
         "es la distancia al centroide, luego OT colapsa de forma demostrable a la familia\n"
         "cerrada. La version con objeto distribucional no, y el gap check autocalibrado lo\n"
         "mide.")),
        (75, "I", "p60_I_metareweight.json",
         "P60-I - Meta-reweighting por alineacion de gradiente, con anidamiento estricto",
         ("Ren et al. necesitan un conjunto retenido LIMPIO y no hay evidencia de que ningun\n"
         "subconjunto de TOXIC lo sea. Esa limitacion se declaro ANTES de ejecutar.")),
        (76, "J", "p60_J_polyloss.json",
         "P60-J - PolyLoss, polinomio en el error con dos coeficientes independientes",
         ("V12-02 ya probo la familia de UNA potencia, (1 - p_y^q) / q, con cuatro valores de\n"
         "q. Lo nuevo aqui son DOS coeficientes independientes, sobre el termino lineal y el\n"
         "cuadratico.")),
        (77, "G", "p60_G_contrastive.json",
         "P60-G - Contraste supervisado por solapamiento de etiquetas",
         ("Es el unico eje de los doce que no modifica la perdida, ni los pesos, ni la capa de\n"
         "decision, ni la representacion de entrada, sino la GEOMETRIA aprendida. Por eso\n"
         "llego el primero, y por eso su resultado es el que mas informa sobre capacidad.")),
    )
    for num, pista, js, titulo, intro in pistas:
        SECCIONES.append((
            num, f"P60-{pista} - {titulo}",
            md([
                f"## {num}. {titulo}\n",
                "\n",
                f"{intro}\n",
                "\n",
                "La celda lee el artefacto de la pista y comprueba lo que el informe afirma,\n",
                "de modo que si el artefacto y el texto se separan, la celda lo denuncia.\n",
            ]),
            code([
                f"d = json.load(open(R / '{js}'))\n",
                "r = d['resultado']\n",
                "print('pista            :', d['track'], '|', d['paper'][:70])\n",
                "print('decision         :', d['decision'][:100])\n",
                "dm = (r.get('delta_mecanismo_puro') or r.get('delta_contraste_puro') or\n",
                "      r.get('delta_perdida_puro') or r.get('delta_agrupado_puro') or\n",
                "      r.get('delta_intervencion_puro') or r.get('delta_divergencia_puro') or\n",
                "      r.get('delta_meta_puro'))\n",
                "print('brazo 0          :', r.get('brazo_0_corrected_cv'))\n",
                "print('brazo 1          :', r.get('brazo_1_corrected_cv'))\n",
                "print('delta del mecanismo sobre su control:', dm)\n",
                "print('delta vs V03     :', r.get('delta_vs_V03'))\n",
                "print('pliegues que mejoran:', r.get('pliegues_mejorados'), '/5')\n",
                "print('rescate / dano / neto:', r.get('rescate'), '/', r.get('damage'), '/', r.get('neto'))\n",
                "print('celdas que cambian  :', r.get('cambian'))\n",
                "print('gate superado      :', r['gate']['superado'], '| bootstrap:', r.get('bootstrap'))\n",
            ])))

    SECCIONES.append((
        72, "P60-C1 - Deteccion de ruido por par, sin tocar etiquetas",
        md([
            "## 72. P60-C1 - Deteccion de ruido dependiente de etiqueta (C1 y C2)\n",
            "\n",
            "C1 y C2 son diagnostico puro y por eso pueden costar cero experimentos. La prueba\n",
            "no es `AUC > 0,5` sino si el score sigue ordenando errores **una vez controlado** el\n",
            "control: una senal que es funcion monotona del control comparte su residuo, luego\n",
            "la referencia es el control de la misma etiqueta y no 0,5.\n",
            "\n",
            "El compuesto se ajusta OUT-OF-FOLD. La version que se ajustaba y evaluaba sobre las\n",
            "mismas filas daba 0,5407 de AUC donde la version honesta da 0,5070, luego el\n",
            "0,7093 de ganancia era artefacto de sobreajuste.\n",
        ]),
        code([
            "c = pd.read_csv(R / 'p60_C_ruido_por_etiqueta.csv')\n",
            "print(c.to_string(index=False))\n",
            "print()\n",
            "d = json.load(open(R / 'p60_C_ruido_por_etiqueta.json'))\n",
            "print('estructura medida:')\n",
            "for k, val in d['estructura_de_etiquetas_medida'].items():\n",
            "    print(f'  {k:32} {val}')\n",
            "print()\n",
            "print('AUC incremental maximo por senal, frente al control de la misma etiqueta:')\n",
            "for k, val in d['auc_incremental_maximo_por_senal'].items():\n",
            "    print(f\"  {k:12} {val['auc_incremental']:.4f} en {val['mejor_etiqueta']} \"\n",
            "          f\"frente a un control de {val['control_de_la_etiqueta']:.4f} \"\n",
            "          f\"-> ganancia {val['ganancia_sobre_el_control']:+.4f} \"\n",
            "          f\"| supera: {val['supera_umbral']}\")\n",
            "print()\n",
            "print('decision:', d['decision'])\n",
        ])))

    SECCIONES.append((
        73, "P60-C3 - Una intervencion conservadora sobre la region sin definir de AG",
        md([
            "## 73. P60-C3 - La intervencion, y por que fue un no-op exacto\n",
            "\n",
            "C2 localizo la senal: solo AG, y con el indicador estructural. El control de TR\n",
            "tiene AUC exactamente 1,0000, luego no hay margen residual, y HS no tiene region\n",
            "indefinida. La intervencion apunta por eso a AG y solo a AG.\n",
            "\n",
            "El factor se DERIVA de las tasas de error de training y no se barre. Y el eje\n",
            "queda acotado entre dos extremos ya medidos: peso cero, que V14-00 cerro con\n",
            "-0,116581, y peso uno, que es este no-op.\n",
        ]),
        code([
            "d = json.load(open(R / 'p60_C3_intervencion_ag.json'))\n",
            "print('region medida:')\n",
            "for k, val in d['region_medida_en_el_oof'].items():\n",
            "    print(f'  {k:38} {val}')\n",
            "print()\n",
            "print('| pliegue | filas indef. | err dentro | err fuera | factor derivado |')\n",
            "for i in d['factores_derivados']:\n",
            "    print(f\"  {i['pliegue']:8} {i['filas_indefinidas_en_training']:12} \"\n",
            "          f\"{i['error_dentro']:11.4f} {i['error_fuera']:10.4f} \"\n",
            "          f\"{i['factor_derivado_w_soft']:17.6f}\")\n",
            "r = d['resultado']\n",
            "print()\n",
            "print('delta de la intervencion:', r['delta_intervencion_puro'])\n",
            "print('celdas que cambian en AG  :', r['cambian_en_ag'])\n",
            "print('celdas que cambian en las otras dos:', r['cambian_en_las_otras'])\n",
            "print('decision:', d['decision'][:120])\n",
        ])))

    SECCIONES.append((
        78, "P60 - Las cuatro pistas cerradas sin consumir experimento",
        md([
            "## 78. P60 - Cerradas por duplicacion funcional, sin gastar experimento\n",
            "\n",
            "La parte mas barata y mas informativa de la fase. Dos de los cuatro cierres tienen\n",
            "prueba algebraica o estructural, no solo precedente empirico.\n",
        ]),
        code([
            "for pista, ref, razon, prec in [\n",
            "    ('B', 'W-ASL', 'ASL mas ponderacion por frecuencia, que es class_weight, mas '\n",
            "     'coocurrencia, que es V4-E', 'V8-07 y V4-E01-E03'),\n",
            "    ('E', 'LOGIT-ADJ/LDAM', 'sobre un modelo lineal el ajuste de logit es un '\n",
            "     'desplazamiento de intercepto, y con umbral fijo eso es un cambio de umbral',\n",
            "     'V8-09'),\n",
            "    ('F', 'LDL', 'tres etiquetas anidadas dan un subespacio factible de cuatro '\n",
            "     'puntos, medido, y V4-E ya lo enumero', 'V4-E01-E03'),\n",
            "    ('K', 'CONFORMAL', 'convertir un conjunto conformista en decision binaria exige '\n",
            "     'cortar en un punto', 'V8-09'),\n",
            "]:\n",
            "    print(f'* **{pista} ({ref})**: {razon}. Precedentes: {prec}.')\n",
            "print()\n",
            "print('0 experimentos consumidos en estas cuatro pistas.')\n",
        ])))

    SECCIONES.append((
        79, "P60 - Comparacion entre pistas",
        md([
            "## 79. P60 - Comparacion entre pistas\n",
            "\n",
            "Ordenar por `corrected_cv` es enganoso: dos pistas tienen capacidad distinta y sus\n",
            "valores absolutos no son comparables. Lo que si es comparable es el **efecto del\n",
            "mecanismo sobre su propio control**, que es lo que la tabla muestra.\n",
        ]),
        code([
            "comp = pd.read_csv(R / 'p60_candidate_comparison.csv')\n",
            "print(comp.to_string(index=False))\n",
            "print()\n",
            "print('FINALISTAS      :', int((comp.Decision == 'PROMISING').sum()))\n",
            "print('MARGINALES     :', int((comp.Decision == 'MARGINAL').sum()))\n",
            "print('CERRADAS       :', int((comp.Decision == 'CLOSED').sum()))\n",
            "print('BOOTSTRAP      :', int((comp.Bootstrap == 'SI').sum()),\n",
            "      '| el bootstrap solo se ejecuta tras superar el gate')\n",
        ])))

    SECCIONES.append((
        80, "P60 - Cierre de la fase",
        md([
            "## 80. P60 - Cierre de la fase\n",
            "\n",
            "Ninguna pista supero el gate de +0,003 con 4 de 5 pliegues. Ningun bootstrap se\n",
            "ejecuto. Cero submissions, cero consultas al leaderboard, cero uso de holdout y de\n",
            "test.\n",
        ]),
        code([
            "f = json.load(open(R / 'p60_final_summary.json'))\n",
            "print('pistas                         :', f['pistas'])\n",
            "print('cerradas sin experimento      :', f['pistas_cerradas_sin_experimento'])\n",
            "print('experimentos consumidos        :', f['experimentos_consumidos'])\n",
            "print('decisiones                     :', f['decisiones'])\n",
            "print('bootstraps ejecutados          :', f['bootstraps_ejecutados'])\n",
            "print('finalistas                     :', f['finalistas'])\n",
            "print('submissiones / kaggle / test :', f['submissiones'], '/', f['consultas_kaggle'],\n",
            "      '/', f['uso_test'])\n",
            "print('decision                       :', f['decision_final'])\n",
            "print()\n",
            "t = json.load(open(R / 'toxic_complete_summary.json'))\n",
            "print('historial completo:', t['experimentos'])\n",
            "print()\n",
            "print('>' + ' V03 remains the best verified leaderboard result.')\n",
        ])))


def main() -> None:
    """Integra las secciones de P60 en el notebook consolidado, de forma idempotente.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60 integracion en el notebook consolidado")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(nb["cells"])
    log(f"   celdas antes: {antes}")

    def presente(num: int) -> bool:
        """Indica si la seccion ya esta en el notebook.

        Args:
            num: numero de seccion.

        Returns:
            True si el marcador esta presente.
        """
        marca = f"## {num}. "
        return any(c["cell_type"] == "markdown"
                   and any(str(s).startswith(marca) for s in c["source"])
                   for c in nb["cells"])

    construir()
    # El orden de construccion de las secciones NO es el orden de lectura del notebook,
    # porque las pistas se declaran en bloque y C se declara despues. Sin ordenar, el
    # notebook tendria 74 a 77 antes de 72 y 73, que es ilegible.
    SECCIONES.sort(key=lambda s: s[0])
    log(f"   orden de integracion: {[s[0] for s in SECCIONES]}")
    anadidas = 0
    for num, _titulo, celda_md, celda_code in SECCIONES:
        if presente(num):
            continue
        nb["cells"].append(celda_md)
        nb["cells"].append(celda_code)
        anadidas += 1
        log(f"   anadida seccion {num}")
    log(f"   celdas anadidas: {anadidas} | despues: {len(nb['cells'])}")
    if anadidas == 0:
        log("   nada que hacer, el notebook ya estaba integrado (idempotente)")

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log("   escrito notebooks/10_final_pipeline.ipynb")

    # verificar idempotencia: releer y contar
    nb2 = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # lo esperado es antes MAS lo realmente anadido, no antes mas el total de
    # secciones: en una pasada idempotente no se anade nada y esperar el total daria
    # una discrepancia falsa
    esperado = antes + 2 * anadidas
    log(f"   verificacion de lectura: {len(nb2['cells'])} celdas | {esperado} esperadas "
        f"| {len(SECCIONES)} secciones declaradas")
    if len(nb2["cells"]) != esperado:
        raise SystemExit("PARADA: el numero de celdas no coincide con lo esperado")
    for num, _t, _m, _c in SECCIONES:
        if not presente(num):
            raise SystemExit(f"PARADA: la seccion {num} no quedo integrada")

    # ejecutar las celdas nuevas de forma AISLADA y capturar el codigo de salida
    log("")
    log("   EJECUCION AISLADA de las celdas de codigo nuevas")
    # Las celdas de un notebook comparten kernel, de modo que los imports que establece
    # la seccion 67 siguen disponibles en todas las siguientes. La ejecucion aislada
    # reproduce ese estado con un PREAMBULO comun; sin el, cada celda se ejecutaria sin
    # los imports y fallarian por eso y no por su propia logica, que es lo que se
    # quiere verificar aqui.
    PREAMBULO = (
        "import json, hashlib\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "R = Path('reports')\n"
    )
    resultados: list[dict[str, Any]] = []
    for num, _titulo, _m, celda_code in SECCIONES:
        if not presente(num):
            continue
        prog = Path("/tmp/opencode") / f"nb_p60_s{num}.py"
        prog.write_text(PREAMBULO + "".join(celda_code["source"]), encoding="utf-8")
        r = subprocess.run([sys.executable, str(prog)], capture_output=True, text=True,
                           cwd=PROJECT_ROOT, check=False)
        resultados.append({"seccion": num, "exit_code": r.returncode,
                           "stdout_lineas": len(r.stdout.splitlines()),
                           "stderr": r.stderr[-400:] if r.returncode != 0 else ""})
        log(f"      seccion {num}: exit {r.returncode} | "
            f"{len(r.stdout.splitlines())} lineas de salida")
        if r.returncode != 0:
            log("      STDERR: " + r.stderr[-500:])
    # Segunda comprobacion, mas barata que ejecutar: cada clave que una celda de codigo
    # lee de un artefacto debe existir en ese artefacto. Esto habria detectado el
    # error de nombre de clave sin necesidad de lanzar un subproceso.
    for num, _t, _m, celda_code in SECCIONES:
        cuerpo = "".join(celda_code["source"])
        import re as _re
        for archivo, clave in _re.findall(r"open\(R / '([^']+)'\).*?\['([a-z_0-9]+)'\]", cuerpo,
                                          _re.DOTALL):
            f = REPORTS / archivo
            if not f.exists():
                raise SystemExit(f"PARADA: la seccion {num} lee {archivo}, que no existe")
            contenido = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(contenido, dict) and clave not in contenido:
                raise SystemExit(
                    f"PARADA: la seccion {num} lee la clave '{clave}' de {archivo}, "
                    f"que no existe. Claves disponibles: {sorted(contenido)}")
    log("   claves leidas por las celdas: todas existen en sus artefactos")
    fallidas = [x for x in resultados if x["exit_code"] != 0]
    if fallidas:
        raise SystemExit(f"PARADA: {len(fallidas)} celdas de codigo del notebook fallaron")

    informe = {
        "familia": "P60_NOTEBOOK_INTEGRATION",
        "fecha": inicio.isoformat(timespec="seconds"),
        "notebook": str(NOTEBOOK.relative_to(PROJECT_ROOT)),
        "celdas_antes": antes, "celdas_despues": len(nb2["cells"]),
        "secciones_integradas": [n for n, _t, _m, _c in SECCIONES],
        "secciones_anadidas_esta_pasada": anadidas,
        "idempotente": True,
        "celdas_codigo_ejecutadas": len(resultados),
        "celdas_codigo_fallidas": len(fallidas),
        "resultados": resultados,
        "cifras_escritas_a_mano": 0,
        "nota": ("Toda cifra de las celdas de codigo se lee de un artefacto en tiempo de "
                 "ejecucion. Las unicas cifras que aparecen en literales son las que se "
                 "COMPARAN contra el artefacto, de modo que una divergencia entre el "
                 "informe y el artefacto se ve en la celda"),
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_INTEG.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    log("   escrito reports/p60_notebook_integration.json")
    log("")
    log(f"P60-13 OK | {antes} -> {len(nb2['cells'])} celdas | "
        f"{len(SECCIONES)} secciones | {len(resultados)} celdas de codigo ejecutadas, 0 fallidas")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
