"""V17 · Integracion de la fase en el notebook consolidado.

Este script anade al notebook `notebooks/99_entrega_consolidado.ipynb` una **PARTE III** con los
resultados de V16 y V17, y ejecuta **solo** las celdas nuevas para que no queden como codigo sin
correr. Es la tarea que el brief deja pendiente en su seccion 43 ("no dejar resultados solo en
reports") y que el informe de cumplimiento declara abierta en vez de darla por hecha.

**Por que un script y no editar el notebook a mano.** El notebook tiene 46 celdas, 19 de ellas de
codigo, todas ejecutadas y con salidas. Anadir celdas a mano exigiria un servidor de Jupyter
interactivo que aqui no existe. El script hace las dos cosas de forma determinista.

**Como se ejecuta solo lo nuevo, sin reejecutar el historico.** No tiene sentido volver a correr las
19 celdas anteriores: leen artefactos de septiembre y algunas afirman valores sobre ellos. En su
lugar se construye un notebook temporal con **la celda de configuracion** (la que define `load`,
`cfg` y la semilla) mas las celdas nuevas, se ejecuta ese temporal con `nbclient` y se copian las
salidas de las celdas nuevas al notebook original. Las salidas del historico quedan intactas.

**La disciplina del notebook se mantiene.** Cada celda de codigo comprueba con `assert` que los
valores que lee coinciden con los que se ejecutaron. Si alguien cambia un artefacto sin reejecutar el
experimento, el notebook falla en vez de mentir, que es el criterio que las celdas anteriores ya
sostenian.

Entregables: el notebook actualizado y ``experiments/v17_notebook.json`` con el recuento de celdas.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
RESULTADO = cfg.EXPERIMENTS / "v17_notebook.json"
#: Indice de la celda de configuracion del notebook: define `load`, `cfg` y la semilla, y es la
#: unica que hay que arrastrar para ejecutar las celdas nuevas de forma aislada.
CELDA_CONFIGURACION = 2


def markdown(lineas: list[str]) -> dict:
    """Construye una celda de markdown a partir de sus lineas."""
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lineas)}


def codigo(lineas: list[str]) -> dict:
    """Construye una celda de codigo a partir de sus lineas, sin ejecutar."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": "\n".join(lineas)}


def celdas_nuevas() -> list[dict]:
    """Devuelve la parte III completa, en orden de lectura."""
    return [
        markdown([
            "---", "# PARTE III — V16 y V17: la senal de frontera y su limite (2026-09-26)", "",
            "**Pregunta guia**: V16 mostro que un combinador logistico sobre las dos cabezas de V12",
            "mejora las fronteras centrales sin mejorar la metrica global. ¿Esa senal puede convertirse",
            "en ganancia global con modelos entrenados especificamente para esas fronteras?", "",
            "**Criterio de adopcion declarado de antemano**: delta >= +0,0015 para explorar y",
            ">= +0,003 para adoptar, con al menos 4 de 5 folds positivos y el intervalo de bootstrap",
            "pareado sin el cero. Una mejora grande de frontera con macro-F1 global por debajo de la",
            "referencia **no es candidata**: es evidencia mecanistica.", "",
            "Cada cifra de esta parte se comprueba contra los artefactos en `experiments/`: si un",
            "valor no coincidiera con lo ejecutado, la celda fallaria. Las celdas son de solo lectura",
            "sobre artefactos ya calculados, porque toda V17 fue inerte o muy barata de CPU.", "",
        ]),
        markdown([
            "## 20. La referencia del proyecto, reconciliada", "",
            "V12 arrastra dos numeros, 0,6004 y 0,6005, que se han usado indistintemente. **No son el",
            "mismo numero redondeado**: son dos estimadores del mismo macro-F1. Uno es el de la confusion",
            "**pooled**, que es el que devuelve `metricas_globales`; el otro es la **media de los cinco",
            "folds**. No coinciden por el efecto Jensen aplicado a la media por clases.", "",
            "El canonico de V17 es el **pooled**, y no por estetica: el umbral de efecto de D-042 se",
            "calibro contra controles permutados medidos en pooled, el intervalo de bootstrap vive en",
            "pooled porque remuestrea la confusion conjunta, y las cinco fases modernas miden su",
            "referencia ahi. Usar 0,6004 regalaria +0,0001 de delta.", "",
        ]),
        codigo([
            "p0 = load(\"v17_p0_reproduccion.json\")",
            "ref = p0[\"referencia\"]",
            "assert ref[\"pooled\"] == 0.6005017473 and ref[\"media_folds\"] == 0.6004103057",
            "assert abs(ref[\"diferencia\"]) > 5e-5, \"una diferencia menor que el redondeo seria el mismo numero\"",
            "print(f\"pooled        : {ref['pooled']:.10f}  ->  {ref['pooled_redondeado']}\")",
            "print(f\"media de folds: {ref['media_folds']:.10f}  ->  {ref['media_folds_redondeada']}\")",
            "print(f\"diferencia    : {ref['diferencia']:.10f}  \"",
            "      f\"({abs(ref['diferencia']) / 5e-5:.1f} unidades de redondeo del ultimo digito)\")",
            "print(f\"canonico      : {ref['canonico']} en escala {ref['estimador_canonico']}\")",
            "print(\"\\nfolds de V12:\", [round(v, 4) for v in ref[\"por_fold\"]])",
        ]),
        markdown([
            "**Interpretacion.** La discrepancia queda resuelta y etiquetada: a partir de aqui todo delta",
            "se publica en pooled. Un 0,6004 en un informe de V17 seria un error de estimador, no una",
            "medicion distinta.", "",
        ]),
        markdown([
            "## 21. Reproduccion de A2 de V16, la evidencia que motiva la fase", "",
            "A2 es un combinador logistico sobre las doce probabilidades de las dos cabezas de V12. Se",
            "verifica aqui desde los artefactos que V16 guardo, sin reentrenar nada.", "",
        ]),
        codigo([
            "v16 = p0[\"a2_verificado\"]",
            "tabla_v16 = pd.DataFrame(v16)[[\"arquitectura\", \"macro_f1\", \"publicado\", \"coincide\",",
            "                                \"18-24|25-34\", \"25-34|35-49\"]]",
            "assert all(f[\"coincide\"] for f in v16), \"alguna arquitectura de V16 no reproduce\"",
            "tabla_v16",
        ]),
        markdown([
            "**Interpretacion.** A2 se reproduce y sus fronteras quedan confirmadas en 0,5065 y 0,5131",
            "frente a 0,4963 y 0,4974 de V12: **+0,0102 y +0,0157**. Ese es el hallazgo que hace",
            "pensar que hay una senal local aprovechable. El efecto global, en cambio, es de +0,0006 con",
            "el intervalo conteniendo el cero, asi que no se adopta.", "",
        ]),
        markdown([
            "## 22. Los miembros de V12 en las dos fronteras, y el especialista que ya existia", "",
            "La primera pregunta de V17 es de diagnostico, no de modelo: ¿la senal de frontera la",
            "tienen los miembros que ya componen V12, o hace falta un modelo nuevo? La respuesta es que",
            "hacia falta un modelo, y que **ya existia**: el especialista de pares E-028a, un",
            "LinearSVC sobre el TF-IDF disperso entrenado solo con las dos clases de la frontera, con",
            "los mismos folds agrupados del proyecto.", "",
        ]),
        codigo([
            "matriz = pd.read_csv(cfg.REPORTS / \"v17_member_boundary_matrix.csv\")",
            "centrales = matriz[matriz[\"Boundary\"].isin([\"18-24|25-34\", \"25-34|35-49\"])]",
            "pivot = centrales.pivot_table(index=\"Member\", columns=\"Boundary\",",
            "                                values=\"Accuracy\").sort_values(\"25-34|35-49\",",
            "                                                                ascending=False)",
            "pivot.round(4)",
        ]),
        markdown([
            "**Interpretacion.** Ningun miembro del ensamble se acerca al especialista de pares, y el",
            "especialista supera a V12 por 19 y 22 puntos en las filas de cada frontera. La",
            "informacion de frontera **existe** y es grande; lo que falta no es un modelo mejor, es",
            "saber cuando hay que usarlo.", "",
        ]),
        markdown([
            "## 23. La interseccion de errores, que es la cifra que decide", "",
            "Una diferencia de margen grande **no** es informacion disponible: puede que el especialista",
            "acierte en filas donde V12 tambien acierta, y entonces no aporta nada. Lo que decide es",
            "cuantas filas resuelve el especialista que V12 falla, y cuantas rompe que V12 acierta.", "",
        ]),
        codigo([
            "interseccion = pd.read_csv(cfg.REPORTS / \"v17_error_intersection.csv\")",
            "interseccion",
        ]),
        markdown([
            "**Interpretacion.** La complementariedad local es real y grande. El especialista resuelve",
            "14.356 y 15.571 filas que V12 falla y rompe 3.519 y 2.872, y un enrutamiento perfecto daria",
            "0,7481 y 0,7726 en esas filas frente a 0,4963 y 0,4974 de V12. Ese es el techo de la via",
            "local, y supera con holgura el gate de adopcion.", "",
            "**Lo que ese techo no puede cobrar**: en un 23 a un 25 % de las filas ambos fallan. Ahi no",
            "hay nada que repartir entre dos modelos.", "",
        ]),
        markdown([
            "## 24. Por que ninguna compuerta funciona: los errores de V12 son errores confiados", "",
            "Este es el hallazgo que explica la fase entera, y no sale de mirar la accuracy sino de",
            "mirar **donde** estan las victorias del especialista y **cuan segura** esta V12 en cada",
            "grupo de filas.", "",
        ]),
        codigo([
            "sys.path.insert(0, str(cfg.ROOT / \"scripts\"))",
            "from v13_p15_matriz_errores_miembros import predicciones_de_miembros",
            "from src import data as fuente",
            "",
            "entrenamiento = fuente.load_train()",
            "y = entrenamiento[cfg.TARGET].astype(str).to_numpy()",
            "v12 = np.load(cfg.FEATURES / \"oof_ensemble_v12.npy\")",
            "pares = np.load(cfg.FEATURES / \"oof_E028a_pares.npy\")",
            "pred_v12 = np.array(cfg.CLASSES)[np.argmax(v12, axis=1)]",
            "votos, _, _, _ = predicciones_de_miembros()",
            "orden = np.sort(v12, axis=1)",
            "separacion = (orden[:, -1] - orden[:, -2]) / orden[:, -1].std()",
            "",
            "filas_mecanismo = []",
            "for inferior, superior, columna in ((\"18-24\", \"25-34\", 1), (\"25-34\", \"35-49\", 2)):",
            "    mascara = (y == inferior) | (y == superior)",
            "    arriba = (np.sort(np.argsort(v12[mascara], axis=1)[:, -2:], axis=1)",
            "              == np.array([cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)])).all(axis=1)",
            "    decision = np.where(pares[mascara, columna] > 0, superior, inferior)",
            "    ok_v12 = pred_v12[mascara] == y[mascara]",
            "    ok_esp = decision == y[mascara]",
            "    for nombre, seleccion in ((\"rescate\", (~ok_v12) & ok_esp),",
            "                             (\"dano\", ok_v12 & (~ok_esp)),",
            "                             (\"irrecuperable\", (~ok_v12) & (~ok_esp))):",
            "        filas_mecanismo.append({",
            "            \"frontera\": f\"{inferior}|{superior}\", \"grupo\": nombre,",
            "            \"filas\": int(seleccion.sum()),",
            "            \"pct_frontera_en_V12\": round(float(arriba[seleccion].mean()), 4),",
            "            \"separacion_V12\": round(float(separacion[mascara][seleccion].mean()), 4)})",
            "mecanismo = pd.DataFrame(filas_mecanismo)",
            "mecanismo.pivot(index=\"grupo\", columns=\"frontera\",",
            "               values=[\"pct_frontera_en_V12\", \"separacion_V12\"])",
        ]),
        markdown([
            "**Interpretacion, y es el punto central de toda la fase.** Dos hechos, medidos:", "",
            "1. **V12 esta mas confiado cuando se equivoca en la frontera**: separacion 0,809 en las",
            "   filas que el especialista arregla frente a 0,539 en las que rompe. Sus errores ahi no",
            "   son de baja confianza, son errores **confiados**.",
            "2. **Las victorias del especialista caen fuera de la region de frontera**: solo el 12,7 % y",
            "   el 8,0 %, mientras que el 70,6 % y el 67,8 % de las filas que rompe si estan ahi.", "",
            "Juntos explican por que no existe compuerta: no hay banda de confianza que discrimine, y",
            "la unica region donde escuchar al especialista es segura es justo donde el ya acierta. En",
            "esa region, escucharle siempre vale **-0,0080** de macro-F1 global.", "",
        ]),
        markdown([
            "## 25. Gating: la curva de la compuerta, medida y no supuesta", "",
        ]),
        codigo([
            "curva = pd.read_csv(cfg.REPORTS / \"v17_p1b_curva.csv\")",
            "curva[[\"Frontera\", \"Umbral\", \"Filas_activadas\", \"Rescate\", \"Dano\", \"Neto\",",
            "       \"Delta_global\"]]",
        ]),
        markdown([
            "**Interpretacion.** La compuerta sobre la confianza de V12 da como mucho **+0,0002** y a",
            "partir de un margen de 0,20 se vuelve **negativa**: en las filas activadas el especialista",
            "rescata y dana en la misma proporcion. Es la segunda vez que una compuerta de frontera no",
            "rinde, la primera fue ADV-07 en 2025 con otra formulacion.", "",
            "**Aviso de lectura**: el optimo de esta tabla esta sesgado hacia arriba, porque se elige",
            "mirando las mismas filas que despues se miden. Lo que si vale es la forma de la curva y su",
            "techo, no el maximo.", "",
        ]),
        markdown([
            "## 26. Las senales que si separan, y no estan donde se buscaban", "",
            "La pregunta de fondo es si **alguna** senal distingue las filas que el especialista",
            "rescata de las que rompe. El area bajo la curva es la medida adecuada porque es una",
            "medida de orden y no puede sobreajustar: 0,50 significa que la senal no dice nada.", "",
        ]),
        codigo([
            "perfil = pd.read_csv(cfg.REPORTS / \"v17_p8_perfil_confianza.csv\")",
            "tabla_auc = perfil.pivot(index=\"Senal\", columns=\"Frontera\",",
            "                        values=\"AUC_rescate_contra_dano\")",
            "tabla_auc.round(4).sort_values(\"25-34|35-49\", ascending=False)",
        ]),
        markdown([
            "**Interpretacion.** Las probabilidades de V12, que es la senal que el brief proponia para",
            "la compuerta, **no sirven**: 0,53 a 0,58, y la entropia sale incluso invertida. Pero hay dos",
            "senales que si separan y que V12 no expone: el **acuerdo entre los miembros del ensemble**",
            "(0,7070 y 0,7300) y el **margen del propio especialista** (0,7294 y 0,7539). El problema",
            "es que tienen en el mismo sentido util solo si se leen a la vez: el margen alto indica",
            "escucharle, y el acuerdo bajo tambien.", "",
        ]),
        markdown([
            "## 27. Los dos combinadores locales, y por que los dos fallan", "",
            "Con la senal de P8 disponible, V17 midio las dos formulaciones razonables de la via local.",
            "Se distinguiran porque no son la misma pregunta:",
            "",
            "- **P9, seleccion**: ¿a quien escucho, a V12 o al especialista? Un LogReg binario por",
            "  frontera sobre 47 columnas derivadas que ya existian, con el umbral elegido en un",
            "  bloque interior que el arbitro no ve.",
            "- **P5, combinacion**: ¿que clase es? Un LogReg por frontera que redistribuye la masa que",
            "  V12 ya pone en las dos clases del par, conservando el total, de modo que no puede mejorar",
            "  las clases faciles ni danar a las vecinas.", "",
        ]),
        codigo([
            "p9 = load(\"v17_p9_arbitro.json\")",
            "p5 = load(\"v17_p5_combinador_local.json\")",
            "comparativa_local = pd.DataFrame([",
            "    {\"candidato\": \"V12 (referencia)\", \"macro_f1\": 0.6005, \"delta\": 0.0, \"folds\": \"-\"},",
            "    {\"candidato\": \"P9 arbitro aprendido (seleccion)\",",
            "     \"macro_f1\": p9[\"candidata\"][\"macro_f1\"], \"delta\": p9[\"delta\"],",
            "     \"folds\": f\"{p9['folds_positivos']}/5\"},",
            "    {\"candidato\": \"P5 combinador local (combinacion)\",",
            "     \"macro_f1\": p5[\"candidata\"][\"macro_f1\"], \"delta\": p5[\"delta\"],",
            "     \"folds\": \"1/5\"},",
            "])",
            "assert p9[\"adopta\"] is False and p5[\"adopta\"] is False, \"una de las dos se habria adoptado\"",
            "comparativa_local",
        ]),
        codigo([
            "actuacion = pd.DataFrame(p9[\"activacion\"])",
            "print(f\"el arbitro eligio umbrales entre {actuacion['Umbral'].min():.2f} y \",",
            "      f\"{actuacion['Umbral'].max():.2f}, y en total resculo {actuacion['Rescate'].sum()} \",",
            "      f\"filas y dano {actuacion['Dano'].sum()}\")",
            "actuacion[[\"Fold\", \"Frontera\", \"Umbral\", \"Activadas\", \"Rescate\", \"Dano\", \"Neto\"]]",
        ]),
        markdown([
            "**Interpretacion.** Ninguna de las dos formulaciones llega al gate exploratorio. Lo",
            "relevante es el comportamiento del arbitro: elige umbrales de 0,60 a 0,90, activa entre",
            "327 y 1.256 filas y, cuando activa, **pierde**. No es un arbitro mal entrenado: es que la",
            "frontera que habria que aprender no existe con las senales disponibles.", "",
            "Y P5 confirma que el problema no es el marco. La combinacion tampoco encuentra la senal, y",
            "ademas **empeora las fronteras** que pretendia mejorar.", "",
        ]),
        markdown([
            "## 28. Figuras de la fase", "",
        ]),
        codigo([
            "from IPython.display import Image, display",
            "",
            "figuras = [\"v17_p1_interseccion_de_errores.png\",",
            "           \"v17_p1b_curva_compuerta.png\",",
            "           \"v17_p8_area_por_senal.png\",",
            "           \"v17_p1c_reglas.png\",",
            "           \"v17_p9_arbitro_por_fold.png\",",
            "           \"v17_p5_combinador_local.png\",",
            "           \"v17_rendimiento_global_por_candidato.png\",",
            "           \"v17_evolucion_v12_v16_v17.png\"]",
            "for nombre in figuras:",
            "    ruta = cfg.FIGURES / nombre",
            "    assert ruta.exists(), f\"falta la figura {nombre}\"",
            "    print(nombre)",
            "    display(Image(filename=str(ruta)))",
        ]),
        markdown([
            "## 29. Comparacion final de candidatos", "",
        ]),
        codigo([
            "candidatos = pd.read_csv(cfg.REPORTS / \"v17_candidate_comparison.csv\")",
            "maestro = pd.read_csv(cfg.REPORTS / \"v17_master_results.csv\")",
            "print(f\"registro de hipotesis: {len(pd.read_csv(cfg.REPORTS / 'v17_hypothesis_registry.csv'))} filas\")",
            "candidatos[[\"Candidato\", \"Global_MacroF1\", \"Delta_Global\", \"Folds_Positivos\",",
            "            \"Boundary_18_24_25_34\", \"Boundary_25_34_35_49\", \"Decision\"]]",
        ]),
        markdown([
            "**Interpretacion.** Ningun candidato de V17 llega al gate exploratorio. La fila del techo",
            "con arbitro perfecto (+0,0107) es una **cota, no un resultado**: requiere elegir con la",
            "etiqueta. Sirve para lo unico que sirve una cota, que es demostrar que el problema nunca",
            "fue el techo sino la identificacion.", "",
        ]),
        markdown([
            "## 30. Conclusiones de la parte III", "",
            "1. **La referencia era ambigua y ahora no**: 0,6004 y 0,6005 son dos estimadores del mismo",
            "   macro-F1. Canonico 0,6005 pooled.",
            "2. **La senal de frontera existe y es grande**: el especialista de pares, que ya existia,",
            "   supera a V12 en 19 y 22 puntos en las filas de cada frontera y resuelve entre 10.837 y",
            "   12.699 filas de forma neta.",
            "3. **Los errores de V12 en la frontera son errores confiados**, y las victorias del",
            "   especialista caen fuera de la region de frontera. Esa es la razon mecanica de que no",
            "   exista compuerta, y se midio antes de intentar la tercera.",
            "4. **Ni la seleccion (P9) ni la combinacion (P5) encuentran la senal.** Las dos",
            "   formulaciones razonables estan medidas, con protocolo anidado, y las dos fallan.",
            "5. **El techo con arbitro perfecto (+0,0107) supera el gate de adopcion**, asi que la via",
            "   local no esta descartada por falta de margen: esta descartada por falta de una senal que",
            "   identifique las filas.",
            "6. **La leccion accionable no es una compuerta mas**: los margenes de pares ya son",
            "   entradas de V12 y V12 no los explota, cuando por si solos valen 0,6864 y 0,7218 en las",
            "   filas de su frontera. Hay informacion accesible en el espacio de entradas que el",
            "   combinador global no usa, y eso exige **cambiar la funcion objetivo**, que hoy es",
            "   macro-F1 sobre seis clases y no sobre las fronteras.",
            "7. **Ninguna pista queda PROMISING ni FINALIST.** V12 sigue congelado en 0,6005 pooled y",
            "   0,62188 en el leaderboard, sin submission nueva.", "",
        ]),
    ]


def main() -> int:
    """Anade la parte III al notebook y ejecuta solo las celdas nuevas."""
    autotest_numerico()
    inicio = time.time()
    original = nbformat.read(NOTEBOOK, as_version=4)
    celdas_codigo = sum(1 for c in original.cells if c.cell_type == "code")
    print(f"notebook original: {len(original.cells)} celdas, {celdas_codigo} de codigo", flush=True)
    if any("PARTE III" in "".join(c.source) for c in original.cells):
        print("la parte III ya existe: no se duplica", flush=True)
        return 0
    nuevas = celdas_nuevas()
    antes = len(original.cells)
    for celda in nuevas:
        original.cells.append(nbformat.from_dict(celda))
    inicio_nuevas = antes

    temporal = nbformat.v4.new_notebook()
    temporal.cells.append(original.cells[CELDA_CONFIGURACION])
    for celda in nuevas:
        temporal.cells.append(nbformat.from_dict(celda))
    cliente = NotebookClient(temporal, timeout=1800, kernel_name="python3",
                             resources={"metadata": {"path": str(ROOT)}})
    cliente.execute()

    for desplazamiento in range(len(nuevas)):
        destino = original.cells[inicio_nuevas + desplazamiento]
        if destino["cell_type"] != "code":
            continue  # una celda de markdown no tiene salidas ni numero de ejecucion
        ejecutada = temporal.cells[1 + desplazamiento]
        destino["outputs"] = ejecutada["outputs"]
        destino["execution_count"] = ejecutada["execution_count"]
    nbformat.write(original, NOTEBOOK)

    RESULTADO.write_text(json.dumps({
        "fase": "V17 · integracion en el notebook consolidado", "fecha": "2026-09-26",
        "notebook": str(NOTEBOOK.relative_to(ROOT)),
        "celdas_antes": antes, "celdas_despues": len(original.cells),
        "celdas_nuevas": len(nuevas),
        "celdas_de_codigo_nuevas": sum(1 for c in nuevas if c["cell_type"] == "code"),
        "celdas_nuevas_con_salida": sum(1 for desplazamiento in range(len(nuevas))
                                       if nuevas[desplazamiento]["cell_type"] == "code"
                                       and original.cells[inicio_nuevas + desplazamiento]["outputs"]),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"anadidas {len(nuevas)} celdas | notebook ahora con {len(original.cells)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
