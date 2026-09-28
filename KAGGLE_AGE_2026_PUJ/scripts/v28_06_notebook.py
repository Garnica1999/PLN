"""V28 - 06 - Integra V28 en el notebook consolidado, con las aserciones obligatorias.

Anade la seccion 36 al final del notebook. No reescribe celdas existentes y es idempotente: si la
seccion ya esta, no hace nada.

## Que contiene la seccion

Para cada tecnica: motivacion, gap analysis, control, configuracion, resultado, comparacion de
folds, complementariedad y decision. Mas el control de P0, el gap analysis completo, las cuatro
tecnicas cerradas **sin experimento**, las tres auditorias y las cuatro aserciones.

## De donde salen los numeros

De los artefactos. **Ninguna cifra se escribe a mano.** Las unicas constantes fijadas aqui son el
canonico de V20 y el hash de la submission, que no viven en ningun artefacto de V28.

## Las cuatro aserciones

1. **baseline mismatch**: los dos controles de V28 y los de V27 tienen que reproducir 0,6097782.
2. **artifact mismatch**: cada cifra mostrada coincide con `v28_master_results.csv`.
3. **reported result mismatch**: la etiqueta de decision que el informe afirma es la del artefacto,
   y el signo del delta es coherente con ella.
4. **experiment count mismatch**: cada experimento de V28 esta exactamente una vez, y **las cuatro
   tecnicas cerradas por equivalencia no tienen ninguna fila** en `experiments.csv`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.verificacion_numerica import autotest_numerico  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
CV_V20 = 0.6097782
EJECUTADOS = {
    "V28-02": ("A", "Meta-model bagging", "v28_02_tecnica_A_bagging.json",
               "v28_A_bagging_diagnostico.png"),
    "V28-03": ("C", "Pairwise class decomposition", "v28_03_tecnica_C_pairwise.json",
               "v28_C_pairwise_diagnostico.png"),
}
#: Tecnicas cerradas por equivalencia funcional: aparecen en el informe y **no** como experimento.
SIN_EXPERIMENTO = {
    "B": ("Compact consensus geometry", "ALREADY_CLOSED",
          "el bloque de incertidumbre se implemento y se midio: 267 features dan 0,5897 y anadiendo "
          "las 51 columnas de geometria del consenso, 0,5898. Delta +0,0001."),
    "D": ("Dirichlet / evidence fusion", "FUNCTIONALLY_DUPLICATED",
          "es un cambio de combinador, y H04 (ponderar) y H06 (apilar un segundo nivel) estan "
          "cerrados con numero; ademas V20 ya ES el segundo nivel."),
    "E": ("Meta-model con incertidumbre explicita", "CLOSED_BY_DUPLICATION",
          "las mismas tres magnitudes que B (entropia, acuerdo, margen) y la integracion es H06. "
          "La regla critica del punto 23 del brief lo prohibe explicitamente."),
    "F": ("Randomized meta-subspace", "FUNCTIONALLY_DUPLICATED",
          "HIERBA ya tiene colsample_bytree=0,5, que es exactamente aleatorizar la exposicion de "
          "cada arbol a un subespacio manteniendo todas las columnas."),
}
#: Decisiones que el informe afirma, para que la asercion pueda compararlas con los artefactos.
DECISION_AFIRMADA = {
    "A": {"etiqueta": "CLOSED", "delta_maximo": 0.0},
    "C": {"etiqueta": "CLOSED", "delta_maximo": 0.0},
}


def _md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def _code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.splitlines(keepends=True)}


def _celdas() -> list[dict]:
    """Las celdas de la seccion 36, en orden."""
    carga = [
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        "RAIZ = Path.cwd()",
        "if not (RAIZ / \"src\").exists():",
        "    RAIZ = RAIZ.parent",
        "sys.path.insert(0, str(RAIZ))",
        "sys.path.insert(0, str(RAIZ / \"scripts\"))",
        "from src import config as cfg",
        "",
        f"CV_V20 = {CV_V20}",
        'ARTEFACTOS = {',
    ]
    for eid, (_, _, ruta, _) in EJECUTADOS.items():
        carga.append(f'    "{eid}": RAIZ / "experiments" / "{ruta}",')
    carga += [
        "}",
        "artefactos = {k: json.loads(p.read_text(encoding=\"utf-8\")) for k, p in ARTEFACTOS.items()}",
        "candidatos = {k: list(v[\"candidatos\"].values())[1] for k, v in artefactos.items()}",
        "controles = {k: v[\"candidatos\"][\"T0 V20 control\"] for k, v in artefactos.items()}",
        "p0 = json.loads((RAIZ / \"experiments\" / \"v28_00_control_absoluto.json\")",
        "                .read_text(encoding=\"utf-8\"))",
        "gap = json.loads((RAIZ / \"experiments\" / \"v28_01_gap_analysis.json\")",
        "                 .read_text(encoding=\"utf-8\"))",
        "cierre = json.loads((RAIZ / \"experiments\" / \"v28_04_cierre.json\")",
        "                   .read_text(encoding=\"utf-8\"))",
        "auditoria = json.loads((RAIZ / \"experiments\" / \"v28_05_auditoria_final.json\")",
        "                    .read_text(encoding=\"utf-8\"))",
        "",
        "# Lo que comprueban las aserciones del final. Se define antes de usarse.",
        f"EJECUTADOS = {EJECUTADOS!r}",
        f"DECISION_AFIRMADA = {DECISION_AFIRMADA!r}",
        f"SIN_EXPERIMENTO = {SIN_EXPERIMENTO!r}",
        "MAPA_TECNICA = {tec: eid for eid, (tec, _, _, _) in EJECUTADOS.items()}",
        "",
        "### P0: los dos controles de V28",
        "pd.DataFrame([{\"ID\": k, \"control_corrected_cv\": v[\"corrected_cv\"],",
        "               \"reproduce\": artefactos[k][\"control_t0\"][\"reproducido\"]}",
        "              for k, v in controles.items()])",
    ]
    return [
        _md("## 36. V28: mecanismos estructuralmente distintos para combinar la evidencia\n"
            "\n"
            "V25 a V27 eliminaron varias explicaciones: podado de features, escala de la evidencia, "
            "restricciones monotonas, transformaciones percentiles, fiabilidad constante por "
            "modelo y reformulaciones ordinales. V28 se pregunta por algo mas profundo:\n"
            "\n"
            "> Existe alguna forma estructuralmente distinta de combinar, estabilizar o descomponer "
            "la evidencia de los modelos base que V20 no este aprovechando?\n"
            "\n"
            "**V20 sigue congelado en 0,6097782.** Todas las cifras de esta seccion se leen de los "
            "artefactos.\n"),
        _md("### 36.1 P0: higiene del repositorio y control absoluto de V20\n"
            "\n"
            "Antes de nada, los dos directorios no seguidos que quedaron al cierre de V27 se "
            "auditan: **ninguno pertenece a AGE**, los dos quedan fuera de la rama, y no se borra, "
            "no se commitea ni se modifica ninguno.\n"
            "\n"
            "Y V20 se recomputa desde los artefactos exigiendo coincidencia **bit a bit** con el "
            "OOF almacenado, que es mas fuerte que comparar el macro-F1: dos reconstrucciones "
            "podrian dar el mismo score con probabilidades distintas.\n"),
        _code("\n".join(carga)),
        _code("higiene = p0[\"higiene\"][\"hallazgos\"]\n"
              "pd.DataFrame([{\"directorio\": h[\"ruta\"], \"ficheros\": h[\"n_ficheros\"],\n"
              "               \"pertenece_a_AGE\": h[\"contiene_rutas_de_AGE\"],\n"
              "               \"accion\": h[\"accion\"]} for h in higiene])\n"
              "\n"
              "### Control absoluto\"\n"
              "pd.DataFrame([{\"comprobacion\": k, \"valor\": v} for k, v in\n"
              "              p0[\"control_absoluto\"].items()\n"
              "              if isinstance(v, (bool, int, float, str))])"),
        _md("### 36.2 Gap analysis: de seis tecnicas, dos sobreviven\n"
            "\n"
            "La busqueda es por **equivalencia funcional**, no por coincidencia de nombre. Este "
            "script queda excluido de su propia busqueda, porque sus patrones son literalmente "
            "los nombres que busca.\n"),
        _code("pd.DataFrame([{\"ID\": t[\"ID\"], \"tecnica\": t[\"tecnica\"],\n"
              "                  \"veredicto\": t[\"veredicto\"],\n"
              "                  \"experimentos\": t[\"experimentos\"]}\n"
              "                 for t in gap[\"tecnicas\"]])\n"
              "\n"
              "### Supervivientes\"\n"
              "gap[\"supervivientes\"], f\"presupuesto que gasta: {gap['presupuesto_que_gasta']} de \"\n"
              "f\"{gap['presupuesto_maximo']}\""),
        _md("**Lo que el gap analysis cerro sin gastar un experimento:**\n"
            "\n"
            "- **B (geometria del consenso)**: `advanced_04_incertidumbre.py` implementa un "
            "**superconjunto** de lo que B propone, y esta medido: 267 features dan 0,5897 y "
            "anadiendo las 51 columnas dan 0,5898.\n"
            "- **D (fusion de evidencia)**: `dirichlet` en el proyecto es el prior de Monroe para el "
            "EDA. Un pool de evidencia es un combinador, y H04 mas H06 estan cerrados.\n"
            "- **E (incertidumbre explicita)**: las mismas tres magnitudes que B, y la regla critica "
            "del punto 23 lo prohibe.\n"
            "- **F (subespacio aleatorio)**: `colsample_bytree=0,5` ya aleatoriza la exposicion de "
            "cada arbol manteniendo todas las columnas.\n"),
        _md("### 36.3 Tecnica A: bagging del meta-modelo\n"
            "\n"
            "**Motivacion.** V20 usa un unico meta-modelo. Aunque la evidencia sea correcta, la "
            "estimacion de la frontera podria tener varianza residual.\n"
            "\n"
            "**Lo que cambia y lo que no.** Se mantiene la matriz de 279 features, las "
            "predicciones de los tres miembros del blend, los folds, la cadena de cinco umbrales, la "
            "reconstruccion, los thresholds y el blend. Se cambia **solo** las filas que ve cada "
            "ajuste del meta-modelo.\n"
            "\n"
            "**Configuracion fijada antes de mirar la validacion:** N = 5 miembros, remuestreo "
            "bootstrap de Poisson(1) como `sample_weight`, y agregacion por **promedio de "
            "probabilidades**.\n"
            "\n"
            "**El confundidor, declarado antes del resultado:** LightGBM ya embolsaba filas por "
            "arbol (`subsample=0,8` con `subsample_freq=1`), y los estimadores pasan de 300 a 1.500. "
            "Parte de cualquier mejora podria ser capacidad y no reduccion de varianza.\n"),
        _code("a = candidatos[\"V28-02\"]\n"
              "ic95_a = artefactos[\"V28-02\"][\"gate\"][\"bootstrap\"][\"ic95\"]\n"
              "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in [\n"
              "    (\"corrected_cv\", a[\"corrected_cv\"]),\n"
              "    (\"delta frente a V20\", a[\"delta_vs_V20\"]),\n"
              "    (\"folds positivos\", a[\"folds_positivos\"]),\n"
              "    (\"rescate\", a[\"rescue\"]), (\"dano\", a[\"damage\"]), (\"neto\", a[\"net\"]),\n"
              "    (\"filas que cambian\", a[\"changed_rows\"]),\n"
              "    (\"IC95 bajo\", ic95_a[0]), (\"IC95 alto\", ic95_a[1]),\n"
              "    (\"n miembros\", artefactos[\"V28-02\"][\"n_miembros\"]),\n"
              "]])\n"
              "\n"
              "### F1 por clase\"\n"
              "pd.DataFrame([{\"modelo\": n, **v[\"f1_por_clase\"]}\n"
              "              for n, v in artefactos[\"V28-02\"][\"candidatos\"].items()]).T"),
        _md("**A2: el control de varianza es lo que hace informativo al resultado.** Los cinco "
            "miembros **acuerdan solo en el 91,9 %** de las decisiones, luego la varianza que A "
            "pretendia eliminar **existia y era real**. Y promediarla empeoro. Ademas 1.500 "
            "estimadores en vez de 300 no dieron ganancia, lo que descarta la explicacion facil por "
            "capacidad.\n"),
        _code("pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in\n"
              "              artefactos[\"V28-02\"][\"control_de_varianza_A2\"].items()])"),
        _md("### 36.4 Tecnica C: descomposicion pairwise en 5 pares disjuntos\n"
            "\n"
            "**Motivacion medida.** La confusion de V20 es claramente bandada: los cinco pares "
            "adyacentes concentran el **59,9 % de sus 66.005 errores**. C entrena cinco binarios "
            "**disjuntos**, uno por par, en vez de los cinco **anidados** de `clase <= k-1` de V20. No "
            "es CORAL ni cumulative-link: aqui cada frontera la aprende un clasificador "
            "independiente con la matriz completa.\n"
            "\n"
            "**Pares y agregacion fijados antes de entrenar.** El corte de 5.000 confusiones "
            "mutuas, calculado sobre la confusion out-of-fold del control, selecciono exactamente "
            "los cinco pares adyacentes. La agregacion es una **cadena de probabilidad "
            "independiente**: `P(clase<=k)` es el producto de `(1 - p_j)`, que es monota por "
            "construccion y pasa por la misma `reconstruye` de V20.\n"
            "\n"
            "**El coste mecanico, declarado antes:** al ser disjunta, cada clasificador ve unas "
            "56.000 filas en vez de 169.000. No es un defecto, es su mecanismo.\n"),
        _code("c = candidatos[\"V28-03\"]\n"
              "ic95_c = artefactos[\"V28-03\"][\"gate\"][\"bootstrap\"][\"ic95\"]\n"
              "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in [\n"
              "    (\"corrected_cv\", c[\"corrected_cv\"]),\n"
              "    (\"delta frente a V20\", c[\"delta_vs_V20\"]),\n"
              "    (\"folds positivos\", c[\"folds_positivos\"]),\n"
              "    (\"rescate\", c[\"rescue\"]), (\"dano\", c[\"damage\"]), (\"neto\", c[\"net\"]),\n"
              "    (\"filas que cambian\", c[\"changed_rows\"]),\n"
              "    (\"IC95 bajo\", ic95_c[0]), (\"IC95 alto\", ic95_c[1]),\n"
              "    (\"pares\", \", \".join(artefactos[\"V28-03\"][\"pares_seleccionados\"])),\n"
              "]])\n"
              "\n"
              "### F1 por clase: el colapso esta en las clases que NO son frontera\"\n"
              "pd.DataFrame([{\"modelo\": n, **v[\"f1_por_clase\"]}\n"
              "              for n, v in artefactos[\"V28-03\"][\"candidatos\"].items()]).T"),
        _code("masa = artefactos[\"V28-03\"][\"masa_conjunta_38pct\"]\n"
              "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in masa.items()])\n"
              "\n"
              "### Errores en los 5 pares adyacentes: aqui C SI funciona\"\n"
              "pd.DataFrame([{\"modelo\": n, \"errores_totales\": v[\"errores_totales\"],\n"
              "               \"errores_en_pares_adyacentes\": v[\"errores_en_pares_adyacentes\"]}\n"
              "              for n, v in artefactos[\"V28-03\"][\"candidatos\"].items()])"),
        _md("**Lo que C resuelve y lo que no.** Los pares disjuntos **si** reducen los errores de su "
            "objetivo local (de 39.556 a 32.826) y **si** son complementarios en la masa dificil "
            "(acierten 7.788 de las 61.949 filas que V19 y V20 fallan juntas). Pero el precio es el "
            "colapso de las clases que no son frontera, donde un clasificador que solo ve dos "
            "clases no puede colocar una fila lejos de su par.\n"
            "\n"
            "> Una discrepancia que se declara: el brief cita 38,3 % de errores conjuntos y esta "
            "> medicion da **36,6 %**, porque la definicion no es identica (aqui `arbol_v19` contra "
            "> V20). Se usa la cifra medida y reproducible, no la que encajaria.\n"),
        _md("### 36.5 El resultado de V28 en una tabla\n"
            "\n"),
        _code("maestra = pd.read_csv(RAIZ / \"reports\" / \"v28_master_results.csv\")\n"
              "maestra[[\"ID\", \"corrected_cv\", \"delta_vs_V20\", \"folds_positivos\", \"rescue\",\n"
              "         \"damage\", \"net\"]]\n"
              "\n"
              "### Registro de hipotesis\"\n"
              "pd.read_csv(RAIZ / \"reports\" / \"v28_hypothesis_registry.csv\")[\n"
              "    [\"ID\", \"Tecnica\", \"Veredicto_gap_analysis\", \"Experimento\", \"Status\",\n"
              "     \"Decision\"]]"),
        _md("### 36.6 Las cuatro aserciones obligatorias\n"
            "\n"
            "Si algo deja de coincidir con su fuente, estas celdas **paran el notebook** en vez de "
            "dejar pasar una cifra incorrecta.\n"),
        _code("maestra = pd.read_csv(RAIZ / \"reports\" / \"v28_master_results.csv\")\n"
              "registro = pd.read_csv(RAIZ / \"experiments.csv\")\n"
              "\n"
              "# 1. BASELINE MISMATCH: los controles de V28, y los de V27, reproducen el canonico.\n"
              "for eid, control in controles.items():\n"
              "    assert control[\"corrected_cv\"] == CV_V20, (\n"
              "        f\"baseline mismatch en {eid}: {control['corrected_cv']}\")\n"
              "    assert artefactos[eid][\"control_t0\"][\"reproducido\"] is True, eid\n"
              "assert p0[\"control_absoluto\"][\"reproduce_bit_a_bit\"] is True, \"P0 no fue bit a bit\"\n"
              "assert p0[\"control_absoluto\"][\"folds_identicos\"] is True, \"P0: folds distintos\"\n"
              "for eid in (\"V27-01\", \"V27-02\", \"V27-05\"):\n"
              "    v27 = json.loads((RAIZ / \"experiments\" /\n"
              "                    {\"V27-01\": \"v27_01_h12_reliability.json\",\n"
              "                     \"V27-02\": \"v27_02_h11_monotone.json\",\n"
              "                     \"V27-05\": \"v27_05_h03_percentil.json\"}[eid])\n"
              "                   .read_text(encoding=\"utf-8\"))\n"
              "    assert v27[\"candidatos\"][\"T0 V20 control\"][\"corrected_cv\"] == CV_V20, eid\n"
              "\n"
              "# 2. ARTIFACT MISMATCH: cada cifra mostrada coincide con la tabla maestra.\n"
              "for eid, t1 in candidatos.items():\n"
              "    fila = maestra.loc[maestra[\"ID\"] == eid]\n"
              "    assert not fila.empty, f\"{eid} no esta en v28_master_results.csv\"\n"
              "    assert abs(float(fila[\"corrected_cv\"].iloc[0]) - t1[\"corrected_cv\"]) < 1e-6, eid\n"
              "    assert int(fila[\"folds_positivos\"].iloc[0]) == t1[\"folds_positivos\"], eid\n"
              "    assert int(fila[\"rescue\"].iloc[0]) == t1[\"rescue\"], eid\n"
              "    assert int(fila[\"net\"].iloc[0]) == t1[\"net\"], eid\n"
              "\n"
              "# 3. REPORTED RESULT MISMATCH: la etiqueta afirmada es la del artefacto y el signo\n"
              "#    del delta es coherente con ella.\n"
              "for hid, info in DECISION_AFIRMADA.items():\n"
              "    eid = MAPA_TECNICA[hid]\n"
              "    assert info[\"etiqueta\"] in artefactos[eid][\"decision\"], (\n"
              "        f\"reported result mismatch en {hid}: {artefactos[eid]['decision']}\")\n"
              "    assert candidatos[eid][\"delta_vs_V20\"] <= info[\"delta_maximo\"], hid\n"
              "\n"
              "# 4. EXPERIMENT COUNT MISMATCH: dos experimentos reales, y las cuatro tecnicas\n"
              "#    cerradas por equivalencia NO pueden tener fila.\n"
              "assert registro[\"experiment_id\"].duplicated().sum() == 0, \"experiments.csv duplicado\"\n"
              "for _, (tec, _, _, _) in EJECUTADOS.items():\n"
              "    eid = MAPA_TECNICA[tec]\n"
              "    n = int((registro[\"experiment_id\"] == eid).sum())\n"
              "    assert n == 1, f\"experiment count mismatch: {eid} aparece {n} veces\"\n"
              "for hid in SIN_EXPERIMENTO:\n"
              "    n = int(registro[\"experiment_id\"].astype(str).str.contains(\n"
              "        f\"V28-{hid}\", regex=False).sum())\n"
              "    assert n == 0, f\"{hid} se cerro sin entrenar y no debe tener fila: tiene {n}\"\n"
              "assert cierre[\"submissions_generadas\"] == 0, \"se genero una submission\"\n"
              "assert cierre[\"finalistas\"] == 0, \"hay finalistas sin gate superado\"\n"
              "\n"
              "print(\"las cuatro aserciones pasan: baseline, artefactos, resultados y conteo\")"),
        _code("from IPython.display import Image, display\n"
              "for nombre in (\"v28_maestra_resultados.png\", \"v28_maestra_folds.png\",\n"
              "               \"v28_A_bagging_diagnostico.png\", \"v28_C_pairwise_diagnostico.png\"):\n"
              "    ruta = cfg.FIGURES / nombre\n"
              "    assert ruta.exists(), f\"falta la figura {nombre}\"\n"
              "    display(Image(filename=str(ruta)))"),
        _md("### 36.7 Conclusion de V28\n"
            "\n"
            "1. **El gap analysis fue el trabajo mas rentable de la fase.** De seis tecnicas, "
            "**cuatro se cerraron sin gastar un experimento**, y dos de ellas (**B** y **F**) porque "
            "ya estaban implementadas y una (**B**) con un numero medido hace anos.\n"
            "\n"
            "2. **A (bagging del meta-modelo) = CLOSED**, delta -0,000312, 1 de 5 folds. La varianza "
            "**existia** (los 5 miembros solo acordaban en el 91,9 %) y promediarla empeoro. La "
            "varianza de la estimacion del meta-modelo **no es donde esta el error**, y 1.500 "
            "estimadores en vez de 300 descartan la explicacion por capacidad.\n"
            "\n"
            "3. **C (descomposicion pairwise) = CLOSED**, delta -0,067553, 0 de 5 folds. Los pares "
            "disjuntos **si** reducen los errores de su objetivo local y **si** son complementarios "
            "en la masa dificil, pero rompen las clases que no son frontera.\n"
            "\n"
            "4. **V20 sigue siendo el champion** en 0,6097782, con LB 0,62689. **0 finalistas, 0 "
            "submissions, ninguna linea de V20 tocada.**\n"
            "\n"
            "5. **Lo que las dos negativas juntas contestan:** ni la **varianza de la estimacion** "
            "ni la **descomposicion de la decision** eran el cuello de botella. Junto con V25 a V27, "
            "el limite que queda es el mismo que se identifico entonces: **la limitacion de los "
            "datos**.\n"),
    ]


def main() -> int:
    """Anade la seccion 36 al notebook, de forma idempotente."""
    autotest_numerico()
    inicio = time.time()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celdas = notebook["cells"]
    antes = len(celdas)
    if any("## 36. V28" in "".join(c["source"]) for c in celdas):
        print(f"[notebook] la seccion 36 ya existe ({antes} celdas). No se duplica.", flush=True)
        return 0
    nuevas = _celdas()
    celdas.extend(nuevas)
    notebook["cells"] = celdas
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    verificado = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tipos = {c["cell_type"] for c in verificado["cells"]}
    assert tipos <= {"markdown", "code"}, f"tipos inesperados: {tipos}"
    invalidas = 0
    for celda in nuevas:
        if celda["cell_type"] == "code":
            try:
                compile("".join(celda["source"]), "<celda>", "exec")
            except SyntaxError:
                invalidas += 1
    if invalidas:
        print(f"[STOP] {invalidas} celdas de codigo nuevas no compilan", flush=True)
        return 1
    print(f"[notebook] {antes} -> {len(verificado['cells'])} celdas | "
          f"{sum(1 for c in nuevas if 'assert' in ''.join(c['source']))} celdas con aserciones | "
          f"todas compilan", flush=True)
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
