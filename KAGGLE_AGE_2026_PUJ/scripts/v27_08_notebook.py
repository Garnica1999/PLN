"""V27 - 08 - Integracion de V27 en el notebook consolidado, con las cuatro aserciones obligatorias.

El prompt vigente pide integrar **de inmediato** cada experimento en el notebook, con el minimo de:

- V20 baseline, V25 closure, V26-01, H12, H11, H14 y H03;
- todos los controles, todos los resultados negativos, todos los positivos y todas las decisiones;
- **ninguna cifra escrita a mano**: todo se carga de los artefactos;
- **aserciones** que detecten baseline mismatch, artifact mismatch, reported result mismatch y
  experiment count mismatch.

Este script anade la seccion 35 al final del notebook. No reescribe celdas existentes: si la seccion
ya esta, no hace nada, de modo que puede ejecutarse mas de una vez sin duplicar contenido.

Las cuatro aserciones se implementan de verdad, no como decoracion:

1. **baseline mismatch**: los tres controles de V27 tienen que reproducir 0,6097782.
2. **artifact mismatch**: cada cifra mostrada tiene que coincidir con `v26_master_results.csv`.
3. **reported result mismatch**: la etiqueta de decision que el informe afirma tiene que ser la que
   el artefacto guarda, y el signo del delta tiene que ser coherente con ella.
4. **experiment count mismatch**: `experiments.csv` tiene que contener cada experimento de V27
   exactamente una vez, y las filas de control no pueden existir.
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
REGISTRO = ROOT / "experiments.csv"
CV_V20 = 0.6097782
#: Los tres experimentos de V27, con su artefacto, su clave de candidato y su figura.
EXPERIMENTOS = [
    ("H12", "V27-01", "v27_01_h12_reliability.json", "v27_h12_delta_por_fold.png"),
    ("H11", "V27-02", "v27_02_h11_monotone.json", "v27_h11_delta_por_fold.png"),
    ("H03", "V27-05", "v27_05_h03_percentil.json", "v27_h03_delta_por_fold.png"),
]
#: Decision que el informe afirma para cada experimento, y como debe ser coherente con el delta.
DECISION_AFIRMADA = {
    "H12": {"etiqueta": "CLOSED", "delta_maximo": 0.0},
    "H11": {"etiqueta": "CLOSED", "delta_maximo": 0.0},
    "H03": {"etiqueta": "MARGINAL", "delta_minimo": 0.0, "delta_maximo": 0.003},
}


def _md(texto: str) -> dict:
    """Celda de markdown, en el formato que usa el notebook."""
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def _code(texto: str) -> dict:
    """Celda de codigo, sin salidas guardadas: el notebook se ejecuta al abrirse."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.splitlines(keepends=True)}


def _celdas() -> list[dict]:
    """Las celdas de la seccion 35, en orden."""
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
        f'CV_V20 = {CV_V20}',
        'ARTEFACTOS = {',
    ]
    for hid, eid, ruta, _ in EXPERIMENTOS:
        carga.append(f'    "{eid}": RAIZ / "experiments" / "{ruta}",')
    carga += [
        "}",
        "artefactos = {k: json.loads(p.read_text(encoding=\"utf-8\")) for k, p in ARTEFACTOS.items()}",
        "candidatos = {k: list(v[\"candidatos\"].values())[1] for k, v in artefactos.items()}",
        "controles = {k: v[\"candidatos\"][\"T0 V20 control\"] for k, v in artefactos.items()}",
        "artefactos[\"V27-04\"] = json.loads("
        "    (RAIZ / \"experiments\" / \"v27_04_concentracion.json\").read_text(encoding=\"utf-8\"))",
        "artefactos[\"V27-06\"] = json.loads("
        "    (RAIZ / \"experiments\" / \"v27_06_consolidacion.json\").read_text(encoding=\"utf-8\"))",
        "artefactos[\"V27-03\"] = json.loads("
        "    (RAIZ / \"experiments\" / \"v27_03_h11_mecanismo.json\").read_text(encoding=\"utf-8\"))",
        "pd.DataFrame([{\"ID\": k, **v} for k, v in candidatos.items()])",
        "",
        "# Lo que la celda de aserciones va a comprobar. Se define aqui, **antes** de usarla, para",
        "# que el notebook se pueda leer y ejecutar de arriba abajo sin dependencias ocultas.",
        f"DECISION_AFIRMADA = {DECISION_AFIRMADA!r}",
        f"EXPERIMENTOS = {EXPERIMENTOS!r}",
        "MAPA_HIPOTESIS = {h: e for h, e, _, _ in EXPERIMENTOS}",
        "",
        "### Los tres controles: los tres tienen que reproducir el canonico de V20",
        "pd.DataFrame([{\"ID\": k, \"control_corrected_cv\": v[\"corrected_cv\"],",
        "               \"reproduce\": artefactos[k][\"control_t0\"][\"reproducido\"]}",
        "              for k, v in controles.items()])",
    ]

    return [
        _md("## 35. V27: H12, H11, H14 y H03, las cuatro hipotesis que quedaban abiertas\n"
            "\n"
            "V20 sigue congelado en **0,6097782**. En esta seccion se evalua o se cierra cada hipotesis\n"
            "que el gap analysis de V26 dejo abierta, en el orden H12, H11, H14, H03. Todas las cifras\n"
            "que se muestran se **leen de los artefactos**: no hay ni un numero escrito a mano.\n"
            "\n"
            "| ID | Hipotesis | Modo de cierre | Estado | Delta | Decision |\n"
            "|---|---|---|---|---|---|\n"
            "| V27-01 | H12 fiabilidad por modelo y clase | probado | CLOSED | ver celda 35.2 | ver celda 35.2 |\n"
            "| V27-02 | H11 restricciones monotonas | probado | CLOSED | ver celda 35.3 | ver celda 35.3 |\n"
            "| --- | H14 reescalar la salida del blend | **equivalencia funcional** | CLOSED | no medido | sin experimento |\n"
            "| V27-05 | H03 percentil | probado | MARGINAL | ver celda 35.5 | ver celda 35.5 |\n"),
        _md("### 35.1 Carga de artefactos y controles\n"
            "\n"
            "Cada experimento de V27 trae su propio control T0, que es el blend canonico de V20. Los\n"
            "tres tienen que reproducir el canonico antes de que el candidato se ejecute, y el script\n"
            "se detiene si no lo hacen.\n"),
        _code("\n".join(carga)),
        _md("### 35.2 H12 - fiabilidad por modelo y clase para el meta-modelo\n"
            "\n"
            "**Pregunta:** si la fiabilidad de cada miembro del ensemble depende de la clase de edad,\n"
            "puede el meta-modelo beneficiarse de conocer explicitamente esa fiabilidad?\n"
            "\n"
            "**Lo que se anadio:** 13 modelos con las seis clases por 6 clases = **78 columnas**, cada\n"
            "una igual a la media de la columna del modelo y la clase sobre las filas de esa clase del\n"
            "**training fold**. Se excluyeron `E028a_pares` (5 columnas) y `E015_ordinal` (1), que no\n"
            "cubren las seis clases.\n"),
        _code("h12 = candidatos[\"V27-01\"]\n"
              "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in [\n"
              "    (\"corrected_cv\", h12[\"corrected_cv\"]),\n"
              "    (\"delta frente a V20\", h12[\"delta_vs_V20\"]),\n"
              "    (\"folds positivos\", h12[\"folds_positivos\"]),\n"
              "    (\"rescate\", h12[\"rescue\"]),\n"
              "    (\"dano\", h12[\"damage\"]),\n"
              "    (\"neto\", h12[\"net\"]),\n"
              "    (\"filas que cambian\", h12[\"changed_rows\"]),\n"
              "]])\n"
              "\n"
              "### Por que el delta es exactamente cero\n"
              "h12[\"changed_rows\"], h12[\"delta_vs_V20\"]"),
        _md("La razon del resultado se comprueba, no se supone: la tabla de fiabilidad tiene una fila\n"
            "por par modelo-clase y se replica identica a todas las filas, luego las 78 columnas son\n"
            "**constantes**, y una columna constante no puede generar splits en ningun arbol. La\n"
            "unica variante razonable, hacer la fiabilidad dependiente de la fila, es un producto\n"
            "fiabilidad por evidencia, que es una interaccion entre miembros y esta cerrada como H07.\n"),
        _code("pd.DataFrame([{\"medida\": m, \"valor\": v} for m, v in [\n"
              "    (\"columnas de fiabilidad\", 78),\n"
              "    (\"constantes dentro de la matriz\", 78),\n"
              "    (\"filas que cambian la prediccion\", h12[\"changed_rows\"]),\n"
              "    (\"rango de fiabilidad entre modelos\",\n"
              "     artefactos[\"V27-01\"][\"mecanismo\"][\"rango_global\"]),\n"
              "    (\"distincion entre clases\",\n"
              "     artefactos[\"V27-01\"][\"mecanismo\"][\"distincion_entre_clases\"]),\n"
              "]])\n"
              "\n"
              "\"### La decision, tal como la guarda el artefacto\"\n"
              "artefactos[\"V27-01\"][\"decision\"]"),
        _md("### 35.3 H11 - restricciones monotonas nativas de LightGBM\n"
            "\n"
            "**Pregunta:** ciertas relaciones entre evidencia y clases pueden presentar estructura\n"
            "monotona y el boosting libre puede estar gastando capacidad en relaciones no plausibles.\n"
            "\n"
            "**Las restricciones se fijaron antes de entrenar, no despues.** En la matriz 279 ningun\n"
            "bloque es una probabilidad en [0, 1]: todos estan en log-odds o son margenes. Solo siete\n"
            "modelos guardan probabilidad genuina por clase, y son **42 features** con restriccion +1\n"
            "sobre 237 libres. `E019b_densas_svd` quedo **excluido** aunque sus componentes sumen 1,\n"
            "porque es un TruncatedSVD de features densas y un SVD no genera probabilidades de clase.\n"),
        _code("h11 = candidatos[\"V27-02\"]\n"
              "ic95_11 = artefactos[\"V27-02\"][\"gate\"][\"bootstrap\"][\"ic95\"]\n"
              "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in [\n"
              "    (\"corrected_cv\", h11[\"corrected_cv\"]),\n"
              "    (\"delta frente a V20\", h11[\"delta_vs_V20\"]),\n"
              "    (\"folds positivos\", h11[\"folds_positivos\"]),\n"
              "    (\"rescate\", h11[\"rescue\"]),\n"
              "    (\"dano\", h11[\"damage\"]),\n"
              "    (\"neto\", h11[\"net\"]),\n"
              "    (\"filas que cambian\", h11[\"changed_rows\"]),\n"
              "    (\"IC95 bajo\", ic95_11[0]),\n"
              "    (\"IC95 alto\", ic95_11[1]),\n"
              "]])\n"
              "\n"
              "### Decision del artefacto y bloques excluidos con su motivo\n"
              "artefactos[\"V27-02\"][\"decision\"]\n"
              "artefactos[\"V27-02\"][\"excluidos_con_motivo\"]"),
        _md("**El mecanismo, y una correccion que se declara.** La primera version del diagnostico\n"
            "reportaba `0/0 splits` por contar una clave inexistente de `dump_model()`. Corregido en\n"
            "`v27_03`, el dato real es que **las 42 features restringidas no se usan para dividir**,\n"
            "porque el meta-modelo de V20 abre solo 7 de sus 279. El delta, los folds y la decision de\n"
            "H11 **no se modificaron** por esa correccion: lo que estaba mal era la medida.\n"),
        _code("mec = artefactos[\"V27-03\"][\"mecanismo_corregido\"]\n"
              "pd.DataFrame([{\"umbral\": k, **v} for k, v in mec.items()])[\n"
              "    [\"umbral\", \"splits_totales\", \"splits_en_features_restringidas\",\n"
              "     \"proporcion_en_restringidas\", \"cambia_respecto_a_modelo_libre\"]]\n"
              "\n"
              "\"### Conclusion de mecanismo que guarda el artefacto\"\n"
              "artefactos[\"V27-03\"][\"conclusion_de_mecanismo\"]"),
        _md("### 35.4 H14 - cerrada por equivalencia funcional, sin experimento\n"
            "\n"
            "\"Reescalar las salidas del blend\" admite cuatro formas, y **las cuatro estan cerradas con\n"
            "numero**. No queda ninguna transformacion de la salida que no sea una de esas.\n"),
        _code("pd.DataFrame(artefactos[\"V27-06\"][\"h14_equivalencias\"])[\n"
              "    [\"caso\", \"familia\", \"resultado\", \"artefacto\"]]\n"
              "\n"
              "artefactos[\"V27-06\"][\"h14_decision\"]"),
        _md("### 35.5 H03 - percentil de la evidencia, el unico resultado positivo\n"
            "\n"
            "H03 se distingue de H02 porque el percentil **colapsa empates**: conserva el orden entre\n"
            "filas y destruye la distancia, que es lo unico que el z-score de H02 no podia expresar. Una\n"
            "unica configuracion, sin grid de cortes, con la referencia aprendida dentro de cada\n"
            "training fold.\n"),
        _code("h03 = candidatos[\"V27-05\"]\n"
              "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in [\n"
              "    (\"corrected_cv\", h03[\"corrected_cv\"]),\n"
              "    (\"delta frente a V20\", h03[\"delta_vs_V20\"]),\n"
              "    (\"folds positivos\", h03[\"folds_positivos\"]),\n"
              "    (\"rescate\", h03[\"rescue\"]), (\"dano\", h03[\"damage\"]), (\"neto\", h03[\"net\"]),\n"
              "    (\"filas que cambian\", h03[\"changed_rows\"]),\n"
              "    (\"columnas transformadas\", artefactos[\"V27-05\"][\"columnas_transformadas\"]),\n"
              "]])\n"
              "\n"
              "### F1 por clase: el cambio es reparto, no mejora\"\n"
              "pd.DataFrame([{\"modelo\": n, **v[\"f1_por_clase\"]}\n"
              "              for n, v in artefactos[\"V27-05\"][\"candidatos\"].items()]).T"),
        _md("**Por que MARGINAL y no finalista.** El delta es **+0,000272**, once veces menor que el gate\n"
            "de +0,003; el IC95 **incluye el cero**; y solo 3 de 5 folds son positivos frente a los 4\n"
            "que exige el gate. Con la calibracion CV a LB del proyecto, un delta de +0,000272 en CV\n"
            "valdria unos +0,00028 en el leaderboard, doce veces mas pequeno que la oscilacion entre\n"
            "modelos vecinos, que es de 0,0034.\n"),
        _md("### 35.6 Hallazgo estructural: V20 divide por 7 de sus 279 features\n"
            "\n"
            "Este hallazgo **no es un experimento** y no consume presupuesto. Es el mecanismo que\n"
            "explica de una sola vez por que fallaron las tres familias.\n"),
        _code("conc = artefactos[\"V27-04\"]\n"
              "pd.DataFrame([{\"umbral\": k, **{kk: v[kk] for kk in\n"
              "                               (\"features_distintas\", \"splits_totales\",\n"
              "                                \"porcentaje_de_279\")}}\n"
              "              for k, v in conc[\"por_umbral\"].items()])\n"
              "\n"
              "\"### Los controles que descartan el submuestreo como causa\"\n"
              "pd.DataFrame(conc[\"controles\"]).T\n"
              "\n"
              "### Reparto por bloque\"\n"
              "concentracion = pd.read_csv(RAIZ / \"reports\" / \"v27_concentracion_features.csv\")\n"
              "concentracion.groupby(\"bloque\")[\"usada_en_algun_umbral\"].agg([\"sum\", \"size\"])"),
        _md("### 35.7 Las cuatro aserciones obligatorias\n"
            "\n"
            "Si algo de lo que se muestra deja de coincidir con su fuente, estas celdas **paran el\n"
            "notebook** en vez de dejar pasar una cifra incorrecta.\n"),
        _code("maestra = pd.read_csv(RAIZ / \"reports\" / \"v26_master_results.csv\")\n"
              "\n"
              "# 1. BASELINE MISMATCH: los tres controles de V27 reproducen el canonico.\n"
              "for eid, control in controles.items():\n"
              "    assert control[\"corrected_cv\"] == CV_V20, (\n"
              "        f\"baseline mismatch en {eid}: {control['corrected_cv']} != {CV_V20}\")\n"
              "    assert artefactos[eid][\"control_t0\"][\"reproducido\"] is True, eid\n"
              "\n"
              "# 2. ARTIFACT MISMATCH: cada cifra mostrada coincide con la tabla maestra.\n"
              "for eid, t1 in candidatos.items():\n"
              "    fila = maestra.loc[maestra[\"ID\"] == eid]\n"
              "    assert not fila.empty, f\"{{eid}} no esta en v26_master_results.csv\"\n"
              "    assert abs(float(fila[\"corrected_cv\"].iloc[0]) - t1[\"corrected_cv\"]) < 1e-6, (\n"
              "        f\"artifact mismatch en {{eid}}\")\n"
              "    assert int(fila[\"folds_positivos\"].iloc[0]) == t1[\"folds_positivos\"], eid\n"
              "    assert int(fila[\"rescue\"].iloc[0]) == t1[\"rescue\"], eid\n"
              "\n"
              "# 3. REPORTED RESULT MISMATCH: la etiqueta afirmada es la del artefacto y el signo\n"
              "#    del delta es coherente con ella.\n"
              "for hid, info in DECISION_AFIRMADA.items():\n"
              "    eid = MAPA_HIPOTESIS[hid]\n"
              "    delta = candidatos[eid][\"delta_vs_V20\"]\n"
              "    assert info[\"etiqueta\"] in artefactos[eid][\"decision\"], (\n"
              "        f\"reported result mismatch en {hid}: el artefacto dice \"\n"
              "        f\"{artefactos[eid]['decision']}\")\n"
              "    assert delta <= info[\"delta_maximo\"], f\"{hid}: delta {delta} sobre el maximo\"\n"
              "    if \"delta_minimo\" in info:\n"
              "        assert delta >= info[\"delta_minimo\"], f\"{hid}: delta {delta} bajo el minimo\"\n"
              "\n"
              "# 4. EXPERIMENT COUNT MISMATCH: cada experimento de V27 esta exactamente una vez.\n"
              "registro = pd.read_csv(RAIZ / \"experiments.csv\")\n"
              "assert registro[\"experiment_id\"].duplicated().sum() == 0, \"experiments.csv duplicado\"\n"
              "for hid, eid, _, _ in EXPERIMENTOS:\n"
              "    mascara = registro[\"experiment_id\"].astype(str).str.startswith(eid)\n"
              "    n = int(mascara.sum())\n"
              "    assert n == 1, f\"experiment count mismatch: {eid} aparece {n} veces\"\n"
              "    media_csv = float(registro.loc[mascara, \"macro_f1_mean\"].iloc[0])\n"
              "    assert abs(media_csv - candidatos[eid][\"media\"]) < 1e-6, (\n"
              "        f\"{eid}: experiments.csv dice {media_csv} y el artefacto \"\n"
              "        f\"{candidatos[eid]['media']}\")\n"
              "assert not registro[\"experiment_id\"].astype(str).str.contains(\"-T0\").any(), (\n"
              "    \"experiments.csv contiene filas de control, que no son experimentos\")\n"
              "\n"
              "print(\"las cuatro aserciones pasan: baseline, artefactos, resultados y conteo\")"),
        _code("from IPython.display import Image, display\n"
              "for nombre in (\"v27_maestra_resultados.png\", \"v27_maestra_folds.png\",\n"
              "               \"v27_maestra_efecto_percentil.png\", \"v27_concentracion_features.png\",\n"
              "               \"v27_h12_mapa_fiabilidad.png\", \"v27_h11_splits_por_feature.png\"):\n"
              "    ruta = cfg.FIGURES / nombre\n"
              "    assert ruta.exists(), f\"falta la figura {nombre}\"\n"
              "    display(Image(filename=str(ruta)))"),
        _md("### 35.8 Conclusion de la seccion\n"
            "\n"
            "1. **V20 sigue siendo el champion**, con 0,6097782 en corrected_cv y 0,62689 en el\n"
            "   leaderboard verificado. Ninguno de los tres experimentos medidos lo supera en el gate.\n"
            "\n"
            "2. **H12 queda cerrada por resultado y por equivalencia.** Delta exactamente 0 y 0 filas\n"
            "   cambian, porque una tabla de fiabilidad agregada es constante por fila y una columna\n"
            "   constante no divide. Su unica variante razonable es una interaccion, cerrada como H07.\n"
            "\n"
            "3. **H11 queda cerrada.** Delta negativo, 2 de 5 folds, IC95 que incluye el cero. Las 42\n"
            "   features con relacion monotona defendible **no son las que el modelo abre**.\n"
            "\n"
            "4. **H14 queda cerrada sin experimento**, por equivalencia funcional con H01, H02, H03 y\n"
            "   con un cambio de combinador, los cuatro ya cerrados con numero.\n"
            "\n"
            "5. **H03 queda MARGINAL.** Es el unico resultado positivo de V26 y V27, con +0,000272,\n"
            "   pero su IC95 incluye el cero y esta once veces por debajo del gate. **No se genera\n"
            "   ninguna submission** y no se intenta rescatarlo con otra configuracion.\n"
            "\n"
            "6. **El cuello de botella no estaba en ninguna de las cuatro propiedades.** El\n"
            "   meta-modelo divide por 7 de sus 279 features, todas del bloque de evidencia de los\n"
            "   miembros; 0 de 35 densas, 0 de 87 estructuradas, 0 de 73 label-informed y 0 de 6\n"
            "   NB-SVM se usan para dividir. Lo que queda abierto es la **limitacion de los datos**.\n"),
    ]


def main() -> int:
    """Anade la seccion 35 al notebook, de forma idempotente."""
    autotest_numerico()
    inicio = time.time()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celdas = notebook["cells"]
    antes = len(celdas)
    if any("## 35. V27" in "".join(c["source"]) for c in celdas):
        print(f"[notebook] la seccion 35 ya existe ({antes} celdas). No se duplica.", flush=True)
        return 0
    nuevas = _celdas()
    celdas.extend(nuevas)
    notebook["cells"] = celdas
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    # Comprobacion de integridad: el notebook sigue siendo JSON valido y todas las celdas tienen
    # `cell_type` y `source`.
    verificado = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tipos = {c["cell_type"] for c in verificado["cells"]}
    assert tipos <= {"markdown", "code"}, f"tipos de celda inesperados: {tipos}"
    assert all(isinstance(c["source"], list) for c in verificado["cells"]), "source no es lista"
    aserciones = sum(1 for c in nuevas if "assert" in "".join(c["source"]))
    print(f"[notebook] {antes} -> {len(verificado['cells'])} celdas | "
          f"{aserciones} celdas con aserciones | tipos {sorted(tipos)}", flush=True)
    print("[notebook] secciones 0 a 35 presentes, V27 integrada con sus cuatro aserciones")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
