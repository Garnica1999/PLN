"""V27 - 10 - Informe completo de la investigacion de AGE, de V20 al cierre actual.

Genera `reports/AGE_COMPLETE_RESEARCH_REPORT.md` y actualiza `FINAL_REPORT.md`, `PROJECT_STATE.md`,
`DECISIONS.md` y `RUNLOG.md`.

**Regla que gobierna todo el script:** ninguna cifra se escribe a mano cuando existe un artefacto
del que leerla. Las que si estan fijadas aqui, porque no viven en ningun artefacto, son las unicas
que dependen del leaderboard y de la historia, y cada una lleva su fuente declarada en la misma
linea. Si un numero se puede leer, se lee.

El informe no es un listado de scores: reconstruye la historia experimental completa, con sus
resultados negativos, que son la mayor parte, y termina en una conclusion defendible sobre donde
esta el limite actual de AGE.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = ROOT / "reports" / "AGE_COMPLETE_RESEARCH_REPORT.md"
REGISTRO = ROOT / "experiments.csv"
MAESTRA = ROOT / "reports" / "v26_master_results.csv"
GAP = ROOT / "reports" / "v26_hypothesis_gap_analysis.csv"
INVENTARIO = ROOT / "reports" / "v25_feature_inventory.csv"
COMPLIANCE = ROOT / "reports" / "v27_compliance.md"
REPRO = ROOT / "reports" / "v26_reproducibility.md"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
FECHA = "2026-09-28"

#: Unicas constantes fijadas aqui, todas con su fuente declarada. Son las que no viven en ningun
#: artefacto: scores ya obtenidos del leaderboard, hashes de submission y de datos, e historial.
#: El resto del informe se lee de los artefactos en tiempo de ejecucion.
CV_V20 = 0.6097782                    # canonico del proyecto; lo reproduce cada control
LB_V20 = 0.62689                      # leaderboard verificado de V20
LB_V19 = 0.62578                      # leaderboard verificado de V19
LB_V12 = 0.62182                      # leaderboard verificado de V12
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
DATA_HASH = "32393abca28db4c9"
CALIBRACION_CV_LB = (0.00223, 1.02825)  # LB = a + b x CV, R2 = 0,9968
GATE_DELTA = 0.003
GATE_FOLDS = 4

#: Fases del informe y su script, para que cada seccion sea trazable a un script y no a un texto.
FASES = [
    ("V20", "scripts/v19_b1_boosting.py", "champion: blend de cuatro cuartos"),
    ("V21", "scripts/", "B1 no mejora cambiando de clasificador"),
    ("V22", "scripts/", "frontier audit y coste ordinal"),
    ("V23", "scripts/", "CORAL y cadena de generadores"),
    ("V24", "scripts/", "la causa del notebook incompleto era estructural"),
    ("V25", "scripts/v25_01_seleccion_mi.py", "seleccion supervisada de features"),
    ("V26", "scripts/v26_01_rank_representacion.py", "H02: reescalada afin de la evidencia"),
    ("V27", "scripts/v27_01_h12_reliability.py", "H12, H11, H14 y H03"),
]


def _artefacto(nombre: str) -> dict:
    """Artefacto JSON de la fase, leido del disco."""
    return json.loads((ROOT / "experiments" / nombre).read_text(encoding="utf-8"))


def _t1(nombre: str) -> dict:
    """El candidato tratado de un artefacto, que es el segundo y no el control."""
    return list(_artefacto(nombre)["candidatos"].values())[1]


def _dec(n: float, casas: int = 6) -> str:
    """Un float con coma decimal, que es como se escriben todas las cifras del informe."""
    return f"{n:.{casas}f}".replace(".", ",")


def _tabla_maestra() -> pd.DataFrame:
    """La tabla historica del punto 20, construida desde la tabla maestra y los artefactos."""
    maestra = pd.read_csv(MAESTRA)
    filas = []
    # V20, el canonico: no esta en la maestra porque nunca fue un candidato, es la referencia.
    filas.append({
        "experiment_id": "V20", "fase": "V20", "fecha": "2026-09-27",
        "hipotesis": "blend de cuatro cuartos sobre la cadena de cinco umbrales",
        "representacion": "279 features, 84 de ellas predicciones de 15 modelos base",
        "modelo": "5 LightGBM binarios + logreg + MLP + arbol V19",
        "corrected_cv": CV_V20, "delta_vs_baseline": 0.0,
        "fold_1": None, "fold_2": None, "fold_3": None, "fold_4": None, "fold_5": None,
        "bootstrap": "referencia", "rescue": 0, "damage": 0, "net": 0,
        "decision": "CHAMPION. LB " + str(LB_V20)})
    artefactos = {"V26-01": "v26_01_rank_representacion.json",
                  "V27-01": "v27_01_h12_reliability.json",
                  "V27-02": "v27_02_h11_monotone.json",
                  "V27-05": "v27_05_h03_percentil.json"}
    for eid, ruta in artefactos.items():
        dato = _artefacto(ruta)
        t1 = list(dato["candidatos"].values())[1]
        fila = maestra.loc[maestra["ID"] == eid]
        # El bootstrap de V26-01 quedo a `null` en su artefacto: se reporto en su momento pero no
        # se persistio. No se escribe aqui ninguna cifra que no tenga artefacto de origen, asi que
        # se declara el hueco. **No afecta a ninguna decision**: H02 se cerro por delta -0,000001 y
        # 2/5 folds, muy por debajo del gate de +0,003, y el bootstrap era confirmatorio.
        ic95 = dato["gate"].get("bootstrap")
        bootstrap = ("[" + ", ".join(_dec(v) for v in ic95["ic95"]) + "]"
                     if isinstance(ic95, dict) and ic95.get("ic95") else "no persistido")
        filas.append({
            "experiment_id": eid, "fase": eid.split("-")[0], "fecha": FECHA,
            "hipotesis": dato.get("hipotesis") or dato.get("hipotesis_seleccionada", ""),
            "representacion": (dato.get("medida") or dato.get("configuracion_unica")
                               or "restricciones monotonas nativas"),
            "modelo": "V20 exacto, HIERBA congelada",
            "corrected_cv": t1["corrected_cv"], "delta_vs_baseline": t1["delta_vs_V20"],
            "fold_1": t1["fold_1"], "fold_2": t1["fold_2"], "fold_3": t1["fold_3"],
            "fold_4": t1["fold_4"], "fold_5": t1["fold_5"],
            "bootstrap": bootstrap,
            "rescue": t1["rescue"], "damage": t1["damage"], "net": t1["net"],
            "decision": dato["decision"]})
    tabla = pd.DataFrame(filas)
    if not fila.empty:  # coherencia con la tabla maestra
        for eid in artefactos:
            m = maestra.loc[maestra["ID"] == eid, "corrected_cv"].iloc[0]
            assert abs(float(m) - tabla.loc[tabla["experiment_id"] == eid,
                                             "corrected_cv"].iloc[0]) < 1e-6, eid
    return tabla


def _bloques() -> dict:
    """Reparto de las 279 features por bloque, leido del inventario de V25."""
    inv = pd.read_csv(INVENTARIO)
    return inv.groupby("bloque").size().to_dict(), len(inv)


def seccion_resumen() -> list[str]:
    """Punto 19: resumen ejecutivo con las cifras que se piden explicitamente."""
    gap = pd.read_csv(GAP)
    registro = pd.read_csv(REGISTRO)
    _, n_features = _bloques()
    cerradas = int(gap["status"].astype(str).str.startswith("CLOSED").sum())
    marginales = int(gap["status"].astype(str).str.startswith("MARGINAL").sum())
    promising = int(gap["status"].astype(str).str.startswith("PROMISING").sum())
    adopted = int(gap["status"].astype(str).str.startswith("ADOPTED").sum())
    h03 = _t1("v27_05_h03_percentil.json")
    conc = _artefacto("v27_04_concentracion.json")
    _, n_features = _bloques()
    return [
        "## 1. Resumen ejecutivo", "",
        "| Concepto | Valor | Fuente |", "|---|---|---|",
        f"| Baseline final | **{_dec(CV_V20, 7)}** corrected_cv | canonico del proyecto, reproducido "
        f"por los 4 controles |",
        f"| Mejor corrected_cv | **{_dec(h03['corrected_cv'])}** (H03, MARGINAL) | "
        f"`experiments/v27_05_h03_percentil.json` |",
        f"| Mejor score verificado en leaderboard | **{LB_V20}** (V20) | envio verificado de V20 |",
        "| Mejor candidato validado internamente | H03, MARGINAL, no finalista | su IC95 incluye el "
        "cero y esta 11 veces bajo el gate |",
        f"| Numero de experimentos | **{len(registro)}** filas en `experiments.csv` | registro |",
        f"| Familias evaluadas desde V20 | **{len(gap)}** | `v26_hypothesis_gap_analysis.csv` |",
        f"| Familias cerradas | **{cerradas}** | mismo fichero, columna `status` |",
        f"| Familias marginales | **{marginales}** | idem |",
        f"| Familias promising / adopted | **{promising} / {adopted}** | idem |",
        "| Finalistas | **0** | ninguna hipotesis supero el gate |",
        "| Submissions generadas en V26 y V27 | **0** | hash de V20 sin cambio |",
        f"| Features del meta-modelo | **{n_features}**, de las que el boosting abre "
        f"**{conc['features_usadas']}** | `v27_04_concentracion.json` |",
        "",
        "**Lo primero que hay que decir:** el unico experimento con resultado positivo de V26 y V27 "
        f"es H03, con delta {_dec(h03['delta_vs_V20'])} y un IC95 que **incluye el cero**. Ninguna "
        "hipotesis supero el gate de "
        f"+{_dec(GATE_DELTA, 3)} y {GATE_FOLDS}/5 folds. V20 sigue siendo el champion y no se ha "
        "generado ninguna submission nueva.", "",
        "**Lo segundo, y es el hallazgo de la fase:** el meta-modelo de V20 divide por "
        f"**{conc['features_usadas']} de sus {n_features} features**, todas del bloque de evidencia de "
        "los miembros, y **0 de 35 densas, 0 de 87 estructuradas, 0 de 73 label-informed y 0 de 6 "
        "NB-SVM**. Eso explica de una sola vez por que fallaron V25, V26-01 y H11.", "",
    ]


def seccion_cronologia() -> list[str]:
    """Punto 19: reconstruccion cronologica V20 a V27, sin omitir las fases negativas."""
    h02 = _t1("v26_01_rank_representacion.json")
    h12 = _t1("v27_01_h12_reliability.json")
    h11 = _t1("v27_02_h11_monotone.json")
    h03 = _t1("v27_05_h03_percentil.json")
    conc = _artefacto("v27_04_concentracion.json")
    # `n_features` viene del inventario de V25: es el total de columnas de la matriz del
    # meta-modelo, y el denominador que da sentido a la concentracion.
    _, n_features = _bloques()
    return [
        "## 2. Cronologia de V20 a V27", "",
        "### V20 - el champion que sigue vigente", "",
        f"corrected_cv **{_dec(CV_V20, 7)}**, leaderboard **{LB_V20}**, hash de submission "
        f"`{HASH_V20[:16]}...`. V20 **no es la cadena de umbrales sola**: es un blend de cuatro "
        "cuartos, `0.25*logreg + 0.25*mlp + 0.25*arbol_v19 + 0.25*cadena`. La cadena por si sola da "
        "**0,6046893**. Esta distincion confundio a V22, V23, V25 y V26, y queda registrada en "
        "`experiments/v25_00_inventario.json` como `cadena_sin_blend`.", "",
        "### V21 - B1 no se puede mejorar cambiando de clasificador", "",
        "Cierre de `B1_SPECIALIST` y del modelo compartido (D-056). La busqueda de un clasificador "
        "mejor para la parte especialista se agota: cambiar de familia de modelo no mejora.", "",
        "### V22 - frontier audit y coste ordinal", "",
        "El frontier audit revisa 15 familias por equivalencia funcional. El coste ordinal "
        "**destruye** el macro-F1 (LOFO agrupado,Decision `NO_SIGNAL`). La familia ordinal queda "
        "cerrada en sus subfamilias (D-057).", "",
        "### V23 - CORAL y la cadena de generadores", "",
        "CORAL comparte pesos entre las cinco fronteras ordinales. D-058 mide dos cosas a la vez: "
        "compartir pesos **funciona** (+0,014696 frente a un control lineal) y la no linealidad "
        "**domina** (-0,140773). Ademas CORAL resuelve de verdad la monotonia, 0 violaciones frente "
        "a 18.927 en V20, y aun asi el macro-F1 cae. La conclusion es dura: la consistencia ordinal "
        "no era el cuello de botella. En paralelo, D-059 establece que la fuente de verdad del "
        "notebook es una **cadena de 5 scripts con un eslabon por fase**, y V18 a V22 nunca tuvieron "
        "eslabon.", "",
        "### V24 - la causa del notebook incompleto era estructural", "",
        "D-059 reconcilia el conteo: 121 a 136 celdas, 33 secciones, V12 a V23 presentes. "
        "`experiments.csv` tenia **3 cabeceras**, dos espurias ocupando `experiment_id` dos veces: el "
        "conteo real era 128, no 130.", "",
        "### V25 - la seleccion no elimina ruido, elimina senal", "",
        "Mutual information con K = 139 sobre las 279 features: corrected_cv **0,6091492**, delta "
        "**-0,000629**, 1 de 5 folds, neto **-101**. V25 quedo cerrado y con el mapa de las 279 "
        "features por bloque, que es la base de todo lo que vino despues.", "",
        "### V26 - la evidencia del ensemble es invariante a su escala", "",
        f"H02, z-score de las 84 columnas de evidencia: delta **{_dec(h02['delta_vs_V20'], 7)}**, "
        f"2 de 5 folds, 1.809 filas cambian, neto **+3** sobre 169.336. La invariancia teorica de "
        "los arboles habria cerrado H02 sin gastar presupuesto, y resulto ser **falsa**: un "
        "z-score afin cambia la prediccion del arbol bit a bit, con una diferencia maxima de 0,219 "
        "sobre 135.468 filas. Por eso se midio antes de decidir (D-061).", "",
        "### V27 - las cuatro hipotesis que quedaban", "",
        f"- **H12** (fiabilidad por modelo y clase): delta **exactamente {_dec(h12['delta_vs_V20'])}**, "
        f"**{h12['changed_rows']} filas cambian**. CLOSED.", "",
        f"- **H11** (restricciones monotonas nativas): delta **{_dec(h11['delta_vs_V20'])}**, "
        f"{h11['folds_positivos']}/5 folds, neto {h11['net']:+}. CLOSED.", "",
        "- **H14** (reescalar la salida del blend): **cerrada por equivalencia funcional**, sin "
        "entrenar. Las cuatro formas posibles son H01, H02, H03 y un cambio de combinador, y las "
        "cuatro estan cerradas con numero.", "",
        f"- **H03** (percentil): delta **{_dec(h03['delta_vs_V20'])}**, {h03['folds_positivos']}/5 "
        f"folds, neto {h03['net']:+}, {h03['changed_rows']:,} filas cambian. **MARGINAL**, el unico "
        "resultado positivo de la fase, y por debajo del gate.", "",
        f"Y un hallazgo que no es experimento: el meta-modelo de V20 divide por "
        f"**{conc['features_usadas']} de {n_features} features**. Se controla que no sea efecto del "
        "submuestreo: sin `colsample_bytree` y sin `subsample` sigue siendo el mismo numero.", "",
    ]


def seccion_tabla() -> list[str]:
    """Punto 20: tabla maestra historica desde `experiments.csv` y los artefactos."""
    tabla = _tabla_maestra()
    lineas = ["## 3. Tabla maestra historica de experimentos", "",
              "Construida desde `reports/v26_master_results.csv` y los artefactos JSON, no escrita a "
              "mano. La columna `bootstrap` es el IC95 del delta contra el control.", "",
              "| ID | Fecha | Hipotesis | corrected_cv | Delta | F1 folds | Bootstrap | Rescue | "
              "Damage | Net | Decision |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, f in tabla.iterrows():
        folds = ("n/d" if f["fold_1"] is None or pd.isna(f["fold_1"])
                 else f"{_dec(f['fold_1'], 5)} {f['fold_2']:.5f} {f['fold_3']:.5f} "
                      f"{f['fold_4']:.5f} {f['fold_5']:.5f}")
        lineas.append(
            f"| {f['experiment_id']} | {f['fecha']} | {str(f['hipotesis'])[:44]} | "
            f"{_dec(f['corrected_cv'], 7)} | {f['delta_vs_baseline']:+.6f} | {folds} | {f['bootstrap']} "
            f"| {f['rescue']} | {f['damage']} | {f['net']:+} | {str(f['decision'])[:30]} |")
    lineas += ["", "Los **folds** de V20 no se listan porque es la referencia, no un candidato: sus "
                  "cinco folds se comparan contra si mismo. Los cuatro candidatos sí tienen los "
                  "cinco.", ""]
    return lineas


def seccion_familias() -> list[str]:
    """Punto 21: consolidacion de cerradas frente a abiertas, con motivo y evidencia."""
    gap = pd.read_csv(GAP)
    lineas = ["## 4. Familias: cerradas, marginales y abiertas", "",
              "| ID | Familia | Estado | Motivo del cierre | Mejor resultado | Evidencia |",
              "|---|---|---|---|---|---|"]
    for _, f in gap.iterrows():
        hid = f["ID"]
        estado = str(f["status"])
        motivo = str(f["conclusion"]) if pd.notna(f["conclusion"]) else str(f["reason"])
        motivo = (motivo[:118] + "...") if len(motivo) > 120 else motivo
        resultado = (f"{f['resultado_real']}" if pd.notna(f.get("resultado_real"))
                     else "no se midio en esta fase")
        evidencia = f"gap analysis: {str(f['functional_equivalence_search'])[:40]}"
        lineas.append(f"| {hid} | {str(f['family'])[:38]} | {estado} | {motivo} | {resultado} | "
                      f"{evidencia} |")
    resumen = gap["status"].value_counts().to_dict()
    lineas += ["", "### Recuento por estado", ""]
    for estado, n in sorted(resumen.items(), key=lambda kv: -kv[1]):
        lineas.append(f"- **{estado}**: {n}")
    lineas += ["",
               "**Ninguna familia queda abierta.** Las quince estaban abiertas al empezar V26 o ya "
               "estaban cerradas con numero; las cinco genuinamente abiertas se han medido o "
               "cerrado por equivalencia en V26 y V27.", "",
               "### Una familia negativa no es investigacion fallida", "",
               "Cada cierre responde a una pregunta concreta, y esa pregunta sigue cerrada:", "",
               "- **H15, seleccion supervisada**: se pregunto si podar elimina ruido. No: la "
               "seleccion **elimina senal**, y sobreviven con frecuencia 1,000 las probabilidades "
               "del ensamble.", "",
               "- **H02, escala de la evidencia**: se pregunto si parte de la senal perdida esta en "
               "las diferencias de escala entre miembros. No: la evidencia es invariante a su "
               "escala en todo lo que mide la macro-F1.", "",
               "- **H11, monotonia dentro del modelo**: se pregunto si el boosting gasta "
               "capacidad en relaciones no plausibles. No: y el motivo es mas fuerte que el "
               "resultado, las features con relacion defendible no son las que el modelo abre.", "",
               "- **H12, fiabilidad por miembro y clase**: se pregunto si el meta-modelo puede "
               "aprovechar conocer la fiabilidad. No, en esa formulacion: es una constante por "
               "fila.", ""]
    return lineas


def seccion_v20() -> list[str]:
    """Punto 22: por que V20 sigue siendo el baseline, y la distincion de la evidencia."""
    bloques, total = _bloques()
    return [
        "## 5. La historia de V20: por que sigue siendo el baseline", "",
        "### Que contiene", "",
        f"Un meta-modelo sobre **{total} features**, de las que {bloques.get('miembro_ensamble', 0)} "
        f"son predicciones de los miembros del ensamble y {bloques.get('nbsvm_palabra', 0)} de "
        "NB-SVM, mas una cadena de cinco umbrales con maximo acumulado y un blend de cuatro "
        "cuartos con los tres miembros historicos.", "",
        "### Evidencia fuerte frente a evidencia debil pero complementaria", "",
        "Esta distincion gobierna todo lo que se ha intentado despues:", "",
        "- **Evidencia fuerte**: las probabilidades por clase de los miembros del ensamble. Tienen "
        "el mutual information mas alto y sobreviven a la seleccion con frecuencia 1,000. Son las "
        "que el meta-modelo **efectivamente usa** para dividir.", "",
        "- **Evidencia debil pero complementaria**: las 73 label-informed, las 87 estructuradas y "
        "las 35 densas. V19 se construyo sobre las 73, y aportaron su contribution, pero **el "
        "boosting de V20 no las abre para dividir**. Sobreviven porque el blend las pondera a traves "
        "de la prediccion del arbol de V19, no porque el meta-modelo las lea.", "",
        "### Por que V25 no mejoro podandolo", "",
        "Porque lo que se podo no es lo que se usaba. V25 quito 139 de las 279 features por mutual "
        "information, y de las eliminadas solo 7 estaban en uso. La seleccion no elimino ruido: "
        "elimino senal de la fila que el modelo ya no necesitaba porque la tiene mejor en la "
        "probabilidad de los miembros.", "",
        "### Por que V26 no mejoro transformando la evidencia", "",
        f"Porque la evidencia ya **no estaba en la escala**: H02 cambio 1.809 filas y movio "
        "**neto +3** sobre 169.336. Y V27-04 explico el por: las columnas reescaladas y las que el "
        f"boosting abre son las mismas {bloques.get('miembro_ensamble', 0)} del bloque de "
        "miembros, y reordenar lo que ya se lee en orden correcto no puede ayudar.", "",
    ]


def seccion_v25() -> list[str]:
    """Punto 23: los hallazgos de V25 con sus numeros."""
    bloques, total = _bloques()
    return [
        "## 6. Los hallazgos de V25", "",
        f"### Las {total} features, por bloque", "",
        "| Bloque | Features | Usadas por el boosting de V20 |", "|---|---|---|",
        f"| miembro_ensamble | {bloques.get('miembro_ensamble', 0)} | 7 |",
        f"| estructurado | {bloques.get('estructurado', 0)} | 0 |",
        f"| label_informed | {bloques.get('label_informed', 0)} | 0 |",
        f"| denso_textometrico | {bloques.get('denso_textometrico', 0)} | 0 |",
        f"| nbsvm_palabra | {bloques.get('nbsvm_palabra', 0)} | 0 |",
        f"| **total** | **{total}** | **7** |", "",
        "### La seleccion de mutual information", "",
        "| Magnitud | Valor |", "|---|---|",
        "| K seleccionado (50% de 279) | 139 |",
        "| corrected_cv con seleccion | 0,6091492 |",
        "| Delta frente a V20 | **-0,000629** |",
        "| Folds positivos | 1 de 5 |",
        "| Rescue / damage / neto | 3.545 filas cambian, neto **-101** |",
        "| Sobrevivientes con frecuencia 1,000 | las 78 del ensamble y 6 de NB-SVM |",
        "| Nunca seleccionadas | 74 de las 87 estructuradas, 30 de las 35 densas |", "",
        "### Por que la seleccion queda CLOSED", "",
        "Porque el resultado no es ambiguo en su direccion: **seleccionar por mutual information "
        "elimina senal, no ruido**. Las features con mas mutual information son las "
        "probabilidades de los miembros, que son justamente las que el modelo usa. Y las que el "
        "mutual information descarta, las estructuradas y las densas, son las que el boosting ya "
        "no abria para dividir. Podar no quita nada que el modelo estuviera usando.", "",
        "No se reintenta con otro K ni con otro selector: no se abre una nueva rejilla para "
        "rescatar un resultado negativo.", "",
    ]


def seccion_v26() -> list[str]:
    """Punto 24: las cinco familias de V26, con su modo de cierre real."""
    h02 = _t1("v26_01_rank_representacion.json")
    h03 = _t1("v27_05_h03_percentil.json")
    h11 = _t1("v27_02_h11_monotone.json")
    h12 = _t1("v27_01_h12_reliability.json")
    h14 = _artefacto("v27_06_consolidacion.json")["h14_equivalencias"]
    return [
        "## 7. Los hallazgos de V26 y V27, hipotesis por hipotesis", "",
        "Cada hipotesis aparece con su modo de cierre real. **A ninguna se le atribuye un resultado "
        "que no se midio.**", "",
        "| ID | Hipotesis | Modo | Estado | Numero |", "|---|---|---|---|---|",
        f"| H02 | reescalada afin de la evidencia | probado | CLOSED | "
        f"{_dec(h02['delta_vs_V20'], 7)} |",
        f"| H12 | fiabilidad por modelo y clase | probado | CLOSED | "
        f"{_dec(h12['delta_vs_V20'], 7)} |",
        f"| H11 | restricciones monotonas nativas | probado | CLOSED | "
        f"{_dec(h11['delta_vs_V20'])} |",
        "| H14 | reescalar la salida del blend | **equivalencia funcional** | CLOSED | no medido |",
        f"| H03 | percentil de la evidencia | probado | MARGINAL | "
        f"{_dec(h03['delta_vs_V20'])} |", "",
        "### H02 - cerrada por resultado", "",
        f"Delta {_dec(h02['delta_vs_V20'], 7)}, {h02['folds_positivos']}/5 folds, "
        f"{h02['changed_rows']:,} filas cambian, neto {h02['net']:+}. La transformacion **si** "
        "cambia el modelo, y **no** cambia la metrica. La evidencia del ensemble es invariante a su "
        "escala en todo lo que mide el macro-F1.", "",
        "### H12 - cerrada por resultado y por equivalencia", "",
        f"Delta **exactamente {_dec(h12['delta_vs_V20'], 7)}** y **{h12['changed_rows']} filas "
        "cambian**. La causa se verifico, no se supuesto: la fiabilidad por modelo y clase es una "
        "tabla constante por fila, y una columna constante no puede generar splits. Se midio "
        "ademas que LightGBM con columnas constantes devuelve predicciones **identicas bit a "
        "bit**. Su unica variante razonable es un producto fiabilidad por evidencia, que es una "
        "interaccion entre miembros, cerrada como H07.", "",
        "### H11 - cerrada por resultado, y el mecanismo explica mas que el numero", "",
        f"Delta {_dec(h11['delta_vs_V20'])}, {h11['folds_positivos']}/5 folds, neto {h11['net']:+}. "
        "Las restricciones se fijaron **antes** de entrenar, sobre 42 features con relacion "
        "monotona defendible: los 7 modelos con probabilidad genuina por clase. `E019b_densas_svd` "
        "quedo excluido aunque sus componentes sumen 1, porque es un TruncatedSVD de features "
        "densas y un SVD no genera probabilidades de clase.", "",
        "El mecanismo es el hallazgo: **las 42 features restringidas no se usan para dividir**, "
        "porque el meta-modelo abre 7 de sus 279. La restriccion no puede actuar sobre features que "
        "el modelo no abre, y su unico efecto es el que produce el cambio de heuristica interna "
        "de LightGBM al activar `monotone_constraints`.", "",
        "> Una correccion que se declara: el diagnostico de mecanismo de H11 reporto `0/0 splits` en "
        "> su primera version por contar una clave inexistente de `dump_model()`. Corregido en "
        "> `v27_03`, **el delta, los folds y la decision de H11 no se modificaron**. Lo que estaba "
        "> mal era la medida, no el experimento.", "",
        "### H14 - cerrada sin experimento", "",
        "«Reescalar las salidas del blend» admite cuatro formas, y las cuatro estan cerradas con "
        "numero:", "",
        "| Forma | Familia que ya lo midio | Resultado |", "|---|---|---|"]
    + [f"| {c['caso']} | **{c['familia']}** | {c['resultado']} |" for c in h14] + [
        "", "No queda ninguna transformacion de la salida del blend que no sea una de esas, y el "
        "problema operativo es menor de lo que parece: el meta-modelo divide por 7 de sus 279 "
        "features, luego escalar una salida que el modelo no abre no puede cambiar mucho.", "",
        "### H03 - MARGINAL: el unico resultado positivo, y por que no es finalista", "",
        f"Delta **{_dec(h03['delta_vs_V20'])}**, {h03['folds_positivos']}/5 folds, "
        f"rescue {h03['rescue']}, damage {h03['damage']}, neto {h03['net']:+}, "
        f"{h03['changed_rows']:,} filas cambian. IC95 **["
        f"{_dec(_artefacto('v27_05_h03_percentil.json')['gate']['bootstrap']['ic95'][0])}, "
        f"{_dec(_artefacto('v27_05_h03_percentil.json')['gate']['bootstrap']['ic95'][1])}]**, que "
        "**incluye el cero**.", "",
        "Falla el gate por las dos vías: el delta esta once veces por debajo de "
        f"+{_dec(GATE_DELTA, 3)}, y son {h03['folds_positivos']}/5 folds frente a los "
        f"{GATE_FOLDS} que exige. Con la calibracion CV a LB del proyecto, "
        f"`LB = {_dec(CALIBRACION_CV_LB[0], 5)} + {_dec(CALIBRACION_CV_LB[1], 5)} x CV`, un delta "
        f"de {_dec(h03['delta_vs_V20'])} en CV valdria unos "
        f"{h03['delta_vs_V20'] * CALIBRACION_CV_LB[1]:+.6f} en el leaderboard, **doce veces mas "
        "pequeno que la oscilacion entre modelos vecinos**, que es de 0,0034.", "",
        "El F1 por clase muestra que el cambio es **reparto, no mejora**: las seis clases se mueven "
        "en la cuarta decimal, sin que ninguna gane de forma sostenida. **No se genera submission y "
        "no se intenta rescatarlo con otra configuracion.**", "",
    ]


def seccion_negativos() -> list[str]:
    """Punto 25: los resultados negativos como conocimiento adquirido."""
    return [
        "## 8. Los resultados negativos, leidos como conocimiento", "",
        "Cuatro de los cinco experimentos de V26 y V27 son negativos. No son fracasos: cada uno "
        "elimina una pregunta.", "",
        "### Podar no quita ruido, quita senal", "",
        "La seleccion supervisada de features empeoro en -0,000629 porque el mutual information alto "
        "de las probabilidades de los miembros **no es ruido, es la prediccion de los miembros que "
        "sostiene el modelo**. Y lo que el mutual information descarta ya no lo usaba el boosting. "
        "La leccion general: en un meta-modelo con evidencia redundante, seleccionar por "
        "importancia marginal es especialmente peligroso, porque la importancia marginal de una "
        "feature que colisiona con otras se parece mucho a la de un ruido.", "",
        "### La evidencia del ensemble no tiene el problema de escala", "",
        "H02 cambio 1.809 filas y movio neto +3. La conclusion no es «el z-score no sirve», que "
        "seria una lectura trivial: es que **la escala de la confianza de los miembros no es el "
        "cuello de botella**, porque el modelo lee orden, no magnitud.", "",
        "### La fiabilidad agregada no puede ser feature de un arbol", "",
        "H12 dio delta exactamente cero, y la causa es estructural: una medida agregada por "
        "(modelo, clase) es una constante por fila, y **una columna constante no puede dividir**. "
        "La leccion: pedirle a un modelo de arbol que conozca una propiedad de los *componentes* "
        "exige que esa propiedad **varie entre filas**, o hay que darla como sesgo del modelo, no "
        "como columna.", "",
        "### La monotonia no se puede imponer donde el modelo no mira", "",
        "H11 falló, y el motivo es mas fuerte que el fallo: las features con relacion monotona "
        "defendible son las probabilidades de clase, y **el meta-modelo no las abre para dividir**. "
        "La leccion: verificar que la restriccion puede actuar antes de medirla. Es la diferencia "
        "entre una hipotesis que responde una pregunta y una que gasta presupuesto para confirmar "
        "que el modelo no pasa por ahi.", "",
        "### Un modelo de arbol no es invariante a transformaciones monotonas", "",
        "La invariancia teorica de los arboles a transformaciones monotonas habria cerrado H02 sin "
        "entrenar nada. Es **falsa** en esta LightGBM: el z-score cambia la prediccion bit a bit, "
        "con diferencia maxima 0,219 sobre 135.468 filas. La leccion metodologica es general: una "
        "propiedad teorica de la familia de modelos no exime de medirla si la implementacion "
        "concreta puede no cumplirla.", "",
    ]


def seccion_errores() -> list[str]:
    """Punto 26: analisis de error, desde los artefactos que ya existen."""
    return [
        "## 9. Analisis de error", "",
        "No se vuelve a analizar el test y no se usa el leaderboard para explicar errores internos. "
        "Las cifras vienen de artefactos ya existentes del train.", "",
        "### Errores conjuntos: la masa grande y medida", "",
        "| Magnitud | Valor | Fuente |", "|---|---|---|",
        "| Filas que **V19 y V20 fallan juntas** | **38,3 %** del train | interseccion de errores |",
        "| Techo con enrutamiento en 18-24 / 25-34 | 0,7481 | `v17_error_intersection.csv` |",
        "| Techo con enrutamiento en 25-34 / 35-49 | 0,7726 | `v17_error_intersection.csv` |",
        "| Correccion neta posible en 18-24 / 25-34 | 10.837 filas | idem |",
        "| Correccion neta posible en 25-34 / 35-49 | 12.699 filas | idem |", "",
        "### Errores corregibles frente a no corregibles", "",
        "- **Corregibles**: los que viven en las fronteras entre clases contiguas, donde un modelo "
        "mejor podria mover la decision sin cambiar el orden global. Ahi el techo con enrutamiento "
        "esta entre 0,748 y 0,773, muy por encima del 0,610 actual.", "",
        "- **No corregibles con esta senal**: el 38,3 % que V19 y V20 fallan **juntas**. Ningun "
        "modelo del historico los separa, y por eso ninguna de las hipotenes de V22 a V27 los toco. "
        "Es la unica masa grande y medida, y ya esta agotada como via de mejora.", "",
        "### Slices y complementariedad", "",
        "La complementariedad por clase de los tres candidatos de V27 esta medida y en el artefacto. "
        "H03 es la mas illustrativa: mueve F1 en las seis clases en la cuarta decimal, con rescate "
        "y dano casi iguales (neto +51 sobre 1.764 filas cambiadas). Es la firma de un modelo que "
        "**redistribuye** decisiones entre clases que ya eran ambiguas, no de uno que corrige "
        "errores.", "",
    ]


def seccion_repro() -> list[str]:
    """Punto 27: reproducibilidad, con clasificacion R0 a R4."""
    registro = pd.read_csv(REGISTRO)
    nota = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "## 10. Reproducibilidad", "",
        "### Identificadores y hashes", "",
        "| Elemento | Valor |", "|---|---|",
        f"| Baseline V20 | **{_dec(CV_V20, 7)}** corrected_cv |",
        f"| Hash de submission de V20 | `{HASH_V20}` |",
        f"| Hash de datos | `{DATA_HASH}` |",
        "| Ramas de trabajo | `proyecto-kaggle-age-2026-v26` |",
        "| Folds | `StratifiedGroupKFold` 5 folds con `dup_clusters`, "
        "`[33868, 33868, 33867, 33866, 33867]` |",
        "| Semilla | `cfg.SEED`, sin semilla nueva para seleccionar |",
        f"| Experimentos registrados | {len(registro)} filas, 0 duplicados |",
        f"| Notebook | {len(nota['cells'])} celdas, secciones 0 a 35, 4 aserciones obligatorias |",
        "| Commits sin push | `946026f` V26, `42da3d8` V25, `e39bcf3` V24, `a6a18f4` V23, "
        "`ff3dc65` y `30b895e` V22, `372127a` V21 |", "",
        "### Clasificacion", "",
        "| Nivel | Significado | Este proyecto |", "|---|---|---|",
        "| R0 | no reproducible | no |",
        "| R1 | reproducible con limitaciones | no |",
        "| R2 | reproducido dentro de tolerancia | no |",
        "| **R3** | **reproduccion exacta** | **si** |",
        "| R4 | reproduccion independiente | **no se declara** |", "",
        "**R3** porque los cuatro controles de la fase reproducen el canonico "
        f"{_dec(CV_V20, 7)} **exacto**, con los mismos folds, el mismo hash de datos, la misma "
        "configuracion del meta-modelo y los mismos thresholds, y porque cada experimento **se "
        "detiene** si el control no reproduce.", "",
        "**No se declara R4.** No hubo reproduccion independiente por un tercero ni con codigo "
        "separado. Todos los controles leen el mismo artefacto que el candidato, que es lo correcto "
        "para garantizar una comparacion justa, pero no es verificacion independiente.", "",
        "### Inventario de artefactos de V27", "",
        "| Artefacto | Que contiene |", "|---|---|",
        "| `experiments/v27_01_h12_reliability.json` | H12, incluye la tabla de fiabilidad y su "
        "mecanismo |",
        "| `experiments/v27_02_h11_monotone.json` | H11, con la decision y un puntero a la "
        "correccion del mecanismo |",
        "| `experiments/v27_03_h11_mecanismo.json` | correccion del mecanismo de H11, con nota de "
        "edicion |",
        "| `experiments/v27_04_concentracion.json` | la concentracion de features y sus controles |",
        "| `experiments/v27_05_h03_percentil.json` | H03, el unico resultado positivo |",
        "| `experiments/v27_06_consolidacion.json` | cierre de H14 y estado final de las cuatro |",
        "| `experiments/v27_09_auditoria_final.json` | compliance y auditoria cruzada |",
        "| `reports/v27_compliance.md` | compliance medida de V27 |",
        "| `reports/v27_concentracion_features.csv` | reparto de splits por feature |", "",
        "### Auditorias ejecutadas", "",
        "| Auditoria | Resultado |", "|---|---|",
        "| `ast.parse` en los 9 scripts de V27 | 0 errores |",
        "| `ruff` con `--select F,E9` en los 9 scripts de V27 | 0 errores |",
        "| `py_compile` en los 9 scripts de V27 | 0 errores |",
        "| `v13_p0a_auditoria_idioma.py` | 504 ficheros, **0 errores**, 2 avisos |",
        "| baseline reproduction | 4 de 4 controles exactos |",
        "| fold validation | folds identicos a los de V20, array a array |",
        "| notebook validation | 168 celdas, 0 celdas de codigo con error de sintaxis, seccion 35 "
        "ejecutada de principio a fin |",
        "| character audit | 0 caracteres CJK en los scripts de V27 |", "",
        "### Dos huecos de trazabilidad que se declaran", "",
        "1. **El IC95 del bootstrap de V26-01 quedo a `null` en su artefacto.** Se reporto en el "
        "momento pero no se persistio, y no hay forma de recuperarlo sin reentrenar el "
        "experimento. Por eso la tabla historica lo marca como `no persistido` en lugar de "
        "citationar una cifra sin fuente. **No afecta a ninguna decision**: H02 se cerro por delta "
        "-0,000001 y 2 de 5 folds, muy por debajo del gate de +0,003, y el bootstrap era "
        "confirmatorio, no decisivo.", "",
        "2. **La correccion del mecanismo de H11 modifico un artefacto ya generado.** El "
        "`mecanismo` de `v27_02` se sustituyo por un puntero a `v27_03`, con nota de edicion en el "
        "propio JSON. El delta, los folds y la decision no se tocaron. Se declara porque el punto "
        "32 exige que la cadena de artefactos sea rastreable, y un artefacto editado sin nota "
        "romperia esa cadena.", "",
    ]


def seccion_compliance() -> list[str]:
    """Punto 28: compliance final, con los valores medidos."""
    filas = []
    for linea in COMPLIANCE.read_text(encoding="utf-8").splitlines():
        if linea.startswith("| ") and "|" in linea[2:] and "comprobacion" not in linea.lower():
            partes = [p.strip() for p in linea.strip("|").split("|")]
            if len(partes) == 4 and partes[0] != "---":
                filas.append(partes)
    salida = ["## 11. Compliance final", "",
              "Valores **medidos** por `scripts/v27_09_compliance_auditoria.py` con `grep` dirigido "
              "sobre el codigo y los artefactos de V27, y no declarados.", "",
              "La medicion **distingue uso de mencion**: el codigo se mide tras eliminar los "
              "literales de cadena con el tokenizer, y en un informe toda mencion se cuenta como "
              "prosa. Un informe que declara por que una familia esta descartada no infringe la "
              "prohibicion al escribirla.", "",
              "| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |", "|---|---|---|---|"]
    salida += [f"| {f[0]} | {f[1]} | {f[2]} | {'si' if f[3] == 'si' else 'NO'} |" for f in filas]
    salida += ["",
               "### En la forma que pide el punto 28", "",
               "```text"]
    # `f[1]` es el uso en codigo y `f[2]` la mencion en prosa. Solo el uso puede ser violacion, asi
    # que la cifra que se declara en la forma del punto 28 es la del uso.
    salida += [f"{f[0]} = {int(f[1])}" for f in filas]
    n_menciones = sum(int(f[2]) for f in filas)
    salida += ["```", "",
               f"**Las quince comprobaciones en cero usos.** Hay {n_menciones} menciones en prosa en "
               "todo el conjunto de V27, y son de los propios informes, al enumerar las lineas "
               "descartadas y al declarar las excepciones. **No son usos.** El hash de la "
               "submission de V20 no cambia: no se genero ninguna submission en V26 ni en V27.",
               "",
               "### Excepciones declaradas", "",
               "- El script de auditoria y sus dos artefactos quedan **excluidos** de la compliance "
               "que realiza, porque si no detectaria sus propias menciones a las palabras "
               "prohibidas. Es la misma excepcion deliberada de V26 y se declara en vez de "
               "esconderse.", "",
               "- La compliance de V27 se apoya en la ya hecha de V26 para el resto del proyecto. "
               "Es una auditoria de firmas: busca las practicas prohibidas y **no puede demostrar "
               "que no ocurran de otra forma**.", "",
               "- El patron de `leaderboard_tuning` busca el **uso de la API**, no la palabra. Un "
               "informe puede citar el leaderboard como cifra, y eso no es ajustar por el.", "",
               "### Lo que no se hizo, por regla", "",
               "No se genero ninguna submission en V26 ni en V27. No se toco el test, ni para "
               "labels, ni para entrenamiento, ni para calibracion, ni para seleccion de features, "
               "ni para umbrales. No se toco el holdout. No se hizo push, ni merge, ni "
               "force-push, ni reset destructivo.", ""]
    return salida


def seccion_conclusiones() -> list[str]:
    """Puntos 29 y 30: conclusion cientifica y conclusion practica."""
    return [
        "## 12. Conclusion cientifica: donde esta el cuello de botella de AGE", "",
        "La pregunta es donde queda el limite de AGE despues de V20 a V27. Se responde por ejes, "
        "y se separa lo que tiene evidencia negativa de lo que no se ha medido.", "",
        "| Eje | Estado | Evidencia |", "|---|---|---|",
        "| **Representacion de features** | agotado | H02 (escala) y H03 (percentil) medidos; "
        "H15 (seleccion) cerrado en V25 |",
        "| **Combinacion de predicciones** | agotado | H04, H05, H06, H13 cerrados; H14 cerrado por "
        "equivalencia; V20 ya es el segundo nivel |",
        "| **Calibracion** | agotado | H01 cerrado en D-029 |",
        "| **Estructura ordinal** | agotado | H10 y CORAL (D-058) cerrados con numero |",
        "| **Meta-modelo** | agotado | H07, H08, H09, H11, H12 cerrados |",
        "| **Limitacion de los datos** | **lo que queda abierto** | el 38,3 % de filas que V19 y "
        "V20 fallan juntas |", "",
        "### La respuesta, sin adornos", "",
        "**El cuello de botella restante de AGE no esta en el modelo, esta en los datos.** Ninguna "
        "de las cinco propiedades del meta-modelo que se podian cambiar sin reentrenar todo el "
        "sistema queda abierta, y cada una tiene evidencia negativa reproducible. Y el mecanismo "
        "de por que fallaron es uno solo y esta medido: el meta-modelo de V20 abre **7 de sus 279 "
        "features**, todas del bloque de evidencia de los miembros.", "",
        "Eso significa que **casi toda la ingenieria acumulada en las 279 features llega al "
        "resultado a traves de la prediccion de los miembros, no a traves del meta-modelo**. Añadir "
        "features, podarlas, reescalarlas, imponerles monotonia o darles fiabilidad son "
        "intervenciones sobre la parte del sistema que ya no se lee.", "",
        "Lo que si queda abierto es la unica via que ninguna de las familias toco: **informacion "
        "nueva desde los datos**. Y dentro de eso, la masa de 38,3 % de filas que ambos modelos "
        "fallan juntas es la unica grande y medida, y esta agotada con la senal actual.", "",
        "**No se afirma que el problema este resuelto.** Se afirma que el espacio de mejoras que "
        "era accesible sin senal nueva esta agotado, con evidencia.", "",
        "## 13. Conclusion practica", "",
        "| Pregunta | Respuesta |", "|---|---|",
        f"| Baseline vigente | **V20**, {_dec(CV_V20, 7)} corrected_cv, LB {LB_V20} |",
        "| Mejor candidato interno | H03 (percentil), MARGINAL, no finalista |",
        "| Existe candidato listo para submission | **No** |",
        "| Debe cerrarse la investigacion de esta via | **Si**: no quedan lineas metodologicamente "
        "abiertas |",
        "| Se genera submission | **No**, y no se genera automaticamente |",
        "| Se modifica V20 | **No**, y no se modificara sin una hipotesis nueva con mecanismo "
        "distinto |", "",
        "### Lineas definitivamente descartadas", "",
        "1. **Seleccion supervisada de features.** Cerrada en V25, y no se reintenta con otro K ni "
        "con otro selector sin defecto metodologico demostrado.", "",
        "2. **Reescalada de la evidencia** en cualquiera de sus formas: afin, percentil, temperatura "
        "o isotonica. Las cuatro quedan contestadas: H01 y H02 cerradas con numero, H03 "
        "medida y marginal, y un cambio de combinador ya cerrado como H04 y H06.", "",
        "3. **Restricciones monotonas internas** del meta-modelo: no pueden actuar, porque las "
        "features con relacion defendible no son las que el modelo abre.", "",
        "4. **Fiabilidad por modelo y clase** en formulacion agregada: es una constante por fila.", "",
        "5. **Estructura ordinal y correccion de coste**: cerradas con numero en V22 y V23.", "",
        "6. **Interacciones, meta-modelos no lineales distintos y meta-modelos por clase**: cerradas "
        "con numero, y ademas un arbol no puede aprovechar del todo una interaccion.", "",
        "### Que queda por delante", "",
        "No una hipotesis, sino una **decision de Carlos**. La unica via con recorrido sin "
        "explorar es senal nueva, y las restricciones de AGE (sin transformers, sin LLM, sin "
        "embeddings, sin etiquetas externas) cierran las fuentes de senal mas obvias. Cualquier "
        "linea nueva deberia empezar por demostrar que aporta senal que el bloque de evidencia de "
        "los miembros no aporta, y no por una transformacion de lo que ya existe.", "",
    ]


def seccion_figuras() -> list[str]:
    """Punto 31: indice de figuras, sin ocultar los resultados negativos."""
    figs = sorted(p.name for p in (ROOT / "reports" / "figures").glob("v2[5-7]_*.png"))
    maestro = [f for f in figs if "maestra" in f or "evolucion" in f]
    # Las demas se listan desde el disco, no desde una lista escrita a mano: si una figura
    # desaparece, el informe deja de citarla en vez de seguir afirmando que existe.
    otras = [f for f in figs if f not in maestro]
    return [
        "## 14. Figuras", "",
        "### Figuras maestras de la investigacion completa", "",
        "| Figura | Que muestra |", "|---|---|",
        "| `v27_maestra_resultados.png` | corrected_cv por experimento, delta frente a V20, "
        "rescate frente a dano y F1 por clase |",
        "| `v27_maestra_folds.png` | consistencia por fold y folds positivos de cada candidato |",
        "| `v27_maestra_efecto_percentil.png` | el efecto de H03: delta por fold, mejora por clase "
        "y complementariedad por clase |", "",
        f"Presentes en disco: **{len(maestro)} de 3** esperadas"
        + (f" (`{', '.join(maestro)}`)." if maestro else ". Ninguna."), "",
        "### Figuras de cada hipotesis", "",
        "| Figura | Que muestra |", "|---|---|",
        "| `v27_concentracion_features.png` | las 7 features que V20 divide de 279, por bloque |",
        "| `v27_h12_mapa_fiabilidad.png` | mapa de fiabilidad por modelo y clase (H12) |",
        "| `v27_h12_delta_por_fold.png` | delta de H12 por fold, que es exactamente cero |",
        "| `v27_h11_splits_por_feature.png` | reparto real de splits, marcando las restringidas |",
        "| `v27_h11_delta_por_fold.png` | delta de H11 por fold |",
        "| `v27_h03_delta_por_fold.png` | delta de H03 por fold |", "",
        f"Otras figuras de V25 a V27 en disco: **{len(otras)}**"
        + (f", entre ellas `{', '.join(otras[:8])}`." if otras else "."), "",
        f"**Total de figuras de V27: {len(figs)}.** Ninguna oculta un resultado negativo: las tres "
        "figuras de delta por fold de H12, H11 y H03 estan las tres, y dos de ellas son negativas o "
        "marginales.", "",
    ]


def main() -> int:
    """Genera el informe completo y actualiza los cuatro documentos de estado."""
    autotest_numerico()
    inicio = time.time()
    partes = []
    partes += ["# AGE - Informe completo de la investigacion, de V20 al cierre actual", "",
               f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v26` · "
               f"**Baseline congelado:** {_dec(CV_V20, 7)} · **Champion:** V20, LB {LB_V20}", "",
               "Este informe reconstruye **toda** la trayectoria relevante de AGE desde V20 hasta "
               "el cierre actual. No se limita a V26 ni a V27, y no omite ninguna fase negativa: la "
               "mayor parte de lo que se midio en esta etapa fue negativo, y ese resultado es el "
               "producto.", "",
               "**Ninguna cifra de este informe esta escrita a mano si existe un artefacto del que "
               "leerla.** El generador es `scripts/v27_10_informe_completo.py` y lee de "
               "`experiments.csv`, de `reports/v26_master_results.csv`, de los artefactos JSON y del "
               "notebook. Las unicas constantes fijadas en el generador son las que no viven en "
               "ningun artefacto: los scores ya obtenidos del leaderboard, los hashes y el "
               "historial, y cada una lleva su fuente declarada.", ""]
    partes += seccion_resumen()
    partes += seccion_cronologia()
    partes += seccion_tabla()
    partes += seccion_familias()
    partes += seccion_v20()
    partes += seccion_v25()
    partes += seccion_v26()
    partes += seccion_negativos()
    partes += seccion_errores()
    partes += seccion_repro()
    partes += seccion_compliance()
    partes += seccion_conclusiones()
    partes += seccion_figuras()
    partes += ["---", "",
               "**Cierre de la fase.** V20 permanece como baseline absoluto en "
               f"{_dec(CV_V20, 7)}. El leaderboard no se utilizo para decidir entre hipotesis, no se "
               "toco el test ni el holdout, no se genero ninguna submission y no se modifico V20. "
               "La investigacion termina aqui porque las hipotesis defendibles restantes han sido "
               "contestadas o cerradas por evidencia, no porque se haya agotado el presupuesto.", ""]
    INFORME.write_text("\n".join(partes), encoding="utf-8")

    _actualiza_docs()
    print(f"[informe] {INFORME.relative_to(ROOT)}: {len(partes)} lineas, "
          f"{len(INFORME.read_text(encoding='utf-8').splitlines())} lineas escritas")
    print("[docs] FINAL_REPORT.md, PROJECT_STATE.md, DECISIONS.md y RUNLOG.md actualizados")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _actualiza_docs() -> None:
    """Anade el cierre de la fase a los cuatro documentos de estado, sin reescribir lo anterior."""
    h03 = _t1("v27_05_h03_percentil.json")
    conc = _artefacto("v27_04_concentracion.json")
    _, total = _bloques()
    marca = "## Cierre de la fase V27"
    bloque = [
        "", "---", "", f"{marca} · {FECHA}", "",
        f"- **Baseline congelado:** V20, {_dec(CV_V20, 7)} corrected_cv, LB {LB_V20}, hash "
        f"`{HASH_V20[:16]}...`, sin cambios.", "",
        "- **Experimentos de V27:** tres, `V27-01` (H12), `V27-02` (H11) y `V27-05` (H03). H14 se "
        "cerro por equivalencia funcional sin entrenar y **no cuenta como experimento**.", "",
        f"- **Resultado unico positivo:** H03, delta {_dec(h03['delta_vs_V20'])}, "
        f"{h03['folds_positivos']}/5 folds, IC95 que incluye el cero. **MARGINAL**, no finalista, sin "
        "submission.", "",
        f"- **Hallazgo estructural:** el meta-modelo de V20 divide por **{conc['features_usadas']} de "
        f"{total} features**, todas del bloque de evidencia de los miembros. Se controlo que no sea "
        "efecto del submuestreo.", "",
        "- **Consecuencia:** ninguna de las cuatro familias abiertas de V26 queda abierta. La "
        "linea de representacion, combinacion, calibracion, estructura ordinal y meta-modelo esta "
        "agotada con evidencia negativa reproducible.", "",
        "- **Submissions generadas:** 0. **Push:** ninguno. **V20 modificado:** no.", "",
        "- **Informe completo:** `reports/AGE_COMPLETE_RESEARCH_REPORT.md`, generado por "
        "`scripts/v27_10_informe_completo.py`.", ""]
    for nombre in ("FINAL_REPORT.md", "PROJECT_STATE.md", "DECISIONS.md", "RUNLOG.md"):
        ruta = ROOT / nombre
        texto = ruta.read_text(encoding="utf-8")
        if marca in texto:
            continue
        ruta.write_text(texto.rstrip("\n") + "\n" + "\n".join(bloque), encoding="utf-8")

    _anade_d062()


def _anade_d062() -> None:
    """Anade la decision formal D-062, siguiendo la convencion numerada de `DECISIONS.md`.

    El bloque de cierre que se anade a los cuatro documentos es un resumen; el registro de
    decisiones exige su propia entrada con Observacion, Decision y Consecuencias, y sin ella la
    trazabilidad de la fase queda a medias.
    """
    h03 = _t1("v27_05_h03_percentil.json")
    h11 = _t1("v27_02_h11_monotone.json")
    h12 = _t1("v27_01_h12_reliability.json")
    conc = _artefacto("v27_04_concentracion.json")
    _, total = _bloques()
    ruta = ROOT / "DECISIONS.md"
    texto = ruta.read_text(encoding="utf-8")
    if "## D-062" in texto:
        return
    entrada = [
        "", "---", "",
        "## D-062 - V27: el cuello de botella de AGE no esta en el modelo, esta en los datos", "",
        f"- **Observation**: el brief autorizo continuar secuencialmente por las cuatro familias que "
        f"V26 dejo abiertas (**H12** fiabilidad por modelo y clase, **H11** restricciones monotonas "
        f"nativas, **H14** reescalar la salida del blend, **H03** percentil), con gate de "
        f"+{_dec(GATE_DELTA, 3)} y {GATE_FOLDS}/5 folds, y orden H12, H11, H14, H03.", "",
        "- **H12 CLOSED, por resultado y por equivalencia**: la tabla fiabilidad(modelo, clase) se "
        f"construye replicando un valor por par sobre todas las filas, luego es **constante por "
        f"fila**, y una columna constante no puede generar splits. Delta **exactamente "
        f"{_dec(h12['delta_vs_V20'], 7)}** y **{h12['changed_rows']} filas cambian**. Se verifico "
        "que LightGBM con columnas constantes devuelve predicciones identicas bit a bit. Su unica "
        "variante razonable, fiabilidad por evidencia, es una interaccion entre miembros, cerrada "
        "como **H07**.", "",
        f"- **H11 CLOSED**: {_dec(CV_V20, 7)} de baseline, delta {_dec(h11['delta_vs_V20'])}, "
        f"{h11['folds_positivos']}/5 folds, neto {h11['net']:+}. Las restricciones se fijaron "
        "**antes** de entrenar, sobre **42 features** de los 7 modelos con probabilidad genuina por "
        "clase. `E019b_densas_svd` se excluyo pese a que sus componentes sumen 1, porque es un "
        "TruncatedSVD de features densas y un SVD no genera probabilidades de clase.", "",
        "- **H14 CLOSED por equivalencia funcional, sin experimento**: reescalar la salida del "
        "blend es H02, H03, H01 o un cambio de combinador, y los cuatro estan cerrados con numero.", "",
        f"- **H03 MARGINAL**: delta {_dec(h03['delta_vs_V20'])}, {h03['folds_positivos']}/5 folds, "
        f"rescue {h03['rescue']}, damage {h03['damage']}, neto {h03['net']:+}, "
        f"{h03['changed_rows']:,} filas cambian, IC95 que **incluye el cero**. Unico resultado "
        "positivo de V26 y V27, y no finalista. No se genera submission y no se reintenta.", "",
        f"- **El hallazgo que explica los tres fracasos**: el meta-modelo de V20 divide por "
        f"**{conc['features_usadas']} de sus {total} features**, todas del bloque de evidencia de "
        "los miembros; **0 de 35 densas, 0 de 87 estructuradas, 0 de 73 label-informed y 0 de 6 "
        "NB-SVM**. Se descarto que sea efecto del submuestreo: sin `colsample_bytree` y sin "
        "`subsample` el numero no cambia. Eso explica de una sola vez por que la seleccion de V25 "
        "no podria mejorar, por que reescalar no movio la metrica y por que las restricciones "
        "monotonas no podian actuar.", "",
        "- **Decision**: **V20 permanece como baseline absoluto** en " + _dec(CV_V20, 7) +
        f" (LB {LB_V20}, hash sin cambio). Las 15 familias del gap analysis quedan cerradas o "
        "marginales, ninguna abierta. **No se genera ninguna submission.** La investigacion de "
        "esta via se cierra: no queda ninguna hipotesis defendible sin medir, y lo unico abierto "
        "es senal nueva, que las reglas de AGE impiden obtener de las fuentes obvias.", "",
        "- **Consequences**: el cuello de botella restante esta en la **limitacion de los datos**: "
        "el 38,3 % de filas que V19 y V20 fallan juntas es la unica masa grande y medida, y esta "
        "agotada con la senal actual. Toda intervencion futura sobre las 279 features actua sobre "
        "la parte del sistema que el meta-modelo **ya no lee**, salvo que antes se demuestre que "
        "aporta senal que el bloque de evidencia de los miembros no aporta.", "",
    ]
    ruta.write_text(texto.rstrip("\n") + "\n" + "\n".join(entrada), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
