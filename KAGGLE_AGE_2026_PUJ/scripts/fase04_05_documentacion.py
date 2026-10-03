"""Fases 4-12 (Etapa B-C) — Construye y ejecuta los notebooks 04 y 05.

``04_feature_engineering.ipynb`` documenta la matriz de representaciones, la descomposición de
la normalización (con el hallazgo de menciones), la comparación estadística pareada, la decisión
de representación y el catálogo de features.

``05_baselines.ipynb`` documenta la tabla B0-B5 más la representación ganadora, con el análisis
por clase.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import nbutil  # noqa: E402

SETUP = """
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / "src" / "config.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg, feature_catalog
from src.config import set_seed

set_seed()
pd.set_option("display.width", 175)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 60)

representaciones = json.loads((cfg.EXPERIMENTS / "representaciones_resultados.json").read_text())
baselines = json.loads((cfg.EXPERIMENTS / "baselines_resultados.json").read_text())
E001 = json.loads((cfg.EXPERIMENTS / "E-001_resultados.json").read_text())
print("resultados cargados:", len(representaciones), "representaciones |", len(baselines), "baselines")
"""

NB04 = [
    nbutil.md(
        """
# 04 · Representaciones y catálogo de features (Etapa B)

**Pregunta.** ¿Qué representación maximiza macro-F1 bajo el protocolo agrupado por clústeres?

**Diseño experimental.** Modelo fijo (LinearSVC C=1) para aislar el efecto de la representación;
validación `StratifiedGroupKFold` de 5 folds agrupada por clúster de near-duplicados;
vectorizadores ajustados **solo** con el train de cada fold. Cada configuración tiene hipótesis
declarada antes de ejecutarse (ver `experiments.csv`).

Este notebook **no re-ejecuta** los experimentos: documenta y analiza sus resultados guardados en
`experiments/representaciones_resultados.json`.
"""
    ),
    nbutil.md("## 0. Configuración"),
    nbutil.code(SETUP),
    nbutil.md("## 1. Matriz de representaciones"),
    nbutil.code(
        """
matriz = pd.DataFrame([
    {"id": key, "representacion": value["nombre"], "texto": value["texto"],
     "macro_f1": round(value["macro_f1_mean"], 4), "std": round(value["macro_f1_std"], 4),
     "weighted": round(value["weighted_f1_mean"], 4),
     "accuracy": round(value["accuracy_mean"], 4), "features": value["n_features"]}
    for key, value in representaciones.items()
]).sort_values("macro_f1", ascending=False).reset_index(drop=True)
matriz.to_csv(cfg.TABLES / "representaciones_comparativa.csv", index=False)
matriz
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** Los n-gramas de carácter dominan: `char_wb 3-6` (0,5413) y `char_wb 3-5`
(0,5408) superan a todos los esquemas de palabra (`word 1-gram` 0,5025; `word 1-2` 0,4971;
`word 1-3` 0,4941). La unión palabra+carácter (E-001, 0,5344) queda **por debajo** del char solo.

**INTERPRETACIÓN.** Coincide con la literatura (Sapkota et al. 2015; PAN 2019, D-008): en texto
informal, corto y transliterado a ASCII, el n-grama de carácter captura morfología, abreviaturas
y ortografía, y no pierde información por tokenización. Añadir word n-grams diluye el modelo
lineal sin aportar señal nueva.

**CONSECUENCIA.** La representación base pasa a ser **char n-grams**, no la unión.
"""
    ),
    nbutil.md("## 2. Descomposición de la normalización: el hallazgo de las menciones"),
    nbutil.code(
        """
descomposicion = matriz[matriz["id"].isin(["E-001", "E-005a", "E-005b", "E-005c", "E-005d", "E-005e"])]
descomposicion = descomposicion.sort_values("macro_f1", ascending=False)
referencia = float(E001["macro_f1_mean"])
descomposicion = descomposicion.assign(delta_vs_crudo=(descomposicion["macro_f1"] - referencia).round(4))
descomposicion[["id", "representacion", "texto", "macro_f1", "delta_vs_crudo", "features"]]
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN (cuantitativa).** Partiendo del texto crudo (0,5344):
- quitar **solo las menciones** -> **0,4094** (−0,1251, 0/5 folds a favor);
- sustituir menciones por el marcador `<user>` -> **0,4107** (−0,1237);
- quitar solo URLs -> 0,5277 (−0,0067);
- quitar solo puntuación -> 0,5264 (−0,0080);
- normalización completa (URLs+menciones+RT+puntuación+minúsculas) -> 0,3902 (−0,1442).

Además, tras normalizar, **7.240 tweets quedan vacíos** (frente a 104 originales) y 13.137 se
quedan con ≤1 token: hay muchísimos tweets que son *solo URL*, *solo menciones* o *solo hashtags*
(p. ej. `#TFZ @M_MartinezRed @RaisAlGhousia …` -> `tfz`).

**INTERPRETACIÓN.** La señal no está solo en lo que se dice, sino en **a quién se menciona** y en
la **forma** del tweet. La identidad de las cuentas mencionadas (fandoms, medios, políticos)
revela el mundo social del autor, que es informativo de la edad. Reemplazar por `<user>` destruye
esa identidad, por eso también cae.

**CONSECUENCIA.** (1) el texto se mantiene **crudo**; (2) se crea un **canal dedicado** de
TF-IDF sobre los tokens de mención (experimento E-008).
"""
    ),
    nbutil.md("## 3. Canal de menciones: confirmación de la hipótesis"),
    nbutil.code(
        """
e008 = representaciones["E-008"]
e003c = representaciones["E-003c"]
print(f"E-003c (char 3-6)              : {e003c['macro_f1_mean']:.4f} ± {e003c['macro_f1_std']:.4f}")
print(f"E-008  (char 3-6 + menciones)  : {e008['macro_f1_mean']:.4f} ± {e008['macro_f1_std']:.4f}")
print(f"mejora                         : {e008['macro_f1_mean'] - e003c['macro_f1_mean']:+.4f}")
print(f"features                       : {e003c['n_features']:,} -> {e008['n_features']:,}")
print()
print("F1 por clase (E-003c -> E-008):")
comparativa = pd.DataFrame({"char_3_6": e003c["per_class_f1"], "char_3_6_+_menciones": e008["per_class_f1"]})
comparativa["delta"] = (comparativa["char_3_6_+_menciones"] - comparativa["char_3_6"]).round(4)
comparativa.round(4)
"""
    ),
    nbutil.md("## 4. Comparación estadística pareada entre folds"),
    nbutil.code(
        """
from scipy import stats

def fold_scores(experiment_id):
    return np.array([fold["macro_f1"] for fold in representaciones[experiment_id]["per_fold"]])

e001_folds = np.array([fold["macro_f1"] for fold in E001["per_fold"]])
comparaciones = [
    ("E-008", "E-003c", "canal de menciones vs char 3-6", fold_scores("E-008"), fold_scores("E-003c")),
    ("E-003c", "E-002b", "char 3-6 vs word 1-2", fold_scores("E-003c"), fold_scores("E-002b")),
    ("E-003c", "E-001", "char 3-6 vs union word+char", fold_scores("E-003c"), e001_folds),
    ("E-005d", "E-001", "quitar menciones vs crudo", fold_scores("E-005d"), e001_folds),
    ("E-003c", "E-003b", "char 3-6 vs char 3-5", fold_scores("E-003c"), fold_scores("E-003b")),
]
rows = []
for left, right, label, scores_left, scores_right in comparaciones:
    difference = scores_left - scores_right
    rows.append({"comparacion": label, "dif_media": round(float(difference.mean()), 4),
                 "folds_gana": f"{int((difference > 0).sum())}/5",
                 "p_wilcoxon": round(float(stats.wilcoxon(scores_left, scores_right).pvalue), 4)})
paired = pd.DataFrame(rows)
paired.to_csv(cfg.TABLES / "representaciones_pareado.csv", index=False)
print("Nota: con 5 folds el p mínimo alcanzable por Wilcoxon es 0,0625; la evidencia es la "
      "consistencia (5/5) más el tamaño de efecto.")
paired
"""
    ),
    nbutil.md(
        """
## 5. Decisión de representación (D-010)

**Representación adoptada:** `TF-IDF char_wb(3,6) sobre texto crudo` ∪ `TF-IDF word(1,1) sobre los
tokens de mención` + `LinearSVC(C=1)`.

Evidencia: mejor macro-F1 de la matriz (0,5557 ± 0,0052), mejora consistente en 5/5 folds sobre
la alternativa char sola (+0,0144) y sobre la baseline E-001 (+0,0213), con costo de features
moderado (314k) y sin ningún preprocesamiento que destruya señal.
"""
    ),
    nbutil.md("## 6. Catálogo de features"),
    nbutil.code(
        """
catalogo = feature_catalog.build_catalog()
catalogo.to_csv(cfg.REPORTS / "feature_catalog.csv", index=False)
print(f"catálogo: {len(catalogo)} filas | grupos: {catalogo['group'].nunique()}")
print(catalogo.groupby("group").size().to_string())
catalogo[["name", "group", "description", "hypothesis", "leakage_risk"]].head(12)
"""
    ),
    nbutil.md(
        """
### Conclusiones de la Etapa B

| # | Observación | Interpretación | Consecuencia |
|---|---|---|---|
| B1 | char n-grams > word n-grams (+0,044) | el carácter captura morfología y ortografía informal | representación base = char_wb 3-6 |
| B2 | La unión word+char diluye (−0,007 vs char solo) | sin señal léxica nueva que aportar | se descarta la unión clásica |
| B3 | Quitar menciones cuesta −0,125 | la identidad de las cuentas mencionadas es señal fuerte | canal dedicado de menciones (+0,014) |
| B4 | Quitar URLs/puntuación cuesta −0,007/−0,008 | la forma del tweet es señal | texto crudo, sin normalizar |
| B5 | 7.240 tweets vacíos tras normalizar | muchos tweets son solo enlaces/menciones | la normalización agresiva es inviable aquí |
| B6 | TF puro ≈ TF-IDF (−0,0005) | el IDF aporta poco con clases balanceadas | se conserva IDF sublinear (sin costo) |

**Siguiente:** Fase 13-19 — modelos clásicos sobre la representación ganadora, análisis de errores,
ajuste de macro-F1 (umbral/prior por clase) y ensambles.
"""
    ),
]

NB05 = [
    nbutil.md(
        """
# 05 · Baselines B0-B5 (Fase 12)

Tabla comparativa de los baselines obligatorios más la representación ganadora de la Etapa B, bajo
el mismo protocolo (StratifiedGroupKFold de 5 folds agrupado por clúster de near-duplicados).
"""
    ),
    nbutil.md("## 0. Configuración"),
    nbutil.code(SETUP),
    nbutil.md("## 1. Tabla de baselines"),
    nbutil.code(
        """
filas = [
    {"id": "B0", "descripcion": "DummyClassifier(prior)", "macro_f1": baselines["B0"]["macro_f1_mean"], "std": baselines["B0"]["macro_f1_std"]},
    {"id": "B1", "descripcion": "TF-IDF word 1-2 + LogisticRegression (OvR)", "macro_f1": baselines["B1"]["macro_f1_mean"], "std": baselines["B1"]["macro_f1_std"]},
    {"id": "B2", "descripcion": "TF-IDF word 1-2 + LinearSVC", "macro_f1": representaciones["E-002b"]["macro_f1_mean"], "std": representaciones["E-002b"]["macro_f1_std"]},
    {"id": "B3", "descripcion": "TF-IDF char_wb 3-5 + LinearSVC", "macro_f1": representaciones["E-003b"]["macro_f1_mean"], "std": representaciones["E-003b"]["macro_f1_std"]},
    {"id": "B4", "descripcion": "TF-IDF word 1-2 + char_wb 3-5 + LinearSVC", "macro_f1": E001["macro_f1_mean"], "std": E001["macro_f1_std"]},
    {"id": "B5", "descripcion": "35 features densas + LightGBM", "macro_f1": baselines["B5"]["macro_f1_mean"], "std": baselines["B5"]["macro_f1_std"]},
    {"id": "E-008", "descripcion": "char_wb 3-6 + canal de menciones + LinearSVC (ganadora)", "macro_f1": representaciones["E-008"]["macro_f1_mean"], "std": representaciones["E-008"]["macro_f1_std"]},
]
tabla = pd.DataFrame(filas)
tabla["macro_f1"] = tabla["macro_f1"].round(4)
tabla["std"] = tabla["std"].round(4)
tabla["vs_azar"] = (tabla["macro_f1"] - 1 / 6).round(4)
tabla.to_csv(cfg.TABLES / "baselines_comparativa.csv", index=False)
tabla
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** El orden de mérito es claro: la señal dispersa de carácter domina (0,54-0,56), las
representaciones de palabra quedan en 0,50, las features densas en 0,36 y el azar en 0,17.

**INTERPRETACIÓN.** Las 35 features densas, pese a tener tamaños de efecto altos por separado
(ε² de `n_chars` = 0,126), no alcanzan a la representación dispersa porque comprimen el tweet en
pocos números. El baseline lineal disperso captura además el léxico y la ortografía.

**CONSECUENCIA.** La Fase 13 (modelos clásicos) debe trabajar sobre la representación dispersa
ganadora; las features densas se reservan para el análisis de errores y como complemento en un
posible ensamble (Fase 20).
"""
    ),
    nbutil.md("## 2. Desempeño por clase de la representación ganadora"),
    nbutil.code(
        """
por_clase = pd.DataFrame({
    "B5_densas": baselines["B5"]["per_class_f1"],
    "B3_char_3_5": representaciones["E-003b"]["per_class_f1"],
    "E-008_ganadora": representaciones["E-008"]["per_class_f1"],
}).round(4)
por_clase["delta_ganadora_vs_B3"] = (por_clase["E-008_ganadora"] - por_clase["B3_char_3_5"]).round(4)
por_clase.to_csv(cfg.TABLES / "baselines_por_clase.csv")
por_clase
"""
    ),
    nbutil.md(
        """
**LECTURA.** Las clases extremas (`13-17`, `65-xx`) son las mejor clasificadas —coherente con el
EDA: son las más distintas en registro y léxico— mientras que las clases contiguas
(`35-49`/`50-64`) concentran la confusión. Esto anticipa el análisis de errores (Fase 17) y el
ajuste de umbral por clase para macro-F1 (Fase 19).

**Cierre de la Etapa B.** Representación decidida por evidencia (D-010), baselines completos y
envío v02 preparado con la ganadora.
"""
    ),
]


def main() -> int:
    """Construye y ejecuta los notebooks 04 y 05."""
    for path, cells, title in (
        (cfg.NOTEBOOKS / "04_feature_engineering.ipynb", NB04, "04 · Representaciones y features"),
        (cfg.NOTEBOOKS / "05_baselines.ipynb", NB05, "05 · Baselines B0-B5"),
    ):
        nbutil.build(path, cells, title=title)
        print(f"[build] {path}")
        print(f"[execute] {nbutil.execute(path, timeout=1800)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
