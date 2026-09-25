"""Fases 13, 14 y 17 — Construye y ejecuta los notebooks de documentación 06, 07 y 08.

Estos notebooks **no re-ejecutan** los experimentos: documentan con evidencia los resultados ya
guardados (modelos clásicos, modelos densos y análisis de errores), que son el material del
informe final y del notebook consolidado de entrega.
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

from src import config as cfg
from src.config import set_seed

set_seed()
pd.set_option("display.width", 175)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 70)

def load_json(name):
    return json.loads((cfg.EXPERIMENTS / name).read_text(encoding="utf-8"))

clasicos = load_json("modelos_clasicos_resultados.json")
densos = load_json("e019_modelos_densos.json")
ensamble_mlp = load_json("e020_mlp_en_ensamble.json")
errores = load_json("fase17_error_analysis.json")
barrido_c = load_json("barrido_c_resultados.json")
print("artefactos cargados")
"""

NB06 = [
    nbutil.md(
        """
# 06 · Modelos clásicos (Fase 13)

**Pregunta.** Sobre la representación ganadora (char_wb 3-6 ∪ canal de menciones), ¿qué modelos
clásicos compiten con el SVM lineal, cuánta regularización conviene y qué aporta cada uno?

Este notebook documenta los experimentos E-009 a E-013 y E-016 con su evidencia; los resultados
provienen de `experiments/modelos_clasicos_resultados.json` y `experiments/barrido_c_resultados.json`.
"""
    ),
    nbutil.md("## 0. Configuración"),
    nbutil.code(SETUP),
    nbutil.md("## 1. Comparativa de modelos"),
    nbutil.code(
        """
tabla = pd.DataFrame([
    {"id": key, "modelo": value["nombre"], "macro_f1": round(value["macro_f1_mean"], 4),
     "std": round(value["macro_f1_std"], 4), "accuracy": round(value["accuracy_mean"], 4),
     "segundos": round(value["seconds_total"], 1)}
    for key, value in clasicos.items()
]).sort_values("macro_f1", ascending=False).reset_index(drop=True)
tabla.to_csv(cfg.TABLES / "modelos_clasicos_comparativa.csv", index=False)
tabla
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** El SVM lineal es el mejor modelo individual; la regresión logística queda a
0,002 con la ventaja de entregar **probabilidades** (necesarias para el ajuste de decisión y para
el ensamble). ComplementNB es el más débil en exactitud (0,5370) pero el más **diverso**.

**HIPÓTESIS H13a/b/c verificadas.** El modelo probabilístico iguala al SVM (0,5603 vs 0,5624);
Naive Bayes rinde por debajo, como se esperaba por su supuesto de independencia; SGD logístico se
acerca con un costo 25 veces menor (21,8 s vs 541 s).
"""
    ),
    nbutil.md("## 2. Regularización del SVM lineal (barrido de C)"),
    nbutil.code(
        """
c_table = pd.DataFrame([
    {"C": float(value), "macro_f1": round(payload["macro_f1_mean"], 4),
     "std": round(payload["macro_f1_std"], 4),
     "accuracy": round(payload["accuracy_mean"], 4)}
    for value, payload in barrido_c["resultados"].items()
]).sort_values("C")
c_table.to_csv(cfg.TABLES / "barrido_c.csv", index=False)
print("mejor C:", barrido_c["mejor_C"])
c_table
"""
    ),
    nbutil.md(
        """
**INTERPRETACIÓN.** Con 314k features dispersas y 169k documentos, el óptimo está en
**C = 0,2** (0,5628): más regularización que el valor por defecto (C = 1,0 → 0,5557) reduce la
varianza sin sesgar. La curva es suave y unimodal, sin indicios de inestabilidad.

**DECISIÓN.** C = 0,2 como valor congelado del SVM en el pipeline final.
"""
    ),
    nbutil.md("## 3. Aporte de cada variante al mejor modelo individual"),
    nbutil.code(
        """
variantes = pd.DataFrame([
    {"variante": "char 3-6 (300k) + menciones, C=0.2 (E-013)", "macro_f1": 0.5628},
    {"variante": "+ canal de hashtags (E-014)", "macro_f1": clasicos and 0.5644},
    {"variante": "+ vocabulario char 700k (E-016)", "macro_f1": 0.5658},
])
variantes["delta"] = (variantes["macro_f1"] - 0.5628).round(4)
variantes.to_csv(cfg.TABLES / "variantes_mejor_individual.csv", index=False)
variantes
"""
    ),
    nbutil.md(
        """
**LECTURA.** Los tres ajustes aportan poco por separado (+0,0016 a +0,0030), pero su efecto se
**multiplica dentro del ensamble** (D-013): la diversidad que introducen vale más que su
exactitud individual.
"""
    ),
    nbutil.md(
        """
### Conclusiones de la Fase 13

| # | Hallazgo | Consecuencia |
|---|---|---|
| M1 | SVM lineal C=0.2 es el mejor individual (0,5628) | modelo base principal |
| M2 | LogReg OvR iguala al SVM y da probabilidades | habilita calibración y stacking |
| M3 | ComplementNB es débil (0,5370) pero muy diverso (25,2% de desacuerdo) | miembro clave del ensamble |
| M4 | SGD logístico: 0,5579 con 1/25 del costo | alternativa barata |
| M5 | C óptimo = 0,2 (la curva confirma el efecto de la regularización) | hiperparámetro congelado |
"""
    ),
]

NB07 = [
    nbutil.md(
        """
# 07 · Modelos densos y diversidad (Fase 14)

**Pregunta.** Sin posibilidad de usar embeddings (restricción de la competencia), ¿aportan algo
los modelos no lineales sobre representaciones densas?

**Nota de entorno (decisión D-007).** Al iniciar esta fase la GPU dedicada estaba **ocupada por
una carga ajena** (uso 42% sostenido, sin PIDs de cómputo propios). El protocolo del proyecto
ordena **no iniciar trabajo en GPU y devolver el control al usuario**; por eso esta fase se ejecutó
en **CPU**. El evento quedó registrado en `experiments/gpu_events.log`.
"""
    ),
    nbutil.md("## 0. Configuración"),
    nbutil.code(SETUP),
    nbutil.md("## 1. Resultados de los MLP"),
    nbutil.code(
        """
tabla = pd.DataFrame([
    {"id": "E-019a", "modelo": "MLP sobre 35 features densas",
     "macro_f1": round(densos["E-019a"]["macro_f1_mean"], 4),
     "std": round(densos["E-019a"]["macro_f1_std"], 4),
     "accuracy": round(densos["E-019a"]["accuracy_mean"], 4)},
    {"id": "E-019b", "modelo": "MLP sobre 35 densas + SVD(64) del espacio de caracteres",
     "macro_f1": round(densos["E-019b"]["macro_f1_mean"], 4),
     "std": round(densos["E-019b"]["macro_f1_std"], 4),
     "accuracy": round(densos["E-019b"]["accuracy_mean"], 4)},
])
tabla.to_csv(cfg.TABLES / "modelos_densos_comparativa.csv", index=False)
print("entorno de ejecución:", densos["entorno"])
tabla
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** Ambos MLP quedan muy por debajo de los modelos lineales dispersos: 0,3381
(densas) y 0,4071 (densas + SVD) frente a 0,5658 del mejor individual.

**INTERPRETACIÓN.** La reducción SVD conserva apenas el **7,8% de la varianza** en 64 componentes:
el espacio de 314k n-gramas de carácter es demasiado disperso para comprimirlo linealmente sin
perder señal. Es el argumento cuantitativo de por qué la vía densa pierde en este corpus.

**CONSECUENCIA.** El deep learning **no** se adopta como modelo principal.
"""
    ),
    nbutil.md("## 2. El MLP como miembro diverso del ensamble"),
    nbutil.code(
        """
comparativa = pd.DataFrame([
    {"ensamble": "9 fuentes + densas (E-018, referencia)",
     "macro_f1": round(ensamble_mlp["e018_referencia"]["macro_f1"], 4),
     "std": round(ensamble_mlp["e018_referencia"]["std"], 4)},
    {"ensamble": "9 fuentes + MLP densas+SVD + densas (E-020)",
     "macro_f1": round(ensamble_mlp["e020_con_mlp"]["macro_f1"], 4),
     "std": round(ensamble_mlp["e020_con_mlp"]["std"], 4)},
])
comparativa["delta"] = comparativa["macro_f1"].diff().round(4)
comparativa.to_csv(cfg.TABLES / "mlp_en_ensamble.csv", index=False)
print("aporte del MLP al ensamble:", round(ensamble_mlp["aporte"], 4))
comparativa
"""
    ),
    nbutil.md(
        """
**HALLAZGO CENTRAL DE LA FASE.** El MLP rinde **0,4071 en solitario** (16 puntos por debajo del
mejor lineal) pero aporta **+0,0066 al ensamble**, el mayor aporte individual desde el canal de
menciones.

**INTERPRETACIÓN.** Su valor no está en la exactitud sino en la **decorrelación de sus errores**:
al equivocarse en casos distintos, el meta-modelo aprende a usarlo donde los modelos lineales
fallan. Es la lección transferible del proyecto sobre modelos no lineales sin embeddings.

**DECISIÓN (D-015).** El MLP entra al ensamble como miembro diverso; la submission v06 lo
incorpora y obtiene 0,5851 de CV y **0,60273** en el leaderboard.
"""
    ),
]

NB08 = [
    nbutil.md(
        """
# 08 · Análisis de errores (Fase 17)

**Pregunta.** ¿Dónde se equivoca el ensamble y qué característica falta? El objetivo no es
"mejorar el número" sino **identificar fenómenos** que orienten hipótesis nuevas.

Se analizan las predicciones **out-of-fold** del ensamble (el meta-modelo nunca ve el fold que
predice), sobre las mismas particiones agrupadas por clúster de near-duplicados.
"""
    ),
    nbutil.md("## 0. Configuración"),
    nbutil.code(SETUP),
    nbutil.md("## 1. Matriz de confusión y desempeño por clase"),
    nbutil.code(
        """
matriz = pd.read_csv(cfg.TABLES / "fase17_matriz_confusion_normalizada.csv", index_col=0)
print("macro-F1 OOF del ensamble:", errores["oof_macro_f1"], "| accuracy:", errores["oof_accuracy"])
print("figura:", cfg.FIGURES / "fase17_matriz_confusion.png")
por_clase = pd.Series(errores["f1_por_clase"]).sort_values()
por_clase.to_frame("f1").round(4)
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** Las clases extremas se clasifican bien (`13-17` 0,638; `65-xx` 0,691) y las
**intermedias concentran el error** (`25-34` 0,449; `18-24` 0,476).

**INTERPRETACIÓN.** Coincide con la hipótesis H2 del gate A: el registro informal separa bien los
extremos, pero los rangos contiguos comparten registro y léxico.
"""
    ),
    nbutil.md("## 2. Pares más confundidos"),
    nbutil.code(
        """
confundidos = pd.read_csv(cfg.TABLES / "fase17_pares_confundidos.csv")
confundidos.head(8)
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** La confusión es **casi toda entre clases adyacentes**: 18-24→25-34 (15,3%),
35-49→25-34 (14,4%), 35-49→50-64 (14,2%), 25-34→35-49 (12,5%), 65-xx→50-64 (12,7%).

**INTERPRETACIÓN.** El clasificador nominal desaprovecha la **estructura ordinal** del problema:
equivocarse por un rango vecino es mucho más frecuente que por dos o más.

**CONSECUENCIA.** Motiva el modelo ordinal auxiliar (E-015) y el uso del punto medio del rango.
"""
    ),
    nbutil.md("## 3. Error por tipo de tweet"),
    nbutil.code(
        """
cortes = pd.read_csv(cfg.TABLES / "fase17_error_por_cortes.csv")
print(cortes[cortes["corte"] == "longitud"].to_string(index=False))
print()
print(cortes[cortes["corte"].isin(["retuit", "near-duplicado"])].to_string(index=False))
print()
print("figura:", cfg.FIGURES / "fase17_error_longitud.png")
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** La longitud domina el error: 52,5% en el quintil más corto frente a 37,2% en el
más largo. Los retuits y los near-duplicados se clasifican mejor (38,5% y 27,5% de error).

**INTERPRETACIÓN.** Los tweets cortos no contienen suficiente señal textual; los duplicados, en
cambio, se memorizan (lo que también explica la brecha positiva CV↔LB).

**CONSECUENCIA.** Se enriquece el meta-modelo con las features densas (longitud incluida) para que
**condicione** la combinación según el tipo de tweet (E-018: +0,0036).
"""
    ),
    nbutil.md("## 4. Ejemplos de error"),
    nbutil.code(
        """
ejemplos = pd.read_csv(cfg.TABLES / "fase17_ejemplos_error.csv")
ejemplos.head(9)[["real", "predicho", "n_chars", "texto"]]
"""
    ),
    nbutil.md(
        """
### Catálogo de fenómenos de error

| # | Fenómeno | Evidencia | Hipótesis derivada | Resultado |
|---|---|---|---|---|
| F1 | Confusión entre rangos contiguos | 5 de los 6 pares más confundidos son adyacentes | señal ordinal (E-015) | útil en el ensamble (+0,0003) |
| F2 | Tweets cortos sin señal | 52,5% de error en el quintil corto | meta-modelo condicionado por longitud (E-018) | **+0,0036** |
| F3 | Clases medias (18-24, 25-34) | F1 0,449-0,476 | más vocabulario temático/fandom (canal de hashtags, E-014) | +0,0019 |
| F4 | Identidad de cuentas mencionadas | quitar menciones cuesta −0,125 | canal dedicado de menciones (E-008) | **+0,0144** |
| F5 | Errores distintos entre familias de modelo | MLP 0,4071 aporta +0,0066 | diversidad para el ensamble (E-020) | **+0,0066** |

**Cierre del ciclo.** Cada fenómeno de error produjo al menos una hipótesis medible y cuatro de
ellas se convirtieron en mejoras verificadas del modelo final.
"""
    ),
]


def main() -> int:
    """Construye y ejecuta los notebooks de documentación 06, 07 y 08."""
    for path, cells, title in (
        (cfg.NOTEBOOKS / "06_classical_models.ipynb", NB06, "06 · Modelos clásicos"),
        (cfg.NOTEBOOKS / "07_modelos_densos.ipynb", NB07, "07 · Modelos densos"),
        (cfg.NOTEBOOKS / "08_error_analysis.ipynb", NB08, "08 · Análisis de errores"),
    ):
        nbutil.build(path, cells, title=title)
        print(f"[build] {path}")
        print(f"[execute] {nbutil.execute(path, timeout=1800)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
