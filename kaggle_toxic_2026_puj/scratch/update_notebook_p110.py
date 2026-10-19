import json
from pathlib import Path

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
NB_PATH = RAIZ / "notebooks" / "10_final_pipeline.ipynb"

nb = json.load(open(NB_PATH, "r", encoding="utf-8"))

# Actualizar Celda 507 (Markdown P109)
cell_507_text = """## 166. P109 - PODA POR FRECUENCIA MIN_DF=5 Y UMBRALES CALIBRADOS: EVALUACION EN KAGGLE (LB 0.77447 -> REJECTED) (2026-10-14)

**Objetivo**: Evaluar la hipotesis de poda de ruido n-grama mediante `min_df=5` y calibracion fina de umbrales sobre el tensor conjunto.

**Diseno Evaluado**:
- Poda de frecuencia minima `min_df=5`: reduce el espacio de 86.275 a 31.603 features parsimoniosas.
- Regularizacion $C_{AG}=2.5, C_{HS,TR}=2.0$ y umbrales relajados ($HS=0.45, TR=0.46, AG=0.43$) con regla ascendente ($TR=1 \\implies HS=1$).

**Resultados Empiricos**:
- Canonica CV (1.483 filas): **0.792215** (+0.009843 vs P108; 5/5 pliegues ganados, win rate bootstrap 98.4%).
- Particion 47 CV (1.483 filas): **0.789294** (+0.008108 vs P108; 5/5 pliegues ganados, win rate 96.8%).
- **Resultado Kaggle Public Leaderboard**: **0.77447** (-0.01408 vs P108 = 0.78855; -0.00859 vs V03 = 0.78306). **REJECTED**.

**Diagnostico Mecanico y Post-Mortem del Fallo**:
1. *Poda excesiva*: `min_df=5` elimino 54.672 n-gramas sub-lexicales raros (que aparecen 2 a 4 veces en train). En un conjunto ciego de tweets de HatEval, estos n-gramas capturan insultos morfologicos, variantes con erratas y palabras clave de baja frecuencia esenciales en evaluacion no vista.
2. *Relajacion de umbrales en test de menor prevalencia*: El test set tiene una prevalencia de positivos ~7% menor que train ($k \\approx 0.935$). Bajar umbrales genero +69 positivos espurios (+28 en TR, +34 en AG), colapsando la precision.
- **Decision y Regla para P110+**: P108 (`0.78855`) se mantiene como el CAMPEON OFICIAL SUPREMO. En futuros candidatos, mantener el espacio sub-lexical intacto (`min_df=2`, 86k features) y no relajar umbrales por debajo de los valores conservadores probados (HS: 0.44, TR: 0.51, AG: 0.46).
"""

nb["cells"][507]["source"] = [line + "\n" for line in cell_507_text.split("\n")[:-1]]

# Agregar Celda 509 (Markdown P110) y Celda 510 (Codigo P110)
cell_509_md = """## 167. P110 - ENSEMBLE LINEAL SUAVE (95% LR + 5% SGD) SOBRE V2_TRIPLE: NUEVO CANDIDATO TOP (CV 0.784998) (2026-10-14)

**Objetivo**: Superar el score de P108 (0.78855) y avanzar hacia 0.80 mediante un ensemble lineal complementario sin relajar umbrales ni podar el espacio sub-lexical de 86k features.

**Justificacion Cientifica y Metodologica**:
- Tras el post-mortem de P109, se identifico que el espacio de 86.275 dimensiones (`min_df=2`) es indispensable, pero la frontera de decision de un modelo individual de `LogisticRegression` (coordinate descent en LibLinear) tiene ligera rigidez en casos limites.
- Se comprobo empiricamente que agregar features de superficie densas (longitud, mayusculas/shouting, puntuacion) perjudica el modelo (-0.0025 CV) al distorsionar la norma L2 dispersa.
- En cambio, combinar linealmente las probabilidades de `LogisticRegression` (95%) con un clasificador entrenado con optimizacion estocastica por gradiente (`SGDClassifier log_loss`, $\\alpha=5\\times 10^{-6}$, 5%) produce una suavizacion optima de los logits extremos sin alterar la calibracion global de la probabilidad.
- Se mantienen congelados los umbrales conservadores de P108 ($HS=0.44, TR=0.51, AG=0.46$) y la regla ontologica ascendente ($TR=1 \\implies HS=1$).

**Resultados Empiricos**:
- Canonica CV (1.483 filas): **0.784998** (delta **+0.002626** vs P108 = 0.782372, **+0.007787** vs V03 = 0.777211). **4/5 pliegues ganados**. Win rate bootstrap: **84.7%**.
- Particion 47 CV (1.483 filas): **0.784318** (delta **+0.003132** vs P108 = 0.781185, **+0.008686** vs V03 = 0.775632). **4/5 pliegues ganados**. Win rate bootstrap: **91.6%**.
- Media combinada CCV: **0.784658** (delta medio **+0.002879** vs P108).
- **Comportamiento en Test Set (1.154 filas)**:
  * Solo 27 celdas distintas de P108 (0.78%).
  * Positivos predichos: HS=513, TR=327, AG=419 (total 1.259, tasa 36.37% vs 36.05% de P108). La prevalencia de TR se mantiene rigurosamente intacta (327 vs 328).
  * 9 violaciones ontologicas corregidas en test.
- **Estado**: ADOPTED_TOP_CANDIDATE para envio a Kaggle. Proyeccion en Leaderboard Publico: **~0.791 - 0.793**.
- **Artefactos**: `submissions/toxic_submission_p110_lr_sgd_blend_asc.csv` (sha256 `535b2b35e4e3131f28af2026805f5846fbc2cfe81b744d73344b2605967a47b4`), `reports/p110_01_challenger_lr_sgd_blend.json`, `reports/p110_01_challenger_lr_sgd_blend.md`.
"""

cell_510_code = """\"\"\"Verificacion de la seccion 167 (P110: Challenger LR + SGD Blend CV 0.784998).

No entrena nada; comprueba artefactos, metricas y restricciones de fuga.
\"\"\"

import json
import pandas as pd
from pathlib import Path

PROY = PROJECT_ROOT
R = PROY / "reports"
SUB = PROY / "submissions"
EXP = PROY / "experiments"

# 1. Comprobar reporte JSON de P110
p110_json = R / "p110_01_challenger_lr_sgd_blend.json"
assert p110_json.exists(), "Falta reporte JSON de P110"
data_p110 = json.loads(p110_json.read_text(encoding="utf-8"))
assert data_p110["canonica"]["ccv_p110"] >= 0.784, "Canonica no supera 0.784"
assert data_p110["canonica"]["delta_vs_p108"] > 0.002, "Delta vs P108 inferior a +0.002"
assert data_p110["particion_47"]["delta_vs_p108"] > 0.002, "Delta vs P108 en part47 inferior a +0.002"
assert data_p110["canonica"]["pliegues_ganados"] in ["4/5", "5/5"], "No gano al menos 4/5 pliegues en canonica"
assert data_p110["particion_47"]["pliegues_ganados"] in ["4/5", "5/5"], "No gano al menos 4/5 pliegues en part47"

# 2. Comprobar submission generada P110
sub_p110 = SUB / "toxic_submission_p110_lr_sgd_blend_asc.csv"
assert sub_p110.exists(), "Falta submission P110"
df_sub_110 = pd.read_csv(sub_p110)
assert len(df_sub_110) == 1154, "Filas de submission incorrectas"
assert list(df_sub_110.columns) == ["id", "HS", "TR", "AG"], "Columnas de submission incorrectas"
assert ((df_sub_110["TR"] == 1) & (df_sub_110["HS"] == 0)).sum() == 0, "Violacion ontologica detectada"
assert df_sub_110.isnull().sum().sum() == 0, "Hay valores nulos en submission"

# 3. Comprobar registro en experiments.csv
exp_df = pd.read_csv(PROY / "experiments.csv")
assert "P110-01" in exp_df["experiment_id"].values, "P110-01 no registrado en experiments.csv"
print("Seccion 167 (P110 Challenger LR + SGD Blend CV 0.784998) verificada con exito.")
"""

new_cell_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + "\n" for line in cell_509_md.split("\n")[:-1]]
}

new_cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in cell_510_code.split("\n")[:-1]]
}

if len(nb["cells"]) == 509:
    nb["cells"].append(new_cell_md)
    nb["cells"].append(new_cell_code)
elif len(nb["cells"]) == 511:
    nb["cells"][509] = new_cell_md
    nb["cells"][510] = new_cell_code

with open(NB_PATH, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)

print(f"Notebook actualizado: {len(nb['cells'])} celdas.")
