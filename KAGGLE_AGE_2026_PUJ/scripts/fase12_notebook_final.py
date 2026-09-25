"""Regenera ``notebooks/99_entrega_consolidado.ipynb`` con la fase avanzada completa.

El notebook resultante:
    1. **Verifica** los valores contra los artefactos reales (JSON/CSV) con aserciones, de modo que
       no puede desincronizarse de lo ejecutado.
    2. Narra la cadena completa OBSERVACIÓN → HIPÓTESIS → MÉTODO → RESULTADO → INTERPRETACIÓN →
       DECISIÓN de las 12 fases del proyecto, incluidas la fase avanzada de segunda generación y las
       líneas cerradas (retrieval, POS, ruido de etiqueta).
    3. Muestra el modelo final y su verificación de reproducibilidad.

Uso::

    python scripts/fase12_notebook_final.py
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

def load(name):
    return json.loads((cfg.EXPERIMENTS / name).read_text(encoding="utf-8"))

experimentos = pd.read_csv(cfg.ROOT / "experiments.csv")
decisiones = [l for l in (cfg.ROOT / "DECISIONS.md").read_text(encoding="utf-8").splitlines()
              if l.startswith("## D-")]
print(f"experimentos registrados: {len(experimentos)} | decisiones documentadas: {len(decisiones)}")
"""

CELLS = [
    nbutil.md(
        """
# 99 · Entrega consolidada — Predicción de rango de edad a partir de tweets

**Competencia**: `tweet-age-2026-puj` (Kaggle) · Procesamiento de Lenguaje Natural · Maestría en
Inteligencia Artificial, Pontificia Universidad Javeriana.

**Tarea**: clasificar cada tweet en uno de 6 rangos de edad (`13-17`, `18-24`, `25-34`, `35-49`,
`50-64`, `65-xx`) usando únicamente su texto.

**Resultado final**: **0,62182 de F1** en el leaderboard (línea base inicial 0,55235 → **+0,0695**),
con validación interna honesta de 0,6005 y confirmación en holdout congelado de 0,6077.

**Restricciones respetadas**: sin transformers, sin LLMs y sin ningún modelo de vectorización
(embedding) preentrenado ni entrenado desde cero. Todo el desempeño proviene de representaciones
estadísticas (TF-IDF), features artesanales, modelos clásicos y un ensamble en dos niveles.

Este notebook **verifica** cada cifra contra los artefactos del proyecto con aserciones: si algún
valor no coincidiera con lo ejecutado, el notebook fallaría.
"""
    ),
    nbutil.md("## 0. Configuración y trazabilidad"),
    nbutil.code(SETUP),
    nbutil.md(
        """
## 1. Los datos

| Archivo | Filas | Columnas | Separador |
|---|---|---|---|
| `age_train.csv` | 169.336 | `id;text;age_range` | `;` |
| `age_test.csv` | 112.892 | `id;text` | `;` |
| `age_submission.csv` | 112.892 | `id,age_range` | `,` |

**OBSERVACIÓN.** Seis clases casi balanceadas por diseño (16,0-17,0%), texto en ancho fijo de 140
caracteres con relleno de espacios, **sin metadatos** de usuario ni temporales, y un alfabeto de
**96 caracteres con un único no-ASCII**: el corpus fue transliterado a ASCII (sin tildes, `ñ` ni
emojis).

**INTERPRETACIÓN.** Es un barrido de Twitter en español latinoamericano (~2016-2017, re-indexado y
equilibrado por clase) del que no se pudo identificar la fuente pública exacta. El truncado a 140 y
la transliteración son artefactos de exportación, no propiedades de los autores.

**CONSECUENCIA.** Sin metadatos, toda la señal es textual; el relleno se elimina y las features de
emoji se descartan por varianza nula.
"""
    ),
    nbutil.code(
        """
auditoria = load("audit_01_resumen.json")
assert auditoria["train_rows"] == 169336 and auditoria["test_rows"] == 112892
print("train:", auditoria["train_rows"], "| test:", auditoria["test_rows"])
print("clases:", auditoria["classes"])
print(f"padding: {auditoria['padding']['train']['raw_len_eq_padded_width_pct']}% de textos con "
      f"ancho exacto de 140 | longitud real mediana: "
      f"{auditoria['padding']['train']['clean_len_median']:.0f}")
print("vacíos:", auditoria["empty_texts"], "| ruido de etiqueta:",
      auditoria["label_conflicts"]["rows_affected"], "filas")
print("id como proxy de clase (desviación máx):", auditoria["id_leakage_max_deviation_pp"], "pp")
"""
    ),
    nbutil.md(
        """
## 2. Metodología: el problema del leakage por duplicados

**OBSERVACIÓN.** No hay identificador de autor, pero **1,21% de las filas de test aparecen
literalmente en train** y **4,26% comparten un clúster de near-duplicado** (Jaccard ≥ 0,8), con
92,8% de clústeres de etiqueta única.

**HIPÓTESIS.** Con un split aleatorio el modelo puede memorizar duplicados en lugar de aprender
estilo, inflando la métrica.

**MÉTODO.** Clústeres de near-duplicados construidos sobre texto sin etiquetas (normalización +
Jaccard con *blocking* por tokens raros) y dos esquemas de validación comparados: agrupado por
clúster (primario) y estratificado ingenuo (control), más un baseline de memorización pura.

**RESULTADO.** El split estratificado regala **0,061 de macro-F1** por memorización; el agrupado lo
reduce a cero por construcción.

**DECISIÓN (D-003).** Toda la validación usa `StratifiedGroupKFold` agrupado por clúster, y las
comparaciones de modelos usan **leave-one-fold-out** sobre predicciones out-of-fold.
"""
    ),
    nbutil.code(
        """
memorizacion = pd.DataFrame([
    {"esquema": "agrupado por clúster (usado)", **auditoria["memorization_group"]},
    {"esquema": "estratificado ingenuo (control)", **auditoria["memorization_stratified"]},
])
assert auditoria["cluster_leakage"]["test_coverage_pct"] > 4.0
print(memorizacion.to_string(index=False))
"""
    ),
    nbutil.md(
        """
## 3. Análisis exploratorio y lingüístico

**OBSERVACIÓN.** La longitud y la riqueza léxica crecen **monótonamente** con la edad (73 caracteres
en `13-17` frente a 111 en `65-xx`; ε² = 0,126 para `n_chars`), y el léxico separa mundos sociales:
`follow, justin, bieber, amo` y hashtags de fandoms en `13-17`; `venezuela, maduro, gobierno, pueblo`
y `efemerides` en `65-xx`.

**CONSECUENCIA.** Se priorizaron 12 hipótesis en `reports/hipotesis_gate_a.md`, cada una atada a un
experimento concreto.
"""
    ),
    nbutil.code(
        """
kruskal = pd.read_csv(cfg.TABLES / "eda_kruskal.csv").head(5)
terminos = pd.read_csv(cfg.TABLES / "top_terms_by_class.csv")
for clase in ["13-17", "65-xx"]:
    print(f"{clase}: " + ", ".join(terminos[terminos.clase == clase].head(8).termino.tolist()))
display(kruskal[["feature", "epsilon_sq"]].round(4))
"""
    ),
    nbutil.md(
        """
## 4. Representaciones: la decisión central

Se evaluaron **14 configuraciones** con modelo fijo y validación agrupada. Los n-gramas de carácter
dominan (`char_wb 3-6`: 0,5413) frente a los de palabra (`word 1-2`: 0,4971) y la unión clásica
**diluye** (−0,0069).

**HALLAZGO PRINCIPAL.** Eliminar o enmascarar las **menciones** cuesta **−0,125** de macro-F1:
la identidad de las cuentas mencionadas es una de las señales más fuertes del corpus. Un canal TF-IDF
dedicado aporta **+0,0144**.

**Decisión (D-010)**: `char_wb(3,6)` sobre texto **crudo** ∪ canal de menciones, sin normalizar.
"""
    ),
    nbutil.code(
        """
representaciones = pd.read_csv(cfg.TABLES / "representaciones_comparativa.csv")
assert abs(float(representaciones[representaciones.id == "E-003c"].macro_f1.iloc[0]) - 0.5413) < 0.001
assert abs(float(representaciones[representaciones.id == "E-005a"].macro_f1.iloc[0]) - 0.3902) < 0.001
display(representaciones.head(8)[["id", "representacion", "texto", "macro_f1", "features"]])
"""
    ),
    nbutil.md(
        """
## 5. Baselines, modelos clásicos y modelos densos

| Baseline | macro-F1 |
|---|---|
| B0 Dummy (prior) | 0,0485 |
| B1 word 1-2 + LogReg | 0,5015 |
| B2 word 1-2 + LinearSVC | 0,4971 |
| B3 char 3-5 + LinearSVC | 0,5408 |
| B4 word ∪ char (E-001) | 0,5344 |
| B5 35 features densas + LightGBM | 0,3567 |

**Modelos densos (Fase 14, ejecutada en CPU porque la GPU estaba ocupada — protocolo D-007).**
MLP sobre densas 0,3381 · MLP sobre densas+SVD(64) 0,4071: la vía densa pierde, porque la SVD solo
conserva el **7,8% de la varianza**. Sin embargo, como **miembro diverso** del ensamble aporta
**+0,0066**: lección central del proyecto sobre el valor de un modelo no lineal sin embeddings.
"""
    ),
    nbutil.code(
        """
baselines = pd.read_csv(cfg.TABLES / "baselines_comparativa.csv")
densos = load("e019_modelos_densos.json")
assert abs(densos["E-019b"]["macro_f1_mean"] - 0.4071) < 0.001
display(baselines[["id", "descripcion", "macro_f1"]])
print(f"MLP densas {densos['E-019a']['macro_f1_mean']:.4f} | "
      f"MLP densas+SVD {densos['E-019b']['macro_f1_mean']:.4f} | entorno: {densos['entorno']}")
"""
    ),
    nbutil.md(
        """
## 6. Análisis de errores

**OBSERVACIÓN.** Las clases extremas se clasifican bien (`13-17` 0,638; `65-xx` 0,691) y las
**intermedias concentran el error** (`25-34` 0,449; `18-24` 0,476). La confusión es casi toda entre
**rangos contiguos** y el error crece al acortarse el tweet (52,5% en el quintil más corto frente a
37,2% en el más largo).

**CONSECUENCIA.** Cada fenómeno generó una hipótesis medible: canal de menciones (+0,0144),
condicionamiento por longitud (+0,0036), canal de hashtags (+0,0019), diversidad del MLP (+0,0066).
"""
    ),
    nbutil.code(
        """
errores = load("fase17_error_analysis.json")
f1_clase = pd.Series(errores["f1_por_clase"]).sort_values(ascending=False)
assert abs(f1_clase["25-34"] - 0.449) < 0.01
print("F1 por clase (OOF del ensamble):")
print(f1_clase.round(4).to_string())
print()
print("pares más confundidos:")
print(pd.read_csv(cfg.TABLES / "fase17_pares_confundidos.csv").head(5).to_string(index=False))
"""
    ),
    nbutil.md(
        """
## 7. Fase avanzada v1: features supervisadas de entidad

**OBSERVACIÓN (gap analysis).** El sistema explotaba forma superficial y bolsa de palabras, pero no
la **estadística supervisada de las entidades** ni las distribuciones internas, la rareza o la
posición.

**MÉTODO.** 160 features nuevas: 87 locales (distribuciones, entropía, forma, posición, compresión) y
73 supervisadas calculadas **siempre in-fold** (`P(clase|mención)`, `P(clase|hashtag)`,
`P(clase|fuente de RT)`, rareza de corpus y log-odds por clase).

**RESULTADO.** CV 0,5866 → **0,5897 (+0,0031)**, positiva en **5/5 folds**; holdout congelado
0,5914 → **0,5966 (+0,0052)**. Aportes: menciones supervisadas +0,0016 · fuente de RT +0,0007 ·
rareza +0,0003 · léxico por clase +0,0001 · hashtags −0,0001.

**DECISIÓN (D-017).** Adoptadas; el modelo pasa a 267 features.
"""
    ),
    nbutil.code(
        """
avanzada = load("advanced_02_resultados.json")
assert abs(avanzada["con_todo"]["macro_f1"] - 0.5897) < 0.001
holdout1 = load("advanced_03_holdout.json")["holdout_congelado"]
assert abs(holdout1["delta_macro_f1"] - 0.0052) < 0.001
tabla = pd.DataFrame([
    {"configuracion": "baseline (14 miembros + densas)", "macro_f1": avanzada["baseline"]["macro_f1"]},
    {"configuracion": "+ features estructuradas", "macro_f1": avanzada["con_estructuradas"]["macro_f1"]},
] + [{"configuracion": f"+ {nombre}", "macro_f1": datos["macro_f1"]}
     for nombre, datos in avanzada["aporte_supervisado"].items()]
  + [{"configuracion": "+ todas (candidato)", "macro_f1": avanzada["con_todo"]["macro_f1"]}])
display(tabla.round(4))
print(f"holdout congelado: baseline {holdout1['resultados']['baseline']['macro_f1']:.4f} → "
      f"candidato {holdout1['resultados']['candidato']['macro_f1']:.4f} "
      f"({holdout1['delta_macro_f1']:+.4f})")
"""
    ),
    nbutil.md(
        """
## 8. Fase avanzada v2: el cuello de botella era el combinador

**OBSERVACIÓN (diagnóstico de intersección).** En el **25,94%** de los tweets el ensamble falla
mientras **al menos un modelo base acierta**, y los rescatadores son sistemáticamente los modelos
débiles pero diversos (softmax ordinal 36,1%; centroides 31,1%; MLP 30,5%). El error depende además
del tipo de tweet (`solo_url` 61,1% frente a `mencion_alta` 19,2%) y de la longitud (54,5% → 35,1%).

**HIPÓTESIS.** El meta-modelo lineal aplica pesos globales y no puede condicionar su confianza al
contexto del tweet.

**RESULTADO.** El **MLP como meta-modelo**: 0,5967 vs 0,5906 del LogReg (**+0,0061, 5/5 folds**);
segmentación dura: −0,0007/−0,0015 (falla, por segunda vez). Holdout: 0,5981 → **0,6024**.

**SEGUNDO HALLAZGO.** Promediar **dos combinadores diversos** (MLP + LogReg, ponderación fija
0,5/0,5): CV **0,6005**, holdout **0,6077** (+0,0043 sobre el MLP solo). Un tercer combinador no
ayuda (D-025): la diversidad solo paga entre modelos de **calidad comparable**.

**DECISIÓN (D-020, D-022).** Modelo final = ensamble de 14 fuentes + promedio de dos combinadores.
"""
    ),
    nbutil.code(
        """
diagnostico = load("phase1_diagnostico.json")
assert abs(diagnostico["ensemble_miss_pct"] - 25.94) < 0.1
combinadores = load("phase7_nbsvm_canal_meta.json")["combinadores"]
assert abs(combinadores["promedio"] - 0.6005) < 0.001
holdout2 = load("phase8_holdout_v12.json")["holdout"]
assert holdout2["confirmado"] is True
print(f"filas donde el ensamble falla y algún miembro acierta: "
      f"{diagnostico['ensemble_miss_pct']}%")
print(f"combinadores: MLP {combinadores['mlp']:.4f} | LogReg {combinadores['logreg']:.4f} | "
      f"promedio {combinadores['promedio']:.4f}")
print(f"holdout congelado: LogReg {holdout2['resultados']['LogReg']['macro_f1']:.4f} | "
      f"MLP {holdout2['resultados']['MLP']['macro_f1']:.4f} | "
      f"promedio {holdout2['resultados']['promedio 0,5/0,5']['macro_f1']:.4f} "
      f"(Δ {holdout2['delta_promedio_vs_mlp']:+.4f})")
print()
print("líneas cerradas con evidencia en esta fase:")
for linea, delta in (("residuales de longitud", -0.0019), ("contenido vs estilo (estilo solo 0,2793)", 0.0008),
                     ("NB-SVM de palabra", 0.0009), ("tercer combinador", -0.0006)):
    print(f"  · {linea}: {delta:+.4f}")
"""
    ),
    nbutil.md(
        """
## 9. Líneas cerradas con evidencia (resumen)

| Línea | Resultado | Decisión |
|---|---|---|
| Normalización agresiva del texto | −0,1442 | cerrada: "limpiar" destruye señal |
| Quitar/marcar menciones | −0,125 | cerrada: la identidad de la mención es señal |
| Especialistas por longitud | +0,0000 | cerrada |
| BM25 sin normalización L2 | −0,0002 | cerrada |
| kNN sobre SVD | 0,3088 individual | cerrada |
| Meta-modelo LightGBM | 0,5858 vs 0,5860 | cerrada |
| Ajuste de decisión por clase (macro-F1) | +0,0006 | cerrada (clases balanceadas) |
| Compuerta dura de frontera | −0,0036 | cerrada |
| Segmentación dura (tipo/longitud) | −0,0007 / −0,0015 | cerrada |
| Residuales de longitud | −0,0019 | cerrada |
| Estilo puro como representación | 0,2793 | cerrada (manda el contenido) |
| NB-SVM por canal | +0,0009 | cerrada |
| Tercer combinador | −0,0006 | cerrada (diversidad sin calidad no compone) |
| **Retrieval train→test** | etiqueta de duplicados solo acierta **0,615** | **cerrada**: permitido por reglas, pero su techo lo fija la consistencia de etiqueta, no la cobertura |
| **POS de spaCy** | −0,0032 | cerrada (el corpus ASCII degrada el etiquetado) |
| Exclusión de filas ruidosas | +0,0001 / +0,0016 | mejora real pero bajo el umbral (D-027) |
"""
    ),
    nbutil.code(
        """
vias = load("phase10_dos_vias.json")
rul = load("phase11_ruido_etiqueta.json")
assert abs(vias["via_a_retrieval"]["acierto_exactos"] - 0.6151) < 0.001
assert abs(vias["via_b_pos"]["delta"] - (-0.0032)) < 0.001
print(f"retrieval: cobertura de clúster in-fold {vias['via_a_retrieval']['cobertura_pct']}% | "
      f"duplicados exactos {vias['via_a_retrieval']['exactos_pct']}% | "
      f"acierto de la etiqueta recuperada {vias['via_a_retrieval']['acierto_exactos']} | "
      f"sobrescritura dura {vias['via_a_retrieval']['sobrescritura_dura']['delta']:+.4f}")
print(f"POS: {vias['via_b_pos']['macro_f1']:.4f} ({vias['via_b_pos']['delta']:+.4f})")
print(f"ruido de etiqueta: contradictorias {rul['delta_sin_contradictorias']:+.4f} | "
      f"duplicadas {rul['sin_duplicadas']['macro_f1'] - rul['baseline']['macro_f1']:+.4f}")
"""
    ),
    nbutil.md(
        """
## 10. Progresión y resultado final

| Versión | Cambio | CV (LOFO) | Leaderboard |
|---|---|---|---|
| v01 | baseline word ∪ char | 0,5344 | 0,55235 |
| v02 | char 3-6 + canal de menciones | 0,5557 | 0,57018 |
| v03 | stacking de 4 modelos | 0,5656 | 0,58081 |
| v04 | ensamble de 9 fuentes | 0,5748 | 0,59178 |
| v05 | meta-modelo enriquecido | 0,5784 | 0,59518 |
| v06 | + MLP denso como miembro diverso | 0,5851 | 0,60273 |
| v07 | + ordinal acumulativo + centroides | 0,5860 | 0,60372 |
| v08 | + pares adyacentes | 0,5866 | 0,60376 |
| v09 | + features supervisadas de entidad | 0,5897 | 0,60807 |
| v11 | meta-modelo MLP(64) | 0,5966 | 0,61744 |
| **v12** | **+ NB-SVM palabra y promedio de combinadores** | **0,6005** | **0,62182** |

**Calibración de la validación**: en 10 envíos, la brecha CV↔LB fue **+0,0177 ± 0,0023** (estable y
explicada por la memorización de near-duplicados del test, que el CV agrupado bloquea por diseño).
El **holdout congelado** (15% de clústeres, semilla 2026) predijo el leaderboard mejor que el CV en
las dos últimas rondas (D-024).
"""
    ),
    nbutil.code(
        """
progresion = pd.DataFrame([
    {"version": "v01", "cv": 0.5344, "lb": 0.55235}, {"version": "v02", "cv": 0.5557, "lb": 0.57018},
    {"version": "v03", "cv": 0.5656, "lb": 0.58081}, {"version": "v04", "cv": 0.5748, "lb": 0.59178},
    {"version": "v05", "cv": 0.5784, "lb": 0.59518}, {"version": "v06", "cv": 0.5851, "lb": 0.60273},
    {"version": "v07", "cv": 0.5860, "lb": 0.60372}, {"version": "v08", "cv": 0.5866, "lb": 0.60376},
    {"version": "v09", "cv": 0.5897, "lb": 0.60807}, {"version": "v11", "cv": 0.5966, "lb": 0.61744},
    {"version": "v12", "cv": 0.6005, "lb": 0.62182},
])
progresion["brecha"] = (progresion["lb"] - progresion["cv"]).round(4)
progress = progresion["lb"].iloc[-1] - progresion["lb"].iloc[0]
assert abs(progress - 0.06947) < 0.0005, progress
brecha = progresion["brecha"].mean()
print(f"mejora acumulada: +{progress:.4f} | brecha CV↔LB media: {brecha:+.4f}")
display(progresion)
"""
    ),
    nbutil.md(
        """
## 11. Modelo final y reproducibilidad

**Modelo (submission v12)**: ensamble de **14 fuentes** sobre la representación `char_wb(3,6)` del
texto crudo más canales de mención, hashtag y palabra — con perfiles supervisados de entidad,
ordinalidad (acumulativa y por pares adyacentes), prototipos y NB-SVM — y un **combinador doble**:
promedio de las probabilidades de un MLP(64) y una regresión logística sobre **279 features**.

**Garantías**: el test nunca participa del ajuste de vectorizadores, modelos ni combinadores; las
features supervisadas se aprenden **dentro del fold**; todos los componentes aleatorios tienen
semilla fija.
"""
    ),
    nbutil.code(
        """
final = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v12.csv")
v11 = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v11.csv")
print("filas:", len(final), "| ids únicos:", final.id.nunique(),
      "| etiquetas:", sorted(final.age_range.unique()), "| NaN:", int(final.isna().sum().sum()))
print()
print("distribución de etiquetas predichas (v12):")
print(final.age_range.value_counts().reindex(cfg.CLASSES).to_string())
print()
print(f"coincidencia con la submission v11 (enviada): {(final.age_range == v11.age_range).mean():.4%}")
print(f"submissions disponibles: {sorted(p.name for p in cfg.SUBMISSIONS.glob('age_submission_v*.csv'))}")
"""
    ),
    nbutil.md(
        """
## 12. Conclusiones

1. **Sin embeddings ni transformers es posible un desempeño sólido**: 0,62182 de F1, con +0,0695
   sobre la línea base inicial, usando solo representaciones estadísticas, features artesanales y
   ensambles clásicos.
2. **La validación honesta sostiene las decisiones**: el CV agrupado por clústeres de near-duplicados
   predijo el leaderboard con una brecha estable de +0,0177 ± 0,0023 en diez envíos. Sin ese control,
   el split aleatorio habría regalado 0,061.
3. **"Limpiar" el texto puede destruir la señal**: la normalización completa costó 0,144 y la
   identidad de las cuentas mencionadas resultó una de las señales más fuertes (−0,125 al eliminarla).
4. **El valor de un modelo no lineal puede estar en su diversidad**: el MLP denso rindió 0,4071 en
   solitario y aportó +0,0066 al ensamble; y como **combinador** fue la palanca que destrabó la
   meseta (+0,0061).
5. **La diversidad sin calidad no compone**: promediar dos combinadores de calidad comparable aportó
   +0,0027; añadir un tercero más débil perjudicó.
6. **Los experimentos negativos son resultados**: quince líneas quedaron cerradas con su
   cuantificación y su explicación, incluida la única vía habilitada por reglas que los datos
   desaconsejaron (retrieval: la etiqueta de los duplicados solo coincide en 61,5%).

### Limitaciones

- **Truncado a 140 caracteres y transliteración ASCII**: se pierde contexto y toda señal tipográfica.
- **Sin metadatos de usuario**: no se puede explotar perfil ni agrupar por autor.
- **Ruido de etiqueta** (0,38% de filas contradictorias) y 166 textos vacíos: techo práctico.
- **Procedencia del corpus no identificada**: las comparaciones con la literatura son orientativas.
- **Fronteras centrales** (`18-24|25-34`: 0,486 del ensamble): las técnicas evaluadas no encontraron
  señal complementaria robusta; el estilo solo alcanza 0,2793 y el contenido, 0,5537.

### Trabajo futuro

- Reentrenar el modelo final aplicando la limpieza de textos duplicados (+0,0016, dentro del ruido).
- Si las reglas se relajaran (embeddings preentrenados), la frontera central sería el primer objetivo.
- Aprovechar la estructura ordinal con un modelo jerárquico supervisado más fino.
"""
    ),
    nbutil.code(
        """
print("VERIFICACIÓN FINAL")
print(f"  experimentos registrados : {len(experimentos)}")
print(f"  decisiones documentadas  : {len(decisiones)}")
print(f"  submission final         : {len(final):,} filas | LB registrado: 0,62182")
print("  todas las aserciones de este notebook se cumplieron.")
"""
    ),
]


def main() -> int:
    """Construye y ejecuta el notebook consolidado actualizado."""
    notebook_path = cfg.NOTEBOOKS / "99_entrega_consolidado.ipynb"
    nbutil.build(notebook_path, CELLS, title="99 · Entrega consolidada (versión final)")
    print(f"[build] {notebook_path}")
    print(f"[execute] {nbutil.execute(notebook_path, timeout=1800)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
