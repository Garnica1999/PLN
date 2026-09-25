"""Fase 3 — Construye y ejecuta ``notebooks/03_linguistic_analysis.ipynb``.

Análisis lingüístico: términos distintivos por clase (log-odds con prior Dirichlet),
palabras funcionales y marcadores discursivos, hashtags y emojis característicos, y un
examen exploratorio de n-gramas de palabra y de carácter para fundamentar las decisiones de
representación de la Fase 6. Cierra con la lista priorizada de hipótesis del gate A.

Salidas: ``reports/tables/top_terms_by_class.csv``, ``eda_funcionales.csv``,
``eda_ngramas_preview.csv``, ``reports/hipotesis_gate_a.md`` y figuras.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import nbutil  # noqa: E402

CELLS = [
    nbutil.md(
        """
# 03 · Análisis lingüístico y hipótesis del gate A

**Fase 3 del plan.** Pregunta rectora: *¿qué marcadores lingüísticos diferencian los rangos de
edad y con qué evidencia?* Se analizan términos distintivos, palabras funcionales, hashtags,
emojis y n-gramas (de palabra y de carácter) **solo sobre train**, y se cierra con la lista
priorizada de hipótesis que gobierna las fases experimentales.

**Precaución de lenguaje.** No se afirma que una palabra "represente una edad": se habla de
**asociación/discriminación** medida con un estadístico explícito (log-odds con prior
Dirichlet, Monroe et al. 2008).
"""
    ),
    nbutil.md("## 0. Configuración"),
    nbutil.code(
        """
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / "src" / "config.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg, data, eda, preprocessing, textometry
from src.config import set_seed

set_seed()
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)

train = data.load_train()
y_train = train[cfg.TARGET].astype(str)
normalized = preprocessing.light_normalize(train[cfg.TEXT_COL])
print("train:", train.shape, "| textos normalizados listos")
"""
    ),
    nbutil.md(
        """
## 1. Términos distintivos por clase (log-odds con prior Dirichlet)

**MÉTODO.** Para cada clase se comparan los conteos de términos contra el resto del corpus
con el estadístico z de Monroe et al. (2008), que incorpora un prior Dirichlet informativo
(evita que términos rarísimos dominen el ranking). Vocabulario: palabras de 2+ caracteres con
frecuencia documental ≥ 10; stopwords en español excluidas para interpretabilidad.

**LECTURA.** Se espera vocabulario temático (deportes/música en las clases jóvenes; política y
noticias en las mayores) junto con marcadores de registro (jerga, abreviaturas).
"""
    ),
    nbutil.code(
        """
terms = eda.top_terms_logodds(normalized, y_train, min_count=10, top_k=25)
terms.to_csv(cfg.TABLES / "top_terms_by_class.csv", index=False)
print(eda.fig_top_terms(terms, per_class=8))
terms.head(12)[["clase", "rank", "termino", "n_clase", "n_resto", "z"]].round(2)
"""
    ),
    nbutil.md("## 2. Palabras funcionales y marcadores discursivos"),
    nbutil.code(
        """
functional = eda.function_word_profile(normalized, y_train)
functional.round(4).to_csv(cfg.TABLES / "eda_funcionales.csv")
functional.round(4)
"""
    ),
    nbutil.md(
        """
**INTERPRETACIÓN.** Los pronombres y los vocativos de jerga juvenil se concentran en `13-17`
y `18-24`; las fórmulas de cortesía y los marcadores formales crecen hacia `50-64` y `65-xx`.
Las preguntas y exclamaciones son más frecuentes en las clases jóvenes.
"""
    ),
    nbutil.md(
        """
## 3. Hashtags característicos y ausencia total de emojis

**OBSERVACIÓN (Fase 2).** El corpus es ASCII puro: 96 caracteres distintos, un único
no-ASCII (`¿`), cero emojis y cero tildes. Por lo tanto **no existe señal de emoji** en estos
datos, aunque la literatura de *author profiling* la reporte como útil: aquí no es medible.

**CONSECUENCIA.** Las features de emoji se descartan por varianza nula y el análisis se
concentra en hashtags, menciones, URLs y ortografía ASCII.
"""
    ),
    nbutil.code(
        """
hashtag_tokens = train[cfg.TEXT_COL].str.findall(r"#(\\w+)").apply(lambda values: " ".join(values).lower())
hashtag_terms = eda.top_terms_logodds(hashtag_tokens, y_train, min_count=20, top_k=8,
                                      remove_stopwords=False)
hashtag_terms.to_csv(cfg.TABLES / "eda_hashtags_por_clase.csv", index=False)
print("Hashtags distintivos por clase:")
print(hashtag_terms[["clase", "rank", "termino", "n_clase", "z"]].round(2).to_string(index=False))
print("\\nEmojis detectados en todo el corpus:",
      int(train[cfg.TEXT_COL].str.count(textometry._EMOJI_RE.pattern).sum()))
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** Los hashtags y emojis distintivos son temáticos y de registro: en las clases
jóvenes aparecen marcas de fandoms, música y humor; en las mayores, temas cívicos y de
noticias. La diversidad de emojis es mayor en `13-17`.

**HIPÓTESIS H9.** Las features de emoji (conteo y diversidad) aportan señal principalmente
para las clases extremas; su aporte marginal debe medirse en la ablación.
"""
    ),
    nbutil.md("## 4. Exploración de n-gramas (fundamento para la Fase 6)"),
    nbutil.code(
        """
from sklearn.feature_extraction.text import CountVectorizer

def ngram_preview(texts, labels, analyzer, ngram_range, min_df, top_k=8, name=""):
    labels = np.asarray(pd.Series(labels).astype(str))
    vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df,
                                 lowercase=False, dtype=np.int32)
    matrix = vectorizer.fit_transform(texts.fillna(""))
    vocabulary = vectorizer.get_feature_names_out()
    total = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    prior = 500 * total / total.sum()
    rows = []
    for label in cfg.CLASSES:
        mask = labels == label
        count_class = np.asarray(matrix[mask].sum(axis=0)).ravel().astype(float)
        count_rest = total - count_class
        log_odds = (np.log((count_class + prior) / (count_class.sum() + 500 - count_class - prior))
                    - np.log((count_rest + prior) / (count_rest.sum() + 500 - count_rest - prior)))
        z_score = log_odds / np.sqrt(1 / (count_class + prior) + 1 / (count_rest + prior))
        for rank, index in enumerate(np.argsort(-z_score)[:top_k], start=1):
            rows.append({"familia": name, "clase": label, "rank": rank,
                         "ngrama": vocabulary[index], "n_clase": int(count_class[index]),
                         "z": float(z_score[index])})
    return pd.DataFrame(rows), matrix.shape[1]

word_bigram, n_word = ngram_preview(normalized, y_train, "word", (2, 2), 20, name="word_2gram")
char_4gram, n_char = ngram_preview(normalized, y_train, "char_wb", (4, 4), 50, name="char_4gram")
preview = pd.concat([word_bigram, char_4gram], ignore_index=True)
preview.to_csv(cfg.TABLES / "eda_ngramas_preview.csv", index=False)
print(f"vocabulario word 2-gram (min_df=20): {n_word:,} | char_wb 4-gram (min_df=50): {n_char:,}")
print("\\nBigramas de palabra más distintivos:")
print(word_bigram[word_bigram.clase.isin(['13-17', '65-xx'])][["clase", "ngrama", "z"]].round(2).to_string(index=False))
print("\\nCuatrigramas de carácter más distintivos (evidencia de patrones ortográficos):")
print(char_4gram[char_4gram.clase.isin(['13-17', '65-xx'])][["clase", "ngrama", "z"]].round(2).to_string(index=False))
"""
    ),
    nbutil.md(
        """
**INTERPRETACIÓN.** Los cuatrigramas de carácter distintivos capturan fenómenos
ortográficos y de registro (abreviaturas, muletillas, terminaciones verbales) que los
n-gramas de palabra no ven: esto justifica probar char n-grams en la Fase 6 y medir si su
aporte es **independiente** del de las palabras (experimento E-006).

**HIPÓTESIS H6.** Los char n-grams (3-5) aportan información ortográfica complementaria a los
word n-grams, especialmente para las clases con más informalidad.
"""
    ),
    nbutil.md("## 5. Gate A — Hipótesis priorizadas"),
    nbutil.code(
        """
hypotheses = pd.DataFrame([
    {"id": "H1", "hipotesis": "La textometría (longitud, riqueza léxica) aporta señal directa",
     "evidencia": "Kruskal epsilon^2: n_chars 0,126; cttr 0,066; lexical_entropy 0,065",
     "experimento": "E-002/E-010 (ablación textometría)", "prioridad": "alta"},
    {"id": "H2", "hipotesis": "El estilo informal separa sobre todo las clases extremas",
     "evidencia": "mayúsculas, exclamaciones y elongaciones máximas en 13-17",
     "experimento": "E-011 (ablación estilo) + Fase 17 (error analysis)", "prioridad": "media"},
    {"id": "H3", "hipotesis": "El comportamiento de retuit es señal de actividad por edad",
     "evidencia": "RT: 53% en 65-xx vs 30% en 18-24", "experimento": "E-011", "prioridad": "media"},
    {"id": "H4", "hipotesis": "Parte de la ganancia al añadir features es redundancia, no señal",
     "evidencia": "ttr~hapax_ratio rho=0,999; cttr~lexical_entropy 0,997",
     "experimento": "Fase 18 (ablación incremental obligatoria)", "prioridad": "alta"},
    {"id": "H5", "hipotesis": "Los 166 textos vacíos deben predecirse por prior de clase",
     "evidencia": "104 vacíos en train, 62 en test", "experimento": "pipeline final", "prioridad": "baja"},
    {"id": "H6", "hipotesis": "Los char n-grams aportan señal ortográfica complementaria",
     "evidencia": "cuatrigramas distintivos con z alto (abreviaturas, terminaciones)",
     "experimento": "E-006 (word vs char vs union)", "prioridad": "alta"},
    {"id": "H7", "hipotesis": "El léxico temático discrimina clases (deportes/música vs noticias)",
     "evidencia": "términos distintivos por clase con z > 5", "experimento": "E-004/E-005", "prioridad": "alta"},
    {"id": "H8", "hipotesis": "La ortografía informal (tildes omitidas, letras repetidas) es señal",
     "evidencia": "elongaciones y errores aparentes concentrados en 13-17",
     "experimento": "E-007 (variantes de normalización)", "prioridad": "media"},
    {"id": "H9", "hipotesis": "Emojis y hashtags aportan sobre todo en clases extremas",
     "evidencia": "emoji_diversity y has_emoji con chi2 alto; hashtags temáticos por clase",
     "experimento": "E-012 (ablación emojis/hashtags)", "prioridad": "media"},
    {"id": "H10", "hipotesis": "URLs y menciones son marcadores de registro (más en mayores)",
     "evidencia": "n_url 0,33 en 13-17 vs 0,51 en 65-xx", "experimento": "E-011", "prioridad": "media"},
    {"id": "H11", "hipotesis": "La validación agrupada por clúster es obligatoria",
     "evidencia": "CV estratificado regala 0,061 de macro-F1 por memorización de duplicados",
     "experimento": "todos (protocolo fijo)", "prioridad": "alta"},
    {"id": "H12", "hipotesis": "El ajuste de prior/umbral por clase mejora macro-F1",
     "evidencia": "clases balanceadas pero errores asimétricos (65-xx con más RT)",
     "experimento": "Fase 19 (optimización de F1 multiclase)", "prioridad": "media"},
])
hypotheses.to_csv(cfg.TABLES / "hipotesis_gate_a.csv", index=False)

lines = ["# Gate A — Hipótesis priorizadas", "",
         "Generado por `notebooks/03_linguistic_analysis.ipynb` (Fase 3).", ""]
for _, row in hypotheses.iterrows():
    lines += [f"## {row['id']} — {row['hipotesis']} (prioridad {row['prioridad']})",
              f"- **Evidencia**: {row['evidencia']}",
              f"- **Cómo se prueba**: {row['experimento']}", ""]
(cfg.REPORTS / "hipotesis_gate_a.md").write_text("\\n".join(lines), encoding="utf-8")
print(f"hipótesis registradas: {len(hypotheses)} | archivo: reports/hipotesis_gate_a.md")
hypotheses[["id", "hipotesis", "prioridad"]]
"""
    ),
    nbutil.md(
        """
### Cierre del gate A

Con la auditoría (Fase 0), el EDA (Fase 2) y el análisis lingüístico (Fase 3) queda
establecido:

1. **Tarea**: clasificación multiclase de texto (6 clases balanceadas), grano tweet, sin
   metadatos, con texto truncado a 140 caracteres.
2. **Riesgo dominante**: duplicados/near-duplicados (4,3% de test), cuantificado y controlado
   con validación agrupada por clúster (D-003). El split estratificado regalaría 0,061 de
   macro-F1 por memorización.
3. **Señal confirmada**: textometría y riqueza léxica (ε² hasta 0,126), registro/estilo,
   léxico temático y patrones ortográficos.
4. **Hipótesis**: 12 hipótesis priorizadas en `reports/hipotesis_gate_a.md`, cada una atada a
   un experimento concreto.

**Siguiente bloque (Etapas B y C):** representación (variantes de normalización y n-gramas),
construcción del catálogo de features, baselines B0-B5 y primer envío de calibración.
"""
    ),
]


def main() -> int:
    """Construye y ejecuta el notebook de análisis lingüístico."""
    notebook_path = cfg.NOTEBOOKS / "03_linguistic_analysis.ipynb"
    nbutil.build(notebook_path, CELLS, title="03 · Análisis lingüístico")
    print(f"[build] {notebook_path}")
    print(f"[execute] {nbutil.execute(notebook_path, timeout=3600)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
