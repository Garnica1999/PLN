"""Fases 21-22 — Construye y ejecuta los notebooks 09 (pipeline final) y 99 (entrega consolidada).

``09_final_pipeline.ipynb`` ejecuta el modelo final de extremo a extremo con ``src/pipeline.py``,
genera la submission, la valida y verifica la reproducibilidad contra la v06 del leaderboard.

``99_entrega_consolidado.ipynb`` es el **artefacto calificado** (30% de la nota): narra el proyecto
completo con el formato OBSERVACIÓN → HIPÓTESIS → MÉTODO → RESULTADO → INTERPRETACIÓN → DECISIÓN,
reproduce las cifras clave y verifica el artefacto final.
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
print("raíz del proyecto:", ROOT)
"""

NB09 = [
    nbutil.md(
        """
# 09 · Pipeline final (Fases 21-22)

**Objetivo.** Ejecutar el **modelo final** de extremo a extremo con una sola llamada, generar la
submission y verificar dos cosas: (1) que el formato cumple la plantilla oficial y (2) que el
resultado es **reproducible** (mismas predicciones que la submission v06 enviada a Kaggle).

El modelo final es el ensamble de 10 fuentes con meta-modelo enriquecido, documentado en
`DECISIONS.md` (D-010, D-012, D-013, D-014, D-015) y con desempeño CV (leave-one-fold-out) de
**0,5851 ± 0,0042** y **0,60273** en el leaderboard público.

**Garantías metodológicas**: los vectorizadores, los modelos base y el meta-modelo se ajustan
**solo con train**; el test se usa exclusivamente para predecir. Todos los componentes aleatorios
llevan semilla fija.
"""
    ),
    nbutil.md("## 0. Configuración"),
    nbutil.code(SETUP),
    nbutil.md(
        """
## 1. Ejecución del pipeline

Se llama a `src.pipeline.run()`, que construye las representaciones, ajusta los 10 modelos base,
ajusta el meta-modelo con las matrices out-of-fold y escribe la submission validada.
Tiempo esperado: ~8-10 minutos en CPU (32 hilos).
"""
    ),
    nbutil.code(
        """
import time
from src import pipeline

started = time.time()
resumen = pipeline.run(output_path=cfg.SUBMISSIONS / "age_submission_final.csv", verbose=True)
print(f"\\ntiempo total: {round(time.time() - started, 1)}s")
"""
    ),
    nbutil.md("## 2. Validación del formato de la submission"),
    nbutil.code(
        """
print(json.dumps(resumen["checks"], indent=2, ensure_ascii=False))
assert all(resumen["checks"].values()), "alguna comprobación de formato falló"
print("\\nTodas las comprobaciones de formato pasan: columnas, orden, filas, ids, etiquetas, NaN.")
"""
    ),
    nbutil.md("## 3. Verificación de reproducibilidad contra la submission v06"),
    nbutil.code(
        """
final = pd.read_csv(cfg.SUBMISSIONS / "age_submission_final.csv")
v06 = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v06.csv")
coincidencia = float((final[cfg.TARGET] == v06[cfg.TARGET]).mean())
print(f"coincidencia con la submission v06 (la enviada a Kaggle): {coincidencia:.4%}")
print(f"filas: {len(final):,} | etiquetas distintas: {sorted(final[cfg.TARGET].unique())}")
assert coincidencia == 1.0, "el pipeline no reproduce la submission v06"
print("Reproducibilidad verificada: el pipeline regenera exactamente la submission v06.")
"""
    ),
    nbutil.md(
        """
## 4. Resumen del modelo final

| Componente | Detalle |
|---|---|
| Representación principal | TF-IDF `char_wb(3,6)` sobre texto **crudo** (sin normalizar) |
| Canales auxiliares | menciones (`@cuenta`), hashtags (`#etiqueta`), palabras (1-2) |
| Modelos base (10) | 2× LinearSVC(C=0.2) · LogReg(OvR) · SGD(log_loss) · ComplementNB · SVM+hashtags · Ridge ordinal · softmax ordinal · LogReg palabra · MLP(densas+SVD) |
| Meta-modelo | StandardScaler + LogisticRegression sobre 90 features (49 scores + 6 del MLP + 35 densas) |
| Validación | StratifiedGroupKFold 5 folds agrupado por clúster de near-duplicados + LOFO |
| CV (LOFO) | **0,5851 ± 0,0042** |
| Leaderboard público | **0,60273** |

**Archivos generados**: `submissions/age_submission_final.csv` (idéntico a la v06),
`experiments/submission_v06_resumen.json` y los artefactos de `reports/`.
"""
    ),
]

NB99 = [
    nbutil.md(
        """
# 99 · Entrega consolidada — Predicción de rango de edad a partir de tweets

**Competencia**: `tweet-age-2026-puj` (Kaggle, host `NLP_Course_PUJ`) · Curso de Procesamiento de
Lenguaje Natural · Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana.

**Tarea**: clasificar cada tweet en uno de **6 rangos de edad** (`13-17`, `18-24`, `25-34`,
`35-49`, `50-64`, `65-xx`) a partir únicamente de su texto.

**Métrica oficial**: F1-Score (leaderboard). **Resultado final: 0,60273** (línea base inicial
0,55235), con una validación interna de 0,5851.

Este notebook consolida el proyecto completo: datos, análisis, representaciones, modelos,
experimentos, errores y decisiones, cada una respaldada por evidencia cuantitativa. Todos los
experimentos están registrados en `experiments.csv` y todas las decisiones en `DECISIONS.md`.
"""
    ),
    nbutil.md("## 0. Configuración y artefactos"),
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

from src import config as cfg
from src.config import set_seed

set_seed()
pd.set_option("display.width", 175)
pd.set_option("display.max_columns", 60)

REPRODUCIR_PIPELINE = False  # poner en True para re-ejecutar el modelo final (~9 min en CPU)
experimentos = pd.read_csv(cfg.ROOT / "experiments.csv")
auditoria = json.loads((cfg.EXPERIMENTS / "audit_01_resumen.json").read_text(encoding="utf-8"))
print(f"experimentos registrados: {len(experimentos)}")
print(f"decisiones documentadas: {len([l for l in (cfg.ROOT / 'DECISIONS.md').read_text(encoding='utf-8').splitlines() if l.startswith('## D-')])}")
"""
    ),
    nbutil.md(
        """
## 1. Los datos

| Archivo | Filas | Columnas | Separador |
|---|---|---|---|
| `age_train.csv` | 169.336 | `id;text;age_range` | `;` |
| `age_test.csv` | 112.892 | `id;text` | `;` |
| `age_submission.csv` | 112.892 | `id,age_range` | `,` |

**OBSERVACIÓN.** Seis clases casi perfectamente balanceadas (16,0-17,0%), texto almacenado en
ancho fijo de 140 caracteres con relleno de espacios, **sin metadatos** de usuario ni temporales, y
un alfabeto de **96 caracteres con un único no-ASCII** (`¿`): el corpus fue transliterado a ASCII,
sin tildes ni emojis.

**INTERPRETACIÓN.** El corpus es un barrido de Twitter en español latinoamericano (~2016-2017,
re-indexado y equilibrado por clase) del que **no se pudo identificar la fuente pública exacta**
(`reports/tables/dataset_provenance.md`). El truncado a 140 y la transliteración ASCII son
artefactos del proceso de exportación, no propiedades de los autores.

**CONSECUENCIA.** Sin metadatos, toda la señal es textual; el relleno se elimina y las features de
emoji se descartan por varianza nula (D-009).
"""
    ),
    nbutil.code(
        """
resumen_datos = {
    "filas_train": auditoria["train_rows"], "filas_test": auditoria["test_rows"],
    "clases": auditoria["classes"], "vacios": auditoria["empty_texts"],
    "duplicados_exactos_test": auditoria["exact_duplicates"]["test_rows_seen_in_train_pct"],
    "cobertura_near_dup_test": auditoria["cluster_leakage"]["test_coverage_pct"],
    "pureza_clusteres": auditoria["cluster_leakage"]["purity_pct"],
    "ruido_etiqueta_pct": auditoria["label_conflicts"]["rows_affected_pct"],
}
pd.Series(resumen_datos, dtype=object).to_frame("valor")
"""
    ),
    nbutil.md(
        """
## 2. Metodología: el problema del leakage por duplicados

**OBSERVACIÓN.** El corpus no tiene identificador de autor, pero **1,2% de las filas de test
aparecen literalmente en train** y **4,3% comparten un clúster de near-duplicado** (Jaccard ≥ 0,8),
con 92,8% de clústeres de etiqueta única.

**HIPÓTESIS.** Si el conjunto de validación se particiona al azar, el modelo puede **memorizar**
duplicados en lugar de aprender estilo, inflando artificialmente la métrica.

**MÉTODO.** Se construyeron clústeres de near-duplicados sobre texto sin etiquetas (normalización +
Jaccard con *blocking* por tokens raros) y se compararon dos esquemas de validación: agrupado por
clúster (primario) y estratificado ingenuo (control). Además, un baseline de "memorización pura"
que predice la etiqueta mayoritaria del clúster.

**RESULTADO.** El esquema estratificado regala **0,061 de macro-F1** por memorización; el agrupado
lo reduce a cero por construcción.

**DECISIÓN.** Toda la validación del proyecto usa `StratifiedGroupKFold` agrupado por clúster
(D-003), y las comparaciones entre modelos se hacen con **leave-one-fold-out** sobre predicciones
out-of-fold.
"""
    ),
    nbutil.code(
        """
memorizacion = pd.DataFrame([
    {"esquema": "agrupado por clúster (usado)", **auditoria["memorization_group"]},
    {"esquema": "estratificado ingenuo (control)", **auditoria["memorization_stratified"]},
])
memorizacion.to_csv(cfg.TABLES / "entrega_memorizacion.csv", index=False)
memorizacion
"""
    ),
    nbutil.md(
        """
## 3. Análisis exploratorio y lingüístico

**OBSERVACIÓN.** La longitud del tweet y la riqueza léxica crecen de forma **monótona** con la
edad: 73 caracteres de media en `13-17` frente a 111 en `65-xx` (Kruskal-Wallis ε² = 0,126 para
`n_chars`). El léxico distintivo separa mundos sociales: `follow, justin, bieber, please` y
hashtags de fandoms en `13-17`; `venezuela, maduro, gobierno, pueblo` y `efemerides` en `65-xx`.

**HIPÓTESIS.** Existe señal de registro (formal/informal), de tema y de ortografía explotable con
representaciones clásicas.

**CONSECUENCIA.** Se priorizaron 12 hipótesis en `reports/hipotesis_gate_a.md`, cada una atada a un
experimento concreto.
"""
    ),
    nbutil.code(
        """
kruskal = pd.read_csv(cfg.TABLES / "eda_kruskal.csv").head(6)
terminos = pd.read_csv(cfg.TABLES / "top_terms_by_class.csv")
for clase in ["13-17", "65-xx"]:
    top = terminos[terminos["clase"] == clase].head(8)["termino"].tolist()
    print(f"términos distintivos {clase}: {', '.join(top)}")
kruskal[["feature", "epsilon_sq"]].round(4)
"""
    ),
    nbutil.md(
        """
## 4. Representaciones: la decisión central del proyecto

Se evaluaron **14 configuraciones** con modelo fijo (LinearSVC) y validación agrupada.

**OBSERVACIÓN.** Los n-gramas de carácter dominan (0,5408-0,5413) frente a los de palabra
(0,4941-0,5025), y la unión clásica palabra+carácter **diluye** (−0,0069).

**HALLAZGO PRINCIPAL.** Eliminar o enmascarar las **menciones** cuesta **−0,125** de macro-F1:
la identidad de las cuentas mencionadas es una de las señales más fuertes del corpus. Un canal
TF-IDF dedicado a los tokens de mención aporta **+0,0144**.

**INTERPRETACIÓN.** En este corpus la señal no está solo en lo que se dice, sino en **a quién se
menciona** y en la **forma** del tweet (URLs, puntuación, retuits). "Limpiar" el texto destruye
señal: tras la normalización completa, 7.240 tweets quedan vacíos.

**DECISIÓN (D-010).** Representación principal: `char_wb(3,6)` sobre texto **crudo** ∪ canal de
menciones; sin normalización.
"""
    ),
    nbutil.code(
        """
matriz = pd.read_csv(cfg.TABLES / "representaciones_comparativa.csv")
matriz.head(10)[["id", "representacion", "texto", "macro_f1", "features"]]
"""
    ),
    nbutil.md(
        """
## 5. Modelos, optimización y ensamble

**OBSERVACIÓN.** El SVM lineal con C = 0,2 es el mejor modelo individual (0,5628); la regresión
logística lo iguala con probabilidades; ComplementNB es débil (0,5370) pero muy diverso.

**MÉTODO.** Sobre esas piezas se construyó un ensamble por **stacking** con evaluación
leave-one-fold-out, añadiendo componentes guiados por el análisis de errores.

**RESULTADO.** Progresión de la validación interna: 0,5344 (baseline) → 0,5557 (menciones) →
0,5656 (stacking) → 0,5748 (+hashtags, +700k, +palabra) → 0,5784 (+densas en el meta) →
**0,5851** (+MLP denso como miembro diverso).

**DECISIÓN.** El modelo final es el ensamble de 10 fuentes con meta-modelo enriquecido (D-013,
D-014, D-015).
"""
    ),
    nbutil.code(
        """
progresion = pd.DataFrame([
    {"hito": "Baseline word+char (E-001)", "cv_macro_f1": 0.5344, "lb": 0.55235},
    {"hito": "char 3-6 + canal de menciones (E-008)", "cv_macro_f1": 0.5557, "lb": 0.57018},
    {"hito": "Stacking de 4 modelos (Fase 20)", "cv_macro_f1": 0.5656, "lb": 0.58081},
    {"hito": "Ensamble de 9 fuentes (Fase 20b)", "cv_macro_f1": 0.5748, "lb": 0.59178},
    {"hito": "Meta-modelo enriquecido (E-018)", "cv_macro_f1": 0.5784, "lb": 0.59518},
    {"hito": "Ensamble de 10 fuentes con MLP (E-020)", "cv_macro_f1": 0.5851, "lb": 0.60273},
])
progresion["brecha_lb_cv"] = (progresion["lb"] - progresion["cv_macro_f1"]).round(4)
progresion.to_csv(cfg.TABLES / "entrega_progresion.csv", index=False)
progresion
"""
    ),
    nbutil.md(
        """
## 6. Análisis de errores

**OBSERVACIÓN.** Las clases extremas se clasifican bien (`13-17` 0,638; `65-xx` 0,691) y las
intermedias concentran el error (`25-34` 0,449; `18-24` 0,476). La confusión es casi toda entre
**rangos contiguos** y el error crece al acortarse el tweet (52,5% en el quintil más corto).

**CONSECUENCIA.** Cada fenómeno generó una hipótesis medible: canal de menciones (+0,0144),
condicionamiento por longitud (+0,0036), canal de hashtags (+0,0019), señal ordinal y diversidad
del MLP (+0,0066).
"""
    ),
    nbutil.code(
        """
f1_clase = pd.Series(json.loads((cfg.EXPERIMENTS / "fase17_error_analysis.json").read_text())["f1_por_clase"])
cortes = pd.read_csv(cfg.TABLES / "fase17_error_por_cortes.csv")
print("F1 por clase (OOF del ensamble):")
print(f1_clase.sort_values(ascending=False).round(4).to_string())
print("\\nError por longitud:")
print(cortes[cortes["corte"] == "longitud"][["bucket", "error_rate"]].to_string(index=False))
"""
    ),
    nbutil.md(
        """
## 7. Modelo final y submission

**MÉTODO.** El pipeline final (`src/pipeline.py`) ajusta 10 modelos base sobre todo el train,
construye el meta-modelo con las matrices out-of-fold y predice el test. La submission se valida
contra la plantilla oficial antes de escribirse.

**RESULTADO.** Formato verificado (columnas, orden, 112.892 filas, ids, 6 etiquetas, sin NaN) y
**reproducibilidad exacta** contra la submission enviada (coincidencia del 100%).

Para re-ejecutar el pipeline completo basta poner `REPRODUCIR_PIPELINE = True` en la celda de
configuración (≈9 minutos en CPU).
"""
    ),
    nbutil.code(
        """
if REPRODUCIR_PIPELINE:
    from src import pipeline
    resumen_final = pipeline.run(output_path=cfg.SUBMISSIONS / "age_submission_final.csv",
                                 verbose=True)
else:
    resumen_final = json.loads((cfg.EXPERIMENTS / "submission_v06_resumen.json").read_text())

final = pd.read_csv(cfg.SUBMISSIONS / "age_submission_final.csv")
v06 = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v06.csv")
print("checks de formato:", resumen_final["checks"])
print(f"coincidencia con la submission enviada (v06): {(final[cfg.TARGET] == v06[cfg.TARGET]).mean():.4%}")
print("\\nDistribución de etiquetas predichas:")
print(final[cfg.TARGET].value_counts().reindex(cfg.CLASSES).to_string())
"""
    ),
    nbutil.md(
        """
## 8. Conclusiones

1. **Sin embeddings ni transformers es posible un desempeño sólido**: 0,60273 de F1 en el
   leaderboard con n-gramas de carácter, canales léxicos y ensambles clásicos.
2. **La validación honesta es la que sostiene las decisiones**: el CV agrupado por clústeres de
   near-duplicados predijo el leaderboard con una brecha estable de +0,0165 ± 0,0013 en seis
   mediciones independientes. Sin ese control, el split aleatorio habría regalado 0,061.
3. **"Limpiar" el texto puede destruir la señal**: la normalización completa costó 0,144 y la
   identidad de las cuentas mencionadas resultó ser una de las señales más fuertes (−0,125 al
   eliminarla). La evidencia manda sobre la intuición.
4. **El valor de un modelo no lineal puede estar en su diversidad, no en su exactitud**: el MLP
   rindió 0,4071 en solitario y aun así aportó +0,0066 al ensamble.
5. **Los experimentos negativos son resultados**: el ajuste de umbral por clase (+0,0006), la
   normalización agresiva (−0,144) y los modelos densos como clasificadores (−0,16) quedaron
   documentados con su cuantificación y su explicación.

### Limitaciones

- **Truncado a 140 caracteres** y transliteración ASCII: se pierde contexto y toda señal
  tipográfica (emojis, tildes), lo que limita el techo alcanzable.
- **Sin metadatos de usuario**: no se puede explotar información de perfil ni agrupar por autor;
  el control de leakage es por clústeres de contenido, no por identidad.
- **Ruido de etiqueta**: 0,38% de las filas tienen etiquetas contradictorias para el mismo texto,
  lo que fija un techo práctico.
- **Procedencia del corpus no identificada**: no se pudo confirmar la fuente pública, por lo que
  las comparaciones con la literatura son orientativas (PAN 2014/2016: 0,29-0,53 de exactitud).
- **Interpretación de reglas**: la prohibición de "modelos de vectorización" se interpretó como
  embeddings (D-001); si el criterio fuera más estricto, la tabla de ablación permite reconstruir
  un modelo sin POS ni lexicones.

### Trabajo futuro

- Explotar la estructura ordinal con un modelo jerárquico (clasificadores binarios encadenados
  entre rangos contiguos).
- Modelos específicos por tipo de tweet (cortos vs largos), dado que la longitud domina el error.
- Si las reglas lo permitieran, embeddings entrenados desde cero sobre este dominio.
"""
    ),
]


def main() -> int:
    """Construye y ejecuta los notebooks 09 y 99."""
    for path, cells, title in (
        (cfg.NOTEBOOKS / "09_final_pipeline.ipynb", NB09, "09 · Pipeline final"),
        (cfg.NOTEBOOKS / "99_entrega_consolidado.ipynb", NB99, "99 · Entrega consolidada"),
    ):
        nbutil.build(path, cells, title=title)
        print(f"[build] {path}")
        print(f"[execute] {nbutil.execute(path, timeout=3600)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
