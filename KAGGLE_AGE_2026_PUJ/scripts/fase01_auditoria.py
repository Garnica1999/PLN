"""Fase 0 — Construye y ejecuta ``notebooks/01_data_audit.ipynb``.

Uso::

    /home/carlos/venv/bin/python scripts/fase01_auditoria.py

El notebook resultante queda ejecutado (con salidas embebidas) y produce:
``reports/data_dictionary.csv``, figuras de auditoría, ``features/dup_clusters.parquet``
y ``experiments/audit_01_resumen.json``.
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
# 01 · Auditoría de datos y leakage — TweetAge_2026_PUJ

**Fase 0 del plan.** Objetivo: caracterizar los datos reales, verificar la integridad de la
descarga, documentar el esquema, cuantificar el ruido de etiqueta y **medir la exposición a
leakage por duplicados**, que es el riesgo dominante de este corpus: no existe identificador
de autor, por lo que no es posible agrupar por usuario y el control debe hacerse por
clústeres de near-duplicados (decisión D-003).

**Reglas respetadas.** El archivo de test no tiene etiquetas y no se usa para ajustar nada:
los clústeres se construyen sobre texto (sin etiquetas) y toda métrica de desempeño se
calcula únicamente dentro de train.

**Salidas.** `reports/data_dictionary.csv`, `reports/figures/audit_*.png`,
`features/dup_clusters.parquet`, `experiments/audit_01_resumen.json`.
"""
    ),
    nbutil.md("## 0. Configuración y entorno"),
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

from src import audit, config as cfg, cv, data, submission
from src.config import set_seed

set_seed()
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)
print("ROOT:", ROOT)
print("python:", sys.version.split()[0], "| pandas:", pd.__version__, "| numpy:", np.__version__)
print("GPU requerida en esta fase: NO (pipeline CPU-first)")
"""
    ),
    nbutil.md(
        """
## 1. Descarga y verificación de integridad

**OBSERVACIÓN esperada:** tres archivos (`age_train.csv`, `age_test.csv`,
`age_submission.csv`) descargados de la competencia privada con autenticación Bearer.

**MÉTODO:** `src.data.download_raw()` (streaming + SHA-256) y `src.data.verify()`, que
revisa el número de campos por registro con `csv.reader` (detecta separadores embebidos y
filas truncadas), unicidad de `id`, coincidencia de ids test<->submission y conteo de textos
vacíos.
"""
    ),
    nbutil.code(
        """
info = data.download_raw()
print("ZIP sha256 :", info["zip"])
print("ZIP bytes  :", f"{int(info['size']):,}", "| cacheado:", info.get("cached"))
print("Contenido  :", info["files"])

checks = data.verify()
print(json.dumps(checks, indent=2, ensure_ascii=False))
print("\\nSHA256SUMS.txt:")
print(cfg.SHA256_FILE.read_text(encoding="utf-8"))

assert checks["integrity_train"]["bad_records"] == 0
assert checks["integrity_test"]["bad_records"] == 0
assert checks["integrity_sample"]["bad_records"] == 0
assert checks["test_ids_equal_sample_ids"] is True
print("Integridad estructural verificada: 0 registros malformados en los tres archivos.")
"""
    ),
    nbutil.md(
        """
## 2. Esquema y diccionario de datos

**INTERPRETACIÓN.** El esquema es mínimo: `id` (identificador), `text` (única señal
disponible) y `age_range` (objetivo, 6 clases). No hay metadatos de usuario ni temporales,
lo que descarta por completo la línea de features de perfil y concentra el modelado en
representación del texto y estilo.
"""
    ),
    nbutil.code(
        """
train = data.load_train()
test = data.load_test()
sample = data.load_sample_submission()

for name, frame in (("train", train), ("test", test), ("submission", sample)):
    print("=" * 78)
    print(json.dumps(audit.basic_profile(frame, name), indent=2, ensure_ascii=False)[:900])

dictionary = audit.build_data_dictionary(train, test, sample)
dictionary.to_csv(cfg.REPORTS / "data_dictionary.csv", index=False)
print("\\nreports/data_dictionary.csv escrito:", len(dictionary), "filas")
dictionary
"""
    ),
    nbutil.md(
        """
## 3. Perfil de texto: padding de ancho fijo (D-005)

**OBSERVACIÓN.** El corpus almacena el texto en un campo de ancho fijo: la mayoría de las
filas miden exactamente 140 caracteres en bruto, pero la longitud real (tras eliminar el
relleno de espacios) es bastante menor.

**INTERPRETACIÓN.** El relleno es un artefacto de exportación (límite histórico de Twitter),
no una señal autoral; el truncado a 140 caracteres sí es una limitación real del corpus.

**CONSECUENCIA.** `src.data.clean_text` aplica `strip`; el truncado se registra como
hipótesis de error analysis (los tweets largos pierden contexto).
"""
    ),
    nbutil.code(
        """
raw_train = (pd.read_csv(cfg.TRAIN_FILE, sep=";", encoding="utf-8-sig", usecols=["text"])["text"]
             .fillna("").astype(str))
raw_test = (pd.read_csv(cfg.TEST_FILE, sep=";", encoding="utf-8-sig", usecols=["text"])["text"]
            .fillna("").astype(str))

lengths = pd.DataFrame([audit.length_profile(train, "train"), audit.length_profile(test, "test")])
lengths.to_csv(cfg.TABLES / "audit_longitudes.csv", index=False)
print(audit.fig_padding_effect(raw_train.str.len(), train["text"].str.len()))
lengths.T
"""
    ),
    nbutil.md(
        """
## 4. Distribución de clases

**OBSERVACIÓN.** Las seis clases están casi perfectamente balanceadas (16,0%–17,0%).

**INTERPRETACIÓN.** El corpus fue construido balanceado por diseño, típico de los esquemas
de *author profiling* (rangos tipo PAN). Un clasificador trivial que prediga la clase
mayoritaria obtendría ≈17% de exactitud.

**CONSECUENCIA.** La métrica macro-F1 y la exactitud serán cercanas entre sí; no se
necesitan técnicas de rebalanceo, y cualquier mejora debe venir de la representación.
"""
    ),
    nbutil.code(
        """
counts = train[cfg.TARGET].value_counts().reindex(cfg.CLASSES)
summary = pd.DataFrame({"tweets": counts, "pct": (100 * counts / len(train)).round(3)})
print(audit.fig_class_distribution(train))
print("Clase mayoritaria:", counts.idxmax(), f"({100*counts.max()/len(train):.2f}%)")
print("Desbalanceo máximo relativo: %.3f" % (counts.max() / counts.min()))
summary
"""
    ),
    nbutil.md(
        """
## 5. Verificación de leakage por `id`

**OBSERVACIÓN.** Si los identificadores se hubieran asignado tras ordenar por clase, `id`
sería un proxy del objetivo y bastaría con explotarlo.

**MÉTODO.** Probabilidad de cada clase por decil de `id` (test de estabilidad).

**RESULTADO.** La desviación máxima respecto a la línea base (1/6) es de ~1 punto
porcentual, sin tendencia monótona.

**DECISIÓN.** Se descarta `id` como feature. Ningún modelo lo usará.
"""
    ),
    nbutil.code(
        """
id_table = audit.id_proxy_check(train, n_bins=10)
deviation = (id_table - 1 / len(cfg.CLASSES)).abs().max()
print(audit.fig_id_proxy(id_table))
print("Desviación máxima por clase (pp):", (deviation * 100).round(2).to_dict())
print("Desviación global máxima (pp): %.2f" % (deviation.max() * 100))
id_table
"""
    ),
    nbutil.md(
        """
## 6. Ruido de etiqueta

**OBSERVACIÓN.** Hay textos idénticos (normalizados) anotados con etiquetas distintas.

**INTERPRETACIÓN.** Ruido irreducible: fija un techo práctico de desempeño y explica parte
de los errores que ningún modelo podrá corregir. Se conserva (eliminar filas sesgaría la
distribución), pero se documenta.
"""
    ),
    nbutil.code(
        """
conflicts = audit.label_conflict_stats(train)
print(json.dumps(conflicts, indent=2, ensure_ascii=False))
"""
    ),
    nbutil.md(
        """
## 7. Duplicados y near-duplicados

**OBSERVACIÓN.** En Twitter abundan los retuits y las variantes del mismo texto (mismo
cuerpo con distinto enlace o menciones). Si un texto de test aparece en train, el modelo
puede memorizarlo en lugar de aprender estilo.

**MÉTODO.** Normalización (minúsculas, sin URLs, sin menciones, sin prefijo `RT`, sin
puntuación) y agrupamiento por texto normalizado idéntico **o** por similitud de **Jaccard
≥ 0,8** entre conjuntos de tokens, con *blocking* por los tokens más raros para evitar el
costo cuadrático. Se agrupa sobre train+test concatenados, sin usar etiquetas.

**NOTA METODOLÓGICA (corrección documentada).** La primera versión agrupaba por la firma de
los primeros 60 caracteres y produjo clústeres espurios: un mismo clúster llegó a reunir 190
tweets con las 6 etiquetas mezcladas, unidos solo por compartir una apertura genérica. Eso
hundía la pureza medida (67%) y hacía ininterpretable la cuantificación del leakage. El
método actual solo une textos realmente similares (comparaciones: ~4,8 millones, presupuesto
no agotado).

**RESULTADO.** Ver tabla y figura: ~1,2% de las filas de test son duplicado exacto de train y
~4,3% comparten clúster de near-duplicado, con ~93% de clústeres de etiqueta única.

**CONSECUENCIA.** Validación agrupada por clúster (D-003) y cuantificación explícita del
puntaje "memorizable" en la sección 8.
"""
    ),
    nbutil.code(
        """
exact = audit.duplicate_stats(train, test)
print(json.dumps(exact, indent=2, ensure_ascii=False))

all_texts = pd.concat([train[cfg.TEXT_COL], test[cfg.TEXT_COL]], ignore_index=True)
cluster_stats: dict = {}
clusters = cv.build_dup_clusters(all_texts, stats=cluster_stats)
clusters.to_parquet(cfg.FEATURES / "dup_clusters.parquet", index=False)
print("Diagnóstico de agrupamiento:", json.dumps(cluster_stats, ensure_ascii=False))

leakage = cv.cluster_leakage_report(clusters, n_train=len(train), y_train=train[cfg.TARGET])
print(json.dumps(leakage, indent=2, ensure_ascii=False))
print(audit.fig_dup_exposure(exact["test_rows_seen_in_train_pct"],
                             leakage["test_coverage_pct"],
                             leakage.get("purity_pct", float("nan"))))

sizes = clusters["cluster"].value_counts()
print("Clústeres:", sizes.size, "| tamaño máximo:", int(sizes.max()),
      "| %% train en clústeres de tamaño>1: %.2f" %  # noqa: UP031
      (100 * clusters["cluster"].iloc[:len(train)].duplicated(keep=False).mean()))

train_cluster_ids = clusters["cluster"].to_numpy()[:len(train)]
purity_frame = pd.DataFrame({"cluster": train_cluster_ids, "y": train[cfg.TARGET].astype(str)})
mode_pred = purity_frame.groupby("cluster")["y"].transform(lambda values: values.mode().iat[0])
print("Pureza a nivel de fila (train, moda del clúster): %.2f%%" % (100 * mode_pred.eq(purity_frame["y"]).mean()))
audit.most_common_labels_per_cluster(clusters, n_train=len(train), y_train=train[cfg.TARGET])
"""
    ),
    nbutil.md(
        """
## 8. ¿Cuánto puntaje es memorizable? (cuantificación del leakage)

**PREGUNTA.** ¿Cuánto del desempeño esperado puede obtenerse memorizando duplicados en
lugar de generalizar estilo?

**MÉTODO.** Dentro de train, con validación cruzada: para cada fila de validación cuyo
clúster tiene miembros en el train del fold, se predice la etiqueta mayoritaria del clúster;
el resto recibe un centinela que nunca acierta. Se compara el esquema **agrupado por
clúster** (primario) contra el **estratificado ingenuo** (control).

**INTERPRETACIÓN.** La diferencia entre ambos esquemas mide el optimismo del split aleatorio
y justifica la validación agrupada en todas las fases siguientes.
"""
    ),
    nbutil.code(
        """
cluster_ids = clusters["cluster"].to_numpy()
y_train = train[cfg.TARGET].astype(str).to_numpy()

folds_group = cv.make_folds(y_train, groups=cluster_ids[:len(train)], scheme="group")
folds_strat = cv.make_folds(y_train, groups=None, scheme="stratified")

mem_group = cv.duplicate_memorization_baseline(cluster_ids[:len(train)], y_train, folds_group)
mem_strat = cv.duplicate_memorization_baseline(cluster_ids[:len(train)], y_train, folds_strat)

comparison = pd.DataFrame({
    "esquema": ["agrupado por clúster (primario)", "estratificado ingenuo (control)"],
    "cobertura_%": [mem_group["coverage_pct"], mem_strat["coverage_pct"]],
    "acierto_en_cubiertas": [mem_group["accuracy_covered"], mem_strat["accuracy_covered"]],
    "macro_f1_memorizacion": [mem_group["macro_f1_memorization_only_mean"],
                              mem_strat["macro_f1_memorization_only_mean"]],
    "macro_f1_std": [mem_group["macro_f1_memorization_only_std"],
                     mem_strat["macro_f1_memorization_only_std"]],
})
comparison.to_csv(cfg.TABLES / "audit_memorizacion.csv", index=False)
print(json.dumps({"group": mem_group, "stratified": mem_strat}, indent=2, ensure_ascii=False))
comparison
"""
    ),
    nbutil.md(
        """
## 9. Validación de la plantilla de submission

**MÉTODO.** `src.submission.validate_submission` comprueba columnas, orden, número de
filas, ids, etiquetas válidas, NaN y duplicados. Se valida la plantilla oficial y una
submission ficticia (todo `13-17`) para demostrar que el validador acepta lo correcto y
rechaza lo incorrecto.

**RESULTADO.** La plantilla oficial pasa todas las comprobaciones; el validador está listo
para usarse en la Fase 22 sin cambios.
"""
    ),
    nbutil.code(
        """
official = submission.validate_submission(sample.copy(), test[cfg.ID_COL], sample)
print("plantilla oficial:", official)

dummy = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL], cfg.TARGET: "13-17"})
print("submission ficticia válida:", submission.validate_submission(dummy, test[cfg.ID_COL], sample))

bad = dummy.copy()
bad.loc[0, cfg.TARGET] = "99-99"
try:
    submission.validate_submission(bad, test[cfg.ID_COL], sample)
    print("ERROR: el validador no detectó la etiqueta inválida")
except submission.SubmissionError as exc:
    print("rechazo correcto de etiqueta inválida:", str(exc)[:120])
"""
    ),
    nbutil.md(
        """
## 10. Gate A — Resumen de la auditoría

Se consolida la evidencia en `experiments/audit_01_resumen.json` y se listan los hechos que
condicionan el resto del proyecto.
"""
    ),
    nbutil.code(
        """
resumen = {
    "competition": cfg.COMPETITION,
    "zip_sha256": info["zip"],
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "classes": {str(key): int(value) for key, value in counts.items()},
    "empty_texts": {"train": int((train[cfg.TEXT_COL] == "").sum()),
                    "test": int((test[cfg.TEXT_COL] == "").sum())},
    "padding": lengths.set_index("file").to_dict(orient="index"),
    "id_leakage_max_deviation_pp": round(float(deviation.max() * 100), 3),
    "label_conflicts": conflicts,
    "exact_duplicates": exact,
    "cluster_leakage": leakage,
    "memorization_group": mem_group,
    "memorization_stratified": mem_strat,
    "submission_format_ok": official,
}
(cfg.EXPERIMENTS / "audit_01_resumen.json").write_text(
    json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: resumen[k] for k in
                  ("train_rows", "test_rows", "id_leakage_max_deviation_pp", "label_conflicts",
                   "memorization_group", "memorization_stratified")}, indent=2, ensure_ascii=False))
print("\\nResumen guardado en experiments/audit_01_resumen.json")
"""
    ),
    nbutil.md(
        """
### Conclusiones del gate A

| # | Observación | Interpretación | Hipótesis | Consecuencia |
|---|---|---|---|---|
| A1 | 6 clases balanceadas (16,0–17,0%) | corpus balanceado por diseño (esquema tipo PAN) | el desbalanceo no es un problema; la señal está en el estilo | sin rebalanceo; macro-F1 como métrica primaria |
| A2 | Texto truncado/padded a 140 chars; mediana real 93 | límite histórico de Twitter, artefacto de exportación | los tweets largos truncados pierden señal y serán más difíciles | `strip` del padding; hipótesis de error analysis por longitud |
| A3 | `id` no es proxy de clase (≤1 pp) | asignación aleatoria de ids | — | `id` descartado como feature |
| A4 | 1,2% de test aparece literal en train; 4,3% comparte clúster de near-duplicado (Jaccard ≥ 0,8), con 93% de clústeres puros | leakage por duplicados (retuits y variantes) | memorizar duplicados infla el score si el CV no los agrupa | CV agrupado por clúster (D-003) y cuantificación del efecto |
| A5 | 0,38% de filas con etiqueta contradictoria | ruido irreducible del corpus | techo práctico de desempeño | se conserva y se documenta |
| A6 | 166 textos vacíos (104 train, 62 test) | tweets sin contenido (solo enlace o vacío) | sin señal textual: predicción por prior | manejo explícito en el pipeline |
| A7 | Sin metadatos de usuario | no hay author profiling multiusuario | la señal es 100% textual | foco en representación léxica, estilométrica y de carácter |

**Próxima fase:** Fase 1 (estado del arte en dos corrientes) y Fase 2-3 (EDA y análisis
lingüístico) con la lista priorizada de hipótesis.
"""
    ),
]


def main() -> int:
    """Construye y ejecuta el notebook de auditoría."""
    notebook_path = cfg.NOTEBOOKS / "01_data_audit.ipynb"
    nbutil.build(notebook_path, CELLS, title="01 · Auditoría de datos y leakage")
    print(f"[build] {notebook_path}")
    result = nbutil.execute(notebook_path, timeout=3600)
    print(f"[execute] {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
