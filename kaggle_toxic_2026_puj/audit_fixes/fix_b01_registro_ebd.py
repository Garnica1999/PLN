"""Fase de correcciones B-01/B-02: repara el bloque EBD del registro.

Reemplaza las 6 filas malformadas (9 campos, id KAGGLE_TOXIC_2026_PUJ) por 6
filas validas EBD-01..06 reconstruidas desde los artefactos de la fase v5
(reports/embedding_downstream_v5.json, reports/embedding_intrinsic_eval.md,
reports/embedding_p2_verdict.json, reports/embedding_downstream_results.csv).
La reconstruccion queda marcada en `notes` (politica de no inventar historia).
"""
from __future__ import annotations

import csv
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")
CSV_PATH = PROJ / "experiments.csv"

COMUN = {
    "competition": "KAGGLE_TOXIC_2026_PUJ",
    "preprocessing": "EMB-NORMAL (identico a P1, verificado token a token sobre los 2885 textos de TOXIC)",
    "date": "2026-09-25",
    "script": "experiments/embedding_02_fasttext_baseline_v5.py",
}
NOTA = ("Artefactos: reports/embedding_intrinsic_eval.md, reports/embedding_downstream_results.csv, "
        "reports/embedding_downstream_v5.json, reports/embedding_p2_verdict.json. "
        "Fila reconstruida por la auditoria forense (B-01): la original estaba corrupta; "
        "el contenido se deriva de esos artefactos.")

FILAS = [
    {
        "experiment_id": "EBD-01",
        "question": "Se puede entrenar un embedding FastText desde cero con el corpus externo no etiquetado y que cobertura tiene sobre el vocabulario de TOXIC?",
        "hypothesis": "El corpus Wikipedia 10M permite entrenar skip-gram con subpalabras y cubrir la mayor parte de los tipos de TOXIC",
        "feature_groups": "ninguno (entrenamiento de representacion)",
        "representation": "FastText skipgram dim=200, subpalabras 3-6, corpus Wikipedia ES 10M tokens",
        "model": "entrenamiento no supervisado desde cero (fastText C++ CLI)",
        "hyperparameters": "dim=200, ws=5, minCount=3, neg=10, epoch=5, thread=16, seed=42, minn=3, maxn=6, bucket=200000",
        "validation": "vocabulario 100098; 8555 tipos TOXIC consultados; sha256 del .vec registrado",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": "0 s (modelo reutilizado del cache con sha256 verificado)",
        "dimension": "200",
        "result": "FASTTEXT-BASELINE entrenado (sha256 e54a363da986aeac...); n_tipos_con_vector 8555 de 8555 consultados; vocab 100098",
        "interpretation": "La representacion se aprende integramente con datos no etiquetados autorizados (D-030); su utilidad se mide en EBD-04/EBD-05",
        "decision": "GO a los controles (semilla y word2vec) y a la evaluacion intrinseca",
        "notes": NOTA,
    },
    {
        "experiment_id": "EBD-02",
        "question": "Cuanto depende la representacion de la semilla de inicializacion?",
        "hypothesis": "La geometria relativa (vecinos) es estable aunque las direcciones absolutas cambien con la semilla",
        "feature_groups": "ninguno",
        "representation": "FastText skipgram dim=200, subpalabras 3-6, corpus 10M",
        "model": "control de semilla (seed 7) del mismo entrenamiento",
        "hyperparameters": "identicos a EBD-01 salvo seed=7",
        "validation": "coseno centrado medio y solapamiento de vecinos top-10 sobre 12 palabras",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": "0 s (cache con sha256 afde390f208b1f51...)",
        "dimension": "200",
        "result": "coseno centrado medio entre semillas -0.034; solapamiento medio de vecinos top-10 0.792",
        "interpretation": "Las direcciones absolutas no son comparables entre semillas pero la estructura de vecindad si; ninguna conclusion downstream depende del azimuth",
        "decision": "Registrado como control; la fase continua con seed=42",
        "notes": NOTA,
    },
    {
        "experiment_id": "EBD-03",
        "question": "Que aportan las subpalabras frente al control Word2Vec sin subpalabras?",
        "hypothesis": "FastText compone vectores para variantes OOV (callate, cllate...) donde Word2Vec no puede",
        "feature_groups": "ninguno",
        "representation": "Control Word2Vec (skipgram minn=0, maxn=0) vs FastText subpalabras 3-6",
        "model": "control estructural (misma arquitectura sin subpalabras)",
        "hyperparameters": "identicos a EBD-01 con minn=0, maxn=0",
        "validation": "tabla de variantes ortograficas OOV del vocabulario TOXIC",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": "0 s (cache con sha256 52f94d37e18cef72...)",
        "dimension": "200",
        "result": "El control Word2Vec no puede componer subpalabras por construccion; la tabla de variantes OOV se corrigio en v5.1 (D-042, correccion de vocabulario)",
        "interpretation": "La comparacion aislada de subpalabras es debil a 10M tokens (senal sublexica limitada); su efecto se mide en la evaluacion intrinseca y en v5.1",
        "decision": "Control completado; sin adopcion",
        "notes": NOTA,
    },
    {
        "experiment_id": "EBD-04",
        "question": "La representacion captura estructura semantica util para el vocabulario de la competencia?",
        "hypothesis": "Los vecinos por coseno centrado seran coherentes para terminos frecuentes del dominio",
        "feature_groups": "ninguno",
        "representation": "FastText 10M (baseline) y control Word2Vec",
        "model": "evaluacion intrinseca (vecinos mas cercanos por coseno centrado)",
        "hyperparameters": "no aplica",
        "validation": "tabla de vecinos para 12 palabras + variantes ortograficas; sin labels externos",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": "", "dimension": "200",
        "result": "Vecinos coherentes (violencia->violento, inmigracion->emigracion...) con limitaciones de registro enciclopedico",
        "interpretation": "La geometria aprendida es razonable; no implica utilidad supervisada (eso lo decide EBD-05)",
        "decision": "Sin adopcion; evidencia intrinseca documentada en reports/embedding_intrinsic_eval.md",
        "notes": NOTA,
    },
    {
        "experiment_id": "EBD-05",
        "question": "Que combinacion de pooling y clasificador rinde mas usando SOLO el embedding?",
        "hypothesis": "La media de vectores (pooling A) con LR supera a los poolings B/C y a SVC",
        "feature_groups": "200d L2 por documento",
        "representation": "FastText/Word2Vec 10M; poolings A (media), B (tfidf), C (media+max)",
        "model": "LogisticRegression C=2.0 balanced / LinearSVC C=1.0 balanced",
        "hyperparameters": "identicos a V03; bloque denso en norma L2",
        "validation": "corrected_cv v2; IC95 pareado 10.000 remuestreos; umbrales ajustados (diagnostico)",
        "f1": "0.6603", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": "51.7 s (rejilla completa)",
        "dimension": "200",
        "result": "Mejor: FASTTEXT-BASELINE|A_media|LR 0.660313 (umbrales ajustados), delta -0.1169 vs V03, IC95 [-0.1397, -0.0942]; LR >= SVC; A >= B/C",
        "interpretation": "La representacion sola esta a ~0.117 de V03; el valor del embedding solo puede venir de la fusion (P3)",
        "decision": "Pasar a fusion con V03 (EBD-07..EBD-09)",
        "notes": NOTA,
    },
    {
        "experiment_id": "EBD-06",
        "question": "Cual es el veredicto de P2 (arquitectura, pooling y clasificador) antes de fusionar?",
        "hypothesis": "El veredicto se deriva de la rejilla completa y fija la configuracion para la fusion",
        "feature_groups": "ninguno (consolidacion)",
        "representation": "FastText 10M / word2vec control",
        "model": "consolidacion de la rejilla (no entrena)",
        "hyperparameters": "no aplica",
        "validation": "comparacion con corrected_cv de V03 (0.777211)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": "51.7 s", "dimension": "200",
        "result": "Configuracion elegida: FASTTEXT-BASELINE + pooling A (media) + LR; brecha -0.1169 frente a V03",
        "interpretation": "La representacion por si sola no compite; la unica via razonable es la fusion dispersa+densa",
        "decision": "GO a P3 (fusion y complementariedad)",
        "notes": NOTA,
    },
]


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    header, cuerpo = filas[0], filas[1:]
    nuevas = []
    reemplazadas = 0
    for fila in cuerpo:
        if len(fila) != len(header) and fila and fila[0] == "KAGGLE_TOXIC_2026_PUJ":
            indice = reemplazadas
            reemplazadas += 1
            if indice < len(FILAS):
                datos = dict(COMUN)
                datos.update(FILAS[indice])
                nueva = [datos.get(col, "") for col in header]
                nuevas.append(nueva)
                continue
            continue
        nuevas.append(fila)
    assert reemplazadas == 6, f"esperaba 6 filas malformadas, encontre {reemplazadas}"
    assert all(len(f) == len(header) for f in nuevas), "quedan filas con campos incorrectos"
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(nuevas)
    ids = [f[0] for f in nuevas]
    print("filas:", len(nuevas), "| unicas:", len(set(ids)))
    print("EBD presentes:", sorted(i for i in ids if i.startswith("EBD-")))


if __name__ == "__main__":
    main()
