"""P59 step 11: track F, generative n-gram model. CLOSED by duplication, no training.

Why this track is closed without consuming an experiment
--------------------------------------------------------
Track F proposes a generative class-conditional n-gram model with one small order
and one smoothing or backoff strategy fixed in advance. It is closed on three
independent legs.

**Leg 1, algebraic.** Over a fixed n-gram feature representation, ANY
class-conditional generative model collapses to a log-linear form:

    log P(c | x) = log P(c) + sum_f theta_{f,c}

with `theta_{f,c}` the class-conditional log-probability of feature `f`. That is
the naive-Bayes / log-odds form. The estimator of `theta` is the only free
choice, and every estimator of it is a smoothing strategy: Laplace, Witten
backoff, Dirichlet priors, and ComplementNB's complement. So "a new generative
n-gram formulation with one new smoothing strategy" is, by construction, a
smoothing variant of a model that is already closed. The feature representation
cannot rescue it either, because the collapse holds for whatever basis the
n-grams live in.

**Leg 2, the family has been executed, repeatedly.**
- `MOD-02_BEST` and `MOD-02_W12`: ComplementNB(alpha=1,0) x3 on the exact union
  TF-IDF of V03. 0,6888 with mean per-column F1 against 0,7587 for the linear
  baseline, **-0,069949 and 0 of 5 folds**. Discarded.
- `E2-15-NB-WORD` and `E2-15-NB-CHAR`: the NB-SVM log-count ratio, fold-local,
  feeding the same logistic regression. **-0,0234 with 0 of 5 folds**, ensemble
  gain **-0,0057**, and a residual correlation with V03 of **0,9243**, which is
  the number that says there is nothing complementary left in it.
- `V37-01`: BM25 word weighting with V03's classifier, k1 1,2 and b 0,75,
  idf `log(1 + (N - df + 0,5)/(df + 0,5))`, document length and fold-local avgdl.
  **-0,0027860149**, 25 rescues against 31 damage, **net -6**, correlation
  **0,9874**. Its own words: "BM25 no aporta una fuente de senal nueva". Closed
  with "No se prueba BM25 sobre char".
- `V4-G01-G03`: logistic regression on likelihood-derived scores, candidate D
  with fragile evidence.
- `V26-03`: Tsetlin machine, a genuinely different discriminative rule over the
  same alphabet, REJECTED and its family closed.
- Zero rows of the registry describe a "modelo generativo" as such; the family
  exists only under the names above.

**Leg 3, and this is the one that closes the exact specification F gives.** The
only free parameter of a generative n-gram model on a fixed representation is
the smoothing. `V8-04` measured that parameter directly: the class-wise
log-count ratio on V03's TF-IDF gives **-0,058496** with smoothing 1e-6 and
**+0,004745** with smoothing 1e-3, a swing of **+0,063241** between two values
that differ only in that constant. Its interpretation states that "el suavizado
sostiene el metodo entero". And its decision pre-registers, in writing: "**No se
barre el suavizado para buscar un valor mejor**, porque seria el rescate por
hiperparametro que las reglas prohiben desde la fase v6".

Track F is specified as exactly one order and exactly one smoothing strategy. So
F is a single point of a sweep that was closed in advance, and the best point of
that sweep already missed its own filter by three thousandths, with an IC95
including zero and 3 of 5 folds.

Legs 1 and 2 say the mechanism is not new. Leg 3 says the one degree of freedom
left inside it is forbidden. Together there is no experiment to run.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_11_generativo_ngram.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
JSON_F = PROJECT_ROOT / "reports" / "p59_F_generativo_ngram.json"
MD_F = PROJECT_ROOT / "reports" / "p59_F_generativo_ngram.md"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-10b] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def main() -> None:
    """Registra el cierre de F por duplicacion funcional, sin entrenar.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-F Generative n-gram. Cierre por duplicacion funcional, SIN entrenar")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ccv_ref = float(f1_score(v["y_true"], v["pred"], average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    log(f"   control: V03 {ccv_ref!r} exacto")

    informe: dict[str, Any] = {
        "familia": "P59_F_GENERATIVE_NGRAM",
        "naturaleza": ("CIERRE POR DUPLICACION FUNCIONAL. 0 experimentos, 0 modelos, 0 scores, "
                       "0 submissions, 0 Kaggle, 0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "track": "P59-F",
        "estado": "CLOSED_BY_DUPLICATION",
        "experimentos_consumidos": 0,
        "dependencia": "ninguna. Solo gap analysis, novedad funcional y presupuesto",
        "control": {"corrected_cv_V03": repr(ccv_ref),
                    "reproduce_exacto": bool(abs(ccv_ref - V03_CCV) <= 1e-12)},
        "F1_gap_check": {
            "veredicto": "EXISTE EQUIVALENCIA FUNCIONAL",
            "pierna_1_algebraica": {
                "colapso": ("sobre una representacion de n-gramas fija, TODO modelo generativo "
                            "condicional de clase colapsa a log P(c|x) = log P(c) + sum_f theta_{f,c}"),
                "consecuencia": ("esa es la forma de naive Bayes y de log-odds. El unico parametro "
                                 "libre es el estimador de theta, y todo estimador de theta es una "
                                 "estrategia de suavizado: Laplace, backoff de Witten, priors de "
                                 "Dirichlet, o el complemento de ComplementNB"),
                "por_que_eso_mata_F": ("una formulacion generativa nueva con una estrategia de "
                                       "suavizado nueva es, por construccion, una variante de "
                                       "suavizado de un modelo ya cerrado. La representacion no "
                                       "salva nada, porque el colapso vale para cualquier base"),
            },
            "pierna_2_familia_ya_ejecutada": [
                {"id": "MOD-02_BEST", "que_hizo": "ComplementNB(alpha=1,0) x3 sobre la union TF-IDF de V03",
                 "resultado": ("0,6888 de F1 media por columna frente a 0,7587 del lineal, "
                               "-0,069949 y 0 de 5 pliegues"), "cierre": "descartada"},
                {"id": "E2-15-NB-WORD", "que_hizo": "ratio log-count NB-SVM fold-local sobre word(1,2) hacia la misma LR",
                 "resultado": ("-0,0234 con 0 de 5 pliegues, ganancia de ensemble -0,0057 y "
                               "correlacion de residuales con V03 de 0,9243"),
                 "cierre": "NO adoptada"},
                {"id": "E2-15-NB-CHAR", "que_hizo": "lo mismo sobre char",
                 "resultado": "-0,0392 con 0 de 5 pliegues", "cierre": "NO adoptada"},
                {"id": "V37-01", "que_hizo": ("ponderacion BM25 en word con el clasificador de V03, "
                                               "k1 1,2 y b 0,75, idf log(1 + (N - df + 0,5)/(df + 0,5)), "
                                               "longitud de documento y avgdl fold-local"),
                 "resultado": ("-0,0027860149, 25 rescates contra 31 danios, neto -6, correlacion "
                               "0,9874, y su propia lectura: BM25 no aporta una fuente de senal nueva"),
                 "cierre": "BM25 = CLOSED. No se prueba BM25 sobre char"},
                {"id": "V4-G01-G03", "que_hizo": "LR sobre puntuaciones de verosimilitud",
                 "resultado": "candidato D con evidencia fragil, IC95 del delta toca cero"},
                {"id": "V26-03", "que_hizo": "Tsetlin Machine, otra regla discriminativa sobre el mismo alfabeto",
                 "resultado": "REJECTED, familia cerrada"},
                {"id": "HIB-H11/H12/H14/UNION", "que_hizo": "bloques de log-odds y mezclas de ratio",
                 "resultado": "descartadas, deltas negativos"},
            ],
            "filas_que_describen_un_modelo_generativo": 0,
            "nombre_por_el_que_existe_la_familia": ("solo como ComplementNB, NB-SVM, BM25/Okapi, "
                                                     "log-odds y likelihood scores"),
            "pierna_3_el_unico_grado_de_libertad_esta_prohibido": {
                "por_que_es_legislativa": ("el unico parametro libre de un modelo generativo de "
                                           "n-gramas sobre representacion fija es el suavizado, y F "
                                           "esta especificado exactamente como un unico orden y una "
                                           "unica estrategia de suavizado"),
                "V8_04_lo_medijo": ("el ratio log-count por etiqueta sobre la TF-IDF de V03 da "
                                    "-0,058496 con suavizado 1e-6 y +0,004745 con 1e-3, un salto de "
                                    "+0,063241 entre dos valores que solo difieren en esa "
                                    "constante. Su interpretacion: el suavizado sostiene el metodo "
                                    "entero"),
                "cierre_pre_registrado": ("V8-04: no se barre el suavizado para buscar un valor "
                                          "mejor, porque seria el rescate por hiperparametro que "
                                          "las reglas prohiben desde la fase v6"),
                "estado_del_mejor_punto": ("la variante de 1e-3 queda a tres milesimas de superar "
                                           "su filtro de detectabilidad, con IC95 que incluye el "
                                           "cero y 3 de 5 pliegues"),
            },
        },
        "decision": (
            "CLOSED_BY_DUPLICATION. 0 experimentos consumidos. No es que el modelo generativo de "
            "n-gramas falle en este corpus: es que no es un experimento. Sobre una representacion "
            "fija colapsa a la forma de naive Bayes, y su unico grado de libertad, el suavizado, "
            "esta pre-registrado como barrido prohibido desde V8-04, donde ademas se midio que "
            "sostiene el metodo entero. Sin orden sweep, sin suavizado sweep, sin backoff sweep, sin "
            "submission. V03 remains the best verified leaderboard result."),
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "entrenamientos": 0, "segundos": 0.1},
    }
    JSON_F.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# P59-F - Modelo generativo de n-gramas: CLOSED_BY_DUPLICATION")
    A("")
    A(f"Generado por `experiments/p59_11_generativo_ngram.py`. Fecha {inicio.isoformat(timespec='seconds')}. "
      "Sin entrenar.")
    A("")
    A("## Que proponia F")
    A("")
    A("Un modelo generativo de n-gramas que exploits evidencia de secuencia o condicionalidad que un "
      "modelo discriminativo sparse no representa igual, con un unico orden pequeno y una unica "
      "estrategia de suavizado o backoff fijada de antemano.")
    A("")
    A("## Control")
    A("")
    A("```text")
    A(f"V03 {ccv_ref!r} reproduce exacto: {abs(ccv_ref - V03_CCV) <= 1e-12}")
    A("```")
    A("")
    A("## F1 gap check: EXISTE EQUIVALENCIA FUNCIONAL")
    A("")
    A("### Pierna 1, algebrica")
    A("")
    A("Sobre una representacion de n-gramas fija, todo modelo generativo condicional de clase "
      "colapsa a")
    A("")
    A("```text")
    A("log P(c | x) = log P(c) + sum_f theta_{f,c}")
    A("```")
    A("")
    A("que es la forma de naive Bayes y de log-odds. El unico parametro libre es el estimador de "
      "`theta`, y todo estimador de `theta` es una estrategia de suavizado: Laplace, backoff de "
      "Witten, priors de Dirichlet, o el complemento de ComplementNB. La representacion no salva "
      "nada, porque el colapso vale para cualquier base.")
    A("")
    A("### Pierna 2, la familia ya se ejecuto")
    A("")
    A("| id | que hizo | resultado | cierre |")
    A("|---|---|---|---|")
    A("| MOD-02_BEST | ComplementNB(alpha=1,0) x3 sobre la union TF-IDF de V03 | 0,6888 frente a "
      "0,7587 del lineal, **-0,069949 y 0 de 5 pliegues** | descartada |")
    A("| E2-15-NB-WORD | ratio log-count NB-SVM fold-local hacia la misma LR | **-0,0234, 0 de 5 "
      "pliegues**, ganancia de ensemble **-0,0057**, correlacion de residuales con V03 **0,9243** | "
      "NO adoptada |")
    A("| E2-15-NB-CHAR | lo mismo sobre char | -0,0392 con 0 de 5 pliegues | NO adoptada |")
    A("| V37-01 | BM25 en word con el clasificador de V03, k1 1,2 y b 0,75 | **-0,0027860149**, "
      "rescates 25 contra danios 31, **neto -6**, correlacion **0,9874** | BM25 = CLOSED, no se "
      "prueba sobre char |")
    A("| V4-G01-G03 | LR sobre verosimilitudes | candidato D con evidencia fragil | --- |")
    A("| V26-03 | Tsetlin Machine | REJECTED | familia cerrada |")
    A("")
    A("Cero filas del registro describen un modelo generativo como tal: la familia existe solo con "
      "los nombres de arriba.")
    A("")
    A("### Pierna 3, y esta cierra la especificacion exacta de F")
    A("")
    A("El unico grado de libertad de un modelo generativo de n-gramas sobre representacion fija es "
      "el **suavizado**, y F esta especificado justo como un orden y una estrategia de suavizado. "
      "`V8-04` midio ese parametro directamente:")
    A("")
    A("```text")
    A("suavizado 1e-6  ->  -0,058496   con IC95 [-0,07611, -0,04061], 0 de 5 pliegues")
    A("suavizado 1e-3  ->  +0,004745   con IC95 [-0,00708, +0,01714], 3 de 5 pliegues")
    A("salto entre dos valores que solo difieren en esa constante:  +0,063241")
    A("```")
    A("")
    A("Su propia lectura: **el suavizado sostiene el metodo entero**. Y su decision pre-registra, "
      "por escrito: *no se barre el suavizado para buscar un valor mejor, porque seria el rescate "
      "por hiperparametro que las reglas prohiben desde la fase v6*. Ademas el mejor punto queda a "
      "tres milesimas de superar su filtro de detectabilidad, con el IC95 incluyendo el cero.")
    A("")
    A("F es un punto suelto de un barrido cerrado de antemano.")
    A("")
    A("## Decision")
    A("")
    A("```text")
    A(f"decision  {informe['decision']}")
    A("```")
    A("")
    A("> V03 remains the best verified leaderboard result.")
    A("")
    A("*Seccion generada por `experiments/p59_11_generativo_ngram.py`. Sin push.*")
    MD_F.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_F, JSON_F):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("   escrito reports/p59_F_generativo_ngram.md y .json")
    log("P59-F VEREDICTO: CLOSED_BY_DUPLICATION | 0 experimentos consumidos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
