"""P13 (Puerta 2): meta-analisis de familias y representaciones (spec seccion 59)."""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")


def fold(text: str) -> str:
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
    for k, v in repl.items():
        text = text.replace(k, v)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch if ord(ch) < 128 else "" for ch in norm if not unicodedata.combining(ch))


FAMILIAS = [
    # familia, n_experimentos, mejor_delta, peor_delta, media_delta, LB, transfer, complementariedad, estado
    ("v1 baseline word", 1, "+0.000", "-0.000", "+0.000", "0.75677 (B1)", "gap -0.0010 (excelente)",
     "n/a", "cerrada (superada por union)"),
    ("v1/v2 union word+char", 3, "+0.0063 (thr)", "-0.0280 (optuna)", "n/a",
     "0.7703-0.7831", "gap -0.013 a -0.029", "n/a", "cerrada; V03 referencia"),
    ("v2 correccion de protocolo", 2, "n/a (metodo)", "n/a", "n/a",
     "n/a", "grupos fusionados acercan local y LB", "n/a", "ADOPTADA (D-015/D-016)"),
    ("v2 metas/candidatos", 12, "+0.0102 (dev)", "-0.0014 (LB)", "n/a",
     "0.7790-0.7816", "gap -0.006 a +0.002", "emocion+sintaxis casi perfecta",
     "cerrada (D-024; V03 se mantiene)"),
    ("v4 reweighting/label/meta/shift", 8, "+0.0082 (A_cost)", "-0.0008", "n/a",
     "sin sondas (solo holdout)", "holdout rechazo D-029", "canales no superan",
     "cerrada (D-029)"),
    ("v5/v5.1 embeddings", 27, "n/a (complementariedad negativa)", "n/a", "n/a",
     "sin sondas", "n/a", "nula con V03", "cerrada (D-032..D-043)"),
    ("v6 tracks", 7, "+0.0016 (I2)", "-0.0105 (H1)", "n/a",
     "sin sondas", "n/a", "nula", "cerrada (D-052); caveat F-22"),
    ("v7 tracks", 3, "-0.0019 (mejor)", "-0.0024", "n/a",
     "sin sondas", "n/a", "nula", "cerrada (D-056)"),
    ("v8 tecnicas + sondas", 10, "+0.0051 (promediado OOF)", "-0.030 (LDA)", "n/a",
     "3 sondas: 0.7747/0.7778/0.7712", "gap -0.0025 a -0.0111", "ninguna",
     "cerrada (D-065/D-067)"),
]

REPRESENTACIONES = [
    # representacion, modelo, loss, features, regimen, mejor_resultado, transfer_gap
    ("TF-IDF word (1,2) min_df2 sublinear", "LR C=1 balanced", "BCE", "~4k", "1731 filas",
     "0.7578 OOF / 0.75677 LB", "-0.0010"),
    ("union word+char_wb", "LR C=2 balanced", "BCE", "24646", "1731 filas",
     "0.7927 OOF his / 0.777211 corr / 0.78306 LB", "-0.0158 (historico)"),
    ("union word+char_wb", "LR C=3.26 (optuna)", "BCE", "~32k", "1731 filas",
     "0.8012-0.8039 OOF / 0.77602 LB", "-0.0279"),
    ("union + 44 meta features", "Ridge alpha=3", "L2 (clasificacion)", "44", "1483 dev OOF",
     "0.7874 dev / 0.78163 LB", "-0.0058"),
    ("union + canales EMO/SYN", "Ridge alpha=0.3-1", "L2", "restringido", "1483 dev",
     "0.7793-0.7815 LB", "+0.0005 a +0.0018"),
    ("TF-IDF union", "LinearSVC C=1", "hinge", "24646", "1731", "0.7809 dev (M3)", "n/a"),
    ("FastText propio 10M-50M (media de tokens)", "LR/Ridge", "BCE/L2", "200 dims", "1483 dev",
     "sin mejora", "n/a"),
    ("SVD(300) de TF-IDF union", "MLP 256-128 (torch)", "BCE", "300", "1731 (hist.)",
     "0.7949 OOF his", "no sondo"),
    ("chars desde cero (CharCNN)", "CNN embeddings propios", "BCE", "vocab char", "1731",
     "0.741 OOF his", "n/a"),
    ("words desde cero (BiLSTM)", "BiLSTM 96", "BCE", "vocab palabra", "1731",
     "0.736 OOF his", "n/a"),
    ("union (5 particiones)", "promedio de modelos", "BCE", "24646", "1483 dev",
     "+0.0051 OOF", "-0.0111 (v12 LB)"),
]


def main() -> None:
    out1 = PROJ / "audit" / "out" / "p13_meta_familias.csv"
    with out1.open("w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["familia", "n_experimentos", "mejor_delta", "peor_delta", "media_delta",
                    "LB_results", "transfer_behavior", "complementariedad", "estado"])
        for fila in FAMILIAS:
            w.writerow([fold(str(x)) for x in fila])
    out2 = PROJ / "audit" / "out" / "p13_meta_representaciones.csv"
    with out2.open("w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["representacion", "modelo", "loss", "features", "regimen",
                    "mejor_resultado", "transfer_gap"])
        for fila in REPRESENTACIONES:
            w.writerow([fold(str(x)) for x in fila])
    print("familias:", len(FAMILIAS), "->", out1)
    print("representaciones:", len(REPRESENTACIONES), "->", out2)


if __name__ == "__main__":
    main()
