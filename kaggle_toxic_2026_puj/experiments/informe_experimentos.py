#!/usr/bin/env python
"""
Informe consolidado de TODOS los experimentos del proyecto.

Genera reports/informe_experimentos_completo.md desde experiments.csv:
  - resumen ejecutivo;
  - el mejor experimento (V03) con descripcion y resultados;
  - tablas por capitulo (un resumen por experimento: tema, resultado, decision);
  - indice de prefijos.

Solo lectura sobre experiments.csv; ASCII puro en la salida.

Uso:
    /home/carlos/venv/bin/python experiments/informe_experimentos.py
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "experiments.csv"
SALIDA = RAIZ / "reports" / "informe_experimentos_completo.md"

FAMILIAS = {"E00", "E", "E2", "PRE", "REP", "MOD", "ENS", "HIB", "DL",
            "EBD", "AUD", "FIN", "TH", "PA", "OPT", "HOLD", "B1", "B5"}


def ascii_safe(texto: str) -> str:
    """Translitera a ASCII y limpia espacios."""
    plano = unicodedata.normalize("NFKD", texto)
    limpio = "".join(c for c in plano if not unicodedata.combining(c))
    limpio = limpio.encode("ascii", "ignore").decode("ascii")
    return " ".join(limpio.split())


def corta(texto: str, n: int) -> str:
    """Recorta con elipsis."""
    texto = ascii_safe(texto).replace("|", "/")
    if len(texto) <= n:
        return texto
    return texto[: n - 3].rstrip() + "..."


CLAVES_RESULTADO = ("corrected_cv", "delta_vs_v03", "delta_vs_30",
                    "delta_pooled", "delta", "f1_media_columnas",
                    "f1_oof_umbrales_optimos", "val_ppl", "lb", "score",
                    "holdout_meta", "auc")


def resumen_resultado(texto: str, limite: int = 150) -> str:
    """Resumen compacto del campo result (JSON -> claves; texto -> recorte)."""
    texto = texto.strip()
    if not texto:
        return ""
    if texto.startswith("{"):
        try:
            d = json.loads(texto)
        except (ValueError, TypeError):
            return corta(texto, limite)
        if isinstance(d, dict):
            partes = []
            for clave, valor in d.items():
                if len(partes) >= 4:
                    break
                if any(c in clave.lower() for c in CLAVES_RESULTADO):
                    if isinstance(valor, float):
                        partes.append(f"{clave}={valor:.6f}")
                    else:
                        partes.append(f"{clave}={str(valor)[:40]}")
            if partes:
                return corta("; ".join(partes), limite)
            return corta("; ".join(f"{k}={str(v)[:30]}"
                                   for k, v in list(d.items())[:3]), limite)
    return corta(texto, limite)


def capitulo(prefijo: str) -> int:
    """Numero de capitulo para un prefijo del registro."""
    if prefijo in FAMILIAS or re.fullmatch(r"S0\d|R[12]", prefijo):
        return 1
    if prefijo.startswith("V") and prefijo[1:].isdigit():
        return 2
    if prefijo.startswith("P") and prefijo[1:].isdigit():
        n = int(prefijo[1:])
        if n <= 72:
            return 3
        if n <= 79:
            return 4
        if n <= 89:
            return 5
        if n <= 97:
            return 6
        return 7
    return 8


TITULOS = {
    1: (("Capitulo 1 - Fundamentos, familias de features y primeras "
         "submissions"),
        ("Metrica oficial, preprocesamiento, canales de features (E2), "
         "familias MOD/ENS/HIB/DL/EBD/FIN/AUD y sondas de submission "
         "(S02-S08, R1/R2).")),
    2: (("Capitulo 2 - Serie V: iteraciones del pipeline y mecanica de "
         "adopcion (V2..V56)"),
        ("Versiones v04-v32+ del modelo original: mecanismos, proyecciones "
         "latentes, features estructurales, auditorias de metrica e "
         "integridad, sondas de leaderboard.")),
    3: ("Capitulo 3 - Fases P59-P72: optimizacion, robustez y estructura",
        ("Residuos y perdidas robustas, estructura del id, bins de "
         "longitud, blends, grafo de vecindad, canal de estructura, "
         "embeddings downstream y controles aleatorios.")),
    4: ("Capitulo 4 - Fases P73-P79: profundizacion",
        ("TextCNN y null replicado, corpus externo, atencion RNN, "
         "selector, tokenizer y cierre de huecos.")),
    5: ("Capitulo 5 - Fases P80-P89: auditorias de novedad y gobernanza",
        ("Auditoria Word2Vec y de afirmaciones, Factorization Machines, "
         "SGNS sobre Twitter/X, gobernanza de anotacion TR, LightGBM y "
         "consistencia.")),
    6: (("Capitulo 6 - Fases P90-P97: calidad de etiquetas, reapertura y "
         "mecanismos"),
        ("Casebook de errores de V03, anotacion humana (P91-P93), "
         "evaluacion de reapertura (P94-P96) y tres mecanismos nuevos "
         "(P97).")),
    7: ("Capitulo 7 - Fases P98-P101: linea contextual y supervision",
        ("C98-1 (30M) y D98-1, S99-01 (60M) y gate de escala, composicion "
         "B1 (ChatSubs) y auditoria de supervision (P101).")),
    8: ("Capitulo 8 - Otros registros",
        "Filas cuyo prefijo no cae en los capitulos anteriores."),
}


def main() -> None:
    """Construye el informe completo."""
    with open(ENTRADA, encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.DictReader(fh))

    por_cap: dict[int, list] = {}
    for r in filas:
        pref = r["experiment_id"].split("-")[0]
        por_cap.setdefault(capitulo(pref), []).append(r)

    fechas = sorted(r["date"] for r in filas if r["date"].strip())
    n_metricas = sum(1 for r in filas
                     if (r["f1"].strip() or r["macro_f1"].strip()
                         or r["weighted_f1"].strip()))

    md = []
    md.append("# Informe consolidado de experimentos - "
              "tweet-toxic-2026-puj")
    md.append("")
    md.append(f"Generado desde `experiments.csv` ({len(filas)} filas) el "
              f"2026-10-14. Rango de fechas: {fechas[0]} a {fechas[-1]}. "
              "Documento derivado y regenerable "
              "(`experiments/informe_experimentos.py`); no modifica ningun "
              "registro.")
    md.append("")
    md.append("## 0. Resumen ejecutivo")
    md.append("")
    md.append("El proyecto es una competencia academica de clasificacion de "
              "toxicidad en tweets con tres etiquetas binarias (HS = discurso "
              "de odio; TR = dirigido a persona/grupo; AG = agresividad). La "
              "metrica oficial es la media column-wise de F1 binaria; la "
              "referencia interna canonica es `corrected_cv` (F1 media por "
              "columnas con particion agrupada, 1.483 filas de dev).")
    md.append("")
    md.append("**El mejor experimento del proyecto es V03** (ver seccion 1): "
              "corrected_cv 0.7772114525165721 y leaderboard 0.78306, "
              "verificado como el maximo real de LB (auditoria V46-00). Nada "
              "lo ha superado en el leaderboard; los candidatos con mejor "
              "metrica local no transfirieron (gap local-LB documentado) o "
              "fueron falsificados en la segunda particion. V03 permanece "
              "CONGELADO como referencia.")
    md.append("")
    md.append(f"Registro: {len(filas)} filas; {n_metricas} con metricas "
              "explicitas; 7 capitulos por era de trabajo. Cada fila "
              "resume: "
              "que se probo (pregunta), el resultado registrado y la "
              "decision/estado. Las decisiones de cierre completas estan en "
              "`DECISIONS.md`; los informes por fase en `reports/`.")
    md.append("")
    md.append("## 1. El mejor experimento: V03 (referencia congelada)")
    md.append("")
    md.append("**Que es.** Pipeline lineal de tres clasificadores logisticos "
              "independientes (uno por etiqueta), sobre una union TF-IDF "
              "word(1,2) + char_wb(3,5), min_df=2, sublinear, con "
              "`LogisticRegression(class_weight='balanced', C=2.0, "
              "liblinear)`; la decision usa umbrales por etiqueta elegidos "
              "sobre probabilidades OOF por rejilla 0.20-0.80 (paso 0.01).")
    md.append("")
    md.append("**Umbrales congelados**: HS 0.44 | TR 0.51 | AG 0.46.")
    md.append("")
    md.append("**Resultados**: corrected_cv 0.7772114525165721 (reproducido "
              "exacto en varias verificaciones independientes: V9-01, V10-00, "
              "V45-00); leaderboard 0.78306 (maximo real verificado del "
              "proyecto, V46-00). Submission congelada "
              "`toxic_submission_v03_lr_union_thr.csv` (sha256 "
              "7d7478f271e6edfd...); modelo `model_v03.joblib` (263ed5bd...).")
    md.append("")
    md.append("**Por que es el mejor.** 22 sondas de leaderboard registradas "
              "y ninguna supero 0.78306; los mejores candidatos locales "
              "(p.ej. S08 meta v2, +0.018 en holdout congelado) no "
              "transfirieron al LB; la serie contextual P98-P100 produjo "
              "fusiones con deltas locales positivos (+0.005055 en P98, "
              "+0.004375 en P99) pero IC que incluye cero, fallo de "
              "atribucion vs el control de 30M o gate fallido; P100 fue "
              "negativo (-0.000591). Conclusion historica: los deltas "
              "locales pequenos no transfieren; V03 se mantiene como "
              "referencia (freno de transferencia documentado en P95).")
    md.append("")
    md.append("**Challengers mas cercanos (todos cerrados)**:")
    md.append("")
    md.append("| Candidato | Metrica local | vs V03 | LB | Estado |")
    md.append("|---|---|---|---|---|")
    md.append("| D98-1 fusion contextual 30M (P98) | 0.7822663 | "
              "+0.005055 (4/5) | no enviado | MARGINAL; falsificado en la 2a "
              "particion (+0.001151, 2/5) |")
    md.append("| S99-01 fusion 60M (P99) | 0.7815860 | +0.004375 (4/5) | no "
              "enviado | GATE_FAILED (-0.00068 vs 30M; 47 negativa) |")
    md.append("| V31-04 interacciones URL x longitud | 0.7813780 | +0.004167 "
              "| 0.78306 (identico a V03, V33-03) | NEW_SIGNAL; sin "
              "transferencia |")
    md.append("| P66-02 blend por arbol | 0.7805183 | +0.003307 (3/5) | no "
              "enviado | MARGINAL |")
    md.append("| S08 meta v2 (Ridge) | holdout 0.8465 | +0.0180 (holdout) | "
              "0.78163 | NO ADOPTADA (no transfiere) |")
    md.append("| B1 composicion ChatSubs (P100) | 0.7766201 | -0.000591 | no "
              "enviado | GATE_FAILED (falsificada) |")
    md.append("")

    for cap in sorted(por_cap):
        titulo, desc = TITULOS[cap]
        grupo = por_cap[cap]
        fechas_cap = sorted(r["date"] for r in grupo if r["date"].strip())
        md.append(f"## {titulo}")
        md.append("")
        md.append(f"{desc}")
        md.append("")
        rango = (f"{fechas_cap[0]} a {fechas_cap[-1]}"
                 if fechas_cap else "sin fecha")
        md.append(f"{len(grupo)} registros ({rango}).")
        md.append("")
        md.append("| ID | Fecha | Tema (que prueba) | Resultado | "
                  "Decision/estado |")
        md.append("|---|---|---|---|---|")
        for r in grupo:
            md.append(
                f"| {corta(r['experiment_id'], 40)} | "
                f"{corta(r['date'], 10)} | "
                f"{corta(r['question'], 150)} | "
                f"{resumen_resultado(r['result'])} | "
                f"{corta(r['decision'], 120)} |")
        md.append("")

    md.append("## Anexo - indice de prefijos")
    md.append("")
    pref_conteo: dict[str, int] = {}
    for r in filas:
        p = r["experiment_id"].split("-")[0]
        pref_conteo[p] = pref_conteo.get(p, 0) + 1
    md.append(f"{len(pref_conteo)} prefijos en total: "
              + ", ".join(f"{p} ({n})" for p, n in sorted(
                  pref_conteo.items(), key=lambda kv: -kv[1])))
    md.append("")
    md.append("Nota final: este informe es descriptivo; la autoridad sobre "
              "decisiones y estados es `DECISIONS.md` + `experiments.csv` + "
              "los informes por fase en `reports/`.")
    md.append("")

    texto = "\n".join(md)
    if any(ord(c) > 127 for c in texto):
        raise SystemExit("Byte > 127 en el informe generado")
    SALIDA.write_text(texto, encoding="ascii")
    print(f"Informe generado: {SALIDA.relative_to(RAIZ)} "
          f"({len(texto)} bytes, {len(md)} lineas)")
    print(f"  filas: {len(filas)} | capitulos: {len(por_cap)} | "
          f"prefijos: {len(pref_conteo)}")


if __name__ == "__main__":
    main()
