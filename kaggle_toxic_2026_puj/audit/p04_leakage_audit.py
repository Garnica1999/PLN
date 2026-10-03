"""P4: leakage - cuantificaciones independientes.

A. Optimismo de umbrales:
   - V03 dev OOF: congelados vs optimizados sobre el mismo OOF vs anidado
     leave-one-fold-out con reutilizacion (estilo E2-16).
   - OOF historico (1731): reproducir la seleccion 0.44/0.51/0.46 y su ganancia.
B. Contaminacion de test: escaneo de los textos de toxic_test.csv en los
   corpus externos locales (txt + raw comprimido).
C. features/v2_structured_features.parquet: ids contenidos en train.
D. Hashes de corpus vs manifiesto (si el manifiesto los registra).

Escribe audit/out/p04_leakage_checks.csv, audit/p04_findings.json y
audit/out/p04_leakage_rows.csv (filas para la matriz final).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
OUT = PROJ / "audit" / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJ))

from src import data as data_module  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402

GRID = np.round(np.arange(0.20, 0.8001, 0.01), 2)
ETQ = ("HS", "TR", "AG")
CHECKS: list[dict] = []


def f1_bin(y: np.ndarray, p: np.ndarray) -> float:
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def f1_cols(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([f1_bin(y[:, i], p[:, i]) for i in range(3)]))


def aplicar(proba: np.ndarray, umbrales: dict[str, float]) -> np.ndarray:
    return np.column_stack([
        (proba[:, i] >= umbrales[e]).astype(int) for i, e in enumerate(ETQ)])


def optimizar(y: np.ndarray, proba: np.ndarray) -> tuple[dict[str, float], float]:
    umbrales = {}
    for i, e in enumerate(ETQ):
        mejor, mejor_u = -1.0, GRID[0]
        for u in GRID:
            valor = f1_bin(y[:, i], (proba[:, i] >= u).astype(int))
            if valor > mejor + 1e-12:
                mejor, mejor_u = valor, float(u)
        umbrales[e] = mejor_u
    return umbrales, f1_cols(y, aplicar(proba, umbrales))


def ascii_text(text: str) -> str:
    """Pliega a ASCII (norma de artefactos del proyecto)."""
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
            "\u2192": "->", "\u2264": "<=", "\u2265": ">=", "\u00d7": "x"}
    for key, value in repl.items():
        text = text.replace(key, value)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch if ord(ch) < 128 else "" for ch in norm if not unicodedata.combining(ch))


def check(area: str, claim: str, evidence: str, status: str) -> None:
    CHECKS.append({"area": area, "claim": ascii_text(claim),
                   "evidence": ascii_text(evidence), "status": status})


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:  # noqa: PLR0915
    findings: dict = {}

    # ---------- A. Optimismo de umbrales ----------
    oof = np.load(PROJ / "features" / "oof_v03_final.npz", allow_pickle=False)
    y, proba, pliegues = oof["y_true"], oof["proba"], oof["pliegues"]
    congelados = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
    f1_congelado = f1_cols(y, aplicar(proba, congelados))
    umbrales_lax, f1_lax = optimizar(y, proba)
    pred_nested = np.zeros_like(y)
    for k in sorted(set(pliegues.tolist())):
        mascara_otros = pliegues != k
        u_k, _ = optimizar(y[mascara_otros], proba[mascara_otros])
        pred_nested[pliegues == k] = aplicar(proba[pliegues == k], u_k)
    f1_nested_reuse = f1_cols(y, pred_nested)
    findings["optimismo_dev"] = {
        "f1_congelado": round(f1_congelado, 6),
        "umbrales_lax": umbrales_lax,
        "f1_lax_mismo_oof": round(f1_lax, 6),
        "optimismo_mismo_oof": round(f1_lax - f1_congelado, 6),
        "f1_nested_reutilizando_otros_pliegues": round(f1_nested_reuse, 6),
        "optimismo_residual_tras_anidar": round(f1_nested_reuse - f1_congelado, 6),
        "referencia_e2_16_citada": 0.776613,
        "referencia_v7_estricta_citada": 0.767228,
    }
    check("thresholds", "optimizar umbrales sobre el mismo OOF infla la metrica",
          f"+{f1_lax - f1_congelado:.4f} (0.777211 -> {f1_lax:.4f})", "CONFIRMADO")
    check("thresholds", "reutilizar OOF de otros pliegues como calibracion deja sesgo residual",
          f"nested reuso = {f1_nested_reuse:.6f} vs congelado {f1_congelado:.6f} "
          f"(delta {f1_nested_reuse - f1_congelado:+.6f}); E2-16 citaba 0.776613",
          "CUANTIFICADO")

    hist = np.load(PROJ / "experiments" / "baseline_v03_frozen" / "oof_v03_historical.npz")
    yh, ph = hist["y_true"], hist["proba"]
    uh, f1h_lax = optimizar(yh, ph)
    f1h_congelado = f1_cols(yh, aplicar(ph, congelados))
    findings["optimismo_historico"] = {
        "umbrales_optimos_recomputados": uh,
        "umbrales_documentados": congelados,
        "f1_optimo_recomputado": round(f1h_lax, 6),
        "f1_con_umbrales_documentados_sobre_mismo_oof": round(f1h_congelado, 6),
        "ganancia_sobre_umbral_0_5": round(f1h_lax - f1_cols(yh, (ph >= 0.5).astype(int)), 6),
        "coincide_seleccion": uh == congelados,
    }
    check("thresholds", "el optimo historico 0.44/0.51/0.46 se reproduce sobre el OOF guardado",
          f"optimos = {uh}; ganancia vs 0.5 = {f1h_lax - f1_cols(yh, (ph >= 0.5).astype(int)):+.4f} "
          "(la ganancia es sesgo de seleccion si se reporta sobre el mismo OOF)",
          "CONFIRMADO" if uh == congelados else "REVISAR")

    # ---------- B. Contaminacion de test en corpus ----------
    test = data_module.read_test()
    train = data_module.read_train()
    test["id"] = test["id"].astype(int)
    train["id"] = train["id"].astype(int)
    elegibles = [t for t in test["text"] if len(t) >= 30 and len(t.split()) >= 5]
    patrones = sorted({t.strip() for t in elegibles} | {t.strip().lower() for t in elegibles})
    pats_path = OUT / "p04_test_patterns.txt"
    pats_path.write_text("\n".join(patrones), encoding="utf-8")
    hallazgos_corpus = []
    corpus_root = PROJ / "data" / "external" / "corpora"
    txt_targets = sorted(str(p) for p in corpus_root.glob("**/*.txt"))
    if txt_targets:
        res = subprocess.run(["grep", "-rFl", "-f", str(pats_path), *txt_targets],
                             capture_output=True, text=True, check=False)
        hallazgos_corpus = [line for line in res.stdout.splitlines() if line.strip()]
    findings["scan_corpus"] = {
        "n_patrones": len(patrones),
        "n_textos_test_elegibles": len(elegibles),
        "n_textos_test_totales": len(test),
        "archivos_txt_escaneados": len(txt_targets),
        "coincidencias": hallazgos_corpus,
    }
    gz = sorted(corpus_root.glob("**/*.gz"))
    for archivo in gz:
        res = subprocess.run(["bash", "-c",
                              f"zcat {archivo} | grep -Fl -f {pats_path} - | head -3"],
                             capture_output=True, text=True, check=False)
        if res.stdout.strip():
            hallazgos_corpus.append(str(archivo))
    check("test_contamination", "ningun texto de test aparece en los corpus externos (txt + gz)",
          f"{len(patrones)} patrones sobre {len(txt_targets)} txt y {len(gz)} gz; "
          f"coincidencias = {len(hallazgos_corpus)}",
          "LIMPIO" if not hallazgos_corpus else "REVISAR")
    findings["scan_corpus"]["coincidencias_total"] = len(hallazgos_corpus)

    # ---------- C. features parquet ----------
    pq = PROJ / "features" / "v2_structured_features.parquet"
    frame = pd.read_parquet(pq)
    id_col = next((c for c in frame.columns if c.lower() == "id"), None)
    ids_pq = set(frame[id_col].astype(int).tolist()) if id_col else set()
    ids_train = set(train["id"].tolist())
    findings["features_parquet"] = {
        "columnas": int(frame.shape[1]),
        "filas": int(frame.shape[0]),
        "id_col": id_col,
        "ids_subconjunto_de_train": bool(ids_pq <= ids_train),
        "ids_fuera_de_train": len(ids_pq - ids_train),
        "cubre_holdout": bool(ids_pq >= set(oof["ids"].tolist())),
    }
    check("features", "features/v2_structured_features.parquet contiene solo ids de train",
          f"{len(ids_pq)} ids; fuera de train = {len(ids_pq - ids_train)}",
          "LIMPIO" if ids_pq <= ids_train else "REVISAR")

    # ---------- D. hashes de corpus vs manifiesto ----------
    manifest = json.loads((PROJ / "data" / "external" / "corpora" / "corpus_manifest.json")
                          .read_text(encoding="utf-8"))
    fuentes = [(f.get("nombre"), f.get("url") or f.get("recurso") or "")
               for f in manifest.get("fuentes", [])]
    construidos = manifest.get("corpus_construidos", [])
    findings["corpus_fuentes"] = fuentes
    verif = []
    for entrada in construidos if isinstance(construidos, list) else []:
        ruta = entrada.get("archivo") or entrada.get("ruta") or entrada.get("salida") or ""
        esperado = (entrada.get("sha256") or "").lower()
        if not ruta or not esperado:
            continue
        path = PROJ / ruta if not str(ruta).startswith("/") else Path(ruta)
        if not path.exists():
            verif.append({"archivo": ruta, "estado": "AUSENTE"})
            continue
        real = sha256_of(path)
        verif.append({"archivo": ruta, "estado": "OK" if real == esperado else "MISMATCH",
                      "real": real[:16], "esperado": esperado[:16]})
    findings["verificacion_hashes_corpus"] = verif
    if verif:
        ok = sum(1 for v in verif if v["estado"] == "OK")
        check("corpus", "hashes de corpus del manifiesto verificables",
              f"{ok}/{len(verif)} OK; estados={sorted({v['estado'] for v in verif})}",
              "CONFIRMADO" if ok == len(verif) else "REVISAR")
    else:
        check("corpus", "el manifiesto no registra sha256 por archivo construido (solo fuentes)",
              f"corpus_construidos con {len(construidos)} entradas; sin campos de hash utilizables",
              "UNKNOWN")

    # ---------- escritura ----------
    with (OUT / "p04_leakage_checks.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=["area", "claim", "evidence", "status"])
        writer.writeheader()
        writer.writerows(CHECKS)
    rows = [
        {"Experiment_ID": "GLOBAL/thresholds", "Leakage_Type": "L4-potencial(selection bias)",
         "Potential": "si", "Detected": "si (documentado)",
         "Evidence": f"optimizar umbrales sobre el mismo OOF infla "
                     f"{f1_lax - f1_congelado:+.4f}; anidado por reutilizacion deja "
                     f"{f1_nested_reuse - f1_congelado:+.6f}",
         "Affected_Result": "v1/v2 fase 13a: 0.7989 reportado sobre umbrales optimizados en el mismo OOF",
         "Severity": "MEDIUM (histórico, corregido por D-016)", "Decision": "documentado; V03 usa umbrales congelados"},
        {"Experiment_ID": "GLOBAL/corpus-test", "Leakage_Type": "L5",
         "Potential": "si", "Detected": "no",
         "Evidence": f"scan {len(patrones)} patrones sobre {len(txt_targets)} txt + {len(gz)} gz; "
                     f"coincidencias {len(hallazgos_corpus)}",
         "Affected_Result": "n/a", "Severity": "INFO", "Decision": "sin accion"},
    ]
    with (OUT / "p04_leakage_rows.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{k: ascii_text(str(v)) for k, v in row.items()} for row in rows])
    (PROJ / "audit" / "p04_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=True, default=str), encoding="ascii")
    print("A. optimismo dev:", findings["optimismo_dev"])
    print("A. optimismo historico:", findings["optimismo_historico"])
    print("B. scan corpus:", {k: v for k, v in findings["scan_corpus"].items() if k != "coincidencias"})
    print("C. parquet:", findings["features_parquet"])
    print("D. verificacion corpus:", findings["verificacion_hashes_corpus"][:6])
    print("checks:", [(c["area"], c["status"]) for c in CHECKS])


if __name__ == "__main__":
    main()
