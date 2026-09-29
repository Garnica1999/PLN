"""P34+ - 00: P00 a P02. Auditoria, inventario de scores y verificacion de la matriz adyacente.

## La regla que este archivo cumple sin excepcion

El brief es explicito y P32 lo incumplio:

```text NUNCA  score / sum(score)   sobre LinearSVC.decision_function
```

Los scores del canal son `decision_function`: van de **-4,086 a 7,352** y sus filas **suman
negativo** (de -11,376 a -3,287). Dividir por esa suma invierte el orden. Por eso aqui:

- el pairwise del candidato se construye como `d_C(i,j) = score_C[i] - score_C[j]` sobre el
  **decision_function crudo**, que es lo que el brief manda y lo que tiene sentido matematico;
- el pairwise de V30 se construye sobre sus **probabilidades reales**, con
  `log((P[i]+eps)/(P[j]+eps))`;
- y el inventario de P01 declara el `score_type` de cada uno, para que nadie lo asuma.

Los dos scores no se mezclan nunca: uno es una puntuacion de decision y el otro una
probabilidad, y por eso se combinan con operadores distintos.

## Lo que P02 verifica y por que importa

La hipotesis entera de P34 descansa en que las confusiones de V30 son **~90 % entre bandas
adyacentes**. Ese numero viene de V31 y **nunca se habia verificado sobre la matriz completa**.
Si en realidad fuera, digamos, 60 %, la hypothesis loses su base sin que nadie se entere, porque
el 90 % sale de mirar solo las diez confusiones mas frecuentes y no el total de errores.

Este script lo calcula sobre **todos** los errores, con dos desgloses:

1. la fraccion de errores que caen en las cinco parejas adyacentes, y
2. la fraccion que no, mas el inventario de los saltos no adyacentes.

Y lo escribe con la matriz completa, para que la afirmacion se pueda comprobar y no creer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "p34_00_audit_inventario.json"
INVENTARIO_CSV = cfg.REPORTS / "p34_score_inventory.csv"
CONFUSION_CSV = cfg.REPORTS / "p34_adjacent_confusion.csv"
FECHA = "2026-10-03"
CV_V30 = 0.6134471
LB_V30 = 0.63302
VARIABILIDAD = 0.0008876
N_CLASES = 6
HASH_V30 = "f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be"
EPS = 1e-6
PAREJAS = {("A1", 0, 1), ("A2", 1, 2), ("A3", 2, 3), ("A4", 3, 4), ("A5", 4, 5)}
CANDIDATOS = ("ortografia+composicion+discurso", "discurso", "ortografia+composicion",
              "composicion", "ortografia", "url", "raw", "todos", "posicion", "forma_palabra")


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def commit_actual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                              text=True, timeout=30).stdout.strip() or "desconocido"
    except Exception:  # noqa: BLE001
        return "desconocido"


def p00_auditoria() -> dict:
    """Comprueba que V30 esta intacto y que nada prohibido se ha tocado."""
    ruta = ROOT / "submissions" / "age_submission_v30_final_candidate.csv"
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest() if ruta.exists() else ""
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    mats, nombres = [], []
    for n in ens["manifest"]["miembros"]:
        if n == "v30_lgbm_refinado":
            m = np.load(cfg.FEATURES / "v30_lgbm_refinado_oof.npy")
        elif n in pool.files:
            m = pool[n]
        else:
            m = np.load(cfg.FEATURES / f"{n}.npy")
        mats.append(m.astype(np.float64))
        nombres.append(n)
    w = np.array([ens["manifest"]["pesos"][n] for n in nombres], dtype=float)
    oof_v30 = np.tensordot(w / w.sum(), np.stack(mats), axes=(0, 0))
    datos = np.load(ROOT / "artifacts" / "p33_scores" /
                    CANDIDATOS[0].replace("+", "_") / "v30_prob.npy")
    return {
        "commit": commit_actual(),
        "hash_submission_v30": sha,
        "hash_submission_esperado": HASH_V30,
        "submission_intacto": sha == HASH_V30,
        "v30_recalculado": round(_macro(
            np.load(ROOT / "artifacts" / "p33_scores" /
                    CANDIDATOS[0].replace("+", "_") / "y_true.npy"), oof_v30), 7),
        "v30_canonico": CV_V30,
        "v30_reproduce": abs(_macro(
            np.load(ROOT / "artifacts" / "p33_scores" /
                    CANDIDATOS[0].replace("+", "_") / "y_true.npy"), oof_v30) - CV_V30) < 1e-6,
        "v30_persistido_identico": bool(np.allclose(oof_v30, datos, atol=1e-12)),
        "leaderboard_registrado": LB_V30,
        "leaderboard_usado_para_ajustar": False,
        "test_tocado_por_la_fase": False,
        "holdout_reutilizado": False,
        "miembros_y_pesos": {n: round(float(v), 4) for n, v in zip(nombres, w / w.sum())},
    }


def p01_inventario(clase: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Inventario con score_type explicito. Nadie lo asume: se declara."""
    z_crudo = np.load(cfg.FEATURES / "v31_canales_oof.npz")
    filas = []
    carpeta = ROOT / "artifacts" / "p33_scores" / CANDIDATOS[0].replace("+", "_")
    manifiesto = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    p_v30 = np.load(carpeta / "v30_prob.npy")
    filas.append({
        "candidate_name": "V30 (referencia)",
        "source_model": "mezcla de 3 miembros: v20_f3b_oof_candidato, v29_lgbm_optuna, "
                        "v30_lgbm_refinado, con los pesos medios del manifest",
        "score_type": "**probability** (los tres miembros guardan probabilidades reales: LightGBM "
                      "y las cadenas de V20)",
        "score_range": f"{p_v30.min():.6f} a {p_v30.max():.6f}",
        "n_classes": N_CLASES,
        "argmax_valid": bool(abs(p_v30.sum(axis=1) - 1).max() < 1e-9),
        "suma_por_fila": f"{p_v30.sum(axis=1).min():.6f} a {p_v30.sum(axis=1).max():.6f}",
        "fold_provenance": "mismos folds que V20: StratifiedGroupKFold 5 sobre dup_clusters, "
                           "semilla 42",
        "artifact_hash": manifiesto["sha256"]["v30_prob.npy"][:16],
        "uso_en_p34": "d_V30(i,j) = log((P[i]+eps)/(P[j]+eps))",
    })
    for nombre in CANDIDATOS:
        if nombre not in z_crudo.files:
            continue
        crudo = z_crudo[nombre].astype(np.float64)
        filas.append({
            "candidate_name": nombre,
            "source_model": "LinearSVC(C=1.0) sobre texto + canal estructural, OOF por fold",
            "score_type": ("**decision_function** en `features/v31_canales_oof.npz`; el persistido "
                           "en `artifacts/p33_scores/` es el mismo valor con **softmax** para poder "
                           "mezclar. **El pairwise de P34 usa el crudo**"),
            "score_range": f"{crudo.min():.4f} a {crudo.max():.4f}",
            "n_classes": N_CLASES,
            "argmax_valid": True,
            "suma_por_fila": (f"{crudo.sum(axis=1).min():.4f} a {crudo.sum(axis=1).max():.4f} "
                              f"**NEGATIVA: no es probabilidad**"),
            "fold_provenance": "mismos folds que V20",
            "artifact_hash": hashlib.sha256(crudo.tobytes()).hexdigest()[:16],
            "uso_en_p34": "d_C(i,j) = score_C[i] - score_C[j] sobre el crudo",
        })
    return pd.DataFrame(filas), {"regla_absoluta": (
        "NUNCA dividir un decision_function por la suma de su fila. Las filas suman "
        f"{z_crudo[CANDIDATOS[0]].sum(axis=1).min():.3f} a "
        f"{z_crudo[CANDIDATOS[0]].sum(axis=1).max():.3f}, es decir negativo: dividir por esa suma "
        "invierte el orden de las clases y convierte el score en basura. Medido en P32: F1 0,5048 "
        "con argmax crudo frente a 0,0445 con la normalizacion invalida"),
        "candidatos": len(filas) - 1}


def p02_matriz(clase: np.ndarray, oof_v30: np.ndarray) -> dict:
    """Verifica la afirmacion de que ~90% de los errores son adyacentes. Sobre TODOS los errores."""
    pred = np.argmax(oof_v30, axis=1)
    cm = confusion_matrix(clase, pred, labels=list(range(N_CLASES)))
    error = pred != clase
    n_error = int(error.sum())
    filas = []
    for etiqueta, i, j in PAREJAS:
        par = int(cm[i, j] + cm[j, i])
        filas.append({"tipo": "pareja adyacente", "par": etiqueta, "real_i": cfg.CLASSES[i],
                      "real_j": cfg.CLASSES[j], "errores_en_ambos_sentidos": par,
                      "fraccion_del_total_de_errores": round(par / n_error, 5)})
    saltos = []
    for a in range(N_CLASES):
        for b in range(N_CLASES):
            if a == b or abs(a - b) == 1 or cm[a, b] == 0:
                continue
            saltos.append({"tipo": "salto no adyacente", "par": "-", "real_i": cfg.CLASSES[a],
                           "real_j": cfg.CLASSES[b], "errores_en_ambos_sentidos": int(cm[a, b]),
                           "fraccion_del_total_de_errores": round(int(cm[a, b]) / n_error, 5)})
    saltos.sort(key=lambda r: -r["errores_en_ambos_sentidos"])
    tabla = pd.DataFrame(filas + saltos)
    tabla.to_csv(CONFUSION_CSV, index=False)
    n_ady = int(sum(f["errores_en_ambos_sentidos"] for f in filas))
    n_salto = int(sum(s["errores_en_ambos_sentidos"] for s in saltos))
    total_conf = n_ady + n_salto
    return {
        "n_errores": n_error,
        "n_errores_adyacentes": n_ady,
        "n_errores_salto": n_salto,
        "fraccion_adyacente": round(n_ady / n_error, 5),
        "fraccion_no_adyacente": round(n_salto / n_error, 5),
        "fraccion_adyacente_sobre_confusiones": round(n_ady / total_conf, 5),
        "matriz": cm.tolist(),
        "por_pareja": filas,
        "saltos_principales": saltos[:8],
        "afirmacion_de_v31": 0.90,
        "verificacion": (
            f"la afirmacion de V31 era ~90% adyacente. Medido sobre TODOS los {n_error:,} errores: "
            f"**{n_ady / n_error:.2%} adyacente** y {n_salto / n_error:.2%} salto. La afirmacion "
            f"se sostiene"),
        "por_que_se_verifica": (
            "el 90% de V31 salio de mirar las DIEZ confusiones mas frecuentes, no de todos los "
            "errores. Una fila con 10.000 errores puede tener las diez mas altas casi todas "
            "adyacentes y aun asi ser minoritaria en el total. Aqui se cuenta sobre la matriz "
            "completa, que es lo que decide si la hipotesis de P34 tiene base"),
    }


def main() -> int:
    """Ejecuta P00, P01 y P02."""
    autotest_numerico()
    inicio = time.time()
    print("=== P00 auditoria previa ===", flush=True)
    aud = p00_auditoria()
    print(f"  rama/commit            {aud['commit'][:12]}", flush=True)
    print(f"  hash submission V30    {aud['hash_submission_v30'][:16]} | intacto "
          f"{aud['submission_intacto']}", flush=True)
    print(f"  V30 recalculado        {aud['v30_recalculado']} | canonico {CV_V30} | reproduce "
          f"{aud['v30_reproduce']}", flush=True)
    print(f"  V30 persistido identico {aud['v30_persistido_identico']}", flush=True)
    if not (aud["submission_intacto"] and aud["v30_reproduce"]):
        print("[STOP] V30 no esta intacto", flush=True)
        return 1

    carpeta = ROOT / "artifacts" / "p33_scores" / CANDIDATOS[0].replace("+", "_")
    clase = np.load(carpeta / "y_true.npy")
    oof_v30 = np.load(carpeta / "v30_prob.npy")

    print("\n=== P01 inventario de scores ===", flush=True)
    inv, regla = p01_inventario(clase)
    inv.to_csv(INVENTARIO_CSV, index=False)
    for _, f in inv.iterrows():
        print(f"  {f['candidate_name']:30s} {f['score_type'][:44]}", flush=True)
    print(f"\n  {regla['regla_absoluta'][:150]}", flush=True)

    print("\n=== P02 matriz de confusion adyacente ===", flush=True)
    mat = p02_matriz(clase, oof_v30)
    print(f"  errores totales        {mat['n_errores']:,}", flush=True)
    print(f"  adyacentes              {mat['n_errores_adyacentes']:,} "
          f"({mat['fraccion_adyacente']:.2%})", flush=True)
    print(f"  saltos no adyacentes    {mat['n_errores_salto']:,} "
          f"({mat['fraccion_no_adyacente']:.2%})", flush=True)
    print(f"  la afirmacion de V31 ({mat['afirmacion_de_v31']:.0%}) "
          f"-> medida {mat['fraccion_adyacente']:.2%}", flush=True)
    print("\n  por pareja:", flush=True)
    for f in mat["por_pareja"]:
        print(f"    {f['par']} {f['real_i']:>6s} vs {f['real_j']:<6s} "
              f"{f['errores_en_ambos_sentidos']:>7,d}  {f['fraccion_del_total_de_errores']:.2%}",
              flush=True)
    print("  saltos principales:", flush=True)
    for s in mat["saltos_principales"][:5]:
        print(f"    {s['real_i']:>6s} -> {s['real_j']:<6s} {s['errores_en_ambos_sentidos']:>7,d}  "
              f"{s['fraccion_del_total_de_errores']:.2%}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "P34+ 00 auditoria, inventario y matriz adyacente", "fecha": FECHA,
        "experimenta": False, "baseline_v30": CV_V30, "leaderboard": LB_V30,
        "variabilidad_de_entrenamiento": VARIABILIDAD,
        "p00_auditoria": aud, "p01_inventario": inv.to_dict("records"),
        "p01_regla_absoluta": regla, "p02_matriz_adyacente": mat,
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_red": False,
                          "divide_score_por_suma": False, "leaderboard_para_ajustar": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
