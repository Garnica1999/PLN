#!/usr/bin/env python
"""
P99-12: evaluaciones downstream de S99-01 (60M) vs 30M vs V03.

Pre-registro: reports/p99_preregistration.md seccion 6.
  - Brazos por particion: A=V03 (frozen), B=fusion 30M (reps congeladas de
    P98), C=fusion 60M (reps de S99-01). Fusion identica a D98-1 (TF-IDF del
    fold + bloque denso; LR C=2 balanced; umbrales V03 congelados).
  - Particiones: canonica (folds de oof_v03_final) y la 47 pre-registrada en
    P98 (verificada por hash contra p98_14_preregistro.json). Sin elegir
    particion; metricas separadas; sin concatenar folds.
  - Controles de reproduccion obligatorios: fusion 30M canonica debe dar
    0.7822662599; la 47 usa artefactos congelados de D98-1b.
  - Gate de escala (P99 seccion 7): delta60 vs V03 >= +0.003 canonica; >=4/5
    pliegues; positivo vs 30M; positivo en la 47; neto > 0; sin leakage.

Salidas: reports/p99_downstream_canonica.json y part47.json,
reports/p99_downstream_results.csv, reports/p99_complementarity.csv,
figuras p99_12_*.png.

Uso:
    /home/carlos/venv/bin/python experiments/p99_12_downstream_s99_01.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.embedding_utils import normalizar_emb_normal, tokenizar  # noqa: E402


def _cargar(ruta: Path, nombre: str):
    """Carga un script del proyecto como modulo."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


m13 = _cargar(RAIZ / "experiments" / "p98_13_downstream_d98_1.py", "p98_13")

SAL_CAN = RAIZ / "reports" / "p99_downstream_canonica.json"
SAL_P47 = RAIZ / "reports" / "p99_downstream_part47.json"
SAL_RES = RAIZ / "reports" / "p99_downstream_results.csv"
SAL_COMP = RAIZ / "reports" / "p99_complementarity.csv"
SAL_REPS = RAIZ / "features" / "p99_reps_dev.npz"
SAL_FIG_CAN = RAIZ / "reports" / "figuras" / "p99_12_canonica.png"
SAL_FIG_47 = RAIZ / "reports" / "figuras" / "p99_12_part47.png"

UMBRALES_V03 = m13.UMBRALES_V03
SEMILLA = m13.SEMILLA
RUTA_REPS_30 = RAIZ / "features" / "p98_13_reps_dev.npz"
RUTA_OOF_47 = RAIZ / "features" / "p98_14_oof_b.npz"
RUTA_PRE_47 = RAIZ / "reports" / "p98_14_preregistro.json"
REF_FUSION_30_CANONICA = 0.7822662599


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def cargar_lm_p99(direccion: str, dispositivo):
    """Carga un checkpoint del LM S99-01."""
    ruta = RAIZ / "models" / f"p99_s99_01_{direccion}.pt"
    datos = torch.load(ruta, map_location=dispositivo, weights_only=True)
    modelo = m13.LMReal(datos["cfg"]).to(dispositivo)
    modelo.load_state_dict(datos["state_dict"])
    modelo.eval()
    return modelo


def construir_reps_60(dispositivo) -> dict:
    """Representaciones 2048-d de S99-01 (misma receta de D98-1)."""
    vocab = json.loads((RAIZ / "features" / "p98_c98_1_vocab.json").read_text(
        encoding="utf-8"))
    word_id = {tok: i + 1 for i, tok in enumerate(vocab["vocab_top"])}
    char_id = {ch: i + 1 for i, ch in enumerate(vocab["chars"])}
    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    train = data_module.read_train()
    mapa = train.assign(_id=train["id"].astype(int)).set_index("_id")
    textos = mapa.loc[ids, "text"].tolist()
    tokenizados = [tokenizar(normalizar_emb_normal(t)) for t in textos]
    largos = np.array([len(t) for t in tokenizados])
    orden = np.argsort(largos)
    reps = np.zeros((len(ids), 2048), dtype=np.float32)
    modelo_ltr = cargar_lm_p99("ltr", dispositivo)
    modelo_rtl = cargar_lm_p99("rtl", dispositivo)
    for inicio in range(0, len(orden), 64):
        idx = orden[inicio:inicio + 64]
        ids_f, chs_f, ids_b, chs_b = [], [], [], []
        for i in idx.tolist():
            a, b = m13.codificar_tweet(tokenizados[i], word_id, char_id, False)
            c, d = m13.codificar_tweet(tokenizados[i], word_id, char_id, True)
            ids_f.append(a)
            chs_f.append(b)
            ids_b.append(c)
            chs_b.append(d)
        hf = m13.extraer_reps(modelo_ltr, ids_f, chs_f, dispositivo)
        hb = m13.extraer_reps(modelo_rtl, ids_b, chs_b, dispositivo)
        for k, i in enumerate(idx.tolist()):
            L = len(tokenizados[i])
            rep_ltr = hf[k][:L]
            rep_rtl = hb[k][:L][::-1]
            reps[i] = np.concatenate([rep_ltr, rep_rtl], axis=1).mean(axis=0)
    return {"ids": ids, "reps": reps, "textos": textos,
            "tokenizados": tokenizados}


def folds_part47(y: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    """Recrea la particion 47 y verifica su hash contra el pre-registro."""
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y])
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues = np.full(len(y), -1, dtype=int)
    for numero, (_, val) in enumerate(divisor.split(
            np.arange(len(y)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues[val] = numero + 1
    assert (pliegues >= 1).all()
    sha = hashlib.sha256(pliegues.astype("int64").tobytes()).hexdigest()
    previo = json.loads(RUTA_PRE_47.read_text(encoding="utf-8"))
    assert previo["particion"]["sha256_pliegues"] == sha, \
        "la particion 47 no reproduce el pre-registro"
    return pliegues


def resumen(pred, y, pliegues, nombre, referencia=None) -> dict:
    """Metricas de una regla en una particion."""
    met = m13.metricas_multietiqueta(y, pred)
    por_pliegue = [float(m13.metricas_multietiqueta(
        y[pliegues == k], pred[pliegues == k])["f1_media_columnas"])
        for k in sorted(np.unique(pliegues).tolist())]
    r = {"regla": nombre, "corrected_cv": met["f1_media_columnas"],
         "f1_por_etiqueta": met["f1_por_etiqueta"],
         "f1_pliegues": por_pliegue,
         "std_pliegues": float(np.std(por_pliegue, ddof=1))}
    if referencia is not None:
        r["correccion_vs_ref"] = m13.correccion(pred, referencia, y)
    return r


def main() -> None:
    """Ejecuta las dos evaluaciones downstream."""
    for ruta in (SAL_CAN, SAL_P47, SAL_RES, SAL_COMP, SAL_REPS, SAL_FIG_CAN,
                 SAL_FIG_47):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    if not torch.cuda.is_available():
        raise SystemExit("GPU/ROCm no disponible")
    dispositivo = torch.device("cuda")
    inicio = time.perf_counter()

    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    y = z["y_true"].astype(int)
    pliegues_can = z["pliegues"].astype(int)
    pred_v03_can = z["pred"].astype(int)
    grupos = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz",
                     allow_pickle=True)["grupos"].astype(np.int64)

    reps30 = np.load(RUTA_REPS_30, allow_pickle=True)["reps"]
    print("extrayendo reps de S99-01 (60M)...", flush=True)
    d60 = construir_reps_60(dispositivo)
    reps60 = d60["reps"]
    textos = d60["textos"]
    tokenizados = d60["tokenizados"]
    np.savez_compressed(SAL_REPS, ids=d60["ids"], reps=reps60)
    print(f"  reps60: {reps60.shape} | sha "
          f"{sha256(SAL_REPS)[:16]} ({round(time.perf_counter()-inicio,1)} s)",
          flush=True)

    # ---------- canonica ----------
    print("canonica: controles y fusiones...", flush=True)
    fus30 = m13.oof_por_pliegue(reps30, textos, y, pliegues_can, "fusion")
    c30 = m13.metricas_multietiqueta(y, (fus30 >= UMBRALES_V03).astype(int))
    control_ok = abs(c30["f1_media_columnas"] - REF_FUSION_30_CANONICA) < 1e-6
    assert control_ok, f"control 30M no reproduce: {c30['f1_media_columnas']}"
    emb30 = m13.oof_por_pliegue(reps30, textos, y, pliegues_can, "embedding")
    emb60 = m13.oof_por_pliegue(reps60, textos, y, pliegues_can, "embedding")
    fus60 = m13.oof_por_pliegue(reps60, textos, y, pliegues_can, "fusion")

    pred_v03c = (emb30 * 0) + pred_v03_can  # alias claro
    r_v03 = resumen(pred_v03_can, y, pliegues_can, "v03_frozen")
    r_f30 = resumen((fus30 >= UMBRALES_V03).astype(int), y, pliegues_can,
                    "fusion_30M", pred_v03_can)
    r_f60 = resumen((fus60 >= UMBRALES_V03).astype(int), y, pliegues_can,
                    "fusion_60M", pred_v03_can)
    r_e30 = resumen((emb30 >= UMBRALES_V03).astype(int), y, pliegues_can,
                    "embedding_30M")
    r_e60 = resumen((emb60 >= UMBRALES_V03).astype(int), y, pliegues_can,
                    "embedding_60M")
    pred_f60 = (fus60 >= UMBRALES_V03).astype(int)
    pred_f30 = (fus30 >= UMBRALES_V03).astype(int)
    delta60_v03 = r_f60["corrected_cv"] - r_v03["corrected_cv"]
    delta60_30 = r_f60["corrected_cv"] - r_f30["corrected_cv"]
    folds60 = int(np.sum(np.array(r_f60["f1_pliegues"])
                         > np.array(r_v03["f1_pliegues"]) + 1e-12))
    boot60_v03 = m13.bootstrap_clusters_delta(y, pred_f60, pred_v03_can, grupos)
    boot60_30 = m13.bootstrap_clusters_delta(y, pred_f60, pred_f30, grupos)
    canonica = {
        "particion": "canonica (folds de oof_v03_final)",
        "reglas": {"v03": r_v03, "fusion_30M": r_f30, "fusion_60M": r_f60,
                   "embedding_30M": r_e30, "embedding_60M": r_e60},
        "control_reproduccion_30M": {"valor": c30["f1_media_columnas"],
                                     "esperado": REF_FUSION_30_CANONICA,
                                     "ok": True},
        "comparacion": {"delta60_vs_v03": round(delta60_v03, 6),
                        "pliegues60_mejorados_vs_v03": folds60,
                        "delta60_vs_30": round(delta60_30, 6),
                        "neto60_vs_v03": r_f60["correccion_vs_ref"]["neto"],
                        "neto60_vs_30": m13.correccion(pred_f60, pred_f30, y)["neto"],
                        "bootstrap_60_vs_v03": boot60_v03,
                        "bootstrap_60_vs_30": boot60_30},
        "tiempo_total_s": round(time.perf_counter() - inicio, 1),
    }

    # ---------- particion 47 ----------
    print("particion 47: verificando y evaluando...", flush=True)
    pliegues_47 = folds_part47(y, grupos)
    oof47 = np.load(RUTA_OOF_47, allow_pickle=True)
    fus30_47 = oof47["proba_fus"]
    pred_v03_47 = oof47["pred_v03"]
    c30_47 = m13.metricas_multietiqueta(y, (fus30_47 >= UMBRALES_V03).astype(int))
    assert abs(c30_47["f1_media_columnas"] - 0.7767830429) < 1e-6
    fus60_47 = m13.oof_por_pliegue(reps60, textos, y, pliegues_47, "fusion")
    pred_f60_47 = (fus60_47 >= UMBRALES_V03).astype(int)
    r_v03_47 = resumen(pred_v03_47, y, pliegues_47, "v03_equivalente_47")
    r_f30_47 = resumen((fus30_47 >= UMBRALES_V03).astype(int), y, pliegues_47,
                       "fusion_30M_47", pred_v03_47)
    r_f60_47 = resumen(pred_f60_47, y, pliegues_47, "fusion_60M_47",
                       pred_v03_47)
    delta60_v03_47 = r_f60_47["corrected_cv"] - r_v03_47["corrected_cv"]
    delta60_30_47 = r_f60_47["corrected_cv"] - r_f30_47["corrected_cv"]
    folds60_47 = int(np.sum(np.array(r_f60_47["f1_pliegues"])
                            > np.array(r_v03_47["f1_pliegues"]) + 1e-12))
    boot60_v03_47 = m13.bootstrap_clusters_delta(y, pred_f60_47, pred_v03_47,
                                                 grupos)
    part47 = {
        "particion": "47 (seed 43; hash verificado contra p98_14_preregistro)",
        "reglas": {"v03_equivalente_47": r_v03_47, "fusion_30M_47": r_f30_47,
                   "fusion_60M_47": r_f60_47},
        "control_reproduccion_30M_47": {"valor": c30_47["f1_media_columnas"],
                                        "esperado": 0.7767830429, "ok": True},
        "comparacion": {"delta60_vs_v03": round(delta60_v03_47, 6),
                        "pliegues60_mejorados_vs_v03": folds60_47,
                        "delta60_vs_30": round(delta60_30_47, 6),
                        "neto60_vs_v03": r_f60_47["correccion_vs_ref"]["neto"],
                        "bootstrap_60_vs_v03": boot60_v03_47},
        "tiempo_total_s": round(time.perf_counter() - inicio, 1),
    }

    # ---------- gate de escala ----------
    condiciones = {
        "delta60_vs_v03_canonica_ge_0003": delta60_v03 >= 0.003,
        "pliegues_canonica_ge_4": folds60 >= 4,
        "positivo_vs_30M": delta60_30 > 0,
        "positivo_en_particion_47": delta60_v03_47 > 0,
        "neto_positivo_canonica": r_f60["correccion_vs_ref"]["neto"] > 0,
        "neto_positivo_47": r_f60_47["correccion_vs_ref"]["neto"] > 0,
    }
    if all(condiciones.values()):
        veredicto = "GATE_PASSED"
    else:
        veredicto = "GATE_FAILED"
    gate = {"condiciones": condiciones, "veredicto": veredicto,
            "nota": "cumplir el gate permite proponer el siguiente "
                    "experimento; no autoriza adopcion"}
    canonica["gate"] = gate
    part47["gate"] = gate
    canonica["experiment_id"] = "P99-13"
    part47["experiment_id"] = "P99-14"

    SAL_CAN.write_text(json.dumps(canonica, indent=2, ensure_ascii=True) + "\n",
                       encoding="utf-8")
    SAL_P47.write_text(json.dumps(part47, indent=2, ensure_ascii=True) + "\n",
                       encoding="utf-8")

    filas = ["particion,regla,corrected_cv,delta_vs_v03,delta_vs_30M,f1_HS,"
             "f1_TR,f1_AG,folds_positivos,neto"]
    filas.append(f"canonica,v03_frozen,{r_v03['corrected_cv']:.10f},0,"
                 f"{r_v03['corrected_cv'] - r_f30['corrected_cv']:+.6f},"
                 f"{r_v03['f1_por_etiqueta']['HS']:.6f},"
                 f"{r_v03['f1_por_etiqueta']['TR']:.6f},"
                 f"{r_v03['f1_por_etiqueta']['AG']:.6f},-,")
    for r, nombre in ((r_f30, "fusion_30M"), (r_f60, "fusion_60M"),
                      (r_e30, "embedding_30M"), (r_e60, "embedding_60M")):
        d30 = r["corrected_cv"] - r_f30["corrected_cv"]
        filas.append(f"canonica,{nombre},{r['corrected_cv']:.10f},"
                     f"{r['corrected_cv'] - r_v03['corrected_cv']:+.6f},"
                     f"{d30:+.6f},{r['f1_por_etiqueta']['HS']:.6f},"
                     f"{r['f1_por_etiqueta']['TR']:.6f},"
                     f"{r['f1_por_etiqueta']['AG']:.6f},-,"
                     f"{r.get('correccion_vs_ref', {}).get('neto', '-')}")
    filas.append(f"part47,v03_equivalente_47,{r_v03_47['corrected_cv']:.10f},"
                 f"0,{r_v03_47['corrected_cv'] - r_f30_47['corrected_cv']:+.6f},"
                 f"{r_v03_47['f1_por_etiqueta']['HS']:.6f},"
                 f"{r_v03_47['f1_por_etiqueta']['TR']:.6f},"
                 f"{r_v03_47['f1_por_etiqueta']['AG']:.6f},-,")
    for r, nombre in ((r_f30_47, "fusion_30M_47"), (r_f60_47, "fusion_60M_47")):
        filas.append(f"part47,{nombre},{r['corrected_cv']:.10f},"
                     f"{r['corrected_cv'] - r_v03_47['corrected_cv']:+.6f},"
                     f"{r['corrected_cv'] - r_f30_47['corrected_cv']:+.6f},"
                     f"{r['f1_por_etiqueta']['HS']:.6f},"
                     f"{r['f1_por_etiqueta']['TR']:.6f},"
                     f"{r['f1_por_etiqueta']['AG']:.6f},-,"
                     f"{r.get('correccion_vs_ref', {}).get('neto', '-')}")
    SAL_RES.write_text("\n".join(filas) + "\n", encoding="utf-8")

    textos_norm = [normalizar_emb_normal(t) for t in textos]
    largo_tokens = np.array([len(t.split()) for t in textos_norm])
    slices = {"longitud_q4": largo_tokens >= np.quantile(largo_tokens, 0.75),
              "longitud_q1": largo_tokens <= np.quantile(largo_tokens, 0.25),
              "url": np.array([" url " in " " + t + " " for t in textos_norm]),
              "hashtag": np.array([" hashtag " in " " + t + " "
                                   for t in textos_norm]),
              "mencion": np.array([" usuario " in " " + t + " "
                                   for t in textos_norm]),
              "elongaciones": np.array([bool(re.search(r"(\w)\1{2,}", t))
                                        for t in textos])}
    lineas = ["particion,slice,n,f1_v03,f1_60M,delta,rescatadas,rotas,neto"]
    for etiqueta, pl, pv, p60 in (
            ("canonica", pliegues_can, pred_v03_can, pred_f60),
            ("part47", pliegues_47, pred_v03_47, pred_f60_47)):
        for nombre, mascara in slices.items():
            corr = m13.correccion(p60[mascara], pv[mascara], y[mascara])
            f1a = float(m13.metricas_multietiqueta(y[mascara], pv[mascara])
                        ["f1_media_columnas"])
            f1b = float(m13.metricas_multietiqueta(y[mascara], p60[mascara])
                        ["f1_media_columnas"])
            lineas.append(f"{etiqueta},{nombre},{int(mascara.sum())},"
                          f"{f1a:.6f},{f1b:.6f},{f1b-f1a:+.6f},"
                          f"{corr['rescatadas']},{corr['rotas']},{corr['neto']}")
    SAL_COMP.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ruta_fig, pliegues, r_a, r_b, titulo in (
            (SAL_FIG_CAN, pliegues_can, r_v03, r_f60, "Canonica"),
            (SAL_FIG_47, pliegues_47, r_v03_47, r_f60_47, "Particion 47")):
        fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
        xs = np.arange(5)
        ejes[0].bar(xs - 0.18, r_a["f1_pliegues"], 0.36, label="V03")
        ejes[0].bar(xs + 0.18, r_b["f1_pliegues"], 0.36, label="fusion 60M")
        ejes[0].set_xticks(xs, [f"p{k}" for k in range(1, 6)])
        ejes[0].set_title(f"F1 por pliegue ({titulo})")
        ejes[0].legend(fontsize=8)
        ejes[1].bar(["delta 60 vs V03", "delta 60 vs 30M"],
                    [r_b["corrected_cv"] - r_a["corrected_cv"],
                     delta60_30 if titulo == "Canonica" else delta60_30_47])
        ejes[1].axhline(0, color="black", lw=0.8)
        ejes[1].axhline(0.003, color="red", lw=0.8, ls="--")
        ejes[1].set_title("Deltas (linea roja: +0.003)")
        fig.tight_layout()
        fig.savefig(ruta_fig, dpi=130)
        plt.close(fig)

    print(f"D98-style downstream completo:", flush=True)
    print(f"  canonica: fusion60 {r_f60['corrected_cv']:.6f} vs V03 "
          f"{r_v03['corrected_cv']:.6f} (delta {delta60_v03:+.6f}, {folds60}/5) "
          f"vs 30M {r_f30['corrected_cv']:.6f} (delta {delta60_30:+.6f})",
          flush=True)
    print(f"  part47: fusion60 {r_f60_47['corrected_cv']:.6f} vs v03eq "
          f"{r_v03_47['corrected_cv']:.6f} (delta {delta60_v03_47:+.6f}, "
          f"{folds60_47}/5) vs 30M {r_f30_47['corrected_cv']:.6f} "
          f"(delta {delta60_30_47:+.6f})", flush=True)
    print(f"  GATE: {veredicto} | condiciones: {condiciones}", flush=True)


if __name__ == "__main__":
    main()
