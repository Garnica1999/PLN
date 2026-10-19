#!/usr/bin/env python
"""
P98-13 (Fase B, D98-1): evaluacion downstream de las representaciones C98-1.

Pre-registro: reports/p98_architecture.md (seccion 6):
  - Reps congeladas: concat(capa 2 LTR, capa 2 RTL) por token; pooling primario
    = media sobre las posiciones de token (2048-d); sin busqueda de pooling.
  - Embedding-only: 3 LR binarias (balanced, C=1.0, liblinear) por pliegue.
  - Fusion estilo v5.1: TF-IDF del pliegue (word 1-2 + char_wb 3-5, min_df=2,
    sublinear) + bloque denso 2048; LR C=2.0 balanced; umbrales V03 congelados.
  - Comparaciones: V03; referencia estatica v5.1 (citada, no reentrenada);
    salvage/damage/neto; bootstrap por clusters (unidad = grupo).

Salidas: features/p98_13_reps_dev.npz, features/p98_13_oof.npz,
reports/p98_13_downstream.json y .md, reports/p98_downstream_results.csv,
reports/p98_complementarity.csv, figura reports/figuras/p98_13_downstream.png.

Uso:
    /home/carlos/venv/bin/python experiments/p98_13_downstream_d98_1.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.embedding_utils import normalizar_emb_normal, tokenizar  # noqa: E402
from src.evaluation import metrica_macro, metricas_multietiqueta  # noqa: E402


def _cargar_modulo(ruta: Path):
    """Carga un script del proyecto como modulo."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


LMReal = _cargar_modulo(RAIZ / "experiments" / "p98_10_piloto_c98_0.py").LMReal

SAL_REPS = RAIZ / "features" / "p98_13_reps_dev.npz"
SAL_OOF = RAIZ / "features" / "p98_13_oof.npz"
SAL_JSON = RAIZ / "reports" / "p98_13_downstream.json"
SAL_MD = RAIZ / "reports" / "p98_13_downstream.md"
SAL_RES = RAIZ / "reports" / "p98_downstream_results.csv"
SAL_COMP = RAIZ / "reports" / "p98_complementarity.csv"
SAL_FIG = RAIZ / "reports" / "figuras" / "p98_13_downstream.png"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA = 20260925
EOS = "<eos>"


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def cargar_lm(direccion: str, dispositivo) -> tuple:
    """Carga un checkpoint del LM C98-1."""
    ruta = RAIZ / "models" / f"p98_c98_1_{direccion}.pt"
    datos = torch.load(ruta, map_location=dispositivo, weights_only=True)
    modelo = LMReal(datos["cfg"]).to(dispositivo)
    modelo.load_state_dict(datos["state_dict"])
    modelo.eval()
    return modelo, datos


def hidden_capa2(modelo, palabras: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
    """Estados ocultos de la 2a capa (o 1a si capas=1), en eval."""
    b, s, el = chars.shape
    c = modelo.char_emb(chars.reshape(b * s, el)).transpose(1, 2)
    c = torch.cat([torch.relu(modelo.conv3(c)).max(dim=2).values,
                   torch.relu(modelo.conv5(c)).max(dim=2).values], dim=1)
    entrada = torch.cat([modelo.word_emb(palabras), c.reshape(b, s, -1)], dim=2)
    h, _ = modelo.lstm1(torch.relu(modelo.proj(entrada)))
    if modelo.lstm2 is not None:
        h, _ = modelo.lstm2(h)
    return h


def codificar_tweet(tokens: list[str], word_id: dict, char_id: dict,
                    invertir: bool) -> tuple[np.ndarray, np.ndarray]:
    """ids de palabra (OOV=0) y filas de chars para un tweet (+EOS)."""
    sec = (tokens[::-1] if invertir else tokens) + [EOS]
    ids = np.zeros(len(sec), dtype=np.int64)
    chs = np.zeros((len(sec), 24), dtype=np.uint8)
    for i, tok in enumerate(sec):
        ids[i] = word_id.get(tok, 0)
        for j, ch in enumerate(tok[:24]):
            chs[i, j] = char_id.get(ch, 0)
    return ids, chs


def extraer_reps(modelo, lote_ids: list[np.ndarray], lote_chs: list[np.ndarray],
                 dispositivo) -> list[np.ndarray]:
    """Estados capa2 por tweet (sin pad), en GPU."""
    largos = [len(x) for x in lote_ids]
    maxlen = max(largos)
    palabras = torch.zeros((len(lote_ids), maxlen), dtype=torch.long)
    chars = torch.zeros((len(lote_ids), maxlen, 24), dtype=torch.long)
    for i, (ids, chs) in enumerate(zip(lote_ids, lote_chs)):
        palabras[i, :len(ids)] = torch.from_numpy(ids)
        chars[i, :len(chs)] = torch.from_numpy(chs.astype(np.int64))
    with torch.no_grad():
        h = hidden_capa2(modelo, palabras.to(dispositivo), chars.to(dispositivo))
    h = h.cpu().numpy()
    return [h[i, :largos[i]] for i in range(len(lote_ids))]


def construir_reps_dev(dispositivo) -> dict:
    """Representaciones 2048-d (media LTR+RTL sobre posiciones de token)."""
    vocab = json.loads((RAIZ / "features" / "p98_c98_1_vocab.json").read_text(
        encoding="utf-8"))
    word_id = {tok: i + 1 for i, tok in enumerate(vocab["vocab_top"])}
    char_id = {ch: i + 1 for i, ch in enumerate(vocab["chars"])}

    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    y = z["y_true"].astype(int)
    pliegues = z["pliegues"].astype(int)
    pred_v03 = z["pred"].astype(int)
    train = data_module.read_train()
    mapa = train.assign(_id=train["id"].astype(int)).set_index("_id")
    textos = mapa.loc[ids, "text"].tolist()

    tokenizados = [tokenizar(normalizar_emb_normal(t)) for t in textos]
    largos = np.array([len(t) for t in tokenizados])
    orden = np.argsort(largos)
    reps = np.zeros((len(ids), 2048), dtype=np.float32)
    modelo_ltr, _ = cargar_lm("ltr", dispositivo)
    modelo_rtl, _ = cargar_lm("rtl", dispositivo)
    lote_max = 64
    for inicio in range(0, len(orden), lote_max):
        idx = orden[inicio:inicio + lote_max]
        ids_f, chs_f = [], []
        ids_b, chs_b = [], []
        for i in idx.tolist():
            a, b = codificar_tweet(tokenizados[i], word_id, char_id, False)
            ids_f.append(a[:])
            chs_f.append(b[:])
            a2, b2 = codificar_tweet(tokenizados[i], word_id, char_id, True)
            ids_b.append(a2[:])
            chs_b.append(b2[:])
        hf = extraer_reps(modelo_ltr, ids_f, chs_f, dispositivo)
        hb = extraer_reps(modelo_rtl, ids_b, chs_b, dispositivo)
        for k, i in enumerate(idx.tolist()):
            L = len(tokenizados[i])
            rep_ltr = hf[k][:L]                     # posiciones de token (sin EOS)
            rep_rtl = hb[k][:L][::-1]               # des-invertida
            reps[i] = np.concatenate([rep_ltr, rep_rtl], axis=1).mean(axis=0)
    return {"ids": ids, "y_true": y, "pliegues": pliegues, "pred_v03": pred_v03,
            "reps": reps, "tokenizados": tokenizados, "textos": textos}


def lr_binaria() -> LogisticRegression:
    """Clasificador pre-registrado para las reps (embedding-only)."""
    return LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                              solver="liblinear", random_state=42)


def lr_fusion() -> LogisticRegression:
    """Clasificador pre-registrado para la fusion v5.1-style."""
    return LogisticRegression(class_weight="balanced", C=2.0, max_iter=1000,
                              solver="liblinear", random_state=42)


def oof_por_pliegue(reps, textos, y, pliegues, modo: str) -> np.ndarray:
    """OOF de la cabeza (embedding-only o fusion) por pliegues canonicos."""
    proba = np.zeros((len(y), 3))
    for k in sorted(np.unique(pliegues).tolist()):
        idx_tr = np.flatnonzero(pliegues != k)
        idx_va = np.flatnonzero(pliegues == k)
        if modo == "embedding":
            x_tr, x_va = reps[idx_tr], reps[idx_va]
            fabrica = lr_binaria
        else:
            palabra = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                      min_df=2, sublinear_tf=True)
            caracter = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                       min_df=2, sublinear_tf=True)
            textos_tr = [textos[i] for i in idx_tr]
            textos_va = [textos[i] for i in idx_va]
            m_tr = sparse.hstack([palabra.fit_transform(textos_tr),
                                  caracter.fit_transform(textos_tr),
                                  sparse.csr_matrix(reps[idx_tr])]).tocsr()
            m_va = sparse.hstack([palabra.transform(textos_va),
                                  caracter.transform(textos_va),
                                  sparse.csr_matrix(reps[idx_va])]).tocsr()
            x_tr, x_va = m_tr, m_va
            fabrica = lr_fusion
        for etiqueta in range(3):
            modelo = fabrica()
            modelo.fit(x_tr, y[idx_tr, etiqueta])
            proba[idx_va, etiqueta] = modelo.predict_proba(x_va)[:, 1]
    return proba


def correccion(pred_a, pred_b, y) -> dict:
    """Salvage/damage/neto por celda de pred_a frente a pred_b."""
    ok_a = pred_a == y
    ok_b = pred_b == y
    rescatadas = int((~ok_b & ok_a).sum())
    rotas = int((ok_b & ~ok_a).sum())
    return {"rescatadas": rescatadas, "rotas": rotas, "neto": rescatadas - rotas}


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    """IC95 del delta pooled (a-b) por clusters."""
    generador = np.random.default_rng(SEMILLA)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        dif[r] = metrica_macro(y[idx], pred_a[idx]) - metrica_macro(y[idx], pred_b[idx])
    c = np.quantile(dif, [0.025, 0.975])
    return {"ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
            "pct_mejora": round(float((dif > 0).mean()), 6)}


def main() -> None:
    """Ejecuta D98-1 completo."""
    for ruta in (SAL_REPS, SAL_OOF, SAL_JSON, SAL_MD, SAL_RES, SAL_COMP, SAL_FIG):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    if not torch.cuda.is_available():
        raise SystemExit("GPU/ROCm no disponible")
    dispositivo = torch.device("cuda")
    inicio = time.perf_counter()

    print("extrayendo representaciones de desarrollo...", flush=True)
    d = construir_reps_dev(dispositivo)
    reps, y, pliegues = d["reps"], d["y_true"], d["pliegues"]
    print(f"  reps: {reps.shape} ({round(time.perf_counter()-inicio,1)} s)",
          flush=True)
    np.savez_compressed(SAL_REPS, ids=d["ids"], y_true=y, pliegues=pliegues,
                        grupos=np.load(RAIZ / "features" /
                                       "v2_oof_base_corrected.npz",
                                       allow_pickle=True)["grupos"],
                        reps=reps)
    grupos = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz",
                     allow_pickle=True)["grupos"].astype(np.int64)

    print("embedding-only (LR por pliegue)...", flush=True)
    proba_emb = oof_por_pliegue(reps, d["textos"], y, pliegues, "embedding")
    print("fusion estilo v5.1 (TF-IDF + denso)...", flush=True)
    proba_fus = oof_por_pliegue(reps, d["textos"], y, pliegues, "fusion")

    pred_emb = (proba_emb >= UMBRALES_V03).astype(int)
    pred_fus = (proba_fus >= UMBRALES_V03).astype(int)
    pred_v03 = d["pred_v03"]

    def resumen_regla(pred, nombre):
        met = metricas_multietiqueta(y, pred)
        por_pliegue = [float(metricas_multietiqueta(
            y[pliegues == k], pred[pliegues == k])["f1_media_columnas"])
            for k in sorted(np.unique(pliegues).tolist())]
        return {"regla": nombre, "corrected_cv": met["f1_media_columnas"],
                "f1_por_etiqueta": met["f1_por_etiqueta"],
                "f1_pliegues": por_pliegue,
                "std_pliegues": float(np.std(por_pliegue, ddof=1)),
                "correccion_vs_v03": correccion(pred, pred_v03, y)}

    r_v03 = resumen_regla(pred_v03, "v03_frozen")
    r_emb = resumen_regla(pred_emb, "embedding_only")
    r_fus = resumen_regla(pred_fus, "fusion_v51_style")

    delta_fus = r_fus["corrected_cv"] - r_v03["corrected_cv"]
    pliegues_mej_fus = int(np.sum(np.array(r_fus["f1_pliegues"])
                                  > np.array(r_v03["f1_pliegues"]) + 1e-12))
    boot_fus = bootstrap_clusters_delta(y, pred_fus, pred_v03, grupos)
    delta_emb = r_emb["corrected_cv"] - r_v03["corrected_cv"]

    largo_tokens = np.array([len(t) for t in d["tokenizados"]])
    urls = np.array([(" url " in " " + " ".join(t) + " ") for t in d["tokenizados"]])
    menciones = np.array([(" usuario " in " " + " ".join(t) + " ")
                          for t in d["tokenizados"]])
    hashtags = np.array([(" hashtag " in " " + " ".join(t) + " ")
                         for t in d["tokenizados"]])
    elong = np.array([bool(re.search(r"(\w)\1{2,}", t)) for t in d["textos"]])
    q = np.quantile(largo_tokens, [0.25, 0.5, 0.75])
    cuartil = np.digitize(largo_tokens, q)
    slices = {"url": urls, "mencion": menciones, "hashtag": hashtags,
              "elongaciones": elong}
    for c in range(4):
        slices[f"longitud_q{c+1}"] = cuartil == c

    filas_comp = []
    for nombre, mascara in slices.items():
        n = int(mascara.sum())
        if n == 0:
            continue
        corr = correccion(pred_fus[mascara], pred_v03[mascara], y[mascara])
        f1_v03 = float(metricas_multietiqueta(y[mascara], pred_v03[mascara])
                       ["f1_media_columnas"])
        f1_fus = float(metricas_multietiqueta(y[mascara], pred_fus[mascara])
                       ["f1_media_columnas"])
        filas_comp.append({"slice": nombre, "n": n,
                           "f1_v03": round(f1_v03, 6),
                           "f1_fusion": round(f1_fus, 6),
                           "delta": round(f1_fus - f1_v03, 6),
                           "rescatadas": corr["rescatadas"],
                           "rotas": corr["rotas"], "neto": corr["neto"]})
    lineas = ["slice,n,f1_v03,f1_fusion,delta,rescatadas,rotas,neto"]
    for f in filas_comp:
        lineas.append(f"{f['slice']},{f['n']},{f['f1_v03']},{f['f1_fusion']},"
                      f"{f['delta']},{f['rescatadas']},{f['rotas']},{f['neto']}")
    SAL_COMP.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    if delta_fus >= 0.003 and pliegues_mej_fus >= 4 and boot_fus["ic95"][0] > 0:
        veredicto = "PROMISING"
    elif delta_fus > 0 and pliegues_mej_fus >= 4:
        veredicto = "MARGINAL_SENAL"
    elif delta_fus > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "INVALIDATED"
    resultado = {
        "experiment_id": "P98-13", "etapa": "D98-1 (downstream)",
        "fecha": "2026-10-13",
        "reglas": {"v03_frozen": r_v03, "embedding_only": r_emb,
                   "fusion_v51_style": r_fus},
        "comparaciones": {
            "fusion_vs_v03": {"delta": round(delta_fus, 6),
                              "pliegues_mejorados": pliegues_mej_fus,
                              "bootstrap_clusters": boot_fus},
            "embedding_only_vs_v03": {"delta": round(delta_emb, 6)},
            "referencia_estatica_v51": {"embedding_only": 0.6523,
                                        "fusion": 0.7818,
                                        "fuente": "reports/embedding_architecture_v5_1.md"},
        },
        "slices": filas_comp,
        "veredicto": veredicto,
        "tiempo_total_s": round(time.perf_counter() - inicio, 1),
    }
    np.savez_compressed(SAL_OOF, ids=d["ids"], y_true=y, pliegues=pliegues,
                        proba_emb=proba_emb, proba_fus=proba_fus,
                        pred_emb=pred_emb, pred_fus=pred_fus, pred_v03=pred_v03)
    SAL_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")

    filas_csv = ["regla,corrected_cv,delta_vs_v03,f1_HS,f1_TR,f1_AG,std_pliegues"]
    for r, nombre in ((r_v03, "v03_frozen"), (r_emb, "embedding_only"),
                      (r_fus, "fusion_v51_style")):
        filas_csv.append(f"{nombre},{r['corrected_cv']:.10f},"
                         f"{r['corrected_cv'] - r_v03['corrected_cv']:+.6f},"
                         f"{r['f1_por_etiqueta']['HS']:.6f},"
                         f"{r['f1_por_etiqueta']['TR']:.6f},"
                         f"{r['f1_por_etiqueta']['AG']:.6f},"
                         f"{r['std_pliegues']:.5f}")
    SAL_RES.write_text("\n".join(filas_csv) + "\n", encoding="utf-8")

    md = [
        "# P98-13 (D98-1): evaluacion downstream de las representaciones C98-1",
        "",
        "Pre-registro: `p98_architecture.md` seccion 6. Reps congeladas "
        "(concat capa2 LTR+RTL, media sobre tokens, 2048-d); umbrales V03.",
        "",
        "| Regla | corrected_cv | delta vs V03 | F1 HS | F1 TR | F1 AG |",
        "|---|---|---|---|---|---|",
    ]
    for r, nombre in ((r_v03, "V03 frozen"), (r_emb, "embedding-only (LR C=1)"),
                      (r_fus, "fusion v5.1-style (TF-IDF + denso)")):
        md.append(f"| {nombre} | {r['corrected_cv']:.10f} | "
                  f"{r['corrected_cv'] - r_v03['corrected_cv']:+.6f} | "
                  f"{r['f1_por_etiqueta']['HS']:.4f} | "
                  f"{r['f1_por_etiqueta']['TR']:.4f} | "
                  f"{r['f1_por_etiqueta']['AG']:.4f} |")
    md += [
        "",
        f"- Fusion vs V03: pliegues mejorados {pliegues_mej_fus}/5; IC95 clusters "
        f"{boot_fus['ic95']}; neto de celdas "
        f"{r_fus['correccion_vs_v03']['neto']:+d}.",
        f"- Referencia estatica v5.1 (citada): embedding-only 0.6523; fusion "
        f"0.7818.",
        "",
        f"**Veredicto: {veredicto}** (gate: delta >= +0.003, >=4/5 pliegues, IC "
        "sin cero).",
    ]
    SAL_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(5)
    ejes[0].bar(xs - 0.18, r_v03["f1_pliegues"], 0.36, label="V03")
    ejes[0].bar(xs + 0.18, r_fus["f1_pliegues"], 0.36, label="fusion")
    ejes[0].set_xticks(xs, [f"p{k}" for k in sorted(np.unique(pliegues).tolist())])
    ejes[0].set_title("F1 por pliegue (umbrales V03)")
    ejes[0].legend(fontsize=8)
    nombres_s = [f["slice"] for f in filas_comp]
    ejes[1].bar(nombres_s, [f["neto"] for f in filas_comp])
    ejes[1].axhline(0, color="black", lw=0.8)
    ejes[1].set_title("Neto de celdas por slice (fusion vs V03)")
    ejes[1].tick_params(axis="x", rotation=40)
    fig.tight_layout()
    SAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAL_FIG, dpi=130)
    plt.close(fig)

    print(f"D98-1: embedding-only {r_emb['corrected_cv']:.6f} | fusion "
          f"{r_fus['corrected_cv']:.6f} (delta {delta_fus:+.6f}, "
          f"{pliegues_mej_fus}/5) | veredicto {veredicto}", flush=True)
    print(f"tiempo: {resultado['tiempo_total_s']} s")


if __name__ == "__main__":
    main()
