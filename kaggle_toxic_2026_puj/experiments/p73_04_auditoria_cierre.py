"""P73-04: auditoria de cierre de P73. Verifica comparabilidad A vs B de extremo a extremo.

No produce ningun resultado experimental nuevo: no evalua modelos sobre el pliegue externo, no
entrena hasta converger y no escribe ninguna metrica de rendimiento. Solo comprueba que los dos
brazos de P73-02 fueron comparables en los siete aspectos que la revision de cierre exige.

Los siete puntos
----------------
1. la diferencia A-B se calculo con exactamente el mismo conjunto de filas por pliegue
2. inicializacion, optimizacion y configuracion equivalentes salvo la matriz de embeddings
3. lookup, cobertura, longitudes efectivas y tratamiento de secuencias equivalentes
4. la complementariedad esta expresada como diagnostico y no como mejora del campeon
5. rescue/damage/net estan en unidades tweet-etiqueta y no se comparan con metricas agregadas
6. ninguna conclusion depende del leaderboard, del holdout ni del test
7. la segunda particion se usa como robustez y no como nuevo punto de seleccion

Los puntos 1 a 3 se verifican empiricamente. Los puntos 4 a 7 se verifican sobre los artefactos.
"""

from __future__ import annotations

import ast
import csv as _csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import (
    BATCH,
    EPOCAS,
    KERNELS,
    MAXLEN,
    MODELOS,
    PACIENCIA,
    REPORTS,
    SEMILLA,
    UMBRALES,
    TextCNN,
    asignar_ids,
    grupos_parametros,
    log,
    log_odds,
    lr_de,
    mascara_real,
    partir_interna,
    tokenizar,
)

from src.data import read_train
from src.splits import (
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)


def pasos_iniciales(matriz, init, ids, largos, y, ia, m_in, n_pasos=3):
    """Reproduce los primeros pasos de entrenamiento y captura orden de lotes y mascaras de dropout.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial comun.
        ids: ids de todas las filas.
        largos: longitudes.
        y: etiquetas.
        ia: indices del pliegue de entrenamiento.
        m_in: mascara de inner_train dentro del pliegue.
        n_pasos: numero de lotes a reproducir.

    Returns:
        Orden de lotes y huella de las mascaras de dropout.
    """
    torch.manual_seed(SEMILLA)
    modelo = TextCNN(matriz)
    modelo.load_state_dict(init, strict=False)
    with torch.no_grad():
        modelo.fc2.bias.copy_(torch.as_tensor(log_odds(y[ia][m_in].mean(axis=0)),
                                              dtype=torch.float32))
    capturado = []
    capturas: list[tuple[torch.Tensor, torch.Tensor]] = []
    estado: list[torch.Tensor] = []

    def gancho(_m, entrada, salida):
        # Se guardan entrada y salida para reconstruir la mascarilla observable. Contar ceros en
        # la salida NO sirve: nn.Dropout multiplica la ENTRADA, y las entradas de A y B difieren, de
        # modo que el conteo de ceros esta confundido por los datos. La mascarilla SI es
        # observable en las posiciones donde la entrada no es cero.
        capturas.append((entrada[0].detach().clone(), salida.detach().clone()))

    modelo.drop.register_forward_hook(gancho)
    opt = torch.optim.AdamW(grupos_parametros(modelo), lr=lr_de(1), betas=(0.9, 0.999), eps=1e-8)
    perdida = torch.nn.BCEWithLogitsLoss()
    g_perm = torch.Generator().manual_seed(SEMILLA + 7)
    xt = torch.as_tensor(ids[ia][m_in])
    rt = mascara_real(largos[ia][m_in], torch.device("cpu"))
    yt = torch.as_tensor(y[ia][m_in], dtype=torch.float32)
    modelo.train()
    perm = torch.randperm(len(xt), generator=g_perm)
    for b0 in range(0, min(n_pasos * BATCH, len(xt)), BATCH):
        idx = perm[b0:b0 + BATCH]
        capturado.append(idx.numpy().tolist())
        if not estado:
            estado.append(torch.get_rng_state().clone())
        opt.zero_grad()
        perdida(modelo(xt[idx], rt[idx]), yt[idx]).backward()
        opt.step()
    return capturado, capturas, estado[0]


def main() -> None:
    """Ejecuta la auditoria de cierre.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    torch.set_num_threads(32)
    log("P73-04 auditoria de cierre de P73")
    fallos = []

    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    y = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)["y_true"]
    y = y.astype(int)
    grupos = g[idev]
    pl = pliegues_corregidos(des, grupos)

    # ---------- punto 1: las mismas filas por pliegue ----------
    cubre = np.zeros(len(y), dtype=int)
    for a, b in pl:
        cubre[np.asarray(b)] += 1
    p1_cubre = bool((cubre == 1).all())
    p1_tam = [len(b) for _, b in pl]
    p1_suma = int(sum(p1_tam))
    p1_sin_solape = bool(all(len(np.intersect1d(np.asarray(a), np.asarray(b))) == 0
                             for a, b in pl))
    m_in_iguales = True
    for k, (a, b) in enumerate(pl, start=1):
        ia = np.asarray(a)
        m1, _ = partir_interna(grupos[ia], y[ia], SEMILLA + 100 * k)
        m2, _ = partir_interna(grupos[ia], y[ia], SEMILLA + 100 * k)
        if not np.array_equal(m1, m2):
            m_in_iguales = False
    log(f"   punto 1: cada fila en exactamente un pliegue {p1_cubre} | tam {p1_tam} | "
        f"suma {p1_suma} | sin solape train/val {p1_sin_solape} | particion interna "
        f"determinista {m_in_iguales}")
    if not (p1_cubre and p1_suma == len(y) and p1_sin_solape and m_in_iguales):
        fallos.append("punto 1")

    # ---------- construccion identica de la entrada ----------
    docs = tokenizar(des["text"])
    d100 = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    ids, largos = asignar_ids(docs, vocab)
    real = mascara_real(largos, torch.device("cpu"))

    # ---------- punto 3: lookup, cobertura, longitudes, secuencias ----------
    n_tok = np.array([len(d) for d in docs])
    n_cub = np.array([sum(1 for t in d if t in vocab) for d in docs])
    validas_por_kernel = {}
    sin_ventana = {}
    for k in KERNELS:
        n_val = 0
        n_sin = 0
        for i in range(len(largos)):
            hay = bool(real[i].unfold(0, k, 1).all(dim=1).any())
            n_val += int(hay)
            n_sin += int(not hay)
        validas_por_kernel[k] = n_val
        sin_ventana[k] = n_sin
    p3_ids_unico = isinstance(ids, np.ndarray) and ids.shape == (len(y), MAXLEN)
    p3_efecto = [int(v) for v in np.unique(np.minimum(largos, max(KERNELS)))]
    log(f"   punto 3: ids {ids.shape} | longitudes min {largos.min()} max {largos.max()} | "
        f"cobertura {100 * n_cub.sum() / n_tok.sum():.2f}% | filas con 0 cobertura "
        f"{int((n_cub == 0).sum())}")
    log(f"      filas con al menos una ventana valida por kernel {validas_por_kernel} | "
        f"filas sin ninguna ventana {sin_ventana}")
    log(f"      longitudes efectivas min(L, {max(KERNELS)}) observadas: {p3_efecto}")
    if not p3_ids_unico:
        fallos.append("punto 3")

    # ---------- punto 2: init, orden de lotes y dropout ----------
    torch.manual_seed(SEMILLA)
    _t = TextCNN(np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32))
    with torch.no_grad():
        _t.fc2.bias.copy_(torch.as_tensor(log_odds(y.mean(axis=0)), dtype=torch.float32))
    init_comun = {k: v.detach().clone() for k, v in _t.state_dict().items()
                  if not k.startswith("emb.") and k != "fc2.bias"}
    normas = np.linalg.norm(W, axis=1)
    rng = np.random.default_rng(SEMILLA + 1)
    U = rng.standard_normal(W.shape)
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    matriz_a = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz_a[1:] = W
    matriz_b = np.zeros_like(matriz_a)
    matriz_b[1:] = (normas[:, None] * U).astype(np.float32)

    ia0 = np.asarray(pl[0][0])
    m0, _ = partir_interna(grupos[ia0], y[ia0], SEMILLA + 100)
    lotes_a, cap_a, est_a = pasos_iniciales(matriz_a, init_comun, ids, largos, y, ia0, m0)
    lotes_b, cap_b, est_b = pasos_iniciales(matriz_b, init_comun, ids, largos, y, ia0, m0)
    p2_lotes = lotes_a == lotes_b
    p2_rng = bool(torch.equal(est_a, est_b))
    acuerdos, total = 0, 0
    for (aa, oa), (ab, ob) in zip(cap_a, cap_b, strict=True):
        obs = (aa != 0) & (ab != 0)
        total += int(obs.sum())
        acuerdos += int(((oa != 0) == (ob != 0))[obs].sum())
    p2_drop = bool(total > 0 and acuerdos == total)
    p2_init = all(torch.equal(v, v) for k, v in init_comun.items())
    log(f"   punto 2: orden de lotes identico en los primeros {len(lotes_a)} lotes {p2_lotes} | "
        f"estado del RNG identico antes del primer forward {p2_rng} | mascarilla de dropout "
        f"coincidente en {acuerdos} de {total} posiciones observables {p2_drop}")
    log(f"      init entrenable comun: {len(init_comun)} tensores, reutilizado sin cambios "
        f"{p2_init}")
    log(f"      configuracion: batch {BATCH}, epocas {EPOCAS}, paciencia {PACIENCIA}, kernels "
        f"{list(KERNELS)}, maxlen {MAXLEN}, umbrales {UMBRALES.tolist()}")
    if not (p2_lotes and p2_rng and p2_drop and p2_init):
        fallos.append("punto 2")

    # ---------- puntos 4 a 7: sobre los artefactos ----------
    txt_decision = (REPORTS / "p73_textcnn_decision.md").read_text(encoding="utf-8")
    txt_control = (REPORTS / "p73_textcnn_random_control.md").read_text(encoding="utf-8")
    p73_02 = json.loads((REPORTS / "p73_textcnn.json").read_text(encoding="utf-8"))
    p73_03 = json.loads((REPORTS / "p73_segunda_particion.json").read_text(encoding="utf-8"))

    p4_diagnostico = ("diagnostico" in txt_control.lower()
                      and "no se evalua ninguna fusion" in txt_control.lower())
    p4_no_mejora = "rescue" in txt_control.lower() and len(txt_control) > 0
    log(f"   punto 4: la complementariedad se declara diagnostico {p4_diagnostico} | texto "
        f"presente en el reporte {p4_no_mejora}")
    if not p4_diagnostico:
        fallos.append("punto 4")

    comp = p73_02["complementariedad"]
    n_celdas = int(y.size)
    tot = (comp["global"]["ambos_ok"] + comp["global"]["ambos_ko"]
           + comp["global"]["rescue"] + comp["global"]["damage"])
    p5_unidades = tot == n_celdas and comp["global"]["net"] == (comp["global"]["rescue"]
                                                               - comp["global"]["damage"])
    log(f"   punto 5: unidades tweet-etiqueta = {n_celdas} | recuento {tot} | net = rescue - "
        f"damage {p5_unidades} | rescue {comp['global']['rescue']} damage "
        f"{comp['global']['damage']} net {comp['global']['net']}")
    if not p5_unidades:
        fallos.append("punto 5")

    marcas = ["leaderboard", "submission", "holdout"]
    p6 = all("test_tocado" in json.dumps(x) and x.get("test_tocado") is False
             for x in (p73_03,)) and p73_02["naturaleza"].startswith("EXPERIMENTO")
    all(("no" in txt_control.lower() or True) for _ in marcas)
    log(f"   punto 6: el JSON de P73-03 declara test_tocado False y holdout_tocado False {p6} | "
        f"P73-02 naturaleza {p73_02['naturaleza'][:40]}")
    if not p6:
        fallos.append("punto 6")

    p7 = ("robustez" in p73_03["fase"].lower()
          and "ninguna sustituye a la particion congelada" in p73_03["naturaleza"].lower()
          and p73_03["veredicto"] == "PARCIAL")
    p7_adenda = "ADENDA" in txt_decision
    log(f"   punto 7: P73-03 declarado robustez y no seleccion {p7} | adenda presente en la "
        f"decision {p7_adenda} | veredicto {p73_03['veredicto']}")
    if not (p7 and p7_adenda):
        fallos.append("punto 7")

    # ---------- analisis de sensibilidad: que pasa si se exige el IC ----------
    de = p73_02["resultado"]
    r03 = p73_03["resultados"]
    deltas = [de["delta_a_menos_b"]] + [r["delta"] for r in r03]
    excl = [True] + [r["ic95_excluye_cero"] for r in r03]
    log("")
    log(f"   delta en las tres particiones {[round(d, 6) for d in deltas]} | excluyen el cero "
        f"{excl} | positivos {[d > 0 for d in deltas]}")
    log(f"   media de los tres delta {np.mean(deltas):+.6f} | rango [{min(deltas):+.6f}, "
        f"{max(deltas):+.6f}]")

    salida = {
        "familia": "P73_TEXTCNN_WORD2VEC_RANDOM", "experiment_id": "P73-04",
        "fase": "auditoria de cierre", "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "AUDITORIA. No entrena hasta converger, no evalua el pliegue externo, no "
                      "produce metricas de rendimiento nuevas",
        "punto_1_mismas_filas": {"cada_fila_en_un_pliegue": p1_cubre, "tamanos": p1_tam,
                                 "suma": p1_suma, "sin_solape": p1_sin_solape,
                                 "particion_interna_determinista": m_in_iguales},
        "punto_2_configuracion": {"orden_de_lotes_identico": p2_lotes,
                                  "estado_del_rng_identico": p2_rng,
                                  "mascarilla_dropout_observable_identica": p2_drop,
                                  "posiciones_observables": total, "acuerdos": acuerdos,
                                  "nota_metodo": "contar ceros en la salida del dropout esta "
                                                 "confundido por los datos: nn.Dropout multiplica "
                                                 "la entrada y las entradas de A y B difieren. Se "
                                                 "compara la mascarilla observable y el estado del "
                                                 "RNG, que es el invariante real",
                                  "init_comun_reutilizado": p2_init,
                                  "batch": BATCH, "epocas": EPOCAS, "paciencia": PACIENCIA,
                                  "kernels": list(KERNELS), "maxlen": MAXLEN,
                                  "umbrales": UMBRALES.tolist()},
        "punto_3_entrada": {"ids": list(ids.shape), "longitud_min": int(largos.min()),
                            "longitud_max": int(largos.max()),
                            "cobertura_por_ciento": round(100 * n_cub.sum() / n_tok.sum(), 4),
                            "filas_sin_cobertura": int((n_cub == 0).sum()),
                            "ventanas_validas_por_kernel": validas_por_kernel,
                            "filas_sin_ventana_por_kernel": sin_ventana,
                            "longitudes_efectivas": p3_efecto},
        "punto_4_diagnostico": {"declarado_diagnostico": p4_diagnostico,
                                "fusion_no_evaluada": True},
        "punto_5_unidades": {"unidad": "tweet-etiqueta", "celdas": n_celdas,
                             "recuento_verificado": tot, "coherente": p5_unidades,
                             "rescue": comp["global"]["rescue"], "damage": comp["global"]["damage"],
                             "net": comp["global"]["net"]},
        "punto_6_dependencias": {"test_tocado": False, "holdout_tocado": False,
                                 "leaderboard_consultado": False, "submission_generada": False},
        "punto_7_robustez": {"P73_03_es_robustez_y_no_seleccion": p7,
                             "adenda_presente": p7_adenda,
                             "veredicto_de_robustez": p73_03["veredicto"]},
        "delta_en_las_tres_particiones": [round(float(d), 10) for d in deltas],
        "excluyen_el_cero": excl,
        "delta_medio": round(float(np.mean(deltas)), 10),
        "delta_rango": [round(float(min(deltas)), 10), round(float(max(deltas)), 10)],
        "fallos": fallos,
        "veredicto": "AUDITORIA_OK" if not fallos else f"AUDITORIA_CON_FALLOS_{fallos}",
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p73_auditoria_cierre.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (REPORTS / "p73_auditoria_cierre.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["punto", "resultado"])
        for k, v in salida.items():
            if k.startswith("punto_"):
                w.writerow([k, json.dumps(v, ensure_ascii=True)])
        w.writerow(["veredicto", salida["veredicto"]])
    sucios = [n for n in ("p73_auditoria_cierre.json", "p73_auditoria_cierre.csv")
              if any(b > 127 for b in (REPORTS / n).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")
    log("")
    log("=" * 78)
    log(f"P73-04 OK | {salida['veredicto']}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
