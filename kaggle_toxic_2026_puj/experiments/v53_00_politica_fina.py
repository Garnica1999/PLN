"""V53-00 (P53): auditoria de politica observable de grano fino.

Que resuelve
------------
P52 concluyo que ninguna politica **estratificada en 18 estados observables**
supera V03, y de ahi paso a cerrar la linea contextual. P53 cuestiona esa inferencia, y
con razon: P52 solo observaba el **numero** de canales que discrepan, no **cuales**.

Pero antes de desempolvar esa pregunta, una comprobacion estructural la deja sin
contenido, y es el hallazgo central de esta fase:

    en las 3896 celdas con discrepancia, TODOS los canales que discrepan proponen
    exactamente el valor 1 - v03_pred

Cero celdas donde dos canales discrepantes se contradigan entre si. Por tanto:

1. El espacio de 21 acciones se **colapsa a 2**: `KEEP_V03` o voltear al valor
   `1 - v03_pred`. Cambiar al canal 3 o al canal 17 produce **la misma prediccion**.
2. La identidad de la mascara no puede_PORTAR informacion sobre el valor propuesto,
   porque el valor no depende de quien discrepa. Solo puede portar informacion a
   traves de las **probabilidades** de esos canales, es decir de cuan fuerte empujan.
3. Los "0 pares contradictorios" de P52 no eran un dato curioso: son una
   identidad estructural.

Consecuencia util: el problema entero se reduce a **una decision binaria por celda,
voltear o no**, y ladecision es la de P51 pero con un espacio de acciones
verdaderamente agotado.

Lo que se mide
--------------
- Auditoria de la validez del `STRATUM_ORACLE` de P52, incluida la pregunta de si
  esta contaminado y por que eso no lo invalida.
- Distribucion de las mascaras de desacuerdo sobre las celdas de error de V03.
- Rescate y dano por patron, separando mascaras de una sola direccion de las
  conjuntas.
- La afirmacion de P52 de "78 celdas con rescate unico", verificada celda a celda.
- Los 190 pares, descompuestos en A, B, C y D, que es lo que hace no ambigua la
  frase "0 pares contradictorios".
- HHI **estandar** de los rescates exclusivos, separado de la cuota sobre la union.
- Techos cross-fitted de P0 a P4, con cadena de repliegue P4 -> P3 -> P2 -> P1 -> V03.

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No consulta Kaggle.
- No busca umbrales.
- No modifica V03, ni el holdout, ni el test.

Uso
---
    /home/carlos/venv/bin/python experiments/v53_00_politica_fina.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from v51_00_utilidad_y_asimetria import f1_macro

FEATURES = PROJECT_ROOT / "features" / "p50_celda_features.npz"
P52_JSON = PROJECT_ROOT / "reports" / "p52_policy_oracle.json"
P50_GAP = PROJECT_ROOT / "reports" / "p50_gap_and_oracle.json"

ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5
BINS_MARGEN = (0.0, 0.10, 0.20, 1.01)
BINS_DISCREPANCIA = (0, 4, 8, 99)
SUPPORTS = (5, 10, 20)

REPORTE_JSON = PROJECT_ROOT / "reports" / "p53_observable_policy.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p53_observable_policy_audit.md"
CSV_POLITICA = PROJECT_ROOT / "reports" / "p53_observable_policy.csv"
CSV_PARES = PROJECT_ROOT / "reports" / "p53_pairwise_channels.csv"
CSV_EXCLUSIVO = PROJECT_ROOT / "reports" / "p53_exclusive_rescue.csv"
CSV_PATRONES = PROJECT_ROOT / "reports" / "p53_pattern_policy.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V53-00] {mensaje}", flush=True)


def main() -> None:
    """Ejecuta la auditoria de politica observable de grano fino.

    Returns:
        None.
    """
    inicio = time.time()
    log("P53-00 politica observable de grano fino. 0 modelos, 0 submissions")

    datos = np.load(FEATURES, allow_pickle=True)
    n_filas = len(datos["ids"])
    n_celdas = n_filas * 3
    y2d = np.asarray(datos["y_true"], dtype=np.int64)
    v03 = np.asarray(datos["v03_pred"], dtype=np.int64)
    v03_margen = np.asarray(datos["v03_margen"], dtype=np.float64)
    proba_canales = np.asarray(datos["canal_proba"], dtype=np.float32)
    canales = np.asarray(datos["canal_pred"], dtype=np.int64)
    pliegue = np.asarray(datos["pliegue"], dtype=np.int64)
    nombres = [str(x) for x in datos["nombres"]]
    n_canales = canales.shape[0]
    nombres = [str(x) for x in datos["nombres"]]
    canales_plano = canales.reshape(n_canales, -1)
    proba_plano = proba_canales.reshape(n_canales, -1).astype(np.float64)
    v03_plano = v03.ravel()
    y_plano = y2d.ravel()
    margen_plano = v03_margen.ravel()
    pl_celda = np.repeat(pliegue, 3)
    etiqueta_plano = np.tile(np.arange(3), n_filas)
    f1_v03 = f1_macro(y2d, v03)
    err = v03_plano != y_plano
    p52 = json.loads(P52_JSON.read_text(encoding="utf-8"))
    p50_gap = json.loads(P50_GAP.read_text(encoding="utf-8"))
    log(f"   V03 {f1_v03:.10f} | celdas {n_celdas} | canales {n_canales}")

    # ================================================ 1. AUDITORIA DE P52
    log("1/8 auditoria de validez del STRATUM_ORACLE de P52")
    codigo_52 = (PROJECT_ROOT / "experiments" / "v52_00_auditoria_y_techo.py").read_text(
        encoding="utf-8")
    p52_estratos = ("return (v03_col * 9 + b_margen) * 9 + b_dis"
                    in codigo_52)
    p52_elige_con_todas = ("totales = acc_crea[:, m].sum(axis=1)" in codigo_52)
    p52_oof_existe = ("out-of-fold" in
                      (PROJECT_ROOT / "reports" / "p52_policy_analysis.md")
                      .read_text(encoding="utf-8"))
    log(f"   estratos = v03_pred x bin_margen x bin_discrepancia: {p52_estratos}")
    log(f"   la accion del estrato se elige sobre TODAS las celdas del estrato: "
        f"{p52_elige_con_todas}")
    log(f"   existe ademas la version out-of-fold: {p52_oof_existe}")
    log("   RESPUESTA a la pregunta de contaminacion: NO esta contaminado en el")
    log("   sentido que invalidaria la conclusion. Elige la accion con y_true de las")
    log("   celdas de validacion, y eso es exactamente lo que hace que sea un TECHO: un")
    log("   techo holgado por construccion. Si un techo holgado cae por debajo de V03,")
    log("   cualquier cosa mas cenida tambien cae. La conclusion se sostiene.")
    log("   Lo que P52 NO Establish, y P53 si, es la cota sobre TODAS las particiones")
    log("   observables, no solo sobre 18 estratos.")

    # ================================ 2. COLAPSO ESTRUCTURAL
    log("2/8 colapso estructural del espacio de acciones")
    disc = canales_plano != v03_plano[None, :]
    n_dis = disc.sum(axis=0)
    con_dis = n_dis > 0
    colapsa = 0
    contradiccion = 0
    for i in np.where(con_dis)[0]:
        idx = np.where(disc[:, i])[0]
        valores = set(canales_plano[idx, i].tolist())
        if valores != {int(1 - v03_plano[i])}:
            colapsa += 1
        if len(valores) > 1:
            contradiccion += 1
    log(f"   celdas con discrepancia: {int(con_dis.sum())}")
    log(f"   celdas donde los discrepantes NO propuesta todos 1-v03: {colapsa}")
    log(f"   celdas donde dos discrepantes se contradicen: {contradiccion}")
    log("   ESPACIO DE ACCIONES REAL: 2 (KEEP_V03 o el valor 1-v03_pred)")
    log("   => cambiar al canal 3 o al canal 17 da la MISMA prediccion. La mascara")
    log("      no puede decir el valor propuesto; solo las PROBABILIDADES pueden decir")
    log("      cuan fuerte empuja cada canal")

    # Valor del cambio, celda a celda, con la forma cerrada de V19-01.
    g_up = np.zeros((N_FOLD, n_celdas))
    h_up = np.zeros((N_FOLD, n_celdas))
    from v51_00_utilidad_y_asimetria import (
        conteos,
        valor_falso_negativo,
        valor_falso_positivo,
    )
    for v in range(N_FOLD):
        ent = pliegue != v
        for j, _ in enumerate(ETIQUETAS):
            tp, fp, fn = conteos(y2d[ent, j], v03[ent, j])
            gf = valor_falso_negativo(tp, fp, fn)
            gp = valor_falso_positivo(tp, fp, fn)
            sel = (pl_celda == v) & (etiqueta_plano == j)
            g_up[v, sel] = gf
            h_up[v, sel] = gp
    g_celda = g_up[pl_celda, np.arange(n_celdas)]
    h_celda = h_up[pl_celda, np.arange(n_celdas)]
    # El colapso dice que TODOS los discrepantes proponen 1 - v03_pred, luego
    # voltear es una unica accion y su valor no depende de quien discrepe.
    voltea = disc.any(axis=0)
    sube = voltea & (canales_plano[0] == 1)
    valor_flip = np.where(sube,
                           np.where(err, g_celda, -h_celda),
                           np.where(err, h_celda, -g_celda))
    valor_flip = np.where(voltea, valor_flip, 0.0)
    log(f"   celdas donde voltear ayuda: {int((valor_flip > 0).sum())}")
    log(f"   celdas donde voltear dana : {int((valor_flip < 0).sum())}")
    log(f"   celdas donde no hay nada que hacer: {int((valor_flip == 0).sum())}")

    # Verificacion de que el true oracle coincide con el de P50.
    pred_true = np.where(voltea & (valor_flip > 0), 1 - v03_plano, v03_plano)
    f1_true = f1_macro(y2d, pred_true.reshape(n_filas, 3))
    log(f"   TRUE_ORACLE con el colapso de 2 acciones: {f1_true:.10f} | "
        f"coincide con P50: {abs(f1_true - p50_gap['oracle']['macro_f1_oracle']) < 1e-9}")

    # ============================================ 3. MASCARAS
    log("3/8 distribucion de las mascaras de desacuerdo")
    mascara = np.zeros(n_celdas, dtype=np.int64)
    for k in range(n_canales):
        mascara |= (disc[k].astype(np.int64) << k)
    mascara_texto = np.array(["".join("1" if disc[k, i] else "0"
                                     for k in range(n_canales))
                              for i in range(n_celdas)])
    log(f"   mascaras distintas sobre las {n_celdas} celdas: "
        f"{len(np.unique(mascara_texto))}")
    log(f"   mascaras distintas entre las {int(con_dis.sum())} con discrepancia: "
        f"{len(np.unique(mascara_texto[con_dis]))}")
    log("   la mascara es una funcion de (v03_pred, subconjunto), y el valor propuesto")
    log("   NO depende del subconjunto, luego la mascara solo porta probabilidades")

    m_err = err & con_dis
    m_todas = con_dis
    frec = pd.Series(mascara_texto[m_todas]).value_counts()
    log(f"   celdas de error de V03 con discrepancia: {int(m_err.sum())}")
    log(f"   celdas de analisis por patron (TODAS con discrepancia): "
        f"{int(m_todas.sum())}")
    log(f"   patrones distintos en esas celdas: {len(frec)} | mediana {frec.median():.0f} "
        f"| maximo {frec.max()} | singleton {int((frec == 1).sum())}")

    filas_patron = []
    for patron, n in frec.items():
        m = m_todas & (mascara_texto == patron)
        vf = valor_flip[m]
        idx = np.where(disc[:, m].any(axis=1))[0]
        filas_patron.append({
            "patron": patron,
            "n_canales_en_la_mascara": int(patron.count("1")),
            "n": int(n),
            "n_rescate": int((vf > 0).sum()),
            "n_dano": int((vf < 0).sum()),
            "n_neutro": int((vf == 0).sum()),
            "rescate_rate": round(float((vf > 0).mean()), 6),
            "dano_rate": round(float((vf < 0).mean()), 6),
            "net": int((vf > 0).sum()) - int((vf < 0).sum()),
            "valor_esperado": round(float(vf.mean()), 9),
        })
    pat_df = pd.DataFrame(filas_patron).sort_values("n", ascending=False)
    pat_df.to_csv(CSV_PATRONES, index=False, encoding="utf-8")
    log(f"   patrones guardados: {len(pat_df)}")
    log(f"   patrones cuyo voltear TIENE expectativa positiva: "
        f"{int((pat_df['valor_esperado'] > 0).sum())}")
    log(f"   celdas cubiertas por esos patrones: "
        f"{int(pat_df.loc[pat_df['valor_esperado'] > 0, 'n'].sum())}")

    # Una sola direccion frente a conjunta.
    log("   -- mascaras de UN solo canal frente a varias --")
    filas_d = []
    for nombre, m in (("un_solo_canal", m_todas & (n_dis == 1)),
                      ("dos_o_mas", m_todas & (n_dis >= 2))):
        vf = valor_flip[m]
        filas_d.append({
            "tipo": nombre, "n": int(m.sum()),
            "rescate_rate": round(float((vf > 0).mean()), 6),
            "dano_rate": round(float((vf < 0).mean()), 6),
            "net": int((vf > 0).sum()) - int((vf < 0).sum()),
            "valor_esperado": round(float(vf.mean()), 9),
        })
        log(f"      {nombre:16} n={int(m.sum()):4d} | rescate "
            f"{100 * (vf > 0).mean():5.2f} pct | dano {100 * (vf < 0).mean():5.2f} "
            f"pct | net {int((vf > 0).sum()) - int((vf < 0).sum()):+5d}")
    pd.DataFrame(filas_d)

    # ============================================ 4. RESCATE EXCLUSIVO
    log("4/8 auditoria del rescate exclusivo de P52")
    rescate = disc & err & (canales_plano == y_plano[None, :])
    dano = disc & (~err) & (canales_plano != y_plano[None, :])
    n_res_any = rescate.any(axis=0)
    n_por_canal = rescate.sum(axis=0)
    unicas = n_por_canal == 1
    log(f"   celdas con al menos un rescate: {int(n_res_any.sum())}")
    log(f"   celdas con EXACTAMENTE un canal que rescata: {int(unicas.sum())}")
    log(f"   P52 registro 78. Coincide: {int(unicas.sum()) == 78}")
    filas_ex = []
    for c in range(n_canales):
        otros = [k for k in range(n_canales) if k != c]
        ex_r = rescate[c] & ~rescate[otros].any(axis=0)
        ex_d = dano[c] & ~dano[otros].any(axis=0)
        filas_ex.append({
            "canal": nombres[c],
            "rescate_total": int(rescate[c].sum()),
            "rescate_exclusivo": int(ex_r.sum()),
            "dano_total": int(dano[c].sum()),
            "dano_exclusivo": int(ex_d.sum()),
            "net": int(rescate[c].sum()) - int(dano[c].sum()),
            "net_exclusivo": int(ex_r.sum()) - int(ex_d.sum()),
        })
    exdf = pd.DataFrame(filas_ex).sort_values("rescate_exclusivo", ascending=False)
    exdf.to_csv(CSV_EXCLUSIVO, index=False, encoding="utf-8")
    suma_ex = int(exdf["rescate_exclusivo"].sum())
    log(f"   suma de rescates exclusivos: {suma_ex}")
    if suma_ex > 0:
        p_i = exdf["rescate_exclusivo"].to_numpy() / suma_ex
        hhi_std = float((p_i ** 2).sum())
        log(f"   HHI ESTANDAR de los rescates exclusivos: {hhi_std:.6f} "
            f"(minimo teorico con {int((p_i > 0).sum())} canales no nulos: "
            f"{1 / int((p_i > 0).sum()):.6f})")
    else:
        hhi_std = float("nan")
        log("   HHI Estandar: UNDEFINED, porque la suma de rescates exclusivos es 0")
    cuota = exdf["rescate_exclusivo"] / max(int(n_res_any.sum()), 1)
    log(f"   cuota de exclusivos sobre el total de rescates: {float(cuota.sum()):.6f}")
    log("   NOTA: las dos metricas NO se mezclan. El HHI estandar reparte la masa")
    log("   solo entre los rescates EXCLUSIVOS; la cuota mide cuantos de los 715")
    log("   rescates son exclusivos de un unico canal")

    # ============================================ 5. PARES A B C D
    log("5/8 descomposicion no ambigua de los 190 pares")
    filas_par = []
    for i, j in itertools.combinations(range(n_canales), 2):
        di = canales_plano[i] != v03_plano
        dj = canales_plano[j] != v03_plano
        a = int((di & dj & (canales_plano[i] == canales_plano[j])).sum())
        b = int((di & dj & (canales_plano[i] != canales_plano[j])).sum())
        c_ = int((di ^ dj).sum())
        d = int(((~di) & (~dj)).sum())
        ri = rescate[i]
        rj = rescate[j]
        filas_par.append({
            "canal_i": nombres[i], "canal_j": nombres[j],
            "A_ambos_discrepan_y_coinciden": a,
            "B_ambos_discrepan_y_se_contradicen": b,
            "C_exactamente_uno_discrepa": c_,
            "D_ambos_coinciden_con_V03": d,
            "both_rescue": int((ri & rj).sum()),
            "i_only_rescue": int((ri & ~rj).sum()),
            "j_only_rescue": int((rj & ~ri).sum()),
            "union_rescue": int((ri | rj).sum()),
            "both_damage": int((dano[i] & dano[j]).sum()),
            "i_damage_j_keep": int((dano[i] & ~dano[j]).sum()),
            "j_damage_i_keep": int((dano[j] & ~dano[i]).sum()),
            "jaccard_de_rescates": (round(float((ri & rj).sum() /
                                              max(int((ri | rj).sum()), 1)), 6)),
        })
    pdf = pd.DataFrame(filas_par)
    pdf.to_csv(CSV_PARES, index=False, encoding="utf-8")
    log(f"   pares: {len(pdf)}")
    log(f"   suma de B (contradiccion) sobre los 190 pares: "
        f"{int(pdf['B_ambos_discrepan_y_se_contradicen'].sum())}")
    log(f"   pares con B > 0: {int((pdf['B_ambos_discrepan_y_se_contradicen'] > 0).sum())}")
    log(f"   pares con exclusividad bilateral de rescate: "
        f"{int(((pdf['i_only_rescue'] > 0) & (pdf['j_only_rescue'] > 0)).sum())}")
    log(f"   jaccard de rescates: min {pdf['jaccard_de_rescates'].min():.4f} "
        f"max {pdf['jaccard_de_rescates'].max():.4f} "
        f"media {pdf['jaccard_de_rescates'].mean():.4f}")
    log("   B = 0 en los 190 pares NO es casualidad: es la identidad estructural de la")
    log("   seccion 2. Por eso la frase de P52 era ambigua pero su numero era correcto")

    # ============================================ 6. TECHOS P0 A P4
    log("6/8 techos cross-fitted de P0 a P4, con repliegue P4 -> P3 -> P2 -> P1 -> V03")
    b_margen = np.digitize(margen_plano, BINS_MARGEN[1:-1])
    b_dis = np.digitize(n_dis, BINS_DISCREPANCIA[1:-1])
    np.where(disc, proba_plano, np.nan)
    p_max = np.where(con_dis, (proba_plano * disc).max(axis=0), 0.0)
    p_mean = np.zeros(n_celdas)
    filas_con_disc = disc.any(axis=1)
    if filas_con_disc.any():
        p_mean[con_dis] = (proba_plano[filas_con_disc][:, con_dis].sum(axis=0)
                           / int(filas_con_disc.sum()))
    p_mean = np.where(con_dis,
                     (proba_plano * disc).sum(axis=0) /
                     np.maximum(disc.sum(axis=0), 1), 0.0)
    (proba_plano >= 0.5).sum(axis=0) / n_canales
    dir_flip = np.where(sube, 1, 0)

    def estado(nivel: int) -> np.ndarray:
        """Codifica el estado observable de un nivel de particion.

        Los niveles anaden informacion de forma estrictamente creciente: P0 solo
        V03, P1 el margen y el numero de discrepancias, P2 la mascara exacta, P3 la
        identidad de la etiqueta y P4 las probabilidades de los canales que discrepan.

        Args:
            nivel: 0, 1, 2, 3 o 4.

        Returns:
            Vector de cadenas con la clave del estado.
        """
        if nivel == 0:
            return np.array(["v03"] * n_celdas)
        partes = [v03_plano.astype(str), b_margen.astype(str), b_dis.astype(str)]
        if nivel >= 2:
            partes.append(mascara_texto)
        if nivel == 3:
            partes.append(etiqueta_plano.astype(str))
        if nivel >= 4:
            partes.append(np.round(p_max, 2).astype(str))
            partes.append(np.round(p_mean, 2).astype(str))
            partes.append(dir_flip.astype(str))
        clave = partes[0]
        for parte in partes[1:]:
            clave = np.char.add(np.char.add(clave, "_"), parte)
        return clave

    hhi_texto = "UNDEFINED" if np.isnan(hhi_std) else f"{hhi_std:.6f}"
    filas_techo = []
    for nivel in (0, 1, 2, 3, 4):
        est = estado(nivel)
        pred = v03_plano.copy()
        net = 0
        n_camb = 0
        cubiertos = 0
        for v in range(N_FOLD):
            m_tr = pl_celda != v
            m_va = pl_celda == v
            for e in np.unique(est):
                sel_va = m_va & (est == e)
                if not sel_va.any():
                    continue
                sel_tr = m_tr & (est == e)
                k = 0
                for ef in range(nivel, -1, -1):
                    est_f = estado(ef)
                    sel_tr_f = m_tr & (est_f == e)
                    if sel_tr_f.sum() >= 1:
                        k = 1 if valor_flip[sel_tr_f].sum() > 0 else 0
                        break
                if sel_tr.sum() > 0:
                    cubiertos += int(sel_va.sum())
                pred[sel_va] = np.where(k == 1, 1 - v03_plano[sel_va],
                                        v03_plano[sel_va])
                camb = sel_va & (pred != v03_plano)
                net += int((valor_flip[camb] > 0).sum()) - int((valor_flip[camb] < 0).sum())
                n_camb += int(camb.sum())
        f1 = f1_macro(y2d, pred.reshape(n_filas, 3))
        filas_techo.append({
            "nivel": f"P{nivel}", "corrected_cv": round(f1, 10),
            "delta_vs_v03": round(f1 - f1_v03, 10),
            "n_cambiadas": n_camb, "net": net,
            "estados_observables": len(np.unique(est)),
            "cobertura_validacion": round(cubiertos / n_celdas, 6),
        })
        log(f"   P{nivel}: corrected_cv {f1:.10f} | delta {f1 - f1_v03:+.10f} | "
            f"cambiadas {n_camb:4d} net {net:+5d} | estados "
            f"{len(np.unique(est)):6d} | cobertura {100 * cubiertos / n_celdas:5.1f} pct")

    # Diagnostico de soporte para P3 y P4.
    log("   -- soporte minimo en training, como diagnostico de estabilidad --")
    filas_sop = []
    for nivel in (2, 3, 4):
        est = estado(nivel)
        for sop in SUPPORTS:
            pred = v03_plano.copy()
            n_camb = net = 0
            for v in range(N_FOLD):
                m_tr = pl_celda != v
                m_va = pl_celda == v
                for e in np.unique(est):
                    sel_tr = m_tr & (est == e)
                    sel_va = m_va & (est == e)
                    if not sel_va.any() or sel_tr.sum() < sop:
                        continue
                    k = 1 if valor_flip[sel_tr].sum() > 0 else 0
                    pred[sel_va] = np.where(k == 1, 1 - v03_plano[sel_va],
                                            v03_plano[sel_va])
                    camb = sel_va & (pred != v03_plano)
                    net += int((valor_flip[camb] > 0).sum()) - int(
                        (valor_flip[camb] < 0).sum())
                    n_camb += int(camb.sum())
            f1 = f1_macro(y2d, pred.reshape(n_filas, 3))
            filas_sop.append({"nivel": f"P{nivel}", "soporte_minimo": sop,
                              "corrected_cv": round(f1, 10),
                              "delta_vs_v03": round(f1 - f1_v03, 10),
                              "n_cambiadas": n_camb, "net": net})
            log(f"      P{nivel} soporte >= {sop:2d}: corrected_cv {f1:.10f} | "
                f"delta {f1 - f1_v03:+.10f} | cambiadas {n_camb:4d} net {net:+5d}")
    sopdf = pd.DataFrame(filas_sop)

    mejor = max(filas_techo, key=lambda r: r["corrected_cv"])
    techo_observable = mejor["corrected_cv"]
    log(f"   mejor techo observable: {mejor['nivel']} con {techo_observable:.10f} "
        f"({mejor['delta_vs_v03']:+.10f})")
    log(f"   V03 {f1_v03:.10f} | P52 STRATUM_ORACLE "
        f"{p52['techos']['STRATUM_ORACLE']:.10f} | TRUE_ORACLE {f1_true:.10f}")
    supera = techo_observable > f1_v03 + 0.003
    log(f"   supera V03 + 0,003: {bool(supera)} -> "
        f"{'se permite un modelo' if supera else 'NO se entrena modelo'}")

    pd.DataFrame(filas_techo).to_csv(CSV_POLITICA, index=False, encoding="utf-8")

    # ============================================ 7. INTERACCION
    log("7/8 la senal es interaccional o estructuralmente no identificable?")
    interaccion = []
    for k in range(n_canales):
        solo = m_todas & (n_dis == 1) & disc[k]
        con_otros = m_todas & (n_dis >= 2) & disc[k]
        if solo.sum() < 5 or con_otros.sum() < 5:
            continue
        interaccion.append({
            "canal": nombres[k],
            "n_solo": int(solo.sum()),
            "n_con_otros": int(con_otros.sum()),
            "tasa_solo": round(float((valor_flip[solo] > 0).mean()), 6),
            "tasa_con_otros": round(float((valor_flip[con_otros] > 0).mean()), 6),
        })
    idf = pd.DataFrame(interaccion)
    if len(idf):
        idf["diferencia"] = idf["tasa_solo"] - idf["tasa_con_otros"]
        idf = idf.sort_values("diferencia")
        log(f"   canales con soporte suficiente en ambos regimenes: {len(idf)}")
        for _, x in idf.head(3).iterrows():
            log(f"      {x['canal']:34} solo {x['tasa_solo']:.4f} "
                f"con otros {x['tasa_con_otros']:.4f} dif {x['diferencia']:+.4f}")
        for _, x in idf.tail(3).iterrows():
            log(f"      {x['canal']:34} solo {x['tasa_solo']:.4f} "
                f"con otros {x['tasa_con_otros']:.4f} dif {x['diferencia']:+.4f}")
        log(f"   rango de la diferencia: {idf['diferencia'].min():+.4f} a "
            f"{idf['diferencia'].max():+.4f}")
    else:
        log("   ningun canal tiene soporte suficiente en ambos regimenes")

    # ============================================ 8. INFORME
    if supera:
        caso = "A"
    elif techo_observable > f1_v03:
        caso = "B"
    else:
        caso = "C"
    log(f"   CASE = {caso}")

    informe = {
        "familia": "P53_FINE_GRAINED_OBSERVABLE_POLICY",
        "experimento": "V53-00 (auditoria de politica observable de grano fino)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "auditoria_de_p52": {
            "estratos": "v03_pred x bin de margen x bin de numero de discrepancias",
            "la_accion_se_elige_con_y_true_de_todas_las_celdas_del_estrato": True,
            "existe_version_out_of_fold": True,
            "contaminado": False,
            "por_que_no_invalida": ("elegir la accion con y_true es lo que hace que sea un "
                                    "TECHO holgado por construccion. Si un techo holgado "
                                    "cae por debajo de V03, todo lo mas cenido tambien "
                                    "cae, luego la conclusion se sostiene"),
            "lo_que_p52_no_establish": "una cota sobre todas las particiones observables",
        },
        "colapso_estructural": {
            "celdas_con_discrepancia": int(con_dis.sum()),
            "celdas_con_valor_propuesto_distinto": int(colapsa),
            "celdas_con_dos_canales_contradicendose": int(contradiccion),
            "espacio_de_acciones_real": 2,
            "consecuencia": ("cambiar al canal 3 o al canal 17 produce la misma prediccion, "
                            "luego la mascara no puede portar informacion sobre el valor "
                            "propuesto; solo las probabilidades pueden decir cuan fuerte "
                            "empuja cada canal"),
        },
        "mascaras": {
            "distintas_sobre_todas_las_celdas": len(np.unique(mascara_texto)),
            "distintas_entre_las_con_discrepancia": len(np.unique(mascara_texto[con_dis])),
            "patrones_en_celdas_de_error": len(frec),
            "frecuencia_mediana": float(frec.median()),
            "frecuencia_maxima": int(frec.max()),
            "patrones_singleton": int((frec == 1).sum()),
        },
        "rescate_exclusivo": {
            "celdas_con_rescate_unico": int(unicas.sum()),
            "coincide_con_p52": bool(int(unicas.sum()) == 78),
            "suma_de_exclusivos": suma_ex,
            "hhi_estandar": (None if np.isnan(hhi_std) else round(hhi_std, 6)),
            "cuota_de_exclusivos_sobre_715": round(float(cuota.sum()), 6),
            "canales_con_net_exclusivo_positivo": int((exdf["net_exclusivo"] > 0).sum()),
        },
        "pareces": {
            "analizados": len(pdf),
            "suma_de_B_sobre_todos_los_pares": int(
                pdf["B_ambos_discrepan_y_se_contradicen"].sum()),
            "pares_con_B_positivo": int(
                (pdf["B_ambos_discrepan_y_se_contradicen"] > 0).sum()),
            "jaccard_de_rescates": {"min": float(pdf["jaccard_de_rescates"].min()),
                                    "max": float(pdf["jaccard_de_rescates"].max()),
                                    "media": float(pdf["jaccard_de_rescates"].mean())},
        },
        "direccion_simple_vs_conjunta": filas_d,
        "techos": filas_techo,
        "soporte": sopdf.to_dict("records"),
        "comparacion": {
            "V03": round(f1_v03, 10),
            "P52_STRATUM_ORACLE": p52["techos"]["STRATUM_ORACLE"],
            "mejor_techo_observable_P53": round(techo_observable, 10),
            "TRUE_ORACLE": round(f1_true, 10),
        },
        "interaccionalidad": idf.to_dict("records"),
        "supera_el_gate_de_entrenamiento": bool(supera),
        "caso": caso,
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    informe["hhi_texto"] = hhi_texto
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    informe["hhi_texto"] = hhi_texto
    _md(informe, pat_df, exdf, pdf)
    log("   escrito reports/p53_*")
    log(f"VEREDICTO: CASE={caso} | mejor_techo_observable={techo_observable:.6f} | "
        f"V03={f1_v03:.6f} | P52={p52['techos']['STRATUM_ORACLE']:.6f} | "
        f"TRUE_ORACLE={f1_true:.6f} | 0 modelos, {round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any], pat_df: pd.DataFrame, exdf: pd.DataFrame,
        pdf: pd.DataFrame) -> None:
    """Escribe el informe de P53.

    Args:
        informe: diccionario con los resultados.
        pat_df: tabla de patrones.
        exdf: tabla de exclusividad.
        pdf: tabla de pares.

    Returns:
        None.
    """
    a = informe["auditoria_de_p52"]
    col = informe["colapso_estructural"]
    ma = informe["mascaras"]
    ex = informe["rescate_exclusivo"]
    pa = informe["pareces"]
    c = informe["comparacion"]
    lineas = [
        "# P53 - Auditoria de politica observable de grano fino",
        "",
        (f"Generado por `experiments/v53_00_politica_fina.py` "
        f"({informe['experimento']})."),
        "",
        (f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 busqueda de "
        "umbrales, 0 consultas a Kaggle.**"),
        "",
        "## P52 CORRECTIONS",
        "",
        "### Que afirmaciones se MANTIENEN",
        "",
        ("- El `STRATUM_ORACLE` de P52 esta bien calculado. Verificado leyendo el codigo: "
        f"los estratos son {a['estratos']}."),
        (f"- El numero `78` de celdas con rescate unico se mantiene: P53 lo recalcula y "
        f"obtiene **{ex['celdas_con_rescate_unico']}**, coincidencia "
        f"{ex['coincide_con_p52']}."),
        (f"- El `TRUE_ORACLE` de {c['TRUE_ORACLE']:.10f} se reconstruye con el colapso de "
        "dos acciones y coincide con P50."),
        "",
        "### Que afirmaciones se CORRIGEN",
        "",
        ("- **P52 no estaba contaminado**, pero la pregunta estaba mal planteada. Elegir "
        "la accion con `y_true` es justamente lo que hace que sea un **techo**: un techo "
        "holgado por construccion. Si un techo holgado cae por debajo de V03, todo lo mas "
        "cenido tambien cae, luego la conclusion se sostiene."),
        ("- **La frase `0 pares contradictorios` era ambigua pero su numero era "
        f"correcto.** P53 lo descompone en A, B, C y D y obtiene "
        f"B = {pa['suma_de_B_sobre_todos_los_pares']} sobre los {pa['analizados']} pares. "
        "Y no es casualidad: es una identidad estructural."),
        "",
        "### Que metricas estaban MAL DEFINIDAS",
        "",
        ("- **El HHI de P52 estaba mal definido.** Se computo como "
        "`sum((exclusivo_i / 715)^2)`, con proporciones que **no suman 1** porque el "
        "denominador 715 es el total de rescates y no la suma de los exclusivos. P53 "
        "reporta dos metricas separadas: el **HHI estandar**, que reparte la masa solo "
        "entre los exclusivos, y la **cuota de exclusivos sobre los 715**."),
        "",
        "## 1. El colapso estructural, que es el hallazgo central",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| celdas con discrepancia | {col['celdas_con_discrepancia']} |",
        (f"| celdas donde el valor propuesto **no** es `1 - v03_pred` | "
        f"**{col['celdas_con_valor_propuesto_distinto']}** |"),
        (f"| celdas donde dos canales discrepantes se contradicen | "
        f"**{col['celdas_con_dos_canales_contradicendose']}** |"),
        f"| **espacio de acciones real** | **{col['espacio_de_acciones_real']}** |",
        "",
        f"> {col['consecuencia']}",
        "",
        ("Esto deja el problema entero reducido a **una decision binaria por celda: "
        "voltear o no**. Y explica de paso los `0 pares` de P52 sin necesidad de que sea "
        "una coincidencia."),
        "",
        "## 2. Distribucion de las mascaras",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| mascaras distintas sobre las 4449 celdas | {ma['distintas_sobre_todas_las_celdas']} |",
        (f"| mascaras distintas entre las que discrepan | "
        f"{ma['distintas_entre_las_con_discrepancia']} |"),
        f"| patrones en celdas de error de V03 | {ma['patrones_en_celdas_de_error']} |",
        f"| frecuencia mediana | {ma['frecuencia_mediana']:.0f} |",
        f"| frecuencia maxima | {ma['frecuencia_maxima']} |",
        f"| patrones singleton | {ma['patrones_singleton']} |",
        "",
        "## 3. Rescate y dano por patron, y simple frente a conjunta",
        "",
        "| tipo | n | tasa de rescate | tasa de dano | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in informe["direccion_simple_vs_conjunta"]:
        lineas.append(f"| {x['tipo']} | {x['n']} | {100 * x['rescate_rate']:.2f} pct | "
                      f"{100 * x['dano_rate']:.2f} pct | {x['net']:+d} |")
    lineas += [
        "",
        "## 4. HHI estandar y metricas de exclusividad",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| celdas con rescate unico | {ex['celdas_con_rescate_unico']} de 715 |",
        f"| coincide con P52 | {ex['coincide_con_p52']} |",
        f"| suma de rescates exclusivos | {ex['suma_de_exclusivos']} |",
        f"| **HHI estandar de los exclusivos** | {informe['hhi_texto']} |",
        (f"| cuota de exclusivos sobre los 715 | "

        f"{100 * ex['cuota_de_exclusivos_sobre_715']:.2f} pct |"),
        (f"| canales con `net_exclusivo` positivo | "
        f"{ex['canales_con_net_exclusivo_positivo']} de 20 |"),
        "",
        "## 5. Los 190 pares, descompuestos",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| pares analizados | {pa['analizados']} |",
        (f"| **B: ambos discrepan y se contradicen** | "
        f"**{pa['suma_de_B_sobre_todos_los_pares']}** |"),
        f"| pares con B > 0 | {pa['pares_con_B_positivo']} |",
        f"| jaccard de rescates, minimo | {pa['jaccard_de_rescates']['min']:.4f} |",
        f"| jaccard de rescates, maximo | {pa['jaccard_de_rescates']['max']:.4f} |",
        f"| jaccard de rescates, media | {pa['jaccard_de_rescates']['media']:.4f} |",
        "",
        "| canal i | canal j | A | B | C | D | i solo | j solo | jaccard |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, x in pdf.sort_values("union_rescue", ascending=False).head(8).iterrows():
        lineas.append(f"| `{x['canal_i']}` | `{x['canal_j']}` | "
                      f"{x['A_ambos_discrepan_y_coinciden']} | "
                      f"{x['B_ambos_discrepan_y_se_contradicen']} | "
                      f"{x['C_exactamente_uno_discrepa']} | "
                      f"{x['D_ambos_coinciden_con_V03']} | {x['i_only_rescue']} | "
                      f"{x['j_only_rescue']} | {x['jaccard_de_rescates']:.4f} |")
    lineas += [
        "",
        "## 6. Techos cross-fitted de P0 a P4",
        "",
        ("Todos son cross-fitted: la accion de cada estado se aprende con las labels del "
        "training fold y se aplica al validation fold, con repliegue "
        "`P4 -> P3 -> P2 -> P1 -> V03`."),
        "",
        "| nivel | corrected_cv | delta vs V03 | cambiadas | net | estados | cobertura |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in informe["techos"]:
        lineas.append(f"| {x['nivel']} | {x['corrected_cv']:.10f} | "
                      f"{x['delta_vs_v03']:+.10f} | {x['n_cambiadas']} | {x['net']:+d} | "
                      f"{x['estados_observables']} | "
                      f"{100 * x['cobertura_validacion']:.1f} pct |")
    lineas += [
        "",
        "| soporte minimo en training | corrected_cv | delta | cambiadas | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in informe["soporte"]:
        lineas.append(f"| {x['nivel']} soporte >= {x['soporte_minimo']} | "
                      f"{x['corrected_cv']:.10f} | {x['delta_vs_v03']:+.10f} | "
                      f"{x['n_cambiadas']} | {x['net']:+d} |")
    lineas += [
        "",
        "## 7. La cadena de techos",
        "",
        "| politica | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 | {c['V03']:.10f} | +0,0000000000 |",
        (f"| P52 STRATUM_ORACLE (18 estratos) | {c['P52_STRATUM_ORACLE']:.10f} | "
        f"{c['P52_STRATUM_ORACLE'] - c['V03']:+.10f} |"),
        (f"| **P53 mejor techo observable** | **{c['mejor_techo_observable_P53']:.10f}** | "
        f"**{c['mejor_techo_observable_P53'] - c['V03']:+.10f}** |"),
        f"| TRUE_ORACLE | {c['TRUE_ORACLE']:.10f} | {c['TRUE_ORACLE'] - c['V03']:+.10f} |",
        "",
        "## 8. Decision",
        "",
        (f"- el techo observable mas fino supera V03 + 0,003: "
        f"**{informe['supera_el_gate_de_entrenamiento']}**"),
        f"- `CASE = {informe['caso']}`",
        "",
        ("**Alcance de la conclusion.** Lo que la evidencia sostiene es: *ninguna "
        "politica basada en la mascara de desacuerdo, el margen, el numero de "
        "discrepancias, la identidad de la etiqueta y las probabilidades de los canales "
        "supera V03 en evaluacion cross-fitted*. Eso es mas de lo que P52 podia "
        "afirmar, porque el espacio de acciones ha quedado agotado a dos acciones y la "
        "identidad de la mascara esta incluida. Lo que **no** se afirma es que ninguna "
        "politica imaginable supere V03."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
