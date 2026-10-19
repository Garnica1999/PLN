"""Consensos Product-of-Experts de V53/V54/V55/V56 (porte fiel de los scripts de construccion).

Product-of-Experts bayesiano en el espacio de log-probabilidades:
    log P_geom(y = k | x) = sum_m w_m log P_m(y = k | x) - log Z(x)

CONSENSOS congela, por version, los miembros EN ORDEN de apilado y los pesos del script
ANTES de la normalizacion final (v53/v54 los derivan de W_V48; v55/v56 son literales):
- v53_02_build_submission.py:51-59,75-76 (7 miembros).
- v54_01_build_submission.py:52-72,89-93 (8 miembros, rama Cramer 0.80 V43 + 0.20 V47).
- v55_01_build_submission.py:53-56,73 (8 miembros, incluye CatBoost V49).
- v56_01_build_submission.py:53-56,74 (9 miembros, incluye V51 y V53).
Los pesos se normalizan al combinarlos (WEIGHTS_GEOM = WEIGHTS_GEOM / WEIGHTS_GEOM.sum()).
ASCII puro. Sin efectos en import.
"""

from __future__ import annotations

import numpy as np

#: Regla plug-in congelada por clase (v53_02_build_submission.py:51; age_umbrales_validacion.json).
PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

#: Pesos V48 de origen de las derivaciones de v53/v54 (v53_02:54-55, v54_01:55-56).
W_V48 = [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549]

#: Peso de Dense-SE-TabularNet en v53 (v53_02:56) y en v54 (v54_01:58).
W_DENSE_V53 = 0.0250
W_DENSE_V54 = 0.0200

#: Reparto de la rama Cramer de v54: 0.80 V43 + 0.20 V47 (v54_01:64-65).
REPARTO_CRAMER_V54 = (0.80, 0.20)

#: Epsilon del clip de probabilidades (v53_02:81, v54_01:98, v55_01:74, v56_01:75).
EPS = 1e-12

#: Especificaciones congeladas por version. Los valores float64 de v53/v54 son el resultado
#: EXACTO de la derivacion jerarquica (ver pesos_geometricos); v55/v56 son los literales.
CONSENSOS = {
    "v53": {
        "miembros": ["v30", "v43", "v42", "v40", "v39_ordinal", "v39_hgb", "v53_dense"],
        "pesos": [0.54873, 0.08112000000000001, 0.16809000000000002, 0.07351500000000001,
                  0.050017500000000006, 0.053527500000000006, 0.025],
        "fuente": "scripts/v53_02_build_submission.py:54-59,75-76",
    },
    "v54": {
        "miembros": ["v30", "v43", "v47", "v42", "v40", "v39_ordinal", "v39_hgb", "v53_dense"],
        "pesos": [0.551544, 0.06522880000000002, 0.016307200000000004, 0.16895200000000002,
                  0.07389200000000001, 0.050274000000000006, 0.053802, 0.02],
        "fuente": "scripts/v54_01_build_submission.py:55-72,89-93",
    },
    "v55": {
        "miembros": ["v30", "v43", "v47", "v42", "v40", "v39_ordinal", "v39_hgb", "v49_catboost"],
        "pesos": [0.52829, 0.07720, 0.02895, 0.17931, 0.07890, 0.05062, 0.04869, 0.00805],
        "fuente": "scripts/v55_01_build_submission.py:53-56,73",
    },
    "v56": {
        "miembros": ["v30", "v43", "v47", "v42", "v40", "v39_ordinal", "v39_hgb",
                     "v51_consistent", "v53_dense"],
        "pesos": [0.55939, 0.06023, 0.02089, 0.16873, 0.06080, 0.04376, 0.05204, 0.01035,
                  0.02381],
        "fuente": "scripts/v56_01_build_submission.py:53-56,74",
    },
}


def pesos_geometricos(version: str) -> np.ndarray:
    """Vector de pesos pre-normalizacion de una version, derivado como el script.

    v53: W_V48 normalizado * (1 - 0.0250) + [0.0250] (v53_02_build_submission.py:54-58).
    v54: W_V48 * (1 - 0.0200) con rama Cramer 0.80/0.20 y + [0.0200] (v54_01:55-71).
    v55/v56: literales del script (v55_01:55, v56_01:55).
    """
    if version == "v53":
        w48 = np.array(W_V48, dtype=np.float64)
        w48 = w48 / w48.sum()
        rem = 1.0 - W_DENSE_V53
        return np.array(list(w48 * rem) + [W_DENSE_V53])
    if version == "v54":
        w48 = np.array(W_V48, dtype=np.float64)
        w48 = w48 / w48.sum()
        base = w48 * (1.0 - W_DENSE_V54)
        reparto_v43, reparto_v47 = REPARTO_CRAMER_V54
        return np.array([base[0], base[1] * reparto_v43, base[1] * reparto_v47, base[2],
                         base[3], base[4], base[5], W_DENSE_V54])
    return np.array(CONSENSOS[version]["pesos"], dtype=np.float64)


def poe_ponderado(modelos: list[np.ndarray], pesos: list[float]) -> np.ndarray:
    """PoE ponderado en log-espacio con normalizacion final (v55_01:77-84, v56_01:77-85).

    Los pesos se normalizan (WEIGHTS_GEOM / WEIGHTS_GEOM.sum()); las probabilidades se
    recortan a [EPS, 1.0] antes del log; se desplaza por el maximo por fila y se
    renormalizan las filas. Entrada: modelos (n, 6) en orden de apilado.
    """
    w = np.asarray(pesos, dtype=np.float64)
    w = w / w.sum()
    log_geom = sum(wi * np.log(np.clip(m, EPS, 1.0)) for wi, m in zip(w, modelos))
    p_geom = np.exp(log_geom - np.max(log_geom, axis=1, keepdims=True))
    return p_geom / p_geom.sum(axis=1, keepdims=True)


def regla_plug_in(prob: np.ndarray) -> np.ndarray:
    """Decision plug-in por clase: argmax(prob * PESOS_PLUG_IN) (v53_02:88-89)."""
    return np.argmax(prob * PESOS_PLUG_IN, axis=1)
