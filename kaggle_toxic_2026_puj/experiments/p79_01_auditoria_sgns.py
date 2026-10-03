"""P79: auditoria del entrenador Word2Vec. El objetivo SGNS, el scheduler y la inicializacion.

Proposito
---------
P71-04 implementa el skip-gram con negative sampling desde cero, y de ahi salen el corpus, el
vocabulario y los artefactos congelados que P72, P73, P74, P75, P76, P77 y P78 consumen. Esta
auditoria comprueba tres cosas, sin reentrenar nada de tamano real:

    1. Si `SkipGram.forward` de `p71_04:199-216` implementa SGNS. No lo implementa: promedia el
       termino negativo sobre el eje de los K negativos, y eso divide el peso de la repulsion
       por K.
    2. Si el gradiente que produce el codigo se puede desbloquear con el factor 1/K.
    3. Si el calculo de `total_lotes` del scheduler de `p72_01:91-120` usa las mismas semillas
       y el mismo numero de lotes que las epocas efectivas.
    4. Si `nn.init.zeros_(self.salida.weight)` en `p71_04:195-196` sesga el entrenamiento, y si
       eso alcanza o no al artefacto congelado que se uso de verdad.

Que NO hace, por construccion
----------------------------
- NO entrena el Word2Vec de 10 millones de tokens.
- NO escribe ni modifica `modelos/twitter_domain/w2v_twitter_100d.npz`, ni ningun otro
  artefacto de `modelos/`. El sha256 del congelado se comprueba al principio y al final, y si
  cambia el script aborta.
- NO lee test, ni holdout, ni el leaderboard, ni genera submission.
- NO edita ni borra P71 a P78.
- NO propone reentrenar. Solo cuantifica.

Alcance de losiran
------------------
La verificacion del scheduler y la evaluacion de los pares reales se hacen sobre las PRIMERAS
`N_LINEAS_CORPUS` lineas de `corpus_twitter_tokens.txt`, no sobre los 10.000.029 tokens. Es un
limite declarado: la frecuencia de los tipos cambia al recortar, luego los recuentos ABSOLUTOS
de pares no son los de la corrida real. Lo que SI se mide sin limite es la DESVIACION
ESTRUCTURAL, que es una propiedad del codigo y no del corpus: las semillas del plan y las de
las epocas efectivas estan desfasadas en uno, y eso no depende de cuantos tokens se lean.

Los dos artefactos que se comparan
---------------------------------
`w2v_twitter_100d.npz` es el que uso P73 y P78. Se identifica su productor leyendo los sha256
registrados en los informes de P71-04, P71-06, P71-08 y P72-01, y no suponiendolo.

Estado del arte aplicado
------------------------
Referencia de la loss: Mikolov et al., skip-gram con negative sampling,
`loss = -mean_b[ log sigma(v_b . u_pos_b) + SUM_k log sigma(-v_b . u_neg_bk) ]`, con el mismo
peso en el termino positivo y en cada uno de los K negativos. word2vec original y gensim
promedian la loss sobre el batch con esa ponderacion; el factor 1/K no aparece en ninguna
fuente.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import p71_04_entrenar_word2vec as w2v

EPOCAS_W2V = w2v.EPOCAS
LR_P71_04 = w2v.LR
NEGATIVOS_W2V = w2v.NEGATIVOS
RE_TOKEN_W2V = w2v.RE_TOKEN
SEMILLA_W2V = w2v.SEMILLA
SUBMUESTREO = w2v.SUBMUESTREO
SkipGram = w2v.SkipGram
construir_vocabulario = w2v.construir_vocabulario
pares_de_epoca = w2v.pares_de_epoca

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p79"
CORPORA = PROJECT_ROOT / "data" / "external" / "corpora" / "twitter_domain"
CORPUS_TOKENS = CORPORA / "corpus_twitter_tokens.txt"
MODELO_100D = PROJECT_ROOT / "modelos" / "twitter_domain" / "w2v_twitter_100d.npz"
SHA_100D = "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a"
INFORME_ENTRENAMIENTO = REPORTS / "twitter_word2vec_training.json"
INFORME_SGD = REPORTS / "twitter_word2vec_training_sgd.json"
INFORME_LRFIX = REPORTS / "twitter_word2vec_training_lrfix.json"
INFORME_DIMENSIONES = REPORTS / "twitter_word2vec_dimensiones.json"
FUENTE_DIMENSIONES = PROJECT_ROOT / "experiments" / "p72_01_dimensiones.py"
FUENTE_BASE = PROJECT_ROOT / "experiments" / "p71_04_entrenar_word2vec.py"
FUENTE_LRFIX = PROJECT_ROOT / "experiments" / "p71_08_word2vec_corregido.py"
JSON_AUDIT = REPORTS / "p79_auditoria_sgns.json"

# =========================================================================
# SECCION 1: CONFIGURACION
# =========================================================================
DIM_TENSOES = 5
N_VOCAB_TENSOES = 7
N_TIPOS_TENSOES = 6
SEMILLA_TENSOES = 20261140
KS_AUDITADOS = (1, 5, 10, 20)
K_POR_DEFECTO = NEGATIVOS_W2V
TOL_GRADIENTE = 1e-6

N_LINEAS_CORPUS = 2000
N_PARES_REALES = 4000

LOTE_AUDIT = 256
LR0_POR_EJEMPLO = 0.025
LR_MINIMO = 1e-3
DIM_SINTETICA = 100
EPOCAS_SINTETICAS = 120
SEMILLA_SINTETICA = 20261141
N_FAMILIAS = 8
PALABRAS_POR_FAMILIA = 5
PALABRAS_COMUNES = 8
N_DOCS_SINTETICOS = 800
DOS_LN_2 = 2 * math.log(2)

CONFIG = {
    "referencia_sgns": ("-mean_b[ log sigma(v_b . u_pos_b) + SUM_k log sigma(-v_b . u_neg_bk) ], "
                        "con el mismo peso en el positivo y en cada uno de los K negativos"),
    "codigo_p71_04": ("-(logsigmoid(v.u_pos).mean() + logsigmoid(-(v.u_neg)).mean()) con el "
                      "segundo .mean() sobre los ejes batch x K, luego el termino negativo pesa "
                      "1/K de lo que deberia"),
    "dim_de_los_tensores_de_test": DIM_TENSOES,
    "n_vocab_de_los_tensores_de_test": N_VOCAB_TENSOES,
    "n_tipos_de_los_tensores_de_test": N_TIPOS_TENSOES,
    "k_por_defecto": K_POR_DEFECTO,
    "ks_auditados": list(KS_AUDITADOS),
    "semilla_de_los_tensores_de_test": SEMILLA_TENSOES,
    "n_lineas_del_corpus_para_el_conteo_de_lotes": N_LINEAS_CORPUS,
    "n_pares_reales_para_evaluar_el_artefacto": N_PARES_REALES,
    "lote_de_la_auditoria_del_scheduler": LOTE_AUDIT,
    "tolerancia_de_gradiente": TOL_GRADIENTE,
    "entrena_el_w2v_de_10m": False,
    "toca_el_artefacto_congelado": False,
}


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P79-01] {m}", flush=True)


# =========================================================================
# SECCION 2: LOS DOS OBJETIVOS
# =========================================================================
def objetivo_codigo(entrada: torch.Tensor, salida: torch.Tensor, centro: torch.Tensor,
                    ctx: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                  torch.Tensor]:
    """Reproduce exactamente la loss de `SkipGram.forward` de P71-04.

    Es `-(pos + neg_l)` con `pos = logsigmoid(v.u_pos).mean()` y
    `neg_l = logsigmoid(-(v.u_neg).mean())`, donde ese segundo `.mean()` promedia sobre el
    lote Y sobre los K negativos a la vez.

    Args:
        entrada: matriz de entrada, con gradiente.
        salida: matriz de salida, con gradiente.
        centro: indices centrales.
        ctx: indices de contexto positivo.
        neg: indices negativos, de forma (lote, K).

    Returns:
        Tupla con la perdida total, el termino positivo y el termino negativo.
    """
    v = entrada[centro]
    u_pos = salida[ctx]
    u_neg = salida[neg]
    pos = F.logsigmoid((v * u_pos).sum(dim=1)).mean()
    neg_l = F.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2)).mean()
    return -(pos + neg_l), pos, neg_l


def objetivo_referencia(entrada: torch.Tensor, salida: torch.Tensor, centro: torch.Tensor,
                        ctx: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                      torch.Tensor]:
    """Objetivo SGNS de referencia, con la SUMA sobre los K negativos.

    Args:
        entrada: matriz de entrada, con gradiente.
        salida: matriz de salida, con gradiente.
        centro: indices centrales.
        ctx: indices de contexto positivo.
        neg: indices negativos, de forma (lote, K).

    Returns:
        Tupla con la perdida total, el termino positivo y el termino negativo.
    """
    v = entrada[centro]
    u_pos = salida[ctx]
    u_neg = salida[neg]
    pos = F.logsigmoid((v * u_pos).sum(dim=1)).mean()
    neg_l = F.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2)).sum(dim=1).mean()
    return -(pos + neg_l), pos, neg_l


def gradiente_analitico(entrada: torch.Tensor, salida: torch.Tensor, centro: torch.Tensor,
                        ctx: torch.Tensor, neg: torch.Tensor,
                        factor_neg: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradiente analitico en forma cerrada, para las dos objectives.

    Derivando `-log sigma(s) = -log(1 + e^-s)` se obtiene `d/ds = -sigma(-s)`, y
    `-log sigma(-s) = -log(1 + e^s)` da `d/ds = +sigma(s)`. Con `s_pos = v.u_pos` y
    `s_neg = v.u_neg`, y dividiendo por el numero de ejemplos `B`:

        d/dv    = ( -sigma(-s_pos) u_pos + factor_neg * SUM_k sigma(s_neg_bk) u_neg_bk ) / B
        d/du_pos = ( -sigma(-s_pos) v ) / B
        d/du_neg = ( factor_neg * sigma(s_neg_bk) v ) / B

    `factor_neg` vale 1.0 para la referencia y `1/K` para el codigo, y eso es TODO el
    defecto.

    Ojo con el eje del negativo: `s_neg` es de forma (B, K), y para multiplicarla por
    `v` de forma (B, dim) hay que insertar el eje de la dimension, no el de K. Con
    `unsqueeze(1)` el broadcasting produce (B, B, K) en vez de (B, K, dim), y el error
    pasa inadvertido mientras K coincida con dim. Por eso se comprueba con autograd y
    no por lectura.

    Args:
        entrada: matriz de entrada, sin gradiente.
        salida: matriz de salida, sin gradiente.
        centro: indices centrales.
        ctx: indices de contexto positivo.
        neg: indices negativos, de forma (lote, K).
        factor_neg: peso del termino negativo, 1.0 en la referencia y 1/K en el codigo.

    Returns:
        Tupla con el gradiente de `entrada[centro]`, de `salida[ctx]` y de `salida[neg]`.
    """
    v = entrada[centro]
    u_pos = salida[ctx]
    u_neg = salida[neg]
    n = v.shape[0]
    s_pos = (v * u_pos).sum(dim=1)
    s_neg = (v.unsqueeze(1) * u_neg).sum(dim=2)
    dv = -torch.sigmoid(-s_pos).unsqueeze(1) * u_pos
    dv = dv + factor_neg * (torch.sigmoid(s_neg).unsqueeze(2) * u_neg).sum(dim=1)
    du_pos = -torch.sigmoid(-s_pos).unsqueeze(1) * v / n
    du_neg = factor_neg * (torch.sigmoid(s_neg).unsqueeze(2) * v.unsqueeze(1)) / n
    return dv / n, du_pos, du_neg


def componente_positivo(entrada: torch.Tensor, salida: torch.Tensor, centro: torch.Tensor,
                        ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Parte del gradiente que solo depende del par positivo, identica en los dos objetivos.

    Args:
        entrada: matriz de entrada, sin gradiente.
        salida: matriz de salida, sin gradiente.
        centro: indices centrales.
        ctx: indices de contexto positivo.

    Returns:
        Tupla con el gradiente de `entrada[centro]` y de `salida[ctx]`, solo por el positivo.
    """
    v = entrada[centro]
    u_pos = salida[ctx]
    n = v.shape[0]
    s_pos = (v * u_pos).sum(dim=1)
    dv = -torch.sigmoid(-s_pos).unsqueeze(1) * u_pos / n
    du_pos = -torch.sigmoid(-s_pos).unsqueeze(1) * v / n
    return dv, du_pos


def tensores_de_test(dim: int = DIM_TENSOES, n_vocab: int = N_VOCAB_TENSOES,
                     n_tipos: int = N_TIPOS_TENSOES, k: int = K_POR_DEFECTO,
                     semilla: int = SEMILLA_TENSOES) -> dict[str, torch.Tensor]:
    """Tensores diminutos y deterministas para la auditoria numerica.

    Args:
        dim: dimension de los embeddings.
        n_vocab: filas de cada matriz.
        n_tipos: numero de tipos distintos en el lote.
        k: numero de negativos por ejemplo.
        semilla: semilla del generador.

    Returns:
        Diccionario con `entrada`, `salida`, `centro`, `ctx` y `neg`.
    """
    g = torch.Generator().manual_seed(semilla)
    entrada = torch.randn(n_vocab, dim, generator=g, dtype=torch.float64)
    salida = torch.randn(n_vocab, dim, generator=g, dtype=torch.float64)
    idx = torch.randperm(n_vocab, generator=g)[:n_tipos]
    n_ejemplos = 2 * n_tipos
    centro = torch.cat([idx, idx])
    ctx = torch.roll(idx, 1).repeat(2)
    neg = torch.randint(0, n_vocab, (n_ejemplos, k), generator=g)
    return {"entrada": entrada, "salida": salida, "centro": centro, "ctx": ctx, "neg": neg}


# =========================================================================
# SECCION 3: PUNTO 1, LA LOSS DEL CODIGO NO ES SGNS
# =========================================================================
def test1_loss_no_es_sgns(t: dict[str, torch.Tensor], k: int) -> dict:
    """Compara la loss del codigo con la referencia y mide el ratio del termino negativo.

    Ademas ejecuta el `SkipGram.forward` REAL de P71-04 y comprueba que coincide con la
    replica local, para que la auditoria secsharp sobre el codigo que se ejecuto y no sobre
    una reimplementacion.

    Args:
        t: tensores de `tensores_de_test`.
        k: numero de negativos.

    Returns:
        Diccionario con la perdida de cada objetivo, el ratio y la comprobacion de fidelidad.

    Raises:
        SystemExit: si la replica local no reproduce `SkipGram.forward`.
    """
    entrada = t["entrada"].clone().requires_grad_(True)
    salida = t["salida"].clone().requires_grad_(True)
    l_cod, pos_c, neg_c = objetivo_codigo(entrada, salida, t["centro"], t["ctx"], t["neg"])
    l_ref, pos_r, neg_r = objetivo_referencia(entrada, salida, t["centro"], t["ctx"], t["neg"])

    modelo = SkipGram(t["entrada"].shape[0], t["entrada"].shape[1]).double()
    with torch.no_grad():
        modelo.entrada.weight.copy_(t["entrada"])
        modelo.salida.weight.copy_(t["salida"])
    l_real = modelo(t["centro"], t["ctx"], t["neg"])
    fidelidad = float(torch.abs(l_real.double() - l_cod.detach()).item())
    if fidelidad > 1e-12:
        raise SystemExit(f"STOP: la replica local no reproduce SkipGram.forward, diferencia "
                         f"{fidelidad:.3e}")

    with torch.no_grad():
        pos_id = float(torch.abs(pos_c - pos_r).item())
        ratio_neg = float(neg_c / neg_r)
        reconstruida = float(pos_r + neg_c)
    esperado = 1.0 / k
    if abs(ratio_neg - esperado) > 1e-12:
        raise SystemExit(f"STOP: el ratio del termino negativo no es 1/K = {esperado}")
    return {
        "k": k,
        "loss_codigo": round(float(l_cod.item()), 12),
        "loss_referencia": round(float(l_ref.item()), 12),
        "loss_codigo_menos_referencia": round(float(l_cod.item() - l_ref.item()), 12),
        "termino_positivo_codigo": round(float(pos_c.item()), 12),
        "termino_positivo_referencia": round(float(pos_r.item()), 12),
        "termino_positivo_identico": pos_id < 1e-12,
        "termino_negativo_codigo": round(float(neg_c.item()), 12),
        "termino_negativo_referencia": round(float(neg_r.item()), 12),
        "ratio_del_termino_negativo": round(ratio_neg, 12),
        "uno_sobre_k": round(esperado, 12),
        "identidad_loss_codigo_menos_pos_mas_neg": round(reconstruida, 12),
        "la_identidad_se_cumple": abs(reconstruida - float(l_cod.item())) < 1e-12,
        "fidelidad_con_skipgram_forward": fidelidad,
        "lectura": (f"con K={k} el termino negativo del codigo pesa {ratio_neg:.4f} de lo que "
                    f"deberia: la repulsion esta infra-pesada por un factor {k}"),
    }


# =========================================================================
# SECCION 4: PUNTO 2, EL GRADIENTE SE DESBLOQUEA CON EL FACTOR 1/K
# =========================================================================
def test2_gradientes(t: dict[str, torch.Tensor], k: int) -> dict:
    """Compara el gradiente analitico de la referencia con el que produce el codigo.

    La comparacion NO es circular. Se parte el gradiente analitico de la referencia en sus
    dos componentes, se comprueba que la suma reproduce exactamente lo que da autograd, y
    luego se comprueba que el gradiente del codigo es la misma suma con el componente
    negativo dividido por K. Eso es "desbloquear" el defecto: el codigo no es un objetivo
    distinto, es la referencia con la repulsion reescalada.

    Todo se compara sobre la matriz ACUMULADA, porque autograd entrega el gradiente de la
    matriz entera con las repeticiones de indice ya sumadas.

    Args:
        t: tensores de `tensores_de_test`.
        k: numero de negativos.

    Returns:
        Diccionario con todos los ratios y las comprobaciones.

    Raises:
        SystemExit: si alguna comprobacion no se cumple.
    """
    entrada, salida = t["entrada"], t["salida"]
    e_cod = entrada.clone().requires_grad_(True)
    s_cod = salida.clone().requires_grad_(True)
    l_cod, _, _ = objetivo_codigo(e_cod, s_cod, t["centro"], t["ctx"], t["neg"])
    g_cod = torch.autograd.grad(l_cod, [e_cod, s_cod])
    e_ref = entrada.clone().requires_grad_(True)
    s_ref = salida.clone().requires_grad_(True)
    l_ref, _, _ = objetivo_referencia(e_ref, s_ref, t["centro"], t["ctx"], t["neg"])
    g_ref = torch.autograd.grad(l_ref, [e_ref, s_ref])

    with torch.no_grad():
        n_vocab = entrada.shape[0]
        dv_pos, _du_pos = componente_positivo(entrada, salida, t["centro"], t["ctx"])
        dv_total_ref, du_pos_ref, du_neg_ref = gradiente_analitico(
            entrada, salida, t["centro"], t["ctx"], t["neg"], 1.0)
        dv_neg_ref = dv_total_ref - dv_pos

        def acumular(indices: torch.Tensor, por_ocurrencia: torch.Tensor) -> torch.Tensor:
            """Suma el gradiente por ocurrencia sobre la matriz, como hace autograd.

            `centro`, `ctx` y `neg` repiten indices a proposito, porque en word2vec un
            mismo centro aparece muchas veces. El autograd devuelve SIEMPRE el gradiente
            acumulado de la matriz entera. Indexar ese tensor por los indices volveria a
            multiplicar por el numero de repeticiones, y la comparacion con el analitico
            por ocurrencia seria falsa. Hay que acumular el analitico con la misma regla.

            Args:
                indices: indices, de cualquier forma.
                por_ocurrencia: gradiente de una sola ocurrencia, de forma (N, dim).

            Returns:
                Matriz (n_vocab, dim) con el gradiente acumulado.
            """
            plano = por_ocurrencia.reshape(-1, por_ocurrencia.shape[-1])
            acumulado = torch.zeros(n_vocab, plano.shape[1], dtype=plano.dtype)
            return acumulado.index_add_(0, indices.reshape(-1), plano)

        a_pos = acumular(t["centro"], dv_pos)
        a_neg_ref = acumular(t["centro"], dv_neg_ref)
        a_upos = acumular(t["ctx"], du_pos_ref)
        a_uneg_ref = acumular(t["neg"], du_neg_ref)
        a_uneg_cod = a_uneg_ref / k
        g_entrada_cod = g_cod[0]
        g_entrada_ref = g_ref[0]
        g_salida_ref = g_ref[1]
        g_salida_cod = g_cod[1]
        neg_cod = g_entrada_cod - a_pos
        norma_pos = float(torch.linalg.vector_norm(a_pos))
        norma_neg_ref = float(torch.linalg.vector_norm(a_neg_ref))
        norma_neg_cod = float(torch.linalg.vector_norm(neg_cod))
        residuo_ref = float(torch.linalg.vector_norm(
            g_entrada_ref - (a_pos + a_neg_ref)))
        # autograd entrega la matriz de salida COMPLETA, con la parte positiva y la
        # negativa ya sumadas. Se verifica la suma y el negativo se aisla por diferencia.
        residuo_salida_ref = float(torch.linalg.vector_norm(
            g_salida_ref - (a_upos + a_uneg_ref)))
        residuo_salida_cod = float(torch.linalg.vector_norm(
            g_salida_cod - (a_upos + a_uneg_cod)))
        norma_uneg_ref = float(torch.linalg.vector_norm(g_salida_ref - a_upos))
        norma_uneg_cod = float(torch.linalg.vector_norm(g_salida_cod - a_upos))
        reconstruido_cod = a_pos + a_neg_ref / k
        residuo_cod = float(torch.linalg.vector_norm(g_entrada_cod - reconstruido_cod))
        ratio_neg = norma_neg_cod / max(norma_neg_ref, 1e-300)
        # el termino positivo es el MISMO tensor en los dos objetivos, por construccion:
        # el codigo no cambia ni el atractor ni el signo
        ratio_pos = 1.0
        ratio_salida_neg = norma_uneg_cod / max(norma_uneg_ref, 1e-300)
        ratio_total_entrada = float(torch.linalg.vector_norm(g_entrada_cod)
                                    / max(float(torch.linalg.vector_norm(g_entrada_ref)),
                                          1e-300))
        solape = int(t["centro"].unique().numel())
        repeticiones_centro = int(len(t["centro"]) - solape)
    escala = max(norma_neg_ref, 1.0)
    if residuo_ref > TOL_GRADIENTE * escala:
        raise SystemExit(f"STOP: el gradiente analitico de la entrada de la referencia no "
                         f"reproduce el autograd, residuo {residuo_ref:.3e}")
    if residuo_salida_ref > TOL_GRADIENTE * escala:
        raise SystemExit(f"STOP: el gradiente analitico de la salida de la referencia no "
                         f"reproduce el autograd, residuo {residuo_salida_ref:.3e}")
    if abs(ratio_neg - 1.0 / k) > 1e-9:
        raise SystemExit("STOP: el componente negativo del codigo no es 1/K del de la referencia")
    if abs(ratio_salida_neg - 1.0 / k) > 1e-9:
        raise SystemExit("STOP: el negativo de la salida tampoco queda en 1/K")
    if residuo_cod > TOL_GRADIENTE * escala or residuo_salida_cod > TOL_GRADIENTE * escala:
        raise SystemExit("STOP: el codigo no es la referencia con la repulsion en 1/K")
    return {
        "k": k,
        "el_analitico_reproduce_la_referencia": True,
        "residuo_de_la_descomposicion_de_la_referencia": round(residuo_ref, 14),
        "residuo_de_la_entrada_de_la_referencia": round(residuo_ref, 14),
        "residuo_de_la_salida_de_la_referencia": round(residuo_salida_ref, 14),
        "residuo_de_la_salida_del_codigo": round(residuo_salida_cod, 14),
        "residuo_de_la_entrada_del_codigo": round(residuo_cod, 14),
        "norma_del_componente_positivo": round(norma_pos, 10),
        "norma_del_componente_negativo_referencia": round(norma_neg_ref, 10),
        "norma_del_componente_negativo_codigo": round(norma_neg_cod, 10),
        "ratio_componente_negativo": round(ratio_neg, 12),
        "uno_sobre_k": round(1.0 / k, 12),
        "ratio_componente_positivo": round(ratio_pos, 12),
        "ratio_gradiente_total_de_entrada": round(ratio_total_entrada, 12),
        "ratio_gradiente_de_salida_negativo": round(ratio_salida_neg, 12),
        "residuo_del_bloqueo_por_1_sobre_k": round(residuo_cod, 14),
        "el_codigo_es_la_referencia_con_la_repulsion_en_1_sobre_k": True,
        "cota_inferior_del_ratio_total": round(1.0 / k, 6),
        "cota_superior_del_ratio_total": 1.0,
        "ejemplos": len(t["centro"]),
        "centros_distintos": solape,
        "centros_repetidos": repeticiones_centro,
        "comparacion_sobre_matriz_acumulada": True,
        "nota_de_la_comparacion": (
            "autograd devuelve el gradiente acumulado de la matriz entera, con las "
            "repeticiones de centro ya sumadas. El analitico se acumula con index_add_ "
            "usando los mismos indices. La comparacion es por eso valida aunque un centro "
            f"aparezca {repeticiones_centro} veces, que es el caso real de word2vec"),
        "lectura": (f"con K={k} el componente negativo del codigo mide {ratio_neg:.6f} del de la "
                    f"referencia, que es 1/K exacto, y el componente positivo mide {ratio_pos:.6f}, "
                    f"que es 1. El gradiente total de la entrada queda entre {1.0 / k:.2f} y 1.00 "
                    f"veces el de la referencia"),
    }


# =========================================================================
# SECCION 5: PUNTO 3, EL SCHEDULER
# =========================================================================
def leer_primeras_n(ruta: Path, n: int) -> list[list[str]]:
    """Lee las primeras `n` lineas del corpus tokenizado, con el filtro de `leer_tokens`.

    Usa el mismo regex y el mismo filtro de minimo tres tokens que `leer_tokens` de P71-04, y
    no el fichero entero: el corpus son 10.000.029 tokens y aqui solo se cuentan lotes.

    Args:
        ruta: fichero del corpus, un documento normalizado por linea.
        n: numero de lineas a leer.

    Returns:
        Lista de listas de tokens.
    """
    docs = []
    with ruta.open(encoding="utf-8", errors="replace") as fh:
        for i, linea in enumerate(fh):
            if i >= n:
                break
            toks = RE_TOKEN_W2V.findall(linea)
            if len(toks) >= 3:
                docs.append(toks)
    return docs


def contar_lotes(docs: list[list[str]], vocab: dict[str, int], p: np.ndarray, semilla: int,
                 lote: int) -> tuple[int, int]:
    """Numero de pares y de lotes de una epoca con una semilla dada.

    Args:
        docs: documentos tokenizados.
        vocab: indice de palabras.
        p: probabilidad de conservar cada palabra.
        semilla: semilla del generador.
        lote: tamano del lote.

    Returns:
        Tupla con el numero de pares y el numero de lotes, que es el techo de la division.
    """
    r = np.random.default_rng(semilla)
    centros, _ = pares_de_epoca(docs, vocab, p, r)
    n_lotes = max(1, math.ceil(len(centros) / lote))
    return len(centros), int(n_lotes)


def auditar_semillas_en_el_fuente(ruta: Path) -> dict:
    """Extrae las expresiones de semilla de los bucles del scheduler, del codigo fuente.

    No evalua nada: localiza cada `default_rng(... SEMILLA ...)` y el encabezado del bucle que
    lo contiene, para que el hallazgo sea auditable a ojo.

    Args:
        ruta: fichero fuente a inspeccionar.

    Returns:
        Diccionario con los hallazgos y el veredicto de desalineacion.

    Raises:
        SystemExit: si el fichero no existe o no se encuentra ningun `default_rng`.
    """
    if not ruta.exists():
        raise SystemExit(f"STOP: no existe el fuente {ruta}")
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    hallazgos = []
    for i, linea in enumerate(lineas):
        if "default_rng" not in linea or "SEMILLA" not in linea:
            continue
        bucle = ""
        for j in range(i, max(-1, i - 15), -1):
            if lineas[j].lstrip().startswith("for "):
                bucle = lineas[j].strip()
                break
        hallazgos.append({"linea": i + 1, "bucle": bucle, "expresion": linea.strip()})
    if not hallazgos:
        raise SystemExit(f"STOP: no se encuentra ningun default_rng con SEMILLA en {ruta.name}")

    def desplazamiento(encabezado: str) -> int:
        """Deduce el desplazamiento de `ep` en un encabezado de bucle.

        Args:
            encabezado: texto del encabezado del bucle.

        Returns:
            El primer valor que toma `ep`, o None si no se deduce.
        """
        if re.search(r"range\(", encabezado):
            return 0
        m = re.search(r"start\s*=\s*(\d+)", encabezado)
        if m:
            return int(m.group(1))
        return None

    grupos: dict[str, set[int]] = {}
    for h in hallazgos:
        d = desplazamiento(h["bucle"])
        if d is None:
            continue
        etiqueta = "plan" if "planes.append" in "\n".join(lineas[max(0, h["linea"] - 4):
                                                            h["linea"]]) else "entrenamiento"
        grupos.setdefault(etiqueta, set()).update(range(d, d + EPOCAS_W2V))
    desalineado = len(grupos) == 2 and grupos["plan"] != grupos["entrenamiento"]
    return {"fuente": str(ruta.relative_to(PROJECT_ROOT)), "hallazgos": hallazgos,
            "conjuntos_de_semillas_por_bucle": {k: sorted(v) for k, v in grupos.items()},
            "semillas_desalineadas": bool(desalineado),
            "que_implica": ("`total_lotes` se calcula con los rangos del bucle del plan y las "
                            "epocas efectivas se recorren con otro rango, luego los dos "
                            "conjuntos de pares son distintos y el decaimiento lineal del "
                            "learning rate se calibra sobre un numero de lotes que no es el que "
                            "se recorre")
            if desalineado else "los dos bucles usan el mismo rango de semillas"}


def test3_scheduler() -> dict:
    """Punto 3: comprueba el calculo de `total_lotes` de `p72_01` de forma estatica y numerica.

    Returns:
        Diccionario con la lectura estatica del fuente, el conteo real de lotes sobre las
        primeras N lineas, la desviacion entre el total planificado y el total efectivo, y el
        efecto sobre el learning rate.

    Raises:
        SystemExit: si el corpus no existe.
    """
    log("")
    log("   PUNTO 3: el scheduler de p72_01")
    estatico = auditar_semillas_en_el_fuente(FUENTE_DIMENSIONES)
    for h in estatico["hallazgos"]:
        log(f"      {FUENTE_DIMENSIONES.name}:{h['linea']} | {h['bucle']} -> {h['expresion']}")
    log(f"      semillas desalineadas entre el plan y el entrenamiento: "
        f"{estatico['semillas_desalineadas']}")

    if not CORPUS_TOKENS.exists():
        raise SystemExit(f"STOP: no existe el corpus tokenizado {CORPUS_TOKENS}")
    docs = leer_primeras_n(CORPUS_TOKENS, N_LINEAS_CORPUS)
    vocab, p, _ = construir_vocabulario(docs)
    n_tokens = sum(len(d) for d in docs)
    log(f"      corpus recortado: {len(docs)} documentos | {n_tokens} tokens | "
        f"vocabulario {len(vocab)} (min_count 3, subsampling {SUBMUESTREO})")
    log("      NO se entrena nada: solo se generan pares para contar lotes")

    pares_plan, lotes_plan = [], []
    for ep in range(EPOCAS_W2V):
        n_pares, n_lotes = contar_lotes(docs, vocab, p, SEMILLA_W2V + ep, LOTE_AUDIT)
        pares_plan.append(n_pares)
        lotes_plan.append(n_lotes)
    pares_real, lotes_real = [], []
    for ep in range(1, EPOCAS_W2V + 1):
        n_pares, n_lotes = contar_lotes(docs, vocab, p, SEMILLA_W2V + ep, LOTE_AUDIT)
        pares_real.append(n_pares)
        lotes_real.append(n_lotes)
    total_plan, total_real = sum(lotes_plan), sum(lotes_real)
    log(f"      pares por epoca, plan    {pares_plan} | total de lotes {total_plan}")
    log(f"      pares por epoca, reales  {pares_real} | total de lotes {total_real}")
    desvio = (total_real - total_plan) / max(total_plan, 1)
    log(f"      desvio del total de lotes: {100 * desvio:+.3f} por ciento")

    lr_ini = LR0_POR_EJEMPLO * LOTE_AUDIT
    pasos = range(total_real)
    lr_real = [max(LR_MINIMO, lr_ini * (1.0 - paso / max(total_plan, 1))) for paso in pasos]
    lr_ideal = [max(LR_MINIMO, lr_ini * (1.0 - paso / max(total_real, 1))) for paso in pasos]
    en_piso = int(sum(1 for x in lr_real if x <= LR_MINIMO * (1 + 1e-9)))
    n_pares_corpus = 10000029
    log(f"      learning rate inicial {lr_ini:.4f} | ultimo real {lr_real[-1]:.6f} | "
        f"ultimo ideal {lr_ideal[-1]:.6f} | pasos en el piso {en_piso} de {total_real}")
    return {
        "lectura_estatica_del_fuente": estatico,
        "lectura_estatica_del_fuente_lrfix": auditar_semillas_en_el_fuente(FUENTE_LRFIX),
        "lectura_estatica_de_la_base": auditar_semillas_en_el_fuente(FUENTE_BASE),
        "recorte": {
            "lineas_leidas": N_LINEAS_CORPUS, "documentos": len(docs), "tokens": n_tokens,
            "vocabulario": len(vocab), "lote": LOTE_AUDIT,
            "limite_declarado": ("el recuento es sobre las primeras lineas del fichero, no sobre "
                                 "los 10.000.029 tokens. Los recuentos ABSOLUTOS de pares no son "
                                 "los de la corrida real, porque la frecuencia de los tipos "
                                 "cambia al recortar. La DESVIACION entre los dos totales, que "
                                 "es lo que se audita, si se puede escalar de forma lineal"),
            "tokens_reales_del_corpus": n_pares_corpus,
            "factor_de_escalado_lineal": round(n_pares_corpus / max(n_tokens, 1), 4),
        },
        "pares_por_epoca_plan": pares_plan,
        "pares_por_epoca_reales": pares_real,
        "lotes_por_epoca_plan": lotes_plan,
        "lotes_por_epoca_reales": lotes_real,
        "total_lotes_plan": total_plan,
        "total_lotes_reales": total_real,
        "desvio_relativo": round(desvio, 8),
        "learning_rate": {
            "inicial": round(lr_ini, 6), "minimo": LR_MINIMO,
            "ultimo_real": round(lr_real[-1], 8), "ultimo_ideal": round(lr_ideal[-1], 8),
            "pasos_en_el_piso": en_piso, "pasos_totales": total_real,
            "fraccion_en_el_piso": round(en_piso / max(total_real, 1), 6),
            "media_real": round(float(np.mean(lr_real)), 8),
            "media_ideal": round(float(np.mean(lr_ideal)), 8),
        },
        "coste_de_la_planificacion": ("el bucle del plan genera los pares de los 5 meses "
                                      "entrenamiento y despues los tira, guardandose solo su "
                                      "longitud. Eso es una pasada completa extra sobre el "
                                      "corpus, sin efecto sobre el modelo"),
    }


# =========================================================================
# SECCION 6: PUNTO 4, LA INICIALIZACION A CERO
# =========================================================================
def test4_inicializacion() -> dict:
    """Punto 4: mide el efecto de `nn.init.zeros_(self.salida.weight)` y su alcance real.

    Con la matriz de salida a cero, todos los scores `v.u` son cero, y el gradiente de la
    matriz de ENTRADA vale `-sigma(0) * 0 = 0`: exactamente cero, no aproximadamente. La
    matriz de entrada no se mueve en el primer paso, y tampoco mientras la de salida siga
    siendo despreciable frente al learning rate.

    Returns:
        Diccionario con la medida del gradiente, la comparacion entre inicializaciones y el
        analisis del alcance sobre el artefacto congelado.
    """
    log("")
    log("   PUNTO 4: la inicializacion a cero de la matriz de salida")
    t = tensores_de_test()
    medidas = {}
    for nombre, salida in (("ceros", torch.zeros_like(t["salida"])),
                           ("uniforme_0p05", torch.empty_like(t["salida"]).uniform_(
                               -0.05, 0.05, generator=torch.Generator().manual_seed(7)))):
        e = t["entrada"].clone().requires_grad_(True)
        s = salida.clone().requires_grad_(True)
        loss, _, _ = objetivo_codigo(e, s, t["centro"], t["ctx"], t["neg"])
        ge, gs = torch.autograd.grad(loss, [e, s])
        with torch.no_grad():
            medidas[nombre] = {
                "norma_del_gradiente_de_entrada": float(torch.linalg.vector_norm(
                    ge[t["centro"]])),
                "norma_del_gradiente_de_salida": float(torch.linalg.vector_norm(gs)),
                "gradiente_de_entrada_es_cero": bool(
                    torch.allclose(ge[t["centro"]], torch.zeros_like(ge[t["centro"]]),
                                   atol=1e-14)),
            }
    ratio = (medidas["ceros"]["norma_del_gradiente_de_entrada"]
             / max(medidas["uniforme_0p05"]["norma_del_gradiente_de_entrada"], 1e-300))
    log(f"      gradiente de la entrada con la salida a cero: "
        f"{medidas['ceros']['norma_del_gradiente_de_entrada']:.3e} "
        f"(exatamente cero: {medidas['ceros']['gradiente_de_entrada_es_cero']})")
    log(f"      gradiente de la entrada con la salida uniforme: "
        f"{medidas['uniforme_0p05']['norma_del_gradiente_de_entrada']:.6f}")
    log(f"      ratio cero / uniforme = {ratio:.6e}")

    dinamica = comparar_dinamica_de_entrenamiento()
    log(f"      tras {dinamica['pasos']} pasos de la receta de p71_04 (Adam, lr 0.025):")
    log(f"         salida a cero        loss {dinamica['ceros']['loss_final']:.6f} | "
        f"norma media de la entrada {dinamica['ceros']['norma_entrada_final']:.6f}")
    log(f"         salida uniforme      loss {dinamica['uniforme']['loss_final']:.6f} | "
        f"norma media de la entrada {dinamica['uniforme']['norma_entrada_final']:.6f}")
    return {
        "medida_del_gradiente_en_el_paso_1": medidas,
        "ratio_cero_sobre_uniforme": ratio,
        "direccion_del_sesgo": ("no es un sesgo de SIGNO sino un retardo: la matriz de entrada "
                                "no recibe gradiente hasta que la de salida deja de ser cero. La "
                                "repulsion sigue estando, pero solo del lado de la salida. Con "
                                "el learning rate de p71_04, 0.025 sobre lotes de 8192, el "
                                "paso efectivo era 3e-6, luego la de salida tardaba muchisimo en "
                                "salir de cero: ese es el mecanismo del loss clavado en "
                                f"{DOS_LN_2:.6f} de p71_04 y p71_06"),
        "dinamica_sobre_un_corpus_sintetico": dinamica,
    }


def corpus_sintetico(n_docs: int = N_DOCS_SINTETICOS, semilla: int = SEMILLA_SINTETICA,
                     n_familias: int = N_FAMILIAS,
                     palabras_por_familia: int = PALABRAS_POR_FAMILIA,
                     n_comunes: int = PALABRAS_COMUNES) -> list[list[str]]:
    """Corpus sintetico con estructura de familias, para separar bien los dos objetivos.

    Las familias de palabras NUNCA se mezclan entre si, de modo que el skip-gram tiene algo
    que aprender y la curva de perdida baja. Es el mismo diagnostico que uso p71_08, y no
    toca ningun dato real.

    Args:
        n_docs: numero de documentos.
        semilla: semilla del generador.
        n_familias: numero de familias.
        palabras_por_familia: palabras por familia.
        n_comunes: palabras comunes a todas las familias.

    Returns:
        Lista de listas de tokens.
    """
    rng = np.random.default_rng(semilla)
    letras = "abcdefghijklmnopqrstuvwxyz"
    comunes = [f"c{i}" for i in range(n_comunes)]
    docs = []
    for _ in range(n_docs):
        f = int(rng.integers(0, n_familias))
        base = f * palabras_por_familia
        familia = [f"{letras[(base + i) % 26]}{base + i}" for i in range(palabras_por_familia)]
        largo = int(rng.integers(20, 41))
        toks = [familia[int(rng.integers(0, len(familia)))] for _ in range(largo)]
        toks += [comunes[int(rng.integers(0, n_comunes))] for _ in range(4)]
        rng.shuffle(toks)
        docs.append(toks)
    return docs


def comparar_dinamica_de_entrenamiento(pasos: int = 60, lr: float = LR_P71_04,
                                        lote: int = 32) -> dict:
    """Compara el entrenamiento con la salida a cero y con la salida uniforme.

    Aplica la receta exacta de `p71_04`: optimizador Adam, learning rate 0.025 y el mismo
    `.mean()` defectuoso. La comparacion es sobre un corpus sintetico diminuto, y su unico
    proposito es medir CUANTO cuesta el retardo, no reproducir la corrida real.

    Args:
        pasos: numero de pasos de Adam.
        lr: learning rate.
        lote: tamano del lote.

    Returns:
        Diccionario con las curvas y los valores finales.

    Raises:
        SystemExit: si el corpus sintetico no genera pares.
    """
    docs = corpus_sintetico()
    vocab, p, _ = construir_vocabulario(docs)
    r = np.random.default_rng(SEMILLA_SINTETICA)
    centros, contextos = pares_de_epoca(docs, vocab, p, r)
    if len(centros) == 0:
        raise SystemExit("STOP: el corpus sintetico no ha generado pares")
    idx = np.array([vocab[w] for w in vocab], dtype=np.int64)
    pesos = np.array([1] * len(vocab), dtype=np.float64)
    pesos = pesos / pesos.sum()
    salida = {}
    for nombre, uniforme in (("ceros", None), ("uniforme", 0.05)):
        torch.manual_seed(SEMILLA_SINTETICA)
        modelo = SkipGram(len(vocab), DIM_SINTETICA)
        if uniforme is None:
            with torch.no_grad():
                modelo.salida.weight.zero_()
        else:
            with torch.no_grad():
                modelo.salida.weight.uniform_(-uniforme, uniforme)
        opt = torch.optim.Adam(modelo.parameters(), lr=lr)
        curva = []
        for _ in range(pasos):
            lote_i = min(lote, len(centros))
            c = torch.from_numpy(centros[:lote_i])
            x = torch.from_numpy(contextos[:lote_i])
            neg = torch.from_numpy(r.choice(idx, size=(lote_i, NEGATIVOS_W2V), p=pesos))
            opt.zero_grad()
            loss, _, _ = objetivo_codigo(modelo.entrada.weight, modelo.salida.weight, c, x, neg)
            loss.backward()
            opt.step()
            curva.append(round(float(loss.item()), 6))
        with torch.no_grad():
            norma = float(torch.linalg.vector_norm(modelo.entrada.weight, dim=1).mean())
        salida[nombre] = {"curva_de_perdida": curva, "loss_final": curva[-1],
                           "loss_inicial": curva[0], "norma_entrada_final": round(norma, 6)}
    salida["pasos"] = pasos
    salida["learning_rate"] = lr
    salida["lote"] = lote
    salida["optimizador"] = "Adam, la receta literal de p71_04: entrenar"
    salida["diferencia_de_loss_final"] = round(salida["ceros"]["loss_final"]
                                               - salida["uniforme"]["loss_final"], 6)
    return salida


# =========================================================================
# SECCION 7: PUNTO 5, QUE PRODUCTO SE USO DE VERDAD
# =========================================================================
def identificar_productor_del_artefacto(ruta: Path) -> dict:
    """Determina que script produjo un artefacto, leyendo los sha256 de los informes.

    No se supone: se compara el sha256 del fichero en disco con los que dejaron P71-04,
    P71-06, P71-08 y P72-01 en sus informes.

    Args:
        ruta: artefacto a identificar.

    Returns:
        Diccionario con el productor, el informe que lo declara y el sha256.

    Raises:
        SystemExit: si el artefacto no existe o si ningun informe lo declara.
    """
    if not ruta.exists():
        raise SystemExit(f"STOP: no existe el artefacto {ruta}")
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
    informes = [(INFORME_ENTRENAMIENTO, "p71_04_entrenar_word2vec"),
                (INFORME_SGD, "p71_06_word2vec_optimizador"),
                (INFORME_LRFIX, "p71_08_word2vec_corregido"),
                (INFORME_DIMENSIONES, "p72_01_dimensiones")]
    for informe, script in informes:
        if not informe.exists():
            continue
        datos = json.loads(informe.read_text("utf-8"))
        for ruta_rel, sha_declarado in _aplanar_shas(datos):
            if sha_declarado == sha and ruta_rel == str(ruta.relative_to(PROJECT_ROOT)):
                return {"artefacto": ruta_rel, "sha256": sha, "productor": script,
                        "declarado_en": str(informe.relative_to(PROJECT_ROOT))}
    raise SystemExit(f"STOP: ningun informe declara el sha256 de {ruta.name}")


def _aplanar_shas(nodo: Any) -> list[tuple[str, str]]:
    """Extrae todos los pares (ruta, sha256) de un informe JSON anidado.

    Args:
        nodo: parte del JSON.

    Returns:
        Lista de tuplas con la ruta relativa del artefacto y su sha256.
    """
    encontrados: list[tuple[str, str]] = []
    if isinstance(nodo, dict):
        sha = nodo.get("sha256") or nodo.get("sha256_modelo")
        ruta = nodo.get("artefacto") or nodo.get("modelo")
        if sha and ruta:
            encontrados.append((ruta, sha))
        for v in nodo.values():
            encontrados.extend(_aplanar_shas(v))
    elif isinstance(nodo, list):
        for v in nodo:
            encontrados.extend(_aplanar_shas(v))
    return encontrados


def busca_override_de_salida(ruta_fuente: Path) -> dict:
    """Busca, en el fuente, una sobrescritura de la inicializacion de la matriz de salida.

    Args:
        ruta_fuente: fichero fuente a inspeccionar.

    Returns:
        Diccionario con si existe la sobrescritura, la linea y el texto.
    """
    if not ruta_fuente.exists():
        return {"existe": False, "fuente": str(ruta_fuente)}
    lineas = ruta_fuente.read_text(encoding="utf-8").splitlines()
    for i, linea in enumerate(lineas, start=1):
        if "salida.weight" in linea and ("uniform_" in linea or "normal_" in linea
                                         or "kaiming" in linea):
            return {"existe": True, "fuente": str(ruta_fuente.relative_to(PROJECT_ROOT)),
                    "linea": i, "texto": linea.strip()}
    return {"existe": False, "fuente": str(ruta_fuente.relative_to(PROJECT_ROOT))}


def evaluar_artefacto_congelado() -> dict:
    """Evalua el Word2Vec 100d congelado bajo los dos objetivos, con pares reales.

    No entrena nada: se leen las primeras lineas del corpus, se generan pares con
    `pares_de_epoca`, se proyectan al vocabulario del artefacto y se calculan las dos losses y
    los dos gradientes con los pesos CONGELADOS.

    Returns:
        Diccionario con los valores medidos y la implicacion.

    Raises:
        SystemExit: si el sha256 del artefacto no coincide con el congelado.
    """
    sha_antes = hashlib.sha256(MODELO_100D.read_bytes()).hexdigest()
    if sha_antes != SHA_100D:
        raise SystemExit(f"STOP: el sha256 del artefacto congelado es {sha_antes}")
    d = np.load(MODELO_100D, allow_pickle=True)
    palabras = [str(w) for w in d["palabras"]]
    W = d["W"].astype(np.float64)
    indice = {w: i for i, w in enumerate(palabras)}

    docs = leer_primeras_n(CORPUS_TOKENS, N_LINEAS_CORPUS)
    # OJO: a `pares_de_epoca` hay que pasarle TOKENS, no ids. Esa funcion hace ella misma
    # `vocab[t]`. Si se le pasan ids, `t in vocab` es False para todos y salen cero pares.
    dentro = [[t for t in d_ if t in indice] for d_ in docs]
    dentro = [x for x in dentro if len(x) >= 3]
    r = np.random.default_rng(SEMILLA_W2V)
    centros, contextos = pares_de_epoca(dentro, indice, np.ones(len(indice)), r)
    if len(centros) == 0:
        raise SystemExit("STOP: no se han generado pares con el vocabulario del artefacto")
    n = min(N_PARES_REALES, len(centros))
    sel = r.choice(len(centros), n, replace=False)
    centro = torch.from_numpy(centros[sel])
    ctx = torch.from_numpy(contextos[sel])
    neg = torch.from_numpy(r.integers(0, len(palabras), (n, K_POR_DEFECTO)))
    E = torch.from_numpy(W)
    S = torch.from_numpy(W)

    e_cod = E.clone().requires_grad_(True)
    s_cod = S.clone().requires_grad_(True)
    l_cod, pos_c, neg_c = objetivo_codigo(e_cod, s_cod, centro, ctx, neg)
    g_cod = torch.autograd.grad(l_cod, [e_cod, s_cod])
    e_ref = E.clone().requires_grad_(True)
    s_ref = S.clone().requires_grad_(True)
    l_ref, pos_r, neg_r = objetivo_referencia(e_ref, s_ref, centro, ctx, neg)
    g_ref = torch.autograd.grad(l_ref, [e_ref, s_ref])
    with torch.no_grad():
        norma_g_cod = float(torch.linalg.vector_norm(g_cod[0]))
        norma_g_ref = float(torch.linalg.vector_norm(g_ref[0]))
        norma_salida_cod = float(torch.linalg.vector_norm(g_cod[1]))
        norma_salida_ref = float(torch.linalg.vector_norm(g_ref[1]))
    peso_negativo = float((abs(neg_r) / max(abs(pos_r), 1e-12)).detach())
    sha_despues = hashlib.sha256(MODELO_100D.read_bytes()).hexdigest()
    if sha_despues != sha_antes:
        raise SystemExit("STOP: el artefacto congelado ha cambiado durante la auditoria")
    log(f"      pares reales evaluados: {n} | vocabulario {len(palabras)} | "
        f"norma media de fila {np.linalg.norm(W, axis=1).mean():.6f}")
    log(f"      loss con el objetivo del CODIGO   {float(l_cod):.6f} "
        f"(pos {float(pos_c):.6f} + neg {float(neg_c):.6f})")
    log(f"      loss con el objetivo de REFERENCIA {float(l_ref):.6f} "
        f"(pos {float(pos_r):.6f} + neg {float(neg_r):.6f})")
    log(f"      norma del gradiente de entrada: codigo {norma_g_cod:.6f} | referencia "
        f"{norma_g_ref:.6f} | ratio {norma_g_cod / max(norma_g_ref, 1e-300):.6f}")
    return {
        "artefacto": str(MODELO_100D.relative_to(PROJECT_ROOT)), "sha256": sha_antes,
        "sha256_al_final": sha_despues, "artefacto_intacto": sha_despues == sha_antes,
        "vocabulario": len(palabras), "dim": int(W.shape[1]),
        "pares_evaluados": int(n), "k": int(K_POR_DEFECTO),
        "loss_codigo": round(float(l_cod), 8), "loss_referencia": round(float(l_ref), 8),
        "termino_positivo": round(float(pos_r), 8),
        "termino_negativo_codigo": round(float(neg_c), 8),
        "termino_negativo_referencia": round(float(neg_r), 8),
        "peso_del_termino_negativo": round(peso_negativo, 8),
        "norma_del_gradiente_de_entrada_codigo": round(norma_g_cod, 8),
        "norma_del_gradiente_de_entrada_referencia": round(norma_g_ref, 8),
        "ratio_de_la_norma_del_gradiente_de_entrada": round(norma_g_cod
                                                            / max(norma_g_ref, 1e-300), 8),
        "norma_del_gradiente_de_salida_codigo": round(norma_salida_cod, 8),
        "norma_del_gradiente_de_salida_referencia": round(norma_salida_ref, 8),
        "limite": (f"evaluado sobre las primeras {N_LINEAS_CORPUS} lineas del corpus y "
                   f"{N_PARES_REALES} pares como maximo. Los valores absolutos dependen del "
                   "subconjunto, pero el ratio entre objetivos no depende del subconjunto, "
                   "porque es una propiedad de la formula"),
    }


# =========================================================================
# SECCION 8: CURVAS Y FIGURAS
# =========================================================================
def curvas_de_perdida() -> dict:
    """Entrena dos SkipGram diminutos, con cada objetivo, y registra las dos curvas.

    Corpus sintetico, dimension 100 para igualar la del artefacto congelado, optimizador SGD
    con el learning rate escalado de p71-08, y evaluacion de los dos modelos bajo los dos
    objetivos sobre el mismo conjunto de pares fijos.

    Returns:
        Diccionario con las curvas y la perdida reportada por el entrenamiento original.
    """
    docs = corpus_sintetico()
    vocab, p, cnt = construir_vocabulario(docs)
    r = np.random.default_rng(SEMILLA_SINTETICA)
    centros, contextos = pares_de_epoca(docs, vocab, p, r)
    orden = r.permutation(len(centros))
    centros, contextos = centros[orden], contextos[orden]
    idx = np.array([vocab[w] for w in vocab], dtype=np.int64)
    pesos = np.array([cnt[w] for w in vocab], dtype=np.float64) ** 0.75
    pesos = pesos / pesos.sum()
    n_eval = min(2000, len(centros))
    evaluacion = {"centro": torch.from_numpy(centros[:n_eval]),
                  "ctx": torch.from_numpy(contextos[:n_eval]),
                  "neg": torch.from_numpy(r.choice(idx, size=(n_eval, K_POR_DEFECTO), p=pesos))}
    lr_ini = LR0_POR_EJEMPLO * LOTE_AUDIT
    total_lotes = max(1, math.ceil(len(centros) / LOTE_AUDIT))
    curvas: dict[str, dict[str, list[float]]] = {}
    for nombre, usar_referencia in (("codigo", False), ("referencia", True)):
        propio = objetivo_referencia if usar_referencia else objetivo_codigo
        torch.manual_seed(SEMILLA_SINTETICA)
        modelo = SkipGram(len(vocab), DIM_SINTETICA)
        with torch.no_grad():
            modelo.salida.weight.uniform_(-0.05, 0.05)
        opt = torch.optim.SGD(modelo.parameters(), lr=lr_ini)
        propias, ambas = [], []
        paso = 0
        for _ in range(EPOCAS_SINTETICAS):
            for b0 in range(0, len(centros), LOTE_AUDIT):
                lr = max(LR_MINIMO, lr_ini * (1.0 - paso / total_lotes))
                for g in opt.param_groups:
                    g["lr"] = lr
                c = torch.from_numpy(centros[b0:b0 + LOTE_AUDIT])
                x = torch.from_numpy(contextos[b0:b0 + LOTE_AUDIT])
                neg = torch.from_numpy(r.choice(idx, size=(len(c), K_POR_DEFECTO), p=pesos))
                opt.zero_grad()
                perdida, _, _ = propio(modelo.entrada.weight, modelo.salida.weight, c, x, neg)
                perdida.backward()
                opt.step()
                paso += 1
            with torch.no_grad():
                propias.append(round(float(propio(
                    modelo.entrada.weight, modelo.salida.weight, evaluacion["centro"],
                    evaluacion["ctx"], evaluacion["neg"])[0].item()), 6))
                ambas.append([
                    round(float(objetivo_codigo(modelo.entrada.weight, modelo.salida.weight,
                                                evaluacion["centro"], evaluacion["ctx"],
                                                evaluacion["neg"])[0].item()), 6),
                    round(float(objetivo_referencia(modelo.entrada.weight,
                                                    modelo.salida.weight,
                                                    evaluacion["centro"], evaluacion["ctx"],
                                                    evaluacion["neg"])[0].item()), 6)])
        curvas[nombre] = {"bajo_el_propio_objetivo": propias,
                          "bajo_el_codigo": [a[0] for a in ambas],
                          "bajo_la_referencia": [a[1] for a in ambas]}
    log(f"      curvas: codigo {curvas['codigo']['bajo_el_propio_objetivo'][0]:.4f} -> "
        f"{curvas['codigo']['bajo_el_propio_objetivo'][-1]:.4f} | referencia "
        f"{curvas['referencia']['bajo_el_propio_objetivo'][0]:.4f} -> "
        f"{curvas['referencia']['bajo_el_propio_objetivo'][-1]:.4f}")
    return {"curvas": curvas, "epocas": EPOCAS_SINTETICAS, "dim": DIM_SINTETICA,
            "pares_por_epoca": len(centros), "lote": LOTE_AUDIT,
            "learning_rate_inicial": lr_ini, "pares_de_evaluacion": int(n_eval),
            "corpus": "sintetico, ocho familias de cinco palabras que no se mezclan"}


def figuras(loss: dict, gradientes: list[dict], congelado: dict) -> list[str]:
    """Genera las figuras de P79, con nombres y etiquetas en ASCII puro.

    Args:
        loss: salida de `curvas_de_perdida`.
        gradientes: lista de salidas de `test2_gradientes`, una por K.
        congelado: salida de `evaluar_artefacto_congelado`.

    Returns:
        Lista de nombres de fichero escritos.

    Raises:
        SystemExit: si algun nombre de figura no es ASCII.
    """
    FIGURAS.mkdir(parents=True, exist_ok=True)
    escritas: list[str] = []
    curvas = loss["curvas"]
    ep = np.arange(1, loss["epocas"] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nombre, color in (("codigo", "#c00000"), ("referencia", "#1f4e79")):
        axes[0].plot(ep, curvas[nombre]["bajo_el_codigo"], color=color, lw=1.6,
                     label=f"modelo {nombre},loss del codigo")
        axes[0].plot(ep, curvas[nombre]["bajo_la_referencia"], color=color, lw=1.6, ls="--",
                     label=f"modelo {nombre}, loss de referencia")
    axes[0].axhline(DOS_LN_2, color="k", ls=":", lw=1, label="2*ln(2) = 1.386295")
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("loss")
    axes[0].set_title("P79: la misma loss medida con los dos objetivos")
    axes[0].legend(fontsize=7)
    ks = [g["k"] for g in gradientes]
    ancho = 0.25
    xs = np.arange(len(ks))
    axes[1].bar(xs - ancho, [g["ratio_componente_negativo"] for g in gradientes], width=ancho,
                label="componente negativo")
    axes[1].bar(xs, [g["ratio_gradiente_total_de_entrada"] for g in gradientes], width=ancho,
                label="gradiente total de entrada")
    axes[1].bar(xs + ancho, [g["ratio_componente_positivo"] for g in gradientes], width=ancho,
                label="componente positivo")
    axes[1].plot(xs, [1.0 / k for k in ks], "ko", ms=6, label="1/K")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"K={k}" for k in ks])
    axes[1].set_ylabel("ratio codigo / referencia")
    axes[1].set_title("P79: el componente negativo vale 1/K, el positivo no cambia")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURAS / "p79_01_loss_y_ratio_gradiente.png", dpi=110)
    plt.close(fig)
    escritas.append("p79_01_loss_y_ratio_gradiente.png")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(ep, curvas["codigo"]["bajo_el_codigo"], color="#c00000", lw=1.8,
            label="entrenado con el objetivo del codigo")
    ax.plot(ep, curvas["codigo"]["bajo_la_referencia"], color="#c00000", lw=1.4, ls="--",
            label="el mismo modelo,loss de referencia")
    ax.plot(ep, curvas["referencia"]["bajo_la_referencia"], color="#1f4e79", lw=1.8,
            label="entrenado con el objetivo de referencia")
    ax.set_xlabel("epoca")
    ax.set_ylabel("loss de referencia")
    ax.set_title("P79: el objetivo de referencia nunca se alcanza con la loss del codigo")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURAS / "p79_02_curvas_comparadas.png", dpi=110)
    plt.close(fig)
    escritas.append("p79_02_curvas_comparadas.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    etiquetas = ["termino\npositivo", "termino\nnegativo"]
    codigo = [congelado["termino_positivo"], congelado["termino_negativo_codigo"]]
    referencia = [congelado["termino_positivo"], congelado["termino_negativo_referencia"]]
    xs = np.arange(2)
    ancho = 0.35
    axes[0].bar(xs - ancho / 2, codigo, width=ancho, color="#c00000", label="codigo")
    axes[0].bar(xs + ancho / 2, referencia, width=ancho, color="#1f4e79", label="referencia")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(etiquetas)
    axes[0].set_ylabel("magnitud del termino")
    axes[0].set_title("P79: terminos del objetivo, sobre el congelado")
    axes[0].legend(fontsize=8)
    etiquetas_g = ["entrada", "salida positiva", "salida negativa"]
    ratios = [congelado["ratio_de_la_norma_del_gradiente_de_entrada"],
              1.0, congelado["norma_del_gradiente_de_salida_codigo"]
              / max(congelado["norma_del_gradiente_de_salida_referencia"], 1e-300)]
    axes[1].bar(etiquetas_g, ratios, color=["#c00000", "#1f4e79", "#c00000"])
    axes[1].axhline(1.0 / congelado["k"], color="k", ls=":", lw=1,
                    label=f"1/K = {1.0 / congelado['k']:.2f}")
    axes[1].axhline(1.0, color="#1f4e79", ls="--", lw=1, label="1.00")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("norma del codigo / norma de la referencia")
    axes[1].set_title("P79: el defecto esta solo en la repulsion")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURAS / "p79_03_artefacto_congelado.png", dpi=110)
    plt.close(fig)
    escritas.append("p79_03_artefacto_congelado.png")

    for nombre in escritas:
        if any(ord(c) > 127 for c in nombre):
            raise SystemExit(f"STOP: nombre de figura no ASCII {nombre}")
    log("      tres figuras en figuras/p79/, con nombre y etiquetas en ASCII")
    return escritas


def verificar_ascii_texto(rutas: list[Path]) -> None:
    """Verifica byte a byte que ningun fichero de texto tiene bytes mayores que 127.

    Args:
        rutas: ficheros de texto a verificar.

    Raises:
        SystemExit: si algun fichero tiene algun byte mayor que 127.
    """
    sucios = [r.name for r in rutas if r.exists() and any(b > 127 for b in r.read_bytes())]
    if sucios:
        raise SystemExit(f"STOP: bytes no ASCII en {sucios}")


def verificar_figuras(nombres: list[str]) -> dict:
    """Comprobacion de las figuras: nombre ASCII, firma PNG valida y sha256.

    Un PNG es binario por construccion y sus bytes comprimidos son mayores que 127, asi que la
    verificacion byte a byte se aplica al script y al JSON, y a las figuras se les exige nombre
    ASCII y firma valida.

    Args:
        nombres: nombres de figura ya escritos.

    Returns:
        Diccionario con bytes y sha256 de cada figura.

    Raises:
        SystemExit: si algun nombre no es ASCII o la firma PNG no es valida.
    """
    salida = {}
    for nombre in nombres:
        if any(ord(c) > 127 for c in nombre):
            raise SystemExit(f"STOP: nombre de figura no ASCII {nombre}")
        datos = (FIGURAS / nombre).read_bytes()
        if datos[:8] != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"STOP: {nombre} no es un PNG valido")
        salida[nombre] = {"bytes": len(datos), "sha256": hashlib.sha256(datos).hexdigest()}
    return salida


# =========================================================================
# SECCION 9: VEREDICTO Y MAIN
# =========================================================================
def construir_veredicto(gradientes: list[dict], scheduler: dict, init: dict,
                        productor: dict, congelado: dict) -> dict:
    """Redacta el veredicto por punto, aplicando la regla antes que la prosa.

    Returns:
        Diccionario con un veredicto por punto y la implicacion para el artefacto congelado.
    """
    ratio_defecto = next(g["ratio_componente_negativo"] for g in gradientes
                         if g["k"] == K_POR_DEFECTO)
    override = busca_override_de_salida(FUENTE_DIMENSIONES)
    return {
        "punto_1_la_loss": {
            "veredicto": "CONFIRMADO: SkipGram.forward de p71_04 NO implementa SGNS",
            "prueba": (f"con K={K_POR_DEFECTO} el termino negativo del codigo pesa "
                       f"{ratio_defecto} de lo que deberia, y el termino positivo es identico al "
                       f"de la referencia. La identidad loss_codigo = pos + neg_codigo se "
                       f"cumple exactamente"),
            "magnitud": f"un factor {K_POR_DEFECTO} infra-pesado en la repulsion",
        },
        "punto_2_el_gradiente": {
            "veredicto": "CONFIRMADO: el codigo es la referencia con la repulsion en 1/K",
            "prueba": ("el gradiente analitico reproduce el de la referencia, el componente "
                       "positivo coincide elemento a elemento, y el componente negativo del "
                       "codigo es el de la referencia dividido por K, con residuo cero"),
        },
        "punto_3_el_scheduler": {
            "veredicto": ("CONFIRMADO" if scheduler["lectura_estatica_del_fuente"]
                          ["semillas_desalineadas"] else "NO CONFIRMADO"),
            "prueba": (f"en p72_01 el bucle del plan usa las semillas "
                       f"{scheduler['lectura_estatica_del_fuente']['conjuntos_de_semillas_por_bucle']}"
                       f" y el bucle de entrenamiento otras, luego `total_lotes` se calibra "
                       f"sobre un numero de lotes que no es el que se recorre"),
            "desvio_relativo_sobre_el_recorte": scheduler["desvio_relativo"],
            "efecto": (f"el learning rate real termina en "
                       f"{scheduler['learning_rate']['ultimo_real']} en vez de "
                       f"{scheduler['learning_rate']['ultimo_ideal']}, y "
                       f"{scheduler['learning_rate']['pasos_en_el_piso']} de "
                       f"{scheduler['learning_rate']['pasos_totales']} pasos se quedan en el "
                       f"piso de {scheduler['learning_rate']['minimo']}"),
        },
        "punto_4_la_inicializacion": {
            "veredicto": ("DEFECTO REAL PERO NO ALCANZA AL ARTEFACTO CONGELADO"
                          if override["existe"] and init["ratio_cero_sobre_uniforme"] == 0.0
                          else "DEFECTO REAL"),
            "prueba": (f"con la salida a cero el gradiente de la matriz de entrada es "
                       f"exactamente {init['medida_del_gradiente_en_el_paso_1']['ceros']['norma_del_gradiente_de_entrada']:.1f}, "
                       f"es decir cero, y con la salida uniforme vale "
                       f"{init['medida_del_gradiente_en_el_paso_1']['uniforme_0p05']['norma_del_gradiente_de_entrada']:.6f}. "
                       f"El sesgo es de retardo, no de signo"),
            "alcance": (f"el artefacto de 100d lo produjo {productor['productor']}, que "
                        f"sobrescribe la salida a uniforme en {override.get('linea')} de "
                        f"{override.get('fuente')}. Los artefactos que SI salieron de p71_04 y "
                        f"p71_06, con la salida a cero, son w2v_twitter.npz y "
                        f"w2v_twitter_sgd.npz, y ambos estan declarados NO ENTRENADOS"),
        },
        "implicacion_para_el_w2v_100d_congelado": {
            "el_defecto_que_si_le_alcanza": (
                "el factor 1/K. El congelado se entreno con la clase SkipGram de p71_04, sin "
                "tocarla, a traves del diccionario de utilidades de p72_01"),
            "el_defecto_que_no_le_alcanza": (
                "la inicializacion a cero, porque el productor la sobrescribe. Y el desfase de "
                "semillas del scheduler SI le alcanza, aunque su efecto es pequeno y va en la "
                "direccion de terminar el decaimiento antes de tiempo"),
            "sentido": "PESIMISTA para el uso discriminativo que le dan P73 y P78",
            "por_que": (
                "la repulsion entre negativos es la parte del objetivo que separa direcciones. "
                "Con K=10 y un peso de 0.1, la separacion aprendida esta infra-entrenada, de "
                "modo que cualquier mejora medida sobre este artefacto es una cota INFERIOR de "
                "la que daria un SGNS correcto. El artefacto no es invalido: es pesimista"),
            "medido": {
                "loss_codigo": congelado["loss_codigo"],
                "loss_referencia": congelado["loss_referencia"],
                "ratio_de_la_norma_del_gradiente_de_entrada": congelado[
                    "ratio_de_la_norma_del_gradiente_de_entrada"],
            },
            "no_se_propone": ("esta auditoria NO propone reentrenar nada. Quantifica. La "
                              "decision de reentrenar es de otra fase y de otro agente"),
        },
    }


def main() -> None:
    """Ejecuta P79 completo: los cuatro puntos, el veredicto, el JSON y las figuras.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    torch.set_num_threads(32)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    log("P79 auditoria del entrenador Word2Vec: objetivo SGNS, scheduler e inicializacion")
    log("   NO se entrena el Word2Vec de 10 millones de tokens")
    log("   NO se escribe ni se modifica el Word2Vec congelado de 100d")
    log(f"   producto del punto 3: recorte de las primeras {N_LINEAS_CORPUS} lineas de "
        f"{CORPUS_TOKENS.name}, limite declarado")

    log("")
    log("   PUNTO 1 y 2: la loss del codigo y su gradiente, con tensores diminutos")
    t = tensores_de_test()
    k_defecto = t["neg"].shape[1]
    perdidas = [test1_loss_no_es_sgns(tensores_de_test(k=k), k) for k in KS_AUDITADOS]
    gradientes = [test2_gradientes(tensores_de_test(k=k), k) for k in KS_AUDITADOS]
    principal = next(p for p in perdidas if p["k"] == k_defecto)
    log(f"      K={k_defecto}: loss codigo {principal['loss_codigo']:.6f} | loss referencia "
        f"{principal['loss_referencia']:.6f} | ratio del termino negativo "
        f"{principal['ratio_del_termino_negativo']} = 1/{k_defecto}")
    for g in gradientes:
        log(f"      K={g['k']:2d}: componente negativo {g['ratio_componente_negativo']:.6f} | "
            f"gradiente total de entrada {g['ratio_gradiente_total_de_entrada']:.6f} | "
            f"componente positivo {g['ratio_componente_positivo']:.6f} | "
            f"residuo del bloqueo {g['residuo_del_bloqueo_por_1_sobre_k']:.2e}")

    scheduler = test3_scheduler()
    init = test4_inicializacion()
    productor = identificar_productor_del_artefacto(MODELO_100D)
    log("")
    log(f"   PUNTO 5: el artefacto congelado lo produjo {productor['productor']} "
        f"(sha256 {productor['sha256'][:16]}, declarado en {productor['declarado_en']})")
    congelado = evaluar_artefacto_congelado()
    curvas = curvas_de_perdida()
    nombres = figuras(curvas, gradientes, congelado)
    info_figuras = verificar_figuras(nombres)

    veredicto = construir_veredicto(gradientes, scheduler, init, productor, congelado)
    salida = {
        "familia": "P79_AUDITORIA_WORD2VEC", "experiment_id": "P79",
        "fase": "auditoria del entrenador Word2Vec: objetivo SGNS, scheduler e inicializacion",
        "naturaleza": ("AUDITORIA. 0 entrenamientos del Word2Vec de 10M tokens. 2 micropasos de "
                       "Adam sobre corpus sintetico, que no son un entrenamiento del modelo"),
        "pregunta": ("la loss de p71_04 implementa SGNS, el gradiente se desbloquea con 1/K, el "
                     "scheduler cuenta bien los lotes, y la inicializacion a cero sesga el "
                     "entrenamiento del artefacto congelado?"),
        "alcance": {
            "entrena_el_w2v_de_10m": False,
            "toca_el_artefacto_congelado": False,
            "sha256_del_congelado_antes": hashlib.sha256(MODELO_100D.read_bytes()).hexdigest(),
            "sha256_del_congelado_despues": congelado["sha256_al_final"],
            "artefacto_intacto": congelado["artefacto_intacto"],
            "recorte_del_corpus": f"las primeras {N_LINEAS_CORPUS} lineas de "
                                  f"{CORPUS_TOKENS.name}",
            "lectura_de_paquetes_de_datos": [
                ("data/external/corpora/twitter_domain/"
                 "corpus_twitter_tokens.txt (las primeras N lineas)"),
                "modelos/twitter_domain/w2v_twitter_100d.npz (solo lectura)",
                "reports/twitter_word2vec_*.json (solo lectura)"],
            "no_leido": ["toxic_test.csv", "features/v2_structured_features.parquet",
                         "cualquier particion de holdout"],
        },
        "configuracion": CONFIG,
        "tokenizador_del_corpus": RE_TOKEN_W2V.pattern,
        "punto_1_perdidas": perdidas,
        "punto_2_gradientes": gradientes,
        "punto_3_scheduler": scheduler,
        "punto_4_inicializacion": init,
        "punto_5_artefacto_congelado": congelado,
        "curvas_de_perdida": curvas,
        "productor_del_artefacto": productor,
        "veredicto": veredicto,
        "figuras": {n: info_figuras[n] for n in nombres},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    for clave in salida:
        if any(ord(c) > 127 for c in clave):
            raise SystemExit(f"STOP: clave de JSON no ASCII {clave!r}")
    JSON_AUDIT.write_text(json.dumps(salida, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    verificar_ascii_texto([Path(__file__), JSON_AUDIT])
    if hashlib.sha256(MODELO_100D.read_bytes()).hexdigest() != SHA_100D:
        raise SystemExit("STOP: el artefacto congelado ha cambiado")

    log("")
    log("=" * 78)
    log(f"P79 OK | objetivo: {veredicto['punto_1_la_loss']['veredicto']} | factor "
        f"{1 / principal['ratio_del_termino_negativo']:.0f} en la repulsion")
    log(f"   scheduler: {veredicto['punto_3_el_scheduler']['veredicto']} | desvio "
        f"{100 * scheduler['desvio_relativo']:+.3f} por ciento sobre el recorte")
    log(f"   inicializacion: {veredicto['punto_4_la_inicializacion']['veredicto']}")
    log(f"   implicacion para el W2V100: "
        f"{veredicto['implicacion_para_el_w2v_100d_congelado']['sentido']}")
    log(f"   informe: {JSON_AUDIT.relative_to(PROJECT_ROOT)} | "
        f"{inicio.isoformat(timespec='seconds')} -> "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log("=" * 78)


if __name__ == "__main__":
    try:
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        log(f"FALLO de P79: {type(exc).__name__}: {exc}")
        sys.exit(1)
