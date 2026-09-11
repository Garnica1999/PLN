"""12_evaluar_vinculo_cp_v2.py
Fecha: 2026-09-08
Descripcion: Recalibracion (Fase K-2) del benchmark M7 del vinculo
    sentencia -> articulos de la Constitucion Politica sobre weak labels de
    M2 (data/entidades_sentencias.parquet, arts_cp). Corrige las causas
    diagnosticadas del v1 (scripts/11_evaluar_vinculo_cp.py) manteniendo el
    protocolo temporal de ADR-5 (P@k, R@k, F1@k, MRR; global, por anio y por
    periodo). Mejoras pre-registradas:
    (1) indice por PASAJES: cada articulo se segmenta en pasajes de
        oraciones (<= CONFIG['pasaje_max_palabras'] palabras) y el score del
        articulo es el MAXIMO sobre sus pasajes (BM25 por pasajes, SOTA
        DoSSIER/COLIEE 2021; Fink 2023);
    (2) COBERTURA COMPLETA con MAX-POOLING por chunks: la hipotesis inicial
        de recortar la query a la ventana inicial (considerandos) fue
        REFUTADA con evidencia sobre la muestra v2: la primera cita de un
        articulo etiquetado aparece a mediana en el 41% del texto (solo
        29.5% de casos la tienen en el primer cuarto) y el sweep de
        calibracion mostro mejora monotona con la cobertura (ventana 3000
        chars F1 0.127 vs texto completo 0.204 en BM25). Por ello la query
        es el texto completo dividido en chunks (BM25: chunks de
        CONFIG['chunk_bm25_chars'] caracteres con solape; E5: chunks de 512
        tokens, ADR-3) y el score del articulo es el MAXIMO entre chunks
        (max-pooling: un articulo citado en cualquier parte del texto
        puntua);
    (3) tokenizacion BM25 con STOPWORDS JURIDICAS (terminos procesales que
        no discriminan entre articulos) y BIGRAMAS de las colocaciones LLR
        ganadoras del EDA de PRIMERA ENTREGA (resultados/eda_extendido/
        colocaciones.csv, top-20 LLR; se excluyen nombres propios, fechas
        y conectivos procesales, exclusion documentada en CONFIG); el
        ablation sobre la muestra v2 confirma que la tokenizacion mejorada
        supera a la simple en F1@5 (0.218 vs 0.206 con chunk 800);
    (4) E5-multilingual-base en GPU (pooling mean-tokens por chunk, score
        del articulo = MAX coseno sobre sus pasajes y chunks de query);
    (5) muestra ampliada a 3 sentencias/anio (90 casos, semilla 42) con
        cupos proporcionales al tipo dentro de cada anio;
    (6) estrategia HIBRIDA pre-registrada: 0.5*BM25 + 0.5*E5 con
        normalizacion min-max por caso.
    La estrategia cita-exacta (regex M2) se mantiene como referencia
    circular por construccion y se reporta por separado.

    ciclo v2 - Fase R (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): re-benchmark
    honesto ANTES de limpiar el indice (la limpieza es la Fase S1). Sin
    cambios de metodologia: misma muestra (90 casos, 3/ano, semilla 42),
    mismos pasajes, mismo indice BM25 y mismas estrategias. Adiciones de
    solo medicion:
    (R1) nDCG@10 estandar BEIR (ganancia binaria; IDCG con min(10, n_gold)
         relevantes al inicio) para TODAS las estrategias;
    (R2) top-10 por caso y estrategia (id + score a 4 decimales) en el
         campo top10_por_caso; la cita-exacta es binaria y recibe score
         sintetico 1.0 en orden de aparicion;
    (R3) diagnostico de contaminacion del indice sucio v1: articulos
         transitorios en top-5/top-10 e IDs malformados ("ACL", Acto
         Legislativo 1 de 2016 mal parseado) en los rankings de bm25 e
         hibrido;
    (R4) salidas NUEVAS sin tocar las previas (ADR-11):
         resultados/vinculo_cp_benchmark_v3.json,
         resultados/vinculo_cp_por_ano_v3.csv, figuras/deriva_retrieval_v3.png
         y figuras/contaminacion_transitorios_v3.png;
    (R5) campos version ("v3-ciclo2-faseR") e indice ("sucio-v1"): esta
         fase SOLO mide el dano, no lo arregla.

    ciclo v2 - Fase S1 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md):
    re-benchmark con el indice LIMPIO tras la correccion del parquet
    (02_parseo_constitucion.py: 45 IDs "Transitorio ACL01016-N" ->
    numeracion canonica + columna acto_legislativo). Sin cambios de
    metodologia ni de muestra (comparabilidad con v3 sucio). Adiciones:
    (S1a) flag --indice {permanentes,completo} (default permanente):
          el indice BM25 se construye SOLO sobre articulos permanentes
          (sin transitorios ni preambulo) porque el diagnostico de la
          Fase R midio 24.4% de transitorios en top-5 contaminando el
          ranking; el modo completo preserva el comportamiento v3 para
          regresion. E5 e hibrido evaluan sobre los mismos articulos del
          indice; cita_exacta no consulta el indice (usa el gold del
          caso) y no cambia;
    (S1b) el GOLD de cada caso (arts_cp de M2) se conserva intacto: si
          un caso citara transitorios legitimamente siguen contando como
          relevantes (verificado: el regex M2 no extrae transitorios,
          0 en 23.742 sentencias, por lo que el impacto es nulo);
    (S1c) salidas NUEVAS (ADR-11):
          resultados/vinculo_cp_benchmark_v3_limpio.json,
          resultados/vinculo_cp_por_ano_v3_limpio.csv,
          figuras/deriva_retrieval_v3_limpio.png,
          figuras/contaminacion_transitorios_v3_limpio.png y
          figuras/comparacion_v3_sucio_vs_limpio.png; los v3 sucios NO
          se tocan;
    (S1d) campos version ("v3-ciclo2-faseS1") e indice ("permanentes-v3"
          o "completo-v3").

    ciclo v2 - Fase T (2026-09-10, plan/PLAN_v2_ciclo_mejora.md):
    reranking de dos etapas con cross-encoder multilingue
    BAAI/bge-reranker-v2-m3 (568M, apache-2.0; arquitectura ganadora en
    statute retrieval segun DoSSIER/COLIEE y LegalMALR 2026). Adiciones:
    (T1) estrategia NUEVA "bm25_rerank": BM25 top-20 candidatos ->
         cross-encoder puntua cada par (query, pasaje) -> reorden por
         score -> top-10. La query es el TEXTO de la sentencia (mismo
         texto que alimenta los chunks de bm25/e5, sin scores) y el
         pasaje es el texto del pasaje del articulo (el mismo que indexa
         BM25); el score del articulo es el MAXIMO entre sus pasajes
         (max-pooling, coherente con ADR-9). Truncamiento del PAR a 512
         tokens (longest_first), fp16 en GPU con batch 16 (rebana a
         mitades ante OOM) y CPU batch 4 si no hay >= 6 GiB de VRAM
         libre (WARNING documentado, no bloquea);
    (T2) flag --rerank/--no-rerank (default ON en esta fase): con
         --no-rerank el script se comporta exactamente como v3_limpio
         (regresion; metricas identicas, salidas redirigidas a
         /tmp/opencode/*_regresion para no pisar los _limpio); con
         rerank activo la salida va a
         resultados/vinculo_cp_benchmark_v4.json con version
         "v4-ciclo2-faseT" y campo reranker (ADR-11: v2/v3/v3_limpio
         NUNCA se sobrescriben);
    (T3) flag --limite N: recorta la muestra temporal a los primeros N
         casos SOLO para smoke tests del pipeline E2E (documentado en el
         JSON como smoke_test=true; el run final va sin limite);
    (T4) la estrategia entra al mismo marco de metricas (P@5, R@5,
          F1@5, MRR, nDCG@10, top10_por_caso, diagnostico) sin cambios
          de protocolo.

    ciclo v2 - Fase T iter2 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md):
    correccion del fallo de v4. El run v4 (bm25_rerank) EMPEORO vs
    bm25 (F1@5 0.233 -> 0.177, MRR 0.765 -> 0.572) porque la query del
    par (query, pasaje) era el TEXTO COMPLETO de la sentencia (~68k
    chars) truncado a 512 tokens por longest_first: el cross-encoder
    solo veia el INICIO de la sentencia (cabecera/metadata procesal) y
    perdia el contenido sustantivo; BM25 en cambio hace max-pooling
    sobre ~113 chunks del texto completo (hallazgo ADR-9) y por eso
    gana. Adiciones:
    (T5) estrategia NUEVA "bm25_rerank_chunks": para cada candidato
         del top-20 BM25 se identifica su MEJOR pasaje (max score BM25
         del caso) y los top-3 chunks de la query contra ese pasaje
         (CONFIG['k_chunks_rerank']); el cross-encoder puntua los 3
         pares (chunk, pasaje) y el score del articulo es el MAXIMO
         entre ellos (max-pooling, coherente con ADR-9). Cada par cabe
         en 512 tokens (chunk ~800 chars + pasaje <= 120 palabras); la
         longitud real de los pares se mide SIN truncar y se reporta
         en rerank_chunks_stats del JSON;
    (T6) la estrategia bm25_rerank de v4 (query completa) se CONSERVA
         con su nombre para comparacion directa y reproducibilidad
         del fallo;
    (T7) scores_bm25_caso ahora TAMBIEN expone la matriz de scores
         (chunk, pasaje) del caso para que bm25_rerank_chunks derive
         el mejor pasaje y los top-3 chunks sin recomputar el indice;
    (T8) salidas NUEVAS v5 (ADR-11: v2/v3/v3_limpio/v4 NUNCA se
         sobrescriben): resultados/vinculo_cp_benchmark_v5.json
         (version "v5-ciclo2-faseT-iter2", campo nota con este
         diagnostico), resultados/vinculo_cp_por_ano_v5.csv,
         figuras/deriva_retrieval_v5.png y
         figuras/contaminacion_transitorios_v5.png.
         scripts/lib/componentes_producto.py NO se toca en esta
         iteracion (el producto solo adoptara el reranker si gana;
         decision del checkpoint).

    ciclo v2 - Fase T iter3 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md):
    ampliacion del recall de la 1a etapa del reranking. El techo de v5
    (bm25_rerank_chunks) es el recall@20 de la primera etapa BM25: los
    articulos citados fuera del top-20 no pueden ser rescatados por el
    cross-encoder (caso 6251: arts 21/209 en posiciones 220/305 del
    ranking BM25; F1@10 0.249 < 0.30). Adiciones:
    (T9) estrategias NUEVAS con candidatos top-50:
         bm25_rerank_chunks_top50 (top-50 BM25) e
         hibrido_rerank_chunks_top50 (top-50 del hibrido pre-registrado
         0.5*bm25_norm + 0.5*e5_norm, que aporta el recall denso que
         BM25 solo no tiene). El rerank por chunks de cada candidato NO
         cambia respecto a v5: mejor pasaje BM25 del articulo + top-3
         chunks contra ese pasaje desde la matriz chunk x pasaje +
         max-pooling de los 3 pares (50x3 = 150 pares/caso, escala
         lineal);
    (T10) las 6 estrategias previas se CONSERVAN intactas:
          ranking_bm25_rerank_chunks (v5) delega en la generalizada
          ranking_rerank_chunks con los mismos argumentos (top-20 BM25)
          -> no-regresion verificada contra el smoke v5;
    (T11) salidas NUEVAS v6 (ADR-11: v2/v3/v3_limpio/v4/v5 NUNCA se
          sobrescriben): resultados/vinculo_cp_benchmark_v6.json
          (version "v6-ciclo2-faseT-iter3", campo nota con el
          diagnostico iter3), resultados/vinculo_cp_por_ano_v6.csv,
          figuras/deriva_retrieval_v6.png y
          figuras/contaminacion_transitorios_v6.png.
    ciclo v2 - Fase W-a (2026-09-11, plan/PLAN_v2_ciclo_mejora.md):
    prior ESTATICO de hubs de la CP como senal aditiva de ranking
    (UA-StatuteRetrieval 2026, arXiv:2607.11109: los articulos hub de
    co-citacion resisten el decay temporal; en la CP los hubs son los
    articulos mas citados del corpus). El prior es la FRECUENCIA GLOBAL
    DE CITACION de cada articulo en las 23.742 sentencias de
    data/entidades_sentencias.parquet (weak labels M2), normalizada a
    [0,1] por el maximo (estilo IDF: estadistica de corpus, identica
    para toda consulta, no informacion por caso; ver
    construir_prior_hubs). NO-LEAKAGE: el prior no depende del conjunto
    de evaluacion ni de que casos se miden; es la misma senal global que
    se aplicaria en produccion. Se comparo en smoke contra el PageRank
    ponderado del grafo de co-citacion (arista = dos articulos citados
    en la misma sentencia; 66.976 pares, 2.261.390 pesos): Spearman
    0.9828 y mismo top-5 (241/86/13/2/1), por lo que se adopta la
    frecuencia normalizada por ser la senal mas simple y estable (la
    alternativa no anade informacion). Adiciones:
    (W1) data/prior_hubs_cp.json (NUEVO): mapa canon -> valor
         normalizado; lo genera --generar-prior o la primera corrida
         que lo necesite (cargar_o_construir_prior_hubs);
    (W2) aplicar_prior_hubs(scores, prior, alpha, norm=True):
         score_final = min-max(scores) + alpha * prior_norm; el min-max
         DENTRO del diccionario hace comparable alpha entre estrategias
         de escalas distintas (BM25, coseno E5, logits del
         cross-encoder); los canones sin valor en el prior reciben 0.0;
    (W3) flag --hubs-alpha FLOAT (default 0.0 = comportamiento actual
         intacto): con alpha>0 el prior se aplica a las estrategias
         rankeables (bm25, e5, hibrido, bm25_rerank*,
         hibrido_rerank*) DESPUES de calcular sus scores base y ANTES
         de top-k/metricas (aplicar_prior_a_casos); cita_exacta NO se
         prioriza (es referencia circular);
    (W4) flag --calibrar-hubs: rejilla {0.0,0.05,0.1,0.15,0.2,0.3,0.5}
         SIN re-ejecutar modelos (los scores base se calculan UNA vez;
         la rejilla solo re-rankea y re-mide). Protocolo honesto
         anti-overfit: seleccion de alpha por F1@5 en la mitad A
         (indices pares) validada en B (impares) y viceversa, mas
         full-90; la recomendacion del JSON solo dice "adoptar" si
         mejora F1@5 en A, B y full-90 a la vez;
    (W5) salidas NUEVAS v7 (ADR-11: v2/v3/v3_limpio/v4/v5/v6 NUNCA se
         sobrescriben): resultados/vinculo_cp_benchmark_v7.json
         (version "v7-ciclo2-faseW-a", con las secciones hubs_prior,
         hubs_alpha, global_alpha0 y calibracion_hubs),
         resultados/vinculo_cp_por_ano_v7.csv,
         figuras/deriva_retrieval_v7.png,
         figuras/contaminacion_transitorios_v7.png y
         figuras/hubs_alpha_v7.png (curva alpha vs F1@5 en A/B/full
         para la mejor estrategia vigente).

    ciclo v2 - Fase W-a iter2 (2026-09-11, plan/PLAN_v2_ciclo_mejora.md):
    extension de la rejilla de alpha del prior de hubs. El run v7 eligio
    alpha=0.5 en las 4 estrategias focales, es decir, el MAXIMO de la
    rejilla {0.0..0.5}: la curva seguia subiendo y el optimo quedo en el
    borde, asi que iter2 extiende la rejilla hasta 2.0
    ({0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0,1.5,2.0}) para localizar el
    maximo interior y medir donde satura la senal. NO cambia nada mas del
    protocolo: misma muestra, misma normalizacion min-max, misma seleccion
    por F1@5 en A validada en B y viceversa + full-90 y mismo criterio
    "adoptar". Adiciones:
    (W6) campo de diagnostico calibracion_hubs.curva_alpha_full: para
         cada estrategia focal (bm25, hibrido, bm25_rerank_chunks_top50,
         hibrido_rerank_chunks_top50) mapea alpha -> F1@5 full-90, para
         ver la forma de la curva y el punto de saturacion;
    (W7) salidas NUEVAS v8 (ADR-11: v2/v3/v3_limpio/v4/v5/v6/v7 NUNCA se
         sobrescriben): resultados/vinculo_cp_benchmark_v8.json
         (version "v8-ciclo2-faseW-a-iter2", campo nota con este
         diagnostico), resultados/vinculo_cp_por_ano_v8.csv,
         figuras/deriva_retrieval_v8.png,
         figuras/contaminacion_transitorios_v8.png y
         figuras/hubs_alpha_v8.png.

    Estado del arte aplicado: BM25 por pasajes con agregacion max, max-pooling
    de chunks de query (cobertura completa), hibrido lexico-denso con
    normalizacion min-max por consulta (pre-registrado 0.5/0.5), E5
    zero-shot para deriva temporal (ADR-3, ADR-6), reranking con
    cross-encoder bge-reranker-v2-m3 (Fase T), prior de hubs de
    co-citacion con rejilla alpha extendida (Fase W-a iter2,
    UA-StatuteRetrieval 2026), protocolo temporal de ADR-5 con P@k,
    R@k, F1@k y MRR.
"""

import argparse
import importlib.util
import itertools
import json
import random
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pqds
import torch
from rank_bm25 import BM25Okapi

CONFIG = {
    "semilla": 42,
    "k": 5,
    "ano_min": 1992,
    "ano_max": 2021,
    "muestra_por_ano": 3,
    # --- cobertura de query (mejora 2) ---
    "ventana_query_chars": 10**9,
    "chunk_bm25_chars": 800,
    "solape_bm25_chars": 200,
    # --- pasajes por articulo (mejora 1) ---
    "pasaje_max_palabras": 120,
    "agregacion_pasajes": "max",
    "agregacion_chunks_bm25": "max",
    "agregacion_chunks_e5": "max",
    # --- E5 (mejora 4) ---
    "tamano_chunk": 512,
    "solape_chunk": 64,
    "reserva_prefijo_tokens": 4,
    "prefijo_query": "query: ",
    "prefijo_pasaje": "passage: ",
    "modelo_e5": "intfloat/multilingual-e5-base",
    "batch_embedding": 32,
    "batch_embedding_oom": 16,
    "vram_minima_gb": 2.0,
    # --- hibrido (mejora 6, pre-registrado) ---
    "pesos_hibrido": {"bm25": 0.5, "e5": 0.5},
    # --- cita exacta (referencia circular) ---
    "ventana_cp": 60,
    # --- reranking dos etapas (ciclo v2 - Fase T) ---
    "reranker": "BAAI/bge-reranker-v2-m3",
    "k_candidatos_rerank": 20,
    "k_rerank_salida": 10,
    "rerank_default": True,
    # --- reranking por chunks (ciclo v2 - Fase T iter2) ---
    "k_chunks_rerank": 3,
    # --- reranking top-50 (ciclo v2 - Fase T iter3): ampliacion del
    # recall de la 1a etapa; el rerank por chunks de cada candidato es
    # identico al de v5 (escala lineal: 50x3 = 150 pares/caso). ---
    "k_candidatos_rerank_top50": 50,
    # --- rutas ---
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_entidades": "data/entidades_sentencias.parquet",
    "ruta_constitucion": "data/constitucion_articulos.parquet",
    "ruta_script_m2": "scripts/03_extraccion_entidades.py",
    "ruta_benchmark_v1": "resultados/vinculo_cp_benchmark.json",
    "ruta_benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
    "ruta_benchmark": "resultados/vinculo_cp_benchmark_v3.json",
    "ruta_por_ano": "resultados/vinculo_cp_por_ano_v3.csv",
    "ruta_figura": "figuras/deriva_retrieval_v3.png",
    "ruta_figura_contaminacion": "figuras/contaminacion_transitorios_v3.png",
    "version_salida": "v3-ciclo2-faseR",
    "version_indice": "sucio-v1",
    # ciclo v2 - Fase S1: modo del indice y salidas con sufijo _limpio.
    # INCLUIR_TRANSITORIOS_BENCH controla el modo por defecto del
    # benchmark (False = solo permanentes, la limpieza de la Fase S1);
    # el flag --indice lo sobreescribe en la linea de comandos.
    "incluir_transitorios_bench": False,
    "sufijo_limpio": "_limpio",
    "sufijo_completo": "_completo",
    "version_salida_limpia": "v3-ciclo2-faseS1",
    "version_indice_limpio": "permanentes-v3",
    "version_indice_completo": "completo-v3",
    "canon_preambulo": "Preambulo",
    "k_ndcg": 10,
    "k_top_guardado": 10,
    "k_diagnostico": (5, 10),
    "prefijo_transitorio": "Transitorio",
    "marca_id_malformado": "ACL",
    "periodos": [
        ("1992-2001", 1992, 2001),
        ("2002-2011", 2002, 2011),
        ("2012-2021", 2012, 2021),
    ],
    # ciclo v2 - Fase T: bm25_rerank se anade al final; con --no-rerank
    # se retira de la lista y el benchmark es identico al v3_limpio.
    # ciclo v2 - Fase T iter2: bm25_rerank_chunks (correccion del
    # truncamiento de la query) se anade al final; bm25_rerank (v4) se
    # conserva para comparacion directa.
    # ciclo v2 - Fase T iter3: bm25_rerank_chunks_top50 e
    # hibrido_rerank_chunks_top50 (candidatos top-50 de BM25 y del
    # hibrido 0.5/0.5) se anaden al final; las 6 estrategias previas se
    # conservan intactas (no-regresion v5).
    "estrategias": [
        "cita_exacta",
        "bm25",
        "e5",
        "hibrido",
        "bm25_rerank",
        "bm25_rerank_chunks",
        "bm25_rerank_chunks_top50",
        "hibrido_rerank_chunks_top50",
    ],
    # Salidas v4 de la Fase T (ADR-11: las v2/v3/v3_limpio no se tocan).
    "version_salida_v4": "v4-ciclo2-faseT",
    "ruta_benchmark_v4": "resultados/vinculo_cp_benchmark_v4.json",
    "ruta_por_ano_v4": "resultados/vinculo_cp_por_ano_v4.csv",
    "ruta_figura_v4": "figuras/deriva_retrieval_v4.png",
    "ruta_figura_contaminacion_v4": "figuras/contaminacion_transitorios_v4.png",
    # Salidas v5 de la Fase T iter2 (ADR-11: las v2/v3/v3_limpio/v4 no
    # se tocan; el run final de 90 casos lo hace ml-trainer despues).
    "version_salida_v5": "v5-ciclo2-faseT-iter2",
    "ruta_benchmark_v5": "resultados/vinculo_cp_benchmark_v5.json",
    "ruta_por_ano_v5": "resultados/vinculo_cp_por_ano_v5.csv",
    "ruta_figura_v5": "figuras/deriva_retrieval_v5.png",
    "ruta_figura_contaminacion_v5": "figuras/contaminacion_transitorios_v5.png",
    "nota_v5": (
        "Iteracion 2 de la Fase T (ciclo v2): la estrategia bm25_rerank "
        "de v4 (query = texto completo de la sentencia, ~68k chars) "
        "EMPEORO vs bm25 (F1@5 0.233 -> 0.177, MRR 0.765 -> 0.572) "
        "porque el par (query, pasaje) se truncaba a 512 tokens y el "
        "cross-encoder solo veia el inicio de la sentencia (cabecera/"
        "metadata procesal), perdiendo el contenido sustantivo. La "
        "correccion v5 (bm25_rerank_chunks) alinea el reranking con el "
        "hallazgo ADR-9: para cada candidato del top-20 BM25 se puntuan "
        "los top-3 chunks de la query contra su mejor pasaje y el score "
        "del articulo es el maximo entre los 3 pares (max-pooling). "
         "bm25_rerank (v4) se conserva en el benchmark para comparacion "
         "directa y reproducibilidad del fallo."
     ),
    # Salidas v6 de la Fase T iter3 (ADR-11: las v2/v3/v3_limpio/v4/v5
    # no se tocan; el run final de 90 casos lo hace ml-trainer despues).
    "version_salida_v6": "v6-ciclo2-faseT-iter3",
    "ruta_benchmark_v6": "resultados/vinculo_cp_benchmark_v6.json",
    "ruta_por_ano_v6": "resultados/vinculo_cp_por_ano_v6.csv",
    "ruta_figura_v6": "figuras/deriva_retrieval_v6.png",
    "ruta_figura_contaminacion_v6": "figuras/contaminacion_transitorios_v6.png",
    "nota_v6": (
        "Iteracion 3 de la Fase T (ciclo v2): el techo de v5 "
        "(bm25_rerank_chunks, F1@10 0.249 < 0.30) es el recall@20 de la "
        "primera etapa BM25: los articulos citados fuera del top-20 no "
        "pueden ser rescatados por el cross-encoder (caso 6251: arts "
        "21/209 en posiciones 220/305 del ranking BM25). Esta iteracion "
        "amplia la primera etapa a top-50 con dos variantes: "
        "bm25_rerank_chunks_top50 (candidatos = top-50 BM25) e "
        "hibrido_rerank_chunks_top50 (candidatos = top-50 del hibrido "
        "pre-registrado 0.5*bm25_norm + 0.5*e5_norm, que aporta el "
        "recall denso que BM25 solo no tiene). El rerank por chunks de "
        "cada candidato es identico al de v5 (mejor pasaje BM25 + "
        "top-3 chunks + max-pooling; 50x3 = 150 pares/caso). Las 6 "
        "estrategias previas se conservan intactas para no-regresion."
    ),
    # --- ciclo v2 - Fase W-a: prior estatico de hubs de la CP ---
    # Prior de frecuencia global de citacion (estilo IDF) calculado
    # sobre las 23.742 sentencias de data/entidades_sentencias.parquet
    # y normalizado a [0,1] por el maximo; ver construir_prior_hubs
    # para la eleccion frecuencia vs PageRank de co-citacion y la nota
    # de no-leakage.
    "ruta_prior_hubs": "data/prior_hubs_cp.json",
    "hubs_alpha_default": 0.0,
    # ciclo v2 - Fase W-a iter2: rejilla extendida. La rejilla v7
    # {0.0..0.5} dejo el optimo en el borde (alpha=0.5 en las 4
    # estrategias focales) y la curva seguia subiendo; iter2 sondea hasta
    # 2.0 para localizar el maximo interior y medir la saturacion. Mismo
    # protocolo de seleccion (F1@5 A/B cruzada + full-90).
    "hubs_alphas_calibracion": [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ],
    # Rejilla original del run v7 (Fase W-a iter1); se conserva como
    # referencia historica y se reporta en el JSON v8.
    "hubs_alphas_calibracion_v7": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
    # Estrategias focales de la Fase W-a (las 4 con evidencia en v7): el
    # diagnostico curva_alpha_full y el resumen final las usan.
    "estrategias_focales_hubs": [
        "bm25",
        "hibrido",
        "bm25_rerank_chunks_top50",
        "hibrido_rerank_chunks_top50",
    ],
    "k_f1_10": 10,
    "version_salida_v7": "v7-ciclo2-faseW-a",
    "ruta_benchmark_v7": "resultados/vinculo_cp_benchmark_v7.json",
    "ruta_por_ano_v7": "resultados/vinculo_cp_por_ano_v7.csv",
    "ruta_figura_v7": "figuras/deriva_retrieval_v7.png",
    "ruta_figura_contaminacion_v7": "figuras/contaminacion_transitorios_v7.png",
    "ruta_figura_hubs_v7": "figuras/hubs_alpha_v7.png",
    "nota_v7": (
        "Fase W-a del ciclo v2: prior estatico de hubs de la CP "
        "(frecuencia global de citacion normalizada, estilo IDF; "
        "UA-StatuteRetrieval 2026, arXiv:2607.11109) como senal aditiva "
        "score_final = min-max(score_base) + alpha * prior_norm. El "
        "PageRank ponderado del grafo de co-citacion se comparo en smoke "
        "(Spearman 0.9828 con la frecuencia, mismo top-5 241/86/13/2/1) "
        "y se descarto por no aportar sobre la senal mas simple. "
        "--hubs-alpha fija el peso (0.0 = sin prior, comportamiento "
        "intacto); --calibrar-hubs recorre la rejilla "
        "{0.0,0.05,0.1,0.15,0.2,0.3,0.5} re-rankeando los scores base ya "
        "calculados (sin re-ejecutar modelos) con validacion cruzada A/B "
        "(indices pares/impares) y full-90. La estrategia cita_exacta no "
        "se prioriza (es referencia circular)."
    ),
    # Salidas v8 de la Fase W-a iter2 (ADR-11: v2/v3/v3_limpio/v4/v5/v6/v7
    # NUNCA se tocan; el run final de 90 casos lo hace ml-trainer despues).
    "version_salida_v8": "v8-ciclo2-faseW-a-iter2",
    "ruta_benchmark_v8": "resultados/vinculo_cp_benchmark_v8.json",
    "ruta_por_ano_v8": "resultados/vinculo_cp_por_ano_v8.csv",
    "ruta_figura_v8": "figuras/deriva_retrieval_v8.png",
    "ruta_figura_contaminacion_v8": "figuras/contaminacion_transitorios_v8.png",
    "ruta_figura_hubs_v8": "figuras/hubs_alpha_v8.png",
    "nota_v8": (
        "Iteracion 2 de la Fase W-a (ciclo v2): la rejilla v7 "
        "{0.0,0.05,0.1,0.15,0.2,0.3,0.5} eligio alpha=0.5 (el maximo de "
        "la rejilla) en las 4 estrategias focales, es decir, el optimo "
        "quedo en el borde y la curva seguia subiendo; iter2 extiende la "
        "rejilla hasta 2.0 "
        "({0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0,1.5,2.0}) para "
        "localizar el maximo interior y medir donde satura la senal. Sin "
        "cambios de metodologia: misma muestra, misma normalizacion "
        "min-max, mismo protocolo de seleccion por F1@5 en A validada en "
        "B y viceversa + full-90 y mismo criterio de recomendacion "
        "('adoptar' solo si mejora F1@5 en A, B y full-90). Se anade el "
        "diagnostico calibracion_hubs.curva_alpha_full (alpha -> F1@5 "
        "full) por estrategia focal. Salidas NUEVAS v8; los artefactos "
        "v7 y anteriores quedan intactos (ADR-11)."
    ),
    "modo_hubs_activo": False,
}

# Stopwords juridicas (mejora 3): terminos procesales de encabezado y de
# citas que aparecen en casi todas las sentencias y no discriminan entre
# articulos de la CP. Lista corta y conservadora, documentada; los bigramas
# compuestos (con "_") nunca se filtran.
CONFIG["stopwords_juridicas"] = frozenset(
    {
        "articulo", "articulos", "ley", "leyes", "numeral", "numerales",
        "inciso", "incisos", "parrafo", "parrafos", "literal", "literales",
        "caso", "casos", "sentencia", "sentencias", "expediente", "bogota",
        "honorable", "magistrado", "magistrados", "ponente", "secretaria",
        "general", "demanda", "demandas", "demandado", "demandados",
        "demandada", "demandadas", "demandante", "demandantes", "dos",
        "mil", "senor", "senora", "senores",
    }
)

# Bigramas LLR ganadores del EDA de PRIMERA ENTREGA (mejora 3): top-20 de
# resultados/eda_extendido/colocaciones.csv (medida llr). Exclusiones
# documentadas: "dos mil" (fecha), "toda vez" y "caso concreto"
# (conectivos procesales), "cepeda espinosa", "jose cepeda" y
# "magistrado ponente" (nombres propios). Quedan 14 colocaciones
# sustantivas del dominio constitucional.
CONFIG["bigramas_llr"] = frozenset(
    {
        ("derechos", "fundamentales"),
        ("corte", "constitucional"),
        ("seguridad", "social"),
        ("constitucion", "politica"),
        ("debido", "proceso"),
        ("derecho", "fundamental"),
        ("minimo", "vital"),
        ("primera", "instancia"),
        ("perjuicio", "irremediable"),
        ("cosa", "juzgada"),
        ("derechos", "humanos"),
        ("corte", "suprema"),
        ("tribunal", "superior"),
        ("segunda", "instancia"),
    }
)

ROOT = Path(__file__).resolve().parent.parent

MEDIDAS = ("p_at_k", "r_at_k", "f1_at_k", "mrr", "ndcg_at_10")

# Estrategias evaluadas en el diagnostico de contaminacion (Fase R): las
# basadas en el indice BM25 sucio v1 (bm25 puro y su hibrido con E5). La
# cita-exacta no consulta el indice y E5 no depende de los IDs.
ESTRATEGIAS_DIAGNOSTICO = ("bm25", "hibrido")

# ciclo v2 - Fase W-a: estrategias que admiten el prior de hubs. La
# cita-exacta queda fuera (mismo regex M2 que genera el gold: priorizarla
# seria circular). El prior se aplica tras calcular los scores base y
# antes de top-k/metricas (ver aplicar_prior_a_casos).
ESTRATEGIAS_RANKEABLES = (
    "bm25",
    "e5",
    "hibrido",
    "bm25_rerank",
    "bm25_rerank_chunks",
    "bm25_rerank_chunks_top50",
    "hibrido_rerank_chunks_top50",
)

# Subconjunto de rerank: su ranking de producto esta truncado a
# CONFIG['k_rerank_salida'] (10); al re-rankear con prior el resultado se
# trunca igual para conservar la semantica de las metricas (el MRR solo
# puede mirar ese top-10).
ESTRATEGIAS_RERANK = (
    "bm25_rerank",
    "bm25_rerank_chunks",
    "bm25_rerank_chunks_top50",
    "hibrido_rerank_chunks_top50",
)

# Clave del caso donde vive el vector de scores base de cada estrategia:
# fuente unica para top10_por_caso y para la calibracion de hubs.
FUENTE_SCORES = {
    "bm25": "scores_bm25",
    "e5": "scores_e5",
    "hibrido": "scores_hibrido",
    "bm25_rerank": "scores_bm25_rerank",
    "bm25_rerank_chunks": "scores_bm25_rerank_chunks",
    "bm25_rerank_chunks_top50": "scores_bm25_rerank_chunks_top50",
    "hibrido_rerank_chunks_top50": "scores_hibrido_rerank_chunks_top50",
}

# Medidas reportadas en la calibracion de hubs (incluye F1@10, pedido
# explicitamente en la Fase W-a).
MEDIDAS_CALIBRACION = (
    "p_at_5",
    "r_at_5",
    "f1_at_5",
    "mrr",
    "ndcg_at_10",
    "p_at_10",
    "f1_at_10",
)

# Claves de CONFIG con rutas relativas a la raiz del proyecto; se resuelven
# contra ROOT al inicio de main() para que el script corra desde cualquier
# directorio de trabajo.
RUTAS_RELATIVAS = (
    "ruta_corpus",
    "ruta_entidades",
    "ruta_constitucion",
    "ruta_script_m2",
    "ruta_benchmark_v1",
    "ruta_benchmark_v2",
    "ruta_benchmark",
    "ruta_por_ano",
    "ruta_figura",
    "ruta_figura_contaminacion",
    # ciclo v2 - Fase T: rutas v4 (se activan con --rerank).
    "ruta_benchmark_v4",
    "ruta_por_ano_v4",
    "ruta_figura_v4",
    "ruta_figura_contaminacion_v4",
    # ciclo v2 - Fase T iter2: rutas v5.
    "ruta_benchmark_v5",
    "ruta_por_ano_v5",
    "ruta_figura_v5",
    "ruta_figura_contaminacion_v5",
    # ciclo v2 - Fase T iter3: rutas v6.
    "ruta_benchmark_v6",
    "ruta_por_ano_v6",
    "ruta_figura_v6",
    "ruta_figura_contaminacion_v6",
    # ciclo v2 - Fase W-a: prior de hubs y rutas v7.
    "ruta_prior_hubs",
    "ruta_benchmark_v7",
    "ruta_por_ano_v7",
    "ruta_figura_v7",
    "ruta_figura_contaminacion_v7",
    "ruta_figura_hubs_v7",
    # ciclo v2 - Fase W-a iter2: rejilla extendida y rutas v8.
    "ruta_benchmark_v8",
    "ruta_por_ano_v8",
    "ruta_figura_v8",
    "ruta_figura_contaminacion_v8",
    "ruta_figura_hubs_v8",
)


def resolver_rutas() -> None:
    """Convierte las rutas relativas de CONFIG en absolutas contra ROOT.

    Permite ejecutar el script desde cualquier directorio de trabajo sin
    depender del CWD (el v1 asumia workdir = raiz del proyecto).
    """
    for clave in RUTAS_RELATIVAS:
        ruta = Path(CONFIG[clave])
        if not ruta.is_absolute():
            CONFIG[clave] = str(ROOT / ruta)


# ===== SECCION 1: utilidades base =====
def configurar_semillas(semilla: int) -> None:
    """Fija semillas de aleatoriedad y hilos de torch.

    Args:
        semilla: Valor entero de semilla.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.set_num_threads(32)


def elegir_dispositivo() -> str:
    """Elige GPU solo si hay VRAM libre suficiente para E5.

    La maquina comparte la GPU con procesos externos (p. ej. llama-server).
    Si la memoria libre es menor que CONFIG['vram_minima_gb'] se degrada a
    CPU y se documenta en el log; no se cambia de modelo ni de metodologia.

    Returns:
        "cuda" si hay >= 2 GiB libres y torch la ve disponible; si no, "cpu".
    """
    if not torch.cuda.is_available():
        return "cpu"
    libre, _ = torch.cuda.mem_get_info()
    minimo = int(CONFIG["vram_minima_gb"] * 1024**3)
    if libre < minimo:
        print(
            "[aviso] VRAM libre insuficiente para E5 "
            f"({libre / 1024**3:.1f} GiB < {CONFIG['vram_minima_gb']} GiB); "
            "se ejecuta E5 en CPU (GPU ocupada por procesos externos)"
        )
        return "cpu"
    return "cuda"


def normalizar_sin_tildes(texto: str) -> str:
    """Convierte un texto a minusculas y sin diacriticos.

    Args:
        texto: Texto original.

    Returns:
        Texto normalizado.
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )
    return sin_tildes.lower()


def tokenizar(texto: str) -> list[str]:
    """Tokeniza para BM25: unigramas sin stopwords juridicas + bigramas LLR.

    Los bigramas se detectan sobre la secuencia completa de unigramas (sin
    filtrar) y se anaden como tokens compuestos "a_b"; despues se filtran
    los unigramas procesales. Asi "corte constitucional" sobrevive como
    bigrama aunque "corte" no este en la lista de stopwords.

    Args:
        texto: Texto a tokenizar.

    Returns:
        Lista de tokens (unigramas utiles + bigramas compuestos).
    """
    tokens = re.findall(r"[a-z0-9]+", normalizar_sin_tildes(texto))
    bigramas = [
        f"{a}_{b}"
        for a, b in itertools.pairwise(tokens)
        if (a, b) in CONFIG["bigramas_llr"]
    ]
    unigramas = [t for t in tokens if t not in CONFIG["stopwords_juridicas"]]
    return unigramas + bigramas


def leer_filas_por_id(ruta: str, columnas: list[str], ids: list[str]) -> pd.DataFrame:
    """Lee solo las filas con los ids dados de un parquet (posible particion).

    Usa pyarrow.dataset con filtro pushdown para no materializar en RAM el
    corpus completo mientras corre el job de lematizacion (05_lematizacion).

    Args:
        ruta: Ruta del parquet (archivo o directorio particionado).
        columnas: Columnas a leer.
        ids: Lista de ids (strings) a conservar.

    Returns:
        DataFrame con las filas cuyos ids estan en la lista.
    """
    dataset = pqds.dataset(ruta, format="parquet")
    tabla = dataset.to_table(
        columns=columnas, filter=pqds.field("id").isin(ids)
    )
    return tabla.to_pandas()


def importar_extraccion_m2() -> object:
    """Importa el modulo 03_extraccion_entidades.py (M2) por importlib.

    El nombre del archivo empieza por digito y no admite import estandar; se
    carga con spec para reutilizar extraer_arts_cp sin duplicar logica.

    Returns:
        Modulo M2 con la funcion extraer_arts_cp.
    """
    ruta = ROOT / CONFIG["ruta_script_m2"]
    spec = importlib.util.spec_from_file_location("extraccion_m2", ruta)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el modulo M2: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# ===== SECCION 2: muestra temporal 3/anio estratificada (mejora 5) =====
def cuotas_proporcionales(conteo: dict[str, int], cupos: int) -> dict[str, int]:
    """Reparte cupos entre tipos proporcionalmente (mayor resto, determinista).

    Args:
        conteo: Dict {tipo: numero de sentencias labeled del anio}.
        cupos: Total de cupos a repartir (CONFIG['muestra_por_ano']).

    Returns:
        Dict {tipo: cupos enteros} con suma <= cupos.
    """
    total = sum(conteo.values())
    if total <= 0 or cupos <= 0:
        return {tipo: 0 for tipo in conteo}
    crudo = {tipo: cupos * n / total for tipo, n in conteo.items()}
    base = {tipo: int(crudo[tipo]) for tipo in conteo}
    faltante = cupos - sum(base.values())
    orden = sorted(conteo, key=lambda t: (-(crudo[t] - base[t]), t))
    for tipo in orden[:faltante]:
        base[tipo] += 1
    return base


def muestrear_anio(grupo: pd.DataFrame, cupos: dict[str, int], rng: random.Random) -> pd.DataFrame:
    """Elige sentencias de un anio segun los cupos por tipo, con relleno.

    Si un tipo no alcanza su cuota, el deficit se rellena muestreando entre
    las filas restantes del anio (cualquier tipo) para no perder casos.

    Args:
        grupo: Filas labeled de un anio (columnas id, tipo, ano).
        cupos: Dict {tipo: cuota} calculado por cuotas_proporcionales.
        rng: Generador aleatorio con semilla fija.

    Returns:
        DataFrame con las filas elegidas del anio.
    """
    indices_elegidos: list[int] = []
    for tipo in sorted(cupos):
        subgrupo = grupo[grupo["tipo"] == tipo].sort_values("id")
        cuota = min(cupos[tipo], len(subgrupo))
        if cuota <= 0:
            continue
        posiciones = rng.sample(range(len(subgrupo)), cuota)
        indices_elegidos.extend(subgrupo.iloc[posiciones].index.tolist())
    faltan = sum(cupos.values()) - len(indices_elegidos)
    if faltan > 0:
        resto = grupo[~grupo.index.isin(indices_elegidos)].sort_values("id")
        if not resto.empty:
            cuota = min(faltan, len(resto))
            posiciones = rng.sample(range(len(resto)), cuota)
            indices_elegidos.extend(resto.iloc[posiciones].index.tolist())
    return grupo.loc[sorted(indices_elegidos)]


def cargar_muestra_temporal(marco: pd.DataFrame) -> pd.DataFrame:
    """Selecciona 3 sentencias por anio (1992-2021) con semilla fija.

    Dentro de cada anio los cupos se reparten proporcionalmente al tipo
    (T/C/SU) con el metodo del mayor resto y el muestreo es determinista
    (orden por id, rng con semilla 42, tipos en orden alfabetico).

    Args:
        marco: DataFrame con columnas id, tipo, ano y labeled (bool).

    Returns:
        DataFrame con <= 90 filas (3/anio) ordenadas por (ano, id).
    """
    rng = random.Random(CONFIG["semilla"])
    filas = []
    etiquetadas = marco[marco["labeled"]]
    for anio in range(CONFIG["ano_min"], CONFIG["ano_max"] + 1):
        grupo = etiquetadas[etiquetadas["ano"] == anio]
        if grupo.empty:
            continue
        conteo = grupo["tipo"].value_counts().to_dict()
        cupos = cuotas_proporcionales(conteo, CONFIG["muestra_por_ano"])
        filas.append(muestrear_anio(grupo, cupos, rng))
    muestra = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    return muestra.sort_values(["ano", "id"]).reset_index(drop=True)


# ===== SECCION 3: pasajes de la Constitucion (mejora 1) =====
def dividir_pasajes(texto: str, max_palabras: int) -> list[str]:
    """Segmenta un articulo en pasajes de oraciones completas.

    El parquet de la Constitucion no conserva saltos de parrafo, por lo que
    la segmentacion es por oraciones (split tras punto) agrupadas hasta
    max_palabras; una oracion mas larga que el limite queda intacta como
    pasaje propio (E5 la trocea internamente por tokens si hace falta).

    Args:
        texto: Texto del articulo.
        max_palabras: Limite de palabras por pasaje.

    Returns:
        Lista de pasajes (>= 1 para todo texto no vacio).
    """
    oraciones = [o.strip() for o in re.split(r"(?<=\.)\s+", texto) if o.strip()]
    pasajes: list[str] = []
    actual: list[str] = []
    palabras = 0
    for oracion in oraciones:
        largo = len(oracion.split())
        if palabras + largo > max_palabras and actual:
            pasajes.append(" ".join(actual))
            actual, palabras = [], 0
        actual.append(oracion)
        palabras += largo
    if actual:
        pasajes.append(" ".join(actual))
    return pasajes if pasajes else ([texto.strip()] if texto.strip() else [])


def preparar_pasajes_constitucion(
    incluir_transitorios: bool,
) -> tuple[pd.DataFrame, list[str], list[np.ndarray], set[str]]:
    """Carga los articulos y los segmenta en pasajes para el indice.

    ciclo v2 - Fase S1: con incluir_transitorios=False el indice se
    construye SOLO sobre articulos permanentes (se excluyen los 112
    transitorios y el preambulo) porque contaminaban el ranking (24.4%
    del top-5 en el diagnostico de la Fase R). El conjunto canon_valido
    (lo que valida el gold) se devuelve sobre el parquet COMPLETO para
    que un caso que citara transitorios legitimamente no los pierda.

    Args:
        incluir_transitorios: Si False, el indice cubre solo articulos
            permanentes; si True, indexa los 497 articulos completos
            (modo de regresion, comportamiento v3).

    Returns:
        Tupla (pasajes, canones, grupos_indices, canon_valido):
        - pasajes: DataFrame con columnas canon, texto y tokens (un
          pasaje por fila, en orden de articulo);
        - canones: Lista de canones de los articulos INDEXADOS (385
          permanentes o 497 completos), alineada a grupos_indices;
        - grupos_indices: Lista (por articulo indexado, mismo orden que
          canones) de arrays con las posiciones de sus pasajes;
        - canon_valido: Conjunto con los 497 canones del parquet
          completo (para validar el gold sin perder transitorios).
    """
    articulos = pd.read_parquet(CONFIG["ruta_constitucion"])
    canones_todos = articulos["numero"].astype(str).tolist()
    canon_valido = set(canones_todos)
    if incluir_transitorios:
        canones = canones_todos
        textos = articulos["texto"].fillna("").astype(str).tolist()
    else:
        mascara = ~articulos["numero"].astype(str).str.startswith(
            CONFIG["prefijo_transitorio"], na=False
        ) & (articulos["numero"].astype(str) != CONFIG["canon_preambulo"])
        indexables = articulos[mascara]
        canones = indexables["numero"].astype(str).tolist()
        textos = indexables["texto"].fillna("").astype(str).tolist()
    filas = []
    grupos_indices = []
    posicion = 0
    for canon, texto in zip(canones, textos, strict=True):
        trozos = dividir_pasajes(texto, CONFIG["pasaje_max_palabras"])
        indices = []
        for trozo in trozos:
            filas.append({"canon": canon, "texto": trozo})
            indices.append(posicion)
            posicion += 1
        grupos_indices.append(np.asarray(indices, dtype=int))
    pasajes = pd.DataFrame(filas)
    pasajes["tokens"] = pasajes["texto"].apply(tokenizar)
    return pasajes, canones, grupos_indices, canon_valido


# ===== SECCION 4: metricas de retrieval (protocolo ADR-5 + Fase R) =====
def ndcg_at_k(relevantes: list[str], ranking: list[str], k: int) -> float:
    """Calcula nDCG@k con ganancia binaria (estandar BEIR, Fase R).

    DCG@k = sum(rel_i / log2(i + 1)) con i desde 1; rel_i = 1 si el
    articulo en la posicion i esta en el gold del caso. IDCG@k coloca
    min(k, n_gold) relevantes al inicio (ganancia ideal). Un caso sin
    gold no puede ocurrir aqui (la muestra exige arts_cp no vacios),
    pero se devuelve 0.0 defensivamente.

    Args:
        relevantes: Canones de articulos relevantes (weak labels).
        ranking: Canones ordenados por el retrieval (mejor primero).
        k: Profundidad de corte (CONFIG['k_ndcg'] = 10).

    Returns:
        nDCG@k en [0.0, 1.0].
    """
    rel = set(relevantes)
    if not rel:
        return 0.0
    dcg = sum(
        1.0 / np.log2(pos + 1)
        for pos, canon in enumerate(ranking[:k], start=1)
        if canon in rel
    )
    n_ideal = min(k, len(rel))
    idcg = sum(1.0 / np.log2(pos + 1) for pos in range(1, n_ideal + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def metricas_ranking(relevantes: list[str], ranking: list[str], k: int) -> dict:
    """Calcula P@k, R@k, F1@k, MRR y nDCG@10 de un ranking frente al gold.

    Args:
        relevantes: Canones de articulos relevantes (weak labels).
        ranking: Canones ordenados por el retrieval (mejor primero).
        k: Profundidad para P@k, R@k y F1@k.

    Returns:
        Dict con p_at_k, r_at_k, f1_at_k, mrr y ndcg_at_10.
    """
    rel = set(relevantes)
    top = ranking[:k]
    recuperados = sum(1 for x in top if x in rel)
    p = recuperados / k
    r = recuperados / len(rel) if rel else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    mrr = 0.0
    for pos, canon in enumerate(ranking, start=1):
        if canon in rel:
            mrr = 1.0 / pos
            break
    return {
        "p_at_k": p,
        "r_at_k": r,
        "f1_at_k": f1,
        "mrr": mrr,
        "ndcg_at_10": ndcg_at_k(relevantes, ranking, CONFIG["k_ndcg"]),
    }


def promediar_metricas(metricas: list[dict]) -> dict:
    """Promedia las cinco medidas sobre varios casos de retrieval.

    Args:
        metricas: Lista de dicts de metricas_ranking.

    Returns:
        Dict con las medias de p_at_k, r_at_k, f1_at_k, mrr y ndcg_at_10.
    """
    return {
        medida: float(np.mean([m[medida] for m in metricas])) for medida in MEDIDAS
    }


def asignar_periodo(anio: int) -> str:
    """Asigna un anio a su periodo cronologico de CONFIG.

    Args:
        anio: Anio de la sentencia.

    Returns:
        Nombre del periodo (p. ej. "1992-2001") que contiene el anio.
    """
    for nombre, inicio, fin in CONFIG["periodos"]:
        if inicio <= anio <= fin:
            return nombre
    return "desconocido"


def metricas_por_caso(casos: list[dict]) -> pd.DataFrame:
    """Evalua las cuatro estrategias sobre cada caso de la muestra.

    Args:
        casos: Lista de dicts con id, ano, arts y rankings por estrategia.

    Returns:
        DataFrame largo con una fila por (id, estrategia) y sus medidas.
    """
    filas = []
    for caso in casos:
        relevantes = caso["arts"]
        for estrategia in CONFIG["estrategias"]:
            medidas = metricas_ranking(
                relevantes, caso["rankings"][estrategia], CONFIG["k"]
            )
            filas.append(
                {
                    "id": caso["id"],
                    "ano": caso["ano"],
                    "estrategia": estrategia,
                    "n_relevantes": len(relevantes),
                    **medidas,
                }
            )
    return pd.DataFrame(filas)


def agregar_por_periodo(por_caso: pd.DataFrame) -> dict:
    """Agrupa metricas por periodo cronologico (ADR-5).

    Args:
        por_caso: DataFrame con columnas ano, estrategia y las medidas.

    Returns:
        Dict anidado {periodo: {estrategia: {medidas medias}}}.
    """
    resultado = {}
    por_caso = por_caso.copy()
    por_caso["periodo"] = por_caso["ano"].apply(asignar_periodo)
    for nombre_periodo, _, _ in CONFIG["periodos"]:
        grupo = por_caso[por_caso["periodo"] == nombre_periodo]
        resultado[nombre_periodo] = {}
        for estrategia in CONFIG["estrategias"]:
            filas = grupo[grupo["estrategia"] == estrategia]
            resultado[nombre_periodo][estrategia] = {
                medida: float(filas[medida].mean()) for medida in MEDIDAS
            }
    return resultado


# ===== SECCION 5: estrategia BM25 por pasajes con query en chunks =====
def construir_indice_bm25(pasajes: pd.DataFrame) -> BM25Okapi:
    """Construye el indice BM25 sobre los PASAJES de los articulos.

    Args:
        pasajes: DataFrame con la columna tokens (un pasaje por fila).

    Returns:
        Indice BM25Okapi listo para consultas.
    """
    return BM25Okapi(pasajes["tokens"].tolist())


def chunks_query(texto: str) -> list[str]:
    """Divide la query en chunks de caracteres con solape (cobertura total).

    La ventana es el texto completo (CONFIG['ventana_query_chars'] = inf
    practico): la evidencia de calibracion refuto la ventana inicial corta
    (las citas aparecen a lo largo de todo el texto).

    Args:
        texto: Texto limpio de la sentencia.

    Returns:
        Lista de chunks (>= 1 si el texto no esta vacio).
    """
    ventana = texto[: CONFIG["ventana_query_chars"]]
    chunks: list[str] = []
    inicio = 0
    while inicio < len(ventana):
        fin = min(inicio + CONFIG["chunk_bm25_chars"], len(ventana))
        chunks.append(ventana[inicio:fin])
        if fin >= len(ventana):
            break
        inicio = fin - CONFIG["solape_bm25_chars"]
    return chunks


def scores_bm25_caso(
    indice: BM25Okapi, grupos_indices: list[np.ndarray], canones: list[str], chunks: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Scores BM25 por articulo y matriz (chunk, pasaje) del caso.

    Por cada chunk de la query se puntuan todos los pasajes; el score del
    pasaje es el maximo entre chunks (CONFIG['agregacion_chunks_bm25']) y el
    del articulo el maximo entre sus pasajes (CONFIG['agregacion_pasajes']).

    ciclo v2 - Fase T iter2: ademas del vector de scores por articulo se
    devuelve la matriz completa (n_chunks, n_pasajes) de scores BM25 para
    que la estrategia bm25_rerank_chunks derive, para cada articulo
    candidato, su mejor pasaje y los top-k chunks de la query contra ese
    pasaje SIN recomputar el indice (misma evidencia que produce el
    ranking BM25, hallazgo ADR-9).

    Args:
        indice: Indice BM25Okapi sobre pasajes.
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        canones: Canones alineados con grupos_indices.
        chunks: Chunks de la ventana de query.

    Returns:
        Tupla (scores, matriz):
        - scores: Array (n_articulos,) alineado con canones;
        - matriz: Array (n_chunks, n_pasajes) con el score BM25 de cada
          chunk contra cada pasaje (fila por chunk de query).
    """
    n_pasajes = indice.corpus_size
    matriz = np.zeros((len(chunks), n_pasajes), dtype=np.float64)
    max_pasajes = np.zeros(n_pasajes, dtype=np.float64)
    for fila, chunk in enumerate(chunks):
        tokens = tokenizar(chunk)
        if not tokens:
            continue
        scores_chunk = indice.get_scores(tokens)
        matriz[fila] = scores_chunk
        np.maximum(max_pasajes, scores_chunk, out=max_pasajes)
    scores = np.zeros(len(canones), dtype=np.float64)
    for i, indices in enumerate(grupos_indices):
        scores[i] = max_pasajes[indices].max()
    return scores, matriz


def ranking_desde_scores(scores: np.ndarray, canones: list[str]) -> list[str]:
    """Ordena los canones por score descendente (empates estables).

    Args:
        scores: Array (n_articulos,) de scores.
        canones: Canones alineados con scores.

    Returns:
        Canones ordenados por score descendente.
    """
    orden = np.argsort(-scores, kind="stable")
    return [canones[i] for i in orden]


# ===== SECCION 6: estrategia E5-multilingual por pasajes (GPU) =====
def chunkear_texto(texto: str, tokenizer: object) -> list[str]:
    """Divide un texto en chunks de <=512 tokens con solape (ADR-3).

    Reserva CONFIG['reserva_prefijo_tokens'] tokens para el prefijo
    "query: "/"passage: " que se anade al codificar.

    Args:
        texto: Texto crudo largo (ventana de query o pasaje).
        tokenizer: Tokenizer del modelo E5.

    Returns:
        Lista de chunks decodificados (sin el prefijo).
    """
    ids = tokenizer.encode(texto, add_special_tokens=False)
    maximo = CONFIG["tamano_chunk"] - CONFIG["reserva_prefijo_tokens"]
    solape = CONFIG["solape_chunk"]
    chunks: list[str] = []
    inicio = 0
    total = len(ids)
    while inicio < total:
        fin = min(inicio + maximo, total)
        chunks.append(tokenizer.decode(ids[inicio:fin], skip_special_tokens=True))
        if fin == total:
            break
        inicio = fin - solape
    return chunks


def _codificar_con_fallback(modelo: object, textos: list[str]) -> np.ndarray:
    """Codifica textos con E5 reintentando con batch menor ante OOM.

    Args:
        modelo: SentenceTransformer cargado en el dispositivo elegido.
        textos: Textos con prefijo ya incluido.

    Returns:
        Matriz de embeddings normalizados (n, dim).
    """
    try:
        return np.asarray(
            modelo.encode(
                textos,
                batch_size=CONFIG["batch_embedding"],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )
    except RuntimeError:
        print(
            "[aviso] OOM en E5: se re-codifica con batch "
            f"{CONFIG['batch_embedding_oom']} (documentado)"
        )
        return np.asarray(
            modelo.encode(
                textos,
                batch_size=CONFIG["batch_embedding_oom"],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )


def media_por_grupos(matriz: np.ndarray, longitudes: list[int]) -> np.ndarray:
    """Agrega por media (chunk-mean de ADR-3) y renormaliza cada grupo.

    Args:
        matriz: Embeddings planos (sum(longitudes), dim).
        longitudes: Numero de chunks de cada documento, en orden.

    Returns:
        Matriz (len(longitudes), dim) con un vector unitario por documento.
    """
    vectores = []
    inicio = 0
    for n_chunks in longitudes:
        bloque = matriz[inicio : inicio + n_chunks]
        inicio += n_chunks
        media = bloque.mean(axis=0)
        norma = float(np.linalg.norm(media))
        vectores.append(media / norma if norma > 0 else media)
    return np.vstack(vectores)


def scores_e5_caso(
    vectores_query: np.ndarray, matriz_pasajes: np.ndarray, grupos_indices: list[np.ndarray]
) -> np.ndarray:
    """Scores E5 por articulo: max coseno entre chunks de query y pasajes.

    Cada chunk de la query se compara con todos los pasajes; el score del
    pasaje es el maximo entre chunks y el del articulo el maximo entre sus
    pasajes (max-pooling en ambos niveles, CONFIG['agregacion_chunks_e5']).

    Args:
        vectores_query: Matriz (n_chunks, dim) de embeddings unitarios de
            los chunks de la query.
        matriz_pasajes: Matriz (n_pasajes, dim) de embeddings unitarios.
        grupos_indices: Arrays de posiciones de pasajes por articulo.

    Returns:
        Array (n_articulos,) de similitudes maximas por articulo.
    """
    similitudes = matriz_pasajes @ vectores_query.T
    maximos_pasajes = similitudes.max(axis=1)
    scores = np.zeros(len(grupos_indices), dtype=np.float64)
    for i, indices in enumerate(grupos_indices):
        scores[i] = maximos_pasajes[indices].max()
    return scores


# ===== SECCION 7: estrategia hibrida pre-registrada (mejora 6) =====
def normalizar_min_max(scores: np.ndarray) -> np.ndarray:
    """Normaliza min-max un vector de scores por caso.

    Si todos los scores son iguales (maximo == minimo) devuelve ceros: esa
    estrategia no discrimina en ese caso y no aporta al hibrido.

    Args:
        scores: Array de scores de una estrategia para un caso.

    Returns:
        Array normalizado a [0, 1].
    """
    minimo, maximo = float(scores.min()), float(scores.max())
    if maximo <= minimo:
        return np.zeros_like(scores)
    return (scores - minimo) / (maximo - minimo)


def scores_hibridos(scores_bm25: np.ndarray, scores_e5: np.ndarray) -> np.ndarray:
    """Combina BM25 y E5 con pesos pre-registrados 0.5/0.5 (min-max por caso).

    Args:
        scores_bm25: Scores BM25 del caso (n_articulos,).
        scores_e5: Scores E5 del caso (n_articulos,).

    Returns:
        Array (n_articulos,) de scores hibridos.
    """
    pesos = CONFIG["pesos_hibrido"]
    return (
        pesos["bm25"] * normalizar_min_max(scores_bm25)
        + pesos["e5"] * normalizar_min_max(scores_e5)
    )


# ===== SECCION 7b: estrategia bm25_rerank (ciclo v2 - Fase T) =====
def ranking_bm25_rerank(
    reranker: object,
    scores_bm25: np.ndarray,
    canones: list[str],
    pasajes: pd.DataFrame,
    grupos_indices: list[np.ndarray],
    texto_query: str,
) -> tuple[list[str], np.ndarray]:
    """Rerankea el top-20 BM25 con el cross-encoder bge-reranker-v2-m3.

    ciclo v2 - Fase T: dos etapas. (1) BM25 aporta los
    CONFIG['k_candidatos_rerank'] = 20 mejores articulos (sobre el indice
    activo: permanentes con --indice permanentes). (2) El cross-encoder
    puntua cada par (query, pasaje) donde la query es el TEXTO de la
    sentencia (el mismo que alimenta los chunks de bm25/e5, sin scores)
    y el pasaje es el texto del pasaje del articulo (el mismo que indexa
    BM25); el score del articulo es el MAXIMO entre sus pasajes
    (max-pooling, coherente con ADR-9). Se reordena por score y se
    devuelve el top-10.

    Args:
        reranker: RerankerBGE ya cargado (scripts/lib/reranker_bge.py).
        scores_bm25: Scores BM25 del caso alineados con canones.
        canones: Canones de los articulos indexados.
        pasajes: DataFrame con columnas canon y texto (un pasaje/fila).
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        texto_query: Texto de la sentencia (ventana de query).

    Returns:
        Tupla (ranking_top10, scores_rerank):
        - ranking_top10: Canones rerankeados (<= 10, los mejores tras
          el cross-encoder);
        - scores_rerank: Array (n_articulos,) con el score del reranker
          en los candidatos del top-20 y -inf en el resto (para
          top10_por_caso; el ranking completo se corta a 10).
    """
    orden_bm25 = np.argsort(-scores_bm25, kind="stable")
    candidatos = orden_bm25[: CONFIG["k_candidatos_rerank"]]
    textos_pasajes = pasajes["texto"].astype(str).tolist()
    queries: list[str] = []
    pasajes_lote: list[str] = []
    mapa_pares: list[tuple[int, int]] = []
    for pos_articulo, indice in enumerate(candidatos):
        for indice_pasaje in grupos_indices[indice]:
            queries.append(texto_query)
            pasajes_lote.append(textos_pasajes[indice_pasaje])
            mapa_pares.append((pos_articulo, indice_pasaje))
    scores_pares = reranker.puntuar(queries, pasajes_lote)
    scores_rerank = np.full(len(canones), -np.inf, dtype=np.float64)
    maximos = np.full(len(candidatos), -np.inf, dtype=np.float64)
    for (pos_articulo, _), score in zip(mapa_pares, scores_pares, strict=True):
        maximos[pos_articulo] = max(maximos[pos_articulo], float(score))
    for pos_articulo, indice in enumerate(candidatos):
        scores_rerank[indice] = maximos[pos_articulo]
    orden_rerank = candidatos[np.argsort(-maximos, kind="stable")]
    ranking = [canones[i] for i in orden_rerank[: CONFIG["k_rerank_salida"]]]
    return ranking, scores_rerank


def ranking_rerank_chunks(
    reranker: object,
    scores_candidatos: np.ndarray,
    matriz_bm25: np.ndarray,
    chunks: list[str],
    canones: list[str],
    pasajes: pd.DataFrame,
    grupos_indices: list[np.ndarray],
    k_candidatos: int,
) -> tuple[list[str], np.ndarray, int]:
    """Rerankea los top-k_candidatos con pares (chunk, pasaje) y max-pooling.

    ciclo v2 - Fase T iter2 (generalizada en iter3): correccion del
    truncamiento de la query que hizo fallar a v4. Para cada articulo
    candidato del top-k_candidatos (ordenado por scores_candidatos):
    (1) su mejor pasaje es el de mayor score BM25 del caso (columna de
    maximos de la matriz chunk x pasaje); (2) los top-
    CONFIG['k_chunks_rerank'] = 3 chunks de la query son los de mayor
    score BM25 contra ese pasaje; (3) el cross-encoder puntua los 3
    pares (chunk, pasaje) y el score del articulo es el MAXIMO entre
    ellos (max-pooling, coherente con ADR-9). Cada par cabe en 512
    tokens (chunk ~800 chars + pasaje <= 120 palabras); la longitud
    real SIN truncar se mide con el tokenizer del reranker y se
    devuelve para el diagnostico del JSON.

    ciclo v2 - Fase T iter3: la funcion se generaliza para las variantes
    top-50. scores_candidatos define QUIENES son los candidatos (scores
    BM25 para bm25_rerank_chunks*; scores hibridos 0.5/0.5 para
    hibrido_rerank_chunks_top50) y k_candidatos define CUANTOS (20 en
    v5, 50 en iter3). La derivacion de los pares NO cambia: siempre usa
    la matriz BM25 chunk x pasaje del caso (mejor pasaje + top-3
    chunks), igual que en v5.

    Args:
        reranker: RerankerBGE ya cargado (scripts/lib/reranker_bge.py).
        scores_candidatos: Scores que ordenan a los candidatos (n_articulos,)
            alineados con canones (BM25 o hibrido segun la estrategia).
        matriz_bm25: Matriz (n_chunks, n_pasajes) de scores BM25 del
            caso (salida de scores_bm25_caso; fuente de los pares).
        chunks: Chunks de la ventana de query (alineados con las filas
            de matriz_bm25).
        canones: Canones de los articulos indexados.
        pasajes: DataFrame con columnas canon y texto (un pasaje/fila).
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        k_candidatos: Numero de candidatos de la 1a etapa (20 en v5,
            50 en las variantes iter3).

    Returns:
        Tupla (ranking_top10, scores_rerank, max_tokens_par):
        - ranking_top10: Canones rerankeados (<= 10);
        - scores_rerank: Array (n_articulos,) con el score del reranker
          en los candidatos y -inf en el resto;
        - max_tokens_par: Maxima longitud en tokens (sin truncar) de
          los pares (chunk, pasaje) puntuados en este caso.
    """
    orden_candidatos = np.argsort(-scores_candidatos, kind="stable")
    candidatos = orden_candidatos[:k_candidatos]
    maximos_pasajes = matriz_bm25.max(axis=0)
    textos_pasajes = pasajes["texto"].astype(str).tolist()
    queries: list[str] = []
    pasajes_lote: list[str] = []
    mapa_pares: list[tuple[int, int]] = []
    for pos_articulo, indice in enumerate(candidatos):
        indices_pasaje = grupos_indices[indice]
        mejor_pasaje = int(indices_pasaje[np.argmax(maximos_pasajes[indices_pasaje])])
        scores_chunks = matriz_bm25[:, mejor_pasaje]
        top_chunks = np.argsort(-scores_chunks, kind="stable")[
            : CONFIG["k_chunks_rerank"]
        ]
        for fila_chunk in top_chunks:
            queries.append(chunks[fila_chunk])
            pasajes_lote.append(textos_pasajes[mejor_pasaje])
            mapa_pares.append((pos_articulo, fila_chunk))
    scores_pares = reranker.puntuar(queries, pasajes_lote)
    scores_rerank = np.full(len(canones), -np.inf, dtype=np.float64)
    maximos = np.full(len(candidatos), -np.inf, dtype=np.float64)
    for (pos_articulo, _), score in zip(mapa_pares, scores_pares, strict=True):
        maximos[pos_articulo] = max(maximos[pos_articulo], float(score))
    for pos_articulo, indice in enumerate(candidatos):
        scores_rerank[indice] = maximos[pos_articulo]
    orden_rerank = candidatos[np.argsort(-maximos, kind="stable")]
    ranking = [canones[i] for i in orden_rerank[: CONFIG["k_rerank_salida"]]]
    max_tokens_par = reranker.max_tokens_par(queries, pasajes_lote)
    return ranking, scores_rerank, max_tokens_par


def ranking_bm25_rerank_chunks(
    reranker: object,
    scores_bm25: np.ndarray,
    matriz_bm25: np.ndarray,
    chunks: list[str],
    canones: list[str],
    pasajes: pd.DataFrame,
    grupos_indices: list[np.ndarray],
) -> tuple[list[str], np.ndarray, int]:
    """Rerankea el top-20 BM25 con pares (chunk, pasaje) (v5, iter2).

    ciclo v2 - Fase T iter3: la estrategia v5 se CONSERVA intacta; esta
    funcion es un delegado fino de ranking_rerank_chunks con los mismos
    argumentos de siempre (candidatos = top-20 BM25,
    CONFIG['k_candidatos_rerank']) para garantizar la no-regresion de
    v5 (mismos scores que el smoke v5).

    Args:
        reranker: RerankerBGE ya cargado (scripts/lib/reranker_bge.py).
        scores_bm25: Scores BM25 del caso alineados con canones.
        matriz_bm25: Matriz (n_chunks, n_pasajes) de scores BM25 del
            caso (salida de scores_bm25_caso).
        chunks: Chunks de la ventana de query (alineados con las filas
            de matriz_bm25).
        canones: Canones de los articulos indexados.
        pasajes: DataFrame con columnas canon y texto (un pasaje/fila).
        grupos_indices: Arrays de posiciones de pasajes por articulo.

    Returns:
        Tupla (ranking_top10, scores_rerank, max_tokens_par) de
        ranking_rerank_chunks con k_candidatos = top-20 BM25.
    """
    return ranking_rerank_chunks(
        reranker,
        scores_bm25,
        matriz_bm25,
        chunks,
        canones,
        pasajes,
        grupos_indices,
        CONFIG["k_candidatos_rerank"],
    )


# ===== SECCION 7c: prior de hubs CP (ciclo v2 - Fase W-a) =====
def construir_prior_hubs(
    entidades: pd.DataFrame, canones: set[str]
) -> dict[str, float]:
    """Construye el prior de hubs CP por frecuencia global de citacion.

    La senal es la FRECUENCIA con que cada articulo de la CP aparece en
    la columna arts_cp de las 23.742 sentencias del corpus (weak labels
    del NER M2), normalizada a [0,1] por el maximo (el articulo mas
    citado recibe 1.0).

    NO-LEAKAGE: es una estadistica GLOBAL de corpus, estilo IDF, identica
    para toda consulta y calculada sobre TODO el corpus; no contiene
    informacion por caso ni privilegia las sentencias del benchmark (el
    prior de un articulo no depende de que casos se evaluan, igual que
    un IDF o las frecuencias de documento de BM25). El articulo mas
    citado (241, presente en ~20.289 de 23.742 sentencias) es un hub
    estructural de la jurisprudencia constitucional, no una pista de la
    respuesta de un caso concreto.

    Eleccion de senal: se comparo en smoke contra el PageRank ponderado
    del grafo de co-citacion (arista = dos articulos citados en la misma
    sentencia; 66.976 pares, 2.261.390 pesos): Spearman 0.9828 y mismo
    top-5 (241, 86, 13, 2, 1). La frecuencia normalizada es la senal mas
    simple y estable, y el PageRank no aporta informacion adicional, por
    lo que se adopta la frecuencia (documentado en el plan de la Fase W).

    Los valores fuera del conjunto de canones (p. ej. typos del regex M2
    como "54A") se ignoran; los canones nunca citados quedan en 0.0.

    Args:
        entidades: DataFrame de M2 con la columna arts_cp (JSON string
            con la lista de articulos citados por sentencia).
        canones: Conjunto de canones validos del parquet de la CP.

    Returns:
        Dict {canon: valor en [0,1]} con TODOS los canones (0.0 para los
        nunca citados); vacio si el corpus no tiene citas resolubles.
    """
    frecuencia: dict[str, int] = {canon: 0 for canon in canones}
    for crudo in entidades["arts_cp"]:
        arts = json.loads(crudo) if crudo else []
        for articulo in arts:
            clave = str(articulo)
            if clave in frecuencia:
                frecuencia[clave] += 1
    maximo = max(frecuencia.values(), default=0)
    if maximo <= 0:
        return {}
    # Orden deterministico por canon para que el JSON sea reproducible.
    return {
        canon: round(valor / maximo, 6)
        for canon, valor in sorted(frecuencia.items())
    }


def cargar_prior_hubs(ruta: str | None = None) -> dict[str, float]:
    """Carga el prior de hubs desde JSON.

    Args:
        ruta: Ruta del JSON; por defecto CONFIG['ruta_prior_hubs'].

    Returns:
        Dict {canon: valor} con los valores convertidos a float.
    """
    ruta_prior = Path(ruta if ruta is not None else CONFIG["ruta_prior_hubs"])
    with ruta_prior.open("r", encoding="utf-8") as manejador:
        crudo = json.load(manejador)
    return {str(canon): float(valor) for canon, valor in crudo.items()}


def cargar_o_construir_prior_hubs() -> dict[str, float]:
    """Carga data/prior_hubs_cp.json o lo construye y guarda si falta.

    La construccion lee data/entidades_sentencias.parquet y el parquet
    completo de la CP; el archivo resultante es independiente del modo
    --indice (se calcula sobre los 497 canones) para que la calibracion
    sea reproducible entre corridas.

    Returns:
        Dict {canon: valor normalizado}.
    """
    ruta_prior = Path(CONFIG["ruta_prior_hubs"])
    if ruta_prior.exists():
        prior = cargar_prior_hubs()
        print(f"Prior hubs: {len(prior)} canones cargados de {ruta_prior}")
        return prior
    entidades = pd.read_parquet(CONFIG["ruta_entidades"], columns=["arts_cp"])
    articulos = pd.read_parquet(CONFIG["ruta_constitucion"], columns=["numero"])
    canones = set(articulos["numero"].astype(str))
    prior = construir_prior_hubs(entidades, canones)
    guardar_json(CONFIG["ruta_prior_hubs"], prior)
    print(f"Prior hubs: {len(prior)} canones construidos en {ruta_prior}")
    return prior


def generar_prior_hubs() -> dict[str, float]:
    """Construye y guarda data/prior_hubs_cp.json; imprime el top-10.

    Returns:
        Dict {canon: valor normalizado} generado (se guarda en
        CONFIG['ruta_prior_hubs']).
    """
    entidades = pd.read_parquet(CONFIG["ruta_entidades"], columns=["arts_cp"])
    articulos = pd.read_parquet(CONFIG["ruta_constitucion"], columns=["numero"])
    canones = set(articulos["numero"].astype(str))
    prior = construir_prior_hubs(entidades, canones)
    guardar_json(CONFIG["ruta_prior_hubs"], prior)
    print(
        f"Prior hubs guardado en {CONFIG['ruta_prior_hubs']} "
        f"({len(prior)} canones)"
    )
    for canon, valor in top_hubs(prior, 10):
        print(f"  {canon:>6}: {valor:.4f}")
    return prior


def top_hubs(prior: dict[str, float], n: int = 10) -> list[tuple[str, float]]:
    """Devuelve los n articulos con mayor valor del prior.

    Args:
        prior: Dict {canon: valor}.
        n: Numero de articulos a devolver.

    Returns:
        Lista de tuplas (canon, valor) ordenada por valor descendente y
        canon ascendente (desempate estable).
    """
    return sorted(prior.items(), key=lambda par: (-par[1], par[0]))[:n]


def aplicar_prior_hubs(
    scores: dict[str, float], prior: dict, alpha: float, norm: bool = True
) -> dict[str, float]:
    """Suma el prior de hubs a los scores base de una estrategia.

    score_final = score_norm + alpha * prior_norm(canon). Con norm=True
    los scores base se normalizan min-max DENTRO del diccionario (rango
    [0,1]) para que alpha sea comparable entre estrategias de escalas
    distintas (BM25, coseno E5, logits del cross-encoder); con norm=False
    se usan los valores crudos (solo diagnostico). Un canon ausente del
    prior recibe 0.0. Con alpha=0 se devuelve una copia intacta (el
    comportamiento de las fases previas no cambia).

    Args:
        scores: Dict {canon: score base} de una estrategia y un caso.
        prior: Dict {canon: valor} del prior de hubs (normalizado).
        alpha: Peso del prior (>= 0).
        norm: Si True, min-max de los scores base antes de sumar.

    Returns:
        Dict {canon: score_final} con las mismas claves que scores.
    """
    if alpha == 0.0 or not scores:
        return dict(scores)
    valores = np.asarray(list(scores.values()), dtype=np.float64)
    if norm:
        minimo, maximo = float(valores.min()), float(valores.max())
        if maximo > minimo:
            valores = (valores - minimo) / (maximo - minimo)
        else:
            valores = np.zeros_like(valores)
    return {
        canon: float(base + alpha * float(prior.get(canon, 0.0)))
        for (canon, _), base in zip(scores.items(), valores, strict=True)
    }


def aplicar_prior_scores(
    scores: np.ndarray, canones: list[str], prior: dict, alpha: float
) -> np.ndarray:
    """Aplica el prior a un vector de scores alineado con canones.

    Los elementos no finitos (p. ej. -inf del reranker para los articulos
    que no son candidatos de la primera etapa) se preservan tal cual: la
    normalizacion min-max y la suma del prior ocurren solo entre los
    candidatos con score finito, de modo que el prior NO puede resucitar
    articulos que quedaron fuera del recall de la primera etapa.

    Args:
        scores: Array (n_articulos,) de scores base.
        canones: Canones alineados con scores.
        prior: Dict {canon: valor} del prior de hubs.
        alpha: Peso del prior (>= 0).

    Returns:
        Array nuevo (n_articulos,) con los scores priorizados; es una
        copia incluso con alpha=0 (nunca se muta el original).
    """
    if alpha == 0.0:
        return scores.copy()
    finitos = np.flatnonzero(np.isfinite(scores))
    if finitos.size == 0:
        return scores.copy()
    dicc = {canones[i]: float(scores[i]) for i in finitos}
    priorizado = aplicar_prior_hubs(dicc, prior, alpha, norm=True)
    resultado = scores.copy()
    for i in finitos:
        resultado[i] = priorizado[canones[i]]
    return resultado


def ranking_con_prior(
    scores: np.ndarray,
    canones: list[str],
    estrategia: str,
    prior: dict,
    alpha: float,
) -> tuple[list[str], np.ndarray]:
    """Re-rankea una estrategia aplicando el prior de hubs.

    Args:
        scores: Array de scores base de la estrategia (puede contener
            -inf en las estrategias de rerank).
        canones: Canones alineados con scores.
        estrategia: Nombre de la estrategia (define si el ranking se
            trunca a CONFIG['k_rerank_salida']).
        prior: Dict {canon: valor} del prior de hubs.
        alpha: Peso del prior.

    Returns:
        Tupla (ranking, scores_final): ranking ordenado por el score
        priorizado (truncado en las estrategias de rerank) y el vector
        de scores final.
    """
    scores_final = aplicar_prior_scores(scores, canones, prior, alpha)
    ranking = ranking_desde_scores(scores_final, canones)
    if estrategia in ESTRATEGIAS_RERANK:
        ranking = ranking[: CONFIG["k_rerank_salida"]]
    return ranking, scores_final


def aplicar_prior_a_casos(
    casos: list[dict], prior: dict, canones: list[str], alpha: float
) -> None:
    """Aplica el prior in-place a las estrategias rankeables de los casos.

    Actualiza tanto el vector de scores (caso[FUENTE_SCORES[estrategia]])
    como el ranking, de modo que las metricas, el top-10 por caso y las
    figuras sean coherentes con el score final. La cita-exacta nunca se
    toca (referencia circular, sin scores).

    Args:
        casos: Lista de dicts con scores_* y rankings por estrategia.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.
        alpha: Peso del prior (> 0).
    """
    for caso in casos:
        for estrategia in CONFIG["estrategias"]:
            if estrategia not in ESTRATEGIAS_RANKEABLES:
                continue
            clave = FUENTE_SCORES[estrategia]
            if clave not in caso:
                continue
            ranking, scores_final = ranking_con_prior(
                caso[clave], canones, estrategia, prior, alpha
            )
            caso[clave] = scores_final
            caso["rankings"][estrategia] = ranking


def metricas_rejilla_hubs(
    casos: list[dict],
    prior: dict,
    canones: list[str],
    alphas: list[float],
    estrategias: list[str],
) -> pd.DataFrame:
    """Re-rankea y re-mide cada (caso, estrategia, alpha) de la rejilla.

    NO re-ejecuta modelos: parte de los scores base ya calculados. Ademas
    de las medidas del protocolo (k=5) calcula P@10 y F1@10 para el
    reporte de la Fase W-a.

    Args:
        casos: Lista de dicts con scores base y gold (arts).
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.
        alphas: Rejilla de pesos del prior.
        estrategias: Estrategias rankeables a evaluar.

    Returns:
        DataFrame largo con columnas indice, id, estrategia, alpha,
        p_at_5, r_at_5, f1_at_5, mrr, ndcg_at_10, p_at_10, f1_at_10.
    """
    filas = []
    for indice, caso in enumerate(casos):
        for estrategia in estrategias:
            scores_base = caso[FUENTE_SCORES[estrategia]]
            for alpha in alphas:
                ranking, _ = ranking_con_prior(
                    scores_base, canones, estrategia, prior, alpha
                )
                m5 = metricas_ranking(caso["arts"], ranking, CONFIG["k"])
                m10 = metricas_ranking(caso["arts"], ranking, CONFIG["k_f1_10"])
                filas.append(
                    {
                        "indice": indice,
                        "id": caso["id"],
                        "estrategia": estrategia,
                        "alpha": alpha,
                        "p_at_5": m5["p_at_k"],
                        "r_at_5": m5["r_at_k"],
                        "f1_at_5": m5["f1_at_k"],
                        "mrr": m5["mrr"],
                        "ndcg_at_10": m5["ndcg_at_10"],
                        "p_at_10": m10["p_at_k"],
                        "f1_at_10": m10["f1_at_k"],
                    }
                )
    return pd.DataFrame(filas)


def promediar_alpha_hubs(
    tabla: pd.DataFrame, estrategia: str, alpha: float, indices: set[int]
) -> dict:
    """Promedia las medidas de calibracion sobre un subconjunto de casos.

    Args:
        tabla: DataFrame de metricas_rejilla_hubs.
        estrategia: Estrategia a filtrar.
        alpha: Peso del prior a filtrar (comparacion exacta de float de
            la misma rejilla, sin aritmetica).
        indices: Posiciones (0-based) de los casos a promediar.

    Returns:
        Dict {medida: media} con las medidas de MEDIDAS_CALIBRACION.
    """
    filas = tabla[
        (tabla["estrategia"] == estrategia)
        & (tabla["alpha"] == alpha)
        & (tabla["indice"].isin(indices))
    ]
    return {
        medida: float(filas[medida].mean()) for medida in MEDIDAS_CALIBRACION
    }


def graficar_hubs_alpha(
    curvas: dict[str, list[float]],
    estrategia: str,
    alpha_elegido: float,
    ruta: str,
) -> None:
    """Genera la figura de calibracion (curva alpha vs F1@5 A/B/full).

    La ruta la fija el llamador (figuras/hubs_alpha_v7.png en la Fase W-a
    original y hubs_alpha_v8.png en la iter2 con la rejilla extendida).

    Args:
        curvas: Dict {"A": [...], "B": [...], "full": [...]} alineado
            con CONFIG['hubs_alphas_calibracion'].
        estrategia: Estrategia representada (la de mejor F1@5 a alpha=0).
        alpha_elegido: Alpha elegido en full-90 (linea vertical).
        ruta: Ruta del PNG de salida.
    """
    alphas = list(CONFIG["hubs_alphas_calibracion"])
    figura, eje = plt.subplots(figsize=(8, 5))
    for nombre, etiqueta, color in (
        ("full", "full-90", "#1f77b4"),
        ("A", "mitad A (pares)", "#2ca02c"),
        ("B", "mitad B (impares)", "#d62728"),
    ):
        eje.plot(
            alphas,
            curvas[nombre],
            marker="o",
            linewidth=1.2,
            color=color,
            label=f"F1@5 {etiqueta}",
        )
    eje.axvline(
        alpha_elegido,
        color="#7f7f7f",
        linestyle="--",
        label=f"alpha full-90 = {alpha_elegido}",
    )
    eje.set_xlabel("alpha (peso del prior de hubs CP)")
    eje.set_ylabel("F1@5")
    eje.set_title(
        f"Fase W-a: prior de hubs vs F1@5 ({estrategia})", fontsize=10
    )
    eje.grid(True, alpha=0.3)
    eje.legend(fontsize=8)
    figura.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def calibrar_hubs(casos: list[dict], prior: dict, canones: list[str]) -> dict:
    """Calibra alpha del prior con validacion cruzada honesta A/B + full.

    Protocolo anti-overfit (Fase W-a): los 90 casos se parten por su
    posicion en la muestra en mitad A (indices pares) y mitad B (indices
    impares). Para cada estrategia se elige el alpha que maximiza F1@5 en
    A y se reporta su F1@5 en B (y viceversa), mas la curva full-90; la
    recomendacion solo es "adoptar" si el alpha elegido por la otra mitad
    y el de full-90 mejoran F1@5 respecto a alpha=0 en las tres vistas.
    Los empates se resuelven por el alpha mas pequeno de la rejilla para
    que el resultado sea determinista.

    ciclo v2 - Fase W-a iter2: la rejilla se extiende hasta 2.0 (el run
    v7 dejo el optimo en el borde alpha=0.5); adicionalmente se reporta
    el diagnostico curva_alpha_full (alpha -> F1@5 full-90) para cada
    estrategia focal, que permite ver la forma de la curva y donde
    satura. El protocolo de seleccion NO cambia.

    Args:
        casos: Lista de dicts con id, arts (gold), rankings y scores_*.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.

    Returns:
        Dict serializable con la rejilla, el protocolo, las curvas F1@5
        (A/B/full), el detalle por estrategia, el diagnostico
        curva_alpha_full por estrategia focal y la recomendacion.
    """
    estrategias = [
        e for e in CONFIG["estrategias"] if e in ESTRATEGIAS_RANKEABLES
    ]
    alphas = list(CONFIG["hubs_alphas_calibracion"])
    tabla = metricas_rejilla_hubs(casos, prior, canones, alphas, estrategias)
    indices_a = set(range(0, len(casos), 2))
    indices_b = set(range(1, len(casos), 2))
    indices_full = set(range(len(casos)))

    def seleccionar_alpha(estrategia: str, indices: set[int]) -> float:
        """Devuelve el alpha de mayor F1@5 (empate: el mas pequeno)."""
        mejor_alpha = alphas[0]
        mejor_f1 = -1.0
        for alpha in alphas:
            f1 = promediar_alpha_hubs(tabla, estrategia, alpha, indices)[
                "f1_at_5"
            ]
            if f1 > mejor_f1 + 1e-12:
                mejor_alpha, mejor_f1 = alpha, f1
        return mejor_alpha

    curvas_f1: dict[str, dict[str, list[float]]] = {}
    por_estrategia: dict[str, dict] = {}
    for estrategia in estrategias:
        curvas_f1[estrategia] = {
            "A": [
                promediar_alpha_hubs(tabla, estrategia, a, indices_a)[
                    "f1_at_5"
                ]
                for a in alphas
            ],
            "B": [
                promediar_alpha_hubs(tabla, estrategia, a, indices_b)[
                    "f1_at_5"
                ]
                for a in alphas
            ],
            "full": [
                promediar_alpha_hubs(tabla, estrategia, a, indices_full)[
                    "f1_at_5"
                ]
                for a in alphas
            ],
        }
        alpha_a = seleccionar_alpha(estrategia, indices_a)
        alpha_b = seleccionar_alpha(estrategia, indices_b)
        alpha_full = seleccionar_alpha(estrategia, indices_full)
        met_a_0 = promediar_alpha_hubs(tabla, estrategia, 0.0, indices_a)
        met_b_0 = promediar_alpha_hubs(tabla, estrategia, 0.0, indices_b)
        met_full_0 = promediar_alpha_hubs(tabla, estrategia, 0.0, indices_full)
        met_a_alpha_b = promediar_alpha_hubs(
            tabla, estrategia, alpha_b, indices_a
        )
        met_b_alpha_a = promediar_alpha_hubs(
            tabla, estrategia, alpha_a, indices_b
        )
        met_full_alpha_full = promediar_alpha_hubs(
            tabla, estrategia, alpha_full, indices_full
        )
        mejora_a = 100 * (met_a_alpha_b["f1_at_5"] - met_a_0["f1_at_5"])
        mejora_b = 100 * (met_b_alpha_a["f1_at_5"] - met_b_0["f1_at_5"])
        mejora_full = 100 * (
            met_full_alpha_full["f1_at_5"] - met_full_0["f1_at_5"]
        )
        adoptar = mejora_a > 0 and mejora_b > 0 and mejora_full > 0
        por_estrategia[estrategia] = {
            "alpha_elegido_A": alpha_a,
            "alpha_elegido_B": alpha_b,
            "alpha_elegido_full": alpha_full,
            "metricas_A_alpha0": met_a_0,
            "metricas_A_en_alpha_B": met_a_alpha_b,
            "metricas_B_alpha0": met_b_0,
            "metricas_B_en_alpha_A": met_b_alpha_a,
            "metricas_full_alpha0": met_full_0,
            "metricas_full_en_alpha_elegido": met_full_alpha_full,
            "mejora_A_pp": round(mejora_a, 3),
            "mejora_B_pp": round(mejora_b, 3),
            "mejora_full_pp": round(mejora_full, 3),
            "recomendacion": "adoptar" if adoptar else "no adoptar",
            "evidencia": (
                f"F1@5 A {met_a_0['f1_at_5']:.3f}->"
                f"{met_a_alpha_b['f1_at_5']:.3f} (alpha_B={alpha_b}, "
                f"{mejora_a:+.1f} pp); B {met_b_0['f1_at_5']:.3f}->"
                f"{met_b_alpha_a['f1_at_5']:.3f} (alpha_A={alpha_a}, "
                f"{mejora_b:+.1f} pp); full {met_full_0['f1_at_5']:.3f}->"
                f"{met_full_alpha_full['f1_at_5']:.3f} "
                f"(alpha_full={alpha_full}, {mejora_full:+.1f} pp)"
            ),
        }
    # ciclo v2 - Fase W-a iter2: diagnostico por estrategia focal de la
    # forma de la curva (alpha -> F1@5 full-90), para ver donde satura.
    focales = [
        e for e in CONFIG["estrategias_focales_hubs"] if e in estrategias
    ]
    curva_alpha_full = {
        estrategia: {
            str(alpha): round(curvas_f1[estrategia]["full"][posicion], 6)
            for posicion, alpha in enumerate(alphas)
        }
        for estrategia in focales
    }
    mejor_estrategia = max(
        estrategias, key=lambda e: curvas_f1[e]["full"][alphas.index(0.0)]
    )
    ruta_figura = CONFIG["ruta_figura_hubs_v8"]
    graficar_hubs_alpha(
        curvas_f1[mejor_estrategia],
        mejor_estrategia,
        por_estrategia[mejor_estrategia]["alpha_elegido_full"],
        ruta_figura,
    )
    return {
        "grid_alpha": alphas,
        "protocolo": (
            "Seleccion de alpha por F1@5 en la mitad A (indices pares) "
            "validada en B (impares) y viceversa, mas full-90; los scores "
            "base se calculan UNA vez y la rejilla solo re-rankea/re-mide "
            "(sin re-ejecutar modelos). Recomendacion 'adoptar' solo si "
            "mejora F1@5 en A, B y full-90. Rejilla extendida hasta 2.0 "
            "en la iter2 de la Fase W-a (el optimo v7 quedo en el borde)."
        ),
        "mitad_A": {
            "criterio": "indices pares de la muestra (0-based)",
            "n_casos": len(indices_a),
        },
        "mitad_B": {
            "criterio": "indices impares de la muestra (0-based)",
            "n_casos": len(indices_b),
        },
        "estrategias_evaluadas": estrategias,
        "estrategias_focales": focales,
        "mejor_estrategia_figura": mejor_estrategia,
        "curvas_f1_at_5": curvas_f1,
        "curva_alpha_full": curva_alpha_full,
        "por_estrategia": por_estrategia,
        "figura": ruta_figura,
    }


# ===== SECCION 8: figura de deriva temporal =====
def graficar_deriva(por_caso: pd.DataFrame) -> None:
    """Genera figuras/deriva_retrieval_v3.png con F1@5 y MRR por anio.

    Con 3 casos/anio las series se promedian por anio (mas estables que
    el v1, que tenia 1 caso/anio).

    Args:
        por_caso: DataFrame largo con ano, estrategia y medidas.
    """
    etiquetas = {
        "cita_exacta": "cita-exacta (regex M2, circular)",
        "bm25": "BM25 (pasajes + chunks + bigramas LLR)",
        "e5": "E5-multilingual (pasajes, max-pooling)",
        "hibrido": "hibrido 0.5*BM25 + 0.5*E5",
        # ciclo v2 - Fase T
        "bm25_rerank": "BM25 top-20 + bge-reranker-v2-m3",
        # ciclo v2 - Fase T iter2
        "bm25_rerank_chunks": "BM25 top-20 + reranker por chunks (max-pooling)",
        # ciclo v2 - Fase T iter3
        "bm25_rerank_chunks_top50": (
            "BM25 top-50 + reranker por chunks (max-pooling)"
        ),
        "hibrido_rerank_chunks_top50": (
            "hibrido top-50 + reranker por chunks (max-pooling)"
        ),
    }
    colores = {
        "cita_exacta": "#7f7f7f",
        "bm25": "#1f77b4",
        "e5": "#d62728",
        "hibrido": "#2ca02c",
        "bm25_rerank": "#9467bd",
        "bm25_rerank_chunks": "#ff7f0e",
        "bm25_rerank_chunks_top50": "#8c564b",
        "hibrido_rerank_chunks_top50": "#e377c2",
    }
    series = (
        por_caso.groupby(["ano", "estrategia"])[list(MEDIDAS)]
        .mean()
        .reset_index()
    )
    figura, ejes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for medida, eje, titulo in (
        ("f1_at_k", ejes[0], "F1@5 sentencia->articulos CP por anio (3 casos/anio)"),
        ("mrr", ejes[1], "MRR por anio"),
    ):
        for estrategia in CONFIG["estrategias"]:
            serie = series[series["estrategia"] == estrategia].sort_values("ano")
            eje.plot(
                serie["ano"],
                serie[medida],
                marker="o",
                markersize=4,
                linewidth=1.1,
                color=colores[estrategia],
                label=etiquetas[estrategia],
            )
        eje.set_ylabel(medida.upper())
        eje.set_title(titulo, fontsize=10)
        eje.grid(True, alpha=0.3)
        eje.legend(fontsize=8, loc="upper right")
    ejes[1].set_xlabel("Anio de la sentencia")
    ejes[1].set_xticks(range(CONFIG["ano_min"], CONFIG["ano_max"] + 1, 3))
    figura.tight_layout()
    Path(CONFIG["ruta_figura"]).parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(CONFIG["ruta_figura"], dpi=150, bbox_inches="tight")
    plt.close(figura)


# ===== SECCION 8b: top-10 por caso y diagnostico de contaminacion (Fase R) =====
def top10_por_caso(casos: list[dict], canones: list[str]) -> dict:
    """Construye el top-10 (id, score) por caso y estrategia (Fase R).

    Para bm25/e5/hibrido se usan los scores reales del caso (orden del
    ranking ya calculado). Para cita_exacta no hay scores: se asigna un
    score sintetico 1.0 en orden de aparicion (la estrategia es binaria;
    el campo existe por homogeneidad del JSON).

    Args:
        casos: Lista de dicts con rankings y scores_* por estrategia.
        canones: Canones alineados con los arrays de scores.

    Returns:
        Dict {id_caso: {estrategia: [{"id": canon, "score": float}]}} con
        hasta CONFIG['k_top_guardado'] entradas por lista.
    """
    salida: dict = {}
    for caso in casos:
        por_estrategia: dict = {}
        for estrategia in CONFIG["estrategias"]:
            ranking = caso["rankings"][estrategia][: CONFIG["k_top_guardado"]]
            if estrategia in FUENTE_SCORES:
                scores = caso[FUENTE_SCORES[estrategia]]
                posicion = {canon: i for i, canon in enumerate(canones)}
                por_estrategia[estrategia] = [
                    {"id": canon, "score": round(float(scores[posicion[canon]]), 4)}
                    for canon in ranking
                ]
            else:
                por_estrategia[estrategia] = [
                    {"id": canon, "score": 1.0} for canon in ranking
                ]
        salida[caso["id"]] = por_estrategia
    return salida


def es_transitorio(canon: str) -> bool:
    """Indica si un canon es de un articulo transitorio de la CP.

    Args:
        canon: Canon del articulo (p. ej. "Transitorio ACL01016-1", "86").

    Returns:
        True si el canon empieza por CONFIG['prefijo_transitorio'].
    """
    return canon.startswith(CONFIG["prefijo_transitorio"])


def es_id_malformado(canon: str) -> bool:
    """Indica si un canon es un ID malformado del parseo v1.

    Los 45 IDs "Transitorio ACL01016-N"/"ACL01017-N" provienen del Acto
    Legislativo 1 de 2016 mal parseado (diagnostico del plan v2); se
    detectan por la subcadena CONFIG['marca_id_malformado'].

    Args:
        canon: Canon del articulo.

    Returns:
        True si el canon contiene la marca de ID malformado.
    """
    return CONFIG["marca_id_malformado"] in canon


def diagnosticar_contaminacion(casos: list[dict]) -> dict:
    """Cuantifica la contaminacion del indice sucio v1 (Fase R).

    Por caso y estrategia (bm25, hibrido) cuenta articulos transitorios
    en top-5/top-10 e IDs malformados en el top-10; agrega globalmente
    % medio de transitorios, casos con al menos uno en top-5 y total de
    IDs malformados aparecidos en rankings.

    Args:
        casos: Lista de dicts con id, ano y rankings por estrategia.

    Returns:
        Dict con la estructura del campo diagnostico_contaminacion del
        JSON de salida (por estrategia y agregados globales).
    """
    k5, k10 = CONFIG["k_diagnostico"]
    resultado: dict = {}
    for estrategia in ESTRATEGIAS_DIAGNOSTICO:
        transitorios_top5: list[int] = []
        transitorios_top10: list[int] = []
        malformados_top10: list[int] = []
        casos_con_transitorio_top5 = 0
        ids_malformados_totales = 0
        for caso in casos:
            ranking = caso["rankings"][estrategia]
            n_transitorios_top5 = sum(1 for c in ranking[:k5] if es_transitorio(c))
            n_transitorios_top10 = sum(1 for c in ranking[:k10] if es_transitorio(c))
            n_malformados_top10 = sum(
                1 for c in ranking[:k10] if es_id_malformado(c)
            )
            transitorios_top5.append(n_transitorios_top5)
            transitorios_top10.append(n_transitorios_top10)
            malformados_top10.append(n_malformados_top10)
            if n_transitorios_top5 > 0:
                casos_con_transitorio_top5 += 1
            ids_malformados_totales += sum(
                1 for c in ranking if es_id_malformado(c)
            )
        resultado[estrategia] = {
            "pct_transitorios_top5": float(np.mean(transitorios_top5) / k5),
            "pct_transitorios_top10": float(np.mean(transitorios_top10) / k10),
            "casos_con_transitorio_top5": casos_con_transitorio_top5,
            "casos_con_transitorio_top10": sum(1 for n in transitorios_top10 if n > 0),
            "ids_malformados_en_rankings": ids_malformados_totales,
            "ids_malformados_top10": int(np.sum(malformados_top10)),
            "por_caso": [
                {
                    "id": caso["id"],
                    "ano": caso["ano"],
                    "transitorios_top5": n5,
                    "transitorios_top10": n10,
                    "malformados_top10": nm,
                }
                for caso, n5, n10, nm in zip(
                    casos,
                    transitorios_top5,
                    transitorios_top10,
                    malformados_top10,
                    strict=True,
                )
            ],
        }
    return resultado


def graficar_contaminacion(diagnostico: dict) -> None:
    """Genera figuras/contaminacion_transitorios_v3.png (Fase R).

    Panel superior: % medio de articulos transitorios en top-5 vs top-10
    (estrategia bm25). Panel inferior: numero de casos con al menos un
    transitorio en top-5 por periodo cronologico (bm25).

    Args:
        diagnostico: Dict de diagnosticar_contaminacion (usa bm25).
    """
    diag_bm25 = diagnostico["bm25"]
    k5, k10 = CONFIG["k_diagnostico"]
    por_caso = diag_bm25["por_caso"]
    periodos = [nombre for nombre, _, _ in CONFIG["periodos"]]
    casos_top5_por_periodo = {nombre: 0 for nombre in periodos}
    for fila in por_caso:
        periodo = asignar_periodo(fila["ano"])
        if fila["transitorios_top5"] > 0:
            casos_top5_por_periodo[periodo] += 1

    figura, ejes = plt.subplots(2, 1, figsize=(9, 7))
    eje = ejes[0]
    barras = eje.bar(
        [f"top-{k5}", f"top-{k10}"],
        [
            diag_bm25["pct_transitorios_top5"] * 100,
            diag_bm25["pct_transitorios_top10"] * 100,
        ],
        color=["#1f77b4", "#ff7f0e"],
        width=0.5,
    )
    eje.bar_label(barras, fmt="%.1f%%", fontsize=9)
    eje.set_ylabel("% de articulos transitorios")
    eje.set_title(
        "Contaminacion del indice CP (sucio-v1): articulos transitorios "
        "en el ranking BM25",
        fontsize=10,
    )
    eje.set_ylim(0, max(5.0, eje.get_ylim()[1] * 1.25))
    eje.grid(True, axis="y", alpha=0.3)

    eje = ejes[1]
    barras = eje.bar(
        periodos,
        [casos_top5_por_periodo[nombre] for nombre in periodos],
        color="#1f77b4",
        width=0.5,
    )
    eje.bar_label(barras, fontsize=9)
    eje.set_ylabel("Numero de casos")
    eje.set_title(
        "Casos con al menos 1 articulo transitorio en top-5 (BM25)",
        fontsize=10,
    )
    eje.set_xlabel("Periodo de la sentencia")
    eje.grid(True, axis="y", alpha=0.3)

    figura.tight_layout()
    Path(CONFIG["ruta_figura_contaminacion"]).parent.mkdir(
        parents=True, exist_ok=True
    )
    figura.savefig(
        CONFIG["ruta_figura_contaminacion"], dpi=150, bbox_inches="tight"
    )
    plt.close(figura)


# ===== SECCION 9: guardado =====
def guardar_json(ruta: str, contenido: dict) -> None:
    """Guarda un dict como JSON UTF-8 con indentacion.

    Args:
        ruta: Ruta del archivo de salida.
        contenido: Dict serializable.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with Path(ruta).open("w", encoding="utf-8") as manejador:
        json.dump(contenido, manejador, ensure_ascii=False, indent=2)


def cargar_globales_v1() -> dict | None:
    """Lee los resultados globales del benchmark v1 para comparar (solo lectura).

    Returns:
        Dict {estrategia: {medidas}} del v1, o None si no existe el archivo.
    """
    ruta = Path(CONFIG["ruta_benchmark_v1"])
    if not ruta.exists():
        return None
    with ruta.open("r", encoding="utf-8") as manejador:
        return json.load(manejador).get("global")


# ===== SECCION 10: orquestacion principal =====
def parsear_args() -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos (Fases S1/T/W).

    ciclo v2 - Fase T: --rerank/--no-rerank activa las estrategias de
    reranking (default ON en esta fase); --limite N recorta la muestra a
    los primeros N casos SOLO para smoke tests del pipeline.
    ciclo v2 - Fase W-a: --hubs-alpha fija el peso del prior de hubs,
    --calibrar-hubs activa la rejilla con validacion cruzada A/B y
    --generar-prior construye data/prior_hubs_cp.json y termina.
    ciclo v2 - Fase W-a iter2: la rejilla de --calibrar-hubs se extiende
    hasta 2.0 (el optimo v7 quedo en el borde) y las salidas son v8.

    Returns:
        Namespace con indice ("permanentes"/"completo"), rerank (bool),
        limite (int | None), hubs_alpha (float), calibrar_hubs (bool) y
        generar_prior (bool).

    Raises:
        SystemExit: Si --hubs-alpha es negativo (parser.error).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark M7 v3 del vinculo sentencia->articulos CP. Fase S1: "
            "--indice permanentes construye el indice solo sobre articulos "
            "permanentes (sin transitorios ni preambulo); --indice completo "
            "preserva el comportamiento v3 (regresion). Fase T iter3: "
            "--rerank (default) anade las estrategias bm25_rerank (v4, "
            "query completa), bm25_rerank_chunks (v5, top-3 chunks con "
            "max-pooling) y las top-50 (bm25_rerank_chunks_top50 e "
            "hibrido_rerank_chunks_top50, para atacar el techo de "
            "recall@20) con bge-reranker-v2-m3 y escribe las salidas v6; "
            "--no-rerank reproduce exactamente el v3_limpio. Fase W-a: "
            "--hubs-alpha FLOAT aplica el prior de hubs de la CP (0.0 = "
            "sin prior) y --calibrar-hubs calibra el alpha con validacion "
            "cruzada A/B; la iter2 de la Fase W-a extiende la rejilla "
            "hasta 2.0 y ambas escriben las salidas v8."
        )
    )
    parser.add_argument(
        "--indice",
        choices=["permanentes", "completo"],
        default=(
            "completo" if CONFIG["incluir_transitorios_bench"] else "permanentes"
        ),
        help=(
            "Modo del indice BM25: 'permanentes' (default, Fase S1) excluye "
            "transitorios y preambulo; 'completo' indexa los 497 articulos."
        ),
    )
    parser.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=CONFIG["rerank_default"],
        help=(
            "Activa las estrategias de reranking (Fase T iter3, default "
            "ON): bm25_rerank (v4, query completa), bm25_rerank_chunks "
            "(v5, top-3 chunks + max-pooling) y las NUEVAS top-50 "
            "(bm25_rerank_chunks_top50 e hibrido_rerank_chunks_top50, "
            "candidatos top-50 de BM25 y del hibrido 0.5/0.5). Salida en "
            "resultados/vinculo_cp_benchmark_v6.json."
        ),
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help=(
            "Desactiva el reranking: el benchmark se comporta exactamente "
            "como v3_limpio (regresion, sin estrategia bm25_rerank); las "
            "salidas van a /tmp/opencode/*_regresion para no sobrescribir "
            "los archivos _limpio."
        ),
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help=(
            "Recorta la muestra temporal a los primeros N casos (SOLO para "
            "smoke tests del pipeline E2E; el run final va sin limite)."
        ),
    )
    # ciclo v2 - Fase W-a: prior de hubs.
    parser.add_argument(
        "--hubs-alpha",
        type=float,
        default=CONFIG["hubs_alpha_default"],
        help=(
            "Peso alpha del prior de hubs CP (Fase W-a; default 0.0 = sin "
            "prior, comportamiento identico al v6). Con alpha>0 el score "
            "final de las estrategias rankeables es min-max(score_base) + "
            "alpha * prior_hubs; cita_exacta no se prioriza. Salida en "
            "resultados/vinculo_cp_benchmark_v8.json."
        ),
    )
    parser.add_argument(
        "--calibrar-hubs",
        action="store_true",
        help=(
            "Calibra --hubs-alpha (Fase W-a iter2): rejilla extendida "
            "{0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0,1.5,2.0} "
            "re-rankeando los scores base ya calculados (sin re-ejecutar "
            "modelos) con validacion cruzada A/B (indices pares/impares) "
            "y full-90; escribe la seccion calibracion_hubs del v8 "
            "(incluido curva_alpha_full) y figuras/hubs_alpha_v8.png."
        ),
    )
    parser.add_argument(
        "--generar-prior",
        action="store_true",
        help=(
            "Construye data/prior_hubs_cp.json desde las 23.742 sentencias "
            "(frecuencia global de citacion normalizada), imprime el "
            "top-10 y termina sin ejecutar el benchmark."
        ),
    )
    args = parser.parse_args()
    if args.hubs_alpha < 0:
        parser.error("--hubs-alpha debe ser >= 0")
    return args


def aplicar_modo_indice(modo: str) -> bool:
    """Ajusta CONFIG segun el modo de indice elegido (Fase S1).

    Deriva las rutas de salida y los campos version/indice del JSON a
    partir del flag --indice, sin tocar NUNCA las salidas v2/v3 previas
    (ADR-11): permanentes escribe los archivos con sufijo _limpio y
    completo con sufijo _completo.

    Args:
        modo: "permanentes" o "completo" (choices de parsear_args).

    Returns:
        True si el indice incluye transitorios (modo completo).

    Raises:
        ValueError: Si modo no es uno de los valores permitidos.
    """
    if modo == "permanentes":
        incluir = False
        sufijo = CONFIG["sufijo_limpio"]
        CONFIG["version_salida"] = CONFIG["version_salida_limpia"]
        CONFIG["version_indice"] = CONFIG["version_indice_limpio"]
    elif modo == "completo":
        incluir = True
        sufijo = CONFIG["sufijo_completo"]
        CONFIG["version_salida"] = CONFIG["version_salida_limpia"]
        CONFIG["version_indice"] = CONFIG["version_indice_completo"]
    else:
        raise ValueError(f"Modo de indice desconocido: {modo}")
    CONFIG["ruta_benchmark"] = (
        f"resultados/vinculo_cp_benchmark_v3{sufijo}.json"
    )
    CONFIG["ruta_por_ano"] = f"resultados/vinculo_cp_por_ano_v3{sufijo}.csv"
    CONFIG["ruta_figura"] = f"figuras/deriva_retrieval_v3{sufijo}.png"
    CONFIG["ruta_figura_contaminacion"] = (
        f"figuras/contaminacion_transitorios_v3{sufijo}.png"
    )
    return incluir


def aplicar_modo_rerank(usar_rerank: bool, limite: int | None = None) -> None:
    """Ajusta CONFIG segun el flag --rerank (Fase T / iter2 / iter3).

    Con rerank activo las estrategias bm25_rerank (v4),
    bm25_rerank_chunks (v5, correccion del truncamiento) y las NUEVAS
    top-50 de la iter3 (bm25_rerank_chunks_top50,
    hibrido_rerank_chunks_top50) entran al benchmark y las salidas
    pasan a los archivos v6 (version "v6-ciclo2-faseT-iter3", campo
    reranker y nota del diagnostico iter3); con --no-rerank todas se
    retiran y el benchmark es identico al v3_limpio en metricas, pero
    las salidas se redirigen a /tmp/opencode con sufijo _regresion para
    NO sobrescribir los archivos _limpio (ADR-11: v2/v3/v3_limpio
    NUNCA se sobrescriben). Con --limite N (smoke test) las salidas se
    redirigen a /tmp/opencode/ para NO tocar resultados/.

    Args:
        usar_rerank: True si el flag --rerank esta activo.
        limite: Valor de --limite (None en el run final).
    """
    if usar_rerank:
        CONFIG["version_salida"] = CONFIG["version_salida_v6"]
        CONFIG["ruta_benchmark"] = CONFIG["ruta_benchmark_v6"]
        CONFIG["ruta_por_ano"] = CONFIG["ruta_por_ano_v6"]
        CONFIG["ruta_figura"] = CONFIG["ruta_figura_v6"]
        CONFIG["ruta_figura_contaminacion"] = CONFIG["ruta_figura_contaminacion_v6"]
    else:
        CONFIG["estrategias"] = [
            e
            for e in CONFIG["estrategias"]
            if e
            not in (
                "bm25_rerank",
                "bm25_rerank_chunks",
                "bm25_rerank_chunks_top50",
                "hibrido_rerank_chunks_top50",
            )
        ]
        if limite is None:
            destino = Path("/tmp/opencode")
            for clave in (
                "ruta_benchmark",
                "ruta_por_ano",
                "ruta_figura",
                "ruta_figura_contaminacion",
            ):
                original = Path(CONFIG[clave])
                CONFIG[clave] = str(
                    destino / f"{original.stem}_regresion{original.suffix}"
                )
    if limite is not None:
        destino = Path("/tmp/opencode")
        for clave in (
            "ruta_benchmark",
            "ruta_por_ano",
            "ruta_figura",
            "ruta_figura_contaminacion",
        ):
            original = Path(CONFIG[clave])
            CONFIG[clave] = str(
                destino / f"{original.stem}_smoke{original.suffix}"
            )


def aplicar_modo_hubs(alpha: float, calibrar: bool, limite: int | None) -> None:
    """Ajusta CONFIG para la Fase W-a iter2 (prior de hubs de la CP).

    Activa el modo hubs (alpha > 0 o --calibrar-hubs) y redirige las
    salidas a los archivos v8 (ADR-11: v2/v3/v3_limpio/v4/v5/v6/v7 NUNCA
    se sobrescriben; el v7 se genero con la rejilla corta, esta iter2 usa
    la extendida hasta 2.0). Con --limite (smoke) las rutas se redirigen
    a /tmp/opencode con sufijo _smoke (incluida la figura de calibracion
    hubs_alpha_v8) para no tocar los artefactos finales. Con alpha=0 y
    sin calibracion el script conserva el comportamiento y las salidas
    previas intactos.

    Args:
        alpha: Valor de --hubs-alpha (0.0 = sin prior).
        calibrar: True si --calibrar-hubs esta activo.
        limite: Valor de --limite (None en el run final).
    """
    CONFIG["modo_hubs_activo"] = bool(alpha > 0.0 or calibrar)
    if not CONFIG["modo_hubs_activo"]:
        return
    CONFIG["version_salida"] = CONFIG["version_salida_v8"]
    CONFIG["ruta_benchmark"] = CONFIG["ruta_benchmark_v8"]
    CONFIG["ruta_por_ano"] = CONFIG["ruta_por_ano_v8"]
    CONFIG["ruta_figura"] = CONFIG["ruta_figura_v8"]
    CONFIG["ruta_figura_contaminacion"] = CONFIG["ruta_figura_contaminacion_v8"]
    if limite is None:
        return
    destino = Path("/tmp/opencode")
    for clave in (
        "ruta_benchmark",
        "ruta_por_ano",
        "ruta_figura",
        "ruta_figura_contaminacion",
        "ruta_figura_hubs_v8",
    ):
        original = Path(CONFIG[clave])
        CONFIG[clave] = str(
            destino / f"{original.stem}_smoke{original.suffix}"
        )


def main() -> None:
    """Orquesta el benchmark M7 v3 (Fase T iter3 + Fase W-a iter2)."""
    inicio_total = time.time()
    args = parsear_args()
    incluir_transitorios = aplicar_modo_indice(args.indice)
    aplicar_modo_rerank(args.rerank, args.limite)
    aplicar_modo_hubs(args.hubs_alpha, args.calibrar_hubs, args.limite)
    resolver_rutas()
    if args.generar_prior:
        generar_prior_hubs()
        return
    configurar_semillas(CONFIG["semilla"])
    dispositivo = elegir_dispositivo()
    print(
        f"Modo indice: {args.indice} (incluir_transitorios="
        f"{incluir_transitorios}) | version salida: "
        f"{CONFIG['version_salida']} | indice: {CONFIG['version_indice']} | "
        f"rerank: {args.rerank} | hubs_alpha: {args.hubs_alpha} | "
        f"calibrar_hubs: {args.calibrar_hubs}"
    )
    print(
        f"GPU disponible: {torch.cuda.is_available()} | "
        f"device E5: {dispositivo}"
    )

    # --- ciclo v2 - Fase W-a: prior de hubs (se carga o construye una vez) ---
    prior_hubs = None
    if CONFIG["modo_hubs_activo"]:
        prior_hubs = cargar_o_construir_prior_hubs()
        print(
            "Top-10 hubs del prior: "
            + ", ".join(
                f"{canon} ({valor:.3f})"
                for canon, valor in top_hubs(prior_hubs, 10)
            )
        )

    # --- pasajes de la Constitucion (mejora 1; Fase S1: modo del indice) ---
    pasajes, canones, grupos_indices, canon_valido = (
        preparar_pasajes_constitucion(incluir_transitorios)
    )
    pasajes_por_articulo = [len(g) for g in grupos_indices]
    print(
        f"Constitucion: {len(canones)} articulos indexados -> "
        f"{len(pasajes)} pasajes (media {np.mean(pasajes_por_articulo):.1f}, "
        f"max {max(pasajes_por_articulo)}) | canon_valido (gold): "
        f"{len(canon_valido)} articulos del parquet completo"
    )

    # --- marco del corpus y weak labels (solo columnas ligeras) ---
    marco = pd.read_parquet(
        CONFIG["ruta_corpus"], columns=["id", "tipo", "ano"]
    )
    marco["id"] = marco["id"].astype(str)
    marco["tipo"] = marco["tipo"].astype(str)
    marco["ano"] = marco["ano"].astype(int)
    entidades = pd.read_parquet(
        CONFIG["ruta_entidades"], columns=["id", "arts_cp"]
    )
    entidades["id"] = entidades["id"].astype(str)
    etiquetas_por_id = dict(zip(entidades["id"], entidades["arts_cp"], strict=True))

    def arts_validos(sid: str) -> list[str]:
        crudo = etiquetas_por_id.get(sid)
        lista = json.loads(crudo) if crudo else []
        return [a for a in lista if a in canon_valido]

    marco["labeled"] = marco["id"].apply(lambda sid: len(arts_validos(sid)) > 0)

    # --- muestra temporal 3/anio estratificada (mejora 5) ---
    muestra = cargar_muestra_temporal(marco)
    ids_muestra = muestra["id"].tolist()
    textos_muestra = leer_filas_por_id(
        CONFIG["ruta_corpus"], ["id", "texto", "texto_limpio"], ids_muestra
    )
    textos_muestra["id"] = textos_muestra["id"].astype(str)
    crudo_por_id = dict(
        zip(textos_muestra["id"], textos_muestra["texto"], strict=True)
    )
    limpio_por_id = dict(
        zip(textos_muestra["id"], textos_muestra["texto_limpio"], strict=True)
    )
    casos = []
    for _, fila in muestra.iterrows():
        sid = str(fila["id"])
        casos.append(
            {
                "id": sid,
                "tipo": str(fila["tipo"]),
                "ano": int(fila["ano"]),
                "texto": str(crudo_por_id.get(sid, "")),
                "texto_limpio": str(limpio_por_id.get(sid, "")),
                "arts": arts_validos(sid),
            }
        )
    print(
        f"Muestra temporal: {len(casos)} sentencias "
        f"({CONFIG['muestra_por_ano']}/anio, semilla {CONFIG['semilla']}), "
        f"tipos {muestra['tipo'].value_counts().to_dict()}"
    )

    # --- ciclo v2 - Fase T: recorte de muestra SOLO para smoke tests ---
    if args.limite is not None:
        casos = casos[: args.limite]
        print(
            f"[smoke] --limite {args.limite}: benchmark recortado a "
            f"{len(casos)} casos (NO es el run final)"
        )

    # --- chunks de query sobre el texto completo (mejora 2) ---
    for caso in casos:
        caso["ventana"] = caso["texto_limpio"][: CONFIG["ventana_query_chars"]]
        caso["chunks_bm25"] = chunks_query(caso["ventana"])
        caso["rankings"] = {}
    chars_ventana = [len(c["ventana"]) for c in casos]
    print(
        f"Query: media {np.mean(chars_ventana):.0f} chars de texto completo, "
        f"media {np.mean([len(c['chunks_bm25']) for c in casos]):.1f} chunks BM25"
    )

    # --- estrategia (a): cita-exacta con regex M2 (referencia circular) ---
    modulo_m2 = importar_extraccion_m2()
    for caso in casos:
        caso["rankings"]["cita_exacta"] = modulo_m2.extraer_arts_cp(
            caso["texto"], CONFIG["ventana_cp"]
        )

    # --- estrategia (b): BM25 por pasajes con query en chunks (mejoras 1-3) ---
    inicio_bm25 = time.time()
    indice_bm25 = construir_indice_bm25(pasajes)
    for caso in casos:
        scores, matriz = scores_bm25_caso(
            indice_bm25, grupos_indices, canones, caso["chunks_bm25"]
        )
        caso["scores_bm25"] = scores
        caso["matriz_bm25"] = matriz
        caso["rankings"]["bm25"] = ranking_desde_scores(scores, canones)
    print(f"BM25 por pasajes: {time.time() - inicio_bm25:.1f} s")

    # --- estrategia (c): E5-multilingual por pasajes en GPU (mejora 4) ---
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models as st_models

    backbone = st_models.Transformer(
        CONFIG["modelo_e5"], max_seq_length=CONFIG["tamano_chunk"]
    )
    pooling = st_models.Pooling(
        backbone.get_embedding_dimension(), pooling_mode="mean"
    )
    modelo_e5 = SentenceTransformer(modules=[backbone, pooling], device=dispositivo)
    tokenizer = modelo_e5.tokenizer

    # Pasajes: chunk-mean por pasaje si excede el limite de tokens.
    grupos_pasajes = [
        chunkear_texto(t, tokenizer) or [""] for t in pasajes["texto"].tolist()
    ]
    longitudes_pasajes = [len(g) for g in grupos_pasajes]
    textos_pasajes = [
        CONFIG["prefijo_pasaje"] + chunk
        for grupo in grupos_pasajes
        for chunk in grupo
    ]
    inicio_e5 = time.time()
    emb_pasajes = _codificar_con_fallback(modelo_e5, textos_pasajes)
    matriz_pasajes = media_por_grupos(emb_pasajes, longitudes_pasajes)
    print(
        f"E5: {len(textos_pasajes)} chunks de {len(pasajes)} pasajes "
        f"codificados en {time.time() - inicio_e5:.1f} s ({dispositivo})"
    )

    # Queries: chunks del texto completo; el score usa max-pooling por
    # chunk (scores_e5_caso), no chunk-mean.
    grupos_query = [chunkear_texto(c["ventana"], tokenizer) or [""] for c in casos]
    longitudes_query = [len(g) for g in grupos_query]
    textos_query = [
        CONFIG["prefijo_query"] + chunk
        for grupo in grupos_query
        for chunk in grupo
    ]
    inicio_query = time.time()
    emb_query = _codificar_con_fallback(modelo_e5, textos_query)
    print(
        f"E5: {len(textos_query)} chunks de query codificados en "
        f"{time.time() - inicio_query:.1f} s "
        f"(media {np.mean(longitudes_query):.1f} chunks/sentencia)"
    )
    inicio = 0
    for caso, n_chunks in zip(casos, longitudes_query, strict=True):
        vectores_query = emb_query[inicio : inicio + n_chunks]
        inicio += n_chunks
        scores = scores_e5_caso(vectores_query, matriz_pasajes, grupos_indices)
        caso["scores_e5"] = scores
        caso["rankings"]["e5"] = ranking_desde_scores(scores, canones)

    # --- estrategia (d): hibrido pre-registrado 0.5/0.5 (mejora 6) ---
    for caso in casos:
        scores = scores_hibridos(caso["scores_bm25"], caso["scores_e5"])
        caso["scores_hibrido"] = scores
        caso["rankings"]["hibrido"] = ranking_desde_scores(scores, canones)

    # --- estrategias (e)-(h): reranking dos etapas (Fase T / iter2 / iter3) ---
    max_tokens_pares: list[int] = []
    max_tokens_pares_top50: list[int] = []
    if args.rerank:
        from lib.reranker_bge import RerankerBGE

        reranker = RerankerBGE()
        inicio_rerank = time.time()
        for caso in casos:
            # (e) bm25_rerank (v4): query completa; se conserva para
            # comparacion directa y reproducibilidad del fallo.
            ranking, scores_rerank = ranking_bm25_rerank(
                reranker,
                caso["scores_bm25"],
                canones,
                pasajes,
                grupos_indices,
                caso["ventana"],
            )
            caso["scores_bm25_rerank"] = scores_rerank
            caso["rankings"]["bm25_rerank"] = ranking
            # (f) bm25_rerank_chunks (v5, iter2): top-3 chunks contra el
            # mejor pasaje del candidato; max-pooling de los pares.
            # CONSERVADA intacta (no-regresion contra el smoke v5).
            ranking, scores_rerank, max_tokens = ranking_bm25_rerank_chunks(
                reranker,
                caso["scores_bm25"],
                caso["matriz_bm25"],
                caso["chunks_bm25"],
                canones,
                pasajes,
                grupos_indices,
            )
            caso["scores_bm25_rerank_chunks"] = scores_rerank
            caso["rankings"]["bm25_rerank_chunks"] = ranking
            max_tokens_pares.append(max_tokens)
            # (g) bm25_rerank_chunks_top50 (iter3): candidatos = top-50
            # BM25; pares identicos a v5 (escala lineal 50x3 = 150).
            ranking, scores_rerank, max_tokens = ranking_rerank_chunks(
                reranker,
                caso["scores_bm25"],
                caso["matriz_bm25"],
                caso["chunks_bm25"],
                canones,
                pasajes,
                grupos_indices,
                CONFIG["k_candidatos_rerank_top50"],
            )
            caso["scores_bm25_rerank_chunks_top50"] = scores_rerank
            caso["rankings"]["bm25_rerank_chunks_top50"] = ranking
            max_tokens_pares_top50.append(max_tokens)
            # (h) hibrido_rerank_chunks_top50 (iter3): candidatos =
            # top-50 del hibrido pre-registrado 0.5/0.5 (recall denso);
            # los pares se derivan de la matriz BM25 chunk x pasaje
            # igual que en v5 (el hibrido solo cambia QUIENES son los
            # candidatos, no como se puntuan los pares).
            ranking, scores_rerank, max_tokens = ranking_rerank_chunks(
                reranker,
                caso["scores_hibrido"],
                caso["matriz_bm25"],
                caso["chunks_bm25"],
                canones,
                pasajes,
                grupos_indices,
                CONFIG["k_candidatos_rerank_top50"],
            )
            caso["scores_hibrido_rerank_chunks_top50"] = scores_rerank
            caso["rankings"]["hibrido_rerank_chunks_top50"] = ranking
            max_tokens_pares_top50.append(max_tokens)
        print(
            f"Rerank bge-reranker-v2-m3: {len(casos)} casos x 4 estrategias "
            f"(top-20 x2 + top-50 x2) en "
            f"{time.time() - inicio_rerank:.1f} s ({reranker.dispositivo}) | "
            f"max tokens par (chunk,pasaje) sin truncar: "
            f"top-20 {max(max_tokens_pares)} | top-50 "
            f"{max(max_tokens_pares_top50)}"
        )

    # --- evaluacion base (alpha=0) de referencia (solo en modo hubs) ---
    globales_alpha0 = None
    if CONFIG["modo_hubs_activo"]:
        por_caso_alpha0 = metricas_por_caso(casos)
        globales_alpha0 = {
            estrategia: promediar_metricas(
                por_caso_alpha0[
                    por_caso_alpha0["estrategia"] == estrategia
                ].to_dict("records")
            )
            for estrategia in CONFIG["estrategias"]
        }

    # --- ciclo v2 - Fase W-a: calibracion del prior de hubs ---
    inicio_calibracion = time.time()
    calibracion_hubs = None
    if args.calibrar_hubs:
        calibracion_hubs = calibrar_hubs(casos, prior_hubs, canones)
        print(
            f"Calibracion hubs (rejilla "
            f"{CONFIG['hubs_alphas_calibracion']}): "
            f"{time.time() - inicio_calibracion:.1f} s "
            "sin re-ejecutar modelos | mejor estrategia figura: "
            f"{calibracion_hubs['mejor_estrategia_figura']}"
        )
        for estrategia, detalle in calibracion_hubs["por_estrategia"].items():
            print(
                f"  {estrategia:32s} {detalle['recomendacion']:>10s} | "
                f"{detalle['evidencia']}"
            )

    # --- aplicacion del alpha efectivo (si > 0) y metricas finales ---
    if args.hubs_alpha > 0.0:
        aplicar_prior_a_casos(casos, prior_hubs, canones, args.hubs_alpha)
        print(
            f"Prior hubs aplicado con alpha={args.hubs_alpha} a "
            f"{[e for e in CONFIG['estrategias'] if e in ESTRATEGIAS_RANKEABLES]}"
        )

    # --- evaluacion global, por anio y por periodo ---
    por_caso = metricas_por_caso(casos)
    globales = {
        estrategia: promediar_metricas(
            por_caso[por_caso["estrategia"] == estrategia].to_dict("records")
        )
        for estrategia in CONFIG["estrategias"]
    }
    por_periodo = agregar_por_periodo(por_caso)

    # --- Fase R: top-10 por caso y diagnostico de contaminacion ---
    top10 = top10_por_caso(casos, canones)
    diagnostico = diagnosticar_contaminacion(casos)

    # recall de extraccion de la cita-exacta (circular, se reporta aparte)
    recalls = []
    for caso in casos:
        predichos = set(caso["rankings"]["cita_exacta"])
        etiquetas = set(caso["arts"])
        recalls.append(len(predichos & etiquetas) / len(etiquetas) if etiquetas else 0.0)
    recall_extraccion = float(np.mean(recalls))

    # --- guardado de resultados ---
    ganadora = max(globales, key=lambda e: globales[e]["f1_at_k"])
    benchmark = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "version": CONFIG["version_salida"],
        "indice": CONFIG["version_indice"],
        "reranker": CONFIG["reranker"] if args.rerank else None,
        "nota": (
            CONFIG["nota_v8"]
            if CONFIG["modo_hubs_activo"]
            else (CONFIG["nota_v6"] if args.rerank else None)
        ),
        "smoke_test": args.limite is not None,
        "descripcion": (
            "Benchmark M7 (ciclo v2, Fase T iter3: ampliacion del recall "
            "de la 1a etapa del reranking con cross-encoder "
            "bge-reranker-v2-m3 sobre el indice limpio de permanentes de "
            "la Fase S1) vinculo sentencia->articulos CP sobre weak "
            "labels M2 (ADR-5): "
            f"{CONFIG['muestra_por_ano']} sentencias/anio 1992-2021, "
            f"semilla {CONFIG['semilla']}, cupos proporcionales al tipo. "
            "Misma muestra, pasajes y estrategias que v3_limpio "
            "(comparabilidad); se reportan las 4 estrategias de rerank: "
            "bm25_rerank (v4, query completa truncada a 512 tokens; "
            "conservada para reproducibilidad del fallo), "
            "bm25_rerank_chunks (v5, top-20 BM25 + top-3 chunks con "
            "max-pooling) y las NUEVAS iter3 con candidatos top-50: "
            "bm25_rerank_chunks_top50 (top-50 BM25) e "
            "hibrido_rerank_chunks_top50 (top-50 del hibrido 0.5/0.5), "
            "para atacar el techo de recall@20 de la 1a etapa (caso "
            "6251: arts 21/209 en posiciones 220/305 de BM25). El "
            "diagnostico de contaminacion se mantiene: con indice "
            "permanente se ESPERA 0 transitorios en top-k (ese es el "
            "resultado de la limpieza, se documenta tal cual sale)."
        ),
        "config": {
            "k": CONFIG["k"],
            "modelo_e5": CONFIG["modelo_e5"],
            "chunk_e5": CONFIG["tamano_chunk"],
            "solape_e5": CONFIG["solape_chunk"],
            "batch_embedding": CONFIG["batch_embedding"],
            "dispositivo_e5": dispositivo,
            "ventana_query_chars": CONFIG["ventana_query_chars"],
            "chunk_bm25_chars": CONFIG["chunk_bm25_chars"],
            "solape_bm25_chars": CONFIG["solape_bm25_chars"],
            "pasaje_max_palabras": CONFIG["pasaje_max_palabras"],
            "agregacion_pasajes": CONFIG["agregacion_pasajes"],
            "agregacion_chunks_bm25": CONFIG["agregacion_chunks_bm25"],
            "agregacion_chunks_e5": CONFIG["agregacion_chunks_e5"],
            "muestra_por_ano": CONFIG["muestra_por_ano"],
            "pesos_hibrido": CONFIG["pesos_hibrido"],
            "stopwords_juridicas": sorted(CONFIG["stopwords_juridicas"]),
            "bigramas_llr": sorted(" ".join(par) for par in CONFIG["bigramas_llr"]),
            "nota_cita_exacta": (
                "La estrategia cita-exacta es circular por construccion "
                "(mismo regex M2 que genero los weak labels): su recall de "
                "extraccion se reporta por separado."
            ),
            "k_ndcg": CONFIG["k_ndcg"],
            "k_top_guardado": CONFIG["k_top_guardado"],
            "k_diagnostico": list(CONFIG["k_diagnostico"]),
            "modo_indice": args.indice,
            "incluir_transitorios": incluir_transitorios,
            "rerank": args.rerank,
            "k_candidatos_rerank": CONFIG["k_candidatos_rerank"],
            "k_rerank_salida": CONFIG["k_rerank_salida"],
            "k_chunks_rerank": CONFIG["k_chunks_rerank"],
            "k_candidatos_rerank_top50": CONFIG["k_candidatos_rerank_top50"],
            "limite_casos": args.limite,
            "nota_gold": (
                "El gold (arts_cp de M2) se valida contra el parquet "
                "completo (497 articulos): los transitorios citados "
                "legitimamente siguen contando como relevantes aunque el "
                "indice no los rankee (verificado: 0 transitorios en "
                "23.742 sentencias)."
            ),
        },
        "pasajes_stats": {
            "n_articulos": len(canones),
            "n_pasajes": len(pasajes),
            "media_por_articulo": float(np.mean(pasajes_por_articulo)),
            "max_por_articulo": int(max(pasajes_por_articulo)),
        },
        "query_stats": {
            "cobertura": "texto completo (ventana infinita; refutada la ventana corta)",
            "media_chars_query": float(np.mean(chars_ventana)),
            "media_chunks_bm25": float(np.mean([len(c["chunks_bm25"]) for c in casos])),
            "media_chunks_e5": float(np.mean(longitudes_query)),
        },
        "rerank_chunks_stats": {
            "nota": (
                "Verificacion iter2/iter3: longitud en tokens SIN "
                "truncar de los pares (chunk, pasaje) puntuados por las "
                "estrategias de rerank por chunks; debe ser <= 512 para "
                "que el cross-encoder vea el par completo (causa del "
                "fallo v4)."
            ),
            "max_tokens_par": int(max(max_tokens_pares)) if max_tokens_pares else None,
            "media_tokens_par": (
                round(float(np.mean(max_tokens_pares)), 1)
                if max_tokens_pares
                else None
            ),
            "pares_por_caso": (
                CONFIG["k_candidatos_rerank"] * CONFIG["k_chunks_rerank"]
            ),
            "top50": {
                "nota": (
                    "Estrategias iter3 (bm25_rerank_chunks_top50 e "
                    "hibrido_rerank_chunks_top50): 50 candidatos x 3 "
                    "chunks = 150 pares/caso."
                ),
                "max_tokens_par": (
                    int(max(max_tokens_pares_top50))
                    if max_tokens_pares_top50
                    else None
                ),
                "media_tokens_par": (
                    round(float(np.mean(max_tokens_pares_top50)), 1)
                    if max_tokens_pares_top50
                    else None
                ),
                "pares_por_caso": (
                    CONFIG["k_candidatos_rerank_top50"]
                    * CONFIG["k_chunks_rerank"]
                ),
            },
        },
        "muestra": [
            {"id": c["id"], "tipo": c["tipo"], "ano": c["ano"], "arts_cp": c["arts"]}
            for c in casos
        ],
        "recall_extraccion_cita_exacta": recall_extraccion,
        "global": globales,
        "por_periodo": por_periodo,
        "top10_por_caso": top10,
        "diagnostico_contaminacion": {
            estrategia: {
                clave: valor
                for clave, valor in diagnostico[estrategia].items()
                if clave != "por_caso"
            }
            for estrategia in ESTRATEGIAS_DIAGNOSTICO
        },
        "estrategia_ganadora_f1_at_k": ganadora,
        "comparacion_v1_global": cargar_globales_v1(),
        "tiempos_segundos": {
            "bm25": round(time.time() - inicio_bm25, 1),
            "e5_pasajes": round(time.time() - inicio_e5 - (time.time() - inicio_query), 1),
            "e5_query": round(time.time() - inicio_query, 1),
            "rerank": (
                round(time.time() - inicio_rerank, 1) if args.rerank else None
            ),
            "calibracion_hubs": (
                round(time.time() - inicio_calibracion, 1)
                if args.calibrar_hubs
                else None
            ),
            "total": round(time.time() - inicio_total, 1),
        },
    }
    # ciclo v2 - Fase W-a: evidencia del prior y de la calibracion A/B.
    if CONFIG["modo_hubs_activo"]:
        benchmark["descripcion"] += (
            " Fase W-a: se anade el prior estatico de hubs de la CP "
            "(frecuencia global de citacion normalizada) como senal "
            "aditiva de ranking con peso alpha; el detalle y la "
            "calibracion A/B (rejilla extendida hasta 2.0 en la iter2, "
            "con curva_alpha_full por estrategia focal) estan en las "
            "secciones hubs_prior, global_alpha0 y calibracion_hubs."
        )
        benchmark["config"].update(
            {
                "hubs_alpha": args.hubs_alpha,
                "calibrar_hubs": args.calibrar_hubs,
                "estrategias_rankeables": [
                    e
                    for e in CONFIG["estrategias"]
                    if e in ESTRATEGIAS_RANKEABLES
                ],
                "estrategias_focales_hubs": list(
                    CONFIG["estrategias_focales_hubs"]
                ),
                "prior_hubs": CONFIG["ruta_prior_hubs"],
                "grid_hubs_alpha": list(CONFIG["hubs_alphas_calibracion"]),
                "grid_hubs_alpha_v7_original": list(
                    CONFIG["hubs_alphas_calibracion_v7"]
                ),
            }
        )
        benchmark["hubs_prior"] = {
            "ruta": CONFIG["ruta_prior_hubs"],
            "senal": (
                "frecuencia global de citacion por articulo CP sobre las "
                "23.742 sentencias de M2, normalizada a [0,1] (estilo IDF)"
            ),
            "no_leakage": (
                "prior global de corpus, identico para toda consulta; no "
                "usa informacion por caso del conjunto de evaluacion"
            ),
            "comparacion_pagerank": (
                "PageRank ponderado del grafo de co-citacion (aristas = "
                "pares co-citados en una sentencia): Spearman 0.9828 con "
                "la frecuencia y mismo top-5 (241/86/13/2/1); se adopta "
                "la frecuencia por ser la senal mas simple"
            ),
            "n_articulos": len(prior_hubs),
            "top10": [
                {"id": canon, "valor": valor}
                for canon, valor in top_hubs(prior_hubs, 10)
            ],
        }
        benchmark["hubs_alpha"] = args.hubs_alpha
        benchmark["global_alpha0"] = globales_alpha0
    if calibracion_hubs is not None:
        benchmark["calibracion_hubs"] = calibracion_hubs
    guardar_json(CONFIG["ruta_benchmark"], benchmark)

    por_caso[["id", "ano", "estrategia", "n_relevantes", *MEDIDAS]].to_csv(
        CONFIG["ruta_por_ano"], index=False, encoding="utf-8"
    )
    graficar_deriva(por_caso)
    graficar_contaminacion(diagnostico)

    # --- resumen final ---
    print(
        f"\n=== Resumen M7 v3 (global, k=5, indice {args.indice}) ==="
    )
    for estrategia in CONFIG["estrategias"]:
        m = globales[estrategia]
        print(
            f"{estrategia:30s} P@5={m['p_at_k']:.3f} R@5={m['r_at_k']:.3f} "
            f"F1@5={m['f1_at_k']:.3f} MRR={m['mrr']:.3f} "
            f"nDCG@10={m['ndcg_at_10']:.3f}"
        )
    print(f"recall extraccion (cita-exacta, circular): {recall_extraccion:.3f}")
    for nombre, _, _ in CONFIG["periodos"]:
        fila = por_periodo[nombre]
        extra_rerank = (
            f" | rerank={fila['bm25_rerank']['f1_at_k']:.3f}"
            if "bm25_rerank" in fila
            else ""
        )
        extra_rerank_chunks = (
            f" | rerank_chunks={fila['bm25_rerank_chunks']['f1_at_k']:.3f}"
            if "bm25_rerank_chunks" in fila
            else ""
        )
        extra_top50 = (
            f" | bm25_top50={fila['bm25_rerank_chunks_top50']['f1_at_k']:.3f}"
            if "bm25_rerank_chunks_top50" in fila
            else ""
        )
        extra_hibrido_top50 = (
            f" | hibrido_top50="
            f"{fila['hibrido_rerank_chunks_top50']['f1_at_k']:.3f}"
            if "hibrido_rerank_chunks_top50" in fila
            else ""
        )
        print(
            f"periodo {nombre}: F1@5 bm25={fila['bm25']['f1_at_k']:.3f} | "
            f"e5={fila['e5']['f1_at_k']:.3f} | "
            f"hibrido={fila['hibrido']['f1_at_k']:.3f}"
            f"{extra_rerank}{extra_rerank_chunks}{extra_top50}"
            f"{extra_hibrido_top50}"
        )
    print(f"ganadora por F1@5: {ganadora}")
    diag_resumen = diagnostico["bm25"]
    print(
        "contaminacion (bm25): "
        f"transitorios top-5 {diag_resumen['pct_transitorios_top5'] * 100:.1f}% | "
        f"top-10 {diag_resumen['pct_transitorios_top10'] * 100:.1f}% | "
        f"casos con transitorio en top-5 "
        f"{diag_resumen['casos_con_transitorio_top5']}/{len(casos)} | "
        f"IDs malformados en rankings "
        f"{diag_resumen['ids_malformados_en_rankings']}"
    )
    if not incluir_transitorios and diag_resumen["pct_transitorios_top5"] == 0.0:
        print(
            "[ok] 0 transitorios en top-k con indice permanente: es el "
            "resultado ESPERADO de la Fase S1 (se documenta tal cual)"
        )
    if CONFIG["modo_hubs_activo"]:
        print(
            "prior hubs: alpha="
            f"{args.hubs_alpha} | top-5: "
            f"{[canon for canon, _ in top_hubs(prior_hubs, 5)]} | figura "
            f"calibracion: {CONFIG['ruta_figura_hubs_v8']}"
        )
        if calibracion_hubs is not None:
            for estrategia in CONFIG["estrategias_focales_hubs"]:
                detalle = calibracion_hubs["por_estrategia"].get(estrategia)
                if detalle is None:
                    continue
                print(
                    f"calibracion {estrategia}: "
                    f"{detalle['recomendacion']} (alpha_full="
                    f"{detalle['alpha_elegido_full']}, mejora full "
                    f"{detalle['mejora_full_pp']:+.1f} pp)"
                )
    print(
        f"tiempo total: {time.time() - inicio_total:.1f} s | "
        f"salidas: {CONFIG['ruta_benchmark']}, {CONFIG['ruta_por_ano']}, "
        f"{CONFIG['ruta_figura']}, {CONFIG['ruta_figura_contaminacion']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"Error fatal: {exc}")
        sys.exit(1)
