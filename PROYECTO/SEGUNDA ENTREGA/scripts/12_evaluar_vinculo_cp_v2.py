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

    ciclo v3 - Sprint A1 (2026-09-11, plan/PLAN_v3_recuperacion_robusta.md):
    "metricas honestas" SIN re-ejecutar modelos. Flag NUEVO --diagnostico
    (default OFF: sin el flag el comportamiento y las salidas actuales
    quedan identicos): reutiliza los scores base ya calculados y produce
    (1) curvas recall@k (CONFIG['k_curvas_recall']) por estrategia focal,
    (2) IC 95% bootstrap (1000 remuestreos de casos con reemplazo, semilla
    42) de nDCG@10/MRR/F1@5/F1@10, (3) desglose por periodo y por terciles
    de longitud del caso, (4) recall de los candidatos top-50 de la
    primera etapa (techo del reranker) y (5) el techo teorico del gold
    (F1@5/F1@10 maximos dado el numero de articulos gold por caso). Se
    comparan dos contextos de scoring: sin hubs (alpha=0) y con los alphas
    adoptados en el run v8 (bm25/hibrido 0.75, rerank top-50 1.5). Las
    variantes de rerank quedan topadas por su set de candidatos para
    k>50: se documenta como recall de candidatos. Salidas NUEVAS
    (ADR-16, nada previo se sobrescribe):
    resultados/c3_diagnostico_recall.json y figuras/c3_recall_curvas.png,
    c3_metricas_ic.png y c3_por_periodo.png; con --limite N todo va a
    /tmp/opencode y en modo diagnostico las salidas estandar del
    benchmark se redirigen a /tmp para no pisar los v6/v8 canonicos.

    ciclo v3 - Sprint A2 (2026-09-11, plan/PLAN_v3_recuperacion_robusta.md):
    comparativa de EMBEDDINGS legales en espanol + profundidad de
    candidatos del rerank. Adiciones:
    (A2.1) flag --embedding-model HF_ID (repetible; acepta un HF id o un
         slug corto e5/alia/bgem3/modernbert): carga modelos
         sentence-transformers arbitrarios para la primera etapa densa.
         E5 se calcula SIEMPRE (estrategias legacy + referencia del IC
         pareado); --embedding-model anade modelos a la matriz. Sin el
         flag y con --k-candidatos-hybrid 50 el comportamiento y las
         salidas son identicos a A1;
    (A2.2) convenciones de prefijos por modelo, verificadas en el
         config_sentence_transformers.json y en la model card de cada
         repo (2026-09-11): E5 publica "query: "/"passage: "; ALIA-
         MrBERT-legal, BGE-M3-Legal-Spanish y modernbert-embed-es-
         legal-v2 declaran prompts query/document VACIOS (la card usa
         prompt_name="query"/encode_query que mapean al prompt vacio),
         por lo que van SIN prefijo. Todos los embeddings se normalizan
         (similitud coseno);
    (A2.3) cache NUEVA de embeddings de pasajes de la CP por modelo en
         data/cache_emb_cp/<slug>.joblib, con metadata de validacion
         (hf_id, prefijos, chunking, n_pasajes y md5 del orden de
         canones). La cache legacy del producto
         (data/cache_e5_cp/pasajes_e5_base.joblib, Fase X) se MIGRA
         para E5 previa validacion (modelo/chunking/n_pasajes + spot
         check de 3 pasajes contra la matriz cacheada);
    (A2.4) flag --k-candidatos-hybrid K (default 50 = top-50 actual):
         profundidad de candidatos de las estrategias hibridas de la
         matriz (la A2 usa 100; la maquinaria admite 200). El rerank
         escala lineal (K x CONFIG['k_chunks_rerank'] pares/caso).
         Nombres NUEVOS hibrido_<slug>_rerank_chunks_top<K>; la E5 a
         K=50 reutiliza la estrategia legacy
         hibrido_rerank_chunks_top50 (cross-check exacto contra v8, sin
         recomputar);
    (A2.5) la comparacion reporta cada estrategia en DOS contextos del
         prior de hubs (alpha=0 sin hubs para atribuir la mejora al
         embedding, y alpha=1.5 adoptado en v8), guarda las metricas
         POR CASO (nDCG@10 y F1@10) y calcula el IC 95% bootstrap
         PAREADO (mismos remuestreos de casos: 1000, seed 42) de
         delta = estrategia_nueva - E5@K (referencia): media e IC de la
         diferencia, mas prob_mejora y veredicto "gana" (p2.5 > 0);
    (A2.6) salidas NUEVAS v9 (ADR-16: v2..v8 NUNCA se sobrescriben):
         resultados/vinculo_cp_benchmark_v9.json (version
         "v9-ciclo3-sprintA2" con las secciones comparacion_a2 y
         embeddings_a2), resultados/vinculo_cp_por_ano_v9.csv,
         figuras/deriva_retrieval_v9.png,
         figuras/contaminacion_transitorios_v9.png y
         figuras/c3_a2_embeddings.png (nDCG@10 por modelo/contexto con
         IC); con --limite N todo va a /tmp/opencode y, si ademas hay
         --diagnostico, sus salidas tambien (para no pisar el artefacto
         A1 congelado).

    ciclo v3 - Sprint A3 (2026-09-11, plan/PLAN_v3_recuperacion_robusta.md):
    comparativa de RERANKERS legales en espanol sobre los candidatos de
    ALIA@100 (el embedding con mayor recall de candidatos de A2: 0.723 vs
    0.619 de E5). Adiciones:
    (A3.1) flag --a3 (default OFF; requiere --rerank): fija la primera
          etapa densa en ALIA-MrBERT legal y la profundidad en 100
          candidatos (slug/k documentados en CONFIG['a3']), reutilizando
          la maquinaria de embeddings de A2 (carga generica
          SentenceTransformer y cache por slug data/cache_emb_cp/alia
          .joblib) y el hibrido pre-registrado 0.5*bm25 + 0.5*alia;
    (A3.2) tres variantes sobre los MISMOS candidatos: bge
          (hibrido_alia_rerank_chunks_top100, referencia; reutiliza la
          estrategia A2 homonima si la matriz A2 ya la calculo a k=100),
          aliarerank (hibrido_alia_rerank_sinai_top100, cross-encoder
          SINAI/ALIA-MrBERT-es-legal-administrative-reranker de 150M,
          Apache-2.0) y rrf (hibrido_alia_rerank_rrf_top100, Reciprocal
          Rank Fusion k=60 entre los rankings de ambos rerankers:
          score_rrf = sum 1/(60+rank_i), normalizada min-max y con el
          prior de hubs alpha como el resto de estrategias);
    (A3.3) eficiencia: los pares (chunk, pasaje) de cada caso se
          construyen UNA sola vez (misma derivacion de v5/A2: mejor
          pasaje BM25 + top-3 chunks) y se puntuan con ambos rerankers,
          evitando duplicar candidatos; el par se trunca a 512 tokens
          (longest_first) con batch 16 fp16 y OOM-halving;
    (A3.4) la comparacion reporta las 3 variantes en DOS contextos del
          prior de hubs (alpha=0 sin hubs y alpha=1.5 adoptado en v8),
          guarda nDCG@10 y F1@10 POR CASO y calcula el IC 95% bootstrap
          PAREADO (1000 remuestreos, seed 42) de delta = variante - bge
          con prob_mejora y veredicto "gana" (p2.5 > 0);
    (A3.5) salidas NUEVAS v10 (ADR-16: v2..v9 NUNCA se sobrescriben):
          resultados/vinculo_cp_benchmark_v10.json (version
          "v10-ciclo3-sprintA3" con la seccion comparacion_a3),
          resultados/vinculo_cp_por_ano_v10.csv,
          figuras/deriva_retrieval_v10.png,
          figuras/contaminacion_transitorios_v10.png y
          figuras/c3_a3_rerankers.png (nDCG@10 y F1@10 por variante y
          contexto con IC bootstrap); con --limite N todo va a
          /tmp/opencode con sufijo _smoke.

    ciclo v3 - Sprint A4 (2026-09-11, plan/PLAN_v3_recuperacion_robusta.md):
    fusion de PRIMERA ETAPA (BM25 + denso) y barrido de profundidad de
    candidatos. Adiciones:
    (A4.1) flag --a4 (default OFF; requiere --rerank): fija el embedding
          denso en ALIA-MrBERT legal (el de mayor recall de candidatos de
          A2: 0.723) y compara 4 variantes de fusion de 1a etapa a 100
          candidatos, TODAS con bge rerank chunks (misma derivacion de
          pares que v5/A2/A3) y contextos alpha=0/1.5:
          - w05 (referencia): hibrido min-max 0.5*BM25 + 0.5*denso, el
            comportamiento actual; reutiliza la estrategia homonima
            hibrido_alia_rerank_chunks_top100 si A2/A3 ya la calcularon;
          - w03: fusion lineal con peso BM25 0.3 / denso 0.7;
          - w07: fusion lineal con peso BM25 0.7 / denso 0.3;
          - rrf1a: Reciprocal Rank Fusion k=60 a nivel de 1a etapa entre
            los rankings COMPLETOS BM25 y denso ALIA (score = suma de
            1/(60+rank), rank 1-based, empates por argsort estable);
            candidatos = top-100 del score RRF.
          CONVENCION documentada: el nombre w<NN> indica el peso de BM25
          en la fusion lineal min-max por caso (peso denso = 1 - w).
    (A4.2) barrido de profundidad 50/100/200 de la fusion elegida
          (CONFIG['a4']['fusion_barrido']; revision A4 2026-09-11:
          default w05, la referencia pre-registrada, NO w07). El smoke
          n=10 favorecia a w07 (sin_hubs 0.514 vs 0.466; con_hubs 0.587
          vs 0.556) con IC pareado que excluye 0, pero con extremos
          inferiores minimos (+0.004/+0.001), sin mitades A/B evaluables
          a n=10, con el precedente directo de A3 (RRF +0.038 a n=10 ->
          -0.007 a n=90) y con MENOR recall de candidatos @100 (0.695 vs
          0.744): no es un criterio robusto. El barrido se hace sobre
          w05 y el IC pareado de las 4 fusiones a @100 del run de 90
          decide; --a4-fusion {w05,w03,w07,rrf1a} permite barrer otra
          sin tocar codigo. El rerank escala 3 pares/candidato: 150
          pares/caso a top-50, 300 a top-100 y 600 a top-200.
    (A4.3) seccion comparacion_a4 con: metricas globales y por caso
          (nDCG@10, F1@10) por variante y contexto, recall de candidatos
          de 1a etapa por fusion y profundidad (techo del rerank) y el IC
          95% bootstrap PAREADO (1000 remuestreos, seed 42) de delta =
          variante - w05@100 (referencia) con prob_mejora y veredicto
          "gana" (p2.5 > 0), reutilizando bootstrap_ic_pareado.
    (A4.4) salidas NUEVAS v11 (ADR-16: v2..v10 NUNCA se sobrescriben):
          resultados/vinculo_cp_benchmark_v11.json (version
          "v11-ciclo3-sprintA4" con la seccion comparacion_a4),
          resultados/vinculo_cp_por_ano_v11.csv,
          figuras/deriva_retrieval_v11.png,
          figuras/contaminacion_transitorios_v11.png,
          figuras/c3_a4_fusion_profundidad.png (fusiones @100 y barrido
          de profundidad con IC bootstrap) y el derivado trazable
          resultados/c3_a4_fusion_profundidad.json (seccion
          comparacion_a4 + origen con archivo/version/md5, patron de
          c3_a3_rerankers.json). Con --limite N todo va a /tmp/opencode
          con sufijo _smoke.

    ciclo v3 - Sprint B1 (2026-10-03, plan/PLAN_v3_recuperacion_robusta.md):
    evaluacion del RERANKER DE DOMINIO fine-tuneado por
    scripts/24_b1_entrenar_reranker.py (pares chunk->pasaje citado con
    hard negatives BM25/co-citados; el modelo se guarda con
    save_pretrained en modelos/b1_reranker/). Adiciones:
    (B1.1) flag --b1 (default OFF; requiere --rerank): puntua la ruta
          GANADORA del producto con rerank (hibrido_rerank_chunks_top50
          + prior de hubs alpha=1.5 + salida k=10, la misma de A4/v8)
          con el reranker B1 sobre los MISMOS candidatos top-50 del
          hibrido 0.5/0.5 y los MISMOS pares (chunk, pasaje) que la
          referencia bge; la estrategia nueva se llama
          hibrido_rerank_chunks_top50_b1 y entra al marco de metricas
          sin cambiar ningun protocolo;
    (B1.2) el modelo por defecto es la ruta local modelos/b1_reranker
          resuelta contra ROOT; si el directorio no existe el script
          falla ANTES de ejecutar con un error claro que indica
          entrenar primero con scripts/24_b1_entrenar_reranker.py. El
          flag --reranker-model HF_ID lo sobreescribe SOLO para pruebas
          de fontaneria (acepta un alias del catalogo de
          scripts/lib/reranker_bge.py, p. ej. bge, una ruta local o un
          HF id);
    (B1.3) seccion comparacion_b1 con metricas globales y por caso
          (nDCG@10, F1@10) en contextos alpha=0/1.5 y el IC 95%
          bootstrap PAREADO (1000 remuestreos, seed 42) de delta = B1 -
          referencia bge del MISMO run (mismos casos, candidatos y
          pares; el delta primario usa la ruta ganadora con_hubs
          alpha=1.5). Si los artefactos congelados v8/v10 existen, se
          reconstruye de su top10_por_caso la serie por caso de la
          referencia bge en el contexto alpha=0 y se anaden (1) la
          verificacion de equivalencia referencia_run vs externa y (2)
          un bootstrap pareado adicional de B1 vs la referencia
          congelada; si no existen, el JSON documenta el fallback
          (referencia bge del run);
    (B1.4) salidas NUEVAS v12 (ADR-16: v2..v11 NUNCA se sobrescriben):
          resultados/vinculo_cp_benchmark_v12.json (version
          "v12-ciclo3-sprintB1" con la seccion comparacion_b1),
          resultados/vinculo_cp_por_ano_v12.csv,
          figuras/deriva_retrieval_v12.png,
          figuras/contaminacion_transitorios_v12.png y el derivado
          trazable resultados/c3_b1_reranker.json (seccion
          comparacion_b1 + origen con archivo/version/md5, patron de
          c3_a4_fusion_profundidad.json). Con --limite N todo va a
          /tmp/opencode con sufijo _smoke (run de fontaneria).

    ciclo v3 - Sprint B2 (2026-10-04, plan/PLAN_v3_recuperacion_robusta.md):
    evaluacion del RETRIEVER DENSO de dominio fine-tuneado por
    scripts/25_b2_retriever_denso.py (bi-encoder contrastivo entrenado
    con los pares de B1; el mejor modelo se guarda con save_pretrained en
    modelos/b2_retriever/). Adiciones:
    (B2.1) flag --b2 (default OFF; requiere --rerank y es excluyente con
          --b1, --a3, --a4 y con los disparadores de la matriz A2
          --k-candidatos-hybrid / --embedding-model multiple): evalua la
          ruta NUEVA hibrido_b2_rerank_chunks_top50_b1, es decir, el
          hibrido pre-registrado 0.5*bm25 + 0.5*B2 sobre los MISMOS
          top-50 candidatos y pares (chunk, pasaje) que la ruta B1,
          rerank por chunks con el reranker B1 (modelos/b1_reranker) y
          prior de hubs alpha=1.5; contextos sin_hubs (0.0) y con_hubs
          (1.5), salida k=10. El denso se carga con el plumbing de A2
          (SentenceTransformer + cache data/cache_emb_cp/<slug>.joblib;
          slug b2_retriever);
    (B2.2) el modelo denso por defecto es la ruta local
          modelos/b2_retriever resuelta contra ROOT; si el directorio no
          existe el script falla ANTES de ejecutar con un error claro
          que indica entrenar primero con
          scripts/25_b2_retriever_denso.py --entrenar. El flag
          --embedding-model (un solo valor) sobreescribe el denso SOLO
          para pruebas de fontaneria y --reranker-model sobreescribe el
          reranker B1 de la ruta (default modelos/b1_reranker);
    (B2.3) seccion comparacion_b2 con metricas globales y por caso
          (nDCG@10, F1@10) de la estrategia B2 en contextos alpha=0/1.5 y
          el IC 95% bootstrap PAREADO (1000 remuestreos, seed 42) de
          delta = B2 - referencia v12 (la ruta B1 con hibrido E5
          congelada: per-case de comparacion_b1.metricas_por_caso; si
          falta, top10_por_caso en el contexto sin_hubs; si no hay
          fuente, fallback documentado sin pareado primario) y la
          verificacion externa vs A2/ALIA@100 (per-case de v9
          comparacion_a2.metricas_por_caso) cuando esta disponible;
    (B2.4) salidas NUEVAS v13 (ADR-16: v2..v12 NUNCA se sobrescriben):
          resultados/vinculo_cp_benchmark_v13.json (version
          "v13-ciclo3-sprintB2" con la seccion comparacion_b2),
          resultados/vinculo_cp_por_ano_v13.csv,
          figuras/deriva_retrieval_v13.png,
          figuras/contaminacion_transitorios_v13.png y el derivado
          trazable resultados/c3_b2_retriever.json (seccion
          comparacion_b2 + origen con archivo/version/md5, patron de
          c3_b1_reranker.json). Con --limite N todo va a /tmp/opencode
          con sufijo _smoke (run de fontaneria).
    (B2.5) flag --b2-variante SLUG (iteracion v14; requiere --b2):
          evalua una variante del retriever denso (p. ej. la 2a epoca
          ep2 -> modelos/b2_retriever_ep2; fail-fast si no existe, sin
          usar el override --embedding-model) con TODAS las salidas
          NUEVAS v14 nombradas por slug: version v14-ciclo3-b2-<slug>,
          resultados/vinculo_cp_benchmark_v14_<slug>.json,
          resultados/vinculo_cp_por_ano_v14_<slug>.csv,
          figuras/deriva_retrieval_v14_<slug>.png,
          figuras/contaminacion_transitorios_v14_<slug>.png y derivado
          resultados/c3_b2_retriever_<slug>.json (v2..v13 intactas); la
          nota v14 menciona la variante y el modelo, y las referencias
          comparativas v12 y ALIA@100 del v9 se mantienen. Con --limite
          los slugs "smoke*" conservan las rutas v14 (evidencia de
          fontaneria) y el resto se redirige a /tmp/opencode.

    Estado del arte aplicado: BM25 por pasajes con agregacion max, max-pooling
    de chunks de query (cobertura completa), hibrido lexico-denso con
    normalizacion min-max por consulta (pre-registrado 0.5/0.5), E5
    zero-shot para deriva temporal (ADR-3, ADR-6), reranking con
    cross-encoder bge-reranker-v2-m3 (Fase T), prior de hubs de
    co-citacion con rejilla alpha extendida (Fase W-a iter2,
    UA-StatuteRetrieval 2026), protocolo temporal de ADR-5 con P@k,
    R@k, F1@k y MRR, IC 95% bootstrap no parametrico del promedio
    (Efron-Tibshirani; Sprint A1 del ciclo v3), embeddings legales ES de
    dominio (ALIA-MrBERT legal, BGE-M3-Legal-Spanish y modernbert-embed-
    es-legal-v2; Sprint A2 del ciclo v3), reranker legal de dominio ES
    (SINAI ALIA MrBERT cross-encoder), fusion RRF de rankings (Cormack
    et al. 2009, k=60; Sprint A3 del ciclo v3), fusion lineal de scores
    de 1a etapa con pesos BM25/denso y RRF de rangos a nivel de 1a etapa
    con barrido de profundidad de candidatos (Sprint A4 del ciclo v3) y
    bootstrap pareado de la diferencia para el gate de adopcion (ADR-16);
    fine-tune de cross-encoder de dominio con hard negatives BM25/
    co-citados sobre bge-reranker-v2-m3 (Nogueira y Cho 2019; Karpukhin
    et al. 2020; Sprint B1 del ciclo v3) y recuperacion densa contrastiva
    de dominio con bi-encoder fine-tuneado (DPR/InfoNCE con negativos
    in-batch + hard negatives; Karpukhin et al. 2020, Gao et al. 2021;
    Sprint B2 del ciclo v3).
"""

import argparse
import hashlib
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

import joblib
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
    # --- ciclo v3 - Sprint A1: diagnostico honesto (recall@k + IC) ---
    # El modo --diagnostico reutiliza los scores base ya calculados; los
    # alphas adoptados con hubs son los del run v8 (calibracion_hubs
    # por_estrategia.*.alpha_elegido_full) para las 4 estrategias focales.
    "version_diagnostico": "c3-a1",
    "ruta_diagnostico": "resultados/c3_diagnostico_recall.json",
    "ruta_figura_diagnostico_recall": "figuras/c3_recall_curvas.png",
    "ruta_figura_diagnostico_ic": "figuras/c3_metricas_ic.png",
    "ruta_figura_diagnostico_periodo": "figuras/c3_por_periodo.png",
    "k_curvas_recall": [10, 20, 50, 100, 200, 384],
    "diagnostico_bootstrap": 1000,
    "diagnostico_seed": 42,
    "alphas_adoptados_hubs": {
        "bm25": 0.75,
        "hibrido": 0.75,
        "bm25_rerank_chunks_top50": 1.5,
        "hibrido_rerank_chunks_top50": 1.5,
    },
    # --- ciclo v3 - Sprint A2: embeddings legales ES + profundidad ---
    # Catalogo de modelos de la matriz A2 (slug corto -> configuracion).
    # Prefijos VERIFICADOS el 2026-09-11 en el
    # config_sentence_transformers.json y en la model card de cada repo:
    # E5 publica "query: "/"passage: "; ALIA-MrBERT-legal, BGE-M3-Legal-
    # Spanish y modernbert-embed-es-legal-v2 declaran prompts
    # query/document VACIOS (sus cards usan prompt_name="query" /
    # encode_query, que mapean al prompt vacio): van SIN prefijo. Todos
    # los embeddings se normalizan para similitud coseno.
    "modelos_embedding": {
        "e5": {
            "hf_id": "intfloat/multilingual-e5-base",
            "prefijo_query": "query: ",
            "prefijo_pasaje": "passage: ",
            "carga": "e5_custom",
            "nota": (
                "Referencia v2..v8; prefijos oficiales E5; carga custom "
                "backbone + mean pooling (identidad legacy intacta)."
            ),
        },
        "alia": {
            "hf_id": "SINAI/ALIA-MrBERT-es-legal-administrative-embeddings",
            "prefijo_query": "",
            "prefijo_pasaje": "",
            "carga": "sentence_transformers",
            "nota": (
                "ModernBERT 150M, mean pooling; prompts vacios en "
                "config_sentence_transformers.json y card con "
                "prompt_name='query' (prompt vacio): sin prefijo."
            ),
        },
        "bgem3": {
            "hf_id": "wilfredomartel/BGE-M3-Legal-Spanish",
            "prefijo_query": "",
            "prefijo_pasaje": "",
            "carga": "sentence_transformers",
            "nota": (
                "XLM-R/BGE-M3 568M, CLS pooling + Normalize; prompts "
                "None en la card y vacios en config_sentence_transformers"
                ".json: sin prefijo."
            ),
        },
        "modernbert": {
            "hf_id": "wilfredomartel/modernbert-embed-es-legal-v2",
            "prefijo_query": "",
            "prefijo_pasaje": "",
            "carga": "sentence_transformers",
            "nota": (
                "ModernBERT 149M, mean pooling + Normalize; prompts "
                "None en la card y vacios en config_sentence_transformers"
                ".json: sin prefijo."
            ),
        },
    },
    "cache_emb_cp_dir": "data/cache_emb_cp",
    "k_candidatos_hybrid": 50,
    "a2_contextos": {"sin_hubs": 0.0, "con_hubs": 1.5},
    "a2_slug_referencia": "e5",
    "a2_bootstrap": 1000,
    "a2_seed_pareado": 42,
    "version_salida_v9": "v9-ciclo3-sprintA2",
    "ruta_benchmark_v9": "resultados/vinculo_cp_benchmark_v9.json",
    "ruta_por_ano_v9": "resultados/vinculo_cp_por_ano_v9.csv",
    "ruta_figura_v9": "figuras/deriva_retrieval_v9.png",
    "ruta_figura_contaminacion_v9": "figuras/contaminacion_transitorios_v9.png",
    "ruta_figura_a2_embeddings": "figuras/c3_a2_embeddings.png",
    "nota_v9": (
        "Ciclo v3 - Sprint A2: comparativa de embeddings legales en "
        "espanol (E5 referencia, ALIA-MrBERT-legal, BGE-M3-Legal-Spanish "
        "y modernbert-embed-es-legal-v2; slugs e5/alia/bgem3/modernbert) "
        "y profundidad de candidatos del rerank hibrido "
        "(--k-candidatos-hybrid, default 50; la matriz A2 usa 100). Cada "
        "estrategia nueva se reporta en dos contextos del prior de hubs "
        "(alpha=0 sin hubs y alpha=1.5 adoptado en v8) con metricas por "
        "caso (nDCG@10 y F1@10) y bootstrap PAREADO (1000 remuestreos de "
        "casos, seed 42) de la diferencia contra la referencia E5@K. E5 "
        "a top-50 reutiliza hibrido_rerank_chunks_top50 (cross-check "
        "exacto contra v8). Salidas NUEVAS v9; v2..v8 quedan intactas "
        "(ADR-16)."
    ),
    # Estado serializable de la matriz A2 (lo rellena aplicar_modo_a2).
    "a2": {
        "activa": False,
        "modelos": [],
        "estrategias": [],
        "k": 50,
        "referencia": None,
        "tiempos": {},
    },
    # --- ciclo v3 - Sprint A3: comparativa de rerankers legales ES ---
    # Los candidatos quedan FIJOS en ALIA@100 (mayor recall de candidatos
    # de A2: 0.723 vs 0.619 de E5). El embedding ALIA reutiliza la carga
    # generica y la cache por slug de A2; si la matriz A2 ya calculo la
    # estrategia homonima a k=100, el vector bge se reutiliza (no se
    # re-puntua). Las 3 variantes: bge (referencia), aliarerank (SINAI) y
    # rrf (RRF k=60 entre los rankings de ambos rerankers). Contextos de
    # hubs fijos alpha=0 y alpha=1.5 (el adoptado para el rerank top-50
    # en v8), con IC bootstrap pareado de la diferencia vs bge.
    "a3": {
        "activa": False,
        "slug_embedding": "alia",
        "k": 100,
        "modelo_reranker": (
            "SINAI/ALIA-MrBERT-es-legal-administrative-reranker"
        ),
        "rrf_k": 60,
        "contextos": {"sin_hubs": 0.0, "con_hubs": 1.5},
        "estrategias": [],
        "referencia": None,
        "reutiliza_a2": False,
        "clave_bge": None,
    },
    "a3_bootstrap": 1000,
    "a3_seed_pareado": 42,
    "version_salida_v10": "v10-ciclo3-sprintA3",
    "ruta_benchmark_v10": "resultados/vinculo_cp_benchmark_v10.json",
    "ruta_por_ano_v10": "resultados/vinculo_cp_por_ano_v10.csv",
    "ruta_figura_v10": "figuras/deriva_retrieval_v10.png",
    "ruta_figura_contaminacion_v10": "figuras/contaminacion_transitorios_v10.png",
    "ruta_figura_a3_rerankers": "figuras/c3_a3_rerankers.png",
    "nota_v10": (
        "Ciclo v3 - Sprint A3: comparativa de RERANKERS legales en "
        "espanol sobre los candidatos de ALIA@100 (el embedding con mayor "
        "recall de candidatos de A2: 0.723 vs 0.619 de E5). Tres variantes "
        "sobre los MISMOS pares (chunk, pasaje) construidos una sola vez: "
        "bge (referencia, bge-reranker-v2-m3), aliarerank "
        "(SINAI/ALIA-MrBERT-es-legal-administrative-reranker, cross-encoder "
        "legal ES de 150M, Apache-2.0) y rrf (Reciprocal Rank Fusion k=60 "
        "de los rankings de ambos rerankers). Cada variante se reporta en "
        "dos contextos del prior de hubs (alpha=0 y alpha=1.5) con "
        "metricas por caso y bootstrap PAREADO (1000 remuestreos, seed 42) "
        "de la diferencia contra bge; en las estrategias de rerank el "
        "universo es el set de candidatos top-100 del hibrido ALIA "
        "(recall de candidatos documentado). Salidas NUEVAS v10; v2..v9 "
        "quedan intactas (ADR-16)."
    ),
    # --- ciclo v3 - Sprint A4: fusion de 1a etapa + barrido de profundidad ---
    # Embedding denso FIJO en ALIA-MrBERT legal (mayor recall de
    # candidatos de A2: 0.723). Convencion de pesos: el nombre de la
    # variante indica el peso de BM25 en la fusion lineal min-max por
    # caso (w05 = 0.5/0.5, w03 = BM25 0.3 / denso 0.7, w07 = BM25 0.7 /
    # denso 0.3; el denso recibe 1 - peso). rrf1a es RRF k=60 entre los
    # rankings COMPLETOS BM25 y denso ALIA (candidatos = top-100 del
    # RRF). El barrido de profundidad 50/100/200 usa
    # CONFIG['a4']['fusion_barrido'] = w05 (revision A4 2026-09-11): el
    # smoke n=10 favorecia a w07, pero el criterio NO es robusto
    # (extremos inferiores +0.004/+0.001, sin A/B a n=10, precedente A3
    # RRF +0.038 a n=10 -> -0.007 a n=90 y menor recall de candidatos
    # @100: 0.695 vs 0.744), asi que el barrido se ancla en la referencia
    # pre-registrada w05. El JSON v11 reporta las 4 fusiones a @100 y
    # --a4-fusion permite fijar otra {w05,w03,w07,rrf1a} sin tocar
    # codigo. El run final de 90 casos decide con el IC pareado completo.
    "a4": {
        "activa": False,
        "slug_embedding": "alia",
        "k_referencia": 100,
        "profundidades": [50, 100, 200],
        "rrf_k": 60,
        "pesos_fusion": {"w05": 0.5, "w03": 0.3, "w07": 0.7},
        "variantes_100": ["w05", "w03", "w07", "rrf1a"],
        "fusion_barrido": "w05",
        "contextos": {"sin_hubs": 0.0, "con_hubs": 1.5},
        "estrategias": [],
        "referencia": None,
    },
    "a4_bootstrap": 1000,
    "a4_seed_pareado": 42,
    "version_salida_v11": "v11-ciclo3-sprintA4",
    "ruta_benchmark_v11": "resultados/vinculo_cp_benchmark_v11.json",
    "ruta_por_ano_v11": "resultados/vinculo_cp_por_ano_v11.csv",
    "ruta_figura_v11": "figuras/deriva_retrieval_v11.png",
    "ruta_figura_contaminacion_v11": "figuras/contaminacion_transitorios_v11.png",
    "ruta_figura_a4_fusion_profundidad": "figuras/c3_a4_fusion_profundidad.png",
    "ruta_a4_fusion_profundidad": "resultados/c3_a4_fusion_profundidad.json",
    "nota_v11": (
        "Ciclo v3 - Sprint A4: fusion de PRIMERA ETAPA y barrido de "
        "profundidad sobre la base ganadora ALIA+bge+hubs (A2/A3). "
        "Embedding denso FIJO en ALIA-MrBERT legal; 4 variantes a 100 "
        "candidatos con bge rerank chunks: w05 (referencia, hibrido "
        "min-max 0.5*BM25 + 0.5*denso; reutiliza la estrategia homonima "
        "de A2/A3 si existe), w03 (BM25 0.3 / denso 0.7), w07 (BM25 0.7 "
        "/ denso 0.3) y rrf1a (RRF k=60 de los rankings completos BM25 y "
        "denso ALIA, candidatos = top-100 del RRF). Barrido de "
        "profundidad 50/100/200 de la fusion elegida (--a4-fusion; "
        "default w05, la referencia pre-registrada por revision A4: "
        "el smoke n=10 de w07 no es criterio robusto). Cada "
        "variante se reporta en los contextos "
        "alpha=0 y alpha=1.5 con metricas por caso y bootstrap PAREADO "
        "(1000 remuestreos, seed 42) de la diferencia contra w05@100. "
        "Salidas NUEVAS v11; v2..v10 quedan intactas (ADR-16)."
    ),
    # --- ciclo v3 - Sprint B1: reranker de dominio fine-tuneado ---
    # La evaluacion B1 es ADITIVA: sin --b1 nada cambia. Puntua la ruta
    # ganadora del producto con rerank (hibrido top-50 + prior de hubs
    # alpha=1.5 + k=10, la misma de A4/v8) con el cross-encoder
    # fine-tuneado por scripts/24_b1_entrenar_reranker.py, guardado con
    # save_pretrained en modelos/b1_reranker/; la ruta local se resuelve
    # contra ROOT y --reranker-model la sobreescribe SOLO para pruebas
    # (alias del catalogo del lib, ruta local o HF id). La referencia es
    # la MISMA ruta con bge-reranker-v2-m3 (mismos casos, candidatos y
    # pares), con bootstrap pareado del delta (ADR-16). Salidas NUEVAS
    # v12; v2..v11 quedan intactas.
    "b1": {
        "activa": False,
        "ruta_modelo": "modelos/b1_reranker",
        "modelo": None,
        "estrategia": "hibrido_rerank_chunks_top50_b1",
        "estrategia_referencia": "hibrido_rerank_chunks_top50",
        "candidatos_k": 50,
        "k": 10,
        "contextos": {"sin_hubs": 0.0, "con_hubs": 1.5},
        "contexto_ganador": "con_hubs",
    },
    "b1_bootstrap": 1000,
    "b1_seed_pareado": 42,
    "version_salida_v12": "v12-ciclo3-sprintB1",
    "ruta_benchmark_v12": "resultados/vinculo_cp_benchmark_v12.json",
    "ruta_por_ano_v12": "resultados/vinculo_cp_por_ano_v12.csv",
    "ruta_figura_v12": "figuras/deriva_retrieval_v12.png",
    "ruta_figura_contaminacion_v12": "figuras/contaminacion_transitorios_v12.png",
    "ruta_b1_reranker": "resultados/c3_b1_reranker.json",
    "nota_v12": (
        "Ciclo v3 - Sprint B1: evaluacion del reranker de dominio "
        "fine-tuneado con los pares de scripts/23_b1_pares_reranker.py y "
        "entrenado por scripts/24_b1_entrenar_reranker.py (modelos/"
        "b1_reranker/). La estrategia evaluada es la ruta ganadora del "
        "producto con rerank: hibrido_rerank_chunks_top50 (top-50 del "
        "hibrido 0.5*bm25 + 0.5*e5) con prior de hubs alpha=1.5 y salida "
        "k=10, puntuada con el reranker B1 sobre los MISMOS candidatos y "
        "pares (chunk, pasaje) que la referencia bge-reranker-v2-m3. La "
        "seccion comparacion_b1 reporta nDCG@10/F1@10 por contexto "
        "(alpha=0/1.5) y el IC 95% bootstrap PAREADO (1000 remuestreos, "
        "seed 42) de delta = B1 - bge del mismo run, mas la verificacion "
        "externa de la referencia y el pareado adicional contra los "
        "top-10 por caso congelados en v8/v10 (contexto alpha=0) cuando "
        "estan disponibles; si no, se documenta el fallback. Salidas "
        "NUEVAS v12 y derivado c3_b1_reranker.json; v2..v11 quedan "
        "intactas (ADR-16)."
    ),
    # --- ciclo v3 - Sprint B2: retriever denso de dominio fine-tuneado ---
    # La evaluacion B2 es ADITIVA: sin --b2 nada cambia. Evalua la ruta
    # hibrido_b2_rerank_chunks_top50_b1: el hibrido pre-registrado
    # 0.5*bm25 + 0.5*B2 (retriever denso fine-tuneado por el script 25,
    # guardado en modelos/b2_retriever/) sobre los MISMOS top-50
    # candidatos y pares (chunk, pasaje) que la ruta B1, con el reranker
    # B1 (modelos/b1_reranker) y prior de hubs alpha=1.5. El denso se
    # carga con el plumbing de A2 (carga generica + cache por slug). La
    # referencia primaria es la ruta B1 con hibrido E5 congelada en v12
    # (comparacion_b1.metricas_por_caso / top10_por_caso) y la
    # verificacion externa es ALIA@100 del v9 (comparacion_a2). Salidas
    # NUEVAS v13; v2..v12 quedan intactas (ADR-16).
    "b2": {
        "activa": False,
        "variante": None,
        "ruta_modelo": "modelos/b2_retriever",
        # ciclo v3 - iteracion v14: convencion de ruta de una variante del
        # retriever (p. ej. ep2 -> modelos/b2_retriever_ep2); el {slug} se
        # formatea en resolver_modelo_b2 al activar --b2-variante.
        "ruta_modelo_variante": "modelos/b2_retriever_{slug}",
        "modelo": None,
        "slug": None,
        "clave_hibrido": None,
        "estrategia": "hibrido_b2_rerank_chunks_top50_b1",
        "estrategia_referencia_v12": "hibrido_rerank_chunks_top50_b1",
        "referencia_externa": {
            "estrategia": "hibrido_alia_rerank_chunks_top100",
            "slug": "alia",
            "k": 100,
        },
        "modelo_reranker_por_defecto": "modelos/b1_reranker",
        "modelo_reranker": None,
        "candidatos_k": 50,
        "k": 10,
        "contextos": {"sin_hubs": 0.0, "con_hubs": 1.5},
        "contexto_ganador": "con_hubs",
    },
    "b2_bootstrap": 1000,
    "b2_seed_pareado": 42,
    "version_salida_v13": "v13-ciclo3-sprintB2",
    "ruta_benchmark_v13": "resultados/vinculo_cp_benchmark_v13.json",
    "ruta_por_ano_v13": "resultados/vinculo_cp_por_ano_v13.csv",
    "ruta_figura_v13": "figuras/deriva_retrieval_v13.png",
    "ruta_figura_contaminacion_v13": "figuras/contaminacion_transitorios_v13.png",
    "ruta_b2_retriever": "resultados/c3_b2_retriever.json",
    "nota_v13": (
        "Ciclo v3 - Sprint B2: evaluacion del retriever denso de dominio "
        "fine-tuneado por scripts/25_b2_retriever_denso.py (modelos/"
        "b2_retriever/) en la 1a etapa del hibrido pre-registrado "
        "0.5*bm25 + 0.5*B2 (top-50 candidatos) con el reranker B1 "
        "(modelos/b1_reranker) sobre los MISMOS pares (chunk, pasaje) "
        "que la ruta B1 del producto; la ruta de producto es con_hubs "
        "alpha=1.5 con salida k=10. La seccion comparacion_b2 reporta "
        "nDCG@10/F1@10 por contexto (alpha=0/1.5) y el IC 95% bootstrap "
        "PAREADO (1000 remuestreos, seed 42) de delta = B2 - referencia "
        "v12 (la ruta B1 con hibrido E5 congelada: comparacion_b1."
        "metricas_por_caso; fallback top10_por_caso en sin_hubs; fallback "
        "final documentado), mas la verificacion externa vs A2/ALIA@100 "
        "de v9 (comparacion_a2.metricas_por_caso) cuando esta disponible. "
        "Salidas NUEVAS v13 y derivado c3_b2_retriever.json; v2..v12 "
        "quedan intactas (ADR-16)."
    ),
    # ciclo v3 - iteracion v14: variante B2 (retriever denso alternativo,
    # p. ej. la 2a epoca ep2 del script 25). Plantillas con {slug}/{modelo}
    # formateadas en aplicar_modo_b2 al activar --b2-variante; las rutas
    # formateadas se asignan a las claves estandar (ruta_benchmark, etc.)
    # antes de resolver_rutas, por lo que estas plantillas NO entran en
    # RUTAS_RELATIVAS. La nota activa queda en CONFIG['nota_v14'].
    "b2_variante_v14": {
        "version_salida": "v14-ciclo3-b2-{slug}",
        "ruta_benchmark": "resultados/vinculo_cp_benchmark_v14_{slug}.json",
        "ruta_por_ano": "resultados/vinculo_cp_por_ano_v14_{slug}.csv",
        "ruta_figura": "figuras/deriva_retrieval_v14_{slug}.png",
        "ruta_figura_contaminacion": (
            "figuras/contaminacion_transitorios_v14_{slug}.png"
        ),
        "ruta_b2_retriever": "resultados/c3_b2_retriever_{slug}.json",
        "nota": (
            "Ciclo v3 - iteracion v14: evaluacion de la variante B2 "
            "'{slug}' del retriever denso de dominio ({modelo}) en la 1a "
            "etapa del hibrido pre-registrado 0.5*bm25 + 0.5*B2 (top-50 "
            "candidatos) con el reranker B1 (modelos/b1_reranker) sobre "
            "los MISMOS pares (chunk, pasaje) que la ruta B1 del "
            "producto; la ruta de producto es con_hubs alpha=1.5 con "
            "salida k=10. La seccion comparacion_b2 reporta nDCG@10/F1@10 "
            "por contexto (alpha=0/1.5) y el IC 95% bootstrap PAREADO "
            "(1000 remuestreos, seed 42) de delta = B2 - referencia v12 "
            "(la ruta B1 con hibrido E5 congelada: comparacion_b1."
            "metricas_por_caso; fallback top10_por_caso en sin_hubs; "
            "fallback final documentado), mas la verificacion externa vs "
            "A2/ALIA@100 de v9 (comparacion_a2.metricas_por_caso) cuando "
            "esta disponible. Salidas NUEVAS v14 nombradas por slug "
            "(vinculo_cp_benchmark_v14_{slug}.json, por_ano/figuras con "
            "sufijo _{slug} y derivado c3_b2_retriever_{slug}.json); "
            "v2..v13 quedan intactas (ADR-16)."
        ),
    },
    "nota_v14": None,
}

# El hf_id de E5 en el catalogo A2 es el mismo del pipeline legacy
# (fuente unica de verdad; se asigna aqui porque un dict literal no
# puede referenciar sus propias claves).
CONFIG["modelos_embedding"]["e5"]["hf_id"] = CONFIG["modelo_e5"]

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

# ciclo v3 - Sprint A2: medidas guardadas POR CASO y con IC pareado en la
# comparacion de embeddings (ademas del protocolo completo de MEDIDAS).
MEDIDAS_A2 = ("ndcg_at_10", "f1_at_10")

# Estrategias evaluadas en el diagnostico de contaminacion (Fase R): las
# basadas en el indice BM25 sucio v1 (bm25 puro y su hibrido con E5). La
# cita-exacta no consulta el indice y E5 no depende de los IDs.
ESTRATEGIAS_DIAGNOSTICO = ("bm25", "hibrido")

# ciclo v2 - Fase W-a: estrategias que admiten el prior de hubs. La
# cita-exacta queda fuera (mismo regex M2 que genera el gold: priorizarla
# seria circular). El prior se aplica tras calcular los scores base y
# antes de top-k/metricas (ver aplicar_prior_a_casos). Es una LISTA
# mutable: la matriz A2 (ciclo v3) anade sus estrategias al activarse.
ESTRATEGIAS_RANKEABLES = [
    "bm25",
    "e5",
    "hibrido",
    "bm25_rerank",
    "bm25_rerank_chunks",
    "bm25_rerank_chunks_top50",
    "hibrido_rerank_chunks_top50",
]

# Subconjunto de rerank: su ranking de producto esta truncado a
# CONFIG['k_rerank_salida'] (10); al re-rankear con prior el resultado se
# trunca igual para conservar la semantica de las metricas (el MRR solo
# puede mirar ese top-10). Lista mutable: la matriz A2 anade estrategias.
ESTRATEGIAS_RERANK = [
    "bm25_rerank",
    "bm25_rerank_chunks",
    "bm25_rerank_chunks_top50",
    "hibrido_rerank_chunks_top50",
]

# Clave del caso donde vive el vector de scores base de cada estrategia:
# fuente unica para top10_por_caso y para la calibracion de hubs.
# Diccionario mutable: la matriz A2 (ciclo v3) registra aqui sus
# estrategias al activarse (aplicar_modo_a2).
FUENTE_SCORES = {
    "bm25": "scores_bm25",
    "e5": "scores_e5",
    "hibrido": "scores_hibrido",
    "bm25_rerank": "scores_bm25_rerank",
    "bm25_rerank_chunks": "scores_bm25_rerank_chunks",
    "bm25_rerank_chunks_top50": "scores_bm25_rerank_chunks_top50",
    "hibrido_rerank_chunks_top50": "scores_hibrido_rerank_chunks_top50",
}

# ciclo v3 - Sprint A1: contextos de scoring del diagnostico. sin_hubs es
# el score base tal cual (alpha=0); con_hubs aplica el prior con los
# alphas adoptados en v8 (CONFIG['alphas_adoptados_hubs']).
CONTEXTO_SIN_HUBS = "sin_hubs"
CONTEXTO_CON_HUBS = "con_hubs"

# Etiquetas y colores de las estrategias focales del diagnostico A1 (los
# mismos colores que graficar_deriva para las estrategias equivalentes).
ETIQUETAS_FOCALES = {
    "bm25": "BM25",
    "hibrido": "hibrido 0.5/0.5",
    "bm25_rerank_chunks_top50": "BM25 top-50 + rerank",
    "hibrido_rerank_chunks_top50": "hibrido top-50 + rerank",
}

COLORES_FOCALES = {
    "bm25": "#1f77b4",
    "hibrido": "#2ca02c",
    "bm25_rerank_chunks_top50": "#8c564b",
    "hibrido_rerank_chunks_top50": "#e377c2",
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
    # ciclo v3 - Sprint A1: diagnostico honesto (JSON y 3 figuras nuevas).
    "ruta_diagnostico",
    "ruta_figura_diagnostico_recall",
    "ruta_figura_diagnostico_ic",
    "ruta_figura_diagnostico_periodo",
    # ciclo v3 - Sprint A2: cache de embeddings y salidas v9.
    "cache_emb_cp_dir",
    "ruta_benchmark_v9",
    "ruta_por_ano_v9",
    "ruta_figura_v9",
    "ruta_figura_contaminacion_v9",
    "ruta_figura_a2_embeddings",
    # ciclo v3 - Sprint A3: salidas v10 y figura de rerankers.
    "ruta_benchmark_v10",
    "ruta_por_ano_v10",
    "ruta_figura_v10",
    "ruta_figura_contaminacion_v10",
    "ruta_figura_a3_rerankers",
    # ciclo v3 - Sprint A4: salidas v11, figura de fusion/profundidad y
    # derivado trazable de la seccion comparacion_a4.
    "ruta_benchmark_v11",
    "ruta_por_ano_v11",
    "ruta_figura_v11",
    "ruta_figura_contaminacion_v11",
    "ruta_figura_a4_fusion_profundidad",
    "ruta_a4_fusion_profundidad",
    # ciclo v3 - Sprint B1: salidas v12 y derivado trazable del reranker
    # de dominio (la ruta local del modelo se resuelve aparte en
    # resolver_modelo_b1).
    "ruta_benchmark_v12",
    "ruta_por_ano_v12",
    "ruta_figura_v12",
    "ruta_figura_contaminacion_v12",
    "ruta_b1_reranker",
    # ciclo v3 - Sprint B2: salidas v13 y derivado trazable del retriever
    # denso de dominio (la ruta local del modelo se resuelve aparte en
    # resolver_modelo_b2).
    "ruta_benchmark_v13",
    "ruta_por_ano_v13",
    "ruta_figura_v13",
    "ruta_figura_contaminacion_v13",
    "ruta_b2_retriever",
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


def construir_pares_rerank_chunks(
    scores_candidatos: np.ndarray,
    matriz_bm25: np.ndarray,
    chunks: list[str],
    grupos_indices: list[np.ndarray],
    pasajes: pd.DataFrame,
    k_candidatos: int,
) -> dict:
    """Construye UNA vez los pares (chunk, pasaje) de los top-k candidatos.

    Derivacion comun de las estrategias de rerank por chunks (v5, top-50
    de la Fase T y A2/A3): para cada articulo candidato (ordenado por
    scores_candidatos): (1) su mejor pasaje es el de mayor score BM25 del
    caso (maximo de la matriz chunk x pasaje); (2) los top-
    CONFIG['k_chunks_rerank'] = 3 chunks de la query son los de mayor
    score BM25 contra ese pasaje. ciclo v3 - Sprint A3: extraida de
    ranking_rerank_chunks (misma logica, mismo orden) para que la
    comparativa de rerankers puntue los MISMOS pares con bge y con el
    reranker SINAI sin duplicarlos.

    Args:
        scores_candidatos: Scores que ordenan a los candidatos
            (n_articulos,) alineados con los canones.
        matriz_bm25: Matriz (n_chunks, n_pasajes) de scores BM25 del
            caso (salida de scores_bm25_caso).
        chunks: Chunks de la ventana de query (filas de matriz_bm25).
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        pasajes: DataFrame con columnas canon y texto (un pasaje/fila).
        k_candidatos: Numero de candidatos de la 1a etapa.

    Returns:
        Dict con candidatos (indices de articulo ordenados por score),
        queries y pasajes (listas alineadas de los pares) y posiciones
        (posicion del articulo candidato en candidatos, por par).
    """
    orden_candidatos = np.argsort(-scores_candidatos, kind="stable")
    candidatos = orden_candidatos[:k_candidatos]
    maximos_pasajes = matriz_bm25.max(axis=0)
    textos_pasajes = pasajes["texto"].astype(str).tolist()
    queries: list[str] = []
    pasajes_lote: list[str] = []
    posiciones: list[int] = []
    for pos_articulo, indice in enumerate(candidatos):
        indices_pasaje = grupos_indices[indice]
        mejor_pasaje = int(
            indices_pasaje[np.argmax(maximos_pasajes[indices_pasaje])]
        )
        scores_chunks = matriz_bm25[:, mejor_pasaje]
        top_chunks = np.argsort(-scores_chunks, kind="stable")[
            : CONFIG["k_chunks_rerank"]
        ]
        for fila_chunk in top_chunks:
            queries.append(chunks[fila_chunk])
            pasajes_lote.append(textos_pasajes[mejor_pasaje])
            posiciones.append(pos_articulo)
    return {
        "candidatos": candidatos,
        "queries": queries,
        "pasajes": pasajes_lote,
        "posiciones": posiciones,
    }


def scores_maxpool_por_candidato(
    scores_pares: np.ndarray,
    posiciones: list[int],
    candidatos: np.ndarray,
    n_canones: int,
) -> np.ndarray:
    """Agrega por max-pooling los scores de los pares al vector de articulos.

    El score de cada articulo candidato es el MAXIMO de los scores de sus
    pares (coherente con ADR-9); los articulos fuera del set de
    candidatos quedan en -inf (no son resultados retornables del rerank).

    Args:
        scores_pares: Array (n_pares,) con el score de cada par en el
            orden devuelto por construir_pares_rerank_chunks.
        posiciones: Posicion del articulo candidato en candidatos por par.
        candidatos: Indices de articulo del set de candidatos.
        n_canones: Tamano del universo de articulos.

    Returns:
        Array (n_canones,) con el maximo por candidato y -inf en el resto.
    """
    maximos = np.full(len(candidatos), -np.inf, dtype=np.float64)
    for posicion, score in zip(posiciones, scores_pares, strict=True):
        maximos[posicion] = max(maximos[posicion], float(score))
    scores = np.full(n_canones, -np.inf, dtype=np.float64)
    for pos_articulo, indice in enumerate(candidatos):
        scores[indice] = maximos[pos_articulo]
    return scores


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

    ciclo v3 - Sprint A3: la derivacion de los pares y el max-pooling se
    delegan en construir_pares_rerank_chunks y
    scores_maxpool_por_candidato (misma logica y mismo orden que antes;
    extraidos para que A3 puntue los pares una sola vez con los dos
    rerankers).

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
    pares = construir_pares_rerank_chunks(
        scores_candidatos,
        matriz_bm25,
        chunks,
        grupos_indices,
        pasajes,
        k_candidatos,
    )
    scores_pares = reranker.puntuar(pares["queries"], pares["pasajes"])
    scores_rerank = scores_maxpool_por_candidato(
        scores_pares, pares["posiciones"], pares["candidatos"], len(canones)
    )
    maximos = scores_rerank[pares["candidatos"]]
    orden_rerank = pares["candidatos"][
        np.argsort(-maximos, kind="stable")
    ]
    ranking = [canones[i] for i in orden_rerank[: CONFIG["k_rerank_salida"]]]
    max_tokens_par = reranker.max_tokens_par(pares["queries"], pares["pasajes"])
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
    # ciclo v3 - Sprint A2: paleta adicional para estrategias nuevas
    # (hibrido_<slug>_rerank_chunks_topK) sin romper las existentes.
    paleta_extra = ["#17becf", "#bcbd22", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    for _i, _e in enumerate(e for e in CONFIG["estrategias"] if e not in colores):
        colores[_e] = paleta_extra[_i % len(paleta_extra)]
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
                color=colores.get(estrategia, "#333333"),
                label=etiquetas.get(estrategia, estrategia),
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


# ===== SECCION 8c: diagnostico honesto (ciclo v3 - Sprint A1) =====
# El diagnostico A1 es SOLO inferencia sobre los scores ya calculados:
# re-rankea y remuestrea, nunca re-ejecuta BM25/E5/reranker (ADR-16:
# artefactos nuevos c3_*; los benchmarks previos quedan intactos).
def recall_at_k(relevantes: list[str], ranking: list[str], k: int) -> float:
    """Calcula el recall@k de un ranking frente al gold binario.

    Args:
        relevantes: Canones de articulos relevantes (weak labels M2).
        ranking: Canones ordenados por el retrieval (mejor primero).
        k: Profundidad de corte.

    Returns:
        Recall@k en [0.0, 1.0]; 0.0 defensivo si el caso no tiene gold.
    """
    rel = set(relevantes)
    if not rel:
        return 0.0
    return sum(1 for canon in ranking[:k] if canon in rel) / len(rel)


def metricas_diagnostico_caso(relevantes: list[str], ranking: list[str]) -> dict:
    """Calcula las 4 medidas del IC bootstrap para un caso.

    Args:
        relevantes: Canones de articulos relevantes (weak labels M2).
        ranking: Canones ordenados por el retrieval (mejor primero).

    Returns:
        Dict con ndcg_at_10, mrr, f1_at_5 y f1_at_10.
    """
    m5 = metricas_ranking(relevantes, ranking, CONFIG["k"])
    m10 = metricas_ranking(relevantes, ranking, CONFIG["k_f1_10"])
    return {
        "ndcg_at_10": m5["ndcg_at_10"],
        "mrr": m5["mrr"],
        "f1_at_5": m5["f1_at_k"],
        "f1_at_10": m10["f1_at_k"],
    }


def bootstrap_ic_media(
    valores: np.ndarray, n_remuestreos: int, semilla: int
) -> dict:
    """IC 95% bootstrap del promedio por remuestreo de casos con reemplazo.

    Args:
        valores: Array (n,) con la medida por caso (orden de la muestra).
        n_remuestreos: Numero de remuestreos bootstrap (1000 en A1).
        semilla: Semilla del generador (42 en A1, reproducible).

    Returns:
        Dict {"n", "media", "p2_5", "p97_5"}; con n=0 todos los valores
        son None y con n=1 los percentiles igualan a la media.
    """
    arreglo = np.asarray(valores, dtype=np.float64)
    n = int(arreglo.size)
    if n == 0:
        return {"n": 0, "media": None, "p2_5": None, "p97_5": None}
    media = float(arreglo.mean())
    if n == 1:
        return {"n": 1, "media": media, "p2_5": media, "p97_5": media}
    generador = np.random.default_rng(semilla)
    indices = generador.integers(0, n, size=(n_remuestreos, n))
    medias = arreglo[indices].mean(axis=1)
    return {
        "n": n,
        "media": media,
        "p2_5": float(np.percentile(medias, 2.5)),
        "p97_5": float(np.percentile(medias, 97.5)),
    }


def ranking_diagnostico(
    caso: dict,
    estrategia: str,
    contexto: str,
    prior: dict,
    canones: list[str],
) -> list[str]:
    """Reconstruye el ranking de una estrategia focal en un contexto.

    Args:
        caso: Dict del caso con scores base (FUENTE_SCORES).
        estrategia: Estrategia focal (clave de FUENTE_SCORES).
        contexto: CONTEXTO_SIN_HUBS (alpha=0) o CONTEXTO_CON_HUBS.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con el vector de scores.

    Returns:
        Lista de canones ordenados por score descendente; en las
        variantes de rerank el ranking se restringe a los candidatos
        (score finito), de modo que recall@k con k mayor que el numero
        de candidatos queda topado por el recall de candidatos.
    """
    if contexto == CONTEXTO_CON_HUBS:
        alpha = float(CONFIG["alphas_adoptados_hubs"].get(estrategia, 0.0))
    else:
        alpha = 0.0
    return ranking_con_alpha(caso, estrategia, alpha, prior, canones)


def asignar_terciles_longitud(casos: list[dict]) -> tuple[list[int], list[int]]:
    """Asigna cada caso a un tercil de longitud en palabras (equilibrado).

    Los terciles parten la muestra ordenada por (palabras, id) en tres
    grupos lo mas iguales posible: con n multiplo de 3 son de n/3; en
    otro caso los primeros reciben un caso extra. La longitud es la del
    texto limpio que alimenta la query (documentado en el JSON).

    Args:
        casos: Lista de dicts con texto_limpio e id.

    Returns:
        Tupla (terciles, palabras): terciles[i] en {0,1,2} y numero de
        palabras del texto limpio del caso i.
    """
    palabras = [len(caso["texto_limpio"].split()) for caso in casos]
    n = len(casos)
    orden = sorted(range(n), key=lambda i: (palabras[i], casos[i]["id"]))
    terciles = [0] * n
    for posicion, indice in enumerate(orden):
        terciles[indice] = min(2, posicion * 3 // n) if n > 0 else 0
    return terciles, palabras


def techo_teorico_gold(casos: list[dict], canones: list[str]) -> dict:
    """Techo estructural del gold actual (F1@5/F1@10 y universo).

    El F1@k maximo por caso asume un orden perfecto del gold:
    P@k = min(k, n_gold)/k y R@k = min(k, n_gold)/n_gold; la media
    sobre casos es el techo que ningun modelo puede superar con este
    gold. El recall del universo es la fraccion media del gold que el
    indice activo puede recuperar (recall@384 devolviendo todo el
    indice).

    Args:
        casos: Lista de dicts con arts (gold validado).
        canones: Canones del indice activo.

    Returns:
        Dict con distribucion de n_gold, F1@5/F1@10 maximos medios y
        recall del universo.
    """
    n_gold = [len(caso["arts"]) for caso in casos]
    indice = set(canones)

    def f1_maximo(gold: int, k: int) -> float:
        """F1@k con orden perfecto del gold para un caso."""
        aciertos = min(k, gold)
        p = aciertos / k
        r = aciertos / gold if gold > 0 else 0.0
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    histograma: dict[str, int] = {}
    for gold in n_gold:
        histograma[str(gold)] = histograma.get(str(gold), 0) + 1
    alcanzables = [
        len(set(caso["arts"]) & indice) / len(caso["arts"])
        for caso in casos
        if caso["arts"]
    ]
    return {
        "n_gold_por_caso": {
            "media": round(float(np.mean(n_gold)), 4),
            "mediana": float(np.median(n_gold)),
            "min": int(min(n_gold)),
            "max": int(max(n_gold)),
            "histograma": {
                clave: histograma[clave]
                for clave in sorted(histograma, key=int)
            },
        },
        "f1_at_5_maximo": {
            "media": round(
                float(np.mean([f1_maximo(g, 5) for g in n_gold])), 6
            )
        },
        "f1_at_10_maximo": {
            "media": round(
                float(np.mean([f1_maximo(g, 10) for g in n_gold])), 6
            )
        },
        "recall_universo": {
            "media": (
                round(float(np.mean(alcanzables)), 6) if alcanzables else None
            ),
            "n_articulos_indice": len(canones),
            "casos_con_gold_fuera_indice": sum(
                1 for caso in casos if set(caso["arts"]) - indice
            ),
        },
        "nota": (
            "F1@k maximo por caso con orden perfecto del gold "
            "(P=min(k,n_gold)/k, R=min(k,n_gold)/n_gold); la media es el "
            "techo estructural del gold debil de M2. recall_universo = "
            "fraccion media del gold alcanzable con el indice activo "
            "(recall@384 si se devuelve el indice completo)."
        ),
    }


def construir_diagnostico_a1(
    casos: list[dict],
    prior: dict,
    canones: list[str],
    limite: int | None = None,
) -> dict:
    """Construye el diagnostico A1 completo reutilizando scores base.

    ciclo v3 - Sprint A1: sin re-ejecutar modelos, reconstruye los
    rankings de las estrategias focales en dos contextos (sin hubs y con
    los alphas adoptados en v8) y calcula curvas recall@k, IC 95%
    bootstrap de nDCG@10/MRR/F1@5/F1@10, desglose por periodo y por
    terciles de longitud, recall de candidatos de la primera etapa y el
    techo teorico del gold.

    Args:
        casos: Lista de dicts con id, ano, arts, texto_limpio y scores.
        prior: Dict {canon: valor} del prior de hubs de la CP.
        canones: Canones alineados con los vectores de scores.
        limite: Valor de --limite (None en el run final; documentado).

    Returns:
        Dict serializable con version "c3-a1", config, curvas_recall,
        recall_candidatos_primera_etapa, ic_bootstrap, por_periodo,
        por_longitud, techo_teorico y nota.
    """
    estrategias = [
        estrategia
        for estrategia in CONFIG["estrategias_focales_hubs"]
        if estrategia in CONFIG["estrategias"] and estrategia in FUENTE_SCORES
    ]
    contextos = (CONTEXTO_SIN_HUBS, CONTEXTO_CON_HUBS)
    ks = list(CONFIG["k_curvas_recall"])
    periodos = [nombre for nombre, _, _ in CONFIG["periodos"]]
    medidas = ("ndcg_at_10", "mrr", "f1_at_5", "f1_at_10")
    n_bootstrap = CONFIG["diagnostico_bootstrap"]
    semilla = CONFIG["diagnostico_seed"]

    # Rankings reconstruidos una sola vez por (contexto, estrategia).
    rankings = {
        contexto: {
            estrategia: [
                ranking_diagnostico(caso, estrategia, contexto, prior, canones)
                for caso in casos
            ]
            for estrategia in estrategias
        }
        for contexto in contextos
    }

    # (1) Curvas recall@k por contexto y estrategia.
    curvas: dict = {}
    for contexto in contextos:
        curvas[contexto] = {}
        for estrategia in estrategias:
            valores_k = {}
            for k in ks:
                valores = [
                    recall_at_k(caso["arts"], ranking, k)
                    for caso, ranking in zip(
                        casos, rankings[contexto][estrategia], strict=True
                    )
                ]
                valores_k[str(k)] = round(float(np.mean(valores)), 6)
            curvas[contexto][estrategia] = valores_k

    # Medidas por caso (una vez) para IC, periodo y longitud.
    valores_caso: dict = {}
    for contexto in contextos:
        valores_caso[contexto] = {}
        for estrategia in estrategias:
            filas_caso = [
                metricas_diagnostico_caso(caso["arts"], ranking)
                for caso, ranking in zip(
                    casos, rankings[contexto][estrategia], strict=True
                )
            ]
            valores_caso[contexto][estrategia] = {
                medida: np.asarray(
                    [fila[medida] for fila in filas_caso], dtype=np.float64
                )
                for medida in medidas
            }

    # (2) IC 95% bootstrap por estrategia focal (full).
    ic_bootstrap = {
        contexto: {
            estrategia: {
                medida: bootstrap_ic_media(
                    valores_caso[contexto][estrategia][medida],
                    n_bootstrap,
                    semilla,
                )
                for medida in medidas
            }
            for estrategia in estrategias
        }
        for contexto in contextos
    }

    # (3) Desglose por periodo (n=30 en el run final).
    por_periodo: dict = {}
    for contexto in contextos:
        por_periodo[contexto] = {}
        for periodo in periodos:
            indices = [
                i
                for i, caso in enumerate(casos)
                if asignar_periodo(caso["ano"]) == periodo
            ]
            por_periodo[contexto][periodo] = {
                "n_casos": len(indices),
                "por_estrategia": {
                    estrategia: {
                        medida: bootstrap_ic_media(
                            valores_caso[contexto][estrategia][medida][indices],
                            n_bootstrap,
                            semilla,
                        )
                        for medida in ("ndcg_at_10", "f1_at_10")
                    }
                    for estrategia in estrategias
                },
            }

    # (4) Desglose por terciles de longitud del caso.
    terciles, palabras = asignar_terciles_longitud(casos)
    por_longitud: dict = {}
    for contexto in contextos:
        por_longitud[contexto] = {}
        for tercil, nombre in enumerate(("tercil_1", "tercil_2", "tercil_3")):
            indices = [i for i, valor in enumerate(terciles) if valor == tercil]
            por_longitud[contexto][nombre] = {
                "n_casos": len(indices),
                "rango_palabras": (
                    [
                        int(min(palabras[i] for i in indices)),
                        int(max(palabras[i] for i in indices)),
                    ]
                    if indices
                    else None
                ),
                "ndcg_at_10": {
                    estrategia: bootstrap_ic_media(
                        valores_caso[contexto][estrategia]["ndcg_at_10"][
                            indices
                        ],
                        n_bootstrap,
                        semilla,
                    )
                    for estrategia in estrategias
                },
            }

    # Recall de los candidatos top-50 de la primera etapa (techo del
    # reranker): bm25 alimenta a bm25_rerank_chunks_top50 e hibrido a
    # hibrido_rerank_chunks_top50.
    k_candidatos = CONFIG["k_candidatos_rerank_top50"]
    recall_candidatos: dict = {}
    for base in ("bm25", "hibrido"):
        if base not in CONFIG["estrategias"]:
            continue
        valores = [
            recall_at_k(caso["arts"], caso["rankings"][base], k_candidatos)
            for caso in casos
        ]
        por_periodo_candidatos = {}
        for periodo in periodos:
            indices = [
                i
                for i, caso in enumerate(casos)
                if asignar_periodo(caso["ano"]) == periodo
            ]
            por_periodo_candidatos[periodo] = (
                round(float(np.mean([valores[i] for i in indices])), 6)
                if indices
                else None
            )
        recall_candidatos[f"{base}_top{k_candidatos}"] = {
            "estrategia_primera_etapa": base,
            "k_candidatos": k_candidatos,
            "media": round(float(np.mean(valores)), 6),
            "por_periodo": por_periodo_candidatos,
        }

    return {
        "version": CONFIG["version_diagnostico"],
        "fecha": datetime.now(timezone.utc).isoformat(),
        "smoke_test": limite is not None,
        "casos": len(casos),
        "config": {
            "limite_casos": limite,
            "muestra_por_ano": CONFIG["muestra_por_ano"],
            "semilla_muestra": CONFIG["semilla"],
            "indice": CONFIG["version_indice"],
            "reranker": (
                CONFIG["reranker"]
                if "bm25_rerank_chunks_top50" in CONFIG["estrategias"]
                else None
            ),
            "estrategias_focales": estrategias,
            "k_curvas_recall": ks,
            "n_remuestreos_bootstrap": n_bootstrap,
            "semilla_bootstrap": semilla,
            "contextos": [CONTEXTO_SIN_HUBS, CONTEXTO_CON_HUBS],
            "alphas_adoptados_hubs": dict(CONFIG["alphas_adoptados_hubs"]),
            "recall_topado_por_candidatos": {
                estrategia: True
                for estrategia in estrategias
                if estrategia in ESTRATEGIAS_RERANK
            },
            "nota_terciles": (
                "Terciles equilibrados por numero de palabras del texto "
                "limpio (la query); rango_palabras documenta los cortes."
            ),
        },
        "curvas_recall": curvas,
        "recall_candidatos_primera_etapa": recall_candidatos,
        "ic_bootstrap": ic_bootstrap,
        "por_periodo": por_periodo,
        "por_longitud": por_longitud,
        "techo_teorico": techo_teorico_gold(casos, canones),
        "nota": (
            "ciclo v3 - Sprint A1 'metricas honestas': el diagnostico "
            "reutiliza los scores base ya calculados (sin re-ejecutar "
            "BM25/E5/reranker) y solo re-rankea/remuestrea. Contextos: "
            "sin_hubs (alpha=0) y con_hubs (alphas adoptados del run v8: "
            "bm25 0.75, hibrido 0.75 y variantes de rerank top-50 1.5). "
            "En las variantes de rerank el recall@k para k>50 esta topado "
            "por el set de candidatos de la primera etapa (recall de "
            "candidatos), no por el universo. El IC 95% es un bootstrap "
            "de casos con reemplazo (1000 remuestreos, seed 42); con n "
            "pequeno (periodos de 30 casos) el IC es orientativo. El "
            "gold es el debil de M2 (NER): el techo teorico documenta su "
            "limite estructural."
        ),
    }


def graficar_recall_curvas(diagnostico: dict) -> None:
    """Genera la figura c3_recall_curvas.png (recall@k por estrategia).

    Args:
        diagnostico: Dict de construir_diagnostico_a1.
    """
    estrategias = diagnostico["config"]["estrategias_focales"]
    ks = list(diagnostico["config"]["k_curvas_recall"])
    figura, eje = plt.subplots(figsize=(9.5, 5.5))
    for estrategia in estrategias:
        color = COLORES_FOCALES.get(estrategia, "#333333")
        etiqueta = ETIQUETAS_FOCALES.get(estrategia, estrategia)
        for contexto, estilo, marcador, alpha in (
            (CONTEXTO_CON_HUBS, "-", "o", 1.0),
            (CONTEXTO_SIN_HUBS, "--", "s", 0.7),
        ):
            valores = [
                diagnostico["curvas_recall"][contexto][estrategia][str(k)]
                for k in ks
            ]
            eje.plot(
                ks,
                valores,
                linestyle=estilo,
                marker=marcador,
                markersize=4,
                linewidth=1.2,
                color=color,
                alpha=alpha,
                label=f"{etiqueta} ({contexto.replace('_', ' ')})",
            )
    eje.axvline(
        CONFIG["k_candidatos_rerank_top50"],
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.0,
        label="techo de candidatos top-50 (rerank)",
    )
    eje.set_xlabel("k (profundidad del ranking)")
    eje.set_ylabel("recall@k")
    eje.set_title(
        "ciclo v3 - Sprint A1: recall@k por estrategia focal "
        "(benchmark sentencia->CP)",
        fontsize=10,
    )
    eje.set_xticks(ks)
    eje.set_ylim(0.0, 1.0)
    eje.grid(True, alpha=0.3)
    eje.legend(fontsize=7, loc="lower right", ncol=2)
    figura.tight_layout()
    Path(CONFIG["ruta_figura_diagnostico_recall"]).parent.mkdir(
        parents=True, exist_ok=True
    )
    figura.savefig(
        CONFIG["ruta_figura_diagnostico_recall"],
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)


def graficar_metricas_ic(diagnostico: dict) -> None:
    """Genera la figura c3_metricas_ic.png (media e IC 95% por estrategia).

    Dos paneles (nDCG@10 y F1@10) con barras pareadas por contexto
    (sin hubs y con los alphas adoptados) y barras de error = IC 95%
    bootstrap.

    Args:
        diagnostico: Dict de construir_diagnostico_a1.
    """
    estrategias = diagnostico["config"]["estrategias_focales"]
    posiciones = np.arange(len(estrategias))
    ancho = 0.38
    figura, ejes = plt.subplots(1, 2, figsize=(12, 5))
    for eje, medida, titulo in (
        (ejes[0], "ndcg_at_10", "nDCG@10"),
        (ejes[1], "f1_at_10", "F1@10"),
    ):
        for desplazamiento, contexto, etiqueta_contexto, color in (
            (-ancho / 2, CONTEXTO_SIN_HUBS, "sin hubs", "#7f7f7f"),
            (
                ancho / 2,
                CONTEXTO_CON_HUBS,
                "con hubs (alpha adoptado)",
                "#1f77b4",
            ),
        ):
            medias, errores_inf, errores_sup = [], [], []
            for estrategia in estrategias:
                datos = diagnostico["ic_bootstrap"][contexto][estrategia][
                    medida
                ]
                media = datos["media"] if datos["media"] is not None else 0.0
                p2_5 = datos["p2_5"] if datos["p2_5"] is not None else media
                p97_5 = (
                    datos["p97_5"] if datos["p97_5"] is not None else media
                )
                medias.append(media)
                errores_inf.append(media - p2_5)
                errores_sup.append(p97_5 - media)
            eje.bar(
                posiciones + desplazamiento,
                medias,
                width=ancho,
                color=color,
                alpha=0.85,
                label=etiqueta_contexto,
                yerr=[errores_inf, errores_sup],
                capsize=3,
            )
        eje.set_xticks(posiciones)
        eje.set_xticklabels(
            [ETIQUETAS_FOCALES.get(e, e) for e in estrategias],
            rotation=15,
            ha="right",
            fontsize=8,
        )
        eje.set_ylabel(titulo)
        eje.set_title(
            f"{titulo}: media e IC 95% bootstrap (1000 remuestreos)",
            fontsize=10,
        )
        eje.set_ylim(0.0, 1.0)
        eje.grid(True, axis="y", alpha=0.3)
        eje.legend(fontsize=8)
    figura.tight_layout()
    Path(CONFIG["ruta_figura_diagnostico_ic"]).parent.mkdir(
        parents=True, exist_ok=True
    )
    figura.savefig(
        CONFIG["ruta_figura_diagnostico_ic"], dpi=150, bbox_inches="tight"
    )
    plt.close(figura)


def graficar_por_periodo(diagnostico: dict) -> None:
    """Genera la figura c3_por_periodo.png (nDCG@10 por periodo).

    Barras agrupadas por periodo y estrategia focal en el contexto con
    hubs adoptados; las barras de error son el IC 95% bootstrap del
    subconjunto (30 casos por periodo en el run final).

    Args:
        diagnostico: Dict de construir_diagnostico_a1.
    """
    periodos = [nombre for nombre, _, _ in CONFIG["periodos"]]
    estrategias = diagnostico["config"]["estrategias_focales"]
    posiciones = np.arange(len(periodos))
    ancho = 0.8 / max(len(estrategias), 1)
    figura, eje = plt.subplots(figsize=(10, 5.5))
    for posicion_estrategia, estrategia in enumerate(estrategias):
        medias, errores_inf, errores_sup = [], [], []
        for periodo in periodos:
            datos = diagnostico["por_periodo"][CONTEXTO_CON_HUBS][periodo][
                "por_estrategia"
            ][estrategia]["ndcg_at_10"]
            media = datos["media"] if datos["media"] is not None else 0.0
            p2_5 = datos["p2_5"] if datos["p2_5"] is not None else media
            p97_5 = datos["p97_5"] if datos["p97_5"] is not None else media
            medias.append(media)
            errores_inf.append(media - p2_5)
            errores_sup.append(p97_5 - media)
        eje.bar(
            posiciones - 0.4 + ancho * (posicion_estrategia + 0.5),
            medias,
            width=ancho,
            color=COLORES_FOCALES.get(estrategia, "#333333"),
            label=ETIQUETAS_FOCALES.get(estrategia, estrategia),
            yerr=[errores_inf, errores_sup],
            capsize=3,
        )
    for posicion, periodo in enumerate(periodos):
        n_casos = diagnostico["por_periodo"][CONTEXTO_CON_HUBS][periodo][
            "n_casos"
        ]
        eje.text(
            posicion,
            -0.06,
            f"n={n_casos}",
            ha="center",
            va="top",
            fontsize=8,
        )
    eje.set_xticks(posiciones)
    eje.set_xticklabels(periodos)
    eje.set_ylabel("nDCG@10")
    eje.set_xlabel("Periodo de la sentencia")
    eje.set_title(
        "ciclo v3 - Sprint A1: nDCG@10 por periodo y estrategia "
        "(con hubs adoptados; barras = IC 95% bootstrap)",
        fontsize=10,
    )
    eje.set_ylim(0.0, 1.0)
    eje.grid(True, axis="y", alpha=0.3)
    eje.legend(fontsize=8)
    figura.tight_layout()
    Path(CONFIG["ruta_figura_diagnostico_periodo"]).parent.mkdir(
        parents=True, exist_ok=True
    )
    figura.savefig(
        CONFIG["ruta_figura_diagnostico_periodo"],
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)


# ===== SECCION 8d: matriz A2 de embeddings legales (ciclo v3 - Sprint A2) =====
# La matriz A2 es ADITIVA: sin flags el script se comporta exactamente
# como en A1. Con --embedding-model y/o --k-candidatos-hybrid != 50 se
# activa (aplicar_modo_a2): se cargan los embeddings alternativos (E5
# siempre como referencia), se cachean los embeddings de pasajes por
# modelo en data/cache_emb_cp/<slug>.joblib y se comparan las estrategias
# hibridas + rerank por chunks a la profundidad pedida en dos contextos
# del prior de hubs (alpha=0 y alpha=1.5) con IC bootstrap de la media y
# bootstrap PAREADO de la diferencia contra E5@K (ADR-16: v2..v8
# intocables; salidas v9 y figura c3_a2_embeddings.png).
#
# Convenciones de prefijos (verificadas 2026-09-11 en el
# config_sentence_transformers.json y la model card de cada repo):
# - E5: "query: "/"passage: " (oficiales del modelo);
# - ALIA-MrBERT-legal, BGE-M3-Legal-Spanish y modernbert-embed-es-legal-
#   v2: prompts query/document VACIOS -> sin prefijo (se documenta en el
#   campo prefijos del JSON v9).
# El pooling lo fija cada repositorio (mean para ALIA/modernbert, CLS
# para BGE-M3) y SentenceTransformer lo respeta; los embeddings se
# normalizan (similitud coseno, igual que E5).
def slug_modelo_embedding(nombre: str) -> str:
    """Resuelve el slug corto de un modelo de embeddings.

    Acepta un slug del catalogo (e5/alia/bgem3/modernbert) o un HF id
    (tabulado o arbitrario). Para HF ids no tabulados deriva un slug
    estable de la ultima parte del id (minusculas, caracteres no
    alfanumericos a "_").

    Args:
        nombre: Slug corto o HF id del modelo.

    Returns:
        Slug corto del modelo.
    """
    if nombre in CONFIG["modelos_embedding"]:
        return nombre
    for slug, configuracion in CONFIG["modelos_embedding"].items():
        if configuracion["hf_id"] == nombre:
            return slug
    base = nombre.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


def config_modelo_embedding(nombre: str) -> dict:
    """Devuelve la configuracion de carga de un modelo de embeddings.

    Los modelos no tabulados se asumen sentence-transformers estandar
    SIN prefijos (documentado): el pooling lo fija el repositorio.

    Args:
        nombre: Slug corto o HF id del modelo.

    Returns:
        Dict con slug, hf_id, prefijo_query, prefijo_pasaje, carga y nota.
    """
    slug = slug_modelo_embedding(nombre)
    tabulado = CONFIG["modelos_embedding"].get(slug)
    if tabulado is None:
        tabulado = {
            "hf_id": nombre,
            "prefijo_query": "",
            "prefijo_pasaje": "",
            "carga": "sentence_transformers",
            "nota": (
                "Modelo no tabulado: SentenceTransformer estandar sin "
                "prefijos (verificar la model card antes de adoptar)."
            ),
        }
    return {"slug": slug, **tabulado}


def cargar_modelo_embedding(configuracion: dict, dispositivo: str) -> tuple:
    """Carga un modelo de embeddings (SentenceTransformer) en el dispositivo.

    E5 conserva la carga legacy exacta (backbone + mean pooling) para
    mantener la identidad con v2..v8; el resto se carga con
    SentenceTransformer estandar y max_seq_length = tamano_chunk (512),
    de modo que los chunks definidos por CONFIG['tamano_chunk'] quepan
    completos (el pool lo fija el repositorio del modelo).

    Args:
        configuracion: Config de config_modelo_embedding.
        dispositivo: "cuda"/"cpu" elegido por elegir_dispositivo.

    Returns:
        Tupla (modelo, tokenizer) listos para encode.
    """
    from sentence_transformers import SentenceTransformer

    if configuracion["carga"] == "e5_custom":
        from sentence_transformers import models as st_models

        backbone = st_models.Transformer(
            configuracion["hf_id"], max_seq_length=CONFIG["tamano_chunk"]
        )
        pooling = st_models.Pooling(
            backbone.get_embedding_dimension(), pooling_mode="mean"
        )
        modelo = SentenceTransformer(
            modules=[backbone, pooling], device=dispositivo
        )
    else:
        modelo = SentenceTransformer(configuracion["hf_id"], device=dispositivo)
        modelo.max_seq_length = CONFIG["tamano_chunk"]
    modelo.eval()
    return modelo, modelo.tokenizer


def ruta_cache_emb_pasajes(slug: str) -> Path:
    """Ruta de la cache de embeddings de pasajes de la CP de un modelo.

    Args:
        slug: Slug corto del modelo.

    Returns:
        Path data/cache_emb_cp/<slug>.joblib dentro del proyecto.
    """
    return Path(CONFIG["cache_emb_cp_dir"]) / f"{slug}.joblib"


def _hash_canones(canones: list[str]) -> str:
    """Huella md5 del orden y contenido de los canones indexados.

    Args:
        canones: Canones del indice activo, en orden.

    Returns:
        Digest md5 hexadecimal.
    """
    return hashlib.md5("|".join(canones).encode("utf-8")).hexdigest()


def _cache_nueva_valida(
    datos: object,
    configuracion: dict,
    pasajes: pd.DataFrame,
    hash_canones: str,
) -> np.ndarray | None:
    """Valida una cache nueva de embeddings de pasajes.

    Args:
        datos: Contenido deserializado de la cache (o None).
        configuracion: Config del modelo (hf_id, prefijos, slug).
        pasajes: DataFrame de pasajes del indice activo.
        hash_canones: md5 de los canones del indice activo.

    Returns:
        Matriz float32 (n_pasajes, dim) si la cache es compatible; None
        si falta cualquier campo, no coincide la configuracion o el
        shape no cuadra (en ese caso se recalcula).
    """
    if not isinstance(datos, dict) or datos.get("formato") != "cache_emb_cp_v1":
        return None
    if datos.get("hf_id") != configuracion["hf_id"]:
        return None
    if datos.get("prefijo_pasaje") != configuracion["prefijo_pasaje"]:
        return None
    if (
        datos.get("tamano_chunk") != CONFIG["tamano_chunk"]
        or datos.get("solape_chunk") != CONFIG["solape_chunk"]
        or datos.get("pasaje_max_palabras") != CONFIG["pasaje_max_palabras"]
    ):
        return None
    if datos.get("n_pasajes") != len(pasajes):
        return None
    if datos.get("hash_canones") != hash_canones:
        return None
    matriz = np.asarray(datos.get("matriz"))
    if matriz.ndim != 2 or matriz.shape[0] != len(pasajes):
        return None
    return matriz.astype(np.float32)


def _cache_legacy_e5(configuracion: dict, pasajes: pd.DataFrame) -> np.ndarray | None:
    """Lee la cache legacy E5 del producto (data/cache_e5_cp) si es valida.

    data/cache_e5_cp/pasajes_e5_base.joblib (Fase X del producto) guarda
    la matriz agregada de los 685 pasajes permanentes con el mismo
    modelo/chunking que este benchmark. Se valida modelo, tamano/solape
    de chunk, pasaje_max_palabras y numero de pasajes; el contenido se
    verifica ademas con un spot check contra el modelo vivo (ver
    cargar_o_calcular_matriz_pasajes).

    Args:
        configuracion: Config del modelo E5.
        pasajes: DataFrame de pasajes del indice activo.

    Returns:
        Matriz float32 (n_pasajes, dim) si la cache legacy es compatible;
        None en caso contrario.
    """
    ruta = ROOT / "data" / "cache_e5_cp" / "pasajes_e5_base.joblib"
    if not ruta.is_file():
        return None
    try:
        datos = joblib.load(ruta)
    except Exception:  # noqa: BLE001 - cache ilegible: recalcular
        return None
    if not isinstance(datos, dict):
        return None
    if datos.get("modelo") != configuracion["hf_id"]:
        return None
    if (
        datos.get("tamano_chunk") != CONFIG["tamano_chunk"]
        or datos.get("solape_chunk") != CONFIG["solape_chunk"]
        or datos.get("n_pasajes") != len(pasajes)
    ):
        return None
    matriz = np.asarray(datos.get("matriz"))
    if matriz.ndim != 2 or matriz.shape[0] != len(pasajes):
        return None
    return matriz.astype(np.float32)


def _spot_check_matriz(
    modelo: object,
    tokenizer: object,
    configuracion: dict,
    pasajes: pd.DataFrame,
    matriz: np.ndarray,
) -> bool:
    """Verifica que una matriz cacheada este alineada con los pasajes.

    Codifica 3 pasajes (primero, medio y ultimo) y compara el coseno
    contra las filas cacheadas; si alguno baja de 0.99 se considera la
    cache desalineada y se recalcula (defensa contra caches legacy sin
    hash de pasajes).

    Args:
        modelo: SentenceTransformer cargado.
        tokenizer: Tokenizer del modelo.
        configuracion: Config del modelo (prefijo de pasaje).
        pasajes: DataFrame de pasajes del indice activo.
        matriz: Matriz cacheada (n_pasajes, dim).

    Returns:
        True si los 3 pasajes de control superan coseno 0.99.
    """
    n = len(pasajes)
    indices = sorted({0, n // 2, n - 1})
    grupos = [
        chunkear_texto(pasajes["texto"].iloc[i], tokenizer) or [""]
        for i in indices
    ]
    longitudes = [len(grupo) for grupo in grupos]
    textos = [
        configuracion["prefijo_pasaje"] + chunk
        for grupo in grupos
        for chunk in grupo
    ]
    planos = _codificar_con_fallback(modelo, textos)
    vectores = media_por_grupos(planos, longitudes)
    for posicion, indice in enumerate(indices):
        referencia = matriz[indice].astype(np.float64)
        vector = vectores[posicion].astype(np.float64)
        norma = float(np.linalg.norm(referencia) * np.linalg.norm(vector))
        coseno = float(np.dot(referencia, vector) / norma) if norma > 0 else 0.0
        if coseno < 0.99:
            print(
                f"[aviso] spot check de cache fallido en pasaje {indice} "
                f"(coseno {coseno:.4f} < 0.99): se recalcula la matriz"
            )
            return False
    return True


def cargar_o_calcular_matriz_pasajes(
    modelo: object,
    tokenizer: object,
    configuracion: dict,
    pasajes: pd.DataFrame,
    canones: list[str],
) -> tuple[np.ndarray, dict]:
    """Carga o calcula la matriz de embeddings de los pasajes de la CP.

    La cache es POR MODELO en data/cache_emb_cp/<slug>.joblib y guarda la
    matriz AGREGADA (un vector unitario por pasaje: chunk-mean
    renormalizado, igual que el pipeline E5) con metadata de validacion
    (hf_id, prefijos, tamano/solape de chunk, pasaje_max_palabras,
    n_pasajes, md5 del orden de canones y dimension). Si la cache nueva
    no existe o no valida, para E5 se intenta MIGRAR la cache legacy del
    producto (validada ademas con spot check de 3 pasajes) y, si no, se
    recalcula desde cero con el modelo.

    Args:
        modelo: SentenceTransformer cargado en el dispositivo.
        tokenizer: Tokenizer del modelo.
        configuracion: Config de config_modelo_embedding.
        pasajes: DataFrame con columna texto (un pasaje por fila).
        canones: Canones del indice activo (orden del indice).

    Returns:
        Tupla (matriz float32 (n_pasajes, dim), stats) con stats de
        origen ("cache", "cache_legacy_migrada" o "calculada"), numero
        de chunks y segundos de codificacion.
    """
    slug = configuracion["slug"]
    ruta = ruta_cache_emb_pasajes(slug)
    hash_canones = _hash_canones(canones)
    stats = {
        "slug": slug,
        "hf_id": configuracion["hf_id"],
        "origen": None,
        "n_pasajes": len(pasajes),
        "n_chunks": None,
        "segundos": None,
    }
    if ruta.is_file():
        try:
            datos = joblib.load(ruta)
        except Exception:  # noqa: BLE001 - cache ilegible: recalcular
            datos = None
        matriz = _cache_nueva_valida(datos, configuracion, pasajes, hash_canones)
        if matriz is not None:
            stats["origen"] = "cache"
            print(f"[info] embeddings pasajes {slug}: cache {ruta}")
            return matriz, stats
    if slug == "e5":
        matriz = _cache_legacy_e5(configuracion, pasajes)
        if matriz is not None and _spot_check_matriz(
            modelo, tokenizer, configuracion, pasajes, matriz
        ):
            stats["origen"] = "cache_legacy_migrada"
            stats["n_chunks"] = int(
                sum(len(chunkear_texto(t, tokenizer) or [""]) for t in pasajes["texto"])
            )
            _guardar_cache_emb_pasajes(
                ruta, configuracion, pasajes, hash_canones, matriz, stats
            )
            print(
                f"[info] embeddings pasajes {slug}: cache legacy del "
                f"producto migrada a {ruta} (spot check superado)"
            )
            return matriz, stats
    inicio = time.time()
    grupos = [
        chunkear_texto(texto, tokenizer) or [""]
        for texto in pasajes["texto"].tolist()
    ]
    longitudes = [len(grupo) for grupo in grupos]
    textos = [
        configuracion["prefijo_pasaje"] + chunk
        for grupo in grupos
        for chunk in grupo
    ]
    planos = _codificar_con_fallback(modelo, textos)
    matriz = media_por_grupos(planos, longitudes)
    stats.update(
        {
            "origen": "calculada",
            "n_chunks": len(textos),
            "segundos": round(time.time() - inicio, 1),
        }
    )
    _guardar_cache_emb_pasajes(
        ruta, configuracion, pasajes, hash_canones, matriz, stats
    )
    print(
        f"[info] embeddings pasajes {slug}: calculados y cacheados en "
        f"{ruta} ({stats['segundos']} s)"
    )
    return matriz, stats


def _guardar_cache_emb_pasajes(
    ruta: Path,
    configuracion: dict,
    pasajes: pd.DataFrame,
    hash_canones: str,
    matriz: np.ndarray,
    stats: dict,
) -> None:
    """Serializa la cache nueva de embeddings de pasajes con su metadata.

    Args:
        ruta: Path de destino (<slug>.joblib).
        configuracion: Config del modelo.
        pasajes: DataFrame de pasajes (para n_pasajes).
        hash_canones: md5 de los canones del indice.
        matriz: Matriz (n_pasajes, dim) de embeddings unitarios.
        stats: Stats del calculo (n_chunks, segundos) para trazabilidad.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "formato": "cache_emb_cp_v1",
            "slug": configuracion["slug"],
            "hf_id": configuracion["hf_id"],
            "prefijo_query": configuracion["prefijo_query"],
            "prefijo_pasaje": configuracion["prefijo_pasaje"],
            "tamano_chunk": CONFIG["tamano_chunk"],
            "solape_chunk": CONFIG["solape_chunk"],
            "pasaje_max_palabras": CONFIG["pasaje_max_palabras"],
            "n_pasajes": len(pasajes),
            "hash_canones": hash_canones,
            "dim": int(matriz.shape[1]),
            "n_chunks": stats.get("n_chunks"),
            "segundos": stats.get("segundos"),
            "fecha": datetime.now(timezone.utc).isoformat(),
            "matriz": np.asarray(matriz, dtype=np.float32),
        },
        ruta,
    )


def calcular_scores_embedding(
    modelo: object,
    tokenizer: object,
    configuracion: dict,
    casos: list[dict],
    matriz_pasajes: np.ndarray,
    grupos_indices: list[np.ndarray],
) -> dict:
    """Codifica los chunks de query y calcula el score denso por articulo.

    Mismo pipeline que E5 (ADR-3/ADR-9): chunks de <= tamano_chunk tokens
    de la ventana completa, prefijo de query del catalogo, max-pooling
    entre chunks y entre pasajes del articulo. Los embeddings quedan en
    caso[f"scores_emb_<slug>"] (n_articulos,).

    Args:
        modelo: SentenceTransformer cargado.
        tokenizer: Tokenizer del modelo.
        configuracion: Config de config_modelo_embedding.
        casos: Lista de dicts con ventana y arts.
        matriz_pasajes: Matriz (n_pasajes, dim) unitaria de pasajes.
        grupos_indices: Arrays de posiciones de pasajes por articulo.

    Returns:
        Dict de stats: n_chunks_query, media_chunks_query,
        longitudes_query (por caso) y segundos.
    """
    slug = configuracion["slug"]
    inicio = time.time()
    grupos_query = [
        chunkear_texto(caso["ventana"], tokenizer) or [""] for caso in casos
    ]
    longitudes = [len(grupo) for grupo in grupos_query]
    textos = [
        configuracion["prefijo_query"] + chunk
        for grupo in grupos_query
        for chunk in grupo
    ]
    emb_query = _codificar_con_fallback(modelo, textos)
    posicion = 0
    for caso, n_chunks in zip(casos, longitudes, strict=True):
        vectores = emb_query[posicion : posicion + n_chunks]
        posicion += n_chunks
        caso[f"scores_emb_{slug}"] = scores_e5_caso(
            vectores, matriz_pasajes, grupos_indices
        )
    return {
        "n_chunks_query": len(textos),
        "media_chunks_query": (
            round(float(np.mean(longitudes)), 2) if longitudes else 0.0
        ),
        "longitudes_query": longitudes,
        "segundos": round(time.time() - inicio, 1),
    }


def ranking_con_alpha(
    caso: dict,
    estrategia: str,
    alpha: float,
    prior: dict,
    canones: list[str],
) -> list[str]:
    """Reconstruye el ranking de una estrategia con un alpha explicito.

    Generalizacion de ranking_diagnostico para la matriz A2 (alpha fijo
    por contexto, p. ej. 1.5): aplica el prior solo entre los scores
    finitos (los no candidatos de un rerank quedan en -inf y no son
    retornables) y restringe el universo del rerank a sus candidatos.

    Args:
        caso: Dict del caso con scores base (FUENTE_SCORES).
        estrategia: Nombre de la estrategia (clave de FUENTE_SCORES).
        alpha: Peso del prior de hubs (0.0 = sin hubs).
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con el vector de scores.

    Returns:
        Lista de canones ordenados por score descendente.
    """
    scores = caso[FUENTE_SCORES[estrategia]]
    if alpha:
        scores = aplicar_prior_scores(scores, canones, prior, alpha)
    if estrategia in ESTRATEGIAS_RERANK:
        # El universo del reranker son SOLO sus candidatos: los no
        # candidatos tienen -inf y no son resultados retornables (el
        # ranking de producto ademas esta truncado a 10). Restringir el
        # ranking a ellos es lo que topa recall@k para k>K.
        finitos = np.flatnonzero(np.isfinite(scores))
        orden = finitos[np.argsort(-scores[finitos], kind="stable")]
        return [canones[i] for i in orden]
    return ranking_desde_scores(scores, canones)


def bootstrap_ic_pareado(
    nuevo: np.ndarray,
    referencia: np.ndarray,
    n_remuestreos: int,
    semilla: int,
) -> dict:
    """IC 95% bootstrap PAREADO de la diferencia nuevo - referencia.

    Los dos arrays comparten el orden de casos y el mismo generador
    (semilla y tamano muestral identicos) produce la misma matriz de
    indices, de modo que el delta se calcula caso a caso: es la version
    pareada del bootstrap de casos con reemplazo (mismos remuestreos
    para todas las estrategias, requisito del gate ADR-16).

    Args:
        nuevo: Array (n,) con la medida por caso de la estrategia nueva.
        referencia: Array (n,) con la medida por caso de la referencia.
        n_remuestreos: Numero de remuestreos bootstrap (1000 en A2).
        semilla: Semilla del generador (42 en A2, reproducible).

    Returns:
        Dict {"n", "media_delta", "p2_5", "p97_5", "prob_mejora",
        "gana"}: media observada de la diferencia, IC 95% de la media
        remuestreada, fraccion de remuestreos con delta > 0 y veredicto
        "gana" (extremo inferior del IC > 0).
    """
    a = np.asarray(nuevo, dtype=np.float64)
    b = np.asarray(referencia, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"arrays pareados con formas distintas: {a.shape} != {b.shape}"
        )
    n = int(a.size)
    if n == 0:
        return {
            "n": 0,
            "media_delta": None,
            "p2_5": None,
            "p97_5": None,
            "prob_mejora": None,
            "gana": False,
        }
    deltas = a - b
    media = float(deltas.mean())
    if n == 1:
        return {
            "n": 1,
            "media_delta": media,
            "p2_5": media,
            "p97_5": media,
            "prob_mejora": 1.0 if media > 0 else 0.0,
            "gana": media > 0,
        }
    generador = np.random.default_rng(semilla)
    indices = generador.integers(0, n, size=(n_remuestreos, n))
    medias = deltas[indices].mean(axis=1)
    p2_5 = float(np.percentile(medias, 2.5))
    p97_5 = float(np.percentile(medias, 97.5))
    return {
        "n": n,
        "media_delta": media,
        "p2_5": p2_5,
        "p97_5": p97_5,
        "prob_mejora": float((medias > 0.0).mean()),
        "gana": bool(p2_5 > 0.0),
    }


def construir_estrategias_a2(modelos: list[dict], k: int) -> list[dict]:
    """Registra las estrategias hibridas de la matriz A2.

    Nombres NUEVOS: hibrido_<slug>_rerank_chunks_top<k>. Excepcion de
    compatibilidad: E5 a k=50 reutiliza la estrategia legacy
    hibrido_rerank_chunks_top50 (cross-check exacto contra v8, sin
    recomputar ni duplicar pares).

    Args:
        modelos: Lista de configs (config_modelo_embedding), E5 primero.
        k: Profundidad de candidatos de la primera etapa.

    Returns:
        Lista de entradas {slug, estrategia, clave_scores, base_hibrido,
        alias_legado, k}.
    """
    entradas = []
    for configuracion in modelos:
        slug = configuracion["slug"]
        base_hibrido = f"scores_hibrido_{slug}"
        if slug == "e5" and k == CONFIG["k_candidatos_rerank_top50"]:
            entradas.append(
                {
                    "slug": slug,
                    "estrategia": "hibrido_rerank_chunks_top50",
                    "clave_scores": FUENTE_SCORES[
                        "hibrido_rerank_chunks_top50"
                    ],
                    "base_hibrido": base_hibrido,
                    "alias_legado": True,
                    "k": k,
                }
            )
            continue
        estrategia = f"hibrido_{slug}_rerank_chunks_top{k}"
        entradas.append(
            {
                "slug": slug,
                "estrategia": estrategia,
                "clave_scores": f"scores_{estrategia}",
                "base_hibrido": base_hibrido,
                "alias_legado": False,
                "k": k,
            }
        )
    return entradas


def construir_modelos_a2(args: argparse.Namespace) -> list[dict]:
    """Construye la lista de modelos de la matriz A2 sin duplicados.

    E5 va SIEMPRE primero (estrategias legacy y referencia del IC
    pareado); --embedding-model anade modelos (repetible, orden
    preservado, deduplicado por slug).

    Args:
        args: Namespace de parsear_args.

    Returns:
        Lista de configs de config_modelo_embedding.
    """
    nombres = [CONFIG["modelo_e5"]] + list(args.embedding_model or [])
    vistos: set[str] = set()
    modelos = []
    for nombre in nombres:
        configuracion = config_modelo_embedding(nombre)
        if configuracion["slug"] in vistos:
            continue
        vistos.add(configuracion["slug"])
        modelos.append(configuracion)
    return modelos


def comparar_variantes_rerank(
    casos: list[dict],
    prior: dict,
    canones: list[str],
    estrategias: list[str],
    contextos: dict[str, float],
    referencia: str | None,
    n_bootstrap: int,
    semilla: int,
) -> dict:
    """Nucleo comun de las comparaciones de variantes de rerank (A2/A3).

    Reconstruye los rankings de cada estrategia en cada contexto del
    prior de hubs con ranking_con_alpha (alpha explicito por contexto;
    en las variantes de rerank el universo queda restringido a sus
    candidatos), calcula las medidas por caso (MEDIDAS_A2), su media con
    IC 95% bootstrap y el IC 95% bootstrap PAREADO de delta =
    estrategia - referencia (mismos remuestreos para todas: semilla y n
    identicos). Lo comparten las comparaciones A2 (embeddings) y A3
    (rerankers) para no duplicar la logica.

    Args:
        casos: Lista de dicts con arts, rankings y scores base.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.
        estrategias: Nombres de las estrategias a comparar (claves de
            FUENTE_SCORES).
        contextos: Dict {nombre_contexto: alpha} (alpha=0 sin hubs).
        referencia: Estrategia de referencia del IC pareado; None o una
            estrategia ausente desactiva el pareado.
        n_bootstrap: Numero de remuestreos bootstrap.
        semilla: Semilla del generador (compartida por todos).

    Returns:
        Dict con metricas_por_caso (medidas redondeadas a 6 decimales),
        metricas_globales (media + IC 95% por contexto y estrategia) y
        pareado_vs_referencia (delta + IC 95% + prob_mejora + "gana").
    """
    # Rankings reconstruidos una sola vez por (contexto, estrategia).
    rankings = {
        nombre: {
            estrategia: [
                ranking_con_alpha(caso, estrategia, alpha, prior, canones)
                for caso in casos
            ]
            for estrategia in estrategias
        }
        for nombre, alpha in contextos.items()
    }

    # Medidas por caso (arrays crudos) para IC de la media y pareado.
    valores: dict = {}
    for nombre in contextos:
        valores[nombre] = {}
        for estrategia in estrategias:
            filas = [
                metricas_diagnostico_caso(caso["arts"], ranking)
                for caso, ranking in zip(
                    casos, rankings[nombre][estrategia], strict=True
                )
            ]
            valores[nombre][estrategia] = {
                medida: np.asarray(
                    [fila[medida] for fila in filas], dtype=np.float64
                )
                for medida in MEDIDAS_A2
            }

    metricas_por_caso = {
        nombre: {
            estrategia: {
                medida: [
                    round(float(valor), 6)
                    for valor in valores[nombre][estrategia][medida]
                ]
                for medida in MEDIDAS_A2
            }
            for estrategia in estrategias
        }
        for nombre in contextos
    }
    metricas_globales = {
        nombre: {
            estrategia: {
                medida: bootstrap_ic_media(
                    valores[nombre][estrategia][medida], n_bootstrap, semilla
                )
                for medida in MEDIDAS_A2
            }
            for estrategia in estrategias
        }
        for nombre in contextos
    }

    # Bootstrap PAREADO contra la referencia (mismos remuestreos para
    # todas las estrategias: semilla y n identicos).
    pareado: dict = {}
    for nombre in contextos:
        pareado[nombre] = {}
        if referencia is None or referencia not in estrategias:
            continue
        for estrategia in estrategias:
            if estrategia == referencia:
                continue
            pareado[nombre][estrategia] = {
                medida: bootstrap_ic_pareado(
                    valores[nombre][estrategia][medida],
                    valores[nombre][referencia][medida],
                    n_bootstrap,
                    semilla,
                )
                for medida in MEDIDAS_A2
            }
    return {
        "metricas_por_caso": metricas_por_caso,
        "metricas_globales": metricas_globales,
        "pareado_vs_referencia": pareado,
    }


def construir_comparacion_a2(
    casos: list[dict],
    prior: dict,
    canones: list[str],
) -> dict:
    """Comparacion A2: 2 contextos + metricas por caso + IC pareado.

    Reutiliza la maquinaria del diagnostico A1 (ranking_con_alpha,
    metricas_diagnostico_caso, bootstrap_ic_media) y anade el bootstrap
    PAREADO contra la referencia E5@K. Los rankings se reconstruyen en
    los dos contextos de CONFIG['a2_contextos'] (sin_hubs alpha=0 y
    con_hubs alpha=1.5, el alpha adoptado en v8): en las variantes de
    rerank el universo queda restringido a los candidatos de la primera
    etapa (recall topado, documentado).

    Args:
        casos: Lista de dicts con arts, rankings y scores base.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.

    Returns:
        Dict serializable con version "c3-a2", config de la matriz,
        recall de candidatos por modelo, metricas_por_caso (arrays
        compactos), metricas_globales (media + IC 95%) y
        pareado_vs_referencia (media delta + IC 95% + prob_mejora).
    """
    entradas = CONFIG["a2"]["estrategias"]
    estrategias = [entrada["estrategia"] for entrada in entradas]
    if not estrategias:
        return {
            "version": "c3-a2",
            "nota": "sin estrategias A2 (la matriz requiere --rerank).",
        }
    contextos = dict(CONFIG["a2_contextos"])
    k = CONFIG["a2"]["k"]
    referencia = CONFIG["a2"]["referencia"]
    n_bootstrap = CONFIG["a2_bootstrap"]
    semilla = CONFIG["a2_seed_pareado"]

    # Nucleo compartido con A3: rankings, metricas por caso/globales e IC
    # pareado contra la referencia E5@K.
    comparacion = comparar_variantes_rerank(
        casos,
        prior,
        canones,
        estrategias,
        contextos,
        referencia,
        n_bootstrap,
        semilla,
    )

    # Recall de los candidatos de la primera etapa por modelo (techo del
    # rerank a la profundidad K); se mide SIN hubs (es la 1a etapa).
    recall_candidatos = {}
    for entrada in entradas:
        valores_recall = [
            recall_at_k(
                caso["arts"],
                ranking_desde_scores(caso[entrada["base_hibrido"]], canones),
                k,
            )
            for caso in casos
        ]
        recall_candidatos[entrada["slug"]] = {
            "estrategia_hibrida": entrada["base_hibrido"],
            "estrategia_rerank": entrada["estrategia"],
            "k_candidatos": k,
            "media": round(float(np.mean(valores_recall)), 6),
        }

    hf_por_slug = {
        configuracion["slug"]: configuracion["hf_id"]
        for configuracion in CONFIG["a2"]["modelos"]
    }
    return {
        "version": "c3-a2",
        "n_casos": len(casos),
        "k_candidatos_hybrid": k,
        "referencia": referencia,
        "contextos": contextos,
        "n_remuestreos_bootstrap": n_bootstrap,
        "semilla_bootstrap": semilla,
        "modelos": [
            {
                "slug": entrada["slug"],
                "hf_id": hf_por_slug.get(entrada["slug"]),
                "estrategia": entrada["estrategia"],
                "alias_legado": entrada["alias_legado"],
            }
            for entrada in entradas
        ],
        "recall_candidatos": recall_candidatos,
        "metricas_por_caso": comparacion["metricas_por_caso"],
        "metricas_globales": comparacion["metricas_globales"],
        "pareado_vs_referencia": comparacion["pareado_vs_referencia"],
        "nota": (
            "Sprint A2: contextos sin_hubs (alpha=0, atribucion al "
            "embedding) y con_hubs (alpha=1.5, adoptado en v8). En las "
            "estrategias de rerank el universo es el set de candidatos "
            "top-K del hibrido del modelo (recall_candidatos documenta "
            "ese techo). metricas_por_caso guarda nDCG@10 y F1@10 por "
            "caso para auditoria y para el test pareado."
        ),
        "nota_pareado": (
            "Bootstrap PAREADO de delta = estrategia_nueva - E5@K "
            "(referencia): las dos series comparten el orden de casos y "
            "el generador (seed 42, 1000 remuestreos) produce la misma "
            "matriz de indices, por lo que el IC es el de la diferencia "
            "por caso. 'gana' exige el extremo inferior del IC > 0 "
            "(criterio de ruido del gate ADR-16)."
        ),
    }


def graficar_a2_embeddings(comparacion: dict) -> None:
    """Genera figuras/c3_a2_embeddings.png (nDCG@10 por modelo/contexto).

    Barras agrupadas por estrategia (modelo@K) y contexto del prior de
    hubs; las barras de error son el IC 95% bootstrap de la media
    (bootstrap_ic_media). La ruta sale de CONFIG
    ('ruta_figura_a2_embeddings'), redirigida a /tmp en modo smoke.

    Args:
        comparacion: Dict de construir_comparacion_a2.
    """
    if "metricas_globales" not in comparacion:
        print("[aviso] sin comparacion A2: no se genera la figura c3_a2")
        return
    modelos = comparacion["modelos"]
    k = comparacion["k_candidatos_hybrid"]
    etiquetas = [
        f"{modelo['slug']}@{k}" + (" (legacy)" if modelo["alias_legado"] else "")
        for modelo in modelos
    ]
    estrategias = [modelo["estrategia"] for modelo in modelos]
    contextos = list(comparacion["contextos"].keys())
    colores = {"sin_hubs": "#4c72b0", "con_hubs": "#dd8452"}
    posiciones = np.arange(len(estrategias))
    ancho = 0.8 / max(len(contextos), 1)
    figura, eje = plt.subplots(figsize=(10.5, 5.5))
    for posicion_contexto, contexto in enumerate(contextos):
        medias, errores_inf, errores_sup = [], [], []
        for estrategia in estrategias:
            datos = comparacion["metricas_globales"][contexto][estrategia][
                "ndcg_at_10"
            ]
            media = datos["media"] if datos["media"] is not None else 0.0
            p2_5 = datos["p2_5"] if datos["p2_5"] is not None else media
            p97_5 = datos["p97_5"] if datos["p97_5"] is not None else media
            medias.append(media)
            errores_inf.append(media - p2_5)
            errores_sup.append(p97_5 - media)
        etiqueta = (
            "sin hubs (alpha=0)"
            if contexto == "sin_hubs"
            else f"hubs (alpha={comparacion['contextos'][contexto]})"
        )
        eje.bar(
            posiciones - 0.4 + ancho * (posicion_contexto + 0.5),
            medias,
            width=ancho,
            color=colores.get(contexto, "#333333"),
            label=etiqueta,
            yerr=[errores_inf, errores_sup],
            capsize=3,
        )
    eje.set_xticks(posiciones)
    eje.set_xticklabels(etiquetas, fontsize=9)
    eje.set_ylabel("nDCG@10")
    eje.set_xlabel("Embedding de la 1a etapa densa (hibrido + rerank por chunks)")
    eje.set_title(
        "ciclo v3 - Sprint A2: nDCG@10 por embedding legal y contexto de "
        "hubs (barras = IC 95% bootstrap)",
        fontsize=10,
    )
    eje.set_ylim(0.0, 1.0)
    eje.grid(True, axis="y", alpha=0.3)
    eje.legend(fontsize=9)
    figura.tight_layout()
    ruta = CONFIG["ruta_figura_a2_embeddings"]
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def redirigir_rutas_a_tmp(claves: tuple[str, ...], sufijo: str) -> None:
    """Redirige rutas de CONFIG a /tmp/opencode con un sufijo.

    Es idempotente: si la ruta ya apunta a /tmp/opencode no se vuelve a
    sufijar (evita duplicar "_smoke_smoke").

    Args:
        claves: Claves de CONFIG con rutas a redirigir.
        sufijo: Sufijo intercalado antes de la extension (p. ej. _smoke).
    """
    destino = Path("/tmp/opencode")
    for clave in claves:
        original = Path(CONFIG[clave])
        if original.parent == destino:
            continue
        CONFIG[clave] = str(
            destino / f"{original.stem}{sufijo}{original.suffix}"
        )


def aplicar_modo_a2(
    modelos: list[dict],
    k: int,
    limite: int | None,
    diagnostico: bool,
) -> None:
    """Activa la matriz A2 de embeddings (ciclo v3 - Sprint A2).

    Registra las estrategias nuevas (hibrido_<slug>_rerank_chunks_top<k>)
    en CONFIG['estrategias'], FUENTE_SCORES, ESTRATEGIAS_RANKEABLES y
    ESTRATEGIAS_RERANK (listas mutables) y redirige las salidas a los
    archivos v9 (ADR-16: v2..v8 nunca se sobrescriben). Con --limite todo
    va a /tmp/opencode con sufijo _smoke; si ademas --diagnostico esta
    activo, las salidas del diagnostico A1 tambien se redirigen (sufijo
    _a2) para no pisar el artefacto A1 congelado. La figura del prior v8
    se redirige de forma defensiva (solo se usaria con --calibrar-hubs).

    Args:
        modelos: Lista de configs (construir_modelos_a2), E5 primero.
        k: Profundidad de candidatos (--k-candidatos-hybrid).
        limite: Valor de --limite (None en el run final).
        diagnostico: True si --diagnostico esta activo.
    """
    CONFIG["a2"]["activa"] = True
    CONFIG["a2"]["k"] = k
    CONFIG["a2"]["modelos"] = [dict(configuracion) for configuracion in modelos]
    entradas = construir_estrategias_a2(modelos, k)
    CONFIG["a2"]["estrategias"] = entradas
    referencia = None
    for entrada in entradas:
        nombre = entrada["estrategia"]
        if nombre not in CONFIG["estrategias"]:
            CONFIG["estrategias"].append(nombre)
        FUENTE_SCORES[nombre] = entrada["clave_scores"]
        if nombre not in ESTRATEGIAS_RANKEABLES:
            ESTRATEGIAS_RANKEABLES.append(nombre)
        if nombre not in ESTRATEGIAS_RERANK:
            ESTRATEGIAS_RERANK.append(nombre)
        if entrada["slug"] == CONFIG["a2_slug_referencia"]:
            referencia = nombre
    CONFIG["a2"]["referencia"] = referencia
    CONFIG["version_salida"] = CONFIG["version_salida_v9"]
    CONFIG["ruta_benchmark"] = CONFIG["ruta_benchmark_v9"]
    CONFIG["ruta_por_ano"] = CONFIG["ruta_por_ano_v9"]
    CONFIG["ruta_figura"] = CONFIG["ruta_figura_v9"]
    CONFIG["ruta_figura_contaminacion"] = CONFIG["ruta_figura_contaminacion_v9"]
    rutas_diagnostico = (
        "ruta_diagnostico",
        "ruta_figura_diagnostico_recall",
        "ruta_figura_diagnostico_ic",
        "ruta_figura_diagnostico_periodo",
    )
    if diagnostico:
        redirigir_rutas_a_tmp(rutas_diagnostico, "_a2")
    if limite is not None:
        redirigir_rutas_a_tmp(
            (
                "ruta_benchmark",
                "ruta_por_ano",
                "ruta_figura",
                "ruta_figura_contaminacion",
                "ruta_figura_a2_embeddings",
            )
            + rutas_diagnostico,
            "_smoke",
        )
    else:
        redirigir_rutas_a_tmp(("ruta_figura_hubs_v8",), "_a2")


# ===== SECCION 8e: comparativa A3 de rerankers legales (ciclo v3 - Sprint A3) =====
# La comparativa A3 es ADITIVA: sin --a3 nada cambia. Fija la primera
# etapa densa en ALIA-MrBERT legal y la profundidad en 100 candidatos
# (CONFIG['a3']); sobre los MISMOS pares (chunk, pasaje) construidos una
# sola vez puntua bge (referencia), el reranker legal SINAI y la fusion
# RRF k=60 de ambos rankings. Salidas NUEVAS v10 y figura c3_a3_rerankers
# (ADR-16: v2..v9 intocables).
def scores_rrf_entre_rerankers(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    candidatos: np.ndarray,
    n_canones: int,
    rrf_k: int = 60,
) -> np.ndarray:
    """Fusiona dos rankings de rerankers con Reciprocal Rank Fusion k.

    score_rrf(i) = 1/(k + rank_a(i)) + 1/(k + rank_b(i)) con rank
    1-based del candidato i en cada ranking (orden por score descendente,
    argsort estable). La normalizacion min-max y el prior de hubs alpha
    se aplican despues, en ranking_con_alpha/aplicar_prior_scores, igual
    que en el resto de estrategias (documentado en comparacion_a3). Los
    articulos fuera del set de candidatos quedan en -inf (no son
    resultados retornables del rerank).

    Args:
        scores_a: Scores del primer reranker (n_articulos,); solo se
            leen las posiciones de candidatos.
        scores_b: Scores del segundo reranker (n_articulos,).
        candidatos: Indices de articulo del set de candidatos (top-K de
            la primera etapa).
        n_canones: Tamano del universo de articulos.
        rrf_k: Constante de RRF (60, Cormack et al. 2009).

    Returns:
        Array (n_canones,) con el score RRF en los candidatos y -inf en
        el resto.
    """
    orden_a = candidatos[np.argsort(-scores_a[candidatos], kind="stable")]
    orden_b = candidatos[np.argsort(-scores_b[candidatos], kind="stable")]
    fusion = np.zeros(n_canones, dtype=np.float64)
    for rank, indice in enumerate(orden_a, start=1):
        fusion[indice] += 1.0 / (rrf_k + rank)
    for rank, indice in enumerate(orden_b, start=1):
        fusion[indice] += 1.0 / (rrf_k + rank)
    scores = np.full(n_canones, -np.inf, dtype=np.float64)
    scores[candidatos] = fusion[candidatos]
    return scores


def construir_estrategias_a3() -> list[dict]:
    """Registra las 3 variantes de la comparativa A3.

    Nombres NUEVOS: hibrido_<slug>_rerank_chunks_top<k> (bge, referencia
    y nombre homonimo al de la matriz A2 para poder reutilizarla),
    hibrido_<slug>_rerank_sinai_top<k> (reranker legal SINAI) y
    hibrido_<slug>_rerank_rrf_top<k> (fusion RRF). El slug y la
    profundidad salen de CONFIG['a3'] (alia/100).

    Returns:
        Lista de entradas {variante, estrategia, clave_scores,
        base_hibrido, hf_id, rol}.
    """
    slug = CONFIG["a3"]["slug_embedding"]
    k = CONFIG["a3"]["k"]
    base_hibrido = f"scores_hibrido_{slug}"
    return [
        {
            "variante": "bge",
            "estrategia": f"hibrido_{slug}_rerank_chunks_top{k}",
            "clave_scores": f"scores_hibrido_{slug}_rerank_chunks_top{k}",
            "base_hibrido": base_hibrido,
            "hf_id": CONFIG["reranker"],
            "rol": "referencia",
        },
        {
            "variante": "aliarerank",
            "estrategia": f"hibrido_{slug}_rerank_sinai_top{k}",
            "clave_scores": f"scores_hibrido_{slug}_rerank_sinai_top{k}",
            "base_hibrido": base_hibrido,
            "hf_id": CONFIG["a3"]["modelo_reranker"],
            "rol": "candidato",
        },
        {
            "variante": "rrf",
            "estrategia": f"hibrido_{slug}_rerank_rrf_top{k}",
            "clave_scores": f"scores_hibrido_{slug}_rerank_rrf_top{k}",
            "base_hibrido": base_hibrido,
            "hf_id": None,
            "rol": "fusion",
        },
    ]


def construir_comparacion_a3(
    casos: list[dict],
    prior: dict,
    canones: list[str],
) -> dict:
    """Comparacion A3: 3 rerankers sobre ALIA@100 en 2 contextos + IC pareado.

    Reutiliza el nucleo comparar_variantes_rerank (el mismo de A2) y
    anade el recall de los candidatos ALIA@100 (techo estructural del
    rerank) con desglose por periodo. Referencia del IC pareado = bge.

    Args:
        casos: Lista de dicts con arts, rankings y scores base (incluye
            scores_hibrido_alia y los scores de las 3 variantes).
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.

    Returns:
        Dict serializable con version "c3-a3", config de la comparativa,
        recall de candidatos ALIA@100, metricas_por_caso, metricas_
        globales y pareado_vs_referencia (delta + IC 95% + prob_mejora).
    """
    entradas = CONFIG["a3"]["estrategias"]
    if not entradas:
        return {
            "version": "c3-a3",
            "nota": "sin estrategias A3 (la comparativa requiere --a3).",
        }
    estrategias = [entrada["estrategia"] for entrada in entradas]
    contextos = dict(CONFIG["a3"]["contextos"])
    k = CONFIG["a3"]["k"]
    referencia = CONFIG["a3"]["referencia"]
    n_bootstrap = CONFIG["a3_bootstrap"]
    semilla = CONFIG["a3_seed_pareado"]

    comparacion = comparar_variantes_rerank(
        casos,
        prior,
        canones,
        estrategias,
        contextos,
        referencia,
        n_bootstrap,
        semilla,
    )

    # Recall de los candidatos ALIA@K: techo del rerank (la 1a etapa es
    # el hibrido BM25+ALIA sin hubs); desglose por periodo como en A1.
    slug = CONFIG["a3"]["slug_embedding"]
    valores_recall = [
        recall_at_k(
            caso["arts"],
            ranking_desde_scores(caso[f"scores_hibrido_{slug}"], canones),
            k,
        )
        for caso in casos
    ]
    por_periodo_recall = {}
    for periodo, _, _ in CONFIG["periodos"]:
        indices = [
            i
            for i, caso in enumerate(casos)
            if asignar_periodo(caso["ano"]) == periodo
        ]
        por_periodo_recall[periodo] = (
            round(float(np.mean([valores_recall[i] for i in indices])), 6)
            if indices
            else None
        )
    recall_candidatos = {
        "estrategia_hibrida": f"scores_hibrido_{slug}",
        "k_candidatos": k,
        "media": round(float(np.mean(valores_recall)), 6),
        "por_periodo": por_periodo_recall,
    }

    return {
        "version": "c3-a3",
        "n_casos": len(casos),
        "k_candidatos": k,
        "rrf_k": CONFIG["a3"]["rrf_k"],
        "referencia": referencia,
        "contextos": contextos,
        "n_remuestreos_bootstrap": n_bootstrap,
        "semilla_bootstrap": semilla,
        "rerankers": [
            {
                "variante": entrada["variante"],
                "estrategia": entrada["estrategia"],
                "hf_id": entrada["hf_id"],
                "rol": entrada["rol"],
            }
            for entrada in entradas
        ],
        "reutiliza_matriz_a2": CONFIG["a3"]["reutiliza_a2"],
        "recall_candidatos": recall_candidatos,
        "metricas_por_caso": comparacion["metricas_por_caso"],
        "metricas_globales": comparacion["metricas_globales"],
        "pareado_vs_referencia": comparacion["pareado_vs_referencia"],
        "nota": (
            "Sprint A3: candidatos FIJOS = top-100 del hibrido "
            f"0.5*bm25 + 0.5*{slug} (recall_candidatos documenta el "
            "techo). Los pares (chunk, pasaje) se construyen UNA vez por "
            "caso (mejor pasaje BM25 + top-3 chunks) y se puntuan con "
            "los dos rerankers; bge reutiliza la estrategia A2 homonima "
            "si la matriz A2 ya la calculo a k=100. La variante rrf "
            "fusiona los rankings de ambos rerankers con Reciprocal "
            "Rank Fusion k=60 (score = suma de 1/(60+rank)), y su score "
            "se normaliza min-max y se prioriza con hubs alpha como el "
            "resto de estrategias (ranking_con_alpha/aplicar_prior_"
            "scores). metricas_por_caso guarda nDCG@10 y F1@10 por caso."
        ),
        "nota_pareado": (
            "Bootstrap PAREADO de delta = variante - bge (referencia): "
            "las dos series comparten el orden de casos y el generador "
            f"(seed {semilla}, {n_bootstrap} remuestreos) produce la "
            "misma matriz de indices, por lo que el IC es el de la "
            "diferencia por caso. 'gana' exige el extremo inferior del "
            "IC > 0 (criterio de ruido del gate ADR-16)."
        ),
    }


def graficar_a3_rerankers(comparacion: dict) -> None:
    """Genera figuras/c3_a3_rerankers.png (nDCG@10 y F1@10 por variante).

    Dos paneles (nDCG@10 y F1@10) con barras agrupadas por variante de
    reranker y contexto del prior de hubs (sin_hubs alpha=0 y con_hubs
    alpha=1.5); las barras de error son el IC 95% bootstrap de la media
    (bootstrap_ic_media). La ruta sale de CONFIG
    ('ruta_figura_a3_rerankers'), redirigida a /tmp en modo smoke.

    Args:
        comparacion: Dict de construir_comparacion_a3.
    """
    if "metricas_globales" not in comparacion:
        print("[aviso] sin comparacion A3: no se genera la figura c3_a3")
        return
    variantes = comparacion["rerankers"]
    estrategias = [variante["estrategia"] for variante in variantes]
    etiquetas = [
        {
            "bge": "bge (referencia)",
            "aliarerank": "ALIA reranker (SINAI)",
        }.get(variante["variante"], f"RRF k={comparacion['rrf_k']}")
        for variante in variantes
    ]
    contextos = list(comparacion["contextos"].keys())
    colores = {"sin_hubs": "#4c72b0", "con_hubs": "#dd8452"}
    posiciones = np.arange(len(estrategias))
    ancho = 0.8 / max(len(contextos), 1)
    figura, ejes = plt.subplots(1, 2, figsize=(12, 5.2))
    for eje, medida, titulo in (
        (ejes[0], "ndcg_at_10", "nDCG@10"),
        (ejes[1], "f1_at_10", "F1@10"),
    ):
        for posicion_contexto, contexto in enumerate(contextos):
            medias, errores_inf, errores_sup = [], [], []
            for estrategia in estrategias:
                datos = comparacion["metricas_globales"][contexto][
                    estrategia
                ][medida]
                media = datos["media"] if datos["media"] is not None else 0.0
                p2_5 = datos["p2_5"] if datos["p2_5"] is not None else media
                p97_5 = (
                    datos["p97_5"] if datos["p97_5"] is not None else media
                )
                medias.append(media)
                errores_inf.append(media - p2_5)
                errores_sup.append(p97_5 - media)
            etiqueta = (
                "sin hubs (alpha=0)"
                if contexto == "sin_hubs"
                else f"hubs (alpha={comparacion['contextos'][contexto]})"
            )
            eje.bar(
                posiciones - 0.4 + ancho * (posicion_contexto + 0.5),
                medias,
                width=ancho,
                color=colores.get(contexto, "#333333"),
                label=etiqueta,
                yerr=[errores_inf, errores_sup],
                capsize=3,
            )
        eje.set_xticks(posiciones)
        eje.set_xticklabels(etiquetas, fontsize=9)
        eje.set_ylabel(titulo)
        eje.set_ylim(0.0, 1.0)
        eje.grid(True, axis="y", alpha=0.3)
        eje.legend(fontsize=8)
    ejes[0].set_xlabel("Reranker sobre candidatos ALIA@100")
    ejes[1].set_xlabel("Reranker sobre candidatos ALIA@100")
    figura.suptitle(
        "ciclo v3 - Sprint A3: rerankers legales sobre candidatos ALIA@100 "
        "(barras = IC 95% bootstrap)",
        fontsize=11,
    )
    figura.tight_layout()
    ruta = CONFIG["ruta_figura_a3_rerankers"]
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def aplicar_modo_a3(limite: int | None) -> None:
    """Activa la comparativa A3 de rerankers legales (ciclo v3 - Sprint A3).

    Fija los candidatos en ALIA@100 (CONFIG['a3']) y registra las 3
    variantes (bge referencia, aliarerank SINAI y rrf k=60) en
    CONFIG['estrategias'], FUENTE_SCORES, ESTRATEGIAS_RANKEABLES y
    ESTRATEGIAS_RERANK (listas mutables). Redirige las salidas a los
    archivos v10 (ADR-16: v2..v9 nunca se sobrescriben). Si la matriz A2
    ya calculo la estrategia homonima de bge (slug alia a k=100), el
    vector bge se reutiliza y no se re-puntua (CONFIG['a3']['clave_bge']).
    Con --limite todo va a /tmp/opencode con sufijo _smoke.

    Args:
        limite: Valor de --limite (None en el run final).
    """
    CONFIG["a3"]["activa"] = True
    entradas = construir_estrategias_a3()
    CONFIG["a3"]["estrategias"] = entradas
    referencia = None
    for entrada in entradas:
        nombre = entrada["estrategia"]
        if nombre not in CONFIG["estrategias"]:
            CONFIG["estrategias"].append(nombre)
        FUENTE_SCORES[nombre] = entrada["clave_scores"]
        if nombre not in ESTRATEGIAS_RANKEABLES:
            ESTRATEGIAS_RANKEABLES.append(nombre)
        if nombre not in ESTRATEGIAS_RERANK:
            ESTRATEGIAS_RERANK.append(nombre)
        if entrada["rol"] == "referencia":
            referencia = nombre
    CONFIG["a3"]["referencia"] = referencia
    # Reutilizacion de la matriz A2: solo si A2 esta activa a la misma
    # profundidad (100) y con el mismo slug de embedding (alia).
    clave_bge = None
    if CONFIG["a2"]["activa"] and CONFIG["a2"]["k"] == CONFIG["a3"]["k"]:
        for entrada in CONFIG["a2"]["estrategias"]:
            if entrada["slug"] == CONFIG["a3"]["slug_embedding"]:
                clave_bge = entrada["clave_scores"]
                break
    CONFIG["a3"]["reutiliza_a2"] = clave_bge is not None
    CONFIG["a3"]["clave_bge"] = clave_bge
    CONFIG["version_salida"] = CONFIG["version_salida_v10"]
    CONFIG["ruta_benchmark"] = CONFIG["ruta_benchmark_v10"]
    CONFIG["ruta_por_ano"] = CONFIG["ruta_por_ano_v10"]
    CONFIG["ruta_figura"] = CONFIG["ruta_figura_v10"]
    CONFIG["ruta_figura_contaminacion"] = CONFIG["ruta_figura_contaminacion_v10"]
    if limite is not None:
        redirigir_rutas_a_tmp(
            (
                "ruta_benchmark",
                "ruta_por_ano",
                "ruta_figura",
                "ruta_figura_contaminacion",
                "ruta_figura_a3_rerankers",
            ),
            "_smoke",
        )


# ===== SECCION 8f: fusion de 1a etapa y profundidad A4 (ciclo v3 - Sprint A4) =====
# La comparativa A4 es ADITIVA: sin --a4 nada cambia. Fija el embedding
# denso en ALIA-MrBERT legal (el de mayor recall de candidatos de A2) y
# compara DOS familias con bge rerank chunks y contextos alpha=0/1.5:
# (1) la fusion de la PRIMERA ETAPA a 100 candidatos en 4 variantes:
#     w05 (referencia, hibrido min-max 0.5*BM25 + 0.5*denso; reutiliza la
#     estrategia homonima de A2/A3 si ya existe), w03 (peso BM25 0.3 /
#     denso 0.7), w07 (peso BM25 0.7 / denso 0.3) y rrf1a (Reciprocal
#     Rank Fusion k=60 entre el ranking BM25 y el ranking denso ALIA);
# (2) el barrido de profundidad top-50/100/200 de la fusion elegida
#     (CONFIG['a4']['fusion_barrido'], default w05 por revision A4: el
#     smoke n=10 de w07 no es criterio robusto). El IC bootstrap pareado
#     (1000 remuestreos,
#     seed 42) de
#     cada variante se calcula contra w05@100 (la referencia vigente).
# Salidas NUEVAS v11 y figuras/c3_a4_fusion_profundidad.png; ademas se
# vuelca la seccion comparacion_a4 a
# resultados/c3_a4_fusion_profundidad.json. Con --limite todo va a
# /tmp/opencode con sufijo _smoke (ADR-16: v2..v10 intocables).
def scores_fusion_lineal(
    scores_bm25: np.ndarray,
    scores_denso: np.ndarray,
    peso_bm25: float,
) -> np.ndarray:
    """Combina BM25 y denso con peso explicito (min-max por caso).

    Convencion A4: el nombre de la variante indica el peso de BM25
    (w05 = 0.5/0.5, w03 = BM25 0.3 / denso 0.7, w07 = BM25 0.7 / denso
    0.3); el peso del denso es 1 - peso_bm25. La normalizacion min-max
    por caso es la del hibrido pre-registrado (scores_hibridos), de modo
    que w05 reproduce exactamente el hibrido 0.5/0.5 de A2/A3.

    Args:
        scores_bm25: Scores BM25 del caso (n_articulos,).
        scores_denso: Scores densos del caso (n_articulos,).
        peso_bm25: Peso de BM25 en [0, 1] (el denso recibe 1 - peso).

    Returns:
        Array (n_articulos,) de scores fusionados.
    """
    return (
        peso_bm25 * normalizar_min_max(scores_bm25)
        + (1.0 - peso_bm25) * normalizar_min_max(scores_denso)
    )


def scores_rrf_primera_etapa(
    scores_bm25: np.ndarray,
    scores_denso: np.ndarray,
    rrf_k: int = 60,
) -> np.ndarray:
    """Fusiona los rankings de 1a etapa BM25 y denso con RRF k.

    Formula (Cormack et al. 2009): score_rrf(i) = 1/(k + rank_bm25(i)) +
    1/(k + rank_denso(i)), con rank 1-based del articulo i en cada
    ranking COMPLETO (ambos rankings cubren los mismos n articulos,
    orden por score descendente con argsort estable). EMPATES: al ser
    argsort estable, los scores iguales conservan el orden de los
    canones en el vector (indice ascendente); si el propio RRF empata
    dos articulos, el desempate final lo hace el argsort estable de
    ranking_desde_scores. Los candidatos del rerank son el top-100 del
    score RRF (CONFIG['a4']['k_referencia']).

    Args:
        scores_bm25: Scores BM25 del caso (n_articulos,).
        scores_denso: Scores densos del caso (n_articulos,).
        rrf_k: Constante de RRF (60).

    Returns:
        Array (n_articulos,) con el score RRF (mayor = mejor).
    """
    n = len(scores_bm25)
    orden_bm25 = np.argsort(-scores_bm25, kind="stable")
    orden_denso = np.argsort(-scores_denso, kind="stable")
    fusion = np.zeros(n, dtype=np.float64)
    for rank, indice in enumerate(orden_bm25, start=1):
        fusion[indice] += 1.0 / (rrf_k + rank)
    for rank, indice in enumerate(orden_denso, start=1):
        fusion[indice] += 1.0 / (rrf_k + rank)
    return fusion


def nombre_estrategia_a4(variante: str, k: int) -> str:
    """Nombre de la estrategia A4 de una fusion a profundidad k.

    w05 conserva el nombre del hibrido 0.5/0.5 de A2/A3
    (hibrido_alia_rerank_chunks_top<k>) para reutilizar el vector si la
    matriz A2/A3 ya lo calculo; el resto usa
    fusion_alia_<variante>_rerank_chunks_top<k>.

    Args:
        variante: w05, w03, w07 o rrf1a.
        k: Profundidad de candidatos de la 1a etapa.

    Returns:
        Nombre unico de la estrategia.
    """
    slug = CONFIG["a4"]["slug_embedding"]
    if variante == "w05":
        return f"hibrido_{slug}_rerank_chunks_top{k}"
    return f"fusion_{slug}_{variante}_rerank_chunks_top{k}"


def base_fusion_a4(variante: str) -> str:
    """Clave del vector de scores de 1a etapa de una fusion A4.

    Args:
        variante: w05, w03, w07 o rrf1a.

    Returns:
        Clave del caso con el vector de scores de la 1a etapa (w05 es el
        hibrido 0.5/0.5 scores_hibrido_alia; el resto, los vectores
        materializados en main: scores_fusion_alia_<variante>).
    """
    slug = CONFIG["a4"]["slug_embedding"]
    if variante == "w05":
        return f"scores_hibrido_{slug}"
    return f"scores_fusion_{slug}_{variante}"


def _entrada_a4(variante: str, k: int, familia: str) -> dict:
    """Construye la entrada de registro de una variante A4 (helper).

    Args:
        variante: w05, w03, w07 o rrf1a.
        k: Profundidad de candidatos de la 1a etapa.
        familia: "fusion_100" o "profundidad".

    Returns:
        Dict con nombre de estrategia, clave de scores, base de la
        fusion, profundidad, peso BM25, tipo de fusion y rol.
    """
    estrategia = nombre_estrategia_a4(variante, k)
    return {
        "variante": variante,
        "familia": familia,
        "estrategia": estrategia,
        "clave_scores": f"scores_{estrategia}",
        "base_fusion": base_fusion_a4(variante),
        "k": k,
        "peso_bm25": CONFIG["a4"]["pesos_fusion"].get(variante),
        "tipo": "rrf" if variante == "rrf1a" else "lineal",
        "rol": "referencia" if variante == "w05" else "candidata",
    }


def construir_estrategias_a4(fusion_barrido: str) -> list[dict]:
    """Registra las variantes de fusion @100 y el barrido de profundidad.

    Devuelve una lista DEDUPLICADA: las 4 fusiones a k=100 (w05, w03,
    w07, rrf1a) mas las profundidades 50/100/200 de la fusion elegida
    para el barrido (la profundidad 100 ya esta en la lista y no se
    duplica). Cada entrada declara su familia para separar las dos
    comparaciones: fusion_100 y profundidad.

    Args:
        fusion_barrido: Variante del barrido (w05/w03/w07/rrf1a).

    Returns:
        Lista de entradas de _entrada_a4 (fusion_100 primero).
    """
    entradas: list[dict] = []
    for variante in CONFIG["a4"]["variantes_100"]:
        entradas.append(
            _entrada_a4(variante, CONFIG["a4"]["k_referencia"], "fusion_100")
        )
    nombres = {entrada["estrategia"] for entrada in entradas}
    for k in CONFIG["a4"]["profundidades"]:
        entrada = _entrada_a4(fusion_barrido, k, "profundidad")
        if entrada["estrategia"] in nombres:
            continue
        entradas.append(entrada)
    return entradas


def construir_comparacion_a4(
    casos: list[dict],
    prior: dict,
    canones: list[str],
) -> dict:
    """Comparacion A4: fusiones @100 + barrido de profundidad + IC pareado.

    Reutiliza el nucleo comparar_variantes_rerank (el mismo de A2/A3) dos
    veces: (1) las 4 fusiones de 1a etapa a 100 candidatos y (2) el
    barrido top-50/100/200 de la fusion elegida, incluyendo w05@100 como
    baseline para que el IC pareado contra la referencia w05@100 este
    siempre disponible. Anade el recall de candidatos de 1a etapa de
    cada fusion a 50/100/200 (techo del rerank) y las convenciones de
    pesos/RRF documentadas en el JSON.

    Args:
        casos: Lista de dicts con arts, rankings y scores base.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.

    Returns:
        Dict serializable con version "c3-a4", convenciones, recall de
        candidatos, fusion_100 y profundidad (metricas_globales,
        metricas_por_caso y pareado_vs_referencia en ambas familias).
    """
    entradas = CONFIG["a4"]["estrategias"]
    if not entradas:
        return {
            "version": "c3-a4",
            "nota": "sin estrategias A4 (la comparativa requiere --a4).",
        }
    contextos = dict(CONFIG["a4"]["contextos"])
    n_bootstrap = CONFIG["a4_bootstrap"]
    semilla = CONFIG["a4_seed_pareado"]
    referencia = CONFIG["a4"]["referencia"]
    k_ref = CONFIG["a4"]["k_referencia"]

    por_estrategia = {entrada["estrategia"]: entrada for entrada in entradas}
    estrategias_100 = [
        entrada["estrategia"]
        for entrada in entradas
        if entrada["familia"] == "fusion_100"
    ]
    fusion_100 = comparar_variantes_rerank(
        casos,
        prior,
        canones,
        estrategias_100,
        contextos,
        referencia,
        n_bootstrap,
        semilla,
    )

    estrategias_prof: list[str] = []
    for nombre in [referencia] + [
        nombre_estrategia_a4(CONFIG["a4"]["fusion_barrido"], k)
        for k in CONFIG["a4"]["profundidades"]
    ]:
        if nombre not in estrategias_prof:
            estrategias_prof.append(nombre)
    profundidad = comparar_variantes_rerank(
        casos,
        prior,
        canones,
        estrategias_prof,
        contextos,
        referencia,
        n_bootstrap,
        semilla,
    )

    recall_candidatos = {}
    for variante in CONFIG["a4"]["variantes_100"]:
        base = base_fusion_a4(variante)
        recall_por_k = {}
        for k in CONFIG["a4"]["profundidades"]:
            valores = [
                recall_at_k(
                    caso["arts"], ranking_desde_scores(caso[base], canones), k
                )
                for caso in casos
            ]
            recall_por_k[str(k)] = round(float(np.mean(valores)), 6)
        recall_candidatos[variante] = {
            "estrategia_1a_etapa": base,
            "tipo": "rrf" if variante == "rrf1a" else "lineal",
            "peso_bm25": CONFIG["a4"]["pesos_fusion"].get(variante),
            "recall_por_k": recall_por_k,
        }

    def _etiquetas(nombres: list[str]) -> list[dict]:
        """Etiquetas serializables de una lista de estrategias A4."""
        return [
            {
                "variante": por_estrategia[nombre]["variante"],
                "estrategia": nombre,
                "k": por_estrategia[nombre]["k"],
                "peso_bm25": por_estrategia[nombre]["peso_bm25"],
                "tipo": por_estrategia[nombre]["tipo"],
                "rol": por_estrategia[nombre]["rol"],
            }
            for nombre in nombres
        ]

    return {
        "version": "c3-a4",
        "n_casos": len(casos),
        "referencia": referencia,
        "variante_referencia": "w05",
        "k_referencia": k_ref,
        "profundidades": list(CONFIG["a4"]["profundidades"]),
        "fusion_barrido": CONFIG["a4"]["fusion_barrido"],
        "criterio_seleccion_barrido": (
            "El barrido de profundidad usa la fusion elegida "
            f"'{CONFIG['a4']['fusion_barrido']}': default w05, la "
            "referencia pre-registrada (revision A4 2026-09-11). El "
            "smoke n=10 favorecia a w07 (sin_hubs 0.514 vs w05 0.466; "
            "con_hubs 0.587 vs w05 0.556; IC pareado vs w05 que excluye "
            "0 en ambos contextos), pero NO es un criterio robusto: los "
            "extremos inferiores son minimos (+0.004/+0.001), a n=10 no "
            "hay mitades A/B evaluables, A3 ya mostro que un lider de "
            "smoke puede revertir (RRF +0.038 a n=10 -> -0.007 a n=90) y "
            "w07 tiene menor recall de candidatos @100 (0.695 vs 0.744 "
            "de w05). El JSON reporta las 4 fusiones a @100 y el run de "
            "90 casos decide con el IC pareado completo; ml-trainer "
            "puede barrer otra fusion sin tocar el codigo con "
            "--a4-fusion {w05,w03,w07,rrf1a}."
        ),
        "pesos_fusion_bm25": dict(CONFIG["a4"]["pesos_fusion"]),
        "rrf_k": CONFIG["a4"]["rrf_k"],
        "convencion": (
            "w05/w03/w07 son fusion lineal min-max por caso: score = "
            "peso_bm25 * minmax(BM25) + (1 - peso_bm25) * minmax(denso "
            "ALIA); el nombre indica el peso de BM25. rrf1a = RRF k=60 "
            "de los rankings COMPLETOS BM25 y denso ALIA: score = "
            "1/(60+rank_bm25) + 1/(60+rank_denso), empates por argsort "
            "estable (orden de canones); candidatos = top-100 del RRF."
        ),
        "contextos": contextos,
        "n_remuestreos_bootstrap": n_bootstrap,
        "semilla_bootstrap": semilla,
        "recall_candidatos": recall_candidatos,
        "fusion_100": {
            "etiquetas": _etiquetas(estrategias_100),
            **fusion_100,
        },
        "profundidad": {
            "fusion": CONFIG["a4"]["fusion_barrido"],
            "etiquetas": _etiquetas(estrategias_prof),
            **profundidad,
        },
        "nota": (
            "Sprint A4: fusion de 1a etapa con embedding denso FIJO en "
            "ALIA-MrBERT legal y bge rerank chunks (misma derivacion de "
            "pares que v5/A2/A3: mejor pasaje BM25 + top-3 chunks). "
            "fusion_100 compara las 4 variantes a 100 candidatos; "
            "profundidad barre 50/100/200 de la fusion elegida e incluye "
            "w05@100 como baseline. recall_candidatos documenta el techo "
            "del rerank por fusion y profundidad. metricas_por_caso "
            "guarda nDCG@10 y F1@10 por caso."
        ),
        "nota_pareado": (
            "Bootstrap PAREADO de delta = variante - w05@100 (referencia "
            f"vigente, estrategia {referencia}): las dos series comparten "
            "el orden de casos y el generador "
            f"(seed {semilla}, {n_bootstrap} remuestreos) produce la "
            "misma matriz de indices, por lo que el IC es el de la "
            "diferencia por caso. 'gana' exige el extremo inferior del "
            "IC > 0 (criterio de ruido del gate ADR-16)."
        ),
    }


def graficar_a4_fusion_profundidad(comparacion: dict) -> None:
    """Genera figuras/c3_a4_fusion_profundidad.png (2 paneles).

    Panel izquierdo: nDCG@10 de las 4 fusiones de 1a etapa a 100
    candidatos (contextos sin_hubs alpha=0 y con_hubs alpha=1.5). Panel
    derecho: nDCG@10 del barrido de profundidad 50/100/200 de la fusion
    elegida (mas w05@100 como baseline si la elegida no es w05). Las
    barras de error son el IC 95% bootstrap de la media
    (bootstrap_ic_media). La ruta sale de CONFIG
    ('ruta_figura_a4_fusion_profundidad'), redirigida a /tmp en smoke.

    Args:
        comparacion: Dict de construir_comparacion_a4.
    """
    if "fusion_100" not in comparacion:
        print("[aviso] sin comparacion A4: no se genera la figura c3_a4")
        return
    contextos = list(comparacion["contextos"].keys())
    colores = {"sin_hubs": "#4c72b0", "con_hubs": "#dd8452"}
    figura, ejes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for eje, seccion, titulo_x in (
        (
            ejes[0],
            comparacion["fusion_100"],
            "Fusion de 1a etapa @100 (BM25 + denso ALIA)",
        ),
        (
            ejes[1],
            comparacion["profundidad"],
            "Profundidad de candidatos (fusion elegida)",
        ),
    ):
        etiquetas_datos = seccion["etiquetas"]
        estrategias = [fila["estrategia"] for fila in etiquetas_datos]
        etiquetas = [
            (
                f"{fila['variante']}@{fila['k']}"
                + (" (ref)" if fila["rol"] == "referencia" else "")
            )
            for fila in etiquetas_datos
        ]
        posiciones = np.arange(len(estrategias))
        ancho = 0.8 / max(len(contextos), 1)
        for posicion_contexto, contexto in enumerate(contextos):
            medias, errores_inf, errores_sup = [], [], []
            for estrategia in estrategias:
                datos = seccion["metricas_globales"][contexto][estrategia][
                    "ndcg_at_10"
                ]
                media = datos["media"] if datos["media"] is not None else 0.0
                p2_5 = datos["p2_5"] if datos["p2_5"] is not None else media
                p97_5 = (
                    datos["p97_5"] if datos["p97_5"] is not None else media
                )
                medias.append(media)
                errores_inf.append(media - p2_5)
                errores_sup.append(p97_5 - media)
            etiqueta = (
                "sin hubs (alpha=0)"
                if contexto == "sin_hubs"
                else f"hubs (alpha={comparacion['contextos'][contexto]})"
            )
            eje.bar(
                posiciones - 0.4 + ancho * (posicion_contexto + 0.5),
                medias,
                width=ancho,
                color=colores.get(contexto, "#333333"),
                label=etiqueta,
                yerr=[errores_inf, errores_sup],
                capsize=3,
            )
        eje.set_xticks(posiciones)
        eje.set_xticklabels(etiquetas, fontsize=9)
        eje.set_ylabel("nDCG@10")
        eje.set_xlabel(titulo_x)
        eje.set_ylim(0.0, 1.0)
        eje.grid(True, axis="y", alpha=0.3)
        eje.legend(fontsize=8)
    figura.suptitle(
        "ciclo v3 - Sprint A4: fusion de 1a etapa y profundidad "
        "(barras = IC 95% bootstrap)",
        fontsize=11,
    )
    figura.tight_layout()
    ruta = CONFIG["ruta_figura_a4_fusion_profundidad"]
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def volcar_comparacion_a4(comparacion: dict) -> None:
    """Vuelca la seccion comparacion_a4 a un JSON derivado trazable.

    El derivado resultados/c3_a4_fusion_profundidad.json sigue el patron
    de c3_a3_rerankers.json (A3): version, fecha, experimento, origen
    con archivo/version/md5/seccion y todo el contenido de la
    comparacion. Con --limite la ruta ya esta redirigida a /tmp/opencode
    (sufijo _smoke).

    Args:
        comparacion: Dict de construir_comparacion_a4.
    """
    ruta_origen = Path(CONFIG["ruta_benchmark"])
    try:
        archivo = str(ruta_origen.relative_to(ROOT))
    except ValueError:
        archivo = str(ruta_origen)
    contenido = {
        "version": comparacion["version"],
        "fecha": datetime.now(timezone.utc).isoformat(),
        "experimento": (
            "Ciclo v3 - Sprint A4: fusion de 1a etapa (BM25 + ALIA) y "
            "barrido de profundidad 50/100/200 con bge rerank chunks "
            f"({comparacion.get('n_casos', 0)} casos)"
        ),
        "origen": {
            "archivo": archivo,
            "version": CONFIG["version_salida"],
            "md5": hashlib.md5(ruta_origen.read_bytes()).hexdigest(),
            "seccion": "comparacion_a4",
        },
        **comparacion,
    }
    guardar_json(CONFIG["ruta_a4_fusion_profundidad"], contenido)


def aplicar_modo_a4(
    limite: int | None, fusion_barrido: str | None = None
) -> None:
    """Activa la fusion de 1a etapa y el barrido de profundidad (A4).

    Fija el embedding denso en ALIA, registra las 4 fusiones @100 y las
    profundidades 50/100/200 de la fusion elegida en CONFIG['estrategias'],
    FUENTE_SCORES, ESTRATEGIAS_RANKEABLES y ESTRATEGIAS_RERANK (listas
    mutables) y redirige las salidas a los archivos v11 (ADR-16: v2..v10
    nunca se sobrescriben). Con --limite todo va a /tmp/opencode con
    sufijo _smoke (benchmark, por_ano, figuras y derivado).

    Args:
        limite: Valor de --limite (None en el run final).
        fusion_barrido: Variante del barrido (--a4-fusion); None conserva
            el default de CONFIG (w05; revision A4: el smoke n=10 de w07
            no es criterio robusto).
    """
    CONFIG["a4"]["activa"] = True
    if fusion_barrido is not None:
        CONFIG["a4"]["fusion_barrido"] = fusion_barrido
    entradas = construir_estrategias_a4(CONFIG["a4"]["fusion_barrido"])
    CONFIG["a4"]["estrategias"] = entradas
    referencia = None
    for entrada in entradas:
        nombre = entrada["estrategia"]
        if nombre not in CONFIG["estrategias"]:
            CONFIG["estrategias"].append(nombre)
        FUENTE_SCORES[nombre] = entrada["clave_scores"]
        if nombre not in ESTRATEGIAS_RANKEABLES:
            ESTRATEGIAS_RANKEABLES.append(nombre)
        if nombre not in ESTRATEGIAS_RERANK:
            ESTRATEGIAS_RERANK.append(nombre)
        if entrada["familia"] == "fusion_100" and entrada["variante"] == "w05":
            referencia = nombre
    CONFIG["a4"]["referencia"] = referencia
    CONFIG["version_salida"] = CONFIG["version_salida_v11"]
    CONFIG["ruta_benchmark"] = CONFIG["ruta_benchmark_v11"]
    CONFIG["ruta_por_ano"] = CONFIG["ruta_por_ano_v11"]
    CONFIG["ruta_figura"] = CONFIG["ruta_figura_v11"]
    CONFIG["ruta_figura_contaminacion"] = CONFIG[
        "ruta_figura_contaminacion_v11"
    ]
    if limite is not None:
        redirigir_rutas_a_tmp(
            (
                "ruta_benchmark",
                "ruta_por_ano",
                "ruta_figura",
                "ruta_figura_contaminacion",
                "ruta_figura_a4_fusion_profundidad",
                "ruta_a4_fusion_profundidad",
            ),
            "_smoke",
        )


# ===== SECCION 8g: reranker de dominio B1 (ciclo v3 - Sprint B1) =====
# La evaluacion B1 es ADITIVA: sin --b1 nada cambia. Con --b1 se puntua
# la ruta ganadora del producto con rerank (hibrido_rerank_chunks_top50
# + prior de hubs alpha=1.5 + k=10, la misma de A4/v8) con el
# cross-encoder fine-tuneado por scripts/24_b1_entrenar_reranker.py
# (modelos/b1_reranker/) y se compara contra la MISMA ruta con el
# bge-reranker-v2-m3 de referencia mediante bootstrap PAREADO de casos
# (1000 remuestreos, seed 42; ADR-16). Salidas NUEVAS v12 y derivado
# c3_b1_reranker.json; v2..v11 quedan intactas.
def resolver_modelo_b1(modelo: str | None) -> str:
    """Resuelve el modelo del reranker B1 (ruta local, alias o HF id).

    Sin --reranker-model el modelo es la ruta local
    CONFIG['b1']['ruta_modelo'] (modelos/b1_reranker) resuelta contra
    ROOT; si el directorio no existe se lanza FileNotFoundError con la
    instruccion de entrenar primero con
    scripts/24_b1_entrenar_reranker.py (fail-fast antes de ejecutar el
    benchmark). Con --reranker-model se acepta (1) un alias del catalogo
    de scripts/lib/reranker_bge.py (p. ej. "bge" para la prueba de
    fontaneria del cableado), (2) una ruta local relativa a ROOT o
    absoluta y (3) cualquier HF id (se devuelve tal cual).

    Args:
        modelo: Valor de --reranker-model (None en el run real).

    Returns:
        Identificador del modelo listo para RerankerBGE(modelo_id=...).

    Raises:
        FileNotFoundError: Si no hay override y la ruta local del
            reranker B1 no existe (hay que entrenarlo con el script 24).
    """
    if modelo is None:
        ruta = Path(CONFIG["b1"]["ruta_modelo"])
        if not ruta.is_absolute():
            ruta = ROOT / ruta
        if not ruta.is_dir():
            raise FileNotFoundError(
                f"no existe el reranker B1 en {ruta}: entrena primero con "
                "'/home/carlos/venv/bin/python "
                "scripts/24_b1_entrenar_reranker.py --entrenar' (o pasa "
                "--reranker-model para una prueba de fontaneria del "
                "cableado)"
            )
        return str(ruta)
    from lib.reranker_bge import RERANKERS_DISPONIBLES

    if modelo in RERANKERS_DISPONIBLES:
        return RERANKERS_DISPONIBLES[modelo]
    ruta = Path(modelo)
    if not ruta.is_absolute() and (ROOT / ruta).exists():
        return str(ROOT / ruta)
    return modelo


def cargar_referencia_bge_externa(casos: list[dict]) -> dict | None:
    """Carga la referencia bge por caso desde los JSON congelados v8/v10.

    El delta pareado primario de B1 usa la referencia bge calculada en el
    MISMO run (mismos casos, candidatos y pares; protocolo identico).
    Como verificacion externa trazable, esta funcion reconstruye la
    medida por caso de la referencia (hibrido_rerank_chunks_top50 con
    bge) desde el campo top10_por_caso de los artefactos congelados
    v8/v10 cuando el archivo existe y cubre TODOS los casos de la
    muestra; los top-10 almacenados bastan para calcular nDCG@10 y
    F1@10 exactos (las medidas de la comparacion dependen solo del
    top-10). Esos artefactos se generaron con el prior en alpha=0, por
    lo que la serie externa corresponde al contexto sin_hubs. Si ningun
    archivo sirve, devuelve None y el JSON de B1 documenta el fallback.

    Args:
        casos: Casos del run actual (id y arts).

    Returns:
        Dict con archivo, version, md5, hubs_alpha, estrategia,
        contexto y medidas_por_caso (nDCG@10/F1@10), o None si no hay
        fuente valida.
    """
    estrategia = CONFIG["b1"]["estrategia_referencia"]
    for clave in ("ruta_benchmark_v8", "ruta_benchmark_v10"):
        ruta = Path(CONFIG[clave])
        if not ruta.exists():
            continue
        try:
            with ruta.open("r", encoding="utf-8") as manejador:
                datos = json.load(manejador)
        except (OSError, json.JSONDecodeError):
            continue
        top10 = datos.get("top10_por_caso") or {}
        if not all(
            str(caso["id"]) in top10
            and estrategia in top10[str(caso["id"])]
            for caso in casos
        ):
            continue
        ndcg: list[float] = []
        f1_10: list[float] = []
        for caso in casos:
            ranking = [
                entrada["id"]
                for entrada in top10[str(caso["id"])][estrategia]
            ]
            fila = metricas_diagnostico_caso(caso["arts"], ranking)
            ndcg.append(fila["ndcg_at_10"])
            f1_10.append(fila["f1_at_10"])
        return {
            "archivo": str(ruta),
            "version": datos.get("version"),
            "md5": hashlib.md5(ruta.read_bytes()).hexdigest(),
            "hubs_alpha": datos.get("hubs_alpha"),
            "contexto": "sin_hubs (alpha del artefacto congelado)",
            "estrategia": estrategia,
            "n_casos": len(casos),
            "medidas_por_caso": {
                "ndcg_at_10": ndcg,
                "f1_at_10": f1_10,
            },
        }
    return None


def construir_comparacion_b1(
    casos: list[dict],
    prior: dict,
    canones: list[str],
) -> dict:
    """Comparacion B1: reranker de dominio vs bge en la ruta ganadora.

    Evalua la ruta ganadora del producto con rerank
    (hibrido_rerank_chunks_top50 con prior de hubs alpha=1.5 y salida
    top-10) con dos rerankers: bge-reranker-v2-m3 (referencia) y el
    cross-encoder fine-tuneado en modelos/b1_reranker (script 24). Los
    dos comparten casos, candidatos (top-50 del hibrido 0.5/0.5), pares
    (chunk, pasaje) y protocolo; el delta se estima con bootstrap
    PAREADO de casos (1000 remuestreos, seed 42; criterio "gana":
    extremo inferior del IC > 0, ADR-16). El delta primario usa la
    referencia bge del MISMO run; si los artefactos congelados v8/v10
    estan disponibles, se anade la verificacion de equivalencia de la
    referencia del run contra la externa y un bootstrap pareado
    adicional de B1 contra la referencia congelada en el contexto
    alpha=0 (los unicos top-10 por caso publicados); si no, la seccion
    documenta el fallback.

    Args:
        casos: Casos con arts, rankings y scores base (incluye
            scores_hibrido_rerank_chunks_top50 y los scores de la
            estrategia B1).
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.

    Returns:
        Dict serializable con version "c3-b1", config de la comparativa,
        metricas por caso/globales por contexto, pareado primario vs la
        referencia del run, referencia externa (o None + fallback) y el
        pareado adicional contra la referencia congelada.
    """
    referencia = CONFIG["b1"]["estrategia_referencia"]
    estrategia_b1 = CONFIG["b1"]["estrategia"]
    contextos = dict(CONFIG["b1"]["contextos"])
    n_bootstrap = CONFIG["b1_bootstrap"]
    semilla = CONFIG["b1_seed_pareado"]
    comparacion = comparar_variantes_rerank(
        casos,
        prior,
        canones,
        [referencia, estrategia_b1],
        contextos,
        referencia,
        n_bootstrap,
        semilla,
    )
    externa = cargar_referencia_bge_externa(casos)
    verificacion_externa = None
    pareado_externa = None
    if externa is not None:
        medidas_ext = externa["medidas_por_caso"]
        verificacion_externa = {
            medida: bootstrap_ic_pareado(
                np.asarray(
                    comparacion["metricas_por_caso"]["sin_hubs"][
                        referencia
                    ][medida],
                    dtype=np.float64,
                ),
                np.asarray(medidas_ext[medida], dtype=np.float64),
                n_bootstrap,
                semilla,
            )
            for medida in MEDIDAS_A2
        }
        pareado_externa = {
            "contexto": "sin_hubs",
            "estrategia": estrategia_b1,
            "medidas": {
                medida: bootstrap_ic_pareado(
                    np.asarray(
                        comparacion["metricas_por_caso"]["sin_hubs"][
                            estrategia_b1
                        ][medida],
                        dtype=np.float64,
                    ),
                    np.asarray(medidas_ext[medida], dtype=np.float64),
                    n_bootstrap,
                    semilla,
                )
                for medida in MEDIDAS_A2
            },
        }
    return {
        "version": "c3-b1",
        "n_casos": len(casos),
        "estrategia_b1": estrategia_b1,
        "estrategia_referencia": referencia,
        "reranker_b1": CONFIG["b1"]["modelo"],
        "ruta_modelo_b1_por_defecto": CONFIG["b1"]["ruta_modelo"],
        "candidatos": CONFIG["b1"]["candidatos_k"],
        "k_salida": CONFIG["b1"]["k"],
        "contexto_ganador": CONFIG["b1"]["contexto_ganador"],
        "contextos": contextos,
        "n_remuestreos_bootstrap": n_bootstrap,
        "semilla_bootstrap": semilla,
        "fuente_referencia_delta_primario": (
            "run (misma corrida, mismo protocolo)"
            if externa is None
            else "run + verificacion externa v8/v10"
        ),
        "referencia_externa": externa,
        "verificacion_referencia_externa": verificacion_externa,
        "pareado_vs_referencia": comparacion["pareado_vs_referencia"],
        "pareado_vs_referencia_externa": pareado_externa,
        "metricas_por_caso": comparacion["metricas_por_caso"],
        "metricas_globales": comparacion["metricas_globales"],
        "nota": (
            "Sprint B1: la estrategia "
            f"{estrategia_b1} reutiliza los candidatos top-"
            f"{CONFIG['b1']['candidatos_k']} del hibrido 0.5*bm25 + "
            "0.5*e5 y los MISMOS pares (chunk, pasaje) que la referencia "
            f"{referencia} (bge-reranker-v2-m3); el unico cambio es el "
            "cross-encoder (modelos/b1_reranker, fine-tune del script "
            "24). El delta primario es pareado contra la referencia bge "
            "del mismo run en los contextos alpha=0/1.5; la ruta "
            "ganadora del producto es con_hubs alpha=1.5 con salida "
            "top-10. Si v8/v10 existen, su top10_por_caso reconstruye la "
            "referencia congelada (contexto alpha=0) para verificar la "
            "equivalencia de protocolo y para un pareado adicional; si "
            "no, la referencia del run es el unico origen (fallback "
            "documentado)."
        ),
        "nota_pareado": (
            "Bootstrap PAREADO de delta = B1 - referencia: las dos series "
            "comparten el orden de casos y el generador "
            f"(seed {semilla}, {n_bootstrap} remuestreos) produce la "
            "misma matriz de indices, por lo que el IC es el de la "
            "diferencia por caso. 'gana' exige el extremo inferior del "
            "IC > 0 (criterio de ruido del gate ADR-16)."
        ),
    }


def volcar_comparacion_b1(comparacion: dict) -> None:
    """Vuelca la seccion comparacion_b1 a un JSON derivado trazable.

    El derivado resultados/c3_b1_reranker.json sigue el patron de
    c3_a4_fusion_profundidad.json (A4): version, fecha, experimento,
    origen con archivo/version/md5/seccion y todo el contenido de la
    comparacion. Con --limite la ruta ya esta redirigida a /tmp/opencode
    (sufijo _smoke).

    Args:
        comparacion: Dict de construir_comparacion_b1.
    """
    ruta_origen = Path(CONFIG["ruta_benchmark"])
    try:
        archivo = str(ruta_origen.relative_to(ROOT))
    except ValueError:
        archivo = str(ruta_origen)
    contenido = {
        "version": comparacion["version"],
        "fecha": datetime.now(timezone.utc).isoformat(),
        "experimento": (
            "Ciclo v3 - Sprint B1: reranker de dominio fine-tuneado "
            "(modelos/b1_reranker) vs bge-reranker-v2-m3 en la ruta "
            "ganadora hibrido_rerank_chunks_top50 + hubs alpha=1.5 "
            f"({comparacion.get('n_casos', 0)} casos)"
        ),
        "origen": {
            "archivo": archivo,
            "version": CONFIG["version_salida"],
            "md5": hashlib.md5(ruta_origen.read_bytes()).hexdigest(),
            "seccion": "comparacion_b1",
        },
        **comparacion,
    }
    guardar_json(CONFIG["ruta_b1_reranker"], contenido)


def aplicar_modo_b1(limite: int | None, modelo: str | None) -> None:
    """Activa la evaluacion del reranker B1 (ciclo v3 - Sprint B1).

    Registra la estrategia nueva hibrido_rerank_chunks_top50_b1 (top-50
    del hibrido 0.5/0.5 puntuados con el cross-encoder B1) en
    CONFIG['estrategias'], FUENTE_SCORES, ESTRATEGIAS_RANKEABLES y
    ESTRATEGIAS_RERANK (listas mutables) y redirige las salidas a los
    archivos v12 (ADR-16: v2..v11 nunca se sobrescriben). Con --limite
    todo va a /tmp/opencode con sufijo _smoke.

    Args:
        limite: Valor de --limite (None en el run final).
        modelo: Modelo resuelto por resolver_modelo_b1 (ruta local del
            B1 o el override de --reranker-model).
    """
    CONFIG["b1"]["activa"] = True
    CONFIG["b1"]["modelo"] = modelo
    estrategia = CONFIG["b1"]["estrategia"]
    if estrategia not in CONFIG["estrategias"]:
        CONFIG["estrategias"].append(estrategia)
    FUENTE_SCORES[estrategia] = f"scores_{estrategia}"
    if estrategia not in ESTRATEGIAS_RANKEABLES:
        ESTRATEGIAS_RANKEABLES.append(estrategia)
    if estrategia not in ESTRATEGIAS_RERANK:
        ESTRATEGIAS_RERANK.append(estrategia)
    CONFIG["version_salida"] = CONFIG["version_salida_v12"]
    CONFIG["ruta_benchmark"] = CONFIG["ruta_benchmark_v12"]
    CONFIG["ruta_por_ano"] = CONFIG["ruta_por_ano_v12"]
    CONFIG["ruta_figura"] = CONFIG["ruta_figura_v12"]
    CONFIG["ruta_figura_contaminacion"] = CONFIG[
        "ruta_figura_contaminacion_v12"
    ]
    if limite is not None:
        redirigir_rutas_a_tmp(
            (
                "ruta_benchmark",
                "ruta_por_ano",
                "ruta_figura",
                "ruta_figura_contaminacion",
                "ruta_b1_reranker",
            ),
            "_smoke",
        )


# ===== SECCION 8h: retriever denso de dominio B2 (ciclo v3 - Sprint B2) =====
# La evaluacion B2 es ADITIVA: sin --b2 nada cambia. Con --b2 se evalua
# la ruta hibrido_b2_rerank_chunks_top50_b1: el hibrido pre-registrado
# 0.5*bm25 + 0.5*B2 (retriever denso fine-tuneado por
# scripts/25_b2_retriever_denso.py, modelos/b2_retriever/) sobre los
# MISMOS top-50 candidatos y pares (chunk, pasaje) que la ruta B1, con el
# reranker B1 (modelos/b1_reranker) y prior de hubs alpha=1.5. El delta
# primario es pareado contra la ruta B1 con hibrido E5 congelada en v12
# (comparacion_b1.metricas_por_caso / top10_por_caso) y la verificacion
# externa es ALIA@100 del v9 (comparacion_a2.metricas_por_caso);
# bootstrap pareado de casos (1000 remuestreos, seed 42; ADR-16). Salidas
# NUEVAS v13 y derivado c3_b2_retriever.json; v2..v12 quedan intactas.
def resolver_modelo_b2(
    modelo: str | None, variante: str | None = None
) -> str:
    """Resuelve el retriever denso B2 (ruta local, variante u override).

    Sin --embedding-model el modelo es la ruta local
    CONFIG['b2']['ruta_modelo'] (modelos/b2_retriever) resuelta contra
    ROOT; con --b2-variante SLUG el modelo es la ruta de la variante
    CONFIG['b2']['ruta_modelo_variante'] formateada con el slug
    (modelos/b2_retriever_<SLUG>, p. ej. modelos/b2_retriever_ep2). En
    ambos casos, si el directorio no existe se lanza FileNotFoundError
    con la instruccion de entrenar primero con
    scripts/25_b2_retriever_denso.py --entrenar (fail-fast antes de
    ejecutar el benchmark). Con --embedding-model se acepta (1) un slug
    del catalogo de embeddings de A2 (e5/alia/bgem3/modernbert), (2) una
    ruta local relativa a ROOT o absoluta y (3) cualquier HF id; ese
    override es SOLO para pruebas de fontaneria del cableado y no se
    combina con --b2-variante (validado en parsear_args).

    Args:
        modelo: Valor de --embedding-model (None en el run real).
        variante: Valor de --b2-variante (None sin variante).

    Returns:
        Identificador del modelo listo para config_modelo_embedding.

    Raises:
        FileNotFoundError: Si no hay override y la ruta local del
            retriever denso B2 (o de su variante) no existe (hay que
            entrenarlo con el script 25).
    """
    if modelo is None:
        if variante is not None:
            ruta = Path(
                CONFIG["b2"]["ruta_modelo_variante"].format(slug=variante)
            )
            if not ruta.is_absolute():
                ruta = ROOT / ruta
            if not ruta.is_dir():
                raise FileNotFoundError(
                    f"no existe la variante B2 '{variante}' en {ruta}: "
                    "entrena el retriever alternativo y guardalo en ese "
                    "directorio (p. ej. la variante ep2 del script 25) o "
                    "ejecuta sin --b2-variante para usar "
                    "modelos/b2_retriever"
                )
            return str(ruta)
        ruta = Path(CONFIG["b2"]["ruta_modelo"])
        if not ruta.is_absolute():
            ruta = ROOT / ruta
        if not ruta.is_dir():
            raise FileNotFoundError(
                f"no existe el retriever denso B2 en {ruta}: entrena "
                "primero con '/home/carlos/venv/bin/python "
                "scripts/25_b2_retriever_denso.py --entrenar' (o pasa "
                "--embedding-model para una prueba de fontaneria del "
                "cableado)"
            )
        return str(ruta)
    ruta = Path(modelo)
    if not ruta.is_absolute() and (ROOT / ruta).exists():
        return str(ROOT / ruta)
    return modelo


def nombre_retriever_b2() -> str:
    """Nombre corto del retriever B2 activo para notas y descripciones.

    Devuelve la ruta del modelo activo relativa a ROOT cuando esta dentro
    del proyecto (p. ej. "modelos/b2_retriever" o
    "modelos/b2_retriever_ep2") y la ruta tal cual cuando el override de
    fontaneria apunta fuera del proyecto. Solo debe llamarse con la
    evaluacion B2 activa (CONFIG['b2']['modelo'] no nulo).

    Returns:
        Ruta corta del modelo denso B2 activo.
    """
    ruta = Path(CONFIG["b2"]["modelo"])
    try:
        return str(ruta.relative_to(ROOT))
    except ValueError:
        return str(ruta)


def _indices_por_id_muestra(datos: dict, casos: list[dict]) -> list[int] | None:
    """Alinea los casos del run con la muestra de un benchmark congelado.

    Los benchmarks v9/v12 guardan el campo muestra con el id de cada caso
    en el orden de sus arrays de metricas_por_caso; esta funcion localiza
    la posicion de cada caso del run actual en ese orden (soporta runs
    recortados con --limite y cambios de orden). Devuelve None si el
    archivo no tiene muestra o si algun caso no aparece en ella (en ese
    caso la fuente no sirve como referencia y se documenta el fallback).

    Args:
        datos: JSON del benchmark congelado (debe incluir muestra).
        casos: Casos del run actual (id).

    Returns:
        Lista de posiciones (una por caso, en el orden del run) en la
        muestra del archivo, o None si no hay alineacion completa.
    """
    muestra = datos.get("muestra")
    if not isinstance(muestra, list):
        return None
    posicion = {
        str(fila.get("id")): indice
        for indice, fila in enumerate(muestra)
        if isinstance(fila, dict)
    }
    indices = [posicion.get(str(caso["id"])) for caso in casos]
    if any(indice is None for indice in indices):
        return None
    return [int(indice) for indice in indices]


def _medidas_alineadas(
    metricas_por_caso: dict,
    contexto: str,
    estrategia: str,
    indices: list[int],
) -> dict | None:
    """Extrae las medidas por caso de una estrategia alineadas por muestra.

    Args:
        metricas_por_caso: Seccion metricas_por_caso del benchmark
            congelado ({contexto: {estrategia: {medida: [valores]}}}).
        contexto: Nombre del contexto (sin_hubs/con_hubs).
        estrategia: Estrategia de la que se extraen las medidas.
        indices: Posiciones de los casos del run en el archivo.

    Returns:
        Dict {medida: [valores por caso]} (MEDIDAS_A2) o None si el
        contexto/estrategia/medida no existe o el indice desborda.
    """
    valores = (metricas_por_caso.get(contexto) or {}).get(estrategia)
    if not valores:
        return None
    try:
        return {
            medida: [float(valores[medida][i]) for i in indices]
            for medida in MEDIDAS_A2
        }
    except (KeyError, IndexError, TypeError):
        return None


def cargar_referencia_b2_v12(casos: list[dict]) -> dict | None:
    """Carga la referencia B1 del v12 por caso (delta primario de B2).

    La referencia primaria de B2 es la ruta B1 con hibrido E5 congelada
    en v12 (hibrido_rerank_chunks_top50_b1). Se reconstruye por caso
    desde el campo comparacion_b1.metricas_por_caso del v12 (contextos
    alpha=0/1.5, alineado por el campo muestra). Si esa seccion no
    estuviera disponible, se usa el fallback de top10_por_caso (ranking
    crudo: solo contexto sin_hubs). Si ninguna fuente cubre TODOS los
    casos del run, devuelve None y la comparacion documenta el fallback
    (sin pareado primario).

    Args:
        casos: Casos del run actual (id y arts).

    Returns:
        Dict con archivo, version, indice, md5, estrategia, contextos,
        origen_datos, n_casos y medidas_por_caso ({contexto: {medida:
        [valores]}}), o None si no hay fuente valida.
    """
    ruta = Path(CONFIG["ruta_benchmark_v12"])
    if not ruta.exists():
        return None
    try:
        with ruta.open("r", encoding="utf-8") as manejador:
            datos = json.load(manejador)
    except (OSError, json.JSONDecodeError):
        return None
    indices = _indices_por_id_muestra(datos, casos)
    if indices is None:
        return None
    estrategia = CONFIG["b2"]["estrategia_referencia_v12"]
    comparacion = datos.get("comparacion_b1") or {}
    metricas_por_caso = comparacion.get("metricas_por_caso") or {}
    medidas_por_caso: dict = {}
    for contexto in CONFIG["b2"]["contextos"]:
        medidas = _medidas_alineadas(
            metricas_por_caso, contexto, estrategia, indices
        )
        if medidas is not None:
            medidas_por_caso[contexto] = medidas
    if medidas_por_caso:
        return {
            "archivo": str(ruta),
            "version": datos.get("version"),
            "indice": datos.get("indice"),
            "md5": hashlib.md5(ruta.read_bytes()).hexdigest(),
            "estrategia": estrategia,
            "contextos": list(medidas_por_caso.keys()),
            "origen_datos": (
                "comparacion_b1.metricas_por_caso (alineado por muestra)"
            ),
            "n_casos": len(casos),
            "medidas_por_caso": medidas_por_caso,
        }
    top10 = datos.get("top10_por_caso") or {}
    if all(
        str(caso["id"]) in top10 and estrategia in top10[str(caso["id"])]
        for caso in casos
    ):
        ndcg: list[float] = []
        f1_10: list[float] = []
        for caso in casos:
            ranking = [
                entrada["id"]
                for entrada in top10[str(caso["id"])][estrategia]
            ]
            fila = metricas_diagnostico_caso(caso["arts"], ranking)
            ndcg.append(fila["ndcg_at_10"])
            f1_10.append(fila["f1_at_10"])
        return {
            "archivo": str(ruta),
            "version": datos.get("version"),
            "indice": datos.get("indice"),
            "md5": hashlib.md5(ruta.read_bytes()).hexdigest(),
            "estrategia": estrategia,
            "contextos": [CONTEXTO_SIN_HUBS],
            "origen_datos": (
                "top10_por_caso (contexto sin_hubs, ranking crudo)"
            ),
            "n_casos": len(casos),
            "medidas_por_caso": {
                CONTEXTO_SIN_HUBS: {
                    "ndcg_at_10": ndcg,
                    "f1_at_10": f1_10,
                }
            },
        }
    return None


def cargar_referencia_externa_b2(casos: list[dict]) -> dict | None:
    """Carga la referencia externa A2/ALIA (v9) por caso de B2.

    Verificacion externa de B2: la ruta ALIA@100 con bge rerank
    (hibrido_alia_rerank_chunks_top100) congelada en v9, reconstruida por
    caso desde comparacion_a2.metricas_por_caso (contextos alpha=0/1.5,
    alineada por el campo muestra). Si el v9 no existe, no tiene la
    seccion o no cubre TODOS los casos del run, devuelve None y la
    comparacion documenta el fallback (sin verificacion externa).

    Args:
        casos: Casos del run actual (id).

    Returns:
        Dict con archivo, version, indice, md5, estrategia, slug y k del
        embedding, contextos, origen_datos, n_casos y medidas_por_caso,
        o None si no hay fuente valida.
    """
    ruta = Path(CONFIG["ruta_benchmark_v9"])
    if not ruta.exists():
        return None
    try:
        with ruta.open("r", encoding="utf-8") as manejador:
            datos = json.load(manejador)
    except (OSError, json.JSONDecodeError):
        return None
    indices = _indices_por_id_muestra(datos, casos)
    if indices is None:
        return None
    externa = CONFIG["b2"]["referencia_externa"]
    estrategia = externa["estrategia"]
    comparacion = datos.get("comparacion_a2") or {}
    metricas_por_caso = comparacion.get("metricas_por_caso") or {}
    medidas_por_caso: dict = {}
    for contexto in CONFIG["b2"]["contextos"]:
        medidas = _medidas_alineadas(
            metricas_por_caso, contexto, estrategia, indices
        )
        if medidas is not None:
            medidas_por_caso[contexto] = medidas
    if not medidas_por_caso:
        return None
    return {
        "archivo": str(ruta),
        "version": datos.get("version"),
        "indice": datos.get("indice"),
        "md5": hashlib.md5(ruta.read_bytes()).hexdigest(),
        "estrategia": estrategia,
        "slug_embedding": externa["slug"],
        "k_candidatos": externa["k"],
        "contextos": list(medidas_por_caso.keys()),
        "origen_datos": (
            "comparacion_a2.metricas_por_caso (alineado por muestra)"
        ),
        "n_casos": len(casos),
        "medidas_por_caso": medidas_por_caso,
    }


def construir_comparacion_b2(
    casos: list[dict],
    prior: dict,
    canones: list[str],
) -> dict:
    """Comparacion B2: retriever denso de dominio en la ruta B1.

    Evalua la estrategia hibrido_b2_rerank_chunks_top50_b1 (hibrido
    0.5*bm25 + 0.5*B2 sobre top-50 candidatos con el reranker B1 y prior
    de hubs) en los contextos alpha=0/1.5 con metricas por caso/globales
    (comparar_variantes_rerank) y calcula el IC 95% bootstrap PAREADO
    (1000 remuestreos, seed 42; criterio "gana": extremo inferior del
    IC > 0, ADR-16) de delta = B2 - referencia v12 (ruta B1 con hibrido
    E5 congelada, per-case de comparacion_b1.metricas_por_caso; fallback
    top10_por_caso en sin_hubs; fallback final documentado) y de la
    verificacion externa vs A2/ALIA@100 (v9 comparacion_a2) cuando esta
    disponible.

    Args:
        casos: Casos con arts, rankings y scores base (incluye
            scores_hibrido_b2_* y los scores de la estrategia B2).
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.

    Returns:
        Dict serializable con version "c3-b2", config de la comparativa,
        metricas por caso/globales por contexto, referencia v12 (o None +
        fallback) con el pareado primario, y referencia externa A2/ALIA
        (o None + fallback) con su pareado.
    """
    estrategia = CONFIG["b2"]["estrategia"]
    variante_b2 = CONFIG["b2"]["variante"]
    etiqueta_retriever = nombre_retriever_b2() + (
        f", variante '{variante_b2}'" if variante_b2 is not None else ""
    )
    contextos = dict(CONFIG["b2"]["contextos"])
    n_bootstrap = CONFIG["b2_bootstrap"]
    semilla = CONFIG["b2_seed_pareado"]
    comparacion = comparar_variantes_rerank(
        casos,
        prior,
        canones,
        [estrategia],
        contextos,
        None,
        n_bootstrap,
        semilla,
    )
    referencia_v12 = cargar_referencia_b2_v12(casos)
    pareado_v12 = None
    if referencia_v12 is not None:
        pareado_v12 = {
            contexto: {
                medida: bootstrap_ic_pareado(
                    np.asarray(
                        comparacion["metricas_por_caso"][contexto][
                            estrategia
                        ][medida],
                        dtype=np.float64,
                    ),
                    np.asarray(medidas[medida], dtype=np.float64),
                    n_bootstrap,
                    semilla,
                )
                for medida in MEDIDAS_A2
            }
            for contexto, medidas in referencia_v12[
                "medidas_por_caso"
            ].items()
        }
    referencia_externa = cargar_referencia_externa_b2(casos)
    pareado_externa = None
    if referencia_externa is not None:
        pareado_externa = {
            contexto: {
                medida: bootstrap_ic_pareado(
                    np.asarray(
                        comparacion["metricas_por_caso"][contexto][
                            estrategia
                        ][medida],
                        dtype=np.float64,
                    ),
                    np.asarray(medidas[medida], dtype=np.float64),
                    n_bootstrap,
                    semilla,
                )
                for medida in MEDIDAS_A2
            }
            for contexto, medidas in referencia_externa[
                "medidas_por_caso"
            ].items()
        }
    return {
        "version": "c3-b2",
        "n_casos": len(casos),
        "estrategia_b2": estrategia,
        "estrategia_referencia_v12": CONFIG["b2"][
            "estrategia_referencia_v12"
        ],
        "retriever_b2": CONFIG["b2"]["modelo"],
        "slug_b2": CONFIG["b2"]["slug"],
        "variante_b2": variante_b2,
        "ruta_modelo_b2_por_defecto": CONFIG["b2"]["ruta_modelo"],
        "reranker_b2": CONFIG["b2"]["modelo_reranker"],
        "candidatos": CONFIG["b2"]["candidatos_k"],
        "k_salida": CONFIG["b2"]["k"],
        "contexto_ganador": CONFIG["b2"]["contexto_ganador"],
        "contextos": contextos,
        "n_remuestreos_bootstrap": n_bootstrap,
        "semilla_bootstrap": semilla,
        "fuente_referencia_delta_primario": (
            "v12 (comparacion_b1 congelada)"
            if referencia_v12 is not None
            else "no disponible (fallback: sin pareado primario)"
        ),
        "referencia_v12": referencia_v12,
        "pareado_vs_referencia_v12": pareado_v12,
        "referencia_externa_a2": referencia_externa,
        "pareado_vs_referencia_externa_a2": pareado_externa,
        "metricas_por_caso": comparacion["metricas_por_caso"],
        "metricas_globales": comparacion["metricas_globales"],
        "nota": (
            "Sprint B2: la estrategia "
            f"{estrategia} usa el retriever denso de dominio "
            f"({etiqueta_retriever}, fine-tune del script 25) como "
            "componente denso del hibrido pre-registrado 0.5*bm25 + "
            "0.5*B2 (top-50 candidatos) y el reranker B1 "
            "(modelos/b1_reranker) sobre los MISMOS pares (chunk, "
            "pasaje) que la ruta B1 del producto (mejor pasaje BM25 + "
            "top-3 chunks + max-pooling); la ruta de producto es "
            "con_hubs alpha=1.5 con salida top-10. El delta primario es "
            "pareado contra la ruta B1 con hibrido E5 congelada en v12 "
            "(comparacion_b1.metricas_por_caso; si falta, top10_por_caso "
            "en el contexto sin_hubs; si no hay fuente, fallback "
            "documentado sin pareado primario). La verificacion externa "
            "compara ademas contra A2/ALIA@100 (v9 comparacion_a2."
            "metricas_por_caso) cuando esta disponible."
        ),
        "nota_pareado": (
            "Bootstrap PAREADO de delta = B2 - referencia: las dos "
            "series comparten el orden de casos y el generador "
            f"(seed {semilla}, {n_bootstrap} remuestreos) produce la "
            "misma matriz de indices, por lo que el IC es el de la "
            "diferencia por caso. 'gana' exige el extremo inferior del "
            "IC > 0 (criterio de ruido del gate ADR-16)."
        ),
    }


def volcar_comparacion_b2(comparacion: dict) -> None:
    """Vuelca la seccion comparacion_b2 a un JSON derivado trazable.

    El derivado resultados/c3_b2_retriever.json sigue el patron de
    c3_b1_reranker.json (B1): version, fecha, experimento, origen con
    archivo/version/md5/seccion y todo el contenido de la comparacion.
    Con --limite la ruta ya esta redirigida a /tmp/opencode (sufijo
    _smoke).

    Args:
        comparacion: Dict de construir_comparacion_b2.
    """
    ruta_origen = Path(CONFIG["ruta_benchmark"])
    try:
        archivo = str(ruta_origen.relative_to(ROOT))
    except ValueError:
        archivo = str(ruta_origen)
    variante_b2 = CONFIG["b2"]["variante"]
    contenido = {
        "version": comparacion["version"],
        "fecha": datetime.now(timezone.utc).isoformat(),
        "experimento": (
            "Ciclo v3 - Sprint B2: retriever denso de dominio "
            f"({nombre_retriever_b2()}, script 25)"
            + (
                f" - variante '{variante_b2}'"
                if variante_b2 is not None
                else ""
            )
            + " en la 1a etapa del "
            "hibrido 0.5/0.5 top-50 con el reranker B1 (modelos/"
            "b1_reranker) y hubs alpha=1.5 "
            f"({comparacion.get('n_casos', 0)} casos)"
        ),
        "origen": {
            "archivo": archivo,
            "version": CONFIG["version_salida"],
            "md5": hashlib.md5(ruta_origen.read_bytes()).hexdigest(),
            "seccion": "comparacion_b2",
        },
        **comparacion,
    }
    guardar_json(CONFIG["ruta_b2_retriever"], contenido)


def calcular_denso_b2(
    casos: list[dict],
    pasajes: pd.DataFrame,
    canones: list[str],
    grupos_indices: list[np.ndarray],
    dispositivo: str,
) -> dict:
    """Calcula los scores densos del retriever B2 y su hibrido con BM25.

    Reutiliza el plumbing de A2: carga generica SentenceTransformer
    (config_modelo_embedding), matriz de pasajes con cache por slug en
    data/cache_emb_cp/<slug>.joblib (cargar_o_calcular_matriz_pasajes) y
    score denso por articulo con max-pooling (calcular_scores_embedding);
    materializa ademas el hibrido pre-registrado 0.5*bm25 + 0.5*B2 en
    caso[scores_hibrido_<slug>], que es la base de candidatos del rerank
    B2. El modelo se libera al terminar (la matriz queda cacheada).

    Args:
        casos: Casos con scores_bm25 y ventana.
        pasajes: DataFrame de pasajes del indice activo.
        canones: Canones alineados con los vectores de scores.
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        dispositivo: "cuda"/"cpu" elegido por elegir_dispositivo.

    Returns:
        Dict de stats del embedding (hf_id, slug, prefijos, dim,
        n_pasajes, origen de la cache de pasajes, chunks y tiempos) para
        el JSON del benchmark.
    """
    configuracion = config_modelo_embedding(CONFIG["b2"]["modelo"])
    modelo, tokenizer = cargar_modelo_embedding(configuracion, dispositivo)
    matriz, stats_pasajes = cargar_o_calcular_matriz_pasajes(
        modelo, tokenizer, configuracion, pasajes, canones
    )
    stats_query = calcular_scores_embedding(
        modelo,
        tokenizer,
        configuracion,
        casos,
        matriz,
        grupos_indices,
    )
    slug = configuracion["slug"]
    for caso in casos:
        caso[f"scores_hibrido_{slug}"] = scores_hibridos(
            caso["scores_bm25"], caso[f"scores_emb_{slug}"]
        )
    print(
        f"Embedding B2 {slug}: dim={matriz.shape[1]} | pasajes "
        f"{stats_pasajes['origen']} | {stats_query['n_chunks_query']} "
        f"chunks de query en {stats_query['segundos']} s ({dispositivo})"
    )
    del modelo
    if dispositivo == "cuda":
        torch.cuda.empty_cache()
    return {
        "hf_id": configuracion["hf_id"],
        "slug": slug,
        "prefijo_query": configuracion["prefijo_query"],
        "prefijo_pasaje": configuracion["prefijo_pasaje"],
        "dim": int(matriz.shape[1]),
        "n_pasajes": int(matriz.shape[0]),
        "origen_pasajes": stats_pasajes["origen"],
        "n_chunks_pasajes": stats_pasajes["n_chunks"],
        "segundos_pasajes": stats_pasajes["segundos"],
        "n_chunks_query": stats_query["n_chunks_query"],
        "media_chunks_query": stats_query["media_chunks_query"],
        "segundos_query": stats_query["segundos"],
    }


def aplicar_modo_b2(
    limite: int | None,
    modelo_denso: str,
    modelo_reranker: str,
    variante: str | None = None,
) -> None:
    """Activa la evaluacion del retriever denso B2 (ciclo v3 - Sprint B2).

    Registra la estrategia nueva hibrido_b2_rerank_chunks_top50_b1
    (hibrido 0.5/0.5 con el denso B2, top-50 candidatos, rerank por
    chunks con el reranker B1) en CONFIG['estrategias'], FUENTE_SCORES,
    ESTRATEGIAS_RANKEABLES y ESTRATEGIAS_RERANK (listas mutables) y
    redirige las salidas a los archivos v13 (ADR-16: v2..v12 nunca se
    sobrescriben).

    Con --b2-variante SLUG (ciclo v3 - iteracion v14) todas las salidas
    pasan a las plantillas v14 nombradas por slug de
    CONFIG['b2_variante_v14']: version v14-ciclo3-b2-<slug>, benchmark
    vinculo_cp_benchmark_v14_<slug>.json, por_ano v14_<slug>.csv,
    figuras v14_<slug>.png y derivado c3_b2_retriever_<slug>.json, con
    la nota v14 activa en CONFIG['nota_v14'] (v2..v13 quedan intactas).
    Con --limite las salidas estandar se redirigen a /tmp/opencode con
    sufijo _smoke (convencion de smokes); excepcion documentada de la
    variante: los slugs de fontaneria "smoke*" conservan las rutas v14
    (evidencia pedida de la fontaneria, sin colision con v2..v13) y el
    resto de slugs con --limite si se redirigen para no dejar un
    benchmark parcial en la ruta final de la variante. Sin --limite la
    figura de calibracion de hubs v8 se redirige de forma defensiva
    (solo se usaria con --calibrar-hubs) para no tocar artefactos
    previos.

    Args:
        limite: Valor de --limite (None en el run final).
        modelo_denso: Retriever denso resuelto por resolver_modelo_b2
            (ruta local de modelos/b2_retriever o su variante, o el
            override de --embedding-model).
        modelo_reranker: Reranker B1 resuelto por resolver_modelo_b1
            (ruta local modelos/b1_reranker o el override de
            --reranker-model).
        variante: Valor de --b2-variante (None sin variante).
    """
    CONFIG["b2"]["activa"] = True
    CONFIG["b2"]["modelo"] = modelo_denso
    CONFIG["b2"]["modelo_reranker"] = modelo_reranker
    CONFIG["b2"]["variante"] = variante
    configuracion = config_modelo_embedding(modelo_denso)
    CONFIG["b2"]["slug"] = configuracion["slug"]
    CONFIG["b2"]["clave_hibrido"] = (
        f"scores_hibrido_{configuracion['slug']}"
    )
    estrategia = CONFIG["b2"]["estrategia"]
    if estrategia not in CONFIG["estrategias"]:
        CONFIG["estrategias"].append(estrategia)
    FUENTE_SCORES[estrategia] = f"scores_{estrategia}"
    if estrategia not in ESTRATEGIAS_RANKEABLES:
        ESTRATEGIAS_RANKEABLES.append(estrategia)
    if estrategia not in ESTRATEGIAS_RERANK:
        ESTRATEGIAS_RERANK.append(estrategia)
    if variante is not None:
        plantillas = CONFIG["b2_variante_v14"]
        CONFIG["version_salida"] = plantillas["version_salida"].format(
            slug=variante
        )
        CONFIG["ruta_benchmark"] = plantillas["ruta_benchmark"].format(
            slug=variante
        )
        CONFIG["ruta_por_ano"] = plantillas["ruta_por_ano"].format(
            slug=variante
        )
        CONFIG["ruta_figura"] = plantillas["ruta_figura"].format(
            slug=variante
        )
        CONFIG["ruta_figura_contaminacion"] = plantillas[
            "ruta_figura_contaminacion"
        ].format(slug=variante)
        CONFIG["ruta_b2_retriever"] = plantillas["ruta_b2_retriever"].format(
            slug=variante
        )
        CONFIG["nota_v14"] = plantillas["nota"].format(
            slug=variante, modelo=modelo_denso
        )
    else:
        CONFIG["version_salida"] = CONFIG["version_salida_v13"]
        CONFIG["ruta_benchmark"] = CONFIG["ruta_benchmark_v13"]
        CONFIG["ruta_por_ano"] = CONFIG["ruta_por_ano_v13"]
        CONFIG["ruta_figura"] = CONFIG["ruta_figura_v13"]
        CONFIG["ruta_figura_contaminacion"] = CONFIG[
            "ruta_figura_contaminacion_v13"
        ]
    if limite is not None and not (
        variante is not None and variante.startswith("smoke")
    ):
        redirigir_rutas_a_tmp(
            (
                "ruta_benchmark",
                "ruta_por_ano",
                "ruta_figura",
                "ruta_figura_contaminacion",
                "ruta_b2_retriever",
            ),
            "_smoke",
        )
    else:
        sufijo_defensivo = (
            f"_b2_{variante}" if variante is not None else "_b2"
        )
        redirigir_rutas_a_tmp(("ruta_figura_hubs_v8",), sufijo_defensivo)


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
    ciclo v3 - Sprint A1: --diagnostico (default OFF) anade el
    diagnostico honesto sobre los scores ya calculados (recall@k, IC 95%
    bootstrap, periodos, longitud, techo del gold) y escribe los
    artefactos c3_*; sin el flag nada cambia.
    ciclo v3 - Sprint A2: --embedding-model HF_ID (repetible) anade
    modelos de embeddings a la matriz A2 (E5 siempre como referencia) y
    --k-candidatos-hybrid K cambia la profundidad de candidatos
    (default 50 = comportamiento actual); con cualquiera de los dos la
    matriz se activa y las salidas pasan a v9 + figura c3_a2_embeddings.
    Ambos requieren --rerank (default ON).
    ciclo v3 - Sprint A3: --a3 (default OFF) activa la comparativa de
    rerankers legales sobre candidatos ALIA@100 fijos (bge referencia,
    SINAI aliarerank y fusion RRF k=60) con IC bootstrap pareado de la
    diferencia vs bge y salidas v10 + figura c3_a3_rerankers. Requiere
    --rerank.
    ciclo v3 - Sprint A4: --a4 (default OFF) activa la fusion de 1a
    etapa (pesos BM25/denso w05/w03/w07 y RRF k=60) sobre denso ALIA
    con bge rerank chunks a 100 candidatos, mas el barrido de
    profundidad 50/100/200 de la fusion elegida (--a4-fusion, default
    w05); salidas v11 + figura c3_a4_fusion_profundidad + derivado
    c3_a4_fusion_profundidad.json. Requiere --rerank.
    ciclo v3 - Sprint B1: --b1 (default OFF) evalua el reranker de
    dominio fine-tuneado (modelos/b1_reranker, entrenado por el script
    24) en la ruta ganadora del producto (hibrido_rerank_chunks_top50 +
    hubs alpha=1.5 + k=10) contra la referencia bge del mismo run;
    salidas v12 + derivado c3_b1_reranker.json. Requiere --rerank.
    --reranker-model HF_ID sobreescribe el modelo B1 SOLO para pruebas
    de fontaneria (alias del catalogo del lib, ruta local o HF id) y
    requiere --b1. Sin override y sin modelos/b1_reranker el parser
    falla con la instruccion de entrenar primero con el script 24.
    ciclo v3 - Sprint B2: --b2 (default OFF) evalua el retriever denso
    de dominio fine-tuneado (modelos/b2_retriever, entrenado por el
    script 25) en la 1a etapa del hibrido 0.5/0.5 top-50 con el reranker
    B1 (modelos/b1_reranker) y hubs alpha=1.5; salidas v13 + derivado
    c3_b2_retriever.json. Requiere --rerank y es excluyente con --b1,
    --a3, --a4 y con los disparadores de la matriz A2. --embedding-model
    (un solo valor) sobreescribe el denso B2 SOLO para pruebas de
    fontaneria y --reranker-model sobreescribe el reranker B1 de la
    ruta. Sin override y sin modelos/b2_retriever el parser falla con la
    instruccion de entrenar primero con el script 25.
    ciclo v3 - iteracion v14: --b2-variante SLUG (default None; requiere
    --b2) evalua una variante del retriever denso de dominio: el modelo
    pasa a modelos/b2_retriever_<SLUG> (fail-fast con mensaje claro si
    no existe; NO se usa el override --embedding-model) y TODAS las
    salidas se redirigen a v14 nombradas por slug (version
    v14-ciclo3-b2-<SLUG>, benchmark vinculo_cp_benchmark_v14_<SLUG>.json,
    por_ano vinculo_cp_por_ano_v14_<SLUG>.csv, figuras
    deriva_retrieval_v14_<SLUG>.png y
    contaminacion_transitorios_v14_<SLUG>.png, y derivado
    c3_b2_retriever_<SLUG>.json) con la nota v14 que menciona la
    variante y el modelo; las referencias comparativas (v12 y ALIA@100
    del v9) se mantienen. SLUG debe cumplir [a-z0-9_], no se combina con
    --embedding-model y con --limite los slugs "smoke*" conservan las
    rutas v14 (evidencia de fontaneria) mientras el resto se redirige a
    /tmp/opencode con sufijo _smoke.

    Returns:
        Namespace con indice ("permanentes"/"completo"), rerank (bool),
        limite (int | None), hubs_alpha (float), calibrar_hubs (bool),
        generar_prior (bool), diagnostico (bool), embedding_model
        (list[str] | None), k_candidatos_hybrid (int), a3 (bool), a4
        (bool), a4_fusion (str | None), b1 (bool), b2 (bool),
        b2_variante (str | None) y reranker_model (str | None, resuelto
        cuando --b1 o --b2 estan activos).

    Raises:
        SystemExit: Si --hubs-alpha es negativo, --k-candidatos-hybrid
            es menor que 1, --a3/--a4/--b1/--b2 se usan con --no-rerank,
            --a4-fusion se usa sin --a4, --reranker-model se usa sin
            --b1 ni --b2, --b1 y --b2 se combinan, --b2 se combina con
            --a3/--a4/--k-candidatos-hybrid o con varios
            --embedding-model, --b2-variante se usa sin --b2, con
            --embedding-model o con un slug fuera de [a-z0-9_], o los
            modelos locales B1/B2 (o la variante B2) no existen sin
            override (parser.error).
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
    parser.add_argument(
        "--diagnostico",
        action="store_true",
        help=(
            "Modo diagnostico honesto (ciclo v3 - Sprint A1, default OFF): "
            "reutiliza los scores base ya calculados (sin re-ejecutar "
            "modelos) para generar curvas recall@k, IC 95%% bootstrap "
            "(1000 remuestreos) de nDCG@10/MRR/F1@10/F1@5, desglose por "
            "periodo y por terciles de longitud, techo teorico del gold y "
            "comparacion sin/con hubs adoptados. Salidas NUEVAS: "
            "resultados/c3_diagnostico_recall.json, figuras/c3_recall_"
            "curvas.png, c3_metricas_ic.png y c3_por_periodo.png; con "
            "--limite N van a /tmp/opencode y las salidas estandar del "
            "benchmark tambien se redirigen a /tmp (no pisa v6/v8)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        dest="embedding_model",
        action="append",
        default=None,
        metavar="HF_ID",
        help=(
            "Modelo de embeddings de la matriz A2 (ciclo v3 - Sprint A2; "
            "repetible para la matriz completa). Acepta un HF id o un "
            "slug corto del catalogo (e5/alia/bgem3/modernbert). E5 se "
            "calcula SIEMPRE como referencia del IC pareado; este flag "
            "anade modelos. Requiere --rerank (default ON) y activa las "
            "salidas v9 + figuras/c3_a2_embeddings.png. Sin el flag (y "
            "con k=50) el comportamiento es identico al actual. Con --b2 "
            "(un solo valor) sobreescribe el retriever denso de la ruta "
            "B2 SOLO para pruebas de fontaneria."
        ),
    )
    parser.add_argument(
        "--k-candidatos-hybrid",
        type=int,
        default=CONFIG["k_candidatos_hybrid"],
        metavar="K",
        help=(
            "Profundidad de candidatos de las estrategias hibridas de la "
            "matriz A2 (Sprint A2; default 50 = top-50 de v8). Crea "
            "hibrido_<slug>_rerank_chunks_topK; el rerank escala lineal "
            "(K x 3 pares/caso). La E5 a K=50 reutiliza la estrategia "
            "legacy hibrido_rerank_chunks_top50 (cross-check con v8). "
            "Valores usados en A2: 100 (y 200 como soporte)."
        ),
    )
    parser.add_argument(
        "--a3",
        action="store_true",
        help=(
            "Comparativa A3 de rerankers legales (ciclo v3 - Sprint A3; "
            "default OFF, requiere --rerank). Fija los candidatos en "
            "ALIA@100 (el embedding con mayor recall de candidatos de A2) "
            "y compara 3 variantes sobre los MISMOS pares (chunk, "
            "pasaje): bge (bge-reranker-v2-m3, referencia), aliarerank "
            "(SINAI/ALIA-MrBERT-es-legal-administrative-reranker, "
            "cross-encoder legal ES de 150M) y rrf (Reciprocal Rank "
            "Fusion k=60 de ambos rankings, con min-max + prior de hubs). "
            "Reporta nDCG@10/F1@10 en contextos alpha=0 y alpha=1.5 con "
            "IC 95%% bootstrap pareado de la diferencia vs bge. Salidas "
            "NUEVAS: resultados/vinculo_cp_benchmark_v10.json (version "
            "v10-ciclo3-sprintA3) y figuras/c3_a3_rerankers.png; con "
            "--limite N todo va a /tmp/opencode."
        ),
    )
    parser.add_argument(
        "--a4",
        action="store_true",
        help=(
            "Fusion de 1a etapa + barrido de profundidad (ciclo v3 - "
            "Sprint A4; default OFF, requiere --rerank). Fija el "
            "embedding denso en ALIA-MrBERT legal y compara con bge "
            "rerank chunks 4 variantes a 100 candidatos: w05 "
            "(referencia 0.5*BM25 + 0.5*denso), w03 (BM25 0.3 / denso "
            "0.7), w07 (BM25 0.7 / denso 0.3) y rrf1a (RRF k=60 de los "
            "rankings completos BM25 y denso). Barre ademas "
            "top-50/100/200 de la fusion elegida y reporta nDCG@10/F1@10 "
            "en contextos alpha=0/1.5 con IC 95%% bootstrap pareado de "
            "la diferencia vs w05@100. Salidas NUEVAS: "
            "vinculo_cp_benchmark_v11.json (version v11-ciclo3-sprintA4), "
            "c3_a4_fusion_profundidad.json y "
            "figuras/c3_a4_fusion_profundidad.png; con --limite N todo "
            "va a /tmp/opencode."
        ),
    )
    parser.add_argument(
        "--a4-fusion",
        dest="a4_fusion",
        choices=["w05", "w03", "w07", "rrf1a"],
        default=None,
        metavar="VARIANTE",
        help=(
            "Fusion cuyo barrido de profundidad se ejecuta en modo --a4 "
            "(w05/w03/w07/rrf1a; default w05, la referencia pre-"
            "registrada: el smoke n=10 de w07 no es criterio robusto). "
            "Las 4 fusiones se reportan a @100 de todos modos."
        ),
    )
    parser.add_argument(
        "--b1",
        action="store_true",
        help=(
            "Evalua el reranker de dominio B1 (ciclo v3 - Sprint B1; "
            "default OFF, requiere --rerank). Puntua la ruta ganadora "
            "del producto con rerank (hibrido_rerank_chunks_top50 + "
            "prior de hubs alpha=1.5 + salida k=10) con el cross-encoder "
            "fine-tuneado en modelos/b1_reranker (script 24) sobre los "
            "MISMOS candidatos y pares que la referencia bge y reporta "
            "el IC 95%% bootstrap pareado de la diferencia vs bge del "
            "mismo run. Salidas NUEVAS: vinculo_cp_benchmark_v12.json "
            "(version v12-ciclo3-sprintB1 con la seccion comparacion_b1), "
            "vinculo_cp_por_ano_v12.csv, deriva_retrieval_v12.png y el "
            "derivado c3_b1_reranker.json; con --limite N todo va a "
            "/tmp/opencode. Si modelos/b1_reranker no existe, el script "
            "falla ANTES de ejecutar con la instruccion de entrenar "
            "primero con scripts/24_b1_entrenar_reranker.py."
        ),
    )
    parser.add_argument(
        "--b2",
        action="store_true",
        help=(
            "Evalua el retriever denso de dominio B2 (ciclo v3 - Sprint "
            "B2; default OFF, requiere --rerank; excluyente con --b1, "
            "--a3, --a4 y con los disparadores de la matriz A2). Evalua "
            "la ruta hibrido_b2_rerank_chunks_top50_b1: hibrido "
            "0.5*bm25 + 0.5*B2 (retriever fine-tuneado en "
            "modelos/b2_retriever, script 25) sobre los MISMOS top-50 "
            "candidatos y pares (chunk, pasaje) que la ruta B1, rerank "
            "por chunks con el reranker B1 (modelos/b1_reranker) y "
            "prior de hubs alpha=1.5. Reporta el IC 95%% bootstrap "
            "pareado de la diferencia vs la referencia v12 (ruta B1 con "
            "hibrido E5: comparacion_b1.metricas_por_caso; fallback "
            "top10_por_caso) y la verificacion externa vs A2/ALIA@100 "
            "de v9. Salidas NUEVAS: vinculo_cp_benchmark_v13.json "
            "(version v13-ciclo3-sprintB2 con la seccion comparacion_b2), "
            "vinculo_cp_por_ano_v13.csv, deriva_retrieval_v13.png y el "
            "derivado c3_b2_retriever.json; con --limite N todo va a "
            "/tmp/opencode. Si modelos/b2_retriever no existe, el script "
            "falla ANTES de ejecutar con la instruccion de entrenar "
            "primero con scripts/25_b2_retriever_denso.py --entrenar; "
            "--embedding-model (un solo valor) sobreescribe el denso "
            "SOLO para pruebas de fontaneria."
        ),
    )
    parser.add_argument(
        "--b2-variante",
        dest="b2_variante",
        default=None,
        metavar="SLUG",
        help=(
            "Variante del retriever denso B2 (ciclo v3 - iteracion v14; "
            "default None, requiere --b2). SLUG [a-z0-9_] resuelve el "
            "denso como modelos/b2_retriever_<SLUG> (p. ej. ep2 -> "
            "modelos/b2_retriever_ep2) y redirige TODAS las salidas a "
            "v14 nombradas por slug: version v14-ciclo3-b2-<SLUG>, "
            "vinculo_cp_benchmark_v14_<SLUG>.json, "
            "vinculo_cp_por_ano_v14_<SLUG>.csv, "
            "deriva_retrieval_v14_<SLUG>.png, "
            "contaminacion_transitorios_v14_<SLUG>.png y derivado "
            "c3_b2_retriever_<SLUG>.json; las referencias comparativas "
            "(v12 y ALIA@100 del v9) se mantienen y la nota v14 menciona "
            "la variante y el modelo. Fail-fast si el directorio de la "
            "variante no existe. No se combina con --embedding-model "
            "(la variante resuelve la ruta por convencion). Con --limite "
            "los slugs 'smoke*' conservan las rutas v14 (fontaneria) y "
            "el resto se redirige a /tmp/opencode con sufijo _smoke."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        dest="reranker_model",
        default=None,
        metavar="HF_ID",
        help=(
            "Override del modelo del reranker B1 (requiere --b1 o --b2; "
            "SOLO para pruebas de fontaneria del cableado). Acepta un "
            "alias del catalogo de scripts/lib/reranker_bge.py (p. ej. "
            "bge para usar el reranker de referencia), una ruta local "
            "relativa a la raiz del proyecto o un HF id. Sin este flag "
            "el reranker es la ruta local modelos/b1_reranker (tanto en "
            "la ruta B1 como en la ruta B2)."
        ),
    )
    args = parser.parse_args()
    if args.hubs_alpha < 0:
        parser.error("--hubs-alpha debe ser >= 0")
    if args.k_candidatos_hybrid < 1:
        parser.error("--k-candidatos-hybrid debe ser >= 1")
    if args.a3 and not args.rerank:
        parser.error("--a3 requiere --rerank (default ON) para puntuar los pares")
    if args.a4 and not args.rerank:
        parser.error("--a4 requiere --rerank (default ON) para puntuar los pares")
    if args.a4_fusion is not None and not args.a4:
        parser.error("--a4-fusion requiere --a4")
    if args.b1 and not args.rerank:
        parser.error(
            "--b1 requiere --rerank (default ON) para puntuar los pares "
            "con el reranker de dominio"
        )
    if args.b2 and not args.rerank:
        parser.error(
            "--b2 requiere --rerank (default ON) para puntuar los pares "
            "con el reranker B1 y el retriever denso de dominio"
        )
    if args.b1 and args.b2:
        parser.error(
            "--b1 y --b2 son mutuamente excluyentes: son dos evaluaciones "
            "de dominio distintas (reranker B1 vs retriever denso B2)"
        )
    if args.b2 and args.a3:
        parser.error(
            "--b2 no se combina con --a3 (comparativa de rerankers): "
            "ejecuta cada sprint por separado"
        )
    if args.b2 and args.a4:
        parser.error(
            "--b2 no se combina con --a4 (fusion de 1a etapa): ejecuta "
            "cada sprint por separado"
        )
    if (
        args.b2
        and args.k_candidatos_hybrid != CONFIG["k_candidatos_hybrid"]
    ):
        parser.error(
            "--b2 no se combina con --k-candidatos-hybrid (matriz A2): "
            "el retriever B2 usa top-50 fijo"
        )
    if (
        args.b2
        and args.embedding_model is not None
        and len(args.embedding_model) != 1
    ):
        parser.error(
            "--b2 acepta como maximo un --embedding-model (override del "
            "retriever denso B2 para pruebas de fontaneria)"
        )
    if args.b2_variante is not None:
        if not args.b2:
            parser.error("--b2-variante requiere --b2")
        if re.fullmatch(r"[a-z0-9_]+", args.b2_variante) is None:
            parser.error(
                "--b2-variante debe ser un slug [a-z0-9_] (p. ej. ep2)"
            )
        if args.embedding_model is not None:
            parser.error(
                "--b2-variante no se combina con --embedding-model: la "
                "variante resuelve el denso como modelos/b2_retriever_"
                "<SLUG> (el override --embedding-model es solo para "
                "fontaneria sin variante)"
            )
    if args.reranker_model is not None and not (args.b1 or args.b2):
        parser.error("--reranker-model requiere --b1 o --b2")
    if args.b1:
        try:
            args.reranker_model = resolver_modelo_b1(args.reranker_model)
        except FileNotFoundError as exc:
            parser.error(str(exc))
    if args.b2:
        try:
            args.reranker_model = resolver_modelo_b1(args.reranker_model)
            modelo_denso = (
                args.embedding_model[0]
                if args.embedding_model is not None
                else None
            )
            args.embedding_model = [
                resolver_modelo_b2(modelo_denso, args.b2_variante)
            ]
        except FileNotFoundError as exc:
            parser.error(str(exc))
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


def aplicar_modo_diagnostico(activo: bool, limite: int | None) -> None:
    """Ajusta CONFIG para el modo --diagnostico (ciclo v3 - Sprint A1).

    El diagnostico es el artefacto del experimento A1; para no
    sobrescribir NINGUN benchmark previo (v6/v8 son canonicos con md5,
    ADR-16) las salidas estandar del benchmark se redirigen a
    /tmp/opencode con sufijo _diagnostico cuando el flag esta activo.
    Con --limite N las salidas nuevas del diagnostico tambien van a
    /tmp/opencode con sufijo _smoke (mismo patron del resto del script).
    Sin el flag la funcion no hace nada: el comportamiento actual queda
    identico.

    Args:
        activo: True si el flag --diagnostico esta presente.
        limite: Valor de --limite (None en el run final).
    """
    if not activo:
        return
    destino = Path("/tmp/opencode")
    sufijo_estandar = "_smoke" if limite is not None else "_diagnostico"
    for clave in (
        "ruta_benchmark",
        "ruta_por_ano",
        "ruta_figura",
        "ruta_figura_contaminacion",
    ):
        original = Path(CONFIG[clave])
        # Idempotencia: aplicar_modo_rerank/aplicar_modo_hubs ya pueden
        # haber redirigido estas rutas a /tmp/opencode (no duplicar sufijo).
        if original.parent == destino:
            continue
        CONFIG[clave] = str(
            destino
            / f"{original.stem}{sufijo_estandar}{original.suffix}"
        )
    if limite is None:
        return
    for clave in (
        "ruta_diagnostico",
        "ruta_figura_diagnostico_recall",
        "ruta_figura_diagnostico_ic",
        "ruta_figura_diagnostico_periodo",
    ):
        original = Path(CONFIG[clave])
        CONFIG[clave] = str(
            destino / f"{original.stem}_smoke{original.suffix}"
        )


def main() -> None:
    """Orquesta el benchmark M7 v3 (Fase T iter3 + W-a iter2 + A1..A4/B1/B2)."""
    inicio_total = time.time()
    args = parsear_args()
    incluir_transitorios = aplicar_modo_indice(args.indice)
    aplicar_modo_rerank(args.rerank, args.limite)
    aplicar_modo_hubs(args.hubs_alpha, args.calibrar_hubs, args.limite)
    aplicar_modo_diagnostico(args.diagnostico, args.limite)
    # ciclo v3 - Sprint A2: la matriz de embeddings se activa con
    # --embedding-model o con una profundidad distinta de la default
    # (50). Requiere reranking: sin el, se ignora con aviso. En modo B2
    # el flag --embedding-model es el override del retriever denso y NO
    # activa la matriz A2 (los combos se validan en parsear_args).
    modelos_a2: list[dict] = []
    if not args.b2 and (
        args.embedding_model
        or args.k_candidatos_hybrid != CONFIG["k_candidatos_hybrid"]
    ):
        if not args.rerank:
            print(
                "[aviso] la matriz A2 requiere --rerank (default ON): se "
                "ignoran --embedding-model y --k-candidatos-hybrid"
            )
        else:
            modelos_a2 = construir_modelos_a2(args)
            aplicar_modo_a2(
                modelos_a2, args.k_candidatos_hybrid, args.limite, args.diagnostico
            )
    # ciclo v3 - Sprint A3: comparativa de rerankers sobre ALIA@100. Se
    # activa DESPUES de A2 para poder reutilizar su estrategia homonima
    # (bge a k=100 con slug alia) si la matriz ya la calculo.
    if args.a3:
        aplicar_modo_a3(args.limite)
    # ciclo v3 - Sprint A4: fusion de 1a etapa y barrido de profundidad.
    # Se activa DESPUES de A2/A3 para poder reutilizar sus estrategias
    # homonimas (w05@100 sobre ALIA) si ya se calcularon.
    if args.a4:
        aplicar_modo_a4(args.limite, args.a4_fusion)
    # ciclo v3 - Sprint B1: evaluacion del reranker de dominio. Se
    # activa DESPUES de A2/A3/A4 para que las salidas v12 manden sobre
    # las v9/v10/v11 si se combinan flags.
    if args.b1:
        aplicar_modo_b1(args.limite, args.reranker_model)
    # ciclo v3 - Sprint B2: evaluacion del retriever denso de dominio.
    # Excluyente con B1/A2/A3/A4 (validado en parsear_args); se activa
    # al final para que las salidas v13 manden sobre v9/v10/v11.
    if args.b2:
        aplicar_modo_b2(
            args.limite,
            args.embedding_model[0],
            args.reranker_model,
            args.b2_variante,
        )
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
    if CONFIG["a2"]["activa"]:
        print(
            "Matriz A2 activa: modelos="
            + ", ".join(modelo["slug"] for modelo in modelos_a2)
            + f" | k_candidatos_hybrid={CONFIG['a2']['k']} | referencia="
            f"{CONFIG['a2']['referencia']} | salida "
            f"{Path(CONFIG['ruta_benchmark']).name} | figura "
            f"{Path(CONFIG['ruta_figura_a2_embeddings']).name}"
        )
    if CONFIG["a3"]["activa"]:
        print(
            "Comparativa A3 activa: candidatos="
            f"{CONFIG['a3']['slug_embedding']}@{CONFIG['a3']['k']} | "
            "variantes="
            + ", ".join(
                f"{entrada['variante']}->{entrada['estrategia']}"
                for entrada in CONFIG["a3"]["estrategias"]
            )
            + f" | rrf_k={CONFIG['a3']['rrf_k']} | reutiliza_A2="
            f"{CONFIG['a3']['reutiliza_a2']} | salida "
            f"{Path(CONFIG['ruta_benchmark']).name} | figura "
            f"{Path(CONFIG['ruta_figura_a3_rerankers']).name}"
        )
    if CONFIG["a4"]["activa"]:
        print(
            "Comparativa A4 activa: embedding="
            f"{CONFIG['a4']['slug_embedding']} | fusiones@100="
            + ", ".join(
                entrada["estrategia"]
                for entrada in CONFIG["a4"]["estrategias"]
                if entrada["familia"] == "fusion_100"
            )
            + f" | barrido={CONFIG['a4']['fusion_barrido']} "
            f"{CONFIG['a4']['profundidades']} | referencia="
            f"{CONFIG['a4']['referencia']} | salida "
            f"{Path(CONFIG['ruta_benchmark']).name} | figura "
            f"{Path(CONFIG['ruta_figura_a4_fusion_profundidad']).name}"
        )
    if CONFIG["b1"]["activa"]:
        print(
            "Evaluacion B1 activa: reranker="
            f"{CONFIG['b1']['modelo']} | ruta por defecto="
            f"{CONFIG['b1']['ruta_modelo']} | estrategia="
            f"{CONFIG['b1']['estrategia']} (candidatos top-"
            f"{CONFIG['b1']['candidatos_k']} del hibrido 0.5/0.5) | "
            f"contextos={dict(CONFIG['b1']['contextos'])} | salida "
            f"{Path(CONFIG['ruta_benchmark']).name} | derivado "
            f"{Path(CONFIG['ruta_b1_reranker']).name}"
        )
    if CONFIG["b2"]["activa"]:
        print(
            "Evaluacion B2 activa: retriever denso="
            f"{CONFIG['b2']['modelo']} | variante="
            f"{CONFIG['b2']['variante']} | slug="
            f"{CONFIG['b2']['slug']} | reranker="
            f"{CONFIG['b2']['modelo_reranker']} | estrategia="
            f"{CONFIG['b2']['estrategia']} (candidatos top-"
            f"{CONFIG['b2']['candidatos_k']} del hibrido 0.5*bm25 + "
            "0.5*B2) | contextos="
            f"{dict(CONFIG['b2']['contextos'])} | salida "
            f"{Path(CONFIG['ruta_benchmark']).name} | derivado "
            f"{Path(CONFIG['ruta_b2_retriever']).name}"
        )

    # --- ciclo v2 - Fase W-a: prior de hubs (se carga o construye una vez) ---
    # ciclo v3 - Sprint A1: --diagnostico tambien necesita el prior (contexto
    # con_hubs con los alphas adoptados) aunque no se aplique al benchmark.
    # ciclo v3 - Sprint A2: la comparacion necesita el prior para el
    # contexto con_hubs (alpha=1.5) aunque no se aplique al benchmark.
    # ciclo v3 - Sprint A3: idem para las variantes de reranker en los
    # contextos alpha=0/1.5.
    # ciclo v3 - Sprint B1: idem para la comparacion del reranker de
    # dominio (contexto ganador alpha=1.5).
    # ciclo v3 - Sprint B2: idem para la comparacion del retriever denso
    # de dominio (contextos alpha=0/1.5).
    prior_hubs = None
    if (
        CONFIG["modo_hubs_activo"]
        or args.diagnostico
        or CONFIG["a2"]["activa"]
        or CONFIG["a3"]["activa"]
        or CONFIG["a4"]["activa"]
        or CONFIG["b1"]["activa"]
        or CONFIG["b2"]["activa"]
    ):
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
    # ciclo v3 - Sprint A2: en modo matriz (CONFIG['a2']['activa']) los
    # embeddings de TODOS los modelos de la matriz (E5 incluido) se
    # cargan por la ruta generica cache-aware; sin la matriz el bloque
    # legacy queda INTACTO (identidad con v2..v8).
    stats_emb_a2: dict = {}
    longitudes_query: list[int] = []
    inicio_e5: float | None = None
    inicio_query: float | None = None
    if not CONFIG["a2"]["activa"]:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import models as st_models

        backbone = st_models.Transformer(
            CONFIG["modelo_e5"], max_seq_length=CONFIG["tamano_chunk"]
        )
        pooling = st_models.Pooling(
            backbone.get_embedding_dimension(), pooling_mode="mean"
        )
        modelo_e5 = SentenceTransformer(
            modules=[backbone, pooling], device=dispositivo
        )
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
        grupos_query = [
            chunkear_texto(c["ventana"], tokenizer) or [""] for c in casos
        ]
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
    else:
        # Matriz A2: un modelo a la vez (VRAM), cache de pasajes por slug.
        for configuracion in CONFIG["a2"]["modelos"]:
            slug = configuracion["slug"]
            modelo, tokenizer = cargar_modelo_embedding(configuracion, dispositivo)
            matriz_modelo, stats_pasajes = cargar_o_calcular_matriz_pasajes(
                modelo, tokenizer, configuracion, pasajes, canones
            )
            stats_query = calcular_scores_embedding(
                modelo,
                tokenizer,
                configuracion,
                casos,
                matriz_modelo,
                grupos_indices,
            )
            if slug == "e5":
                for caso in casos:
                    caso["scores_e5"] = caso["scores_emb_e5"]
                    caso["rankings"]["e5"] = ranking_desde_scores(
                        caso["scores_e5"], canones
                    )
                longitudes_query = stats_query["longitudes_query"]
            stats_emb_a2[slug] = {
                "hf_id": configuracion["hf_id"],
                "prefijo_query": configuracion["prefijo_query"],
                "prefijo_pasaje": configuracion["prefijo_pasaje"],
                "dim": int(matriz_modelo.shape[1]),
                "n_pasajes": int(matriz_modelo.shape[0]),
                "origen_pasajes": stats_pasajes["origen"],
                "n_chunks_pasajes": stats_pasajes["n_chunks"],
                "segundos_pasajes": stats_pasajes["segundos"],
                "n_chunks_query": stats_query["n_chunks_query"],
                "media_chunks_query": stats_query["media_chunks_query"],
                "segundos_query": stats_query["segundos"],
            }
            print(
                f"Embedding A2 {slug}: dim={matriz_modelo.shape[1]} | "
                f"pasajes {stats_pasajes['origen']} | "
                f"{stats_query['n_chunks_query']} chunks de query en "
                f"{stats_query['segundos']} s ({dispositivo})"
            )
            del modelo
            if dispositivo == "cuda":
                torch.cuda.empty_cache()

    # --- estrategia (d): hibrido pre-registrado 0.5/0.5 (mejora 6) ---
    for caso in casos:
        scores = scores_hibridos(caso["scores_bm25"], caso["scores_e5"])
        caso["scores_hibrido"] = scores
        caso["rankings"]["hibrido"] = ranking_desde_scores(scores, canones)

    # --- ciclo v3 - Sprint A2: hibridos por modelo (base de candidatos) ---
    # El hibrido es siempre el mismo 0.5/0.5 pre-registrado; lo unico que
    # cambia es el embedding denso (scores_emb_<slug>). Para E5 el vector
    # es identico al de la estrategia (d), pero se materializa aparte
    # para que la matriz A2 no dependa de claves legacy.
    if CONFIG["a2"]["activa"]:
        for configuracion in CONFIG["a2"]["modelos"]:
            slug = configuracion["slug"]
            clave_emb = f"scores_emb_{slug}"
            for caso in casos:
                caso[f"scores_hibrido_{slug}"] = scores_hibridos(
                    caso["scores_bm25"], caso[clave_emb]
                )

    # --- ciclo v3 - Sprint A3/A4: embedding ALIA de la 1a etapa
    # (candidatos @100 en A3; denso fijo de las fusiones en A4). Se
    # calcula una sola vez: si la matriz A2 ya lo aporto (slug alia a
    # k=100) o ya esta materializado, scores_hibrido_alia existe y la
    # cache por slug data/cache_emb_cp/alia.joblib se reutiliza igual.
    stats_emb_a3: dict = {}
    falta_alia = "scores_emb_alia" not in casos[0]
    if (CONFIG["a3"]["activa"] or CONFIG["a4"]["activa"]) and falta_alia:
        slug_alia = (
            CONFIG["a3"]["slug_embedding"]
            if CONFIG["a3"]["activa"]
            else CONFIG["a4"]["slug_embedding"]
        )
        configuracion_alia = config_modelo_embedding(slug_alia)
        modelo_alia, tokenizer_alia = cargar_modelo_embedding(
            configuracion_alia, dispositivo
        )
        matriz_alia, stats_pasajes_alia = cargar_o_calcular_matriz_pasajes(
            modelo_alia,
            tokenizer_alia,
            configuracion_alia,
            pasajes,
            canones,
        )
        stats_query_alia = calcular_scores_embedding(
            modelo_alia,
            tokenizer_alia,
            configuracion_alia,
            casos,
            matriz_alia,
            grupos_indices,
        )
        for caso in casos:
            caso["scores_hibrido_alia"] = scores_hibridos(
                caso["scores_bm25"], caso["scores_emb_alia"]
            )
        stats_emb_a3 = {
            "hf_id": configuracion_alia["hf_id"],
            "prefijo_query": configuracion_alia["prefijo_query"],
            "prefijo_pasaje": configuracion_alia["prefijo_pasaje"],
            "dim": int(matriz_alia.shape[1]),
            "n_pasajes": int(matriz_alia.shape[0]),
            "origen_pasajes": stats_pasajes_alia["origen"],
            "n_chunks_pasajes": stats_pasajes_alia["n_chunks"],
            "segundos_pasajes": stats_pasajes_alia["segundos"],
            "n_chunks_query": stats_query_alia["n_chunks_query"],
            "media_chunks_query": stats_query_alia["media_chunks_query"],
            "segundos_query": stats_query_alia["segundos"],
        }
        print(
            f"Embedding A3 {configuracion_alia['slug']}: dim="
            f"{matriz_alia.shape[1]} | pasajes "
            f"{stats_pasajes_alia['origen']} | "
            f"{stats_query_alia['n_chunks_query']} chunks de query en "
            f"{stats_query_alia['segundos']} s ({dispositivo})"
        )
        del modelo_alia
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    # --- ciclo v3 - Sprint B2: retriever denso de dominio (1a etapa) ---
    # El denso B2 se carga con el plumbing de A2 (carga generica + cache
    # por slug) y se materializa su hibrido 0.5/0.5 con BM25; el rerank
    # con el reranker B1 se hace despues, sobre los mismos top-50
    # candidatos y pares (chunk, pasaje) que la ruta B1.
    stats_emb_b2: dict = {}
    if CONFIG["b2"]["activa"]:
        stats_emb_b2 = calcular_denso_b2(
            casos, pasajes, canones, grupos_indices, dispositivo
        )

    # --- ciclo v3 - Sprint A4: vectores de fusion de 1a etapa (ALIA) ---
    # w05 reutiliza scores_hibrido_alia (0.5/0.5, materializado por el
    # bloque ALIA o por la matriz A2); w03/w07/rrf1a se materializan aqui
    # (baratos: normalizacion min-max/argsort por caso). Estos vectores
    # solo ordenan a los candidatos del rerank y miden su recall.
    if CONFIG["a4"]["activa"]:
        inicio_fusion_a4 = time.time()
        for caso in casos:
            caso["scores_fusion_alia_w03"] = scores_fusion_lineal(
                caso["scores_bm25"],
                caso["scores_emb_alia"],
                CONFIG["a4"]["pesos_fusion"]["w03"],
            )
            caso["scores_fusion_alia_w07"] = scores_fusion_lineal(
                caso["scores_bm25"],
                caso["scores_emb_alia"],
                CONFIG["a4"]["pesos_fusion"]["w07"],
            )
            caso["scores_fusion_alia_rrf1a"] = scores_rrf_primera_etapa(
                caso["scores_bm25"],
                caso["scores_emb_alia"],
                CONFIG["a4"]["rrf_k"],
            )
        print(
            "Fusiones A4 de 1a etapa (w05 0.5/0.5 reutilizado + w03/w07/"
            "rrf1a materializados): "
            f"{time.time() - inicio_fusion_a4:.1f} s"
        )

    # --- estrategias (e)-(h): reranking dos etapas (Fase T / iter2 / iter3) ---
    max_tokens_pares: list[int] = []
    max_tokens_pares_top50: list[int] = []
    max_tokens_pares_a2: dict[str, list[int]] = {}
    reranker = None
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
        # (i) ciclo v3 - Sprint A2: estrategias hibridas por modelo a la
        # profundidad --k-candidatos-hybrid (matriz A2). Mismos pares
        # (chunk, pasaje) derivados de la matriz BM25 (3 por candidato,
        # escala lineal); E5 a k=50 es la estrategia legacy y no se
        # recalcula (alias_legado).
        if CONFIG["a2"]["activa"]:
            inicio_rerank_a2 = time.time()
            for entrada in CONFIG["a2"]["estrategias"]:
                if entrada["alias_legado"]:
                    continue
                slug = entrada["slug"]
                max_tokens_pares_a2[slug] = []
                for caso in casos:
                    ranking, scores_rerank, max_tokens = ranking_rerank_chunks(
                        reranker,
                        caso[entrada["base_hibrido"]],
                        caso["matriz_bm25"],
                        caso["chunks_bm25"],
                        canones,
                        pasajes,
                        grupos_indices,
                        CONFIG["a2"]["k"],
                    )
                    caso[entrada["clave_scores"]] = scores_rerank
                    caso["rankings"][entrada["estrategia"]] = ranking
                    max_tokens_pares_a2[slug].append(max_tokens)
            print(
                f"Rerank A2: {len(CONFIG['a2']['estrategias'])} estrategias "
                f"({', '.join(e['slug'] for e in CONFIG['a2']['estrategias'])}) "
                f"x {len(casos)} casos a k={CONFIG['a2']['k']} "
                f"({CONFIG['a2']['k'] * CONFIG['k_chunks_rerank']} pares/caso) "
                f"en {time.time() - inicio_rerank_a2:.1f} s | max tokens par: "
                + ", ".join(
                    f"{slug}={max(valores)}"
                    for slug, valores in max_tokens_pares_a2.items()
                )
            )
        print(
            f"Rerank bge-reranker-v2-m3: {len(casos)} casos x 4 estrategias "
            f"(top-20 x2 + top-50 x2) en "
            f"{time.time() - inicio_rerank:.1f} s ({reranker.dispositivo}) | "
            f"max tokens par (chunk,pasaje) sin truncar: "
            f"top-20 {max(max_tokens_pares)} | top-50 "
            f"{max(max_tokens_pares_top50)}"
        )

    # --- ciclo v3 - Sprint A3: rerankers legales sobre ALIA@100 ---
    # Eficiencia (A3.3): para cada caso los pares (chunk, pasaje) se
    # construyen UNA sola vez (construir_pares_rerank_chunks) y se
    # puntuan con AMBOS rerankers; el vector bge se reutiliza de la
    # matriz A2 si ya se calculo (clave_bge). La variante rrf fusiona
    # los rankings de los dos rerankers con k=60.
    stats_rerank_a3: dict = {}
    if CONFIG["a3"]["activa"]:
        from lib.reranker_bge import RerankerBGE

        if reranker is None:
            reranker = RerankerBGE()
        reranker_sinai = RerankerBGE(
            modelo_id=CONFIG["a3"]["modelo_reranker"]
        )
        entradas_a3 = CONFIG["a3"]["estrategias"]
        k_a3 = CONFIG["a3"]["k"]
        clave_bge_a3 = CONFIG["a3"]["clave_bge"]
        max_tokens_bge_a3: list[int] = []
        max_tokens_sinai_a3: list[int] = []
        n_chunks_a3 = CONFIG["a3"]["k"] * CONFIG["k_chunks_rerank"]
        inicio_rerank_a3 = time.time()
        for caso in casos:
            pares_a3 = construir_pares_rerank_chunks(
                caso["scores_hibrido_alia"],
                caso["matriz_bm25"],
                caso["chunks_bm25"],
                grupos_indices,
                pasajes,
                k_a3,
            )
            candidatos_a3 = pares_a3["candidatos"]
            if clave_bge_a3 and clave_bge_a3 in caso:
                scores_bge_a3 = caso[clave_bge_a3]
            else:
                scores_bge_a3 = scores_maxpool_por_candidato(
                    reranker.puntuar(pares_a3["queries"], pares_a3["pasajes"]),
                    pares_a3["posiciones"],
                    candidatos_a3,
                    len(canones),
                )
            scores_sinai_a3 = scores_maxpool_por_candidato(
                reranker_sinai.puntuar(pares_a3["queries"], pares_a3["pasajes"]),
                pares_a3["posiciones"],
                candidatos_a3,
                len(canones),
            )
            scores_rrf_a3 = scores_rrf_entre_rerankers(
                scores_bge_a3,
                scores_sinai_a3,
                candidatos_a3,
                len(canones),
                CONFIG["a3"]["rrf_k"],
            )
            scores_por_variante = {
                "bge": scores_bge_a3,
                "aliarerank": scores_sinai_a3,
                "rrf": scores_rrf_a3,
            }
            for entrada in entradas_a3:
                clave_a3 = entrada["clave_scores"]
                caso[clave_a3] = scores_por_variante[entrada["variante"]]
                caso["rankings"][entrada["estrategia"]] = ranking_desde_scores(
                    caso[clave_a3], canones
                )[: CONFIG["k_rerank_salida"]]
            max_tokens_bge_a3.append(
                reranker.max_tokens_par(pares_a3["queries"], pares_a3["pasajes"])
            )
            max_tokens_sinai_a3.append(
                reranker_sinai.max_tokens_par(
                    pares_a3["queries"], pares_a3["pasajes"]
                )
            )
        stats_rerank_a3 = {
            "nota": (
                "Sprint A3: los pares (chunk, pasaje) se construyen una "
                "sola vez por caso y se puntuan con los dos rerankers; "
                "max_tokens_par se mide por tokenizer (bge y SINAI)."
            ),
            "pares_por_caso": n_chunks_a3,
            "max_tokens_par_bge": int(max(max_tokens_bge_a3)),
            "max_tokens_par_sinai": int(max(max_tokens_sinai_a3)),
            "segundos": round(time.time() - inicio_rerank_a3, 1),
            "reutiliza_bge_a2": clave_bge_a3 is not None,
        }
        print(
            f"Rerank A3 (ALIA@{k_a3}): {len(casos)} casos x "
            f"{n_chunks_a3} pares con bge y SINAI en "
            f"{stats_rerank_a3['segundos']} s | max tokens par: bge "
            f"{stats_rerank_a3['max_tokens_par_bge']} | sinai "
            f"{stats_rerank_a3['max_tokens_par_sinai']} | rrf_k="
            f"{CONFIG['a3']['rrf_k']} | bge reutilizado de A2: "
            f"{clave_bge_a3 is not None}"
        )
        del reranker_sinai
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    # --- ciclo v3 - Sprint A4: fusion de 1a etapa con bge rerank chunks ---
    # Cada variante (4 fusiones @100 + profundidades 50/200 de la fusion
    # elegida) construye sus pares (chunk, pasaje) con la MISMA derivacion
    # de v5/A2/A3 (mejor pasaje BM25 + top-3 chunks) a partir de su vector
    # de 1a etapa. Si la estrategia ya existe (reutilizacion de A2/A3, p.
    # ej. w05@100) no se re-puntua: "reutiliza alias si aplica". El coste
    # escala lineal: 3 pares por candidato (600 pares/caso a top-200).
    stats_rerank_a4: dict = {}
    if CONFIG["a4"]["activa"]:
        if reranker is None:
            from lib.reranker_bge import RerankerBGE

            reranker = RerankerBGE()
        entradas_a4 = CONFIG["a4"]["estrategias"]
        max_tokens_a4: dict[str, list[int]] = {}
        calculadas_a4: list[str] = []
        reutilizadas_a4: list[str] = []
        inicio_rerank_a4 = time.time()
        for entrada in entradas_a4:
            if entrada["clave_scores"] in casos[0]:
                reutilizadas_a4.append(entrada["estrategia"])
                continue
            max_tokens_a4[entrada["estrategia"]] = []
            for caso in casos:
                ranking, scores_rerank, max_tokens = ranking_rerank_chunks(
                    reranker,
                    caso[entrada["base_fusion"]],
                    caso["matriz_bm25"],
                    caso["chunks_bm25"],
                    canones,
                    pasajes,
                    grupos_indices,
                    entrada["k"],
                )
                caso[entrada["clave_scores"]] = scores_rerank
                caso["rankings"][entrada["estrategia"]] = ranking
                max_tokens_a4[entrada["estrategia"]].append(max_tokens)
            calculadas_a4.append(entrada["estrategia"])
        stats_rerank_a4 = {
            "nota": (
                "Sprint A4: la fusion fija quienes son los candidatos "
                "(top-k de su score de 1a etapa) y los pares (chunk, "
                "pasaje) se derivan con la misma logica de v5/A2/A3 "
                "(mejor pasaje BM25 + top-3 chunks, 3 pares/candidato)."
            ),
            "calculadas": calculadas_a4,
            "reutilizadas": reutilizadas_a4,
            "max_tokens_par": {
                nombre: int(max(valores))
                for nombre, valores in max_tokens_a4.items()
            },
            "segundos": round(time.time() - inicio_rerank_a4, 1),
        }
        print(
            f"Rerank A4 (denso {CONFIG['a4']['slug_embedding']}): "
            f"{len(calculadas_a4)} estrategias x {len(casos)} casos "
            f"({CONFIG['a4']['k_referencia']} y "
            f"{CONFIG['a4']['profundidades']} candidatos; 3 pares/candidato)"
            f" en {stats_rerank_a4['segundos']} s | reutilizadas: "
            f"{reutilizadas_a4 or 'ninguna'} | max tokens par: "
            + (
                ", ".join(
                    f"{nombre.rsplit('_top', 1)[-1]}={valor}"
                    for nombre, valor in stats_rerank_a4[
                        "max_tokens_par"
                    ].items()
                )
                if stats_rerank_a4["max_tokens_par"]
                else "n/a"
            )
        )

    # --- ciclo v3 - Sprint B1: reranker de dominio fine-tuneado ---
    # Los pares (chunk, pasaje) se derivan con la MISMA logica de v5/A2/
    # A3/A4 (mejor pasaje BM25 + top-3 chunks) a partir de los mismos
    # candidatos top-50 del hibrido 0.5/0.5 que la referencia bge, de
    # modo que el unico cambio del par (B1 vs referencia) es el
    # cross-encoder.
    stats_rerank_b1: dict = {}
    if CONFIG["b1"]["activa"]:
        from lib.reranker_bge import RerankerBGE

        if reranker is None:
            reranker = RerankerBGE()
        reranker_b1 = RerankerBGE(modelo_id=CONFIG["b1"]["modelo"])
        estrategia_b1 = CONFIG["b1"]["estrategia"]
        clave_b1 = FUENTE_SCORES[estrategia_b1]
        max_tokens_b1: list[int] = []
        inicio_rerank_b1 = time.time()
        for caso in casos:
            ranking, scores_rerank, max_tokens = ranking_rerank_chunks(
                reranker_b1,
                caso["scores_hibrido"],
                caso["matriz_bm25"],
                caso["chunks_bm25"],
                canones,
                pasajes,
                grupos_indices,
                CONFIG["b1"]["candidatos_k"],
            )
            caso[clave_b1] = scores_rerank
            caso["rankings"][estrategia_b1] = ranking
            max_tokens_b1.append(max_tokens)
        stats_rerank_b1 = {
            "nota": (
                "Sprint B1: mismos candidatos top-50 del hibrido 0.5/0.5 "
                "y mismos pares (chunk, pasaje) que la referencia bge; el "
                "unico cambio es el cross-encoder del modelo B1."
            ),
            "modelo": CONFIG["b1"]["modelo"],
            "pares_por_caso": (
                CONFIG["b1"]["candidatos_k"] * CONFIG["k_chunks_rerank"]
            ),
            "max_tokens_par": int(max(max_tokens_b1)),
            "segundos": round(time.time() - inicio_rerank_b1, 1),
        }
        print(
            f"Rerank B1 ({CONFIG['b1']['modelo']}): {len(casos)} casos x "
            f"{stats_rerank_b1['pares_por_caso']} pares en "
            f"{stats_rerank_b1['segundos']} s | max tokens par: "
            f"{stats_rerank_b1['max_tokens_par']}"
        )
        del reranker_b1
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    # --- ciclo v3 - Sprint B2: reranker B1 sobre la 1a etapa densa B2 ---
    # Los pares (chunk, pasaje) se derivan con la MISMA logica de v5/A2/
    # A3/A4 (mejor pasaje BM25 + top-3 chunks) a partir de los top-50
    # candidatos del hibrido 0.5*bm25 + 0.5*B2, de modo que el unico
    # cambio frente a la ruta B1 (hibrido E5) es el denso de la 1a
    # etapa; el cross-encoder es el MISMO B1.
    stats_rerank_b2: dict = {}
    if CONFIG["b2"]["activa"]:
        from lib.reranker_bge import RerankerBGE

        reranker_b2 = RerankerBGE(modelo_id=CONFIG["b2"]["modelo_reranker"])
        estrategia_b2 = CONFIG["b2"]["estrategia"]
        clave_b2 = FUENTE_SCORES[estrategia_b2]
        max_tokens_b2: list[int] = []
        inicio_rerank_b2 = time.time()
        for caso in casos:
            ranking, scores_rerank, max_tokens = ranking_rerank_chunks(
                reranker_b2,
                caso[CONFIG["b2"]["clave_hibrido"]],
                caso["matriz_bm25"],
                caso["chunks_bm25"],
                canones,
                pasajes,
                grupos_indices,
                CONFIG["b2"]["candidatos_k"],
            )
            caso[clave_b2] = scores_rerank
            caso["rankings"][estrategia_b2] = ranking
            max_tokens_b2.append(max_tokens)
        stats_rerank_b2 = {
            "nota": (
                "Sprint B2: la 1a etapa es el hibrido 0.5*bm25 + 0.5*B2 "
                f"(slug {CONFIG['b2']['slug']}) con top-50 candidatos; "
                "los pares (chunk, pasaje) y el cross-encoder son los "
                "MISMOS de la ruta B1, por lo que el unico cambio es el "
                "denso de la 1a etapa."
            ),
            "modelo": CONFIG["b2"]["modelo"],
            "reranker": CONFIG["b2"]["modelo_reranker"],
            "pares_por_caso": (
                CONFIG["b2"]["candidatos_k"] * CONFIG["k_chunks_rerank"]
            ),
            "max_tokens_par": int(max(max_tokens_b2)),
            "segundos": round(time.time() - inicio_rerank_b2, 1),
        }
        print(
            f"Rerank B2 ({CONFIG['b2']['modelo_reranker']} sobre denso "
            f"{CONFIG['b2']['slug']}): {len(casos)} casos x "
            f"{stats_rerank_b2['pares_por_caso']} pares en "
            f"{stats_rerank_b2['segundos']} s | max tokens par: "
            f"{stats_rerank_b2['max_tokens_par']}"
        )
        del reranker_b2
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    # --- ciclo v3 - Sprint A1: diagnostico honesto (reutiliza scores base) ---
    # Se ejecuta ANTES de aplicar el alpha efectivo a los casos para leer
    # siempre los scores base intactos (el diagnostico reconstruye sus
    # propios rankings sin mutar los casos).
    diagnostico_a1 = None
    if args.diagnostico:
        inicio_diagnostico = time.time()
        diagnostico_a1 = construir_diagnostico_a1(
            casos, prior_hubs, canones, args.limite
        )
        guardar_json(CONFIG["ruta_diagnostico"], diagnostico_a1)
        graficar_recall_curvas(diagnostico_a1)
        graficar_metricas_ic(diagnostico_a1)
        graficar_por_periodo(diagnostico_a1)
        print(
            f"Diagnostico A1: {len(casos)} casos, "
            f"bootstrap {CONFIG['diagnostico_bootstrap']} remuestreos "
            f"(seed {CONFIG['diagnostico_seed']}) en "
            f"{time.time() - inicio_diagnostico:.1f} s -> "
            f"{CONFIG['ruta_diagnostico']}"
        )

    # --- ciclo v3 - Sprint A2: comparacion de embeddings y pareado ---
    # Se ejecuta ANTES de aplicar el alpha efectivo del benchmark (los
    # contextos de la matriz usan alphas fijos 0.0/1.5 y no deben leer
    # scores mutados); reutiliza la maquinaria del diagnostico A1.
    comparacion_a2 = None
    if CONFIG["a2"]["activa"]:
        inicio_comparacion = time.time()
        comparacion_a2 = construir_comparacion_a2(casos, prior_hubs, canones)
        print(
            f"Comparacion A2: k={CONFIG['a2']['k']}, "
            f"{len(CONFIG['a2']['estrategias'])} estrategias x 2 contextos "
            f"(referencia {CONFIG['a2']['referencia']}) en "
            f"{time.time() - inicio_comparacion:.1f} s"
        )
        for contexto in comparacion_a2["contextos"]:
            for entrada in CONFIG["a2"]["estrategias"]:
                datos = comparacion_a2["metricas_globales"][contexto][
                    entrada["estrategia"]
                ]["ndcg_at_10"]
                print(
                    f"  A2 {contexto:8s} {entrada['slug']:11s} "
                    f"nDCG@10={datos['media']:.3f} "
                    f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}]"
                )
        for contexto, por_estrategia in comparacion_a2[
            "pareado_vs_referencia"
        ].items():
            for estrategia, medidas in por_estrategia.items():
                delta = medidas["ndcg_at_10"]
                veredicto = "gana" if delta["gana"] else "no concluyente"
                print(
                    f"  pareado {contexto:8s} {estrategia:38s} "
                    f"d_nDCG@10={delta['media_delta']:+.3f} "
                    f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                    f"{veredicto}"
                )

    # --- ciclo v3 - Sprint A3: comparacion de rerankers y pareado ---
    # Igual que A2, se ejecuta ANTES de aplicar el alpha efectivo del
    # benchmark (los contextos usan alphas fijos 0.0/1.5 y no deben leer
    # scores mutados).
    comparacion_a3 = None
    if CONFIG["a3"]["activa"]:
        inicio_comparacion_a3 = time.time()
        comparacion_a3 = construir_comparacion_a3(casos, prior_hubs, canones)
        print(
            f"Comparacion A3: ALIA@{CONFIG['a3']['k']}, "
            f"{len(CONFIG['a3']['estrategias'])} variantes x 2 contextos "
            f"(referencia {CONFIG['a3']['referencia']}) en "
            f"{time.time() - inicio_comparacion_a3:.1f} s"
        )
        for contexto in comparacion_a3["contextos"]:
            for entrada in CONFIG["a3"]["estrategias"]:
                datos = comparacion_a3["metricas_globales"][contexto][
                    entrada["estrategia"]
                ]["ndcg_at_10"]
                f1_10 = comparacion_a3["metricas_globales"][contexto][
                    entrada["estrategia"]
                ]["f1_at_10"]
                print(
                    f"  A3 {contexto:8s} {entrada['variante']:11s} "
                    f"nDCG@10={datos['media']:.3f} "
                    f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}] | "
                    f"F1@10={f1_10['media']:.3f} "
                    f"[{f1_10['p2_5']:.3f}, {f1_10['p97_5']:.3f}]"
                )
        for contexto, por_estrategia in comparacion_a3[
            "pareado_vs_referencia"
        ].items():
            for estrategia, medidas in por_estrategia.items():
                delta = medidas["ndcg_at_10"]
                veredicto = "gana" if delta["gana"] else "no concluyente"
                print(
                    f"  pareado {contexto:8s} {estrategia:38s} "
                    f"d_nDCG@10={delta['media_delta']:+.3f} "
                    f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                    f"{veredicto} (prob_mejora="
                    f"{delta['prob_mejora']:.3f})"
                )
        print(
            "  recall candidatos ALIA@100="
            f"{comparacion_a3['recall_candidatos']['media']:.3f} | "
            "reutiliza A2="
            f"{comparacion_a3['reutiliza_matriz_a2']}"
        )

    # --- ciclo v3 - Sprint A4: fusion de 1a etapa y barrido ---
    # Igual que A2/A3, se ejecuta ANTES de aplicar el alpha efectivo del
    # benchmark (los contextos usan alphas fijos 0.0/1.5 y no deben leer
    # scores mutados).
    comparacion_a4 = None
    if CONFIG["a4"]["activa"]:
        inicio_comparacion_a4 = time.time()
        comparacion_a4 = construir_comparacion_a4(casos, prior_hubs, canones)
        print(
            f"Comparacion A4: {len(casos)} casos, "
            f"{len(comparacion_a4['fusion_100']['etiquetas'])} fusiones@100 "
            f"+ barrido {CONFIG['a4']['fusion_barrido']}"
            f"{CONFIG['a4']['profundidades']} en "
            f"{time.time() - inicio_comparacion_a4:.1f} s"
        )
        for contexto in comparacion_a4["contextos"]:
            for fila in comparacion_a4["fusion_100"]["etiquetas"]:
                datos = comparacion_a4["fusion_100"]["metricas_globales"][
                    contexto
                ][fila["estrategia"]]["ndcg_at_10"]
                print(
                    f"  A4 {contexto:8s} {fila['variante']:6s} "
                    f"nDCG@10={datos['media']:.3f} "
                    f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}]"
                )
        for contexto, por_estrategia in comparacion_a4["fusion_100"][
            "pareado_vs_referencia"
        ].items():
            for estrategia, medidas in por_estrategia.items():
                delta = medidas["ndcg_at_10"]
                print(
                    f"  pareado {contexto:8s} {estrategia:44s} "
                    f"d_nDCG@10={delta['media_delta']:+.3f} "
                    f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                    f"prob_mejora={delta['prob_mejora']:.3f} "
                    f"{'gana' if delta['gana'] else 'no concluyente'}"
                )
        for contexto in comparacion_a4["contextos"]:
            for fila in comparacion_a4["profundidad"]["etiquetas"]:
                datos = comparacion_a4["profundidad"]["metricas_globales"][
                    contexto
                ][fila["estrategia"]]["ndcg_at_10"]
                print(
                    f"  A4 profundidad {contexto:8s} "
                    f"{fila['variante']}@{fila['k']:3d} "
                    f"nDCG@10={datos['media']:.3f} "
                    f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}]"
                )
        print(
            "  recall candidatos (50/100/200): "
            + " | ".join(
                f"{variante}="
                + "/".join(
                    f"{valor:.3f}"
                    for valor in datos["recall_por_k"].values()
                )
                for variante, datos in comparacion_a4[
                    "recall_candidatos"
                ].items()
            )
        )

    # --- ciclo v3 - Sprint B1: comparacion del reranker de dominio ---
    # Igual que A2/A3/A4, se ejecuta ANTES de aplicar el alpha efectivo
    # del benchmark (los contextos usan alphas fijos 0.0/1.5 y no deben
    # leer scores mutados).
    comparacion_b1 = None
    if CONFIG["b1"]["activa"]:
        inicio_comparacion_b1 = time.time()
        comparacion_b1 = construir_comparacion_b1(casos, prior_hubs, canones)
        print(
            f"Comparacion B1: {len(casos)} casos, estrategia "
            f"{CONFIG['b1']['estrategia']} vs "
            f"{CONFIG['b1']['estrategia_referencia']} en "
            f"{time.time() - inicio_comparacion_b1:.1f} s | referencia "
            f"externa: "
            f"{(comparacion_b1['referencia_externa'] or {}).get('archivo', 'no disponible (fallback: run)')}"
        )
        for contexto in comparacion_b1["contextos"]:
            for etiqueta, estrategia in (
                ("bge", CONFIG["b1"]["estrategia_referencia"]),
                ("b1", CONFIG["b1"]["estrategia"]),
            ):
                datos = comparacion_b1["metricas_globales"][contexto][
                    estrategia
                ]
                print(
                    f"  B1 {contexto:8s} {etiqueta:3s} "
                    f"nDCG@10={datos['ndcg_at_10']['media']:.3f} "
                    f"[{datos['ndcg_at_10']['p2_5']:.3f}, "
                    f"{datos['ndcg_at_10']['p97_5']:.3f}] | "
                    f"F1@10={datos['f1_at_10']['media']:.3f} "
                    f"[{datos['f1_at_10']['p2_5']:.3f}, "
                    f"{datos['f1_at_10']['p97_5']:.3f}]"
                )
        for contexto, por_estrategia in comparacion_b1[
            "pareado_vs_referencia"
        ].items():
            for estrategia, medidas in por_estrategia.items():
                delta = medidas["ndcg_at_10"]
                print(
                    f"  pareado {contexto:8s} {estrategia:38s} "
                    f"d_nDCG@10={delta['media_delta']:+.3f} "
                    f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                    f"prob_mejora={delta['prob_mejora']:.3f} "
                    f"{'gana' if delta['gana'] else 'no concluyente'}"
                )
        if comparacion_b1["pareado_vs_referencia_externa"] is not None:
            externo = comparacion_b1["pareado_vs_referencia_externa"]
            for medida, delta in externo["medidas"].items():
                print(
                    f"  pareado externo {externo['contexto']:8s} "
                    f"{medida:10s} d={delta['media_delta']:+.3f} "
                    f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                    f"prob_mejora={delta['prob_mejora']:.3f}"
                )

    # --- ciclo v3 - Sprint B2: comparacion del retriever denso ---
    # Igual que A2/A3/A4/B1, se ejecuta ANTES de aplicar el alpha
    # efectivo del benchmark (los contextos usan alphas fijos 0.0/1.5 y
    # no deben leer scores mutados).
    comparacion_b2 = None
    if CONFIG["b2"]["activa"]:
        inicio_comparacion_b2 = time.time()
        comparacion_b2 = construir_comparacion_b2(casos, prior_hubs, canones)
        referencia_v12 = comparacion_b2["referencia_v12"] or {}
        print(
            f"Comparacion B2: {len(casos)} casos, estrategia "
            f"{CONFIG['b2']['estrategia']} (denso "
            f"{CONFIG['b2']['slug']}) en "
            f"{time.time() - inicio_comparacion_b2:.1f} s | referencia "
            "v12: "
            f"{referencia_v12.get('origen_datos', 'no disponible (fallback)')}"
        )
        for contexto in comparacion_b2["contextos"]:
            datos = comparacion_b2["metricas_globales"][contexto][
                CONFIG["b2"]["estrategia"]
            ]
            print(
                f"  B2 {contexto:8s} "
                f"nDCG@10={datos['ndcg_at_10']['media']:.3f} "
                f"[{datos['ndcg_at_10']['p2_5']:.3f}, "
                f"{datos['ndcg_at_10']['p97_5']:.3f}] | "
                f"F1@10={datos['f1_at_10']['media']:.3f} "
                f"[{datos['f1_at_10']['p2_5']:.3f}, "
                f"{datos['f1_at_10']['p97_5']:.3f}]"
            )
        pareado_v12 = comparacion_b2["pareado_vs_referencia_v12"] or {}
        for contexto, medidas in pareado_v12.items():
            delta = medidas["ndcg_at_10"]
            print(
                f"  pareado v12 {contexto:8s} d_nDCG@10="
                f"{delta['media_delta']:+.3f} "
                f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                f"prob_mejora={delta['prob_mejora']:.3f} "
                f"{'gana' if delta['gana'] else 'no concluyente'}"
            )
        pareado_a2 = (
            comparacion_b2["pareado_vs_referencia_externa_a2"] or {}
        )
        for contexto, medidas in pareado_a2.items():
            delta = medidas["ndcg_at_10"]
            print(
                f"  pareado externo A2 {contexto:8s} d_nDCG@10="
                f"{delta['media_delta']:+.3f} "
                f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                f"prob_mejora={delta['prob_mejora']:.3f}"
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
    # Nota de la version activa: la variante v14 manda sobre v13; el resto
    # de la cadena reproduce la seleccion historica de notas.
    if CONFIG["b2"]["variante"] is not None:
        nota_salida = CONFIG["nota_v14"]
    elif CONFIG["b2"]["activa"]:
        nota_salida = CONFIG["nota_v13"]
    elif CONFIG["b1"]["activa"]:
        nota_salida = CONFIG["nota_v12"]
    elif CONFIG["a4"]["activa"]:
        nota_salida = CONFIG["nota_v11"]
    elif CONFIG["a3"]["activa"]:
        nota_salida = CONFIG["nota_v10"]
    elif CONFIG["a2"]["activa"]:
        nota_salida = CONFIG["nota_v9"]
    elif CONFIG["modo_hubs_activo"]:
        nota_salida = CONFIG["nota_v8"]
    elif args.rerank:
        nota_salida = CONFIG["nota_v6"]
    else:
        nota_salida = None
    benchmark = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "version": CONFIG["version_salida"],
        "indice": CONFIG["version_indice"],
        "reranker": CONFIG["reranker"] if args.rerank else None,
        "nota": nota_salida,
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
            "k_candidatos_hybrid": CONFIG["a2"]["k"],
            "modelos_embedding_a2": [
                {
                    "slug": modelo["slug"],
                    "hf_id": modelo["hf_id"],
                    "prefijo_query": modelo["prefijo_query"],
                    "prefijo_pasaje": modelo["prefijo_pasaje"],
                    "nota": modelo["nota"],
                }
                for modelo in CONFIG["a2"]["modelos"]
            ],
            "a2_contextos": dict(CONFIG["a2_contextos"]),
            "a2_referencia": CONFIG["a2"]["referencia"],
            "a3_k_candidatos": CONFIG["a3"]["k"],
            "a3_slug_embedding": CONFIG["a3"]["slug_embedding"],
            "a3_rrf_k": CONFIG["a3"]["rrf_k"],
            "a3_contextos": dict(CONFIG["a3"]["contextos"]),
            "a3_referencia": CONFIG["a3"]["referencia"],
            "a3_reutiliza_matriz_a2": CONFIG["a3"]["reutiliza_a2"],
            "a3_rerankers": [
                {
                    "variante": entrada["variante"],
                    "estrategia": entrada["estrategia"],
                    "hf_id": entrada["hf_id"],
                    "rol": entrada["rol"],
                }
                for entrada in CONFIG["a3"]["estrategias"]
            ],
            "a4_slug_embedding": CONFIG["a4"]["slug_embedding"],
            "a4_k_referencia": CONFIG["a4"]["k_referencia"],
            "a4_profundidades": list(CONFIG["a4"]["profundidades"]),
            "a4_rrf_k": CONFIG["a4"]["rrf_k"],
            "a4_pesos_fusion_bm25": dict(CONFIG["a4"]["pesos_fusion"]),
            "a4_variantes_100": list(CONFIG["a4"]["variantes_100"]),
            "a4_fusion_barrido": CONFIG["a4"]["fusion_barrido"],
            "a4_contextos": dict(CONFIG["a4"]["contextos"]),
            "a4_referencia": CONFIG["a4"]["referencia"],
            "a4_estrategias": [
                {
                    "variante": entrada["variante"],
                    "familia": entrada["familia"],
                    "estrategia": entrada["estrategia"],
                    "k": entrada["k"],
                    "peso_bm25": entrada["peso_bm25"],
                    "tipo": entrada["tipo"],
                    "rol": entrada["rol"],
                }
                for entrada in CONFIG["a4"]["estrategias"]
            ],
            "b1_modelo": CONFIG["b1"]["modelo"],
            "b1_ruta_modelo": CONFIG["b1"]["ruta_modelo"],
            "b1_estrategia": CONFIG["b1"]["estrategia"],
            "b1_estrategia_referencia": CONFIG["b1"]["estrategia_referencia"],
            "b1_candidatos": CONFIG["b1"]["candidatos_k"],
            "b1_k_salida": CONFIG["b1"]["k"],
            "b1_contextos": dict(CONFIG["b1"]["contextos"]),
            "b1_contexto_ganador": CONFIG["b1"]["contexto_ganador"],
            "b1_bootstrap": CONFIG["b1_bootstrap"],
            "b1_seed_pareado": CONFIG["b1_seed_pareado"],
            "b2_modelo": CONFIG["b2"]["modelo"],
            "b2_slug": CONFIG["b2"]["slug"],
            "b2_variante": CONFIG["b2"]["variante"],
            "b2_ruta_modelo": CONFIG["b2"]["ruta_modelo"],
            "b2_reranker": CONFIG["b2"]["modelo_reranker"],
            "b2_estrategia": CONFIG["b2"]["estrategia"],
            "b2_estrategia_referencia_v12": CONFIG["b2"][
                "estrategia_referencia_v12"
            ],
            "b2_candidatos": CONFIG["b2"]["candidatos_k"],
            "b2_k_salida": CONFIG["b2"]["k"],
            "b2_contextos": dict(CONFIG["b2"]["contextos"]),
            "b2_contexto_ganador": CONFIG["b2"]["contexto_ganador"],
            "b2_bootstrap": CONFIG["b2_bootstrap"],
            "b2_seed_pareado": CONFIG["b2_seed_pareado"],
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
            "e5_pasajes": (
                round(inicio_query - inicio_e5, 1)
                if inicio_e5 is not None and inicio_query is not None
                else None
            ),
            "e5_query": (
                round(time.time() - inicio_query, 1)
                if inicio_query is not None
                else None
            ),
            "embeddings_a2": stats_emb_a2 or None,
            "embeddings_a3": stats_emb_a3 or None,
            "rerank": (
                round(time.time() - inicio_rerank, 1) if args.rerank else None
            ),
            "rerank_a3": stats_rerank_a3.get("segundos"),
            "rerank_a4": stats_rerank_a4.get("segundos"),
            "rerank_b1": stats_rerank_b1.get("segundos"),
            "embeddings_b2": stats_emb_b2 or None,
            "rerank_b2": stats_rerank_b2.get("segundos"),
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
    # ciclo v3 - Sprint A2: comparativa de embeddings + IC pareado.
    if comparacion_a2 is not None:
        benchmark["comparacion_a2"] = comparacion_a2
        # Seccion prometida en A2.6/descripcion: stats de carga y codificacion
        # por modelo (hf_id, prefijos, dim, origen de cache, tiempos).
        benchmark["embeddings_a2"] = stats_emb_a2 or None
        benchmark["descripcion"] += (
            " Sprint A2: comparativa de embeddings legales ES (E5 ref, "
            "ALIA-MrBERT legal, BGE-M3-Legal, modernbert legal) en el "
            "hibrido BM25+denso con rerank por chunks a k candidatos "
            f"{CONFIG['a2']['k']} y 2 contextos de hubs (alpha=0 y 1.5), "
            "con IC bootstrap pareado de la diferencia vs la referencia; "
            "detalle en comparacion_a2 y embeddings_a2."
        )
    # ciclo v3 - Sprint A3: comparativa de rerankers legales + IC pareado.
    if comparacion_a3 is not None:
        benchmark["comparacion_a3"] = comparacion_a3
        benchmark["rerank_chunks_stats"]["a3"] = stats_rerank_a3
        benchmark["descripcion"] += (
            " Sprint A3: comparativa de rerankers legales sobre los "
            f"candidatos de {CONFIG['a3']['slug_embedding']}@"
            f"{CONFIG['a3']['k']} (bge referencia, SINAI aliarerank y "
            f"fusion RRF k={CONFIG['a3']['rrf_k']}) con los MISMOS pares "
            "(chunk, pasaje) puntuados por ambos rerankers y 2 contextos "
            "de hubs (alpha=0 y 1.5), con IC bootstrap pareado de la "
            "diferencia vs bge; detalle en comparacion_a3 y "
            "rerank_chunks_stats.a3."
        )
    # ciclo v3 - Sprint A4: fusion de 1a etapa + barrido de profundidad.
    if comparacion_a4 is not None:
        benchmark["comparacion_a4"] = comparacion_a4
        benchmark["embeddings_a4_alia"] = stats_emb_a3 or None
        benchmark["rerank_chunks_stats"]["a4"] = stats_rerank_a4
        benchmark["descripcion"] += (
            " Sprint A4: fusion de 1a etapa con denso FIJO ALIA "
            "(w05 0.5/0.5, w03/w07 con peso BM25 0.3/0.7 y rrf1a RRF "
            "k=60 de los rankings completos) y barrido de profundidad "
            "50/100/200 de la fusion elegida, todos con bge rerank "
            "chunks y 2 contextos de hubs (alpha=0 y 1.5), con IC "
            "bootstrap pareado de la diferencia vs w05@100; detalle en "
            "comparacion_a4 y rerank_chunks_stats.a4."
        )
    # ciclo v3 - Sprint B1: reranker de dominio fine-tuneado.
    if comparacion_b1 is not None:
        benchmark["comparacion_b1"] = comparacion_b1
        benchmark["rerank_chunks_stats"]["b1"] = stats_rerank_b1
        benchmark["descripcion"] += (
            " Sprint B1: evaluacion del reranker de dominio fine-tuneado "
            "(modelos/b1_reranker, script 24) en la ruta ganadora del "
            "producto (hibrido_rerank_chunks_top50 + hubs alpha=1.5 + "
            "k=10) contra la referencia bge del mismo run, con IC "
            "bootstrap pareado de la diferencia y verificacion externa "
            "de la referencia congelada v8/v10 cuando esta disponible; "
            "detalle en comparacion_b1 y rerank_chunks_stats.b1."
        )
    # ciclo v3 - Sprint B2: retriever denso de dominio fine-tuneado.
    if comparacion_b2 is not None:
        benchmark["comparacion_b2"] = comparacion_b2
        benchmark["embeddings_b2"] = stats_emb_b2 or None
        benchmark["rerank_chunks_stats"]["b2"] = stats_rerank_b2
        etiqueta_retriever_b2 = nombre_retriever_b2() + (
            f" (variante '{CONFIG['b2']['variante']}')"
            if CONFIG["b2"]["variante"] is not None
            else ""
        )
        benchmark["descripcion"] += (
            " Sprint B2: evaluacion del retriever denso de dominio "
            f"fine-tuneado ({etiqueta_retriever_b2}, script 25) en la 1a "
            "etapa del hibrido 0.5*bm25 + 0.5*B2 (top-50 candidatos) con "
            "el reranker B1 (modelos/b1_reranker) y hubs alpha=1.5, con "
            "IC bootstrap pareado de la diferencia vs la referencia v12 "
            "(ruta B1 con hibrido E5 congelada) y verificacion externa "
            "vs A2/ALIA@100 (v9) cuando esta disponible; detalle en "
            "comparacion_b2, embeddings_b2 y rerank_chunks_stats.b2."
        )
    guardar_json(CONFIG["ruta_benchmark"], benchmark)

    por_caso[["id", "ano", "estrategia", "n_relevantes", *MEDIDAS]].to_csv(
        CONFIG["ruta_por_ano"], index=False, encoding="utf-8"
    )
    graficar_deriva(por_caso)
    graficar_contaminacion(diagnostico)
    if comparacion_a2 is not None:
        graficar_a2_embeddings(comparacion_a2)
    if comparacion_a3 is not None:
        graficar_a3_rerankers(comparacion_a3)
    if comparacion_a4 is not None:
        volcar_comparacion_a4(comparacion_a4)
        graficar_a4_fusion_profundidad(comparacion_a4)
    if comparacion_b1 is not None:
        volcar_comparacion_b1(comparacion_b1)
    if comparacion_b2 is not None:
        volcar_comparacion_b2(comparacion_b2)

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
    if diagnostico_a1 is not None:
        print("\n=== Diagnostico A1: IC 95% bootstrap (media [p2.5, p97.5]) ===")
        for contexto in (CONTEXTO_SIN_HUBS, CONTEXTO_CON_HUBS):
            for estrategia in diagnostico_a1["config"]["estrategias_focales"]:
                fila = diagnostico_a1["ic_bootstrap"][contexto][estrategia]
                ndcg = fila["ndcg_at_10"]
                f1_10 = fila["f1_at_10"]
                print(
                    f"{contexto:8s} {estrategia:30s} "
                    f"nDCG@10={ndcg['media']:.3f} "
                    f"[{ndcg['p2_5']:.3f}, {ndcg['p97_5']:.3f}] | "
                    f"F1@10={f1_10['media']:.3f} "
                    f"[{f1_10['p2_5']:.3f}, {f1_10['p97_5']:.3f}]"
                )
        techo = diagnostico_a1["techo_teorico"]
        print(
            "techo teorico del gold: "
            f"F1@5 max={techo['f1_at_5_maximo']['media']:.3f} | "
            f"F1@10 max={techo['f1_at_10_maximo']['media']:.3f} | "
            f"n_gold medio={techo['n_gold_por_caso']['media']:.1f} | "
            f"recall universo={techo['recall_universo']['media']:.3f}"
        )
        candidatos = diagnostico_a1["recall_candidatos_primera_etapa"]
        print(
            "recall candidatos top-50: "
            + " | ".join(
                f"{nombre}={datos['media']:.3f}"
                for nombre, datos in candidatos.items()
            )
        )
    if comparacion_a3 is not None:
        print(
            "\n=== Sprint A3: rerankers legales sobre candidatos "
            f"{CONFIG['a3']['slug_embedding']}@{CONFIG['a3']['k']} "
            "(nDCG@10 [IC 95%]) ==="
        )
        for contexto in comparacion_a3["contextos"]:
            for entrada in CONFIG["a3"]["estrategias"]:
                datos = comparacion_a3["metricas_globales"][contexto][
                    entrada["estrategia"]
                ]["ndcg_at_10"]
                print(
                    f"{contexto:8s} {entrada['variante']:11s} "
                    f"nDCG@10={datos['media']:.3f} "
                    f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}]"
                )
        for contexto, por_estrategia in comparacion_a3[
            "pareado_vs_referencia"
        ].items():
            for estrategia, medidas in por_estrategia.items():
                delta = medidas["ndcg_at_10"]
                print(
                    f"pareado {contexto:8s} {estrategia:38s} "
                    f"d_nDCG@10={delta['media_delta']:+.3f} "
                    f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                    f"prob_mejora={delta['prob_mejora']:.3f} | "
                    f"{'gana' if delta['gana'] else 'no concluyente'}"
                )
    if comparacion_a4 is not None:
        print(
            "\n=== Sprint A4: fusion de 1a etapa @100 y profundidad "
            "(nDCG@10 [IC 95%]) ==="
        )
        for contexto in comparacion_a4["contextos"]:
            resumen_fusiones = " | ".join(
                f"{fila['variante']}="
                f"{comparacion_a4['fusion_100']['metricas_globales'][contexto][fila['estrategia']]['ndcg_at_10']['media']:.3f}"
                for fila in comparacion_a4["fusion_100"]["etiquetas"]
            )
            print(f"fusiones {contexto:8s}: {resumen_fusiones}")
            resumen_prof = " | ".join(
                f"top{fila['k']}="
                f"{comparacion_a4['profundidad']['metricas_globales'][contexto][fila['estrategia']]['ndcg_at_10']['media']:.3f}"
                for fila in comparacion_a4["profundidad"]["etiquetas"]
            )
            print(
                f"profundidad {contexto:8s} "
                f"({comparacion_a4['fusion_barrido']}): {resumen_prof}"
            )
    if comparacion_b1 is not None:
        print(
            "\n=== Sprint B1: reranker de dominio vs bge en la ruta "
            "ganadora (nDCG@10 [IC 95%]) ==="
        )
        for contexto in comparacion_b1["contextos"]:
            for etiqueta, estrategia in (
                ("bge", CONFIG["b1"]["estrategia_referencia"]),
                ("b1", CONFIG["b1"]["estrategia"]),
            ):
                datos = comparacion_b1["metricas_globales"][contexto][
                    estrategia
                ]["ndcg_at_10"]
                print(
                    f"{contexto:8s} {etiqueta:3s} "
                    f"nDCG@10={datos['media']:.3f} "
                    f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}]"
                )
        contexto_ganador = CONFIG["b1"]["contexto_ganador"]
        delta = comparacion_b1["pareado_vs_referencia"][contexto_ganador][
            CONFIG["b1"]["estrategia"]
        ]["ndcg_at_10"]
        print(
            f"pareado {contexto_ganador} B1-bge: "
            f"d_nDCG@10={delta['media_delta']:+.3f} "
            f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
            f"prob_mejora={delta['prob_mejora']:.3f} | "
            f"{'gana' if delta['gana'] else 'no concluyente'}"
        )
    if comparacion_b2 is not None:
        print(
            "\n=== Sprint B2: retriever denso de dominio en la ruta B1 "
            "(nDCG@10 [IC 95%]) ==="
        )
        for contexto in comparacion_b2["contextos"]:
            datos = comparacion_b2["metricas_globales"][contexto][
                CONFIG["b2"]["estrategia"]
            ]["ndcg_at_10"]
            print(
                f"{contexto:8s} B2 "
                f"nDCG@10={datos['media']:.3f} "
                f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}]"
            )
        for etiqueta, por_contexto in (
            (
                "v12 (B1 con hibrido E5)",
                comparacion_b2["pareado_vs_referencia_v12"],
            ),
            (
                "externo A2/ALIA@100",
                comparacion_b2["pareado_vs_referencia_externa_a2"],
            ),
        ):
            if not por_contexto:
                print(f"pareado {etiqueta}: no disponible (fallback)")
                continue
            for contexto, medidas in por_contexto.items():
                delta = medidas["ndcg_at_10"]
                print(
                    f"pareado {etiqueta} {contexto:8s} "
                    f"d_nDCG@10={delta['media_delta']:+.3f} "
                    f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
                    f"prob_mejora={delta['prob_mejora']:.3f} | "
                    f"{'gana' if delta['gana'] else 'no concluyente'}"
                )
    salidas = (
        f"{CONFIG['ruta_benchmark']}, {CONFIG['ruta_por_ano']}, "
        f"{CONFIG['ruta_figura']}, {CONFIG['ruta_figura_contaminacion']}"
    )
    if args.diagnostico:
        salidas += (
            f", {CONFIG['ruta_diagnostico']}, "
            f"{CONFIG['ruta_figura_diagnostico_recall']}, "
            f"{CONFIG['ruta_figura_diagnostico_ic']}, "
            f"{CONFIG['ruta_figura_diagnostico_periodo']}"
        )
    if comparacion_a3 is not None:
        salidas += f", {CONFIG['ruta_figura_a3_rerankers']}"
    if comparacion_a4 is not None:
        salidas += (
            f", {CONFIG['ruta_figura_a4_fusion_profundidad']}, "
            f"{CONFIG['ruta_a4_fusion_profundidad']}"
        )
    if comparacion_b1 is not None:
        salidas += f", {CONFIG['ruta_b1_reranker']}"
    if comparacion_b2 is not None:
        salidas += f", {CONFIG['ruta_b2_retriever']}"
    print(
        f"tiempo total: {time.time() - inicio_total:.1f} s | "
        f"salidas: {salidas}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"Error fatal: {exc}")
        sys.exit(1)
