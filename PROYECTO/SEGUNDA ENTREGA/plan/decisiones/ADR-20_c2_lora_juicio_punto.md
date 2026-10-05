# ADR-20: C2-LoRA — tarea de juicio punto a punto y bf16-LoRA (no QLoRA) en 16 GB ROCm

- **Status:** Propuesto (2026-10-05)
- **Date:** 2026-10-05
- **Context:** El frente C2 zero-shot cerró negativo (query understanding Δ nDCG@10 −0,198; rerank listwise
  Δ −0,217; `metricas.json c2_query_llm`/`c2_rerank_llm`). La única vía restante documentada es el fine-tuning
  LoRA de `SINAI/ALIA-es-legal-administrative-7B-Instruct` con datos propios (ítem 1 de
  `PENDIENTES_Y_FUTURO.md`). Recursos: GPU RX 9070 XT 16 GB (ROCm 7.2, gfx1201), 32 hilos, 31 GB RAM, GPU serial;
  modelo base 7,77B params, vocab 256k, contexto 8k, bf16 (15,54 GB decimales); `peft`+`accelerate`+`transformers`
  instalados, `trl` y `bitsandbytes` ausentes; Python 3.14.
- **Decision:**
  1. **Tarea principal:** juicio binario punto a punto "¿el fragmento de sentencia cita el artículo N de la CP?"
     (respuesta restringida "Sí"/"No") entrenado con los pares B1 (150k citas silver vs 149.999 hard negatives),
     evaluado como juez (gold humano B3, 820 ítems) y como reranker (benchmark 90 casos, protocolo pareado).
  2. **Precisión:** bf16-LoRA (r=16, α=32, dropout 0,05, all-linear) con recetas de memoria (`logits_to_keep`,
     gradient checkpointing, batch 1 × grad-accum 16, max_seq 768, `expandable_segments`); QLoRA NF4 solo como
     contingencia tras smoke que valide bitsandbytes en ROCm.
  3. **Dataset:** subset determinista de 20.000 ejemplos (semilla 42) con split por documento idéntico al de B1
     (21.844/1.150 docs) y exclusión dura de las 90 sentencias del benchmark y de los packs gold.
- **Evidence:**
  - Presupuesto de VRAM: 14,47 GiB pesos + 0,41 GiB LoRA/AdamW + 0,25 GiB activaciones (checkpointing) +
    ~0,4 GiB contexto ≈ 15,5–15,8 GiB vs 15,93 GiB → margen 0,1–0,4 GiB; exige GPU exclusiva (precedente B1:
    contención por Chrome).
  - QLoRA: bitsandbytes 0.50.2 publica rueda py3-none-manylinux con build ROCm 7.2.4 y target gfx1201
    (docs oficiales, verificado 2026-10-05); instalaría en py3.14, pero los kernels NF4 de entrenamiento no están
    validados en este stack y añaden dependencia frágil; el ahorro de memoria no compensa el riesgo en la vía
    crítica (el proyecto ya tiene stack GPU estable con peft/transformers).
  - SOTA: LoRA r=8–16 es suficiente en instruction tuning y olvida menos (Biderman 2024, arXiv:2405.09673); QLoRA
    reporta r=64/α=16/lr 2e-4/seq 1024 (Dettmers 2023, arXiv:2305.14314); adaptación de dominio legal con datos
    propios es el patrón ganador (SaulLM-7B arXiv:2403.03883; LegalMALR arXiv:2601.17692; COLIEE 2026
    arXiv:2607.11400).
  - Ruido del target: 182/559 candidatos citados (silver) fueron grado 0 humano en B3 → el juez aprende el criterio
    silver (como B1); la correlación con relevancia humana se mide explícitamente.
- **Alternatives Considered:**
  - Rerank listwise como tarea principal: supervisión de orden débil (etiquetas binarias), salida JSON larga,
    coste alto y baseline zero-shot ya refutado (0,473); se pospone a v3 condicionada a resultados.
  - Query understanding con LoRA: no existen targets de consulta; evaluación solo aguas abajo; rechazado.
  - Vínculo generativo sentencia→artículos: 384 artículos en contexto 8k; inviable y sin señal nueva vs retriever.
  - Full fine-tuning / modelos menores: imposible en 16 GB / rompe comparabilidad con C2 zero-shot.
  - QLoRA como Plan A: rechazado por dependencia y validación (queda como contingencia).
- **Consequences:**
  - (+) Experimento ejecutable en GPU serial con presupuesto acotado (4–10 h de entrenamiento + 2–4 h de
    evaluación); comparación controlada bf16 base vs LoRA, más referencia de producto GGUF Q4 congelada.
  - (+) La tarea punto a punto permite inferencia por logits (1 forward/pair), fusionable con B1 y con gate de
    latencia evaluable (cascada top-20).
  - (–) Margen de VRAM estrecho: cualquier proceso GPU concurrente puede provocar OOM; obligatorio el ladder de
    mitigación y checkpoints reanudables.
  - (–) El juez hereda el techo del silver (~33 % de ruido en positivos); la adopción de producto exige superar a
    v13 con IC95 + A/B + gold humano, un listón alto.
  - Artefactos: `plan/PLAN_c2_lora.md` (spec), scripts 33/34, `data/c2_lora/sft_v1/`,
    `modelos/c2_lora/`, `resultados/c2_lora/*.json`, `figuras/c2_lora_*.png`, REGISTRO `C2-LoRA`.
