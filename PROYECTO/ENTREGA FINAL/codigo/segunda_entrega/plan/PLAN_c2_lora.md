# PLAN C2-LoRA — Juicio de relevancia artículo-fragmento con LoRA sobre ALIA-7B

**Fecha:** 2026-10-05 · **Fase:** ciclo v3, frente C2 (rescate tras zero-shot negativo) · **Autor:** ml-architect
**Responde a:** `metricas.json c2_query_llm` (Δ nDCG@10 −0,198) y `c2_rerank_llm` (Δ −0,217); ítem 1 de
`PENDIENTES_Y_FUTURO.md`. **Premisa:** el fallo zero-shot (ALIA GGUF Q4, dominio administrativo español) es de
dominio y de tarea, no de formato; el ajuste con datos propios debe medirse con el protocolo pareado que refutó C2.
**Decisión de diseño registrada en `plan/decisiones/ADR-20_c2_lora_juicio_punto.md`.**

## 1. Investigación SOTA

### 1.1 Métodos y referencias (verificadas por arXiv API / docs oficiales, 2026-10-05)

| Método / referencia | Aporte | Métrica reportada (dominio) | Viabilidad 16 GB ROCm |
|---|---|---|---|
| LoRA — Hu et al. 2021, arXiv:2106.09685 | Adaptadores de bajo rango (r≪d); sin latencia extra al fusionar | RoBERTa/GPT-2/3: paridad con full-FT en varios benchmarks | Base del stack `peft` ya instalado |
| QLoRA — Dettmers et al. 2023, arXiv:2305.14314 | NF4 + doble cuantización + paged optimizers; r=64, α=16, do=0.1, lr 2e-4, seq 1024, bs 16 | 65B en 48 GB; GPT-4-level en Vicuna | Posible pero exige `bitsandbytes` (contingencia) |
| LoRA Learns Less — Biderman et al. 2024, arXiv:2405.09673 | En instruction tuning r=8–16 basta; menor olvido que full-FT | Programación/matemáticas: LoRA < full-FT en rango bajo, pero más robusto fuera de dominio | Recomienda r bajo para tarea estrecha |
| LoRA+ — Hayou et al. 2024, arXiv:2402.12354 | LR diferenciado para A/B (B ×2–16) | Mejora en GLUE/ViT con mismo coste | Mejora opcional (iteración v2) |
| LegalMALR — Li et al. 2026, arXiv:2601.17692 | Query understanding multiagente + rerank LLM para statute retrieval chino | nDCG/recall superiores a densos+rerankers ligeros (chino) | Patrón SOTA del frente C2; el remedio al zero-shot es adaptación con datos propios |
| COLIEE 2026 Team DU — Alshehri et al. 2026, arXiv:2607.11400 | Ensambles cross-architecture + rerank por features + prompting RAG | 96,3 % accuracy en statute entailment (1.º/33 equipos) | Respalda la ruta de fusión/cascada con B1 |
| SaulLM-7B — Colombo et al. 2024, arXiv:2403.03883 | Adaptación legal = continued pretraining + instruction tuning | SOTA legal inglés (30B tokens) | Inviable full-FT; análogo factible = LoRA SFT |
| Salamandra — Gonzalez-Agirre et al. 2025, arXiv:2502.08489 | Familia base 2B/7B/40B, 35 lenguas europeas; contexto 8k | Base del modelo ALIA | Modelo local ya descargado (~15,5 GB bf16) |
| Lawyer LLaMA — Huang et al. 2023, arXiv:2305.15062 | Instruction tuning legal + retrieval augmentation | Mejoras en consultas legales (chino/inglés) | Precedente de SFT de dominio |
| bitsandbytes ROCm docs (URL en §1.3) | Ruedas PyPI ROCm 7.2.4 con target gfx1201 (RX 9070 XT) | — | Solo contingencia QLoRA |
| SINAI (HF): `ALIA-es-legal-administrative-7B-Instruct` (Apache-2.0) y dataset sintético de instrucciones | Modelo y datos base | — | Modelo local verificado (`config.json`, tokenizer 256k) |

### 1.2 Hiperparámetros heredados (baseline del primer experimento)

| Config | QLoRA (Dettmers 2023) | LoRA instruct (Biderman 2024) | **Este plan — v1** |
|---|---|---|---|
| r / alpha / dropout | 64 / 16 / 0,1 | 8–16 / 16–32 / 0,05 | **16 / 32 / 0,05** |
| lr / scheduler | 2e-4 paged AdamW | 1e-4–2e-4 | **1e-4, cosine, warmup 3 %** |
| batch efectivo | 16 | 16–128 | **1 × acumulación 16** |
| max_seq / epochs | 1024 / 1–3 | 1024–2048 / 1–3 | **768 / 1 (máx 2, early stopping)** |
| target modules | all-linear | q,v típico | **q,k,v,o,gate,up,down** |

### 1.3 Decisión bf16-LoRA vs QLoRA (razonada)

- **bf16-LoRA (Plan A):** sin dependencias nuevas. Pesos base 14,47 GiB + LoRA r=16 all-linear (~37 M params:
  0,41 GiB con AdamW fp32) + activaciones con checkpointing seq 768 batch 1 (~0,25 GiB) + contexto ROCm (~0,4 GiB)
  ≈ **15,5–15,8 GiB de 15,93 GiB** → margen 0,1–0,4 GiB: viable solo con GPU exclusiva. Obligatorio:
  `logits_to_keep` (vocab 256k: logits completos costarían ~1 GiB), checkpointing, batch 1, `expandable_segments`.
- **QLoRA NF4 (contingencia):** bitsandbytes 0.50.2 publica rueda `py3-none-manylinux` con build ROCm 7.2.4 y
  gfx1201; en teoría instalable en Python 3.14 y con ~10 GiB de margen. Riesgo: kernels NF4 de entrenamiento no
  validados en py3.14 + torch 2.14 + ROCm 7.2 (ninguna pieza GPU del proyecto usa bnb). Solo si el ladder de OOM
  falla y un smoke de 20 steps valida bnb.
- Descartados: full fine-tuning (imposible en 16 GB) y modelos menores (rompen la comparabilidad con C2 zero-shot).

## 2. Diseño del dataset

### 2.1 Tarea elegida y argumentos

**Tarea principal: juicio binario punto a punto "¿el fragmento de sentencia cita el artículo N de la CP?" con
respuesta generada restringida a "Sí"/"No"** (supervisión = etiqueta de los pares B1: 150.000 citas silver vs
149.999 negativos minados por BM25/co-cita/aleatorios).

- Supervisión directa y abundante, ya anti-leakage (B1 excluye las 90 del benchmark); es el análogo LLM de B1 →
  comparación justa 7B vs 568M sobre la misma señal.
- Evaluable en 3 niveles: (a) val por doc (misma partición que B1); (b) gold humano B3 (820 grados, 50 casos,
  subconjunto del benchmark = holdout limpio); (c) rerank end-to-end con el protocolo pareado de scripts 30/31.
- Uso barato en inferencia: P("Sí") por logits en un forward (sin generar), fusionable/cascada con B1.
- Descartadas: (i) **rerank listwise como tarea principal** (supervisión de orden débil, salida JSON larga, coste
  alto y ya refutada zero-shot; queda como extensión futura); (ii) **query understanding** (no hay targets de
  consulta; su evaluación es aguas abajo); (iii) **vínculo generativo sentencia→artículos** (384 artículos en
  contexto 8k, inviable y sin señal nueva frente al retriever).
- Caveat de etiqueta (documentado): 182/559 candidatos "citados" del gold B3 fueron grado 0 humano (ruido ~33 % del
  silver); el juez aprende el criterio silver igual que B1, y la correlación con relevancia humana se **mide**, no
  se asume.

### 2.2 Formato (chat template del tokenizer, verificado)

- `system`: "Eres un asistente jurídico experto en derecho constitucional colombiano. Respondes únicamente 'Sí' o 'No'."
- `user`: `FRAGMENTO DE SENTENCIA:\n{chunk}\n\nARTÍCULO {art} DE LA CONSTITUCIÓN:\n{pasaje}\n\n¿El fragmento cita el artículo {art}? Responde solo "Sí" o "No".`
- `assistant`: `Sí` (label 1) / `No` (label 0). EOS `<|im_end|>` (id 5). Tokens de respuesta verificados: `Sí`=34351,
  `No`=2125 (1 token cada uno; re-verificar en runtime).
- Render con `tokenizer.apply_chat_template(add_generation_prompt=True)`; loss solo en las posiciones de respuesta.

### 2.3 Construcción determinista (script 33 `--dataset`)

- Fuente: 23 shards `data/pares_b1/pares_reranker_b1_shard*.parquet` (299.999 filas, 22.994 docs, 384 artículos).
- Filtros duros: `doc_id ∉ 90 benchmark` (`resultados/vinculo_cp_benchmark_v2.json.muestra`; **abortar** si aparece
  alguno); dedupe por `(doc_id, articulo, chunk_idx, pasaje_id)` y por md5 de `(texto_chunk, texto_pasaje, label)`.
- Split por documento replicando B1: `docs=sorted(...)`, `random.Random("42:split:doc")`, `val_frac=0.05` →
  21.844 train / 1.150 val docs; assert disjunción y coincidencia exacta con los docs de B1.
- Muestreo train (semilla 42): RNG por doc `random.Random("42:c2lora:doc:{doc_id}")`, hasta 3 positivos y 2
  negativos por doc (mezcla de fuentes heredada de B1; reportar la realizada), cap global **20.000** (10.000/10.000)
  con shuffle `random.Random("42:c2lora:global")`.
- Val: `val_eval.parquet` = 1.000 fijos estratificados (500/500) para evaluación periódica; `val.parquet` = hasta
  3.000 (1.500/1.500) para métricas finales; solo docs val.
- Textos: chunk ≤ 600 chars, pasaje ≤ 800 chars (≈ 400–500 tokens); truncado del tokenizer a `max_seq=768` como red
  de seguridad; medir % truncado (criterio < 2 %).
- Salidas: `data/c2_lora/sft_v1/{train,val,val_eval}.parquet` + `resultados/c2_lora/dataset_v1.json` (conteos,
  balance, mix de fuentes, longitudes, md5) + `figuras/c2_lora_dataset.png`.

### 2.4 Anti-leakage

- Nunca entrenar con las 90 del benchmark (incluye los 50 casos del gold B3); nunca entrenar con los grados
  humanos; val = docs de B1; semillas fijas y artefactos con md5. No se genera split test interno: el **test externo**
  son el benchmark (90 casos, silver) y el gold B3 (50 casos/820 ítems, humano), usados solo en evaluación.

## 3. Entrenamiento — `scripts/33_c2_lora_train.py`

### 3.1 Esquema de secciones (estilo del proyecto, sin emojis)

1. CONFIG (rutas, semilla 42, PEFT, entrenamiento, caps, smoke).
2. Utilidades: logging, semillas, md5, `versiones_entorno`, guardas de no-sobrescritura.
3. `--dataset`: construcción de §2.3.
4. Modelo: tokenizer + `AutoModelForCausalLM` bf16, `attn_implementation="sdpa"`,
   `gradient_checkpointing_enable(use_reentrant=False)`, `enable_input_require_grads()`, `use_cache=False`,
   `LoraConfig`, `get_peft_model`.
5. Collator completion-only: render del chat, labels −100 en prompt, padding, `logits_to_keep` = tokens de respuesta
   (verificar nombre de la firma en transformers 5.16; fallback: slicing de logits a las últimas posiciones).
6. Bucle manual: AdamW, cosine + warmup 3 % (min_lr 10 %), autocast bf16 (sin GradScaler), grad clip 1.0,
   grad-accum, eval cada 250 steps, early stopping (patience 4), checkpoints reanudables, VRAM pico, tokens/s.
7. Evaluación interna val: loss, accuracy, F1, AUC (P("Sí") por logits), matriz de confusión, figura.
8. Cierre: `resultados/c2_lora/entrenamiento_v1.json` (config, curvas, mejor paso, tiempos, huellas).
9. CLI: `--dataset --entrenar --todo --smoke --max-ejemplos --max-horas --reanudar --max-seq --lr --r`.

### 3.2 Config PEFT concreta

`LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])` · lr 1e-4 · wd 0,0 ·
AdamW (0,9/0,999; eps 1e-8) · batch 1 × acumulación 16 · max_seq 768 · epochs 1 (máx 2) · packing **no**
(ejemplos homogéneos y enmascarado completion-only; no aporta). Checkpoints cada 250 steps en
`modelos/c2_lora/checkpoints/ultimo/` (adapter + optimizer + scheduler + step + RNG) y mejor en
`modelos/c2_lora/adapter_v1/`.

### 3.3 VRAM y tiempo

- VRAM (desglose en §1.3): pico esperado 15,5–15,8 GiB; guard previo: abortar si `torch.cuda.mem_get_info` < 15,5 GiB
  libres; cerrar Chrome/navegador (precedente B1: contención GPU por Chrome).
- **Paso 0 obligatorio:** benchmark de throughput (50 steps) en el smoke; esperado 300–800 tok/s.
- Tiempo estimado: 20.000 ejemplos × ~500 tok ≈ 10 M tokens → **3,5–9 h por época** (sesión con `--max-horas 6` +
  `--reanudar`). Piloto de señal: 4.000 ejemplos → 0,7–2 h. Eval periódica +2–4 % de overhead.
- Comando:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --dataset --entrenar > experimentos/33_c2_lora_train.log 2>&1 &`

### 3.4 Mitigación de riesgos de ejecución (ladder OOM/estabilidad)

1. Reducir `max_seq` 768→512. 2. `expandable_segments:True` + GPU exclusiva. 3. Reducir target_modules a
q,k,v,o (LoRA ~19 M). 4. QLoRA NF4 (contingencia §1.3) o lr 5e-5 si hay inestabilidad/NaN. Ante muerte silenciosa
(precedente B1): `setsid`, log con flush, checkpoint cada 250 steps y `--reanudar`. Compatibilidad py3.14 +
transformers 5.16 + peft 0.21: el **smoke de 20 steps es gate obligatorio** antes del run largo.

## 4. Evaluación y comparación — `scripts/34_c2_lora_eval.py`

### 4.1 Protocolo (mismo rigor que scripts 30/31)

- **E1 val interna:** accuracy/F1/AUC sobre `val.parquet`; B1 recalculado sobre el mismo subconjunto para
  comparación pareada (referencia B1 val AUC 0,8513).
- **E2 gold B3 (820 ítems / 50 casos):** Spearman (score vs grado), AUC(≥1), AUC(=2), exactitud binaria; nDCG@10
  graduado por caso (ganancia 2^g−1) reutilizando la maquinaria de `scripts/16_anotador_gold.py` con la ruta nueva;
  reportar `n_items_fuera_pool` y comparar con el control zero-shot bf16, con los grados congelados qwen3.8/gemma4
  (`gold_mini/respuestas_llm*.jsonl`; κ humano 0,3448 como referencia) y con v13 (0,7852).
- **E3 benchmark (90 casos, referencia silver):** reutilizar por importlib la maquinaria de `scripts/12` y `23`
  (hibrido top-50, chunks 800/200, pasajes 120 palabras, max-pooling por artículo) para que los pares coincidan con
  la inferencia B1; score LoRA = P("Sí") por logits; rutas: `c2_lora_top50` (completa),
  `c2_lora_cascada_b1_top20` (B1 top-50 → top-20 → LoRA; gate de latencia) y `c2_lora_fusion_b1` (α=0,5 primario;
  rejilla 0,25/0,75 como sensibilidad). Métricas nDCG@10 (primaria), F1@10, MRR, P@5, R@5; **bootstrap pareado
  1000/42** vs v13 recalculada, vs C2(b) zero-shot congelado (`resultados/c2_llm/eval_rerank_llm.json` →
  `evaluacion.por_caso`), vs C2(a) (0,493) y vs B1; consistencia A/B por mitades de casos; latencia por caso
  (media/p95/max; gate p95 ≤ 30 s).
- **Control zero-shot controlado:** misma pasada con `model.disable_adapter()` (base bf16, mismos prompts) → la
  comparación "LoRA vs zero-shot" primaria es bf16-vs-bf16; el GGUF Q4 congelado queda como referencia de producto
  (caveat de cuantización documentado).
- Cache de scores por md5(prompt) en `resultados/c2_lora/cache_scores_v1.jsonl` para reutilizar E2/E3.

### 4.2 Carga del adaptador para inferencia

- Por defecto `PeftModel.from_pretrained(base_bf16, modelos/c2_lora/adapter_v1)` sin merge (no duplica 15,5 GB);
  `model.eval()` + `torch.inference_mode()`. `--merge` opcional (`merge_and_unload()`) con paridad verificada en 100
  prompts (max|ΔP| ≤ 1e-3) para despliegue; conversión a GGUF Q4_K_M para ollama solo como paso de producto opcional
  (documentar, no requerido por este plan).

### 4.3 Criterios de éxito / parada

| # | Criterio | Umbral |
|---|---|---|
| S1 | Integridad de datos | 0 docs benchmark en train/val; split idéntico a B1; balance 50±2 %; truncados < 2 % |
| S2 | Aprendizaje (val) | acc ≥ 0,78 y AUC ≥ 0,85 (≥ B1 val: acc 0,7671/AUC 0,8513); Δ acc vs base bf16 ≥ +0,15 (IC95 > 0) |
| S3 | Juicio vs gold humano | AUC(≥1) ≥ 0,75 y Spearman ≥ 0,30; ≥ base bf16 + 0,10 AUC (IC95 > 0) |
| S4 | Rerank end-to-end | LoRA ≥ zero-shot bf16 + 0,05 nDCG@10 (IC95 > 0) = adaptación funciona. **Adopción producto:** cascada o fusión > v13 (0,6907) con Δ ≥ +0,01, IC95 > 0, A/B consistente, gold humano Δ ≥ −0,005 y p95 ≤ 30 s |
| S5 | Eficiencia | ≤ 12 h entrenamiento default; 0 OOM fatales; pico VRAM ≤ 15,9 GiB |
| Parada | Diagnóstico | val AUC < 0,70 tras 1.000 steps → parar y diagnosticar; 3 evals sin mejora → early stop; si LoRA ≤ zero-shot bf16 + 0,02 en E3 → cerrar el frente LLM como negativo (ADR) |

## 5. Riesgos y límites

- **Anti-leakage:** verificación dura en el script (abort si doc del benchmark); gold B3 es holdout (⊆ benchmark);
  val = docs B1; jamás calibrar umbrales/α con test (α primario pre-registrado; rejilla solo como sensibilidad).
- **Sobreajuste y ruido:** r=16 + dropout 0,05 + 1–2 épocas + early stopping; vigilar colapso "todo Sí/No" con la
  matriz de confusión; ~33 % de los "citados" silver son grado 0 humano → techo esperado similar al de B1
  (v12_b1 gold 0,6903); el experimento mide adaptación, no verdad absoluta.
- **Comparabilidad:** GGUF Q4 zero-shot vs bf16 LoRA (confound de cuantización) → se resuelve con el control bf16
  sin adaptador; listwise zero-shot vs juicio punto a punto (interfaces distintas) → la comparación de producto es
  vía E3 con las rutas completas; silver vs gold humano se reportan por separado.
- **Coste/operación/dependencias:** 4–10 h de entrenamiento + 2–4 h de evaluación con GPU serial; muertes silenciosas
  mitigadas con checkpoints/reanudación; disco: adapter ~75 MB, checkpoints ~0,5 GB, merge opcional 15,5 GB; verificar
  los 4 safetensors del modelo base contra `model.safetensors.index.json` antes de entrenar; stack bleeding-edge
  (py3.14/transformers 5.16/peft 0.21/ROCm) cubierto por el smoke gate.

## 6. Fases, tareas y presupuesto

| Fase | Tarea (script) | Acceptance criteria | Depende de |
|---|---|---|---|
| F1 | Verificar modelo base + smoke 20 steps (`33 --smoke`) | Carga bf16 OK, 20 steps sin OOM, throughput medido | Descarga completa |
| F2 | Dataset (`33 --dataset`) | S1 completo + `dataset_v1.json` con md5 | F1 |
| F3 | Piloto de señal (`33 --entrenar --max-ejemplos 4000 --max-horas 3`) | Curva val creciente; AUC ≥ 0,75 al final del piloto | F2 |
| F4 | Entrenamiento completo (`33 --entrenar`, reanudable) | S2 y S5 | F3 |
| F5 | Evaluación (`34 --gold --benchmark --control`) | S3, S4 y JSONs `eval_gold_v1.json` / `eval_benchmark_v1.json` | F4 |
| F6 | Registro + ADR + cierre | `REGISTRO_EXPERIMENTOS.md` + `metricas.json` (`c2_lora_*`) + figura + decisión GO/PARTIAL/STOP | F5 |

**Experimento 1:** `c2_lora_juicio_punto_v1` (script 33 + 34). Iteraciones sugeridas: `v2` = r=32 o 2 épocas o
LoRA+ si hay underfitting; `v3` = extensión a listwise (adaptador aparte) si S4 es positivo pero la adopción falla.
Figuras: `figuras/c2_lora_curvas.png`, `fig_c2_lora_gold.png`, `fig_c2_lora_eval_benchmark.png`. Logs:
`experimentos/33_c2_lora_train.log`, `experimentos/34_c2_lora_eval.log`.

## 7. Open Questions

1. ¿Conviene un umbral de decisión sobre P("Sí") para la ruta de fusión, o el score continuo basta? (Se decide con
   val, nunca con test; skill `threshold-optimization` si aplica).
2. ¿El piloto de 4.000 ejemplos basta para estimar la meseta, o el juez necesita diversidad de más docs? (Decidir
   con la curva S2 y extender `--max-ejemplos` si sigue mejorando).
3. ¿Vale la pena la extensión listwise (v3) o el frente LLM se cierra con el juicio punto a punto? (Depende de S4).
