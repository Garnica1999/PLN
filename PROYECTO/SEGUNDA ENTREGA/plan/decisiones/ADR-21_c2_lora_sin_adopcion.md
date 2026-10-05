# ADR-21: C2-LoRA — el LLM afina pero no supera a los encoders de dominio (sin adopción)

- **Status:** Aceptado (2026-10-05)
- **Date:** 2026-10-05
- **Context:** Tras el C2 zero-shot negativo (ver ADR-20 y `plan/PLAN_c2_lora.md`), se entrenó un
  LoRA (r=16, α=32, qkvo, lr 1e-4, 1 época, 20k pares de B1, bf16 + gradient checkpointing,
  max_seq 512) sobre `SINAI/ALIA-es-legal-administrative-7B-Instruct` para el juicio punto a
  punto "¿el fragmento cita el artículo N de la CP?" (Sí/No). Se evaluó en 3 niveles: val por
  documento, gold humano B3 (820 ítems) y rerank end-to-end con el top-50 recomputado (90 casos).
- **Decision:** **No adoptar** el LoRA en el producto: ninguna variante (rerank top-50, cascada
  con B1, fusiones α∈{0,25;0,5;0,75}) supera a la ruta v2.2 (v13) con IC 95 %; las fusiones
  empatan (+0,0002 a +0,0016, ns) con p95 45,6 s > gate de 30 s. El frente LLM C2 queda
  **cerrado**; el producto conserva los encoders de dominio (B2 + B1).
- **Evidence:**
  - Nivel 1: LoRA acc 0,769 / AUC 0,847 vs base 0,503 / 0,647 → Δacc **+0,265 [0,24; 0,29]**
    prob 1,0 (a un pelo de los umbrales 0,78 / 0,85).
  - Nivel 2 (gold B3): AUC(≥1) 0,654 / Spearman 0,297 vs base 0,597 / 0,200 (Δ significativas);
    la referencia LLM zero-shot puntúa mejor en ese gold (AUC 0,727 / Spearman 0,515).
  - Nivel 3 (top-50 recomputado, 90 casos): top50 0,6842 vs v13 0,6907 (Δ −0,0065 ns); vs base
    bf16 **+0,0518 [0,033; 0,071] prob 1,0** (criterio de adaptación ≥+0,05 → GO); fusiones
    0,6909–0,6923 vs v13 (+0,0002/+0,0015/+0,0016, ns).
  - Incidente: falso stop en el paso 1000 por un bug de padding derecho en la evaluación
    interna del script 33 (corregido; verificado contra evaluador independiente; reanudado
    hasta completar la época con AUC final 0,848).
- **Alternatives Considered:**
  - Adoptar la fusión α=0,75 (+0,0016 ns, p95 45,6 s): rechazada (sin significancia y latencia
    fuera del gate).
  - Reentrenar con más datos/épocas: poco probable que cierre el gap vs encoders (la fusión ya
    empata sin superar); coste GPU alto.
  - Cerrar el frente LLM generativo: aceptado (el valor marginal no justifica más ciclos).
- **Consequences:**
  - (+) El adaptador queda como artefacto documentado y reutilizable (juicio artículo↔fragmento
    con AUC 0,85; útil para análisis offline).
  - (–) El frente LLM generativo no entra al producto; la vía de mejora de M7 sigue siendo
    encoders de dominio (B1/B2).
  - Artefactos: `resultados/c2_lora/*`, `modelos/c2_lora/adapter_v1/`, `scripts/33_c2_lora_train.py`,
    `scripts/34_c2_lora_eval.py`, `metricas.json c2_lora_eval`, REGISTRO `C2-LoRA`.
