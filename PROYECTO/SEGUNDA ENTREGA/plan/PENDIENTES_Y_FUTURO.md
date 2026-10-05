# Pendientes y trabajo futuro — Juris-PLN (memoria viva)

**Fecha:** 2026-10-05 · Mantenimiento: actualizar al cerrar cada ciclo. Fuente de verdad del
estado: `plan/ESTADO_Y_PENDIENTES.md` + `plan/REGISTRO_EXPERIMENTOS.md` + `resultados/metricas.json`.

## A. Pendientes que dependen de anotación humana (Carlos)

| Ítem | Pack listo | Falta | Cómo |
|---|---|---|---|
| **B3 mini-gold (M7)** | `resultados/gold_mini/` (50 casos, 820 decisiones) + pre-anotación LLM + check (kappa 0,651) | Spot-check humano (10-15 casos ≈ 30-60 min; el total son ~2-4 h) | `scripts/16_anotador_gold.py --anotar --anotador carlos` → `--calcular-acuerdo` |
| **O1 (M4)** | `resultados/validacion_m4/` (200 docs) + 2 proxies LLM (0,415 / 0,299) | Anotación humana (3-10 h) → precisión real del silver | `scripts/21_pack_validacion_m4.py --anotar --anotador carlos` → `--calcular-acuerdo` |
| **C1 gold estándar 150 (M7)** | No iniciado | Anotación grande; la herramienta de B3 sirve de base (extender muestreo a 150) | Por definir |

## B. Trabajo futuro técnico (documentado para retomar sin contexto)

1. **C2-LoRA**: fine-tuning LoRA de `ALIA-es-legal-administrative-7B-Instruct` con datos propios
   (pares de citas B1 + casos) — única vía restante del frente LLM tras el C2 zero-shot negativo
   (query understanding Δ −0,198; rerank listwise Δ −0,217). Modelo base ya descargado (GGUF
   Q4_K_M en ollama como `alia-legal`); para LoRA se usaría el checkpoint HF
   `SINAI/ALIA-es-legal-administrative-7B-Instruct`.
2. **2ª época de B2** (retriever denso): requiere añadir un modo `--continuar` al script 25
   (hoy exige reanudar con el mismo `--epochs/--batch`); candidato: lr 1e-5 desde el checkpoint
   `modelos/b2_retriever/checkpoints/epoca_01`. Ganancia esperada incierta.
3. **B4** — BETO chunk-mean++ (más épocas / max_chunks 96): opcional del plan v3; probable
   retorno bajo (BETO sigue por debajo de LogReg en M4).
4. **M4 con hipótesis nuevas** (tras O1): H3 agregación aprendida de chunks (GPU), features por
   secciones/boilerplate (H2-pendiente), clases raras (cRT/aumentación near-distribution). No
   repetir class weights/calibración/jerárquico/EuroVoc (ya probados).
5. **Calibración de probabilidades del producto** (temperature/isotonic en validación) o marcar
   explícitamente como no calibradas (hoy `predict_proba` crudo).
6. **Reintento de los 48 stubs de C3** (sentencias 2022-2026 aún no publicadas como HTML
   estático en la relatoría): relanzar `scripts/26_c3_descarga_2022_2026.py --descargar` en el
   futuro (es resumible y reintenta stubs).
7. **C3 en el corpus principal**: hoy el corpus 2022-2026 vive separado
   (`data/corpus_2022_2026/`); opción de unificarlo con el histórico para análisis conjuntos.
8. **Demo interactiva (streamlit)**: opcional heredada del plan E2.
9. **Barrido md5 completo** de artefactos congelados (observación QA v2.2) — manifest generado
   en `resultados/md5_manifest_v3.json`.
10. **Reemplazo terminológico "silver NER" → "silver"** en docs antiguos de `plan/` (en curso al
    cierre v3; los nuevos documentos ya usan "silver").

## C. Cierre del ciclo v3 (en ejecución 2026-10-05)

- C5: `documento_e2/informe_final_ciclo_v3.md` + deck + README de entrega v2.2/v3.
- C6: comparativa v1 → v2 → v3 (incluida en el informe).
- C4: consolidación documental del producto v3 (v2.2 técnica + QA v2.2).
- Git: commit local del ciclo v3 en rama `feature/e2-ciclo-v3` (push/PR pendiente de
  credenciales de Carlos; git solo bajo pedido).
- Reindex de codebase-memory tras el commit.

## D. Criterios para retomar (reglas vigentes)

- Gates de adopción: IC 95 % bootstrap + consistencia A/B; smokes no deciden.
- Test canónico M4 reutilizado → usar el **holdout sellado** (`data/holdout_m4_bloqueado.parquet`)
  para gates finales de F2.
- Anti-leakage: excluir las 90 sentencias del benchmark en cualquier entrenamiento de dominio.
- Terminología: **silver** = referencia automática (NER); **gold** = anotación humana.
