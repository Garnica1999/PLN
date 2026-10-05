# Pendientes y trabajo futuro — Juris-PLN (memoria viva)

**Fecha:** 2026-10-05 · Mantenimiento: actualizar al cerrar cada ciclo. Fuente de verdad del
estado: `plan/ESTADO_Y_PENDIENTES.md` + `plan/REGISTRO_EXPERIMENTOS.md` + `resultados/metricas.json`.

## A. Pendientes que dependen de anotación humana (Carlos)

| Ítem | Pack listo | Falta | Cómo |
|---|---|---|---|
| **B3 mini-gold (M7) — COMPLETADO 2026-10-05** | Carlos anotó **820/820** decisiones (50 casos) | Gate humano **CONFIRMA v2.2**: B2 0,7852 [0,745;0,826] vs B1 0,6903, Δ +0,0948 IC>0; kappa carlos|llm 0,345 | Resultados: `metricas.json b3_gold_humano` + `resultados/gold_mini/` (respaldos LLM-only en `backup/`) |
| **O1 (M4) — COMPLETADO 2026-10-05** | Carlos anotó **200/200** docs (modo lote ciego) | **El silver NO es fiable**: acuerdo humano–silver 23,5 % (κ 0,15); proxies LLM 41,5 %/29,9 %; ADR-19 | Resultados: `metricas.json o1_validacion_humana` + `resultados/validacion_m4/` (respaldo solo-LLM en `backup/`) |
| **C1 gold estándar (M7) — PACK LISTO 2026-10-05** | `resultados/gold_150/` (150 casos: 40 benchmark restantes + 110 C3 2022-2026; 2.740 candidatos) | Anotar cuando quieras (modo lote; guía propia; todos los casos sin contaminación para B1/B2) | `scripts/16_anotador_gold.py --anotar --pack gold_150 --anotador carlos` |

## B. Trabajo futuro técnico (documentado para retomar sin contexto)

> Estado 2026-10-05: ítem 1 (LoRA C2) en curso e ítem 2 (2ª época B2) hecho sin adopción; el ítem 4 queda desaconsejado por ADR-19; quedan B4 y la unificación C3 como opcionales; el resto quedó ejecutado o documentado en esta misma lista.

1. **C2-LoRA — EN CURSO 2026-10-05**: spec `plan/PLAN_c2_lora.md` + ADR-20 (juicio punto a punto "¿cita el fragmento el artículo N?" Sí/No); ALIA-7B descargado (`modelos/alia_7b_instruct`); `scripts/33_c2_lora_train.py` + dataset (`data/c2_lora/sft_v1/`, 20k/3k/1k, balance 50/50, sin leakage); el smoke ajusta la escalera de VRAM (bf16 all-linear no cabe en 16 GB → `--max-seq 512 --target-modules q_proj,k_proj,v_proj,o_proj`); entrenamiento pendiente de lanzar.
2. **2ª época de B2 — HECHA 2026-10-05 (sin adopción)**: entrenada con `--variante ep2`/`--inicializar-desde` (lr 1e-5 desde ep1); val marginal (+0,0007 acc) y benchmark v14 **sin mejora significativa** (Δ nDCG@10 +0,0013 ns sin hubs / −0,0067 ns con hubs) → **ep1 se mantiene**. El script 12 gana `--b2-variante` (v14) para futuras variantes del retriever.
3. **B4** — BETO chunk-mean++ (más épocas / max_chunks 96): opcional del plan v3; probable
   retorno bajo (BETO sigue por debajo de LogReg en M4).
4. **M4 con hipótesis nuevas — DESACONSEJADO por ADR-19**: el silver no es un gold fiable (acuerdo humano 23,5 %); no invertir en H3/H2-pendiente/clases raras sobre este target. Solo con una etiqueta mejor (gold humano tipo C1 para M4) o una redefinición de tarea (área = tema, no citas). No repetir class weights/calibración/jerárquico/EuroVoc (ya probados).
5. **Calibración de probabilidades del producto**: documentado como no calibradas en `arquitectura.md` (§9; interpretar como ranking/confianza relativa); el experimento temperature/isotonic queda como opcional.
6. **Reintento de los 48 stubs de C3 — HECHO 2026-10-05**: reintentados; siguen sin publicarse como HTML estático (ok=0, stub=48, error=0; corpus sin cambios). Reintentar en el futuro si la relatoría los publica (`scripts/26_c3_descarga_2022_2026.py --descargar`).
7. **C3 en el corpus principal**: hoy el corpus 2022-2026 vive separado
   (`data/corpus_2022_2026/`); opción de unificarlo con el histórico para análisis conjuntos.
8. **Demo interactiva (streamlit) — HECHA 2026-10-05**: `scripts/demo_streamlit.py` (reutiliza el pipeline del producto; verificada con AppTest; requiere `streamlit` instalado en el venv). Uso: `python -m streamlit run scripts/demo_streamlit.py`.
9. **Barrido md5 completo** de artefactos congelados (observación QA v2.2) — manifest generado
   en `resultados/md5_manifest_v3.json`.
10. **Reemplazo terminológico "silver NER" → "silver"** en docs antiguos de `plan/` (en curso al
    cierre v3; los nuevos documentos ya usan "silver").

## C. Cierre del ciclo v3 — COMPLETADO 2026-10-05

- C5/C6/C4: informe v3 (`documento_e2/informe_final_ciclo_v3.md`), deck (+10 slides),
  README (sección 10) y comparativa v1→v2→v3; producto v2.2 consolidado (ADR-18, QA GO).
- **Gate B3 humano COMPLETADO**: Carlos anotó 820/820; confirma v2.2 (B2 0,7852 vs B1
  0,6903; Δ +0,0948 IC>0); adenda del informe añadida.
- Git: commits locales + **push + PR #33** en `feature/e2-ciclo-v3` → `main`
  (https://github.com/Garnica1999/PLN/pull/33; merge pendiente de Carlos); herramienta 16
  con lotes + demo + gate B3 incluidos.
- **Zip de entrega v3**: `PROYECTO/juris_pln_e2_entrega_v3.zip` (39,5 MB, 267 entradas;
  espejo del v2 + artefactos v3; excluye data/, modelos B1/B2, npz y lematización).
- Reindex de codebase-memory hecho tras el commit.

## D. Criterios para retomar (reglas vigentes)

- Gates de adopción: IC 95 % bootstrap + consistencia A/B; smokes no deciden.
- Test canónico M4 reutilizado → usar el **holdout sellado** (`data/holdout_m4_bloqueado.parquet`)
  para gates finales de F2.
- Anti-leakage: excluir las 90 sentencias del benchmark en cualquier entrenamiento de dominio.
- Terminología: **silver** = referencia automática (NER); **gold** = anotación humana.
