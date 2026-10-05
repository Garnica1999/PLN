# Plan de adopción v2.2 — B1 (reranker de dominio) + B2 (retriever denso)

**Estado: PREPARADO, NO EJECUTADO.** Se ejecuta SOLO cuando **B3 (mini-gold humano)** confirme
el gate Sprint B. Fecha: 2026-10-04. Autor: orquestador (ML lead) a partir de los resultados
de los experimentos B1/B2.

## 1. Alcance

- Sustituir en la ruta `--rerank` del producto: E5 → **B2** (`modelos/b2_retriever`) y
  bge-reranker-v2-m3 → **B1** (`modelos/b1_reranker`). Arquitectura sin cambios: híbrido
  0,5·BM25 + 0,5·denso (candidatos top-50) → rerank por chunks → prior de hubs α=1,5; k=10.
- El modo default rápido (BM25 + hubs α=0,75) **no cambia**.
- Evidencia (silver, 90 casos, v13): nDCG@10 **0,6926/0,6907** (sin/con hubs) vs 0,625 de v12
  (Δ +0,0673/+0,0617; IC95 > 0; prob_mejora=1,000); F1@5 **0,478**; sin caídas por periodo.
- Pendiente de control humano (B3) y de medir formalmente recall@50 (gate "+5 pp").

## 2. Cambios de código (ml-coder; ADR-16: archivos vivos + resultados nuevos)

- `scripts/lib/componentes_producto.py`:
  - CONFIG: `ruta_embedding_b2 = "modelos/b2_retriever"`, `ruta_reranker_b1 = "modelos/b1_reranker"`.
  - `VinculoCP`: cargar B2 con la misma interfaz que E5 (SentenceTransformer; ALIA usa prompts
    vacíos, dim 768, max_seq_length 512) y B1 con `RerankerBGE(modelo_id=<ruta local>)`.
  - Modo rerank usa B2+B1; fallback documentado a E5/bge con warning si faltan los modelos
    (o error claro en modo estricto).
- `scripts/lib/reranker_bge.py`: sin cambios (ya acepta modelo local).
- CLI `10_evaluar_sentencia.py`: mismos flags; versión interna "producto v2.2"; salida JSON con
  estructura idéntica.
- Congelamiento: md5 de artefactos v1/v2 intactos; ningún `resultados/*.json` previo se toca.

## 3. QA v2.2

- `scripts/checks_qa_final.py` **v3** (nuevo; v2 preservado como `checks_qa_final_v2.py`):
  mismos 10 casos E2E con expectativas actualizadas (md5 de modelos B1/B2; ranking cambia por
  diseño); control negativo contra los JSON v2.
- Equivalencia producto↔benchmark en 3 casos (ids y scores exactos vs v13).
- Latencia: default sin cambios; rerank con B2+B1 (esperado ≈ v2: 14-18 s; medir; B2 150M es
  más liviano que E5-base, B1 568M igual que bge).
- Drift esperado: clasificador M4 idéntico; retrieval cambia por diseño (documentado).
- Rollback: componentes v2 desde `juris_pln_e2_entrega_v2.zip`; md5 congelados como referencia.

## 4. Criterios de adopción (gate Sprint B, ADR-16)

- **B3**: nDCG@10 con gold humano ≥ 0,66 y sin regresión por periodo; acuerdo aceptable.
- QA v2.2 GO y latencias dentro de los límites (≤10 s default / ≤30 s rerank).
- Si B3 no confirma → **no adopción**; se documenta y el producto queda v2.

## 5. Pasos

1. (Pre) Carlos anota B3 → `16_anotador_gold.py --evaluar` → informe de gate.
2. Decisión de adopción (Carlos + evidencia).
3. ml-coder integra → ml-reviewer revisa → ml-qa-engineer ejecuta QA v2.2 + latencias.
4. Nuevo ADR de adopción; actualizar README/deck/documentación; registrar en
   `REGISTRO_EXPERIMENTOS.md` y `metricas.json`.

## 6. Riesgos y mitigaciones

- Sobreajuste al benchmark de 90 → B3 humano + IC + estabilidad por periodo.
- Latencia del rerank → medir y documentar; opción de cachés existentes.
- VRAM en carga (B1+B2 ≈ 1,8 GB fp16) → guards existentes (`elegir_dispositivo`).
- Rollback simple (artefactos congelados + zip v2).
