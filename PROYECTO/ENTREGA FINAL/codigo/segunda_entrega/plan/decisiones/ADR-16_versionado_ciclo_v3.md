# ADR-16: Política de versionado del ciclo v3 y criterios de adopción

- **Status:** Aceptado
- **Date:** 2026-09-11
- **Context:** El ciclo v2 quedó completo y congelado (retrieval v8, producto v2, QA v2 GO).
  Arranca el ciclo v3 "Recuperación robusta" (Sprint A/B/C) con experimentos nuevos
  (modelos legales ES, RRF, fine-tuning de dominio, gold anotado, LLM local). Se
  necesita una política de versionado y numeración que evite colisiones con la
  numeración del ciclo anterior (los benchmarks v2..v8 ya están usados) y que haga
  trazable cada decisión a su experimento, premisa y evidencia.
- **Decision:**
  1. **Versiones congeladas**: v1 (implementación original) y v2 (ciclo de mejora)
     son INTOCABLES (md5 canónicos en cada JSON). El ciclo actual es **v3**.
  2. **Numeración de resultados**: los benchmarks de retrieval continúan la serie
     (`vinculo_cp_benchmark_v9.json`, `v10`, `v11`, ...); los análisis específicos
     del ciclo usan prefijo `c3_*` (p. ej. `c3_diagnostico_recall.json`,
     `c3_a3_rerankers.json`, `c3_analisis_errores.md`, `gold_mini.json`).
  3. **metricas.json**: los registros del ciclo v3 van a la sección `ciclo_v3`
     (subsecciones por fase/experimento), sin tocar `ciclo_v2` ni claves previas.
  4. **Producto**: permanece en "v2" durante el Sprint A; las adopciones ganadoras
     se documentan como v2.1 (y v2.2 tras Sprint B); el cierre del Sprint C
     produce la versión v3 (docstrings + changelog interno).
  5. **Criterios de adopción (no negociables)**:
     - Toda adopción requiere Δ en la métrica objetivo superior al **IC 95% bootstrap**
       (1000 remuestreos) y consistencia en las mitades A/B del benchmark (o en el gold).
     - Nunca se calibra ni se selecciona con el conjunto de test/reporte final.
     - Anti-leakage: los experimentos de entrenamiento excluyen las 90 sentencias del
       benchmark y respetan `data/splits_m4.parquet`.
     - Si un método no supera el gate, se documenta como **negativo con la misma
       rigurosidad que un positivo** (evidencia, causa, artefacto) y no se adopta.
  6. **Registro de experimentos**: cada experimento se añade a
     `plan/REGISTRO_EXPERIMENTOS.md` (hipótesis, premisa, diseño, resultado,
     decisión, artefacto) antes de avanzar de fase; el ml-reviewer verifica el
     registro en cada gate.
  7. Los zips de entrega (v1, v2, v3) y el código se versionan por ramas + PR
     (push/PR a cargo de Carlos por credenciales del entorno).
- **Alternatives Considered:**
  - Reiniciar la numeración en `v3_*`: descartado (colisiona con la serie v3..v8
    del ciclo v2 que identifica artefactos históricos).
  - Publicar adopciones directamente sin IC bootstrap: descartado (el ciclo v2
    demostró que mejoras de +1-2 pp pueden estar dentro del ruido).
- **Consequences:**
  - (+) Cada decisión del ciclo v3 es auditable: métrica -> experimento -> premisa -> ADR.
  - (+) Los artefactos v1/v2 siguen siendo reproducibles y comparables.
  - (-) Requiere disciplina de registro por experimento (verificado por el reviewer).
