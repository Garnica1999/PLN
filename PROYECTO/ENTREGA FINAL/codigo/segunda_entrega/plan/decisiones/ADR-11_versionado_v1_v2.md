# ADR-11: Política de versionado v1/v2 y cambio de convención de scripts

- **Status:** Aceptado
- **Date:** 2026-09-10
- **Context:** El ciclo E2 v1 está completo, auditado y entregado (QA GO,
  informe CRISP-DM, deck, zip). Al iniciar el ciclo de mejora v2 (benchmark más
  profundo + mejoras incrementales al retrieval y al producto), Carlos definió
  una política distinta a la convención v1 ("1 experimento = 1 script nuevo,
  nunca editar los previos"): las mejoras del producto se aplican SOBRE los
  archivos existentes, preservando los resultados de experimentación v1 como
  evidencia. Además, el diagnóstico en vivo (caso T-6251: 0/2 artículos
  correctos, top-5 contaminado de transitorios, 45 IDs malformados) exige
  modificar scripts ya entregados (02, 10, 12, lib/componentes_producto.py)
  en lugar de crear variantes paralelas.
- **Decision:**
  1. La implementación entregada queda congelada como **implementación v1**:
     los valores en `resultados/*.json` son INTOCABLES y sirven como baseline
     de comparación. Ningún script ni resultado v1 se sobrescribe.
  2. El ciclo de mejora actual es **v2** (especificado en
     `plan/PLAN_v2_ciclo_mejora.md`, aprobado 2026-09-10).
  3. **Convención de código v2**: las mejoras se aplican SOBRE los archivos
     existentes (02_parseo_constitucion.py, 10_evaluar_sentencia.py,
     12_evaluar_vinculo_cp_v2.py, lib/componentes_producto.py), con la versión
     documentada en el docstring del archivo. La convención v1 "1 experimento
     = 1 script nuevo" queda sustituida para el producto.
  4. **Resultados nuevos con sufijo**: cada benchmark nuevo escribe a un
     archivo nuevo (`vinculo_cp_benchmark_v3.json`, `_v4.json`, ...); los
     archivos v1/v2 previos nunca se sobrescriben.
  5. Los datos regenerados conservan respaldo: el parquet de la constitución
     v1 se respalda como `data/constitucion_articulos_v1.parquet` antes de
     regenerar.
- **Alternatives Considered:**
  - Mantener la convención v1 (scripts nuevos 13, 14...): descartada por
    decisión de Carlos; duplicaría el pipeline del producto y dificultaría el
    mantenimiento de un único archivo vivo.
  - Sobrescribir también los resultados v1: descartada explícitamente; los
    valores v1 son evidencia del estado entregado y baseline del ciclo.
- **Consequences:**
  - (+) Un único pipeline vivo del producto, sin ramas paralelas de código.
  - (+) Trazabilidad completa: v1 congelada como baseline, v2 medible contra
    ella en cada fase (v3 -> v3-limpio -> v4).
  - (+) El diff de código entre v1 y v2 es auditable con git cuando Carlos lo
    pida.
  - (-) Requiere disciplina: cada modificación debe documentar la versión en
    el docstring y NO tocar los JSON de resultados previos.
  - (-) El QA de regresión (checks_qa_final.py) debe actualizarse en la fase X
    para validar la v2 sin invalidar los checks v1.
