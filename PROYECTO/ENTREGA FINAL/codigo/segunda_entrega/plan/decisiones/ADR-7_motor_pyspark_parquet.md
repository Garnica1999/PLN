# ADR-7 — Motor PySpark + Parquet particionado

- **Status:** Aceptado (2026-09-08; verificado por orquestador: métricas coinciden con `metricas_formato.json`, parquet 1.5G/341 archivos, E1 intacta)
- **Date:** 2026-09-08
- **Context:** El corpus E2 son 23.750 filas en `../PRIMERA ENTREGA/data/Juris1992_2022.csv` (1,675,758,001 bytes, 1.561 GB; columnas `['', 'Texto']`). E1 demostro que 8 filas no son sentencias (6 autos A-xxx + 2 basura HTML) y que la limpieza exacta (minusculas, enie protegida, NFD sin diacriticos, filtro `[a-zñ]`, split, filtro 1-letra no gramatical + `art`) es costosa (~200 s en streaming). Las fases D-M necesitan lecturas repetidas por tipo y ano sin reparsear el CSV gigante.
- **Decision:** Adopcion directa por criterio: migrar una vez a `data/corpus_sentencias.parquet` particionado Hive `tipo=/ano=` con columnas `(id, tipo, numero, ano, texto, texto_limpio)` usando PySpark `local[*]` driver 8g con Java 17 (`/usr/lib/jvm/java-17-openjdk`). La limpieza E1 se porta verbatim (mismas funciones `normalizar`, `limpiar_texto`, `parsear_prefijo`, `filtrar_tokens`) y se verifica contra `df_metadatos.csv` de E1. Implementado en `scripts/00_csv_a_parquet.py`; metricas en `resultados/metricas_formato.json`; log en `experimentos/migracion_parquet.log`.
- **Alternatives Considered:**
  - Seguir en CSV: descartado (sin particionado, lectura completa de 1.6 GB en cada experimento, sin tipos).
  - polars/duckdb: descartado por criterio (sin cluster, sin ecosistema Spark para el criterio de adopcion; se reserva como alternativa si Spark resulta inviable en RAM).
  - Fallback pandas+pyarrow (`dataset.write_dataset`): implementado como fallback documentado dentro del mismo script; se uso en la primera ejecucion tras OOM de Spark y produce el mismo particionado logico.
- **Consequences:** Lecturas E2 por tipo/ano sin reparsear; E1 queda intacto (solo lectura). Coste: dataset final 1.465 GB (doble columna texto+texto_limpio) y 341 archivos Hive; primer intento Spark con 32 slices fallo por OOM (`task 59680 KiB`, `Java heap space`), corregido con 256 slices sin `count()` previo. Riesgo: Java 26 por defecto incompatible con Spark 4.2.0 (`ClassNotFoundException: jdk.internal.ref.Cleaner`); mitigado fijando `JAVA_HOME` a Java 17.

## Tabla de metricas reales de adopcion (2026-09-08, ejecucion 2)

| Metrica | Valor |
|---|---|
| Motor usado | spark (PySpark 4.2.0, Java 17.0.19; Java 26 por defecto incompatible, error exacto en log) |
| Filas entrada / salida / excluidas | 23750 / 23742 / 8 (auto 6, basura_html 2; posiciones 254, 13086, 13087, 14072, 14073, 16200, 16268, 21338) |
| Distribucion por tipo | T 17099, C 6246, SU 397 |
| Rango de anos | 1992-2021 (2021=20 docs) |
| Tamano CSV vs Parquet | 1,675,758,001 bytes (1.561 GB) vs 1,573,196,021 bytes (1.465 GB, 341 archivos Hive `tipo=/ano=` + `_SUCCESS`/`.crc`) |
| Tiempo total | 237.5 s (pasada CSV ~202 s + verificacion + escritura Spark) |
| RAM pico | 7966.5 MB (7.78 GB) |
| Verificacion limpieza E1 | 23742/23742 documentos coinciden en `n_palabras_clean` con `df_metadatos.csv` (100.0000 %, 0 divergencias) |
| Columnas Parquet | id, tipo, numero, ano, texto, texto_limpio |
| Fallback | false (intento 1 OOM documentado; intento 2 Spark OK tras 256 slices) |
| Rutas | `data/corpus_sentencias.parquet`, `resultados/metricas_formato.json`, `experimentos/migracion_parquet.log` |
