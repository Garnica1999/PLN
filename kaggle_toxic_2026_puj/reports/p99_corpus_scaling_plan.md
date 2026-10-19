# P99 - Plan de construccion del corpus S99-01 (60M tokens)

Documento operativo del pre-registro (`p99_preregistration.md`). Todo el
material es LOCAL y autorizado; sin descargas.

## 1. Fuentes y hashes de entrada (verificados en P0 de P99)

| Fuente | Ruta | sha256 (16) | Tokens archivo | Docs |
|---|---|---|---|---|
| Wikipedia es 50M | data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt | 8088bce2a9283fbc | 50.000.028 | 21.529 |
| mC4 es 10M | data/external/corpora/mc4_es/emb_normal/tokens_10M.txt | 361498bf3eca104d | 10.000.239 | 16.738 |
| Twitter COVID-es | data/external/corpora/twitter_domain/corpus_twitter_tokens.txt | 1ef872c189d0f405 | 10.304.387 | 361.994 |

## 2. Reglas de seleccion (replican y extienden las de p98_11)

1. **Wikipedia**: prefijo POR DOCUMENTOS en orden del archivo hasta
   >= 41.0M tokens (C98-1 uso prefijo hasta 18.0M; el prefijo ampliado lo
   contiene). Holgura disponible hasta 50M si la deduplicacion exige mas.
2. **mC4**: prefijo POR DOCUMENTOS hasta >= 10.0M (usa el archivo completo;
   C98-1 uso 7.5M).
3. **Twitter**: MISMA permutacion con seed 20260925 que p98_11 (se re-deriva
   la permutacion completa de 361.994 documentos), extendida hasta >= 9.5M
   tokens seleccionados (C98-1 corto en 4.5M sobre la misma permutacion; el
   nuevo conjunto es superconjunto).
4. **Deduplicacion exacta** por documento normalizado (minusculas, sin
   tildes; sha1) en el orden wiki->mc4->twitter conservando la primera
   aparicion; se registran removidos por fuente.
5. **Guardia de tamano**: si el total post-dedup < 60.000.000 tokens, se
   extiende el prefijo de Wikipedia (paso por documentos) hasta >= 60.0M.
6. **Shuffle final** de documentos con seed 20260925 (documentos intactos).
7. **Validacion de lenguaje**: los mismos 3.320 documentos de P98 (2% por
   fuente, seed 20260925+7 sobre las claves de C98-1); se verifica que todos
   estan presentes en S99-01 (por hash de documento) y se excluyen del stream
   de entrenamiento en el script de entrenamiento.

## 3. Verificaciones obligatorias del manifiesto resultante

- C98-1 es subconjunto de S99-01: los 166.022 documentos de referencia presentes EXACTOS
  (mismo sha1 normalizado); tokens de referencia 29.737.183 presentes.
- Nuevos unicos = tokens_finales - 29.737.183 (>= 30.0M esperado).
- Overlap exacto con el train de TOXIC: 0 (auditoria incluida).
- Overlap cruzado entre fuentes tras dedup: registrado.
- Los 3.320 documentos de validacion de lenguaje presentes.
- Cifras por fuente (seleccionada, removida, final) y documentos finales.

## 4. Salidas

```text
data/external/corpora/p99_s99_01/tokens_60m.txt      corpus final
artifacts/p99_s99_01_manifest.json                   manifiesto completo
reports/p99_corpus_results.csv                       resumen por fuente
```

## 5. Coste y tiempos

Seleccion + dedup + escritura: CPU-minutos (scripts reproducibles; semilla
fija). Sin GPU. Sin descargas. Sin modificacion de los archivos de origen.

## 6. Plan alternativo (si algo falla)

- Si material local < 60M: detener (no descargar), registrar disponible y
  pedir aprobacion.
- Si el total cae entre 59.0-60.0M tras agotar la holgura de Wikipedia:
  detener y reportar antes de entrenar (no entrenar por debajo de 60M sin
  aprobacion, salvo que la seccion 7 del pre-registro lo autorice -
  actualmente exige >= 60.0M).
