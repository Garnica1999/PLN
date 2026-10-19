# P100 - Auditorias S1/S2 (validacion por agentes)

Fecha: 2026-10-13. Registro de las validaciones solicitadas por el propietario
("validar codigo, afirmaciones e hipotesis") al cierre de S1 (fuentes) y S2
(pre-registro).

## 1. ml-reviewer - codigo S1

- Alcance: experiments/p100_00_registro_fuentes.py y sus 3 salidas.
- Veredicto: APROBADO (confianza alta).
- Verificaciones: ruff limpio; ejecucion exit=0; CSV de 17 columnas identicas
  en orden a p98 y 10 filas parseables; ASCII 0 no permitidos; logica md5 sin
  falsos OK; sin red; no toca toxic_test/labels; no modifica historicos.
- Observaciones menores aplicadas despues: presencia obligatoria de
  LICENSE/README (AUSENTE (ERROR)); raise explicito en vez de assert.
- Observacion aceptada sin cambio: columna sha256 del registro contiene md5
  oficiales prefijados ("md5 ...") para fuentes no descargadas.

## 2. ml-qa-engineer - afirmaciones S1

- Alcance: reports/p100_corpus_registry.csv, p100_corpus_research.md,
  p100_source_decisions.md contra evidencia local y sondas read-only.
- Veredicto: CLAIMS_VERIFICADOS (confianza alta).
- Reprodujo de forma independiente: licencias (CC BY 4.0 Miami; CC BY-NC 4.0
  ChatSubs); error /get0/ (393 bytes; md5 b6bce183...); pagina /get/ (56717
  bytes; md5 feb2aa20...); 555 carpetas 2020-04-24..2021-10-31; muestras de
  IDs; md5 de LICENSE/README; estados del registro (10 fuentes; cero
  ALLOWED pleno); coherencia con las reglas del propietario.
- Observaciones menores aplicadas: redaccion del estado del tar.gz; lista de
  9 archivos por dia completa (incluye es_es).

## 3. ml-architect - hipotesis y pre-registro S2

- Alcance: reports/p100_preregistration.md (draft) contra encargo y fase
  previa.
- Veredicto inicial: GO_CON_CAMBIOS (confianza alta; cifras replicadas con
  p98_11/p98_13, no por lectura).
- Cambios requeridos y aplicados:
  1) social real 4.227.450 (post-dedup) y total nominal 29.727.450 (paridad
     C98-1); anclaje a conteos reales y sha 9f562718...; pool twitter
     1ef872c1 declarado como pool.
  2) medidor fijo de 3.320 documentos evaluado en ambos brazos; sin dominio
     chatsubs en validacion; parada temprana y semilla de barajado
     precisadas.
  3) criterio de atribucion (delta vs fusion30 > 0; si <= 0, H100-01
     falsificada aunque el gate vs V03 pase); "consistente, no marginal"
     definido (delta47 > 0 y >=3/5 pliegues); P5 anadida.
  4) obligaciones ChatSubs y md5 verificado del tar.gz; exposicion ~117-118M.
- Verificacion final: GO FINAL (confianza alta) tras aplicar los cambios y la
  reformulacion residual de la seccion 6.

## 4. Verificacion del artefacto ChatSubs

- ChatSubs.tar.gz: 1.377.313.069 bytes; md5 local
  afd2d3967cf8cdf78883a7baa8d061e4 = md5 oficial del registro Zenodo
  8220853 (verificado por el revisor, el QA y el arquitecto).

## 5. Estado

- S1 cerrada (registro y decisiones congelados); S2 congelada (pre-registro
  con GO final). Siguiente: S3 (construccion del corpus B1 y manifiesto).
