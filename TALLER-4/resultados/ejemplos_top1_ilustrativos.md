# Ejemplos top-1 ilustrativos — Fase 6 (modelo IR TF-IDF base)

Seleccion curada de 5 pares query (entrevista) → top-1 (segmento de libro) de
`resultados/evaluacion_muestra_etiquetada.csv`, para el informe y el notebook.
Etiquetas del etiquetado humano-proxy (2026-09-04; contexto ampliado del
corpus). Mezcla: 2 relacionados buenos, 2 parciales y 1 falso positivo
(etiquetado `parcial` por prudencia, caso mas cercano a un falso positivo duro
de la muestra).

## 1. Relacionado bueno — el segmento cita el mismo tipo de testimonio

- **id_doc**: `5eb4eb12da6da3b502db359f.pdf`
- **id_segmento**: `HALLAZGOS-p129-las-masacres` | **tomo**: HALLAZGOS | **score**: 0.1319
- **Por que**: la entrevista narra una amenaza de masacre a una comunidad
  campesina (actores armados, plazo para salir, "objetivo militar", asesinato
  de un companero); el segmento "Las masacres" abre precisamente con ese mismo
  tipo de relato. Match tematico genuino de hecho y poblacion.

## 2. Relacionado bueno — caso etnico especifico bien recuperado

- **id_doc**: `5fee5bef9e8b013f26931324.pdf`
- **id_segmento**: `ETNICO-p448-violencias-danos-y-resistencias-de-los-pueblos-etnicos-en-el-conflicto-armado` | **tomo**: ETNICO | **score**: 0.1235
- **Por que**: entrevista a un lider raizal sobre el impacto del conflicto
  (narcotrafico) y la discriminacion racial en San Andres; el segmento cita el
  caso raizal (informe DAS "plan secreto"). Recupera la poblacion etnica
  correcta y su afectacion especifica, no un segmento generico.

## 3. Parcial — solape de tema amplio, desviacion del hecho concreto

- **id_doc**: `5eb4eb13da6da3b502db35e7.pdf`
- **id_segmento**: `NARRATIVA-p547-86-negociar-como-si-no-hubiera-guerra-y-pelear-como-si-no-hubiera-negociacion` | **tomo**: NARRATIVA | **score**: 0.1643 (el mas alto de los ejemplos)
- **Por que**: la entrevista es de un analista de relaciones internacionales
  sobre asistencia militar de Estados Unidos (Plan Colombia, Guerra Fria); el
  top-1 trata la negociacion de La Habana y la muerte de Alfonso Cano. Ambos
  son "guerra y salidas del conflicto", pero el score alto no refleja el tema
  central de la query: el match especifico (cooperacion EEUU) no aparece en el
  top-3.

## 4. Parcial — caso representativo del sesgo de especificidad

- **id_doc**: `5ef4e445f1cdbf68d06e9f17.pdf`
- **id_segmento**: `FUERA-p99-de-la-guerra-total-a-la-paz-incompleta-2006-2016` | **tomo**: FUERA | **score**: 0.1623
- **Por que**: joven que socializo los acuerdos de paz (2015-2016), sufrio
  amenazas y salio del pais; el top-1 es la historia macro 2006-2016 (AUC,
  FARC, secuestros). Coincide la epoca ("paz incompleta") pero no su vivencia
  personal del exilio: el match mas especifico ("impacto del exilio") quedo
  fuera del top-1. Ilustra el fenomeno de 7/30 queries con top-1 parcial y
  mejor match en r2-r3.

## 5. Falso positivo — top-1 TESTIMONIAL por solape de registro oral

- **id_doc**: `5f148660f1cdbf68d0e27e16.pdf`
- **id_segmento**: `TESTIMONIAL-p401-senderos-pedregosos` | **tomo**: TESTIMONIAL | **score**: 0.1539
- **Por que**: entrevista a una mujer indigena amazonica excombatiente de las
  FARC (reclutamiento en el Frente Amazonas, fusilamiento de companeros,
  desmovilizacion); el top-1 es un testimonio oral de un joven nasa del Cauca
  en la guerra. Ambos son relatos coloquiales de vida dura en el conflicto,
  pero no comparten poblacion, region ni hecho: el ranking los emparejo por el
  registro oral compartido, no por el contenido. Es el caso mas cercano a un
  falso positivo duro de la muestra (etiquetado `parcial` por prudencia: la
  evidencia revisada no permite afirmar solape tematico real).

---

**Lectura para el informe**: el modelo base recupera bien cuando la query tiene
terminos de dominio especificos (masacres, raizal, detenciones masivas), pero
tiende a emparejar por solape lexico general del conflicto y por el registro
oral cuando la entrevista es testimonial, produciendo top-1 plausibles pero de
baja especificidad (P@1 estricto 0.53; P@1 aceptable 1.0 sobre 30 pares).
