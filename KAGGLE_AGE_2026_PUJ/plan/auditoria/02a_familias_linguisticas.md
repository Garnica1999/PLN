# Auditoria V1-V37 · Fase 2a - Familias linguisticas (NLP)

Revision delegada a `nlp-specialist` sobre solo lectura, con verificacion propia del auditor
principal de cada hallazgo. Ningun archivo de experimento fue editado.

## Alcance revisado

17 IDs: `AGE-LING-P0-GAP`, `AGE-LING-P1-LEGALITY`, `AGE-LING-RESOLUTION`,
`age_linguistic_01_anotacion`, `AGE-LING-P2-PREREG`, `AGE-LING-P2-POS-SEQ`, `AGE-LING-P6-PREREG`,
`AGE-LING-P6-NER`, `AGE-LING-P7-PREREG`, `AGE-LING-P7-TYPED`, `AGE-LING-REMAINING-PREREG`,
`AGE-LING-P3-LEMMA-POS`, `AGE-LING-P4-MORFOLOGIA`, `AGE-LING-P5-DEPENDENCIAS`,
`age_linguistic_10_dep_edges`, `V37-PREREG`, `V37-DIAG-POS`.

## Veredicto por experimento

| ID | veredicto | severidad |
|---|---|---|
| `AGE-LING-P0-GAP` | correcto | baja |
| `AGE-LING-P1-LEGALITY` | correcto | baja |
| `AGE-LING-RESOLUTION` | correcto con reservas | baja |
| `age_linguistic_01_anotacion` | correcto con reservas | media |
| `AGE-LING-P2-PREREG` | correcto | baja |
| `AGE-LING-P2-POS-SEQ` | correcto | baja |
| `AGE-LING-P6-PREREG` | correcto | baja |
| `AGE-LING-P6-NER` | **defecto confirmado** | media |
| `AGE-LING-P7-PREREG` | correcto | baja |
| `AGE-LING-P7-TYPED` | correcto | baja |
| `AGE-LING-REMAINING-PREREG` | correcto | baja |
| `AGE-LING-P3` / `P4` / `P5` | correcto | baja |
| `age_linguistic_10_dep_edges` | correcto | baja |
| `V37-PREREG` | correcto con reservas | baja |
| `V37-DIAG-POS` | correcto con reservas | baja |

Ningun cierre por delta negativo se apoya en un punto sin intervalo: P2 cierra con IC95 que excluye
cero, P7 cierra como "no detectable" (su IC **incluye** cero), R3 cierra con IC que excluye cero.
Es el comportamiento correcto.

## Defectos confirmados (con medicion propia del auditor)

### N-01 (media) El brazo `N1b` filtra spans sobre indices desalineados

`scripts/age_linguistic_06_ner_estructura.py:82` define `TOKEN_RE`, un tokenizador regex distinto
del de spaCy, y `:168-171` lo usa para decidir si un span es mencion/RT/hashtag/URL, indexando el
array de superficie con los indices del span.

Medicion propia sobre el parquet congelado y el texto de `src.data.load_train()`:

```
COINCIDENCIA TOKEN_RE vs len(pos_seq.split()): 11221/20000 = 56,11 %
desplazamiento: min -14  max +48  mediana -1  |  documentos desalineados 8779 = 43,90 %
```

Ejemplo (fila 3): el texto empieza `#TFZ @M_MartinezRed ...`; `TOKEN_RE` produce
`['#TFZ', '@M_MartinezRed', ...]` mientras `pos_seq` empieza `['PROPN','PROPN','SPACE','NOUN',...]`,
porque spaCy parte `#TFZ` en `#` + `TFZ` + espacio.

Consecuencia: `N1b` y la formulacion de D-078 ("N1 sin spans sobre menciones/RT/hashtags/URLs")
describen algo que el codigo no calcula. **N1 y N2 no estan afectados**: usan `ent_seq` y `pos_seq`
directamente.

**Impacto en la conclusion**: ninguno. Los seis brazos quedan bajo el gate de integracion, asi que un
brazo mal alineado no cambia el cierre de la familia NER. **Re-declarar N1b exigiria re-ejecucion**, y
el proyecto prohibe re-ejecutar sin autorizacion.

### N-02 (media) `es_core_news_lg` no esta fijado en el entorno

`requirements.txt:49` fija solo `es_core_news_md` (URL de wheel + sha256). No hay linea para `lg`.

La procedencia si queda registrada (`experiments/age_linguistic_resolucion.json:14-35`,
`reports/linguistic_annotation_registry.csv:4`, con sha256 de `meta.json`), pero **el binario del
anotador que aparece en todos los resultados md/lg no es instalable desde el entorno declarado**.

Consecuencia practica: los experimentos V37 y las familias md/lg no son reproducibles en una maquina
limpia a partir del repositorio. Es el hallazgo mas accionable de esta fase.

### N-03 (media) El 16,35 % de acuerdo md-lg se presenta sin controlar por longitud

`reports/age_linguistic_anotacion_comparativa.md:25-34` titula "Acuerdo md vs lg" con el 16,35 %.

Medicion propia sobre los dos parquets congelados (169.336 filas, `fila` verificada igual en ambos):

```
acuerdo por token            = 86,269 %   (2.235.135 de 2.590.877 comparados)
secuencias completas iguales = 16,345 %   (27.678 de 169.336)
  longitud  1- 3 tokens: 36,76 %
  longitud  4- 7 tokens: 28,01 %
  longitud  8-11 tokens: 20,23 %
  longitud 12-15 tokens: 14,95 %
  longitud 16-23 tokens:  9,89 %
```

La coincidencia de secuencia completa cae monotonamente con la longitud y queda **explicada por
longitud x acuerdo por token**: el 16,35 % no mide calidad de anotacion sino longitud de documento.

V37 lo interpreta correctamente (`reports/v37_diagnostico_pos.md:98`, "limitacion del conjunto de
datos"), pero la correccion **no se aplico al informe original**, que sigue siendo la referencia de
esa familia.

### N-04 (media) Claim sin sustento en el informe de anotacion

`reports/age_linguistic_anotacion_comparativa.md:92-93`: "El acuerdo por token ... es la cota superior
de lo que puede mover el canal en P2". No hay derivacion ni medicion que ligue el acuerdo entre
anotadores a un delta de macro-F1. Y el unico efecto medido del canal (P2, -0,0086357) es de magnitud
muy superior a lo que el 13,7 % de desacuerdo token sugiere acotar.

### N-05 (baja) El SHA256 del pre-registro V37 declarado no corresponde al fichero

`DECISIONS.md:3405` y `PROJECT_STATE.md:37` declaran `30d5d3cf...` para
`experiments/age_linguistic_13_preregistro_v37.json`. Hash real en disco:

```
1d6d6573d825dac9eb841e69979320ecf18dcd75dbee034aab28388ee9fb7115
```

El string `30d5d3cf` no aparece en ningun otro artefacto. La cadena de procedencia que D-089 llama
"inmutable" no cierra. (Consecuencia limitada: los anclajes SHA256 **internos** del pre-registro si
coinciden con la spec y con las enmiendas.)

### N-06 (baja) `n_process` falseable en la ficha de procedencia

`scripts/age_linguistic_00_anotacion.py:116` cae a `n_process=1` si el pipe falla, pero `:127`
escribe `"n_process": 4` literal. La salida es determinista (orden por `i`), asi que el artefacto no se
corrompe; el metadato de procedencia si puede mentir. No se puede determinar si el fallback se
disparo: no hay log de esa corrida.

### N-07 (baja) Claim bibliografico retirado que sigue en el spec congelado

`plan/V37_DIAGNOSTICO_POS_PLAN.md:563` y `:1087` conservan "tres treelets atomicos (etiquetas POS)
significativos" para genero en Johannsen et al. (2015), que la enmienda E14 retira. `:1084` lista el
0,62 de Cruz/Haro/Ortega sin la aclaracion de que es el accuracy del ensemble apilado completo.

Superado por `plan/V37_ENMIENDAS_01.md:162-169` y `reports/v37_diagnostico_pos.md:124`, que si llevan
la atribucion correcta. Riesgo: cita aislada del spec.

**Nota**: ninguna de las fases P2/P6/P7/P3/P4/P5 contiene citas de Flekova, Cruz, Posadas-Duran,
Modaresi ni Johannsen. Las tres atribuciones mal asignadas estan **corregidas** en los informes; el
problema residual es solo el spec congelado.

### N-08 (baja) Cifra obsoleta dentro de las enmiendas

`plan/V37_ENMIENDAS_01.md:153` dice "desajuste de presencia (~6,0 %)". El valor medido y publicado es
**0,00 %** (`experiments/age_linguistic_14_diagnostico_pos.json` ->
`fase_b.sanity_check_proyeccion_universal_E13.desajuste_de_presencia_relativo: 0.0`).

## Reserva de fondo: la premisa del anotador era secundaria

Los `meta.json` declaran `tag_acc` 0,96118 (md) vs 0,96147 (lg), `pos_acc` 0,98495 vs 0,98518,
`ents_f` 0,8929 vs 0,8970, `dep_las` 0,8804 vs 0,8818. **"lg es mejor" no se sostiene para etiquetas**:
la unica diferencia real es de capacidad y tok2vec (631 MB vs 54 MB).

El mandato de D-076 (comparar md vs lg) se cumplio literalmente y los resultados lo confirman (md ~ lg
en los siete brazos). Pero **ningun informe extrae la leccion**: la capacidad del anotador no era la
restriccion activa. Es una leccion no extraida, no un error de ejecucion.

## Lo que el especialista declara SIN defecto

- **Ningun control negativo degenerado.** El patron de V36-E1 (permutacion biyectiva o barajado intra
  documento que deja la matriz identica) **no se repite**. P2 `barajar_intra` SI discrimina
  (-0,008031 intra vs -0,0039528 filas) porque el vectorizador usa `ngram_range=(1,3)` sobre las
  secuencias POS: con bolsa de unigramas habria sido renombracion, pero no lo es. P6/P7/P3/P4/P5
  permutan de forma discrriminante. V37 anade control de FPR conjunta con 0 celdas bajo permutacion
  frente a 20 reales. Todos los instrumentos **pueden** rechazar, y varios rechazaron.
- **Hashes reproducibles.** `tobytes()` aparece una sola vez en el scope lingüistico, sobre arrays
  numericos float contiguos. El defecto `data_hash` de V36 no se propago.
- **Validacion en cada corrida**: `age_linguistic_12_remaining_channels.py:519-537` re-valida el
  artefacto de dependencias en cada corrida (n_tokens identico + recomposicion `dep->POS_cabeza` con 0
  discrepancias exigidas, STOP si falla). El hash no es decorativo.

## V37: la causa declarada y la causa operativa no coinciden (reserva, no defecto)

`scripts/age_linguistic_14_diagnostico_pos.py:2699-2701` devuelve `DIAGNOSTICO_CERRADO` en **G2**
(inestabilidad md-lg) sin evaluar V1..V6. Pero el commit y el acta D-089 dicen "CERRADA POR POTENCIA".

Verificado: V5 falla en **ambos** anotadores (`A1_md` 0,17205317 y `A1_lg` 0,17390555, ambos bajo el
sanity floor 0,176667) y V4 falla con `delta_estandarizado_min` 0,000371 frente a umbral 0,008. O sea
que la conclusion "cerrada por potencia" es **correcta**, pero la rama que la produce es G2.

No es un defecto de conclusion sino una discrepancia entre la causa declarada y el punto de corte del
codigo. Conviene corregir el texto de D-089, no el codigo (el codigo hizo lo que el pre-registro
ordenaba: cortar en el primer gate fallido).

## Estado

Fase 2a **cerrada**: 8 defectos confirmados, 4 de ellos media. Ninguno altera una conclusion de
adopcion. Los mas accionables: **N-02** (entorno no reproducible para `lg`) y **N-01** (N1b
desalineado, sin impacto por estar bajo el gate).
