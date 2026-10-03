# Texto tipado (P7) - auditoria de novedad y especificacion

**Fecha**: 2026-09-30 | **Metodo**: auditoria POR OPERACION sobre el codigo ejecutable (270 ficheros entre scripts/, src/ y notebooks/); el nombre de la tecnica no cuenta como evidencia. Se excluye al propio auditor y al productor de la anotacion.

## Clasificacion

| ID | Familia | Estado |
|---|---|---|
| P7-TEXT-TIPADO | texto tipado (texto original + lemma_POS) | **GENUINELY_NEW** |
| P7-DUP-LEMA-AGG | variantes ya cubiertas de lema | **CLOSED** |
| P7-DUP-SUFIJOS | morfologia de superficie | **CLOSED** |

## Evidencia del barrido

- Consumidores de `lemma_pos_seq` como representacion entrenada (excluido el productor): **0**
- Constructores de tokens tipados fuera del productor: **0**
- Literales de token tipado (tipo `palabra_VERB`) en codigo: **2** [{'archivo': 'v29_01_inventario.py', 'ejemplos': ['oof_ADV']}, {'archivo': 'age_linguistic_p0p1_gap_y_legalidad.py', 'ejemplos': ['casa_NOUN', 'coches_NOUN', 'compra_VERB', 'querer_VERB']}]
- Joins de lema/POS en concatenaciones: **0**

La columna `lemma_pos_seq` existe SOLO como anotacion almacenada (la crea `scripts/age_linguistic_00_anotacion.py` al construir los parquets). Ninguna fase la ha consumido como features de un modelo. La clasificacion es **GENUINELY_NEW**.

## Cobertura del bloque tipado (artefactos congelados)

| Anotador | Docs no vacios | Tokens | Vocab unigramas | Vocab bigramas |
|---|---|---|---|---|
| md | 169,232 (99.94%) | 2,590,877 | 284,904 | 978,705 |
| lg | 169,232 (99.94%) | 2,590,877 | 283,659 | 979,878 |

Ejemplo de secuencia tipada (md): `ud_PROPN y_CCONJ su_DET flia_NOUN ser_AUX el_DET mas_ADV beneficiado_ADJ con_ADP obredecht_PROPN ese_PRON no_ADV ser_AUX`

## Legalidad (D-076, sin herramientas nuevas)

- No se usa ningun anotador nuevo: se reutilizan las anotaciones CONGELADAS de spaCy (md/lg, no-transformer) con SHA256 registrados en `experiments/age_linguistic_01_anotacion.json`.
- Solo etiquetas y lemas (forma); vectores y embeddings siguen prohibidos y no se tocan.
- Sin datos externos, sin labels de edad externas, sin metadatos de usuario, sin test, sin holdout.

## Especificacion de la representacion (congelada antes de ejecutar)

```text
B = V30 + plug-in + texto tipado
texto tipado = [bloque A: texto original] + [bloque B: tokens lemma_POS]
  bloque A: TF-IDF word (1,2), min_df=3, token_pattern \b\w+\b   (contenido lexical)
  bloque B: TF-IDF word (1,2), min_df=5, token_pattern \S+      (lemma_ETIQUETA)
  vectorizadores SEPARADOS (sin colision de vocabulario; cada uno se ajusta SOLO
  en el train de cada fold); cada bloque queda L2 por fila (normalizacion por bloque)
  escala por bloque: 1.0 fija en ambos (neutro; precedente V31-03), definida ANTES
```

Unigramas y bigramas de lemma_POS (sin trigramas). No se anaden NER, dependencias, POS puro, morfologia, longitud, URL, hashtags ni features densas.

## Comparacion explicita

| Fase | Representacion | Que conserva | Que anade |
|---|---|---|---|
| P2 (cerrado) | secuencia de POS | nada lexical | funcion abstracta de cada token |
| P6 (cerrado) | tipos de entidad | nada lexical | estructura de entidades |
| **P7** | texto original + lemma_POS | **el contenido lexical del texto** | **funcion linguistica por token, ligada a su lexema** |

La hipotesis de P7 no es 'mas NLP': es que el componente lexical tipado puede aportar informacion complementaria al campeon que ni el TF-IDF convencional ni POS aislado capturan. P2 y P6 no se extrapolan a P7.
