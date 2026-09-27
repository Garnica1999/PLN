# V31-01: Fase 1 de STRUCTURAL_TEXT_FEATURES, bloque URL estructural

Generado por `experiments/v31_01_url_estructural.py` (V31-01 (Fase 1 de STRUCTURAL_TEXT_FEATURES: bloque URL estructural)).

**Estado `OOF_EVALUATED`. Alcance: solo el bloque URL. No se prueban mencion, hashtag ni enfasis, y no se combinan bloques**

## 1. El control

corrected_cv del control: **0.7772114525**
Reproduce V03: **True**

| etiqueta | F1 control |
|---|---:|
| HS | 0.7617 |
| TR | 0.8157 |
| AG | 0.7542 |

## 2. El challenger: V03 + bloque URL

13 features, todas de estructura textual:

- `url_n`
- `url_longitud_total`
- `url_longitud_media`
- `url_longitud_max`
- `url_con_query`
- `url_con_fragment`
- `url_profundidad_path`
- `url_con_puerto`
- `url_longitud_dominio`
- `url_subdominios`
- `url_proporcion_dominio`
- `url_es_corta`
- `url_n_parametros`

El dominio completo **no** se usa como categoria: un dominio con pocas observaciones es
una tabla de lookup. Se usan longitud del dominio, numero de subdominios y TLD.

| etiqueta | F1 control | F1 challenger | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.7639 | +0.0022 |
| TR | 0.8157 | 0.8135 | -0.0022 |
| AG | 0.7542 | 0.7549 | +0.0007 |

**corrected_cv 0.7772114525 -> 0.7774199655 | delta +0.0002085130**

train F1 del challenger 0.9710 | gap train->OOF +0.1936

## 3. Rescates y danos

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 15 | 10 | +5 | 25 |
| TR | 1 | 3 | -2 | 4 |
| AG | 19 | 16 | +3 | 35 |

**rescates 35 | danios 29 | neto +6 | celdas 64**

## 4. Metricas por pliegue

| pliegue | F1 media |
|---:|---:|
| 1 | 0.7982 |
| 2 | 0.7925 |
| 3 | 0.7830 |
| 4 | 0.7418 |
| 5 | 0.7715 |

## 5. Lo que este script no hace

- No prueba mencion, hashtag ni enfasis.
- No combina bloques.
- No usa interacciones: eso es el Control B, que solo se ejecuta si este muestra senal.
- No genera submission.
- No lee el test.

Coste: 22.3 s, memoria 229 MB, 2 modelos, 0 submissions.

