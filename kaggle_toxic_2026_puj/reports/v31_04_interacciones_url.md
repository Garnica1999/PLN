# V31-04: Control B de STRUCTURAL_TEXT_FEATURES, interacciones del bloque URL

Generado por `experiments/v31_04_interacciones_url.py` (V31-04 (Control B de STRUCTURAL_TEXT_FEATURES: interacciones del bloque URL)).

**Estado `OOF_EVALUATED`. Alcance: interacciones del bloque URL, que es el unico de
los tres que mostro senal positiva en el Control A.**

## 1. Las interacciones

18 features en total: 13 del bloque y
5 de interaccion. Se codifican como cruces discretos,
no como productos de variables continuas de alta cardinalidad. La longitud se agrupa en
tres tramos (corto <= 12 tokens, medio <= 22, largo > 22).

- `_int_url_corta_x_corto`
- `_int_url_query_x_largo`
- `_int_url_x_corto`
- `_int_url_x_largo`
- `_int_url_x_medio`

## 2. Resultado

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 (control) | 0.7772114525 | 0 |
| Control A: solo features | 0.7774199655 | +0.0002085130 |
| Control B: + interacciones | 0.7813779888 | +0.0041665363 |

**El Control B aporta +0.0039580233 sobre el Control A.**

| etiqueta | F1 control | F1 Control B | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.7634 | +0.0017 |
| TR | 0.8157 | 0.8184 | +0.0027 |
| AG | 0.7542 | 0.7623 | +0.0081 |

train F1 0.9711 | gap +0.1897

## 3. Rescates y danos del Control B

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 29 | 26 | +3 | 55 |
| TR | 5 | 3 | +2 | 8 |
| AG | 36 | 24 | +12 | 60 |

**rescates 70 | danios 53 | neto +17**

## 4. Lo que este script no hace

- No prueba interacciones de mencion ni de enfasis: sus Controles A fueron negativos.
- No combina bloques.
- No genera submission. No lee el test.

Coste: 8.2 s, memoria 229 MB, 3 modelos, 0 submissions.

