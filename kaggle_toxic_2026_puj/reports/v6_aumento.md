# Track J: aumento de datos (V6-08)

Generado por `experiments/v6_08_aumento.py` (V6-08 (J0-J2)).

## 1. Que se prueba y por que no es una repeticion de PRE

El gap analysis (V6-00) hizo una distincion que importa: **J no es una repeticion de
PRE-00 a PRE-09**. Esas diez variantes eran *normalizacion*: quitan ruido de
superficie y **sustituyen** el texto. El aumento **crea instancias nuevas** a partir
de las existentes, y la variacion se aplica **solo al entrenamiento**, nunca a
validacion ni a test.

Aunque el mecanismo sea distinto, el precedente de superficie es
adverso: diez transformaciones deterministas, todas descartadas. Si el resultado
vuelve a ser negativo, el cierre abarca tanto el aumento como la superficie, y eso
es un cierre mas solido, no mas debil.

## 2. Transformaciones y por que son seguras

| transformacion | que hace | por que preserva la etiqueta |
|---|---|---|
| `espaciado` | separa las letras de una palabra | no cambia ni una palabra |
| `elongacion` | alarga la ultima letra de una palabra | solo anade un caracter a un item ya existente |
| `puntuacion` | duplica un signo | no toca el lexema |

Lo que **no** se hace, y es lo importante: no se sustituyen palabras por
sinonimos, no se reordena, no se cambia la negacion ni el objetivo. Cualquier
transformacion que toque el significado es una fuente de etiquetas falsas.

## 3. Resultados

| variante | filas de train | corrected_cv | delta | IC95 | HS | TR | AG |
|---|---|---|---|---|---|---|---|
| J0-control | 1186 | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 |
| J1-una-copia | 2372 | 0.771128 | -0.0061 | [-0.0141, +0.0018] | 0.7600 | 0.8054 | 0.7480 |
| J2-dos-copias | 3558 | 0.770898 | -0.0063 | [-0.0145, +0.0016] | 0.7607 | 0.8063 | 0.7457 |

Las filas de train crecen de
1186 a 3558,
un aumento real de datos y no una sustitucion de texto.

## 4. Gate de dos niveles

| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |
|---|---|---|---|---|
| J0-control | no (+0.0000) | no | 0/5 | no supera |
| J1-una-copia | no (-0.0061) | no | 1/5 | no supera |
| J2-dos-copias | no (-0.0063) | no | 1/5 | no supera |

Fecha: 2026-09-26T01:42:48+00:00. Duracion: 23.8 s.
