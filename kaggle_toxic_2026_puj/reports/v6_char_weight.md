# Track I: ponderacion explicita entre word y char_wb (V6-04)

Generado por `experiments/v6_04_char_weight.py` (V6-04 (I0-I3)).

## 1. Que se prueba y por que no es una repeticion

El gap analysis (V6-00) confirmo por busqueda en el codigo que **no existe ningun
`char_weight` ni `word_weight` en el proyecto**. Lo que si se midio fue el valor de
cada bloque por separado y su union: V2-BASE-M1 (solo word), V2-BASE-M4 (solo
char) y V2-BASE-M2/M3 (la union). Se compararon **dos bloques y su union sin
escalar**, pero nunca con una escala explicita dentro de la union.

El peso heredado de `hstack` no es una decision nuestra: cada `TfidfVectorizer`
normaliza por su propia norma de L2, asi que la escala relativa la fija el numero de
caracteristicas activas por documento, no el criterio del proyecto. Este track
responde a la pregunta pendiente con cuatro puntos fijos.

## 2. Resultados

| variante | char_weight | corrected_cv | delta | IC95 | HS | TR | AG |
|---|---|---|---|---|---|---|---|
| I0-control | 1.0 | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 |
| I1-char0.5 | 0.5 | 0.765493 | -0.0117 | [-0.0199, -0.0037] | 0.7604 | 0.8076 | 0.7286 |
| I2-char1.5 | 1.5 | 0.778779 | +0.0016 | [-0.0048, +0.0078] | 0.7642 | 0.8117 | 0.7605 |
| I3-char2.0 | 2.0 | 0.772508 | -0.0047 | [-0.0142, +0.0044] | 0.7568 | 0.8022 | 0.7584 |

**El control `char_weight = 1.0` debe reproducir 0.777211 exactamente.** Con la
multiplicacion identidad, si el control no coincide, los otros tres puntos no son
comparables y la conclusion no valeria. Es el mismo control que B0 (D-045) y A0
(D-046), y cumple la misma funcion: demostrar que la vara mide lo que dice medir.

## 3. Gate de dos niveles

| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |
|---|---|---|---|---|
| I0-control | no (+0.0000) | no | 0/5 | no supera |
| I1-char0.5 | no (-0.0117) | no | 0/5 | no supera |
| I2-char1.5 | no (+0.0016) | no | 3/5 | no supera |
| I3-char2.0 | no (-0.0047) | no | 2/5 | no supera |

## 4. La forma del perfil dice donde esta el optimo, y por que no se afina

El perfil tiene **optimo interior**: sube de 0.5 a 1.5 y baja en 2.0, con el maximo en
el peso 1.5 (+0.0016). Es decir, el peso
heredado de `hstack` **no era optimo**, y la pregunta del track tiene respuesta: el
balance entre evidencia lexica y sublexica deberia inclinarse un poco hacia el
bloque de caracteres.

Aun asi, **no se afina**, por dos razones concretas y no por prudencia vaga:

1. El mejor delta es +0.0016, **menos de la mitad del gate
   (+0.003)**, y su IC95 [-0.0048,
   +0.0078] **incluye el 0**: no hay senal estadistica.
2. Solo 3/5 pliegues mejoran, por
   debajo del minimo de 4.

Afinar alrededor de 1.5 con dos o tres puntos mas costaria unos 25 segundos y, si el
maximo real esta cerca de +0.002, seguiria por debajo del gate: **la escala no era el
cuello de botella**, que era la pregunta del track. Se deja escrito el numero para
que la decision sea revisable, y no se gasta presupuesto en una linea cuyo techo
estimado es la mitad del umbral de adopcion.

Fecha: 2026-09-26T01:32:40+00:00. Duracion: 33.3 s.
