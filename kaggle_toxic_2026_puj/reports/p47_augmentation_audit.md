# P47 - Precedente de augmentation train-only

Generado por `experiments/v47_00_auditoria_datos.py` el 2026-09-28T00:29:46+00:00.

> La directiva propone abrir `deterministic label-preserving augmentation` solo si la auditoria confirma que la familia **no** fue ejecutada. Este informe muestra que si lo fue.

## 1. El experimento que ya existe

- experimento: `V6-08`
- script: `experiments/v6_08_aumento.py` (existe: True)
- sha256 del script: `7cc796cbc18ac70ee4614b7f329f524ecc1b9283e80ef49f933bf7575af02acb`

## 2. Transformaciones implementadas

| transformacion | tipo |
|---|---|
| `espaciado` | variante de espacios: inserta un espacio entre caracteres |
| `elongacion` | alargamiento de una vocal al final de una palabra |
| `repeticion_puntuacion` | duplica un signo de puntuacion |

Las tres son **formalmente ortograficas y preservan la etiqueta por construccion**, que es exactamente el perfil que la directiva describe. La directiva lista como candidatos `elongation normalization/re-expansion`, `repeated punctuation` y `whitespace variants`: los tres ya estan cubiertos.

## 3. Colocacion en el pipeline

- dentro del training fold: construir_textos_aumentados recibe solo los indices de entrenamiento del pliegue
- grupos preservados: **True** (cada copia conserva el indice y por tanto el grupo del original)
- el control reproduce V03 exactamente: **True**

Es decir: la colocacion que la directiva exige en su seccion 14, `split -> train fold -> augmentation -> vectorizer -> classifier -> validation`, ya es la que usa V6-08.

## 4. Resultado

| variante | delta vs V03 |
|---|---:|
| J1, una copia por fila | **-0.0061** |
| J2, dos copias por fila | negativo |

**Track J cerrado por resultado negativo. Sin ampliar rejilla.**

## 5. Conclusion

**CASE D. La augmentation ya existe historicamente.** No se repite bajo otra denominacion.

Auditoria previa que ya lo habia registrado: V41-01: DATA_AUDIT_COMPLETE. Familias A (ortografica) y D (mixup) cerradas.

Lo unico que V41-01 dejo nominalmente sin equivalente son *word dropout conservador* y *masking estructural*. **No entran en el alcance de esta fase** por dos razones: la directiva acota los candidatos a transformaciones ortograficas de alta confianza, y ambas operaciones tocan el contenido de las palabras, que es justo lo que la seccion 12 prohibe por riesgo de alterar la etiqueta. Con el precedente adverso de V6-08 (-0.0061) y de las diez normalizaciones PRE-00..PRE-09 (todas descartadas), abrirlas seria gasto sin hipotesis alguna.

