# V13 · P15 — Que miembro acierta cuando V12 falla

Diagnostico sobre probabilidades fuera de muestra ya calculadas: no entrena, no toca el test y no genera submission. Complementa a `phase13`, que midio el rescate por familia;  aqui baja al grano de miembro, que es el que decide si un especialista de frontera (P3) tiene una base real.

## 1. Referencia

- Filas: 169,336
- Exactitud de V12: **0.6007** · macro-F1 de referencia 0,6004
- Errores de V12: **67,610** (39.93 % de las filas)
- De esos errores, acierta **al menos un miembro: 45,213** (66.87 %); el resto (22,397) es techo de informacion para todo el ensemble.

## 2. Matriz por miembro

| Miembro | Familia | Exactitud | Rescates | Exclusivos | Danos | Neto | Margen rescate | Margen fallo | Separacion |
|---|---|---|---|---|---|---|---|---|---|
| `oof_E019b_densas_svd` | denso | 0.4107 | 13,031 | **3,828** | 45,210 | -32,179 | 0.645 | 0.761 | -0.116 |
| `oof_E015_softmax` | ordinal | 0.3919 | 15,569 | **2,949** | 50,939 | -35,370 | 2.941 | 3.029 | -0.088 |
| `oof_E028a_pares` | ordinal | 0.3149 | 14,913 | **2,591** | 63,319 | -48,406 | 0.255 | 0.267 | -0.013 |
| `oof_E017_probabilidades` | prob_lineal | 0.5012 | 11,909 | **2,451** | 28,760 | -16,851 | 0.689 | 0.869 | -0.180 |
| `oof_E021b_rocchio` | prototipos | 0.4040 | 13,435 | **2,023** | 46,753 | -33,318 | 0.249 | 0.260 | -0.010 |
| `oof_E-010` | bayes | 0.5408 | 10,866 | **1,157** | 21,012 | -10,146 | 0.265 | 0.468 | -0.203 |
| `oof_E-011` | prob_lineal | 0.5567 | 10,495 | **377** | 17,949 | -7,454 | 0.677 | 1.025 | -0.348 |
| `oof_ADV05_nbsvm` | nbsvm | 0.5635 | 9,499 | **347** | 15,804 | -6,305 | 0.173 | 0.369 | -0.196 |
| `oof_E021a_ordinal` | ordinal | 0.5480 | 10,484 | **170** | 19,419 | -8,935 | 0.410 | 0.663 | -0.252 |
| `oof_E014_margenes` | svm_lineal | 0.5645 | 9,752 | **95** | 15,887 | -6,135 | 0.303 | 0.547 | -0.244 |
| `oof_E016_margenes` | svm_lineal | 0.5661 | 9,760 | **28** | 15,630 | -5,870 | 0.302 | 0.548 | -0.247 |
| `oof_E-009` | prob_lineal | 0.5591 | 10,193 | **20** | 17,236 | -7,043 | 0.535 | 0.870 | -0.335 |
| `oof_E013_margenes` | svm_lineal | 0.5629 | 9,888 | **8** | 16,291 | -6,403 | 0.308 | 0.554 | -0.246 |

## 3. Lectura

- **El rescate se reparte o se concentra.** El miembro con mas rescates exclusivos cubre 5.66 % del error de V12; los tres primeros cubren 13.86 %. Si esa cifra es pequeña, no hay 'conocimiento complementario' repartido entre modelos sino uno o dos modelos disgrazados de varios.
- **El margen discrimina**: la separacion va de -0.348 a -0.010. Todas las separaciones son pequenas: la confianza del miembro no identifica las filas que rescata, y por tanto una puerta por confianza no seria mas que ruido con un nombre elegante.
- **Techo con puerta por confianza**: "V12 si acierta, si no el miembro mas confiado" alcanza 0.6821 de exactitud frente a 0.6007 de V12. Ese es el maximo absoluto de lo que cualquier especializacion por miembro puede conseguir sin information nueva.
- **Cubrimiento greedy**: 13 miembros bastan para cubrir el 66.9 % del error de V12.

## 4. Familias

| Familia | Miembros | Rescates (cualquiera) | Exclusiva de la familia | Exactitud media |
|---|---|---|---|---|
| ordinal | 3 | 26,872 (15.87 %) | **22.53 %** | 0.4182 |
| prototipos | 1 | 13,435 (7.93 %) | **19.87 %** | 0.4040 |
| denso | 1 | 13,031 (7.70 %) | **19.27 %** | 0.4107 |
| bayes | 1 | 10,866 (6.42 %) | **16.07 %** | 0.5408 |
| prob_lineal | 3 | 19,651 (11.61 %) | **14.65 %** | 0.5390 |
| nbsvm | 1 | 9,499 (5.61 %) | **14.05 %** | 0.5635 |
| svm_lineal | 3 | 11,140 (6.58 %) | **2.00 %** | 0.5645 |

## 5. Fronteras objetivo

El cruce de una frontera son las filas cuya clase verdadera es una de las dos y que V12 etiqueta con la otra. Es el terreno de P3.

| Frontera | Mejor miembro | Aciertos en el cruce | % del cruce | Exactitud en el par |
|---|---|---|---|---|
| 18-24|25-34 (10,122 cruces) | `oof_E015_softmax` | 3,845 | 37.99 % | 0.3659 |
| 18-24|25-34 (referencia) | V12 | 0 | 0.00 % | 0.4963 |
| 25-34|35-49 (7,252 cruces) | `oof_E015_softmax` | 2,632 | 36.29 % | 0.5143 |
| 25-34|35-49 (referencia) | V12 | 0 | 0.00 % | 0.4974 |

## 6. Consecuencia para P3

- Los tres miembros con mas rescates exclusivos son `oof_E019b_densas_svd` (denso), `oof_E015_softmax` (ordinal), `oof_E028a_pares` (ordinal); en conjunto cubren 13.86 % del error de V12.
- Si esa cobertura es pequeña, un especialista binario de frontera **no puede apoyarse en los miembros actuales** y deberia construirse sobre variables nuevas (P1, P17, P19, P20) en lugar de sobre una recombinacion de las existentes.
- Si la separacion de margen es alta, la puerta de §37 (`gate(especialista, V12)`) tiene una senal de activacion utilizable; si es baja, la puerta tendra que decidirse por la clase mas probable de V12 y no por la confianza del miembro.

Referencia de `phase13` para contraste: 0.6005 de macro-F1 y 7 familias de miembros.

