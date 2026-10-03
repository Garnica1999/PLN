# V24: cierre de la fase. Gate competitivo resuelto

Worktree `/home/carlos/Git/PLN-v24`, rama `v24`, base `ebc2083`. **0 submissions gastadas en
V24. 3 disponibles.**

## 1. Estado actual

**V03 = 0.78306 LB**, congelado, sha256
`7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`.

**Submissions: 22 en el registro, 3 disponibles, 0 enviadas en V24.**

## 2. La tabla del encargo (seccion 23)

| candidato | mecanismo | CV | gap train-OOF | complejidad | rescates | danos | LB | estado |
|---|---|---:|---:|---|---:|---:|---:|---|
| V03 (champion) | TF-IDF union + LR L2 C=2.0 | 0.777211 | +0.1842 | densa, norma 22.8 | -- | -- | **0.78306** | **champion** |
| svd:rbf_svc (V23) | SVC RBF sobre SVD(300) | 0.790409 | n/d | 70.9 % soporte | 136 | 104 | 0.77938 | REJECTED |
| LR L1 C=4.0 | penalizacion L1 | 0.783642 | +0.1455 | 1.3 % no nula | 208 | 205 | no enviada | no enviada |
| kNN / retrieval (V24-03) | voto de vecinos coseno | -- | -- | redundancia 93 % | 3 | 17 | no enviada | **cerrada** |
| factorizacion asimetrica AG (V6-B) | p(AG\|X) en dos ramas | -- | -- | rama de 22 positivos | 1 | 35 | no enviada | **cerrada** |

## 3. Decision final

# V03 REMAINS CHAMPION

Las tres submissions se conservan. V25 no se abre.

## 4. El argumento que decide, y por que no es un argumento de pauta

El gate de P1 estimaba que hace falta un `dOOF` de **+0.03756** para empatar el champion.
El especialista no lo trato como una ley, y hizo bien: lo tradujo a la escala del
instrumento.

**El umbral de gate compra +0.0092 de leaderboard, y el error estandar del leaderboard con
1.154 filas es 0.01219. O sea, el umbral esta a 0.75 sigma del propio ruido de la
medida.** Para resolver dos sigma haria falta un `dOOF` de aproximadamente +0.11, es
decir **8.3 veces** el mejor `dOOF` historico.

Y el techo *oracle* de la sonda de kNN, que es "si supieramos que modelo acierta en cada
celda", da **+0.0811**. El umbral de gate esta al **46 % de un techo inalcanzable**, y el
mejor `dOOF` real de veintitres fases, +0.0132, esta al **16 %**.

| candidato | dOOF | dLB esperado por el modelo de P1 |
|---|---:|---:|
| kNN, mejor mezcla (w = 0.30) | +0.004952 | **-0.007988** |
| mejor fusion de embeddings (V5.1) | +0.005748 | -0.007793 |
| LR L1 C=4.0 | +0.006431 | -0.007626 |
| V23 svd:rbf_svc | +0.013198 | **-0.005969** |

**Todos predicen perdida, y el mejor de ellos perdio de verdad.** El modelo de P1
acierto el orden y la magnitud en V23: predijo -0.00597 y el leaderboard dio -0.00368.

## 5. Por que no puede haber un salto cualitativo

Cuatro razones independientes, y ninguna de ellas es "no se ha intentado":

1. **V03 no esta limitado por capacidad.** Su F1 de entrenamiento es 0.9614, luego ya
   explica el 96 % del train. Lo que limita es la informacion de 1.483 etiquetas.
2. **Las dos unicas cosas que igualaron a V03 fueron recodificaciones del mismo TF-IDF.**
   Una con SVD y kernel RBF, otra con FastText. Ninguna genero informacion nueva:recodificaron
   la misma matriz de 27.431 features en otro espacio.
3. **El mejor techo OOF de todo el repositorio es +0.0064** (fusion de embeddings), y el
   unico resultado por encima de +0.01 es el kernel, que sobreajusta a 76 % de soporte.
4. **La vecindad no ordena el error**, y eso ya se midio en V4-F03 con un IC95 de
   [-0.013, +0.0491]. Volver a medirlo no lo cambia.

El unico mecanismo que rompe el bloqueo es informacion de etiqueta **ajena** a esas 1.483,
y el concurso la prohibe.

## 6. kNN / retrieval: los seis puntos, con cifras verificadas

El gap analysis de V6 dejo este mecanismo como el unico hueco abierto. El especialista no
opino: lo ejecuto. Los numeros se verificaron contra `reports/v24_03_sonda_knn.json` y los
757 errores de V03 se recalcularon de forma independiente.

| | espacio union | espacio char |
|---|---:|---:|
| **A.** correlacion con la probabilidad de V03 | +0.9291 | +0.8512 |
| **B.** errores de V03 con vecino confiable | **12 de 757 = 1.59 %** | 23 de 757 = 3.04 % |
| **C.** precision al contradecir a V03 | **0.1500** (3 de 20) | 0.1071 (6 de 56) |
| **D.** McNemar a favor de V03 | 621 contra 284, p < 1e-29 | 866 contra 320 |
| **D.** Jaccard de los conjuntos de error | 0.343 | 0.269 |
| **E.** mejor mezcla (w = 0.30) | 103 celdas, +29 netas, dOOF **+0.004952** | dOOF -0.001903 |
| **E.** regla de sustitucion directa | **-14 netas**, dOOF -0.005516 | dOOF -0.014514 |
| **F.** techo oracle | dOOF +0.081055 | dOOF +0.091253 |

- **A.** Solo 7 % de varianza marginal respecto a la probabilidad de V03.
- **B.** El 1.59 % es la cifra que decide: la similitud top-1 mediana es 0.2953 y **solo 6
  filas de 1.483 superan 0.7**. No existe un vecindario denso.
- **C.** Peor que el azar.
- **D.** No es complementario: lo reproduce y pierde. El 62 % de los errores de V03 son
  tambien errores de kNN.
- **E.** La mejor formulacion da +0.005 de dOOF, que es **6.7 % de un sigma** del
  leaderboard. Y el modelo de P1 le asigna un dLB de -0.00799.
- **F.** Existe una formulacion no equivalente (el voto como prediccion y no como feature),
  y se ejecuto. Generalizar un resultado nulo da un resultado nulo.

## 7. Auditoria del codigo que ejecuto el especialista

`experiments/v24_03_sonda_knn_predictor.py` y `experiments/v24_03b_perfil_de_celdas.py`:

- Usan `read_train()`. **No leen el test en ningun momento.**
- Usan `holdout_v2` y `pliegues_corregidos`, el protocolo canonico agrupado. Al estar los
  pliegues agrupados por near-duplicate, **un vecino nunca puede provener del grupo de la
  fila de validacion**, sin necesidad de una exclusion adicional.
- Los vectores se ajustan **solo** con el pliegue de entrenamiento.
- No hay etiquetas externas ni embeddings preentrenados.
- Los dos scripts pasan `ruff` y tienen 0 bytes no-ASCII.

## 8. Correccion a una premisa de P0

P0 concluyo que el `I_word_char_weight` era una laguna abierta. **No lo era**: es el
capitulo 3 de V16, ya medido, con -100 celdas netas y dOOF -0.0226. La laguna era de P0, no
del proyecto. Se corrige aqui.

## 9. Lo que deja V24

1. **Un modelo de transferencia utilizable.** `dLB = -0.00920 + 0.2448 * dOOF`, con
   p = 0.0342 sobre 19 observaciones. No es una ley, pero predijo el caso disponible.
2. **Que el intercepto negativo explica el historial entero**: con `dOOF` nulo, el `dLB`
   esperado es -0.0092. Las 19 de 21 caidas no eran ruido, eran el valor por defecto.
3. **Que ningun indicador de complejidad individual sirve como filtro.** Ridge no
   sobreajusta y tiene el peor saldo de celdas (-80); V23 si sobreajusta y tiene el mejor
   (+32); y el saldo de celdas no predice el leaderboard.
4. **Que el techo competitive esta documentado y es estructural**, no una falta de ideas.
5. **Que quedan 3 submissions** y que gastarlas contra un umbral situado al 46 % de un
   techo inalcanzable seria jugarlas por inercia, no por evidencia.
