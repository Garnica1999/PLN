# V11-07: correccion de odds por shift de prevalencia, sobre shift simulado (P3)

Generado por `experiments/v11_07_correccion_prior.py` (V11-07 (P3, correccion de prior bajo shift simulado)).

## 1. La pregunta central de la v11

Hasta ahora se ha establecido que los estimadores funcionan (V11-06) y que la correccion
de prior es el mecanismo adecuado en teoria. Falta lo unico que importa: **si aplicarla
mejora la metrica de la competencia**.

La correccion es un desplazamiento constante en el espacio de logit:

    logit_corregido = logit(p) + logit(pi_destino) - logit(pi_origen)

Con prevalencia destino menor el desplazamiento es negativo, luego la probabilidad baja
y el umbral efectivo sube, que es la direccion correcta.

## 2. Lo que V9-02 no separo, y aqui si

V9-02 ajusto un umbral corregido y lo probo, lo que mezcla **dos** cosas: la correccion
de las probabilidades y una eleccion de umbral. Aqui la correccion se mide **con los
umbrales congelados de V03**, que es la version que no toca la capa de decision, de modo
que la ganancia se puede atribuir a una causa y no a la otra.

## 3. Delta por etiqueta y por factor simulado

| etiqueta | factor | base | corregida | delta | celdas |
| --- | --- | --- | --- | --- | --- |
| HS | 0.85 | 0.720457 | 0.718460 | -0.001997 | 13 |
| HS | 0.90 | 0.727972 | 0.726344 | -0.001628 | 2 |
| HS | 0.94 | 0.733507 | 0.732474 | -0.001032 | 2 |
| HS | 0.95 | 0.734830 | 0.733791 | -0.001039 | 2 |
| HS | 1.00 | 0.741113 | 0.741113 | +0.000000 | 0 |
| HS | 1.05 | 0.746892 | 0.746953 | +0.000061 | 1 |
| TR | 0.85 | 0.785953 | 0.785854 | -0.000099 | 0 |
| TR | 0.90 | 0.791403 | 0.791403 | +0.000000 | 0 |
| TR | 0.95 | 0.796345 | 0.796345 | +0.000000 | 0 |
| TR | 0.96 | 0.797278 | 0.797278 | +0.000000 | 0 |
| TR | 1.00 | 0.800845 | 0.800845 | +0.000000 | 0 |
| TR | 1.05 | 0.804961 | 0.806056 | +0.001095 | 6 |
| AG | 0.85 | 0.721362 | 0.719968 | -0.001393 | 9 |
| AG | 0.90 | 0.728362 | 0.726748 | -0.001613 | 9 |
| AG | 0.94 | 0.733511 | 0.733665 | +0.000154 | 0 |
| AG | 0.95 | 0.734741 | 0.734893 | +0.000152 | 0 |
| AG | 1.00 | 0.740579 | 0.740579 | +0.000000 | 0 |
| AG | 1.05 | 0.745941 | 0.747135 | +0.001194 | 1 |

## 4. Las tres cifras del protocolo, en el factor central de cada etiqueta

| etiqueta | factor | delta en el factor verdadero | IC95 pareado | excluye el cero | positiva en la mayoria de factores |
| --- | --- | --- | --- | --- | --- |
| HS | 0.94 | -0.001032 | [+0.000000, +0.000000] | no | no |
| TR | 0.96 | +0.000000 | [+0.000000, +0.000000] | no | no |
| AG | 0.94 | +0.000154 | [+0.000000, +0.002385] | no | si |

## 5. Gates pre-registrados

- **Gate A, transferencia**: al menos 2 de 3 etiquetas con delta positivo y 1 intervalo
  que excluya el cero. Etiquetas positivas: **1/3**;
  intervalos que excluyen el cero: **0/3**.
  **Superado: False.**
- **Gate B, estabilidad**: al menos 2 de 3 etiquetas positivas en la mayoria de los
  factores baridos, es decir que hay una region y no un punto. **1/3**. Superado: False.**

**La via puede continuar: False.**

## 6. Veredicto

**VIA CERRADA.** La correccion de odds no mejora la metrica de forma que se pueda
distinguir del ruido, o no hay region estable de mejora. El mecanismo es el unico
posible, asi que no se reintenta con mas variantes de la correccion.

Fecha: 2026-09-26T22:01:12+00:00. Duracion: 20.8 s.
