# P34 - decision

```text
ESTADO = INVALIDATED
```

## Por que INVALIDATED y no solo CLOSED

La diferencia entre los dos no es de grado, es de naturaleza. **CLOSED** significa que se busco algo y no se encontro. **INVALIDATED** significa que se encontro algo que **parecia** el resultado buscado y se demostro que no lo era.

En P34: `real - barajado = +0,0002070` sugeria que la correspondencia del residuo aportaba algo. Pero `real - aleatorio = -0,0000797`: el ruido de la misma media y desviacion rinde **mas**. El resultado apparent se atribuye a que se permitio sumar un termino al score, no a la senal linguistica. Eso es exactamente lo que la propia regla del brief marca como INVALIDATED.

| Control | Delta | Lectura |
|---|---|---|
| real | -0.0002039 | la senal |
| barajado | -0.0004109 | sin correspondencia fila-residuo |
| aleatorio | -0.0001242 | sin senal, misma forma |

## Lo que queda cerrado con este veredicto

- **La correccion local de score sobre bandas adyacentes**: cerrada como familia, no como configuracion. No se reintenta con otro `beta`, otra pareja ni otra transformacion.
- **La lectura del 90 % adyacente**: corregida a 59,94 % en este informe. Los informes de V31 no se tocan.

## Lo que este veredicto NO cierra

No cierra la **existencia** de senal linguistica: P33 midio AUC 0,5411 en el residuo global. Cierra su **capacidad de mejorar la decision** a traves de sumar su residuo al score, ya sea sobre las seis clases o sobre un par.

## Continuidad

La regla de continuidad se cumple: no se ha parado en el primer negativo. Se evaluaron las cinco parejas (A1 a A5), con sus controles, su bootstrap y su desglose de alta confianza. El paso siguiente es un **gap analysis actualizado** que busque un mecanismo materialmente distinto, no otra forma de sumar el mismo residuo al score.

**Cifras**: V30 0.6134471 | P34 real 0.6132432 (-0.0002039) | bootstrap -0.0002207, IC95 [-0.0006645, +0.0002135]
