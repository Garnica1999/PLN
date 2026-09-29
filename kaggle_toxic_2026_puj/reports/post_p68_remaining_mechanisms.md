# Post-P68 - Mecanismos restantes

Generado por `experiments/p68_02_h2_auditoria.py`. Fecha 2026-09-29T17:26:55+00:00.

## Los ocho ejes que el encargo senala, y quien los recorrio

| eje | experimentos que lo cubren | resultado |
|---|---|---|
| estructura de los ejemplos | P68-H1 | INVALIDATED: la estructura predice dificultad (AUC 0,608), no ganador (0,549) |
| estructura local | P59-02, P59-10, V29-03, V29-04 | la geometria local predice el error con AUC 0,725 pero no es accionable; los vecinos puros no co |
| geometria de representacion | V25, V27, V26, V43 | FM, kernel de espectro, Tsetlin y PLS: los cuatro cerrados |
| distribucion del entrenamiento | V6-03, P60-06, P60-04 | transporte, peso de clase y tamano de grupo: los tres cerrados |
| descomposicion de decisiones | P68-H1, P50, P51 | enrutado por estructura INVALIDATED, y el gating por canal con mapa de rescates INVALIDATED con  |
| estructura especifica de los errores | V4-B, V4-C, P60-08, V28 | dureza, violacion, contradiccion y region: los cuatro cerrados |
| transferencia al entrenamiento | V30-00, V6-07, P59-10 | peso robusto, deteccion de par inconsistente y regularizacion por vecindad: los tres cerrados |
| ponderacion por correccion ajena | P68-H2 | DUPLICATE: subconjunto estricto del indicador de error ya medido |

**Ejes sin cita historica: `0`.**

## La condicion de cierre del encargo

> Solo cerrar la investigacion si el gap analysis posterior demuestra que no quedan
> mecanismos genuinamente nuevos, o que las unicas lineas restantes son variantes de
> mecanismos ya invalidados o cerrados.

```text
P68-H2 = DUPLICATE                     True
ejes sin recorrer = 0                  True
condicion de cierre cumplida            True
```

## El estado cientifico al cerrar

```text
V03 sigue siendo la referencia congelada, 0,7772114525165721.
P67 no produjo mejora adoptable. P68-H1 tampoco. P68-H2 no se ejecuto.
Existe senal residual en MI_arbol, medida y no explotada.
El techo oracle de los modelos existentes sobre las celdas delegables es +0,0391,
  y cuatro mecanismos distintos han fallado al intentar acercarse a el.
```

## Lo que queda abierto, dicho con precision

No queda **ningun mecanismo nuevo** dentro de los ejes conocidos. Lo que queda abierto
es una pregunta, no una tecnica:

> Existe informacion en los modelos alternativos que ninguno de los mecanismos
> probados ha sabido explotar. El techo medido lo confirma. Lo que falta no es una
> variante de lo probado, sino una razon por la que la informacion sea alcanzable
> desde el entrenamiento, y esa razon no existe hoy.

## Lo que haria falta para reabrir

```text
un mecanismo que NO sea: ponderacion, concatenacion, blend, residualizacion,
  correccion de umbral, enrutado por estructura, gating por canal, ensemble de pesos,
  perdida modificada, regularizacion por vecindad, ni transporte
una medida de por que la informacion seria alcanzable desde el entrenamiento
  y no solo desde la decision
```

Hasta entonces, cerrar es el resultado honesto y no una rendicion: cuatro mecanismos
probados y uno descartado por algebra es mas de lo que la mayoria de los proyectos
puede afirmar sobre su propio techo.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p68_02_h2_auditoria.py`. Sin push.*