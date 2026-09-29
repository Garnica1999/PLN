# P67 - Los cuatro controles que pueden cerrar la pregunta por si solos

Generado por `experiments/p67_01_controles_decisivos.py`. Fecha 2026-09-29T16:41:34+00:00.

## Por que estos controles y no mas variantes de la correccion

Una revision independiente de P66 pregunto que control falta antes de probar mas formulas de correccion, y la respuesta fue que mas formulas no ayudarian. Tres de ellos se calculan aqui porque cada uno puede CERRAR la pregunta por si solo.

## C-A, el reentreno anidado del umbral, que es el mas barato y el mas decisivo

```text
umbrales CONGELADOS    [0.44, 0.51, 0.46]
umbrales REENTRENADOS  [0.385, 0.445, 0.52]
V03 con umbral congelado     0.7772114525
V03 con umbral reentrenado  0.7742383012  delta -0.0029731513
referencia: correccion residual de MI_arbol  +0,0026743167
```

Un escalar por etiqueta es el modelo de calibracion **mas barato posible**. Si lo mas barato iguala o supera al mecanismo caro, no hay caso para el mecanismo caro, y eso cierra la pregunta sin necesidad de medir nada mas.

## C-B, sustitucion cruzada de candidato

| candidato | delta real | delta barajado | positivo |
|---|---:|---:|---|
| MI_arbol | +0.0026743167 | -0.0014390421 | True |
| negacion | +0.0011549795 | -0.0054335321 | True |
| word_1_3 | -0.0001400584 | +0.0038763165 | False |
| cohesion | +0.0000325761 | -0.0070646241 | True |
| huecos_1_2 | +0.0003505310 | -0.0064731429 | True |
| contraste | +0.0000000000 | -0.0017878950 | False |
| huecos_6_10 | -0.0028447979 | +0.0004009382 | False |

**4 de 7** dan delta positivo. Si la mayoria lo dan, el efecto es de la OPERACION y no de MI_arbol.

## C-D, la tasa base que el '47 de 53' de P66 no tenia

Los errores de un clasificador con umbral se concentran **estructuralmente** cerca de la frontera, luego la fraccion de rescates que cae en la banda no significa nada sin su denominador. La version informativa es el enriquecimiento.

| etiqueta | tasa base de errores en banda | rescates en banda | rescates fuera de banda | ENRIQUECIMIENTO |
|---|---:|---:|---:|---:|
| HS | 0.4019 | 1.0000 | 0 | 0.4850 |
| TR | 0.3232 | 0.7500 | 4 | 2.5058 |
| AG | 0.4007 | 0.9167 | 2 | 1.0241 |

Un enriquecimiento **cerca de 1,0** dice que los rescates se concentran en la banda exactamente como se concentran los errores, luego la concentracion no es evidencia de nada. Un enriquecimiento **alto** seria la senal que P66 no pudo ver por falta de denominador.

## C-E, el residuo contra la ETIQUETA, estratificado por banda

El AUC contra el **error** de V03 esta confundido: ese error es funcion determinista de z_V03 y del umbral, y el residuo contiene por construccion un termeno -b*z_V03. Contra la ETIQUETA no hay ese confound, y estratificar separa las dos hipotesis directamente.

| candidato | var del residuo | AUC global | AUC dentro de la banda | AUC fuera de la banda |
|---|---:|---:|---:|---:|
| MI_arbol | 0.8338 | 0.572628 | 0.567424 | 0.579597 |
| negacion | 0.2043 | 0.472985 | 0.492127 | 0.465626 |
| word_1_3 | 0.1469 | 0.499251 | 0.484706 | 0.504894 |
| cohesion | 0.1120 | 0.455979 | 0.451340 | 0.452727 |
| huecos_1_2 | 0.2193 | 0.516186 | 0.524327 | 0.516491 |
| contraste | 0.0807 | 0.437045 | 0.457961 | 0.432401 |
| huecos_6_10 | 0.1863 | 0.471673 | 0.444546 | 0.476209 |

## Veredicto

```text
reentreno del umbral          -0.0029731513
sustitucion cruzada           4 de 7 candidatos positivos
enriquecimiento de los rescates 1.3383
AUC del residuo fuera de la banda 0.579597

CATEGORIA DEL MECANISMO: MIXED_CON_ATRIBUCION_PARCIAL

Los cuatro controles NO dicen lo mismo y eso es el hallazgo. El reentreno anidado de UN escalar por etiqueta da -0.0029731513, es DECISIVO: la calibracion NO explica el efecto y V03 no estaba simplemente mal calibrado en esta particion. Y el AUC del residuo de MI_arbol se mantiene fuera de la banda, en 0.579597, lo que un movimiento de frontera no puede producir. PERO la misma correccion da positivo con 4 de los 7 candidatos, luego una parte del efecto pertenece a la OPERACION y no a MI_arbol. El enriquecimiento de los rescates es 1.3383, mayor que uno, luego hay concentracion real en la banda pero no exclusiva
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p67_01_controles_decisivos.py`. Sin push.*