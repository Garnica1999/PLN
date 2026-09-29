# P33 - decision

```text
ESTADO = CLOSED con evidencia residual registrada
```

La taxonomia del proyecto distingue varios estados y **este caso ocupa dos a la vez**, que es lo que hay que decir en vez de forzar uno solo:

| Nivel | Estado | Motivo |
|---|---|---|
| Integracion global de seis clases | **CLOSED** | la correccion no mejora el F1, el IC95 contiene el cero y el control de ruido gana a la senal |
| Evidencia residual en si | **COMPLEMENTARY_BUT_NOT_TRANSFERABLE** | AUC 0,5411 en las seis clases: hay informacion que V30 no expresa, y no se ha conseguido transferirla |

Que no sea `PROMISING` ni `FINALIST` no requiere discusion: el delta es negativo, los folds positivos son 1 de 5 y el control de capacidad falla.

**Lo que queda abierto y lo que no.**

- **Cerrado**: la correccion global de seis clases sobre el score. Cualquier variante de esa idea (distinta transformacion, distinto `beta`, mas canales) es un duplicado.
- **Abierto, pero no como score correction**: el residuo como *evidencia* podria consumirse en un mecanismo que no sea sumar un termino al score. Esa es la unica via que P33 deja viva, y P34 se uso para explorarla.

**Lo que NO se afirma.**

- No se afirma que la informacion linguistica no exista: existe y esta medida.
- No se afirma que V30 sea optimo: solo que estos mecanismos no la aprovechan.

**Cifras de referencia.**

- V30: 0.6134471 (leaderboard 0.63302)
- P33 real: 0.6132319 (-0.0002152)
- Bootstrap: -0.0002162, IC95 [-0.0006956, +0.0002533]

**Siguiente paso tomado**: P34, que USO esa evidencia en un mecanismo local distinto. Tambien quedo invalidado.
