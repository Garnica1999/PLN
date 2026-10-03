# V13 · P16c — Reconstrucción estable desde V12

P16 no pudo concluir nada: su regresión lineal era inestable porque V12 está muy colineal (VIF mediano 18,9 y 528 en los scores de miembros, según P0b), y un R2 negativo ahí no significa que la información falte sino que el estimador se dispara. Este experimento parametriza el ajuste por **grados de libertad efectivos** en vez de por una penalización elegida a ojo, y toma el mejor R2 fuera de muestra de cada columna sobre una rejilla de complejidad (2, 5, 10, 20, 50, 100).

La escala de referencia la da P0b: 87 de las 278 direcciones de V12 explican el 95 % de su varianza, así que un ajuste con 20 grados de libertad debería capturar lo esencial si existiera.

## 1. Mejor R2 por columna

| Bloque | Columnas | Mejor R2 mediano | Mejor R2 máximo | Columnas con R2 > 0,5 | Columnas con R2 > 0,8 |
|---|---|---|---|---|---|
| P1 tokens | 36 | **0.527** | 0.968 | 21 | 2 |
| P17 pronombres | 26 | **0.211** | 0.382 | 0 | 0 |
| P20 modalidad | 23 | **0.037** | 0.268 | 0 | 0 |

## 2. Control del bloque reconstruido

| Bloque | Modelo | Grados de libertad | Control original | Control reconstruido | Suelo de azar |
|---|---|---|---|---|---|
| P1 tokens | logreg | 2 | 0.3915 | **0.0485** | 0.1667 |
| P1 tokens | linearsvc | 2 | 0.3841 | **0.0485** | 0.1667 |
| P17 pronombres | logreg | 2 | 0.1918 | **0.0485** | 0.1667 |
| P17 pronombres | linearsvc | 2 | 0.1881 | **0.0485** | 0.1667 |
| P20 modalidad | logreg | 2 | 0.1270 | **0.0485** | 0.1667 |
| P20 modalidad | linearsvc | 2 | 0.1259 | **0.0485** | 0.1667 |

## 3. Lectura

- **P1 tokens**: el mejor R2 de cualquier columna llega a 0.968 y la mediana se queda en 0.527. 21 columnas superan 0,5: parte de la información **sí** estaba dentro de V12.
- **P17 pronombres**: el mejor R2 de cualquier columna llega a 0.382 y la mediana se queda en 0.211. Ninguna columna supera 0,5, de modo que **V12 no contiene esa información** de forma linealmente accesible: el problema de las fusiones no es la redundancia.
- **P20 modalidad**: el mejor R2 de cualquier columna llega a 0.268 y la mediana se queda en 0.037. Ninguna columna supera 0,5, de modo que **V12 no contiene esa información** de forma linealmente accesible: el problema de las fusiones no es la redundancia.

## 4. Consecuencia para el plan

- Si los tres bloques quedan por debajo de 0,5 en todas sus columnas, la redundancia lineal queda **descartada** como explicación de que cuatro pistas seguidas no hayan movido la macro-F1. Abrir más pistas de variables del mismo tipo (P2 de dominios de URL, P6 de centroides, P18, P21, P24) tiene el mismo resultado esperado, y el §28 pide no ejecutar por cantidad.
- Lo que queda en pie es el otro diagnóstico: la información existe, es nueva, y el combinador no la usa. Eso es un problema de **arquitectura**, no de representación, y apunta a §37: una puerta `gate(especialista, V12)` que active el bloque solo donde tenga sentido, en lugar de promediarlo sobre todas las filas.
- P16b mide exactamente si esa puerta tendría techo: en qué filas el bloque es mejor que V12 y cuánto separates ese caso.

