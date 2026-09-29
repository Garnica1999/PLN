# P33+ - anomalia de parseo de URL (`t`, `t.`, `t.c`)

## Que es

**1,685 filas** contienen una URL cuyo dominio no es un dominio valido, sino un prefijo de `t.co` cortado al final del texto. Ocurrencias totales: 1,685.

| Dominio | Ocurrencias |
|---|---|
| `t` | 593 |
| `t.` | 565 |
| `t.c` | 527 |

## Es consistente con un `t.co` truncado

- URLs de `t.co` **enteras**: 0
- Prefijos truncados (`t`, `t.`, `t.c`): 1,685
- Longitud mediana del token de ruta de un `t.co` entero: 0 caracteres (min 0, max 0)

**si: los truncados son prefijos exactos de `t.co` (`t`, `t.`, `t.c`) y no aparecen como dominio propio en ninguna otra parte del corpus. No son un dominio distinto, son el mismo `t.co` cortado al final del texto**

## Distribucion por clase

| Clase | Filas afectadas |
|---|---|
| 13-17 | 129 |
| 18-24 | 121 |
| 25-34 | 185 |
| 35-49 | 342 |
| 50-64 | 314 |
| 65-xx | 594 |

## ¿Afecta a la normalizacion de texto existente

- Longitud media del texto afectado: 138.35
- Longitud media tras quitar las URLs: 122.21
- Fraccion de texto suprimida: 0.11667

**quitar la URL deja casi todo el texto, asi que la normalizacion existente no esta perdiendo contenido de edad: la URL es un sufijo que se elimina limpio. Por eso la anomalia **no requiere ninguna intervencion****

## Decision

**DATASET/URL PARSING ANOMALY**, conservada como diagnostico. **No se construye ninguna feature** a partir de ella, no se reinterpreta como senal util, y no se entrena ningun modelo para estas filas, tal como pide el brief

## Por que esta en un informe propio y no en una linea de codigo

Porque una anomalia de parseo que no documenta nadie se convierte enrumor: el dia que alguien vea `t.c` en un Print y piense que es un dominio raro, va a construir una feature sobre arena. Al dejar por escrito que son 1.689 filas de un `t.co` cortado, que la normalizacion las borra limpiamente y que **no** se usa ninguna, queda una decision y no una laguna.
