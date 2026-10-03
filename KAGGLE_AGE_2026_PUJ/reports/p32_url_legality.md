# P32+ - P02.4, auditoria de legalidad del retrieval de URL

**Clasificacion: AMBIGUOUS. Estado: QUARANTINED.**

## Lo que se comprobo, antes de decidir

| Comprobacion | Resultado |
|---|---|
| Reglas locales de AGE en el arbol | 0 ficheros |
| API de Kaggle accesible (solo lectura) | False |
| Fichero de reglas de otra competicion | si, y no se usa |

## Por que AMBIGUOUS y no ALLOWED ni PROHIBITED

Las reglas oficiales de la compilacion **no estan disponibles** en este entorno: no hay copia local de las de AGE, la API de Kaggle no autentica, y el unico fichero de reglas presente en el arbol pertenece a **otra** compilacion (toxicidad: HS, TR, AG) pese a t titularse 'Reglas oficiales de la competencia'. Usarlo como fuente daria por cubierto un requisito que no lo esta, y eso es peor que no tener nada.

Sin el texto de las reglas no se puede establecer que la recuperacion este permitida. Y recuperarla podria ser informacion externa, transductiva o fuga de metadatos, que es exactamente el riesgo que el brief manda|posponer.

## Decision

QUARANTINED sin excepcion, tal como manda el brief para el caso ambiguo. No se hace ninguna peticion HTTP, no se recupera contenido, y ningun candidato que dependa de contenido recuperado puede llegar a una submission. Esto cierra de una vez P2.8, P2.9, P2.10, P12 y H-P32-10

## Que se cierra con esto

| Track | Estado |
|---|---|
| P2.8 recuperacion de contenido | **QUARANTINED** |
| P2.9 experimentos de retrieval (U1-U5) | **QUARANTINED** |
| P2.10 cache de retrieval | **QUARANTINED**, no se crea |
| P12 reemplazo de URL por contenido | **QUARANTINED** |
| H-P32-10 del gap analysis | **QUARANTINED** |

## Lo que sigue abierto y es legal

P2.5, P2.6 y P2.7 siguen siendo legales y ejecutables: usan **solo** el texto que ya esta en el dataset, sin descargar nada. Es la unica via de la URL que esta permitida con lo que se sabe

## Como se levanta la cuarentena

que el organizador confirme por escrito si la recuperacion del contenido de una URL propia esta permitida y que no se combina con metadatos. Sin ese texto, la clasificacion no puede pasar de AMBIGUOUS

## Comprobaciones negativas verificadas por inspeccion

| Restriccion | Estado |
|---|---|
| peticiones HTTP en toda la fase | **0** |
| contenido recuperado de URLs | **0** |
| metadatos de usuario, engagement o perfil | **0** |
| uso del test para decidir nada | **0** |

La ultima tabla no es una declaracion: `scripts/p32_21_compliance.py` la comprueba recorriendo el **AST** de cada script de la fase buscando imports y llamadas.
