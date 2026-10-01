# Accion 4 · Re-optimizar V29/V30 con inner split AGRUPADO

## Que sepidio

Carlos autorizo "el 4": re-optimizar los hiperparametros del camino al campeon usando inner split
**agrupado** en lugar del ingenuo, ahora que la accion 3 demostro que el sesgo es material
(3,907 % de solape de cluster frente a 0,000 %).

## Lo que se midio que justifica hacerlo

`scripts/auditoria_medir_inner_split_optuna.py`, sobre los 5 folds externos canonicos:

| Nivel | solape de cluster en la validacion interior |
|---|---|
| inner ingenuo (codigo real, 5 scripts del camino al campeon) | **3,907 %** |
| inner agrupado (propuesto) | **0,000 %** |

Criterio declarado antes de correr: diferencia > 2 pp = material. Medida: 3,907 pp. Consistente en los
5 folds (3,787 % a 4,152 %).

## El problema de fondo, y por que NO se puede prometer un resultado

**Re-optimizar no es una mejora esperada: es una correccion metodologica cuyo signo es desconocido.**

El leakage del 3,9 % estaba en la **seleccion** de hiperparametros, no en la **estimacion**. La
estimacion externa de V29 y V30 se hizo con el outer split agrupado y es valida. Si el leakage inflaba
la seleccion, el re-optimizado puede **bajar** de rendimiento: se estaban eligiendo hiperparametros
que explotan la redundancia del corpus, y un esquema honesto puede preferir otros peores.

Ademas hay un precedente en el propio proyecto que ya anticipa este tipo de sorpresa:
`scripts/v30_02_ensemble.py:257` registra que el mejor `corrected_cv` de V30 fue **0.6130051**,
mientras el promedio ponderado de pesos da **0.6134471**, y el acta **D-066** lo reconoce como
auto-falsacion. La eleccion post-hoc de un metodo sobre dos candidatos ya esta contaminada por
seleccion; re-optimizar no la arregla, la agrava.

## Coste

| Corrida | Trials por fold | Tiempo real medido |
|---|---|---|
| `v29_04_optuna_individual.py` (LightGBM + MLP) | 30 | 5.152 s (~86 min) |
| `v30_01_optuna_refinamiento.py` (refinamiento local) | 15 | 2.449 s (~41 min) |
| `v30_02_ensemble.py` (pesos) | 300/fold | 62,6 s |

Re-optimizar V29 **y** V30 son ~2 h 10 min de CPU. Los recursos estan disponibles (32 hilos,
14 GB libres).

## Por que NO lanzo esto de inmediato

Re-optimizar sin pre-registro seria exactamente el defecto que la auditoria reprobo en otras fases: un
cambio de metodo ejecutado sin criterio de decision declarado antes. Con multiplicidad ya
identificada como el defecto estructural del proyecto (A-01), hacer un experimento mas sin
pre-registro y sin umbral declarado seria incoherente con el acta D-090 recien cerrada.

Lo que hace falta, y no es barato de hacer bien:

1. **Pre-registro congelado** con criterio de decision ANTES de correr: ¿que se adopta si el
   re-optimizado rinde MAS, y que se hace si rinde MENOS? La opcion honesta por defecto es: si rinde
   menos, **no se adopta y se documenta que el leakage habia inflado la seleccion**, porque ese
   hallazgo tiene valor cientifico aunque cueste puntuacion.
2. **Amplitud**: re-optimizar solo V30 (41 min) o tambien V29 (86 min) cambia la pregunta. V30 es el
   refinamiento del campeon; V29 es donde se eligieron los tres miembros.
3. **Confirmacion**: un delta de CV no basta en este proyecto. El gate historico exige IC95 pareado
   excluyendo 0 y confirmacion en particion alterna. Sin las dos, no hay adopcion.

## Mi recomendacion

**Pre-registro primero, corrida despues.** Es una linea de coste bajo y es lo que el proyecto ya hace
bien: de las cuatro adopciones reales, las cuatro tienen pre-registro.

Si Carlos prefiere ahorrar las ~2 h y evitar el riesgo, la alternativa de menor coste y mayor valor
es **no re-optimizar y cerrar la reserva asi**:

> "La seleccion de hiperparametros del campeon se hizo con un inner split que permitia 3,907 % de
> solape de near-duplicates. La estimacion externa no estaba afectada. Por tanto **la cifra publicada
> del campeon es valida**, pero **el proceso de seleccion no es reproducible bajo un esquema honesto**."

Esa frase es defendible, cuesta cero horas de CPU, y es mas informativa que un numero que puede bajar.

## Estado

**PENDIENTE DE DECISION DE CARLOS.** No se ha ejecutado nada. El resto de las tres acciones
(1 fijar `es_core_news_lg`, 2 etiquetar el IC95, 3 medir el sesgo) estan **cerradas**.