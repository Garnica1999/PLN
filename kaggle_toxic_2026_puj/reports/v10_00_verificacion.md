# V10-00: verificacion de partida y preparacion de la fase v10

Generado por el trabajo de preparacion de la fase v10. Este script no existe como fichero
propio: la verificacion se ejecuto en linea y su resultado quedo en
`reports/v10_00_verificacion.json`, que es la evidencia de este experimento.

## Que se verifica y por que antes de nada

La fase v10 arranca con gates pre-registrados, y un pre-registro sin un baseline verificado
no vale nada: los gates estan escritos en numeros que salen de esa verificacion. Ademas, la
auditoria forense anterior demostro que varias decisiones antiguas dependian de cifras de
referencia mal redondeadas, de modo que comprobar que la referencia se reproduce **exacta**,
y no aproximadamente, es parte del entregable y no un detalle.

## Resultado de la verificacion

| comprobacion | valor obtenido | valor esperado | veredicto |
|---|---|---|---|
| `corrected_cv` reconstruido desde el OOF | 0.7772114525165721 | 0.7772114525165721 | coincide exacto |
| `sha256` de la submission v03 | 7d7478f2...4520e | 7d7478f2...4520e | identica |
| filas de la submission v03 | 1154 | 1154 | correcto |
| filas de `experiments.csv` | 207 | 207 | correcto |
| claves unicas del registro | 207 | 207 | correcto |
| sondas en el registro del leaderboard | 15 | 15 | correcto |
| prevalencias de entrenamiento | 0.4367 / 0.2998 / 0.3824 | idem | correcto |
| grupos fusionados | 1645 | 1645 | correcto |
| artefactos copiados | 22 en `features/`, 9 en `models/` | idem | correcto |

F1 por etiqueta de la referencia, con los umbrales congelados de 0.44, 0.51 y 0.46:

| etiqueta | F1 |
|---|---|
| HS | 0.761689 |
| TR | 0.815730 |
| AG | 0.754215 |

## Preparacion del worktree

La fase v10 trabaja en `/home/carlos/Git/PLN-v10`, rama `proyecto-kaggle-toxic-2026-v10`,
creada desde `23aac93`. Se eligio un worktree propio, y no trabajar en el v8, por dos razones:
el v8 ya acumula catorce scripts de la fase v8 mas los de la auditoria, y mezclar ahi haria
imposible rastrear que cambio pertenece a que fase; y el prompt pide commits con prefijo de
fase, que en una rama compartida se mezclarian.

`data/` queda **enlazado** a la copia compartida, que es donde viven los datos oficiales y el
`.vec` de FastText que necesita la via C. `features/` y `models/` se **copian**, no se
enlazan, porque son artefactos derivados que la auditoria demostro que se necesitan: sin esa
copia los scripts fallan, y ya ocurrio una vez durante la propia auditoria.

## Advertencia que queda escrita antes de ver resultados

**El holdout2 que se creara en la sub-via siguiente no es un conjunto virginal.** Se construye
con un pliegue del desarrollo actual, y esas 1483 filas ya participaron en las decisiones de
las fases v1 a v9.

Su valor real es **detectar que un candidato esta sobreajustado a su gate**, que es un uso
legitimo y necesario, y **no** es probar generalizacion sobre datos nunca vistos. Decir lo
contrario despues de ver los numeros seria una reinterpretacion, y por eso queda escrito en
`tasks/plan_v10.md` **antes** de ejecutar nada.

De ahi una decision de orden que conviene explicitar: **holdout2 se crea antes de la via A**,
no despues. Si se creara despues, las candidatas de la via A se elegirian sobre las mismas
filas que luego se usarian para juzgarlas, y el juez dejaria de ser independiente de la
seleccion.

## Que queda registrado

- `reports/v10_00_verificacion.{md,json}`: evidencia de este experimento.
- `tasks/plan_v10.md`: pre-registro con los gates de las cinco sub-vias.
- `tasks/todo_v10.md`: estado de las tareas.
- Fila `V10-00` en `experiments.csv`.
