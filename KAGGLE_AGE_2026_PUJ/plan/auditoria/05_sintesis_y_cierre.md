# Auditoria V1-V37 · Fase 5 - Sintesis, cierre y trazabilidad

## Dictamen global

**El protocolo del proyecto esta limpio en lo esencial.** No hubo `fit` sobre test, ni calibracion de
umbrales sobre test, ni uso del leaderboard como conjunto de validacion ni como funcion objetivo. `id`
nunca se uso como feature, no hay leakage de autor ni temporal, y el campeon congelado es reproducible
bit a bit. Las cuatro adopciones reales tienen pre-registro, control negativo y confirmacion externa.

**Sobre 195 filas del registro: 151 ejecuciones + 44 documentos + 1 corrida abortada.** Toda la
evidencia citada existe (0 huecos).

**20 defectos confirmados**, ninguno invalida una adopcion.

| Severidad | Cantidad | Defectos |
|---|---|---|
| alta | 2 | K-01 (techo de LB 0,70), A-01 (multiplicidad) |
| media | 9 | K-02, K-03, N-01, N-02, N-03, A-02, A-03, A-04, A-06 |
| baja | 6 | K-04, N-04, N-05, N-06, N-07, N-08, A-07 |

## Lo que cambia una conclusion publicada

### K-01: el techo de LB 0,70 no demuestra lo que afirma

Unico hallazgo que invalida una afirmacion del proyecto. `scripts/v35_00g_techo_analisis.py:69` define
`TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)`: la rejilla **empieza en 0,30 y sube**, y el "techo" se tomo
como su maximo, mientras la curva del oraculo es monotona creciente al bajar tau.

Reproduccion propia: en el punto exacto que el propio D-087 usa como objetivo (tau = 0,2930), el
oraculo da **+0,0646**, por encima del +0,0626 exigido. El techo de +0,0568 era un artefacto del
redondeo de la rejilla.

**Lo que si se sostiene, y es mas fuerte**: un recalibrador tendria que acertar el ~16 % de los errores
con `p(y_true) >= 0,293`, es decir conocimiento casi-oracular. LB 0,70 sigue siendo inalcanzable en la
practica; **la prueba publicada no lo demuestra**.

### A-01: la multiplicidad de ~151 corridas no esta cuantificada

Con SE ~ 0,00188, un gate de `Delta >= +0,0015` implica **~32 falsos positivos esperados**; `+0,0030`
implica **~8**. `grep` de multiplicidad / multiple comparisons / FWER / family-wise en los cuatro
documentos maestros devuelve **cero**.

**Por que no invalida las conclusiones**: las cuatro adopciones tienen IC95 pareado excluyendo 0 y
confirmacion externa. La afirmacion correcta es "ninguno paso el gate **y su bootstrap pareado excluyendo
0**".

## Correcciones aplicadas

| Que | Donde | Verificacion |
|---|---|---|
| Fila `V37-DIAG-POS` escrita con 21 de 22 campos | `experiments.csv` | patrón de `V36-E1`; los 17 campos descriptivos coinciden con el original |
| SHA256 del pre-registro V37 mal transcrito | `DECISIONS.md`, `PROJECT_STATE.md` | corregido a `1d6d6573…`, que es el que registra el JSON |
| Clasificador que martaba V33/V34/V35 como no ejecutados | `scripts/auditoria_v1v37_inventario.py` | ahora `EJECUTADO`; 151 en vez de 148 |
| `seccion_de()` devolvia el H1 del documento | mismo script | resuelve por posicion de celda |
| Checks C1/C4/C5 tautologicos o codigo muerto | mismo script | los 6 checks tienen poder de fallo |
| Idempotencia rota (celdas crecian 4 por pasada) | `scripts/auditoria_v1v37_notebook.py` | estable en 3 pasadas |

## Autocorrecciones durante la propia auditoria

Cinco defectos los encontro y corrigio el propio auditor, y dos los detecto un revisor:

- La primera reparacion de `V37-DIAG-POS` **consumo el hueco equivocado** y **intexto texto nuevo** en
  `decision`. Detectado por `ml-qa-engineer`. La version final mueve la narrativa a `resultado` y
  recupera el cierre ya escrito, sin inventar nada.
- Fragmento en chino que se colo desde la salida de un subagente al transcribir su veredicto
  ("son" + tres ideogramas), detectado al verificar la codificacion de los documentos. No lo reproduzco
  aqui para no reintroducirlo.
- El script del notebook acumulo 4 defectos (249 bytes no-ASCII, la cabecera contada como capitulo,
  la idempotencia rota, dos checks mal escritos) antes de tocar el notebook real.

Es el valor de la leccion de **D-086** pagada por tercera vez: **el autoejercicio estructural detecto
defectos antes de gastar presupuesto y antes de tocar artefactos**.

## Reservas abiertas (no medidas, declaradas como tales)

1. **Sesgo del inner split de Optuna** (A-02): medido sin impacto en solapamiento exacto (0,963 %
   ingenuo vs 1,030 % agrupado); el impacto en la metrica de seleccion **no esta medido**.
2. **`max_iter=40` del MLP** (A-03): miembro del campeon con peso 0,25 sin revisar.
   `experiments/v29_05_optuna_mlp.json` **no existe**: la via se diseño y no se ejecuto.
3. **IC95 [0,635, 0,653] de D-087**: rango declarado, no estimado.
4. **Registro de envios**: 4 envios posteriores (V19, V20, V30, plug-in) sin linea de autorizacion de
   Carlos en `DECISIONS.md` ni en `reports/`. No hay evidencia de infraccion, si ausencia de evidencia.

Ninguna exige cambiar el campeon.

## Deuda documental declarada

- El notebook llega hasta **V35**. **V36** (D-088) y **V37** (D-089) no tienen seccion propia.
- 4 ficheros de `submissions/` no llevan marca de NO-ADOPTADO en el nombre.
- `config/hardware.json` reescrito a mano no coincide con lo que genera `scripts/fase00_entorno.py`.
- `experiments/auditoria_v1v37_huecos.csv` tiene 0 filas (esperado).

## Estado final

| Componente | Estado |
|---|---|
| Inventario | congelado, 197 filas, 6/6 checks, ruff limpio, ASCII puro |
| Capitulo del notebook | anadido, 5/5 checks, 326 celdas originales intactas |
| Acta | **D-090** en `DECISIONS.md` |
| Registro | 2 filas nuevas: `AUD-INV`, `AUD-REV` |
| Campeon | **intacto**, 0,6159732487657944, verificado por hash y recomputado |
| Push / merge / rebase / reset | **ninguno** |
