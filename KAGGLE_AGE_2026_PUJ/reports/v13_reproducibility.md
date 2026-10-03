# V13 · Reproducibilidad y cumplimiento

Documento de cierre de la fase. Complementa a `reports/v13_final_research.md`, que cuenta qué se
encontró; este cuenta **cómo se vuelve a obtener cada número** y **qué restricciones se cumplieron**.

## 1. Entorno verificado

- Intérprete: `/home/carlos/venv/bin/python` (Python 3.14.7). Ningún experimento usa otro.
- `numpy 1.26.4`, `scikit-learn 1.8.0`, `scipy 1.17.1`.
- GPU presente pero **no usada por V13**: ninguno de los experimentos de la fase instancia
  `torch`. El único import del proyecto es `src/gpu.py:130`, que detecta la tarjeta (D-038).
- CPU de 32 hilos. Los experimentos usan el paralelismo por defecto de BLAS y de los propios
  estimadores; no se fijó `OMP_NUM_THREADS` en ningún script porque el fallo de D-040 no depende de
  los hilos (se reproduce con 1 y con 32).

## 2. Advertencia que hay que leer antes de reproducir nada

En este entorno, `numpy 1.26.4` sobre `Python 3.14.7` calcula **mal** el producto matricial cuando
los dos operandos son vistas de la misma matriz, es decir en el patrón `X @ X.T` (D-040). También
fallan `np.corrcoef` y `np.cov` con `rowvar=True` sobre matrices 2D, que hacen internamente esa
auto-transpuesta.

Rutas **verificadas como sanas** en este entorno: `A @ B.T` con matrices distintas, `A.T @ A`,
`Z.T @ z` (matriz-vector), `np.dot` entre vectores, `np.einsum`, `np.corrcoef` y `np.cov` con
argumentos 1D, `np.linalg.norm`, `np.var`, `np.linalg.solve` y `np.linalg.eigvalsh`.

Reproducción mínima del fallo:

```python
import numpy as np
X = np.random.default_rng(0).normal(size=(2000, 5))
print((X @ X.T)[0, 0], (X[:, 0] ** 2).sum())   # 0.44  12.97  -> FALLA
print(np.corrcoef(X).shape)                    # (2000, 2000) -> deberia ser (5, 5)
print(np.corrcoef(X.T).shape)                  # (5, 5)       -> OK
```

Defensa permanente: `src/verificacion_numerica.py`.

- `autotest_numerico()` comprueba seis rutas en las que el proyecto se apoya y **lanza**
  `ErrorNumerico` si alguna falla. Está en la primera línea de la mayoría de los scripts de V13.
- `grama_exacta(matriz)` calcula productos cruzados por sumas explícitas, sin BLAS. Coincide con
  `scipy.stats.spearmanr` con una desviación máxima de **1,2e-16**. Es la vía que usan la criba de
  redundancia y la reconstrucción de P16c.

**Impacto en el proyecto: ninguno.** Los diez sitios del repositorio que usan producto matricial o
`corrcoef` usan rutas sanas, y el `MLPClassifier` de producción da predicciones idénticas con la ruta
parcheada. V12 (CV 0,6004 / LB 0,62188) y todas las decisiones anteriores siguen vigentes.

## 3. Orden de ejecución y tiempo

| # | Script | Qué produce | Minutos aprox. |
|---|---|---|---|
| 1 | `scripts/v13_p0a_auditoria_idioma.py` | `reports/v13_auditoria_idioma.md` | 0,2 |
| 2 | `scripts/v13_p0b_colinealidad.py` | `reports/v13_colinealidad.md` y 4 CSV | 0,5 |
| 3 | `scripts/v13_p15_matriz_errores_miembros.py` | `reports/v13_p15_member_error_matrix.md` y 3 CSV | 0,2 |
| 4 | `scripts/v13_p1a_evidencia_tokens.py` | `reports/v13_p1_evidencia_tokens.md`, 5 matrices por fold | 4 |
| 5 | `scripts/v13_p1c_identidades.py` | `reports/v13_p1_identidades.md` | 9 |
| 6 | `scripts/v13_p1b_fusion_tokens.py` | `reports/v13_p1_fusion.md` | 95 |
| 7 | `scripts/v13_p17a_pronombres.py` | `reports/v17_pronombres.md` | 2 |
| 8 | `scripts/v13_p17b_pronombres.py` | `reports/v17_pronombres_fusion.md` | 2 |
| 9 | `scripts/v13_p20_modalidad.py` | `reports/v20_modalidad.md` | 2 |
| 10 | `scripts/v13_p3a_especialistas.py` | `reports/v3_especialistas.md` | 2 |
| 11 | `scripts/v13_p19_conjunciones.py` | `reports/v19_conjunciones.md` | 4 |
| 12 | `scripts/v13_p16c_reconstruccion.py` | `reports/v16c_reconstruccion.md` | 9 |
| 13 | `scripts/v13_p16b_activacion.py` | `reports/v16b_activacion.md` | 3 |
| 14 | `scripts/v13_p16d_puerta.py` | `reports/v16d_puerta.md` | 4 |
| 15 | `scripts/v13_cierre.py` | `reports/v13_master_results.csv` y las dos tablas agregadas | 0,1 |

Dependencias: el 6 necesita las matrices del 4; el 12 y el 14 también. El 5 rehace el bloque del 4
con otro vocabulario y sirve de validación cruzada del 4 (reproduce 0,3997 exacto). Los demás son
independientes entre sí.

Lo caro es el 6: cuatro evaluaciones completas del combinador (la candidata y tres controles
permutados) más cuatro evaluaciones por cabeza. Los diagnósticos no entrenan nada.

Artefactos de datos en `features/` (no versionados, sobreviven a `git clean`):

- `v13_p1_tokens_fold{0..4}.npy` y `v13_p1_tokens_nombres.json`
- `v13_p1b_oof.npy` (probabilidades OOF de la candidata de P1b)
- `v17_pronombres.npy`, `v20_modalidad.npy`, `v19_conjunciones.npy` y sus nombres

## 4. Decisiones de diseño que conviene conocer al reproducir

**Los folds.** Todos los experimentos usan `cv.make_folds(y, groups=clusters, n_splits=5, n_repeats=1,
scheme="group")` sobre `features/dup_clusters.parquet` y la misma `cfg.SEED`. No se ha creado nunca
una partición nueva de folds: el holdout histórico queda intacto y sin usar.

**Las features supervisadas dentro del fold.** P1a hace ajuste cruzado de 5 folds interiores dentro de
cada train exterior: las características de cada fila del train salen de un perfil calculado sobre las
otras cuatro partes, y las de validación, del train exterior completo. El vocabulario y las
frecuencias de documento se calculan **siempre** sobre el train interior. El test no se toca en
ningún punto de la fase.

**El bootstrap.** Es **pareado**: remuestrea la tabla conjunta de (clase verdadera, predicción de la
referencia, predicción de la candidata), de modo que los dos modelos se evalúan sobre el mismo
remuestreo de filas. Se validó contra el método clásico de remuestreo de filas: difieren en 5e-5. Con
predicciones idénticas da exactamente [0, 0], que es la comprobación de que está bien construido.

**El pre-chequeo de rescate.** P17b, P20, P16b y P16d aplican una puerta antes de pagar una
evaluación cara: se mide el rescate y el daño del bloque sobre el OOF de V12 y, si el neto no llega al
umbral, la pista se cierra. Esa puerta evitó ~70 minutos de CPU en P17 y evitó por completo tres
evaluaciones del combinador.

**Las tres correcciones de P16.** La primera versión (P16) fue **inconcluyente** por un error de
diseño mío: las ecuaciones normales no incluían término independiente, de modo que la predicción se
forzaba a pasar por el origen y todos los R2 salían negativos. La segunda corrección tipificó las
columnas, porque sin tipificar la gramática de V12 tiene una dirección que acapara la traza entera
(8,99e12 de 8,99e12) y los grados de libertad efectivos se hunden a 1. La tercera parametrizó la
penalización por grados de libertad efectivos en vez de por un alfa fijo. Un resultado que necesitó
tres correcciones para no ser falso es un resultado que hay que leer con cautela, y por eso P16c
publica **las dos** medidas: el R2 (accesibilidad de varianza, 0,522 de mediana) y el control
(información de clase, 0,0485), que divergen.

## 5. Estado del repositorio

- Rama: `kaggle-age-2026-puj-v13`, basada en `kaggle-age-2026-puj-lexicons`.
- Todo es **local**. No hay push, ni PR, ni merge, ni cambio de remotos, por indicación expresa.
- `git stash list` conserva `stash@{0}` como respaldo de la recuperación de la fase de lexicones; no
  se ha eliminado.
- Queda un fichero sin versionar ajeno a este proyecto: `kaggle_toxic_2026_puj/tools/`, que pertenece
  a otra sesión del repositorio y se retiró del índice a propósito.
