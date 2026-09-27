# Contexto de la cadena - V25 (en curso)

Actualizado 2026-09-27. Worktree `/home/carlos/Git/PLN-v25`, rama `v25`, base `ad761ba`.
**V22, V23 y V24 cerradas sin candidato. V25 en curso. Sin submission en V22-V25.**
**V03 = 0.78306 LB, intacto, sigue siendo el campeon.**

## Decisiones de cierre por fase

| fase | veredicto | motivo en una linea |
|---|---|---|
| V21 | `NO CANDIDATE JUSTIFIES SUBMISSION` | WS-3 y VS-100 son la misma perturbacion (80.3 % de cambios solapados); L1 C=4.0 da +0.006431 con neto de +3 celdas |
| V22 | `CERRAR V22` | Hipotesis 1 (mismatch de preprocesamiento) **refutada**: HISTORICO menos CRUDO = +0.012871, ya se aplicaba. Hipotesis 2 (All-But-The-Top) **refutada**: D=1/2/5 empeoran y PC1 solo explica el 5.3 % de la varianza |
| V23 | `REJECTED` (submission `56599802`, LB **0.77938**, -0.00368) | El mejor conjunto de evidencia interna de la historia (corrected_cv 0.790410, +0.0132, 5/5 pliegues, McNemar p=0.0452) y aun asi perdio. Mecanismo: 60-76 % de vectores de soporte |
| V24 | `NO REALISTIC CANDIDATE - CLOSE` | El modelo de transferencia exige `dOOF = +0.03756`; el techo historico es +0.0132 |
| V25 | en curso | Factorization Machine, la unica mecanica de la literatura sin ejecutar |

## V23: la leccion que gobierna todo lo demas

La mejor evidencia OOF interna de la historia **no** se transfirio. El gap local-LB paso
de **+0.005849** (V03) a **-0.011030**. Ademas, con `svd:rbf_svc` la prediccion de
`n_vaportes` fue un factor util: 1321/1044/1317 de 1731 vectores de soporte (60-76 %).

## V24: el techo, medido

- Modelo de transferencia: `dLB = -0.00920 + 0.2448 * dOOF` (r=+0.4876, p=0.0342, n=19).
- Para empatar con V03: `dOOF = +0.03756`. Umbral a **0.75 SE** del propio LB (SE=0.01219)
  y al **46 %** del techo oracle de kNN (+0.0811).
- kNN ejecutado y descartado: solo **12 de 757** errores de V03 (1.59 %) tienen vecino
  confiable; precision al contradecir V03 **0.1500**; mejor `dOOF` **+0.004952**.

## V25 P0: la auditoria de mecanismos

`docs/v25_mechanism_audit.md`, 13 mecanismos. La FM es **genuinamente nueva**: 0 ocurrencias
de `FactorizationMachine`, `xlearn`, `libffm`, `fastFM` o `polynomial degree=2` en todo el
repo. Las multiplicaciones elemento a elemento que ya existian (`matriz.multiply(vector)`,
norma de Frobenius) no producen terminos cruzados, y `v12_03_entrenamiento_por_pares.py`
trabaja con pares de **etiquetas**, no de features.

## V25 P1: cuatro hallazgos de implementacion que no son bugs Mein

1. **liblinear se detiene en 5 o 6 iteraciones** (`n_iter_` del artefacto congelado). Su
   solucion **no es el optimo** del objetivo que minimiza: un L-BFGS completo llega a un
   valor **menor** (845.861649 frente a 852.508395 en el pliego 1 de HS) y cae en otro punto
   de una superficie de F1 plana (-0.0016 de OOF). Por eso el camino de control **delega**
   en la ruta exacta de V03; el L-BFGS propio queda como diagnostico.
2. **El intercepto es esencial y no es trivial.** `fit_intercept=True` con interceptos
   reales (-0.357 / -1.000 / -0.735) y **no** regularizado. Sin el, la tasa de positivos con
   umbral 0.44 pasa de 0.4545 a 0.4949.
3. **`v = 0` es un punto estacionario exacto** de la FM: la interaccion es par en `v`, asi
   que su derivada en el origen es cero y un optimizador de gradiente que arranque ahi no
   se mueve nunca. Hay que inicializar `V` al azar, como en la practica estandar.
4. **La interaccion se satura en rango 4**: `k=8` y `k=16` dan resultados identicos a `k=4`
   (normas coinciden en cuatro decimales). El rango efectivo del problema es bajo.

Ademas, dos errores reales los detecto la verificacion por diferencias finitas **antes de
entrenar nada**: un doble factor `C` en el gradiente (el optimo salia x2) y un objetivo que
evaluaba `0.5*||w||^2` pero no `0.5*||V||^F^2` mientras el gradiente si lo anadia. En los
nueve componentes comprobados, `gradiente_analitico - gradiente_numerico` era exactamente
`v[j,f]`.

## Revision bibliografica de V25

`docs/v25_literature_review.md`, 9 referencias verificadas por Crossref, ACL Anthology y
arXiv. **7 de 9 coinciden en todos los campos.** Tres correcciones materiales:

- **Rendle 2010**: el DOI `10.1109/ICDM.2010.127` es correcto, pero el arXiv de reserva
  `2010.10593` **apunta a otro trabajo** (CMIM, Sylvain et al., ICASSP 2021). No hay preprint
  sustituto de Rendle en arXiv.
- **All-But-The-Top**: **falta un autor, Suma Bhat** (son tres). El error estaba propagado en
  `reports/v21_p1_01_embeddings_desde_cero.md` en dos sitios; **corregido**.
- **GAttention**: el apellido es Vazquez y el nombre de pila Horacio Jarquin; lo declarado
  funde ambos.

De las nueve, **solo All-But-The-Top se ha ejecutado alguna vez**, en V22, con veredicto
negativo. Las otras ocho no se han medido nunca.

## Fugas y trampas que sobreviven

- **Fuga historica, detectada dos veces**: V4-C05 usa `dureza["error_score"]` (la prediccion
  OOF de V03 sobre todo el desarrollo) como `sample_weight`. No es candidata valida.
- **Trampa de dtype**: `read_test()` devuelve `id` como `object` (el CSV esta en latin-1 con
  `;`). Forzar `.astype(np.int64)` antes de comparar o escribir.
- **Trampa de `.vec`**: V5.1 consulta el **binario** de fastText (SIFTS en
  `data/external/embeddings/*_sifts.txt`); el `.vec` solo exporta el vocabulario. De 6.059
  tipos consultados solo 3.777 aparecen. Verificar siempre contra
  `vocabulario_toxic(read_train())`.
- **Bug operativo**: al parchear por numero de linea, un fallo en una asercion aborta antes
  del `write_text` y deja la edicion sin aplicar. Para ISC004 usar `ruff check --fix
  --unsafe-fixes`.
- **Precedente, CORREGIDO el 2026-09-27**: **0 de 22 submissions han superado a V03.**
  Las 22 estan por debajo, y V03 es el maximo historico absoluto. La segunda mejor es
  `v08_meta_v2` con LB 0.781630, o sea **-0.00143** frente a V03. La cifra que yo venia
  repitiendo ("19 de 21 por debajo, y v10 +0.00047 y v11 +0.00184 subieron") era una lectura
  erronea de la columna `diferencia` del log: esa columna es **local menos kaggle**, es decir
  cuanto fallo la transferencia, NO cuanto se supero a V03. Leida al reves convierteizu
  fracasos en victorias. Hay que distinguir siempre las tres bases: score de LB, diferencia
  frente a V03, y diferencia frente al propio local.

## Estado

- **V03 = 0.78306**, sha256 `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`.
- **Submissions: 22 en el log, 1 enviada en V23, 3 disponibles.**
- Las **3 submissions restantes se reservan**. La primera solo con evidencia OOF favorable,
  complejidad razonable y mecanismo real. No enviar variantes multiples para ver cual funciona.
- **Pendiente ajeno al trabajo**: rotar el token `ghp_rR3A...` ya expuesto, y revisar el WIP
  de KAGGLE_AGE en `git stash`.

## Fase V44 - SPARSE ONLINE OPTIMIZATION y PLS correcto (2026-09-27)

**Nota de reconciliacion (2026-09-27).** Este bloque antes mezclaba tres fases distintas:
el texto que se rotulaba "V44-00" era en realidad **V40-00**, el que se rotulaba "V44-02
PLS sobre SVD-300" eran los numeros de **V40-01**, y el PLS de "V44-01" era en realidad el
**V44-02** de verdad. Las fases V40 y V42 ya estaban descritas en sus proprios sitios; aqui
solo se deja V44 con sus cifras reales, tomadas de los JSON de `reports/`.

**V44-00 AUDIT_ONLY**: Ni FTRL, ni Coin Betting, ni PiSTOL aparecian en el registry. El unico
optimizador online historico era SGDClassifier (V35), ya cerrado. La familia VW estaba
genuinamente sin probar.

**V44-01 VW optimizers**: REJECTED. Los tres por debajo de V03, y el mejor sigue a -0.140.

| estrategia | corrected_cv | delta vs V03 | rescates | danos | net |
|---|---:|---:|---:|---:|---:|
| V03 (control) | 0.777211 | 0 | - | - | - |
| FTRL | 0.597204 | -0.180007 | 355 | 1857 | -1502 |
| Coin Betting | 0.629918 | -0.147294 | 335 | 1631 | -1296 |
| PiSTOL | 0.636881 | -0.140331 | 332 | 1597 | -1265 |

`SPARSE_ONLINE_OPTIMIZATION = CLOSED`. Limitacion registrada: `v44_01` no fija los hilos de
Vowpal Wabbit, asi que la corrida no es bit-reproducible; el JSON es el registro de la corrida
efectuada. El veredicto no depende de ello: los tres estan entre -0.140 y -0.180.

**V44-02 PLS correcto**: corrige el error algebraico de V40/V41/V42, donde la entrada del PLS
era una matriz de 3 columnas (rango 3) y no el `document x term` de 20441 dimensiones. Esta
vez la entrada es el TF-IDF de V03 y Y = [HS, TR, AG].

| variante | corrected_cv | delta vs V03 | train F1 | gap |
|---|---:|---:|---:|---:|
| V03 (control) | 0.777211 | 0 | 0.970865 | +0.193653 |
| SVD matched | 0.662013 | -0.115198 | 0.665763 | +0.003750 |
| PLS | 0.736782 | -0.040430 | 0.964178 | +0.227396 |

PLS supera a SVD a igual dimension (+0.074768), o sea que la supervision
si orienta las componentes, pero ninguna de las dos alcanza a V03. **PLS = CLOSED.**

**V44-03 integracion**: notebook 177 -> 181 celdas, seccion 39 = frontier audit. Presenta
**Optuna como unico candidato pendiente**; PLS ya estaba cerrado en V44-02 cuando se integro.

**Conclusion de la fase**: ni los optimizadores online ni la proyeccion latente supervisada
aportan nada transferible sobre V03. V03 intacto. 0 submissions.
