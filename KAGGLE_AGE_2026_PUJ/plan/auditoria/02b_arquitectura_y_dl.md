# Auditoria V1-V37 · Fase 2b - Arquitectura metodologica y redes neuronales

Revision delegada a `ml-architect` (diseno experimental transversal) y `dl-specialist` (regimenes de
entrenamiento NN), ambas sobre solo lectura. Ningun artefacto de experimento fue editado.

## Inventario real de experimentos con red neuronal

`dl-specialist` localizo el alcance real, que no coincide con lo que el registro sugiere:

| Tipo | Experimentos |
|---|---|
| `torch` + `backward()` | `V32-E1`, `V18-P3`, `V18-P4`, `V18-P5` |
| `MLPClassifier` de sklearn | `E-019a`, `E-019b`, `V19-B1` (via `EncodedMLP`) |
| Consumen un MLP pero no entrenan NN | `E-020`, `E-018` |
| **No son NN** | `E-018` (LogReg), `V17-P9-ARBITRO` (LogReg binaria), `ADV-01..07` (LightGBM/NBSVM) |

**Ninguna red neuronal fue adoptada como campeon.** `v30_prob.npy` sin escrituras en todo el scope.
El campeon V30 + plug-in sigue en 0.6159732487657944 (baseline recomputado por `V32-E1` con
`baseline_ok: true`).

## Veredicto por familia (de `ml-architect`)

| Familia | n | veredicto | severidad |
|---|---|---|---|
| V1-V4 fundacional (E-*, B*, ADV-*) | 37 | correcto con reservas | media |
| V5-V12 fases PH* + V13-V17 | 76 | correcto con reservas | media |
| V13-V17 busqueda residual | 13 | correcto con reservas | baja |
| V18-V23 frontera y ordinal | 12 | **correcto** | baja |
| V24-V29 seleccion y ensemble | 15 | correcto con reservas | media |
| V30-V31 campeon y tracks | 14 | correcto con reservas | media |
| V32-V37 reapertura y falsacion | 28 | **correcto** | baja (1 alta) |

`V18-V23` y `V32-V37` son los mas correctos en el sentido fuerte: metodos con control de atribucion limpio, gates
fail-fast con anclas medidas, gemelos con presupuesto identico, y auto-falsaciones que se disparan solas.

## Defectos confirmados

### A-01 (ALTA) La multiplicidad de las ~151 corridas no esta cuantificada

Con `macro_f1_std` mediano 0.0042 en las 76 filas que lo declaran, el error estandar de la media de 5
folds es **SE ~ 0.00188**. Con ese SE:

| Gate | z aproximado | Falsos positivos esperados en 151 corridas |
|---|---|---|
| Delta >= +0.0015 | z ~ 0.80 | **~32** |
| Delta >= +0.0030 | z ~ 1.60 | **~8** |

Un Bonferroni de familia 151 exigiria Delta >= **0.0147**, tres veces el gate vigente.

**Verificacion propia**: `grep -niE "multiplicidad|multiple comparison|FWER|family-wise|sesgo de
seleccion"` sobre `DECISIONS.md`, `PROJECT_STATE.md`, `FINAL_REPORT.md` e `INFORME_COMPLETO.md` devuelve
**cero resultados**. El unico `Bonferroni` del proyecto es el interno de V37 (18 pruebas por clase).

**Por que importa**: el argumento central del proyecto es "151 corridas, 39 mecanismos cerrados, ninguno
paso el gate". Ese argumento **no esta protegido contra multiplicidad**. El riesgo esta **ignorado, no
subestimado**.

**Lo que salva las conclusiones**: las cuatro adopciones reales (V19-B1, V20, V30, plug-in) tienen
**IC95 pareado excluyendo 0** y confirmacion externa (LB +0.00273 para el plug-in). El bootstrap pareado
corrige la varianza de estimacion; no corrige la seleccion del maximo sobre 151, pero las tres
adopciones que importan no dependen de esa correccion.

**No es un defecto de ejecucion sino de marco analitico.**

### A-02 (media) Seleccion de hiperparametros contra validacion con leakage de near-duplicates

`src/cv.py:11` documenta `scheme="stratified"` como *control para cuantificar el leakage*. Se usa como
selector en el camino al campeon:

| Script | linea | uso |
|---|---|---|
| `scripts/v29_04_optuna_individual.py` | 166 | inner split de Optuna |
| `scripts/v29_06_ensemble.py` | 300 | inner split de pesos |
| `scripts/v30_01_optuna_refinamiento.py` | 223 | inner split de Optuna |
| `scripts/v30_02_ensemble.py` | 95 | inner split de pesos |
| `scripts/v31_05_integracion.py` | 235 | inner split de integracion |
| `scripts/fase01_auditoria.py` | 298 | **uso legitimo: es el control** |

5 de 6 son selectores. Todos usan `groups=np.arange(len(entrenamiento))`, es decir grupo identidad,
que **no agrupa**.

**Medicion del auditor: NO cuantificada, sin impacto medible en overlap exacto.** Primer intento
invalido (uni folds de repeticiones distintas por ignorar `n_repeats=3` y dio 76 % de solapamiento,
imposible con ~15.000 filas duplicadas). Corregido con `train_idx` explicitos:

```
outer AGRUPADO (primario)   overlap texto exacto = 0,452 %
outer INGENUO (control)     overlap texto exacto = 1,087 %
inner INGENUO (el codigo)   overlap texto exacto = 0,963 %
inner AGRUPADO (loideal)    overlap texto exacto = 1,030 %
```

El agrupado tampoco da 0 % porque los clusters agrupan por Jaccard, no solo igualdad. **Conclusion
honesta: no hay sesgo medible en el solapamiento de texto exacto, y el impacto en la metrica de
seleccion no esta medido.** Queda abierto.

### A-03 (media) `max_iter=40` en el MLP, congelado 37 fases y miembro del campeon con peso 0.25

`scripts/phase2_ponderacion.py:116-118`: `MLPClassifier((64,), alpha=1e-3, batch_size=512,
learning_rate_init=1e-3, max_iter=40, early_stopping=True, n_iter_no_change=5)`.

Con early stopping activo y `n_iter_no_change=5`, 40 iteraciones es un techo de convergencia muy
agresivo. El modelo rinde 0.3381 (E-019a) / 0.4071 (E-019b), se incluye en V20 con peso 0.25
(`v20_f3c_candidato.py:135`) y nunca se re-parametrizo.

La via de revision existia: `v29_04_optuna_individual.py:130-141` abre espacio MLP con `batch_size`
hasta 4096 y `alpha` hasta 1e-1. **Pero el artefacto `experiments/v29_05_optuna_mlp.json` no existe**:
la revision se diseño y no se ejecuto.

### A-04 (media) `V32-E1`: la causa atribuida ("la perdida") es la presupuesto

El informe declara `NO_ATRIBUIBLE` **a la perdida surrogate**. Eso no se sostiene: A1 (perdida
surrogate) agoto las 60 epocas en **5/5 folds** (mejor epoca 54-59 de 60), o sea que nunca convergio,
mientras A2 (cross-entropy) si paro por early stopping (42-56 epocas). A1 vs A2 esta confundido con el
presupuesto, no con la funcion de perdida.

**El veredicto (no adoptar) si se sostiene**, por otra via: el MLP rindio 0.5989/0.6028 frente a
campeon 0.6160 y a LogReg V12 ~0.6005 en el mismo espacio de 279 features.

Presupuesto real de A1: 15 pasos/epoca x 60 = <=900 pasos, lote 8192. El limite no es memoria
(8192x279 float32 = 9,1 MB, fold residente ~175 MB contra 15,9 GB de VRAM) sino el numero de pasos.

### A-05 (media) El IC95 [0.635, 0.653] de D-087 no esta calculado en ningun artefacto

`scripts/v35_00g_techo_analisis.py` produce solo `techo_de_la_curva` (0.0567955), `objetivo_delta`
(0.0626) y `fraccion_de_errores_a_corregir` (0.1615). El IC95 y el `P(LB>=0.70) ~ 1%` salen de una
"tasa de captura del 4 %-9 %" aplicada a mano y publicados en tres documentos maestros.

Un IC95 sin modelo probabilistico detras es un intervalo **declarado, no estimado**.

**La conclusion "LB 0.70 no alcanzable" si se sostiene**, por el argumento del oraculo restringido
(+0.0568 < +0.0626 exigido). **El numero 0.642 y su IC no.**

### A-06 (baja) `V18-P3/P4/P5` no son reproducibles

`src/config.py:75-85` (`set_seed`) solo siembra `random` y `numpy`. Ninguno de los tres scripts llama
a `torch.manual_seed`, asi que la inicializacion de `nn.Linear` queda fuera de control. Contradice el
item de reproducibilidad de `experiments/auditorias.json`.

Mitigante: son cabezas lineales de una capa con resultado negativo en 5/3/3 brazos, robusto al ruido de
inicializacion.

### A-07 (baja) `E-013` reporta el `mejor_C` sobre los mismos folds que lo eligieron

`scripts/experimento_e013_barrido_c.py:103-114`: `best_c = max(C_VALUES, key=macro_f1_mean)`. El
numero publicado (0.5628 con C=0.2) es la media de esos mismos 5 folds, sin inner split. Sesgo al
alza pequeno (5 candidatos), pero es metrica de seleccion sin corregir.

## Lo que se declara SIN defecto

### Continuidad de la particion

Primitiva unica: `cv.make_folds(..., scheme="group")` con `dup_clusters.parquet`. Los 32+ scripts de
las familias no linguisticas usan `scheme="group", n_splits=5, seed=cfg.SEED`: **ninguno recrea splits
aleatorios**.

Las dos semillas alternativas (`SEMILLA_SEGUNDA = 987654` en `v30_05_segunda_particion.py:57` y
`age_plugin_confirmacion.py:123`) usan **los mismos grupos repartidos de otro modo**, o sea
reejecucion y no particion independiente. Esto esta **declarado explicitamente**
(`age_plugin_confirmacion.py:32-36`, `DECISIONS.md:2489`), y el script de confirmacion **no contiene
ningun optimizador** (`:14`: "solo aplica"). No se necesita control de multiplicidad por particion
porque la alternativa nunca se uso para elegir.

### Cadena de evidencia en las cuatro adopciones

V19-B1 (arbol pre-registrado + control barajado + 2a particion), V20 (4 cuartos con pesos
pre-registrados 0.25), V30 (Optuna con espacio local justificado + IC95 + gate) y plug-in
(pre-registro `bd08aba` con SHA256, congelado antes de la 2a particion, con prediccion falsable
"cero cambios en alta confianza" confirmada 2 veces + test). **Cero adopciones sin pre-registro.** El
plug-in tiene incluso commit verificable (`git cat-file -t bd08aba`, 2026-09-29 14:54) anterior al
resultado.

### Correspondencia protocolo-realidad en V32-E1

`verificar_constantes` compara 40 campos del pre-registro contra el codigo y los 7 SHA256 de entradas
congeladas: todo `True`. Lote, LR, WD, epocas, paciencia, fraccion interna, clip, eps, semilla, hilos,
bootstrap, gates, pesos del plug-in y compliance son identicos.

### Sin fuga en ninguna red ni meta-modelo

Escalador ajustado solo en fold-train; combinador anidado ajustado en `fold != k` sobre probabilidades
OOF; `E-020` sobre OOF real de E-019b; `V18-P5` con cross-fitting explicito del teacher. Coherente con
`experiments/auditorias.json` -> `meta_entrenado_con_oof: true`.

### Politica de early stopping

Registrada con fecha y autor en V32-E1: "Carlos: **early stopping estricto** (opcion A) y entrenamiento
en GPU" (`v32_00_preregistro.py:21`), mas `compliance.es_estricto: true`. V18 declara "epocas fijas sin
early stopping" dentro del pre-registro (declaracion, no consulta fechada). `E-019` usa el ES implicito
de sklearn sin declararlo.

### Ausencia de scheduler y AMP en V32-E1

Ambas ausencias estan justificadas por escrito y son coherentes con un run de minutos que congela la
numerica (sonda de determinismo bit a bit: `max|Δp| = 0.0`). **Valido para la hipotesis, no demostrado
optimo.**

## Reservas (valido, no demostrado optimo)

- **`HIERBA` deliberadamente sub-parametrizado** (8 hojas, `max_depth=3`, `min_child=200`): el proyecto
  ya lo sabe y lo mide (D-065 registra que los mejores trials de V29 usan 43-87 hojas). Es un baseline
  conservador elegido como punto de comparacion, y la revision de capacidad se hizo (V29->V30, veredicto
  `EQUIVALE_A_V29`). No es un defecto desconocido.
- **Relajacion de D-042** (Delta +0.003/4-de-5 -> +0.001/3-de-5) se apoyo en ruido medido (+-0.0003 en
  controles permutados). Base legitima pero multiplicidad-inconsistente. **Ninguna familia la uso para
  adoptar**, asi que el dano es potencial, no realizado.
- **V29 Optuna con 10 trials/fold** en vez de 30-50: desviacion de presupuesto declarada en D-064.
- **Pesos del plug-in** derivados de media aritmetica de 5 LOFO y aplicados globalmente: sobreajuste por
  diseno, **declarado** en D-073, con el numero honesto (nested, +0.0020172) identificado y LB externo
  (+0.00273) como confirmacion independiente.
- **Conflicto de drivers en D-066**: dos procesos de Optuna escribiendo el mismo artefacto. Declarado en
  D-064 y resuelto (sobrevivio el de 30 trials). Defecto operativo real pero bien gestionado.

## Hallazgo tecnico reutilizable del scope NN

Un bloque de 6 probabilidades **sin informacion** (A3, etiquetas barajadas) ya cuesta **-0.00090**,
casi igual que un modelo legitimo (A2, CE, **-0.00093**). A1 da **-0.00342**, o sea 0.0025 peor que
ruido, con 13.151 filas volteadas.

Es decir: el **sobre-peso del integrador** es un coste mecanico real y medido. Un gate que compara
candidatos contra el campeon sin un suelo de este tipo puede rechazar por el integrador, no por el
candidato.

## Veredicto global de Fase 2b

Las familias **V18-V23** y **V32-V37** son el estandar del proyecto: pre-registros con SHA256, controles
barajados, gemelos con presupuesto igual, sondas bit-a-bit, y auto-falsaciones que se disparan solas
(D-073 paro la submission por confundir tres numeros; D-066 detecto que la cifra publicada no era
reproducible y el LB resolvio cual era).

Las tres familias con reservas son **V1-V4** (parametros heredados congelados sin revision),
**V24-V29** (inner split de Optuna sin agrupar) y **V30-V31** (seleccion post-hoc sobre 2 candidatos y
pesos no persistidos, ambos autodetectados).

La reserva **transversal y principal** es la multiplicidad (**A-01**): no es un defecto de ejecucion sino
de marco analitico, y es el unico punto donde el proyecto afirma mas de lo que su diseno sostiene.
