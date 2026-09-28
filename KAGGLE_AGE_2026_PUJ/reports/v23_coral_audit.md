# V23 · Auditoria de la fase CORAL

**Estado: PASS**

## 1. Ficheros fuente

| Fichero | Papel |
|---|---|
| `src/ordinal_coral.py` | implementacion cumulative-link, numpy y scipy |
| `scripts/v23_00_tests_coral.py` | TEST A, B, C y D |
| `scripts/v23_01_coral.py` | experimento V23-01 con sus dos controles |
| `scripts/v23_02_auditoria.py` | este script |
| `notebooks/99_entrega_consolidado.ipynb` | entrega, seccion 31 anadida |

## 2. Scripts comprobados

| Script | `py_compile` | import | ejecucion | artefactos | serializacion |
|---|---|---|---|---|---|
| `src/ordinal_coral.py` | OK | OK | OK | OK | OK |
| `scripts/v23_00_tests_coral.py` | OK | OK | OK | OK | OK |
| `scripts/v23_01_coral.py` | OK | OK | OK | OK | OK |
| `scripts/v23_02_auditoria.py` | OK | OK | OK | OK | OK |

El criterio del brief es que **compilar no basta**: los tres scripts productivos se ejecutaron de verdad y sus artefactos estan en disco.

## 3. Tests unitarios

- Los cuatro tests: **pasan**.

- **TEST A** sintetico perfectamente separable: PASA — macro-F1 >= 0.80, seis clases presentes y violaciones = 0
- **TEST B** senal compartida a la escala de AGE: PASA — converge, supera el prior y no satura (fraccion_saturada = 0)
- **TEST C** orden de los umbrales durante y despues: PASA — cero incumplimientos en los 60 pasos y tras el ajuste
- **TEST D** gradiente analitico contra diferencias finitas: PASA — error relativo < 1e-6 y diferencia de perdida < 1e-9

## 4. Comprobaciones de sintaxis

- `py_compile` sobre los tres scripts y el modulo: **OK**.
- `import` de los tres scripts: **OK**.
- Ejecucion real completa de los tres: **OK**.

## 5. Auditoria de caracteres

- Ficheros analizados: **7**.
- Hallazgos: **38**.

- Hallazgos en el **codigo y metadata de V23**: **0**.
- Hallazgos en el **notebook consolidado**: **38**, todos **preexistentes**: hay 38 en el notebook tambien antes de esta fase y **V23 no anade ninguno**.

Se comprueban: alfabetos no latinos, caracteres de control, ancho cero, BOM, comillas tipograficas, guiones Unicode y espacios Unicode. **Ninguno aparece en el codigo ni en la metadata de V23.**

### Deuda Unicode preexistente, documentada y no tocada

El notebook consolidado arrastra caracteres que ninguna sesion de V23 ha escrito. El proyecto excluye `notebooks/` de su auditoria de idioma historica (`EXCLUIR` en `scripts/v13_p0a_auditoria_idioma.py`), luego nunca se detectaron. Se dejan como estan porque reescribir contenido de sesiones anteriores no pertenece a esta fase. Se listan por nombre Unicode y no de forma literal, para que este script no contenga lo que reporta.

| Codepoint | Nombre Unicode | Veces | Aviso en la auditoria historica |
|---|---|---|---|
| U+2212 | MINUS SIGN | 15 | no |
| U+2192 | RIGHTWARDS ARROW | 8 | si |
| U+00A7 | SECTION SIGN | 8 | no |
| U+222A | UNION | 3 | no |
| U+2194 | LEFT RIGHT ARROW | 3 | si |
| U+00B2 | SUPERSCRIPT TWO | 1 | no |

Los puntos del rango de flechas son **avisos** y no errores en la auditoria historica del proyecto: estan en `RANGOS_AVISO`. Los demas estan fuera de la lista de permitidos y si serian errores aqui.


## 6. Auditoria del notebook

- Celdas: **103** | nbformat **4.5**.
- Identificadores unicos: **True** (repetidos: ninguno).
- Titulos de seccion unicos: **True** (repetidos: ninguno).
- Secciones de titulo: **32**.
- Artefactos referenciados: **40**, faltantes: **0** .
- Figuras referenciadas: **0**, faltantes: **0** .

### Deuda documentada, no resuelta

El notebook consolidado llega hasta la seccion 30, que cierra V17. **No contiene V18, V19, V20, V21 ni V22**, y el champion actual es V20. El brief pide integrar V23 y no pide rellenar las anteriores, asi que se deja constancia en vez de escribir contenido que esta sesion no ha verificado.

## 7. Coherencia numerica

| Comprobacion | Resultado |
|---|---|
| control a reproduce | OK |
| control a igual a historico | OK |
| violaciones v20 igual a D057 | OK |
| coral sin violaciones | OK |
| celdas suman el total | OK |
| neto igual a delta de aciertos | OK |
| csv igual a json | OK |
| tests pasan | OK |
| sin cifras manuales | OK |

Ninguna cifra del notebook se escribe a mano: todas se cargan de `experiments/v23_00_tests.json` y `experiments/v23_01_coral.json`.

## 8. Hashes de artefactos

| Artefacto | SHA256 |
|---|---|
| `experiments/v23_00_tests.json` | `4065c1f046693635171e581e4fcabfec` |
| `experiments/v23_01_coral.json` | `abea3b75f0efd83f1f644dc2d9c4a430` |
| `reports/v23_00_tests_coral.md` | `359363218fda45075440c70c882b86d4` |
| `reports/v23_coral_analysis.md` | `5e04ee38246c565ef83749cf4dc2b9f5` |
| `reports/v23_coral_results.csv` | `5e421fa3cf5cf45c4d08415aba9bfaa6` |
| `features/v23_coral_cdf_oof.npy` | `31f62e443d7889f457ea168dadd90586` |
| `features/v23_binarios_lineales_oof.npy` | `86fd9ec415e6a57046b27fc656fb17b7` |
| `reports/figures/v23_coral_monotonicidad.png` | `a0e5fcbb0942bed5ae662062ea9ed561` |
| `reports/figures/v23_coral_complementariedad.png` | `015dfcd0b65aabe373c29c7c8f4b90a5` |
| `reports/figures/v23_coral_folds.png` | `96dc87727328640da7bd7d7ad1872356` |

Artefactos ausentes: **0** 

## 9. Estado final

- Sintaxis: **PASS**
- Import: **PASS**
- Ejecucion: **PASS**
- Artefactos: **PASS**
- Serializacion: **PASS**
- Unicode de V23: **PASS** (0 hallazgos; la deuda preexistente del notebook se documenta aparte y no se cuenta)
- Notebook: **PASS**
- Coherencia numerica: **PASS**
- Ejecucion real de la seccion: **16 celdas, 0 errores**
- V20 intacto: **PASS** (control A reproduce, no se toco su artefacto)
- Test intacto: **PASS** (nunca se lee ni se escribe)
- Holdout intacto: **PASS** (nunca se usa en esta fase)

**CORAL = CLOSED**, sin submission.
