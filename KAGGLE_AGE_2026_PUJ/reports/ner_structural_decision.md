# P6 - NER estructural: decision

**Fecha**: 2026-09-29 | **Pre-registro**: `reports/ner_structural_preregistro.md` | **Baseline**: 0.6159732 (regla de la submission, recomputado)

## Veredicto

| Representacion | Anotador | delta (real) | folds+ | IC95 | delta (barajado) | Gate |
|---|---|---|---|---|---|---|
| N1 | lg | -0.0010228 | 2/5 | [-0.0018194; -0.0002481] | -0.0011874 | no pasa |
| N1 | md | -0.0009257 | 1/5 | [-0.0017437; -0.0001536] | -0.0011904 | no pasa |
| N1b | lg | -0.0013202 | 0/5 | [-0.0021230; -0.0005463] | -0.0011727 | no pasa |
| N1b | md | -0.0010038 | 2/5 | [-0.0018144; -0.0002271] | -0.0012511 | no pasa |
| N2 | lg | -0.0010122 | 1/5 | [-0.0017945; -0.0002442] | -0.0011698 | no pasa |
| N2 | md | -0.0012316 | 1/5 | [-0.0019965; -0.0004430] | -0.0010243 | no pasa |

**Resultado global**: NINGUN BRAZO PASA

## Lectura

- La comparacion decisiva es real vs barajado (mismo pipeline completo): si el barajado rinde igual, lo que aporta el bloque es capacidad o ruido, no informacion de NER.
- **N1/lg**: canal solo 0.1878; delta -0.0010228 vs barajado -0.0011874; ratio de peso del bloque canal en el combinador 0.0689 (cerca de 0 = recalibracion de V30).
- **N1/md**: canal solo 0.1834; delta -0.0009257 vs barajado -0.0011904; ratio de peso del bloque canal en el combinador 0.0662 (cerca de 0 = recalibracion de V30).
- **N1b/lg**: canal solo 0.1522; delta -0.0013202 vs barajado -0.0011727; ratio de peso del bloque canal en el combinador 0.1615 (cerca de 0 = recalibracion de V30).
- **N1b/md**: canal solo 0.1638; delta -0.0010038 vs barajado -0.0012511; ratio de peso del bloque canal en el combinador 0.0699 (cerca de 0 = recalibracion de V30).
- **N2/lg**: canal solo 0.2310; delta -0.0010122 vs barajado -0.0011698; ratio de peso del bloque canal en el combinador 0.1466 (cerca de 0 = recalibracion de V30).
- **N2/md**: canal solo 0.2278; delta -0.0012316 vs barajado -0.0010243; ratio de peso del bloque canal en el combinador 0.1314 (cerca de 0 = recalibracion de V30).

## Atribucion por representacion

- N1b aisla la parte lexica (sin marcas de Twitter): comparar N1 vs N1b separa 'estructura de entidades lexicas' de 'proxy de menciones/RT/hashtags/URLs' (canales V31 cerrados).

## Compliance (registrado, no declarado)

| Restriccion | Estado |
|---|---|
| lectura de test / labels de test | 0 |
| entrenamiento o calibracion con test | 0 |
| holdout | 0 |
| leaderboard (tuning) | 0 |
| embeddings / transformers / LLM | 0 |
| datos o etiquetas externas | 0 |
| auditoria automatica de scripts de la fase | {'hallazgos': [{'script': 'age_linguistic_06_ner_estructura.py', 'hallazgo': 'lectura de test'}, {'script': 'age_linguistic_06_ner_estructura.py', 'hallazgo': 'embeddings'}, {'script': 'age_linguistic_06_ner_estructura.py', 'hallazgo': 'LLM'}], 'limpio': False} |

Nota: las etiquetas HS/TR/AG que menciona la plantilla del brief no aplican a AGE (son de la competencia de toxicidad); el desglose se hace por las 6 clases de edad.

## Limites declarados

- Una sola semilla de barajado (presupuesto del ciclo); el revisor ML recomendo varias y queda como recomendacion no ejecutada.
- La anotacion NER sobredetecta (23,7% de tokens; MISC 57%); el claim es 'senal empirica en anotacion ruidosa', no semantica.
- Los pesos del plug-in se aplican a p_combinada (suposicion de escala pre-registrada).
- Acuerdo md vs lg en la secuencia de entidades: 28,91% (limite de fiabilidad).

## Addendum: verificacion de los hallazgos automaticos de compliance (2026-09-30)

La auditoria automatica del runner reporto 3 hallazgos, TODOS contra
`scripts/age_linguistic_06_ner_estructura.py` y todos son **AUTO-COINCIDENCIAS**: el escaner
busca las cadenas de patron ("age_test", "sentence_transformers", "openai...") y esas cadenas
estan ESCRITAS en su propio codigo (la definicion de los patrones, lineas 288-291).
Verificacion manual dirigida del uso REAL en todos los scripts de la fase (grep sobre
operaciones, no sobre palabras):

| Patron | Uso real |
|---|---|
| lectura de test / labels de test | 0 |
| embeddings / sentence-transformers | 0 |
| transformers / LLM / APIs | 0 |
| red (requests / urllib) | 0 |

Excluyendo el propio escaner, la auditoria da limpio. Limite conocido y registrado: el escaner
no se autoexcluye (leccion ya documentada en el ciclo P0/P1; el proximo escaner lo corrige).
