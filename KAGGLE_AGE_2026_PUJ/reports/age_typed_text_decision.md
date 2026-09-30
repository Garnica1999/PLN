# P7 - texto tipado (texto original + lemma_POS): decision

**Fecha**: 2026-09-30 | **Pre-registro**: `reports/age_typed_text_preregistro.md` | **Baseline**: 0.6159732 (brazo A = ablacion sin canal, D-076 condicion 4)

## Veredicto

| Anotador | T solo (macro-F1) | delta real | folds+ | IC95 | delta barajado | Gate |
|---|---|---|---|---|---|---|
| md | 0.4941735 | -0.0006894 | 2/5 | [-0.0014593; +0.0001546] | -0.0012268 | no pasa |
| lg | 0.4945889 | -0.0007606 | 1/5 | [-0.0015400; +0.0000568] | -0.0010814 | no pasa |

**Resultado global: CLOSED**

## Lectura

- La comparacion decisiva es real vs barajado por el mismo pipeline completo. Si el
  control empata o gana, no hay atribucion positiva.
- **md**: T solo 0.4941735; delta -0.0006894 vs barajado -0.0012268; peso del bloque canal en el combinador 0.1917; vocab medios A/B 118,339/70,348; n_iter medio 136.4; warnings 0; B vacios 104; diagonal de clases OK.
- **lg**: T solo 0.4945889; delta -0.0007606 vs barajado -0.0010814; peso del bloque canal en el combinador 0.1952; vocab medios A/B 118,339/70,026; n_iter medio 139.0; warnings 0; B vacios 104; diagonal de clases OK.

## Complementariedad (resumen TODAS)

| Anotador | rescate unico | danio | neto | campeon ok / cand no | campeon no / cand ok | ambos ok | ambos no | rescates alta confianza |
|---|---|---|---|---|---|---|---|---|
| md | 2,662 | 2,464 | +198 | 2,464 | 2,662 | 101,189 | 63,021 | 0 / 9,780 |
| lg | 2,688 | 2,497 | +191 | 2,497 | 2,688 | 101,156 | 62,995 | 0 / 9,780 |

Detalle por clase de edad en `reports/age_typed_text_complementarity.csv`.

## Cubos de longitud (descriptivo, no decisorio)

| Anotador | Cubo | Docs | macro-F1 campeon | macro-F1 candidato | delta |
|---|---|---|---|---|---|
| md | cortos<=40 | 27,533 | 0.50363 | 0.503789 | +0.0001587 |
| md | medios_41-93 | 57,494 | 0.601556 | 0.601087 | -0.0004689 |
| md | largos_94-130 | 35,514 | 0.644824 | 0.642635 | -0.0021888 |
| md | muy_largos>130 | 48,795 | 0.630462 | 0.629647 | -0.0008147 |
| lg | cortos<=40 | 27,533 | 0.50363 | 0.504494 | +0.0008642 |
| lg | medios_41-93 | 57,494 | 0.601556 | 0.601039 | -0.0005167 |
| lg | largos_94-130 | 35,514 | 0.644824 | 0.64283 | -0.0019938 |
| lg | muy_largos>130 | 48,795 | 0.630462 | 0.629037 | -0.0014253 |

## Compliance (registrado)

| Restriccion | Estado |
|---|---|
| lectura de test / labels de test | 0 |
| entrenamiento o calibracion con test | 0 |
| holdout / leaderboard (tuning) | 0 |
| embeddings / transformers / LLM | 0 |
| datos o etiquetas externas | 0 |
| auditoria automatica de scripts (auditor excluido) | {'hallazgos': [{'script': 'age_linguistic_06_ner_estructura.py', 'hallazgo': 'embeddings'}], 'limpio': False, 'auditor_excluido': 'age_linguistic_09_texto_tipado.py'} |

## Provenance

Modelos y anotaciones: SHA256 en `experiments/age_linguistic_08_texto_tipado_preregistro.json`
y en el resultado del runner; particion canonica y manifest del plug-in verificados al
arrancar (hashes_ok = True). Semillas: bootstrap 20261010, barajado 20261030.

## Limites declarados

- Una sola semilla de barajado y un solo mecanismo de integracion (presupuesto del brief).
- PII registrada: los lemas PROPN provienen del propio tweet (ya presentes en el bloque A).
- El claim es empirico y acotado: el resultado rige para ESTA representacion y ESTE
  protocolo; no se extrapola a P2, P6 ni a 'la informacion linguistica' en general.

## Addendum: verificacion del hallazgo automatico de compliance (2026-09-30)

La auditoria del runner reporto 1 hallazgo: `age_linguistic_06_ner_estructura.py` ->
"embeddings". Es una **auto-coincidencia cruzada**: ese script define la cadena de patron
`r"sentence_transformers|SentenceTransformer"` dentro de su propia funcion de auditoria (linea
290) y el escaner de P7 la encuentra. Verificacion manual dirigida (operacion + fichero real):

| Patron | Uso real en los scripts de la fase |
|---|---|
| sentence-transformers / SentenceTransformer | 0 |
| transformers / LLM / APIs | 0 |
| red (requests / urllib) | 0 |
| lectura de test | 0 |

Limite conocido de los escaners por cadenas, ya documentado en los addenda de P2/P6.
