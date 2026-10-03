# Ciclo final - familias restantes (P3/P4/P5): decision

**Fecha**: 2026-09-30 | **Pre-registro**: `reports/age_remaining_channels_preregistro.md` | **Baseline**: 0.6159732 (brazo A = ablacion sin canal, D-076 cond.4)

## Veredicto por familia

| Familia | Anotador | canal solo | delta real | folds+ | IC95 | delta barajado | Gate |
|---|---|---|---|---|---|---|---|
| R3_lemma_POS | md | 0.4558531 | -0.0009172 | 2/5 | [-0.0017130; -0.0000503] | -0.0011973 | no pasa |
| R3_lemma_POS | lg | 0.4562059 | -0.0009230 | 1/5 | [-0.0017484; -0.0000837] | -0.0011973 | no pasa |
| R4_morfologia | md | 0.2803143 | -0.0011668 | 1/5 | [-0.0019643; -0.0003826] | -0.0011917 | no pasa |
| R4_morfologia | lg | 0.2763103 | -0.0010333 | 2/5 | [-0.0018078; -0.0002723] | -0.0011973 | no pasa |
| R5_dependencias | md | 0.2668566 | -0.0009132 | 1/5 | [-0.0016948; -0.0001418] | -0.0011973 | no pasa |
| R5_dependencias | lg | 0.2677221 | -0.0005440 | 2/5 | [-0.0013490; +0.0002280] | -0.0011973 | no pasa |

**Resultado global: NINGUN BRAZO PASA**

## Lectura

- La comparacion decisiva es real vs barajado por el mismo pipeline completo.
- **R3_lemma_POS/md**: delta -0.0009172 vs barajado -0.0011973; peso del bloque canal 0.1889; vocab medio 70,348; vacios 104; n_iter medio 100.2; warnings 0.
- **R3_lemma_POS/lg**: delta -0.0009230 vs barajado -0.0011973; peso del bloque canal 0.1956; vocab medio 70,026; vacios 104; n_iter medio 116.8; warnings 0.
- **R4_morfologia/md**: delta -0.0011668 vs barajado -0.0011917; peso del bloque canal 0.0838; vocab medio 8,024; vacios 4343; n_iter medio 66.8; warnings 0.
- **R4_morfologia/lg**: delta -0.0010333 vs barajado -0.0011973; peso del bloque canal 0.1056; vocab medio 8,151; vacios 4901; n_iter medio 74.4; warnings 0.
- **R5_dependencias/md**: delta -0.0009132 vs barajado -0.0011973; peso del bloque canal 0.0903; vocab medio 14,154; vacios 104; n_iter medio 65.8; warnings 0.
- **R5_dependencias/lg**: delta -0.0005440 vs barajado -0.0011973; peso del bloque canal 0.1154; vocab medio 14,562; vacios 104; n_iter medio 93.6; warnings 0.

## Serie completa de canales linguisticos estructurados (referencia md)

| Fase | Canal solo | delta (md) | Veredicto |
|---|---|---|---|
| P2 POS-secuencia | 0.2750921 | -0.0086357 | NEGATIVO |
| P6 NER estructural | 0.1833881 | -0.0009257 | CLOSED |
| P7 texto tipado | 0.4941735 | -0.0006894 | CLOSED |
| R3_lemma_POS (lemma_POS) | 0.4558531 | -0.0009172 | CLOSED |
| R4_morfologia (morfologia) | 0.2803143 | -0.0011668 | CLOSED |
| R5_dependencias (dependencias) | 0.2668566 | -0.0009132 | CLOSED |

## Complementariedad (resumen TODAS)

| Familia | Anotador | rescate | danio | neto | alta confianza |
|---|---|---|---|---|---|
| R3_lemma_POS | md | 2,730 | 2,555 | +175 | 0 / 9,780 |
| R3_lemma_POS | lg | 2,777 | 2,607 | +170 | 0 / 9,780 |
| R4_morfologia | md | 2,376 | 2,245 | +131 | 0 / 9,780 |
| R4_morfologia | lg | 2,399 | 2,245 | +154 | 0 / 9,780 |
| R5_dependencias | md | 2,390 | 2,219 | +171 | 0 / 9,780 |
| R5_dependencias | lg | 2,470 | 2,232 | +238 | 0 / 9,780 |

## Compliance (registrado)

- Auditoria por operacion: {'reales': [], 'auto_coincidencias_patron': [{'script': 'age_linguistic_06_ner_estructura.py', 'linea': 290, 'hallazgo': 'embeddings'}, {'script': 'age_linguistic_09_texto_tipado.py', 'linea': 250, 'hallazgo': 'embeddings'}], 'limpio': True, 'auditor_excluido': 'age_linguistic_12_remaining_channels.py'}
- test = 0; holdout = 0; leaderboard = 0; submissions = 0; modelos nuevos = 0; licencias nuevas = 0; descargas = 0.

## Provenance

Hashes de anotacion, dep_edges y particion canonica verificados al arrancar
(`hashes_ok`); validacion del artefacto derivado re-hecha en el runner (0 discrepancias
exigidas). Semillas: bootstrap 20261010, barajado intra-fold 20261031.

## Limites declarados

- Control intra-fold (como P7): sus deltas barajados no son comparables numericamente
  con los globales de P2/P6.
- R5 pierde el POS del dependiente y sus bigramas son adyacencia lineal, no caminos del
  arbol (decision fija pre-registrada).
- El claim es empirico y acotado: 'el canal X no aporta mejora al campeon bajo el
  mecanismo evaluado'; no se extrapola a toda la informacion linguistica.
