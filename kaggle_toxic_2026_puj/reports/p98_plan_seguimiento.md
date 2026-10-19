# P98 - Plan de seguimiento (documento vivo)

Actualizado al cierre de la Fase A. Se regenera al cerrar cada etapa.

## Fase A - gobernanza e infraestructura (COMPLETA)

| Paso | Estado | Artefacto |
|---|---|---|
| Nota de provenance de la autorizacion | HECHO | `docs/p98_provenance_autorizacion_bilstm.md` |
| Registro de fuentes y licencias (17 filas) | HECHO | `reports/p98_corpus_registry.csv`, `reports/p98_corpus_licenses.md` |
| Manifiesto operativo de corpus | HECHO | `artifacts/p98_corpus_manifest.json` |
| Auditoria de hashes de los 6 artefactos | HECHO | via `experiments/p98_00_registro_fuentes.py` |
| Auditoria de solapamiento vs train | HECHO: CUMPLE | `reports/p98_auditoria_solapamiento.json/.md` |
| Infraestructura sintetica (ROCm, checkpoint, determinismo) | HECHA | `reports/p98_infra_test.json/.md`; `models/p98_infra_test.pt` |
| Arquitectura v1 pre-registrada | HECHA | `reports/p98_architecture.md` |

Datos clave de Fase A: ROCm estable y **determinismo exacto** (delta 0.00e+00);
223k tokens/s en caliente (50.8k en la primera corrida, cache de kernels) en la config sintetica (50x mas pequena que la objetivo);
checkpoint roundtrip identico; auditoria de solapamiento con 0 coincidencias
exactas y 1 coincidencia de 8-gramas que resulto ser puntuacion.

## Fase B - experimentos (pendiente; presupuesto <=6 GPU-h por etapa, <=24 total)

| Id | Etapa | Estado | Tope | Gate |
|---|---|---|---|---|
| C98-0 | Piloto real (wikipedia 5M) | HECHO: PILOTO_OK (val ppl 651->351; ~0.16 GPU-h) | 2 GPU-h | Escalera fija: [1024,2] a 17.133 tok/s |
| C98-1 | Preentrenamiento LTR+RTL | HECHO: LTR ppl 94.8 / RTL 91.7 (4+4 epocas; 3.775 GPU-h) | 6 GPU-h | Representaciones extraibles |
| D98-1 | Frozen reps + LR por pliegue; embedding-only y fusion | HECHO: fusion 0.782266 (delta +0.005055, 4/5, IC incluye cero; neto +11) -> MARGINAL_SENAL | ~0 GPU-h | Replicar particion (D98-1b) antes de cualquier declaracion |
| C98-2 | Escala ~100M | CERRADO: falsificacion en D98-1b | 6 GPU-h | La senal no sobrevivio a la particion; no se escala |
| Libros | Variante independiente <=5M tokens | CONDICIONAL | 2 GPU-h | Solo si C98-0/D98-1 muestran senal |
| D98-2 | Cabeza secuencial | NO PROCEDE: D98-1b falsificada (2/5 pliegues) | 2 GPU-h | Criterios del propietario: cierre sin escalar |
| D98-3 | Fine-tuning por pliegue | CONDICIONAL | 2 GPU-h | Evidencia que justifique el coste |
| Replica | Segunda particion agrupada | HECHA: D98-1b FALSIFICADA (delta +0.001151, 2/5, IC incluye 0) | 2 GPU-h | Linea contextual actual CERRADA con evidencia |

## Cuarentenas y pendientes de gobierno

- C98-4 (corpus de odio): CUARENTENA ACTIVA (decision 2026-10-13). Solo se
  documentara la justificacion si C98-1/D98-1 muestran senal complementaria.
- OSCAR/CulturaX/HPLT: registradas sin descarga; familia Common Crawl unica.
- TASS y sentimiento/emocion: NEEDS_EXPLICIT_PERMISSION (no bloquean el plan).

## Invariantes activos

V03 congelado; toxic_test, holdouts y leaderboard sin uso; sin submissions; sin
commits; ASCII; un solo script por experimento; controles registrados.
