# P98-02: auditoria de solapamiento corpus vs train

| Corpus | Documentos | Exactas | 8-gramas | Tokens 8-g |
|---|---|---|---|---|
| wikipedia_10M | 2847 | 0 | 0 | 1500000 |
| wikipedia_50M | 21529 | 0 | n/a (solo documento) | n/a |
| mc4_10M | 16738 | 0 | 0 | 1500000 |
| twitter | 361994 | 0 | 1 | 10304423 |

## Diagnostico del par detectado

- fila train 1375 vs documento corpus 721: 8-grama compartido (puntuacion), Jaccard de tokens 0.038462 (1 tokens comunes de 33 en el documento).

Sin accion: la coincidencia es de puntuacion (sin contenido compartido).

**Veredicto: CUMPLE** (0 coincidencias exactas; coincidencias de 8-gramas dentro del maximo near-dup ya adjudicado en P73-01 y por debajo del umbral 0.2).

auditorias previas citadas: v5 (fuga 8-gramas wikipedia: 0), P73-01 (twitter vs desarrollo: 0 exacto/normalizado; 76 filas con candidato near-dup; max Jaccard 0.142857 < 0.2). El caso detectado aqui se adjudica contra ese maximo documentado. Frente a toxic_test no se escanea (regla vigente): incertidumbre residual documentada.
