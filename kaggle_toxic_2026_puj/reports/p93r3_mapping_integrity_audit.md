# P93-R3 - Auditoria de integridad del mapeo (176 celdas)

Fecha: 2026-10-13. Recomputation independiente desde los archivos crudos.

## 1. Recomputacion de las celdas candidatas

- Definicion aplicada: celdas (tweet, etiqueta) donde A y B coinciden en una
  respuesta binaria y esta difiere del gold interno (clave P92-W).
- Celdas recomputadas: 176; recuentos por etiqueta: HS=23, TR=87, AG=66.
- Coincidencia EXACTA con el archivo de candidatos de P93: True (conjunto por review_id+etiqueta).
- Union por identificador (review_id/adjudication_review_id), nunca por
  posicion de fila; biyeccion P93R <-> review_id verificada (127).
- Sin intercambio de etiquetas: los conjuntos por HS/TR/AG coinciden en
  composicion y recuentos con el archivo original.

## 2. Reproduccion de los recuentos de P93-R2

- C1 sobre las 176: coincide=174, UTD=2, difiere=0 (esperado 174/2/0).
- C2 sobre las 176: coincide=159, UTD=8, difiere=9 (esperado 159/8/9).
- Reproducidos con codigo independiente; el resultado de C1 es inusual
  pero NO es un error de mapeo.

## 3. Firma de copia (candidatas vs no candidatas)

- C1: {'cand_utd': 2, 'no_cand_coincide': 109, 'cand_coincide': 174}
- C2: {'cand_coincide': 159, 'no_cand_coincide': 96, 'cand_utd': 8, 'no_cand_utd': 8, 'cand_difiere': 9, 'no_cand_difiere': 5}
- Interpretacion: si C1/C2 hubieran copiado A/B, la coincidencia seria
  perfecta en TODAS las celdas A==B (candidatas y no candidatas). Las tasas
  observadas se comparan aqui y en el informe de cegamiento; ver dicts.
- C1 escribio 49 notas propias y C2 32 (con 6 tweet_note), con referencias a
  la guia v2.1; C1 dejo 2 UTD y C2 difiere en 9 celdas candidatas: evidencia
  de juicio independiente, no de copia mecanica.

## 4. Conclusion

- Mapeo: INTEGRO (sin errores de union, orden o etiqueta).
- El valor llamativo de C1 (174/176) queda explicado como convergencia
  humana con la evidencia disponible; no se usa como prueba de correccion.
