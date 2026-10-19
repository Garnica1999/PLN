# P93-R5 - Comparacion con el gold y registro de correcciones candidatas

Fecha: 2026-10-13. Comparacion AUTORIZADA explicitamente por el propietario;
sin tercera revision humana. Unidad: celda (tweet, etiqueta). V03 y gold intactos.

## 1. Auditoria de entrada

- Celdas candidatas: 176 (HS 23, TR 87, AG 66); tweets unicos: 127.
- Decisiones humanas finales bloqueadas: 171 celdas con acuerdo binario C1/C2; 5 casos sin consenso (cola de pendientes).
- Version de guia en todas las decisiones: v2.1 (archivo distribuido: fd8d4c65ccd194fb).
- Hashes de ronda 1 y ronda 2 intactos; manifiesto P93-R4 consistente
  (SECOND_ROUND_RESPONSES_VALIDATED / FINAL_HUMAN_DECISIONS_LOCKED).
- Correspondencia por ID y etiqueta verificada en: decisiones, candidatos,
  asignacion P93S1/S93S2, clave interna y grid de 750 celdas (sin uniones por posicion).

## 2. Elegibilidad de la poblacion

- Las 265 ids del muestreo v2.1 y los 127 tweets de las 176 celdas
  pertenecen a train (particion dev); el id 22621 aparece en train con comillas
  malformadas y solo se usa normalizado para el emparejamiento.
- Interseccion con holdout historico (particion 'holdout'): 0; con holdout2: 0;
  con test: 0. No se leyeron contenido, labels ni scores de particiones reservadas
  (solo columnas de id).
- Canonicos verificados: train 3d40e7345b588860, OOF 559265816ce4f631,
  submission 7d7478f271e6edfd, particion e5b62e5f22c4c63a, modelo 263ed5bd0fc0de7c.

## 3. Comparacion de las 171 decisiones con el gold

| estado | celdas | HS | TR | AG |
|---|---|---|---|---|
| HUMAN_FINAL_MATCHES_GOLD | 0 | 0 | 0 | 0 |
| POSSIBLE_LABEL_ISSUE | 14 | 2 | 4 | 8 |
| HUMAN_FINAL_DIFFERS_FROM_GOLD | 157 | 21 | 81 | 55 |
| CONFIRMED_ANNOTATION_ERROR | 0 | 0 | 0 | 0 |

- Coincidencias con el gold: 0 de 171.
- Discrepancias adjudicadas: 171 de 171 (todas registradas; ninguna confirmada como error).
- Con justificacion documentada (POSSIBLE_LABEL_ISSUE): 14.
- Sin justificacion documentada (HUMAN_FINAL_DIFFERS_FROM_GOLD): 157.
- CONFIRMED_ANNOTATION_ERROR: 0. Para confirmar se requiere justificacion
  documentada y evidencia que excluya toda interpretacion alternativa razonable;
  esta fase no realiza esa evaluacion adversarial caso a caso.

Regla operativa aplicada: coincidencia C1/C2 (o con A/B) NO se considera
justificacion suficiente; solo el rationale final referido a la politica v2.1 cuenta.

## 4. Los cinco casos pendientes

| caso | etiqueta | C1 final | C2 final | estado |
|---|---|---|---|---|
| P92W-0035 | AG | NEGATIVE | UNABLE_TO_DETERMINE | UNRESOLVED_NO_THIRD_REVIEW |
| P92W-0061 | AG | NEGATIVE | UNABLE_TO_DETERMINE | UNRESOLVED_NO_THIRD_REVIEW |
| P92W-0094 | TR | POSITIVE | UNABLE_TO_DETERMINE | UNRESOLVED_NO_THIRD_REVIEW |
| P92W-0240 | TR | POSITIVE | UNABLE_TO_DETERMINE | UNRESOLVED_NO_THIRD_REVIEW |
| P92W-0243 | AG | NEGATIVE | POSITIVE | UNRESOLVED_NO_THIRD_REVIEW |

No se asigna respuesta de C1, C2, A ni B; no se abre tercera revision;
no bloquean el analisis de las otras 171 celdas.

## 5. Limitaciones de muestreo (explicitas)

- Las 176 celdas se seleccionaron por coincidencia A/B en binario distinta del gold:
  muestra ENRIQUECIDA por discrepancias. No permite estimar tasa global de error
  del gold, prevalencia poblacional ni porcentaje global de labels incorrectos.
- Componentes del muestreo original separados (sin combinar ingenuamente):
  celdas comparadas por componente: {'B': 133, 'A': 43}; candidatas: {'B': 14};
  en el grid del estudio (750 celdas) la discrepancia A/B observada fue 176/750,
  cifra descriptiva SOLO de esta muestra, no poblacional.
- El componente A (referencia aleatoria) tambien esta condicionado por la seleccion
  de candidatas; tampoco soporta inferencia poblacional directa.

## 6. Limitaciones de independencia (explicitas)

- El cegamiento de la ronda 1 no pudo verificarse por completo (P93-R3).
- La ronda 2 mostro las respuestas de A/B por diseno: las decisiones finales NO son
  independientes de esas respuestas. La concordancia humana NO mide de forma
  independiente el error del gold.

## 7. Procedencia

- Decisiones finales: p93r4_final_decisions.csv (hash 98e4264b0fb98b7a...).
- Gold interno: clave p92w_internal_sampling_key.csv y candidatos P93 (gold 0/1 por celda).
- Sin cambios en fuentes; artefactos nuevos listados en p93r5_manifest.json.

## 8. Bloque de decisiones

```
GOLD_COMPARISON = AUTHORIZED_AND_COMPLETED
THIRD_REVIEW = NOT_REQUIRED_FOR_THIS_PHASE
FINAL_HUMAN_DECISIONS = LOCKED
FIVE_UNRESOLVED_CASES = KEEP_UNRESOLVED
SOURCE_LABEL_MODIFICATION = NOT_AUTHORIZED
MODEL_RETRAINING = NOT_AUTHORIZED
SUBMISSIONS = NOT_AUTHORIZED
V03 = FROZEN
```
