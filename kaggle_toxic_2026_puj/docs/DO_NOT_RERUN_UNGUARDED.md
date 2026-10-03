# DO NOT RERUN UNGUARDED

Scripts que, al re-ejecutarse, destruyen artefactos existentes sin fallar.

## experiments/p66_01_regenerar_scores.py

```text
marca           = DO_NOT_RERUN_UNGUARDED
artefacto       = reports/p66_score_fusion.csv
registros       = 214
restaurado      = si
codigo cambiado = no
```

Escribe su tabla de verificacion de 8 filas encima de los 214 registros de la fase
de fusion, porque comparte nombre de fichero con `p66_02_fusion_espacio_scores.py`.

Ejecutar SIEMPRE seguido de:

```bash
git checkout HEAD -- reports/p66_score_fusion.csv \
                      reports/p66_score_regeneration.json
```

Detalle completo en `reports/p66_artifact_damage.md`.
