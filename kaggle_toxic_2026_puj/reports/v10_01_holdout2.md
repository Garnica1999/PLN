# P0b: holdout2 congelado y referencia recalculada en dev2 (V10-01)

Generado por `experiments/v10_01_holdout2.py` (V10-01 (P0b, holdout2)).

## 1. La advertencia, escrita antes de ver resultados

**Holdout2 no es un conjunto virginal.** Se construye con un pliegue del desarrollo
actual, y esas 1483 filas ya participaron en las decisiones de las fases v1 a v9.

Su valor real es **detectar que un candidato esta sobreajustado a su gate**, que es un
uso legitimo y necesario, y **no** es probar generalizacion sobre datos nunca vistos.
Una mejora aprobada aqui debe leerse como *sobrevive a un conjunto que ya se uso para
decidir otras cosas*.

Por eso holdout2 se crea **antes** de la via A y no despues: si se creara despues, las
candidatas de la via A se elegirian sobre las mismas filas que luego se usarian para
juzgarlas, y el juez dejaria de ser independiente de la seleccion.

## 2. Composicion

| conjunto | filas | grupos |
|---|---|---|
| desarrollo original | 1483 | 1415 |
| holdout2 | 247 | 235 |
| dev2 | 1236 | 1180 |

**Grupos compartidos entre holdout2 y dev2: 0.** La
particion se construye con los mismos grupos fusionados que en v2, near-duplicado
unido a pseudo-autor, de modo que la limpieza es de grupo y no solo de fila.

Prevalencias:

| conjunto | HS | TR | AG |
|---|---|---|---|
| holdout2 | 0.4372 | 0.2996 | 0.3765 |
| dev2 | 0.4369 | 0.3002 | 0.3835 |
| desarrollo original | 0.4367 | 0.2998 | 0.3824 |

## 3. La referencia recalculada en dev2

- `corrected_cv2`: **0.763436**
- frente a los 0.777211 del
  desarrollo original, desviacion **-0.013776**
- tolerancia declarada: 0.005
- dentro de tolerancia: **False**

F1 por etiqueta: HS 0.7604, TR 0.7940, AG 0.7359

La referencia se recalcula con **cinco pliegues propios dentro de dev2**, y no ajustando
y evaluando sobre las mismas filas: una cifra de resubstitucion no es comparable con los
0.7772 historicos, que son un promedio fuera de muestra.

**Los 0.777211 historicos no se sustituyen.** Se conservan como referencia del
desarrollo completo, y la nueva cifra convive con ellos para las comparaciones dentro
de dev2.

## 4. Comprobaciones de integridad

- particion reproducible con la misma semilla: **True**
- grupos compartidos: **0**
- solape con la configuracion del holdout v2: 0
- `sha256` de `config/holdout2.json`: f71ad8ac4959dbdcccc7da23827a36924acd74f344522b81f219a9122d95b3c1
- `sha256` de `data/processed/holdout2_ids.csv`: f6c6ed958abf728b8210d15c26ec43fbd71659a0918d36089a9c6d6cef1639dc

## 5. Regla de uso, congelada con el conjunto

- **maximo 3 evaluaciones**, y solo para decisiones de adopcion
  pre-registradas.
- prohibido optimizar nada sobre holdout2.

## 6. Veredicto

**GATE NO SUPERADO.** A partir de
aqui ninguna decision de adopcion puede apoyarse solo en el OOF, y cualquier candidata
que llegue a la mesa de finales necesita una evaluacion unica aqui.

Fecha: 2026-09-26T07:41:40+00:00. Duracion: 5.7 s.
