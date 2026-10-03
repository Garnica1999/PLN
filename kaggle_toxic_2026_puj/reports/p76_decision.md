# Decision de P76

**CASO B: CALIBRACION_MEJORA_COMPLEMENTARIEDAD_PERO_NO_F1**

```text
V03 campeon                     0.7772114525
V03 calibrado                   0.7685825353
CNN crudo                       0.5838755350
CNN calibrado                   0.5738807461
macro Brier V03 raw -> cal      0.125771 -> 0.122455
macro Brier CNN raw -> cal      0.192150 -> 0.188485
mejor fusion                    C_V03raw_CNNcal_alpha0.05  delta -0.0017743921  folds 2/5
gate superado por               ninguna
finalista                       None
```

La calibracion **hace visible mejor** la informacion complementaria, pero no basta para convertirla en mejora del objetivo. Se cierra la fusion calibrada salvo que aparezca un mecanismo especifico y falsable.

## Estado

```text
V03 = FROZEN
P72 = CLOSED
P73 = COMPLEMENTARY
P74 = EVIDENCIA REPRODUCIBLE
P75 = COMPLEMENTARIEDAD SIN TRANSFERENCIA
P76 = EJECUTADA
TEXTCNN = ACTIVA PARA INVESTIGACION CONTROLADA
test = 0
holdout = 0
leaderboard = 0
submission = 0
```
