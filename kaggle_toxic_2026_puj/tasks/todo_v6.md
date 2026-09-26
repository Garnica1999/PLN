# Tareas de la fase v6

## P0 - Gap analysis (hecho)

- [x] Rama `proyecto-kaggle-toxic-2026-v6` creada desde v5.1
- [x] `experiments/v6_00_gap_analysis.py`: 15 verificaciones programaticas sobre el codigo real
- [x] `reports/v6_hypothesis_gap_analysis.md` y `.csv` con las ocho columnas de la spec
- [x] `reports/v6_verificaciones_gap_v6.json` con la evidencia cruda
- [x] C2, C3 y C4 **cerradas** por duplicacion (V4-D, E2-10)
- [x] Hallazgo de integridad: 6 filas corruptas en experiments.csv; el real es 168

## P0-bis - Diagnostico (hecho)

- [x] `experiments/v6_01_estructura_errores.py`: interseccion de errores y estructura de etiquetas
- [x] Interseccion de solo 9-12 filas: techo de +0.0027, por debajo del gate
- [x] **D y E cerradas preventivamente** por techo teorico
- [x] TR implica HS de forma determinista: el track B tiene soporte empirico
- [x] P(AG|HS=0) = 0.026 con 22 ejemplos: justifica B2r

## B - Factorizacion condicional (cerrado negativo)

- [x] `v6_02_label_factorization.py`: B0 control, B1, B1c, B2, B3
- [x] B0 control reproduce V03 exacto (0.777211): deltas atribuibles
- [x] B1 -0.0107, B2 -0.0008, B1c/B3 -0.0115: **0/5 pasan el Gate 1**
- [x] Causa medida: solo 20 filas de prediccion (1.35 %) violan la invariante
- [x] **Cerrado por la seccion 54. Sin ampliar rejilla.**

## A - Ponderacion por grupo (cerrado negativo)

- [x] `v6_03_ponderacion_grupo.py`: A0 control, A1 1/sqrt(g), A2 1/g, A3 con tope
- [x] A0 control reproduce V03 exacto: deltas atribuibles
- [x] Grupos contados **solo** sobre el train de cada pliegue (si no, leakage)
- [x] A1 +0.0006, A2 +0.0002, A3 -0.0149: **0/4 pasan el Gate 1**
- [x] Solo el 7.1 % de filas puede cambiar de peso
- [x] Patron monotono: penalizar mas a los repetidos empeora (la repeticion es señal)
- [x] **Cerrado por la seccion 54. No reabrir ni en variantes mas suaves.**

## C1 - Skip-grams (cerrado negativo neutro)

- [x] `v6_05_skipgrams.py`: sin hueco (control), hueco 1, huecos 1 y 2
- [x] El control detecto un bug de tokenizacion y lo corrigio (emitia solo unigramas)
- [x] Control ya reproduce V03 exacto: 0.777211
- [x] Vocabulario +83 % con resultado -0.0001: **neutro**, ni anade ni resta
- [x] **C cerrada entera**: C2, C3, C4 por duplicacion y C1 aqui

## I - Ponderacion word/char (cerrado negativo)

- [x] `v6_04_char_weight.py`: char_weight en {1.0 control, 0.5, 1.5, 2.0}
- [x] I0 control reproduce V03 exacto: deltas atribuibles
- [x] Optimo **interior** en 1.5 (+0.0016), pero IC95 incluye el 0 y 3/5 pliegues
- [x] Techo = mitad del gate: **la escala no era el cuello de botella**
- [x] **Cerrado por la seccion 54. No se afina alrededor de 1.5.**

## H - Bagging por grupos (cerrado por protocolo)

- [x] `v6_06_bagging_grupos.py`: H0 control y H1 promedio de 5 con remuestreo de grupos
- [x] H0 control reproduce V03 exacto
- [x] **Diversidad medida: correlacion 0.9573**, o sea hay cinco modelos distintos
- [x] -0.0105 con IC95 que excluye el 0 y 0/5 pliegues
- [x] La hipotesis de techo se **refuto a si misma**; el cierre es de protocolo
- [x] Reabrible solo con una particion de calibracion separada del OOF

## F y J (pendiente)
- [ ] F: score de sospecha por par instancia-etiqueta
- [ ] J: aumento label-preserving, al final

## Cierre

- [ ] `reports/v6_master_results.csv` y `reports/v6_hypothesis_registry.csv`
- [ ] Seccion v6 en el notebook consolidado
- [ ] `reports/v6_final_research.md` y el resto de entregables
- [ ] Auditoria final de reglas
