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

## A - Ponderacion por grupo (pendiente)

- [ ] A1 `w=1`, A2 `w=1/sqrt(g)`, A3 `w=1/g` y variante con tope
- [ ] Grupos calculados **solo** sobre el train del pliegue

## C1 - Skip-grams (pendiente)

- [ ] Solo C1. Hueco acotado 1, y 2 unicamente si el 1 aporta

## I - Ponderacion word/char (pendiente)

- [ ] char_weight en {0.5, 1.0, 1.5, 2.0} con word_weight fijo en 1
- [ ] Es la linea mas barata que queda: cuatro corridas

## H, F, J (pendiente, tras B y A)

- [ ] H: bagging por grupos
- [ ] F: score de sospecha por par instancia-etiqueta
- [ ] J: aumento label-preserving, al final

## Cierre

- [ ] `reports/v6_master_results.csv` y `reports/v6_hypothesis_registry.csv`
- [ ] Seccion v6 en el notebook consolidado
- [ ] `reports/v6_final_research.md` y el resto de entregables
- [ ] Auditoria final de reglas
