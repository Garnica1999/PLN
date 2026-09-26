# v6: cumplimiento de las restricciones (V6-09)

Generado por `experiments/v6_09_entregables.py` (V6-09 (Pcierre, ANALYSIS)).

## 1. Restricciones de la especificacion

| restriccion | estado | como se verifica |
|---|---|---|
| sin transformers, LLM ni embeddings preentrenados de terceros | cumplida | ningun script de v6 importa transformers, sentence-transformers ni HF |
| sin etiquetas externas ni pseudo-labeling | cumplida | ninguna etiqueta entra de fuera de `toxic_train.csv` |
| sin usar `toxic_test.csv` para entrenar | cumplida | solo se lee para generar el fichero de submission, que en v6 no se genero |
| sin usar el holdout | cumplida | el holdout esta agotado (5/5) y la fase lo declara; el Gate 2 usaria otra particion agrupada |
| sin usar el leaderboard para ajustar hiperparametros | cumplida | **0 sondas de LB y 0 submissions en toda la fase** |
| los 19 artefactos FastText de v5/v5.1 son reutilizables | cumplido en principio | no se gasto presupuesto en ellos: D se cerro por techo teorico |
| presupuesto de 24 principales y 6 de reserva | cumplido de sobra | **7 tracks ejecutados, ninguno adopted** |
| un experimento = un script nuevo e inmutable | cumplido | 9 scripts nuevos `v6_00` a `v6_08`, ninguno edita a otro |
| ejecutar con `/home/carlos/venv/bin/python` | cumplido | todos los scripts |

## 2. Gasto de la fase

- scripts nuevos: **9** (2 de analisis, 7 de experimento)
- tracks con variantes medidas: **7** (B, A, I, C1, H, F, J)
- tracks cerrados **sin gastar**: **3** (D y E por techo, C2/C3/C4 por duplicacion, G marcado no ejecutar)
- variantes que superan el Gate 1: **0**
- submissiones al leaderboard: **0**
- sondas de LB: **0**

El presupuesto autorizado era de 24 experimentos principales y 6 de reserva. El
gasto real fue de **7**. La razon no es que faltara tiempo: es que el gap analysis
(V6-00) y el diagnostico (V6-01) **cerraron 4 tracks antes de gastar una unidad**, y
los tracks ejecutados se cerraron uno a uno por resultado negativo. El presupuesto
no obliga a gastarlo, y gastar por gastar seria exactamente el error que D-044
intenta evitar.

## 3. Estado del repositorio

- rama: `proyecto-kaggle-toxic-2026-v6`
- commits de la fase: **9**, ultimo `834d565`
- **sin push**: la autorizacion de push nunca se pidio y no se hace sin ella
- secretos en ficheros versionados: ninguno en los ficheros de la fase

## 4. Recordatorio de seguridad pendiente desde v2

El token clasico de GitHub (`ghp_rR3A...`) que aparecio en el historial del proyecto
**sigue pendiente de rotar**. Esta fase no lo ha usado ni lo ha reintroducido, pero
la rotacion es responsabilidad de Carlos y no depende de este trabajo.
