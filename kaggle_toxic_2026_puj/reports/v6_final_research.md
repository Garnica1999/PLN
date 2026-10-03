# v6: informe de cierre de la fase (V6-09)

Generado por `experiments/v6_09_entregables.py` (V6-09 (Pcierre, ANALYSIS)).

## 1. Que buscaba la fase y que ha encontrado

La fase v6 no buscaba un modelo mejor, sino **mecanismos que el proyecto no hubiera
evaluado de verdad**. La respuesta es que **no quedaba ninguno con margen**: de los
11 hipotesis del registro, **7 se midieron y ninguna supero
el Gate 1**, 2 quedaron cerradas **sin gastar** y 1 se
marcaron como no ejecutables.

El resultado no es que V03 sea inalcanzable, sino algo mas preciso y mas util: **se
sabe ahora que tipo de mecanismo no puede aportar nada y por que**. Los
**7 controles de reproduccion**, cada uno dando 0.777211 exacto,
convierten cada resultado negativo en una afirmacion y no en una sospecha.

## 2. Inventario de la fase

- scripts nuevos: **9** (2 de analisis, 7 de experimento)
- variantes medidas: **25**, de las cuales **7** son controles de reproduccion
- variantes que superan el Gate 1: **0**
- deltas negativos entre variantes no control: **14** de 18
- **V03 sigue congelada**: `corrected_cv` 0.777211, LB 0.78306. Nada de la fase se adopto, y **0 submissions** se hicieron.

## 3. Los siete tracks ejecutados y por que se cerraron

| track | mejor delta | IC95 | por que se cerro |
|---|---|---|---|
| B (V6-02) | -0.0008 | [-0.0075, +0.0057] | la invariante TR implica HS es cierta pero solo se viola en el 1.35 % de las predicciones; el producto atenua p(TR) donde V03 acertaba |
| A (V6-03) | +0.0006 | [-0.0029, +0.0040] | solo el 7.1 % de las filas puede cambiar de peso, y penalizar mas a los repetidos empeora: **la repeticion es senal** |
| I (V6-04) | +0.0016 | [-0.0048, +0.0078] | el optimo es interior (1.5) pero el techo es la mitad del gate: **la escala no era el cuello de botella** |
| C1 (V6-05) | -0.0001 | [-0.0085, +0.0082] | los skip-grams son **neutros**: +83 % de caracteristicas sin mover el resultado |
| H (V6-06) | -0.0105 | [-0.0183, -0.0030] | **por protocolo**: hay diversidad real (corr. 0.9573) pero los umbrales de V03 no son validos para un promedio |
| F (V6-07) | +0.0001 | [-0.0033, +0.0035] | la deteccion encuentra **dificultad, no ruido**: V03 acierta 10.5 % en las filas sospechosas frente a 28.3 % en las demas |
| J (V6-08) | -0.0061 | [-0.0141, +0.0018] | aumentar empeora y **doblar las copias no empeora mas**: es sesgo de las copias, no cantidad |

## 4. Los tracks que se cerraron sin gastar

| track | veredicto | causa |
|---|---|---|
| D (TextCNN) | techo teorico | la interseccion de errores da un techo de **+0.0027**, por debajo del gate de +0.003 |
| E (kNN) | techo teorico | mismo techo: +0.0027 como cota de cualquier linea que solo recupere filas ajenas |
| C2, C3, C4 | duplicacion | ya medidos: V4-D dio -0.0038 y -0.0346; E2-10 dio -0.028 a -0.039 |
| G | no ejecutar | el score y su margen ya son entrada de chains y del meta desde E2-01 y E2-18 |

Cerrar por techo **antes** de gastar es la decision que mas presupuesto ahorro de la
fase, y se tomo apoyandose en una medicion (la interseccion de errores) y no en una
suposicion.

## 5. Los tres candidatos mas cercanos al gate

| variante | delta | a que distancia del gate | IC95 | pliegues |
|---|---|---|---|---|
| I2-char1.5 | +0.0016 | 0.0014 | [-0.0048, +0.0078] | 3 |
| A1-sqrt | +0.0006 | 0.0024 | [-0.0029, +0.0040] | 3 |
| A2-inversa | +0.0002 | 0.0028 | [-0.0045, +0.0051] | 2 |

El mas cercano es **I2-char1.5** con +0.0016, a
0.0014 del gate, con un IC95 que **incluye el 0**. No es un
candidato: es el techo empirico de la linea mas barata que quedaba.

## 6. Lo que la fase deja escrito, mas alla de los negativos

Cinco conclusiones sobreviven al cierre y son las que hacen que la fase valiera la
pena:

1. **Un tuit repetido es senal, no ruido de recoleccion** (track A). Penalizar los
   grupos repetidos empeora de forma monotonica. No reabrir esa linea ni en variantes
   mas suaves.
2. **La repeticion aporta informacion recuperable: casi ninguna** (P0-bis). El mejor
   modelo existente acierta 9 a 12 filas de 1.483 donde V03 falla, y combinarlos
   **elige** entre sus aciertos en vez de sumarlos. Toda linea cuyo techo sea
   recuperar filas ajenas esta condenada por construccion.
3. **La deteccion de sospecha sirve para medir dificultad, no para limpiar** (track F).
   Marca donde V03 falla, y bajar el peso de esas filas aleja al modelo de sus
   errores.
4. **La escala entre word y char no era el cuello de botella** (track I), aunque el
   optimo este en 1.5 y no en el 1 heredado.
5. **No hay forma limpia de medir un promedio de modelos con umbrales calibrados para uno** (track H). Es una limitacion del protocolo, y queda planteada la condicion
   que la resolveria: una particion de calibracion separada del OOF de evaluacion.

## 7. Entregables de la fase

| entregable | contenido |
|---|---|
| `reports/v6_final_research.md` | este informe |
| `reports/v6_master_results.csv` | una fila por variante, con metricas y gate |
| `reports/v6_hypothesis_registry.csv` | una fila por hipotesis, con veredicto y causa |
| `reports/v6_candidate_comparison.csv` | los tres mas cercanos al gate |
| `reports/v6_error_intersection.csv` | interseccion de errores canonizada |
| `reports/v6_compliance.md` | cumplimiento de las restricciones |
| `reports/v6_reproducibility.md` | como reproducir cada resultado |
| `reports/v6_hypothesis_gap_analysis.md` | el gap analysis con verificacion programatica |
| `reports/v6_error_structure.md` | interseccion de errores y estructura de etiquetas |
| `reports/figuras/fig_58` a `fig_65` | ocho figuras de la fase |

Fecha: 2026-09-26T01:45:15+00:00. Duracion de la consolidacion: 0.0 s.
