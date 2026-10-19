# P106-01: Nuevos experimentos para acercarse a 0.80

Fecha: 2026-10-14. Duracion: 315.03 s.

## Resumen de Resultados

| Variante | delta Canonico | Pliegues Pos. | Celdas Cambiadas | IC95 Bootstrap | delta Part. 47 | Veredicto Gate A3 |
|---|---:|---:|---:|---|---:|---|
| V1_boost_tr_hs | +0.000239 | 2/5 | 4 | [-0.000673, +0.001337] | +0.000555 | **GATE_FAILED** |
| V2_desactivacion_afecto | -0.000489 | 1/5 | 3 | [-0.001556, +0.000388] | -0.000093 | **GATE_FAILED** |
| V3_umbrales_3d_anidados | +0.001667 | 4/5 | 67 | [-0.002194, +0.005539] | -0.002058 | **GATE_FAILED** |
| V4_fusion_logits_linearsvc | -0.000218 | 2/5 | 19 | [-0.002620, +0.002024] | +0.000091 | **GATE_FAILED** |

## Conclusiones
- V1 (Boost TR->HS): Demuestra pequeno delta positivo pero celdas insuficientes (<200) e IC que incluye cero.
- V2 (Desactivacion afecto/broma): Muestra delta nulo/marginal, confirmando que la distincion broma vs odio requiere contexto extralinguistico.
- V3 (Umbrales 3D anidados): La seleccion anidada no supera a los umbrales globales de V03 (confirma la estabilidad del punto de operacion congelado).
- V4 (Fusion Logits LinearSVC): La regularizacion aditiva de dos modelos lineales en logits mantiene el nivel de V03 sin producir salto significativo.

V03 remains the best verified leaderboard result.
