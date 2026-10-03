# V25 · 02 — Auditoria de la integracion documental

**Estado: PASS**

## 1. Notebook

- Celdas: **136** | secciones: **34**.
- Identificadores unicos: **True**.
- Titulos unicos: **True**.
- Seccion 33 de V25 presente: **True**.
- Artefactos referenciados: **67**, faltantes: **0**.

## 2. Ejecucion real

- Celdas de codigo ejecutadas: **11**.
- Errores: **0**.

- `e21d474c: experimentos registrados: 128 | decisiones documentadas: 59`
- `v25sel0002: features auditadas: 279`
- `v25sel0003: veredicto: LEGAL: construidas con las etiquetas del training fold y con la misma particion de folds que V20`
- `v25sel0006: T0 reproduce: True | valor: 0.6097782 | T1: 0.6091492 | delta: -0.008383`
- `v25sel0012: features: 279 | en los 5 folds: 137 | en ninguno: 138`
- `v25sel0015: v25_frecuencia_seleccion.png | <IPython.core.display.Image object>`

## 3. Unicode (requisito bloqueante)

- **Introducidos por V25: 0** (requisito bloqueante)
- **Heredados del notebook, de sesiones anteriores: 38**, medidos antes de esta fase y no atribuibles a V25
- En los scripts de V25: **0**

Se comprueban: alfabetos no latinos, CJK, cirilico, caracteres de control, ancho cero, BOM, comillas tipograficas, guiones Unicode y espacios Unicode.

## 4. Coherencia numerica

| Comprobacion | Resultado |
|---|---|
| t0 es el canonico | OK |
| t1 coincide con el json | OK |
| inventario de 279 | OK |
| inventario 73 label informed | OK |
| frecuencia de 279 | OK |
| decision cerrada | OK |
| sin submission | OK |

## 5. Resultado de la fase, que el notebook documenta

- Control T0: **0.6097782** (reproduce: True).
- T1 seleccion MI top 0.5 (K = 139): **0.6091492** | delta **-0.000629** | **1/5** folds positivos.
- Decision: **SUPERVISED_FEATURE_SELECTION = CLOSED** | submission: **False**.

- Inventario: **279 features** auditadas, de las cuales **73** label-informed.
