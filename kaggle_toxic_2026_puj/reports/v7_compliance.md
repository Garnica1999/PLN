# v7: cumplimiento de las restricciones

Generado por `experiments/v7_04_entregables.py` (V7-04 (cierre, analisis)).

## 1. Recuento programatico de las prohibiciones

Una lista de prohibiciones solo vale algo si se cuenta sobre el codigo real. Cada
fila es un recuento, no una declaracion.

| prohibicion | apariciones en el codigo de la fase | ficheros | veredicto |
|---|---|---|---|
| transformers | 0 | ninguno | sin infracciones |
| modelos de lenguaje | 0 | ninguno | sin infracciones |
| embeddings de terceros | 0 | ninguno | sin infracciones |
| etiquetas externas | 0 | ninguno | sin infracciones |
| pseudoetiquetado | 0 | ninguno | sin infracciones |
| uso de etiquetas de test | 0 | ninguno | sin infracciones |
| entrenamiento con texto de test | 0 | ninguno | sin infracciones |
| uso del holdout | 0 | ninguno | sin infracciones |
| nuevo corpus externo | 0 | ninguno | sin infracciones |

**Infracciones detectadas: 0.**

## 2. Restricciones que no se comprueban por recuento

| restriccion | como se cumple |
|---|---|
| V03 no se modifica | sus artefactos son de solo lectura para la fase; el OOF canonico se carga y nunca se escribe |
| la submission historica no se toca | la fase no genera ninguna submission |
| cero submissions automaticas | **0 submissions y 0 sondas de leaderboard** en toda la fase |
| presupuesto autorizado de 25 mas 10 | **3 tracks ejecutados**: se ejecuto lo que la auditoria dejo abierto, no lo que autorizaba el presupuesto |
| un experimento es un script nuevo e inmutable | cuatro scripts nuevos `v7_` a `v7_`, ninguno edita a otro |
| el holdout no se usa | el protocolo de la fase es el de desarrollo; el holdout sigue agotado desde hace varias fases |
| ejecuta con el interprete del proyecto | todos los scripts con `/home/carlos/venv/bin/python` |

## 3. Caracteres

El verificador encuentra **0** caracteres no permitidos en
los ficheros propios del proyecto. Los simbolos matematicos griegos de las tablas y
los alfabetos no latinos del informe de cobertura de vocabulario se admiten porque en
esos dos casos son notacion y dato, no prosa.

Detalle: **ninguno**.

## 4. Estado del repositorio

- rama: `proyecto-kaggle-toxic-2026-v7`
- commits de la fase: **4**, ultimo `18c6e7d`
- **sin push**: la autorizacion de push nunca se pidio y no se hace sin ella
- worktree en `/home/carlos/Git/PLN-v7`, aislado del arbol principal para no tocar el
  trabajo sin versionar del proyecto KAGGLE_AGE

## 5. Recordatorio de seguridad pendiente desde la fase v2

El token clasico de GitHub (`ghp_rR3A...`) que aparecio en el historial del proyecto
**sigue pendiente de rotar**. Esta fase no lo ha usado ni lo ha reintroducido, pero la
rotacion es responsabilidad de Carlos.
