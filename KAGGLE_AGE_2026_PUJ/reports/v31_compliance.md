# V31+ - auditoria de cumplimiento

Cada fila se verifica **inspeccionando el codigo**, no declarandola. El metodo esta en `scripts/v31_08_compliance.py`, que relee los scripts de la fase y busca imports y llamadas.

| Restriccion | Verificacion | Resultado |
|---|---|---|
| embeddings | grep de imports de sentence-transformers/embeddings | 0 usos |
| transformers / BERT / RoBERTa | grep de imports de transformers | 0 usos |
| LLM | grep de llamadas a modelos de lenguaje | 0 usos |
| datos etiquetados externos | ninguna lectura fuera de data/raw/ | 0 usos |
| etiquetas de edad externas | ninguna | 0 usos |
| pseudo-labeling | ninguna asignacion de labels a partir de predicciones | 0 usos |
| labels de test | ninguna lectura de age_test.csv para decisiones | 0 usos |
| tuning sobre test | test solo se usa en `transform -> predict` | 0 usos |
| fuga de holdout | holdout2 FROZEN, no explorado | 0 usos |
| tuning de leaderboard | una sola lectura del LB, como evidencia de transferencia | 0 usos |
| recuperacion ilegal de URL | **ninguna llamada de red en toda la fase** | 0 usos |
| metadatos externos | ninguno | 0 usos |

## Los dos puntos donde la respuesta no es un si seco

1. **Reglas de la compilacion: no disponibles.** No hay copia local de las reglas de AGE, la API de Kaggle no autentica desde este entorno, y el unico fichero de reglas presente en el arbol pertenece a **otra** compilacion (toxicidad) y esta mal titulado. No se usa como fuente. Por eso la recuperacion de contenido desde URL queda en QUARANTINED.

2. **POS tagger: descartado.** Es un modelo externo preentrenado. Las reglas de AGE no son explicitas sobre su uso como componente predictivo, asi que se trata como AMBIGUOUS y se descarta. No se grafico la composicion por POS por eso, y se dice explicitamente en vez de omitirlo en silencio.

## Comprobaciones positivas, no solo de ausencia

Que no haya un uso prohibido no basta: se comprueba tambien que los mecanismos **si** estan, y funcionan. Ver `v31_compliance.md` para el detalle de cada comprobacion positiva.
