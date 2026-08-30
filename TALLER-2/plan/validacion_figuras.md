# Validación de figuras y del trabajo del TALLER-2 (ontología + tesauro)

**Fecha**: 2026-08-30 · **Objetivo**: verificar las 14 figuras guardadas en `imagenes/`, extraer información de lo realizado y validar si la ontología y el trabajo con el tesauro fueron correctos.
**Método**: inspección visual de cada PNG (visión) + validación cruzada contra `data/tesauro_orginal_cev.xlsx`, `data/grupos_seleccionados.json`, `data/wordnet_relaciones.csv` y `ontologia/tesauro_cev.owl` (owlready2/HermiT).

---

## 1. Inventario y metadatos (14 figuras)

| Figura | Tamaño | Nivel |
|---|---|---|
| `barras_top_grupos.png` | 1195×1039 | EDA |
| `profundidad_grupos.png` | 1330×889 | EDA |
| `distribucion_tipos_termino.png` | 1189×739 | EDA |
| `heatmap_terminos_grupos_nivel.png` | 1342×1039 | EDA |
| `arbol_jerarquico_armas.png` | 2389×1339 | EDA |
| `cobertura_wordnet_grupos.png` | 1517×889 | Selección |
| `arbol_ontologia.png` | 3517×2458 | Ontología |
| `grafo_ontologia_completo.png` | 4786×3985 | Ontología |
| `grafo_ontologia_grupos.png` | 2586×2186 | Ontología |
| `cobertura_wordnet_conceptos.png` | 1626×799 | WordNet |
| `heatmap_relaciones_grupos.png` | 1682×874 | WordNet |
| `top_relaciones_conceptos.png` | 1565×1024 | WordNet |
| `grafo_relaciones_conceptos.png` | 2076×1625 | WordNet |
| `profundidad_versus_cobertura.png` | 1626×949 | WordNet |

Todas existen, se generan a 150 dpi y tienen tamaños razonables.

## 2. Validación de datos (¿lo realizado es correcto?)

**Tesauro (Excel) → JSON de selección**: los 6 grupos coinciden EXACTAMENTE (Actores 82, Impactos 107, Armas 94, Poblaciones 213, Hechos 90, Afrontamientos 23 = 609). Cada término del JSON proviene de la hoja `Tesauro` real. **CORRECTO.**

**JSON → Ontología OWL**: 616 clases por raíz (Actores 81, Hechos 91, Impactos 107, Poblaciones 217 [incluye 4 nodos sintéticos], Armas 96 [incluye 2], Afrontamientos 24 [incluye 1]); 534 individuos (términos hoja); 17 object properties; 6 clases definidas; HermiT consistente y clasifica. **CORRECTO** (la diferencia +1/+4/+2/+1 son nodos sintéticos que completan la jerarquía, convención documentada).

**Figuras vs datos**: cada cifra dibujada coincide con el CSV/Excel/OWL verificado (cobertura 50,5% = 308/610; por grupo 79/107, 71/89, 56/81, 49/96, 36/214, 17/23).

## 3. Hallazgos por figura y lo que revelan del trabajo

### 3.1 EDA (tesauro correctamente caracterizado)
- **barras_top_grupos**: Organizaciones (6.405) domina el tesauro; los demás grupos son pequeños en comparación. Justifica el criterio de tamaño (50-450) para la selección.
- **profundidad_grupos**: Armas y Estructuras armadas llegan a 10 niveles; los grupos seleccionados tienen medianas 3-5. Revela que **Armas fue una elección con riqueza jerárquica** (hasta 10 niveles, la más profunda de los seleccionados).
- **distribucion_tipos_termino**: solo 40 términos simples (1 segmento) de 12.341 — el tesauro es casi totalmente compuesto (4.429 en 3 segmentos). Explica por qué el WordNet exacto falla (los compuestos no existen como lemas) y por qué el fallback por segmentos es necesario.
- **heatmap_terminos_grupos_nivel**: Organizaciones concentra 2.347 términos en nivel 5; los grupos seleccionados están concentrados en niveles 3-4. La jerarquía del tesauro es profunda y desigual.
- **arbol_jerarquico_armas**: estructura jerárquica correcta (Armas → Instrumentos/Medios → Armas blancas / Armas de fuego → ...). **Problema estético**: etiquetas de niveles 4-6 solapadas (ilegibles) — debilidad menor de legibilidad.

### 3.2 Selección (criterio validado)
- **cobertura_wordnet_grupos**: muestra correctamente el umbral 1% y los grupos que lo superan (Profesiones 17,3%, Actores 14,6%, Impactos 11,2%, Armas 7,5%, Hechos 2,2%, Poblaciones 1,4%). Poblaciones apenas lo supera — dato crítico de la selección (es el grupo más grande de los elegidos, 213, pero casi sin cobertura).

### 3.3 Ontología (validada visualmente)
- **arbol_ontologia**: 6 raíces (incluido Afrontamientos en marrón) + subárbol de Armas; **CORRECTO** y legible.
- **grafo_ontologia_completo** (1.158 nodos, 1.286 aristas): los nodos se **aglomeran por grupo** (naranja Poblaciones, verde Armas, azul Impactos, rojo Actores, morado Hechos, marrón Afrontamientos) — valida que la ontología preserva la licuación jerárquica por grupo. La superclase `ConceptoDelTesauro` (gris) es el punto central. **CORRECTO**.
- **grafo_ontologia_grupos** (7 nodos, 12 aristas): muestra el **ciclo del conflicto** — Hechos→Actores (perpetradoPor), Armas→Actores/Hechos, Impactos→Hechos/Poblaciones, Poblaciones→Hechos/Impactos/Afrontamientos, Afrontamientos→Poblaciones/Hechos. Posible observar la **cadena causal completa** en un solo grafo. **CORRECTO** pero **etiquetas truncadas** en los nodos ("robaciones..." = Poblaciones, "stentias no..." = Afrontamientos) — se desbordan del círculo.

### 3.4 WordNet (hallazgos clave validados)
- **cobertura_wordnet_conceptos**: 50,5% global (308/610); Afrontamientos 73,9% (17/23) — el grupo incorporado del taller ajeno tiene buena cobertura vía fallback; Poblaciones 16,8% (36/214) — el límite del recurso léxico.
- **heatmap_relaciones_grupos**: normalizado por tamaño; Impactos domina hiponimia (93,2/concepto); Hechos y Afrontamientos dominan coordinados (79,6 y 66,9). Revela la estructura del ruido.
- **top_relaciones_conceptos**: los conceptos con más relaciones son **núcleos polisémicos** ("Cambios forzados en la corporabilidad y performatividad del género", "Impactos en las relaciones y arreglos de género" = 2.804; "familia", "vida" = 1.325). **Hallazgo crítico**: el 36,2% de relaciones se concentra en 15 conceptos polisémicos — confirma que el fallback por núcleos dispara hiponimia/coordinados de forma ruidosa.
- **grafo_relaciones_conceptos**: componente gigante central (vida, familia, género — conecta grupos) + periferia de Armas aislada ("Pistola Beretta", "FuGA", "M16A1" — vocabulario militar sin equivalente en WordNet). Valida la limitación de cobertura del dominio especializado.
- **profundidad_versus_cobertura**: Spearman −0,180, Pearson −0,139 — la profundidad NO explica la cobertura; los niveles 5-6 (Armas) tienen ~0 relaciones mientras los niveles 3 (polisémicos) tienen picos. **Conclusión metodológica**: la especificidad del dominio, no la posición jerárquica, determina la cobertura.

## 4. Veredicto

**El trabajo con el tesauro y la ontología está CORRECTO**:
- Los 6 grupos del JSON provienen del Excel real y coinciden término a término (609).
- La ontología (623 clases, 534 individuos, 17 props, 6 clases definidas) refleja fielmente la jerarquía del tesauro (616 clases por raíz + superclase + definidas), es consistente (HermiT) y el grafo muestra la aglomeración por grupo.
- WordNet se aplicó correctamente sobre 610 conceptos; las cifras de todas las figuras coinciden con el CSV.

**Problemas de legibilidad detectados (mejora para rúbrica criterio 5)**:
1. `arbol_jerarquico_armas.png`: etiquetas de niveles 4-6 solapadas (ilegibles).
2. `grafo_ontologia_grupos.png`: etiquetas truncadas/desbordadas en los nodos grandes.
3. `cobertura_wordnet_conceptos.png`: orden de barras no decreciente (TOTAL primero, sin ordenar por %) — confusión menor.
4. `top_relaciones_conceptos.png`: leyenda bien, pero los nombres largos se enciman ligeramente con las barras.

**Estado (corregido 2026-08-30)**: las 4 figuras con problemas de legibilidad fueron **regeneradas y verificadas**:
1. `arbol_jerarquico_armas.png`: etiquetas de niveles 5-6 rotadas a 90° (sin solapamiento).
2. `grafo_ontologia_grupos.png`: etiquetas completas fuera del nodo con caja blanca (sin truncado); título con 6 grupos/12 aristas.
3. `cobertura_wordnet_conceptos.png`: barras ordenadas de mayor a menor + TOTAL en rojo.
4. `top_relaciones_conceptos.png`: nombres recortados con '…' y figura más ancha.
Verificación: tests 67 passed, ruff 0 errores, notebook end-to-end 0 errores con las figuras nuevas embebidas.
