# FASE 6 — Despliegue

## 6.1 La submission final

El pipeline de despliegue es deliberadamente simple y verificado: se toma el consenso geométrico de los
seis componentes sobre el test, se aplica la regla plug-in congelada, y se escribe el archivo de envío
con el formato exacto de la competición (`id, age_range`). El proyecto sometió esa construcción a un
protocolo de integridad de ocho comprobaciones (número de filas, columnas exactas, identificadores únicos
y en el mismo orden del test, ausencia de nulos, clases válidas, todas las clases presentes y sin
duplicados); cualquier fallo detiene la generación. El notebook repite ese protocolo y verifica la
integridad del archivo resultante con las ocho comprobaciones y su huella SHA-256, sin depender de
artefactos externos. La comparación con la **submission sellada del proyecto** (el archivo exacto que
alcanzó 0.63724 en Kaggle) quedó registrada en el control de calidad de la entrega: 112.142 de 112.892
filas idénticas (99,3356%), con las 750 diferencias concentradas en las dos columnas sensibles a zlib y
en la deriva de los componentes neuronales (§6.2).

```python
# Construccion de la submission final y verificacion de integridad (8 comprobaciones).
submission = pd.DataFrame({"id": test["id"].to_numpy(),
                           "age_range": [CLASSES[i] for i in pred_final]})

comprobaciones = {
    "filas == 112892": len(submission) == 112892,
    "columnas exactas [id, age_range]": list(submission.columns) == ["id", "age_range"],
    "ids unicos": bool(submission["id"].is_unique),
    "ids == test y mismo orden": bool((submission["id"].to_numpy() == test["id"].to_numpy()).all()),
    "sin valores nulos": bool(submission.notna().all().all()),
    "clases validas": bool(submission["age_range"].isin(CLASSES).all()),
    "todas las clases presentes": bool(submission["age_range"].nunique() == 6),
    "sin filas duplicadas": bool(not submission.duplicated().any()),
}
for nombre, resultado in comprobaciones.items():
    print(f"  [{'OK' if resultado else 'FALLA'}] {nombre}")
assert all(comprobaciones.values()), "la submission no paso las comprobaciones"

submission.to_csv("submission.csv", index=False)
if Path("/kaggle/working").exists():
    submission.to_csv("/kaggle/working/submission.csv", index=False)
sha_reconstruida = hashlib.sha256(Path("submission.csv").read_bytes()).hexdigest()
print("\nSHA256 reconstruida :", sha_reconstruida)
print("SHA256 del proyecto :", "bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62")
```

```python
# Figura 6.1: distribucion final de las decisiones sobre el test.
fig, eje = plt.subplots(figsize=(8.5, 3.2))
conteo_final = pd.Series([CLASSES[i] for i in pred_final]).value_counts().reindex(CLASSES)
eje.bar(conteo_final.index.astype(str), conteo_final.values, color="#2b6cb0")
for posicion, valor in enumerate(conteo_final.values):
    eje.text(posicion, valor + 150, f"{valor:,}", ha="center", fontsize=8)
eje.set_title("Distribucion de la submission final (V48 + plug-in)")
eje.set_ylabel("tuits")
plt.tight_layout()
plt.show()
```

## 6.2 Reproducibilidad e idempotencia

La reproducibilidad del notebook descansa en cuatro piezas: (1) el **protocolo de folds generado con
la regla documentada** (clústeres de near-duplicados + `StratifiedGroupKFold` con semilla 42; Pedregosa et al., «Scikit-learn: Machine Learning in Python», Journal of Machine Learning Research, 2011),
verificado por propiedades en la propia ejecución —en la versión 1.8.0 reproduce exactamente la
partición canónica del proyecto; en otras versiones la asignación y los valores absolutos pueden
desplazarse—; (2) las **configuraciones congeladas** de todos los
modelos (semillas, hiperparámetros y pesos — nada se busca durante la ejecución); (3) la **cadena
embebida** (ninguna dependencia de scripts o artefactos externos); y (4) el **corpus autónomo** (se
localiza o descarga al inicio). Ejecutar el notebook dos veces produce exactamente los mismos valores.

Tres tolerancias declaradas, todas verificadas sobre la cadena reconstruida (mismo servidor, versiones
exactas). Primera: `compression_ratio` y `compression_gain` — 2 de las 87 columnas estructuradas —
dependen del tamaño de salida de `zlib.compress` (nivel 6) de la librería del sistema; entre builds de
zlib el tamaño comprimido varía en unos bytes y, en tuits cortos, el ratio cambia hasta 0.115 (máximo
medido; media 5.4e-5 sobre las 169.336 filas). Cualquier modelo que use la matriz de 279 columnas
hereda esa sensibilidad: las decisiones de los miembros coinciden entre el 99,96% y el 100%; los
reajustes de árboles (V20, V29, V30 y la segunda partición), entre el 93% y el 97%, porque tanto las
hojas como el punto de parada del early stopping dependen de esas columnas. Segunda: las métricas
agregadas quedan dentro de 1e-3 del valor documentado (en las corridas de verificación, las
comparaciones crudas van de 1e-5 a 8e-4 según el componente, y la regla plug-in —un decisor afinado
sobre las probabilidades originales— es del mismo orden); cada celda imprime el valor exacto de esta
ejecución junto al documentado. Tercera: los componentes neuronales son deterministas en un mismo
entorno, pero entre GPUs de proveedores distintos (AMD ROCm vs NVIDIA CUDA) aparecen diferencias del
orden de 1e-6 en las probabilidades —típicamente por debajo del 0.2% de las decisiones, hasta ~1.2% en
la red densa V53—, que no alteran las métricas ni la práctica totalidad de las decisiones. La idempotencia es exacta: los reajustes de LightGBM fijan el determinismo explícito
(`deterministic` y `force_col_wise`), de modo que dos ejecuciones completas en la misma máquina y
versiones reproducen los mismos valores celda a celda. La comparación con la submission sellada quedó registrada en el control de calidad de la entrega
(99,3356% de filas idénticas); el notebook, para mantenerse autónomo, verifica en cada ejecución el
formato y la huella SHA-256 de su propia reconstrucción.

```python
# Resumen final de la corrida: versiones, formas y valores clave reconstruidos.
print("=== resumen de la corrida ===")
print("python:", sys.version.split()[0], "| numpy:", np.__version__,
      "| pandas:", pd.__version__, "| scikit-learn:", sklearn.__version__)
print(f"matriz train: {matriz.shape} | matriz test: {matriz_test.shape}")
print(f"V20 (OOF)          : {macro_f1(clase, np.argmax(v20_oof, axis=1)):.7f} (doc 0.6097782)")
print(f"V30 (OOF)          : {cv_v30_oof:.7f} (doc 0.6134471)")
print(f"V30 + plug-in (OOF): {cv_plugin_oof:.10f} (doc 0.6159732487657944)")
print(f"V48 + plug-in (OOF): {cv_v48:.7f} (doc 0.6186451)")
print(f"leaderboard del proyecto: 0.63724 (puesto #1; +0.00196 sobre el segundo lugar al cierre, 0.63528)")
```

## 6.3 Conclusiones

**Resultado.** El proyecto construyó, desde cero y sin embeddings, un clasificador de rango de edad que
alcanzó **0.63724 de macro-F1 en el leaderboard público de Kaggle** (primer lugar del torneo), partiendo
de un baseline de 0.55235: una mejora acumulada de **+0.08489**. El modelo entregado combina seis
componentes diversos (dos ensembles de árboles y cuatro redes tabulares con pérdidas ordinales y de
transporte) (Ke et al., «LightGBM: A Highly Efficient Gradient Boosting Decision Tree», NeurIPS 2017; Gorishniy, Rubachev, Khrulkov y Babenko, «Revisiting Deep Learning Models for Tabular Data», NeurIPS 2021; Frank y Hall, «A Simple Approach to Ordinal Classification», ECML 2001; Frogner, Zhang, Mobahi, Araya y Poggio, «Learning with a Wasserstein Loss», NeurIPS 2015) mediante un consenso geométrico (Hinton, «Training Products of Experts by Minimizing Contrastive Divergence», Neural Computation, 2002) y una regla de decisión por clase (Lipton, Elkan y Narayanaswamy, «Thresholding Classifiers to Maximize F1 Score», arXiv, 2014), ambos congelados.

**Hallazgos centrales.** (1) La fuga por near-duplicados (Kaufman et al., 2012) era el riesgo de medición dominante: el
protocolo de validación agrupada cambió todas las comparaciones y sigue siendo conservador frente al
leaderboard. (2) La estructura ordinal del problema es explotable: las pérdidas ordinales y de Cramér (Frank y Hall, 2001; Cramér, 1928),
la cadena de umbrales y el análisis de fronteras convergen en el mismo mensaje. (3) El consenso
multiplicativo (Product-of-Experts; Hinton, 2002) domina a la mezcla aritmética en esta familia de modelos. (4) Una
regla de decisión bien calibrada por clase (Lipton et al., 2014) aporta más de lo que sugiere su simplicidad.

**Lo que no funcionó, y por qué importa.** El proyecto documentó decenas de líneas cerradas con
evidencia — representaciones lingüísticas completas (POS, NER, morfología, dependencias), familias de
modelos (CatBoost, FTRL, MLP denso, CORAL, ordinales) (Prokhorenkova, Gusev, Vorobev, Dorogush y Gulin, «CatBoost: unbiased boosting with categorical features», NeurIPS 2018; McMahan et al., 2013; Cao, Mirjalili y Raschka, «Rank consistent ordinal regression for neural networks with application to age estimation», Pattern Recognition Letters, 2020; Frank y Hall, 2001), canales de rasgos y técnicas de optimización — y
las falsaciones V44–V52 delimitan el espacio con precisión. Los fracasos medidos no son un apéndice: son
lo que permitió dejar de buscar donde no había señal y concentrar el esfuerzo en las cuatro adopciones
que sí transfirieron al leaderboard. El patrón más repetido, y el más útil para futuros trabajos: una
representación puede tener señal real en aislamiento y no aportar nada al integrarse, porque el modelo
que ya existe la expresa mejor.

**La frontera 2026.** Tras el récord de V48, el proyecto siguió empujando el consenso con una red densa
multi-escala, dos variantes del transporte de Cramér y un árbol simétrico (sección 4.6): el techo de
validación local subió a 0.6189929 (V56, nueve modelos, 5/5 folds positivos, p = 0.0010 frente a V48) y
la auditoría matemática de V57 ratificó la selección dual V48/V56 —V48 conserva el mejor leaderboard
(0.63724), V56 la mejor validación local—. La cadena que este notebook reconstruye paso a paso sigue
siendo la del campeón de leaderboard.
