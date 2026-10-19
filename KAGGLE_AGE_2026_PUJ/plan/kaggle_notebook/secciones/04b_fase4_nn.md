## 4.4 Los componentes neuronales V38–V43: la superación del techo lineal

Tras el plug-in, el diagnóstico del proyecto fue claro: el modelo lineal había llegado a su techo
(oráculo de 0.6935, sección 5.4) y el margen restante exigía **modelos no lineales sobre la misma matriz
de 279 columnas**. Esta sub-fase construyó seis componentes, cada uno añadiendo una técnica sobre el
anterior, todos evaluados con los folds canónicos y las mismas reglas:

- **V38 — TabularResNet:** una red residual para datos tabulares (proyección, dos bloques residuales con
  normalización por lotes (Ioffe y Szegedy, «Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift», ICML 2015) y activación SiLU, dropout 0.2; Gorishniy, Rubachev, Khrulkov y Babenko, «Revisiting Deep Learning Models for Tabular Data», NeurIPS 2021) entrenada con entropía cruzada sobre GPU.
  Superó al techo lineal en validación (CV +0.00068 sobre V30) y demostró que había señal no lineal.
- **V39 — pérdida ordinal + trío:** la misma red con una pérdida que penaliza la **distancia ordinal**
  esperada (equivocarse por un rango cuesta menos que por cinco; Frogner, Zhang, Mobahi, Araya y Poggio, «Learning with a Wasserstein Loss», NeurIPS 2015) y un trío con HistGradientBoosting; la
  combinación con V30 (0.84/0.12/0.04) pasó el gate.
- **V40 — atención Squeeze-and-Excitation:** compuertas que re-ponderan las 256 dimensiones internas de
  cada bloque según su utilidad (mecanismo SE, reducción 4; Hu, Shen y Sun, «Squeeze-and-Excitation Networks», CVPR 2018). La pérdida ordinal se mantiene.
- **V41 — Stochastic Weight Averaging (SWA):** promediar los pesos de las últimas 8 épocas del
  entrenamiento para aterrizar en una zona más plana del espacio de parámetros (mejor generalización; Izmailov, Podoprikhin, Garipov, Vetrov y Wilson, «Averaging Weights Leads to Wider Optima and Better Generalization», UAI 2018).
- **V42 — suavizado de etiquetas + SWA:** el suavizado 0.10 reparte un poco de masa de la clase correcta
  hacia las demás, atenuando la sobreconfianza (Szegedy, Vanhoucke, Ioffe, Shlens y Wojna, «Rethinking the Inception Architecture for Computer Vision», CVPR 2016; Müller, Kornblith y Hinton, «When Does Label Smoothing Help?», NeurIPS 2019), en la formulación de etiquetas suaves ordinales revisada de SORD (Díaz y Marathe, «Soft Labels for Ordinal Regression», CVPR 2019) combinada con la penalización de distancia; con enfriamiento cosenoidal del learning rate (Loshchilov y Hutter, «SGDR: Stochastic Gradient Descent with Warm Restarts», ICLR 2017) y 10
  checkpoints SWA. OOF standalone documentado: 0.613276.
- **V43 — pérdida de Cramér-von Mises:** la pérdida de mayor fundamento matemático de la familia:
  compara la **función de distribución acumulada** de la predicción contra la de la etiqueta
  (distancia cuadrática acumulada, la W₂ de Wasserstein en una dimensión; Cramér, «On the composition of elementary errors: First paper: Mathematical deductions», Scandinavian Actuarial Journal, 1928; von Mises, «Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik», Franz Deuticke, 1931; Székely y Rizzo, «Energy statistics: A class of statistics based on energy distances», Journal of Statistical Planning and Inference, 2013), lo que premia predicciones
  «cerca» en el orden incluso cuando fallan. OOF standalone documentado: 0.612840.

Cuatro de los seis componentes (V39-ordinal, V40, V42 y V43), más el HGB de V39 y el ensemble V30,
alimentan al campeón V48. La reproducibilidad es honesta: las
redes se entrenan con semillas fijas por fold y su salida de test es el promedio de los cinco modelos
(promedio que actúa como un mini-ensamble); el HGB y las redes de V38/V39 usan un modelo final entrenado
sobre todo el train. Entre entornos distintos (por ejemplo, GPU AMD vs NVIDIA) puede haber diferencias
numéricas del orden de 1e-6 —inherentes al hardware—, pero las decisiones y las métricas se mantienen
estables; el notebook lo declara explícitamente en su protocolo de reproducibilidad.

```python
# Componentes neuronales V38-V43: reconstruccion 5-fold completa (GPU si esta disponible).
componentes_nn = {}
referencias_nn = {"v38": None, "v39_ordinal": None, "v40": None, "v41": None,
                  "v42": 0.613276, "v43": 0.612840}
for nombre in ["v38", "v39_ordinal", "v40", "v41", "v42", "v43"]:
    t0 = time.time()
    oof_nn, test_nn = entrenar_componente(nombre, matriz, matriz_test, clase, fold_idx)
    componentes_nn[nombre] = (oof_nn, test_nn)
    cv_nn = macro_f1(clase, np.argmax(oof_nn * PESOS_PLUG_IN, axis=1))
    referencia = referencias_nn.get(nombre)
    texto_ref = f" (doc {referencia:.6f})" if referencia else ""
    print(f"  {nombre:12} OOF standalone {cv_nn:.6f}{texto_ref} | {time.time() - t0:.0f}s")
```

```python
# El HGB de V39 (miembro de arboles) y la figura de la familia neuronal completa.
t0 = time.time()
hgb_oof, hgb_test = hgb_componente(matriz, matriz_test, clase, fold_idx)
cv_hgb = macro_f1(clase, np.argmax(hgb_oof * PESOS_PLUG_IN, axis=1))
print(f"  v39_hgb      OOF standalone {cv_hgb:.6f} | {time.time() - t0:.0f}s")

nombres = list(componentes_nn) + ["v39_hgb"]
valores = [macro_f1(clase, np.argmax(componentes_nn[n][0] * PESOS_PLUG_IN, axis=1))
           for n in componentes_nn] + [cv_hgb]
fig, eje = plt.subplots(figsize=(9.5, 3.4))
eje.bar(nombres, valores, color="#2b6cb0", alpha=0.9)
eje.axhline(cv_v30_oof, color="#c53030", linestyle="--", linewidth=1,
            label=f"V30 lineal ({cv_v30_oof:.4f})")
for posicion, valor in enumerate(valores):
    eje.text(posicion, valor + 0.0004, f"{valor:.4f}", ha="center", fontsize=8)
eje.set_ylim(0.58, 0.63)
eje.set_ylabel("macro-F1 OOF standalone")
eje.set_title("Familia neuronal V38-V43 + HGB (cada uno por separado)")
eje.legend(fontsize=8)
plt.tight_layout()
plt.show()
```
