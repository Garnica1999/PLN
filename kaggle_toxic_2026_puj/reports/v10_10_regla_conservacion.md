# V10-10: regla de conservacion de caracteristicas correlacionadas

Generado por `experiments/v10_10_regla_conservacion.py` (V10-10 (regla de conservacion de caracteristicas correlacionadas)).

## 1. Por que hace falta este experimento

V10-09 elimino una de cada par correlacionado y empeoro mucho: quitar solo el 1,38 por
ciento de las caracteristicas, 283 de 20.441, costo -0.030626 de F1.

Ese numero es demasiado grande para atribuirlo al nivel de correlacion. Si al quitar 283
caracteristicas de 20.441 se pierde 0.03 de metrica, o esas 283 valen mucho, o **la regla
que decide cual se conserva esta descartando siempre la equivocada**.

Y hay una razon para sospechar de la regla. V10-09 conservaba de cada par la
caracteristica mas informativa **en solitario**. Para un modelo lineal eso es lo
contrario de lo que conviene: entre dos rasgos correlacionados, lo que vale es lo que
**cada uno aporta de lo que el otro no aporta**, y quedarse siempre con el mas fuerte
elimina sistematicamente la pieza que desambiguaba al primero.

## 2. Las cuatro reglas y el control que decide la lectura

| regla | que conserva | caracteristicas eliminadas | media | delta | mejoran |
|---|---|---|---|---|---|
| sin_eliminar | nada, es la referencia | 0 | 0.775910 | +0.000000 | 0/5 |
| fuerte_en_solitario | la de mayor |AUC - 0.5|, la de V10-09 | 317 | 0.744657 | -0.031252 | 0/5 |
| debil_en_solitario | la de **menor** |AUC - 0.5|, la contraprueba | 265 | 0.744031 | -0.031879 | 0/5 |
| menos_redundante | la que correlaciona menos con el resto | 275 | 0.742025 | -0.033885 | 0/5 |
| aleatoria | una al azar, el control decisivo | 273 | 0.747023 | -0.028887 | 0/5 |

## 3. Gate A, el control decide como leer todo lo demas

- delta de la regla fuerte: **-0.031252**
- delta de la regla aleatoria: **-0.028887**
- brecha entre las dos: **+0.002365**
- **la regla importa de verdad: False**

Si la regla aleatoria se parece a la regla fuerte, el efecto es de cuantas caracteristicas se quitan y no de cuales, y la via se cierra. Si la brecha supera 0.005, la regla importa de verdad y se evalua el gate B.

## 4. Gate B, mejora sobre V03

- mejor regla: **aleatoria**
- delta medio de al menos 0.002: **False**
- al menos 4 de 5 particiones: **False**
- **gate B superado: False**

## 5. Veredicto

**GATE B NO SUPERADO.** Ninguna regla de conservacion mejora a V03.

La conclusion honesta tiene dos partes y conviene no confundirlas. **El umbral esta
cerrado**: elegir que descartar por nivel de correlacion no mejora a V03, con ningun
umbral y con ninguna regla. Eso responde a la pregunta que se hizo.

Y hay una lectura adicional que V10-09 no podia dar: si la brecha entre la regla
fuerte y la aleatoria es grande, entonces **el mecanismo importa mas que el numero**,
y eso es informacion sobre como funciona el modelo, aunque no sirva para enviar nada.

Fecha: 2026-09-26T08:17:31+00:00. Duracion: 115.3 s.
