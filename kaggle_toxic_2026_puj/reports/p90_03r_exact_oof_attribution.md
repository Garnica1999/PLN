# P90-03-R - Recuperacion de la atribucion exacta OOF de V03

Generado por `experiments/p90_03r_reconstruccion_oof.py` y `experiments/p90_03rb_informe.py`.
Commit auditado: `f0e8065b (script historico)` | HEAD git en el manifiesto.
Fecha: 2026-10-10. Fase diagnostica autorizada: reconstruccion del pipeline historico por fold
para recuperar la atribucion exacta. Sin cambios en V03, sin test, sin holdout, sin submissions,
sin commits.

## 1. Resumen ejecutivo

**La reproduccion es exacta bit a bit.** Los cinco folds se reconstruyeron con el procedimiento
historico (`experiments/14_oof_corregido_v2.py`): vectorizadores TF-IDF word(1,2)+char_wb(3,5)
ajustados por fold y tres LogisticRegression (C=2.0, balanced, liblinear, random_state=42) por
fold. Las probabilidades reconstruidas coinciden con el OOF canonico con **error maximo 0.0**
(0 discordancias binarias; `corrected_cv` = 0.7772114525165721, diferencia 0.0).
Gate superado con las tolerancias predefinidas (rtol 1e-7, atol 1e-8, ccv 1e-7).

Con los modelos fold-specific reales, la descomposicion `logit = intercepto + suma(contribuciones)`
se verifica en las 4449 filas-etiqueta (residuo maximo 4.17e-14) y se
recuperan las contribuciones exactas de los 757 errores. **La comparacion con P90.03 reclasifica
varias conclusiones auxiliares**: se confirman el predominio de la acumulacion, el papel de
"perra", las contribuciones negativas de "la puta"/"mi perra" y las diferencias FP/FN; se
**contradicen** la dominancia del canal char en los FN y el conflicto word-char en FP de AG
(eran artefactos del modelo final in-sample). Decision final: **EXACT_OOF_ATTRIBUTION_RECOVERED**.

## 2. Alcance y autorizacion

Fase P90.03-R: reconstruir el pipeline historico por fold exclusivamente para recuperar la
informacion de atribucion no persistida. No cambia V03, thresholds, folds ni protocolo; no busca
hiperparametros; no genera submissions. La reconstruccion se uso como instrumento de atribucion
y el `corrected_cv` reconstruido solo como control de reproduccion.

## 3. Auditoria de fuentes

| fuente | sha256[:16] | rol |
|---|---|---|
| features/oof_v03_final.npz | 559265816ce4f631 | OOF canonico (intacto) |
| features/v2_oof_base_corrected.npz | 27dfa83736b97951 | padre del OOF (linaje M2, 0.0) |
| experiments/14_oof_corregido_v2.py | f0e8065bfbbc210e | procedimiento historico |
| src/splits.py | 755d3cef5b06c585 | folds y grupos (usados via OOF persistido) |
| src/data.py | 79c1f04d2be4bae6 | lectura del train |
| data/raw/toxic_train.csv | 3d40e7345b588860 | textos y etiquetas |
| reports/eda/fase2_lexico_inducido_agresividad.csv | abcb61f0d6f93f33 | categorizacion descriptiva |

Git: rama main, HEAD sin cambios (e43c594), cambios locales de P90.02/P90.03 preservados.
Software: Python 3.14.7, scikit-learn 1.8.0, numpy 1.26.4, scipy 1.17.1.
Los pliegues persistidos en el OOF se recomputaron con `pliegues_corregidos` y coinciden
exactamente (folds_recomputed_equal = True).

## 4. Ausencia de artefactos fold-specific originales

No existen joblib/pkl de los modelos ni vectorizadores por fold: el script historico los ajusta
en memoria y guarda solo probabilidades. Esta fase reconstruye esos artefactos
**exclusivamente para atribucion**, sin persistirlos como candidatos ni modificar el protocolo.

## 5. Procedimiento de reconstruccion (exacto)

Replicado de `14_oof_corregido_v2.py`: `train_dev` = filas dev del train en el orden original
(verificado identico al OOF: ids y y_true); folds = `pliegues` persistidos (1..5, cubren 1483
exactamente una vez); por fold: `TfidfVectorizer(analyzer=word, ngram_range=(1,2), min_df=2,
sublinear_tf=True)` + `TfidfVectorizer(analyzer=char_wb, ngram_range=(3,5), min_df=2,
sublinear_tf=True)` ajustados solo con el entrenamiento del fold; union = hstack(word, char);
tres `LogisticRegression(C=2.0, class_weight=balanced, max_iter=1000, solver=liblinear,
random_state=42)`; prediccion solo sobre la validacion del fold. Tamanos por fold: train
1186/1186/1187/1187/1186, val 297/297/296/296/297; features por fold 20441-21044.

## 6. Control de reproduccion

| criterio | resultado |
|---|---|
| allclose(rtol=1e-7, atol=1e-8) | True |
| error absoluto maximo | 0.000e+00 |
| error absoluto medio | 0.000e+00 |
| percentil 99 del error absoluto | 0.000e+00 |
| discordancias binarias | 0 |
| corrected_cv reconstruido | 0.7772114525165721 |
| diferencia de corrected_cv (tolerancia 1e-7) | 0.000e+00 |
| folds recomputados == persistidos | True |
| gate superado | True |

Criterio predefinido y no ampliado: `np.allclose(rtol=1e-7, atol=1e-8)`, 0 discordancias
binarias, |delta corrected_cv| <= 1e-7. **Reproduccion exacta: el error maximo es 0.0 en todas
las folds y etiquetas.**

## 7. Metodologia de atribucion exacta

Para cada fila de validacion del fold que la predijo: `logit = intercepto + suma(x_k*coef_k)`;
`probabilidad = sigmoid(logit)`; descomposicion por feature (`contribucion = tfidf * coef`) con
canales word/char separados; residuo de descomposicion por fila (maximo
4.17e-14). Las tablas guardan token/n-grama, canal, valor TF-IDF,
coeficiente y contribucion local de las features top-10 positivas y negativas de cada
fila-etiqueta (`p90_03r_feature_contributions.csv`, 4449 filas) y de cada error
(`p90_03r_exact_oof_attribution.csv`, 757 filas). Recuentos verificados: 757 celdas / 441
tweets; estratos {'HS': [40, 87, 189], 'TR': [13, 40, 111], 'AG': [31, 80, 166]}.

## 8. Resultados HS

- **HS FP (173)**: word medio +0.332, char medio +0.452
  (ambos positivos: el fold modelo cometio el error); top-1 del lexico 88/173;
  acumulacion 108/173; conflicto word-char 60/173;
  char dominante 73/173 (42 por ciento).
  Top positivos: perra 32, los 20, zorra 8, cllate 8. Top negativos: de 62, la 49, https 38,
  en 31, mi perra 9.
- **HS FN (143)**: word medio -0.450, char medio -0.129
  (ambos negativos: la decision real fue a la baja); acumulacion 113/143;
  conflicto word-char 69/143; lexico en top-10 61/143.
  Top negativos: de 66, la 63, en 44, el 37, https 23, rabe 10, polla 10.

## 9. Resultados TR

- **TR FP (82)**: word +0.981, char +0.914;
  top-1 del lexico 63/82 (77 por ciento); mismo signo
  80/82; conflicto word-char solo 2/82.
  Top negativos: de 24, la 23, la puta 8, mi perra 8.
- **TR FN (82)**: word +0.009, char -0.024
  (margenes netos casi nulos); mismo signo 58/82; conflicto word-char
  24/82. Top positivos: puta 15, perra 10, sudaca 7; top negativos:
  de 41, la 23, el 20, en 19, https 14.

## 10. Resultados AG

- **AG FP (135)**: word +0.623, char +0.615;
  top-1 del lexico 85/135; acumulacion 82/135;
  conflicto word-char 19/135 (14 por ciento, muy inferior al auxiliar).
- **AG FN (142)**: word -0.310, char +0.037;
  conflicto word-char 79/142 (56 por ciento); acumulacion
  116/142; top negativos: la 45, en 44, de 41, el 34, https 22,
  rabe 8, polla 8.

## 11. Firmas exactas

| etiqueta | tipo | firma | n |
|---|---|---|---|
| HS | FP | acumulacion de muchas features | 65 |
| HS | FP | conflicto positivas-negativas | 35 |
| HS | FP | conflicto word-char | 60 |
| HS | FP | lexico dominante sin contexto | 6 |
| HS | FP | no interpretable con los artefactos disponibles | 6 |
| HS | FP | pocas features dominantes | 1 |
| HS | FN | acumulacion de muchas features | 61 |
| HS | FN | conflicto positivas-negativas | 5 |
| HS | FN | conflicto word-char | 69 |
| HS | FN | sin evidencia hostil activada | 8 |
| TR | FP | acumulacion de muchas features | 28 |
| TR | FP | conflicto positivas-negativas | 31 |
| TR | FP | conflicto word-char | 2 |
| TR | FP | lexico dominante sin contexto | 7 |
| TR | FP | no interpretable con los artefactos disponibles | 12 |
| TR | FP | pocas features dominantes | 2 |
| TR | FN | acumulacion de muchas features | 40 |
| TR | FN | conflicto positivas-negativas | 14 |
| TR | FN | conflicto word-char | 24 |
| TR | FN | pocas features dominantes | 2 |
| TR | FN | sin evidencia hostil activada | 2 |
| AG | FP | acumulacion de muchas features | 70 |
| AG | FP | conflicto positivas-negativas | 24 |
| AG | FP | conflicto word-char | 19 |
| AG | FP | lexico dominante sin contexto | 10 |
| AG | FP | no interpretable con los artefactos disponibles | 8 |
| AG | FP | pocas features dominantes | 4 |
| AG | FN | acumulacion de muchas features | 50 |
| AG | FN | conflicto positivas-negativas | 9 |
| AG | FN | conflicto word-char | 79 |
| AG | FN | sin evidencia hostil activada | 4 |

## 12. Casos clave

- **perra como top-1 positivo de FP**: HS 32, TR 27,
  AG 31; repeticion por fold: HS f1:6; f2:5; f3:9; f4:7; f5:5; TR f1:4; f2:5; f3:6; f4:9; f5:3;
  AG f1:3; f2:7; f3:8; f4:8; f5:5.
- **la puta en top-5 negativos de errores**: HS 11, TR
  12, AG 8. **mi perra**: HS
  9, TR 9, AG 8.
- **Caso 23839 (HS FP fronterizo, d=4.6e-05)**: logit -0.240977 = intercepto -0.253207 +
  word -0.143680 + char +0.155910. "perra" (word) contribuye +0.558860 y "mi perra" (word,
  coef -1.6009) contribuye -0.551078: el bigrama afiliativo casi cancela al insulto y la
  prediccion queda a 4.6e-05 del threshold. Mecanismo exacto del FP afiliativo, documentado.

## 13. Errores alejados del threshold (exactos)

| etiqueta | tipo | n lejos | acumulacion | pocas | conflicto wc | top features | top-1 por fold |
|---|---|---|---|---|---|---|---|
| HS | FP | 112 | 68 | 1 | 18 | perra:27; los:9; zorra:8; eres:4; novia es:3; subsaharianos:3; cllate:3; guarra: | f1:perra(5); f2:zorra(3); f3:perra(8); f4:perra(7); f5:perra(4) |
| HS | FN | 77 | 65 | 0 | 22 | los:13; al:4; se:4; con:4; si:3; nos:2; es lo:2; mi polla:2 | f1:los(3); f2:los(4); f3:los(3); f4:perra(2); f5:los(2) |
| TR | FP | 54 | 15 | 1 | 0 | perra:20; zorra:6; te:5; eres:5; puta:5; callate puta:2; novia es:2; callate:2 | f1:te(3); f2:perra(4); f3:perra(4); f4:perra(6); f5:perra(3) |
| TR | FN | 57 | 43 | 1 | 22 | puta:9; perra:6; te:5; sudaca:5; zorra:3; al:2; mi:2; mi polla:2 | f1:zorra(1); f2:puta(4); f3:tu(1); f4:sudaca(3); f5:puta(3) |
| AG | FP | 77 | 40 | 3 | 1 | perra:26; zorra:9; guarra:4; te:3; callate:3; cllate:3; callate puta:2; tu eres: | f1:perra(3); f2:perra(6); f3:perra(6); f4:perra(7); f5:perra(4) |
| AG | FN | 89 | 72 | 0 | 45 | al:9; puta:4; nos:4; estos:4; perra:4; los:3; si:3; te:3 | f1:si(3); f2:estos(2); f3:que(2); f4:al(3); f5:al(3) |

## 14. Slices (exactas)

| slice | label | errores dentro | word medio | char medio | activas dentro | activas fuera |
|---|---|---|---|---|---|---|
| URLs | HS | 112 | -0.26439 | 0.205212 | 226.99 | 167.14 |
| URLs | TR | 39 | 0.079164 | 0.170965 | 185.03 | 155.16 |
| URLs | AG | 102 | -0.118495 | 0.303418 | 235.3 | 171.07 |
| hashtags | HS | 37 | -0.076919 | 0.217364 | 215.95 | 184.7 |
| hashtags | TR | 12 | 0.680407 | 0.291214 | 189.42 | 160.12 |
| hashtags | AG | 30 | 0.064146 | 0.343704 | 229.7 | 190.48 |
| colectivos_superficie | HS | 100 | -0.130557 | 0.133144 | 247.5 | 160.97 |
| colectivos_superficie | TR | 25 | 0.184826 | 0.079141 | 220.88 | 151.72 |
| colectivos_superficie | AG | 95 | -0.021853 | 0.18693 | 261.33 | 159.96 |
| menciones | HS | 123 | 0.070988 | 0.288437 | 184.36 | 190.9 |
| menciones | TR | 83 | 0.698218 | 0.591258 | 162.53 | 161.99 |
| menciones | AG | 107 | 0.379841 | 0.48005 | 188.63 | 198.56 |
| mencion_inicial | HS | 100 | 0.126689 | 0.316838 | 172.41 | 195.74 |
| mencion_inicial | TR | 78 | 0.737754 | 0.639959 | 157.71 | 166.4 |
| mencion_inicial | AG | 88 | 0.469192 | 0.523824 | 170.81 | 205.86 |
| cita | HS | 10 | -0.215637 | 0.145595 | 292.2 | 184.96 |
| cita | TR | 1 | -0.309544 | -0.506347 | 335.0 | 161.2 |
| cita | AG | 6 | -0.831332 | 0.996072 | 237.5 | 193.78 |

En los errores de HS dentro de URLs/colectivos el canal word es **negativo** en media
(URLs HS -0.264; colectivos HS -0.131) mientras el char es positivo: la senal que empuja el FP
en esas regiones es mayoritariamente de caracteres, con mas features activas que fuera. Grupos
pequenos (hashtags TR 12) con incertidumbre alta.

## 15. Comparacion con P90.03

| observacion P90.03 (auxiliar) | valor auxiliar | valor exacto | clasificacion |
|---|---|---|---|
| Acumulacion domina frente a pocas features dominantes | aux: FN 76-86 por ciento acumulacion, 0-3 pocas | exacto: FN 79/67/82 por ciento; FP 62/34/61 por ciento; pocas 0-5 | CONFIRMED_BY_EXACT_OOF |
| Conflicto word-char frecuente en FP de HS | aux: 75/173 (43 por ciento) | exacto: 60/173 (35 por ciento) | PARTIALLY_CONFIRMED |
| Conflicto word-char frecuente en FP de AG | aux: 68/135 (50 por ciento) | exacto: 19/135 (14 por ciento) | NOT_CONFIRMED |
| Conflicto word-char en FN de HS/AG bajo | aux: 17 y 3 por ciento | exacto: 69/143 (48) y 79/142 (56) | CONTRADICTED |
| Canal char dominante en FN de HS/AG | aux: 89/143 (62) y 79/142 (56) | exacto: 2/143 (1.4) y 30/142 (21) | CONTRADICTED |
| 'perra' como top-1 positivo en FP | aux: 32/27/30 (HS/TR/AG) | exacto: 32/27/31 | CONFIRMED_BY_EXACT_OOF |
| Contribuciones negativas de 'la puta' y 'mi perra' | aux: presentes (mi perra top-neg en 9/8/7 FP) | exacto: mi perra en top-5 neg de errores 9/9/8; la puta 11/12/8 | CONFIRMED_BY_EXACT_OOF |
| URLs/colectivos: mas features activas y mezcla distinta | aux: URLs HS 248.9 vs 185.0; colectivos 272.7 vs 177.5 | exacto: URLs HS 227.0 vs 167.1; colectivos 247.5 vs 161.0 | CONFIRMED_BY_EXACT_OOF |
| TR FN con canales del mismo signo (100 por ciento) | aux: 82/82 | exacto: 58/82 (71 por ciento) | NOT_CONFIRMED |
| FP vs FN: firmas de canal opuestas | aux: FP con word medio negativo (modelo final no cometia el error) | exacto: FP word/char medios +0.33/+0.45 (HS), +0.98/+0.91 (TR), +0.62/+0.61 (AG); FN -0.45/-0.13 (HS), +0.01/-0.02 (TR), -0.31/+0.04 (AG) | CONFIRMED_BY_EXACT_OOF |

La diferencia clave entre ambas atribuciones: la auxiliar usaba un modelo entrenado con las
mismas filas que intentaba explicar (in-sample), por lo que a menudo **no cometia el error** y
sus canales aparecian en direcciones distorsionadas (p. ej. word medio negativo en FP). La
atribucion exacta usa el modelo del fold que realmente produjo cada prediccion y muestra las
direcciones reales: FP con ambos canales positivos y FN con canales negativos o casi nulos.

## 16. Limitaciones

- La reconstruccion es exacta en este entorno (sklearn 1.8.0); versiones distintas de las
  librerias podrian introducir diferencias minimas (el gate lo detectaria).
- Las contribuciones describen la decision del modelo lineal; no son causalidad semantica.
- El analisis de slices pequenas (hashtags, cita) tiene incertidumbre alta.
- No se persisten los modelos reconstruidos (solo sus checksums); la reproduccion se controla
  por el gate y por los checksums registrados.

## 17. Decision final

**EXACT_OOF_ATTRIBUTION_RECOVERED.** Reproduccion exacta (error 0.0), descomposicion verificada
(residuo maximo 4.17e-14) y 757 errores con atribucion exacta por
fold. Las conclusiones confirmadas/contradichas estan en la seccion 15. Esta fase no propone
modelos ni hipotesis nuevas: su resultado es una explicacion fiable de las predicciones
historicas de V03.

