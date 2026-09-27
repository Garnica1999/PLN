"""V18 · P5 — Destilacion del conocimiento de frontera, con cross-fitting estricto.

P1, P3 y P4 cerraron en negativo, y P5 esta autorizado excepcionalmente porque **no repite ninguno de
los tres mecanismos**:

- P1 **redistribuyo** masa entre clases y pago la mejora de frontera fuera de ella, sobre la clase
  facil vecina.
- P3 **puso a competir** las dos fronteras, que estan acopladas por la clase `25-34`.
- P4 anadio una senal auxiliar **demasiado debil** para desplazar el optimo global, porque su objetivo
  era la etiqueta verdadera, que ya esta en la entropia cruzada.

P5 plantea algo distinto: **transferir** al modelo general el conocimiento de un especialista que ya
demostro resolver las fronteras, **sin routing en inferencia**. Y el contexto lo respalda: V17 mostro
que la senal del especialista es fuerte y que el problema del gating era no poder distinguir cuando
era correcto. Aqui no hay nada que distinguir en inferencia: la senal se aprende durante el
entrenamiento.

**La diferencia tecnica con P4, que es la que hace este experimento nuevo.** En P4 el termino
auxiliar se media contra la **etiqueta verdadera**, que es informacion que la entropia cruzada ya
contiene, y por eso era redundante. Aqui el objetivo auxiliar es la **prediccion suave del
teacher**, que es informacion nueva: el teacher ya vio las filas del par y su decision aporta algo que
el modelo global no consigue deducir por si solo.

**Que teacher se usa, y por que este.** El especialista ya validado es el binario sobre las **279
variables de V12**, medido en P0' con el protocolo completo de cinco folds: **0,7136** en
`18-24|25-34` y **0,7493** en `25-34|35-49`. Se usa ese y no el de TF-IDF disperso (0,6864 y 0,7218,
el de E-028a) por dos razones medidas: es mejor, y esta en **la misma representacion que el student**,
que es la condicion necesaria para que el student pueda expresar lo que se le ensena. Un teacher en
otra representacion ensenaria algo que el student no puede representar, y el experimento no
responderia a la pregunta.

**Cross-fitting, que es la parte donde este tipo de experimento se contamina.** El protocolo es
anidado en tres niveles, exactamente el que pide el brief:

```
para cada fold exterior:
    dividir outer_train en inner folds (agrupados por duplicados)
    para cada inner fold k:
        teacher_k se ajusta en inner_train
        teacher_k predice SOLO inner_validation
    teacher_OOF cubre TODO outer_train, y cada ejemplo fue predicho
        por un teacher que no lo uso para entrenarse
    el student se entrena en outer_train con:
        etiquetas verdaderas (objetivo principal) + teacher_OOF (auxiliar)
    el student se evalua SOLO en outer_validation
```

El teacher ajustado sobre todo `outer_train` **si** predice `outer_validation`, y esas predicciones
se usan **solo para diagnostico**: nunca entran en el entrenamiento del student. Publicar el
diagnostico del teacher sobre `outer_validation` es justamente lo que permite responder si el student
recupera parte de su rendimiento sin necesitar un gate.

**Las señales del teacher**, todas suaves y todas de margen, en linea con lo que pide el brief: la
probabilidad de la clase superior de cada pareja, que es equivalente a su margen por la sigmoida. Se
aplican **solo a las filas de su pareja**, porque fuera de ella el teacher extrapola y su salida no
significa nada. **No hay pseudo-etiqueta dura en ningun punto, y la etiqueta verdadera nunca se
reemplaza**: es siempre el objetivo principal.

**Cuadrícula, y donde se selecciona.** Se evaluan seis combinaciones de peso de destilacion y
temperatura, y **se eligen dentro del train exterior**, en una particion interior agrupada, por
macro-F1 de seis clases. La validacion exterior no participa en la eleccion ni en el entrenamiento.
La temperatura afecta a la senal del teacher antes de entrar en la perdida, que es donde tiene
sentido en una destilacion de senal suave.

**Comparacion obligatoria**: V12, el control del mismo pipeline student **sin** destilacion, y el
student con destilacion. Se reportan pooled, media de folds, delta, IC95 pareado, folds positivos, F1
por clase, las dos fronteras, matriz de confusion, el cambio en la clase `25-34`, rescate y dano, y el
comportamiento del student frente al teacher.

Entregables: ``experiments/v18_p5_destilacion.json``, ``reports/v18_p5_destilacion.md``,
``reports/v18_p5_destilacion.csv``, ``reports/v18_distillation.csv`` y dos figuras.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402
from v13_p1b_fusion_tokens import (PARES, bootstrap_diferencia, logreg,  # noqa: E402
                                   metricas_frontera, metricas_globales)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v18_p5_destilacion.md"
TABLA = cfg.REPORTS / "v18_p5_destilacion.csv"
DISTILACION = cfg.REPORTS / "v18_distillation.csv"
RESULTADO = cfg.EXPERIMENTS / "v18_p5_destilacion.json"
#: Las dos fronteras centrales por indice de clase.
FRONTERAS_INDICE = ((1, 2), (2, 3))
#: Cuadrícula **predefinida** de peso de destilacion y temperatura. Se evalúan las seis y se elige
#: dentro del train exterior; nunca con la validacion exterior.
CUADRICULA = ((0.3, 1.0), (0.3, 2.0), (0.3, 4.0), (1.0, 1.0), (1.0, 2.0), (1.0, 4.0))
#: Inner folds para el cross-fitting del teacher. Cinco, agrupados, como el resto del proyecto.
INNER_FOLDS = 5
#: Fraccion del train exterior que se aparta para elegir la combinacion de la cuadricula.
FRACCION_SELECCION = 0.25
#: Entrenamiento pre-registrado del student, identico al de P3 y P4 para que los tres sean
#: comparables.
EPOCAS = 30
LOTE = 512
TASA_APRENDIZAJE = 1e-3
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
GATE_EXPLORATORIO = 0.0015
GATE_ADOPCION = 0.003
FOLDS_POSITIVOS_MINIMOS = 4
SEMILLA = cfg.SEED
ETIQUETAS_ENTERO = list(range(len(cfg.CLASSES)))


def cruzar(inner: np.ndarray, clusters: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reparte filas en inner folds AGRUPADOS por duplicados.

    Es la pieza que garantiza que la senal del teacher no ve, en ninguna de sus partes, la fila que
    predice. Si un grupo de casi duplicados cruzara entre inner_train e inner_validation, el teacher
    habria visto una version de la fila que luego predice, y la destilacion estaria contaminada sin
    que nada lo delatara.
    """
    generador = np.random.default_rng(SEMILLA)
    grupos_unicos = np.unique(clusters[inner])
    generador.shuffle(grupos_unicos)
    reparto = np.array_split(grupos_unicos, INNER_FOLDS)
    bloques = []
    for grupo in reparto:
        en_validacion = set(grupo.tolist())
        mascara = np.array([g in en_validacion for g in clusters[inner]])
        bloques.append((inner[~mascara], inner[mascara]))
    return bloques


def senal_del_teacher(matriz: np.ndarray, y: np.ndarray, filas_entrenamiento: np.ndarray,
                      filas_validacion: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Ajusta un teacher binario por pareja y devuelve su probabilidad en las filas pedidas.

    El teacher se ajusta **solo** en las filas de su pareja dentro de `filas_entrenamiento`. La
    probabilidad que devuelve es la de la clase superior de la pareja, y es una senal continua
    equivalente al margen por la sigmoida, no una etiqueta dura.
    """
    salida: dict[tuple[int, int], np.ndarray] = {}
    for inferior, superior in FRONTERAS_INDICE:
        # `FRONTERAS_INDICE` son indices de clase y `y` son cadenas, luego la comparacion va con los
        # nombres. Comparar indice contra cadena daba falso siempre y dejaba al teacher sin filas.
        nombre_inferior, nombre_superior = cfg.CLASSES[inferior], cfg.CLASSES[superior]
        modelo = logreg()
        filas = filas_entrenamiento[(y[filas_entrenamiento] == nombre_inferior)
                                    | (y[filas_entrenamiento] == nombre_superior)]
        if len(filas) == 0:
            raise RuntimeError(f"el inner train no tiene filas de la pareja {nombre_inferior}"
                               f" | {nombre_superior}")
        modelo.fit(matriz[filas], (y[filas] == nombre_superior).astype(int))
        salida[(inferior, superior)] = modelo.predict_proba(matriz[filas_validacion])[:, 1]
    return salida


def perdida_destilacion(logits: torch.Tensor, indice_real: torch.Tensor,
                        senal: dict[tuple[int, int], torch.Tensor], peso: float,
                        temperatura: float) -> torch.Tensor:
    """Entropia cruzada verdadera mas la destilacion de las dos senales de frontera.

    La senal del teacher se suaviza con la temperatura antes de entrar, y la perdida solo se aplica
    a las filas de cada pareja, porque fuera de ella el teacher extrapola. La etiqueta verdadera es
    siempre el objetivo principal: aqui no se sustituye por nada.
    """
    perdida = torch.nn.functional.cross_entropy(logits, indice_real)
    if peso == 0.0:
        return perdida
    total = logits.new_zeros(())
    filas = 0
    for inferior, superior in FRONTERAS_INDICE:
        mascara = (indice_real == inferior) | (indice_real == superior)
        if not bool(mascara.any()):
            continue
        objetivo = (senal[(inferior, superior)][mascara] / temperatura).clamp(1e-4, 1 - 1e-4)
        diferencia = logits[mascara, superior] - logits[mascara, inferior]
        total = total + torch.nn.functional.binary_cross_entropy_with_logits(diferencia, objetivo)
        filas += int(mascara.sum())
    return perdida + peso * total / max(filas, 1)


def entrenar_student(matriz_train: np.ndarray, indice_train: np.ndarray, matriz_validacion: np.ndarray,
                     senal: dict[tuple[int, int], np.ndarray], peso: float, temperatura: float,
                     generador: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Entrena el student de seis clases y devuelve probabilidades de train y de validacion."""
    escalador = StandardScaler().fit(matriz_train)
    datos_train = torch.tensor(escalador.transform(matriz_train), dtype=torch.float32,
                               device=DISPOSITIVO)
    datos_validacion = torch.tensor(escalador.transform(matriz_validacion), dtype=torch.float32,
                                   device=DISPOSITIVO)
    objetivo = torch.tensor(indice_train, dtype=torch.long, device=DISPOSITIVO)
    senal_t = {par: torch.tensor(valores, dtype=torch.float32, device=DISPOSITIVO)
               for par, valores in senal.items()}
    modelo = torch.nn.Linear(matriz_train.shape[1], len(cfg.CLASSES)).to(DISPOSITIVO)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    permutacion = torch.randperm(len(datos_train), generator=generador).to(DISPOSITIVO)
    for _ in range(EPOCAS):
        modelo.train()
        for desde in range(0, len(permutacion), LOTE):
            indices = permutacion[desde:desde + LOTE]
            optimizador.zero_grad()
            perdida = perdida_destilacion(modelo(datos_train[indices]), objetivo[indices],
                                           {par: valor[indices] for par, valor in senal_t.items()},
                                           peso, temperatura)
            perdida.backward()
            optimizador.step()
    modelo.eval()
    with torch.no_grad():
        return (torch.softmax(modelo(datos_train), dim=1).cpu().numpy().astype(float),
                torch.softmax(modelo(datos_validacion), dim=1).cpu().numpy().astype(float))


def main() -> int:
    """Destila el conocimiento de los dos especialistas de frontera en el student global."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    pred_v12 = np.array(cfg.CLASSES)[np.argmax(referencia, axis=1)]

    # El performance del teacher sobre las filas de cada frontera, ya medido en P0' con el protocolo
    # completo. Se usa como referencia de "cuanto hay disponible" y no se recalcula aqui.
    teacher_frontera = {"18-24|25-34": 0.7136, "25-34|35-49": 0.7493}
    print(f"[referencia] V12 {metricas_referencia['macro_f1']} | fronteras V12 "
          f"{fronteras_referencia['18-24|25-34']['accuracy']} y "
          f"{fronteras_referencia['25-34|35-49']['accuracy']} | teacher validado en frontera "
          f"{teacher_frontera}", flush=True)

    generador = torch.Generator().manual_seed(SEMILLA)
    # El MLP no cambia: se entrena una vez por fold y se reutiliza en los tres brazos.
    predicciones_mlp = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        red = EncodedMLP().fit(base[entrenamiento], y[entrenamiento])
        probabilidades = red.model.predict_proba(red.scaler.transform(base[validacion]))
        completa = np.zeros((len(validacion), len(cfg.CLASSES)))
        for posicion, clase_nombre in enumerate(red.encoder.classes_):
            completa[:, cfg.CLASSES.index(clase_nombre)] = probabilidades[:, posicion]
        predicciones_mlp[validacion] = completa

    resultados: list[dict] = []
    filas: list[dict] = []
    for nombre, peso, temperatura in (("control_sin_destilacion", 0.0, 1.0),
                                      *[("student_destilado", p, t) for p, t in CUADRICULA]):
        prob_student = np.zeros((len(y), len(cfg.CLASSES)))
        combinaciones_elegidas = []
        # Las claves van con los NOMBRES de clase, que es como se rellenan mas abajo.
        exactitud_teacher_validacion = {f"{cfg.CLASSES[a]}|{cfg.CLASSES[b]}": []
                                        for a, b in FRONTERAS_INDICE}
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])

            # ---- 1) teacher_OOF sobre TODO outer_train, por cross-fitting en inner folds
            # `entrenamiento` es una concatenacion de folds y **no esta ordenado**, asi que se usa un
            # mapa de indice global a posicion en lugar de `searchsorted`, que daria posiciones sin
            # sentido sobre un array desordenado.
            posicion = np.full(len(y), -1, dtype=np.int64)
            posicion[entrenamiento] = np.arange(len(entrenamiento))
            senal_oof = {par: np.zeros(len(entrenamiento)) for par in FRONTERAS_INDICE}
            for inner_train, inner_validacion in cruzar(entrenamiento, clusters):
                if len(inner_validacion) == 0 or len(inner_train) == 0:
                    continue
                senal = senal_del_teacher(base, y, inner_train, inner_validacion)
                posiciones = posicion[inner_validacion]
                assert (posiciones >= 0).all(), "una fila de inner_validacion no esta en outer_train"
                for par in FRONTERAS_INDICE:
                    senal_oof[par][posiciones] = senal[par]

            # ---- 2) seleccion de la cuadricula DENTRO del train exterior
            if peso == 0.0:
                elegido = (0.0, 1.0)
            else:
                generador_seleccion = np.random.default_rng(SEMILLA)
                grupos = np.unique(clusters[entrenamiento])
                generador_seleccion.shuffle(grupos)
                corte = int(len(grupos) * FRACCION_SELECCION)
                en_seleccion = set(grupos[:corte].tolist())
                mascara = np.array([g in en_seleccion for g in clusters[entrenamiento]])
                filas_seleccion = entrenamiento[mascara]
                filas_ajuste = entrenamiento[~mascara]
                senal_ajuste = {par: senal_oof[par][~mascara] for par in FRONTERAS_INDICE}
                senal_seleccion = {par: senal_oof[par][mascara] for par in FRONTERAS_INDICE}
                mejor, mejor_valor = (peso, temperatura), -np.inf
                for candidato_peso, candidata_t in CUADRICULA:
                    # El student se ajusta en el bloque de ajuste con la senal **de ese bloque**, y
                    # puntua en el bloque de seleccion. Pasar aqui la senal del bloque de seleccion
                    # daria longitudes distintas y el loto se indexaria fuera de rango.
                    _, prob = entrenar_student(base[filas_ajuste], clase[filas_ajuste],
                                              base[filas_seleccion], senal_ajuste,
                                              candidato_peso, candidata_t, generador)
                    valor = float(f1_score(y[filas_seleccion],
                                           np.array(cfg.CLASSES)[np.argmax(prob, axis=1)],
                                           average="macro", labels=cfg.CLASSES))
                    if valor > mejor_valor:
                        mejor, mejor_valor = (candidato_peso, candidata_t), valor
                elegido = mejor
            # El student final se entrena sobre TODO outer_train, luego necesita la senal que cubre
            # todo outer_train. La seleccion solo ha servido para fijar (lambda, T).
            senal_elegida = {par: senal_oof[par] for par in FRONTERAS_INDICE}

            # ---- 3) student final en TODO outer_train, evaluado solo en outer_validation
            _, prob_validacion = entrenar_student(base[entrenamiento], clase[entrenamiento],
                                                  base[validacion], senal_elegida, elegido[0],
                                                  elegido[1], generador)
            prob_student[validacion] = prob_validacion
            combinaciones_elegidas.append({"fold": fold["fold"], "lambda": elegido[0],
                                           "temperatura": elegido[1]})

            # ---- 4) diagnostico: teacher ajustado sobre TODO outer_train, que predice
            # outer_validation. Estas predicciones NO entrenan al student en ningun momento.
            if peso != 0.0:
                senal_diagnostico = senal_del_teacher(base, y, entrenamiento, validacion)
                for inferior, superior in FRONTERAS_INDICE:
                    nombre_inferior, nombre_superior = cfg.CLASSES[inferior], cfg.CLASSES[superior]
                    etiqueta = f"{nombre_inferior}|{nombre_superior}"
                    # Nombre distinto del de la lista de resultados: usar `filas` aqui la
                    # sombreaba y rompia el `append` final.
                    filas_frontera = ((y[validacion] == nombre_inferior)
                                      | (y[validacion] == nombre_superior))
                    decision = np.where(senal_diagnostico[(inferior, superior)] > 0.5,
                                        nombre_superior, nombre_inferior)
                    # `decision` cubre toda la validacion exterior, luego hay que indexarlo con la
                    # mascara de las filas del par antes de comparar.
                    exactitud_teacher_validacion[etiqueta].append(
                        float((decision[filas_frontera]
                               == y[validacion][filas_frontera]).mean()))
            print(f"    [{nombre}] fold {fold['fold']}: lambda {elegido[0]}, T {elegido[1]}",
                  flush=True)

        combinada = 0.5 * prob_student + 0.5 * predicciones_mlp
        metricas = metricas_globales(combinada, clase)
        metricas_cabeza = metricas_globales(prob_student, clase)
        fronteras = metricas_frontera(combinada, clase, PARES)
        por_fold = [float(f1_score(y[fold["val_idx"]],
                                    np.array(cfg.CLASSES)[np.argmax(combinada[fold["val_idx"]],
                                                                    axis=1)],
                                    average="macro", labels=cfg.CLASSES)) for fold in folds]
        por_fold_referencia = [float(f1_score(y[fold["val_idx"]],
                                              np.array(cfg.CLASSES)[np.argmax(referencia[fold["val_idx"]],
                                                                               axis=1)],
                                              average="macro", labels=cfg.CLASSES)) for fold in folds]
        folds_positivos = sum(1 for a, b in zip(por_fold, por_fold_referencia) if a > b)
        delta = metricas["macro_f1"] - metricas_referencia["macro_f1"]
        intervalo = bootstrap_diferencia(referencia, combinada, clase, len(cfg.CLASSES))
        pred_student = np.array(cfg.CLASSES)[np.argmax(combinada, axis=1)]
        rescate = int(((pred_v12 != y) & (pred_student == y)).sum())
        dano = int(((pred_v12 == y) & (pred_student != y)).sum())
        confusion = confusion_matrix(clase, np.argmax(combinada, axis=1), labels=ETIQUETAS_ENTERO)
        criterios = {"efecto_minimo": bool(delta >= GATE_ADOPCION),
                     "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
                     "significacion": bool(intervalo[0] > 0)}
        resultado = {
            "brazo": nombre, "lambda": peso, "temperatura": temperatura,
            "combinaciones_elegidas": combinaciones_elegidas,
            "macro_cabeza": metricas_cabeza["macro_f1"],
            "macro_combinada": metricas["macro_f1"], "accuracy": metricas["accuracy"],
            "delta": round(delta, 6), "media_folds": round(float(np.mean(por_fold)), 6),
            "media_folds_referencia": round(float(np.mean(por_fold_referencia)), 6),
            "por_fold": [round(v, 4) for v in por_fold], "folds_positivos": folds_positivos,
            "bootstrap_ic95": [round(v, 6) for v in intervalo],
            "f1_por_clase": metricas["f1_por_clase"],
            "f1_clase_25_34": metricas["f1_por_clase"]["25-34"],
            "f1_clase_25_34_referencia": metricas_referencia["f1_por_clase"]["25-34"],
            "frontera_18_24_25_34": fronteras["18-24|25-34"]["accuracy"],
            "frontera_25_34_35_49": fronteras["25-34|35-49"]["accuracy"],
            "rescate": rescate, "dano": dano, "neto": rescate - dano,
            "matriz_confusion": confusion.tolist(),
            "teacher_por_fold_validacion": {k: [round(v, 4) for v in valores]
                                            for k, valores in exactitud_teacher_validacion.items()},
            "criterios": criterios, "adopta": all(criterios.values()),
        }
        resultados.append(resultado)
        filas.append({"brazo": nombre, "lambda": peso, "temperatura": temperatura,
                      "macro_combinada": metricas["macro_f1"], "delta": round(delta, 4),
                      "folds_positivos": folds_positivos,
                      "frontera_1": resultado["frontera_18_24_25_34"],
                      "frontera_2": resultado["frontera_25_34_35_49"],
                      "f1_25_34": resultado["f1_clase_25_34"],
                      "rescate": rescate, "dano": dano})
        print(f"[{nombre}] {metricas['macro_f1']} | delta {delta:+.4f} | folds {folds_positivos}/5 | "
              f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | fronteras {resultado['frontera_18_24_25_34']} "
              f"y {resultado['frontera_25_34_35_49']} | rescate {rescate:,} dano {dano:,} | adopta "
              f"{resultado['adopta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA, index=False)
    pd.DataFrame([{"brazo": r["brazo"], "frontera": k,
                   "student": v, "v12": fronteras_referencia[k]["accuracy"],
                   "teacher_validado": teacher_frontera.get(k),
                   "recuperacion_sobre_disponible": round(
                       (v - fronteras_referencia[k]["accuracy"])
                       / (teacher_frontera[k] - fronteras_referencia[k]["accuracy"]), 4)
                   if teacher_frontera.get(k) else None}
                  for r in resultados
                  for k, v in (("18-24|25-34", r["frontera_18_24_25_34"]),
                               ("25-34|35-49", r["frontera_25_34_35_49"]))]).to_csv(
                       DISTILACION, index=False)

    # ---------------------------------------------------------------- figuras
    figura = cfg.FIGURES / "v18_p5_destilacion.png"
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5.5))
    nombres = [r["brazo"] + (f" T{r['temperatura']:.0f}" if r["lambda"] else "") for r in resultados]
    colores = ["#1a9850" if r["adopta"] else "#2166ac" for r in resultados]
    eje = ejes[0]
    eje.bar(range(len(nombres)), [r["macro_combinada"] for r in resultados], color=colores)
    eje.axhline(metricas_referencia["macro_f1"], color="grey", linestyle=":", linewidth=1)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, rotation=25, fontsize=7)
    eje.set_title("Macro-F1 del sistema combinado", fontsize=10)
    eje = ejes[1]
    for posicion, resultado in enumerate(resultados):
        eje.plot(cfg.CLASSES, [resultado["f1_por_clase"][c] for c in cfg.CLASSES], marker="o",
                 label=nombres[posicion], color=colores[posicion], alpha=0.85)
    eje.plot(cfg.CLASSES, [metricas_referencia["f1_por_clase"][c] for c in cfg.CLASSES],
             linestyle=":", color="grey", label="V12", linewidth=1.5)
    eje.set_ylabel("F1 de la clase")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=6)
    eje = ejes[2]
    indice = np.arange(len(resultados))
    eje.bar(indice - 0.2, [r["frontera_18_24_25_34"] for r in resultados], 0.4,
            label="student 18-24|25-34", color="#b2182b")
    eje.bar(indice + 0.2, [r["frontera_25_34_35_49"] for r in resultados], 0.4,
            label="student 25-34|35-49", color="#2166ac")
    eje.axhline(teacher_frontera["18-24|25-34"], color="#b2182b", linestyle="--", linewidth=1)
    eje.axhline(teacher_frontera["25-34|35-49"], color="#2166ac", linestyle="--", linewidth=1)
    eje.set_xticks(indice)
    eje.set_xticklabels(nombres, rotation=25, fontsize=7)
    eje.legend(fontsize=6)
    eje.set_title("Fronteras del student frente al teacher validado", fontsize=10)
    fig.suptitle("V18 P5: destilacion del conocimiento de frontera, cross-fitted", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V18 · P5 — Destilacion del conocimiento de frontera", "",
        "## 1. Diseno", "",
        "P5 no repite P1, P3 ni P4: aqui el objetivo auxiliar no es la etiqueta verdadera (que ya "
        "esta en la entropia cruzada) sino la **prediccion suave del teacher**, que es informacion "
        "nueva. Y **no hay routing en inferencia**: la senal se descarga durante el entrenamiento.", "",
        "**Teacher**: el especialista binario sobre las **279 variables de V12**, ya validado en P0' "
        f"con el protocolo completo ({teacher_frontera['18-24|25-34']} y "
        f"{teacher_frontera['25-34|35-49']} en las filas de cada frontera), no el de TF-IDF disperso, "
        "porque es mejor y esta en la misma representacion que el student, que es la condicion para "
        "que el student pueda expresar lo que se le ensena.", "",
        "**Cross-fitting**: el teacher se cross-fita en cinco inner folds **agrupados** del train "
        "exterior, de modo que cada ejemplo del train exterior fue predicho por un teacher que no lo "
        "uso para entrenarse. El teacher ajustado sobre todo el train exterior predice la validacion "
        "exterior **solo para diagnostico**, y esas predicciones nunca entran en el entrenamiento del "
        "student.", "",
        f"**Cuadrica** de {len(CUADRICULA)} combinaciones de peso y temperatura, **seleccionada dentro "
        f"del train exterior** en una particion interior del {FRACCION_SELECCION:.0%}. La validacion "
        "exterior no participa en la eleccion.", "",
        "**Senales**: la probabilidad de la clase superior de cada pareja, equivalente a su margen "
        "por la sigmoida, aplicada **solo a las filas de esa pareja**. Sin pseudo-etiqueta dura en "
        "ningun punto, y la etiqueta verdadera es siempre el objetivo principal.", "",
        "## 2. Resultado", "",
        "| Brazo | Lambda | T | macro-F1 | Delta | Media folds | Folds pos. | IC95 | 18-24\\|25-34 | "
        "25-34\\|35-49 | F1 25-34 | Rescate | Dano | Adopta |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for resultado in resultados:
        lineas.append(
            f"| {resultado['brazo']} | {resultado['lambda']} | {resultado['temperatura']:.0f} | "
            f"{resultado['macro_combinada']} | {resultado['delta']:+.4f} | "
            f"{resultado['media_folds']} | {resultado['folds_positivos']}/5 | "
            f"[{resultado['bootstrap_ic95'][0]:+.4f}, {resultado['bootstrap_ic95'][1]:+.4f}] | "
            f"{resultado['frontera_18_24_25_34']} | {resultado['frontera_25_34_35_49']} | "
            f"{resultado['f1_clase_25_34']} | {resultado['rescate']:,} | {resultado['dano']:,} | "
            f"{'si' if resultado['adopta'] else '**no**'} |")
    lineas.append(f"| V12 (referencia) | - | - | {metricas_referencia['macro_f1']} | - | "
                  f"{float(np.mean(por_fold_referencia)):.4f} | - | - | "
                  f"{fronteras_referencia['18-24|25-34']['accuracy']} | "
                  f"{fronteras_referencia['25-34|35-49']['accuracy']} | "
                  f"{metricas_referencia['f1_por_clase']['25-34']} | - | - | - |")
    lineas += ["", "## 3. Cuanto recupera el student del teacher disponible", "",
              "| Brazo | Frontera | V12 | Student | Teacher validado | Recuperacion |",
              "|---|---|---|---|---|---|"]
    for _, fila in pd.read_csv(DISTILACION).iterrows():
        recuperacion = ("" if pd.isna(fila["recuperacion_sobre_disponible"])
                        else f"{fila['recuperacion_sobre_disponible']:.1%}")
        lineas.append(f"| {fila['brazo']} | {fila['frontera']} | {fila['v12']} | {fila['student']} | "
                      f"{fila['teacher_validado']} | {recuperacion} |")
    lineas += ["", "## 4. Combinacion elegida por fold (seleccion interior)", "",
              "| Brazo | Fold | Lambda | Temperatura |", "|---|---|---|---|"]
    for resultado in resultados:
        for eleccion in resultado["combinaciones_elegidas"]:
            lineas.append(f"| {resultado['brazo']} | {eleccion['fold']} | {eleccion['lambda']} | "
                          f"{eleccion['temperatura']:.0f} |")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V18 · P5 destilacion del conocimiento de frontera", "fecha": "2026-09-26",
        "teacher": {"tipo": "especialista binario sobre las 279 variables de V12",
                    "validado_en_frontera": teacher_frontera,
                    "nota": "ya validado en P0' con el protocolo completo de 5 folds"},
        "cross_fitting": {"inner_folds": INNER_FOLDS, "agrupado": True,
                          "cobertura": "todo outer_train, cada ejemplo predicho por un teacher que "
                                       "no lo uso para entrenarse",
                          "diagnostico_teacher_sobre_outer_validacion": True},
        "cuadricula": [{"lambda": p, "temperatura": t} for p, t in CUADRICULA],
        "seleccion": f"dentro del train exterior, particion interior agrupada del "
                     f"{FRACCION_SELECCION:.0%}",
        "pre_registrado": {"epocas": EPOCAS, "lote": LOTE, "tasa_aprendizaje": TASA_APRENDIZAJE},
        "senales": ["probabilidad de la clase superior de cada pareja", "solo filas de la pareja",
                    "sin pseudo-etiqueta dura", "la etiqueta verdadera es el objetivo principal"],
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "f1_por_clase": metricas_referencia["f1_por_clase"],
                       "fronteras": {k: v["accuracy"] for k, v in fronteras_referencia.items()}},
        "resultados": resultados, "dispositivo": DISPOSITIVO,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    mejor = max(resultados, key=lambda r: r["macro_combinada"])
    print(f"\n{sum(1 for r in resultados if r['adopta'])} de {len(resultados)} brazos adoptan | "
          f"mejor: {mejor['brazo']} {mejor['macro_combinada']} ({mejor['delta']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
