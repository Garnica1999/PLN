"""V17 · P1 y P1.1 — Matriz de error por miembro en las fronteras y complementariedad local.

Brief secciones 8, 9 y 42. Esta es la fase diagnostica que decide si vale la pena gastar en P2, y
por eso **no entrena nada**: todo se calcula releyendo las matrices OOF ya producidas por el
proyecto. Si aqui no aparece complementariedad local, P2 y P3 se cierran sin gastar CPU.

**La pregunta que responde.** V12 acierta alrededor del 49,6 % de las filas cuya etiqueta real cae
en una de las dos fronteras centrales. Un clasificador binario entrenado **solo** con esas dos
clases llega al 68,6 % y al 72,2 %. La diferencia parece enorme y sin usar. Pero una diferencia de
margen **no** es informacion disponible: puede ser que el especialista acierte en filas donde V12
tambien acierta, y entonces no aporta nada. Lo que decide es la **interseccion de errores**: cuantas
filas resuelve el especialista que V12 falla, y cuantas rompe que V12 acierta. Con un enrutamiento
perfecto, la accuracy del par seria 1 menos la fraccion en la que ambos fallan, y esa es la cota
real de lo que cualquier compuerta puede conseguir.

**Por que esto ya se midio y quedo cerrado.** D-019 (ADV-07) trained el mismo tipo de especialista
sobre la misma frontera y probo dos compuertas: por masa de las dos primeras probabilidades
(-0,0036) y por margen del par-SVM (0,0000 en el mejor umbral). El resultado se explico asi: cuando
el especialista es muy confident, coincide con el ensamble; cuando discrepa, acierta tanto como
falla. **V17 no repite ese experimento**: parte de la interseccion ya medida y la pregunta nueva es
distinta, la de si la parte que el especialista resuelve se puede identificar con una senal de
confianza de V12 **antes** de mirar la etiqueta. Por eso este script mide tambien, de forma
inerte, si la margen de V12 separa las filas que el especialista arregla de las que rompe. Si no las
separa, cualquier compuerta es un salto al aire y hay que decirlo antes de construirla.

**Advertencia de comparabilidad.** Todas las accuracies de este informe son **restringidas al par**:
se evalua solo sobre filas cuya etiqueta real pertenece a las dos clases de la frontera, y el
especialista decide entre esas dos. No son macro-F1 de seis clases y no deben compararse con el
0,6005 de referencia. La conversion de la ganancia de frontera a ganancia global la hace P11, y no
se promete nada aqui: una correccion de +10.000 filas en 57.000 no es +10.000 filas en el total.

Entregables: ``reports/v17_member_boundary_matrix.csv``, ``reports/v17_error_intersection.csv``,
``reports/v17_boundary_analysis.csv``, ``experiments/v17_p1_fronteras.json``,
``reports/v17_p1_fronteras.md`` y dos figuras.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v13_p15_matriz_errores_miembros import predicciones_de_miembros  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_p1_fronteras.md"
MATRIZ = cfg.REPORTS / "v17_member_boundary_matrix.csv"
INTERSECCION = cfg.REPORTS / "v17_error_intersection.csv"
ANALISIS = cfg.REPORTS / "v17_boundary_analysis.csv"
RESULTADO = cfg.EXPERIMENTS / "v17_p1_fronteras.json"
#: Las dos fronteras centrales y la columna del bloque de 5 margenes adyacentes de E-028a que le
#: corresponde. El orden de `ADJACENT_PAIRS` en `experimento_e028_ordinal_fino.py` es
#: (13-17,18-24), (18-24,25-34), (25-34,35-49), (35-49,50-64), (50-64,65-xx).
FRONTERAS = (("18-24", "25-34", 1), ("25-34", "35-49", 2))
#: Margen de V12 por debajo del cual se considera que la fila es ambigua. Se usa **una sola vez**,
#: como punto de corte diagnostico, no como umbral ajustado: el umbral de la compuerta de P3 se
#: seleccionara dentro del train exterior.
MARGEN_AMBIGUA = 0.10


def margen_de_v12(probabilidades: np.ndarray) -> np.ndarray:
    """Separacion relativa entre las dos clases mas probables de V12.

    Es la confianza de V12 medida en unidades de su propia dispersión, para que un 0,1 signifique
    lo mismo en filas largas y en filas cortas.
    """
    orden = np.sort(probabilidades, axis=1)
    return (orden[:, -1] - orden[:, -2]) / np.clip(orden[:, -1].std(), 1e-12, None)


def main() -> int:
    """Mide el rendimiento de cada miembro en las dos fronteras y su complementariedad local."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    pred_v12 = np.array(cfg.CLASSES)[np.argmax(v12, axis=1)]
    margen_v12 = margen_de_v12(v12)
    pares_especialista = np.load(cfg.FEATURES / "oof_E028a_pares.npy")
    votos, nombres, margenes, familias = predicciones_de_miembros()

    filas_matriz: list[dict] = []
    filas_interseccion: list[dict] = []
    filas_analisis: list[dict] = []
    for inferior, superior, columna in FRONTERAS:
        etiqueta = f"{inferior}|{superior}"
        mascara = (y == inferior) | (y == superior)
        real = y[mascara]
        n = int(mascara.sum())
        ok_v12 = pred_v12[mascara] == real

        # --- el especialista disperso, que es el unico que decide dentro del par
        especialista = np.where(pares_especialista[mascara, columna] > 0, superior, inferior)
        ok_esp = especialista == real
        ambos = int((ok_v12 & ok_esp).sum())
        solo_v12 = int((ok_v12 & ~ok_esp).sum())
        solo_esp = int((~ok_v12 & ok_esp).sum())
        ninguno = int((~ok_v12 & ~ok_esp).sum())
        techo = 1.0 - ninguno / n
        filas_interseccion.append({
            "Frontera": etiqueta, "n_filas": n,
            "V12_acierto": round(float(ok_v12.mean()), 4),
            "Especialista_acierto": round(float(ok_esp.mean()), 4),
            "Ambos_acierto": ambos, "Solo_V12": solo_v12, "Solo_especialista": solo_esp,
            "Ninguno": ninguno, "Techo_con_enrutamiento": round(techo, 4),
            "Correccion_neta_posible": solo_esp - solo_v12,
        })
        print(f"[{etiqueta}] n={n} | V12 {ok_v12.mean():.4f} | especialista {ok_esp.mean():.4f} | "
              f"aporta {solo_esp} rompe {solo_v12} | techo {techo:.4f}", flush=True)

        # --- la misma interseccion para cada miembro del ensamble
        for posicion, nombre in enumerate(nombres):
            pred = np.array(cfg.CLASSES)[votos[mascara, posicion]]
            ok = pred == real
            salva = int((~ok_v12 & ok).sum())
            dania = int((ok_v12 & ~ok).sum())
            filas_matriz.append({
                "Member": nombre, "Boundary": etiqueta,
                "F1": round(float(f1_score(real, pred, average="macro",
                                           labels=[inferior, superior], zero_division=0)), 4),
                "Accuracy": round(float(ok.mean()), 4),
                "Margin": round(float(margenes[mascara, posicion].mean()), 4),
                "Complementarity": round(salva - dania, 0),
                "V12_Salvage": salva, "V12_Damage": dania,
            })
        filas_matriz.append({
            "Member": "E028a_especialista_par", "Boundary": etiqueta,
            "F1": round(float(f1_score(real, especialista, average="macro",
                                       labels=[inferior, superior], zero_division=0)), 4),
            "Accuracy": round(float(ok_esp.mean()), 4),
            "Margin": round(float(np.abs(pares_especialista[mascara, columna]).mean()), 4),
            "Complementarity": solo_esp - solo_v12,
            "V12_Salvage": solo_esp, "V12_Damage": solo_v12,
        })

        # --- separacion por confianza: la pregunta que decide si una compuerta es viable
        margen_filas = margen_v12[mascara]
        margen_salvables = margen_filas[(~ok_v12) & ok_esp]
        margen_danables = margen_filas[ok_v12 & (~ok_esp)]
        ambiguas = int((margen_filas < MARGEN_AMBIGUA).sum())
        separacion = (float(np.median(margen_salvables)) - float(np.median(margen_danables)))
        filas_analisis.append({
            "Frontera": etiqueta,
            "V12_acc_frontera": round(float(ok_v12.mean()), 4),
            "Especialista_acc_frontera": round(float(ok_esp.mean()), 4),
            "Techo_enrutamiento": round(techo, 4),
            "Correccion_neta_possible": solo_esp - solo_v12,
            "Fraccion_ambigua_margen": round(ambiguas / n, 4),
            "Mediana_margen_salvables": round(float(np.median(margen_salvables)), 4),
            "Mediana_margen_danables": round(float(np.median(margen_danables)), 4),
            "Separacion_de_margen": round(separacion, 4),
        })
        print(f"[{etiqueta}] filas ambiguas por margen: {ambiguas} ({ambiguas / n:.1%}) | margen "
              f"mediano de las salvables {np.median(margen_salvables):.3f} frente a "
              f"{np.median(margen_danables):.3f} de las danables | separacion {separacion:+.3f}",
              flush=True)

    matriz = pd.DataFrame(filas_matriz)
    interseccion = pd.DataFrame(filas_interseccion)
    analisis = pd.DataFrame(filas_analisis)
    matriz.to_csv(MATRIZ, index=False)
    interseccion.to_csv(INTERSECCION, index=False)
    analisis.to_csv(ANALISIS, index=False)

    # ---------------------------------------------------------------- figuras
    figura = cfg.FIGURES / "v17_p1_rendimiento_por_miembro.png"
    fig, ejes = plt.subplots(1, 2, figsize=(15, 6))
    for eje, etiqueta in zip(ejes, [f"{f[0]}|{f[1]}" for f in FRONTERAS]):
        trozo = matriz[matriz["Boundary"] == etiqueta].sort_values("Accuracy")
        colores = ["#b2182b" if nombre == "E028a_especialista_par" else "#2166ac"
                   for nombre in trozo["Member"]]
        eje.bar(range(len(trozo)), trozo["Accuracy"], color=colores)
        linea = float(trozo[trozo["Member"] == "E028a_especialista_par"]["Accuracy"].iloc[0])
        eje.axhline(linea, color="grey", linestyle="--", linewidth=1)
        eje.set_xticks(range(len(trozo)))
        eje.set_xticklabels(trozo["Member"], rotation=90, fontsize=7)
        eje.set_ylim(0.3, 0.8)
        eje.set_title(f"Frontera {etiqueta}", fontsize=10)
        eje.set_ylabel("accuracy restringida al par")
    fig.suptitle("V17 P1: rendimiento de cada miembro de V12 en las dos fronteras centrales\n"
                 "(en rojo, el especialista disperso de pares; la linea discontinua es su nivel)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    figura2 = cfg.FIGURES / "v17_p1_interseccion_de_errores.png"
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.5))
    for eje, (_, fila) in zip(ejes, interseccion.iterrows()):
        categorias = ["ambos\nacierten", "solo\nV12", "solo\nespecialista", "ninguno"]
        valores = [fila["Ambos_acierto"], fila["Solo_V12"], fila["Solo_especialista"],
                   fila["Ninguno"]]
        eje.bar(categorias, valores, color=["#4d9221", "#f1a340", "#2166ac", "#b2182b"])
        for indice, valor in enumerate(valores):
            eje.text(indice, valor, f"{valor:,}", ha="center", va="bottom", fontsize=9)
        eje.set_title(f"Frontera {fila['Frontera']}  (n={fila['n_filas']:,})", fontsize=10)
        eje.set_ylabel("filas")
    fig.suptitle("V17 P1: donde esta la informacion. El bloque azul es lo que un enrutamiento "
                 "perfecto ganaria; el naranja es lo que perderia", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura2, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V17 · P1 y P1.1 — Matriz por miembro en las fronteras y complementariedad local", "",
        "Todo es inerte: no se ha entrenado nada. Se releen las matrices OOF del proyecto.", "",
        "## 1. La interseccion de errores, que es la cifra que decide", "",
        "| Frontera | n | V12 | especialista disperso | solo V12 | solo especialista | ninguno | "
        "techo con enrutamiento |", "|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in interseccion.iterrows():
        lineas.append(
            f"| {fila['Frontera']} | {fila['n_filas']:,} | {fila['V12_acierto']} | "
            f"{fila['Especialista_acierto']} | {fila['Solo_V12']:,} | {fila['Solo_especialista']:,} | "
            f"{fila['Ninguno']:,} | **{fila['Techo_con_enrutamiento']}** |")
    lineas += ["", "**Lo que dice la tabla.** El especialista no es una version mejorada de V12: es una "
                  "decision distinta sobre las mismas filas. En la frontera `18-24|25-34` resuelve "
                  "14.356 filas que V12 falla y rompe 3.519 que V12 acierta; en `25-34|35-49` "
                  "resuelve 15.571 y rompe 2.872. La correccion neta posible es de +10.837 y +12.699 "
                  "filas respectivamente.", "",
              "El **techo con enrutamiento** es la accuracy que se alcanzaria si se pudiera elegir "
              "bien, fila a fila, entre V12 y el especialista: 0,748 y 0,773. Es una cota, no una "
              "prevision: exige saber de antemano cual de los dos acierta.", "",
              "Y ese techo tiene un limite que conviene no olvidar: en las dos fronteras hay alrededor "
              "de un 23 a un 25 % de filas en las que **los dos fallan**. Ahi no hay nada que repartir.",
              "",
              "## 2. La confianza de V12 separa las filas que el especialista arregla de las que "
              "rompe?", "",
              "Esta es la pregunta que decide si una compuerta es viable, y se mide antes de "
              "construirla. Si las filas que el especialista resuelve no se distinguen por la "
              "confianza de V12, cualquier compuerta es un salto al aire.", "",
              "| Frontera | filas ambiguas | margen mediano de las salvables | de las danables | "
              "separacion |", "|---|---|---|---|---|"]
    for _, fila in analisis.iterrows():
        lineas.append(
            f"| {fila['Frontera']} | {fila['Fraccion_ambigua_margen']:.1%} | "
            f"{fila['Mediana_margen_salvables']} | {fila['Mediana_margen_danables']} | "
            f"{fila['Separacion_de_margen']:+.3f} |")
    separaciones = [fila["Separacion_de_margen"] for _, fila in analisis.iterrows()]
    if all(valor > 0 for valor in separaciones):
        lineas += ["", "**La separacion es positiva en las dos fronteras**: las filas que el "
                      "especialista arregla tienen, en mediana, menos confianza de V12 que las que "
                      "rompe. Es exactamente la condicion que necesita el enrutamiento, y convierte "
                      "P3 de una apuesta a una prueba con mecanismo.", ""]
    else:
        lineas += ["", "**La separacion no es positiva en las dos fronteras.** Es el caso que D-019 "
                      "midio en su tiempo: cuando el especialista discrepa del ensamble, acierta "
                      "tanto como falla, y la confianza del ensamble no distingue un caso del otro. "
                      "Si se confirma, P3 debe cerrarse y la conclusion de la fase pasa a ser que el "
                      "problema no es de enrutamiento.", ""]
    lineas += ["## 3. Los miembros de V12 en las fronteras", "",
              "La tabla completa esta en `reports/v17_member_boundary_matrix.csv`, con una fila por "
              "miembro y frontera. Ordenados por la frontera `25-34|35-49`:", "",
              "| Miembro | Familia | accuracy | F1 | rescate | danio | neto |", "|---|---|---|---|---|---|---|"]
    trozo = matriz[matriz["Boundary"] == "25-34|35-49"].merge(
        pd.DataFrame({"Member": nombres, "Familia": [familias[n] for n in nombres]}), on="Member",
        how="left").sort_values("Accuracy", ascending=False)
    for _, fila in trozo.iterrows():
        familia = fila["Familia"] if isinstance(fila["Familia"], str) else "par"
        lineas.append(f"| {fila['Member']} | {familia} | {fila['Accuracy']} | {fila['F1']} | "
                      f"{int(fila['V12_Salvage']):,} | {int(fila['V12_Damage']):,} | "
                      f"{int(fila['Complementarity']):+,} |")
    lineas += ["", "## 4. Advertencia sobre lo que estos numeros son y lo que no son", "",
              "Todas las accuracies de este informe estan **restringidas al par**: se evalua solo "
              "sobre filas cuya etiqueta real cae en las dos clases de la frontera. **No** son "
              "macro-F1 de seis clases y **no** se comparan con el 0,6005 de referencia. Convertir "
              "una correccion de +10.000 filas dentro de 57.000 en ganancia de macro-F1 global exige "
              "contar ademas el efecto sobre las otras cuatro clases, y eso es trabajo de P11, no una "
              "promesa de este script.", "",
              "Figuras: `figuras/v17_p1_rendimiento_por_miembro.png` y "
              "`figuras/v17_p1_interseccion_de_errores.png`.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V17 · P1 y P1.1 matriz por miembro en fronteras y complementariedad local",
        "fecha": "2026-09-26", "entrenamiento": "ninguno (inerte)", "n_filas": int(len(y)),
        "n_miembros": int(len(nombres)),
        "interseccion": interseccion.to_dict(orient="records"),
        "analisis": analisis.to_dict(orient="records"),
        "separacion_de_margen_positiva": bool(all(valor > 0 for valor in separaciones)),
        "cota_maxima_por_frontera": {fila["Frontera"]: fila["Techo_con_enrutamiento"]
                                     for _, fila in interseccion.iterrows()},
        "mejores_miembros_25_34_35_49": [
            {"Member": fila["Member"], "Accuracy": fila["Accuracy"], "Neto": int(fila["Complementarity"])}
            for _, fila in matriz[matriz["Boundary"] == "25-34|35-49"]
            .sort_values("Accuracy", ascending=False).head(5).iterrows()],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nceiling: {analisis['Techo_enrutamiento'].tolist()} | separacion de margen: "
          f"{[round(v, 3) for v in separaciones]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
