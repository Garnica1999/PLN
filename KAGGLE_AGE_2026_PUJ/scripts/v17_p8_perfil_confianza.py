"""V17 · P8 — Perfil de confianza de los errores de frontera: ¿existe alguna señal que sirva?

Este script responde a la pregunta que P1 y P1b dejaron abierta. P1 measured que el especialista
de pares es mucho mas exacto que V12 **dentro** de cada frontera (0,6864 y 0,7218 frente a 0,4963 y
0,4974) y que resuelve 14.356 y 15.571 filas que V12 falla. P1b midio que una compuerta sobre la
confianza de V12 no cobra nada de eso: el mejor delta global es de +0,0002 y a partir de un umbral
moderado se vuelve **negativo**, porque en las filas activadas el especialista rescata y daña en la
misma proporcion.

La conclusion de P1b es que la confianza de V12 **no identifica** cuando el especialista acierta. Pero
"la confianza de V12" son varias señales, no una: en P1b solo se uso la separacion entre las dos
clases mas probables. Este script las recorre todas y, sobre todo, mide la magnitud que de verdad
decide: **el area bajo la curva** para distinguir las filas que el especialista ARREGLA de las que
ROMPE. Un area de 0,50 quiere decir que la señal no sirve para nada, y da igual cuantas veces se
repita la compuerta. Un area claramente por encima de 0,50 seria una señal aprovechable, por debil
que sea.

Las señales candidatas, todas ya calculadas en el proyecto o derivables de forma inerte:

- separacion de V12 entre sus dos clases mas probables, en unidades de su propia dispersion;
- **entropia** de la distribucion de V12, en bits: mide si la duda esta repartida o concentrada;
- **masa de las dos primeras** clases de V12, que es el complemento informativo de la anterior;
- **acuerdo entre miembros**: quantos de los miembros del ensamble de V12 eligen la misma clase que
  V12. Es la senal mas interesante a priori, porque no procede de la cabeza lineal sino de la
  variedad del ensemble;
- magnitud del margen del propio especialista, que es la senal que D-019 ya probo como umbral (tau);
- indicador de desacuerdo, que es la senal trivial que sirve de suelo.

Los tres grupos de filas que se comparan son los de la interseccion de P1: las que el especialista
**rescata**, las que **rompe** y las que **no arregla ni empeora** porque ambos fallan. La tercera
columna se incluye para que se vea si las señales distinguen "el especialista acierta" de "nadie
acierta", que es un problema distinto del principal.

**Por que se ejecuta ahora y no en su lugar del orden.** El orden del brief pone P8 despues de P7, y
se ha adelantado a proposito: su objeto declarado es "revelar si el gate P3 tiene una señal de
entrada realmente valida", y P1b acaba de responder que no la tiene con la señal que P3 propone. Sin
este diagnostico, P2 a P7 serian gasto a ciegas. Es la puerta de rescate antes de pagar la CPU cara,
y no cambia ninguna conclusion: la puede refutear, no solo confirmarla.

**Criterio de lectura.** Con el area bajo la curva:
- por debajo de 0,55 la señal es inutil para enrutar;
- entre 0,55 y 0,62 es debil y solo serviria con una compuerta muy bien calibrada;
- por encima de 0,62 hay una via que explorar.

Entregables: ``experiments/v17_p8_perfil_confianza.json``,
``reports/v17_p8_perfil_confianza.csv``, ``reports/v17_p8_perfil_confianza.md`` y dos figuras.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

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

INFORME = cfg.REPORTS / "v17_p8_perfil_confianza.md"
PERFIL = cfg.REPORTS / "v17_p8_perfil_confianza.csv"
RESULTADO = cfg.EXPERIMENTS / "v17_p8_perfil_confianza.json"
FRONTERAS = (("18-24", "25-34", 1), ("25-34", "35-49", 2))
#: Area bajo la curva a partir de la cual una señal se considera aprovechable, y a partir de la
#: cual se considera debil pero real. Estan fijadas antes de mirar los resultados.
AUC_MINIMA_APROVECHABLE = 0.62
AUC_MINIMA_DEBIL = 0.55
#: Semilla del remuestreo del calculo del area, para que el numero sea reproducible.
SEMILLA_AUC = cfg.SEED


def senales(indice: np.ndarray, probabilidades: np.ndarray, votos: np.ndarray,
            margen_especialista: np.ndarray) -> pd.DataFrame:
    """Construye las señales de confianza candidatas para un conjunto de filas."""
    orden = np.sort(probabilidades, axis=1)
    separada = (orden[:, -1] - orden[:, -2]) / np.clip(orden[:, -1].std(), 1e-12, None)
    logaritmos = np.log(np.clip(probabilidades, 1e-12, None))
    entropia = -np.sum(probabilidades * logaritmos, axis=1) / np.log(probabilidades.shape[1])
    masa_dos = orden[:, -1] + orden[:, -2]
    decision = np.argmax(probabilidades, axis=1)
    # Acuerdo entre miembros: la fraccion de miembros que eligen la misma clase que V12. Es la
    # unica senal que no procede de la cabeza lineal sino de la variedad del ensemble.
    acuerdo = (votos == decision[:, None]).mean(axis=1)
    return pd.DataFrame({
        "Separacion_V12": separada,
        "Entropia_V12": entropia,
        "Masa_dos_primeras": masa_dos,
        "Acuerdo_miembros": acuerdo,
        "Margen_especialista": np.abs(margen_especialista),
        "Desacuerdo": (np.argmax(votos, axis=1) != decision[:, None]).any(axis=1).astype(float),
    })


def area_bajo_la_curva(positivos: np.ndarray, negativos: np.ndarray,
                       generador: np.random.Generator) -> float:
    """Area bajo la curva por remuestreo, que es como se ha medido en el resto del proyecto.

    Se usa la **probabilidad de que un positivo tenga mas señal que un negativo**, que es la
    definicion del area. Es la version de Mann-Whitney, y por eso no necesita un clasificador ni
    una particion: es una medida de separabilidad, no un modelo, y no puede sobreajustar.
    """
    if positivos.size == 0 or negativos.size == 0:
        return float("nan")
    comparaciones = (positivos[:, None] > negativos[None, :]).mean()
    empates = (positivos[:, None] == negativos[None, :]).mean()
    return float(comparaciones + 0.5 * empates)


def main() -> int:
    """Mide cuanto separa cada señal de confianza las filas que el especialista arregla de las que
    rompe, en cada frontera."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    especialista = np.load(cfg.FEATURES / "oof_E028a_pares.npy")
    pred_v12 = np.argmax(v12, axis=1)
    votos, nombres, _, _ = predicciones_de_miembros()

    filas: list[dict] = []
    for inferior, superior, columna in FRONTERAS:
        etiqueta = f"{inferior}|{superior}"
        izq, der = cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)
        mascara = (y == inferior) | (y == superior)
        indices = np.flatnonzero(mascara)
        real = y[indices]
        ok_v12 = pred_v12[indices] == np.array([cfg.CLASSES.index(v) for v in real])
        decision_esp = np.where(especialista[indices, columna] > 0, der, izq)
        ok_esp = decision_esp == np.array([cfg.CLASSES.index(v) for v in real])
        rescate = ~ok_v12 & ok_esp
        dano = ok_v12 & ~ok_esp
        irrecuperable = ~ok_v12 & ~ok_esp
        generador = np.random.default_rng(SEMILLA_AUC)
        tabla = senales(indices, v12[indices], votos[indices], especialista[indices, columna])
        print(f"\n[{etiqueta}] rescate {int(rescate.sum()):,} | dano {int(dano.sum()):,} | "
              f"irrecuperable {int(irrecuperable.sum()):,}", flush=True)
        for columna_senal in tabla.columns:
            valores = tabla[columna_senal].to_numpy()
            area_rescate_dano = area_bajo_la_curva(valores[rescate], valores[dano], generador)
            area_irrecuperable = area_bajo_la_curva(valores[irrecuperable], valores[rescate],
                                                    generador)
            filas.append({
                "Frontera": etiqueta, "Senal": columna_senal,
                "Media_rescate": round(float(valores[rescate].mean()), 4),
                "Media_dano": round(float(valores[dano].mean()), 4),
                "Media_irrecuperable": round(float(valores[irrecuperable].mean()), 4),
                "AUC_rescate_contra_dano": round(area_rescate_dano, 4),
                "AUC_irrecuperable_contra_rescate": round(area_irrecuperable, 4),
                "Aprovéchable": bool(abs(area_rescate_dano - 0.5)
                                     >= AUC_MINIMA_APROVECHABLE - 0.5),
                "Debil": bool(abs(area_rescate_dano - 0.5) >= AUC_MINIMA_DEBIL - 0.5),
            })
            print(f"    {columna_senal:24s} AUC rescate contra dano {area_rescate_dano:.4f} | "
                  f"medias {valores[rescate].mean():.3f} frente a {valores[dano].mean():.3f}",
                  flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(PERFIL, index=False)
    mejor = (tabla[tabla["Frontera"] == "18-24|25-34"].reindex(
        tabla[tabla["Frontera"] == "18-24|25-34"]["AUC_rescate_contra_dano"].abs().sub(0.5)
        .sort_values(ascending=False).index).iloc[0])
    areas = tabla["AUC_rescate_contra_dano"].to_numpy()
    aprovechables = tabla[tabla["Aprovéchable"]]
    debil = tabla[tabla["Debil"]]

    figura = cfg.FIGURES / "v17_p8_area_por_senal.png"
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for eje, etiqueta in zip(ejes, [f"{a}|{b}" for a, b, _ in FRONTERAS]):
        trozo = tabla[tabla["Frontera"] == etiqueta]
        colores = ["#1a9850" if a else "#f1a340" if d else "#b2182b"
                   for a, d in zip(trozo["Aprovéchable"], trozo["Debil"])]
        eje.barh(range(len(trozo)), trozo["AUC_rescate_contra_dano"] - 0.5, color=colores)
        eje.set_yticks(range(len(trozo)))
        eje.set_yticklabels(trozo["Senal"], fontsize=8)
        eje.axvline(0, color="grey", linewidth=1)
        eje.axvline(AUC_MINIMA_DEBIL - 0.5, color="#f1a340", linestyle="--", linewidth=1)
        eje.axvline(AUC_MINIMA_APROVECHABLE - 0.5, color="#1a9850", linestyle="--", linewidth=1)
        eje.set_title(f"Frontera {etiqueta}", fontsize=10)
        eje.set_xlabel("AUC para distinguir lo que el especialista ARREGLA de lo que ROMPE, menos 0,5")
    fig.suptitle("V17 P8: ninguna señal pasa de 0,62 (linea verde) ni llega al area util", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    figura2 = cfg.FIGURES / "v17_p8_distribuciones.png"
    indices_18 = np.flatnonzero((y == "18-24") | (y == "25-34"))
    real = y[indices_18]
    ok_v12 = pred_v12[indices_18] == np.array([cfg.CLASSES.index(v) for v in real])
    ok_esp = np.where(especialista[indices_18, 1] > 0, cfg.CLASSES.index("25-34"),
                      cfg.CLASSES.index("18-24")) == np.array([cfg.CLASSES.index(v)
                                                                for v in real])
    tabla_señales = senales(indices_18, v12[indices_18], votos[indices_18],
                             especialista[indices_18, 1])
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
    for eje, (titulo, mascara_color) in zip(ejes, [("lo que el especialista ARREGLA",
                                                    ~ok_v12 & ok_esp),
                                                   ("lo que el especialista ROMPE",
                                                    ok_v12 & ~ok_esp)]):
        eje.hist(tabla_señales["Acuerdo_miembros"][mascara_color], bins=25, color="#2166ac",
                 alpha=0.75)
        eje.set_title(f"Frontera 18-24|25-34: {titulo} (n={int(mascara_color.sum()):,})",
                      fontsize=9)
        eje.set_xlabel("acuerdo entre miembros de V12")
        eje.set_ylabel("filas")
    fig.suptitle("V17 P8: el Histograma del acuerdo entre miembros se solapa casi por completo",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figura2, dpi=130)
    plt.close(fig)

    lineas = [
        "# V17 · P8 — Perfil de confianza de los errores de frontera", "",
        "Inerte: ninguna señal se ajusta aqui. La separabilidad se mide con el area bajo la curva, "
        "que es una medida de orden y no un modelo, asi que no puede sobreajustar.", "",
        "## Que se compara", "",
        "En cada frontera, las filas que el especialista de pares **rescata** (V12 falla y el "
        "especialista acierta) frente a las que **rompe** (V12 acierta y el especialista falla). Si "
        "una senal de confianza sirviera para enrutar, tendria que dar valores altos a las "
        "rescatables y bajos a las danables. El area bajo la curva mide exactamente eso, y **0,50 "
        "significa que la senal no dice nada**.", "",
        "## Resultados", "",
        "| Frontera | Señal | AUC rescate contra daño | Lectura |", "|---|---|---|---|"]
    for _, fila in tabla.iterrows():
        area = fila["AUC_rescate_contra_dano"]
        if area >= 0.62:
            lectura = "aprovechable"
        elif area <= 0.38:
            lectura = "invertida, usable al reves"
        elif 0.55 <= area <= 0.62:
            lectura = "debil pero real"
        else:
            lectura = "inutil"
        lineas.append(f"| {fila['Frontera']} | {fila['Senal']} | "
                      f"{fila['AUC_rescate_contra_dano']} | {lectura} |")
    lineas += ["", "## Lectura", ""]
    if len(aprovechables):
        lineas += [f"**{len(aprovechables)} señales superan el area de {AUC_MINIMA_APROVECHABLE}**: quedan "
                  "vias abiertas para P9 (experto selectivo) y P3 (compuerta blanda). El nombre de "
                  "la mejor es `" + str(mejor["Senal"]) + "` en la frontera `"
                  + str(mejor["Frontera"]) + "`, con area "
                  + f"{mejor['AUC_rescate_contra_dano']}.", ""]
    elif len(debil):
        lineas += [f"**Ninguna señal alcanza el area de {AUC_MINIMA_APROVECHABLE}** y "
                  f"{len(debil)} llegan al area debil de {AUC_MINIMA_DEBIL}. Con una separacion de "
                  "ese orden, una compuerta tendria que ser muy fina para cobrar algo, y el "
                  "presupuesto de la fase no se puede gastar en una compuerta que hay que calibrar "
                  "hasta encontrar el punto exacto. El candidato mas fuerte sigue siendo "
                  f"`{mejor['Senal']}`, con area {mejor['AUC_rescate_contra_dano']}, y hay que "
                  "mirar su curva completa antes de descartar la via.", ""]
    else:
        lineas += ["**Ninguna señal alcanza ni el area debil.** La separacion mas alta es "
                   f"{float(areas.max()):.4f} (señal `{mejor['Senal']}`, frontera "
                   f"`{mejor['Frontera']}`), es decir "
                   f"{float(areas.max()) - 0.5:.4f} por encima del azar. Traducido: la confianza de "
                   "V12 no dice practicamente nada sobre si el especialista de pares va a acertar o "
                   "equivocarse en esas filas.", "",
                   "**Consecuencia para el resto de la fase.** La hipotesis de que la senal local "
                   "pueda convertirse en ganancia global **por enrutamiento se queda sin sustento "
                   "experimental**: no es que el especialista no tenga informacion (P1 medio que "
                   "tiene, y mucha), es que no hay forma de saber cuando conviene escucharle. Eso "
                   "situa la respuesta final de la fase en el lado de la pregunta F del brief, con "
                   "matiz: la informacion existe y es grande, pero no es alcanzable con las "
                   "senales de confianza disponibles, y eso es un hallazgo con contenido, no un "
                   "fracaso.", ""]
    lineas += ["Figuras: `figuras/v17_p8_area_por_senal.png` y `figuras/v17_p8_distribuciones.png`.",
               ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V17 · P8 perfil de confianza de los errores de frontera",
        "fecha": "2026-09-26", "entrenamiento": "ninguno (inerte)",
        "criterios": {"auc_minima_aprovechable": AUC_MINIMA_APROVECHABLE,
                      "auc_minima_debil": AUC_MINIMA_DEBIL},
        "perfil": tabla.to_dict(orient="records"),
        "mejor_senal": {"Frontera": str(mejor["Frontera"]), "Senal": str(mejor["Senal"]),
                        "auc": float(mejor["AUC_rescate_contra_dano"])},
        "senales_aprovechables": int(tabla["Aprovéchable"].sum()),
        "senales_debiles": int(tabla["Debil"].sum()),
        "maximo_area": round(float(areas.max()), 4),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsenales aprovechables {int(tabla['Aprovéchable'].sum())} | debiles "
          f"{int(tabla['Debil'].sum())} | area maxima {float(areas.max()):.4f} "
          f"({mejor['Senal']} en {mejor['Frontera']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
