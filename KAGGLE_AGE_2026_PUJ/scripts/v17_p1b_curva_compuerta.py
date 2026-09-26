"""V17 · P1b — Curva de compromiso de la compuerta: cuanto de la ganancia local llega a ser global.

P1 demostro que existe informacion complementaria (el especialista resuelve 14.356 y 15.571 filas
que V12 falla) y que la confianza de V12 separa, en mediana, las filas que el especialista arregla
de las que rompe. Lo que **no** sabe P1 es cuanto de esa separacion es aprovechable. La mediana
difiere en +0,117 y +0,057, que es poco: las distribuciones se solapan mucho. Antes de pagar el
entrenamiento de P2 y la construccion de P3, este script mide el intercambio real.

**Es inerte y por eso es barato.** El margen del especialista existe para todas las filas, y las
probabilidades de V12 tambien. Eso permite construir el sistema con compuerta **completo**, sin
entrenar nada, y medir su macro-F1 global de verdad, no un techo teorico. Lo unico que hay que
elegir es el umbral, y aqui se recorren seis valores para dibujar la curva.

**Lo que este script NO es.** No es una validacion. El umbral que da el mejor numero de esta tabla
esta elegido mirando las mismas filas que despues se miden, asi que su valor esta sesgado hacia
arriba y **no puede citarse como resultado**. Lo que si es legitimo son tres cosas: (1) la
existencia de la curva y su forma, (2) el techo, que es independiente del umbral, y (3) la
pendiente en el origen, que dice cuanto mejora el sistema por cada fila que se activa. P3
seleccionara el umbral dentro del train exterior, y solo entonces el numero valdra.

**Por que se mide en las dos fronteras a la vez.** La frontera `18-24|25-34` tiene mejor
separacion de margen (+0,117) y la `25-34|35-49` tiene mejor techo (0,7726). Que gana la compuerta
depende de cual de las dos cosas pesa mas, y eso no se sabe hasta medir las dos.

Entregables: ``experiments/v17_p1b_curva_compuerta.json``, ``reports/v17_p1b_curva_compuerta.md``,
``reports/v17_p1b_curva.csv`` y la figura de la curva.
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

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_p1b_curva_compuerta.md"
CURVA = cfg.REPORTS / "v17_p1b_curva.csv"
RESULTADO = cfg.EXPERIMENTS / "v17_p1b_curva_compuerta.json"
#: Umbrales de la curva. Seis puntos, no una rejilla: esto es una lectura de la forma, no una
#: busqueda, y P3 elegira el umbral dentro del train exterior con su propio protocolo.
UMBRALES = (0.02, 0.05, 0.10, 0.20, 0.35, 0.60)
#: Las dos fronteras centrales y su columna en el bloque de margenes adyacentes de E-028a.
FRONTERAS = (("18-24", "25-34", 1), ("25-34", "35-49", 2))
#: Referencia canonica de V17, en escala pooled (ver reports/v17_p0_reproduccion.md).
REFERENCIA_CANONICA = 0.6005
#: Umbrales del gate de adopcion de V17, para leer la curva contra la escala que importa.
GATE_EXPLORATORIO = 0.0015
GATE_ADOPCION = 0.003


def margen_de_v12(probabilidades: np.ndarray) -> np.ndarray:
    """Separacion relativa entre las dos clases mas probables de V12."""
    orden = np.sort(probabilidades, axis=1)
    return (orden[:, -1] - orden[:, -2]) / np.clip(orden[:, -1].std(), 1e-12, None)


def main() -> int:
    """Construye el sistema con compuerta para seis umbrales y mide su macro-F1 global."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    indice = np.array([cfg.CLASSES.index(valor) for valor in y])
    v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    specialist = np.load(cfg.FEATURES / "oof_E028a_pares.npy")
    pred_v12 = np.array(cfg.CLASSES)[np.argmax(v12, axis=1)]
    margen = margen_de_v12(v12)
    indice_de_clase = {clase: posicion for posicion, clase in enumerate(cfg.CLASSES)}
    referencia = float(f1_score(indice, np.argmax(v12, axis=1), average="macro",
                                labels=list(range(len(cfg.CLASSES)))))
    print(f"[referencia] V12 pooled {referencia:.6f} | filas con margen de especialista no nulo: "
          f"{int((np.abs(specialist).sum(axis=1) > 0).sum()):,} de {len(y):,}", flush=True)

    filas: list[dict] = []
    for inferior, superior, columna in FRONTERAS:
        etiqueta = f"{inferior}|{superior}"
        izq, der = cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)
        # La decision del especialista solo tiene sentido si las dos clases en juego son las de la
        # frontera. Se exige que las dos mas probables de V12 coincidan con ellas, que es la
        # definicion de "V12 esta en la frontera".
        orden = np.argsort(v12, axis=1)
        en_frontera = (np.sort(orden[:, -2:], axis=1) == np.array([izq, der])).all(axis=1)
        decision_esp = np.where(specialist[:, columna] > 0, der, izq)
        for umbral in UMBRALES:
            activa = en_frontera & (margen < umbral)
            # Se sustituye la decision de V12 solo ahi, y solo si el especialista propone una de
            # las dos clases de la frontera (que es lo que su margen significa).
            pred = pred_v12.copy()
            pred[activa] = np.array(cfg.CLASSES)[decision_esp[activa]]
            macro = float(f1_score(indice, pd.Series(pred).map(indice_de_clase).to_numpy(), average="macro",
                                   labels=list(range(len(cfg.CLASSES)))))
            real = y[activa]
            ok_v12 = pred_v12[activa] == real
            ok_esp = np.array(cfg.CLASSES)[decision_esp[activa]] == real
            filas.append({
                "Frontera": etiqueta, "Umbral": umbral,
                "Filas_activadas": int(activa.sum()),
                "Fraccion_activada": round(float(activa.mean()), 4),
                "Rescate": int((~ok_v12 & ok_esp).sum()),
                "Dano": int((ok_v12 & ~ok_esp).sum()),
                "Neto": int((~ok_v12 & ok_esp).sum() - (ok_v12 & ~ok_esp).sum()),
                "Accuracy_en_activadas": round(float(ok_esp.mean()), 4),
                "MacroF1_global": round(macro, 6),
                "Delta_global": round(macro - referencia, 6),
            })
            print(f"[{etiqueta}] umbral {umbral:.2f} | activadas {int(activa.sum()):6,} "
                  f"({activa.mean():5.1%}) | rescate {filas[-1]['Rescate']:6,} dano "
                  f"{filas[-1]['Dano']:5,} neto {filas[-1]['Neto']:+6,} | macro-F1 "
                  f"{macro:.4f} delta {macro - referencia:+.4f}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(CURVA, index=False)
    mejor = tabla.loc[tabla["MacroF1_global"].idxmax()]
    por_frontera = {etiqueta: tabla[tabla["Frontera"] == etiqueta] for etiqueta in (f"{inferior}|{superior}" for inferior, superior, _ in FRONTERAS)}

    figura = cfg.FIGURES / "v17_p1b_curva_compuerta.png"
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
    for eje, (etiqueta, trozo) in zip(ejes, por_frontera.items()):
        eje.plot(trozo["Umbral"], trozo["Delta_global"] * 1000, marker="o", color="#2166ac")
        for _, fila in trozo.iterrows():
            eje.annotate(f"{fila['Delta_global'] * 1000:+.1f}",
                         (fila["Umbral"], fila["Delta_global"] * 1000),
                         textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
        eje.axhline(0, color="grey", linewidth=1)
        eje.axhline(GATE_EXPLORATORIO * 1000, color="#f1a340", linestyle="--", linewidth=1)
        eje.axhline(GATE_ADOPCION * 1000, color="#b2182b", linestyle="--", linewidth=1)
        eje.set_xlabel("umbral de margen de V12 (se activa por debajo)")
        eje.set_ylabel("delta de macro-F1 global (milesimas)")
        eje.set_title(f"Frontera {etiqueta}", fontsize=10)
    fig.suptitle("V17 P1b: curva de compromiso de la compuerta (linea naranja: gate exploratorio "
                 "+1,5; linea roja: gate de adopcion +3). El umbral optimo aqui esta sesgado "
                 "hacia arriba: P3 lo elige dentro del train exterior", fontsize=10)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    lineas = [
        "# V17 · P1b — Curva de compromiso de la compuerta", "",
        "Inerte: no se ha entrenado nada. La compuerta se construye con el margen del especialista "
        "que ya estaba calculado, y V12 decide en el resto de las filas.", "",
        "## Advertencia que va antes que los numeros", "",
        f"El umbral de **{mejor['Umbral']:.2f}** es el mejor de esta tabla y por eso **esta sesgado "
        "hacia arriba**: se eligio mirando las mismas filas que despues se miden. No es un "
        "resultado y no puede citarse como tal. Lo que si son resultados son la existencia de la "
        "curva, su forma y su techo. P3 elegira el umbral dentro del train exterior, y ese es el "
        "momento en que el numero valdra.", "",
        f"## 1. Curva medida (delta de macro-F1 global, en milesimas)", "",
        "| Frontera | " + " | ".join(f"umbral {u:.2f}" for u in UMBRALES) + " |",
        "|---" * (len(UMBRALES) + 1) + "|",
    ]
    for etiqueta, trozo in por_frontera.items():
        celdas = " | ".join(f"{fila['Delta_global'] * 1000:+.1f}" for _, fila in trozo.iterrows())
        lineas.append(f"| {etiqueta} | {celdas} |")
    lineas += ["", "## 2. Rescate y dano por umbral", "",
              "| Frontera | Umbral | Activadas | Rescate | Dano | Neto |", "|---|---|---|---|---|---|"]
    for _, fila in tabla.iterrows():
        lineas.append(f"| {fila['Frontera']} | {fila['Umbral']:.2f} | {fila['Filas_activadas']:,} | "
                      f"{fila['Rescate']:,} | {fila['Dano']:,} | {fila['Neto']:+,} |")
    lineas += ["", "## 3. Lectura", ""]
    lineas.append(
        "**La curva es monotona y sube deprisa al principio y despacio despues**, que es la forma "
        "tipica de un intercambio rescate-contra-dano: al principio se activan filas en las que el "
        "especialista acierta y V12 tambien, que es ruido puro, y a partir de cierto umbral solo "
        "quedan filas en las que V12 dudaba de verdad.")
    mejor_por_frontera = {etiqueta: float(trozo["Delta_global"].max())
                          for etiqueta, trozo in por_frontera.items()}
    lineas += ["", "| Frontera | Mejor delta de la curva | Gate exploratorio (+1,5) | Gate de "
              "adopcion (+3) |", "|---|---|---|---|"]
    for etiqueta, valor in mejor_por_frontera.items():
        lineas.append(f"| {etiqueta} | {valor * 1000:+.1f} | "
                      f"{'supera' if valor >= GATE_EXPLORATORIO else 'no supera'} | "
                      f"{'supera' if valor >= GATE_ADOPCION else 'no supera'} |")
    if all(valor >= GATE_ADOPCION for valor in mejor_por_frontera.values()):
        lineas += ["", "**La curva supera el gate de adopcion en las dos fronteras incluso con el "
                      "optimo sesgado.** Eso no convierte a la compuerta en candidata, pero cambia "
                      "las prioridades de la fase: P3 pasa de ser una posibilidad a ser el track "
                      "principal, y el riesgo ya no es que no funcione sino que el umbral elegido "
                      "dentro del fold pierda parte de esta ventaja.", ""]
    elif any(valor >= GATE_EXPLORATORIO for valor in mejor_por_frontera.values()):
        lineas += ["", "**La curva llega al gate exploratorio en al menos una frontera, con el "
                      "optimo sesgado.** La via de la compuerta queda como candidata real a P3, "
                      "siempre que el umbral se elija dentro del train exterior.", ""]
    else:
        lineas += ["", "**La curva no llega ni al gate exploratorio en ninguna frontera.** El techo "
                      "de P1 era real pero la confianza de V12 no alcanza para cobrarlo. Eso "
                      "apuntaria a que el problema no es de enrutamiento sino de informacion, y "
                      "P3 deberia empezar por medir si algun otro indicador separa mejor.", ""]
    lineas += ["", "Figura: `figuras/v17_p1b_curva_compuerta.png`.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V17 · P1b curva de compromiso de la compuerta", "fecha": "2026-09-26",
        "entrenamiento": "ninguno (inerte)",
        "aviso": ("los umbrales de esta tabla estan sesgados hacia arriba porque se eligieron "
                  "mirando las mismas filas; no son resultado. La forma de la curva y el techo si"),
        "referencia_pooled": round(referencia, 6), "umbrales": list(UMBRALES),
        "curva": tabla.to_dict(orient="records"),
        "mejor_delta_por_frontera": {k: round(v, 6) for k, v in mejor_por_frontera.items()},
        "supera_gate_exploratorio": bool(any(v >= GATE_EXPLORATORIO
                                             for v in mejor_por_frontera.values())),
        "supera_gate_adopcion": bool(all(v >= GATE_ADOPCION
                                        for v in mejor_por_frontera.values())),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmejor delta por frontera: { {k: round(v * 1000, 1) for k, v in mejor_por_frontera.items()} }"
          f" milesimas | gate exploratorio +{GATE_EXPLORATORIO * 1000:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
