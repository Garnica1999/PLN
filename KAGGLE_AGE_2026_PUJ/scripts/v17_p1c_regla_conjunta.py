"""V17 · P1c — La señal sirve cuando se leen las dos señales a la vez, y en sentido contrario.

P8 encontro dos señales que separan las filas que el especialista ARREGLA de las que ROMPE, y
encontró tambien algo que no era obvio al leer los numeros: **las dos señales apuntan en
sentidos opuestos**.

- El **margen del especialista** es mayor en las filas que arregla (0,330 frente a 0,134), asi que
  manda escucharle cuando el especialista esta seguro.
- El **acuerdo entre miembros** es *menor* en las filas que arregla (0,487 frente a 0,265), asi que
  manda escucharle cuando el ensemble esta repartido.

Eso tiene sentido y no es casualidad: si los catorce miembros coinciden, V12 acierta; si se
dispersan, la señal del especialista pesa mas que el promedio. La consecuencia practica es que la
compuerta que P3 propone (una sola señal de ambiguedad de V12) no era solo debil, era
**conceptualmente incompleta**: le falta la mitad de la condicion.

**Que mide este script, y que no.** Es inerte: construye el sistema con compuerta completo, con el
margen del especialista y los votos de los miembros que el proyecto ya tiene calculados, y mide su
macro-F1 global de verdad. Lo que hace es responder una pregunta previa a gastar CPU en el protocolo
anidado: **la separabilidad que vio P8 se traduce en ganancia global, o se queda en una cifra
bonita que no mueve nada?**

Tres lecturas, y las tres importan:
- **univariadas**: una regla que solo mira el acuerdo entre miembros, y otra que solo mira el
  margen del especialista. Sirven para ver cuanto aporta cada señal por separado.
- **bivariada**: la conjuncion, que es la regla que P8 sugiere. Se recorre una rejilla pequeña y
  deliberadamente gruesa (cuatro por cuatro), porque esto es una lectura de la forma, no una
  busqueda de Configuration.
- **oracle**: la cota. Se activa exactamente en las filas que el especialista rescue, sin mirar
  ninguna señal. No es alcanzable, pero es el techo contra el que hay que leer todo lo demas, y no
  sufre el sesgo de seleccion que si sufre el optimo de la rejilla.

**Sobre la tercera frontera.** El brief pide no gastar por simetria y considerar `35-49|50-64` solo
si el enfoque funciona. Aqui no se entrena ningun especialista nuevo: se mide el margen que E-028a
ya tiene para ese par, y se mide su techo y su separabilidad. Eso es precisamente lo que decide si
la via se extiende o no, y no cuesta CPU. Las cifras van marcadas como diagnostico, no como
candidato.

**El aviso de siempre, y aqui es especialmente importante.** El optimo de la rejilla esta sesgado
hacia arriba, porque se elige mirando las mismas filas que despues se miden. **Ninguno de los
numeros de la rejilla puede citarse como resultado.** Los que si valen son el techo, la pendiente en
el origen y la comparacion entre univariada y bivariada. El numero publicable saldra de P9, con
inner OOF.

Entregables: ``experiments/v17_p1c_regla_conjunta.json``,
``reports/v17_p1c_regla_conjunta.md``, ``reports/v17_p1c_regla_conjunta.csv`` y una figura.
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
from v17_p8_perfil_confianza import area_bajo_la_curva  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_p1c_regla_conjunta.md"
CURVA = cfg.REPORTS / "v17_p1c_regla_conjunta.csv"
RESULTADO = cfg.EXPERIMENTS / "v17_p1c_regla_conjunta.json"
#: Las dos fronteras centrales, con su columna en el bloque de margenes adyacentes de E-028a.
CENTRALES = (("18-24", "25-34", 1), ("25-34", "35-49", 2))
#: La tercera frontera, solo como diagnostico de decision (el brief no permite gastar por simetria,
#: asi que aqui no se entrena nada: se mide el margen que ya existe).
TERCERA = ("35-49", "50-64", 3)
#: Rejilla deliberadamente gruesa de umbrales de acuerdo entre miembros. Se activa por DEBAJO, porque
#: el acuerdo alto significa que V12 acierta.
UMBRALES_ACUERDO = (0.20, 0.35, 0.50, 0.65)
#: Umbrales del margen del especialista. Se activa por ENCIMA, porque el margen alto significa que
#: el especialista esta seguro.
UMBRALES_MARGEN = (0.00, 0.15, 0.35, 0.80)
REFERENCIA_CANONICA = 0.6005
GATE_EXPLORATORIO = 0.0015
GATE_ADOPCION = 0.003
SEMILLA = cfg.SEED


def acuerdo_entre_miembros(votos: np.ndarray, decision: np.ndarray) -> np.ndarray:
    """Fraccion de miembros que eligen la misma clase que el sistema. Mas alto es mas acuerdo."""
    return (votos == decision[:, None]).mean(axis=1)


def main() -> int:
    """Mide el intercambio de la regla univariada, de la conjuncion y del techo."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    indice = np.array([cfg.CLASSES.index(valor) for valor in y])
    v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    especialista = np.load(cfg.FEATURES / "oof_E028a_pares.npy")
    pred_v12 = np.array(cfg.CLASSES)[np.argmax(v12, axis=1)]
    indice_de_clase = {clase: posicion for posicion, clase in enumerate(cfg.CLASSES)}
    votos, _, _, _ = predicciones_de_miembros()
    decision_v12 = np.argmax(v12, axis=1)
    acuerdo = acuerdo_entre_miembros(votos, decision_v12)
    referencia = float(f1_score(indice, decision_v12, average="macro",
                                labels=list(range(len(cfg.CLASSES)))))
    print(f"[referencia] V12 pooled {referencia:.6f}", flush=True)

    filas: list[dict] = []

    def medir(nombre: str, etiqueta: str, inferior: str, superior: str, columna: int,
              activa: np.ndarray) -> dict:
        """Aplica una regla de activacion y devuelve su intercambio y su macro-F1 global."""
        izq, der = cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)
        decision_esp = np.where(especialista[:, columna] > 0, der, izq)
        pred = pred_v12.copy()
        pred[activa] = np.array(cfg.CLASSES)[decision_esp[activa]]
        macro = float(f1_score(indice, pd.Series(pred).map(indice_de_clase).to_numpy(),
                               average="macro", labels=list(range(len(cfg.CLASSES)))))
        real = y[activa]
        ok_v12 = pred_v12[activa] == real
        ok_esp = np.array(cfg.CLASSES)[decision_esp[activa]] == real
        rescate = int((~ok_v12 & ok_esp).sum())
        dano = int((ok_v12 & ~ok_esp).sum())
        fila = {"Regla": nombre, "Frontera": etiqueta, "Regla_texto": "",
                "Filas_activadas": int(activa.sum()),
                "Fraccion_activada": round(float(activa.mean()), 5),
                "Rescate": rescate, "Dano": dano, "Neto": rescate - dano,
                "MacroF1_global": round(macro, 6), "Delta_global": round(macro - referencia, 6)}
        filas.append(fila)
        print(f"  [{nombre:34s}] activadas {fila['Filas_activadas']:6,} "
              f"({fila['Fraccion_activada']:5.1%}) | rescate {rescate:6,} dano {dano:5,} "
              f"neto {rescate - dano:+6,} | macro-F1 {macro:.4f} "
              f"delta {macro - referencia:+.4f}", flush=True)
        return fila

    for inferior, superior, columna in CENTRALES:
        etiqueta = f"{inferior}|{superior}"
        izq, der = cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)
        # Condicion de contexto: V12 esta decidiendo entre las dos clases de la frontera.
        orden = np.argsort(v12, axis=1)
        en_frontera = (np.sort(orden[:, -2:], axis=1) == np.array([izq, der])).all(axis=1)
        decision_esp = np.where(especialista[:, columna] > 0, der, izq)
        rescue_real = (pred_v12 != y) & (np.array(cfg.CLASSES)[decision_esp] == y)
        print(f"\n[{etiqueta}] condicion de contexto: {int(en_frontera.sum()):,} filas", flush=True)

        # --- techo: activar exactamente donde el especialista rescue. No es alcanzable.
        medir("techo (oracle)", etiqueta, inferior, superior, columna, en_frontera & rescue_real)
        # --- univariada: solo acuerdo entre miembros, por debajo del umbral
        for umbral in UMBRALES_ACUERDO:
            fila = medir(f"univariada: acuerdo < {umbral:.2f}", etiqueta, inferior, superior,
                         columna, en_frontera & (acuerdo < umbral))
            fila["Regla_texto"] = f"las dos clases mas probables de V12 son las de la frontera y el "\
                                  f"acuerdo entre miembros es menor que {umbral:.2f}"
        # --- univariada: solo margen del especialista, por encima del umbral
        for umbral in UMBRALES_MARGEN:
            fila = medir(f"univariada: margen > {umbral:.2f}", etiqueta, inferior, superior,
                         columna, en_frontera & (np.abs(especialista[:, columna]) > umbral))
            fila["Regla_texto"] = f"las dos clases mas probables de V12 son las de la frontera y el "\
                                  f"margen del especialista es mayor que {umbral:.2f}"
        # --- bivariada: la conjuncion que P8 sugiere, con la rejilla gruesa
        for umbral_acuerdo in UMBRALES_ACUERDO:
            for umbral_margen in UMBRALES_MARGEN:
                fila = medir(f"conjunta: acuerdo < {umbral_acuerdo:.2f} y margen > "
                             f"{umbral_margen:.2f}", etiqueta, inferior, superior, columna,
                             en_frontera & (acuerdo < umbral_acuerdo)
                             & (np.abs(especialista[:, columna]) > umbral_margen))
                fila["Regla_texto"] = f"contexto de frontera, acuerdo menor que "\
                                      f"{umbral_acuerdo:.2f} y margen mayor que {umbral_margen:.2f}"

    # ---------------------------------------------------------------- tercera frontera, solo diagnostico
    inferior, superior, columna = TERCERA
    etiqueta = f"{inferior}|{superior}"
    izq, der = cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)
    orden = np.argsort(v12, axis=1)
    en_frontera = (np.sort(orden[:, -2:], axis=1) == np.array([izq, der])).all(axis=1)
    decision_esp = np.where(especialista[:, columna] > 0, der, izq)
    rescue_real = (pred_v12 != y) & (np.array(cfg.CLASSES)[decision_esp] == y)
    mascara = (y == inferior) | (y == superior)
    real = y[mascara]
    ok_v12 = pred_v12[mascara] == real
    ok_esp = np.array(cfg.CLASSES)[decision_esp[mascara]] == real
    generador = np.random.default_rng(SEMILLA)
    # El acuerdo esta invertido: manda el valor NEGADO, que es lo que crece cuando los miembros se
    # dispersan y el especialista conviene.
    auc_acuerdo = area_bajo_la_curva((-acuerdo[mascara])[(~ok_v12) & ok_esp],
                                     (-acuerdo[mascara])[ok_v12 & (~ok_esp)], generador)
    auc_margen = area_bajo_la_curva(np.abs(especialista[mascara, columna])[(~ok_v12) & ok_esp],
                                    np.abs(especialista[mascara, columna])[ok_v12 & (~ok_esp)],
                                    generador)
    print(f"\n[{etiqueta}] DIAGNOSTICO, sin entrenar: V12 {ok_v12.mean():.4f} frente a "
          f"especialista {ok_esp.mean():.4f} | techo con enrutamiento "
          f"{1.0 - float((~ok_v12 & ~ok_esp).mean()):.4f} | AUC acuerdo (invertido) "
          f"{auc_acuerdo:.4f} y margen {auc_margen:.4f}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(CURVA, index=False)
    sin_techo = tabla[tabla["Regla"] != "techo (oracle)"]
    mejor_por_frontera = {etiqueta: float(sin_techo[sin_techo["Frontera"] == etiqueta]
                                           ["Delta_global"].max())
                          for etiqueta in (f"{a}|{b}" for a, b, _ in CENTRALES)}
    techo_por_frontera = {etiqueta: float(tabla[(tabla["Frontera"] == etiqueta)
                                                & (tabla["Regla"] == "techo (oracle)")]
                                          ["Delta_global"].max())
                          for etiqueta in (f"{a}|{b}" for a, b, _ in CENTRALES)}
    mejor_conjunta = float(sin_techo[sin_techo["Regla"].str.startswith("conjunta")]
                           ["Delta_global"].max())
    mejor_univariada = float(sin_techo[sin_techo["Regla"].str.startswith("univariada")]
                             ["Delta_global"].max())

    figura = cfg.FIGURES / "v17_p1c_reglas.png"
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for eje, etiqueta in zip(ejes, [f"{a}|{b}" for a, b, _ in CENTRALES]):
        trozo = sin_techo[sin_techo["Frontera"] == etiqueta]
        tipos = {"univariada": "#7fbf7b", "conjunta": "#2166ac"}
        for tipo in tipos:
            parte = trozo[trozo["Regla"].str.startswith(tipo)]
            if parte.empty:
                continue
            parte = parte.sort_values("Delta_global")
            eje.scatter(range(len(parte)), parte["Delta_global"] * 1000, s=34,
                        c=tipos[tipo], label=tipo, alpha=0.85)
        techo = techo_por_frontera[etiqueta] * 1000
        eje.axhline(techo, color="#b2182b", linestyle=":", linewidth=1.4,
                    label=f"techo (oracle) {techo:+.1f}")
        eje.axhline(0, color="grey", linewidth=1)
        eje.axhline(GATE_EXPLORATORIO * 1000, color="#f1a340", linestyle="--", linewidth=1)
        eje.axhline(GATE_ADOPCION * 1000, color="#4d4d4d", linestyle="--", linewidth=1)
        eje.set_xticks([])
        eje.set_xlabel("reglas, ordenadas de menor a mayor delta")
        eje.set_title(f"Frontera {etiqueta} (n con contexto de frontera: "
                      f"{int(tabla[(tabla['Frontera'] == etiqueta)]['Filas_activadas'].max()):,})",
                      fontsize=10)
    ejes[0].set_ylabel("delta de macro-F1 global (milesimas)")
    ejes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("V17 P1c: regla univariada frente a conjuncion. Los optimos estan sesgados al "
                 "alza; el techo no", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    alcanza_exploratorio = all(v >= GATE_EXPLORATORIO for v in mejor_por_frontera.values())
    alcanza_adopcion = all(v >= GATE_ADOPCION for v in mejor_por_frontera.values())
    lineas = [
        "# V17 · P1c — Regla conjunta: las dos señales en sentido contrario", "",
        "Inerte. Construye el sistema con compuerta completo y mide su macro-F1 global de verdad.",
        "",
        "> **Aviso.** El optimo de cada rejilla esta sesgado hacia arriba, porque se elige mirando "
        "las mismas filas que despues se miden. **Ninguno de los numeros de la rejilla es "
        "resultado.** Los que valen son el techo, la comparacion entre univariada y conjunta, y el "
        "numero publicable saldra de P9 con inner OOF.", "",
        "## 1. El hallazgo que cambia la regla", "",
        "Las dos señales de P8 apuntan en sentidos **contrarios**, y eso es lo que faltaba:", "",
        "- el **margen del especialista** es mayor en las filas que arregla, asi que manda "
        "escucharle cuando el especialista esta seguro;",
        "- el **acuerdo entre miembros** es menor en las filas que arregla, asi que manda "
        "escucharle cuando el ensemble esta repartido.", "",
        "La compuerta de una sola señal que propone el brief era, por tanto, incompleta: le "
        "faltaba la mitad de la condicion.", "",
        "## 2. Resultados por frontera (delta global, milesimas)", "",
        "| Regla | " + " | ".join(f"{a}|{b}" for a, b, _ in CENTRALES) + " |",
        "|---" * (1 + len(CENTRALES)) + "|",
        f"| techo (oracle, no alcanzable) | "
        + " | ".join(f"{techo_por_frontera[f'{a}|{b}'] * 1000:+.1f}" for a, b, _ in CENTRALES)
        + " |",
        f"| mejor univariada (sesgada) | "
        + " | ".join(f"{float(sin_techo[(sin_techo['Frontera'] == f'{a}|{b}') & sin_techo['Regla'].str.startswith('univariada')]['Delta_global'].max()) * 1000:+.1f}"
                    for a, b, _ in CENTRALES) + " |",
        f"| mejor conjunta (sesgada) | "
        + " | ".join(f"{float(sin_techo[(sin_techo['Frontera'] == f'{a}|{b}') & sin_techo['Regla'].str.startswith('conjunta')]['Delta_global'].max()) * 1000:+.1f}"
                    for a, b, _ in CENTRALES) + " |",
        "", f"**Mejor regla conjunta en cualquier frontera: {mejor_conjunta * 1000:+.1f}** frente a "
        f"**{mejor_univariada * 1000:+.1f}** de la mejor univariada. Gate exploratorio "
        f"+{GATE_EXPLORATORIO * 1000:.1f}, gate de adopcion +{GATE_ADOPCION * 1000:.1f}.", "",
        "## 3. Todas las reglas, con su intercambio", "",
        "| Frontera | Regla | Activadas | Fraccion | Rescate | Dano | Neto | Delta global |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in tabla.sort_values(["Frontera", "Delta_global"], ascending=[True, False]).iterrows():
        lineas.append(f"| {fila['Frontera']} | {fila['Regla']} | {fila['Filas_activadas']:,} | "
                      f"{fila['Fraccion_activada']:.1%} | {fila['Rescate']:,} | {fila['Dano']:,} | "
                      f"{fila['Neto']:+,} | {fila['Delta_global'] * 1000:+.1f} |")
    lineas += ["", "## 4. Tercera frontera, solo diagnostico", "",
              f"No se ha entrenado nada: se mide el margen que E-028a ya produce para `35-49|50-64`. "
              f"V12 acierta {ok_v12.mean():.4f} de esas filas y el especialista "
              f"{ok_esp.mean():.4f}, con un techo de {1.0 - float((~ok_v12 & ~ok_esp).mean()):.4f}. "
              f"El area bajo la curva del acuerdo (leida al reves) es {auc_acuerdo:.4f} y la del "
              f"margen {auc_margen:.4f}.", ""]
    if auc_margen >= 0.62 and auc_acuerdo >= 0.62:
        lineas += ["Las dos señales separan tambien en la tercera frontera, con la misma magnitud "
                   "que en las centrales. El brief permite considerar `35-49|50-64` cuando las "
                   "centrales funcionen, y esta medicion dice que el mecanismo no es una "
                   "casualidad de esas dos parejas.", ""]
    else:
        lineas += ["Alguno de los dos valores no llega al area util, asi que la tercera frontera **no "
                   "se abre** por ahora, que es lo que el brief pide: no gastar por simetria. La "
                   "decision se difiere a que las centrales funcionen.", ""]
    lineas += ["## 5. Decision", ""]
    if alcanza_adopcion:
        lineas += ["**La regla Justa llega al gate de adopcion en las dos fronteras incluso con el "
                   "optimo sesgado.** Procede P9 con arbitro aprendido y protocolo anidado "
                   "estricto, y la regla que se valide sera la conjunta, no la univariada.", ""]
    elif alcanza_exploratorio:
        lineas += ["**La regla llega al gate exploratorio en las dos fronteras con el optimo "
                   "sesgado.** Procede P9. El numero real saldra del protocolo anidado y puede "
                   "caer por debajo, que es justo lo que P9 tiene que medir.", ""]
    else:
        lineas += ["**La regla no llega ni al gate exploratorio con el optimo sesgado.** Con esta "
                   "formulacion, P9 no esta indicado: la separabilidad de P8 no se traduce en "
                   "macro-F1 global. Antes de pagar el protocolo anidado habria que buscar una "
                   "señal mas, no afinar la que hay.", ""]
    lineas.append("Figura: `figuras/v17_p1c_reglas.png`.")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V17 · P1c regla conjunta de las dos senales", "fecha": "2026-09-26",
        "entrenamiento": "ninguno (inerte)",
        "aviso": ("los optimos de la rejilla estan sesgados al alza y no son resultado; el techo y "
                  "la comparacion entre univariada y conjunta si lo son"),
        "referencia_pooled": round(referencia, 6),
        "umbrales_acuerdo": list(UMBRALES_ACUERDO), "umbrales_margen": list(UMBRALES_MARGEN),
        "curva": tabla.to_dict(orient="records"),
        "mejor_univariada": round(mejor_univariada, 6),
        "mejor_conjunta": round(mejor_conjunta, 6),
        "mejor_por_frontera": {k: round(v, 6) for k, v in mejor_por_frontera.items()},
        "techo_por_frontera": {k: round(v, 6) for k, v in techo_por_frontera.items()},
        "alcanza_gate_exploratorio": bool(alcanza_exploratorio),
        "alcanza_gate_adopcion": bool(alcanza_adopcion),
        "tercera_frontera_diagnostico": {
            "frontera": etiqueta,
            "v12": round(float(ok_v12.mean()), 4), "especialista": round(float(ok_esp.mean()), 4),
            "techo": round(1.0 - float((~ok_v12 & ~ok_esp).mean()), 4),
            "auc_acuerdo_invertido": round(auc_acuerdo, 4), "auc_margen": round(auc_margen, 4),
            "entrenada": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntecho { {k: round(v * 1000, 1) for k, v in techo_por_frontera.items()} } | mejor "
          f"univariada {mejor_univariada * 1000:+.1f} | mejor conjunta {mejor_conjunta * 1000:+.1f}"
          f" | alcanza exploratorio: {alcanza_exploratorio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
