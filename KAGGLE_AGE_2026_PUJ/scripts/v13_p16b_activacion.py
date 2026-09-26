"""V13 · P16b — ¿Dónde es útil el bloque de evidencia por token?

P16 mostró que el bloque de P1 **no** es una copia lineal de V12: las 279 variables explican solo
una fracción pequena de su varianza (R2 mediano 0,09) y el control sobre la reconstrucción cae por
debajo del suelo de azar. P1b mostró que, aun así, fusionarlo da -0,0002 y que un bloque **permutado**
da exactamente lo mismo. Las dos cosas juntas son una paradoja que este experimento resuelve:

* el bloque tiene señal propia (0,3997 aislado),
* su información no está en V12,
* y sin embargo añadirlo no mejora nada.

La explicación mas economica es que **la señal este concentrada en una minoria de filas**: la banda
de tokens específicos cubre el 13 % de los tokens del documento y aporta 1,7 tokens conocidos por
documento de media. Si el bloque solo acierta en las filas donde esta activo, entonces promediarlo
con V12 en todas las filas lo diluye, y una puerta (la forma `gate(especialista, V12)` de §37) si
tendria base. Si el bloque es mediocre incluso en sus filas activas, ninguna puerta lo salva y la
pista se cierra definitivamente.

Este experimento mide, sobre artefactos ya calculados y sin entrenar nada:

1. Cuantas filas tienen el bloque activo, por umbral de tokens conocidos.
2. En cada subconjunto, la exactitud de V12 y la del bloque, y la del mejor modelo posible
   (V12 si acierta, si no el bloque), que es el techo de una puerta perfecta.
3. El **rescate por franja de actividad**: si el rescate se concentra en las filas activas, la
   puerta tiene base; si se reparte por igual, no.
4. Lo mismo restringido a las dos fronteras centrales, que es donde §36 exige mirar.

El test no se toca. Ninguna submission se genera aqui.

Entregables: ``reports/v16b_activacion.md``, ``reports/v16b_activacion_franjas.csv``,
``experiments/v16b_p16b_activacion.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v13_p1b_fusion_tokens import PARES, metricas_globales  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v16b_activacion.md"
FRANJAS = cfg.REPORTS / "v16b_activacion_franjas.csv"
RESULTADO = cfg.EXPERIMENTS / "v16b_p16b_activacion.json"

#: Umbrales de tokens conocidos en la banda especifica (la informativa). Cero significa "el
#: documento no aporta ningun token de la banda".
UMBRALES = (0, 1, 2, 3, 5)


def main() -> int:
    """Mide donde aporta el bloque y si una puerta tendria base."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    nombres = json.loads((cfg.FEATURES / "v13_p1_tokens_nombres.json").read_text(encoding="utf-8"))
    columna_actividad = nombres.index("especifica_n_conocidos")
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    acierto_v12 = np.argmax(referencia, axis=1) == clase

    # El OOF del bloque se calcula una vez por fold, con su matriz propia, para que ninguna fila
    # vea su etiqueta (mismo criterio que P1a).
    from v13_p1b_fusion_tokens import oof_una_cabeza
    acierto_bloque = np.zeros(len(y), dtype=bool)
    actividad = np.zeros(len(y))
    for fold in folds:
        matriz = np.load(cfg.FEATURES / f"v13_p1_tokens_fold{fold['fold']}.npy")
        probabilidades = oof_una_cabeza(matriz.astype(np.float64), y, folds, "logreg")
        acierto_bloque[fold["val_idx"]] = (np.argmax(probabilidades[fold["val_idx"]], axis=1)
                                           == clase[fold["val_idx"]])
        actividad[fold["val_idx"]] = matriz[fold["val_idx"], columna_actividad]
    # En las filas de train, la matriz del propio fold trae una estimacion distinta: se usa la del
    # fold en el que la fila es validacion, que es la unica comparable con la referencia de V12.
    print(f"[bloque] actividad: media {np.expm1(actividad).mean():.2f} tokens conocidos | "
          f"filas con al menos 1: {int((actividad > 0).sum()):,} "
          f"({100 * float((actividad > 0).mean()):.1f} %)", flush=True)

    filas = []
    for umbral in UMBRALES:
        activo = actividad >= np.log1p(umbral) - 1e-9
        inactivo = ~activo
        for ambito, mascara in [("todas", np.ones(len(y), dtype=bool)),
                                ("activas", activo), ("inactivas", inactivo)]:
            if not mascara.any():
                continue
            for nombre, mascara_frontera in [("global", mascara)] + [
                    (etiqueta, mascara & np.isin(clase, (cfg.CLASSES.index(a), cfg.CLASSES.index(b))))
                    for (a, b), etiqueta in PARES.items()]:
                if not mascara_frontera.any():
                    continue
                base = acierto_v12[mascara_frontera]
                bloque = acierto_bloque[mascara_frontera]
                techo = base | bloque
                filas.append({
                    "umbral_tokens": umbral, "ambito": ambito, "frontera": nombre,
                    "filas": int(mascara_frontera.sum()),
                    "accuracy_v12": round(float(base.mean()), 4),
                    "accuracy_bloque": round(float(bloque.mean()), 4),
                    "techo_puerta": round(float(techo.mean()), 4),
                    "rescate": int((~base & bloque).sum()),
                    "dano": int((base & ~bloque).sum()),
                    "neto": int((~base & bloque).sum() - (base & ~bloque).sum()),
                })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(FRANJAS, index=False)

    activa_global = tabla[(tabla["umbral_tokens"] == 0) & (tabla["ambito"] == "activas")
                          & (tabla["frontera"] == "global")].iloc[0]
    print(f"[franja activa] filas {int(activa_global['filas']):,} | V12 "
          f"{activa_global['accuracy_v12']} | bloque {activa_global['accuracy_bloque']} | "
          f"techo de puerta {activa_global['techo_puerta']} | neto "
          f"{int(activa_global['neto']):+,}", flush=True)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V13 · P16b — ¿Dónde es útil el bloque de evidencia por token?", "",
        "## 1. La paradoja que hay que explicar", "",
        "El bloque de P1 tiene señal propia (0,3997 en control aislado), su información **no** está "
        "en V12 (P16: R2 mediano 0,09 y el control sobre la reconstrucción cae al suelo de azar) y, "
        "aun así, fusionarlo da -0,0002, igual que un bloque permutado. La explicación más "
        "económica es que la señal esté concentrada en una minoría de filas y el promedio la "
        "diluya. Eso es lo que se mide aquí, sin entrenar nada.", "",
        "## 2. Actividad del bloque", "",
        f"- Tokens conocidos de la banda específica por documento: media "
        f"{np.expm1(actividad).mean():.2f}",
        f"- Documentos con al menos uno: **{int((actividad > 0).sum()):,}** "
        f"({100 * float((actividad > 0).mean()):.1f} % del total)", "",
        "## 3. Exactitud por franja de actividad", "",
        "Umbral = número mínimo de tokens conocidos de la banda específica para considerar el "
        "documento activo. **Techo de puerta** es el máximo alcanzable con una puerta perfecta "
        "(V12 si acierta, si no el bloque).", "",
        "| Umbral | Ámbito | Frontera | Filas | V12 | Bloque | Techo de puerta | Rescate | Daño | Neto |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fila in filas:
        if fila["frontera"] != "global":
            continue
        lineas.append(f"| {fila['umbral_tokens']} | {fila['ambito']} | global | "
                      f"{fila['filas']:,} | {fila['accuracy_v12']} | {fila['accuracy_bloque']} | "
                      f"**{fila['techo_puerta']}** | {fila['rescate']:,} | {fila['dano']:,} | "
                      f"{fila['neto']:+,} |")
    lineas += ["", "## 4. Las dos fronteras centrales (§36)", "",
               "| Umbral | Frontera | Filas | V12 | Bloque | Techo de puerta | Neto |",
               "|---|---|---|---|---|---|---|"]
    for fila in filas:
        if fila["frontera"] == "global" or fila["ambito"] != "activas":
            continue
        lineas.append(f"| {fila['umbral_tokens']} | {fila['frontera']} | {fila['filas']:,} | "
                      f"{fila['accuracy_v12']} | {fila['accuracy_bloque']} | "
                      f"{fila['techo_puerta']} | {fila['neto']:+,} |")
    lineas += ["", "## 5. Lectura", ""]
    lineas += [
        f"- En las {int(activa_global['filas']):,} filas activas, el bloque acierta "
        f"{activa_global['accuracy_bloque']} frente a {activa_global['accuracy_v12']} de V12, y una "
        f"puerta perfecta llegaria a {activa_global['techo_puerta']}.",
        f"- El techo de puerta sobre las filas activas es la cota de §37: si esa cifra no supera al "
        f"0,6004 de V12 por un margen amplio, **ninguna puerta puede adopter el bloque** y la "
        f"pista P1 se cierra de forma definitiva, no solo por no cumplir el gate de la fusion.",
        "- Con una puerta imperfecta el techo es inalcanzable: la regla de §37 solo activa el "
        "especialista cuando V12 es ambiguo, y P15 ya mostro que la confianza de los miembros no "
        "identifica las filas que rescuarian. El margen entre el techo y una puerta real es "
        "precisamente lo que hay que mirar antes de gastar una hora de CPU en P3.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P16b activacion del bloque de P1", "fecha": "2026-09-25",
        "tokens_conocidos_medios": round(float(np.expm1(actividad).mean()), 3),
        "filas_activas": int((actividad > 0).sum()),
        "porcentaje_activo": round(100 * float((actividad > 0).mean()), 2),
        "referencia_v12": metricas_globales(referencia, clase),
        "tabla": filas, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nartefactos: {INFORME.name}, {FRANJAS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
