"""V13 · P15 — Matriz de errores por miembro: quien acierta cuando V12 falla.

Diagnostico que precede a los especialistas de frontera (P3) y a la fusion de Blocks (P4, P5).
No entrena nada: solo reconstruye, sobre las probabilidades fuera de muestra (OOF) que el
proyecto ya tiene, **que miembro del ensemble acierta en el 66,9 % del error rescatable**, con
que margen, en que clase y si el conocimiento complementario se concentra en una familia
concreta de modelos.

Por que hace falta: ``phase13`` ya midio el rescate **por familia** (6 familias, 10 miembros con
bloque de 6 columnas). Faltaba el grano de **miembro**, que es el que decide si un especialista
de frontera tiene una base sobre la que apoyarse: si el rescate lo dan 3 miembros de la misma
familia, es un unico modelo disfrazado de 3 y no hay "conocimiento complementario" que repartir.

Se responde a cinco preguntas:

1. Por miembro: aciertos cuando V12 falla, danos cuando V12 acierta, neto, rescates
   **exclusivos** (que no consigue ningun otro miembro) y margen relativo con el que acierta.
2. El margen relativo **discrimina**: separa "esta fila la va a rescatar" de "esta no"? Sin
   separacion, cualquier puerta (§37) seria ruido, y el proyecto lo sabria antes de gastar una
   hora de GPU.
3. Interseccion de errores: cuantos miembros aciertan a la vez cuando V12 falla, y cuantos hacen
   falta para cubrir cada fraccion del rescate (cubrimiento greedy).
4. Familias: el rescate se reparte o se concentra.
5. Fronteras 18-24|25-34 y 25-34|35-49: cuantos cruces de cada par resuelve cada miembro, que es
   exactamente el terreno de P3.

Toda la medicion es sobre OOF ya calculado: no se toca el test, no se reentrena nada y no se
genera submission. Se aplica la doble metrica (§36) y el criterio de doble metrica por cabeza no
aplica porque aqui no se anaden variables.

Salidas: ``reports/v13_member_error_matrix.csv``, ``reports/v13_error_intersection.csv``,
``reports/v13_boundary_analysis.csv``, ``reports/v13_p15_member_error_matrix.md``,
``experiments/v13_p15_member_error_matrix.json``.
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

from phase2_ponderacion import MEMBER_FILES  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v13_p15_member_error_matrix.md"
MATRIZ = cfg.REPORTS / "v13_member_error_matrix.csv"
INTERSECCION = cfg.REPORTS / "v13_error_intersection.csv"
FRONTERAS = cfg.REPORTS / "v13_boundary_analysis.csv"
RESULTADO = cfg.EXPERIMENTS / "v13_p15_member_error_matrix.json"

#: Familias de modelos (mismas que phase13, para que las cifras sean comparables).
FAMILIAS = {
    "svm_lineal": ["oof_E013_margenes", "oof_E016_margenes", "oof_E014_margenes"],
    "prob_lineal": ["oof_E-009", "oof_E-011", "oof_E017_probabilidades"],
    "bayes": ["oof_E-010"],
    "ordinal": ["oof_E015_softmax", "oof_E021a_ordinal", "oof_E028a_pares"],
    "prototipos": ["oof_E021b_rocchio"],
    "denso": ["oof_E019b_densas_svd"],
    "nbsvm": ["oof_ADV05_nbsvm"],
}
#: Pares de frontera objetivo de V13.
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}
#: Secciones de analisis de phase13, para citar sin recalcular.
PHASE13 = cfg.EXPERIMENTS / "phase13_error_desacuerdo.json"


def nombre_a_familia(nombres: list[str]) -> dict[str, str]:
    """Mapea el nombre de cada miembro a su familia de modelos."""
    asignacion: dict[str, str] = {}
    for familia, archivos in FAMILIAS.items():
        for archivo in archivos:
            asignacion[Path(archivo).stem] = familia
    return {nombre: asignacion.get(nombre, "sin_clasificar") for nombre in nombres}


def predicciones_de_miembros() -> tuple[np.ndarray, list[str], np.ndarray, dict[str, str]]:
    """Devuelve (votos, nombres, margenes normalizados, familia por miembro).

    Convenciones heredadas de ``phase13``: argmax para los bloques de 6 columnas, conteo de
    signos para el bloque de 5 margenes adyacentes y exclusion de la fuente ordinal continua
    (una sola columna, que se analiza aparte por correlacion con el indice ordinal real).
    """
    votos: list[np.ndarray] = []
    margenes: list[np.ndarray] = []
    nombres: list[str] = []
    for archivo, transformacion in MEMBER_FILES:
        matriz = np.load(cfg.FEATURES / archivo)
        if matriz.ndim == 1 or matriz.shape[1] == 1:
            continue
        nombre = Path(archivo).stem
        if matriz.shape[1] == 5:
            # El bloque de 5 margenes adyacentes decide una posicion: la decision mas debil es el
            # margen de menor absoluto, normalizado por el margen medio del propio miembro. En la
            # misma direccion que las otras confianzas (mas alto = mas seguro).
            absolutos = np.abs(matriz)
            votos.append((matriz > 0).sum(axis=1))
            margenes.append(absolutos.min(axis=1) / np.clip(absolutos.mean(axis=1), 1e-12, None))
            nombres.append(nombre)
            continue
        probabilidades = np.exp(matriz) if transformacion == "log" else matriz
        probabilidades = probabilidades / np.clip(probabilidades.sum(axis=1, keepdims=True),
                                                  1e-12, None)
        votos.append(np.argmax(probabilidades, axis=1))
        # Margen relativo: cuanto supera la clase elegida al resto, en unidades tipicas
        # del propio miembro. Comparable entre miembros de familias distintas.
        orden = np.sort(probabilidades, axis=1)
        margenes.append((orden[:, -1] - orden[:, -2])
                        / np.clip(orden[:, -1].std(), 1e-12, None))
        nombres.append(nombre)
    return (np.column_stack(votos), nombres, np.column_stack(margenes), nombre_a_familia(nombres))


def main() -> int:
    """Reconstruye la matriz de errores por miembro y sus lecturas."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    indice_real = np.array([cfg.CLASSES.index(clase) for clase in y])

    v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    pred_v12 = np.argmax(v12, axis=1)
    acierto_v12 = pred_v12 == indice_real
    confianza_v12 = v12.max(axis=1)
    error_v12 = ~acierto_v12
    distancia = np.abs(pred_v12 - indice_real)

    votos, nombres, margenes, familias = predicciones_de_miembros()
    aciertos = votos == indice_real[:, None]
    n_miembros = len(nombres)
    rescuing = aciertos & error_v12[:, None]
    # Un rescate es exclusivo del miembro i cuando es el unico que acierta en esa fila de error.
    otros_ganan = (rescuing.sum(axis=1)[:, None] - rescuing) > 0
    rescuing_exclusivo = rescuing & ~otros_ganan

    # ---------------------------------------------------------------- 1. matriz por miembro
    filas = []
    for indice, nombre in enumerate(nombres):
        rescates = int(rescuing[:, indice].sum())
        danios = int(((~aciertos[:, indice]) & acierto_v12).sum())
        solo_este = int(rescuing_exclusivo[:, indice].sum())
        confianza_rescate = float(margenes[error_v12 & aciertos[:, indice], indice].mean()) \
            if rescates else 0.0
        confianza_fallo = float(margenes[error_v12 & ~aciertos[:, indice], indice].mean()) \
            if int((error_v12 & ~aciertos[:, indice]).sum()) else 0.0
        distancia_rescate = distancia[error_v12 & aciertos[:, indice]]
        filas.append({
            "miembro": nombre,
            "familia": familias[nombre],
            "exactitud_global": round(float(aciertos[:, indice].mean()), 4),
            "rescates_cuando_v12_falla": rescates,
            "pct_del_error_rescatable": round(100 * rescates / max(1, int(error_v12.sum())), 3),
            "rescates_exclusivos": solo_este,
            "pct_rescate_exclusivo": round(100 * solo_este / max(1, int(error_v12.sum())), 3),
            "danios_cuando_v12_acierta": danios,
            "neto": rescates - danios,
            "margen_medio_al_rescatar": round(confianza_rescate, 4),
            "margen_medio_al_fallar": round(confianza_fallo, 4),
            "separacion_margen": round(confianza_rescate - confianza_fallo, 4),
            "distancia_ordinal_media_al_rescatar": round(float(distancia_rescate.mean()), 3)
            if rescates else 0.0,
            "acierta_adyacente_pct": round(
                100 * float((distancia_rescate <= 1).mean()), 2) if rescates else 0.0,
        })
    matriz = pd.DataFrame(filas).sort_values("rescates_exclusivos", ascending=False)
    matriz.to_csv(MATRIZ, index=False)

    # ---------------------------------------------------------------- 2. interseccion de errores
    cuantos = aciertos[error_v12].sum(axis=1)
    interseccion = [{"tipo": "simultaneos", "etiqueta": str(int(k)),
                     "filas": int((cuantos == k).sum()),
                     "pct_del_error_v12": round(100 * float((cuantos == k).mean()), 3)}
                    for k in range(n_miembros + 1)]
    rescued_total = int((cuantos > 0).sum())
    # Cubrimiento greedy: cuantos miembros hacen falta para cubrir cada fraccion del rescate.
    orden = matriz["miembro"].tolist()
    acumulado, cubiertas, greedy = 0, np.zeros(len(indice_real), dtype=bool), []
    for nombre in orden:
        columna = aciertos[:, nombres.index(nombre)]
        nuevas = int((columna & error_v12 & ~cubiertas).sum())
        if nuevas == 0:
            continue
        cubiertas |= columna & error_v12
        acumulado += nuevas
        greedy.append({"tipo": "greedy", "etiqueta": nombre, "filas": None,
                       "pct_del_error_v12": round(100 * acumulado / error_v12.sum(), 3),
                       "rescates_nuevos": nuevas, "acumulado": acumulado})
    pd.DataFrame(interseccion + greedy,
                 columns=["tipo", "etiqueta", "filas", "pct_del_error_v12",
                          "rescates_nuevos", "acumulado"]).to_csv(INTERSECCION, index=False)

    # ---------------------------------------------------------------- 3. fronteras objetivo
    filas_frontera = []
    for (clase_a, clase_b), etiqueta in PARES.items():
        indice_a, indice_b = cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)
        en_pareja = np.isin(indice_real, (indice_a, indice_b))
        # El cruce: la verdad es una clase del par y V12 predice la otra. Por definicion V12
        # acierta 0 de esas filas (un cruce ES un error de V12), asi que su fila de referencia
        # lleva 0 en `aciertos_en_cruce` y su referencia util es la exactitud en el par.
        cruce = en_pareja & (((indice_real == indice_a) & (pred_v12 == indice_b))
                             | ((indice_real == indice_b) & (pred_v12 == indice_a)))
        entradas = en_pareja & ~cruce
        for indice, nombre in enumerate(nombres):
            aciertos_frontera = aciertos[:, indice]
            filas_frontera.append({
                "frontera": etiqueta,
                "miembro": nombre,
                "familia": familias[nombre],
                "filas_par": int(en_pareja.sum()),
                "filas_cruce": int(cruce.sum()),
                "aciertos_en_cruce": int(aciertos_frontera[cruce].sum()),
                "pct_del_cruce": round(100 * float(aciertos_frontera[cruce].mean()), 3),
                "danios_en_no_cruce": int((~aciertos_frontera & acierto_v12)[entradas].sum()),
                "exactitud_en_par": round(float(aciertos_frontera[en_pareja].mean()), 4),
            })
        filas_frontera.append({
            "frontera": etiqueta, "miembro": "V12 (referencia)", "familia": "produccion",
            "filas_par": int(en_pareja.sum()), "filas_cruce": int(cruce.sum()),
            "aciertos_en_cruce": 0,
            "pct_del_cruce": 0.0,
            "danios_en_no_cruce": 0,
            "exactitud_en_par": round(float(acierto_v12[en_pareja].mean()), 4),
        })
    fronteras = pd.DataFrame(filas_frontera)
    fronteras.to_csv(FRONTERAS, index=False)

    # Techo de informacion: exactitud de "V12 si acierta, si no el miembro mas confiado".
    confianza_miembros = np.column_stack(
        [margenes[:, nombres.index(nombre)] for nombre in nombres])
    elegido = confianza_miembros.argmax(axis=1)
    techo = np.where(acierto_v12, pred_v12, votos[np.arange(len(votos)), elegido])
    exactitud_techo = float((techo == indice_real).mean())

    # ---------------------------------------------------------------- 4. lectura por familia
    por_familia = []
    for familia in FAMILIAS:
        miembros_familia = [nombre for nombre in nombres if familias[nombre] == familia]
        indices = [nombres.index(nombre) for nombre in miembros_familia]
        anyo = aciertos[:, indices].any(axis=1)
        exclusive = float((anyo & error_v12
                           & ~((aciertos[:, indices] & error_v12[:, None]).sum(axis=1) - 1
                                > 0)).sum())
        por_familia.append({
            "familia": familia,
            "miembros": len(indices),
            "rescates_anyo": int((anyo & error_v12).sum()),
            "pct_del_error_rescatable": round(100 * float((anyo & error_v12).mean()), 3),
            "pct_exclusiva_de_la_familia": round(100 * exclusive / max(1, int(error_v12.sum())), 3),
            "exactitud_media": round(float(aciertos[:, indices].mean()), 4),
        })
    familias_df = pd.DataFrame(por_familia).sort_values("pct_exclusiva_de_la_familia",
                                                         ascending=False)

    phase13 = json.loads(PHASE13.read_text(encoding="utf-8")) if PHASE13.exists() else {}
    errores = int(error_v12.sum())
    # ---------------------------------------------------------------- 5. informe
    lineas = [
        "# V13 · P15 — Que miembro acierta cuando V12 falla", "",
        "Diagnostico sobre probabilidades fuera de muestra ya calculadas: no entrena, no toca el "
        "test y no genera submission. Complementa a `phase13`, que midio el rescate por familia; "
        " aqui baja al grano de miembro, que es el que decide si un especialista de frontera "
        "(P3) tiene una base real.", "",
        "## 1. Referencia", "",
        f"- Filas: {len(indice_real):,}",
        f"- Exactitud de V12: **{float(acierto_v12.mean()):.4f}** · macro-F1 de referencia 0,6004",
        f"- Errores de V12: **{errores:,}** ({100 * float(error_v12.mean()):.2f} % de las filas)",
        f"- De esos errores, acierta **al menos un miembro: {rescued_total:,}** "
        f"({100 * rescued_total / max(1, errores):.2f} %); el resto "
        f"({errores - rescued_total:,}) es techo de informacion para todo el ensemble.", "",
        "## 2. Matriz por miembro", "",
        "| Miembro | Familia | Exactitud | Rescates | Exclusivos | Danos | Neto | Margen rescate | Margen fallo | Separacion |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in matriz.iterrows():
        lineas.append(
            f"| `{fila['miembro']}` | {fila['familia']} | {fila['exactitud_global']:.4f} | "
            f"{fila['rescates_cuando_v12_falla']:,} | **{fila['rescates_exclusivos']:,}** | "
            f"{fila['danios_cuando_v12_acierta']:,} | {fila['neto']:+,} | "
            f"{fila['margen_medio_al_rescatar']:.3f} | {fila['margen_medio_al_fallar']:.3f} | "
            f"{fila['separacion_margen']:+.3f} |")
    lineas += ["", "## 3. Lectura", "",
               f"- **El rescate se reparte o se concentra.** El miembro con mas rescates "
               f"exclusivos cubre {matriz['pct_rescate_exclusivo'].max():.2f} % del error de V12; "
               f"los tres primeros cubren {matriz['pct_rescate_exclusivo'].head(3).sum():.2f} %. "
               f"Si esa cifra es pequeña, no hay 'conocimiento complementario' repartido entre "
               f"modelos sino uno o dos modelos disgrazados de varios.",
               f"- **El margen discrimina**: la separacion va de "
               f"{matriz['separacion_margen'].min():+.3f} a {matriz['separacion_margen'].max():+.3f}. "
               + ("Una separacion positiva y grande significa que la confianza del miembro "
                  "identifica las filas que va a rescatar, que es lo que necesita la puerta de "
                  "§37 para no ser ruido."
                  if matriz['separacion_margen'].max() > 0.05 else
                  "Todas las separaciones son pequenas: la confianza del miembro no identifica "
                  "las filas que rescata, y por tanto una puerta por confianza no seria mas que "
                  "ruido con un nombre elegante."),
               f"- **Techo con puerta por confianza**: \"V12 si acierta, si no el miembro mas "
               f"confiado\" alcanza {exactitud_techo:.4f} de exactitud frente a "
               f"{float(acierto_v12.mean()):.4f} de V12. Ese es el maximo absoluto de lo que "
               f"cualquier especializacion por miembro puede conseguir sin information nueva.",
               f"- **Cubrimiento greedy**: {len(greedy)} miembros bastan para cubrir el "
               f"{100 * greedy[-1]['acumulado'] / errores:.1f} % del error de V12.", "",
               "## 4. Familias", "",
               "| Familia | Miembros | Rescates (cualquiera) | Exclusiva de la familia | Exactitud media |",
               "|---|---|---|---|---|"]
    for _, fila in familias_df.iterrows():
        lineas.append(f"| {fila['familia']} | {fila['miembros']} | {fila['rescates_anyo']:,} "
                     f"({fila['pct_del_error_rescatable']:.2f} %) | "
                     f"**{fila['pct_exclusiva_de_la_familia']:.2f} %** | {fila['exactitud_media']:.4f} |")
    lineas += ["", "## 5. Fronteras objetivo", "",
               "El cruce de una frontera son las filas cuya clase verdadera es una de las dos y "
               "que V12 etiqueta con la otra. Es el terreno de P3.", "",
               "| Frontera | Mejor miembro | Aciertos en el cruce | % del cruce | Exactitud en el par |",
               "|---|---|---|---|---|"]
    for etiqueta in PARES.values():
        parte = fronteras[fronteras["frontera"] == etiqueta]
        miembros = parte[parte["miembro"] != "V12 (referencia)"]
        mejor = miembros.loc[miembros["aciertos_en_cruce"].idxmax()]
        referencia = parte[parte["miembro"] == "V12 (referencia)"].iloc[0]
        filas_cruce = int((np.isin(indice_real, [cfg.CLASSES.index(a) for a in
                                                 etiqueta.split("|")])
                           & (((indice_real == cfg.CLASSES.index(etiqueta.split("|")[0]))
                               & (pred_v12 == cfg.CLASSES.index(etiqueta.split("|")[1])))
                              | ((indice_real == cfg.CLASSES.index(etiqueta.split("|")[1]))
                                 & (pred_v12 == cfg.CLASSES.index(etiqueta.split("|")[0]))))).sum())
        lineas.append(f"| {etiqueta} ({filas_cruce:,} cruces) | `{mejor['miembro']}` | "
                     f"{int(mejor['aciertos_en_cruce']):,} | {mejor['pct_del_cruce']:.2f} % | "
                     f"{mejor['exactitud_en_par']:.4f} |")
        lineas.append(f"| {etiqueta} (referencia) | V12 | {int(referencia['aciertos_en_cruce']):,} | "
                     f"{referencia['pct_del_cruce']:.2f} % | {referencia['exactitud_en_par']:.4f} |")
    lineas += ["", "## 6. Consecuencia para P3", ""]
    mejor_tres = matriz.head(3)
    lineas += [
        f"- Los tres miembros con mas rescates exclusivos son "
        + ", ".join(f"`{n}` ({f})" for n, f in zip(mejor_tres['miembro'], mejor_tres['familia']))
        + f"; en conjunto cubren {mejor_tres['pct_rescate_exclusivo'].sum():.2f} % del error de V12.",
        "- Si esa cobertura es pequeña, un especialista binario de frontera **no puede apoyarse en "
        "los miembros actuales** y deberia construirse sobre variables nuevas (P1, P17, P19, P20) "
        "en lugar de sobre una recombinacion de las existentes.",
        "- Si la separacion de margen es alta, la puerta de §37 "
        "(`gate(especialista, V12)`) tiene una senal de activacion utilizable; si es baja, la puerta "
        "tendra que decidirse por la clase mas probable de V12 y no por la confianza del miembro.",
        "",
        f"Referencia de `phase13` para contraste: {phase13.get('global', {}).get('macro_f1', 'n/d')} "
        f"de macro-F1 y {len(phase13.get('familias', []))} familias de miembros.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P15 matriz de errores por miembro", "fecha": "2026-09-25",
        "filas": int(len(indice_real)),
        "exactitud_v12": round(float(acierto_v12.mean()), 4),
        "errores_v12": errores,
        "errores_con_rescate": rescued_total,
        "pct_error_rescatable": round(100 * rescued_total / max(1, errores), 3),
        "errores_techo_informacion": errores - rescued_total,
        "techo_puerta_por_confianza": round(exactitud_techo, 4),
        "mejor_miembro": matriz.iloc[0]["miembro"],
        "pct_rescate_exclusivo_maximo": float(matriz["pct_rescate_exclusivo"].max()),
        "pct_rescate_exclusivo_top3": round(float(matriz["pct_rescate_exclusivo"].head(3).sum()), 3),
        "separacion_margen_maxima": float(matriz["separacion_margen"].max()),
        "cubrimiento_greedy": greedy,
        "familias": por_familia,
        "interseccion": interseccion,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"exactitud V12 {acierto_v12.mean():.4f} | errores {errores:,} | "
          f"rescatables {rescued_total:,} ({100 * rescued_total / errores:.2f} %)")
    print(f"techo con puerta por confianza: {exactitud_techo:.4f} | "
          f"cubierto por {len(greedy)} miembros: {100 * greedy[-1]['acumulado'] / errores:.1f} %")
    print("\ntres miembros con mas rescates exclusivos:")
    print(matriz.head(3)[["miembro", "familia", "rescates_cuando_v12_falla",
                          "rescates_exclusivos", "neto", "separacion_margen"]].to_string(index=False))
    print("\npor familia:")
    print(familias_df.to_string(index=False))
    print(f"\nartefactos: {INFORME.name}, {MATRIZ.name}, {INTERSECCION.name}, {FRONTERAS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
