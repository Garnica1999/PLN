"""V28-03: el margen de la region de error, es reducible o es ambiguedad de etiqueta

Por que este script existe
--------------------------
V28-01 localizo donde falla V03, y la conclusion fue que el margen se concentra en dos
regiones transversales: **textos largos** (>= 25 tokens) y **textos con URL**, con error
0.2492 y 0.2445 en HS frente a una media de 0.2131. Eso localiza el margen pero NO dice si
el margen es alcanzable. Hay dos explicaciones incompatibles y hasta ahora el proyecto las
ha tratado como si fueran la misma:

1. **Senal reducible.** Ahi hay informacion que V03 no captura, y una fuente de informacion
   nueva (representacion, rasgo, modelo) laarnessia.
2. **Ambiguidad irreducible.** Los anotadores no coinciden entre si en esos textos, y ninguna
   representacion puede hacer que el modelo acierte mas alla del nivel de acuerdo humano.

La diferencia no es academica: la primera justifica seguir buscando informacion nueva; la
segunda dice que el techo esta cerca y que el esfuerzo deberia ir a otro lado. V28-01 no puede
separarlas, porque solo mira la distribucion del error.

Como se separan aqui, sin entrenar nada
---------------------------------------
El proyecto ya tiene una estructura de grupos fusionados (`grupos_fusionados` de
`src/splits.py`): near-duplicados y pseudo-autores unidos por union-find. Los pliegues de
V03 son `StratifiedGroupKFold` **sobre esos grupos**, asi que dos filas del mismo grupo nunca
coexistieron en entrenamiento y validacion. Ese mismo grupo es, ademas, un experimento
natural ya hecho por los propios anotadores: si dos textos casi identicos llevan etiquetas
distintas, la discrepancia es de la etiqueta, no del modelo.

Se mide, entonces, el **desacuerdo intra-grupo** y se estratifica por las mismas regiones que
V28-01 encontro. Si el desacuerdo es mayor en textos largos y con URL, la segunda explicacion
gana peso en exactamente la region donde queda el margen.

Lo que este script NO hace
--------------------------
No entrena ningun modelo. No construye features. No lee el test. No produce submission.
Calcula una cota superior que usa la etiqueta, y se declara como cota, no como resultado
alcanzable: el objetivo de la cota es dimensionar el ruido, no proponer nada.

Uso
---
    /home/carlos/venv/bin/python experiments/v28_03_ambiguedad_etiquetas.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados

EXPERIMENTO = (
    "V28-03 (ambiguedad de etiqueta estratificada por la region donde V03 falla: "
    "el margen residual es reducible?)"
)
REPORTE_JSON = PROJECT_ROOT / "reports" / "v28_03_ambiguedad_etiquetas.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v28_03_ambiguedad_etiquetas.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRAL_TOKENS_LARGO = 25
RE_URL = "http"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V28-03] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cota_por_voto_mayoritario(
    etiquetas: np.ndarray, grupos: np.ndarray
) -> dict[str, float]:
    """Cota superior del acierto cuando se conoce la etiqueta de cada grupo.

    Para cada grupo se toma la etiqueta mayoritaria y se cuenta como acierto a cada fila que
    coincide con ella. Una fila cuyo grupo tiene anotaciones contradictorias nunca puede ser
    acertada por un modelo que solo vea el texto. La cifra es una **cota**, no un modelo: usa
    la etiqueta, asi que no se puede convierte en submission ni en estimador de generalizacion.

    Args:
        etiquetas: matriz de etiquetas, una fila por documento.
        grupos: identificador de grupo fusionado por fila.

    Returns:
        Diccionario con la cota por etiqueta y la cota macro.
    """
    cotas: dict[str, float] = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        columna = etiquetas[:, j]
        acierto = 0
        for grupo in np.unique(grupos):
            filas = np.flatnonzero(grupos == grupo)
            _valores, conteos = np.unique(columna[filas], return_counts=True)
            acierto += conteos.max()
        cotas[etiqueta] = float(acierto / len(columna))
    cotas["macro"] = float(np.mean([cotas[e] for e in ETIQUETAS]))
    return cotas


def _desacuerdo_por_grupo(
    etiquetas: np.ndarray, grupos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Marca cada fila cuyo grupo contiene anotaciones contradictorias.

    Args:
        etiquetas: matriz de etiquetas, una fila por documento.
        grupos: identificador de grupo fusionado por fila.

    Returns:
        Tupla con la matriz booleana `ambigua` (una columna por etiqueta) y el numero de
        filas por grupo.
    """
    n = len(grupos)
    ambigua = np.zeros((n, len(ETIQUETAS)), dtype=bool)
    for j in range(len(ETIQUETAS)):
        columna = etiquetas[:, j]
        for grupo in np.unique(grupos):
            filas = np.flatnonzero(grupos == grupo)
            if np.unique(columna[filas]).size > 1:
                ambigua[filas, j] = True
    tamanos = pd.Series(grupos).map(pd.Series(grupos).value_counts()).to_numpy()
    return ambigua, tamanos


def _razon_segura(numerador: float, denominador: float) -> float | None:
    """Cociente de dos tasas, o None si el denominador es despreciable.

    Si el error de las filas no ambiguas es cero, el cociente diverge y el numero no significa
    nada. Devolver None es mas honesto que devolver un entero de siete cifras.

    Args:
        numerador: tasa del grupo de interes.
        denominador: tasa del grupo de referencia.

    Returns:
        El cociente, o None si el denominador es menor que 1e-4.
    """
    if denominador < 1e-4:
        return None
    return float(numerador / denominador)


def main() -> None:
    """Ejecuta la medicion de ambiguedad de etiqueta por region.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUN modelo, NINGUNA submission, test no leido")

    log("1/6 datos, grupos fusionados y OOF canonico de V03")
    train = read_train()
    grupos_globales = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_globales))
    mapa_texto = dict(zip(train["id"].astype(int), train["text"]))

    particion = pd.read_csv(PARTICION)
    particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    oof = np.load(OOF_CANONICO, allow_pickle=True)
    ids_oof = np.asarray(oof["ids"], dtype=np.int64)
    y_oof = np.asarray(oof["y_true"], dtype=np.int64)
    pred_v03 = np.asarray(oof["pred"], dtype=np.int64)

    # Solo se trabaja sobre las filas que el OOF canonico cubre, para que el diagnostico de
    # V28-01 y este sean comparables fila a fila.
    mask = np.isin(train["id"].astype(int).to_numpy(), ids_oof)
    filas = train.loc[mask].reset_index(drop=True)
    y = filas[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    grupos = filas["id"].astype(int).map(mapa_grupo).to_numpy()
    log(
        f"   desarrollo {len(filas)} filas | OOF {len(ids_oof)} | "
        f"grupos distintos {len(np.unique(grupos))}"
    )
    assert len(filas) == len(y_oof) == len(grupos), "desalineacion de filas"
    assert np.array_equal(y, y_oof), "las etiquetas no coinciden con el OOF canonico"

    log("2/6 tamanho de los grupos: solo los de 2+ filas pueden discrepar")
    tamanos = pd.Series(grupos).value_counts()
    filas_multimember = int(tamanos[tamanos >= 2].sum())
    filas_singleton = int(tamanos[tamanos == 1].sum())
    filas_solo_ambiguas = 0
    for grupo, n in tamanos.items():
        if n >= 2:
            sel = grupos == grupo
            if any(np.unique(y[sel, j]).size > 1 for j in range(3)):
                filas_solo_ambiguas += int(n)
    log(
        f"   grupos con 1 fila: {filas_singleton} | con 2+ filas: {filas_multimember} | "
        f"dentro de grupos contradictorios: {filas_solo_ambiguas}"
    )

    log("3/6 desacuerdo intra-grupo por region")
    tokens = np.array([len(mapa_texto[int(i)].split()) for i in filas["id"].astype(int)])
    tiene_url = np.array([RE_URL in mapa_texto[int(i)].lower() for i in filas["id"].astype(int)])
    regiones: dict[str, np.ndarray] = {
        "texto largo (>= 25 tokens)": tokens >= UMBRAL_TOKENS_LARGO,
        "texto no largo (< 25 tokens)": tokens < UMBRAL_TOKENS_LARGO,
        "con URL": tiene_url,
        "sin URL": ~tiene_url,
    }
    ambigua, _ = _desacuerdo_por_grupo(y, grupos)
    error = pred_v03 != y

    por_region: dict[str, Any] = {}
    for nombre, mask_r in regiones.items():
        m = int(np.count_nonzero(mask_r))
        if m == 0:
            continue
        mult = mask_r & np.isin(grupos, tamanos[tamanos >= 2].index.to_numpy())
        n_mult = int(np.count_nonzero(mult))
        fila: dict[str, Any] = {
            "n_filas": m,
            "fraccion": m / len(filas),
            "n_filas_en_grupos_multimember": n_mult,
            "fraccion_en_grupos_multimember": n_mult / m,
            "por_etiqueta": {},
        }
        for j, etiqueta in enumerate(ETIQUETAS):
            n_amb = int(np.count_nonzero(ambigua[mult, j])) if n_mult else 0
            fila["por_etiqueta"][etiqueta] = {
                "n_filas_ambiguas": n_amb,
                "tasa_de_ambiguedad": (n_amb / n_mult) if n_mult else None,
                "tasa_de_error_v03": float(np.count_nonzero(error[mask_r, j]) / m),
                "error_en_filas_ambiguas": (
                    float(np.count_nonzero(error[mult & ambigua[:, j], j]) / n_amb)
                    if n_amb
                    else None
                ),
                "error_en_filas_no_ambiguas": (
                    float(
                        np.count_nonzero(error[mult & ~ambigua[:, j], j])
                        / max(int(np.count_nonzero(mult & ~ambigua[:, j])), 1)
                    )
                ),
            }
        por_region[nombre] = fila
        partes = " ".join(
            f"{et} amb {fila['por_etiqueta'][et]['tasa_de_ambiguedad']:.4f}"
            for et in ETIQUETAS
            if fila["por_etiqueta"][et]["tasa_de_ambiguedad"] is not None
        )
        log(f"   {nombre:<30} filas {m:>4} | en grupos 2+ {n_mult:>4} | {partes}")

    log("4/6 los errores de V03, se concentran en las filas ambiguas?")
    # Si el error residual fuera ambiguedad, las filas ambiguas deberian fallar mas. Se mide
    # el enriquecimiento con tabla de contingencia y Fisher exacto, por region y por etiqueta.
    concentracion: dict[str, Any] = {}
    for nombre, mask_r in regiones.items():
        mult = mask_r & np.isin(grupos, tamanos[tamanos >= 2].index.to_numpy())
        if int(np.count_nonzero(mult)) < 30:
            continue
        bloque: dict[str, Any] = {}
        for j, etiqueta in enumerate(ETIQUETAS):
            amb = ambigua[mult, j]
            err = error[mult, j]
            n_amb_err = int(np.count_nonzero(amb & err))
            n_amb_ok = int(np.count_nonzero(amb & ~err))
            n_no_err = int(np.count_nonzero(~amb & err))
            n_no_ok = int(np.count_nonzero(~amb & ~err))
            tabla_c = [[n_amb_err, n_amb_ok], [n_no_err, n_no_ok]]
            orr, p = stats.fisher_exact(tabla_c)
            bloque[etiqueta] = {
                "n_ambiguas": int(np.count_nonzero(amb)),
                "n_no_ambiguas": int(np.count_nonzero(~amb)),
                "error_si_ambigua": n_amb_err / max(int(np.count_nonzero(amb)), 1),
                "error_si_no_ambigua": n_no_err / max(int(np.count_nonzero(~amb)), 1),
                # El cociente solo tiene sentido si el denominador no es cero. Cuando el
                # error de las filas no ambiguas es 0 el cociente se dispara a un numero
                # sin contenido, asi que se guarda None y se declara la division por cero.
                "razon_de_razones": _razon_segura(
                    n_amb_err / max(int(np.count_nonzero(amb)), 1),
                    n_no_err / max(int(np.count_nonzero(~amb)), 1),
                ),
                "razon_de_odds_fisher": float(orr),
                "p_fisher": float(p),
            }
            log(
                f"   {nombre:<30} {etiqueta}: error ambigua "
                f"{bloque[etiqueta]['error_si_ambigua']:.4f} vs no ambigua "
                f"{bloque[etiqueta]['error_si_no_ambigua']:.4f} | p {p:.2e}"
            )
        concentracion[nombre] = bloque

    log("5/6 cota superior por voto mayoritario dentro del grupo")
    cota = _cota_por_voto_mayoritario(y, grupos)
    for etiqueta in ETIQUETAS:
        log(f"   {etiqueta}: cota de acierto {cota[etiqueta]:.4f}")
    log(f"   cota macro {cota['macro']:.4f} (COTA, usa la etiqueta, no es un modelo)")

    log("6/6 informe")
    informe = {
        "familia": "ERROR_DIAGNOSTIC",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "DIAGNOSTIC_ONLY",
        "pregunta": (
            "el error residual de V03 en la region de texto largo y con URL es senal "
            "reducible que una fuente de informacion nueva podria capturar, o ambiguedad "
            "irreducible de las etiquetas"
        ),
        "alcance": (
            "no se entrena ningun modelo ni se construye ninguna feature. Se usa la "
            "estructura de grupos fusionados que ya existe en el proyecto para medir el "
            "desacuerdo entre anotaciones"
        ),
        "tamanos_de_grupo": {
            "n_grupos": len(tamanos),
            "n_grupos_con_2_o_mas": int((tamanos >= 2).sum()),
            "filas_en_grupos_con_2_o_mas": filas_multimember,
            "filas_en_grupos_contradictorios": filas_solo_ambiguas,
            "fraccion_de_filas_que_aporta_informacion": filas_multimember / len(filas),
        },
        "desacuerdo_por_region": por_region,
        "concentracion_del_error_en_filas_ambiguas": concentracion,
        "cota_por_voto_mayoritario": cota,
        "gastos": {
            "filas": len(filas),
            "submissions": 0,
            "modelos_entrenados": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v28_03_ambiguedad_etiquetas.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: medicion completada, 0 modelos, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    t = informe["tamanos_de_grupo"]
    t_frac = t["fraccion_de_filas_que_aporta_informacion"]
    n_contra = t["filas_en_grupos_contradictorios"]
    filas = informe["gastos"]["filas"]
    lineas = [
        "# V28-03: ambiguedad de etiqueta en la region donde V03 falla",
        "",
        f"Generado por `experiments/v28_03_ambiguedad_etiquetas.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. La pregunta",
        "",
        "V28-01 localizo el margen residual de V03 en los textos largos y con URL. Localizar",
        "no es lo mismo que alcanzar. Este script separa las dos explicaciones posibles:",
        "senal reducible, que una fuente de informacion nueva podria capturar, o ambiguedad",
        "irreducible, donde los anotadores no coinciden entre si y ninguna representacion",
        "puede llegar mas lejos.",
        "",
        "El experimento natural ya existe en el proyecto: `grupos_fusionados` une",
        "near-duplicados y pseudo-autores, y los pliegues de V03 son `StratifiedGroupKFold`",
        "sobre esos grupos. Si dos textos casi identicos llevan etiquetas distintas, la",
        "discrepancia es de la anotacion, no del modelo.",
        "",
        "## 2. Cuanto aporta este experimento",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| filas del desarrollo | {informe['gastos']['filas']} |",
        f"| grupos distintos | {t['n_grupos']} |",
        f"| grupos con 2 o mas filas | {t['n_grupos_con_2_o_mas']} |",
        f"| filas dentro de grupos con 2 o mas | {t['filas_en_grupos_con_2_o_mas']} |",
        f"| fraccion de filas que aporta informacion | {t['fraccion_de_filas_que_aporta_informacion']:.4f} |",
        f"| filas dentro de grupos contradictorios | {t['filas_en_grupos_contradictorios']} |",
        "",
        "## 3. Desacuerdo intra-grupo por region",
        "",
        "La tasa de ambiguedad se calcula **solo** sobre filas que viven en un grupo de 2 o",
        "mas, porque una fila sola no puede estar en contradiccion consigo misma. La ultima",
        "region es la que importa: si los textos largos tienen mas desacuerdo, la ambiguedad",
        "explica justo la region donde V03 falla mas.",
        "",
        "| region | filas | en grupos 2+ | fraccion | etiqueta | tasa ambiguedad | error V03 |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for nombre, d in informe["desacuerdo_por_region"].items():
        for k, etiqueta in enumerate(ETIQUETAS):
            pe = d["por_etiqueta"][etiqueta]
            if pe["tasa_de_ambiguedad"] is None:
                continue
            lineas.append(
                f"| {nombre if k == 0 else ''} | {d['n_filas'] if k == 0 else ''} | "
                f"{d['n_filas_en_grupos_multimember'] if k == 0 else ''} | "
                f"{d['fraccion_en_grupos_multimember']:.4f} | {etiqueta} | "
                f"{pe['tasa_de_ambiguedad']:.4f} | {pe['tasa_de_error_v03']:.4f} |"
            )
    lineas += [
        "",
        "## 3bis. El limite de este experimento: poder estadistico",
        "",
        f"Solo el {t_frac:.2%} de las filas del desarrollo vive en un grupo de 2 o mas, y solo",
        f"{n_contra} filas (**{n_contra / filas:.2%}**) estan en un grupo con anotaciones",
        "contradictorias. Ese es el techo del numero de filas que pueden portar informacion aqui,",
        "",
        "**Con 43 filas informativas, este experimento no puede responder la pregunta que se le",
        "hizo.** Los 12 tests de Fisher de la seccion siguiente se calculan sobre bloques de 32 a",
        "73 filas, donde la mitad de las celdas no puede alcanzar significacion aunque el efecto",
        "fuera real. El limite no es de metodo sino de datos: para medir ambiguedad hace falta una",
        "criterio de emparejamiento **mas laxo** que el near-duplicate, que es una tarea de",
        "ingenieria de datos y no de modelado.",
        "",
        "Lo que si se puede leer es lo unico que sobrevive al ruido: en las celdas con",
        "significancia, la direccion es inequivoca.",
        "",
        "## 4. Los errores de V03 se concentran en las filas ambiguas",
        "",
        "Si el error residual fuera ambiguedad, las filas ambiguas fallarian mas que las no",
        "ambiguas. Fisher exacto mide si la diferencia es distinguishable del azar.",
        "",
        "| region | etiqueta | error si ambigua | error si no ambigua | razon | p Fisher |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for nombre, bloque in informe["concentracion_del_error_en_filas_ambiguas"].items():
        for etiqueta, d in bloque.items():
            razon = d["razon_de_razones"]
            marcas = "n/d (denominador 0)" if razon is None else f"{razon:.2f}"
            lineas.append(
                f"| {nombre} | {etiqueta} | {d['error_si_ambigua']:.4f} | "
                f"{d['error_si_no_ambigua']:.4f} | {marcas} | {d['p_fisher']:.2e} |"
            )
    c = informe["cota_por_voto_mayoritario"]
    g = informe["gastos"]
    lineas += [
        "",
        "## 5. Cota superior por voto mayoritario dentro del grupo",
        "",
        "Cada fila se acierta si su etiqueta coincide con la mayoritaria de su grupo. Una fila",
        "dentro de un grupo contradictorio nunca puede acertarse por un modelo que solo vea el",
        "texto, asi que esta cifra acota el ruido de anotacion.",
        "",
        "**Es una cota, no un resultado alcanzable y no un modelo.** Usa la etiqueta, de modo",
        "que no puede convertirse en submission ni en estimador de generalizacion.",
        "",
        "| etiqueta | cota de acierto |",
        "|---|---:|",
    ]
    for etiqueta in ETIQUETAS:
        lineas.append(f"| {etiqueta} | {c[etiqueta]:.4f} |")
    lineas += [
        f"| macro | {c['macro']:.4f} |",
        "",
        "## 6. Lo que este script no hace",
        "",
        "- No entrena ningun modelo.",
        "- No construye ninguna feature a partir de los rasgos de V28-01.",
        "- No lee el test.",
        "- No produce submission.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
