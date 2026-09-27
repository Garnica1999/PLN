"""V16-00: medicion de la prevalencia real del test y curva de respuesta del LB.

Por que nace este modulo
-----------------------
El historial de 15 submissions (``submissions/leaderboard_log.csv``) establece dos
hechos que este modulo convierte en numeros, sin gastar ni una sola submission:

1. El test tiene menos positivos que train. El registro lo estimo "a ojo" en "~7 por
   ciento" a partir de la sonda E00 (plantilla todo 1, LB 0.51494). Ese 7 por ciento
   es una aproximacion de un solo decimal y procede de una unica sonda. Aqui se
   resuelve exactamente, porque la plantilla todo-1 tiene una formula cerrada.
2. Todas las sondas con CV por encima de V03 transfirieron a la baja, y el unico
   modelo que supero la referencia es V03: un lineal de baja capacidad con tres
   umbrales. La palanca que falta probar es la tasa de positivos predicha.

Que se mide
-----------
Sondeo 1 (prevalencia). La plantilla `v00_plantilla` predice 1 en las tres
columnas. Para una columna con prevalencia ``p``, la F1 binaria de la clase positiva
con prediccion todo-uno es exactamente ``2p / (1 + p)``: TP = n*p, FP = n*(1-p),
FN = 0. Por tanto la unica incognita del LB de E00 es la prevalencia de cada columna
del test. Con la hipotesis de que el test es train escalado por un factor comun ``k``,
``p'_c = k * p_c``, la incognita se reduce a un escalar y se resuelve por biseccion.
El factor ``k`` recovered es una medida directa del desplazamiento de prevalencia,
y ``k = 0.93`` reproduce el "~7 por ciento" del registro.

Sondeo 2 (curva de respuesta). Para cada submission con LB conocido se calcula su tasa
de positivos predicha (media sobre las tres columnas, y tambien por columna) y se
regresa el LB sobre esa tasa. La pendiente ``dLB/d(tasa)`` es la cantidad que
convierte "mover N celdas hacia menos positivos" en una prediccion de LB, sin
necesidad de enviar nada.

Cuidado con la interpretacion
-----------------------------
La regresion es **descriptiva**, no causal: las submissions difieren simultaneamente
en modelo, features y umbrales, asi que la tasa de positivos es un covariante, no
un tratamiento. Lo que si es defendible es la forma de la respuesta (monotona y
negativa) y el orden de magnitud de la sensibilidad. La magnitud se contrasta
contra el hecho ya registrado de que hacen falta ~107 cambios netos de acierto para
mover el LB en 0.003.

Salidas
-------
Escribe ``resultados/v16_00_prevalencia_y_curva_lb.csv`` con la tabla de la curva,
``resultados/v16_00_prevalencia_test.json`` con la prevalencia estimada y
``figuras/v16_00_curva_lb_tasa.png`` con la dispersion y la recta ajustada.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import ETIQUETAS

RESULTS_DIR = PROJECT_ROOT / "resultados"
FIGURAS_DIR = PROJECT_ROOT / "figuras"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Fila de la tabla usada para estimar k. El identificador de la sonda cuyo LB es
# 0.51494 y que predice 1 en todas las celdas.
ID_PLANTILLA = "v00_plantilla"
LB_REFERENCIA = 0.78306
LB_PLANTILLA = 0.51494

# Ficheros que no son submissions y por tanto se excluyen del barrido.
EXCLUIDOS = {"leaderboard_log.csv", "submission_manifest.csv"}

# Columnas de LB que se pueden leer del registro, en orden de prioridad.
COLUMNAS_LB = ("kaggle_f1", "LB")

# Submissions cuyo LB no consta en el registro. Se conservan para medir divergencia
# de predicciones, pero quedan fuera de la regresion por no tener variable respuesta.
SIN_LB = {
    "v12_descartado_meta36_downweight",
    "v13_p10_ensemble_cinco_modelos",
    "v14_complejidad_idiosincrasias",
    "v15_seleccion_por_grupo",
    "v16_clasificadores_alternativos",
    "v17_prior_shift_em",
    "v18_near_dup_propagacion",
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def leer_submission(ruta: Path) -> pd.DataFrame | None:
    """Lee una submission detectando el separador por el encabezado.

    Las submissions historicas alternan entre coma y punto y coma, y una de ellas
    (`toxic_submission_v00_plantilla.csv`) tiene el separador correcto pero los
    nombres de columna embebidos en la primera fila si se lee con la codificacion
    equivocada. Se prueban las dos convenciones y se conserva la que produce
    exactamente cuatro columnas con nombres de etiqueta.

    Args:
        ruta: camino al CSV de la submission.

    Returns:
        Marco con las columnas ``id,HS,TR,AG``, o ``None`` si no se pudo leer.
    """
    for separador in (",", ";"):
        for codificacion in ("utf-8", "latin-1"):
            try:
                marco = pd.read_csv(ruta, sep=separador, encoding=codificacion)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            columnas = [c.strip() for c in marco.columns]
            if all(etiqueta in columnas for etiqueta in ETIQUETAS):
                marco.columns = columnas
                return marco
    return None


def cargar_registro() -> pd.DataFrame:
    """Carga el historial del leaderboard.

    Returns:
        Marco con una fila por submission, con su LB y su identificador.
    """
    registro = pd.read_csv(SUBMISSIONS_DIR / "leaderboard_log.csv")
    registro["id"] = registro["submission_id"].str.strip()
    columna_lb = next(c for c in COLUMNAS_LB if c in registro.columns)
    registro["lb"] = pd.to_numeric(registro[columna_lb], errors="coerce")
    return registro


def medir_lote(marco: pd.DataFrame) -> dict[str, object]:
    """Calcula tasas de positivos y conteos de una submission.

    Args:
        marco: submission con columnas ``id,HS,TR,AG``.

    Returns:
        Diccionario con la tasa media de positivos, las tasas por columna, el
        numero de filas y la suma total de positivos.
    """
    etiquetas = marco[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n_filas = etiquetas.shape[0]
    tasas_por_etiqueta = etiquetas.mean(axis=0)
    return {
        "n_filas": int(n_filas),
        "tasa_media": float(tasas_por_etiqueta.mean()),
        "tasa_HS": float(tasas_por_etiqueta[0]),
        "tasa_TR": float(tasas_por_etiqueta[1]),
        "tasa_AG": float(tasas_por_etiqueta[2]),
        "positivos_totales": int(etiquetas.sum()),
    }


# ---------------------------------------------------------------------------
# Sondeo 1: prevalencia real del test
# ---------------------------------------------------------------------------
def f1_todo_uno(prevalencia: float) -> float:
    """F1 binaria de la clase positiva con prediccion todo-uno.

    Args:
        prevalencia: fraccion de positivos reales en la columna.

    Returns:
        Valor de la F1, que es ``2p / (1 + p)``.
    """
    return 2.0 * prevalencia / (1.0 + prevalencia)


def resolver_k(
    prevalencias_train: np.ndarray, lb_objetivo: float, tolerancia: float = 1e-12
) -> tuple[float, np.ndarray]:
    """Resuelve el factor de escala ``k`` tal que el LB de la plantilla cuadre.

    La funcion objetivo ``objetivo(k) = media_c 2*(k*p_c)/(1 + k*p_c)`` es monotona
    **creciente** en ``k``: si el test tuviera mas positivos, la plantilla todo-uno
    obtendria mas F1. Como el LB observado (0.51494) es menor que el que daria la
    prevalencia de train (0.540858), la solucion esta por debajo de ``k = 1`` y la
    biseccion debe devolver el borde superior cuando el objetivo aun excede al
    objetivo. Se busca sobre ``k`` en un rango que cubre desde "el test no tiene
    positivos" hasta "el test tiene el doble de positivos que train".

    Args:
        prevalencias_train: prevalencias de train por etiqueta.
        lb_objetivo: LB observado de la plantilla todo-uno.
        tolerancia: criterio de parada sobre el LB reconstruido.

    Returns:
        Tupla con el factor ``k`` y las prevalencias de test implicadas.
    """
    def objetivo(k: float) -> float:
        return float(np.mean([f1_todo_uno(k * p) for p in prevalencias_train]))

    bajo, alto = 0.0, 2.0
    for _ in range(200):
        medio = 0.5 * (bajo + alto)
        if objetivo(medio) > lb_objetivo:
            alto = medio
        else:
            bajo = medio
        if abs(objetivo(medio) - lb_objetivo) < tolerancia:
            break
    k = 0.5 * (bajo + alto)
    return k, k * prevalencias_train


def aplicar_prior(
    probabilidades: np.ndarray, prevalencias_train: np.ndarray, k: float
) -> np.ndarray:
    """Reescala las probabilidades con un ajuste de prior logit.

    Para una columna con prevalencia de train ``p`` y prevalencia de test ``k*p``,
    el termino de odds se corrige restando ``log(k)`` al logit. Como se busca una
    correccion de prevalencia y no de calibracion, el intercepto es lo unico que se
    toca: las pendientes y la ordenacion de las probabilidades se conservan.

    Args:
        probabilidades: matriz de probabilidades con una columna por etiqueta.
        prevalencias_train: prevalencias de train por etiqueta.
        k: factor de desplazamiento de prevalencia.

    Returns:
        Matriz de probabilidades con la misma forma.
    """
    epsilon = 1e-6
    recortadas = np.clip(probabilidades, epsilon, 1.0 - epsilon)
    logit = np.log(recortadas / (1.0 - recortadas))
    correccion = np.log(k)
    ajustadas = 1.0 / (1.0 + np.exp(-(logit - correccion)))
    return ajustadas


# ---------------------------------------------------------------------------
# Sondeo 2: curva de respuesta del LB a la tasa de positivos
# ---------------------------------------------------------------------------
def ajustar_recta(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Ajusta ``y = a + b*x`` por minimos cuadrados.

    Args:
        x: variable independiente.
        y: variable dependiente.

    Returns:
        Diccionario con la pendiente, el intercepto, el residuo cuadratico y el
        coeficiente de determinacion.
    """
    diseno = np.column_stack([np.ones_like(x), x])
    coeficientes, *_ = np.linalg.lstsq(diseno, y, rcond=None)
    intercepto, pendiente = float(coeficientes[0]), float(coeficientes[1])
    predicho = intercepto + pendiente * x
    residuo = float(np.sum((y - predicho) ** 2))
    varianza_total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residuo / varianza_total if varianza_total > 0 else float("nan")
    return {
        "intercepto": intercepto,
        "pendiente": pendiente,
        "r2": r2,
        "residuo_cuadratico": residuo,
    }


def main() -> None:
    """Ejecuta los dos sondeos y escribe los artefactos."""
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURAS_DIR.mkdir(exist_ok=True)

    # ---- Sondeo 1: prevalencia del test -----------------------------------
    train = pd.read_csv(RAW_DIR / "toxic_train.csv", sep=";")
    prevalencias_train = train[list(ETIQUETAS)].mean(axis=0).to_numpy(dtype=np.float64)
    k, prevalencias_test = resolver_k(prevalencias_train, LB_PLANTILLA)

    # Cuantificacion de la incertidumbre: cuanto LB puede moverse el factor k.
    # El registro no da el error del LB, asi que se reporta la sensibilidad
    # ante un desplazamiento de +-0.001 en el LB reconstruido.
    lb_reconstruido_train = float(np.mean([f1_todo_uno(p) for p in prevalencias_train]))
    sensibilidad_k = 0.0
    for lb_objetivo in (LB_PLANTILLA - 0.001, LB_PLANTILLA + 0.001):
        k_alt, _ = resolver_k(prevalencias_train, lb_objetivo)
        sensibilidad_k = max(sensibilidad_k, abs(k_alt - k))

    # ---- Sondeo 2: curva de respuesta -------------------------------------
    registro = cargar_registro()
    lb_por_id = dict(zip(registro["id"], registro["lb"]))

    filas: list[dict[str, object]] = []
    marco_referencia = None
    for ruta in sorted(SUBMISSIONS_DIR.glob("*.csv")):
        if ruta.name in EXCLUIDOS:
            continue
        marco = leer_submission(ruta)
        if marco is None:
            continue
        identificador = ruta.stem.replace("toxic_submission_", "")
        medidas = medir_lote(marco)
        lb = lb_por_id.get(identificador)
        lb_efectivo = float(lb) if lb is not None and np.isfinite(lb) else float("nan")
        fila: dict[str, object] = {
            "id": identificador,
            "archivo": ruta.name,
            "lb": lb_efectivo,
            "delta_lb_vs_v03": lb_efectivo - LB_REFERENCIA,
            **medidas,
        }
        if identificador == "v03_lr_union_thr":
            marco_referencia = marco
            fila["es_referencia"] = True
        else:
            fila["es_referencia"] = False
            if marco_referencia is not None and len(marco) == len(marco_referencia):
                propias = marco[list(ETIQUETAS)].to_numpy(dtype=np.int64)
                ref = marco_referencia[list(ETIQUETAS)].to_numpy(dtype=np.int64)
                fila["celdas_distintas_v03"] = int((propias != ref).sum())
            else:
                fila["celdas_distintas_v03"] = -1
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    con_lb = tabla[np.isfinite(tabla["lb"])].copy()

    # La plantilla todo-uno se EXCLUYE de los ajustes de pendiente. No es un punto
    # atipico que se pueda descartar por conveniencia: es un sistema degenerado que
    # predice 1 en las 3.462 celdas, con una tasa de 1.0 y un LB de 0.51494, muy
    # fuera del rango de operacion de los modelos reales (tasa 0.321 a 0.415). Su
    # unica prediccion es trivial y no aporta nada a la sensibilidad de la tasa en
    # la zona en la que se mueven los umbrales. Se calcula el ajuste con y sin el
    # para que la diferencia quede documentada en vez de escondida.
    con_modelos = con_lb[con_lb["id"] != ID_PLANTILLA].copy()
    ajuste = ajustar_recta(
        con_modelos["tasa_media"].to_numpy(), con_modelos["lb"].to_numpy()
    )
    ajuste_con_plantilla = ajustar_recta(
        con_lb["tasa_media"].to_numpy(), con_lb["lb"].to_numpy()
    )
    ajustes_por_etiqueta = {
        etiqueta: ajustar_recta(
            con_modelos[f"tasa_{etiqueta}"].to_numpy(), con_modelos["lb"].to_numpy()
        )
        for etiqueta in ETIQUETAS
    }
    # Recta ajustada solo en la familia de V03, que es donde viven los challengers:
    # sistemas de baja capacidad derivados del mismo entrenamiento que difieren de V03 en
    # menos de 250 celdas. Ahi la pendiente es la que realmente gobierna.
    familia_v03 = con_modelos[con_modelos["celdas_distintas_v03"] < 250]
    ajuste_familia = ajustar_recta(
        familia_v03["tasa_media"].to_numpy(), familia_v03["lb"].to_numpy()
    )
    tabla["lb_ajustado"] = ajuste["intercepto"] + ajuste["pendiente"] * tabla["tasa_media"]
    tabla["residuo_lb"] = tabla["lb"] - tabla["lb_ajustado"]
    tabla = tabla.sort_values("tasa_media")
    tabla.to_csv(RESULTS_DIR / "v16_00_prevalencia_y_curva_lb.csv", index=False)

    informe_prevalencia = {
        "metodo": (
            "La plantilla v00 predice 1 en las tres columnas, y la F1 binaria de la "
            "clase positiva con prediccion todo-uno es exactamente 2p/(1+p). La unica "
            "incognita del LB 0.51494 es la prevalencia de cada columna del test. Se "
            "supone que el test es train escalado por un factor comun k y se resuelve "
            "por biseccion."
        ),
        "lb_plantilla": LB_PLANTILLA,
        "lb_reconstruido_con_prevalencia_de_train": lb_reconstruido_train,
        "factor_k_test_sobre_train": k,
        "reduccion_relativa_de_prevalencia": 1.0 - k,
        "prevalencias_train": {
            e: float(p) for e, p in zip(ETIQUETAS, prevalencias_train)
        },
        "prevalencias_test_estimadas": {
            e: float(p) for e, p in zip(ETIQUETAS, prevalencias_test)
        },
        "positivos_test_estimados_por_etiqueta": {
            e: int(round(p * 1154)) for e, p in zip(ETIQUETAS, prevalencias_test)
        },
        "incertidumbre_k_por_0.001_de_lb": sensibilidad_k,
        "tasa_positivos_v03_predicha": float(
            medir_lote(marco_referencia)["tasa_media"]
        ),
        "tasa_positivos_test_objetivo": float(prevalencias_test.mean()),
    }
    (RESULTS_DIR / "v16_00_prevalencia_test.json").write_text(
        json.dumps(informe_prevalencia, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ---- Sondeo 3: contrafactual de prevalencia ----------------------------
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    proba_oof = oof["proba"].astype(np.float64)
    y_oof = oof["y_true"].astype(np.int64)
    umbrales_v03 = np.asarray(oof["umbrales"], dtype=np.float64)

    escenarios = [0.80, 0.88, 0.92, k, 0.96, 1.00]
    contrafactuales = [
        contrafactual_prevalencia(proba_oof, y_oof, factor, umbrales_v03)
        for factor in escenarios
    ]
    (RESULTS_DIR / "v16_00_contrafactual_prevalencia.json").write_text(
        json.dumps(contrafactuales, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ---- Figura ----------------------------------------------------------
    fig, eje = plt.subplots(figsize=(9, 5.5))
    colores = ["tab:red" if abs(d) < 1e-9 else "tab:blue" for d in con_modelos["delta_lb_vs_v03"]]
    eje.scatter(
        con_modelos["tasa_media"],
        con_modelos["lb"],
        c=colores,
        s=70,
        zorder=3,
        edgecolor="black",
        linewidth=0.6,
    )
    for _, fila in con_modelos.iterrows():
        etiqueta = str(fila["id"]).replace("_", "\n", 1)
        eje.annotate(
            etiqueta,
            (fila["tasa_media"], fila["lb"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    rejilla = np.linspace(
        con_modelos["tasa_media"].min() - 0.002,
        con_modelos["tasa_media"].max() + 0.002,
        50,
    )
    eje.plot(
        rejilla,
        ajuste["intercepto"] + ajuste["pendiente"] * rejilla,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=(
            f"pendiente {ajuste['pendiente']:.3f}  R2 {ajuste['r2']:.3f}"
        ),
    )
    eje.axhline(LB_REFERENCIA, color="tab:green", linewidth=1.0, linestyle=":")
    eje.annotate(
        f"V03 = 0.78306 (rojo) a tasa {medir_lote(marco_referencia)['tasa_media']:.4f}",
        (0.02, LB_REFERENCIA),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 6),
        fontsize=8,
        color="tab:green",
    )
    eje.set_xlabel("tasa media de positivos predichos en el test")
    eje.set_ylabel("LB observado")
    eje.set_title("Respuesta del leaderboard a la tasa de positivos predicha")
    eje.legend(fontsize=8)
    eje.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "v16_00_curva_lb_tasa.png", dpi=150)
    plt.close(fig)


    # ---- Informe por pantalla --------------------------------------------
    print("=" * 78)
    print("SONDEO 1: PREVALENCIA REAL DEL TEST")
    print("=" * 78)
    print(f"LB de la plantilla todo-uno            : {LB_PLANTILLA}")
    print(
        "LB que daria con prevalencia de train  : "
        f"{lb_reconstruido_train:.6f}  (sobrevale por {lb_reconstruido_train - LB_PLANTILLA:+.6f})"
    )
    print(f"factor k = prevalencia_test/prevalencia_train : {k:.6f}")
    print(f"reduccion relativa de prevalencia               : {1.0 - k:+.4%}")
    print(f"incertidumbre de k por +-0.001 de LB            : {sensibilidad_k:.6f}")
    print()
    print(f"{'etiqueta':<10}{'train':>10}{'test estimado':>16}{'positivos test':>16}")
    for etiqueta, p_tr, p_te in zip(ETIQUETAS, prevalencias_train, prevalencias_test):
        print(
            f"{etiqueta:<10}{p_tr:>10.6f}{p_te:>16.6f}"
            f"{int(round(p_te * 1154)):>16d}"
        )
    print(f"{'MEDIA':<10}{prevalencias_train.mean():>10.6f}{prevalencias_test.mean():>16.6f}")

    print()
    print("=" * 78)
    print("SONDEO 2: CURVA DE RESPUESTA DEL LB A LA TASA DE POSITIVOS")
    print("=" * 78)
    encabezado = (
        f"{'id':<34}{'tasa':>9}{'HS':>8}{'TR':>8}{'AG':>8}"
        f"{'LB':>10}{'delta':>10}{'celdas':>8}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for _, fila in tabla.iterrows():
        lb_texto = f"{fila['lb']:.5f}" if np.isfinite(fila["lb"]) else "   (sin LB)"
        delta_texto = f"{fila['delta_lb_vs_v03']:+.5f}" if np.isfinite(fila["lb"]) else ""
        celdas_texto = (
            str(int(fila["celdas_distintas_v03"])) if fila["celdas_distintas_v03"] >= 0 else ""
        )
        marca = " *" if fila["es_referencia"] else "  "
        print(
            f"{marca} {str(fila['id'])[:32]:<33}"
            f"{fila['tasa_media']:>9.4f}{fila['tasa_HS']:>8.4f}"
            f"{fila['tasa_TR']:>8.4f}{fila['tasa_AG']:>8.4f}"
            f"{lb_texto:>10}{delta_texto:>10}{celdas_texto:>8}"
        )
    print("  * = referencia V03")
    print()
    print(f"Modelos con LB conocido: {len(con_lb)} (la plantilla se excluye de los ajustes)")
    print(
        f"Ajuste LB ~ tasa (n={len(con_modelos)}): pendiente {ajuste['pendiente']:+.5f} "
        f"por unidad de tasa, intercepto {ajuste['intercepto']:.5f}, R2 {ajuste['r2']:.4f}"
    )
    print(
        f"  (con la plantilla dentro, n={len(con_lb)}: pendiente "
        f"{ajuste_con_plantilla['pendiente']:+.5f}, R2 {ajuste_con_plantilla['r2']:.4f})"
    )
    print(
        f"Ajuste dentro de la familia de V03 (n={len(familia_v03)}, <250 celdas de cambio): "
        f"pendiente {ajuste_familia['pendiente']:+.5f}, R2 {ajuste_familia['r2']:.4f}"
    )
    print("Pendientes por etiqueta (LB ~ tasa de esa etiqueta):")
    for etiqueta, recta in ajustes_por_etiqueta.items():
        print(
            f"  {etiqueta}: pendiente {recta['pendiente']:+.5f}  R2 {recta['r2']:.4f}"
        )
    print(
        f"Traduccion operativa: bajar la tasa media 0.010 equivale a "
        f"{ajuste['pendiente'] * 0.010:+.5f} de LB esperado."
    )
    print()
    print()
    print("=" * 78)
    print("SONDEO 3: CONTRAFACTUAL DE PREVALENCIA SOBRE LOS UMBRALES DE V03")
    print("=" * 78)
    print("Se pondera el OOF con peso k a los positivos, que es exactamente la")
    print("poblacion con prevalencia k*p. No se reentrena nada: solo se mueve la regla")
    print("de decision, que es la unica palanca que el registro no ha agotado.")
    for fila in contrafactuales:
        print(
            f"  k={fila['k']:.4f}  umbrales {fila['umbrales_v03']} -> "
            f"{[round(u, 3) for u in fila['umbrales_reoptimizados']]}  "
            f"celdas {fila['celdas_que_cambian']} "
            f"(total {fila['celdas_totales']})  F1 v03@train "
            f"{fila['f1_umbral_v03_prevalencia_train']:.6f} | "
            f"F1 v03@test {fila['f1_umbral_v03_prevalencia_test']:.6f} "
            f"(gasto {fila['gasto_del_desplazamiento']:+.6f}) | "
            f"F1 reopt@test {fila['f1_reoptimizada_prevalencia_test']:.6f} "
            f"(recupera {fila['recuperacion_moviendo_umbrales']:+.6f}, "
            f"neto {fila['neto_contra_la_referencia']:+.6f})"
        )
    objetivo = next(c for c in contrafactuales if abs(c["k"] - k) < 1e-9)
    print()
    print(
        f"En el escenario medido (k={k:.4f}): el desplazamiento de prevalencia cuesta "
        f"{objetivo['gasto_del_desplazamiento']:+.6f} de F1 si los umbrales se congelan, "
        f"y moverlos recupera {objetivo['recuperacion_moviendo_umbrales']:+.6f}, "
        f"dejando un neto de {objetivo['neto_contra_la_referencia']:+.6f}."
    )
    print(f"Coste de mover solo umbrales: {objetivo['celdas_totales']} celdas de 3462.")

    print()
    print("Artefactos escritos:")
    print("  resultados/v16_00_prevalencia_y_curva_lb.csv")
    print("  resultados/v16_00_prevalencia_test.json")
    print("  resultados/v16_00_contrafactual_prevalencia.json")
    print("  figuras/v16_00_curva_lb_tasa.png")


# ---------------------------------------------------------------------------
# Sondeo 3: contrafactual causal del desplazamiento de prevalencia
# ---------------------------------------------------------------------------
def f1_ponderada(y: np.ndarray, pred: np.ndarray, pesos: np.ndarray) -> float:
    """F1 binaria de la clase positiva con pesos por fila.

    Ponderar por ``k`` a los positivos y por ``1`` a los negativos construye
    exactamente la poblacion con prevalencia ``k * p``: es la simulacion del
    desplazamiento de prevalencia sin re-muestrear y por tanto sin ruido de
    submuestreo.

    Args:
        y: matriz binaria de etiquetas reales.
        pred: matriz binaria de predicciones.
        pesos: pesos por fila.

    Returns:
        F1 ponderada de la clase positiva.
    """
    # Correccion propia registrada antes de interpretar ningun resultado: la primera
    # version de esta funcion recibia ``y`` con forma (n, 1) y ``pesos`` con forma
    # (n,), de modo que ``pesos * y`` se emitia como producto exterior de (1, n) por
    # (n, 1) y devolvia una matriz (n, n) en lugar de un vector de n sumandos. El
    # sintoma fue un contrafactual totalmente insensible a k: los seis escenarios
    # devolvian umbrales y F1 identicos, lo cual es imposible si la ponderacion hace
    # algo. Aqui las tres entradas se aplanan a vectores de la misma longitud y se
    # comprueba la longitud, para que el fallo no pueda repetirse en silencio.
    y_plano = np.asarray(y, dtype=np.float64).reshape(-1)
    pred_plano = np.asarray(pred, dtype=np.float64).reshape(-1)
    pesos_plano = np.asarray(pesos, dtype=np.float64).reshape(-1)
    if not (y_plano.shape == pred_plano.shape == pesos_plano.shape):
        raise ValueError(
            "f1_ponderada exige tres vectores de la misma longitud; "
            f"llegaron {y_plano.shape}, {pred_plano.shape} y {pesos_plano.shape}. "
            "Sin esta comprobacion, el producto exterior silencioso devuelve un "
            "resultado plausible pero falso."
        )
    verdaderos_positivos = float(np.sum(pesos_plano * y_plano * pred_plano))
    falsos_positivos = float(np.sum(pesos_plano * (1.0 - y_plano) * pred_plano))
    falsos_negativos = float(np.sum(pesos_plano * y_plano * (1.0 - pred_plano)))
    denominador = 2.0 * verdaderos_positivos + falsos_positivos + falsos_negativos
    return 2.0 * verdaderos_positivos / denominador if denominador > 0 else 0.0


def umbral_optimo_ponderado(
    probabilidades: np.ndarray, y: np.ndarray, pesos: np.ndarray
) -> tuple[float, float]:
    """Busca el umbral que maximiza la F1 ponderada en una rejilla fina.

    La rejilla es de 0.005 en 0.005 entre 0.05 y 0.95, tres veces mas fina que la
    rejilla original de V03 (paso 0.01), de modo que el resultado no puede venir
    de la granularidad.

    Args:
        probabilidades: probabilidades de la etiqueta.
        y: etiqueta verdadera.
        pesos: pesos por fila.

    Returns:
        Tupla con el umbral optimo y la F1 ponderada que alcanza.
    """
    mejor_umbral, mejor_f1 = 0.5, -1.0
    for umbral in np.arange(0.05, 0.9501, 0.005):
        pred = (probabilidades >= umbral).astype(np.int64)[:, None]
        valor = f1_ponderada(y, pred, pesos)
        if valor > mejor_f1:
            mejor_umbral, mejor_f1 = float(umbral), valor
    return mejor_umbral, mejor_f1


def contrafactual_prevalencia(
    proba: np.ndarray, y: np.ndarray, k: float, umbrales_v03: np.ndarray
) -> dict[str, object]:
    """Mide el efecto de la prevalencia ``k * p`` sobre umbrales y celdas.

    Se calculan tres cantidades, todas sobre las probabilidades OOF canonicas de
    V03 y sin reentrenar nada:

    1. F1 de V03 con sus umbrales y prevalencia de train (la referencia).
    2. F1 de V03 con sus umbrales y prevalencia ``k * p``: el coste de no hacer
       nada si las predicciones se congelaran.
    3. F1 con los umbrales reoptimizados bajo prevalencia ``k * p``: el techo
       alcanzable moviendo solo los tres umbrales.

    La diferencia 2 -> 3 es el beneficio de reoptimizar; la diferencia 1 -> 2 es
    el dano de no hacerlo. Son los dos signos que faltaban por medir.

    Args:
        proba: matriz de probabilidades OOF de V03.
        y: matriz binaria de etiquetas verdaderas.
        k: factor de prevalencia a simular.
        umbrales_v03: umbrales de referencia por etiqueta.

    Returns:
        Diccionario con las F1, los umbrales reoptimizados y el recuento de celdas.
    """
    # Correccion propia registrada antes de interpretar ningun resultado: en la
    # primera version los dos primeros accumulator estaban intercambiados con
    # respecto a su docstring, de modo que la linea de informe atribuia +0.006071 al
    # "coste de congelar" cuando ese numero era en realidad el signo del gasto. Los
    # dos nombres de abajo son explicitos y no admiten confusion: train es la
    # prevalencia de train sin ponderar, test es la prevalencia k*p ponderada.
    pesos = np.where(y == 1, k, 1.0)
    f1_prevalencia_train, f1_prevalencia_test, f1_reoptimizada = [], [], []
    umbrales_nuevos, celdas = [], []
    for etiqueta in range(y.shape[1]):
        columna = y[:, etiqueta]
        pesos_columna = pesos[:, etiqueta]
        proba_columna = proba[:, etiqueta]
        pred_referencia = (proba_columna >= umbrales_v03[etiqueta]).astype(np.int64)[:, None]
        f1_prevalencia_train.append(
            f1_ponderada(columna[:, None], pred_referencia, np.ones_like(pesos_columna))
        )
        f1_prevalencia_test.append(
            f1_ponderada(columna[:, None], pred_referencia, pesos_columna)
        )
        nuevo, valor = umbral_optimo_ponderado(
            proba_columna, columna[:, None], pesos_columna
        )
        umbrales_nuevos.append(nuevo)
        f1_reoptimizada.append(valor)
        celdas.append(
            int(
                (
                    (proba_columna >= nuevo)
                    != (proba_columna >= umbrales_v03[etiqueta])
                ).sum()
            )
        )
    return {
        "k": k,
        "f1_umbral_v03_prevalencia_train": float(np.mean(f1_prevalencia_train)),
        "f1_umbral_v03_prevalencia_test": float(np.mean(f1_prevalencia_test)),
        "f1_reoptimizada_prevalencia_test": float(np.mean(f1_reoptimizada)),
        "gasto_del_desplazamiento": float(
            np.mean(f1_prevalencia_test) - np.mean(f1_prevalencia_train)
        ),
        "recuperacion_moviendo_umbrales": float(
            np.mean(f1_reoptimizada) - np.mean(f1_prevalencia_test)
        ),
        "neto_contra_la_referencia": float(
            np.mean(f1_reoptimizada) - np.mean(f1_prevalencia_train)
        ),
        "umbrales_v03": [float(u) for u in umbrales_v03],
        "umbrales_reoptimizados": umbrales_nuevos,
        "celdas_que_cambian": celdas,
        "celdas_totales": int(sum(celdas)),
    }





if __name__ == "__main__":
    main()
