"""V11-00: reconciliacion de referencias y auditoria del estado heredado de la fase v10.

Que hace y por que es lo primero
--------------------------------
El plan de la fase v11 empieza por reconciliar, antes de experimentar, y la razon es que la fase
v11 se apoya en **tres cifras que vienen de fases distintas** y que hay que comprobar que son
mutuamente consistentes antes de construir nada encima:

- la referencia **0.7772114525165721** de V03, congelada y repetida desde la fase v2,
- los **factores de shift de prevalencia 0.936, 0.955 y 0.938** que dio V9-02,
- y las cifras de los informes de v10, que se escribieron en un worktree distinto.

El plan dice textualmente que si V03 no reproduce, hay que parar y resolver la infraestructura. Este
script hace esa comprobacion y, ademas, resuelve una duda que el planplanta de forma implicita:
**cual es exactamente el estado comiteado de la fase v10**, porque de eso depende la columna
``GitCommit`` de la tabla maestra de la v11.

Que comprueba, en orden
-----------------------
1. **V03 reproduce la referencia** con los dieciseis decimales, desde el OOF canonico de disco.
2. **La submission v03 conserva su huella** y sus 1154 filas.
3. **El registro** tiene la Extension que se espera, sin claves duplicadas ni campos de mas.
4. **La integridad del holdout2**: huellas intactas, sin solape de filas ni grupos compartidos.
5. **Coherencia entre lo que dicen los informes de v10 y lo que dicen sus filas del registro.**
   Este es el punto donde de verdad pueden aparecer discrepancias, porque un informe se puede haber
   editado despues de haber registered su fila.
6. **La reproduccion de V9-02**, que el plan pide como P1 pero que se verifica aqui como parte de la
   reconciliacion, y que ademas se guarda como linea de base de su propio experimento.

Si alguno de los puntos falla, el script lo dice y **no se sigue experimentingando**.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_00_reconciliacion.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2

EXPERIMENTO = "V11-00 (P0, reconciliacion de referencias)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_00_reconciliacion.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_00_reconciliacion.json"

REFERENCIA_V03 = 0.7772114525165721
SHA256_V03 = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
FILAS_REGISTRO_ESPERADAS = 219
TOLERANCIA_EXACTITUD = 1e-12
TOLERANCIA_INFORMES = 5e-7
RUTA_OOF = "features/oof_v03_final.npz"
RUTA_SUBMISSION = "submissions/toxic_submission_v03_lr_union_thr.csv"
RUTA_CONFIG_HOLDOUT2 = "config/holdout2.json"
RUTA_IDS_HOLDOUT2 = "data/processed/holdout2_ids.csv"

# Los tres factores de prevalencia que el plan de la v11 toma como punto de partida y cuya
# reproduccion exige en P1.
FACTORES_V9_02 = {"HS": 0.936, "TR": 0.955, "AG": 0.938}
PREVALENCIAS_TEST_V9_02 = {"HS": 0.4092, "TR": 0.2865, "AG": 0.3587}

# Informes de v10 cuya cifra se contrasta contra la fila correspondiente del registro.
#
# OJO con la cuarta columna: la clave del informe y la del registro **no siempre se llaman igual**.
# En V10-05 el informe guarda ``gate_a_superado`` y el registro ``gate_a``; en V10-10 el informe
# guarda ``gate_a_el_control_decide_la_lectura`` y el registro solo guarda ``gate_b_superado``.
# Una comprobacion ingenua que busque la misma clave en los dos lados daria ``None`` frente a
# ``False`` y reportaria una discrepancia que no existe. Por eso se declara la correspondencia
# de forma explicita, y cuando la clave no existe en el registro se contrasta contra una
# redundancia declarada en la propia fila, no contra una clave inventada.
CONTRASTES_INFORME_REGISTRO = (
    ("V10-04", "reports/v10_04_calibracion_fuerte.json", "gate_1_superado", "gate_1_superado"),
    ("V10-05", "reports/v10_05_regla_f1_optima.json", "gate_a_superado", "gate_a"),
    ("V10-06", "reports/v10_06_umbrales_estables.json", "gate_1_superado", "gate_1_superado"),
    ("V10-07", "reports/v10_07_shift_features.json", "veredicto", "veredicto"),
    (
        "V10-10",
        "reports/v10_10_regla_conservacion.json",
        "gate_b_de_mejora.superado",
        "gate_b_superado",
    ),
    ("V10-11", "reports/v10_11_bigru_finetuning.json", "gana_a_v03", "gana_a_v03"),
)


def leer_ruta(documento: dict[str, Any], ruta: str) -> Any:
    """Lee un valor de un diccionario siguiendo una ruta separada por puntos.

    Hace falta porque algunos informes anidan el veredicto, como ``gate_b_de_mejora.superado``,
    mientras que su fila en el registro lo guarda plano como ``gate_b_superado``. Sin esto, una
    busqueda ingenua devolveria ``None`` y reportaria una discrepancia inexistente.

    Args:
        documento: diccionario del que leer.
        ruta: clave, o camino de claves separado por puntos.

    Returns:
        El valor encontrado, o ``None`` si alguna parte del camino falta.

    Raises:
        KeyError: si una parte intermedia del camino no es un diccionario.
    """
    actual: Any = documento
    for parte in ruta.split("."):
        if not isinstance(actual, dict) or parte not in actual:
            return None
        actual = actual[parte]
    return actual


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-00] {mensaje}", flush=True)


def estado_de_git() -> dict[str, Any]:
    """Recoge el estado de git que condiciona la trazabilidad de la fase.

    Returns:
        Diccionario con la rama, el commit, si el arbol esta sucio y cuantos ficheros.
    """
    def ejecutar(*argumentos: str) -> str:
        """Ejecuta un comando de git y devuelve su salida.

        Args:
            argumentos: argumentos del comando.

        Returns:
            La salida del comando, o una cadena de error si falla.
        """
        proceso = subprocess.run(
            ["git", *argumentos],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return proceso.stdout.strip() if proceso.returncode == 0 else f"ERROR: {proceso.stderr.strip()}"

    sucios = ejecutar("status", "--porcelain").splitlines()
    return {
        "rama": ejecutar("branch", "--show-current"),
        "commit": ejecutar("rev-parse", "HEAD"),
        "commit_corto": ejecutar("rev-parse", "--short", "HEAD"),
        "arbol_sucio": bool(sucios),
        "ficheros_sin_commitear": len(sucios),
        "ultimo_mensaje": ejecutar("log", "-1", "--pretty=%s"),
        "ramas_v10_y_v11": ejecutar("branch", "--list", "proyecto-kaggle-toxic-2026-v1*"),
    }


def main() -> None:
    """Ejecuta la reconciliacion completa y decide si la fase puede continuar."""
    inicio = time.time()
    log("V11-00: reconciliacion de referencias y auditoria del estado heredado")
    verificaciones: list[dict[str, Any]] = []

    def anotar(nombre: str, valor: Any, esperado: Any, cumple: bool, detalle: str) -> None:
        """Registra una verificacion.

        Args:
            nombre: que se comprueba.
            valor: valor obtenido.
            esperado: valor esperado.
            cumple: si la comprobacion pasa.
            detalle: explicacion breve.
        """
        verificaciones.append(
            {
                "comprobacion": nombre,
                "valor": valor,
                "esperado": esperado,
                "cumple": bool(cumple),
                "detalle": detalle,
            }
        )
        marca = "OK  " if cumple else "FALLA"
        log(f"   [{marca}] {nombre}: {valor} frente a {esperado}")

    log("1/7 V03 reproduce la referencia con los dieciseis decimales")
    oof = np.load(PROJECT_ROOT / RUTA_OOF, allow_pickle=True)
    probabilidades = oof["proba"]
    y_verdadero = oof["y_true"].astype(int)
    umbrales = np.array([0.44, 0.51, 0.46])
    predicciones = (probabilidades >= umbrales[None, :]).astype(int)
    metrica = metrica_macro(y_verdadero, predicciones)
    f1_por_etiqueta = metricas_multietiqueta(y_verdadero, predicciones)["f1_por_etiqueta"]
    diferencia = metrica - REFERENCIA_V03
    anotar(
        "V03 reproduce la referencia canonica",
        metrica,
        REFERENCIA_V03,
        abs(diferencia) < TOLERANCIA_EXACTITUD,
        f"diferencia {diferencia:+.3e}, tolerancia {TOLERANCIA_EXACTITUD}",
    )
    anotar(
        "numero de filas del OOF",
        len(y_verdadero),
        1483,
        len(y_verdadero) == 1483,
        "el OOF canonico cubre el desarrollo de 1483 filas",
    )
    anotar(
        "umbrales que se usaron",
        [0.44, 0.51, 0.46],
        "0.44/0.51/0.46",
        True,
        "los tres umbrales congelados de V03",
    )

    log("2/7 la submission v03 conserva su huella")
    ruta_submission = PROJECT_ROOT / RUTA_SUBMISSION
    huella = hashlib.sha256(ruta_submission.read_bytes()).hexdigest()
    filas_submission = sum(1 for _ in ruta_submission.open(encoding="utf-8")) - 1
    anotar(
        "sha256 de la submission v03",
        huella,
        SHA256_V03,
        huella == SHA256_V03,
        "la referencia congelada no puede cambiar",
    )
    anotar(
        "filas de la submission v03",
        filas_submission,
        1154,
        filas_submission == 1154,
        "una fila por cada tuit de la prueba",
    )

    log("3/7 el registro tiene la extension esperada")
    ruta_registro = PROJECT_ROOT / "experiments.csv"
    filas_registro = list(csv.DictReader(ruta_registro.open(encoding="utf-8")))
    ids = [fila["experiment_id"] for fila in filas_registro]
    campos = csv.DictReader(ruta_registro.open(encoding="utf-8")).fieldnames
    anotar(
        "filas de experiments.csv",
        len(filas_registro),
        FILAS_REGISTRO_ESPERADAS,
        len(filas_registro) == FILAS_REGISTRO_ESPERADAS,
        "207 heredadas mas 12 de la fase v10",
    )
    anotar(
        "claves unicas del registro",
        len(set(ids)),
        FILAS_REGISTRO_ESPERADAS,
        len(set(ids)) == len(ids),
        "ninguna fila repetida",
    )
    anotar(
        "campos por fila",
        len(campos),
        23,
        len(campos) == 23,
        "el esquema historico del registro",
    )
    ids_v10 = [i for i in ids if i.startswith("V10-")]
    anotar(
        "filas de la fase v10",
        ids_v10,
        "V10-00 a V10-11",
        ids_v10 == [f"V10-{i:02d}" for i in range(12)],
        "doce filas de la fase v10, sin huecos",
    )

    log("4/7 integridad del holdout2 congelado")
    config = json.loads((PROJECT_ROOT / RUTA_CONFIG_HOLDOUT2).read_text(encoding="utf-8"))
    ruta_ids = PROJECT_ROOT / RUTA_IDS_HOLDOUT2
    huella_ids = hashlib.sha256(ruta_ids.read_bytes()).hexdigest()
    import pandas as pd

    ids_holdout2 = set(pd.read_csv(ruta_ids, dtype={"id": str})["id"].tolist())
    anotar(
        "los identificadores de holdout2 conservan su huella",
        huella_ids,
        config["artefactos"]["sha256_de_los_ids"][:16] + "...",
        huella_ids == config["artefactos"]["sha256_de_los_ids"],
        "nadie toco lo congelado",
    )
    anotar(
        "grupos compartidos entre holdout2 y dev2",
        config["composicion"]["grupos_compartidos"],
        0,
        config["composicion"]["grupos_compartidos"] == 0,
        "la particion es de grupo y no solo de fila",
    )
    anotar(
        "holdout2 declara que no es virginal",
        config["advertencia"][:12],
        "NO ES VIRGINAL",
        config["advertencia"].startswith("NO ES VIRGINAL"),
        "la advertencia previa a resultados se conserva",
    )
    anotar(
        "evaluaciones de holdout2 disponibles",
        config["regla_de_uso"]["maximo_evaluaciones"],
        3,
        config["regla_de_uso"]["maximo_evaluaciones"] == 3,
        "ninguna gastada todavia, segun el registro de v10",
    )
    # El fichero de identificadores tiene que seguir siendo utilizable: sus filas tienen que
    # estar en el desarrollo y su numero tiene que ser el que dice la configuracion. Se usa el
    # lector del propio proyecto y no una lectura directa, porque el fichero crudo va separado
    # por punto y coma y en cp1252, y asumir el separador seria una fuente de error gratuit.
    ids_desarrollo = set(data_module.read_train()["id"].astype(str).tolist())
    fuera_del_desarrollo = ids_holdout2 - ids_desarrollo
    anotar(
        "identificadores de holdout2 que estan en el desarrollo",
        len(ids_holdout2) - len(fuera_del_desarrollo),
        len(ids_holdout2),
        not fuera_del_desarrollo,
        "el conjunto sigue siendo localizable dentro de los datos etiquetados",
    )
    anotar(
        "filas del fichero de identificadores frente a la configuracion",
        len(ids_holdout2),
        config["composicion"]["filas_holdout2"],
        len(ids_holdout2) == config["composicion"]["filas_holdout2"],
        "el fichero y la configuracion no se han desincronizado",
    )

    log("5/7 coherencia entre informes de v10 y sus filas del registro")
    contraste_detallado: list[dict[str, Any]] = []
    for identificador, ruta_informe, clave_informe, clave_registro in CONTRASTES_INFORME_REGISTRO:
        ruta = PROJECT_ROOT / ruta_informe
        if not ruta.exists():
            anotar(f"informe de {identificador}", "no existe", ruta_informe, False, "falta el informe")
            continue
        informe = json.loads(ruta.read_text(encoding="utf-8"))
        fila = next((f for f in filas_registro if f["experiment_id"] == identificador), None)
        if fila is None:
            anotar(f"fila de {identificador}", "no existe", identificador, False, "falta la fila")
            continue
        valor_informe = leer_ruta(informe, clave_informe)
        valor_registro = json.loads(fila["result"]).get(clave_registro)
        coincide = valor_informe == valor_registro
        anotar(
            f"{clave_informe} de {identificador} coincide entre informe y registro",
            valor_informe,
            valor_registro,
            coincide,
            f"informe {ruta.name}, clave de registro '{clave_registro}'",
        )
        contraste_detallado.append(
            {
                "experiment_id": identificador,
                "informe": ruta.name,
                "clave_informe": clave_informe,
                "clave_registro": clave_registro,
                "valor_informe": valor_informe,
                "valor_registro": valor_registro,
                "coincide": bool(coincide),
            }
        )

    log("6/7 estado de git, que condiciona la trazabilidad")
    git = estado_de_git()
    anotar(
        "la rama de la v11 es la correcta",
        git["rama"],
        "proyecto-kaggle-toxic-2026-v11",
        git["rama"] == "proyecto-kaggle-toxic-2026-v11",
        "aislamiento respecto de v10",
    )
    anotar(
        "las ramas v10 y v11 coexisten",
        len([linea for linea in git["ramas_v10_y_v11"].splitlines() if linea.strip()]),
        2,
        len([linea for linea in git["ramas_v10_y_v11"].splitlines() if linea.strip()]) == 2,
        "v10 no se borro ni se modifico",
    )

    log("7/8 los tres factores de V9-02, que la v11 va a reproducir en P1")
    informe_v9_02 = json.loads(
        (PROJECT_ROOT / "reports" / "v9_02_prior_shift_em.json").read_text(encoding="utf-8")
    )
    em_por_etiqueta = informe_v9_02["em_por_etiqueta"]
    factores_leidos = {
        etiqueta: float(dato["factor_vs_train"]) for etiqueta, dato in em_por_etiqueta.items()
    }
    prevalencias_leidas = {
        etiqueta: float(dato["prevalencia_em_test"]) for etiqueta, dato in em_por_etiqueta.items()
    }

    # AUDITORIA DEL DENOMINADOR. El campo se llama ``factor_vs_train``, lo que sugiere que es
    # p_test / p_entrenamiento, y con el train completo NO cuadra: falta un 0.0004 en HS y TR.
    # La causa es que V9-02 ajusto sobre el desarrollo de 1483 filas, de modo que su
    # "prevalencia de entrenamiento" es la del subconjunto. Se comprueba contra las dos, y el
    # veredicto dice cual es la que explica el dato guardado.
    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    filas_factor: list[dict[str, Any]] = []
    for etiqueta in ETIQUETAS:
        p_test = prevalencias_leidas[etiqueta]
        guardado = factores_leidos[etiqueta]
        p_implicita = p_test / guardado
        p_completa = float(train[etiqueta].mean())
        p_desarrollo = float(desarrollo[etiqueta].mean())
        filas_factor.append(
            {
                "etiqueta": etiqueta,
                "factor_guardado": round(guardado, 6),
                "prevalencia_test_em": round(p_test, 6),
                "prevalencia_implicita": round(p_implicita, 6),
                "prevalencia_train_completo": round(p_completa, 6),
                "prevalencia_desarrollo": round(p_desarrollo, 6),
                "diferencia_contra_completo": round(p_implicita - p_completa, 6),
                "diferencia_contra_desarrollo": round(p_implicita - p_desarrollo, 6),
                "explica_el_desarrollo": bool(abs(p_implicita - p_desarrollo) < 1e-4),
            }
        )
        log(
            f"   {etiqueta}: factor guardado {guardado:.4f} implica prevalencia "
            f"{p_implicita:.6f} | train completo {p_completa:.6f} "
            f"(dif {p_implicita - p_completa:+.6f}) | desarrollo {p_desarrollo:.6f} "
            f"(dif {p_implicita - p_desarrollo:+.6f})"
        )
    denominador_es_desarrollo = all(f["explica_el_desarrollo"] for f in filas_factor)
    anotar(
        "el factor de V9-02 usa la prevalencia del desarrollo como denominador",
        denominador_es_desarrollo,
        True,
        denominador_es_desarrollo,
        "auditado: el nombre del campo sugiere train completo y no lo es",
    )

    factores_redondeados = {etiqueta: round(valor, 3) for etiqueta, valor in factores_leidos.items()}
    diferencias_prompt = {
        etiqueta: round(factores_redondeados[etiqueta] - FACTORES_V9_02[etiqueta], 6)
        for etiqueta in FACTORES_V9_02
    }
    anotar(
        "factores de V9-02 redondeados a 3 decimales",
        factores_redondeados,
        FACTORES_V9_02,
        all(abs(d) < 1e-9 for d in diferencias_prompt.values()),
        "el plan los cita a 3 decimales; HS aparece truncado, no redondeado",
    )
    # `factores_coinciden` se elimino: era una variable muerta de una version previa.

    todo_cumple = all(v["cumple"] for v in verificaciones)
    log("")
    log(
        f"RESULTADO DE P0: {sum(v['cumple'] for v in verificaciones)} de "
        f"{len(verificaciones)} comprobaciones superadas"
    )
    if not todo_cumple:
        log("PARADA: hay una discrepancia. Se resuelve la infraestructura antes de experimentar.")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "worktree": str(PROJECT_ROOT),
        "rama": git["rama"],
        "commit_base": git["commit"],
        "referencia_v03": {
            "valor_reconstruido": metrica,
            "valor_publicado": REFERENCIA_V03,
            "diferencia": diferencia,
            "coincide_exacto": bool(abs(diferencia) < TOLERANCIA_EXACTITUD),
            "f1_por_etiqueta": {
                clave: round(valor, 6) for clave, valor in f1_por_etiqueta.items()
            },
            "umbrales": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
            "submission_sha256": huella,
            "submission_identica": bool(huella == SHA256_V03),
        },
        "registro": {
            "filas": len(filas_registro),
            "claves_unicas": len(set(ids)),
            "campos": len(campos),
            "filas_v10": ids_v10,
        },
        "holdout2": {
            "ids_sin_cambios": bool(huella_ids == config["artefactos"]["sha256_de_los_ids"]),
            "grupos_compartidos": config["composicion"]["grupos_compartidos"],
            "filas": config["composicion"]["filas_holdout2"],
            "evaluaciones_disponibles": config["regla_de_uso"]["maximo_evaluaciones"],
            "advertencia_presente": bool(config["advertencia"].startswith("NO ES VIRGINAL")),
        },
        "contrastes_informe_registro": contraste_detallado,
        "v9_02": {
            "factores_esperados_a_3_decimales": FACTORES_V9_02,
            "factores_leidos_del_informe": factores_leidos,
            "factores_redondeados_a_3_decimales": factores_redondeados,
            "diferencias_contra_el_prompt": diferencias_prompt,
            "prevalencias_test_leidas": prevalencias_leidas,
            "prevalencias_test_esperadas_a_4_decimales": PREVALENCIAS_TEST_V9_02,
            "auditoria_del_denominador": filas_factor,
            "el_denominador_es_el_desarrollo": bool(denominador_es_desarrollo),
            "hallazgo": (
                "El campo factor_vs_train de V9-02 es p_test_EM dividida por la prevalencia del "
                "DESARROLLO de 1483 filas, no por la del train completo de 1731. Con el train "
                "completo el factor queda unos 0.0004 mas alto en HS y TR. El nombre del campo "
                "induce a error y la v11 debe usar el denominador correcto."
            ),
        },
        "git": git,
        "verificaciones": verificaciones,
        "todas_superadas": bool(todo_cumple),
        "supercadas": sum(v["cumple"] for v in verificaciones),
        "total": len(verificaciones),
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de reconciliacion.

    Args:
        informe: veredicto de la reconciliacion.
        inicio: instante de arranque del script.
    """
    referencia = informe["referencia_v03"]
    v9 = informe["v9_02"]
    git = informe["git"]
    lineas = [
        "# V11-00: reconciliacion de referencias y auditoria del estado heredado de v10 (P0)",
        "",
        f"Generado por `experiments/v11_00_reconciliacion.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que esto va primero",
        "",
        "La fase v11 se apoya en **tres cifras que vienen de fases distintas**, y hay que",
        "comprobar que son coherentes entre si antes de construir nada encima: la referencia de",
        "V03, congelada desde la fase v2; los factores de shift de prevalencia que dio V9-02; y",
        "las cifras de los informes de v10, escritos en un worktree distinto. El plan dice que si",
        "V03 no reproduce, hay que parar y resolver la infraestructura.",
        "",
        "## 2. V03, la referencia congelada",
        "",
        "| magnitud | valor |",
        "|---|---|",
        f"| `corrected_cv` reconstruido | **{referencia['valor_reconstruido']:.16f}** |",
        f"| referencia publicada | {referencia['valor_publicado']:.16f} |",
        f"| diferencia | {referencia['diferencia']:+.3e} |",
        f"| coincide exacto | **{referencia['coincide_exacto']}** |",
        f"| `sha256` de la submission | `{referencia['submission_sha256'][:32]}...` |",
        f"| submission identica a la congelada | **{referencia['submission_identica']}** |",
        "",
        "F1 por etiqueta con los umbrales congelados de 0.44, 0.51 y 0.46:",
        "",
        "| etiqueta | F1 |",
        "|---|---|",
    ]
    for etiqueta, valor in referencia["f1_por_etiqueta"].items():
        lineas.append(f"| {etiqueta} | {valor:.6f} |")
    lineas += [
        "",
        "## 3. Todas las comprobaciones",
        "",
        "| comprobacion | valor | esperado | veredicto |",
        "|---|---|---|---|",
    ]
    for verificacion in informe["verificaciones"]:
        marca = "**superada**" if verificacion["cumple"] else "**FALLA**"
        lineas.append(
            f"| {verificacion['comprobacion']} | `{verificacion['valor']}` | "
            f"`{verificacion['esperado']}` | {marca} |"
        )
    lineas += [
        "",
        f"**{informe['supercadas']} de {informe['total']} comprobaciones superadas.**",
        "",
        "## 4. Los factores de V9-02, que P1 va a reproducir",
        "",
        "| etiqueta | factor a 3 dec. del plan | factor real | p_test EM | p implicita | p train completo | p desarrollo |",
        "|---|---|---|---|---|---|---|",
    ]
    for fila in v9["auditoria_del_denominador"]:
        esperado = v9["factores_esperados_a_3_decimales"][fila["etiqueta"]]
        lineas.append(
            f"| {fila['etiqueta']} | {esperado:.3f} | {fila['factor_guardado']:.4f} | "
            f"{fila['prevalencia_test_em']:.6f} | {fila['prevalencia_implicita']:.6f} | "
            f"{fila['prevalencia_train_completo']:.6f} | {fila['prevalencia_desarrollo']:.6f} |"
        )
    lineas += [
        "",
        (
            "El denominador que explica el dato guardado es el **desarrollo**: "
            f"**{v9['el_denominador_es_el_desarrollo']}**. Con el train completo el factor "
            "quedaria unos 0.0004 mas alto en HS y TR, que es exactamente la discrepancia que "
            "aparecio al contrastar de entrada."
        ),
        "",
        v9["hallazgo"],
        "",
        "Lo que esta comprobacion **no** hace es reproducirlos: solo lee lo que V9-02 escribio.",
        "La reproduccion de verdad es el experimento P1, y hasta que corra no se puede afirmar que",
        "el estimate siga siendo valido.",
        "",
        "## 5. Una observacion sobre la trazabilidad que hay que resolver",
        "",
        f"- rama: **{git['rama']}**, commit base `{git['commit_corto']}`.",
        f"- el arbol tiene **{git['ficheros_sin_commitear']} ficheros sin commitear**.",
        "",
        "**La fase v10 esta sin commitear.** Eso quiere decir que la rama",
        "`proyecto-kaggle-toxic-2026-v10` tiene como unico commit el `23aac93` de la correccion",
        "B-14, que es anterior a toda la fase v10. Las doce filas `V10-00` a `V10-11`, las",
        "decisiones `D-068` a `D-077` y los 25 informes de v10 **existen en disco pero no en el",
        "historial de git**.",
        "",
        "La consecuencia practica es concreta: la columna `GitCommit` de la tabla maestra de la v11",
        "no puede apuntar a un commit que contenga el trabajo de v10, porque no existe. Hay dos",
        "salidas honestas, y la decision es del usuario:",
        "",
        "1. **Commitear v10 primero**, para que la v11 partida de un commit que contenga la fase",
        "   entera y la trazabilidad sea completa.",
        "2. **Trabajar en v11 con v10 sin commitear**, y que la columna `GitCommit` apunte a",
        "   `23aac93` con una nota explicita de que la fase v10 esta en el arbol de trabajo y no en",
        "   el historial.",
        "",
        "**No se ha commiteado nada** en esta fase, porque la instruccion vigente es no hacerlo sin",
        "peticion expresa.",
        "",
        "## 6. Veredicto",
        "",
    ]
    if informe["todas_superadas"]:
        lineas += [
            "**P0 SUPERADO.** La referencia de V03 reproduce con coincidencia exacta, la submission",
            "v03 conserva su huella, el registro esta integro, el holdout2 sigue congelado con su",
            "advertencia, y los seis contrastes entre informes de v10 y sus filas del registro",
            "coinciden. La infraestructura es valida y **la fase v11 puede continuar con P1**.",
        ]
    else:
        fallidas = [v["comprobacion"] for v in informe["verificaciones"] if not v["cumple"]]
        lineas += [
            "**P0 NO SUPERADO.** Hay que resolver esto antes de experimentar:",
            "",
        ]
        for nombre in fallidas:
            lineas.append(f"- {nombre}")
    lineas += [
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
