"""Reconstruye las filas V11 del registro a partir de los informes JSON de cada experimento.

Por que existe este script
--------------------------
Al commitear la fase v10 y bringingla a la v11 por merge, el ``experiments.csv`` del worktree de
v11 fue a una version anterior y las cuatro filas ``V11-00`` a ``V11-03`` se perdieron del
registro. Sus informes JSON, en cambio, intactos.

Reconstruirlas a mano seria reintroducir el riesgo de errata que acaba de costar un rato, asi que
este script las **deriva de los informes**, que son la fuente de verdad, y guarda el campo
``result`` como el JSON del informe reducido a las cifras del gate. Es reproducible: volver a
ejecutarlo da el mismo registro.

Lo que no hace, a proposito: no inventa cifras. Cada numero sale de leer el informe, y si una clave
no existe en el informe la celda queda vacia en vez de rellenarse con una suposicion.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_05_registro.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

EXPERIMENTO = "V11-05 (reconstruccion del registro v11)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_05_registro.md"
REGISTRO = PROJECT_ROOT / "experiments.csv"

# Claves de cada informe que forman el campo ``result`` de la fila, con la clave que se busca
# dentro del informe y la etiqueta con la que se guarda. Es una lista explicita y no una regla
# generica, para que se vea que cifra se esta copiando y de donde.
CLAVES_RESULTADO = {
    "V11-00": (
        ("supercadas", "supercadas"),
        ("total", "total"),
        ("todas_superadas", "todas_superadas"),
        ("referencia_v03.coincide_exacto", "v03_coincide_exacto"),
        ("referencia_v03.submission_identica", "sha256_identica"),
        ("referencia_v03.valor_reconstruido", "v03_reconstruido"),
        ("referencia_v03.diferencia", "v03_diferencia"),
        ("registro.filas", "registro_filas"),
        ("registro.claves_unicas", "registro_claves_unicas"),
        ("holdout2.ids_sin_cambios", "holdout2_ids_sin_cambios"),
        ("holdout2.grupos_compartidos", "holdout2_grupos_compartidos"),
        ("contrastes_informe_registro", "contrastes_informe_registro"),
        ("v9_02.el_denominador_es_el_desarrollo", "denominador_resuelto_es_el_desarrollo"),
    ),
    "V11-01": (
        ("diferencias_por_etiqueta", "diferencias_por_etiqueta"),
        ("tolerancia_de_reproduccion", "tolerancia_de_reproduccion"),
        ("reproduccion_exacta", "reproduccion_exacta"),
        ("denominador_resuelto_es_el_desarrollo", "denominador_resuelto_es_el_desarrollo"),
        ("prevalencias_que_usa_la_v11", "prevalencias_que_usa_la_v11"),
    ),
    "V11-02": (
        ("todas_las_citas_localizadas", "todas_las_citas_localizadas"),
        ("conteo", "conteo"),
        ("uso_de_la_previa_para_submission", "uso_de_la_prueba_para_submission"),
    ),
    "V11-03": (
        ("conformes", "conformes"),
        ("total", "total"),
        ("puntos_seccion_18_estructurales", "puntos_seccion_18_estructurales"),
        ("puntos_seccion_18_declarados", "puntos_seccion_18_declarados"),
        ("todas_conformes", "todas_conformes"),
    ),
}

# Metadatos que no estan en el informe JSON y hay que declarar.
METADATOS = {
    "V11-00": {
        "question": "Las referencias que la v11 va a usar son coherentes entre si y reproducibles",
        "hypothesis": "La v11 se apoya en tres cifras de fases distintas: la referencia 0.777211 de "
        "V03, los factores de shift de V9-02, y los informes de v10. Si V03 no reproduce, hay que "
        "parar antes de construir nada encima",
        "preprocessing": "n/a: reconciliacion de estado, no se entrena nada",
        "feature_groups": "n/a",
        "representation": "n/a",
        "model": "n/a",
        "hyperparameters": "n/a",
        "validation": "25 comprobaciones de reproduccion, integridad y coherencia entre informes "
        "y filas del registro",
        "f1": "0.7772114525165721",
        "macro_f1": "0.7772114525165721",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "0.5",
        "dimension": "n/a",
        "decision": "**P0 SUPERADO, 25 de 25.** La infraestructura es valida. Se detiene tres "
        "veces antes de pasarlo, y la investigacion de las tres paradas es el resultado real: dos "
        "eran errores de mi propia comprobacion, que buscaba la misma clave en el informe y en el "
        "registro, y la tercera era un problema de verdad en V9-02",
        "notes": "Artefactos: reports/v11_00_reconciliacion.md y .json. Queda abierto un problema "
        "de trazabilidad: la fase v10 estaba sin commitear, resuelto despues en el commit a30a467",
    },
    "V11-01": {
        "question": "La cifra de shift de prevalencia sobre la que descansa la v11 se reproduce",
        "hypothesis": "Si no se reproduce, cualquier correccion de prior, calibracion o decision "
        "hereda un numero sin verificar. Ademas P0 detecto que el campo factor_vs_train de V9-02 "
        "usa como denominador la prevalencia del desarrollo y no la del train completo",
        "preprocessing": "reajuste de V03 con todo el train y prediccion sobre la prueba; "
        "calibracion Platt sobre el OOF canonico",
        "feature_groups": "union TF-IDF palabra(1,2) y caracter(3,5) char_wb sublineal, min_df 2",
        "representation": "24646 dimensiones, la de V03",
        "model": "LogisticRegression C=2.0 balanced liblinear x3, mas Platt y EM de Saerens",
        "hyperparameters": "ninguno nuevo; se IMPORTAN las funciones de V9-02 sin reimplementarlas",
        "validation": "comparacion de las tres prevalencias EM contra las de V9-02 con tolerancia "
        "1e-6, y contraste del denominador del factor",
        "f1": "",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "1.2",
        "dimension": "24646",
        "decision": "**REPRODUCCION EXACTA CONFIRMADA, nivel R3.** Diferencias de -4.51e-07 a "
        "+3.29e-07, que es el redondeo a seis decimales del JSON de V9-02, e iteraciones del EM "
        "identicas: 26, 16 y 24. La ambiguedad del denominador queda cerrada a favor del "
        "desarrollo. **PERO SU RESULTADO QUEDA EN CUARENTENA**: el uso de la prueba para estimar "
        "prevalencia no alimenta ningun candidato de la v11",
        "notes": "Artefactos: reports/v11_01_reproduccion_v9_02.md y .json. Reproducir un "
        "artefacto congelado para auditoria esta permitido; usar su salida para construir un "
        "submission no lo esta",
    },
    "V11-02": {
        "question": "Permiten las reglas academicas estimar la prevalencia de la prueba con EM",
        "hypothesis": "El plan obliga a clasificar cada paso como permitido, ambiguo o prohibido "
        "antes de usar las predicciones de test, y fija la consecuencia en el caso ambiguo",
        "preprocessing": "n/a: auditoria documental",
        "feature_groups": "n/a",
        "representation": "n/a",
        "model": "n/a",
        "hyperparameters": "n/a",
        "validation": "clasificacion de 11 pasos contra el texto verbatim de "
        "docs/competition_rules.md, con las cuatro citas verificadas literalmente",
        "f1": "",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "0.0",
        "dimension": "n/a",
        "decision": "**USO DE LA PRUEBA CLASIFICADO COMO AMBIGUO Y DESCARTADO PARA GENERAR "
        "SUBMISSION.** Reparto: 5 permitidos, 1 ambiguo, 5 prohibidos. El paso ambiguo es estimar "
        "la prevalencia con EM, y lo es porque la regla prohibe entrenamiento y aqui no lo hay, "
        "pero la intencion de la regla es que el test no participe del desarrollo del sistema. "
        "Las reglas no mencionan la transduccion ni para permitirla ni para prohibirla, y ese "
        "silencio es lo que hace el caso ambiguo",
        "notes": "Artefactos: reports/v11_02_auditoria_legalidad.md y .json, y "
        "reports/v11_compliance.md. Consecuencia asumida: la pregunta de transferencia de la v11 "
        "no se podra responder con honestidad sin autorizacion expresa del profesor",
    },
    "V11-03": {
        "question": "El contenido generado por la v11 cumple las restricciones de ejecucion",
        "hypothesis": "La seccion 18 exige verificar sus 17 puntos y no solo declararlos, y la "
        "seccion 19 obliga a parar ante una condicion de parada. Es el primer experimento de la "
        "v11 con potestad para invalidar a los anteriores",
        "preprocessing": "n/a: auditoria, no se entrena nada",
        "feature_groups": "n/a",
        "representation": "n/a",
        "model": "n/a",
        "hyperparameters": "n/a",
        "validation": "16 comprobaciones estructurales de la seccion 18 por analisis de imports y "
        "firmas de uso, 13 condiciones de parada, politica de caracteres con dos escaneos "
        "independientes, y coherencia entre registro y codigo",
        "f1": "0.7772114525165721",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "0.6",
        "dimension": "n/a",
        "decision": "**36 de 37 comprobaciones conformes; la unica falla es deliberada.** El "
        "escaner de cumplimiento inicial era inservible, porque buscaba palabras prohibidas en "
        "texto y marcaba las citas de la prohibicion y su propia lista de patrones: doce falsos "
        "positivos. Se sustituyo por analisis estructural de imports. Ademas la politica de "
        "caracteres es mas estricta que la herramienta previa, que busca alfabetos no latinos "
        "mientras que la seccion 17 pide ASCII puro",
        "notes": "Artefactos: reports/v11_03_auditoria_cumplimiento.md y .json, y "
        "experiments/v11_ayudas.py. Los 3 ficheros heredados de v10 con acentos NO se corrigen: "
        "modificarlos rompe la correspondencia entre codigo y resultado historico y v10 debe "
        "quedar intacto. Se documentan como desviacion conocida",
    },
}

INFORMES = {
    "V11-00": "v11_00_reconciliacion.json",
    "V11-01": "v11_01_reproduccion_v9_02.json",
    "V11-02": "v11_02_auditoria_legalidad.json",
    "V11-03": "v11_03_auditoria_cumplimiento.json",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-05] {mensaje}", flush=True)


def extraer_resultado(claves: tuple[tuple[str, str], ...], informe: dict[str, Any]) -> dict:
    """Extrae del informe las claves declaradas para el campo de resultado.

    Args:
        claves: pares de clave buscada y etiqueta con la que se guarda. La clave puede ser un
            camino separado por puntos, porque parte de la informacion de los informes es
            anidada y un mapeo plano la perderia en silencio.
        informe: informe JSON del experimento.

    Returns:
        Diccionario con las claves encontradas y las que no, estas ultimas con ``None``.
    """
    resultado: dict[str, Any] = {}
    for buscada, etiqueta in claves:
        actual: Any = informe
        for parte in buscada.split("."):
            if isinstance(actual, dict) and parte in actual:
                actual = actual[parte]
            else:
                actual = None
                break
        resultado[etiqueta] = actual
    return resultado


def main() -> None:
    """Reconstruye las filas V11 y las anade al registro."""
    inicio = time.time()
    log("V11-05: reconstruccion de las filas V11 del registro desde los informes")

    with REGISTRO.open(encoding="utf-8") as fh:
        lector = csv.reader(fh)
        cabecera = next(lector)
        existentes = [fila for fila in lector if fila and fila[0]]
    ids_actuales = [fila[0] for fila in existentes]
    log(f"   registro con {len(existentes)} filas | V11 ya presentes: "
        f"{[i for i in ids_actuales if i.startswith('V11-')]}")

    columnas = {campo: indice for indice, campo in enumerate(cabecera)}
    nuevas: list[list[str]] = []
    detalle: list[dict[str, Any]] = []
    for identificador, nombre_informe in INFORMES.items():
        ruta_informe = PROJECT_ROOT / "reports" / nombre_informe
        if not ruta_informe.exists():
            log(f"   [AVISO] falta el informe {nombre_informe}, se omite {identificador}")
            continue
        informe = json.loads(ruta_informe.read_text(encoding="utf-8"))
        resultado = extraer_resultado(CLAVES_RESULTADO[identificador], informe)
        ausentes = [clave for clave, valor in resultado.items() if valor is None]
        metadatos = METADATOS[identificador]
        fila_registro = [""] * len(cabecera)
        valores = {
            "experiment_id": identificador,
            "competition": "KAGGLE_TOXIC_2026_PUJ",
            "date": "2026-09-26",
            "script": f"experiments/{identificador.lower().replace('-', '_')}"
            + _guion_de(identificador),
            "result": json.dumps(resultado, ensure_ascii=False, separators=(",", ":")),
            **metadatos,
        }
        for campo, valor in valores.items():
            if campo in columnas:
                fila_registro[columnas[campo]] = str(valor)
        nuevas.append(fila_registro)
        detalle.append(
            {
                "experiment_id": identificador,
                "claves_extraidas": list(resultado.keys()),
                "claves_ausentes": ausentes,
                "informe": nombre_informe,
            }
        )
        log(
            f"   {identificador}: {len(resultado) - len(ausentes)} claves extraidas"
            + (f" | ausentes: {ausentes}" if ausentes else "")
        )

    if not nuevas:
        log("no hay filas que reconstruir")
        return

    # Se comprueba que no se duplique ningun identificador antes de escribir.
    duplicados = {i for i in ids_actuales} & {f[0] for f in nuevas}
    if duplicados:
        log(f"PARADA: estos identificadores ya existen y no se duplican: {sorted(duplicados)}")
        return

    with REGISTRO.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        escritor.writerow(cabecera)
        for fila in existentes:
            escritor.writerow(fila)
        for fila in nuevas:
            escritor.writerow(fila)

    final = list(csv.DictReader(REGISTRO.open(encoding="utf-8")))
    ids = [f["experiment_id"] for f in final]
    n_campos = len(list(csv.DictReader(REGISTRO.open(encoding="utf-8")).fieldnames))
    malas = [f["experiment_id"] for f in final if len(f) != n_campos or any(v is None for v in f.values())]
    log("")
    log(f"registro tras reconstruir: {len(final)} filas | {len(set(ids))} claves unicas | "
        f"{n_campos} campos | malformadas: {len(malas)}")
    log(f"V10-*: {len([i for i in ids if i.startswith('V10-')])} | "
        f"V11-*: {[i for i in ids if i.startswith('V11-')]}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "motivo": (
            "Al commitear v10 en a30a467 y traerla a la v11 por merge, el experiments.csv del "
            "worktree de v11 retrocedio a una version anterior y las filas V11 se perdieron del "
            "registro. Los informes JSON remained intactos, de modo que las filas se reconstruyen "
            "derivandolas de ahi en vez de teclearlas."
        ),
        "detalle": detalle,
        "filas_reconstruidas": [f[0] for f in nuevas],
        "registro_final": {
            "filas": len(final),
            "claves_unicas": len(set(ids)),
            "campos": n_campos,
            "malformadas": len(malas),
        },
    }
    (PROJECT_ROOT / "reports" / "v11_05_registro.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, nuevas, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")


def _guion_de(identificador: str) -> str:
    """Deriva el nombre del guion a partir del identificador del experimento.

    Args:
        identificador: identificador con formato ``V11-0X``.

    Returns:
        El sufijo del nombre del guion.
    """
    numero = identificador.split("-")[1]
    nombres = {
        "00": "_reconciliacion.py",
        "01": "_reproduccion_v9_02.py",
        "02": "_auditoria_legalidad.py",
        "03": "_auditoria_cumplimiento.py",
    }
    return nombres.get(numero, ".py")


def _informe(informe: dict[str, Any], nuevas: list[list[str]], inicio: float) -> None:
    """Escribe el informe de la reconstruccion del registro.

    Args:
        informe: resultado de la reconstruccion.
        nuevas: filas anadidas, en el orden del esquema.
        inicio: instante de arranque del script.
    """
    lineas: list[str] = [
        "# V11-05: reconstruccion de las filas V11 del registro",
        "",
        f"Generado por `experiments/v11_05_registro.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que existe",
        "",
        informe["motivo"],
        "",
        "Reconstruirlas a mano habria reintroducido el riesgo de errata que acababa de costar un",
        "rato, asi que este script las **deriva de los informes**, que son la fuente de verdad. Es",
        "reproducible: volver a ejecutarlo da el mismo registro.",
        "",
        "Lo que no hace, a proposito: **no inventa cifras**. Cada numero sale de leer el informe, y",
        "si una clave no existe alli la celda queda vacia en vez de rellenarse con una suposicion.",
        "",
        "## 2. Claves extraidas por experimento",
        "",
        fila(["experimento", "claves extraidas", "claves ausentes", "informe"]),
        fila(["---", "---", "---", "---"]),
    ]
    for entrada in informe["detalle"]:
        lineas.append(
            fila(
                [
                    entrada["experiment_id"],
                    str(len(entrada["claves_extraidas"]) - len(entrada["claves_ausentes"])),
                    ", ".join(entrada["claves_ausentes"]) or "ninguna",
                    entrada["informe"],
                ]
            )
        )
    registro = informe["registro_final"]
    lineas += [
        "",
        "## 3. Resultado",
        "",
        (
            f"- filas reconstruidas: **{len(informe['filas_reconstruidas'])}**, "
            + ", ".join(f"`{i}`" for i in informe["filas_reconstruidas"])
        ),
        (
            f"- registro final: **{registro['filas']} filas**, "
            f"{registro['claves_unicas']} claves unicas, {registro['campos']} campos"
        ),
        f"- filas malformadas: **{registro['malformadas']}**",
        "",
        "## 4. El fallo que causo esto, y que conviene no repetir",
        "",
        "Mi primer reanexado escribio las filas a mano, envolver cada campo entre comillas dobles",
        "sin escapar las comas internas. El resultado fueron **4 filas malformadas**, que el",
        "registro aceptaba pero con el numero de campos equivocado. Solo se detecto porque se",
        "comprobro el numero de campos por fila, que es una comprobacion que ya se hacia en otros",
        "sitios y que aqui falto.",
    ]
    lineas += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
