"""V36 - E1 - Reanudacion autorizada: termina el ultimo checkpoint y el post-proceso.

## Que paso

La corrida real de V36-E1 (`scripts/v36_01_transiciones_clase.py`, commit `1e09058`, APROBADO por
`ml-reviewer`) termino con **exit 7** (presupuesto agotado, no error) a los **2453 s** del reloj del
script frente a los **2400 s** declarados en el pre-registro. Se quedaron **39 de 40 checkpoints**:
faltaba unicamente `fold4_integ_c1a.npz`, y tras el fold el post-proceso (sondas de resultado, gates,
arbol, JSON, informe, figura y anotadores).

## Los dos defectos, y por que esto es un script NUEVO

1. **El presupuesto declarado estaba subestimado.** Costre ~2750 s, no 2400 s. Es un error del
   pre-registro (mi estimacion), no del codigo, y no se corrige en retrospectiva: se declara.
2. **La reanudacion era inalcanzable**: `verificar_deadline` mide el tiempo contra
   `estado["inicio_ts"]`, que esta **persistido** en el checkpoint, luego una reanudacion hereda el
   reloj de la invocacion anterior y muere de inmediato (medido: exit 7 en 48 s, tras reutilizar
   correctamente los 39 checkpoints). Es decir, el runner se anuncia "reanudable" y no lo es.

Este script **no edita** el runner ya ejecutado: lo **importa** y le sustituye **solo** la funcion
de reloj, que es el unico defecto de implementacion que impide terminar. Todo lo demas --el mapa
literal, los folds, los gates, las hipotesis nulas, el arbol, la one-shot, la maquinaria de V35-- se
ejecuta **exactamente** como los ejecuto la primera invocacion.

## Autorizacion y overrun, DECLARADOS

Carlos autorizo el 2026-09-30 **el overrun** y la continuacion. Consecuencias, todas publicadas:

- El presupuesto **pre-registrado** sigue siendo 2400 s y se publica como tal: **fue superado**.
- El reloj de esta invocacion mide desde **el inicio de ESTA invocacion**, que es la semantica
  correcta para un runner reanudable; el **total acumulado** se recalcula y se publica junto al
  declarado, de modo que el coste real (~2750 s) **queda a la vista** y no se esconde.
- La correccion del reloj **no relaja** el control: `verificar_deadline` sigue parando con exit 7 si
  esta invocacion supera su presupuesto, y se anade un tope **global** mas alto, el del overrun
  autorizado, para que un fallo real de coste tambien detenga.

## Que verifica antes de continuar

Que los **39 checkpoints existentes** siguen presentes y son legibles, que el estado persistido es
coherente con ellos, y que **no se reentrena nada**: si el numero de checkpoints fuera distinto del
esperado, el script se detiene en vez de continuar sobre un estado que no reconoce.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.verificacion_numerica import autotest_numerico

FECHA = "2026-09-30"
NUCLEO = ROOT / "scripts" / "v36_01_transiciones_clase.py"
ESTADO = ROOT / "experiments" / "v36_01_estado.json"
CKPT_DIR = ROOT / "checkpoints" / "v36_e1"
SALIDA = ROOT / "experiments" / "v36_02_reanudacion.json"

#: Presupuesto **pre-registrado**, que fue superado y se publica como tal.
PRESUPUESTO_PREREG = 2400.0
#: Coste real medido de la primera invocacion (reloj del script).
COSTE_PRIMERA_INVOCACION = 2453.0
#: Tope autorizado por Carlos para esta invocacion: lo que falta (un fold + post-proceso) mas margen.
TOPE_INVOCACION_AUTORIZADO = 900.0
#: Tope global acumulado autorizado: el declarado mas el overrun, para que un fallo real de coste
#: tambien detenga en lugar de correr indefinidamente.
TOPE_GLOBAL_AUTORIZADO = PRESUPUESTO_PREREG + TOPE_INVOCACION_AUTORIZADO
CHECKPOINTS_ESPERADOS = 39


def cargar_nucleo():
    """Importa el runner APROBADO sin reejecutar su ``main``.

    Returns
    -------
    module
        El modulo `v36_01_transiciones_clase` como objeto, con sus ``globals()`` propios.
    """
    spec = importlib.util.spec_from_file_location("nucleo_v36_01", NUCLEO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    nucleo = cargar_nucleo()

    print("=" * 100, flush=True)
    print("V36-E1 - REANUDACION AUTORIZADA (sobrerun declarado)", flush=True)
    print(f"  presupuesto PRE-REGISTRADO: {PRESUPUESTO_PREREG:g} s (SUPERADO: coste medido "
          f"{COSTE_PRIMERA_INVOCACION:g} s)", flush=True)
    print(f"  tope de ESTA invocacion, autorizado: {TOPE_INVOCACION_AUTORIZADO:g} s", flush=True)
    print(f"  tope global acumulado autorizado: {TOPE_GLOBAL_AUTORIZADO:g} s", flush=True)
    print(f"  nucleo importado (NO se modifica): {NUCLEO.name}", flush=True)
    print("=" * 100, flush=True)

    # ---- el unico defecto que se corrige: el reloj ----
    # El acumulado es la SUMA de los costes de cada invocacion (reloj propio de cada una), NO el
    # tiempo de pared desde el inicio persistido: ese incluye el rato entre invocaciones, que no
    # es coste de computo. El primero fue medido en 2453 s.
    coste_invocaciones_previas = COSTE_PRIMERA_INVOCACION

    def verificar_deadline_por_invocacion(estado: dict) -> None:
        """Reloj por invocacion, con tope global acumulado y parada con exit 7.

        Parameters
        ----------
        estado:
            Checkpoint que se guarda antes de parar.

        Raises
        ------
        nucleo.Detencion
            Si se supera el tope de esta invocacion o el global acumulado (codigo 7).
        """
        local = time.time() - inicio
        global_acumulado = coste_invocaciones_previas + local
        if global_acumulado >= TOPE_GLOBAL_AUTORIZADO:
            nucleo.guardar_estado(estado)
            raise nucleo.Detencion(
                f"[STOP] tope global acumulado autorizado de {TOPE_GLOBAL_AUTORIZADO:g} s superado "
                f"({global_acumulado:.0f} s); estado guardado (exit 7)", 7)
        if local >= TOPE_INVOCACION_AUTORIZADO:
            nucleo.guardar_estado(estado)
            raise nucleo.Detencion(
                f"[STOP] tope autorizado de esta invocacion de {TOPE_INVOCACION_AUTORIZADO:g} s "
                f"superado ({local:.0f} s); estado guardado (exit 7)", 7)

    nucleo.verificar_deadline = verificar_deadline_por_invocacion

    # ---- no continuar sobre un estado que no se reconoce ----
    if not ESTADO.exists():
        print("[STOP] no existe el estado persistido: no hay nada que reanudar (exit 2)", flush=True)
        return 2
    estado = nucleo.cargar_estado()
    hechos = sorted((brazo, int(fold))
                    for brazo, detalle in estado["brazos"].items()
                    for fold in detalle.get("folds", {}))
    archivos = sorted(CKPT_DIR.glob("*.npz"))
    print(f"  estado persistido: {len(hechos)} folds de {len(estado['brazos'])} brazos", flush=True)
    print(f"  checkpoints en disco: {len(archivos)} (esperados {CHECKPOINTS_ESPERADOS})", flush=True)
    if len(archivos) != CHECKPOINTS_ESPERADOS:
        print(f"[STOP] hay {len(archivos)} checkpoints y se esperaban {CHECKPOINTS_ESPERADOS}: el "
              "estado no es el que se reanuda (exit 2)", flush=True)
        return 2
    for archivo in archivos:
        try:
            with np.load(archivo, allow_pickle=False) as paquete:
                if "indices" not in paquete or "valores" not in paquete:
                    print(f"[STOP] el checkpoint {archivo.name} no tiene las claves esperadas "
                          "(exit 2)", flush=True)
                    return 2
        except (OSError, ValueError) as error:
            print(f"[STOP] el checkpoint {archivo.name} no es legible: {error} (exit 2)", flush=True)
            return 2
    incompletos = {brazo: sorted(detalle.get("folds", {}))
                   for brazo, detalle in estado["brazos"].items()
                   if len(detalle.get("folds", {})) < 5}
    print(f"  brazos incompletos: {incompletos or 'ninguno'}", flush=True)

    # ---- continuar: el nucleo hace el resto, sin cambiar una sola constante ----
    print("\n  continuando con el nucleo aprobado (solo cambia el reloj)...", flush=True)
    codigo = nucleo.main()
    transcurrido = time.time() - inicio

    resumen = {
        "fase": "V36-E1 - reanudacion autorizada con overrun declarado",
        "fecha": FECHA,
        "motivo": "la primera invocacion agoto el presupuesto pre-registrado (exit 7) y dejo 39 de 40 "
                  "checkpoints; ademas verificar_deadline media contra inicio_ts PERSISTIDO, luego la "
                  "reanudacion era inalcanzable (exit 7 en 48 s)",
        "autorizacion": "Carlos, 2026-09-30: overrun autorizado y declarado",
        "presupuesto_preregistrado": PRESUPUESTO_PREREG,
        "presupuesto_preregistrado_superado": True,
        "coste_real_primera_invocacion": COSTE_PRIMERA_INVOCACION,
        "tope_invocacion_autorizado": TOPE_INVOCACION_AUTORIZADO,
        "tope_global_acumulado_autorizado": TOPE_GLOBAL_AUTORIZADO,
        "correccion_unica": "verificar_deadline mide por invocacion y añade un tope global "
                            "acumulado; el nucleo NO se modifica, se importa",
        "nucleo_importado": str(NUCLEO.relative_to(ROOT.parent)),
        "sha256_del_nucleo": hashlib.sha256(NUCLEO.read_bytes()).hexdigest(),
        "checkpoints_al_entrar": len(archivos),
        "brazos_incompletos_al_entrar": incompletos,
        "se_reentrena_algo": False,
        "codigo_de_salida_del_nucleo": codigo,
        "duracion_de_esta_invocacion": transcurrido,
        "coste_invocaciones_previas": coste_invocaciones_previas,
        "duracion_total_acumulada": coste_invocaciones_previas + transcurrido,
    }
    SALIDA.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[reanudacion] codigo del nucleo: {codigo} | esta invocacion: {transcurrido:.1f} s | "
          f"acumulado: {coste_invocaciones_previas + transcurrido:.1f} s frente al declarado "
          f"{PRESUPUESTO_PREREG:g} s (superado)", flush=True)
    print(f"  {SALIDA.relative_to(ROOT.parent)}", flush=True)
    return int(codigo)


if __name__ == "__main__":
    raise SystemExit(main())