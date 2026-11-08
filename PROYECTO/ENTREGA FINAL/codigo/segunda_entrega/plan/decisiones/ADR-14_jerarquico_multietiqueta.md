# ADR-14: Clasificación jerárquica / multietiqueta / calibración (Fase U)

- **Status:** Aceptado
- **Date:** 2026-09-11
- **Context:** El plan v2 fijó para Fase U la meta macro-F1 test >= 0.32 (LogReg
  plano v1: 0.2870) explorando (A) cascada jerárquica con macro-grupos, (B)
  multietiqueta por bloques citados con umbrales optimizados en validación, (C)
  calibración del clasificador plano (sesgos por clase optimizados en val). Se
  añadieron después (D) cascada con taxonomía EuroVoc (Fase W-b) y (E) features
  de léxico EuroVoc (1.268 términos ES). Protocolo: features v1 (TF-IDF lemas
  enmascarado, hparams del joblib v1), splits canónicos intactos, selección por
  val, test una sola vez.
- **Decision:** Ninguna variante alcanzó la meta 0.32. El ClasificadorArea del
  producto se mantiene en LogReg v1 (sin calibrar). C se registra como la mejor
  variante (+3.8 % sobre v1) y queda disponible como artefacto
  (`modelos/m4_jerarquico/ganadora_C_calibracion.joblib`) si se decide adoptarla
  en el futuro.
- **Evidencia (resultados/m4_jerarquico.json + m4_eurovoc.json, test):**
  - A jerárquica: test 0.2697 (coarse 0.4033); B multietiqueta single-label:
    0.2500 (métrica nativa multietiqueta: macro 0.4792 / micro 0.7895);
    **C calibración: test 0.2980** (val 0.3792) → +3.8 % vs v1 pero < 0.32;
    D EuroVoc: 0.2625; E léxico EuroVoc: 0.2771. Ninguna supera a C.
  - Meta 0.32: NO cumplida por ninguna variante.
- **Alternatives Considered:**
  - Adoptar C igualmente (+3.8 %): descartado por el gate del plan (meta no
    cumplida) y por el principio de minimizar cambios del producto sin mejora
    sustancial; queda documentada y disponible.
  - Iterar más variantes (features nuevas): pausado (retorno decreciente; el
    techo está en las features/datos, no en la arquitectura).
- **Consequences:**
  - (+) Se descartó con evidencia la hipótesis de que jerarquía/multietiqueta
    mejoraran sustancialmente la clasificación; el baseline plano es robusto.
  - (+) La calibración por sesgos (C) funcionó mejor que las variantes
    estructurales y queda como mejora futura de bajo costo (+3.8 %).
  - (-) El análisis de errores no cambió el modelo del producto: se mantiene
    el macro-F1 0.2870 en la ruta adoptada.
