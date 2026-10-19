# Porte V47/V51/V53 a `repro/cadena/neuronales.py` (2026-10-18)

## Que se anadio
- `SEChannelBlock` y `TabularDenseNet` (scripts/v53_01:65-142); `WeightedCramerLoss` (scripts/v47_01:96-120); `ConsistentSmoothedCramerLoss` (scripts/v51_01:53-74; v53_01:145-163).
- `CONFIGS` v47/v51/v53 + ramas en `_construir_modelo`/`_construir_loss`; `cfg["clip"]` en `entrenar_componente` y `cfg["bn_batch"]` en `_entrenar_una`. V38-V43 sin cambios (sin clip/bn_batch -> camino previo).

## Recetas exactas (`entrenar_componente("v47"|"v51"|"v53", ...)`)
- V47: SE-TabularResNet, WeightedCramer (gamma [1.2,1.5,1.3,1.1,1.0], lambda 0.50, smoothing 0.10), recorte Z [-6,+6], seeds 42+fold*100, 16 epocas, batch 256, AdamW lr 1e-3 wd 1e-4, Cosine(T_max=16, eta_min=1e-4), SWA desde ep. 6, update_bn con loader 512 sin barajar; test = media/5.
- V51: SE-TabularResNet, ConsistentSmoothedCramer (CDF objetivo (1-eps)*I(y<=k) + eps*(k+1)/K), sin recorte, mismo resto; test = media/5.
- V53: TabularDenseNet (init 192, growth 128, drop 0.20; 3 bloques densos+SE y transicion a 256), ConsistentSmoothedCramer, recorte Z [-6,+6], mismo resto; test = media/5.

## Verificacion (smoke fold 0, venv local ROCm, RX 9070 XT)
- Parte A (bit a bit vs scripts originales): state_dict identicos (SETabularResNet v47, TabularDenseNet v53) y perdidas identicas en float32.
- Parte B (fold 0 via `entrenar_componente`, folds 1-4 neutralizados; OOF vs `features/v4X_oof_*.npy`, n=33.868 filas):

| comp | max|d| | media|d| | acuerdo argmax |
|------|----------|-----------|----------------|
| v47 | 1.04e-06 | 2.22e-08 | 100.0000% |
| v51 | 0.0 | 0.0 | 100.0000% |
| v53 | 0.0 | 0.0 | 100.0000% |

- Deriva de v47 (~1e-6) es la esperada: el original infiere OOF en lotes de 512 y el porte en una pasada (orden de reduccion FP); v51/v53 originales ya inferian en una pasada -> coincidencia bit a bit.
- Tiempos fold 0 en GPU: v47 30.6 s, v51 28.8 s, v53 37.1 s. Script de smoke en `/tmp/opencode/smoke_v47_v51_v53.py` (fuera del repo).
