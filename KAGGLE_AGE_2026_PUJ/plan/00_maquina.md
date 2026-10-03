# Fase 00 — Perfil del entorno de ejecución

- **Fecha**: 2026-09-25
- **Intérprete canónico**: `/home/carlos/venv/bin/python` (Python 3.14.7)
- **CPU**: AMD Ryzen 9 5950X 16-Core Processor — 16 núcleos físicos, 32 hilos lógicos
- **RAM**: 31.2 GB
- **GPU**: AMD Radeon RX 9070 XT (dedicada, 15.9 GB, torch_cuda=True)
- **Sistema**: Linux 7.2.6-1-cachyos

## Decisiones de recursos derivadas del perfil
- `n_jobs=-1` y 32 hilos para BLAS/torch (CPUs con 32 hilos).
- **La fase de lexicones es CPU pura**: los agregados léxicos y el combinador lineal no requieren GPU (coherente con D-007).
- No se instalan paquetes: el venv ya contiene nltk 3.9.4, pandas 2.3.3, scikit-learn 1.8.0 y scipy 1.17.1, que es todo lo necesario.

_Generado por `scripts/fase00_entorno.py`; no asumir hardware sin reejecutarlo._
