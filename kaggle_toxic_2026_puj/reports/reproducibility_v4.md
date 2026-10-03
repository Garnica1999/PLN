# Reproducibilidad de la fase v4

Fecha: 2026-09-25.

## 1. Doble ejecucion con SHA256

Script verificado: `experiments/28_hardness_v4.py`.

| Artefacto | Primera ejecucion | Segunda ejecucion | Coincide |
|---|---|---|---|
| hardness_v4.csv | `7162b9c09cee6fad` | `7162b9c09cee6fad` | si |
| hardness_v4.json | `6ea9856efe52a4fb` | `92657e2d03ee63de` | no (JSON con marca de tiempo) |
| hardness_v4.npz | `f9613389d15420d8` | `f9613389d15420d8` | si |

El JSON incluye marca de tiempo, por lo que la comparacion de integridad se hace sobre el CSV y el NPZ, que son deterministas dado que el script usa semillas fijas y no paralelismo no determinista.

## 2. Inventario de artefactos

Artefactos presentes: **41**.

| Artefacto | sha256 |
|---|---|
| experiments/baseline_v03_frozen/model_v03.joblib | `263ed5bd0fc0de7cabfbaa85e88fca6f` |
| experiments/baseline_v03_frozen/submission_v03.csv | `7d7478f271e6edfd51afff85bad3eaa0` |
| features/hardness_v4.npz | `f9613389d15420d8b21db4853315aa37` |
| features/label_consistency_v4.npz | `8c90b9dd7f68fb8f3afdc9687b5adb99` |
| features/oof_candidates_v4.npz | `c5d7992cd853936f8d243d7a19560c5b` |
| features/oof_joint_label_v4.npz | `3bbd11c62cb034f69d6309beb7eb9cdd` |
| features/oof_meta_shift_v4.npz | `106d9c0fd43c70572dfcf507b2d0129e` |
| features/oof_reweight_v4.npz | `a7d0ffbcfae9402172c109c2eabcb762` |
| features/oof_tracks_g_v4.npz | `ae276a4a533e7bb5a53ac4721863a515` |
| features/oof_weighting_v4.npz | `92f2ef7ee9563eaa97ca64ece7b0a7b3` |
| reports/baseline_v03_v4_check.json | `1c4e0bd0b7fda11b3a07fe5ae62db579` |
| reports/class_conditional_lm_v4.csv | `9e390a3c23787d2dec1d7bdb050f0af8` |
| reports/class_conditional_lm_v4.json | `8201cc60b4d0896a9f0d5d22617e42f1` |
| reports/correction_matrix_v4.csv | `7484a334521b6557d1b077b89818e853` |
| reports/correction_opportunities_v4.csv | `99ce5922ef8309887ecd0cf25a99baa9` |
| reports/discovery_v4.json | `5ce4c78da09006a7d6b47afcc3b73c25` |
| reports/distribution_shift_v4.csv | `99c9f5091b84b35052d2c47cb4f57151` |
| reports/error_taxonomy_v4.md | `9e2c36094f1caecb80c2a9439d420206` |
| reports/hardness_v4.csv | `7162b9c09cee6fadab3727d517ae88e9` |
| reports/hardness_v4.json | `92657e2d03ee63de2b4edd47aaf6f8cc` |
| reports/hardness_vs_noise_v4.md | `656f188ba7ac5e6bc16edfcd8053be5c` |
| reports/implicit_hate_diagnostic_v4.json | `49decd7355d97d47d8698031b84dca4d` |
| reports/joint_label_v4.csv | `c8e4cfae60c16068fb56a61e8948e24b` |
| reports/joint_label_v4.json | `697b7f0c3176a7b9268b6a099f230d20` |
| reports/label_consistency_v4.csv | `9ef10beac8fc9cdd1acc6cf23c5adb18` |
| reports/label_consistency_v4.json | `5ef0922554645e2754aee4030c766597` |
| reports/label_consistency_v4_ejemplos.md | `32d61fad2bd395b7bf413696149b571f` |
| reports/meta_shift_v4.csv | `d772e52822f30466262411efefd2e0e4` |
| reports/meta_shift_v4.json | `398bcdb33ce212385decbdfaaf13395f` |
| reports/research_backlog_v4.md | `b96bc88cfb9c4e8f761655b20687321d` |
| reports/reweighting_v4.csv | `201a7705d2315c2e9d659f480d51b5c0` |
| reports/reweighting_v4.json | `189027f8769cefc9d2077ba96125cf54` |
| reports/signal_coverage_v4.csv | `a5f221512ec7b5605aad96ae93a63169` |
| reports/structural_sparse_v4.csv | `1ac778026de17c96882901916ac949ae` |
| reports/structural_sparse_v4.json | `62be6ce4e6415e4be295cdb860d5c9e7` |
| reports/supervised_weighting_v4.csv | `1e0700efd79c44af63a2517b4460a8d6` |
| reports/supervised_weighting_v4.json | `36de8e207e87e94949211e7c9f236435` |
| reports/v4_candidate_comparison.md | `39a3386e52b5b32dfdf5b5ae8718c9e1` |
| reports/v4_candidates.csv | `8735f8bab9364cd35b23d8b214aa4855` |
| reports/v4_candidates.json | `63d5e85df8f8161e634bb9c69249b286` |
| reports/v4_gap_analysis.md | `7322a02898ee431f2f96513624578f35` |
