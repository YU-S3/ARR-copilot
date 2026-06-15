# SCM V3.1 Experiment Report

- Seeds: 42, 2024, 2025, 2026, 2027
- Split: train/valid/test = 0.60/0.20/0.20
- Mainline calibration: SCM-v2 best and SCM_v3 best with uncalibrated, temperature, OVR sigmoid, OVR isotonic.
- TabDDPM grid: fixed generated class-0 sample sizes crossed with epoch settings.

## Ranked Summary

| family               | variant                                          | model_name                         | accuracy_mean | balanced_accuracy_mean | macro_f1_mean | class0_recall_mean | brier_multiclass_mean | ece_confidence_mean | augmented_size_mean |
| -------------------- | ------------------------------------------------ | ---------------------------------- | ------------- | ---------------------- | ------------- | ------------------ | --------------------- | ------------------- | ------------------- |
| mainline_calibration | xgb_scm_v2_best__ovr_isotonic                    | xgb_scm_v2_best                    | 0.9733        | 0.9723                 | 0.9529        | 0.9714             | 0.0460                | 0.0293              | 47.2000             |
| mainline_calibration | xgb_scm_v2_best__temperature                     | xgb_scm_v2_best                    | 0.9627        | 0.9650                 | 0.9334        | 0.9714             | 0.0599                | 0.0242              | 47.2000             |
| mainline_calibration | xgb_scm_v2_best__uncalibrated                    | xgb_scm_v2_best                    | 0.9627        | 0.9650                 | 0.9334        | 0.9714             | 0.0626                | 0.0621              | 47.2000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__temperature  | scm_v3_res_norule_single_flat_nocf | 0.9600        | 0.9643                 | 0.9316        | 0.9714             | 0.0724                | 0.0345              | 53.6000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__uncalibrated | scm_v3_res_norule_single_flat_nocf | 0.9600        | 0.9643                 | 0.9316        | 0.9714             | 0.0736                | 0.0621              | 53.6000             |
| mainline_calibration | xgb_scm_v2_best__ovr_sigmoid                     | xgb_scm_v2_best                    | 0.9707        | 0.9628                 | 0.9463        | 0.9429             | 0.1024                | 0.1949              | 47.2000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__ovr_isotonic | scm_v3_res_norule_single_flat_nocf | 0.9707        | 0.9628                 | 0.9497        | 0.9429             | 0.0544                | 0.0340              | 53.6000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__ovr_sigmoid  | scm_v3_res_norule_single_flat_nocf | 0.9600        | 0.9405                 | 0.9254        | 0.8857             | 0.1107                | 0.1901              | 53.6000             |

## Notes

- V3.1 is a follow-up validation experiment, not a replacement for the original V3 report.
- Calibration methods are fit on the validation split and evaluated only on the test split.
- TabDDPM grid variants are compared against the dual mainlines using the same split protocol.
