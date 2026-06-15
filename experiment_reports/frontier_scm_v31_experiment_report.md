# SCM V3.1 Experiment Report

- Seeds: 42, 2024, 2025, 2026, 2027
- Split: train/valid/test = 0.60/0.20/0.20
- Mainline calibration: SCM-v2 best and SCM_v3 best with uncalibrated, temperature, OVR sigmoid, OVR isotonic.
- TabDDPM grid: fixed generated class-0 sample sizes crossed with epoch settings.

## Ranked Summary

| accuracy_mean | balanced_accuracy_mean | macro_f1_mean | class0_recall_mean | brier_multiclass_mean | ece_confidence_mean | augmented_size_mean |
| ------------- | ---------------------- | ------------- | ------------------ | --------------------- | ------------------- | ------------------- |
| 0.9707        | 0.9048                 | 0.8992        | 0.7500             | 0.0520                | 0.0427              | 43.0000             |
| 0.9707        | 0.9048                 | 0.8992        | 0.7500             | 0.0564                | 0.0695              | 43.0000             |
| 0.9680        | 0.9035                 | 0.8960        | 0.7500             | 0.0599                | 0.0398              | 43.0000             |
| 0.9547        | 0.8856                 | 0.8672        | 0.7000             | 0.0708                | 0.0690              | 24.0000             |
| 0.9653        | 0.8848                 | 0.8884        | 0.7000             | 0.0598                | 0.0344              | 54.8000             |
| 0.9653        | 0.8848                 | 0.8884        | 0.7000             | 0.0631                | 0.0701              | 54.8000             |
| 0.9653        | 0.8848                 | 0.8860        | 0.7000             | 0.0610                | 0.0273              | 54.8000             |
| 0.9520        | 0.8803                 | 0.8690        | 0.7000             | 0.0710                | 0.0741              | 24.0000             |
| 0.9627        | 0.8742                 | 0.8728        | 0.6500             | 0.0646                | 0.0642              | 12.0000             |
| 0.9600        | 0.8729                 | 0.8636        | 0.6500             | 0.0680                | 0.0626              | 48.0000             |
| 0.9547        | 0.8682                 | 0.8579        | 0.6500             | 0.0720                | 0.0737              | 48.0000             |
| 0.9600        | 0.8575                 | 0.8519        | 0.6000             | 0.0766                | 0.0653              | 12.0000             |
| 0.9653        | 0.8274                 | 0.8405        | 0.5000             | 0.1023                | 0.1695              | 43.0000             |
| 0.9573        | 0.7907                 | 0.8054        | 0.4000             | 0.1076                | 0.1642              | 54.8000             |

## Notes

- V3.1 is a follow-up validation experiment, not a replacement for the original V3 report.
- Calibration methods are fit on the validation split and evaluated only on the test split.
- TabDDPM grid variants are compared against the dual mainlines using the same split protocol.
