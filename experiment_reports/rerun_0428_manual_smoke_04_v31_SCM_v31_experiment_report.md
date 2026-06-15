# SCM V3.1 Experiment Report

- Seeds: 42
- Split: train/valid/test = 0.60/0.20/0.20
- Mainline calibration: SCM-v2 best and SCM_v3 best with uncalibrated, temperature, OVR sigmoid, OVR isotonic.
- TabDDPM grid: fixed generated class-0 sample sizes crossed with epoch settings.

## Ranked Summary

| family               | variant                                          | model_name                         | accuracy_mean | balanced_accuracy_mean | macro_f1_mean | class0_recall_mean | brier_multiclass_mean | ece_confidence_mean | augmented_size_mean |
| -------------------- | ------------------------------------------------ | ---------------------------------- | ------------- | ---------------------- | ------------- | ------------------ | --------------------- | ------------------- | ------------------- |
| mainline_calibration | xgb_scm_v2_best__ovr_sigmoid                     | xgb_scm_v2_best                    | 0.9733        | 0.9792                 | 0.9609        | 1.0000             | 0.1138                | 0.1853              | 23.0000             |
| mainline_calibration | xgb_scm_v2_best__ovr_isotonic                    | xgb_scm_v2_best                    | 0.9600        | 0.9713                 | 0.9374        | 1.0000             | 0.0679                | 0.0182              | 23.0000             |
| mainline_calibration | xgb_scm_v2_best__temperature                     | xgb_scm_v2_best                    | 0.9600        | 0.9713                 | 0.9374        | 1.0000             | 0.0710                | 0.0126              | 23.0000             |
| mainline_calibration | xgb_scm_v2_best__uncalibrated                    | xgb_scm_v2_best                    | 0.9600        | 0.9713                 | 0.9374        | 1.0000             | 0.0766                | 0.0499              | 23.0000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__temperature  | scm_v3_res_norule_single_flat_nocf | 0.9467        | 0.9634                 | 0.9160        | 1.0000             | 0.0942                | 0.0140              | 31.0000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__uncalibrated | scm_v3_res_norule_single_flat_nocf | 0.9467        | 0.9634                 | 0.9160        | 1.0000             | 0.0966                | 0.0398              | 31.0000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__ovr_isotonic | scm_v3_res_norule_single_flat_nocf | 0.9333        | 0.9158                 | 0.8982        | 0.8571             | 0.0990                | 0.0354              | 31.0000             |
| mainline_calibration | scm_v3_res_norule_single_flat_nocf__ovr_sigmoid  | scm_v3_res_norule_single_flat_nocf | 0.9333        | 0.8761                 | 0.8761        | 0.7143             | 0.1301                | 0.1751              | 31.0000             |

## Notes

- V3.1 is a follow-up validation experiment, not a replacement for the original V3 report.
- Calibration methods are fit on the validation split and evaluated only on the test split.
- TabDDPM grid variants are compared against the dual mainlines using the same split protocol.
