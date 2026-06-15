# SCM_v3 Generalization Diagnostics

- Seeds: 42, 2024, 2025, 2026, 2027
- Split: train/valid/test = 0.60/0.20/0.20
- Compared models: xgb_reference_adasyn, xgb_scm_v2_best, scm_v3_res_norule_single_flat_nocf, xgb_tabddpm_proto

## Test Mean Metrics

| model_name                         | accuracy | balanced_accuracy | macro_f1 | class0_recall | brier_multiclass | ece_confidence |
| ---------------------------------- | -------- | ----------------- | -------- | ------------- | ---------------- | -------------- |
| xgb_scm_v2_best                    | 0.9707   | 0.9048            | 0.8992   | 0.7500        | 0.0564           | 0.0695         |
| scm_v3_res_norule_single_flat_nocf | 0.9653   | 0.8848            | 0.8884   | 0.7000        | 0.0631           | 0.0701         |
| xgb_tabddpm_proto                  | 0.9520   | 0.8690            | 0.8536   | 0.6500        | 0.0646           | 0.0795         |
| xgb_reference_adasyn               | 0.9573   | 0.8542            | 0.8442   | 0.6000        | 0.0648           | 0.0689         |

## Gap Summary

| index | model_name                         | macro_f1_train_test_gap_mean | macro_f1_train_test_gap_std | balanced_accuracy_train_test_gap_mean | balanced_accuracy_train_test_gap_std | macro_f1_train_valid_gap_mean | macro_f1_train_valid_gap_std | balanced_accuracy_train_valid_gap_mean | balanced_accuracy_train_valid_gap_std | brier_multiclass_test_mean | brier_multiclass_test_std | ece_confidence_test_mean | ece_confidence_test_std |
| ----- | ---------------------------------- | ---------------------------- | --------------------------- | ------------------------------------- | ------------------------------------ | ----------------------------- | ---------------------------- | -------------------------------------- | ------------------------------------- | -------------------------- | ------------------------- | ------------------------ | ----------------------- |
| 0     | scm_v3_res_norule_single_flat_nocf | 0.1116                       | 0.0879                      | 0.1152                                | 0.0918                               | 0.1240                        | 0.0290                       | 0.1147                                 | 0.0305                                | 0.0631                     | 0.0307                    | 0.0701                   | 0.0202                  |
| 1     | xgb_reference_adasyn               | 0.1558                       | 0.0934                      | 0.1458                                | 0.1267                               | 0.1525                        | 0.0655                       | 0.1477                                 | 0.0965                                | 0.0648                     | 0.0373                    | 0.0689                   | 0.0049                  |
| 2     | xgb_scm_v2_best                    | 0.1008                       | 0.1056                      | 0.0952                                | 0.1065                               | 0.1317                        | 0.0857                       | 0.1438                                 | 0.0838                                | 0.0564                     | 0.0338                    | 0.0695                   | 0.0216                  |
| 3     | xgb_tabddpm_proto                  | 0.1464                       | 0.0815                      | 0.1310                                | 0.1114                               | 0.1498                        | 0.0879                       | 0.1370                                 | 0.1071                                | 0.0646                     | 0.0188                    | 0.0795                   | 0.0114                  |

## Bootstrap Test CI

| model_name                         | metric                   | mean   | ci_low | ci_high |
| ---------------------------------- | ------------------------ | ------ | ------ | ------- |
| scm_v3_res_norule_single_flat_nocf | accuracy                 | 0.9654 | 0.9440 | 0.9840  |
| scm_v3_res_norule_single_flat_nocf | balanced_accuracy        | 0.8866 | 0.8154 | 0.9504  |
| scm_v3_res_norule_single_flat_nocf | macro_f1                 | 0.8900 | 0.8282 | 0.9421  |
| scm_v3_res_norule_single_flat_nocf | class0_recall            | 0.7062 | 0.5000 | 0.8847  |
| scm_v3_res_norule_single_flat_nocf | class0_average_precision | 0.7899 | 0.6147 | 0.9250  |
| scm_v3_res_norule_single_flat_nocf | brier_multiclass         | 0.0630 | 0.0390 | 0.0895  |
| scm_v3_res_norule_single_flat_nocf | ece_confidence           | 0.0588 | 0.0462 | 0.0726  |
| xgb_reference_adasyn               | accuracy                 | 0.9570 | 0.9333 | 0.9760  |
| xgb_reference_adasyn               | balanced_accuracy        | 0.8540 | 0.7757 | 0.9268  |
| xgb_reference_adasyn               | macro_f1                 | 0.8549 | 0.7851 | 0.9174  |
| xgb_reference_adasyn               | class0_recall            | 0.5996 | 0.3684 | 0.8182  |
| xgb_reference_adasyn               | class0_average_precision | 0.7480 | 0.5559 | 0.8931  |
| xgb_reference_adasyn               | brier_multiclass         | 0.0653 | 0.0423 | 0.0931  |
| xgb_reference_adasyn               | ece_confidence           | 0.0591 | 0.0480 | 0.0721  |
| xgb_scm_v2_best                    | accuracy                 | 0.9707 | 0.9520 | 0.9867  |
| xgb_scm_v2_best                    | balanced_accuracy        | 0.9037 | 0.8393 | 0.9617  |
| xgb_scm_v2_best                    | macro_f1                 | 0.8996 | 0.8394 | 0.9511  |
| xgb_scm_v2_best                    | class0_recall            | 0.7462 | 0.5554 | 0.9231  |
| xgb_scm_v2_best                    | class0_average_precision | 0.7358 | 0.5199 | 0.9167  |
| xgb_scm_v2_best                    | brier_multiclass         | 0.0562 | 0.0340 | 0.0815  |
| xgb_scm_v2_best                    | ece_confidence           | 0.0590 | 0.0432 | 0.0728  |
| xgb_tabddpm_proto                  | accuracy                 | 0.9519 | 0.9280 | 0.9733  |
| xgb_tabddpm_proto                  | balanced_accuracy        | 0.8703 | 0.7936 | 0.9380  |
| xgb_tabddpm_proto                  | macro_f1                 | 0.8582 | 0.7927 | 0.9162  |
| xgb_tabddpm_proto                  | class0_recall            | 0.6546 | 0.4286 | 0.8637  |
| xgb_tabddpm_proto                  | class0_average_precision | 0.7685 | 0.6059 | 0.8970  |
| xgb_tabddpm_proto                  | brier_multiclass         | 0.0646 | 0.0436 | 0.0883  |
| xgb_tabddpm_proto                  | ece_confidence           | 0.0634 | 0.0535 | 0.0760  |
