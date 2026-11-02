# Evaluation methodology and recorded results

All numbers below are recorded project artifacts. No evaluation was run while preparing this public snapshot.

## Email

The saved results report a 16,416-row stratified holdout. The aggregate audit counts are 7,847 legitimate and 8,569 phishing rows. Confusion-matrix order is actual legitimate/phishing by predicted legitimate/phishing.

| Model | Accuracy | Phishing precision | Phishing recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression pipeline | 98.47% | 98.29% | 98.79% | 98.54% | 99.87% | 7,700 | 147 | 104 | 8,465 |
| Multinomial Naive Bayes | 96.37% | 98.84% | 94.15% | 96.44% | 99.70% | 7,752 | 95 | 501 | 8,068 |

The email training code performs 5-fold stratified CV on the training partition with shuffle=True and random_state=42. The saved CSV reports means and standard deviations. However, the same 20% holdout is scored for both models and the model with the higher holdout F1 is selected for serialization. Therefore, the holdout is not independent of model selection and should not be presented as a final unbiased estimate.

## Deployed 25-feature URL model

The recorded row-wise holdout contains 19,183 rows: 9,602 legitimate and 9,581 phishing. Its confusion matrix is actual legitimate/phishing by predicted legitimate/phishing.

| Accuracy | Phishing precision | Phishing recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 93.37% | 94.88% | 91.68% | 93.25% | 97.57% | 9,128 | 474 | 797 | 8,784 |

A prior PSL-based audit replayed the row-wise split and found 5,142 shared registrable domains. These shared domains account for 12,640 test rows (65.891675%) and 39,353 training rows (51.287632%). The audit used a pinned PSL snapshot with PRIVATE suffixes enabled. The overlap does not prove that every prediction is leaked, but it limits claims about unseen-domain generalization. One exact URL string was also found in both partitions.

A candidate domain-disjoint subset was sized in prior analysis, but no performance result for that subset is included here. No domain-grouped retraining experiment has been performed.

## Separate URL cross-validation artifact

results/url_cross_validation_results.csv is from training/cross_validate_models.py, which loads combined_features_train.csv and evaluates different model configurations. It is not cross-validation of the deployed 25-feature Random Forest. Do not compare it as if it were the deployed model CV result.

## Counterfactual sensitivity summary

results/email_sensitivity_audit/group_summary.csv reports token-group masking on the existing email holdout. These are transformed-input sensitivity observations, not new independent evaluations, proof of causal feature importance, or evidence of improved production performance. Date-like masks can remove month or year tokens used in ordinary prose. Individual contribution scores were not computed.

## Reproducibility and interpretation

The source scripts define the described splits and model configurations, but datasets are absent from this snapshot and exact source versions/licenses are not established. Results have not been independently reproduced from this release. Global email coefficient rankings were omitted because derived lexical terms were not cleared for public release. See dataset-provenance.md and reproducing-results.md.