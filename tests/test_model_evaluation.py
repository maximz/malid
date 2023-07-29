#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import malid.external.model_evaluation_scores
from malid.external import model_evaluation


def test_roc_auc_score_with_missing_labels_in_yscore():
    # confirm main functionality: handle missing classes automatically
    assert (
        malid.external.model_evaluation_scores.roc_auc_score(
            y_true=["Healthy", "Ebola", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.33, 0.33, 0.34],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        == 0.875
        # and confirm that adding missing classes as columns of 0s does not change ROC score
        == sklearn.metrics.roc_auc_score(
            y_true=["Healthy", "Ebola", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.0, 0.1, 0.8],
                    [0.33, 0.0, 0.33, 0.34],
                    [0.1, 0.0, 0.8, 0.1],
                    [0.05, 0.0, 0.05, 0.9],
                    [0.8, 0.0, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "Ebola", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        # and confirm that this is different from removing the entries for the missing classes
        != sklearn.metrics.roc_auc_score(
            y_true=["Healthy", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
    )

    # pass through if no missing classes
    assert (
        malid.external.model_evaluation_scores.roc_auc_score(
            y_true=["Healthy", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        == sklearn.metrics.roc_auc_score(
            y_true=["Healthy", "HIV", "Healthy", "Covid"],
            y_score=np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                ]
            ),
            labels=["Covid", "HIV", "Healthy"],
            multi_class="ovo",
            average="macro",
        )
        == 1.0
    )

    # further confirmation that adding columns of 0 probabilities (for missing classes) does not change ROC AUC
    y_true = np.array([0, 0, 1, 2])
    y_scores = np.array(
        [[0.1, 0.9, 0.0], [0.3, 0.6, 0.1], [0.35, 0.6, 0.05], [0.8, 0.2, 0.0]]
    )
    y_scores2 = np.array(
        [
            [0.1, 0.9, 0.0, 0.0],
            [0.3, 0.6, 0.1, 0.0],
            [0.35, 0.6, 0.05, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ]
    )
    assert (
        sklearn.metrics.roc_auc_score(
            y_true, y_scores, multi_class="ovo", labels=[0, 1, 2], average="macro"
        )
        == sklearn.metrics.roc_auc_score(
            y_true, y_scores2, multi_class="ovo", labels=[0, 1, 2, 3], average="macro"
        )
        == 0.25
    )
    assert (
        sklearn.metrics.roc_auc_score(
            y_true, y_scores, multi_class="ovo", labels=[0, 1, 2], average="weighted"
        )
        == sklearn.metrics.roc_auc_score(
            y_true,
            y_scores2,
            multi_class="ovo",
            labels=[0, 1, 2, 3],
            average="weighted",
        )
        == 0.21875
    )


@pytest.mark.xfail(raises=ValueError)
def test_roc_auc_with_missing_labels_must_be_multiclass_ovo_mode():
    malid.external.model_evaluation_scores.roc_auc_score(
        y_true=["Healthy", "Ebola", "HIV", "Healthy", "Covid"],
        y_score=np.array(
            [
                [0.1, 0.1, 0.8],
                [0.33, 0.33, 0.34],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
                [0.8, 0.1, 0.1],
            ]
        ),
        labels=["Covid", "HIV", "Healthy"],
        multi_class="ovr",
        average="macro",
    )


def test_roc_auc_score_doesnt_have_to_sum_to_one():
    assert malid.external.model_evaluation_scores.roc_auc_score(
        y_true=["Covid", "HIV", "Healthy"],
        y_score=np.array(
            [
                [0.1, 0.1, 0.8],
                [0.33, 0.33, 0.34],
                [0.1, 0.8, 0.1],
            ]
        ),
        labels=["Covid", "HIV", "Healthy"],
        multi_class="ovo",
        average="macro",
    ) == malid.external.model_evaluation_scores.roc_auc_score(
        y_true=["Covid", "HIV", "Healthy"],
        y_score=np.array(
            [
                [0.2, 0.1, 0.8],
                [0.66, 0.33, 0.34],
                [0.2, 0.8, 0.1],
            ]
        ),
        labels=["Covid", "HIV", "Healthy"],
        multi_class="ovo",
        average="macro",
    )


def test_roc_auc_score_with_missing_labels_in_ytest():
    # confirm main functionality: handle missing classes automatically
    # also tests that roc auc score does not have to sum to 1
    assert np.allclose(
        [
            malid.external.model_evaluation_scores.roc_auc_score(
                y_true=["Covid", "Covid", "Covid", "Healthy", "Healthy"],
                y_score=np.array(
                    [
                        [0.1, 0.1, 0.8],
                        [0.4, 0.5, 0.1],
                        [0.8, 0.1, 0.1],
                        [0.33, 0.33, 0.34],
                        [0.2, 0.2, 0.6],
                    ]
                ),
                labels=["Covid", "HIV", "Healthy"],
                multi_class="ovo",
                average="macro",
            ),
            # and confirm that removing missing classes does not change ROC score (though sklearn requires renorm to sum to 1)
            sklearn.metrics.roc_auc_score(
                y_true=["Covid", "Covid", "Covid", "Healthy", "Healthy"],
                y_score=np.array(
                    [
                        [0.2, 0.8],
                        [0.8, 0.2],
                        [0.8, 0.2],
                        [0.49, 0.51],
                        [0.25, 0.75],
                    ]
                )[:, 1],
                labels=["Covid", "Healthy"],
                multi_class="ovo",
                average="macro",
            ),
        ],
        2 / 3,
    )


def test_roc_auc_score_with_so_many_missing_labels_that_only_one_label_is_left():
    # 3 initial categorical labels
    # but all y_true belongs to a single label
    y_true = pd.Series(
        ["surviving_class", "surviving_class"],
        dtype=pd.CategoricalDtype(
            categories=["surviving_class", "other_class", "dropped_class"]
        ),
    )
    # classifier only had 2 of the 3 initial classes
    y_score = np.array([[0.6, 0.4], [0.5, 0.5]])
    labels = np.array(["surviving_class", "dropped_class"])

    (
        y_score_modified,
        labels_modified,
    ) = malid.external.model_evaluation_scores._inject_missing_labels(
        y_true=y_true,
        y_score=y_score,
        labels=labels,
    )
    assert np.array_equal(y_score_modified, np.array([[0.6], [0.5]]))
    assert np.array_equal(labels_modified, ["surviving_class"])

    for average in ["weighted", "macro"]:
        with pytest.raises(
            ValueError,
            match="Only one class present in y_true. Probability-based score is not defined in that case.",
        ):
            assert (
                malid.external.model_evaluation_scores.roc_auc_score(
                    y_true=y_true,
                    y_score=y_score,
                    labels=labels,
                    average=average,
                )
                == 0.5
            )


models_factory = lambda: {
    "dummy": DummyClassifier(strategy="stratified"),
    "logistic_multinomial": LogisticRegression(multi_class="multinomial"),
    "logistic_ovr": LogisticRegression(multi_class="ovr"),
    "randomforest": RandomForestClassifier(),
    "linearsvm": SVC(kernel="linear"),
    "nonlinear_svm": SVC(kernel="rbf"),
}


@pytest.fixture(name="models_factory")
def models_factory_fixture():
    # Pass the lambda function factory as a fixture
    # https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
    return models_factory


all_model_names = list(models_factory().keys())
tree_models = ["randomforest"]
# multiclass OvR:
ovr_models = ["logistic_multinomial", "logistic_ovr"]
# multiclass OvO:
ovo_models = ["linearsvm"]
# no coefs_, feature_importances_, or feature names:
dummy_models = ["dummy"]
# no coefs_ or feature_importances_ but does have feature names:
no_coefs_models = ["nonlinear_svm"]


def test_model_evaluation(sample_data, sample_data_two, models_factory, tmp_path):
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)
    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()
    for model in all_model_names:
        print(model)
        print(
            experiment_set_global_performance.model_global_performances[
                model
            ].full_report()
        )
        experiment_set_global_performance.model_global_performances[
            model
        ].confusion_matrix_fig()
        print()

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    assert combined_stats.loc["logistic_multinomial"].equals(
        pd.Series(
            {
                "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy global": "1.000",
                "MCC global": "1.000",
                "sample_size": 10,
                "n_abstentions": 0,
                "sample_size including abstentions": 10,
                "abstention_rate": 0.0,
                "missing_classes": False,
            }
        )
    ), f"Observed: {combined_stats.loc['logistic_multinomial'].to_dict()}"
    assert (
        experiment_set_global_performance.model_global_performances[
            "logistic_multinomial"
        ].full_report()
        == """Per-fold scores:
ROC-AUC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
ROC-AUC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
Accuracy: 1.000 +/- 0.000 (in 2 folds)
MCC: 1.000 +/- 0.000 (in 2 folds)

Global scores:
Accuracy: 1.000
MCC: 1.000

Global classification report:
              precision    recall  f1-score   support

       Covid       1.00      1.00      1.00         4
       Ebola       1.00      1.00      1.00         2
         HIV       1.00      1.00      1.00         2
     Healthy       1.00      1.00      1.00         2

    accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10
"""
    )

    experiment_set_global_performance.export_all_models(
        func_generate_classification_report_fname=lambda model_name: tmp_path
        / f"{model_name}.classification_report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: tmp_path
        / f"{model_name}.confusion_matrix.png",
        confusion_matrix_figsize=(4, 4),
        dpi=72,
    )


def test_model_evaluation_with_abstention(
    sample_data, sample_data_two, models_factory, tmp_path
):
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            test_metadata = pd.DataFrame(
                {
                    "patient_id": range(X_test.shape[0]),
                    "alternate_ground_truth_column": y_test,
                }
            )
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=test_metadata,
                test_abstentions=["Healthy", "HIV"],
                test_abstention_metadata=pd.DataFrame(
                    {
                        "patient_id": [-1, -2],
                        "alternate_ground_truth_column": ["Healthy", "HIV"],
                    }
                ),
                test_sample_weights=np.ones(X_test.shape[0]),
                test_abstention_sample_weights=np.ones(2),
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)
    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()
    for model in all_model_names:
        print(model)
        print(
            experiment_set_global_performance.model_global_performances[
                model
            ].full_report()
        )
        experiment_set_global_performance.model_global_performances[
            model
        ].confusion_matrix_fig()
        print()

    all_entries = experiment_set_global_performance.model_global_performances[
        "logistic_multinomial"
    ].get_all_entries()
    print(all_entries)
    assert not all_entries["y_true"].isna().any()
    assert all_entries["difference_between_top_two_predicted_probas"].isna().sum() == 4
    assert all_entries.shape[0] == 14

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    assert combined_stats.loc["logistic_multinomial"].equals(
        pd.Series(
            {
                "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy with sample weights per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC with sample weights per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy global": "1.000",
                "Accuracy with sample weights global": "1.000",
                "MCC global": "1.000",
                "MCC with sample weights global": "1.000",
                "Accuracy per fold with abstention": "0.714 +/- 0.000 (in 2 folds)",
                "Accuracy with sample weights per fold with abstention": "0.714 +/- 0.000 (in 2 folds)",
                "MCC per fold with abstention": "0.703 +/- 0.000 (in 2 folds)",
                "MCC with sample weights per fold with abstention": "0.703 +/- 0.000 (in 2 folds)",
                "Unknown/abstention proportion per fold with abstention": "0.286 +/- 0.000 (in 2 folds)",
                "Accuracy global with abstention": "0.714",
                "Accuracy with sample weights global with abstention": "0.714",
                "MCC global with abstention": "0.703",
                "MCC with sample weights global with abstention": "0.703",
                "Unknown/abstention proportion global with abstention": "0.286",
                "Abstention label global with abstention": "Unknown",
                "sample_size": 10,
                "n_abstentions": 4,
                "sample_size including abstentions": 14,
                "abstention_rate": 4 / 14,
                "missing_classes": False,
            },
        )
    ), f"Observed: {combined_stats.loc['logistic_multinomial'].to_dict()}"
    assert (
        experiment_set_global_performance.model_global_performances[
            "logistic_multinomial"
        ].full_report()
        == """Per-fold scores without abstention:
ROC-AUC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
ROC-AUC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
Accuracy: 1.000 +/- 0.000 (in 2 folds)
Accuracy with sample weights: 1.000 +/- 0.000 (in 2 folds)
MCC: 1.000 +/- 0.000 (in 2 folds)
MCC with sample weights: 1.000 +/- 0.000 (in 2 folds)

Global scores without abstention:
Accuracy: 1.000
Accuracy with sample weights: 1.000
MCC: 1.000
MCC with sample weights: 1.000

Per-fold scores with abstention (note that abstentions not included in probability-based scores):
Accuracy: 0.714 +/- 0.000 (in 2 folds)
Accuracy with sample weights: 0.714 +/- 0.000 (in 2 folds)
MCC: 0.703 +/- 0.000 (in 2 folds)
MCC with sample weights: 0.703 +/- 0.000 (in 2 folds)
Unknown/abstention proportion: 0.286 +/- 0.000 (in 2 folds)

Global scores with abstention:
Accuracy: 0.714
Accuracy with sample weights: 0.714
MCC: 0.703
MCC with sample weights: 0.703
Unknown/abstention proportion: 0.286
Abstention label: Unknown

Global classification report with abstention:
              precision    recall  f1-score   support

       Covid       1.00      1.00      1.00         4
       Ebola       1.00      1.00      1.00         2
         HIV       1.00      0.50      0.67         4
     Healthy       1.00      0.50      0.67         4
     Unknown       0.00      0.00      0.00         0

    accuracy                           0.71        14
   macro avg       0.80      0.60      0.67        14
weighted avg       1.00      0.71      0.81        14
"""
    )

    experiment_set_global_performance.export_all_models(
        func_generate_classification_report_fname=lambda model_name: tmp_path
        / f"{model_name}.classification_report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: tmp_path
        / f"{model_name}.confusion_matrix.png",
        dpi=72,
    )


def test_ModelSingleFoldPerformance_constructor_patterns(sample_data, models_factory):
    (X_train, y_train, X_test, y_test) = sample_data
    clf = models_factory()["logistic_multinomial"].fit(X_train, y_train)

    # typical: pass clf and X_test
    model_evaluation.ModelSingleFoldPerformance(
        model_name="logistic_multinomial",
        fold_id=0,
        clf=clf,
        X_test=X_test,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )

    # atypical: pass computed fields directly
    model_evaluation.ModelSingleFoldPerformance(
        model_name="logistic_multinomial",
        fold_id=0,
        y_true=y_test,
        y_pred=clf.predict(X_test),
        class_names=clf.classes_,
        X_test_shape=X_test.shape,
        y_decision_function=clf.decision_function(X_test),
        y_preds_proba=clf.predict_proba(X_test),
        fold_label_train="train",
        fold_label_test="test",
    )


@pytest.mark.xfail
def test_ModelSingleFoldPerformance_constructor_requires_one_of_two_patterns(
    sample_data,
):
    (X_train, y_train, X_test, y_test) = sample_data

    # passing none of the above does not work
    model_evaluation.ModelSingleFoldPerformance(
        model_name="logistic_multinomial",
        fold_id=0,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )


def test_model_evaluation_with_alternate_column_name(
    sample_data, sample_data_two, models_factory, tmp_path
):
    """test scenario where we provide metadata object and an alternate column name, including metadata for abstentions."""
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            test_metadata = pd.DataFrame(
                {
                    "patient_id": range(X_test.shape[0]),
                    "alternate_ground_truth_column": y_test,
                }
            )
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=test_metadata,
                test_abstentions=["Healthy", "HIV"],
                test_abstention_metadata=pd.DataFrame(
                    {
                        "patient_id": [-1, -2],
                        "alternate_ground_truth_column": ["Healthy", "HIV"],
                    }
                ),
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)

    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize(
        global_evaluation_column_name="alternate_ground_truth_column"
    )

    for model in all_model_names:
        print(model)
        print(
            experiment_set_global_performance.model_global_performances[
                model
            ].full_report()
        )
        experiment_set_global_performance.model_global_performances[
            model
        ].confusion_matrix_fig()
        print()

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    assert combined_stats.loc["logistic_multinomial"].equals(
        pd.Series(
            {
                "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy global": "1.000",
                "MCC global": "1.000",
                "Global evaluation column name global": "alternate_ground_truth_column",
                "Accuracy per fold with abstention": "0.714 +/- 0.000 (in 2 folds)",
                "MCC per fold with abstention": "0.703 +/- 0.000 (in 2 folds)",
                "Unknown/abstention proportion per fold with abstention": "0.286 +/- 0.000 (in 2 folds)",
                "Accuracy global with abstention": "0.714",
                "MCC global with abstention": "0.703",
                "Unknown/abstention proportion global with abstention": "0.286",
                "Abstention label global with abstention": "Unknown",
                "Global evaluation column name global with abstention": "alternate_ground_truth_column",
                "sample_size": 10,
                "n_abstentions": 4,
                "sample_size including abstentions": 14,
                "abstention_rate": 4 / 14,
                "missing_classes": False,
            },
        )
    ), f"Observed: {combined_stats.loc['logistic_multinomial'].to_dict()}"
    assert (
        experiment_set_global_performance.model_global_performances[
            "logistic_multinomial"
        ].full_report()
        == """Per-fold scores without abstention:
ROC-AUC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
ROC-AUC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
Accuracy: 1.000 +/- 0.000 (in 2 folds)
MCC: 1.000 +/- 0.000 (in 2 folds)

Global scores without abstention using column name alternate_ground_truth_column:
Accuracy: 1.000
MCC: 1.000
Global evaluation column name: alternate_ground_truth_column

Per-fold scores with abstention (note that abstentions not included in probability-based scores):
Accuracy: 0.714 +/- 0.000 (in 2 folds)
MCC: 0.703 +/- 0.000 (in 2 folds)
Unknown/abstention proportion: 0.286 +/- 0.000 (in 2 folds)

Global scores with abstention using column name alternate_ground_truth_column:
Accuracy: 0.714
MCC: 0.703
Unknown/abstention proportion: 0.286
Abstention label: Unknown
Global evaluation column name: alternate_ground_truth_column

Global classification report with abstention using column name alternate_ground_truth_column:
              precision    recall  f1-score   support

       Covid       1.00      1.00      1.00         4
       Ebola       1.00      1.00      1.00         2
         HIV       1.00      0.50      0.67         4
     Healthy       1.00      0.50      0.67         4
     Unknown       0.00      0.00      0.00         0

    accuracy                           0.71        14
   macro avg       0.80      0.60      0.67        14
weighted avg       1.00      0.71      0.81        14
"""
    )

    experiment_set_global_performance.export_all_models(
        func_generate_classification_report_fname=lambda model_name: tmp_path
        / f"{model_name}.classification_report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: tmp_path
        / f"{model_name}.confusion_matrix.png",
        dpi=72,
    )


@pytest.mark.xfail
def test_metadata_object_required_for_alternate_column(
    sample_data, sample_data_two, models_factory
):
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            # TODO: try the train method from model_evaluation, along with export
            clf = clf.fit(X_train, y_train)
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_abstentions=["Healthy", "HIV"],
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)
    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize(
        global_evaluation_column_name=[model_evaluation.Y_TRUE_VALUES, "column_dne"]
    )


def test_model_evaluation_with_multiple_alternate_column_names(
    sample_data, sample_data_two, models_factory, tmp_path
):
    """test scenario where we provide metadata object and an alternate column name, including metadata for abstentions."""
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            test_metadata = pd.DataFrame(
                {
                    "patient_id": range(X_test.shape[0]),
                    "alternate_ground_truth_column": [f"alternate_{s}" for s in y_test],
                }
            )
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=test_metadata,
                test_abstentions=["Healthy", "HIV"],
                test_abstention_metadata=pd.DataFrame(
                    {
                        "patient_id": [-1, -2],
                        "alternate_ground_truth_column": [
                            "alternate_Healthy",
                            "alternate_HIV",
                        ],
                    }
                ),
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)

    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize(
        global_evaluation_column_name=[
            model_evaluation.Y_TRUE_VALUES,
            "alternate_ground_truth_column",
        ]
    )

    for model in all_model_names:
        print(model)
        cm = experiment_set_global_performance.model_global_performances[
            model
        ].confusion_matrix()
        print(cm)
        print()

        assert np.array_equal(
            cm.index,
            [
                "Covid, alternate_Covid",
                "Ebola, alternate_Ebola",
                "HIV, alternate_HIV",
                "Healthy, alternate_Healthy",
            ],
        )

        # Confirm repr of Y_TRUE_VALUES is printed correctly
        assert (
            "Global classification report with abstention using column name [<default y_true column>, 'alternate_ground_truth_column']"
            in experiment_set_global_performance.model_global_performances[
                model
            ].full_report()
        )


def test_feature_importances_and_names_with_default_feature_names(
    sample_data, sample_data_two, models_factory
):
    """Numpy data -> default feature names"""
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
            )

            ## Test feature names and feature importances for single fold
            if model_name in dummy_models:
                # dummy has no coefs, and also swallows feature names
                assert single_perf.feature_names is None
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in ovr_models:
                # multiclass OvR linear model is special cased
                assert np.array_equal(single_perf.feature_names, [0, 1, 2, 3, 4])
                assert single_perf.feature_importances is None
                assert type(single_perf.multiclass_feature_importances) == pd.DataFrame
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.index, clf.classes_
                )
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.columns,
                    single_perf.feature_names,
                )
            elif model_name in ovo_models or model_name in no_coefs_models:
                # multiclass OvO linear model does not support feature importances,
                # but does store feature names.
                assert np.array_equal(single_perf.feature_names, [0, 1, 2, 3, 4])
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in tree_models:
                # These work just like binary linear models
                assert np.array_equal(single_perf.feature_names, [0, 1, 2, 3, 4])
                assert type(single_perf.feature_importances) == pd.Series
                assert np.array_equal(
                    single_perf.feature_importances.index, single_perf.feature_names
                )
                assert single_perf.multiclass_feature_importances is None
            else:
                raise ValueError("Did not expect other model types")
            model_outputs.append(single_perf)

    ## Test with multiple folds at aggregated level
    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()

    for model in dummy_models:
        # no feature importances
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in ovr_models:
        # multiclass OvR
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        dict_of_dfs = experiment_set_global_performance.model_global_performances[
            model
        ].multiclass_feature_importances
        assert np.array_equal(list(dict_of_dfs.keys()), [0, 1])
        for df in dict_of_dfs.values():
            assert np.array_equal(df.index, ["Covid", "Ebola", "HIV", "Healthy"])
            assert np.array_equal(df.columns, [0, 1, 2, 3, 4])

    for model in ovo_models:
        # multiclass OvO is not supported
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in no_coefs_models:
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in tree_models:
        # These work just like binary linear models
        df = experiment_set_global_performance.model_global_performances[
            model
        ].feature_importances
        assert df.index.name == "fold_id"
        assert np.array_equal(df.index, [0, 1])
        assert np.array_equal(df.columns, [0, 1, 2, 3, 4])
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )


def test_feature_importances_and_names_with_custom_feature_names(
    sample_data, sample_data_two, models_factory
):
    """Pandas data -> custom feature names"""
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            X_train_df = pd.DataFrame(X_train).rename(columns=lambda s: f"feature_{s}")
            X_test_df = pd.DataFrame(X_test).rename(columns=lambda s: f"feature_{s}")

            clf = clf.fit(X_train_df, y_train)
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test_df,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
            )

            ## Test feature names and feature importances for single fold

            if model_name in dummy_models:
                # dummy has no coefs, and also swallows feature names
                assert single_perf.feature_names is None
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in ovr_models:
                # multiclass OvR linear model is special cased
                assert np.array_equal(
                    single_perf.feature_names,
                    ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                )
                assert single_perf.feature_importances is None
                assert type(single_perf.multiclass_feature_importances) == pd.DataFrame
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.index, clf.classes_
                )
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.columns,
                    single_perf.feature_names,
                )
            elif model_name in ovo_models or model_name in no_coefs_models:
                # multiclass OvO linear model does not support feature importances,
                # but does store feature names.
                assert np.array_equal(
                    single_perf.feature_names,
                    ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                )
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in tree_models:
                # These work just like binary linear models
                assert np.array_equal(
                    single_perf.feature_names,
                    ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                )
                assert type(single_perf.feature_importances) == pd.Series
                assert np.array_equal(
                    single_perf.feature_importances.index, single_perf.feature_names
                )
                assert single_perf.multiclass_feature_importances is None
            else:
                raise ValueError("Did not expect other model types")
            model_outputs.append(single_perf)

    ## Test with multiple folds at aggregated level
    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()

    for model in dummy_models:
        # no feature importances
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in ovr_models:
        # multiclass OvR
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        dict_of_dfs = experiment_set_global_performance.model_global_performances[
            model
        ].multiclass_feature_importances
        assert np.array_equal(list(dict_of_dfs.keys()), [0, 1])
        for df in dict_of_dfs.values():
            assert np.array_equal(df.index, ["Covid", "Ebola", "HIV", "Healthy"])
            assert np.array_equal(
                df.columns,
                ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
            )

    for model in ovo_models:
        # multiclass OvO is not supported
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in no_coefs_models:
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in tree_models:
        # These work just like binary linear models
        df = experiment_set_global_performance.model_global_performances[
            model
        ].feature_importances
        assert df.index.name == "fold_id"
        assert np.array_equal(df.index, [0, 1])
        assert np.array_equal(
            df.columns,
            ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )


def test_sentinel_value():
    # we want to make sure our Y_TRUE_VALUES behaves
    import copy, pickle

    assert (
        str(model_evaluation.Y_TRUE_VALUES)
        == repr(model_evaluation.Y_TRUE_VALUES)
        == "<default y_true column>"
    )
    assert model_evaluation.Y_TRUE_VALUES is model_evaluation.Y_TRUE_VALUES
    assert model_evaluation.Y_TRUE_VALUES is not object()
    assert model_evaluation.Y_TRUE_VALUES is pickle.loads(
        pickle.dumps(model_evaluation.Y_TRUE_VALUES)
    )
    assert (
        copy.deepcopy(model_evaluation.Y_TRUE_VALUES) is model_evaluation.Y_TRUE_VALUES
    )


def test_ModelSingleFoldPerformance_copy(sample_data, models_factory):
    (X_train, y_train, X_test, y_test) = sample_data
    model_name, clf = next(iter(models_factory().items()))
    clf = clf.fit(X_train, y_train)
    single_perf = model_evaluation.ModelSingleFoldPerformance(
        model_name=model_name,
        fold_id=0,
        clf=clf,
        X_test=X_test,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )
    single_perf_copy = single_perf.copy()
    assert np.array_equal(single_perf.y_pred, single_perf_copy.y_pred)


def test_ModelSingleFoldPerformance_apply_abstention_mask(
    sample_data, sample_data_two, models_factory
):
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=np.vstack([X_test, X_test[:2]]),
                y_true=np.hstack([y_test, ["Healthy1", "HIV1"]]),
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=pd.DataFrame(
                    {
                        "patient_id": np.hstack([np.arange(X_test.shape[0]), [-1, -2]]),
                    }
                ),
                test_sample_weights=np.hstack([np.ones(X_test.shape[0]), [1.1, 1.1]]),
            )

            # Confirm initial state
            assert (
                single_perf.y_true.shape[0]
                == single_perf.y_pred.shape[0]
                == single_perf.X_test_shape[0]
                == single_perf.test_metadata.shape[0]
                == single_perf.test_sample_weights.shape[0]
                == y_test.shape[0] + 2
            )
            if single_perf.y_decision_function is not None:
                # it's None for "dummy" model
                assert single_perf.y_decision_function.shape[0] == y_test.shape[0] + 2
            if single_perf.y_preds_proba is not None:
                # it's None for "linearsvm" model
                assert single_perf.y_preds_proba.shape[0] == y_test.shape[0] + 2
            assert single_perf.test_abstentions is None
            assert single_perf.test_abstention_metadata is None
            assert single_perf.test_abstention_sample_weights is None

            # Apply mask
            single_perf = single_perf.apply_abstention_mask(
                single_perf.test_metadata["patient_id"] < 0
            )

            # Confirm new sizes
            assert (
                single_perf.y_true.shape[0]
                == single_perf.y_pred.shape[0]
                == single_perf.X_test_shape[0]
                == single_perf.test_metadata.shape[0]
                == single_perf.test_sample_weights.shape[0]
                == y_test.shape[0]
            )
            if single_perf.y_decision_function is not None:
                # it's None for "dummy" model
                assert single_perf.y_decision_function.shape[0] == y_test.shape[0]
            if single_perf.y_preds_proba is not None:
                # it's None for "linearsvm" model
                assert single_perf.y_preds_proba.shape[0] == y_test.shape[0]
            assert (
                single_perf.test_abstentions.shape[0]
                == single_perf.test_abstention_metadata.shape[0]
                == single_perf.test_abstention_sample_weights.shape[0]
                == 2
            )
            assert all(single_perf.test_metadata["patient_id"] >= 0)
            assert all(single_perf.test_abstention_metadata["patient_id"] < 0)
            assert np.array_equal(single_perf.test_abstentions, ["Healthy1", "HIV1"])
            assert all(single_perf.test_abstention_sample_weights == 1.1)
            assert all(single_perf.test_sample_weights == 1.0)

            model_outputs.append(single_perf)
    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()
    for model in all_model_names:
        print(model)
        print(
            experiment_set_global_performance.model_global_performances[
                model
            ].full_report()
        )
        experiment_set_global_performance.model_global_performances[
            model
        ].confusion_matrix_fig()
        print()

    all_entries = experiment_set_global_performance.model_global_performances[
        "logistic_multinomial"
    ].get_all_entries()
    print(all_entries)
    assert not all_entries["y_true"].isna().any()
    assert all_entries["difference_between_top_two_predicted_probas"].isna().sum() == 4
    assert all_entries.shape[0] == 14

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    assert combined_stats.loc["logistic_multinomial"].equals(
        pd.Series(
            {
                "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy with sample weights per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC with sample weights per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy global": "1.000",
                "Accuracy with sample weights global": "1.000",
                "MCC global": "1.000",
                "MCC with sample weights global": "1.000",
                "Accuracy per fold with abstention": "0.714 +/- 0.000 (in 2 folds)",
                "Accuracy with sample weights per fold with abstention": "0.694 +/- 0.000 (in 2 folds)",
                "MCC per fold with abstention": "0.718 +/- 0.000 (in 2 folds)",
                "MCC with sample weights per fold with abstention": "0.704 +/- 0.000 (in 2 folds)",
                "Unknown/abstention proportion per fold with abstention": "0.286 +/- 0.000 (in 2 folds)",
                "Accuracy global with abstention": "0.714",
                "Accuracy with sample weights global with abstention": "0.694",
                "MCC global with abstention": "0.718",
                "MCC with sample weights global with abstention": "0.704",
                "Unknown/abstention proportion global with abstention": "0.286",
                "Abstention label global with abstention": "Unknown",
                "sample_size": 10,
                "n_abstentions": 4,
                "sample_size including abstentions": 14,
                "abstention_rate": 4 / 14,
                "missing_classes": False,
            },
        )
    ), f"Observed: {combined_stats.loc['logistic_multinomial'].to_dict()}"


def test_ModelSingleFoldPerformance_apply_abstention_mask_empty_mask(
    sample_data, models_factory
):
    # Edge case: mask nothing
    (X_train, y_train, X_test, y_test) = sample_data
    model_name = "logistic_multinomial"
    clf = models_factory()[model_name]
    clf = clf.fit(X_train, y_train)
    single_perf = model_evaluation.ModelSingleFoldPerformance(
        model_name=model_name,
        fold_id=0,
        clf=clf,
        X_test=X_test,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )

    original_scores = single_perf.scores()

    # Apply mask of all False
    single_perf = single_perf.apply_abstention_mask(np.full(y_test.shape[0], False))

    # Confirm new sizes
    assert (
        single_perf.y_true.shape[0]
        == single_perf.y_pred.shape[0]
        == single_perf.y_decision_function.shape[0]
        == single_perf.y_preds_proba.shape[0]
        == single_perf.X_test_shape[0]
        == y_test.shape[0]
    )
    # assert single_perf.test_abstentions is None # TODO: This is broken - but we want to move towards this behavior anyway
    assert single_perf.test_abstention_metadata is None
    assert single_perf.test_abstention_sample_weights is None

    assert original_scores == single_perf.scores()


def test_ModelSingleFoldPerformance_apply_abstention_mask_entire_mask(
    sample_data, sample_data_two, models_factory
):
    # Edge case: mask everything
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            single_perf = model_evaluation.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
            )

            # Apply mask of all True
            single_perf = single_perf.apply_abstention_mask(
                np.full(y_test.shape[0], True)
            )

            # Confirm new sizes
            assert (
                single_perf.y_true.shape[0]
                == single_perf.y_pred.shape[0]
                == single_perf.X_test_shape[0]
                == 0
            )
            if single_perf.y_decision_function is not None:
                # it's None for "dummy" model
                assert single_perf.y_decision_function.shape[0] == 0
            if single_perf.y_preds_proba is not None:
                # it's None for "linearsvm" model
                assert single_perf.y_preds_proba.shape[0] == 0
            assert single_perf.test_abstentions.shape[0] == y_test.shape[0]

            model_outputs.append(single_perf)
    experiment_set = model_evaluation.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()

    # Carrying around len(y_true) == 0 objects should be fine, until it's time to compute scores
    with pytest.raises(ValueError, match="Cannot compute scores when y_true is empty"):
        experiment_set_global_performance.get_model_comparison_stats()


def test_featurized_data_apply_abstention_mask():
    fd = model_evaluation.FeaturizedData(
        X=pd.DataFrame(np.ones((3, 5))),
        y=np.array([0, 1, 2]),
        # note the mixed types
        sample_names=np.array([1, "2", 3], dtype="object"),
        metadata=pd.DataFrame({"sample_name": [1, 2, 3]}),
        sample_weights=np.array([0.1, 0.2, 0.3]),
        extras={"key": "value"},
    )
    fd.X.values[2, :] = 0
    fd_new = fd.apply_abstention_mask(mask=(np.array(fd.X) == 0).all(axis=1))
    assert fd_new.X.shape == (2, 5)
    assert (np.array(fd_new.X) == 1).all()
    assert np.array_equal(fd_new.y, [0, 1])
    assert np.array_equal(fd_new.sample_names, np.array([1, "2"], dtype="object"))
    assert np.array_equal(fd_new.metadata["sample_name"].values, [1, 2])
    assert np.array_equal(fd_new.abstained_sample_names, [3])
    assert np.array_equal(fd_new.abstained_sample_y, [2])
    assert np.array_equal(fd_new.abstained_sample_metadata["sample_name"].values, [3])
    assert np.array_equal(fd_new.sample_weights, [0.1, 0.2])
    assert fd_new.extras == {"key": "value"}


def test_metric_comparison():
    # comparison by value, not by name
    assert model_evaluation.Metric(
        value=1.0, friendly_name="metric1"
    ) > model_evaluation.Metric(value=0.5, friendly_name="metric1")
    assert model_evaluation.Metric(
        value=1.0, friendly_name="metric1"
    ) > model_evaluation.Metric(value=0.5, friendly_name="metric2")
    assert model_evaluation.Metric(
        value=1.0, friendly_name="metric1"
    ) == model_evaluation.Metric(value=1.0, friendly_name="metric2")
