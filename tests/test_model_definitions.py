import numpy as np
from malid.train import model_definitions
import pandas as pd
from wrap_glmnet import GlmnetLogitNetWrapper
import pytest


def test_convert_to_patient_level_scorer():
    sklearn_cv_scorer_at_patient_level = (
        model_definitions.convert_to_patient_level_scorer("neg_log_loss")
    )
    # Confirm it generates negative values due to picking up the sklearn metric sign settings
    class MockClassifier:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]])

    assert (
        sklearn_cv_scorer_at_patient_level(
            MockClassifier(),
            np.random.randn(4, 3),
            np.array(["spam", "ham", "ham", "spam"]),
            groups=np.array([1, 1, 2, 2]),
            sample_weight=np.array([1, 1, 0.5, 1]),
        )
        < 0
    )


@pytest.fixture
def data():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(15, 5)).rename(columns=lambda s: f"col{s}")
    y = np.array(["Covid19", "Healthy", "HIV"] * 5)
    participant_label = np.array(
        [
            "covid1",
            "healthy1",
            "hiv1",
            "covid2",
            "healthy2",
            "hiv2",
            "covid3",
            "healthy3",
            "hiv3",
            "covid1",
            "healthy1",
            "hiv1",
            "covid2",
            "healthy2",
            "hiv2",
        ]
    )
    return X, y, participant_label


@pytest.fixture
def data_binary():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(24, 5)).rename(columns=lambda s: f"col{s}")
    y = np.array(["Covid19", "Healthy"] * 12)
    participant_label = np.array(
        [
            "covid1",
            "healthy1",
            "covid2",
            "healthy2",
            "covid3",
            "healthy3",
        ]
        * 4
    )
    return X, y, participant_label


@pytest.mark.parametrize(
    "dataset", [pytest.lazy_fixture("data"), pytest.lazy_fixture("data_binary")]
)
def test_patient_level_scorers(dataset):
    # Like wrap_glmnet test_scorer
    for scorer in [
        model_definitions.glmnet_rocauc_scorer_at_patient_level,
        model_definitions.glmnet_deviance_scorer_at_patient_level,
        model_definitions.glmnet_matthews_corrcoef_scorer_at_patient_level,
    ]:
        X, y, groups = dataset
        clf = GlmnetLogitNetWrapper(alpha=1, n_lambda=5, n_splits=3, scoring=scorer)
        clf = clf.fit(X, y, groups=groups)
        assert clf.cv_mean_score_final_ is not None, f"scorer {scorer} failed"
