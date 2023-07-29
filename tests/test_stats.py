import numpy as np
from sklearn.linear_model import LogisticRegression

from malid import stats


def test_linear_model_supervised_embedding():
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        class_weight="balanced",
        random_state=0,
    )
    n_samples = 100
    n_features = 10
    n_classes = 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)
    clf = clf.fit(X, y)
    assert len(clf.classes_) == n_classes

    embedding = stats.linear_model_supervised_embedding(clf, X)
    assert embedding.shape == (n_samples, n_classes)
