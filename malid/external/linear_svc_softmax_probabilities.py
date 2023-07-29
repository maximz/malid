import numpy as np
from sklearn.svm import LinearSVC

import malid.external.genetools_stats


class LinearSVCSoftmaxProbabilities(LinearSVC):
    """LinearSVC has decision_function but not predict_proba, unless you wrap with a CalibratedClassifier.

    The goal: compute ROC AUC.
    roc_auc_score does not accept decision function for multiclass scenarios; it requires probabilities that sum to 1.
    This wrapper of LinearSVC adds a predict_proba function by softmaxing the decision_function outputs.

    Alternative:

    ```python
        CalibratedClassifierCV(
            OneVsOneClassifier(
                svm.LinearSVC(dual=False, random_state=0),
                n_jobs=n_jobs,
            ),
            cv=5,
            method="sigmoid",
        )
    ```

    """

    def predict_proba(self, X):
        """Predicted probabilities by softmaxing decision function. Output shape `(n_samples, n_classes)`.
        Note that in binary case, decision_function will have shape `(n_samples, )`, but predict_proba will still have shape `(n_samples, 2)`, per sklearn convention.
        """
        distances = self.decision_function(X)
        if distances.ndim == 1:
            # In binary case, decision_function will have shape `(n_samples, )`
            # To run softmax, we need a 2d array of [-distance, +distance]
            # And we want to return a predict_proba with shape `(n_samples, 2)`
            # see https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/linear_model/_logistic.py#L1471
            distances = np.c_[-distances, distances]
        return malid.external.genetools_stats.softmax(distances)
