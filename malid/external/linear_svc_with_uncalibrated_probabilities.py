import numpy as np
from sklearn.svm import LinearSVC

import genetools.stats


class LinearSVCWithUncalibratedProbabilities(LinearSVC):
    """LinearSVC has decision_function but not predict_proba, unless you wrap with a CalibratedClassifier.

    The goal: compute ROC AUC.
    roc_auc_score does not accept decision function for multiclass scenarios; it requires probabilities that sum to 1.
    This wrapper of LinearSVC adds a predict_proba function by:
    - softmaxing the logits from decision_function() in the multiclass case,
    - or sigmoiding the logits from decision_function() in the binary case, as in standard logistic regression.

    Important context for why these probabilities are uncalibrated: https://stats.stackexchange.com/questions/143152/why-do-one-versus-all-multi-class-svms-need-to-be-calibrated

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
        """Predicted probabilities by softmaxing decision function if multiclass (or sigmoiding if binary, like logistic regression). Output shape `(n_samples, n_classes)`.
        Note that in binary case, decision_function will have shape `(n_samples, )`, but predict_proba will still have shape `(n_samples, 2)`, per sklearn convention.
        """
        distances = self.decision_function(X)
        return genetools.stats.run_sigmoid_if_binary_and_softmax_if_multiclass(
            distances
        )
