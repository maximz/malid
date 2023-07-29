from typing import List, Union, Optional

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import matthews_corrcoef

from extendanything import ExtendAnything


class AdjustedProbabilitiesDerivedModel(ExtendAnything, ClassifierMixin, BaseEstimator):
    """Wrapper around any multiclass classification model to adjust decision thresholds that determine predicted labels.
    The output of predict() and predict_proba() are different:
    - predict_proba() probabilities are rescaled and may no longer sum to 1.
    - predict() is the argmax of the rescaled predict_proba() probabilities.

    Applying the class reweighting operation to the predicted probabilities does not affect the final ROC AUC score:
    the ROC AUC stays the same after multiplying the class probabilities by their weights.

    But the ROC AUC can change if you then renormalize the probabilities to sum to 1 again,
    because the rank orderings between examples may change.
    Don't renormalize.

    Note: sklearn's built-in ROC AUC score does not accept multiclass probabilities that don't sum to 1.
    Use model_evaluation roc_auc_score instead.

    The intuition for how this adjusts decision thresholds between classes:
    By downweighing a class by dividing its probabilities by a factor of n, it now has to be n times as strong to outrank its competitor classes.
    For each example: Apply these class-specific factors, then choose dominant class probability as the winning label.

    Based on:
    - https://stats.stackexchange.com/a/310956/297
    - https://stats.stackexchange.com/a/449903/297
    """

    # TODO: Can we make AdjustedProbabilitiesDerivedModel be a generic type that accepts a type T and shadows all T’s type params?

    # ExtendAnything means will pass through to base instance's attributes.
    # Used to map e.g. classes_ -> inner_clf.classes_ and predict_proba -> inner_clf.predict_proba.

    # TODO: is there a type hint for all sklearn models?
    # BaseEstimator is odd to put here: https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
    # Preferably something that requires predict_proba() to exist.
    def __init__(
        self, inner_clf: BaseEstimator, class_weights: Union[np.ndarray, List[float]]
    ) -> None:
        # sets self._inner
        super().__init__(inner_clf)
        # also make it available as self.inner_clf so that sklearn clone works
        self.inner_clf = inner_clf

        self.class_weights = class_weights

    @staticmethod
    def _compute_adjusted_class_probabilities(
        predict_probas: np.ndarray,
        class_weights: Union[np.ndarray, List[float]],
    ) -> np.ndarray:
        # Element-wise multiply - i.e. reweight the classes
        # Don't renormalize each row to sum to 1.
        return np.multiply(predict_probas, class_weights)

    def predict_proba(self, X) -> np.ndarray:
        """
        Reweight predicted class probability vectors.
        """
        return self._compute_adjusted_class_probabilities(
            self._inner.predict_proba(X), self.class_weights
        )

    @staticmethod
    def _choose_winning_label(
        classes: Union[List[str], np.ndarray], adjusted_probabilities: np.ndarray
    ) -> Union[str, np.ndarray]:
        # Argmax along last axis,
        # which is axis 1 for a 2D array and axis 0 for a 1D array.

        # This handles edge case when adjusted_probailities is a 1d array, i.e. X had only one example.
        # Don't want to convert to 2d array because in this case.
        # Return a scalar (this is the str in the union type), rather than an array with one entry.

        # Defensive casting:
        return np.array(classes)[np.argmax(np.array(adjusted_probabilities), axis=-1)]

    def predict(self, X) -> Union[str, np.ndarray]:
        """Reweight predicted class probability vectors to choose predicted class label for each example."""
        return self._choose_winning_label(self.classes_, self.predict_proba(X))

    @staticmethod
    def _get_weights_to_optimize_decision_threshold(
        predicted_probabilities_validation: np.ndarray,
        classes: Union[List[str], np.ndarray],
        y_validation_true: np.ndarray,
        score_func=matthews_corrcoef,
    ) -> np.ndarray:
        """
        Given predict_proba(X) and y_true, find class weights to maximize accuracy (or another score) of choosing predicted labels y_pred.

        Usage: train a model on a training set, tune its decision thresholds on a validation set with this method, then apply to a test set.

        score_func must accept these parameters: score_func(y_true, y_pred)
        """

        def _score(class_weights, probabilities, y_true, classes):
            # We could follow the standard usage of AdjustedProbabilitiesDerivedModel as follows:
            # adjusted_model = AdjustedProbabilitiesDerivedModel(model, class_weights)
            # y_pred = adjusted_model.predict(X)

            # But for efficiency we only run predict_proba() once and store the result, which will not change as we modify class weights.
            # That's provided in `probabilities` argument.

            y_pred = AdjustedProbabilitiesDerivedModel._choose_winning_label(
                classes,
                AdjustedProbabilitiesDerivedModel._compute_adjusted_class_probabilities(
                    probabilities, class_weights
                ),
            )

            # Compute and return negative since we are minimizing.
            score = score_func(y_true, y_pred)
            return -1.0 * score

        # Technically, the only constraint we need is >0,
        # so that adjusted probabilities are >0 and normalizing rows by sum works right.
        # But to constrain the search, we'll use the range 0 to 1. (Consider a 1-100 range instead?)
        bounds = optimize.Bounds([1e-5] * len(classes), [1] * len(classes))

        # class weights will be supplied as first argument. args are additional arguments
        result = optimize.differential_evolution(
            _score,
            bounds,
            args=(
                np.array(predicted_probabilities_validation),
                np.array(y_validation_true),
                classes,
            ),
        )
        best_class_weights = result.x
        return best_class_weights

    @classmethod
    def adjust_model_decision_thresholds(
        cls,
        model: BaseEstimator,
        y_validation_true: np.ndarray,
        score_func=matthews_corrcoef,
        X_validation: Optional[np.ndarray] = None,
        predicted_probabilities_validation: Optional[np.ndarray] = None,
    ) -> "AdjustedProbabilitiesDerivedModel":
        """
        Tune a multiclass model's decision thresholds against a validation set:
        Tune class weights to achieve higher classification performance with this set of predicted class probabilities

        Provide X_validation or predicted_probabilities_validation.
        """
        if X_validation is None and predicted_probabilities_validation is None:
            raise ValueError(
                "Must provide X_validation or predicted_probabilities_validation."
            )

        if predicted_probabilities_validation is None:
            # Get predict_probas once, then optimize on them; don't call predict_proba over and over again.
            predicted_probabilities_validation = model.predict_proba(X_validation)
            # Optionally reorder if we have Pandas objects
            if hasattr(y_validation_true, "index"):
                if hasattr(predicted_probabilities_validation, "index"):
                    # This takes precedence
                    y_validation_true = y_validation_true.loc[
                        predicted_probabilities_validation.index
                    ]
                elif hasattr(X_validation, "index"):
                    y_validation_true = y_validation_true.loc[X_validation.index]

        best_class_weights = cls._get_weights_to_optimize_decision_threshold(
            predicted_probabilities_validation=predicted_probabilities_validation,
            classes=model.classes_,
            y_validation_true=y_validation_true,
            score_func=score_func,
        )
        # Return new model (wrapper)
        return cls(model, best_class_weights)
