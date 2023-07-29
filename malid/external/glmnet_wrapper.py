from typing import Optional, Union
import numpy as np
from typing_extensions import Self
import glmnet
from malid.external.model_evaluation_scores import roc_auc_score
from extendanything import ExtendAnything
from sklearn.utils.validation import _get_feature_names
import sklearn.model_selection
import sklearn.utils.class_weight
import functools
import logging

logger = logging.getLogger(__name__)


def apply_glmnet_wrapper():
    """
    Replace a glmnet internal function with a wrapper, so that we can override internal CV.

    python-glmnet forces use of their own cross validation splitters.
    But we can inject our own right before _score_lambda_path is called (the function that actually uses the splitter).
    See https://github.com/civisanalytics/python-glmnet/blob/813c06f5fcc9604d8e445bd4992f53c4855cc7cb/glmnet/logistic.py#L245
    """

    def wrap_with_cv_update(original_method):
        @functools.wraps(original_method)
        def wrapped_method_that_replaces_original_method(est, *args, **kwargs):
            # Wrapped function is called f"{original_method.__name__}"
            # est is the estimator instance

            # Before calling _score_lambda_path, glmnet sets est._cv.
            # We will modify est._cv if est._cv_override is set and available.
            # Pass through unmodified otherwise.
            if hasattr(est, "_cv_override") and est._cv_override is not None:
                est._cv = est._cv_override

            # Proceed with original method
            return original_method(est, *args, **kwargs)

        return wrapped_method_that_replaces_original_method

    # Hot swap the function
    import glmnet.util

    wrapped_func = wrap_with_cv_update(glmnet.util._score_lambda_path)
    glmnet.util._score_lambda_path = wrapped_func
    # It's likely it has already been imported by glmnet.logistic, because glmnet.__init__ import glmnet.logistic,
    # so we need to hot swap the imported version too.
    import glmnet.logistic

    glmnet.logistic._score_lambda_path = wrapped_func


# Do this replacement at import time.
apply_glmnet_wrapper()


class GlmnetLogitNetWrapper(ExtendAnything):
    """
    Wrapper around python-glmnet's LogitNet that exposes some additional features:

    - standard sklearn API properties
    - control over choice of lambda
    - multiclass ROC-AUC scorer for internal cross validation
    - automatic class weight rebalancing as in sklearn

    Use this wrapper in place of python-glmnet's LogitNet.
    """

    # ExtendAnything passes everything else on to _inner automatically

    # TODO: support balanced class weights.
    # TODO: support specifying exact cross validation splits, not just the number and the group labels.

    @staticmethod
    def rocauc_scorer(
        clf,
        X: np.ndarray,
        y_true: np.ndarray,
        lamb: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Multiclass ROC-AUC scorer for LogitNet's internal cross validation.
        To use, pass `scoring=GlmnetLogitNetWrapper.rocauc_scorer` to the model constructor.
        """
        # Make a multiclass CV scorer for ROC-AUC for glmnet models.
        # `scoring="roc_auc"`` doesn't suffice: multiclass not supported.

        # We have the same problem for sklearn models, but we can't repurpose sklearn's make_scorer output here.
        # The scorer function here has a different signature than sklearn's make_scorer output function: lambdas are passed to the scorer.
        # Specifically, the scoring function will be called with arguments: `(clf, X[test_inx, :], y[test_inx], lamb)`, where `lamb` is an array.

        # Unfortunately, glmnet.scorer.make_scorer is not sufficient, because it does not set the "labels" parameter used in roc_auc_score.
        # This does not work:
        # rocauc_scorer = glmnet.scorer.make_scorer(
        #     roc_auc_score,
        #     average="weighted",
        #     multi_class="ovo",
        #     needs_proba=True,
        # )

        # Instead we roll our own.

        # y_preds_proba shape is (n_samples, n_classes, n_lambdas)
        y_preds_proba = clf.predict_proba(X, lamb=lamb)

        # One score per lambda. Shape is (n_lambdas,)
        scores = np.array(
            [
                roc_auc_score(
                    y_true=y_true,
                    y_score=y_preds_proba[:, :, lambda_index],
                    average="weighted",
                    labels=clf.classes_,
                    multi_class="ovo",
                    sample_weight=sample_weight,
                )
                for lambda_index in range(y_preds_proba.shape[2])
            ]
        )
        return scores

    def __init__(
        self,
        use_lambda_1se=True,
        require_cv_group_labels=False,
        n_splits: Optional[int] = 3,
        internal_cv: Optional[sklearn.model_selection.BaseCrossValidator] = None,
        class_weight: Optional[Union[dict, str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a LogitNet model. All kwargs are passed to LogitNet.

        Extra arguments:
        - `use_lambda_1se` determines which lambda is used for prediction, coef_, and intercept_:
            If True (default), use lambda_1se (larger lambda but performance still within 1 standard error).
            This is the default behavior of LogitNet.
            This lambda and its index are available as properties lambda_best_ and lambda_best_inx_.

            If False, use the lambda value that achieved highest cross validation performance for prediction.
            This lambda and its index are available as properties lambda_max_ and lambda_max_inx_.

            Pass `use_lambda_1se=False` to switch LogitNet's default behavior to use the lambda with highest CV performance,
            rather than use a simpler model that still performs within 1 standard error of the best observed performance.

        - `require_cv_group_labels` (disabled by default) determines whether to require groups parameter in fit() call.
            Glmnet is often used with internal cross-validation.
            If `require_cv_group_labels` is True, then the `groups` parameter must be passed to fit(), otherwise an error is thrown.
            This adds a safety net to make sure that the user is aware that they are using internal cross-validation
            and that the internal cross-validation is performing the correct splits.

        - `class_weight`: dict or 'balanced' or None, defaults to 'balanced'.
            Behaves just like class_weight in sklearn models, see e.g. LogisticRegression.

        - `internal_cv`: an optional sklearn-style cross validation split generator like KFold, StratifiedKFold, etc.
            Optional override of glmnet's default cross validation split generator.
            Note that specifying internal_cv will cause the n_splits argument to be overwritten with the value of `internal_cv.get_n_splits()`.
            If internal_cv is not specified, then n_splits will be used with glmnet's default CV split strategy. Either n_splits or internal_cv must be specified.
        """
        # for sklearn clone compatibility, in the constructor we should only set these variables, and not do any modifications yet
        self.use_lambda_1se = use_lambda_1se
        self.require_cv_group_labels = require_cv_group_labels
        self.class_weight = class_weight
        self.internal_cv = internal_cv
        self.n_splits = (
            n_splits  # may be null for now - we will fill later from internal_cv
        )

        if self.n_splits is None and self.internal_cv is None:
            raise ValueError("Either n_splits or internal_cv must be specified.")

        # sets self._inner
        super().__init__(
            glmnet.LogitNet(
                # default if not provided. we will override later with the correct value from internal_cv
                n_splits=n_splits if n_splits is not None else 3,
                **kwargs,
            )
        )

    ######
    # Fix some inconsistencies in the python-glmnet API
    # (Note: because of how ExtendAnything works, these only affect outside users of the wrapper.
    # Glmnet internal code will still see Glmnet's internal values for these properties, not our modified version.)

    @property
    def lambda_best_(self) -> float:
        # lambda_max_ is a scalar, but lambda_best_ is wrapped in an array for some reason
        # unwrap it
        return self._inner.lambda_best_.item(0)

    @property
    def lambda_best_inx_(self) -> float:
        # lambda_max_inx_ is a scalar, but lambda_best_inx_ is wrapped in an array for some reason
        # unwrap it
        return self._inner.lambda_best_inx_.item(0)

    ######
    # Internal properties to determine which lambda to use for prediction, based on self.use_lambda_1se setting

    @property
    def _lambda_for_prediction_(self) -> float:
        if self.use_lambda_1se:
            return self.lambda_best_
        else:
            return self.lambda_max_

    @property
    def _lambda_inx_for_prediction_(self) -> float:
        if self.use_lambda_1se:
            return self.lambda_best_inx_
        else:
            return self.lambda_max_inx_

    ######

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Self:
        # just in case, make sure the glmnet wrapper has been applied
        # (consider checking if already wrapped first, so we don't wrap an already-wrapped function?)
        apply_glmnet_wrapper()

        if self.internal_cv is not None:
            # Set a new attribute on the inner glmnet.LogitNet object.
            # This will get picked up by our wrapped version of _score_lambda_path.
            # See https://github.com/civisanalytics/python-glmnet/blob/813c06f5fcc9604d8e445bd4992f53c4855cc7cb/glmnet/logistic.py#L245
            self._inner._cv_override = self.internal_cv

            # Fill wrapper's n_splits from internal_cv
            self.n_splits = self.internal_cv.get_n_splits()

            # Check n_splits, then fill wrapped object's n_splits accordingly.
            if self.n_splits < 3:
                logger.warning(
                    f"Cross-validation strategy only performs {self.n_splits} splits. Python-Glmnet will not perform cross validation unless n_splits >= 3. We are setting the wrapped glmnet object's n_splits=3 to avoid this issue."
                )
                self._inner.n_splits = 3
            else:
                self._inner.n_splits = self.n_splits

        if groups is None and self.require_cv_group_labels:
            raise ValueError(
                "GlmnetLogitNetWrapper requires groups parameter in fit() call because require_cv_group_labels was set to True."
            )

        if self.class_weight is not None:
            # Use sklearn to compute class weights, then map to individual sample weights
            sample_weight_computed = sklearn.utils.class_weight.compute_sample_weight(
                class_weight=self.class_weight, y=y
            )
            if sample_weight is None:
                # No sample weights were provided. Just use the ones derived from class weights.
                sample_weight = sample_weight_computed
            else:
                # Sample weights were already provided. We need to combine with class-derived weights.
                # First, confirm shape matches
                if sample_weight.shape[0] != sample_weight_computed.shape[0]:
                    raise ValueError(
                        "Provided sample_weight has different number of samples than y."
                    )
                # Then, multiply the two
                sample_weight = sample_weight * sample_weight_computed

        # Fit as usual
        self._inner = self._inner.fit(
            X=X, y=y, sample_weight=sample_weight, groups=groups, **kwargs
        )

        # Add properties to be compatible with sklearn API
        self.n_features_in_ = X.shape[1]

        feature_names = _get_feature_names(X)
        # If previously fitted, delete attribute
        if hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")
        # Set new attribute if feature names are available (otherwise leave unset)
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        return self

    ######
    # Allow user to choose which lambda to use as default

    def predict(
        self, X: np.ndarray, lamb: Optional[Union[float, np.ndarray]] = None
    ) -> np.ndarray:
        return self._inner.predict(
            X, lamb=self._lambda_for_prediction_ if lamb is None else lamb
        )

    def predict_proba(
        self, X: np.ndarray, lamb: Optional[Union[float, np.ndarray]] = None
    ) -> np.ndarray:
        return self._inner.predict_proba(
            X, lamb=self._lambda_for_prediction_ if lamb is None else lamb
        )

    def decision_function(
        self, X: np.ndarray, lamb: Optional[Union[float, np.ndarray]] = None
    ) -> np.ndarray:
        return self._inner.decision_function(
            X, lamb=self._lambda_for_prediction_ if lamb is None else lamb
        )

    # TODO: in predict(), decision_function(), predict_proba(), check feature names match those in fit()?
    # https://github.com/scikit-learn/scikit-learn/blob/d52e946fa4fca4282b0065ddcb0dd5d268c956e7/sklearn/base.py#L364

    @property
    def coef_(self) -> np.ndarray:
        return self._inner.coef_path_[:, :, self._lambda_inx_for_prediction_]

    @property
    def intercept_(self) -> np.ndarray:
        return self._inner.intercept_path_[:, self._lambda_inx_for_prediction_]

    ######
    # Expose new properties based on the best lambda

    @property
    def cv_mean_score_final_(self) -> float:
        return self._inner.cv_mean_score_[self._lambda_inx_for_prediction_]

    @property
    def cv_standard_error_final_(self) -> float:
        return self._inner.cv_standard_error_[self._lambda_inx_for_prediction_]

    ######
    # Add analysis methods
    def plot_cross_validation_curve(self, scorer_name: str):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.errorbar(
            np.log(self.lambda_path_), self.cv_mean_score_, yerr=self.cv_standard_error_
        )
        plt.axvline(np.log(self.lambda_best_), color="k")
        plt.axvline(np.log(self.lambda_max_), color="r")
        plt.xlabel("log(lambda)")
        plt.ylabel(scorer_name)
        return fig
