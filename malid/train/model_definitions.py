import logging
from typing import Callable, List, Optional, Union

import numpy as np
import sklearn.model_selection
from malid.external.standard_scaler_that_preserves_input_type import (
    StandardScalerThatPreservesInputType,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, get_scorer, matthews_corrcoef, log_loss
from multiclass_metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from malid.datamodels import CVScorer, healthy_label
from malid.external.linear_svc_with_uncalibrated_probabilities import (
    LinearSVCWithUncalibratedProbabilities,
)
import choosegpu
from xgboost_label_encoding import XGBoostClassifierWithLabelEncoding
from wrap_glmnet import GlmnetLogitNetWrapper
from malid.train.one_vs_rest_except_negative_class_classifier import (
    OneVsRestClassifier,  # our custom version of sklearn's OvR wrapper
    OneVsRestExceptNegativeClassClassifier,
    OneVsNegativeClassifier,
)
import inspect

logger = logging.getLogger(__name__)

# Declaring this out here so we can modify for faster automated tests
DEFAULT_N_LAMBDAS_FOR_TUNING = 100


def _convert_glmnet_lambda_to_sklearn_C(lamb: float, n_train: int):
    """
    Convert a glmnet lambda to an equivalent sklearn C value.
    See _make_sklearn_linear_model docs.
    """
    return 1 / (lamb * n_train)


def _make_sklearn_linear_model(l1_ratio: float, C: float = 1.0, max_iter=10000):
    """Create sklearn lasso/elasticnet/ridge with a fixed default lambda (not cross validated)."""
    return LogisticRegression(
        # Note that sklearn LogisticRegression evaluates only one regularization penalty strength, which defaults to C=1
        # The use of C is for consistency with SVM conventions.
        # Here's how to convert C to lambda, the standard notation for lasso that's used by the glmnet models below:
        # lambda = 1 / (C * n), where n is the number of training examples.
        # Or to convert a glmnet lambda to an equivalent sklearn C value: C = 1 / (lambda * n).
        # Therefore the default C=1 here is equivalent to lambda=1/n in glmnet.
        C=C,
        #
        penalty="elasticnet",
        # multinomial objective with lasso (L1) or elasticnet penalty: only supported by "saga" solver.
        solver="saga",
        # multi_class="auto" means multinomial/softmax regression if the data is multiclass, or standard logistic regression if the data is binary.
        # (multi_class="multinomial" is not the same, because it requests a binary softmax classifier in binary cases instead of normal logistic regression.)
        multi_class="auto",
        # class_weight "balanced" accounts for unbalanced classes. see https://stackoverflow.com/a/30982811/130164
        # (these class weights will be multiplied with sample_weight if sample weights are passed through the fit method.)
        class_weight="balanced",
        random_state=0,
        # Set a high maximum number of iterations like 10000, otherwise the model may not converge, especially under poor choices of lambda.
        max_iter=max_iter,
        l1_ratio=l1_ratio,
    )


def _aggregate_predictions_to_patient_level(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate y_true and predicted class probabilities from the sequence level to the patient level.
    Patients are defined by the groups parameter, which is a vector of patient IDs."""
    unique_groups = np.unique(groups)

    # defensive casts
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # first y_true entry for each patient is saved as the patient's y_true
    aggregated_y_true = np.array([y_true[groups == g][0] for g in unique_groups])

    # take mean of all predictions for a patient, to get patient-level predicted class probabilities
    # use sample weights if provided
    aggregated_y_pred = np.vstack(
        [
            np.average(
                y_pred[groups == g],
                axis=0,
                weights=sample_weight[groups == g]
                if sample_weight is not None
                else None,
            )
            for g in unique_groups
        ]
    )

    return aggregated_y_true, aggregated_y_pred


class convert_to_patient_level_scorer:
    """
    Wrapper function to transform base_scorer into a patient-level scorer.

    Parameters:
    - base_scorer:
        * metric name recognized by sklearn,
        * functions produced by sklearn.metrics.make_scorer
        * custom scoring functions with signature score(y_true, y_predicted_probas), possibly with more parameters

    Caution: does not accept custom scoring functions with signature (estimator, X, y_true).

    Score functions should return a scalar score where greater is better.
    (If you pass a sklearn metric name string or a function produced by sklearn.metrics.make_scorer, we will automatically use the scorer's greater_is_better configuration.)

    Returns:
    - Function to compute patient-level metric. Requires groups parameter to be passed.

    Implementation note:
    Originally, this was:
        def convert_to_patient_level_scorer(base_scorer: Union[str, Callable]):
            def _get_score_func_and_sign(scorer) -> tuple[Callable, int]:
                ...
            def wrapper(
                estimator,
                X: np.ndarray,
                y_true: np.ndarray,
                groups: np.ndarray,
                sample_weight: Optional[np.ndarray] = None,
                *args,
                **kwargs,
            ):
                ...
            return wrapper
    However, Joblib can't pickle nested functions like wrapper() inside convert_to_patient_level_scorer():
    e.g. _pickle.PicklingError: Can't pickle <function convert_to_patient_level_scorer.<locals>.wrapper at 0x7f57b0798ee0>: it's not found as malid.train.model_definitions.convert_to_patient_level_scorer.<locals>.wrapper
    We followed the solution at https://stackoverflow.com/a/12022055/130164, creating a class with a __call__ method
    """

    def __init__(self, base_scorer: Union[str, Callable]):
        self.base_scorer = base_scorer

    # This wrapper follows conventional sklearn scorer signature (estimator, X, y_true): https://github.com/scikit-learn/scikit-learn/blob/2a2772a87b6c772dc3b8292bcffb990ce27515a8/sklearn/metrics/_scorer.py#L211
    # Unfortunately glmnet.scorer sometimes breaks that convention by renaming estimator to clf or y_true to y.
    def __call__(
        self,
        estimator,
        X: np.ndarray,
        y_true: np.ndarray,
        groups: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        (
            aggregated_y_true,
            aggregated_y_probas,
        ) = _aggregate_predictions_to_patient_level(
            y_true,
            estimator.predict_proba(X),
            groups,
            # If provided, use sample_weight when aggregating. Don't pass to group-level metric function.
            sample_weight=sample_weight,
        )

        def _get_score_func_and_sign(
            scorer: Union[str, Callable]
        ) -> tuple[Callable, int]:
            """Returns function with signature: score(y_true, y_predicted_probas), possibly with more parameters"""
            if isinstance(scorer, str):
                # Convert sklearn metric name strings to functions
                scorer = get_scorer(scorer)

            if hasattr(scorer, "_score_func"):
                # For functions produced by sklearn.metrics.make_scorer, extract inner score function and sign
                return scorer._score_func, scorer._sign

            # Custom functions: pass through and assume greater is better
            return scorer, 1

        # Arrive at a function with signature score(y_true, y_predicted_probas), possibly with more parameters
        score_func, sign = _get_score_func_and_sign(self.base_scorer)

        # Check if scorer accepts a 'groups' argument, a 'labels' argument, or arbitrary kwargs
        scorer_parameters = inspect.signature(score_func).parameters
        scorer_supports_labels = "labels" in scorer_parameters.keys()
        scorer_supports_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in scorer_parameters.values()
        )
        if labels is not None and (scorer_supports_labels or scorer_supports_kwargs):
            return sign * score_func(
                aggregated_y_true, aggregated_y_probas, labels=labels, *args, **kwargs
            )
        else:
            return sign * score_func(
                aggregated_y_true, aggregated_y_probas, *args, **kwargs
            )


# Extend glmnet_wrapper scorers
def glmnet_rocauc_scorer_at_patient_level(
    clf: GlmnetLogitNetWrapper,
    X: np.ndarray,
    y_true: np.ndarray,
    lamb: np.ndarray,
    groups: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
):

    # y_preds_proba shape is (n_samples, n_classes, n_lambdas)
    y_preds_proba = clf.predict_proba(X, lamb=lamb)

    # One score per lambda. Shape is (n_lambdas,)
    scores = np.empty(y_preds_proba.shape[2])

    for lambda_index in range(y_preds_proba.shape[2]):
        (
            aggregated_y_true,
            aggregated_y_probas,
        ) = _aggregate_predictions_to_patient_level(
            y_true, y_preds_proba[:, :, lambda_index], groups
        )
        scores[lambda_index] = roc_auc_score(
            y_true=aggregated_y_true,
            y_score=aggregated_y_probas,
            average="weighted",
            labels=clf.classes_,
            multi_class="ovo",
            sample_weight=sample_weight,
        )
    return scores


def glmnet_deviance_scorer_at_patient_level(
    clf: GlmnetLogitNetWrapper,
    X: np.ndarray,
    y_true: np.ndarray,
    lamb: np.ndarray,
    groups: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
):
    # y_preds_proba shape is (n_samples, n_classes, n_lambdas)
    # for binary log_loss, y_pred shape (n_samples,) is accepted too, but not required to extract positive class in this way.
    y_preds_proba = clf.predict_proba(X, lamb=lamb)

    # One score per lambda. Shape is (n_lambdas,)
    scores = np.empty(y_preds_proba.shape[2])

    for lambda_index in range(y_preds_proba.shape[2]):
        (
            aggregated_y_true,
            aggregated_y_probas,
        ) = _aggregate_predictions_to_patient_level(
            y_true, y_preds_proba[:, :, lambda_index], groups
        )
        scores[lambda_index] = log_loss(
            y_true=aggregated_y_true,
            y_pred=aggregated_y_probas,
            # provide labels explicitly to avoid error
            labels=clf.classes_,
            sample_weight=sample_weight,
        )
    # greater is worse; we want to minimize log loss
    return -1 * scores


def glmnet_matthews_corrcoef_scorer_at_patient_level(
    clf: GlmnetLogitNetWrapper,
    X: np.ndarray,
    y_true: np.ndarray,
    lamb: np.ndarray,
    groups: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
):
    # Get predicted probabilities with shape (n_samples, n_classes, n_lambdas)
    y_preds_proba = clf.predict_proba(X, lamb=lamb)

    # One score per lambda. Shape is (n_lambdas,)
    scores = np.empty(y_preds_proba.shape[2])
    for lambda_index in range(y_preds_proba.shape[2]):
        # matthews_corrcoef is a label based score, so we need to choose a winning label for each patient
        # Aggregate the predicted sequence probabilities to patient-level probabilities, with mean
        (
            aggregated_y_true,
            aggregated_y_probas,
        ) = _aggregate_predictions_to_patient_level(
            y_true, y_preds_proba[:, :, lambda_index], groups
        )

        # Then take argmax to arrive at patient-level predictions.
        predicted_label_indices = np.argmax(aggregated_y_probas, axis=1)
        # Map the argmax prediction indices to class labels using clf.classes_
        aggregated_y_pred = clf.classes_[predicted_label_indices]

        scores[lambda_index] = matthews_corrcoef(
            y_true=aggregated_y_true, y_pred=aggregated_y_pred
        )

    return scores


###


def make_models(
    # Set nested/internal cross validation parameters for hyperparameter tuning.
    # Cross-validation split generators are provided at model instantiation time,
    # while data-dependent parameters, such as `groups` (used to ensure that all sequences/samples from the same patient are in train or in test, even in internal/nested cross validation), are given at fit() time.
    #
    # Edge case: many sklearn "*CV" models take a range of hyperparameters (data-independent) at instantiation time, and then do a grid search over these hyperparameters,
    # even though other specialized models like glmnet treat these as data-dependent parameters to be computed at fit() time.
    # (For sklearn *CV models that don't take a `groups` parameter at fit time — e.g. GridSearch accepts at fit time, but LogisticRegressionCV does not —
    # we should make wrappers to accept the groups parameter at fit() time. Don't instantiate the model until then; see the glmnet wrapper as an example.)
    internal_cv: sklearn.model_selection.BaseCrossValidator,
    # Configure the scoring metric for the internal CV.
    cv_scoring=CVScorer.Deviance,
    chosen_models: Optional[List[str]] = None,
    n_jobs=5,
    n_lambdas_to_tune: Optional[int] = None,
):
    """
    Get models to fit.
    """
    # set global random seed - but not sure we need this, and this usage is deprecated.
    np.random.seed(0)

    # If user has not explicitly opted into GPU, disable GPU usage (relevant for XGBoost)
    choosegpu.configure_gpu(enable=False, overwrite_existing_configuration=False)

    # Configure the number of lambdas to tune for glmnet models
    if n_lambdas_to_tune is None:
        n_lambdas_to_tune = DEFAULT_N_LAMBDAS_FOR_TUNING

    CVScorer.validate(cv_scoring)
    if cv_scoring == CVScorer.AUC:
        # Pass a multiclass CV scorer for ROC-AUC
        # scoring: can't use predefined value "roc_auc" here: errors with "multiclass format is not supported"
        # instead pass a custom scorer object
        # Make cross validation ROC-AUC scorer to pass in arguments to roc_auc_score
        sklearn_cv_scorer = make_scorer(
            roc_auc_score, average="weighted", multi_class="ovo", needs_proba=True
        )
        glmnet_cv_scorer = GlmnetLogitNetWrapper.rocauc_scorer
        sklearn_cv_scorer_at_patient_level = convert_to_patient_level_scorer(
            sklearn_cv_scorer
        )
        glmnet_cv_scorer_at_patient_level = glmnet_rocauc_scorer_at_patient_level
    elif cv_scoring == CVScorer.MCC:
        sklearn_cv_scorer = "matthews_corrcoef"  # a string suffices
        import glmnet.scorer

        glmnet_cv_scorer = glmnet.scorer.make_scorer(matthews_corrcoef)
        sklearn_cv_scorer_at_patient_level = convert_to_patient_level_scorer(
            sklearn_cv_scorer
        )
        glmnet_cv_scorer_at_patient_level = (
            glmnet_matthews_corrcoef_scorer_at_patient_level
        )
    elif cv_scoring == CVScorer.Deviance:
        # Deviance is the default for R glmnet.
        glmnet_cv_scorer = GlmnetLogitNetWrapper.deviance_scorer
        # Minimizing the deviance is equivalent to minimizing the log loss.
        # But sklearn CV scorers are maximized, so we need to maximize the negative log loss.
        sklearn_cv_scorer = "neg_log_loss"
        sklearn_cv_scorer_at_patient_level = convert_to_patient_level_scorer(
            sklearn_cv_scorer
        )
        glmnet_cv_scorer_at_patient_level = glmnet_deviance_scorer_at_patient_level
    else:
        raise ValueError(f"Unknown cv_scoring argument: {cv_scoring}")

    def _make_glmnet_linear_model(
        l1_ratio: float, score_at_sequence_level: bool = True
    ):
        """Create glmnet lasso/elasticnet/ridge with data-dependent lambda (set through internal cross validation)."""
        # Use glmnet for data-dependent lambda path and other optimizations.
        # As in sklearn (e.g. GridSearchCV), we will define the CV strategy (a split generator function, essentially) upfront,
        # and then provide the relevant data at fit() time.
        # Note that we should pass groups parameter to .fit() to ensure that for the nested cross-validation,
        # each patient's entire set of sequences is always in either the training set or the test set, but never in both in the same fold.
        # By nested cross validation, we mean that this is subdividing whatever training set is used into a further set of folds.

        # TODO(later): automatically tune the elasticnet L1/L2 ratio, over the range 0, 0.1, 0.2, ..., 1.0.
        # To do this, run cv.glmnet repeatedly with the different L1/L2 ratios (cv.glmnet only tunes lambda; it does not auto-tune the alpha ratio).
        # Each cv.glmnet call should return the average AUC over its internal CV folds. Keep the alpha that works best.
        # This means we’ll arrive at a different alpha for each (outer) fold, but that’s ok; we’ll also have different lambdas.
        # Group all these models together (i.e. the best one from each outer fold) under the single name “elasticnet_cv”.
        # Then bring similar tuning to the other models in this file.
        return GlmnetLogitNetWrapper(
            alpha=l1_ratio,  # lasso is 1, ridge is 0
            # specify cross validation fold generator.
            # the one we pass is going to be executed at fit() time: it will stratify on y, and also account for groups.
            # notice that the data-dependent y and groups parameters for designing the internal folds are passed at .fit() time,
            # while the split strategy function (including the number of splits) is provided up-front here (consistent with sklearn)
            internal_cv=internal_cv,
            # the alternative would be to leave internal_cv unset,
            # and instead just specify a number of internal cross validation splits for glmnet to create with its default strategy:
            # n_splits=3,
            n_lambda=n_lambdas_to_tune,
            # Standardize outside Glmnet, to be consistent with other models (all mdoels assume that the input data has been standardized before hitting the model).
            # Also this prevents LogitNet from transforming coefficients back to the non-standardized raw scale.
            standardize=False,
            scoring=glmnet_cv_scorer
            if score_at_sequence_level
            else glmnet_cv_scorer_at_patient_level,
            n_jobs=n_jobs,
            verbose=True,
            # random_state only used for the nested CV, not for the actual solving
            random_state=0,
            # Use the lambda with the best CV score, rather than a slightly simpler model with a lambda whose performance is within 1 standard error of the best lambda.
            # The more parsimonious model (lambda_1se=True) is a good choice when focused on interpretation, but for all the base models, we prioritize prediction over interpretation.
            # We would expect lambda_max to generalize better than lambda_1se, because it retains more variables so we have more robustness in a sense (not hanging our hat entirely on a small set of variables).
            # For the final metamodel, we can report and interpret a lambda_1se version too.
            # (In fact, we have logic in training_utils.modify_fitted_model_lambda_setting() that copies each fitted GlmnetLogitNetWrapper and toggles its lambda_1se setting, so we can report both without refitting.)
            use_lambda_1se=False,
            # extra check to ensure that the nested CV is not using the same patient's sequences in both the training and test sets
            require_cv_group_labels=True,
            # class_weight "balanced" accounts for unbalanced classes. see https://stackoverflow.com/a/30982811/130164
            # (these class weights will be multiplied with sample_weight if sample weights are passed through the fit method.)
            class_weight="balanced",
        )

    # NOTE: Models assume that the input data has been standardized (mean 0, variance 1). (Calling code may wrap the model in a Pipeline with a scaler.)
    models = {
        # Most frequent strategy for dummy: always predict mode
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent", random_state=0
        ),
        # Stratified strategy for dummy: predict according to source distribution
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=0),
        #
        # We switched our linear models to use glmnet instead of scikit-learn. Here's why we no longer use sklearn linear models:
        # - They have a default fixed lambda hyperparameter, as opposed to a data-dependent lambda hyperparameter.
        # - They don't have smart logic for tuning the lambda hyperparameter with internal cross validation, whereas glmnet has clever tricks.
        # - The default sklearn lambda, 1/n_train, is not guaranteed to give good performance, or even to have the model fit converge.
        # In our experience, Glmnet tries the sklearn default lambda and considers it alongside other possible lambdas.
        # (Exception: When n_train is very big, Glmnet may skip the sklearn default lambda, and we saw convergence issues in sklearn even after bumping the number of iterations for sklearn's solver.)
        #
        # However, if you already know the optimal lambda and only fit one model with that optimal configuration, sklearn with the saga solver is a bit faster than glmnet.
        #
        # Glmnet linear models with data-dependent lambda hyperparameter, tuned using internal cross validation:
        "lasso_cv": _make_glmnet_linear_model(l1_ratio=1),
        "elasticnet_cv0.75": _make_glmnet_linear_model(l1_ratio=0.75),
        "elasticnet_cv": _make_glmnet_linear_model(l1_ratio=0.5),
        "elasticnet_cv0.25": _make_glmnet_linear_model(l1_ratio=0.25),
        "ridge_cv": _make_glmnet_linear_model(l1_ratio=0),
        #
        # Glmnet linear models with lambda tuned on a patient-level metric:
        # Caution: this is patient-level, not specimen-level, because our group variable is "participant_label"
        "lasso_cv_patient_level_optimization": _make_glmnet_linear_model(
            l1_ratio=1, score_at_sequence_level=False
        ),
        "elasticnet_cv0.75_patient_level_optimization": _make_glmnet_linear_model(
            l1_ratio=0.75, score_at_sequence_level=False
        ),
        "elasticnet_cv_patient_level_optimization": _make_glmnet_linear_model(
            l1_ratio=0.5, score_at_sequence_level=False
        ),
        "elasticnet_cv0.25_patient_level_optimization": _make_glmnet_linear_model(
            l1_ratio=0.25, score_at_sequence_level=False
        ),
        "ridge_cv_patient_level_optimization": _make_glmnet_linear_model(
            l1_ratio=0, score_at_sequence_level=False
        ),
        #
        # Also a sklearn unregularized logistic regression, for comparison:
        "logisticregression_unregularized": LogisticRegression(
            penalty=None,
            solver="saga",
            # multi_class="auto" means multinomial/softmax regression if the data is multiclass, or standard logistic regression if the data is binary.
            # (multi_class="multinomial" is not the same, because it requests a binary softmax classifier in binary cases instead of normal logistic regression.)
            multi_class="auto",
            # class_weight "balanced" accounts for unbalanced classes. see https://stackoverflow.com/a/30982811/130164
            # (these class weights will be multiplied with sample_weight if sample weights are passed through the fit method.)
            class_weight="balanced",
            random_state=0,
            # Set high maximum number of iterations, otherwise the model may not converge.
            max_iter=10000,
        ),
        #
        # Instead of multinomial formulation, train separate binomial one-vs-rest models.
        # The goal is to decouple the classes: At the sequence level, we don't want to predict that a sequence is "disease A" based on how much it looks or doesn't look like disease B or disease C.
        # We will thread this through to the aggregation level, where we will not allow feature leakage like using aggregated P(sequence is HIV) features to predict P(patient has Covid).
        # Use GlmnetLogitNetWrapper as base estimator.
        "elasticnet_cv_ovr": OneVsRestClassifier(
            estimator=_make_glmnet_linear_model(l1_ratio=0.5),
            normalize_predicted_probabilities=False,
            n_jobs=n_jobs,
            # Allow some classes to fail to train, e.g. if they have too few samples and therefore internal cross validation for some binary OvR problems fails.
            allow_some_classes_to_fail_to_train=True,
        ),
        "elasticnet_cv0.75_ovr": OneVsRestClassifier(
            estimator=_make_glmnet_linear_model(l1_ratio=0.75),
            normalize_predicted_probabilities=False,
            n_jobs=n_jobs,
            allow_some_classes_to_fail_to_train=True,
        ),
        "elasticnet_cv0.25_ovr": OneVsRestClassifier(
            estimator=_make_glmnet_linear_model(l1_ratio=0.25),
            normalize_predicted_probabilities=False,
            n_jobs=n_jobs,
            allow_some_classes_to_fail_to_train=True,
        ),
        "lasso_cv_ovr": OneVsRestClassifier(
            estimator=_make_glmnet_linear_model(l1_ratio=1.0),
            normalize_predicted_probabilities=False,
            n_jobs=n_jobs,
            allow_some_classes_to_fail_to_train=True,
        ),
        "ridge_cv_ovr": OneVsRestClassifier(
            estimator=_make_glmnet_linear_model(l1_ratio=0.0),
            normalize_predicted_probabilities=False,
            n_jobs=n_jobs,
            allow_some_classes_to_fail_to_train=True,
        ),
        #
        "rf_multiclass": RandomForestClassifier(
            n_jobs=min(2, n_jobs),
            random_state=0,
            # Balanced class weights computed based on the bootstrap sample for every tree grown:
            class_weight="balanced_subsample",
        ),
        # XGBoost wrapped in label encoding (native XGBoost doesn't support string labels)
        # Note that this may attempt to use GPU, so user should call `choosegpu.configure_gpu(enable=False)`. Above we default to CPU if user has not explicitly opted into GPU use.
        "xgboost": XGBoostClassifierWithLabelEncoding(
            n_estimators=100,
            n_jobs=n_jobs,
            random_state=0,
            seed=0,
            class_weight="balanced",
            # TODO: set eval_metric?
        ),
        "linearsvm_ovr": LinearSVCWithUncalibratedProbabilities(  # replacement for svm.LinearSVC
            dual=False,
            multi_class="ovr",
            random_state=0,
            class_weight="balanced",
            # Set high maximum number of iterations, otherwise the model may not converge.
            max_iter=10000,
        ),
    }

    if not chosen_models:
        chosen_models = list(models.keys())

    # filter to selected models
    return {k: v for k, v in models.items() if k in chosen_models}
