import logging
from typing import List, Optional

import numpy as np
import sklearn.base
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from malid.external.model_evaluation_scores import roc_auc_score
from sklearn.pipeline import make_pipeline

from malid.datamodels import CVScorer
from malid.external.linear_svc_softmax_probabilities import (
    LinearSVCSoftmaxProbabilities,
)
import choosegpu
from xgboost_label_encoding import XGBoostClassifierWithLabelEncoding
from malid.external.glmnet_wrapper import GlmnetLogitNetWrapper

logger = logging.getLogger(__name__)

# Declaring this out here so we can modify for faster automated tests
DEFAULT_N_LAMBDAS_FOR_TUNING = 100


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
    cv_scoring=CVScorer.AUC,
    chosen_models: Optional[List[str]] = None,
    n_jobs=5,
):
    """
    Get models to fit.
    """
    # set global random seed - but not sure we need this, and this usage is deprecated.
    np.random.seed(0)

    # If user has not explicitly opted into GPU, disable GPU usage (relevant for XGBoost)
    choosegpu.configure_gpu(enable=False, overwrite_existing_configuration=False)

    CVScorer.validate(cv_scoring)
    if cv_scoring == CVScorer.AUC:
        # Pass a multiclass CV scorer for ROC-AUC
        # scoring: can't use predefined value "roc_auc" here: errors with "multiclass format is not supported"
        # instead pass a custom scorer object
        # Make cross validation ROC-AUC scorer to pass in arguments to roc_auc_score
        rocauc_multiclass_cv_scorer = make_scorer(
            roc_auc_score, average="weighted", multi_class="ovo", needs_proba=True
        )
        sklearn_cv_scorer = rocauc_multiclass_cv_scorer
        glmnet_cv_scorer = GlmnetLogitNetWrapper.rocauc_scorer
    elif cv_scoring == CVScorer.MCC:
        sklearn_cv_scorer = "matthews_corrcoef"  # a string suffices
        import glmnet.scorer

        glmnet_cv_scorer = glmnet.scorer.make_scorer(matthews_corrcoef)
    else:
        raise ValueError(f"Unknown cv_scoring argument: {cv_scoring}")

    models = {
        # Most frequent strategy for dummy: always predict mode
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent", random_state=0
        ),
        # Stratified strategy for dummy: predict according to source distribution
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=0),
        "lasso_multiclass": LogisticRegression(
            # Note that sklearn LogisticRegression evaluates only one regularization penalty strength, which defaults to C=1
            # The use of C is for consistency with SVM conventions.
            # Here's how to convert C to lambda, the standard notation for lasso that's used by the glmnet models below:
            # lambda = 1 / (C * n), where n is the number of training examples.
            # Or to convert a glmnet lambda to an equivalent sklearn C value: C = 1 / (lambda * n).
            # Therefore the default C=1 here is equivalent to lambda=1/n in glmnet.
            penalty="l1",
            # multinomial objective with lasso (L1) penalty: only "saga" solver supported
            solver="saga",
            multi_class="multinomial",
            # class_weight "balanced" accounts for unbalanced classes. see https://stackoverflow.com/a/30982811/130164
            class_weight="balanced",
            random_state=0,
        ),
        "lasso_cv": make_pipeline(
            # Force scaling
            StandardScaler(),
            # Use glmnet for data-dependent lambda path and other optimizations.
            # As in sklearn (e.g. GridSearchCV), we will define the CV strategy (a split generator function, essentially) upfront,
            # and then provide the relevant data at fit() time.
            # Note that we should pass groups parameter to .fit() to ensure that for the nested cross-validation,
            # each patient's entire set of sequences is always in either the training set or the test set, but never in both in the same fold.
            # By nested cross validation, we mean that this is subdividing whatever training set is used into a further set of folds.
            GlmnetLogitNetWrapper(
                alpha=1,  # lasso
                # specify cross validation fold generator.
                # the one we pass is going to be executed at fit() time: it will stratify on y, and also account for groups.
                # notice that the data-dependent y and groups parameters for designing the internal folds are passed at .fit() time,
                # while the split strategy function (including the number of splits) is provided up-front here (consistent with sklearn)
                internal_cv=internal_cv,
                # the alternative would be to leave internal_cv unset,
                # and instead just specify a number of internal cross validation splits for glmnet to create with its default strategy:
                # n_splits=3,
                n_lambda=DEFAULT_N_LAMBDAS_FOR_TUNING,
                # standardize outside in the pipeline, to be consistent with other models
                # also this prevents LogitNet from transforming coefficients back to the non-standardized raw scale
                standardize=False,
                scoring=glmnet_cv_scorer,
                n_jobs=n_jobs,
                verbose=True,
                # random_state only used for the nested CV, not for the actual solving
                random_state=0,
                # use slightly simpler model than the lambda with the best CV score
                use_lambda_1se=True,
                # extra check to ensure that the nested CV is not using the same patient's sequences in both the training and test sets
                require_cv_group_labels=True,
                class_weight="balanced",
            ),
        ),
        "ridge_cv": make_pipeline(
            # Force scaling
            StandardScaler(),
            # Use glmnet for data-dependent lambda path and other optimizations
            # Note that we should pass groups parameter to .fit() to ensure that for the nested cross-validation,
            # each patient's entire set of sequences is always in either the training set or the test set, but never in both in the same fold.
            # By nested cross validation, we mean that this is subdividing whatever training set is used into a further set of folds.
            GlmnetLogitNetWrapper(
                alpha=0,  # ridge
                # specify cross validation fold generator.
                # the one we pass is going to be executed at fit() time: it will stratify on y, and also account for groups.
                # notice that the data-dependent y and groups parameters for designing the internal folds are passed at .fit() time,
                # while the split strategy function (including the number of splits) is provided up-front here (consistent with sklearn)
                internal_cv=internal_cv,
                # the alternative would be to leave internal_cv unset,
                # and instead just specify a number of internal cross validation splits for glmnet to create with its default strategy:
                # n_splits=3,
                n_lambda=DEFAULT_N_LAMBDAS_FOR_TUNING,
                # standardize outside in the pipeline, to be consistent with other models
                # also this prevents LogitNet from transforming coefficients back to the non-standardized raw scale
                standardize=False,
                scoring=glmnet_cv_scorer,
                n_jobs=n_jobs,
                verbose=True,
                # random_state only used for the nested CV, not for the actual solving
                random_state=0,
                # use slightly simpler model than the lambda with the best CV score
                use_lambda_1se=True,
                # extra check to ensure that the nested CV is not using the same patient's sequences in both the training and test sets
                require_cv_group_labels=True,
                class_weight="balanced",
            ),
        ),
        "elasticnet_cv": make_pipeline(
            # Force scaling
            StandardScaler(),
            # Use glmnet for data-dependent lambda path and other optimizations
            # Note that we should pass groups parameter to .fit() to ensure that for the nested cross-validation,
            # each patient's entire set of sequences is always in either the training set or the test set, but never in both in the same fold.
            # By nested cross validation, we mean that this is subdividing whatever training set is used into a further set of folds.
            GlmnetLogitNetWrapper(
                alpha=0.5,
                # specify cross validation fold generator.
                # the one we pass is going to be executed at fit() time: it will stratify on y, and also account for groups.
                # notice that the data-dependent y and groups parameters for designing the internal folds are passed at .fit() time,
                # while the split strategy function (including the number of splits) is provided up-front here (consistent with sklearn)
                internal_cv=internal_cv,
                # the alternative would be to leave internal_cv unset,
                # and instead just specify a number of internal cross validation splits for glmnet to create with its default strategy:
                # n_splits=3,
                n_lambda=DEFAULT_N_LAMBDAS_FOR_TUNING,
                # standardize outside in the pipeline, to be consistent with other models
                # also this prevents LogitNet from transforming coefficients back to the non-standardized raw scale
                standardize=False,
                scoring=glmnet_cv_scorer,
                n_jobs=n_jobs,
                verbose=True,
                # random_state only used for the nested CV, not for the actual solving
                random_state=0,
                # use slightly simpler model than the lambda with the best CV score
                use_lambda_1se=True,
                # extra check to ensure that the nested CV is not using the same patient's sequences in both the training and test sets
                require_cv_group_labels=True,
                class_weight="balanced",
            ),
        ),
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
        "linearsvm_ovr": LinearSVCSoftmaxProbabilities(  # replacement for svm.LinearSVC
            dual=False,
            multi_class="ovr",
            random_state=0,
            class_weight="balanced",
        ),
    }

    if not chosen_models:
        chosen_models = list(models.keys())

    # filter to selected models
    return {k: v for k, v in models.items() if k in chosen_models}
