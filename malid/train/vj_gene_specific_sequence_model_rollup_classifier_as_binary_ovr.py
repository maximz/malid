from typing import Optional, Union
from typing_extensions import Self
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
from malid.train.one_vs_rest_except_negative_class_classifier import (
    OneVsNegativeClassifier,
    OneVsRestClassifier,
    _inner_estimators_has,
)


class BinaryOvRClassifierWithFeatureSubsettingByClass(OneVsRestClassifier):
    """
    OneVsRestClassifier where each binary classifier uses a subset of features that start with the positive class name (with a given delimeter).

    Based on OneVsRestClassifier, but only a subset of features is available and delivered to each inner classifier.

    For example, when training the Covid vs rest classifier, we will use Covid-specific features only.
    It would not make sense for the Covid aggregation model to assign a positive coefficient to the "mean HIV sequence probability" feature; we will get rid of that feature altogether.

    A multinomial formulation would require consistent features, because it's a single model that classifies among multiple classes simultaneously. (Each decision boundary separates one class from the rest, but the boundaries are learned jointly, considering the relationships between the classes. There would be a single region in feature space for each class.)

    Instead, we use an OvR strategy. Each binary classifier will receive a subset of the total original features.

    Important: This model requires feature names to be provided in the input data. So all X parameters must be a pandas dataframe with named columns.

    Features for each submodel will be chosen by matching the beginning of the feature name to the submodel's positive class.
    `feature_name_class_delimeter` can optionally be provided to enforce a delimeter.
    For example, with feature_name_class_delimeter == "_", features named "Covid19_*" will be used for the Covid19 vs rest classifier.
    Any features that do not match a class name will be silently ignored.

    Extra parameters on top of usual malid.train.OneVsRestClassifier parameters:
        - estimator: base estimator to train. Either a single BaseEstimator, or a dict mapping class names to pre-specified specialized BaseEstimator's for them.
        - feature_name_class_delimeter: will be used when assigning features to each class. Optional; defaults to empty string.
    """

    def __init__(
        self,
        estimator: Union[BaseEstimator, dict[Union[int, str], BaseEstimator]],
        *,
        feature_name_class_delimeter: str = "",
        normalize_predicted_probabilities: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        allow_some_classes_to_fail_to_train: bool = False,
    ):
        self.feature_name_class_delimeter = feature_name_class_delimeter
        super().__init__(
            estimator=estimator,
            normalize_predicted_probabilities=normalize_predicted_probabilities,
            n_jobs=n_jobs,
            verbose=verbose,
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )

    def _subset_X_features_for_positive_class(
        self, X: pd.DataFrame, positive_class: str
    ) -> pd.DataFrame:
        """
        Used in _fit_binary to subset X to the features that are relevant to the positive class.
        Given a pandas dataframe X, return X subsetted to columns that start with f"{positive_class}{self.feature_name_class_delimeter}"
        All rows are preserved.
        """
        bool_array = X.columns.str.startswith(
            f"{positive_class}{self.feature_name_class_delimeter}"
        )
        if bool_array.sum() == 0:
            raise ValueError(
                f"No features found for positive class {positive_class}. Check that feature names are provided in X."
            )
        return X.loc[:, bool_array]

    @staticmethod
    def _verify_feature_names_provided(X: pd.DataFrame):
        """Require that X has feature names"""
        if type(X) is not pd.DataFrame:
            raise ValueError(f"X must be a pandas dataframe with feature names")

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Self:
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.
        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.
        Returns
        -------
        self
        """
        self._verify_feature_names_provided(X)
        super().fit(X, y, sample_weight=sample_weight, groups=groups, **kwargs)

        # Set feature_names_in_ and n_features_in_ attributes based on the full set of features
        # (Otherwise at this moment it is based on the subset of features that were used for fitting the first submodel)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        return self

    @available_if(_inner_estimators_has("predict_proba"))
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability estimates. The returned estimates for all classes are ordered by label of classes.
        Probabilities from the OvR classifiers are *not* normalized to sum to 1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        T : (sparse) array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        self._verify_feature_names_provided(X)
        return super().predict_proba(X)

    # Not all estimators implement decision_function, so we need to check before advertising that we have this functionality.
    # Otherwise other code will attempt to call this, and hit errors.
    @available_if(_inner_estimators_has("decision_function"))
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Decision function for the OneVsRestClassifier.

        Return the distance of each sample from the decision boundary for each
        class. This can only be used with estimators which implement the
        `decision_function` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes) or (n_samples,) for \
            binary classification.
            Result of calling `decision_function` on the final estimator.
        """
        self._verify_feature_names_provided(X)
        return super().decision_function(X)


class OneVsNegativeClassifierWithFeatureSubsettingByClass(
    OneVsNegativeClassifier, BinaryOvRClassifierWithFeatureSubsettingByClass
):
    """
    Combine this behavior into one model:
    - Contrast each positive class against a designated negative class.
    - For each binary classifier, use a subset of features that start with the positive class name (with a given delimeter).

    Implementation note: the multiple inheritance is tricky here:
    - We can't pass *args, **kwargs around because the base class inherits from sklearn which forbids this.
    - When each constructor calls the next superclass in the MRO, it won't necessarily have the right arguments provided.
        That's why we can't put BinaryOvRClassifierWithFeatureSubsettingByClass first: it wouldn't have a negative_class argument that OneVsNegativeClassifier (next in the MRO) expects.
        We get lucky that BinaryOvRClassifierWithFeatureSubsettingByClass has a default value for feature_name_class_delimeter, so we don't need to provide it when calling from OneVsNegativeClassifier.
    - In trickier situations, we would have to combine the classes manually, e.g. inherit only from BinaryOvRClassifierWithFeatureSubsettingByClass and copy OneVsNegativeClassifier's logic in here.
    """

    def __init__(
        self,
        estimator: Union[BaseEstimator, dict[Union[int, str], BaseEstimator]],
        negative_class: str,
        *,
        feature_name_class_delimeter: str = "",
        normalize_predicted_probabilities: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        allow_some_classes_to_fail_to_train: bool = False,
    ):
        OneVsNegativeClassifier.__init__(
            self,
            estimator=estimator,
            negative_class=negative_class,
            normalize_predicted_probabilities=normalize_predicted_probabilities,
            n_jobs=n_jobs,
            verbose=verbose,
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )
        BinaryOvRClassifierWithFeatureSubsettingByClass.__init__(
            self,
            estimator=estimator,
            feature_name_class_delimeter=feature_name_class_delimeter,
            normalize_predicted_probabilities=normalize_predicted_probabilities,
            n_jobs=n_jobs,
            verbose=verbose,
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )


# TODO: Create similar combination of OneVsRestExceptNegativeClassClassifier and BinaryOvRClassifierWithFeatureSubsettingByClass.
