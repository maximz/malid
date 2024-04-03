from typing import Generator, List, Optional, Union
from collections import Mapping
from typing_extensions import Self
import numpy as np
from dataclasses import dataclass
import sklearn.base
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
import crosseval
from sklearn.utils.metaestimators import available_if
from malid.external.logging_context_for_warnings import ContextLoggerWrapper
from log_with_context import add_logging_context

logger = ContextLoggerWrapper(name=__name__)


def _inner_estimators_has(attr):
    """Check if self.estimator or self.estimators_[0].clf has attr.

    Based on: sklearn.multiclass._estimators_has
    This version knows to look at the self.estimators_[0].clf attribute on our InnerEstimator data class.

    If `self.estimators_[0]` has the attr, then its safe to assume that other
    values has it too. This function is used together with `avaliable_if`.
    """
    return lambda self: (
        hasattr(self.estimator, attr)
        or (
            hasattr(self, "estimators_")
            and isinstance(self.estimators_[0], InnerEstimator)
            and hasattr(self.estimators_[0].clf, attr)
        )
    )


# declare this class globally to be pickleable for joblib: https://stackoverflow.com/a/16377267/130164
@dataclass
class InnerEstimator:
    clf: BaseEstimator
    negative_class: str
    positive_class: str


@dataclass
class _BinarySubmodelJobConfiguration:
    i: int
    X_possibly_subset: np.ndarray
    y_binary_and_possibly_subset: np.ndarray
    negative_class: str
    positive_class: str
    sample_weight_possibly_subset: Optional[np.ndarray] = None
    groups_possibly_subset: Optional[np.ndarray] = None


class OneVsRestClassifier(ClassifierMixin, BaseEstimator):
    """
    Custom version of sklearn OneVsRestClassifier, giving us more control of the behavior.

    Parameters:
        - estimator: base estimator to train.
            Either a single BaseEstimator, or a dict mapping class names to pre-specified specialized BaseEstimator's for them.
            The provided estimator will be cloned (i.e. any existing fitted parameters will be reset) before fitting each submodel.
        - normalize_predicted_probabilities: whether to normalize predicted probabilities to sum to 1: discouraged!
        - allow_some_classes_to_fail_to_train (default False):
            Whether to continue training even if some classes fail to train (e.g. due to insufficient data for internal cross validation in that binary OvR problem). We will throw an error if *no* classes are trained successfully.
            This option has no effect when the whole problem is binary, because binary means a single inner classifier in this OvR wrapper (basically pass-through), and an error is thrown if that single classifier fails to fit, regardless of this option's setting.

    The resulting class probabilities no longer sum to 1: they come from different models with different calibrations.
    Scikit-learn's OneVsRestClassifier strangely normalizes the probabilities to sum to 1 (except in multilabel settings), but we feel that's not appropriate:
    We don't have reason to believe the probabilities from each classifiers are going to be calibrated and that the sum is already going to be close to 1 (which could make a slight nudge to actually sum to 1 justifiable).
    Our decision pairs nicely with our custom roc_auc_score that does not require probabilities to sum to 1.
    """

    def __init__(
        self,
        estimator: Union[BaseEstimator, dict[Union[int, str], BaseEstimator]],
        *,
        normalize_predicted_probabilities: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        allow_some_classes_to_fail_to_train: bool = False,
    ):
        self.estimator = estimator
        self.normalize_predicted_probabilities = normalize_predicted_probabilities
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.allow_some_classes_to_fail_to_train = allow_some_classes_to_fail_to_train

    @property
    def n_classes_(self):
        """Number of classes."""
        return len(self.classes_)

    def _fit_binary(
        self,
        clf: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        negative_class: str,
        positive_class: str,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Union[InnerEstimator, None]:
        """Fit binary model for a single class. All rows of X are preserved; we just change the y labels."""
        try:
            with add_logging_context(
                positive_class=positive_class, negative_class=negative_class
            ):
                clf, _ = crosseval.train_classifier(
                    clf=clf,
                    model_name=f"{self.__class__.__name__}_{clf.__class__.__name__}_{positive_class}",
                    X_train=self._subset_X_features_for_positive_class(
                        X, positive_class
                    ),
                    y_train=y,
                    # Feed weights and groups through. (We confirm this with automated tests.)
                    train_sample_weights=sample_weight,
                    train_groups=groups,
                )
                return InnerEstimator(
                    clf=clf,
                    negative_class=negative_class,
                    positive_class=positive_class,
                )
        except Exception as e:
            # Also intercept errors, and re-raise them with positive_class, negative_class mentioned.
            msg = f"_fit_binary failed for positive class {positive_class}, negative class {negative_class}: {e}"
            if self.allow_some_classes_to_fail_to_train:
                # Optionally allow some classes to fail to train.
                # Log a warning and return None, so that we can continue training other classes.
                logger.warning(msg)
                return None
            raise RuntimeError(msg) from e

    def _subset_X_features_for_positive_class(
        self, X: np.ndarray, positive_class: str
    ) -> np.ndarray:
        """
        Used by some derived child classes. By default, no-op: keep all columns of X.
        """
        return X

    def _filter_label_binarizer_classes(
        self, label_binarizer_classes: np.ndarray
    ) -> np.ndarray:
        """Used by some derived child classes. By default, no-op: keep all classes."""
        return np.ones(label_binarizer_classes.shape, dtype=bool)

    def _generate_jobs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        columns: Generator,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[_BinarySubmodelJobConfiguration, None, None]:
        """Generate submodel fit job configurations. (Ok to overload but make sure positive_class entries match self.classes_.)"""
        for i, column in enumerate(columns):
            yield _BinarySubmodelJobConfiguration(
                i=i,
                X_possibly_subset=X,
                y_binary_and_possibly_subset=column,
                # negative examples include all classes besides positive_class
                negative_class="not %s" % self.classes_[i],
                positive_class=self.classes_[i],
                sample_weight_possibly_subset=sample_weight,
                groups_possibly_subset=groups,
            )

    def fit(
        self,
        X: np.ndarray,
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
        # Make OvR indicator matrix:
        # Y will have shape (n_samples, n_classes), with column order corresponding to label_binarizer.classes_
        # (This logic, including use of sparse LabelBinarizer, is from sklearn's OneVsRestClassifier:
        # "A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.")
        label_binarizer = LabelBinarizer(sparse_output=True)
        Y = label_binarizer.fit_transform(y)
        Y = Y.tocsc()

        class_mask = self._filter_label_binarizer_classes(label_binarizer.classes_)
        self.classes_ = label_binarizer.classes_[class_mask]
        if len(label_binarizer.classes_) == 2:
            # Binary problem, so Y has shape n x 1, meaning we can't cut it further.
            # Make sure we are not being asked to cut it further: confirm that the class mask is all true.
            if not class_mask.all():
                raise ValueError(
                    f"Applying class mask to binary problem leaves less than two classes in data; nothing to be trained"
                )
            Y_masked = (
                Y  # no masking needed for binary problem; Y is already one column.
            )
        else:
            # Not a binary problem. Apply the class mask.
            Y_masked = Y[:, class_mask]
        columns = (col.toarray().ravel() for col in Y_masked.T)  # this is a generator

        if len(self.classes_) < 2:
            raise ValueError(f"Only one class in data; nothing to be trained")

        # Check if self.estimator is dict:
        if isinstance(self.estimator, Mapping):
            # Specialized estimators for each class have been provided
            # Validate that the dict keys match the classes
            if set(self.estimator.keys()) != set(self.classes_):
                raise ValueError(
                    f"Keys of provided estimator dict ({self.estimator.keys()}) must match classes: {self.classes_}"
                )

        self.estimators_: List[Union[InnerEstimator, None]] = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, backend="loky"
        )(
            # Generate one job for each positive class binary problem.
            delayed(self._fit_binary)(
                clf=sklearn.base.clone(
                    self.estimator[binary_submodel_job_configuration.positive_class]
                )
                if isinstance(self.estimator, Mapping)
                else sklearn.base.clone(self.estimator),
                X=binary_submodel_job_configuration.X_possibly_subset,
                y=binary_submodel_job_configuration.y_binary_and_possibly_subset,
                negative_class=binary_submodel_job_configuration.negative_class,
                positive_class=binary_submodel_job_configuration.positive_class,
                sample_weight=binary_submodel_job_configuration.sample_weight_possibly_subset,
                groups=binary_submodel_job_configuration.groups_possibly_subset,
            )
            for binary_submodel_job_configuration in self._generate_jobs(
                X=X, y=y, columns=columns, sample_weight=sample_weight, groups=groups
            )
        )
        # Remove any None values from self.estimators_, which may exist if we allowed some classes to fail to train.
        self.estimators_: List[InnerEstimator] = [
            est for est in self.estimators_ if est is not None
        ]
        # Confirm that we trained at least one class
        if len(self.estimators_) == 0:
            raise ValueError(
                f"Failed to train any classes: all _fit_binary calls failed"
            )
        # Modify self.classes_ to remove any classes that failed to train:
        if len(label_binarizer.classes_) == 2:
            # In the case of the entire problem being a binary classification:
            # - self.classes_ should contain two classes: the negative class and the positive class.
            # - self.estimators_ should contain only a single inner classifier.
            # - Failing to train is not an option, because there's only one classifier to train altogether. Above, we already checked and errored out if self.estimators_ is empty (i.e. if the one classifier we tried to train ended up failing).
            # (If the whole problem is binary, the binary OvR wrapper is basically pass-through.)

            # For binary classification, include both classes:
            # This setting is special because we don't want to remove the negative class from self.classes_.
            self.classes_ = label_binarizer.classes_
        else:
            # For multiclass setting:
            # The OvR wrapper may have been configured to allow some classes to fail to train.
            # Use the positive classes from trained estimators.
            # Note that _generate_jobs must generate jobs with positive_class being a valid entry of self.classes_
            self.classes_ = np.array([est.positive_class for est in self.estimators_])

        def delattr_if_exists(attrname):
            if hasattr(self, attrname):
                delattr(self, attrname)

        if hasattr(self.estimators_[0].clf, "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].clf.n_features_in_
        else:
            # Latest fitted models do not have n_features_in_:
            # Remove this attribute if it exists (may have been set by a previous call to fit)
            # (This is unlikely)
            delattr_if_exists("n_features_in_")

        if hasattr(self.estimators_[0].clf, "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].clf.feature_names_in_
        else:
            # Latest fitted models do not have feature_names_in_:
            # Remove this attribute if it exists (may have been set by a previous call to fit)
            # (This is common if a previous call to fit() used a numpy array instead of a pandas dataframe with column names)
            delattr_if_exists("feature_names_in_")

        return self

    @available_if(_inner_estimators_has("predict_proba"))  # uses predict_proba
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multi-class targets using underlying estimators.

        This chooses labels based on highest predicted class probability -- which is dangerous! The probabilities are not calibrated.
        Still, we'll keep this function for compatibility with sklearn.
        TODO: Switch to this multilabel behavior? https://github.com/scikit-learn/scikit-learn/blob/6f9c6629e505c5892ded725efa86f91c8fb986e4/sklearn/multiclass.py#L453C11-L464

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.
        Returns
        -------
        y : (sparse) array-like of shape (n_samples,)
            Predicted multi-class targets.
        """
        check_is_fitted(self)

        # predict_proba will handle calling each submodel
        predicted_class_probabilities = self.predict_proba(X)

        predicted_classes = self.classes_[predicted_class_probabilities.argmax(axis=1)]
        return predicted_classes

    @available_if(_inner_estimators_has("predict_proba"))
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates. The returned estimates for all classes are ordered by label of classes.
        Probabilities from the OvR classifiers are *not* normalized to sum to 1, unless self.normalize_predicted_probabilities is True (discouraged!).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        T : (sparse) array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)

        # Y[i, j] gives the probability that sample i has the label j.
        Y = np.array(
            [
                est.clf.predict_proba(
                    self._subset_X_features_for_positive_class(X, est.positive_class)
                )[:, 1]
                for est in self.estimators_
            ]
        ).T

        if len(self.estimators_) == 1:
            # Two classes means only one estimator, but we still want to return probabilities for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if self.normalize_predicted_probabilities:
            # Normalize probabilities to sum to 1.
            Y /= np.sum(Y, axis=1)[:, np.newaxis]

        return Y

    @available_if(_inner_estimators_has("decision_function"))
    def decision_function(self, X: np.ndarray) -> np.ndarray:
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
        check_is_fitted(self)

        if len(self.estimators_) == 1:
            # Two classes means only one estimator
            est = self.estimators_[0]
            return est.clf.decision_function(
                self._subset_X_features_for_positive_class(X, est.positive_class)
            )

        return np.array(
            [
                est.clf.decision_function(
                    self._subset_X_features_for_positive_class(X, est.positive_class)
                ).ravel()
                for est in self.estimators_
            ]
        ).T


class OneVsRestExceptNegativeClassClassifier(OneVsRestClassifier):
    """
    Based on OneVsRestClassifier, with 'other' class excluded.

    For example, when training a sequence classifier, we should not learn a Healthy class.
    At the sequence level, we should only conclude "absence of disease signal".

    A multinomial formulation would directly learn a "healthy" class, because it's a single model that classifies among multiple classes simultaneously. (Each decision boundary separates one class from the rest, but the boundaries are learned jointly, considering the relationships between the classes. There would be a single region in feature space for each class, including a “healthy” region.)

    Instead, we use an OvR strategy. We will still use healthy donor-originating sequences as negative examples, but we won't learn a specific "healthy" class.

    Approach:
    - Only use sequences from healthy people as negative examples for "each disease vs rest" binary classifiers.
    - Train separate binary classifiers, each distinguishing between "Disease i" and "all other diseases and sequences from healthy donors".
    - Result: disease probabilities for each sequence. We no longer have a “healthy” probability per sequence. The disease class probabilities also no longer sum to 1.

    Parameters:
        - estimator: base estimator to train
        - other_class_label: will not train this class vs rest
        - normalize_predicted_probabilities: whether to normalize predicted probabilities to sum to 1

    Expected behavior:
        - predict_proba() returns probability estimates for all but the Other class.
        - predict() will never return the Other class as a prediction.
        - each inner binary estimator is fitted on negative examples that include: a) all classes besides positive_class, b) all examples from self.other_class_label
        - will refuse to fit if there are fewer than 2 classes remaining after the Other class is removed
        - classes_ will not include the Other class
    """

    def __init__(
        self,
        estimator: Union[BaseEstimator, dict[Union[int, str], BaseEstimator]],
        other_class_label: str,
        *,
        normalize_predicted_probabilities: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        allow_some_classes_to_fail_to_train: bool = False,
    ):
        """other_class_label: will not train this class vs rest"""
        self.other_class_label = other_class_label
        super().__init__(
            estimator=estimator,
            normalize_predicted_probabilities=normalize_predicted_probabilities,
            n_jobs=n_jobs,
            verbose=verbose,
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )

    def _filter_label_binarizer_classes(
        self, label_binarizer_classes: np.ndarray
    ) -> np.ndarray:
        if self.other_class_label is None:
            raise ValueError(
                f"Other class must be specified for {self.__class__.__name__}"
            )

        if self.other_class_label not in label_binarizer_classes:
            raise ValueError(
                f"The specified negative class '{self.other_class_label}' is not in y, which has classes {label_binarizer_classes}"
            )
        return label_binarizer_classes != self.other_class_label


class OneVsNegativeClassifier(OneVsRestClassifier):
    """
    OneVsRestClassifier that contrasts each positive class against a designated negative class.

    Based on OneVsRestClassifier, we train separate binary classifiers, each distinguishing between one class and a common negative class.

    For example, with "Healthy" as the negative class, we would fit "Covid vs Healthy" and "HIV vs Healthy".
    This is different from fitting "Covid vs Rest", which means "Covid vs (HIV + Healthy)". Instead, only Healthy is used as negative examples.

    This design helps decouple the classes. For example, we will avoid predicting that an item is Covid based on how much it does not look like HIV.

    Important: We no longer have Healthy probabilities per item. The disease class probabilities also no longer sum to 1.
    (TODO: Consider Rob's optimization approach for unifying the negative class Healthy predictions into a single set of probabilities: https://projecteuclid.org/journals/annals-of-statistics/volume-26/issue-2/Classification-by-pairwise-coupling/10.1214/aos/1028144844.full)

    Extra parameters beyond OneVsRestClassifier's standard parameters:
        - negative_class: label of the negative class

    Expected behavior:
        - We no longer have a “healthy” probability per sequence: predict_proba() returns probability estimates for all but the negative class. The disease class probabilities also no longer sum to 1.
        - predict() will never return the negative class as a prediction.
        - will refuse to fit if there are fewer than 2 classes remaining after the negative class is removed
        - classes_ will not include the negative class
    """

    def __init__(
        self,
        estimator: Union[BaseEstimator, dict[Union[int, str], BaseEstimator]],
        negative_class: str,
        *,
        normalize_predicted_probabilities: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        allow_some_classes_to_fail_to_train: bool = False,
    ):
        self.negative_class = negative_class
        super().__init__(
            estimator=estimator,
            normalize_predicted_probabilities=normalize_predicted_probabilities,
            n_jobs=n_jobs,
            verbose=verbose,
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )

    def _filter_label_binarizer_classes(
        self, label_binarizer_classes: np.ndarray
    ) -> np.ndarray:
        if self.negative_class is None:
            raise ValueError(
                f"Negative class must be specified for {self.__class__.__name__}"
            )

        if self.negative_class not in label_binarizer_classes:
            raise ValueError(
                f"The specified negative class '{self.negative_class}' is not in y, which has classes {label_binarizer_classes}"
            )

        return label_binarizer_classes != self.negative_class

    def _generate_jobs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        columns: Generator,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[_BinarySubmodelJobConfiguration, None, None]:
        """Generate submodel fit job configurations."""
        # compute length of columns, which will exhaust this generator
        computed_length_of_generator = sum(1 for _ in columns)
        for i in range(computed_length_of_generator):
            # Construct subproblem:
            negative_class = self.negative_class
            positive_class = self.classes_[i]

            # Subset to entries where y_true belongs to either the negative class or the postiive class
            mask = np.logical_or(y == negative_class, y == positive_class)

            X_subset = X[mask]

            y_subset_and_binary = np.zeros(y[mask].shape, int)
            y_subset_and_binary[y[mask] == positive_class] = 1

            if sample_weight is not None:
                sample_weight_subset = sample_weight[mask]
            else:
                sample_weight_subset = None

            if groups is not None:
                groups_subset = groups[mask]
            else:
                groups_subset = None

            yield _BinarySubmodelJobConfiguration(
                i=i,
                X_possibly_subset=X_subset,
                y_binary_and_possibly_subset=y_subset_and_binary,
                negative_class=negative_class,
                positive_class=positive_class,
                sample_weight_possibly_subset=sample_weight_subset,
                groups_possibly_subset=groups_subset,
            )
