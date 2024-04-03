from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from malid import io, config
from malid.datamodels import (
    GeneLocus,
    GeneralAnndataType,
    TargetObsColumnEnum,
    SampleWeightStrategy,
)
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from crosseval import FeaturizedData


class MetadataFeaturizerMixin(metaclass=ABCMeta):
    @abstractmethod
    def featurize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Featurize dataset."""
        pass


class ImmuneClassifierMixin(metaclass=ABCMeta):
    def __init__(
        self,
        *args,
        # Keyword-only arguments:
        fold_id: int,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        models_base_dir: Optional[Path] = None,
        # These might not be set by every classifier:
        sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    ):
        """
        Common interface to access classifier metadata.

        In the case of multiple inheritance, ImmuneClassifierMixin does *not* forward unused arguments to the next class in the Method Resolution Order.
        The user must explicitly initialize other base classes.
        """

        if sample_weight_strategy is None:
            # This might not be set by every classifier
            sample_weight_strategy = config.sample_weight_strategy

        GeneLocus.validate(gene_locus)
        GeneLocus.validate_single_value(gene_locus)
        TargetObsColumnEnum.validate(target_obs_column)
        SampleWeightStrategy.validate(sample_weight_strategy)

        self.fold_id = fold_id
        self.fold_label_train = fold_label_train
        self.gene_locus = gene_locus
        self.target_obs_column = target_obs_column
        self.sample_weight_strategy = sample_weight_strategy

        # If a custom models_base_dir is given, overwrite the default property
        if models_base_dir is not None:
            self._custom_models_base_dir = Path(models_base_dir)
        else:
            self._custom_models_base_dir = None

    # output_base_dir and models_base_dir are properties, not strings, so that they do not go stale when the class is pickled to disk as a submodel in a MetamodelConfig object.
    # However, the user is allowed to override them by setting them directly.
    @property
    def output_base_dir(self):
        return self._get_output_base_dir(
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
            sample_weight_strategy=self.sample_weight_strategy,
        )

    @property
    def models_base_dir(self):
        # Custom directory provided during initialization takes precedence over default.
        return self._custom_models_base_dir or self._get_model_base_dir(
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
            sample_weight_strategy=self.sample_weight_strategy,
        )

    @staticmethod
    @abstractmethod
    def _get_output_base_dir(
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        ...

    @staticmethod
    @abstractmethod
    def _get_model_base_dir(
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        ...

    @abstractmethod
    def featurize(self, dataset: GeneralAnndataType) -> FeaturizedData:
        """Featurize dataset."""
        pass

    def tune_model_decision_thresholds_to_validation_set(self, validation_set=None):
        """Adjust model decision thresholds against validation set.

        Provide validation_set or it will be loaded using this clf's fold_id setting.
        """
        if not self.fold_label_train.startswith("train_smaller"):
            raise ValueError(
                "Fold label train must start with train_smaller because we will tune on validation set. Do not tune on validation set if model was trained on train-full (which includes validation set)."
            )

        if validation_set is None:
            # load validation set.
            # if using classifier on high quality sequences only, load that subset only
            validation_set = io.load_fold_embeddings(
                fold_id=self.fold_id,
                fold_label="validation",
                target_obs_column=self.target_obs_column,
                gene_locus=self.gene_locus,
                sample_weight_strategy=self.sample_weight_strategy,
            )

        validation_featurized = self.featurize(
            dataset=validation_set,
        )

        clf_tuned = AdjustedProbabilitiesDerivedModel.adjust_model_decision_thresholds(
            model=self,
            X_validation=validation_featurized.X,
            y_validation_true=validation_featurized.y,
        )
        return clf_tuned

    # TODO: Can we move evaluate_performance() from threshold-tuning notebooks here?
    # Not sure, because want to call adjusted-threshold wrapper class's predict();
    # putting the logic here would restrict it to calling base classifier's predict().
