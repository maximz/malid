from abc import ABCMeta, abstractmethod
from pathlib import Path

import anndata
import pandas as pd

from malid import io
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
)
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from malid.external.model_evaluation import FeaturizedData


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
        models_base_dir: Path,
        # These might not be set by every classifier:
        sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE,
        **kwargs,
    ):
        # Forward all unused arguments in the case of multiple inheritance (https://stackoverflow.com/a/50465583/130164)
        super().__init__(*args, **kwargs)

        GeneLocus.validate(gene_locus)
        GeneLocus.validate_single_value(gene_locus)
        TargetObsColumnEnum.validate(target_obs_column)
        SampleWeightStrategy.validate(sample_weight_strategy)

        self.fold_id = fold_id
        self.fold_label_train = fold_label_train
        self.gene_locus = gene_locus
        self.target_obs_column = target_obs_column
        self.models_base_dir = models_base_dir
        self.sample_weight_strategy = sample_weight_strategy

    @abstractmethod
    def featurize(self, dataset: anndata.AnnData) -> FeaturizedData:
        """Featurize dataset."""
        pass

    def tune_model_decision_thresholds_to_validation_set(self, validation_set=None):
        """Adjust model decision thresholds against validation set.

        Provide validation_set or it will be loaded using this clf's fold_id setting.
        """
        if self.fold_label_train != "train_smaller":
            raise ValueError(
                "Fold label train must be train_smaller because we will tune on validation set"
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
