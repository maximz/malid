import logging
from pathlib import Path
from typing import Optional, Union, List

import anndata
import joblib
import numpy as np
import pandas as pd

from malid import helpers, config
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    combine_classification_option_names,
)
from extendanything import ExtendAnything
from malid.external.genetools_arrays import make_dummy_variables_in_specific_order
from malid.external.model_evaluation import FeaturizedData
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)

logger = logging.getLogger(__name__)


class SequenceClassifier(ImmuneClassifierMixin, ExtendAnything):
    """Wrapper around sequence classification models. Featurize manually with featurize() before calling predict(), predict_proba(), or decision_function()."""

    # ExtendAnything means will pass through to base instance's attributes, exposing the base model's classes_, predict(), predict_proba(), etc.

    def __init__(
        self,
        fold_id: int,
        model_name_sequence_disease: str,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
        models_base_dir: Optional[Path] = None,
    ):
        if models_base_dir is None:
            models_base_dir = self._get_model_base_dir(
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
            )
        models_base_dir = Path(models_base_dir)

        # Load and wrap classifier
        fname = (
            models_base_dir
            / f"{fold_label_train}_model.{model_name_sequence_disease}.{fold_id}.joblib"
        )
        # sets self._inner to loaded model, to expose its attributes
        super().__init__(
            inner=joblib.load(fname),
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
            models_base_dir=models_base_dir,
        )
        self.model_name_sequence_disease = model_name_sequence_disease

    @staticmethod
    def _get_model_base_dir(
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        return (
            config.paths.sequence_models_dir
            / gene_locus.name
            / combine_classification_option_names(
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
            )
        )

    def featurize(
        self,
        dataset: anndata.AnnData,
    ) -> FeaturizedData:
        """Featurize sequences in repertoire, returning FeaturizedData object.
        Calls internal static method equivalent (available if you don't have an instantiated SequenceClassifier object.)
        Allows missing isotypes.
        """
        is_raw = False
        if helpers.should_switch_to_raw(dataset):
            # Use raw data
            # Note this log line will fire once per specimen when using rollup sequence classifier because we run model on each specimen separately.
            # We could alternatively run once on full test set, then groupby specimen_label to aggregate, but this seems fine.
            # maybe turn the logging off.
            logger.debug(
                f"Switching SequenceClassifier (fold {self.fold_id}, model {self.model_name_sequence_disease}, fold_label_train={self.fold_label_train}) to raw data for alternative target {self.target_obs_column}"
            )
            is_raw = True

        return self._featurize(
            repertoire=dataset,
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
            sample_weight_strategy=self.sample_weight_strategy,
            allow_missing_isotypes=True,
            is_raw=is_raw,
        )

    @classmethod
    def _featurize(
        cls,
        repertoire: anndata.AnnData,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
        allow_missing_isotypes=True,
        is_raw=False,
    ) -> FeaturizedData:
        """Featurize sequences in repertoire, returning FeaturizedData object."""
        GeneLocus.validate(gene_locus)
        GeneLocus.validate_single_value(gene_locus)
        TargetObsColumnEnum.validate(target_obs_column)
        SampleWeightStrategy.validate(sample_weight_strategy)

        # Choose scaled or unscaled data
        data_X = np.array(repertoire.raw.X if is_raw else repertoire.X)
        data_X = cls._add_extra_columns_to_embedding_vectors(
            data_X=data_X,
            isotype_groups=repertoire.obs["isotype_supergroup"],
            v_genes=repertoire.obs["v_gene"],
            gene_locus=gene_locus,
            allow_missing_isotypes=allow_missing_isotypes,
        )

        def _get_weights(repertoire):
            """Get sequence weights if loaded and requested. Will throw error if not available in repertoire."""
            if sample_weight_strategy == SampleWeightStrategy.NONE:
                # By default, all data points have uniform sample weight
                return None

            if sample_weight_strategy == SampleWeightStrategy.ISOTYPE_USAGE:
                if gene_locus == GeneLocus.TCR:
                    # this is a no-op - only one isotype
                    return None
                return repertoire.obs["sample_weight_isotype_rebalance"]
            elif sample_weight_strategy == SampleWeightStrategy.GRAPH_CONFIDENCE:
                return repertoire.obs["sequence_confidence_rating"]
            else:
                raise NotImplementedError(
                    f"Unsupported sample_weight_strategy: {sample_weight_strategy}"
                )

        return FeaturizedData(
            X=data_X,
            y=repertoire.obs[target_obs_column.value.obs_column_name],
            sample_names=repertoire.obs_names,
            metadata=repertoire.obs,
            sample_weights=_get_weights(repertoire),
        )

    @staticmethod
    def _add_extra_columns_to_embedding_vectors(
        data_X: np.ndarray,
        isotype_groups: Union[pd.Series, List[str]],
        v_genes: Union[pd.Series, List[str]],
        gene_locus: GeneLocus,
        allow_missing_isotypes=True,
    ) -> np.ndarray:
        def add_isotype_dummy_variables_to_embedding_vectors(
            isotype_groups: Union[pd.Series, List[str]],
            gene_locus: GeneLocus,
            allow_missing_isotypes=False,
        ) -> pd.DataFrame:
            """Create dummy variables corresponding to isotype group of each sequence to the sequence embedding vectors.
            This enforces a consistent ordering of the dummy variables, and ensures the provided isotype groups belong to a subset of allowed isotype groups.

            Usage: add_isotype_dummy_variables_to_embedding_vectors(adata.obs["isotype_supergroup"], GeneLocus.BCR)

            If allow_missing_isotypes is set to True (defaults to False), don't error if any expected isotype group (i.e. any of helpers.isotype_groups_kept) is missing. Just set those dummy variables to 0 always.
            Will still error if there are any isotype groups provided as input that are not in helpers.isotype_groups_kept. Such isotype groups should be filtered out before running this method.
            """
            return make_dummy_variables_in_specific_order(
                values=isotype_groups,
                expected_list=helpers.isotype_groups_kept[gene_locus],
                allow_missing_entries=allow_missing_isotypes,
            )

        def add_v_gene_dummy_variables_to_embedding_vectors(
            v_genes: Union[pd.Series, List[str]], gene_locus: GeneLocus
        ) -> pd.DataFrame:
            """Create dummy variables corresponding to V gene of each sequence to the sequence embedding vectors.
            This enforces a consistent ordering of the dummy variables, and inserts zeroes for any V genes that are not in this dataset but in the full list of V genes seen anywhere.
            Will throw an error if you pass in any v_genes that are not found in the list of all possible observed V genes.

            Usage: add_v_gene_dummy_variables_to_embedding_vectors(adata.obs["v_gene"], GeneLocus.BCR)
            """
            return make_dummy_variables_in_specific_order(
                values=v_genes,
                expected_list=helpers.all_observed_v_genes()[gene_locus],
                # input dataset may only have a subset of all possible V genes, of course
                allow_missing_entries=True,
            )

        GeneLocus.validate_single_value(gene_locus)
        # Attach isotype group information as dummy variables.
        # Allow some isotype groups to be missing (we are not guaranteed to have sampled all isotypes from these test patients)
        hstack_items = []
        hstack_items.append(
            # Cast to X's dtype, so that the eventual hstacking doesn't promote X's dtype to some higher-precision dtype like float64.
            add_isotype_dummy_variables_to_embedding_vectors(
                isotype_groups=isotype_groups,
                gene_locus=gene_locus,
                allow_missing_isotypes=allow_missing_isotypes,
            ).values.astype(data_X.dtype)
        )

        # Attach V gene information as dummy variables
        if config.include_v_gene_as_dummy_variable:
            hstack_items.append(
                # Cast to X's dtype, so that the eventual hstacking doesn't promote X's dtype to some higher-precision dtype like float64.
                add_v_gene_dummy_variables_to_embedding_vectors(
                    v_genes=v_genes, gene_locus=gene_locus
                ).values.astype(data_X.dtype)
            )

        # Perform the horizontal stacking.
        # (Don't do it separately for each hstack item, because this causes a copy.)
        return np.hstack((data_X, *hstack_items))

    # TODO: add get_hidden_state() logic here that will call self.model.get_hidden_state() if that exists or raise NotImplementedError
