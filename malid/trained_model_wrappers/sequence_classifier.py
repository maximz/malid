import logging
from pathlib import Path
from typing import Optional, Union, List

import anndata
import joblib
import numpy as np
import pandas as pd

from malid import helpers, config
from malid.apply_embedding import get_embedder_used_for_embedding_anndata
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    combine_classification_option_names,
    GeneralAnndataType,
)
from extendanything import ExtendAnything
from genetools.arrays import make_dummy_variables_in_specific_order
from crosseval import FeaturizedData
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AbstractSequenceClassifier(ImmuneClassifierMixin, ABC):
    """
    A type denoting a sequence-level classifier.
    Must implement what ImmuneClassifierMixin requires.
    """

    pass


class SequenceClassifier(AbstractSequenceClassifier, ExtendAnything):
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
        # Control the order of superclass initialization.
        # 1. Call ImmuneClassifierMixin's constructor
        ImmuneClassifierMixin.__init__(
            self,
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
            models_base_dir=models_base_dir,
        )

        # 2. Now that models_base_dir is set, construct the file path
        fname = (
            self.models_base_dir
            / f"{self.fold_label_train}_model.{model_name_sequence_disease}.{self.fold_id}.joblib"
        )

        # 3. Call ExtendAnything's constructor to load and wrap classifier
        # self._inner will now be the loaded model, and its attributes will be exposed (pass-through)
        ExtendAnything.__init__(self, inner=joblib.load(fname))

        # Set other attributes.
        self.model_name_sequence_disease = model_name_sequence_disease

    @staticmethod
    def _get_directory_suffix(
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        return Path(gene_locus.name) / combine_classification_option_names(
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )

    @classmethod
    def _get_model_base_dir(
        cls,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        return config.paths.sequence_models_dir / cls._get_directory_suffix(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )

    @classmethod
    def _get_output_base_dir(
        cls,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        return config.paths.sequence_models_output_dir / cls._get_directory_suffix(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
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
        # If include_v_gene_as_dummy_variable is not specified explicitly, its default will be set based on the embedder used to make this anndata
        include_v_gene_as_dummy_variable: Optional[bool] = None,
        include_isotype_as_dummy_variable: bool = True,
        # SHM rate column is included if include_v_mut_feature is True and the gene locus is BCR. For GeneLocus.TCR, this value is always 0, so we do not include the column.
        include_v_mut_feature: bool = True,
    ) -> FeaturizedData:
        """
        Featurize sequences in repertoire, returning FeaturizedData object.

        include_v_gene_as_dummy_variable: default is None, which means to use the default for the EmbedderSequenceContent used for the embedding in the anndata.
        """
        GeneLocus.validate(gene_locus)
        GeneLocus.validate_single_value(gene_locus)
        TargetObsColumnEnum.validate(target_obs_column)
        SampleWeightStrategy.validate(sample_weight_strategy)

        if include_v_gene_as_dummy_variable is None:
            # Embedder-dependent modeling feature flag:
            # Only include V gene as dummy variable in model3 if embedding full sequence or CDR1+2+3, without prioritizing CDR3 only.
            # This looks up the current embedder_sequence_content setting for the embedder name recorded in the anndata.
            include_v_gene_as_dummy_variable = get_embedder_used_for_embedding_anndata(
                repertoire
            ).embedder_sequence_content.include_v_gene_as_dummy_variable

        # Choose scaled or unscaled data based on is_raw
        data_X = np.array(repertoire.raw.X if is_raw else repertoire.X)

        # Add requested dummy variables
        data_X = cls._add_extra_columns_to_embedding_vectors(
            data_X=data_X,
            isotype_groups=repertoire.obs["isotype_supergroup"],
            v_genes=repertoire.obs["v_gene"],
            gene_locus=gene_locus,
            include_v_gene_as_dummy_variable=include_v_gene_as_dummy_variable,
            include_isotype_as_dummy_variable=include_isotype_as_dummy_variable,
            allow_missing_isotypes=allow_missing_isotypes,
        )

        # Add V gene somatic hypermutation rate (v_mut) feature, for BCR only.
        # (TCR v_mut is always 0, so we don't need to add it.)
        if include_v_mut_feature and gene_locus == GeneLocus.BCR:
            data_X = np.column_stack(
                # Cast to X's dtype so that hstacking doesn't promote X's dtype to some higher-precision dtype like float64.
                (data_X, repertoire.obs["v_mut"].to_numpy(dtype=data_X.dtype))
            )

        # TODO: Overload FeaturizedData as in SubsetRollupClassifierFeaturizedData to indicate that X is always a ndarray.
        return FeaturizedData(
            X=data_X,
            y=repertoire.obs[target_obs_column.value.obs_column_name],
            sample_names=repertoire.obs_names,
            metadata=repertoire.obs,
            sample_weights=cls._get_sequence_weights(
                repertoire, gene_locus, sample_weight_strategy
            ),
        )

    @staticmethod
    def _get_sequence_weights(
        adata: GeneralAnndataType,
        gene_locus: GeneLocus,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Optional[pd.Series]:
        """Get sequence weights if loaded and requested. Will throw error if not available in repertoire."""
        if sample_weight_strategy == SampleWeightStrategy.NONE or (
            sample_weight_strategy == SampleWeightStrategy.ISOTYPE_USAGE
            and gene_locus == GeneLocus.TCR
        ):
            # By default, all data points have uniform sample weight
            return None

        # One or more custom sample weight strategies have been selected.
        # We will get all the weights and then element-wise multiply them.
        weights = []

        if (
            SampleWeightStrategy.ISOTYPE_USAGE in sample_weight_strategy
            and gene_locus == GeneLocus.BCR
        ):
            weights.append(adata.obs["sample_weight_isotype_rebalance"])

        if SampleWeightStrategy.CLONE_SIZE in sample_weight_strategy:
            weights.append(adata.obs["sample_weight_clone_size"])

        if len(weights) == 0:
            raise NotImplementedError(
                f"Unsupported sample_weight_strategy: {sample_weight_strategy}"
            )

        # Confirm weights have same shapes
        shapes = np.array([arr.shape[0] for arr in weights])
        if not (shapes == shapes[0]).all():
            raise ValueError(f"Sample weights have different shapes: {shapes}")

        # Combine weights by elementwise multiplication
        combined = np.multiply.reduce(weights)
        if not combined.shape[0] == shapes[0]:
            raise ValueError(
                f"Combined sample weights have unexpected shape: {combined.shape[0]} (expected {shapes[0]})"
            )
        return combined

    @staticmethod
    def _add_extra_columns_to_embedding_vectors(
        data_X: np.ndarray,
        isotype_groups: Union[pd.Series, List[str]],
        v_genes: Union[pd.Series, List[str]],
        gene_locus: GeneLocus,
        include_v_gene_as_dummy_variable: bool,
        include_isotype_as_dummy_variable: bool,
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
        hstack_items = []

        # Attach isotype group information as dummy variables.
        # Allow some isotype groups to be missing (we are not guaranteed to have sampled all isotypes from these test patients)
        if include_isotype_as_dummy_variable:
            hstack_items.append(
                # Cast to X's dtype, so that the eventual hstacking doesn't promote X's dtype to some higher-precision dtype like float64.
                add_isotype_dummy_variables_to_embedding_vectors(
                    isotype_groups=isotype_groups,
                    gene_locus=gene_locus,
                    allow_missing_isotypes=allow_missing_isotypes,
                ).to_numpy(dtype=data_X.dtype)
            )

        # Attach V gene information as dummy variables
        if include_v_gene_as_dummy_variable:
            hstack_items.append(
                # Cast to X's dtype, so that the eventual hstacking doesn't promote X's dtype to some higher-precision dtype like float64.
                add_v_gene_dummy_variables_to_embedding_vectors(
                    v_genes=v_genes, gene_locus=gene_locus
                ).to_numpy(dtype=data_X.dtype)
            )

        # Perform the horizontal stacking.
        # (Don't do it separately for each hstack item, because this causes a copy.)
        return np.column_stack((data_X, *hstack_items))

    # TODO: add get_hidden_state() logic here that will call self.model.get_hidden_state() if that exists or raise NotImplementedError
