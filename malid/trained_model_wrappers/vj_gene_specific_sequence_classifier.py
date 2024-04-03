from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Generator, Optional, List, Tuple, Type, Union
from enum import Enum, auto
from enum_mixins import ValidatableEnumMixin
import anndata
import dataclasses
import joblib
import numpy as np
import pandas as pd
from slugify import slugify
from malid import helpers
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from crosseval import Classifier, FeaturizedData
from malid.trained_model_wrappers import AbstractSequenceClassifier, SequenceClassifier

logger = logging.getLogger(__name__)

# split_key could be a string, or an arbitrary-length tuple of strings
SplitKeyType = Union[str, tuple[str, ...]]


@dataclasses.dataclass(eq=False)
class SubsetClassifierFeaturizedData(FeaturizedData):
    # Overload FeaturizedData:
    # X has different type than in typical FeaturizedData
    X: anndata.AnnData


class VJGeneSpecificSequenceClassifier(AbstractSequenceClassifier):
    """
    V-J gene specific sequence classifier.

    Set model_name_sequence_disease to the name of the model class that is used, such as "elasticnet_cv".

    When creating a RollupSequenceClassifier:
    - set model_name_sequence_disease to be the model class, such as "elasticnet_cv", just like above.
    - initialize VJGeneSpecificSequenceClassifier ourselves and pass the VJGeneSpecificSequenceClassifier as the sequence_classifier argument to RollupSequenceClassifier's constructor.
    """

    split_on: List[str] = ["v_gene", "j_gene"]
    split_short_name: str = "vj_gene_specific"

    models_: Dict[SplitKeyType, Classifier]
    classes_: np.ndarray

    def __init__(
        self,
        fold_id: int,
        model_name_sequence_disease: str,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
        models_base_dir: Optional[Path] = None,
        # Optionally skip loading from disk (used in tests)
        models: Optional[Dict[SplitKeyType, Classifier]] = None,
        classes: Optional[np.ndarray] = None,
    ):
        # After this superclass call, self.models_base_dir will be available regardless of whether the user provided a custom override to the default base dir property.
        super().__init__(
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
            models_base_dir=models_base_dir,
        )

        self.model_name_sequence_disease = model_name_sequence_disease
        if models is None or classes is None:
            # Load classifiers
            loaded: dict = joblib.load(
                self.models_base_dir
                / f"{self.fold_label_train}_model.{self.model_name_sequence_disease}.{self.fold_id}.split_models.joblib"
            )
            self.models_ = loaded["models"]
            self.classes_ = loaded["classes"]
        else:
            self.models_ = models
            self.classes_ = classes

    @classmethod
    def _get_directory_suffix(cls) -> Path:
        return Path(f"{cls.split_short_name}_models")

    @classmethod
    def _get_model_base_dir(
        cls,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        return (
            SequenceClassifier._get_model_base_dir(
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
            )
            / cls._get_directory_suffix()
        )

    @classmethod
    def _get_output_base_dir(
        cls,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> Path:
        return (
            SequenceClassifier._get_output_base_dir(
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
            )
            / cls._get_directory_suffix()
        )

    def predict(self, dataset: anndata.AnnData) -> pd.Series:
        """Will return NaN for sequences that had no compatible model."""
        predictions_proba: pd.DataFrame = self.predict_proba(dataset)
        return predictions_proba.idxmax(axis=1)

    @classmethod
    def _featurize_split(
        cls,
        adata_split: anndata.AnnData,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
    ) -> FeaturizedData:
        return SequenceClassifier._featurize(
            repertoire=adata_split,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
            #
            # Go back to the raw, unscaled language model embeddings. We will scale them separately for each V-J gene submodel.
            # This is because we are zooming into a tiny subset of the data which is likely quite similar in LLM embedding space.
            # If we just use the original scaled LLM embeddings, these input features would no longer be expected to have zero mean and unit variance or anything close to it.
            # We want to revert to raw embeddings and rescale from scratch for just the subset of interest.
            # (For comparison, in old-school model3, this flag depended on the value of target_obs_column:
            # - we skipped reverting to raw for TargetObsColumnEnum.disease because that was the happy path: all rows included, so can use already-available scaled embeddings.
            # - other target_obs_columns would subset the rows, so we had to revert to raw and rescale.)
            is_raw=True,
            #
            # Disable adding V gene dummy variable if splitting on V gene or V family
            include_v_gene_as_dummy_variable="v_gene" not in cls.split_on
            and "v_family" not in cls.split_on,
            #
            # Disable adding isotype dummy variable if splitting on isotype
            include_isotype_as_dummy_variable="isotype_supergroup" not in cls.split_on,
            #
            # Add v_mut (somatic hypermutation (SHM) rate) as a feature always
            include_v_mut_feature=True,
        )

    def predict_proba(self, dataset: anndata.AnnData) -> pd.DataFrame:
        """Score sequences to appropriate models. Some will have no model, and will have probabilities = NaN."""
        scores = []
        for split_key, adata_split in self.generate_training_chunks(dataset):
            # split_key could be (v_gene, j_gene), for example.
            model = self.models_.get(split_key, None)
            if model is None:
                # These sequences had no compatible model. Skip
                continue
            featurized_split = self._featurize_split(
                adata_split,
                gene_locus=self.gene_locus,
                target_obs_column=self.target_obs_column,
                sample_weight_strategy=self.sample_weight_strategy,
            )
            scores.append(
                pd.DataFrame(
                    model.predict_proba(
                        featurized_split.X,
                    ),
                    index=featurized_split.sample_names,
                    columns=model.classes_,
                )
            )

        # Combine across models
        combined_across_models = (
            pd.concat(scores, axis=0) if len(scores) > 0 else pd.DataFrame()
        )

        # Some models may have had fewer classes. Fill those in as 0s.
        combined_across_models = combined_across_models.reindex(
            columns=self.classes_
        ).fillna(0)

        # Return in order of input, with NaNs for sequences that were not scored by any model (i.e. they did not have a compatible V-J gene combination)
        return combined_across_models.reindex(index=dataset.obs.index)

    def featurize(self, dataset: anndata.AnnData) -> SubsetClassifierFeaturizedData:
        """Featurize.
        Unconventional: FeaturizedData.X will be an anndata, not a data_X matrix - because predict_proba() expects anndata.
        The other fields may contain more sequences than survive the prediction process (i.e. some may be abstained on because there are no V/J-gene specific models compatible with these sequences).
        """
        return SubsetClassifierFeaturizedData(
            X=dataset,  # Note: violates type of normal FeaturizedData
            y=dataset.obs[self.target_obs_column.value.obs_column_name],
            sample_names=dataset.obs_names,
            metadata=dataset.obs,
            sample_weights=SequenceClassifier._get_sequence_weights(
                dataset, self.gene_locus, self.sample_weight_strategy
            ),
        )

    @classmethod
    def generate_training_chunks(
        cls,
        full_anndata: anndata.AnnData,
        return_copies: bool = False,
        skip_chunks: Optional[List[SplitKeyType]] = None,
    ) -> Generator[Tuple[SplitKeyType, anndata.AnnData], None, None]:
        # If return_copies is True (default False), returns copied anndatas instead of views.
        chunks_generator = helpers.anndata_groupby_obs(
            full_anndata,
            cls.split_on,
            return_copies=return_copies,
            observed=True,
            sort=False,
        )
        # Optionally skip chunks, by creating a derived generator that filters out the skipped chunks
        if skip_chunks is not None:
            # Use a generator expression to filter out the chunks to be skipped
            # This is still a lazy-executed generator.
            chunks_generator = (
                (chunk_key, anndata_chunk)
                for chunk_key, anndata_chunk in chunks_generator
                if chunk_key not in skip_chunks
            )
        yield from chunks_generator

    @classmethod
    def n_training_chunks(cls, full_anndata: anndata.AnnData) -> int:
        return full_anndata.obs.groupby(cls.split_on, observed=True).ngroups

    @classmethod
    def generate_training_chunk_hashes(
        cls, full_anndata: anndata.AnnData
    ) -> dict[SplitKeyType, str]:
        # Get group keys
        group_keys: list[SplitKeyType] = list(
            full_anndata.obs.groupby(cls.split_on, observed=True).groups.keys()
        )

        # Convert to filename strings
        def join_group_key(group_key: SplitKeyType) -> str:
            if isinstance(group_key, str):
                return group_key
            else:
                return "_".join(group_key)

        hashes = {
            group_key: slugify(join_group_key(group_key)) for group_key in group_keys
        }

        # Verify no collisions
        if len(set(hashes.values())) != len(hashes):
            raise ValueError("Filename hashes are not unique")

        return hashes


class VGeneSpecificSequenceClassifier(VJGeneSpecificSequenceClassifier):
    """
    V gene specific sequence classifier.
    Modifies the V-J gene splitting classifier to only split by V gene.
    """

    split_on: List[str] = ["v_gene"]
    split_short_name: str = "v_gene_specific"


class VGeneIsotypeSpecificSequenceClassifier(VJGeneSpecificSequenceClassifier):
    """
    Split by V gene and isotype.
    """

    split_on: List[str] = ["v_gene", "isotype_supergroup"]
    split_short_name: str = "vgene_isotype_specific"


class VFamilyIsotypeSpecificSequenceClassifier(VJGeneSpecificSequenceClassifier):
    """
    Split by V family and isotype.
    """

    split_on: List[str] = ["v_family", "isotype_supergroup"]
    split_short_name: str = "vfamily_isotype_specific"


class SequenceSubsetStrategy(ValidatableEnumMixin, Enum):
    split_VJ = auto()
    split_V = auto()
    split_Vgene_and_isotype = auto()
    split_Vfamily_and_isotype = auto()

    @property
    def base_model(self) -> Type[VJGeneSpecificSequenceClassifier]:
        """
        Returns base classifier type:
        either Type[VJGeneSpecificSequenceClassifier] or a child class type
        """
        if self == SequenceSubsetStrategy.split_VJ:
            return VJGeneSpecificSequenceClassifier
        elif self == SequenceSubsetStrategy.split_V:
            return VGeneSpecificSequenceClassifier
        elif self == SequenceSubsetStrategy.split_Vgene_and_isotype:
            return VGeneIsotypeSpecificSequenceClassifier
        elif self == SequenceSubsetStrategy.split_Vfamily_and_isotype:
            return VFamilyIsotypeSpecificSequenceClassifier
        else:
            raise ValueError(f"Unknown SequenceSubsetStrategy: {self}")
