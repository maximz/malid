import dataclasses
import logging
from pathlib import Path
from typing import Optional, List

import anndata
import genetools
import joblib
import numpy as np
import pandas as pd
from enum import Enum, auto
from enum_mixins import ValidatableEnumMixin

from malid import helpers
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
)
from extendanything import ExtendAnything
from crosseval import FeaturizedData
from malid.external.standard_scaler_that_preserves_input_type import (
    StandardScalerThatPreservesInputType,
)
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)
from malid.trained_model_wrappers import (
    VJGeneSpecificSequenceClassifier,
)
from malid.trained_model_wrappers.rollup_sequence_classifier import (
    _compute_maximum_entropy_cutoff,
    _entropy_threshold,
    _trim_bottom_only,
    _weighted_median,
)
from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SequenceSubsetStrategy,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=False)
class SubsetRollupClassifierFeaturizedData(FeaturizedData):
    # Overload FeaturizedData for better type hinting:
    # X has less fleixlbe type than in typical FeaturizedData
    X: pd.DataFrame


# TODO(Python 3.10): Subclass FeaturizedData to define the "extras" structure more explicitly than the base class does.
# @dataclasses.dataclass(eq=False)
# class FeaturizedDataExtrasForVJGeneSpecificSequenceModelRollupClassifier:
#     fill_missing: bool
#     standardization_transformer_before_reweighing_by_subset_frequencies: Optional[StandardScalerThatPreservesInputType]
#
# @dataclasses.dataclass(eq=False)
# class FeaturizedDataForVJGeneSpecificSequenceModelRollupClassifier(FeaturizedData):
#     extras: FeaturizedDataExtrasForVJGeneSpecificSequenceModelRollupClassifier


class AggregationStrategy(ValidatableEnumMixin, Enum):
    # Weighted aggregation strategies.
    mean = auto()
    median = auto()
    trim_bottom_five_percent = auto()
    entropy_ten_percent_cutoff = auto()
    entropy_twenty_percent_cutoff = auto()

    @property
    def model_name_suffix(self) -> str:
        # Companion to ModelNameSuffixes Enum.
        # e.g. _mean_aggregated
        return f"_{self.name}_aggregated"


class VJGeneSpecificSequenceModelRollupClassifier(
    ImmuneClassifierMixin, ExtendAnything
):
    # ExtendAnything means will pass through to base instance's attributes, exposing the base model's classes_, predict(), predict_proba(), etc.

    sequence_classifier: VJGeneSpecificSequenceClassifier  # or a child class

    def __init__(
        self,
        fold_id: int,
        # Important distinction: Base model vs rollup model may have different names and likely different fold labels
        base_sequence_model_name: str,
        rollup_model_name: str,
        base_model_train_fold_label: str,  # What was the base model trained on?
        fold_label_train: str,  # Train fold label for this rollup model
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        # To make this model clearer for downstream users, we're making sample_weight_strategy optional since the default SampleWeightStrategy.ISOTYPE_USAGE doesn't have any impact on the default SequenceSubsetStrategy.split_Vgene_and_isotype
        # self.sample_weight_strategy will get set to config.sample_weight_strategy in ImmuneClassifierMixin's constructor.
        sample_weight_strategy: Optional[SampleWeightStrategy] = None,
        sequence_models_base_dir: Optional[Path] = None,
        # The default is to split by V gene and isotype, but other strategies are available, and the class is still named after the original attempt to split by V gene and J gene.
        sequence_subset_strategy: SequenceSubsetStrategy = SequenceSubsetStrategy.split_Vgene_and_isotype,
        # Optionally provide the sequence classifier directly. Otherwise it will be loaded through the constructor corresponding to the sequence_subset_strategy parameter.
        sequence_classifier: Optional[
            VJGeneSpecificSequenceClassifier
        ] = None,  # or a child class
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
        )

        # 2. Load base model, which is necessary to figure out our rollup model's models_base_dir.
        # Load base model if not provided
        if sequence_classifier is not None:
            self.sequence_classifier = sequence_classifier
            # Validation:
            if (
                self.sequence_classifier.model_name_sequence_disease
                != base_sequence_model_name
            ):
                raise ValueError(
                    f"Provided sequence classifier has model name {self.sequence_classifier.model_name_sequence_disease}, but base_sequence_model_name is {base_sequence_model_name}"
                )
            if self.sequence_classifier.fold_label_train != base_model_train_fold_label:
                raise ValueError(
                    f"Provided sequence classifier has fold label {self.sequence_classifier.fold_label_train}, but base_model_train_fold_label is {base_model_train_fold_label}"
                )
        else:
            # The SequenceSubsetStrategy parameter in the constructor determines whether we load by default from VJGeneSpecificSequenceClassifier or from a child class.
            self.sequence_classifier = sequence_subset_strategy.base_model(
                fold_id=fold_id,
                # Pass base model name and base model train fold here
                model_name_sequence_disease=base_sequence_model_name,
                fold_label_train=base_model_train_fold_label,
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
                models_base_dir=sequence_models_base_dir,
            )

        # 3. Now that self.sequence_classifier is available, our models_base_dir is also available.
        # Construct the file path.
        fname = (
            self.models_base_dir
            / f"{self.fold_label_train}_model.{rollup_model_name}.{self.fold_id}.joblib"
        )

        # 4. Call ExtendAnything's constructor to load and wrap classifier
        # self._inner will now be the loaded model, and its attributes will be exposed (pass-through)
        ExtendAnything.__init__(self, inner=joblib.load(fname))

        # Set other attributes.
        self.rollup_model_name = rollup_model_name

        # Don't need to store base_sequence_model_name and base_model_train_fold_label:
        # They should be available from the sequence_classifier object:
        # base_sequence_model_name == self.sequence_classifier.model_name_sequence_disease
        # base_model_train_fold_label == self.sequence_classifier.fold_label_train

        # Load and unpack model settings dictionary from disk, which was created during train procedure using make_model_settings().
        # (Do all loading like this up front in init(), rather than deferred in featurize() or elsewhere, because we may pickle out this object when saving a metamodel and want to make sure everything is bundled in.)
        self.model_settings = joblib.load(
            self.models_base_dir
            / f"{self.fold_label_train}_model.{self.rollup_model_name}.{self.fold_id}.settings_joblib"
        )
        # Determine whether this model expects missing values to be filled in featurization. True for most model names except for some special ones like xgboost_with_nans.
        # If are_nans_present is True, then fill_missing should be False.
        self.fill_missing: bool = not self.model_settings["are_nans_present"]
        # Extract other model settings necessary for featurization:
        self.reweigh_by_subset_frequencies: bool = self.model_settings[
            "reweigh_by_subset_frequencies"
        ]
        self.standardization_transformer_before_reweighing_by_subset_frequencies: Optional[
            StandardScalerThatPreservesInputType
        ] = self.model_settings[
            "standardization_transformer_before_reweighing_by_subset_frequencies"
        ]
        self.aggregation_strategy = AggregationStrategy[
            # Get enum value by name
            self.model_settings["aggregation_strategy_name"]
        ]

    @staticmethod
    def make_model_settings(
        are_nans_present: bool,
        reweigh_by_subset_frequencies: bool,
        standardization_transformer_before_reweighing_by_subset_frequencies: Optional[
            StandardScalerThatPreservesInputType
        ],
        aggregation_strategy: AggregationStrategy,
    ) -> dict:
        """Create dictionary containing model settings to be saved to disk."""
        # See how these are reloaded from disk in constructor above.
        return dict(
            are_nans_present=are_nans_present,
            reweigh_by_subset_frequencies=reweigh_by_subset_frequencies,
            standardization_transformer_before_reweighing_by_subset_frequencies=standardization_transformer_before_reweighing_by_subset_frequencies,
            aggregation_strategy_name=aggregation_strategy.name,
        )

    @staticmethod
    def _get_directory_suffix(
        base_sequence_model_name: str,
        base_model_train_fold_label: str,
        split_short_name: str,  # e.g. "vj_gene_specific"
    ) -> Path:
        # Shared between _get_model_base_dir() and _get_output_base_dir()
        return (
            Path(f"rollup_models_specialized_for_{split_short_name}")
            / f"base_model_{base_sequence_model_name}_trained_on_{base_model_train_fold_label}"
        )

    # Directory containing rollup models:
    @classmethod
    def _get_model_base_dir(
        cls,
        sequence_models_base_dir: Path,
        base_sequence_model_name: str,
        base_model_train_fold_label: str,
        split_short_name: str,  # e.g. "vj_gene_specific"
    ) -> Path:
        """Pass base model name and base model train fold here."""
        return sequence_models_base_dir / cls._get_directory_suffix(
            base_sequence_model_name=base_sequence_model_name,
            base_model_train_fold_label=base_model_train_fold_label,
            split_short_name=split_short_name,
        )

    @property
    def models_base_dir(self):
        # overrides default implementation on ImmuneClassifierMixin,
        # because _get_model_base_dir signature is different,
        # and because we don't allow user to pass in an override.
        return self._get_model_base_dir(
            sequence_models_base_dir=self.sequence_classifier.models_base_dir,
            base_sequence_model_name=self.sequence_classifier.model_name_sequence_disease,
            base_model_train_fold_label=self.sequence_classifier.fold_label_train,
            split_short_name=self.sequence_classifier.split_short_name,
        )

    @property
    def output_base_dir(self) -> Path:
        return self._get_output_base_dir(
            sequence_model_output_base_dir=self.sequence_classifier.output_base_dir,
            base_sequence_model_name=self.sequence_classifier.model_name_sequence_disease,
            base_model_train_fold_label=self.sequence_classifier.fold_label_train,
            split_short_name=self.sequence_classifier.split_short_name,
        )

    @classmethod
    def _get_output_base_dir(
        cls,
        sequence_model_output_base_dir: Path,
        base_sequence_model_name: str,
        base_model_train_fold_label: str,
        split_short_name: str,  # e.g. "vj_gene_specific"
    ) -> Path:
        """Pass base model name and base model train fold here."""
        return sequence_model_output_base_dir / cls._get_directory_suffix(
            base_sequence_model_name=base_sequence_model_name,
            base_model_train_fold_label=base_model_train_fold_label,
            split_short_name=split_short_name,
        )

    @staticmethod
    def _generate_sequence_predictions(
        gene_subset_specific_sequence_model: VJGeneSpecificSequenceClassifier,  # or a child class
        adata: anndata.AnnData,
    ) -> pd.DataFrame:
        """
        Generate sequence predictions for all sequences in the dataset.
        Concatenate predictions along with original sequence category info
        Note: the predictions will be returned as NaNs for sequences whose V/J gene pair is not present as a submodel within the gene_subset_specific_sequence_model collection.

        This is called from _featurize() below. But if we are going to run _featurize several times with different options, it is more efficient to call this function manually, store the results, and pass to all _featurize() calls.
        """
        sequence_featurized_data = gene_subset_specific_sequence_model.featurize(adata)

        weights = sequence_featurized_data.sample_weights
        if weights is None:
            # Force sample weight column to be float NaN (avoid pandas FutureWarning)
            weights = np.nan
        if not isinstance(weights, pd.Series):
            # Convert to pandas Series
            weights = pd.Series(weights, index=adata.obs_names, name="weight")

        preds = pd.concat(
            [
                gene_subset_specific_sequence_model.predict_proba(
                    sequence_featurized_data.X
                ),
                adata.obs["specimen_label"],
                weights.rename("weight"),
            ]
            + [
                # Inherit the split criteria from the sequence model
                adata.obs[split_col]
                for split_col in gene_subset_specific_sequence_model.split_on
            ],
            axis=1,
        )
        return preds

    @staticmethod
    def _normalize_row_sum_within_each_isotype(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pass in a dataframe of subset frequencies in this format:
            Index: specimen_label
            Columns: multi-index of (v_gene, isotype_supergroup)

        Make rows sum to 1 within each isotype_supergroup.

        (This is not an inplace operation. It returns a new dataframe.)
        """
        subset_frequencies_norm_special = df.copy()
        for (
            isotype_supergroup
        ) in subset_frequencies_norm_special.columns.get_level_values(
            "isotype_supergroup"
        ).unique():
            # For e.g. IgA, divide each IgA column by the sum of all IgA columns
            # Repeat for all isotypes
            subset_frequencies_norm_special.loc[
                :, pd.IndexSlice[:, isotype_supergroup]
            ] = subset_frequencies_norm_special.loc[
                :, pd.IndexSlice[:, isotype_supergroup]
            ].div(
                subset_frequencies_norm_special.loc[
                    :, pd.IndexSlice[:, isotype_supergroup]
                ].sum(axis=1),
                axis=0,
            )
        # Each isotype sums to 1. For BCR there are three isotypes. So subset_frequencies_norm_special.sum(axis=1) will overall sum to 3 per row.
        return subset_frequencies_norm_special

    @classmethod
    def _featurize(
        cls,
        data: anndata.AnnData,
        gene_subset_specific_sequence_model: VJGeneSpecificSequenceClassifier,  # or a child class
        fill_missing: bool,
        aggregation_strategy: AggregationStrategy,  # .mean, .median, etc
        feature_order: Optional[List[str]] = None,
        reweigh_by_subset_frequencies: bool = False,
        # standardization_transformer_before_reweighing_by_subset_frequencies is ignored (but passed through) unless reweigh_by_subset_frequencies is True
        standardization_transformer_before_reweighing_by_subset_frequencies: Optional[
            StandardScalerThatPreservesInputType
        ] = None,
        # If calling _featurize() repeatedly, we have the option of passing _generate_sequence_predictions() output in here to avoid repeated work:
        pregenerated_sequence_predictions: Optional[pd.DataFrame] = None,
    ) -> SubsetRollupClassifierFeaturizedData:
        """
        We have split sequences by V and J gene and trained a sequence model for each subset.
        Now we will aggregate sequence-level predictions to a specimen prediction.
        Previous approach: combine all models as if their probabilities are on the same scale
        This approach: train a second-stage model with a #target_classes * #models vector per specimen:

            X = [
                [average predicted class probabilities for all V1,J1 sequences from this individual],
                [average predicted class probabilities for all V2,J1 sequences from this individual],
                ...
            ]
            y = target class.

        Later, we can check which V/J genes are important to this second-stage model.

        _featurize produces a vector per specimen of length n_classes x n_splits.
        The second-stage model maps this specimen-level feature vector to the specimen-level target class lael.

        How it works:
        - Given a specimen with a bunch of sequences, split into subsets by V,J gene.
        - Run each subset through the corresponding sequence model with the same V,J gene.
        - Average the predicted class probabilities within each subsets ==> n_classes-dim vector for each subset. (Averaging is performed if aggregation_strategy=AggregationStrategy.mean; other options are available.)
        - Concatenate all the n_classes-dim vectors into one row per specimen.
        - Train a second stage model with that featurization as input.

        Note that column order is not deterministic.
        For example, column order may change if a specimen lacks a particular V/J gene pair.
        """
        GeneLocus.validate_single_value(gene_subset_specific_sequence_model.gene_locus)
        TargetObsColumnEnum.validate(
            gene_subset_specific_sequence_model.target_obs_column
        )
        SampleWeightStrategy.validate(
            gene_subset_specific_sequence_model.sample_weight_strategy
        )

        if pregenerated_sequence_predictions is None:
            sequence_predictions = cls._generate_sequence_predictions(
                gene_subset_specific_sequence_model=gene_subset_specific_sequence_model,
                adata=data,
            )
        else:
            sequence_predictions = pregenerated_sequence_predictions

        # Average predictions within each specimen, subset by V-J group
        mean_by_specimen_and_vgene, column_map = cls._featurize_sequence_predictions(
            sequence_predictions,
            split_on=gene_subset_specific_sequence_model.split_on,
            aggregation_strategy=aggregation_strategy,
        )

        if mean_by_specimen_and_vgene.isna().all().any():
            # Sanity check: Confirm there are no columns of *all* NaNs.
            # If any V-J group was missing from the provided submodels in gene_subset_specific_sequence_model, then mean_by_specimen_and_vgene will have columns of all NaNs (the corresponding predictions for those V-J groups).
            #
            # To elaborate:
            # - Columns of all NaNs would occur in the situation where there are sequence subsets (e.g. V-J gene pairs) for which we have no trained sequence model.
            # - In that situation, there will be no predicted probabilities for those anndata sequence entries in any sample. So we'd have a column of all NaNs in this aggregated dataframe.
            #
            # However, _featurize_sequence_predictions now takes care of this edge case by dropping any original sequence rows that had no sequence probabilities generated, i.e. sequence rows where there was no submodel available for that sequence subset so there was nothing to aggregate.
            #
            # (Note: mean_by_specimen_and_vgene can still have columns with *some* but not all NaNs. That's expected if a specimen does not have a particular gene subset - no sequence scores there to aggregate - but the gene subset exists and was scored in other specimens.)
            raise ValueError(
                "There should not be any columns of all NaNs after aggregation procedure"
            )

        if feature_order is not None:
            # If training feature_order provided:
            # Subset to V-J gene pairs that are present in the training data, and add any missing V-J gene pairs with N/A predictions.

            # Rationale:
            # We don't want the model to generate features for all possible V-J gene pairs it has submodels for,
            # just the gene pairs that are actually in the training data. And by passing feature order to test featurize step, we can apply consistent fill N/A logic below.

            mean_by_specimen_and_vgene = mean_by_specimen_and_vgene.reindex(
                columns=feature_order, fill_value=np.nan
            )

        # Adjust column map accordingly after possibly dropping N/A columns and reindexing
        column_map = {
            k: v
            for k, v in column_map.items()
            if k in mean_by_specimen_and_vgene.columns
        }

        # The column_map indicates how to map the mean_by_specimen_and_vgene string column names back to the original class_name and split_on property values.
        # For example, it translates Healthy_V1_J1 back to its original meaning: P(Healthy|V1,J1).
        # column_map_df: row names are column names of mean_by_specimen_and_vgene; column names are class_name and split_on parameters
        column_map_df = pd.DataFrame.from_dict(column_map, orient="index")

        # Fill N/A predictions, which can arise if a specimen lacks a particular V/J gene pair.
        # First, report number of N/As:
        num_na = mean_by_specimen_and_vgene.isna().sum().sum()
        num_total = (
            mean_by_specimen_and_vgene.shape[0] * mean_by_specimen_and_vgene.shape[1]
        )
        logger.info(
            f"Number of VJGeneSpecificSequenceModelRollupClassifier featurization matrix N/As due to specimens not having any sequences with particular V/J gene pairs: {num_na} / {num_total} = {num_na / num_total:.2%}"
        )
        missing_fill_value = 1.0 / len(gene_subset_specific_sequence_model.classes_)
        if fill_missing:
            mean_by_specimen_and_vgene.fillna(missing_fill_value, inplace=True)
        # TODO: Switch to tree models that support N/As

        if reweigh_by_subset_frequencies:
            ## Reweigh by split_on frequencies within specimen.
            if not fill_missing:
                # Undefined for now: How does this reweigh-by-frequencies path interact with NaNs if not fill_missing?
                # For now, we will restrict to fill_missing True only.
                raise ValueError("reweigh_by_subset_frequencies requires fill_missing")

            # 1) we need to scale before reweighing. (We'll also scale again after reweighing, because all our classifiers/models are really Pipelines that start with StandardScaler)
            if (
                standardization_transformer_before_reweighing_by_subset_frequencies
                is None
            ):
                # If a fitted transformer is not supplied, create and fit it here.
                standardization_transformer_before_reweighing_by_subset_frequencies = (
                    StandardScalerThatPreservesInputType().fit(
                        mean_by_specimen_and_vgene
                    )
                )
            mean_by_specimen_and_vgene = standardization_transformer_before_reweighing_by_subset_frequencies.transform(
                mean_by_specimen_and_vgene
            )

            # 2) Compute subset frequencies within each specimen, which we will use to reweigh the specimen-level predictions (the features we are preparing for the upcoming model).
            # These are raw counts, not normalized, for now.
            # Rows are indexed by specimen_label
            # Columns are indexed by split_on values. E.g. if split_on is V gene and J gene, columns are a multiindex with tuple values like (V1, J1) and multiindex names "v_gene", "j_gene"
            subset_frequencies = (
                data.obs.groupby(
                    ["specimen_label"] + gene_subset_specific_sequence_model.split_on,
                    observed=True,
                )
                .size()
                .unstack(gene_subset_specific_sequence_model.split_on, fill_value=0)
            )

            # Reindex to match mean_by_specimen_and_vgene specimen order
            subset_frequencies = subset_frequencies.loc[
                mean_by_specimen_and_vgene.index
            ]

            # Now drop any subset frequency columns that are not present in mean_by_specimen_and_vgene.
            columns_that_are_present = [
                column_tuple
                for column_tuple in subset_frequencies.columns
                if (
                    column_map_df[
                        gene_subset_specific_sequence_model.split_on
                    ].drop_duplicates()
                    == column_tuple
                )
                .all(axis=1)
                .any()
            ]
            normalize_after_subsetting = True
            # Later, consider moving the `subset_frequencies = subset_frequencies[columns_that_are_present]` line to after the normalization operation below.
            # This is scaffolded with normalize_after_subsetting, even though this is not wired yet up into training and downstream loading+featurizing.
            # Rationale for normalizing first, then dropping rare V genes after:
            # We believe healthy people have a baseline V gene usage distribution, and sick people have a peaky/concentrated distribution relative to the healthy baseline.
            # Imagine 4 V genes. Suppose a healthy person has 1 sequence from each of the 4 V genes, but a sick person has 2 sequences each from only 2 of the V genes.
            # So the usage proportions are 25%-25%-25%-25% and 50%-50%-0%-0%.
            # The last two V genes are going to be dropped for being rare.
            # So the final data is either 25%-25% and 50%-50% if we normalize_rows before subsetting to the kept V genes, or 50%-50% and 50%-50% if we normalize_rows after subsetting to the kept V genes.
            # Notice how the people are indistiguishable in the latter case.
            # However, in practice we don't think this will be as important - we will leave this as a future improvement.
            # We should also try this in Model 1.
            if normalize_after_subsetting:
                # The default option is to drop columns before normalizing.
                subset_frequencies = subset_frequencies[columns_that_are_present]

            # Then normalize remainder to sum to 1.
            # This normalization should be done separately *within each isotype*.
            # The reason why is that isotype usage frequencies are technical artifacts of the sample preparation and sequencing process, whereas V gene usage is a biological signal.
            # So we should not just do: subset_frequencies = genetools.stats.normalize_rows(subset_frequencies)
            if (
                gene_subset_specific_sequence_model.split_on
                != subset_frequencies.columns.names
            ):
                # Sanity check:
                # The split_on property of the sequence model should match the multi-index level names of the columns of subset_frequencies.
                # This will also be caught in _normalize_row_sum_within_each_isotype if we access an invalid level name.
                raise ValueError(
                    f"Unexpected: split_on property of sequence model {gene_subset_specific_sequence_model.split_on} does not match the multi-index level names of the columns of subset_frequencies {subset_frequencies.columns.names}"
                )
            if "isotype_supergroup" in gene_subset_specific_sequence_model.split_on:
                # If we are splitting by isotype, normalize counts within each isotype.
                # This normalization should be done separately *within each isotype* because isotype usage frequencies are technical artifacts of the sample preparation and sequencing process, whereas V gene usage is a biological signal.
                subset_frequencies = cls._normalize_row_sum_within_each_isotype(
                    subset_frequencies
                )
            else:
                # We are not splitting by isotype.
                # Normalize rows (specimens) to sum to 1, without regard for isotype.
                subset_frequencies = genetools.stats.normalize_rows(subset_frequencies)

            if not normalize_after_subsetting:
                # Alternative: wait until after normalization to drop any subset frequency columns that are not present in mean_by_specimen_and_vgene (i.e. dropping the rare V genes for which no models were trained).
                # This code path is skipped for now - just scaffolded for later.
                subset_frequencies = subset_frequencies[columns_that_are_present]

            # 3) Reweigh by subset frequencies within each specimen.
            # Iterate over the columns of subset_frequencies:
            for subset_tuple in subset_frequencies.columns:
                frequencies_of_this_subset_in_all_specimens = subset_frequencies[
                    subset_tuple
                ]

                # Get all corresponding mean_by_specimen_and_vgene columns related to this subset
                # For example, for subset V1,J1, get columns "Covid19_V1_J1", "HIV_V1_J1", and so on.
                related_column_names = column_map_df[
                    column_map_df.apply(
                        lambda row: tuple(
                            row[gene_subset_specific_sequence_model.split_on].values
                        )
                        == subset_tuple,
                        axis=1,
                    )
                ].index

                for col in related_column_names:
                    # Reweigh each corresponding column of mean_by_specimen_and_vgene, with a different weight applied for each specimen, based on that subset's frequency in that specimen.
                    # These are pandas multiplies, so they will be aligned by specimen_label index, but just in case, we make it explicit with a .loc.
                    # Note that mean_by_specimen_and_vgene[col] may have np.nan entries if we don't fill missing values in mean_by_specimen_and_vgene. After element-wise multiplication, they will stay nan.
                    mean_by_specimen_and_vgene[
                        col
                    ] *= frequencies_of_this_subset_in_all_specimens.loc[
                        mean_by_specimen_and_vgene.index
                    ]

            # Done. The scaler and reweighing-column-map will be stored as part of the model settings to be reloaded from disk for featurization in future (see make_model_settings()).

        specimen_metadata = helpers.extract_specimen_metadata_from_anndata(
            adata=data,
            gene_locus=gene_subset_specific_sequence_model.gene_locus,
            target_obs_column=gene_subset_specific_sequence_model.target_obs_column,
        ).loc[mean_by_specimen_and_vgene.index]

        return SubsetRollupClassifierFeaturizedData(
            X=mean_by_specimen_and_vgene,
            y=specimen_metadata[
                gene_subset_specific_sequence_model.target_obs_column.value.obs_column_name
            ],
            metadata=specimen_metadata,
            sample_names=specimen_metadata.index,
            extras={
                "fill_missing": fill_missing,
                "missing_fill_value": missing_fill_value,
                "split_on": gene_subset_specific_sequence_model.split_on,
                "column_map": column_map,
                "reweigh_by_subset_frequencies": reweigh_by_subset_frequencies,
                "standardization_transformer_before_reweighing_by_subset_frequencies": standardization_transformer_before_reweighing_by_subset_frequencies,
            },
        )

    @staticmethod
    def _fill_missing_later(
        featurized_data: SubsetRollupClassifierFeaturizedData,
    ) -> SubsetRollupClassifierFeaturizedData:
        """
        Fill missing values in FeaturizedData object produced by VJGeneSpecificSequenceModelRollupClassifier.
        Use this if we had set fill_missing=False originally, and now want to apply fill_missing=True.
        """
        return dataclasses.replace(
            featurized_data,
            X=featurized_data.X.fillna(featurized_data.extras["missing_fill_value"]),
        )

    @staticmethod
    def _featurize_sequence_predictions(
        sequence_predicted_probas: pd.DataFrame,
        # Pass the split criteria from the sequence model, e.g. ['v_gene', 'j_gene']
        split_on: List[str],
        aggregation_strategy: Optional[AggregationStrategy] = AggregationStrategy.mean,
    ) -> tuple[pd.DataFrame, dict[str, dict]]:
        """
        Transform predicted probabilities dataframe.

        Input: sequences x classes dataframe (raw sequence-level predicted probabilities from the V-J gene specific submodel corresponding to each sequence's V gene and J gene)
        Also includes columns for specimen_label and for each split_on property (e.g. V gene, J gene)

        Output: specimens x (classes x V,J pairs) dataframe (mean predicted probabilities in a specimen, subset by V-J group).
        Column name format: f"{class_name}_{v_gene}_{j_gene}"
        Example column names: Covid19_V1_J1, Covid19_V2_J1, etc.
        """

        # Get the list of class names to aggregate. Exclude specimen_label, weight, and split_on columns.
        columns_to_aggregate = list(
            set(sequence_predicted_probas.columns)
            - set(["specimen_label", "weight"])
            - set(split_on)
        )

        # First, drop any rows where all columns to aggregate are NaN.
        # This can happen if a specimen has a particular sequence subset for which we have no trained sequence model.
        # In that situation, there will be no predicted probabilities for those anndata sequence entries.
        # We will drop those rows because there are no sequence probabilities to aggregate for that sequence subset.
        # Drop rows where all the columns to aggregate are NaN:
        # (note that this creates a view, not a copy)
        sequence_predicted_probas = sequence_predicted_probas.dropna(
            subset=columns_to_aggregate, how="all"
        )
        # This means we don't need to handle NaNs in our aggregation functions explicitly.
        # In fact, let's follow up with a sanity check that there are no NaNs left. (So far we only dropped rows of all-NaNs, not any-NaNs)
        if sequence_predicted_probas[columns_to_aggregate].isna().any().any():
            raise ValueError(
                f"There should not be any NaNs remaining in sequence_predicted_probas for columns {columns_to_aggregate}. There should be sequence submodels to create scores for all of these sequences."
            )

        # We will perform weighted aggregation with weights based on sequence-level sample weights.
        # For example, those may be ISOTYPE_USAGE weights which would matter if our model split strategy does not split on isotype_supergroup.
        # If *all* weights are NaN, fill weights with 1.
        if sequence_predicted_probas["weight"].isna().all():
            # Avoid SettingWithCopyWarning:
            # sequence_predicted_probas may be a view after dropna() above,
            # so don't do this directly: sequence_predicted_probas["weight"] = 1
            sequence_predicted_probas = sequence_predicted_probas.assign(weight=1)

        # Confirm we don't have partially-NaN weights:
        if sequence_predicted_probas["weight"].isna().any():
            raise ValueError(
                "Cannot perform weighted aggregation if some sequence-level weights are not provided while others are."
            )

        # We are going to do a groupby-apply-unstack operation, but here it's split up over multiple lines for clarity. We'll go ahead and call the final variable means_unstacked already though.
        means_unstacked = sequence_predicted_probas.groupby(
            ["specimen_label"] + split_on,
            # Use observed=True to only include real V-J gene combinations
            # (Note that we don't need to worry about whether some of VJGeneSpecificSequenceClassifier's submodels don't predict all classes.
            # VJGeneSpecificSequenceClassifier.predict_proba already compensates for that, ensuring all class predictions are present, filling N/As with 0.)
            observed=True,
        )

        # Perform aggregation.
        # After this .mean() or similar operation, we will have a dataframe with:
        # - rows identified by (specimen_label, v_gene, j_gene) if split_on is V gene and J gene
        # - columns identified by class name
        AggregationStrategy.validate(aggregation_strategy)
        if aggregation_strategy == AggregationStrategy.mean:
            # OLD: unweighted mean:
            # Select columns_to_aggregate so that we don't aggregate the weights column. We essentially ignore and drop the weights column.
            # means_unstacked = means_unstacked[columns_to_aggregate].mean()

            # Weighted mean:
            # Use groupby-apply with numpy's average function for the weighted mean calculation
            means_unstacked = means_unstacked.apply(
                lambda grp: pd.Series(
                    np.average(
                        grp[columns_to_aggregate], weights=grp["weight"], axis=0
                    ),
                    index=columns_to_aggregate,
                )
            )

            # Slower alternative: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#named-aggregation
            # weighted_mean_func = lambda x: np.average(x, weights=df.loc[x.index, "weight"])
            # result = df.groupby(["specimen_label", "v_gene", "j_gene"], observed=True).agg(
            #     **{col: (col, weighted_mean_func) for col in cols_to_average}
            # )

        elif aggregation_strategy == AggregationStrategy.median:
            # OLD: unweighted median:
            # Select columns_to_aggregate so that we don't aggregate the weights column. We essentially ignore and drop the weights column.
            # means_unstacked = means_unstacked[columns_to_aggregate].median()

            # Weighted median:
            means_unstacked = means_unstacked.apply(
                lambda grp: _weighted_median(
                    seq_probs=grp[columns_to_aggregate],
                    seq_weights=grp["weight"],
                    normalize_final_probabilities_to_sum_to_1=False,
                ),
            )

        elif aggregation_strategy == AggregationStrategy.trim_bottom_five_percent:
            # Trim bottom 5% of probabilities, while respecting the weights
            means_unstacked = means_unstacked.apply(
                lambda grp: _trim_bottom_only(
                    seq_probs=grp[columns_to_aggregate],
                    seq_weights=grp["weight"],
                    proportiontocut=0.05,
                    normalize_final_probabilities_to_sum_to_1=False,
                ),
            )

        elif aggregation_strategy in [
            AggregationStrategy.entropy_ten_percent_cutoff,
            AggregationStrategy.entropy_twenty_percent_cutoff,
        ]:
            # 10% or 20% reduction
            max_entropy_cutoff = _compute_maximum_entropy_cutoff(
                n_classes=len(columns_to_aggregate)
            )
            reduction_factor = (
                0.9
                if aggregation_strategy
                == AggregationStrategy.entropy_ten_percent_cutoff
                else 0.8
            )
            reduced_cutoff = max_entropy_cutoff * reduction_factor
            means_unstacked = means_unstacked.apply(
                lambda grp: _entropy_threshold(
                    seq_probs=grp[columns_to_aggregate],
                    seq_weights=grp["weight"],
                    max_entropy=reduced_cutoff,
                    normalize_final_probabilities_to_sum_to_1=False,
                )
            )
        else:
            raise ValueError(
                f"Unsupported aggregation_strategy: {aggregation_strategy}"
            )

        # Unstack.
        means_unstacked = (
            # Rename the column index to be "class_name". Used below when we work with .columns.names after unstacking (.columns.names[0] will be "class_name").
            means_unstacked.rename_axis("class_name", axis="columns")
            #
            # After this .unstack() operation, we have a dataframe with:
            # - rows identified by a single specimen_label
            # - columns identified by a (class_name, v_gene, j_gene) tuple if split_on is V gene and J gene
            # note that after this, .columns is now a MultiIndex, with .columns.names being ['class_name', 'v_gene', 'j_gene'] (if those are the split_on properties)
            .unstack(split_on)
        )

        # Combine column names, and store the mapping
        new_columns = means_unstacked.columns.to_series().apply("_".join)
        column_map = {}
        for new_column_name, column_tuple in zip(new_columns, means_unstacked.columns):
            # the column tuple contains the target class name (e.g. Covid19, HIV if disease is the target), followed by values for each split_on property (e.g. V gene, J gene)
            # this will produce a dict like {"class_name": "Covid19", "v_gene": "V1", "j_gene": "J1"} if those are the split_on properties
            column_map[new_column_name] = dict(
                zip(means_unstacked.columns.names, column_tuple)
            )
        # Sanity check length
        if len(new_columns) < len(means_unstacked.columns):
            # Our underscore join of split_on values resulted in duplicate column names / was not unique.
            raise ValueError(
                f"Duplicate column names detected when combining split_on values with underscores: {means_unstacked.columns}"
            )

        means_unstacked.columns = new_columns
        return means_unstacked, column_map

    def featurize(
        self,
        dataset: anndata.AnnData,
    ) -> SubsetRollupClassifierFeaturizedData:
        return self._featurize(
            data=dataset,
            gene_subset_specific_sequence_model=self.sequence_classifier,
            # Should we fill missing values in featurized data?
            fill_missing=self.fill_missing,
            # Which aggregation strategy should we use?
            aggregation_strategy=self.aggregation_strategy,
            # Impose consistent feature order
            feature_order=self.feature_names_in_,
            # Should we reweigh by subset frequencies within specimen?
            reweigh_by_subset_frequencies=self.reweigh_by_subset_frequencies,
            # standardization_transformer_before_reweighing_by_subset_frequencies is used only if reweigh_by_subset_frequencies is True:
            standardization_transformer_before_reweighing_by_subset_frequencies=self.standardization_transformer_before_reweighing_by_subset_frequencies,
        )
