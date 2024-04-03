import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import joblib
import numpy as np
import pandas as pd

from malid import config, helpers
from malid.datamodels import (
    GeneLocus,
    GeneralAnndataType,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    combine_classification_option_names,
)
from extendanything import ExtendAnything
from crosseval import FeaturizedData
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)

from fisher import pvalue_npy

logger = logging.getLogger(__name__)


class ExactMatchesClassifier(ImmuneClassifierMixin, ExtendAnything):
    """
    Wrapper around pre-trained exact-matches classification models.

    Reimplementation of exact match selection criteria from Emerson 2017.
    Sequences are selected as being disease-associated at a cross-validated p-value threshold.
    Specimens are featurized by their number of exact matches to these disease-associated sequence lists.

    We did not implement their generative model, but rather the standard discriminative classifier they mention as an alternative,
    because our problem is multiclass, not binary, and to be consistent with our other classifiers (we care about comparing the featurizations).

    To compare apples-to-apples / to be consistent with the convergent cluster classifier, we marked specimens that had 0 exact matches to disease-associated sequences as abstentions.
    These had no evidence of disease signal. It would be misleading to train the model that a row of all 0s is a disease specimen.
    Abstentions are penalized by the MCC metric used in the p-value tuning procedure.
    """

    # ExtendAnything means will pass through to base instance's attributes, exposing the base model's classes_, predict(), predict_proba(), etc.

    @staticmethod
    def _compute_fisher_scores_for_sequences(
        train_df: pd.DataFrame, target_obs_column: TargetObsColumnEnum
    ) -> pd.DataFrame:
        """
        Compute fisher scores for sequences in train_df.
        Returns DataFrame of fisher scores for sequences in train_df
        """
        TargetObsColumnEnum.validate(target_obs_column)

        # First create 2x2 contingency table, counting the number of unique participants that are/aren't in a disease class and have/don't have a particular V/J/CDR3
        # (For BCR, we are NOT considering CDR1 and CDR2 mutations)

        # We already know v_gene and j_gene are categorical - this is important for speed
        # Grouping by cdr3 length is redundant but reduces the search space for cdr3 sequence column.

        # df of sequence x disease classes. cells are unique participants with that sequence and that disease class.
        number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in = (
            train_df.groupby(
                [
                    "v_gene",
                    "j_gene",
                    "cdr3_aa_sequence_trim_len",
                    "cdr3_seq_aa_q_trim",
                    target_obs_column.value.obs_column_name,
                ],
                observed=True,
            )["participant_label"]
            .nunique()
            .unstack(fill_value=0)
        )

        # row sums
        count_sum = number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in.sum(
            axis=1
        )

        # total number of unique participants by class
        nunique_participants_by_class = train_df.groupby(
            [target_obs_column.value.obs_column_name]
        )["participant_label"].nunique()
        nunique_participants_total = nunique_participants_by_class.sum()

        # subtract
        num_unique_participants_in_each_class_that_dont_have_this_sequence = (
            nunique_participants_by_class
            - number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in
        )

        results = {}
        for (
            col
        ) in (
            number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in.columns
        ):

            # number of unique participants that are in this class and have this sequence
            right_class_and_have_sequence = number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in[
                col
            ]

            # number of unique participants that are in another class but have this sequence
            wrong_class_but_have_sequence = (
                count_sum
                - number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in[
                    col
                ]
            )

            # number of unique participants that are in this class but do not have this sequence
            # is the same as [total number of unique participants in this class] - [number of unique participants that are in this class and have this sequence]
            # right_class_but_dont_have_sequence = nunique_participants_by_class[col] - number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in[col]
            # same as:
            # right_class_but_dont_have_sequence = nunique_participants_by_class[col] - right_class_and_have_sequence
            # same as vectorized subtraction for all columns:
            right_class_but_dont_have_sequence = (
                num_unique_participants_in_each_class_that_dont_have_this_sequence[col]
            )

            # number of unique participants that are in another class and do not have this sequence
            # is the same as [total number of unique participants in another class] - [number of unique participants that are in another class and have this sequence]
            number_unique_participants_in_another_class = (
                nunique_participants_total - nunique_participants_by_class[col]
            )
            wrong_class_and_dont_have_sequence = (
                number_unique_participants_in_another_class
                - wrong_class_but_have_sequence
            )

            # Contigency table is:
            # ---------------------------------------------|---------------------- |
            # is other class |                                                     |
            # ----------------      # unique participants with this property       -
            # is this class  |                                                     |
            # ---------------|-----------------------------|---------------------- |
            #                | does not have this sequence | has this sequence --- |

            # Run fisher test - vectorized version
            # returns: "lefts, rights, twos"
            # Sticking to default (int64) gives error: Buffer dtype mismatch, expected 'uint_t' but got 'long'
            fisher_dtype = np.uint
            _, one_sided_p_value_right_tail, _ = pvalue_npy(
                wrong_class_and_dont_have_sequence.values.astype(fisher_dtype),
                wrong_class_but_have_sequence.values.astype(fisher_dtype),
                right_class_but_dont_have_sequence.values.astype(fisher_dtype),
                right_class_and_have_sequence.values.astype(fisher_dtype),
            )

            # sanity check shape
            assert (
                one_sided_p_value_right_tail.shape[0]
                == number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in.shape[
                    0
                ]
            )

            # save result
            results[col] = one_sided_p_value_right_tail

        # DataFrame of p-value for each disease (columns) for each unique sequence in train_df
        results_df = pd.DataFrame(
            results,
            index=number_of_unique_participants_by_disease_type_that_each_sequence_occurs_in.index,
        )
        # return in sorted column order
        return results_df[sorted(results_df.columns)]

    @staticmethod
    def _filter_fisher_scores_to_pvalue_threshold(
        results: pd.DataFrame, p_value_threshold: float
    ) -> pd.DataFrame:
        # optionally filter down if we have already decided on best p-val
        return results[results.min(axis=1) <= p_value_threshold]

    def featurize(self, dataset: GeneralAnndataType) -> FeaturizedData:
        """
        Pass adata.
        Make sure all data is from the same fold ID and fold label, and match the classifier's fold settings.
        For each patient, get total number of unique seqs, and total number of unique seqs that match each disease-specific list at or below a given P val threshold.
        """
        return self._featurize(
            df=dataset.obs,
            sequences_with_fisher_result=self.sequences_with_fisher_result,
            p_value_threshold=self.p_value_threshold,
            feature_order=self.feature_names_in_,
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
        )

    @classmethod
    def _featurize(
        cls,
        df: pd.DataFrame,
        sequences_with_fisher_result: pd.DataFrame,
        p_value_threshold: float,
        feature_order: List[str],
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
    ) -> FeaturizedData:
        """
        Pass adata.obs.
        Make sure all data is from the same fold ID and fold label, and match the classifier's fold settings.
        For each patient, get total number of unique seqs, and total number of unique seqs that match each disease-specific list at or below a given P val threshold.
        """
        # Dedupe sequences by specimen,
        # and create multi-index, which is used for further aggregation below.
        # The resulting dataframe has no non-index columns.
        df_deduped = (
            df[
                [
                    # We only need these columns.
                    "v_gene",
                    "j_gene",
                    "cdr3_aa_sequence_trim_len",
                    "cdr3_seq_aa_q_trim",
                    "specimen_label",
                ]
            ]
            .drop_duplicates()
            .set_index(
                [
                    "v_gene",
                    "j_gene",
                    "cdr3_aa_sequence_trim_len",
                    "cdr3_seq_aa_q_trim",
                    "specimen_label",
                ]
            )
        )

        # Merge score columns in
        df_deduped_scored = pd.merge(
            df_deduped,
            sequences_with_fisher_result,
            left_on=sequences_with_fisher_result.index.names,
            right_index=True,
            how="left",
            validate="m:1",
        )
        if df_deduped_scored.shape[0] != df_deduped.shape[0]:
            raise ValueError(
                f"Expected {df_deduped.shape[0]} rows after merge, but got {df_deduped_scored.shape[0]}"
            )

        # Note that all up to here is independent of the p-value threshold,
        # so we could optimize further by only doing the above once before looping over p-values for the below.

        # After merging:
        # Given a p-value, featurize to specimen-level feature vectors
        num_unique_sequences_per_specimen_that_are_predictive_of_a_class = (
            # This produces a dataframe indexed by the same multi-index as df_deduped_scored,
            # with columns named the same as sequences_with_fisher_result.columns,
            # and entries are all True or False.
            (
                df_deduped_scored[sequences_with_fisher_result.columns]
                <= p_value_threshold
            )
            # So then groupby specimen_label from the multi-index, and sum
            .groupby("specimen_label").sum()
        )
        total_num_unique_sequences_per_specimen = df_deduped_scored.groupby(
            "specimen_label"
        ).size()
        feature_vectors = (
            # multiply by 1000 first so we don't arrive at a very tiny float
            # this is ok because everything will be z-scored when input to the classifier
            (
                num_unique_sequences_per_specimen_that_are_predictive_of_a_class * 1000
            ).divide(total_num_unique_sequences_per_specimen, axis="index")
        )

        # Consider later:
        # We are using k/n as features, with k being disease hits and n being the total number of sequences per specimen,
        # but could also include total number of sequences n as an extra feature, if we are concerned about sampling depth.

        # Reorder feature matrix according to feature_names_in_ order, inserting 0s for any missing features
        feature_vectors = feature_vectors.reindex(columns=feature_order).fillna(0)

        # Extract specimen metadata
        obs = helpers.extract_specimen_metadata_from_obs_df(
            df=df, gene_locus=gene_locus, target_obs_column=target_obs_column
        )

        # Find abstentions: any missing specimens that did not have a single sequence match a predictive sequence
        # These specimens will have all 0s in their feature vectors
        abstained_specimen_names = num_unique_sequences_per_specimen_that_are_predictive_of_a_class[
            (
                num_unique_sequences_per_specimen_that_are_predictive_of_a_class.values
                == 0
            ).all(axis=1)
        ].index.tolist()
        abstained_specimen_metadata = obs.loc[abstained_specimen_names].copy()

        # Make order match
        scored_specimens = feature_vectors.index[
            ~feature_vectors.index.isin(abstained_specimen_names)
        ]
        feature_vectors = feature_vectors.loc[scored_specimens]
        obs = obs.loc[scored_specimens]

        # extract target metadata column
        target_obs_col_name = target_obs_column.value.obs_column_name
        y = obs[target_obs_col_name]
        abstained_specimens_ground_truth_labels = abstained_specimen_metadata[
            target_obs_col_name
        ]

        # Confirm no rows of all 0s -- these should be abstained specimens
        if (feature_vectors.values == 0).all(axis=1).any():
            raise ValueError(
                "Some specimens (feature matrix rows) have all 0s. These should be abstentions."
            )

        # Clear RAM
        del df_deduped
        del df_deduped_scored
        gc.collect()

        return FeaturizedData(
            X=feature_vectors,
            y=y,
            metadata=obs,
            sample_names=scored_specimens,
            abstained_sample_y=abstained_specimens_ground_truth_labels,
            abstained_sample_names=abstained_specimen_names,
            abstained_sample_metadata=abstained_specimen_metadata,
            extras={
                "num_unique_sequences_per_specimen_that_are_predictive_of_a_class": num_unique_sequences_per_specimen_that_are_predictive_of_a_class.loc[
                    scored_specimens
                ],
                "total_num_unique_sequences_per_specimen": total_num_unique_sequences_per_specimen.loc[
                    scored_specimens
                ],
                "p_value_threshold": p_value_threshold,
            },
        )

    @staticmethod
    def _get_model_base_dir(
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        # sample_weight_strategy is disregarded; just included for compatibility with ImmuneClassifierMixin
        sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    ):
        return (
            config.paths.exact_matches_models_dir
            / gene_locus.name
            / combine_classification_option_names(target_obs_column)
        )

    @staticmethod
    def _get_output_base_dir(
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        # sample_weight_strategy disregarded
        sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    ) -> Path:

        return (
            config.paths.exact_matches_output_dir
            / gene_locus.name
            / combine_classification_option_names(
                target_obs_column=target_obs_column,
            )
        )

    def __init__(
        self,
        fold_id: int,
        model_name: str,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        models_base_dir: Optional[Path] = None,
        # sample_weight_strategy is not used by ExactMatchesClassifier,
        # but can optionally be passed in to be consistent with ImmuneClassifierMixin.
        sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    ):
        # Control the order of superclass initialization.
        # 1. Call ImmuneClassifierMixin's constructor
        ImmuneClassifierMixin.__init__(
            self,
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            models_base_dir=models_base_dir,
            sample_weight_strategy=sample_weight_strategy,
        )

        # 2. Now that models_base_dir is set, construct the file path
        self.model_name = model_name
        fname = (
            self.models_base_dir
            / f"{self.fold_label_train}_model.{self.model_name}.{self.fold_id}.joblib"
        )

        # 3. Call ExtendAnything's constructor to load and wrap classifier
        # self._inner will now be the loaded model, and its attributes will be exposed (pass-through)
        ExtendAnything.__init__(self, inner=joblib.load(fname))

        # Load and set other attributes.
        # (Do all loading like this up front in init(), rather than deferred in featurize() or elsewhere, because we may pickle out this object when saving a metamodel and want to make sure everything is bundled in.)

        self.p_value_threshold = joblib.load(
            self.models_base_dir
            / f"{self.fold_label_train}_model.{self.model_name}.{self.fold_id}.p_value.joblib"
        )

        # Load class-association p-values for each training set sequence
        self.sequences_with_fisher_result = joblib.load(
            self.models_base_dir
            / f"{self.fold_label_train}_model.{self.fold_id}.{self.fold_label_train}.sequences_joblib"
        )
        # filter sequences by p-value so that we don't waste time in the eventual pd.merge on sequences that are irrelevant
        self.sequences_with_fisher_result = (
            self._filter_fisher_scores_to_pvalue_threshold(
                self.sequences_with_fisher_result,
                p_value_threshold=self.p_value_threshold,
            )
        )
