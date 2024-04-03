from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import anndata
import joblib
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.base import BaseEstimator, ClassifierMixin

import genetools.arrays
from malid import helpers
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from crosseval import FeaturizedData
from malid.trained_model_wrappers import AbstractSequenceClassifier, SequenceClassifier
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)

logger = logging.getLogger(__name__)


def _trimmed_mean(
    sequence_class_prediction_probabilities: pd.DataFrame,
    sequence_weights: pd.Series,
    proportiontocut: float = 0.1,
    # Normalize the final Series of aggregated class probabilities to sum to 1
    normalize_final_probabilities_to_sum_to_1: bool = True,
) -> pd.Series:
    # take weighted trimmed mean, then rewrap as pandas series
    # that means trim off ends first, then take weighted mean of remainder
    # the challenge is applying correct original sequence weights to smaller trimmed arrays for each class
    # we must take trimmed mean for each column (each class) independently!

    # input: probabilities is matrix, weights is 1d vector (one weight per row of matrix)

    # tile the weights horizontally
    weights_horizontally_cloned = np.tile(
        np.array(sequence_weights)[np.newaxis, :].transpose(),
        sequence_class_prediction_probabilities.shape[1],
    )
    if (
        sequence_class_prediction_probabilities.shape
        != weights_horizontally_cloned.shape
    ):
        raise ValueError(
            "Shape mismatch between sequence_class_prediction_probabilities and weights_horizontally_cloned"
        )

    # get trimming mask. the selected indices will be different for each column.
    trimming_mask = genetools.arrays.get_trim_both_sides_mask(
        sequence_class_prediction_probabilities,
        proportiontocut=proportiontocut,
        axis=0,
    )

    # apply this trimming mask to the probabilities and to the row weights
    sequence_class_prediction_probabilities_trimmed = np.take_along_axis(
        np.array(sequence_class_prediction_probabilities), trimming_mask, axis=0
    )  # interestingly this does not work: np.array(sequence_class_prediction_probabilities)[trimming_mask]
    sequence_weights_trimmed = np.take_along_axis(
        weights_horizontally_cloned, trimming_mask, axis=0
    )  # interestingly, this *does* work instead: np.array(sequence_weights)[trimming_mask]
    if (
        sequence_class_prediction_probabilities_trimmed.shape
        != sequence_weights_trimmed.shape
    ):
        raise ValueError(
            "Shape mismatch between sequence_class_prediction_probabilities_trimmed and sequence_weights_trimmed"
        )

    column_averages = np.average(
        a=sequence_class_prediction_probabilities_trimmed,
        weights=sequence_weights_trimmed,
        axis=0,
    )

    disease_probabilities = pd.Series(
        column_averages,
        index=sequence_class_prediction_probabilities.columns,
    )
    if normalize_final_probabilities_to_sum_to_1:
        disease_probabilities = disease_probabilities / disease_probabilities.sum()
    return disease_probabilities


def _weighted_median(
    seq_probs: pd.DataFrame,
    seq_weights: pd.Series,
    # Normalize the final Series of aggregated class probabilities to sum to 1
    normalize_final_probabilities_to_sum_to_1: bool = True,
) -> pd.Series:
    """
    Compute weighted median of sequence predictions.
    Note that the weighted median is not the same as "weighted trim-both-sides-by-50% mean". (Though unweighted median is the same as unweighted trim-both-by-50% mean.)
    Weighted median = factor in the weights when finding the center of the array.
    """
    # calculate weighted median of each column
    disease_probabilities = pd.Series(
        {
            disease: genetools.arrays.weighted_median(seq_probs[disease], seq_weights)
            for disease in seq_probs.columns
        }
    )
    if normalize_final_probabilities_to_sum_to_1:
        # normalize "weighted median of each column" series to sum to 1
        disease_probabilities = disease_probabilities / disease_probabilities.sum()
    return disease_probabilities


def _compute_maximum_entropy_cutoff(n_classes: int) -> float:
    """
    Compute the maximum entropy cutoff for a n-dim vector of class probabilities.
    This is the entropy of the n-dim vector of all [1/n] entries.
    See _entropy_threshold docs.
    """
    return scipy.stats.entropy(np.ones(n_classes) / n_classes)


def _entropy_threshold(
    seq_probs: pd.DataFrame,
    seq_weights: pd.Series,
    max_entropy: float,
    # Normalize the final Series of aggregated class probabilities to sum to 1
    normalize_final_probabilities_to_sum_to_1: bool = True,
) -> pd.Series:
    """
    Filter out sequence predictions with high entropy. This isolates (keeps) sequences that are "opinionated."
    (Note that high entropy means that the sequence is not very informative about the disease. The closer a n-dim vector is to being all [1/n], the higher its entropy.)

    Note:
    Above a certain entropy, the entropy threshold applies no filtering, acting as an untrimmed mean.
    That maximum entropy threshold is the entropy of the n-dim vector of all [1/n] entries:
        maximum_entropy_cutoff = scipy.stats.entropy(np.ones(n_classes) / n_classes)
    Skip any entropy thresholds that are >= maximum_entropy_cutoff; they would apply no filtering.
    Use _compute_maximum_entropy_cutoff() to calculate this value.
    """
    # Calculate entropy row-wise (for each sequence)
    seq_prob_entropy = seq_probs.apply(scipy.stats.entropy, axis=1)
    # Create mask to filter out sequences greater than or equal to the maximum entropy threshold
    mask = seq_prob_entropy < max_entropy

    # Check if the mask has excluded every data point
    if not mask.any():
        # Handle the edge case when mask excludes all data points: return a uniform distribution
        disease_probabilities = pd.Series(
            np.ones(len(seq_probs.columns)) / len(seq_probs.columns),
            index=seq_probs.columns,
        )
        return disease_probabilities

    disease_probabilities = pd.Series(0, index=seq_probs.columns)

    for i in seq_probs.columns:
        mutual_info_subset_of_disease_probabilities = seq_probs.loc[mask, i]
        mutual_info_subset_of_weights = seq_weights.loc[mask]
        sum_of_weights = mutual_info_subset_of_weights.sum()
        if sum_of_weights > 0:
            disease_probabilities[i] = (
                mutual_info_subset_of_disease_probabilities
                * mutual_info_subset_of_weights
            ).sum() / sum_of_weights
        else:
            # Handle division by zero edge case: assign 0
            disease_probabilities[i] = 0

    if normalize_final_probabilities_to_sum_to_1:
        total_prob_sum = disease_probabilities.sum()
        if total_prob_sum > 0:
            disease_probabilities = disease_probabilities / total_prob_sum
        else:
            # Handle division by zero edge case: assign equal probabilities to all classes
            disease_probabilities = pd.Series(
                np.ones(len(disease_probabilities)) / len(disease_probabilities),
                index=disease_probabilities.index,
            )

    return disease_probabilities


def _trim_bottom_only(
    seq_probs: pd.DataFrame,
    seq_weights: pd.Series,
    proportiontocut=0.1,
    # Normalize the final Series of aggregated class probabilities to sum to 1
    normalize_final_probabilities_to_sum_to_1: bool = True,
) -> pd.Series:
    # TODO: generalize _trimmed_mean to support trimming bottom only, top only, or both sides. i.e. accept trim_lower, trim_upper parameters.
    disease_probabilities = pd.Series(0, index=seq_probs.columns)
    for i in seq_probs.columns:
        disease_sorted = seq_probs[i].sort_values()
        # apply trimming mask to the probabilities
        disease_trimmed = disease_sorted.iloc[int(proportiontocut * len(seq_probs)) :]
        # apply trimming mask to the row weights
        weights_trimmed = seq_weights.loc[disease_trimmed.index]
        disease_probabilities[i] = (disease_trimmed * weights_trimmed).sum() / (
            weights_trimmed.sum()
        )
    if normalize_final_probabilities_to_sum_to_1:
        disease_probabilities = disease_probabilities / disease_probabilities.sum()
    return disease_probabilities


class RollupSequenceClassifier(ImmuneClassifierMixin, BaseEstimator, ClassifierMixin):
    """
    Roll-up the per-sequence classifier's outputs to a per-specimen prediction.

    This model breaks the convention of calling featurize(adata) before executing the model.
    Instead, pass an anndata to predict() or predict_proba() directly.
    """

    sequence_classifier: AbstractSequenceClassifier

    def __init__(
        self,
        fold_id: int,
        model_name_sequence_disease: str,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
        sequence_models_base_dir: Optional[Path] = None,
        # Optionally provide the sequence classifier directly. Otherwise it will be loaded through the SequenceClassifier constructor.
        sequence_classifier: Optional[AbstractSequenceClassifier] = None,
    ):
        """
        Create a RollupSequenceClassifier.
        Loads sequence model from disk (cannot be fitted here).
        Requires fold_id to load appropriate model and preprocessing.

        Supports tuning decision thresholds:
        Don't overoptimize at the sequence classifier stage, since those ground truth labels are not accurate.
        Instead tune the decision thresholds *after* rolling up to specimen-level predictions, where we now do have accurate ground truth labels (patient diagnosis).
        """
        # TODO: Rename model_name_sequence_disease to not include "disease"; we can have other classification targets.

        if sequence_classifier is not None:
            self.sequence_classifier = sequence_classifier
        else:
            self.sequence_classifier = SequenceClassifier(
                fold_id=fold_id,
                model_name_sequence_disease=model_name_sequence_disease,
                fold_label_train=fold_label_train,
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
                models_base_dir=sequence_models_base_dir,
            )

        self.model_name_sequence_disease = model_name_sequence_disease
        self.sequence_models_base_dir = self.sequence_classifier.models_base_dir
        # These will be validated and stored on self
        super().__init__(
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )

    # Directory containing rollup models:
    @staticmethod
    def _get_model_base_dir(sequence_models_base_dir: Path) -> Path:
        return sequence_models_base_dir / "rollup_models"

    @property
    def models_base_dir(self):
        # overrides default implementation on ImmuneClassifierMixin,
        # because _get_model_base_dir signature is different,
        # and because we don't allow user to pass in an override.
        return self._get_model_base_dir(
            sequence_models_base_dir=self.sequence_models_base_dir
        )

    @property
    def output_base_dir(self) -> Path:
        return self._get_output_base_dir(
            sequence_model_output_base_dir=self.sequence_classifier.output_base_dir
        )

    @staticmethod
    def _get_output_base_dir(sequence_model_output_base_dir: Path) -> Path:
        return sequence_model_output_base_dir / "rollup_models"

    @property
    def classes_(self):
        return self.sequence_classifier.classes_

    def featurize(self, dataset: anndata.AnnData) -> FeaturizedData:
        """Featurize.
        Unconventional: FeaturizedData.X will be an anndata, not a data_X matrix - because predict_proba() expects anndata.
        The other fields may contain more specimens than survive the rollup process (i.e. some may be abstained on).
        """
        specimen_metadata = helpers.extract_specimen_metadata_from_anndata(
            adata=dataset,
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
        )

        return FeaturizedData(
            # TODO: Overload FeaturizedData to make X an AnnData, just like SubsetClassifierFeaturizedData child class
            X=dataset,  # Note: violates type
            y=specimen_metadata[self.target_obs_column.value.obs_column_name],
            metadata=specimen_metadata,
            sample_names=specimen_metadata.index,
        )

    def predict_proba(
        self,
        dataset: anndata.AnnData,
        # TODO(refactor): Make strategy an enum
        # TODO(refactor): set strategy and proportiontocut in the constructor, not as arguments to predict_proba
        proportiontocut: float = 0.1,
        strategy: str = "trimmed_mean",
    ) -> pd.DataFrame:
        """
        Predict class probabilities for each specimen in dataset.
        Returns DataFrame indexed by specimen label, with columns corresponding to classifier classes.
        Unconventional: accepts anndata argument, not featurized data matrix.
        """
        # Run model on each specimen
        # Alternative considered: Run on full data set first, then groupby specimen_label and compute rollup.

        # Refactored into two steps so we can try many rollup strategies quickly without redoing step 1 every time.

        # Step 1: generate sequence predictions within each specimen
        sequence_predictions_by_specimen = (
            self._predict_sequence_probas_all_repertoires(dataset)
        )

        # Step 2: rollup the sequence predictions
        return self._rollup_sequence_probas_all_repertoires(
            sequence_predictions_by_specimen=sequence_predictions_by_specimen,
            proportiontocut=proportiontocut,
            strategy=strategy,
        )

    def _predict_sequence_probas_all_repertoires(
        self, dataset: anndata.AnnData
    ) -> Dict[
        str,
        Tuple[
            pd.DataFrame,
            Union[pd.Series, np.ndarray],
            Optional[Union[pd.Series, np.ndarray]],
        ],
    ]:
        sequence_predictions_by_specimen = {}
        for specimen_label, specimen_repertoire in helpers.anndata_groupby_obs(
            dataset, "specimen_label", observed=True, sort=False
        ):
            sequence_predictions_by_specimen[
                specimen_label
            ] = self._predict_sequence_probas_single_repertoire(
                repertoire=specimen_repertoire
            )
        return sequence_predictions_by_specimen

    def _predict_sequence_probas_single_repertoire(
        self,
        repertoire: anndata.AnnData,
    ) -> Tuple[
        pd.DataFrame,
        Union[pd.Series, np.ndarray],
        Optional[Union[pd.Series, np.ndarray]],
    ]:
        """
        return sequence predictions - operates on one specimen only
        """
        featurized = self.sequence_classifier.featurize(repertoire)

        # one vector per sequence, of prediction *probabilities* for whether sequence is Covid, HIV, etc or Healthy
        # i.e. already softmaxed
        sequence_class_prediction_probabilities = pd.DataFrame(
            self.sequence_classifier.predict_proba(
                featurized.X,
            ),
            index=featurized.sample_names,
            columns=self.sequence_classifier.classes_,
        )
        sequence_y_labels = featurized.y
        sequence_weights = featurized.sample_weights
        return (
            sequence_class_prediction_probabilities,
            sequence_y_labels,
            sequence_weights,
        )

    def _rollup_sequence_probas_all_repertoires(
        self,
        sequence_predictions_by_specimen: Dict[
            str,
            Tuple[
                pd.DataFrame,
                Union[pd.Series, np.ndarray],
                Optional[Union[pd.Series, np.ndarray]],
            ],
        ],
        proportiontocut: float,
        strategy: str,
    ) -> pd.DataFrame:
        specimen_results_proba: List[pd.Series] = [
            self._rollup_sequence_probas_for_single_repertoire(
                sequence_class_prediction_probabilities=sequence_class_prediction_probabilities,
                proportiontocut=proportiontocut,
                strategy=strategy,
                sequence_weights=sequence_weights,
            ).rename(specimen_label)
            for specimen_label, (
                sequence_class_prediction_probabilities,
                _,
                sequence_weights,
            ) in sequence_predictions_by_specimen.items()
        ]

        # Return in expected column order
        specimen_results_proba_combined = pd.DataFrame(specimen_results_proba)
        return specimen_results_proba_combined[self.classes_]

    def _rollup_sequence_probas_for_single_repertoire(
        self,
        sequence_class_prediction_probabilities: pd.DataFrame,
        proportiontocut: float,
        strategy: str,
        sequence_weights: Optional[Union[pd.Series, np.ndarray]],
    ) -> pd.Series:
        """
        rollup predictions (probabilistic). operates on one specimen only
        """

        # Drop rows (sequences) with all NaN probas, then fill remaining NaNs with 0.
        # This situation is possible if sequence_classifier is a VJGeneSpecificSequenceClassifier which does not necessarily have a model available for every V-J gene combination.
        # (If we drop any rows, we also need to drop corresponding entries from sequence_weights if that was supplied.)

        # Identify row indices with all NaN values
        all_nan_rows = sequence_class_prediction_probabilities.isna().all(axis=1)

        # Remove rows with all NaN values. This should be equivalent to sequence_class_prediction_probabilities.dropna(how="all", axis=0)
        sequence_class_prediction_probabilities = (
            sequence_class_prediction_probabilities.loc[~all_nan_rows]
        )

        # Fillna on the rest
        sequence_class_prediction_probabilities = (
            sequence_class_prediction_probabilities.fillna(0.0)
        )

        if sequence_class_prediction_probabilities.empty:
            # No rows remaining
            # Return no predictions
            return pd.Series(
                np.nan, index=sequence_class_prediction_probabilities.columns
            )

        if sequence_weights is None:
            sequence_weights = pd.Series(
                1, index=sequence_class_prediction_probabilities.index
            )
        else:
            # defensive cast to pd.Series
            # also apply same no-NaN-rows filter
            sequence_weights = pd.Series(
                sequence_weights[~all_nan_rows],
                index=sequence_class_prediction_probabilities.index,
            )

        if strategy == "trimmed_mean":
            # if given sample weights: take trimmed mean
            disease_probabilities = _trimmed_mean(
                sequence_class_prediction_probabilities,
                sequence_weights,
                proportiontocut,
            )
        elif strategy == "weighted_median":
            disease_probabilities = _weighted_median(
                sequence_class_prediction_probabilities, sequence_weights
            )
        elif strategy == "trim_bottom_only":
            disease_probabilities = _trim_bottom_only(
                sequence_class_prediction_probabilities,
                sequence_weights,
                proportiontocut,
            )
        elif strategy == "entropy_threshold":
            disease_probabilities = _entropy_threshold(
                seq_probs=sequence_class_prediction_probabilities,
                seq_weights=sequence_weights,
                max_entropy=proportiontocut,
            )
        else:
            raise ValueError("Invalid rollup strategy")

        return disease_probabilities

    def predict(self, dataset: anndata.AnnData) -> pd.Series:
        """
        Predict disease label for each specimen in dataset, by rolling up all sequence predictions in each specimen.
        Returns Series indexed by specimen label.
        Unconventional: accepts anndata argument, not featurized data matrix.
        """
        predictions_proba: pd.DataFrame = self.predict_proba(dataset)
        return predictions_proba.idxmax(axis=1)


##########


def _apply_saved_decision_threshold_tuning(inner_clf, fname):
    """Reapply saved decision thresholds to a classifier object."""
    wrapped_clf_from_disk = joblib.load(fname)
    if type(wrapped_clf_from_disk) != AdjustedProbabilitiesDerivedModel:
        # No AdjustedProbabilitiesDerivedModel wrapper in the version saved to disk.
        return inner_clf
    return AdjustedProbabilitiesDerivedModel(
        inner_clf, class_weights=wrapped_clf_from_disk.class_weights
    )


def load_rollup_sequence_classifier_including_optional_tuned_thresholds(
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    fold_label_train: str,  # What base models were trained on
    model_name_sequence_disease: str,
    sequence_models_base_dir: Optional[Path] = None,
) -> Union[RollupSequenceClassifier, AdjustedProbabilitiesDerivedModel]:
    """
    If `model_name_sequence_disease` ends in "-rollup_tuned", will load decision-threshold tuned version of rollup model for sequence classifier without that suffix.
    """
    use_tuned_rollup_model = False
    if model_name_sequence_disease.endswith("-rollup_tuned"):
        # Load decision-threshold-tuned version of rollup model.
        use_tuned_rollup_model = True
        model_name_sequence_disease = model_name_sequence_disease[
            : -len("-rollup_tuned")
        ]

    rollup_clf = RollupSequenceClassifier(
        fold_id=fold_id,
        model_name_sequence_disease=model_name_sequence_disease,
        fold_label_train=fold_label_train,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
        sequence_models_base_dir=sequence_models_base_dir,
    )
    if use_tuned_rollup_model:
        # Don't use entire wrapped saved model directly because want to use latest code for RollupSequenceClassifier
        # Instead create latest RollupSequenceClassifier as above, and then apply any desired wrapping weights
        rollup_clf = _apply_saved_decision_threshold_tuning(
            rollup_clf,
            rollup_clf.models_base_dir
            / f"{fold_label_train}_model.{model_name_sequence_disease}-rollup_tuned.{fold_id}.joblib",
        )
    return rollup_clf
