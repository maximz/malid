from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union
import warnings
import anndata
import joblib
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.base import BaseEstimator, ClassifierMixin

import malid.external.genetools_arrays
from malid import helpers
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from malid.external.model_evaluation import FeaturizedData
from malid.trained_model_wrappers import SequenceClassifier
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)

logger = logging.getLogger(__name__)


def _trimmed_mean(
    sequence_class_prediction_probabilities: pd.DataFrame,
    sequence_weights: pd.Series,
    proportiontocut: float = 0.1,
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
    trimming_mask = malid.external.genetools_arrays.get_trim_both_sides_mask(
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
    disease_probabilities = disease_probabilities / disease_probabilities.sum()
    return disease_probabilities


def _weighted_median(seq_probs: pd.DataFrame, seq_weights: pd.Series):
    """
    Compute weighted median of sequence predictions.
    Note that the weighted median is not the same as "weighted trim-both-sides-by-50% mean". (Though unweighted median is the same as unweighted trim-both-by-50% mean.)
    Weighted median = factor in the weights when finding the center of the array.
    """
    disease_probabilities = pd.Series(0, index=seq_probs.columns)
    for i in seq_probs.columns:
        disease_sorted = seq_probs[i].sort_values()
        cumsum = seq_weights.loc[disease_sorted.index].cumsum()
        cutoff = seq_weights.sum() / 2.0
        disease_probabilities[i] = disease_sorted.loc[cumsum >= cutoff].iloc[0]
    disease_probabilities = disease_probabilities / disease_probabilities.sum()
    return disease_probabilities


def _entropy_threshold(
    seq_probs: pd.DataFrame, seq_weights: pd.Series, max_entropy=1.4
):
    """
    Filter sequence predictions with high entropy. This isolates sequences that are "opinionated."
    (Note that high entropy means that the sequence is not very informative about the disease. The closer a n-dim vector is to being all [1/n], the higher its entropy.)
    """
    seq_prob_entropy = seq_probs.apply(scipy.stats.entropy, axis=1)
    mask = seq_prob_entropy < max_entropy

    disease_probabilities = pd.Series(0, index=seq_probs.columns)

    for i in seq_probs.columns:
        mutual_info_subset_of_disease_probabilities = seq_probs.loc[mask, i]
        mutual_info_subset_of_weights = seq_weights.loc[mask]
        disease_probabilities[i] = (
            mutual_info_subset_of_disease_probabilities * mutual_info_subset_of_weights
        ).sum() / (mutual_info_subset_of_weights.sum())

    disease_probabilities = disease_probabilities / disease_probabilities.sum()
    return disease_probabilities


def _trim_bottom_only(
    seq_probs: pd.DataFrame, seq_weights: pd.Series, proportiontocut=0.1
):
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
    disease_probabilities = disease_probabilities / disease_probabilities.sum()
    return disease_probabilities


class RollupSequenceClassifier(ImmuneClassifierMixin, BaseEstimator, ClassifierMixin):
    """
    Roll-up the per-sequence classifier's outputs to a per-specimen prediction.

    This model breaks the convention of calling featurize(adata) before executing the model.
    Instead, pass an anndata to predict() or predict_proba() directly.
    """

    def __init__(
        self,
        fold_id: int,
        model_name_sequence_disease: str,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        sample_weight_strategy: SampleWeightStrategy,
        sequence_models_base_dir: Optional[Path] = None,
    ):
        """
        Create a RollupSequenceClassifier.
        Loads sequence model from disk (cannot be fitted here).
        Requires fold_id to load appropriate model and preprocessing.

        Supports tuning decision thresholds:
        Don't overoptimize at the sequence classifier stage, since those ground truth labels are not accurate.
        Instead tune the decision thresholds *after* rolling up to specimen-level predictions, where we now do have accurate ground truth labels (patient diagnosis).
        """

        self.clf_sequence_disease = SequenceClassifier(
            fold_id=fold_id,
            model_name_sequence_disease=model_name_sequence_disease,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
            models_base_dir=sequence_models_base_dir,
        )

        self.model_name_sequence_disease = model_name_sequence_disease
        self.sequence_models_base_dir = self.clf_sequence_disease.models_base_dir
        # These will be validated and stored on self
        super().__init__(
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
            # directory containing rollup models
            models_base_dir=self._get_model_base_dir(
                sequence_models_base_dir=self.sequence_models_base_dir
            ),
        )

    @staticmethod
    def _get_model_base_dir(sequence_models_base_dir: Path) -> Path:
        return sequence_models_base_dir / "rollup_models"

    @property
    def classes_(self):
        return self.clf_sequence_disease.classes_

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
        specimen_results_proba: List[pd.Series] = []
        for specimen_label, specimen_repertoire in helpers.anndata_groupby_obs(
            dataset, "specimen_label", observed=True, sort=False
        ):
            specimen_results_proba.append(
                self._predict_proba_single_repertoire(
                    repertoire=specimen_repertoire,
                    proportiontocut=proportiontocut,
                    strategy=strategy,
                ).rename(specimen_label)
            )
        specimen_results_proba = pd.DataFrame(specimen_results_proba)

        # Return in expected column order
        return specimen_results_proba[self.classes_]

    def _predict_proba_single_repertoire(
        self,
        repertoire: anndata.AnnData,
        proportiontocut: float = 0.1,
        strategy: str = "trimmed_mean",
    ) -> pd.Series:
        """
        rollup predictions (probabilistic). operates on one specimen only
        """
        featurized = self.clf_sequence_disease.featurize(repertoire)

        # one vector per sequence, of prediction *probabilities* for whether sequence is Covid, HIV, etc or Healthy
        # i.e. already softmaxed
        sequence_class_prediction_probabilities = pd.DataFrame(
            self.clf_sequence_disease.predict_proba(
                featurized.X,
            ),
            index=featurized.sample_names,
            columns=self.clf_sequence_disease.classes_,
        )

        sequence_weights = featurized.sample_weights
        if sequence_weights is None:
            sequence_weights = pd.Series(
                1, index=sequence_class_prediction_probabilities.index
            )
        else:
            # defensive cast to pd.Series
            sequence_weights = pd.Series(
                sequence_weights, index=sequence_class_prediction_probabilities.index
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
                sequence_class_prediction_probabilities,
                sequence_weights,
                proportiontocut,
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
