from __future__ import annotations
import gc
from itertools import combinations
from pathlib import Path
from typing import Optional

import anndata
import genetools
import joblib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap
from genetools.palette import HueValueStyle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from slugify import slugify

import malid.external.genetools_arrays
import malid.external.genetools_plots
from malid import helpers, stats, io
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    UmapSupervisionStrategy,
    healthy_label,
)
import choosegpu
from malid.trained_model_wrappers import BlendingMetamodel
from malid.trained_model_wrappers import SequenceClassifier

# Feature flag
compute_knn_index = False  # TODO: Reenable.

import logging

logger = logging.getLogger(__name__)

disease_color_palette = helpers.disease_color_palette.copy()
disease_color_palette[healthy_label] = HueValueStyle(
    color=disease_color_palette[healthy_label], zorder=-15, alpha=0.5
)

## Data


def load_overlay_data_for_fold(
    fold_id,
    specimen_metadata,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    foreground_peakset_fold_label: str,  # which fold label to use as foreground overlay for peak timepoints
    foreground_include_peak: bool,  # whether to plot people that are ONLY in peak timepoint set
    foreground_include_offpeak: bool,  # whether to plot people that are ONLY in offpeak timepoint set
    offpeak_include_diseases_outside_training_set: bool,  # whether to plot offpeak patients whose diseases are never in training set
    filter_naive_isotypes: bool,
):
    """Generator of overlay patient dicts for a particular fold id."""
    ## Get test sets, both peak and off-peak
    try:
        # May be nonexistent, e.g. global fold -1 does not have a "test" set.
        adata_test_peak = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label=foreground_peakset_fold_label,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )
    except FileNotFoundError:
        logger.warning(
            f"{foreground_peakset_fold_label} (peak) anndata not found on disk for fold {fold_id}, {gene_locus}",
            exc_info=True,
        )
        adata_test_peak = None

    try:
        adata_test_offpeak = io.load_fold_embeddings_off_peak(
            fold_id=fold_id,
            gene_locus=gene_locus,
            sample_weight_strategy=sample_weight_strategy,
        )
        if not offpeak_include_diseases_outside_training_set:
            adata_test_offpeak = adata_test_offpeak[
                adata_test_offpeak.obs["disease"].isin(
                    helpers.diseases_in_peak_timepoint_dataset()
                )
            ]
    except FileNotFoundError:
        logger.warning(
            f"Off peak anndata not found on disk for fold {fold_id}, {gene_locus}",
            exc_info=True,
        )
        adata_test_offpeak = None

    if filter_naive_isotypes:
        if adata_test_peak is not None:
            adata_test_peak = remove_naive_isotypes(adata_test_peak)
        if adata_test_offpeak is not None:
            adata_test_offpeak = remove_naive_isotypes(adata_test_offpeak)

    # Choose which patients will get plotted
    test_patients_to_process = set()
    if foreground_include_peak and adata_test_peak is not None:
        # process each participant with any peak samples
        test_patients_to_process = test_patients_to_process.union(
            adata_test_peak.obs["participant_label"].unique()
        )

    if foreground_include_offpeak and adata_test_offpeak is not None:
        # also process each participant with any off-peak samples
        test_patients_to_process = test_patients_to_process.union(
            adata_test_offpeak.obs["participant_label"].unique()
        )

    if len(test_patients_to_process) == 0:
        logger.error(f"No overlay patients selected for fold {fold_id}, {gene_locus}.")
        # return empty generator
        return
        yield

    ## Make overlay patient dicts
    # subset big anndatas by participant and timepoint.
    # merge any specimens that are replicates at the same timepoint.
    for participant_label in test_patients_to_process:
        yield make_participant_dict(
            fold_id=fold_id,
            participant_label=participant_label,
            adata_test_peak=adata_test_peak,
            adata_test_offpeak=adata_test_offpeak,
            specimen_metadata=specimen_metadata,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )


def get_top_ranked_subset_obsnames(
    adata,
    class_label_column,
    rank_by_column,
    difference_between_top_two_predicted_probas_column,
    # one of these two must be provided:
    top_n_per_class: Optional[int] = 5000,
    top_percent_per_class: Optional[float] = None,
    must_match_column=None,
    difference_between_top_two_predicted_probas_cutoff=0.0,
    # then this can optionally be provided to subsample a certain N from the set chosen above
    subsample_n_from_chosen: Optional[int] = None,
) -> np.ndarray:
    """
    Returns a list of obsnames from an anndata object, sorted by rank_by_column.
    Get top X% subset (per class!) of training set for this fold to form the background for the plots, and to form the basis for the UMAP transformations.
    Optionally choose a further n=[subsample_n_from_chosen] subset from each class's selected set at the end.
    """
    if top_n_per_class is None and top_percent_per_class is None:
        raise ValueError(
            "Either top_n_per_class or top_percent_per_class must be provided."
        )
    if top_n_per_class is not None and top_percent_per_class is not None:
        raise ValueError(
            "Only one of top_n_per_class or top_percent_per_class must be provided."
        )

    subset_obsnames = []
    for class_name, obs_single_class in adata.obs.groupby(
        class_label_column, observed=True
    ):
        if must_match_column is not None:
            # subset further
            obs_single_class = obs_single_class[
                obs_single_class[must_match_column]
                == obs_single_class[class_label_column]
            ]
        # subset further
        obs_single_class = obs_single_class[
            obs_single_class[difference_between_top_two_predicted_probas_column]
            >= difference_between_top_two_predicted_probas_cutoff
        ]
        if top_percent_per_class is not None:
            # take top N fraction of each class
            result = malid.external.genetools_arrays.get_top_n_percent(
                df=obs_single_class,
                col=rank_by_column,
                fraction=top_percent_per_class,
            )
        else:
            # take top N
            result = malid.external.genetools_arrays.get_top_n(
                df=obs_single_class,
                col=rank_by_column,
                n=top_n_per_class,
            )
        # Optionally subset further
        if subsample_n_from_chosen is not None:
            result = result.sample(
                n=min(subsample_n_from_chosen, result.shape[0]),
                replace=False,
                random_state=1,
            )
        subset_obsnames.append(result.index.values)
    return np.hstack(subset_obsnames)


def make_participant_dict(
    fold_id: int,
    participant_label: str,
    adata_test_peak: Optional[anndata.AnnData],
    adata_test_offpeak: Optional[anndata.AnnData],
    specimen_metadata: pd.DataFrame,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
):
    """create overlay participant dicts"""

    # get specimen timepoint metadata for all (peak + non-peak) samples
    all_specimens_for_this_participant = specimen_metadata[
        specimen_metadata["participant_label"] == participant_label
    ]
    participant_disease_true_label = all_specimens_for_this_participant[
        "disease.separate_past_exposures"
    ].iloc[0]
    assert all(
        all_specimens_for_this_participant["disease.separate_past_exposures"]
        == participant_disease_true_label
    )

    def get_subset_anndata(adata_full, specimen_label):
        return adata_full[adata_full.obs["specimen_label"] == specimen_label]

    adata_subsets = []

    # Subset big anndatas by participant and timepoint. merge any specimens that are replicates at the same timepoint.
    # Don't sort in this groupby so that original timepoint order is kept
    for (
        timepoint,
        is_peak,
    ), specimens_at_timepoint in all_specimens_for_this_participant.groupby(
        ["timepoint_formatted", "is_peak"], observed=True, sort=False
    ):
        adata_partial = []
        # get anndata for each replicate
        for _, specimen in specimens_at_timepoint.iterrows():
            # get anndata from peak or non-peak set
            adata_to_use = adata_test_peak if is_peak else adata_test_offpeak
            adata = get_subset_anndata(
                adata_full=adata_to_use,
                specimen_label=specimen["specimen_label"],
            )

            if adata.shape[0] > 0:
                adata_partial.append(adata)
            else:
                # Empty subset anndata - warn and skip
                logger.warning(
                    f"Skipping empty subset anndata: {specimen['specimen_label']}, is_peak {is_peak}, timepoint {timepoint}, participant {participant_label}"
                )

        if len(adata_partial) == 0:
            logger.warning(
                f"No subsets to combine for is_peak={is_peak}, timepoint={timepoint}, participant={participant_label}"
            )
            continue

        # combine anndatas across replicates from same timepoint
        # note: this makes a copy of the anndatas, even if this is a list containing one anndata-view
        adata = anndata.concat(adata_partial)

        # note that this new anndata will still have multiple "specimen_label"s in its obs if we combined multiple replicates
        # but this will cause issues because the repertoire stats classifier featurization does a groupby on specimen_label, i.e. it will produce a separate feature vector for each replicate
        # we don't want that, so recode the specimen labels
        new_specimen_label = "|".join(adata.obs["specimen_label"].unique())
        # we've combined multiple replicates (with different source specimen_labels) into one anndata
        # rename all specimen labels to a combined string that contains all the specimen labels, delimited by |
        adata.obs["specimen_label"] = new_specimen_label

        # repeat this for disease_subtype in case the replicates had different disease_subtypes, which would be odd
        new_disease_subtype_label = "|".join(adata.obs["disease_subtype"].unique())
        adata.obs["disease_subtype"] = new_disease_subtype_label

        # also merge numeric columns that may vary between replicates of the same specimen, like isotype_proportion columns
        replicate_metadata = helpers.extract_specimen_metadata_from_obs_df(
            df=adata.obs, gene_locus=gene_locus, target_obs_column=target_obs_column
        )
        if replicate_metadata.shape[0] > 1:
            # need to merge multiple metadata rows for the same specimen_label
            # replace numeric columns with mean across replicates
            replacements = replicate_metadata.select_dtypes(include=["number"]).mean()
            adata.obs = adata.obs.assign(**replacements)
            # confirm we now have one metadata row for this specimen.
            assert (
                helpers.extract_specimen_metadata_from_obs_df(
                    df=adata.obs,
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_column,
                ).shape[0]
                == 1
            ), f"replicate metadata merging failed for {participant_label}, {new_specimen_label}, {gene_locus}"

        # store combined anndata for this timepoint and is_peak status
        adata_subsets.append(
            {
                "specimen_label": new_specimen_label,
                "timepoint": timepoint,
                "adata": adata,
                "is_peak": is_peak,
                "number_of_replicates_combined": len(adata_partial),
                "specimen_disease_subtype": new_disease_subtype_label,
            }
        )

    # It's theoretically possible that adata_subsets is empty list here.
    participant_dict = {
        "participant_label": participant_label,
        "fold_id": fold_id,
        "adata_subsets": adata_subsets,
        "disease_true_label": participant_disease_true_label,
    }

    return participant_dict


## Models


def load_models_for_fold(
    fold_id,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    metamodel_name: str,
    individual_sequence_model_name: str,
):
    blending_metamodel_clf = BlendingMetamodel.from_disk(
        fold_id=fold_id,
        metamodel_name=metamodel_name,
        base_model_train_fold_name="train_smaller",
        metamodel_fold_label_train="validation",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        metamodel_flavor="default",
    )
    individual_sequence_model_clf = SequenceClassifier(
        fold_id=fold_id,
        model_name_sequence_disease=individual_sequence_model_name,
        fold_label_train="train_smaller",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    return (
        blending_metamodel_clf,
        individual_sequence_model_clf,
    )


def run_blended_model(
    clf: BlendingMetamodel, subset_adata: anndata.AnnData, gene_locus: GeneLocus
):
    """Run blending model"""
    # Blending model assumes multiple specimens in featurized set.
    # So even if running on single specimen like in supervised_embedding context,
    # return lists.

    ## Featurize
    # Wrap input anndata as Dict[GeneLocus, anndata.AnnData], but use only the single-locus metamodel for now.
    # TODO: consider using the multi-locus metamodel here to improve specimen predictions.
    # However that might break the idea that we are looking only at BCRs or only at TCRs visually.
    featurized = clf.featurize({gene_locus: subset_adata})

    # Handle abstention
    if featurized.X.shape[0] == 0:
        return np.array(["Unknown"]), np.array([np.nan])

    # get predicted label
    blended_model_predicted_label = clf.predict(featurized.X)
    # prediction confidence of winning label
    blended_results_proba = clf.predict_proba(featurized.X)
    blended_model_prediction_confidence = blended_results_proba.max(axis=1)

    return (
        blended_model_predicted_label,
        blended_model_prediction_confidence,
    )


def _get_top_two_probabilities(probabilities):
    """
    Given predicted class probabilities, returns dataframe with columns:
    - "max_probability"
    - "second_highest_probability"
    - "difference_between_top_two_predicted_probas"
    - "max_probability_label"
    - "second_highest_probability_label"
    """
    # get winning label probability, 2nd highest probability, and the diff
    predicted_point_probabilities_top_two = pd.DataFrame(
        probabilities.apply(lambda row: pd.Series(row.nlargest(2).values), axis=1),
        index=probabilities.index,
    )
    predicted_point_probabilities_top_two.columns = [
        "max_probability",
        "second_highest_probability",
    ]
    # get diff between top two probabilities
    predicted_point_probabilities_top_two[
        "difference_between_top_two_predicted_probas"
    ] = (
        predicted_point_probabilities_top_two["max_probability"]
        - predicted_point_probabilities_top_two["second_highest_probability"]
    )

    # And attach the top two predicted labels
    top_two_predicted_labels = pd.DataFrame(
        probabilities.apply(lambda row: pd.Series(row.nlargest(2).index), axis=1),
        index=probabilities.index,
    )
    top_two_predicted_labels.columns = [
        "max_probability_label",
        "second_highest_probability_label",
    ]

    return pd.concat(
        [predicted_point_probabilities_top_two, top_two_predicted_labels],
        axis=1,
    )


def embed_sequences(
    clf,
    subset_adata,
    umap_transformer=None,
    pca_transformer=None,
    umap_supervised_labels=None,
    attach_true_disease_label=False,
    attach_embeddings=False,
):
    """
    Using individual-sequence model, generate predictions and embeddings for each sequence.
    Then scale and UMAP the suprevised embeddings to get 2d coordinates (optionally supply pre-trained transformations).

    For a linear model, the embedding is a linear transformation using the coefficients.

    parameters:
    - clf: per-sequence model
    - subset_adata: anndata object
    - transformer:
        optionally supply pre-trained umap transformation.
        if None (default), a new umap transformation (with scaling) will be trained.
    - umap_supervised_labels:
        if training a UMAP transformation (i.e. transformer=None) and umap_supervised_labels is not None,
        these labels will be used to supervise the UMAP.
        the labels must be numeric categories. for partial supervision, replace masked categories with the label -1 (special value that UMAP expects).
    - attach_true_disease_label:
        if True, attach the obs columns "disease.separate_past_exposures" (renamed to "label.patient_of_origin") and "disease_subtype" (renamed to "label.patient_of_origin_subtype") to the output.

    returns:
    - dataframe indexed with embeddings anndata's obs_names, contains these columns for each data point:
        - "X_umap1"
        - "X_umap2"
        - "label.predicted_by.individual_sequence_model"
        - "label.predicted_by.individual_sequence_model.confidence"
        - "label.patient_of_origin" and "label.patient_of_origin_subtype" (if attach_true_disease_label=True)
    - scale+UMAP transformer
    """

    # Running clf.featurize() on the full subset_adata might produce a giant (N x 1900+) object
    # To save RAM, we run this one batch at a time.
    model_predictions_by_point_all = []
    embeddings_all = []

    if subset_adata.shape[0] < 100000:
        # But if the input data is small, then batching may be an unnecessary slowdown.
        # Disable batching for small datasets.
        batches = [(None, subset_adata)]
    else:
        # An easy way to batch this is to divide by specimen.
        batches = helpers.anndata_groupby_obs(
            subset_adata, "specimen_label", observed=True
        )

    for _, specimen_subset in batches:
        featurized_sequences = clf.featurize(specimen_subset)

        # index with anndata's obs_names
        predicted_point_probabilities = pd.DataFrame(
            clf.predict_proba(featurized_sequences.X),
            index=featurized_sequences.sample_names,
            columns=clf.classes_,
        )

        # get winning label probability, 2nd highest probability, and the diff. and the top two labels
        predicted_point_probabilities_top_two = _get_top_two_probabilities(
            predicted_point_probabilities
        ).rename(
            columns={
                "max_probability": "label.predicted_by.individual_sequence_model.confidence",
                "second_highest_probability": "second_highest_confidence.predicted_by.individual_sequence_model",
                "difference_between_top_two_predicted_probas": "difference_between_top_two_predicted_probas.by.individual_sequence_model",
                "max_probability_label": "label.predicted_by.individual_sequence_model",
                "second_highest_probability_label": "second_highest_label.predicted_by.individual_sequence_model",
            }
        )

        # get supervised embeddings
        if hasattr(clf, "coef_"):
            # this is a linear model
            # set column names because the embedding columns correspond to classes
            embeddings = pd.DataFrame(
                stats.linear_model_supervised_embedding(clf, featurized_sequences.X),
                index=featurized_sequences.sample_names,
                columns=clf.classes_,
            )
        else:
            return ValueError(
                "Unkown model type - unable to generate supervised embedding."
            )

        # store results from this batch
        model_predictions_by_point_all.append(predicted_point_probabilities_top_two)
        embeddings_all.append(embeddings)

    # Concatenate all batches
    embeddings_all = pd.concat(embeddings_all, axis=0)
    model_predictions_by_point_all = pd.concat(model_predictions_by_point_all, axis=0)

    # Arrange in original obsnames order
    if not (
        embeddings_all.shape[0]
        == model_predictions_by_point_all.shape[0]
        == subset_adata.shape[0]
    ):
        raise ValueError("Shape mismatch")
    embeddings_all = embeddings_all.loc[subset_adata.obs_names]
    model_predictions_by_point_all = model_predictions_by_point_all.loc[
        subset_adata.obs_names
    ]

    # Train UMAP transformer
    if umap_transformer is None:
        if (
            umap_supervised_labels is not None
            and umap_supervised_labels.shape[0] != subset_adata.shape[0]
        ):
            raise ValueError(
                "umap_supervised_labels must have same number of entries as subset_adata has rows"
            )

        umap_transformer = make_pipeline(
            preprocessing.StandardScaler(),
            umap.UMAP(n_neighbors=15, n_components=2, random_state=0),
        )
        # fit supervised UMAP - note: can't accept strings; must use numeric categories
        # UMAP treats "-1" is a special "masked label" value for partial supervision
        umap_transformer = umap_transformer.fit(
            embeddings_all.values,
            y=np.array(umap_supervised_labels)
            if umap_supervised_labels is not None
            else None,
        )

    # Train PCA transformer
    if pca_transformer is None:
        pca_transformer = make_pipeline(
            preprocessing.StandardScaler(), PCA(n_components=2, random_state=0)
        )
        pca_transformer = pca_transformer.fit(embeddings_all.values)

    # Apply UMAP transformer
    umap_points = pd.DataFrame(
        umap_transformer.transform(embeddings_all.values),
        columns=["X_umap1", "X_umap2"],
        index=embeddings_all.index,
    )

    # Apply PCA transformer
    pca_points = pd.DataFrame(
        pca_transformer.transform(embeddings_all.values),
        columns=["X_pca1", "X_pca2"],
        index=embeddings_all.index,
    )

    # Join pandas objects with same index
    embedded_points = pd.concat(
        [
            umap_points,
            pca_points,
            # Attach true disease label if requested
            subset_adata.obs["disease.separate_past_exposures"].rename(
                "label.patient_of_origin"
            )
            if attach_true_disease_label
            else None,
            # Attach true disease_subtype label if requested
            subset_adata.obs["disease_subtype"].rename(
                "label.patient_of_origin_subtype"
            )
            if attach_true_disease_label
            else None,
            # Attach predicted label and probability
            # This is prediction confidence from individual sequence model
            model_predictions_by_point_all,
        ],
        axis=1,
    )

    if attach_embeddings:
        embedded_points = pd.concat(
            [
                embedded_points,
                embeddings_all.rename(columns=lambda s: f"embedding.{s}"),
            ],
            axis=1,
        )

    if embedded_points.isna().any().any():
        raise ValueError(
            "NaN in embedded_points, perhaps indices did not match before pd.concat"
        )

    return embedded_points, umap_transformer, pca_transformer


## Plots


def plot_fold_background(
    background_points_df: pd.DataFrame,
    # Faster plotting: https://jbendeaton.com/blog/2011/speed-up-plot-rendering-in-pythonmatplotlib
    rasterized=True,
):
    """plots fold background. returns (fig, hue name) generator"""
    for viz_type in ["pca", "umap"]:
        lims = None
        for description, hue_key, palette in zip(
            [
                "Predicted sequence label",
                "Patient of origin label",
                "Patient of origin - subtype",
            ],
            [
                "label.predicted_by.individual_sequence_model",
                "label.patient_of_origin",
                "label.patient_of_origin_subtype",
            ],
            [
                disease_color_palette,
                disease_color_palette,
                sc.plotting.palettes.default_28,
            ],
        ):
            # Background for this fold
            fig, ax = genetools.plots.scatterplot(
                data=background_points_df,
                x_axis_key=f"X_{viz_type}1",
                y_axis_key=f"X_{viz_type}2",
                hue_key=hue_key,
                discrete_palette=palette,
                enable_legend=True,
                alpha=0.5,
                marker=".",
                marker_size=5,
                figsize=(5, 5),
                legend_title=description,
                remove_x_ticks=True,
                remove_y_ticks=True,
                rasterized=rasterized,
            )

            # Put sample sizes in legend
            genetools.plots.add_sample_size_to_legend(
                ax=ax,
                data=background_points_df,
                hue_key=hue_key,
            )

            if lims is None:
                # store axis limits
                lims = (ax.get_xlim(), ax.get_ylim())
            else:
                # apply stored axis limits, to make all plots consistent
                # not necessary here because each point plotted in all figs,
                # but will be necessary in plots below.
                ax.set_xlim(lims[0])
                ax.set_ylim(lims[1])

            yield (fig, hue_key, viz_type)

        # diagnostics: plot some raw probability metrics for each class
        for class_name, class_grp in background_points_df.groupby(
            "label.predicted_by.individual_sequence_model", observed=True
        ):
            for description, hue_key in zip(
                [
                    f"Probability of {class_name}",
                    "Difference between top two probabilities",
                ],
                [
                    "label.predicted_by.individual_sequence_model.confidence",
                    "difference_between_top_two_predicted_probas.by.individual_sequence_model",
                ],
            ):
                # Background for this fold
                fig, ax = genetools.plots.scatterplot(
                    data=class_grp,
                    x_axis_key=f"X_{viz_type}1",
                    y_axis_key=f"X_{viz_type}2",
                    hue_key=hue_key,
                    continuous_hue=True,
                    enable_legend=True,
                    alpha=0.8,
                    marker=".",
                    marker_size=10,
                    figsize=(5, 5),
                    legend_title=description,
                    remove_x_ticks=True,
                    remove_y_ticks=True,
                    rasterized=rasterized,
                )

                if lims is None:
                    raise ValueError(
                        "lims should be set here - so that we can make plots consistent"
                    )
                else:
                    # apply stored axis limits, to make all plots consistent
                    # not necessary here because each point plotted in all figs,
                    # but will be necessary in plots below.
                    ax.set_xlim(lims[0])
                    ax.set_ylim(lims[1])

                yield (fig, f"{class_name}-{hue_key}", viz_type)


def plot_embedding_pairwise_scatterplots(df):
    """For each fold's background set, plot pairwise scatterplots of raw embeddings.
    Returns a generator of (fig, x_col, y_col), or an empty list if no plots to be made."""
    embedding_cols = df.columns[df.columns.str.startswith("embedding.")]
    if len(embedding_cols) == 0:
        # we did not attach raw embeddings - none to plot
        return []

    hue_key = "label.predicted_by.individual_sequence_model"

    # Plot all pairs of embedding columns
    for x_col, y_col in combinations(embedding_cols, 2):
        fig, ax = genetools.plots.scatterplot(
            data=df,
            x_axis_key=x_col,
            y_axis_key=y_col,
            hue_key=hue_key,
            discrete_palette=disease_color_palette,
            enable_legend=True,
            alpha=0.5,
            marker=".",
            marker_size=malid.external.genetools_plots.get_point_size(df.shape[0]),
            figsize=(5, 5),
            legend_title="Predicted sequence label",
            remove_x_ticks=True,
            remove_y_ticks=True,
            # Faster plotting: https://jbendeaton.com/blog/2011/speed-up-plot-rendering-in-pythonmatplotlib
            rasterized=True,
        )

        # Put sample sizes in legend
        genetools.plots.add_sample_size_to_legend(
            ax=ax,
            data=df,
            hue_key=hue_key,
        )
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        yield (x_col, y_col, fig)


def _stack_background_and_foreground_points_into_line_segments(
    background_points_in_order, foreground_points_in_order
):
    # reshape operation refactored out for automated tests
    return np.hstack(
        [
            foreground_points_in_order.values,
            background_points_in_order.values,
        ]
    ).reshape(-1, 2, 2)


def plot_by_participant(
    participant_dict, background_points, legend_hues=None, viz_type="umap"
):
    """Overlay single participant on fold's training set as background (background_points)

    Returns dict mapping figure name -> figure object

    If legend_hues supplied, show all these hues in legend, even if no points from them.
    """

    def _plot_all_timepoints(
        participant_dict,
        filter_to_overall_prediction_or_healthy=False,
        filter_by_prediction_confidence_value: Optional[float] = None,
        filter_to_top_percent_of_prediction_confidences: Optional[float] = None,
        filter_to_difference_between_top_two_predicted_probas_cutoff: Optional[
            float
        ] = None,
        filter_to_background_set_thresholds: bool = False,
        plot_nearest_neighbor_connections=False,
    ):
        patient_id = participant_dict["participant_label"]
        fold_id = participant_dict["fold_id"]
        timepoint_adatas = participant_dict["adata_subsets"]

        # create subplot panels
        def _mksubplots():
            n_ax = len(timepoint_adatas)
            fig, axarr = plt.subplots(nrows=1, ncols=n_ax, figsize=(n_ax * 5, 5))
            if n_ax == 1:
                # if only a single column, axarr won't be an array yet, it will just be a single axis object
                # for consistency, make this into an array with a single axis object
                axarr = [axarr]
            return fig, axarr

        fig_full, axarr_full = _mksubplots()

        # get foreground point collections for each timepoint
        timepoint_foregrounds = []
        for timepoint_data in timepoint_adatas:
            overall_prediction = timepoint_data["blended_model_predicted_label"]
            specimen_name = f"{patient_id} at {timepoint_data['timepoint']}{' (non-peak)' if not timepoint_data['is_peak'] else ''}: true label {timepoint_data['specimen_disease_subtype']}"
            specimen_description = [
                specimen_name,
                f"Single-locus metamodel prediction: {overall_prediction} ({timepoint_data['blended_model_prediction_confidence'] * 100:0.1f}%)",
            ]

            # copy before making changes, so plots don't have side effects on each other
            foreground_points = timepoint_data["foreground_points"].copy()

            # Store iloc index of each point in the foreground_points dataframe
            # So that we can find the right kNN query indices regardless of any filters applied here
            foreground_points["iloc"] = np.arange(foreground_points.shape[0])

            if filter_to_overall_prediction_or_healthy:
                # filter to foreground points whose predicted labels == overall predicted label for entire sample or == healthy
                foreground_points = foreground_points[
                    (
                        foreground_points[
                            "label.predicted_by.individual_sequence_model"
                        ]
                        == overall_prediction
                    )
                    | (
                        foreground_points[
                            "label.predicted_by.individual_sequence_model"
                        ]
                        == healthy_label
                    )
                ]
            if filter_by_prediction_confidence_value is not None:
                foreground_points = foreground_points[
                    foreground_points[
                        "label.predicted_by.individual_sequence_model.confidence"
                    ]
                    >= filter_by_prediction_confidence_value
                ]
                specimen_description.append(
                    f"Filtered to sequence-model prediction confidence $\geq {filter_by_prediction_confidence_value}$"
                )
            if filter_to_difference_between_top_two_predicted_probas_cutoff is not None:
                foreground_points = foreground_points[
                    foreground_points[
                        "difference_between_top_two_predicted_probas.by.individual_sequence_model"
                    ]
                    >= filter_to_difference_between_top_two_predicted_probas_cutoff
                ]
                specimen_description.append(
                    f"Filtered to difference_between_top_two_predicted_probas.by.individual_sequence_model >= {filter_to_difference_between_top_two_predicted_probas_cutoff:0.2f}"
                )
            if filter_to_top_percent_of_prediction_confidences is not None:
                # Take top N% of all points
                # Don't do this per-class, because it would force inclusion of every class, even those that are not likely for an individual
                foreground_points = malid.external.genetools_arrays.get_top_n_percent(
                    df=foreground_points,
                    col="label.predicted_by.individual_sequence_model.confidence",
                    fraction=filter_to_top_percent_of_prediction_confidences,
                )
                specimen_description.append(
                    f"Filtered to top {filter_to_top_percent_of_prediction_confidences*100:0.1f}% by sequence-model prediction confidence"
                )

            if filter_to_background_set_thresholds:
                # Learn the top N% cutoff for each class from the full background set, and then filter to those cutoffs in the foreground
                # We already have cutoffs almost like that: the ones used for subsetting the full background set to the plotted background points.
                # So just reapply those cutoffs to the foreground set.
                # TODO: explicitly compute the cutoffs previously and pass in, rather than recomputing on the fly many times
                for classname, background_class_df in background_points.groupby(
                    "label.predicted_by.individual_sequence_model", observed=True
                ):
                    # For each class in the background set: get min probability amongst the points selected for plotting
                    class_cutoff = background_class_df[
                        "label.predicted_by.individual_sequence_model.confidence"
                    ].min()
                    # Apply that threshold to the foreground set for this class
                    foreground_points = foreground_points[
                        (
                            (
                                foreground_points[
                                    "label.predicted_by.individual_sequence_model"
                                ]
                                == classname
                            )
                            & (
                                foreground_points[
                                    "label.predicted_by.individual_sequence_model.confidence"
                                ]
                                >= class_cutoff
                            )
                        )
                        | (
                            foreground_points[
                                "label.predicted_by.individual_sequence_model"
                            ]
                            != classname
                        )
                    ]
                specimen_description.append(
                    f"Filtered foreground set to background set's prediction confidence thresholds"
                )

            # store
            timepoint_foregrounds.append((foreground_points, specimen_description))

        # choose a consistent foreground point size for all subplots
        max_foreground_point_count = max(
            foreground_points.shape[0]
            for (foreground_points, specimen_description) in timepoint_foregrounds
        )

        for ix, (
            (foreground_points, specimen_description),
            ax,
            other_timepoint_data,
        ) in enumerate(zip(timepoint_foregrounds, axarr_full, timepoint_adatas)):
            # is_final_plot_in_row = ix == len(timepoint_adatas) - 1
            _, ax = plot_overlay(
                background_points=background_points,
                representation=viz_type,
                background_marker_size=malid.external.genetools_plots.get_point_size(
                    background_points.shape[0]
                ),
                foreground_points=foreground_points,
                foreground_palette=disease_color_palette,
                foreground_hue_key="label.predicted_by.individual_sequence_model",
                foreground_marker=".",
                foreground_marker_size=malid.external.genetools_plots.get_point_size(
                    max_foreground_point_count
                ),
                foreground_legend_title=f"Predicted sequence label {'(filtered)' if filter_to_overall_prediction_or_healthy else ''}",
                # enable_legend=True,  # is_final_plot_in_row,
                foreground_legend_hues=legend_hues
                if not filter_to_overall_prediction_or_healthy
                else None,
                plot_title="\n".join(specimen_description + [viz_type]),
                ax=ax,
                rasterized=True,
            )

            # Put sample sizes in legend
            genetools.plots.add_sample_size_to_legend(
                ax=ax,
                data=foreground_points,
                hue_key="label.predicted_by.individual_sequence_model",
            )

            # Plot nearest neighbor connections
            if plot_nearest_neighbor_connections:
                if "neighbor_links_df" not in other_timepoint_data:
                    raise ValueError(
                        f"Expected to find 'neighbor_links_df' in timepoint data but didn't. Available keys: {other_timepoint_data.keys()}"
                    )
                # each row is a link between one test set point and one training set point,
                # both specified by numerical index (not obsname)
                neighbor_links_df = other_timepoint_data["neighbor_links_df"]

                # We may have filtered down foreground_points. But we stored original iloc as a column.
                # Restrict neighbor_links_df to the remaining foreground_points:
                neighbor_links_df = neighbor_links_df[
                    neighbor_links_df["center_id"].isin(
                        foreground_points["iloc"].values
                    )
                ]

                # Extract background, foreground point pairs
                # foreground query point is stored as "center" in neighbor_links_df
                foreground_points_in_order = foreground_points.set_index("iloc").loc[
                    neighbor_links_df["center_id"].values
                ]
                # background training point is stored as "neighbor" in neighbor_links_df
                background_points_in_order = background_points.iloc[
                    neighbor_links_df["neighbor_id"].values
                ]

                # merge PCA or UMAP coordinates of foreground/center ID and background/neighbor ID
                points = _stack_background_and_foreground_points_into_line_segments(
                    background_points_in_order=background_points_in_order[
                        [f"X_{viz_type}1", f"X_{viz_type}2"]
                    ],
                    foreground_points_in_order=foreground_points_in_order[
                        [f"X_{viz_type}1", f"X_{viz_type}2"]
                    ],
                )

                # Plot line segments
                # https://stackoverflow.com/a/8068758/130164
                ax.add_collection(
                    LineCollection(
                        points,
                        linewidths=0.5,
                        colors="lightgray",
                        zorder=-1,
                        rasterized=True,
                        alpha=0.5,
                    )
                )
        return fig_full

    # Make figures
    yield (
        "all",
        _plot_all_timepoints(
            participant_dict, filter_to_overall_prediction_or_healthy=False
        ),
    )
    yield (
        "filtered",
        _plot_all_timepoints(
            participant_dict, filter_to_overall_prediction_or_healthy=True
        ),
    )
    # yield ("all_predictionconfidence_topset", _plot_all_timepoints(
    #     participant_dict,
    #     filter_to_overall_prediction_or_healthy=False,
    #     filter_by_prediction_confidence_value=0.8,
    # ))
    # yield ("filtered_predictionconfidence_topset", _plot_all_timepoints(
    #     participant_dict,
    #     filter_to_overall_prediction_or_healthy=True,
    #     filter_by_prediction_confidence_value=0.8,
    # ))
    yield (
        "all_predictionconfidence_top_percentile",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=False,
            filter_to_top_percent_of_prediction_confidences=0.3,
        ),
    )
    yield (
        "filtered_predictionconfidence_top_percentile",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=True,
            filter_to_top_percent_of_prediction_confidences=0.3,
        ),
    )
    # Apply thresholds from background set
    yield (
        "all_predictionconfidence_background_set_thresholds",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=False,
            filter_to_background_set_thresholds=True,
        ),
    )
    yield (
        "filtered_predictionconfidence_background_set_thresholds",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=True,
            filter_to_background_set_thresholds=True,
        ),
    )
    # also filter by difference between top two predicted probabilities:
    yield (
        "all_high_difference_between_toptwo",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=False,
            filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
        ),
    )
    yield (
        "filtered_high_difference_between_toptwo",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=True,
            filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
        ),
    )
    # combine
    yield (
        "all_high_difference_between_toptwo_and_predictionconfidence_top_percentile",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=False,
            filter_to_top_percent_of_prediction_confidences=0.3,
            filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
        ),
    )
    yield (
        "filtered_high_difference_between_toptwo_and_predictionconfidence_top_percentile",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=True,
            filter_to_top_percent_of_prediction_confidences=0.3,
            filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
        ),
    )
    yield (
        "all_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=False,
            filter_to_background_set_thresholds=True,
            filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
        ),
    )
    yield (
        "filtered_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds",
        _plot_all_timepoints(
            participant_dict,
            filter_to_overall_prediction_or_healthy=True,
            filter_to_background_set_thresholds=True,
            filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
        ),
    )

    if compute_knn_index:
        yield (
            "all_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=False,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "filtered_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=True,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "all_predictionconfidence_top_percentile_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=False,
                filter_to_top_percent_of_prediction_confidences=0.3,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "filtered_predictionconfidence_top_percentile_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=True,
                filter_to_top_percent_of_prediction_confidences=0.3,
                plot_nearest_neighbor_connections=True,
            ),
        )
        # Apply thresholds from background set
        yield (
            "all_predictionconfidence_background_set_thresholds_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=False,
                filter_to_background_set_thresholds=True,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "filtered_predictionconfidence_background_set_thresholds_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=True,
                filter_to_background_set_thresholds=True,
                plot_nearest_neighbor_connections=True,
            ),
        )
        # also filter by difference between top two predicted probabilities:
        yield (
            "all_high_difference_between_toptwo_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=False,
                filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "filtered_high_difference_between_toptwo_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=True,
                filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
                plot_nearest_neighbor_connections=True,
            ),
        )
        # combine
        yield (
            "all_high_difference_between_toptwo_and_predictionconfidence_top_percentile_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=False,
                filter_to_top_percent_of_prediction_confidences=0.3,
                filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "filtered_high_difference_between_toptwo_and_predictionconfidence_top_percentile_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=True,
                filter_to_top_percent_of_prediction_confidences=0.3,
                filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "all_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=False,
                filter_to_background_set_thresholds=True,
                filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
                plot_nearest_neighbor_connections=True,
            ),
        )
        yield (
            "filtered_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds_with_knn_superimposed",
            _plot_all_timepoints(
                participant_dict,
                filter_to_overall_prediction_or_healthy=True,
                filter_to_background_set_thresholds=True,
                filter_to_difference_between_top_two_predicted_probas_cutoff=0.05,
                plot_nearest_neighbor_connections=True,
            ),
        )


## Run


def generate_transformation(
    supervision_type: UmapSupervisionStrategy,
    individual_sequence_model_clf: SequenceClassifier,
    adata_train: anndata.AnnData,
    umap_labels_column: str,
    adata_train_high_confidence_subset_obsnames: np.ndarray,
):
    if supervision_type == UmapSupervisionStrategy.PARTIALLY_SUPERVISED:
        ## Full data, partially supervised
        # Supply partially supervised labels for UMAP:
        # - supply full dataset
        # - only use labels for points with high confidence (transform to numeric codes)
        # - mask the remaining labels (set value -1).
        adata_to_umap = adata_train
        umap_supervised_labels = (
            adata_to_umap.obs[umap_labels_column].astype("category").cat.codes.copy()
        )
        umap_supervised_labels[
            ~umap_supervised_labels.index.isin(
                adata_train_high_confidence_subset_obsnames
            )
        ] = -1
    elif supervision_type == UmapSupervisionStrategy.FULLY_SUPERVISED:
        # Use only the high-confidence subset for this UMAP.
        # Supply supervised labels for UMAP
        adata_to_umap = adata_train[adata_train_high_confidence_subset_obsnames]
        umap_supervised_labels = (
            adata_to_umap.obs[umap_labels_column].astype("category").cat.codes.copy()
        )
    elif supervision_type == UmapSupervisionStrategy.UNSUPERVISED:
        # Use only the high-confidence subset for this UMAP.
        adata_to_umap = adata_train[adata_train_high_confidence_subset_obsnames]
        umap_supervised_labels = None
    else:
        raise ValueError(f"Unrecognized UmapSupervisionStrategy: {supervision_type}")

    # Run UMAP and PCA
    background_points, umap_transformer, pca_transformer = embed_sequences(
        individual_sequence_model_clf,
        adata_to_umap,
        umap_transformer=None,
        pca_transformer=None,
        umap_supervised_labels=umap_supervised_labels,
        attach_true_disease_label=True,
        attach_embeddings=True,
    )

    return background_points, umap_transformer, pca_transformer


def plot_overlay(
    background_points,
    representation: str,
    background_marker_size: float,
    foreground_points: pd.DataFrame,
    foreground_palette,
    foreground_hue_key: str,
    foreground_marker_size: float,
    foreground_legend_title: str,
    plot_title: str,
    foreground_marker="o",
    foreground_legend_hues=None,
    ax=None,
    # Faster plotting: https://jbendeaton.com/blog/2011/speed-up-plot-rendering-in-pythonmatplotlib
    rasterized=True,
):
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    # plot the background, with lower zorder than any palette zorders
    genetools.plots.scatterplot(
        data=background_points,
        x_axis_key=f"X_{representation}1",
        y_axis_key=f"X_{representation}2",
        hue_key=None,
        ax=ax,
        alpha=0.3,
        marker=".",
        marker_size=background_marker_size,
        na_color="lightgray",
        marker_zorder=-25,
        remove_x_ticks=True,
        remove_y_ticks=True,
        rasterized=rasterized,
    )

    # plot the foreground
    genetools.plots.scatterplot(
        data=foreground_points,
        x_axis_key=f"X_{representation}1",
        y_axis_key=f"X_{representation}2",
        hue_key=foreground_hue_key,
        discrete_palette=foreground_palette,
        ax=ax,
        enable_legend=True,
        alpha=0.8,
        marker=foreground_marker,
        marker_size=foreground_marker_size,
        legend_title=foreground_legend_title,
        legend_hues=foreground_legend_hues,
        remove_x_ticks=True,
        remove_y_ticks=True,
        rasterized=rasterized,
    )
    ax.set_title(plot_title)
    return fig, ax


def remove_naive_isotypes(adata: anndata.AnnData):
    return adata[adata.obs["isotype_supergroup"] != "IGHD-M"]


def run_for_fold(
    fold_id: int,
    gene_locus: GeneLocus,
    specimen_metadata: pd.DataFrame,
    metamodel_name: str,
    individual_sequence_model_name: str,
    output_dir: Path,
    foreground_output_dir: Path,
    background_fold_label: str,  # which fold label to plot as background reference map
    foreground_peakset_fold_label: str,  # which fold label to use as foreground overlay for peak timepoints
    foreground_include_peak: bool,  # whether to plot people that are ONLY in peak timepoint set
    foreground_include_offpeak: bool,  # whether to plot people that are ONLY in offpeak timepoint set
    offpeak_include_diseases_outside_training_set: bool,  # whether to plot offpeak patients whose diseases are never in training set
    supervision_type: UmapSupervisionStrategy,
    # one of these two must be provided:
    high_confidence_top_n_per_class: Optional[int] = None,
    high_confidence_top_percent_per_class: Optional[float] = 0.2,
    filter_naive_isotypes=False,
    difference_between_top_two_predicted_probas_cutoff=0.0,
    # then this can optionally be provided to subsample a certain N from the set chosen above
    subsample_n_from_chosen: Optional[int] = None,
    target_obs_column: TargetObsColumnEnum = TargetObsColumnEnum.disease,
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE,
):
    """
    the sample_weight_strategy will not be used to reweight sequences for the sequence-level UMAP,
    but will be used to load the sequence-level model that was trained with knoweldge of sequence weights,
    and to reweight sequences for the specimen-level processing in the metamodel
    """
    ## Based on feature flags, enable/disable GPU
    # Allow GPU usage globally, for kNN index, if computing kNN graph to superimpose on UMAP
    choosegpu.configure_gpu(enable=compute_knn_index)

    # defensive cast
    output_dir = Path(output_dir)
    foreground_output_dir = Path(foreground_output_dir)

    # create GeneLocus subdirectories
    output_dir = output_dir / gene_locus.name
    foreground_output_dir = foreground_output_dir / gene_locus.name
    output_dir.mkdir(parents=True, exist_ok=True)
    foreground_output_dir.mkdir(parents=True, exist_ok=True)

    # Load background data for fold
    logger.info(
        f"Loading background data from fold {fold_id}, {gene_locus}, with sample_weight_strategy {sample_weight_strategy}"
    )
    # Load peak training set for this fold - this will be the gray background
    adata_train = io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label=background_fold_label,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    if filter_naive_isotypes:
        adata_train = remove_naive_isotypes(adata_train)

    # Load models
    logger.info(
        f"Loading models from fold {fold_id}, {gene_locus}, with sample_weight_strategy {sample_weight_strategy}"
    )
    (blending_metamodel_clf, individual_sequence_model_clf,) = load_models_for_fold(
        fold_id=fold_id,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        metamodel_name=metamodel_name,
        individual_sequence_model_name=individual_sequence_model_name,
        sample_weight_strategy=sample_weight_strategy,
    )

    # Get plot background, creating UMAP transformation used for each overlaid patient. This is dependent on how we threshold the adata_train set
    logger.info(f"Computing plot background (training set) from fold {fold_id}")

    # Use model predictions to choose subset of sequences
    # If training umap transformer only on background points with high "label.predicted_by.individual_sequence_model.confidence",
    # need to first get those predicted probabilities.

    # TODO: batching?
    # TODO: don't have this be rerun on the subset later if already computed here?

    # Get probabilities and index with anndata's obs_names
    featurized_sequences = individual_sequence_model_clf.featurize(adata_train)
    predicted_point_probabilities = pd.DataFrame(
        individual_sequence_model_clf.predict_proba(featurized_sequences.X),
        index=featurized_sequences.sample_names,
        columns=individual_sequence_model_clf.classes_,
    )

    # get winning label probability, 2nd highest probability, and the diff, as well as the top two labels
    predicted_point_probabilities_top_two = _get_top_two_probabilities(
        predicted_point_probabilities
    ).rename(
        columns={
            "max_probability": "label.predicted_by.individual_sequence_model.confidence",
            "second_highest_probability": "second_highest_confidence.predicted_by.individual_sequence_model",
            "difference_between_top_two_predicted_probas": "difference_between_top_two_predicted_probas.by.individual_sequence_model",
            "max_probability_label": "label.predicted_by.individual_sequence_model",
            "second_highest_probability_label": "second_highest_label.predicted_by.individual_sequence_model",
        }
    )

    # store
    adata_train.obs[
        "label.predicted_by.individual_sequence_model"
    ] = predicted_point_probabilities_top_two[
        "label.predicted_by.individual_sequence_model"
    ]
    adata_train.obs[
        "label.predicted_by.individual_sequence_model.confidence"
    ] = predicted_point_probabilities_top_two[
        "label.predicted_by.individual_sequence_model.confidence"
    ]
    adata_train.obs[
        "difference_between_top_two_predicted_probas.by.individual_sequence_model"
    ] = predicted_point_probabilities_top_two[
        "difference_between_top_two_predicted_probas.by.individual_sequence_model"
    ]

    # Train umap transformer only on background points with high "label.predicted_by.individual_sequence_model.confidence"
    # Procedure:
    # 1) For each class, take all background sequences predicted to be in that class
    # 2) Subset to only those that also originated from a patient of that class (i.e. their transferred pseudolabel matches predicted label)
    # 3) Sort by prediction confidence
    # 4) Take top N=high_confidence_top_n_per_class from the filtered and sorted list of this class's sequences
    # 5) Do this for all classes and combine
    adata_train_high_confidence_subset_obsnames = get_top_ranked_subset_obsnames(
        adata=adata_train,
        class_label_column="label.predicted_by.individual_sequence_model",
        rank_by_column="label.predicted_by.individual_sequence_model.confidence",
        top_n_per_class=high_confidence_top_n_per_class,
        top_percent_per_class=high_confidence_top_percent_per_class,
        must_match_column="disease.separate_past_exposures",
        difference_between_top_two_predicted_probas_cutoff=difference_between_top_two_predicted_probas_cutoff,
        difference_between_top_two_predicted_probas_column="difference_between_top_two_predicted_probas.by.individual_sequence_model",
        subsample_n_from_chosen=subsample_n_from_chosen,
    )

    # Run UMAP and PCA
    background_points, umap_transformer, pca_transformer = generate_transformation(
        supervision_type=supervision_type,
        individual_sequence_model_clf=individual_sequence_model_clf,
        adata_train=adata_train,
        umap_labels_column="label.predicted_by.individual_sequence_model",  # alternative: "disease.separate_past_exposures"
        adata_train_high_confidence_subset_obsnames=adata_train_high_confidence_subset_obsnames,
    )

    # Generate kNN index from logits of high-confidence subset of training set
    knn_index = None
    embedding_columns = background_points.columns[
        background_points.columns.str.startswith("embedding.")
    ]
    if compute_knn_index:
        from malid.knn import _fit_knn_index

        background_logits = background_points[embedding_columns]
        # Don't PCA because already fairly low-dimensional logits (one per class)
        # But maybe scale? TODO
        knn_index = _fit_knn_index(X=background_logits, metric="euclidean")

    # Plot fold background UMAP and PCA
    for (fig, hue_name, viz_type) in plot_fold_background(background_points):
        plt.title(f"Fold {fold_id} {viz_type} - subset of training set (background)")
        genetools.plots.savefig(
            fig,
            (
                output_dir
                if hue_name == "label.patient_of_origin"
                else foreground_output_dir
            )
            / f"background.fold.{fold_id}.{slugify(hue_name)}.{viz_type}.png",
            dpi=72,
        )
        genetools.plots.savefig(
            fig,
            foreground_output_dir
            / f"background.fold.{fold_id}.{slugify(hue_name)}.{viz_type}.pdf",
        )
        plt.close(fig)

    # Plot pairwise scatterplots from raw embeddings (pre PCA or UMAP)
    pairwise_scatterplots = []
    for (x_col, y_col, fig) in plot_embedding_pairwise_scatterplots(background_points):
        plt.title(f"Fold {fold_id} - pairwise scatterplot: logits - {x_col} vs {y_col}")

        fname_out = f"pairwise_scatterplot.{fold_id}.{slugify(x_col)}.{slugify(y_col)}"
        genetools.plots.savefig(
            fig, foreground_output_dir / (fname_out + ".png"), dpi=72
        )
        genetools.plots.savefig(fig, foreground_output_dir / (fname_out + ".pdf"))
        pairwise_scatterplots.append(
            {
                "x_col": x_col,
                "y_col": y_col,
                "fname_out": foreground_output_dir / (fname_out + ".png"),
            }
        )
        plt.close(fig)

    joblib.dump(
        pairwise_scatterplots,
        foreground_output_dir / f"pairwise_scatterplot.{fold_id}.list.joblib",
    )

    ## Save out background points
    background_points.to_csv(
        foreground_output_dir / f"background_points.fold.{fold_id}.embeddings.tsv",
        sep="\t",
    )
    joblib.dump(
        {"umap": umap_transformer, "pca": pca_transformer},
        foreground_output_dir / f"umap_pca_transformers.fold.{fold_id}.joblib",
    )

    ## Get plot foreground - run models on each test sample
    participant_labels = []
    # Load data for fold into participant dictionaries (with a generator)
    logger.info(
        f"Loading foreground data from fold {fold_id} with sample_weight_strategy {sample_weight_strategy}"
    )
    for participant_dict in load_overlay_data_for_fold(
        fold_id=fold_id,
        specimen_metadata=specimen_metadata,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
        foreground_peakset_fold_label=foreground_peakset_fold_label,
        foreground_include_peak=foreground_include_peak,
        foreground_include_offpeak=foreground_include_offpeak,
        offpeak_include_diseases_outside_training_set=offpeak_include_diseases_outside_training_set,
        filter_naive_isotypes=filter_naive_isotypes,
    ):
        participant_label = participant_dict["participant_label"]
        if len(participant_dict["adata_subsets"]) == 0:
            logger.warning(
                f"No adata_subsets for overlay participant {participant_label} from fold {participant_dict['fold_id']} -- skipping."
            )
            continue
        participant_labels.append(participant_label)
        logger.info(
            f"Processing overlay participant {participant_label} from fold {participant_dict['fold_id']}"
        )
        for subset_dict in participant_dict["adata_subsets"]:
            subset_adata = subset_dict["adata"]
            helpers.confirm_all_sequences_from_same_specimen(subset_adata)

            # Run blended model and individual-sequence model. get prediction confidences too
            (
                blended_model_predicted_label,
                blended_model_prediction_confidence,
            ) = run_blended_model(
                clf=blending_metamodel_clf,
                subset_adata=subset_adata,
                gene_locus=gene_locus,
            )

            # Get first entry because only one specimen
            blended_model_predicted_label = blended_model_predicted_label[0]
            blended_model_prediction_confidence = blended_model_prediction_confidence[0]

            foreground_points, _, _ = embed_sequences(
                individual_sequence_model_clf,
                subset_adata,
                umap_transformer=umap_transformer,
                pca_transformer=pca_transformer,
                attach_embeddings=True,
            )

            subset_dict.update(
                {
                    "adata": subset_adata,
                    "blended_model_predicted_label": blended_model_predicted_label,
                    "blended_model_prediction_confidence": blended_model_prediction_confidence,
                    "foreground_points": foreground_points,
                }
            )

            # Query kNN index
            if compute_knn_index:
                from malid.knn import _get_neighbors

                # for each test set point, get single closest neighbor in train set
                foreground_logits = foreground_points[embedding_columns]
                # Don't PCA because already fairly low-dimensional logits (one per class)
                # But consider scaling?
                subset_dict.update(
                    {
                        "neighbor_links_df": _get_neighbors(
                            knn_index=knn_index,
                            data_X_contiguous=foreground_logits,
                            n_neighbors=1,
                        )
                    }
                )

        # Plot fold foreground
        for viz_type in ["umap", "pca"]:
            for fig_name, fig in plot_by_participant(
                participant_dict,
                background_points,
                legend_hues=individual_sequence_model_clf.classes_,
                viz_type=viz_type,
            ):
                genetools.plots.savefig(
                    fig,
                    foreground_output_dir
                    / f"timecourse.fold.{fold_id}.{participant_dict['participant_label']}.{fig_name}.scatter.{viz_type}.png",
                    dpi=300,
                )
                genetools.plots.savefig(
                    fig,
                    foreground_output_dir
                    / f"timecourse.fold.{fold_id}.{participant_dict['participant_label']}.{fig_name}.scatter.{viz_type}.pdf",
                )
                plt.close(fig)

        # Done with this participant, so trash all subset_dicts to free RAM
        del participant_dict["adata_subsets"]
        gc.collect()

    # Export participant_labels that we analyze - for summary nb
    joblib.dump(
        participant_labels,
        foreground_output_dir / f"participant_labels.fold.{fold_id}.joblib",
    )

    # Clear dataset cache
    del adata_train
    io.clear_cached_fold_embeddings()
    gc.collect()
