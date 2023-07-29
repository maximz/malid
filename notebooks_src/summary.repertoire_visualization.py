# %%

# %%
import numpy as np
import pandas as pd

# %%
from malid import config, helpers
from malid.datamodels import GeneLocus
from summarynb import show, image, chunks, table
import joblib

# %%
from IPython.display import display, Markdown

# %%
from slugify import slugify

# %%
import os

# %%
config.embedder.name

# %%
output_dirs = {
    gene_locus: config.paths.supervised_embedding_output_dir / gene_locus.name
    for gene_locus in config.gene_loci_used
}
foreground_output_dirs = {
    gene_locus: config.paths.supervised_embedding_foreground_output_dir
    / gene_locus.name
    for gene_locus in config.gene_loci_used
}
output_dirs, foreground_output_dirs

# %%

# %% [markdown]
# # Compare pairwise scatterplots side-by-side across folds

# %%
diseases = helpers.diseases_in_peak_timepoint_dataset()
for gene_locus in config.gene_loci_used:
    pairwise_scatterplots = pd.concat(
        [
            pd.DataFrame(
                joblib.load(
                    foreground_output_dirs[gene_locus]
                    / f"pairwise_scatterplot.{fold_id}.list.joblib"
                )
            ).assign(fold_id=fold_id)
            for fold_id in config.all_fold_ids
        ],
        axis=0,
    )
    display(Markdown(f"## Fold training set - predicted label - UMAP - {gene_locus}"))
    show(
        [
            foreground_output_dirs[gene_locus]
            / f"background.fold.{fold_id}.label-predicted-by-individual-sequence-model.umap.png"
            for fold_id in config.all_fold_ids
        ],
        headers=[f"fold {fold_id}" for fold_id in config.all_fold_ids],
    )

    display(
        Markdown(f"## Fold training set - true (noisy) label - UMAP - {gene_locus}")
    )
    show(
        [
            output_dirs[gene_locus]
            / f"background.fold.{fold_id}.label-patient-of-origin.umap.png"
            for fold_id in config.all_fold_ids
        ],
        headers=[f"fold {fold_id}" for fold_id in config.all_fold_ids],
    )

    display(Markdown(f"## Fold training set - predicted label - PCA - {gene_locus}"))
    show(
        [
            foreground_output_dirs[gene_locus]
            / f"background.fold.{fold_id}.label-predicted-by-individual-sequence-model.pca.png"
            for fold_id in config.all_fold_ids
        ],
        headers=[f"fold {fold_id}" for fold_id in config.all_fold_ids],
    )

    display(Markdown(f"## Fold training set - true (noisy) label - PCA - {gene_locus}"))
    show(
        [
            output_dirs[gene_locus]
            / f"background.fold.{fold_id}.label-patient-of-origin.pca.png"
            for fold_id in config.all_fold_ids
        ],
        headers=[f"fold {fold_id}" for fold_id in config.all_fold_ids],
    )

    display(
        Markdown(
            f"## Diagnostics per class: max label confidence - UMAP - {gene_locus}"
        )
    )
    for fold_id in config.all_fold_ids:
        display(Markdown(f"### Fold {fold_id}"))
        show(
            [
                foreground_output_dirs[gene_locus]
                / f"background.fold.{fold_id}.{slugify(disease)}-label-predicted-by-individual-sequence-model-confidence.umap.png"
                for disease in diseases
            ],
            headers=diseases,
        )

    display(
        Markdown(
            f"## Diagnostics per class: difference between top two probabilities - UMAP - {gene_locus}"
        )
    )
    for fold_id in config.all_fold_ids:
        display(Markdown(f"### Fold {fold_id}"))
        show(
            [
                foreground_output_dirs[gene_locus]
                / f"background.fold.{fold_id}.{slugify(disease)}-difference-between-top-two-predicted-probas-by-individual-sequence-model.umap.png"
                for disease in diseases
            ],
            headers=diseases,
        )

    for (x_col, y_col), grp in pairwise_scatterplots.groupby(
        ["x_col", "y_col"], observed=True
    ):
        display(Markdown(f"## {x_col} vs {y_col} - {gene_locus}"))
        show(
            grp["fname_out"].tolist(),
            headers=[f"fold {fold_id}" for fold_id in grp["fold_id"]],
        )

# %%

# %%

# %% [markdown]
# # Supervised embeddings for off-peak specimens

# %%

# %%
for fold_id in config.all_fold_ids:
    for gene_locus in config.gene_loci_used:
        patient_names = sorted(
            joblib.load(
                foreground_output_dirs[gene_locus]
                / f"participant_labels.fold.{fold_id}.joblib"
            )
        )
        display(Markdown(f"## Fold {fold_id}, {gene_locus}"))
        show(
            [
                foreground_output_dirs[gene_locus]
                / f"background.fold.{fold_id}.label-predicted-by-individual-sequence-model.umap.png",
                output_dirs[gene_locus]
                / f"background.fold.{fold_id}.label-patient-of-origin.umap.png",
            ],
            headers=[
                "Fold training set - predicted label",
                "Fold training set - true (noisy) label",
            ],
            max_width=500,
        )

        pairwise_scatterplots = joblib.load(
            foreground_output_dirs[gene_locus]
            / f"pairwise_scatterplot.{fold_id}.list.joblib"
        )
        show(chunks([d["fname_out"] for d in pairwise_scatterplots], 3), max_width=500)

        for patient in patient_names:
            disease_label = (
                helpers.get_specimens_for_participant(patient)[
                    "disease.separate_past_exposures"
                ]
                .drop_duplicates()
                .squeeze()
            )
            display(Markdown(f"### {patient} - {disease_label}, {gene_locus}"))
            show(
                [
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered.scatter.umap.png",
                    ],
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_with_knn_superimposed.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_with_knn_superimposed.scatter.umap.png",
                    ],
                ],
                headers=["All", "Filtered to predicted class + Healthy only"],
                max_width=1000,
            )

            show(
                [
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_predictionconfidence_top_percentile.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_predictionconfidence_top_percentile.scatter.umap.png",
                    ],
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_predictionconfidence_top_percentile_with_knn_superimposed.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_predictionconfidence_top_percentile_with_knn_superimposed.scatter.umap.png",
                    ],
                ],
                headers=[
                    "All - top percentile of prediction confidence",
                    "Filtered to predicted class + Healthy only",
                ],
                max_width=1000,
            )

            show(
                [
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_predictionconfidence_background_set_thresholds.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_predictionconfidence_background_set_thresholds.scatter.umap.png",
                    ],
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_predictionconfidence_background_set_thresholds_with_knn_superimposed.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_predictionconfidence_background_set_thresholds_with_knn_superimposed.scatter.umap.png",
                    ],
                ],
                headers=[
                    "All - with background's thresholds of prediction confidence",
                    "Filtered to predicted class + Healthy only",
                ],
                max_width=1000,
            )

            show(
                [
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_high_difference_between_toptwo.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_high_difference_between_toptwo.scatter.umap.png",
                    ],
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_high_difference_between_toptwo_with_knn_superimposed.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_high_difference_between_toptwo_with_knn_superimposed.scatter.umap.png",
                    ],
                ],
                headers=[
                    "All - high difference between top 2 predicted probas",
                    "Filtered to predicted class + Healthy only",
                ],
                max_width=1000,
            )

            show(
                [
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_high_difference_between_toptwo_and_predictionconfidence_top_percentile.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_high_difference_between_toptwo_and_predictionconfidence_top_percentile.scatter.umap.png",
                    ],
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_high_difference_between_toptwo_and_predictionconfidence_top_percentile_with_knn_superimposed.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_high_difference_between_toptwo_and_predictionconfidence_top_percentile_with_knn_superimposed.scatter.umap.png",
                    ],
                ],
                headers=[
                    "All - high diff between top2 probas, and top percentile of remaining",
                    "Filtered to predicted class + Healthy only",
                ],
                max_width=1000,
            )

            show(
                [
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds.scatter.umap.png",
                    ],
                    [
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.all_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds_with_knn_superimposed.scatter.umap.png",
                        foreground_output_dirs[gene_locus]
                        / f"timecourse.fold.{fold_id}.{patient}.filtered_high_difference_between_toptwo_and_predictionconfidence_background_set_thresholds_with_knn_superimposed.scatter.umap.png",
                    ],
                ],
                headers=[
                    "All - high diff between top2 probas, and background's thresholds of prediction confidence",
                    "Filtered to predicted class + Healthy only",
                ],
                max_width=1000,
            )

# %%

# %%
