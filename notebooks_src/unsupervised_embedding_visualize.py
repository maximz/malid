# %%

# %% [markdown]
# # Plot PCA and UMAP of language model embedding of test sets ("unsupervised embedding")
#
# - Data was scaled using training set's scaler
# - PCA was computed already, using training set's PCA
# - UMAP computed from scratch (first on training set, then applied to test set)
#
# This is different from supervised embedding which comes out of the trained classifiers.

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns
import pandas as pd
import genetools
from genetools.palette import HueValueStyle
import gc
from slugify import slugify
from typing import Optional

import malid.external.genetools_plots
import malid.external.genetools_scanpy_helpers

sns.set_style("dark")

import scanpy as sc

from malid import config, io, logger, helpers
from malid.datamodels import TargetObsColumnEnum, healthy_label
from malid.external import genetools_plots
import choosegpu

choosegpu.configure_gpu(enable=True, memory_pool=True)

# %%

# %%

# %%
disease_color_palette = helpers.disease_color_palette.copy()
disease_color_palette[healthy_label] = HueValueStyle(
    color=disease_color_palette[healthy_label], zorder=-15, alpha=0.5
)


# %%
def plot_background(
    all_points, representation, fold_id, fold_label, gene_locus, plt_quantile=0.01
):
    # plot the background, with lower zorder than any palette zorders
    fig, ax = genetools.plots.scatterplot(
        data=all_points,
        x_axis_key=f"X_{representation}1",
        y_axis_key=f"X_{representation}2",
        hue_key="disease",
        discrete_palette=disease_color_palette,
        alpha=0.5,
        marker=".",
        marker_size=malid.external.genetools_plots.get_point_size(all_points.shape[0]),
        figsize=(5, 5),
        legend_title="Disease",
        enable_legend=True,
        remove_x_ticks=True,
        remove_y_ticks=True,
    )

    ax.set_title(
        f"Fold {fold_id} {fold_label}, {representation}, {gene_locus} - all diseases"
    )

    # Zoom in
    ax.set_xlim(
        np.quantile(all_points[f"X_{representation}1"], plt_quantile),
        np.quantile(all_points[f"X_{representation}1"], 1 - plt_quantile),
    )
    ax.set_ylim(
        np.quantile(all_points[f"X_{representation}2"], plt_quantile),
        np.quantile(all_points[f"X_{representation}2"], 1 - plt_quantile),
    )

    # Put sample sizes in legend
    genetools.plots.add_sample_size_to_legend(
        ax=ax,
        data=all_points,
        hue_key="disease",
    )

    return fig, ax


# %%
def plot_within_disease(
    all_points,
    disease,
    representation,
    fold_id,
    fold_label,
    gene_locus,
    plt_quantile=0.01,
):
    foreground_points = all_points[all_points["disease"] == disease]

    fig, ax = plt.subplots(figsize=(5, 5))

    foreground_hue_key = "study_name"
    foreground_palette = helpers.study_name_color_palette
    foreground_legend_title = "study name"
    foreground_marker_size = malid.external.genetools_plots.get_point_size(
        all_points.shape[0]
    )
    foreground_marker = "o"

    plot_title = (
        f"{disease}, fold {fold_id} {fold_label}, {representation}, {gene_locus}"
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
        alpha=0.5,
        marker=foreground_marker,
        marker_size=foreground_marker_size,
        legend_title=foreground_legend_title,
        remove_x_ticks=False,
        remove_y_ticks=False,
    )
    ax.set_title(plot_title)

    # Zoom in
    ax.set_xlim(
        np.quantile(all_points[f"X_{representation}1"], plt_quantile),
        np.quantile(all_points[f"X_{representation}1"], 1 - plt_quantile),
    )
    ax.set_ylim(
        np.quantile(all_points[f"X_{representation}2"], plt_quantile),
        np.quantile(all_points[f"X_{representation}2"], 1 - plt_quantile),
    )

    ax.set_aspect("equal", "datalim")  # change axes limits to get 1:1 aspect

    # Put sample sizes in legend
    genetools.plots.add_sample_size_to_legend(
        ax=ax,
        data=foreground_points,
        hue_key=foreground_hue_key,
    )

    return fig, ax


# %%
def plot_within_disease_overall_density(
    all_points,
    disease,
    representation,
    fold_id,
    fold_label,
    gene_locus,
    plt_quantile=0.01,
):
    xcol = f"X_{representation}1"
    ycol = f"X_{representation}2"
    xlims = (
        np.quantile(all_points[xcol], plt_quantile),
        np.quantile(all_points[xcol], 1 - plt_quantile),
    )
    ylims = (
        np.quantile(all_points[ycol], plt_quantile),
        np.quantile(all_points[ycol], 1 - plt_quantile),
    )

    foreground_points = all_points[all_points["disease"] == disease]

    # filter down
    # TODO: do this early in other methods too?
    foreground_points = foreground_points[
        (foreground_points[xcol] >= xlims[0])
        & (foreground_points[xcol] <= xlims[1])
        & (foreground_points[ycol] >= ylims[0])
        & (foreground_points[ycol] <= ylims[1])
    ]

    fig, ax = plt.subplots(figsize=(5, 5))

    # set minimum count for background cells: https://stackoverflow.com/a/5405654/130164
    # also set grid size
    hexplotted = ax.hexbin(
        foreground_points[xcol],
        foreground_points[ycol],
        mincnt=10,
        gridsize=25,
        cmap="Blues",
    )

    # Add color bar.
    # see also https://stackoverflow.com/a/44642014/130164
    # Pull colorbar out of axis by creating a special axis for the colorbar - rather than distorting main ax.
    # specify width and height relative to parent bbox
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    colorbar_ax = inset_axes(
        ax,
        width="5%",
        height="80%",
        loc="center left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    colorbar = fig.colorbar(hexplotted, cax=colorbar_ax, label="Density")

    # set global "current axes" back to main axes,
    # so that any calls like plt.title target main ax rather than inset colorbar_ax
    plt.sca(ax)

    ax.set_title(
        f"{disease}, fold {fold_id} {fold_label}, {representation}, {gene_locus}"
    )

    ax.set_aspect("equal", "datalim")  # change axes limits to get 1:1 aspect

    return fig, ax


# %%
def plot_within_disease_relative_density(
    all_points,
    disease,
    representation,
    positive_class,  # which study name to use as numerator in proportion
    fold_id,
    fold_label,
    gene_locus,
    n_bins=25,  # per dimension
    minimal_bin_density_quantile: Optional[float] = 0.50,  # drop bins with low counts
    plt_quantile=0.01,  # zoom in
):
    foreground_points = all_points[all_points["disease"] == disease]

    xlims = (
        np.quantile(all_points[f"X_{representation}1"], plt_quantile),
        np.quantile(all_points[f"X_{representation}1"], 1 - plt_quantile),
    )
    ylims = (
        np.quantile(all_points[f"X_{representation}2"], plt_quantile),
        np.quantile(all_points[f"X_{representation}2"], 1 - plt_quantile),
    )

    fig, ax, description = genetools_plots.two_class_relative_density_plot(
        data=foreground_points,
        x_key=f"X_{representation}1",
        y_key=f"X_{representation}2",
        hue_key="study_name",
        positive_class=positive_class,
        colorbar_label=f"Proportion of {positive_class} vs all {disease}",
        quantile=minimal_bin_density_quantile,
        figsize=(5, 5),
        n_bins=n_bins,
        range=(xlims, ylims),  # Only use zoom extent
    )

    ax.set_title(
        f"{disease}, fold {fold_id} {fold_label}, {representation}, {gene_locus}\n{description}"
    )

    # Zoom in
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_aspect("equal", "datalim")  # change axes limits to get 1:1 aspect

    return fig, ax


# %%

# %%
fold_label = "test"

# %%
for gene_locus in config.gene_loci_used:
    for fold_id in config.cross_validation_fold_ids:
        logger.info(f"Processing fold {fold_id}-{fold_label}, {gene_locus}")

        # Load test set
        adata = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=gene_locus,
            target_obs_column=TargetObsColumnEnum.disease,
        )
        assert not adata.obs["study_name"].isna().any()

        # Construct UMAP for each test set anndata
        # It's dependent on training set
        # Fit on training set (loaded and thrown away), apply to test set. Both are already scaled.
        # (This has already been done for PCA)
        _, adata = malid.external.genetools_scanpy_helpers.umap_train_and_test_anndatas(
            adata_train=io.load_fold_embeddings(
                fold_id=fold_id,
                fold_label="train_smaller",
                gene_locus=gene_locus,
                target_obs_column=TargetObsColumnEnum.disease,
            ),
            adata_test=adata,
            n_neighbors=15,
            n_components=2,
            inplace=True,
            random_state=0,
            use_rapids=True,
            use_pca=True,
        )

        # add PCA and UMAP to obs
        obsm_df = adata.obsm.to_df()
        adata.obs = genetools.helpers.horizontal_concat(
            adata.obs,
            obsm_df[obsm_df.columns[obsm_df.columns.str.startswith("X_umap")]],
        )
        adata.obs = genetools.helpers.horizontal_concat(
            adata.obs,
            obsm_df[["X_pca1", "X_pca2"]],
        )

        all_points = adata.obs

        for representation in ["umap", "pca"]:
            # Plot all diseases
            fig, ax = plot_background(
                all_points=all_points,
                representation=representation,
                fold_id=fold_id,
                fold_label=fold_label,
                gene_locus=gene_locus,
            )
            genetools.plots.savefig(
                fig,
                config.paths.output_dir
                / f"language_model_embedding.all_diseases.{representation}.fold_{fold_id}_{fold_label}.{gene_locus.name}.png",
                dpi=72,
            )
            plt.close(fig)

            # Compare by batch
            for disease in all_points["disease"].unique():
                fig, ax = plot_within_disease(
                    all_points=all_points,
                    disease=disease,
                    representation=representation,
                    fold_id=fold_id,
                    fold_label=fold_label,
                    gene_locus=gene_locus,
                )
                genetools.plots.savefig(
                    fig,
                    config.paths.high_res_outputs_dir
                    / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.fold_{fold_id}_{fold_label}.{gene_locus.name}.png",
                    dpi=72,
                )
                plt.close(fig)

                # Plot overall density
                fig, ax = plot_within_disease_overall_density(
                    all_points=all_points,
                    disease=disease,
                    representation=representation,
                    fold_id=fold_id,
                    fold_label=fold_label,
                    gene_locus=gene_locus,
                )
                genetools.plots.savefig(
                    fig,
                    config.paths.high_res_outputs_dir
                    / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.overall_density.fold_{fold_id}_{fold_label}.{gene_locus.name}.png",
                    dpi=72,
                )
                plt.close(fig)

                # Plot relative density, if we have two batches only
                study_names = all_points[all_points["disease"] == disease][
                    "study_name"
                ].unique()
                if len(study_names) == 2:
                    fig, ax = plot_within_disease_relative_density(
                        all_points=all_points,
                        disease=disease,
                        representation=representation,
                        # which study name to use as numerator in proportion
                        positive_class=study_names[1],
                        fold_id=fold_id,
                        fold_label=fold_label,
                        gene_locus=gene_locus,
                        minimal_bin_density_quantile=None,
                    )
                    genetools.plots.savefig(
                        fig,
                        config.paths.high_res_outputs_dir
                        / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.relative_density.fold_{fold_id}_{fold_label}.{gene_locus.name}.png",
                        dpi=72,
                    )
                    plt.close(fig)

                    fig, ax = plot_within_disease_relative_density(
                        all_points=all_points,
                        disease=disease,
                        representation=representation,
                        # which study name to use as numerator in proportion
                        positive_class=study_names[1],
                        fold_id=fold_id,
                        fold_label=fold_label,
                        gene_locus=gene_locus,
                        minimal_bin_density_quantile=0.50,
                    )
                    genetools.plots.savefig(
                        fig,
                        config.paths.high_res_outputs_dir
                        / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.relative_density_with_min_density_requirement.fold_{fold_id}_{fold_label}.{gene_locus.name}.png",
                        dpi=72,
                    )
                    plt.close(fig)

        del adata
        io.clear_cached_fold_embeddings()
        gc.collect()

# %%

# %%

# %%

# %%

# %%

# %%

# %%
