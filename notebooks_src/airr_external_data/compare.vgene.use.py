# %%

# %% [markdown]
# # Compare V gene and J gene use in in-house data versus Adaptive data
#
# We have already run `vgene_usage_stats.ipynb` for both datasets

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

from malid import config, io, helpers
from malid.datamodels import (
    healthy_label,
    GeneLocus,
    TargetObsColumnEnum,
    CrossValidationSplitStrategy,
    DataSource,
)
import gc
import joblib
from kdict import kdict
import itertools
import genetools
from pathlib import Path
from slugify import slugify
import scanpy as sc
import anndata


# %%

# %%

# %%
def get_dirs(
    gene_locus: GeneLocus, cross_validation_split_strategy: CrossValidationSplitStrategy
):
    paths = config.make_paths(
        embedder=config.embedder,
        cross_validation_split_strategy=cross_validation_split_strategy,
        dataset_version=config.dataset_version,
    )
    output_dir = (
        paths.model_interpretations_for_selected_cross_validation_strategy_output_dir
        / gene_locus.name
    )
    highres_output_dir = (
        paths.high_res_outputs_dir_for_cross_validation_strategy
        / "model_interpretations"
        / gene_locus.name
    )

    # Create directories - though these directories should already have been created by sequence model interpretations notebooks
    output_dir.mkdir(parents=True, exist_ok=True)
    highres_output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, highres_output_dir


# %%
# reimport (can resume here)
def import_v_gene_counts(
    gene_locus: GeneLocus, cross_validation_split_strategy: CrossValidationSplitStrategy
):
    output_dir, highres_output_dir = get_dirs(
        gene_locus, cross_validation_split_strategy
    )

    specimen_v_gene_counts_df = pd.read_csv(
        highres_output_dir / "v_gene_counts_by_specimen.tsv.gz", sep="\t"
    )

    # subselect to test folds only (which also excludes global fold -1), and set index
    specimen_v_gene_counts_df_test_only = specimen_v_gene_counts_df[
        specimen_v_gene_counts_df["fold_label"] == "test"
    ]

    # confirm only one entry per specimen now
    assert not specimen_v_gene_counts_df_test_only["specimen_label"].duplicated().any()
    specimen_v_gene_counts_df_test_only = specimen_v_gene_counts_df_test_only.set_index(
        "specimen_label"
    ).drop(["fold_id", "fold_label"], axis=1)

    # fill na
    specimen_v_gene_counts_df_test_only = specimen_v_gene_counts_df_test_only.fillna(0)

    v_gene_cols = specimen_v_gene_counts_df_test_only.columns
    v_gene_cols = v_gene_cols[~v_gene_cols.isin(["disease"])]

    # get filtered subset of v_gene_cols, produced previously
    v_gene_cols_filtered = pd.read_csv(output_dir / "meaningful_v_genes.txt")[
        "v_gene"
    ].values
    assert all(vgene in v_gene_cols for vgene in v_gene_cols_filtered)  # sanity check

    return specimen_v_gene_counts_df_test_only, v_gene_cols, v_gene_cols_filtered


# %%

# %%
extra_specimen_metadata = helpers.get_all_specimen_info()
extra_specimen_metadata

# %%
(
    specimen_v_gene_counts_df_test_only,
    v_gene_cols,
    v_gene_cols_filtered,
) = import_v_gene_counts(
    gene_locus=GeneLocus.TCR,
    cross_validation_split_strategy=CrossValidationSplitStrategy.in_house_peak_disease_timepoints,
)

# Get V gene usage proportions per specimen
v_gene_usage_proportions_by_specimen = pd.concat(
    [
        genetools.stats.normalize_rows(
            specimen_v_gene_counts_df_test_only[v_gene_cols]
        ),
        specimen_v_gene_counts_df_test_only["disease"],
    ],
    axis=1,
)

v_gene_usage_proportions_by_specimen_annot = genetools.helpers.merge_into_left(
    v_gene_usage_proportions_by_specimen[v_gene_cols],
    extra_specimen_metadata.set_index("specimen_label"),
)


# drop specimens with group N/A
v_gene_usage_proportions_by_specimen_annot = (
    v_gene_usage_proportions_by_specimen_annot.dropna(subset=["study_name"])
)

v_gene_usage_proportions_by_specimen_annot_melt = pd.melt(
    v_gene_usage_proportions_by_specimen_annot,
    id_vars=["disease", "study_name"],
    value_vars=v_gene_cols,
    var_name="V gene",
    value_name="Proportion",
)
v_gene_usage_proportions_by_specimen_annot_melt

# %%

# %%
(
    specimen_v_gene_counts_df_test_only_adaptive,
    v_gene_cols_adaptive,
    v_gene_cols_filtered_adaptive,
) = import_v_gene_counts(
    gene_locus=GeneLocus.TCR,
    cross_validation_split_strategy=CrossValidationSplitStrategy.adaptive_peak_disease_timepoints,
)

# Get V gene usage proportions per specimen
v_gene_usage_proportions_by_specimen_adaptive = pd.concat(
    [
        genetools.stats.normalize_rows(
            specimen_v_gene_counts_df_test_only_adaptive[v_gene_cols_adaptive]
        ),
        specimen_v_gene_counts_df_test_only_adaptive["disease"],
    ],
    axis=1,
)

v_gene_usage_proportions_by_specimen_annot_adaptive = genetools.helpers.merge_into_left(
    v_gene_usage_proportions_by_specimen_adaptive[v_gene_cols_adaptive],
    extra_specimen_metadata.set_index("specimen_label"),
)


# drop specimens with group N/A
v_gene_usage_proportions_by_specimen_annot_adaptive = (
    v_gene_usage_proportions_by_specimen_annot_adaptive.dropna(subset=["study_name"])
)

v_gene_usage_proportions_by_specimen_annot_melt_adaptive = pd.melt(
    v_gene_usage_proportions_by_specimen_annot_adaptive,
    id_vars=["disease", "study_name"],
    value_vars=v_gene_cols_adaptive,
    var_name="V gene",
    value_name="Proportion",
)
v_gene_usage_proportions_by_specimen_annot_melt_adaptive

# %%

# %%

# %%
v_gene_usage_proportions_by_specimen_annot_melt_combined = pd.concat(
    [
        v_gene_usage_proportions_by_specimen_annot_melt,
        v_gene_usage_proportions_by_specimen_annot_melt_adaptive,
    ],
    axis=0,
)
v_gene_usage_proportions_by_specimen_annot_melt_combined

# %%

# %%
group_color_palette = {
    group: color
    for group, color in zip(
        v_gene_usage_proportions_by_specimen_annot_melt_combined["study_name"].unique(),
        sc.plotting.palettes.default_102,
    )
}


def plot_per_disease(
    v_gene_usage_proportions_by_specimen_annot_melt_combined: pd.DataFrame,
    disease: str,
    filtered=False,
):
    height = 7 if filtered else 13
    selected_v_gene_order = pd.Series(v_gene_cols_filtered if filtered else v_gene_cols)

    selected_v_gene_order = pd.Series(
        sorted(
            v_gene_usage_proportions_by_specimen_annot_melt_combined["V gene"].unique()
        )
    )

    v_gene_usage_proportions_by_specimen_annot_melt_this_disease = (
        v_gene_usage_proportions_by_specimen_annot_melt_combined[
            v_gene_usage_proportions_by_specimen_annot_melt_combined["disease"]
            == disease
        ]
    )
    # Sort for consistent group order
    groups_this_disease = sorted(
        v_gene_usage_proportions_by_specimen_annot_melt_this_disease[
            "study_name"
        ].unique()
    )
    if len(groups_this_disease) == 0:
        print(f"No study_name group info for {disease} - skipping")
        return None

    # Divide the disease plot into subplots by group
    fig, axarr = plt.subplots(
        nrows=1,
        ncols=len(groups_this_disease),
        figsize=(3 * len(groups_this_disease), height),
        sharex=True,  # Make xlims consistent for better readability
        sharey=False,  # Repeat the V gene in each axis for better readability
    )
    for (group, ax) in zip(groups_this_disease, axarr):
        data = v_gene_usage_proportions_by_specimen_annot_melt_this_disease[
            v_gene_usage_proportions_by_specimen_annot_melt_this_disease["study_name"]
            == group
        ]
        sns.barplot(
            # Switch to friendly V gene names
            data=data.assign(
                **{"V gene": data["V gene"].replace(helpers.v_gene_friendly_names)}
            ),
            x="Proportion",
            y="V gene",
            # reference V gene order, possibly filtered down
            order=sorted(selected_v_gene_order.replace(helpers.v_gene_friendly_names)),
            ax=ax,
            color=group_color_palette[group],
            # Compute 95% confidence intervals around a sample mean by bootstrapping:
            # sampling distribution of mean generated by repeated sampling and recording mean each time.
            # the standard error is basically the standard deviation of many sample means
            # we plot mean +/- 1.96*standard error. gives you average value +/- X at the 95% confidence level.
            errorbar=("ci", 95),
            # errorbar="sd", # instead draw the standard deviation of the observations, instead of bootstrapping to get 95% confidence intervals
            # capsize=.025
        )

        ax.set_title(f"{group}", fontweight="bold")
        ax.set_xlabel("Proportion of specimen\n(mean +/- 95% confidence interval)")
        ax.set_ylabel(None)
        sns.despine(ax=ax)

    axarr[0].set_ylabel("V gene")
    fig.suptitle(disease, fontsize="x-large", fontweight="bold")
    plt.tight_layout()
    return fig


# %%
# Plot for all V genes
fig = plot_per_disease(
    v_gene_usage_proportions_by_specimen_annot_melt_combined=v_gene_usage_proportions_by_specimen_annot_melt_combined,
    disease="Covid19",
    filtered=False,
)

# %%
# Plot for all V genes
fig = plot_per_disease(
    v_gene_usage_proportions_by_specimen_annot_melt_combined=v_gene_usage_proportions_by_specimen_annot_melt_combined,
    disease="Healthy/Background",
    filtered=False,
)

# %%

# %%

# %% [markdown]
# # Embed on UMAP and see where it lands?

# %%
ec_data_c = pd.concat(
    [
        v_gene_usage_proportions_by_specimen_annot_adaptive[v_gene_cols_adaptive],
        v_gene_usage_proportions_by_specimen_annot[v_gene_cols],
    ],
    axis=0,
)
ec_data_c

# %%
ec_obs_c = pd.concat(
    [
        v_gene_usage_proportions_by_specimen_annot_adaptive[
            ["study_name", "disease", "data_source"]
        ],
        v_gene_usage_proportions_by_specimen_annot[
            ["study_name", "disease", "data_source"]
        ],
    ],
    axis=0,
)
ec_obs_c

# %%
ec_data_c.isna().any()

# %%
ec_data_c = ec_data_c.dropna(axis=1)

# %%

# %%
meaningful_v_genes = list(v_gene_cols_filtered)
meaningful_v_genes

# %%
adata_subset = anndata.AnnData(
    ec_data_c[[v for v in meaningful_v_genes if v in ec_data_c.columns]], obs=ec_obs_c
)
adata_subset

# %%
adata_subset.shape

# %%
adata_subset.obs["label"] = (
    adata_subset.obs["disease"] + " - " + adata_subset.obs["study_name"]
)

# %%
# make a new label like Covid-19 (Mal-ID), Covid-19 (Adaptive), etc.
adata_subset.obs.drop_duplicates()

# %%
adata_subset.obs["data_source"] = adata_subset.obs["data_source"].map(
    {DataSource.adaptive: "Adaptive", DataSource.in_house: "Mal-ID"}
)
assert not adata_subset.obs["data_source"].isna().any()

# %%
adata_subset.obs["label2"] = (
    adata_subset.obs["disease"].astype(str) + " - " + adata_subset.obs["data_source"]
)
adata_subset.obs["label2"].value_counts()

# %%
adata_subset.obs.drop_duplicates()

# %%
sc.pp.scale(adata_subset)

# %%
sc.tl.pca(adata_subset)

# %%
sc.pp.neighbors(adata_subset)
sc.tl.umap(adata_subset)

# %%
sc.pl.pca(adata_subset, color="label", alpha=0.5)

# %%
sc.pl.umap(adata_subset, color="label", alpha=0.5)

# %%

# %%
sc.pl.pca(adata_subset, color="data_source", alpha=0.5)

# %%
sc.pl.umap(adata_subset, color="data_source", alpha=0.5)

# %%

# %%
sc.pl.pca(adata_subset, color="label2", alpha=0.5)

# %%
sc.pl.umap(adata_subset, color="label2", alpha=0.5)

# %%
fig_umap = sc.pl.umap(
    adata_subset,
    color=["data_source", "label2"],
    alpha=0.5,
    return_fig=True,
    title=["TRBV gene proportions UMAP", ""],
    ncols=1,
)
sns.despine(fig=fig_umap)
genetools.plots.savefig(
    fig_umap,
    config.paths.model_interpretations_for_selected_cross_validation_strategy_output_dir
    / GeneLocus.TCR.name
    / f"v_gene_proportions_by_specimen.adaptive_vs_malid.png",
    dpi=300,
)

# %%

# %%

# %%

# %%
# TODO: add the other external cohorts that are more similar to ours

# %%

# %%
