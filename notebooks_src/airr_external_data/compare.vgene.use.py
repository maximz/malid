# %%

# %% [markdown]
# # Compare V gene and J gene use in our cohorts versus some example Emerson/Immunecode repertoires
#
# Based on `vgene_usage_stats.ipynb`

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

from malid import config, io, helpers
from malid.datamodels import healthy_label, GeneLocus, TargetObsColumnEnum
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
def get_dirs(gene_locus: GeneLocus):
    output_dir = config.paths.model_interpretations_output_dir / gene_locus.name
    highres_output_dir = (
        config.paths.high_res_outputs_dir / "model_interpretations" / gene_locus.name
    )

    # Create directories - though these directories should already have been created by sequence model interpretations notebooks
    output_dir.mkdir(parents=True, exist_ok=True)
    highres_output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, highres_output_dir


# %%
# reimport (can resume here)
def import_v_gene_counts(gene_locus: GeneLocus):
    output_dir, highres_output_dir = get_dirs(gene_locus)

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
extra_specimen_metadata = extra_specimen_metadata[
    extra_specimen_metadata["in_training_set"]
].copy()
extra_specimen_metadata

# %%
(
    specimen_v_gene_counts_df_test_only,
    v_gene_cols,
    v_gene_cols_filtered,
) = import_v_gene_counts(gene_locus=GeneLocus.TCR)

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
def make_counts(fname, disease, study_name):
    external_df = pd.read_parquet(fname)
    counts = (
        external_df["v_gene"]
        .cat.remove_unused_categories()
        .value_counts(normalize=True)
    )
    return (
        counts.rename_axis(index="V gene")
        .reset_index(name="Proportion")
        .assign(disease=disease, study_name=study_name)
    )


# %%

# %%
# TODO: switch this to exactly the same ones used in external cohort evaluation
emerson_fnames = list(config.paths.external_processed_data.glob("P00*.parquet")) + list(
    config.paths.external_processed_data.glob("Keck*.parquet")
)
np.random.shuffle(emerson_fnames)
immunecode_fnames = list(
    config.paths.external_processed_data.glob("ImmuneCode*.parquet")
)
len(emerson_fnames), len(immunecode_fnames)

# %%

# %%
external_counts = []

for fname in immunecode_fnames[: len(immunecode_fnames)]:
    print(fname)
    external_counts.append(
        make_counts(fname=fname, disease="Covid19", study_name="Immunecode")
    )

for fname in emerson_fnames[: len(immunecode_fnames)]:
    print(fname)
    external_counts.append(
        make_counts(fname=fname, disease="Healthy/Background", study_name="Emerson")
    )

# %%

# %%

# %%

# %%
v_gene_usage_proportions_by_specimen_annot_melt_combined = pd.concat(
    [v_gene_usage_proportions_by_specimen_annot_melt] + external_counts,
    axis=0,
)
v_gene_usage_proportions_by_specimen_annot_melt_combined

# %%
group_color_palette = {
    group: color
    for group, color in zip(
        v_gene_usage_proportions_by_specimen_annot_melt_combined["study_name"].unique(),
        sc.plotting.palettes.default_20,
    )
}


def plot_per_disease(
    v_gene_usage_proportions_by_specimen_annot_melt_combined: pd.DataFrame,
    disease: str,
    filtered=False,
):
    height = 6 if filtered else 12
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
            ci=95,
            # ci="sd", # instead draw the standard deviation of the observations, instead of bootstrapping to get 95% confidence intervals
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
ec_combined = pd.concat(
    [
        ec.pivot(index=["study_name", "disease"], columns="V gene", values="Proportion")
        for ec in external_counts
    ],
    axis=0,
)
ec_combined

# %%
ec_data = ec_combined.reset_index(drop=True)
ec_obs = ec_combined.index.to_frame().reset_index(drop=True)

# %%
ec_data_c = pd.concat(
    [ec_data, v_gene_usage_proportions_by_specimen_annot[v_gene_cols]], axis=0
)
ec_obs_c = pd.concat(
    [ec_obs, v_gene_usage_proportions_by_specimen_annot[["study_name", "disease"]]],
    axis=0,
)

# %%
ec_data_c.isna().any()

# %%
ec_data_c = ec_data_c.dropna(axis=1)

# %%

# %%
# TODO: switch to V genes from model1's choices?
meaningful_v_genes = pd.read_csv(
    config.paths.model_interpretations_output_dir
    / GeneLocus.TCR.name
    / "meaningful_v_genes.txt",
    sep="\t",
)["v_gene"].tolist()
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
adata_subset.obs["label2"] = (
    adata_subset.obs["disease"].astype(str)
    + " - "
    + adata_subset.obs["study_name"]
    .map({"Immunecode": "Adaptive", "Emerson": "Adaptive"})
    .fillna("Mal-ID")
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
sc.pl.pca(adata_subset, color="label2", alpha=0.5)

# %%
fig_umap = sc.pl.umap(
    adata_subset,
    color="label2",
    alpha=0.5,
    return_fig=True,
    title="TRBV gene proportions UMAP",
)
genetools.plots.savefig(
    fig_umap,
    config.paths.model_interpretations_output_dir
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
