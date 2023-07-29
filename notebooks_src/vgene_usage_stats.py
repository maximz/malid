# %%
import numpy as np
import pandas as pd

from malid import config, io, helpers
from malid.datamodels import healthy_label, GeneLocus, TargetObsColumnEnum
import gc
import joblib
from kdict import kdict
import itertools
import genetools
from pathlib import Path
from slugify import slugify
from typing import Tuple
import scanpy as sc

# %%
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns


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

# %% [markdown]
# # Compute V gene counts overall by disease cohort
#
# Run once. Saved to disk and reloaded by later functions.

# %%

# %%
def get_vgene_and_cdr3length_counts(
    gene_locus: GeneLocus,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_labels = ["train_smaller", "validation", "test"]

    specimen_v_gene_counts = []
    specimen_cdr3_length_counts = []
    for fold_id in config.all_fold_ids:
        for fold_label in fold_labels:
            if fold_id == -1 and fold_label == "test":
                # skip: global fold does not have a test set
                continue
            adata = io.load_fold_embeddings(
                fold_id=fold_id,
                fold_label=fold_label,
                gene_locus=gene_locus,
                target_obs_column=TargetObsColumnEnum.disease,
            )
            df = adata.obs
            for specimen_label, subset_obs in adata.obs.groupby(
                "specimen_label", observed=True
            ):
                v_gene_counts = (
                    subset_obs["v_gene"].cat.remove_unused_categories().value_counts()
                )
                cdr3_length_counts = subset_obs[
                    "cdr3_aa_sequence_trim_len"
                ].value_counts()
                specimen_description = subset_obs[["specimen_label", "disease"]].iloc[0]
                specimen_v_gene_counts.append(
                    {
                        "fold_id": fold_id,
                        "fold_label": fold_label,
                        **v_gene_counts.to_dict(),
                        **specimen_description.to_dict(),
                    }
                )
                specimen_cdr3_length_counts.append(
                    {
                        "fold_id": fold_id,
                        "fold_label": fold_label,
                        **cdr3_length_counts.to_dict(),
                        **specimen_description.to_dict(),
                    }
                )

            del df, adata
            io.clear_cached_fold_embeddings()
            gc.collect()

    specimen_v_gene_counts_df = pd.DataFrame(specimen_v_gene_counts)
    specimen_cdr3_length_counts_df = pd.DataFrame(specimen_cdr3_length_counts)
    return specimen_v_gene_counts_df, specimen_cdr3_length_counts_df


# %%

# %%
# export
for gene_locus in config.gene_loci_used:
    # Compute (slow)
    (
        specimen_v_gene_counts_df,
        specimen_cdr3_length_counts_df,
    ) = get_vgene_and_cdr3length_counts(gene_locus=gene_locus)

    # Export
    output_dir, highres_output_dir = get_dirs(gene_locus)
    specimen_v_gene_counts_df.to_csv(
        highres_output_dir / "v_gene_counts_by_specimen.tsv.gz", sep="\t", index=None
    )
    specimen_cdr3_length_counts_df.to_csv(
        highres_output_dir / "cdr3_length_counts_by_specimen.tsv.gz",
        sep="\t",
        index=None,
    )


# %%

# %%

# %% [markdown]
# # Analyze V gene counts and CDR3 length distribution

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
    # TODO: switch to V genes from model1's choices?
    v_gene_cols_filtered = pd.read_csv(output_dir / "meaningful_v_genes.txt")[
        "v_gene"
    ].values
    assert all(vgene in v_gene_cols for vgene in v_gene_cols_filtered)  # sanity check

    return specimen_v_gene_counts_df_test_only, v_gene_cols, v_gene_cols_filtered


# %%
def import_cdr3_length_counts(gene_locus: GeneLocus):
    output_dir, highres_output_dir = get_dirs(gene_locus)

    specimen_cdr3_length_counts_df = pd.read_csv(
        highres_output_dir / "cdr3_length_counts_by_specimen.tsv.gz", sep="\t"
    )

    # subselect to test folds only (which also excludes global fold -1), and set index
    specimen_cdr3_length_counts_df_test_only = specimen_cdr3_length_counts_df[
        specimen_cdr3_length_counts_df["fold_label"] == "test"
    ]

    # confirm only one entry per specimen now
    assert (
        not specimen_cdr3_length_counts_df_test_only["specimen_label"]
        .duplicated()
        .any()
    )
    specimen_cdr3_length_counts_df_test_only = (
        specimen_cdr3_length_counts_df_test_only.set_index("specimen_label").drop(
            ["fold_id", "fold_label"], axis=1
        )
    )

    # drop any columns that are all N/A
    specimen_cdr3_length_counts_df_test_only = (
        specimen_cdr3_length_counts_df_test_only.dropna(axis=1, how="all")
    )

    # fill remaining N/As with 0
    specimen_cdr3_length_counts_df_test_only = (
        specimen_cdr3_length_counts_df_test_only.fillna(0)
    )

    cdr3_length_cols = specimen_cdr3_length_counts_df_test_only.columns
    cdr3_length_cols = cdr3_length_cols[~cdr3_length_cols.isin(["disease"])]

    # Convert cols to ints
    specimen_cdr3_length_counts_df_test_only.rename(
        columns={i: int(i) for i in cdr3_length_cols}, inplace=True
    )
    # Get latest column list
    cdr3_length_cols = specimen_cdr3_length_counts_df_test_only.columns
    cdr3_length_cols = cdr3_length_cols[~cdr3_length_cols.isin(["disease"])]

    # Fill in skips as all 0s
    for cdr3_len in np.arange(min(cdr3_length_cols), max(cdr3_length_cols)):
        if cdr3_len not in cdr3_length_cols:
            specimen_cdr3_length_counts_df_test_only[cdr3_len] = 0.0

    # Get latest column list
    cdr3_length_cols = specimen_cdr3_length_counts_df_test_only.columns
    cdr3_length_cols = cdr3_length_cols[~cdr3_length_cols.isin(["disease"])]

    return specimen_cdr3_length_counts_df_test_only, cdr3_length_cols


# %%

# %%

# %%

# %% [markdown]
# # For specimens of each disease type, plot average (+/- std) of V gene proportions and CDR3 length distributions

# %%

# %%
def analyze_v_gene_proportions(gene_locus: GeneLocus):
    output_dir, highres_output_dir = get_dirs(gene_locus)

    (
        specimen_v_gene_counts_df_test_only,
        v_gene_cols,
        v_gene_cols_filtered,
    ) = import_v_gene_counts(gene_locus=gene_locus)

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

    v_gene_usage_proportions_by_specimen_melt = pd.melt(
        v_gene_usage_proportions_by_specimen,
        id_vars=["disease"],
        value_vars=v_gene_cols,
        var_name="V gene",
        value_name="Proportion",
    )

    def plot(v_gene_usage_proportions_by_specimen_melt, filtered=False):
        height = 6 if filtered else 12
        selected_v_gene_order = pd.Series(
            v_gene_cols_filtered if filtered else v_gene_cols
        )

        diseases = sorted(v_gene_usage_proportions_by_specimen_melt["disease"].unique())

        fig, axarr = plt.subplots(
            nrows=1,
            ncols=len(diseases),
            figsize=(3 * len(diseases), height),
            sharex=True,  # Make xlims consistent for better readability
            sharey=False,  # Repeat the V gene in each axis for better readability
        )
        for (disease, ax) in zip(diseases, axarr):
            data = v_gene_usage_proportions_by_specimen_melt[
                v_gene_usage_proportions_by_specimen_melt["disease"] == disease
            ]
            sns.barplot(
                # Switch to friendly V gene names
                data=data.assign(
                    **{"V gene": data["V gene"].replace(helpers.v_gene_friendly_names)}
                ),
                x="Proportion",
                y="V gene",
                # reference V gene order, possibly filtered down
                order=sorted(
                    selected_v_gene_order.replace(helpers.v_gene_friendly_names)
                ),
                ax=ax,
                color=helpers.disease_color_palette[disease],
                # Compute 95% confidence intervals around a sample mean by bootstrapping:
                # sampling distribution of mean generated by repeated sampling and recording mean each time.
                # the standard error is basically the standard deviation of many sample means
                # we plot mean +/- 1.96*standard error. gives you average value +/- X at the 95% confidence level.
                ci=95,
                # ci="sd", # instead draw the standard deviation of the observations, instead of bootstrapping to get 95% confidence intervals
                # capsize=.025
            )

            ax.set_title(disease, fontweight="bold")
            ax.set_xlabel("Proportion of specimen\n(mean +/- 95% confidence interval)")
            ax.set_ylabel(None)
            sns.despine(ax=ax)

        axarr[0].set_ylabel("V gene")
        plt.tight_layout()
        return fig

    ## Plot for all V genes
    fig = plot(
        v_gene_usage_proportions_by_specimen_melt=v_gene_usage_proportions_by_specimen_melt,
        filtered=False,
    )
    genetools.plots.savefig(
        fig,
        highres_output_dir / "v_gene_proportions_by_specimen.by_disease.png",
        dpi=300,
    )

    ## Repeat, for subset of V genes
    fig = plot(
        v_gene_usage_proportions_by_specimen_melt=v_gene_usage_proportions_by_specimen_melt,
        filtered=True,
    )
    genetools.plots.savefig(
        fig,
        highres_output_dir
        / "v_gene_proportions_by_specimen.filtered_v_genes.by_disease.png",
        dpi=300,
    )


# %%
for gene_locus in config.gene_loci_used:
    analyze_v_gene_proportions(gene_locus)


# %%

# %%

# %%
def analyze_cdr3_length_distribution(gene_locus: GeneLocus):
    output_dir, highres_output_dir = get_dirs(gene_locus)

    (
        specimen_cdr3_length_counts_df_test_only,
        cdr3_length_cols,
    ) = import_cdr3_length_counts(gene_locus=gene_locus)

    # Get CDR3 length usage distribution per specimen
    cdr3_length_distribution_by_specimen = pd.concat(
        [
            genetools.stats.normalize_rows(
                specimen_cdr3_length_counts_df_test_only[cdr3_length_cols]
            ),
            specimen_cdr3_length_counts_df_test_only["disease"],
        ],
        axis=1,
    )

    cdr3_length_distribution_by_specimen_melt = pd.melt(
        cdr3_length_distribution_by_specimen,
        id_vars=["disease"],
        value_vars=cdr3_length_cols,
        var_name="CDR3 length",
        value_name="Proportion",
    )

    def plot(cdr3_length_distribution_by_specimen_melt, filtered=False):
        height = 6 if filtered else 12
        diseases = sorted(cdr3_length_distribution_by_specimen_melt["disease"].unique())

        fig, axarr = plt.subplots(
            nrows=1,
            ncols=len(diseases),
            figsize=(3 * len(diseases), height),
            sharex=True,  # Make xlims consistent for better readability
            sharey=False,  # Repeat the V gene in each axis for better readability
        )
        for (disease, ax) in zip(diseases, axarr):
            sns.barplot(
                data=cdr3_length_distribution_by_specimen_melt[
                    cdr3_length_distribution_by_specimen_melt["disease"] == disease
                ],
                x="Proportion",
                y="CDR3 length",
                orient="h",
                order=reversed(sorted(cdr3_length_cols)),
                ax=ax,
                color=helpers.disease_color_palette[disease],
                # Compute 95% confidence intervals around a sample mean by bootstrapping:
                # sampling distribution of mean generated by repeated sampling and recording mean each time.
                # the standard error is basically the standard deviation of many sample means
                # we plot mean +/- 1.96*standard error. gives you average value +/- X at the 95% confidence level.
                ci=95,
                # ci="sd", # instead draw the standard deviation of the observations, instead of bootstrapping to get 95% confidence intervals
                # capsize=.025
            )

            ax.set_title(disease, fontweight="bold")
            ax.set_xlabel("Proportion of specimen\n(mean +/- 95% confidence interval)")
            ax.set_ylabel(None)
            sns.despine(ax=ax)

        axarr[0].set_ylabel(f"CDR3 length ({gene_locus})")
        plt.tight_layout()
        return fig

    ## Plot
    fig = plot(
        cdr3_length_distribution_by_specimen_melt=cdr3_length_distribution_by_specimen_melt,
        filtered=False,
    )


#     genetools.plots.savefig(
#         fig,
#         highres_output_dir / "v_gene_proportions_by_specimen.by_disease.png",
#         dpi=300,
#     )

# %%
for gene_locus in config.gene_loci_used:
    analyze_cdr3_length_distribution(gene_locus)

# %%

# %%

# %% [markdown]
# ## Same but now versus ancestry
#
# Covid19 and Healthy cohorts have many ancestry groups represented

# %%
extra_specimen_metadata = helpers.get_all_specimen_info()
extra_specimen_metadata = extra_specimen_metadata[
    extra_specimen_metadata["in_training_set"]
].copy()

extra_specimen_metadata

# %%
extra_specimen_metadata["disease_subtype"].value_counts()

# %%
extra_specimen_metadata["study_name"].value_counts()

# %%
extra_specimen_metadata["disease_severity"].value_counts()

# %%
extra_specimen_metadata.groupby(["study_name", "disease_severity"]).size()


# %%

# %%

# %%

# %%
def analyze_v_gene_proportions_by_subgroup(gene_locus: GeneLocus, group_key: str):
    output_dir, highres_output_dir = get_dirs(gene_locus)

    (
        specimen_v_gene_counts_df_test_only,
        v_gene_cols,
        v_gene_cols_filtered,
    ) = import_v_gene_counts(gene_locus=gene_locus)

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
    assert (
        v_gene_usage_proportions_by_specimen_annot.shape[0]
        == v_gene_usage_proportions_by_specimen.shape[0]
    )
    assert not v_gene_usage_proportions_by_specimen_annot["disease"].isna().any()

    # drop specimens with group N/A
    v_gene_usage_proportions_by_specimen_annot = (
        v_gene_usage_proportions_by_specimen_annot.dropna(subset=[group_key])
    )

    v_gene_usage_proportions_by_specimen_annot_melt = pd.melt(
        v_gene_usage_proportions_by_specimen_annot,
        id_vars=["disease", group_key],
        value_vars=v_gene_cols,
        var_name="V gene",
        value_name="Proportion",
    )

    group_color_palette = {
        group: color
        for group, color in zip(
            sorted(v_gene_usage_proportions_by_specimen_annot_melt[group_key].unique()),
            sc.plotting.palettes.default_20,
        )
    }

    def plot_per_disease(
        v_gene_usage_proportions_by_specimen_annot_melt: pd.DataFrame,
        disease: str,
        filtered=False,
    ):
        height = 6 if filtered else 12
        selected_v_gene_order = pd.Series(
            v_gene_cols_filtered if filtered else v_gene_cols
        )

        v_gene_usage_proportions_by_specimen_annot_melt_this_disease = (
            v_gene_usage_proportions_by_specimen_annot_melt[
                v_gene_usage_proportions_by_specimen_annot_melt["disease"] == disease
            ]
        )
        # Sort for consistent group order
        groups_this_disease = sorted(
            v_gene_usage_proportions_by_specimen_annot_melt_this_disease[
                group_key
            ].unique()
        )
        if len(groups_this_disease) == 0:
            print(f"No {group_key} group info for {disease} - skipping")
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
                v_gene_usage_proportions_by_specimen_annot_melt_this_disease[group_key]
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
                order=sorted(
                    selected_v_gene_order.replace(helpers.v_gene_friendly_names)
                ),
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

            # compute sample size per disease and group, using dataframe from before melting but after N/As were dropped
            sample_size = v_gene_usage_proportions_by_specimen_annot[
                (v_gene_usage_proportions_by_specimen_annot["disease"] == disease)
                & (v_gene_usage_proportions_by_specimen_annot[group_key] == group)
            ].shape[0]

            ax.set_title(f"{group} $(n={sample_size})$", fontweight="bold")
            ax.set_xlabel("Proportion of specimen\n(mean +/- 95% confidence interval)")
            ax.set_ylabel(None)
            sns.despine(ax=ax)

        axarr[0].set_ylabel("V gene")
        fig.suptitle(f"{disease} ({gene_locus})", fontsize="x-large", fontweight="bold")
        plt.tight_layout()
        return fig

    # Make a plot for each disease
    for disease in ["Covid19", "Lupus", healthy_label]:
        # Plot for all V genes
        fig = plot_per_disease(
            v_gene_usage_proportions_by_specimen_annot_melt=v_gene_usage_proportions_by_specimen_annot_melt,
            disease=disease,
            filtered=False,
        )
        if fig is not None:
            genetools.plots.savefig(
                fig,
                highres_output_dir
                / f"v_gene_proportions_by_specimen.disease.{slugify(disease)}.by_{group_key}.png",
                dpi=300,
            )

        # Repeat, for subset of V genes
        fig = plot_per_disease(
            v_gene_usage_proportions_by_specimen_annot_melt=v_gene_usage_proportions_by_specimen_annot_melt,
            disease=disease,
            filtered=True,
        )
        if fig is not None:
            genetools.plots.savefig(
                fig,
                output_dir
                / f"v_gene_proportions_by_specimen.filtered_v_genes.disease.{slugify(disease)}.by_{group_key}.png",
                dpi=300,
            )

    return v_gene_usage_proportions_by_specimen_annot_melt


# %%
def analyze_cdr3_length_distribution_by_subgroup(gene_locus: GeneLocus, group_key: str):
    output_dir, highres_output_dir = get_dirs(gene_locus)

    (
        specimen_cdr3_length_counts_df_test_only,
        cdr3_length_cols,
    ) = import_cdr3_length_counts(gene_locus=gene_locus)

    # Get CDR3 length usage distribution per specimen
    cdr3_length_distribution_by_specimen = pd.concat(
        [
            genetools.stats.normalize_rows(
                specimen_cdr3_length_counts_df_test_only[cdr3_length_cols]
            ),
            specimen_cdr3_length_counts_df_test_only["disease"],
        ],
        axis=1,
    )

    cdr3_length_distribution_by_specimen_annot = genetools.helpers.merge_into_left(
        cdr3_length_distribution_by_specimen[cdr3_length_cols],
        extra_specimen_metadata.set_index("specimen_label"),
    )
    assert (
        cdr3_length_distribution_by_specimen_annot.shape[0]
        == cdr3_length_distribution_by_specimen.shape[0]
    )
    assert not cdr3_length_distribution_by_specimen_annot["disease"].isna().any()

    # drop specimens with group N/A
    cdr3_length_distribution_by_specimen_annot = (
        cdr3_length_distribution_by_specimen_annot.dropna(subset=[group_key])
    )

    cdr3_length_distribution_by_specimen_annot_melt = pd.melt(
        cdr3_length_distribution_by_specimen_annot,
        id_vars=["disease", group_key],
        value_vars=cdr3_length_cols,
        var_name="CDR3 length",
        value_name="Proportion",
    )

    group_color_palette = {
        group: color
        for group, color in zip(
            sorted(cdr3_length_distribution_by_specimen_annot_melt[group_key].unique()),
            sc.plotting.palettes.default_20,
        )
    }

    def plot_per_disease(
        cdr3_length_distribution_by_specimen_annot_melt: pd.DataFrame,
        disease: str,
        filtered=False,
    ):
        height = 6 if filtered else 12
        cdr3_length_distribution_by_specimen_annot_melt_this_disease = (
            cdr3_length_distribution_by_specimen_annot_melt[
                cdr3_length_distribution_by_specimen_annot_melt["disease"] == disease
            ]
        )
        # Sort for consistent group order
        groups_this_disease = sorted(
            cdr3_length_distribution_by_specimen_annot_melt_this_disease[
                group_key
            ].unique()
        )
        if len(groups_this_disease) == 0:
            print(f"No {group_key} group info for {disease} - skipping")
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
            sns.barplot(
                data=cdr3_length_distribution_by_specimen_annot_melt_this_disease[
                    cdr3_length_distribution_by_specimen_annot_melt_this_disease[
                        group_key
                    ]
                    == group
                ],
                x="Proportion",
                y="CDR3 length",
                orient="h",
                order=reversed(sorted(cdr3_length_cols)),
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

            # compute sample size per disease and group, using dataframe from before melting but after N/As were dropped
            sample_size = cdr3_length_distribution_by_specimen_annot[
                (cdr3_length_distribution_by_specimen_annot["disease"] == disease)
                & (cdr3_length_distribution_by_specimen_annot[group_key] == group)
            ].shape[0]

            ax.set_title(f"{group} $(n={sample_size})$", fontweight="bold")
            ax.set_xlabel("Proportion of specimen\n(mean +/- 95% confidence interval)")
            ax.set_ylabel(None)
            sns.despine(ax=ax)

        axarr[0].set_ylabel("CDR3 length")
        fig.suptitle(f"{disease} ({gene_locus})", fontsize="x-large", fontweight="bold")
        plt.tight_layout()
        return fig

    # Make a plot for each disease
    for disease in ["Covid19", "Lupus", healthy_label]:
        # Plot for all V genes
        fig = plot_per_disease(
            cdr3_length_distribution_by_specimen_annot_melt=cdr3_length_distribution_by_specimen_annot_melt,
            disease=disease,
            filtered=False,
        )
    #         if fig is not None:
    #             genetools.plots.savefig(
    #                 fig,
    #                 highres_output_dir
    #                 / f"v_gene_proportions_by_specimen.disease.{slugify(disease)}.by_{group_key}.png",
    #                 dpi=300,
    #             )

    return cdr3_length_distribution_by_specimen_annot_melt


# %%

# %%
v_gene_use_proportions_by_specimen_and_ethnicity = {}
for gene_locus in config.gene_loci_used:
    v_gene_use_proportions_by_specimen_and_ethnicity[
        gene_locus
    ] = analyze_v_gene_proportions_by_subgroup(
        gene_locus, group_key="ethnicity_condensed"
    )

# %%
for gene_locus in config.gene_loci_used:
    analyze_cdr3_length_distribution_by_subgroup(
        gene_locus, group_key="ethnicity_condensed"
    )

# %%

# %%

# %% [markdown]
# ## Investigate means directly for two V genes of interest

# %%
v_gene_usage_proportions_by_specimen_annot_melt = (
    v_gene_use_proportions_by_specimen_and_ethnicity[GeneLocus.BCR]
)
v_gene_usage_proportions_by_specimen_annot_melt

# %%
v_gene_usage_proportions_by_specimen_annot_melt[
    (
        v_gene_usage_proportions_by_specimen_annot_melt["V gene"].isin(
            ["IGHV5-a", "IGHV4-b"]
        )
    )
    & (
        v_gene_usage_proportions_by_specimen_annot_melt["disease"].isin(
            [healthy_label, "Covid19"]
        )
    )
].groupby(["disease", "V gene", "ethnicity_condensed"])["Proportion"].mean().apply(
    # print as percentage
    lambda mean: f"{mean:0.2%}"
)

# %%

# %%
# repeat, without disease filter
v_gene_usage_proportions_by_specimen_annot_melt[
    v_gene_usage_proportions_by_specimen_annot_melt["V gene"].isin(
        ["IGHV5-a", "IGHV4-b"]
    )
].groupby(["disease", "V gene", "ethnicity_condensed"])["Proportion"].mean().apply(
    # print as percentage
    lambda mean: f"{mean:0.2%}"
)

# %%
# repeat, without disease filter
v_gene_usage_proportions_by_specimen_annot_melt[
    v_gene_usage_proportions_by_specimen_annot_melt["V gene"].isin(
        ["IGHV5-a", "IGHV4-b"]
    )
].groupby(["V gene", "ethnicity_condensed"])["Proportion"].mean().apply(
    # print as percentage
    lambda mean: f"{mean:0.2%}"
)

# %%

# %% [markdown]
# # Look at V gene usage and CDR3 length distribution by `study_name`

# %%
v_gene_use_proportions_by_specimen_and_study_name = {}
for gene_locus in config.gene_loci_used:
    v_gene_use_proportions_by_specimen_and_study_name[
        gene_locus
    ] = analyze_v_gene_proportions_by_subgroup(gene_locus, group_key="study_name")

# %%
for gene_locus in config.gene_loci_used:
    analyze_cdr3_length_distribution_by_subgroup(gene_locus, group_key="study_name")

# %%

# %%
# Some TCR V genes are present in some study_names (batches) but not others
means = (
    v_gene_use_proportions_by_specimen_and_study_name[GeneLocus.TCR]
    .groupby(["disease", "study_name", "V gene"])["Proportion"]
    .mean()
)
means[means == 0.0]

# %%
# Some BCR V genes are present in some study_names (batches) but not others
means = (
    v_gene_use_proportions_by_specimen_and_study_name[GeneLocus.BCR]
    .groupby(["disease", "study_name", "V gene"])["Proportion"]
    .mean()
)
means[means == 0.0]

# %%

# %%

# %%
all_expected_v_genes = helpers.all_observed_v_genes()
[(gl, lst.shape) for gl, lst in all_expected_v_genes.items()]

# %%
v_gene_use_proportions_by_specimen_and_study_name[GeneLocus.BCR][
    "V gene"
].unique().shape

# %%
v_gene_use_proportions_by_specimen_and_study_name[GeneLocus.TCR][
    "V gene"
].unique().shape

# %%
# Any BCR V genes in our entire parquet dataset after ETL step, but not in the plots above (i.e. not in peak timepoint test set?):
set.symmetric_difference(
    set(v_gene_use_proportions_by_specimen_and_study_name[GeneLocus.BCR]["V gene"]),
    all_expected_v_genes[GeneLocus.BCR],
)

# %%
# Any TCR V genes in our entire parquet dataset after ETL step, but not in the plots above (i.e. not in peak timepoint test set?):
set.symmetric_difference(
    set(v_gene_use_proportions_by_specimen_and_study_name[GeneLocus.TCR]["V gene"]),
    all_expected_v_genes[GeneLocus.TCR],
)

# %%

# %%

# %%

# %% [markdown]
# # Look at V gene usage and CDR3 lengths by Covid19 severity

# %%
for gene_locus in config.gene_loci_used:
    analyze_v_gene_proportions_by_subgroup(gene_locus, group_key="disease_severity")

# %%
for gene_locus in config.gene_loci_used:
    analyze_cdr3_length_distribution_by_subgroup(
        gene_locus, group_key="disease_severity"
    )

# %%

# %%

# %% [markdown]
# # Plot V gene usage of each specimen on PCA/UMAP, and compare intra-disease batch distances vs inter-disease distances
#
# Using cosine distance between V gene use proportion vectors per specimen (or mean across specimens in a disease-batch)

# %%
import anndata
import scanpy as sc
from scipy.spatial.distance import pdist, squareform


# %%
def plot_triangular_heatmap(
    df,
    cmap="Blues",
    colorbar_label="Distance",
    figsize=(8, 6),
    vmin=None,
    vmax=None,
    annot=True,
):

    with sns.axes_style("white"):
        # Based on https://seaborn.pydata.org/examples/many_pairwise_correlations.html

        fig, ax = plt.subplots(figsize=figsize)

        # Upper triangle mask
        triangle_mask = np.triu(np.ones_like(df, dtype=np.bool))

        sns.heatmap(
            df,
            cmap=cmap,
            # Draw the heatmap with the mask and correct aspect ratio
            mask=triangle_mask,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={"shrink": 0.5, "label": colorbar_label},
            linewidths=0,
            ax=ax,
            # force all tick labels to be drawn
            xticklabels=True,
            yticklabels=True,
            annot=annot,
        )

        xticklabels = ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0,
        )
        genetools.plots.wrap_tick_labels(
            ax, wrap_x_axis=True, wrap_y_axis=False, wrap_amount=10
        )

        return fig


# %%
# consider euclidean distance too?
def v_gene_use_plot_by_batch(
    gene_locus: GeneLocus, distance_metric="cosine", filter_v_genes=True
):
    output_dir, highres_output_dir = get_dirs(gene_locus)

    # Reload
    (
        specimen_v_gene_counts_df_test_only,
        v_gene_cols,
        v_gene_cols_filtered,
    ) = import_v_gene_counts(gene_locus=gene_locus)

    if filter_v_genes:
        # Use filtered set of V genes - ignore rare ones that may make specimens look artificially distinct
        v_gene_cols = v_gene_cols_filtered

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
    assert (
        v_gene_usage_proportions_by_specimen_annot.shape[0]
        == v_gene_usage_proportions_by_specimen.shape[0]
    )
    assert not v_gene_usage_proportions_by_specimen_annot["disease"].isna().any()
    assert not v_gene_usage_proportions_by_specimen_annot["study_name"].isna().any()

    ## PCA and UMAP of V gene use by specimen, colored by disease + batch
    adata_vgene_use = anndata.AnnData(
        X=v_gene_usage_proportions_by_specimen_annot[v_gene_cols],
        obs=v_gene_usage_proportions_by_specimen_annot[
            v_gene_usage_proportions_by_specimen_annot.columns[
                ~v_gene_usage_proportions_by_specimen_annot.columns.isin(v_gene_cols)
            ]
        ],
    )

    adata_vgene_use.obs["disease_and_batch"] = (
        adata_vgene_use.obs["disease"] + " - " + adata_vgene_use.obs["study_name"]
    )

    adata_vgene_use.raw = adata_vgene_use
    sc.pp.scale(adata_vgene_use)
    sc.pp.pca(adata_vgene_use)
    sc.pp.neighbors(adata_vgene_use, metric=distance_metric)
    sc.tl.umap(adata_vgene_use)

    fig_pca = sc.pl.pca(
        adata_vgene_use,
        color="disease_and_batch",
        alpha=0.5,
        return_fig=True,
        title=f"V gene proportions PCA by disease+batch\n{gene_locus}",
    )
    genetools.plots.savefig(
        fig_pca,
        highres_output_dir
        / f"v_gene_proportions_by_specimen.pca.color_by_disease_batch.png",
        dpi=300,
    )
    fig_umap = sc.pl.umap(
        adata_vgene_use,
        color="disease_and_batch",
        alpha=0.5,
        return_fig=True,
        title=f"V gene proportions UMAP by disease+batch\n{gene_locus}",
    )
    genetools.plots.savefig(
        fig_umap,
        highres_output_dir
        / f"v_gene_proportions_by_specimen.umap.color_by_disease_batch.png",
        dpi=300,
    )

    ## Heatmap of pairwise distance between means of each disease+batch category
    means = v_gene_usage_proportions_by_specimen_annot.groupby(
        ["disease", "study_name"]
    )[v_gene_cols].mean()
    # distance matrix
    dist_mat = pd.DataFrame(
        squareform(pdist(means, metric=distance_metric)),
        # combine disease and study name multiindex
        index=means.index.map(" - ".join),
        columns=means.index.map(" - ".join),
    )
    #     ax = sns.heatmap(dist_mat, cmap="Blues", square=True)
    #     ax.set_xticklabels(
    #         ax.get_xticklabels(), rotation=60, size=12, horizontalalignment="right"
    #     )

    fig_means = plot_triangular_heatmap(dist_mat, vmin=0)
    plt.title(
        f"{distance_metric} distance between disease+batch mean V gene usage\n{gene_locus}"
    )
    genetools.plots.savefig(
        fig_means,
        highres_output_dir
        / f"v_gene_proportions_by_specimen.means_by_disease_batch.distance_heatmap.png",
        dpi=300,
    )

    ## Repeat with median to be resilient to outliers:
    # Heatmap of pairwise distance between medians of each disease+batch category
    medians = v_gene_usage_proportions_by_specimen_annot.groupby(
        ["disease", "study_name"]
    )[v_gene_cols].median()
    # distance matrix
    dist_mat = pd.DataFrame(
        squareform(pdist(medians, metric=distance_metric)),
        # combine disease and study name multiindex
        index=medians.index.map(" - ".join),
        columns=medians.index.map(" - ".join),
    )
    fig_medians = plot_triangular_heatmap(dist_mat, vmin=0)
    plt.title(
        f"{distance_metric} distance between disease+batch median V gene usage\n{gene_locus}"
    )
    genetools.plots.savefig(
        fig_medians,
        highres_output_dir
        / f"v_gene_proportions_by_specimen.medians_by_disease_batch.distance_heatmap.png",
        dpi=300,
    )

    ## Get pairwise distances between specimens. Plot as boxplots grouped by specimen info.

    # Sort order so that when we construct pairwise distance dataframe and take upper triangle, we are left with all comparisons as A vs B only, not some as A vs B and others as B vs A.
    v_gene_usage_proportions_by_specimen_annot.sort_values(
        ["disease", "study_name"], inplace=True
    )

    new_index = pd.MultiIndex.from_frame(
        v_gene_usage_proportions_by_specimen_annot.reset_index()[
            ["specimen_label", "disease", "study_name"]
        ]
    )
    all_dists = pd.DataFrame(
        squareform(
            pdist(
                v_gene_usage_proportions_by_specimen_annot[v_gene_cols],
                metric=distance_metric,
            )
        ),
        index=new_index.rename([f"{name}_1" for name in new_index.names]),
        columns=new_index.rename([f"{name}_2" for name in new_index.names]),
    )

    # Make sure no nans to begin with, because we're going to set lower triangle to nan and drop all nans.
    assert not all_dists.isna().any().any()

    # Upper triangle mask to remove symmetric duplicate entries
    upper_triangle_mask = np.triu(np.ones_like(all_dists, dtype=bool))
    all_dists = all_dists.where(upper_triangle_mask)

    # Move to long form
    # Special stack logic to handle multiindex
    # Also drop N/As
    all_dists = all_dists.stack(
        level=list(range(all_dists.columns.nlevels)), dropna=True
    ).reset_index(name="distance")

    all_dists["comparison"] = (
        all_dists["disease_1"]
        + " - "
        + all_dists["study_name_1"]
        + " vs "
        + all_dists["disease_2"]
        + " - "
        + all_dists["study_name_2"]
    )

    fig_boxplot, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=all_dists,
        x="distance",
        y="comparison",
        order=all_dists.groupby("comparison")["distance"].median().sort_values().index,
        ax=ax,
    )
    ax.set_xlabel(f"{distance_metric} distance")
    ax.set_title(
        f"Pairwise distances between specimens V gene use proportions\n{gene_locus}"
    )
    genetools.plots.savefig(
        fig_boxplot,
        highres_output_dir
        / f"v_gene_proportions_by_specimen.pairwise_distances.boxplot_by_disease_batch.png",
        dpi=300,
    )

    return adata_vgene_use


# %%

# %%

# %%

# %%
adata_vgene_use_bcr = v_gene_use_plot_by_batch(gene_locus=GeneLocus.BCR)

# %%
adata_vgene_use_tcr = v_gene_use_plot_by_batch(gene_locus=GeneLocus.TCR)

# %%

# %%
for gene_locus, adata_vgene_use_single_locus in [
    (GeneLocus.BCR, adata_vgene_use_bcr),
    (GeneLocus.TCR, adata_vgene_use_tcr),
]:
    output_dir, highres_output_dir = get_dirs(gene_locus)
    for color in [
        "disease_subtype",
        "disease_and_batch",
        "disease_severity",
        "ethnicity_condensed",
        "age",
        "age_group",
        "age_group_binary",
        "age_group_pediatric",
        "sex",
    ]:
        fig_umap = sc.pl.umap(
            adata_vgene_use_single_locus,
            color=color,
            alpha=0.5,
            return_fig=True,
            title=f"V gene proportions UMAP by {color}\n{gene_locus.name}",
        )
        genetools.plots.savefig(
            fig_umap,
            highres_output_dir
            / f"v_gene_proportions_by_specimen.umap.color_by_variable.{color}.png",
            dpi=300,
        )

# %%

# %%

# %%

# %%

# %%
