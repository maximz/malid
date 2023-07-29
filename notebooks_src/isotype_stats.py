# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd

from malid import io
from malid import config, helpers
import gc
import joblib
from kdict import kdict
import itertools

# %%
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

from malid.datamodels import GeneLocus, TargetObsColumnEnum

# %%

# %% [markdown]
# # Isotype counts overall by disease cohort. BCR only.

# %%
fold_labels = ["train_smaller", "validation", "test"]

# %%

# %%
specimen_isotype_counts = []
for fold_id in config.all_fold_ids:
    for fold_label in fold_labels:
        if fold_id == -1 and fold_label == "test":
            # skip: global fold does not have a test set
            continue
        adata = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=GeneLocus.BCR,
            target_obs_column=TargetObsColumnEnum.disease,
            load_isotype_counts_per_specimen=False,  # because not created yet - gets created here!
        )
        df = adata.obs
        for specimen_label, subset_obs in adata.obs.groupby(
            "specimen_label", observed=True
        ):
            # get value counts, but first defensively cast to categorical and remove any unused categories — so we don't mark 0 counts for filtered out isotypes.
            isotype_counts = (
                subset_obs["isotype_supergroup"]
                .astype("category")
                .cat.remove_unused_categories()
                .value_counts()
            )
            specimen_description = subset_obs[["specimen_label", "disease"]].iloc[0]
            specimen_isotype_counts.append(
                {
                    "fold_id": fold_id,
                    "fold_label": fold_label,
                    **isotype_counts.to_dict(),
                    **specimen_description.to_dict(),
                }
            )

        del df, adata
        io.clear_cached_fold_embeddings()
        gc.collect()

# %%
specimen_isotype_counts[0]

# %%
specimen_isotype_counts_df = pd.DataFrame(specimen_isotype_counts)
specimen_isotype_counts_df

# %%
# export
specimen_isotype_counts_df.to_csv(
    config.paths.dataset_specific_metadata / "isotype_counts_by_specimen.tsv",
    sep="\t",
    index=None,
)

# %%

# %%

# %%
# reimport (start here if resuming)
specimen_isotype_counts_df = pd.read_csv(
    config.paths.dataset_specific_metadata / "isotype_counts_by_specimen.tsv", sep="\t"
)
specimen_isotype_counts_df

# %%

# %%
specimen_isotype_counts_df.sort_values(
    ["fold_id", "fold_label", "disease", "specimen_label"], inplace=True
)

# %%

# %%
# subselect to test folds only, and set index
specimen_isotype_counts_df_test_only = specimen_isotype_counts_df[
    specimen_isotype_counts_df["fold_label"] == "test"
]
# confirm only one entry per specimen now
assert not specimen_isotype_counts_df_test_only["specimen_label"].duplicated().any()
specimen_isotype_counts_df_test_only = specimen_isotype_counts_df_test_only.set_index(
    "specimen_label"
).drop(["fold_id", "fold_label"], axis=1)
specimen_isotype_counts_df_test_only

# %%
# option 1: for each disease, sum across all specimens. plot totals
specimen_isotype_counts_df_test_only.groupby("disease").sum()

# %%
pd.melt(
    specimen_isotype_counts_df_test_only.groupby("disease").sum().reset_index(),
    id_vars=["disease"],
    value_vars=["IGHD-M", "IGHA", "IGHG"],
)

# %%
sns.barplot(
    data=pd.melt(
        specimen_isotype_counts_df_test_only.groupby("disease").sum().reset_index(),
        id_vars=["disease"],
        value_vars=["IGHD-M", "IGHA", "IGHG"],
    ),
    x="disease",
    y="value",
    hue="variable",
)

# %%

# %%
# - option 2: normalize each specimen to sum to 1. for each disease, sum across all specimens. plot totals
import genetools

pd.concat(
    [
        genetools.stats.normalize_rows(
            specimen_isotype_counts_df_test_only[["IGHD-M", "IGHA", "IGHG"]]
        ),
        specimen_isotype_counts_df_test_only["disease"],
    ],
    axis=1,
)

# %%
sns.barplot(
    data=pd.melt(
        pd.concat(
            [
                genetools.stats.normalize_rows(
                    specimen_isotype_counts_df_test_only[["IGHD-M", "IGHA", "IGHG"]]
                ),
                specimen_isotype_counts_df_test_only["disease"],
            ],
            axis=1,
        )
        .groupby("disease")
        .sum()
        .reset_index(),
        id_vars=["disease"],
        value_vars=["IGHD-M", "IGHA", "IGHG"],
    ),
    x="disease",
    y="value",
    hue="variable",
)

# %%

# %%
# - option 3: normalize each specimen to sum to 1. for each disease, sum across all specimens, and normalize those 3 to sum to 1. plot those totals

genetools.stats.normalize_rows(
    pd.concat(
        [
            genetools.stats.normalize_rows(
                specimen_isotype_counts_df_test_only[["IGHD-M", "IGHA", "IGHG"]]
            ),
            specimen_isotype_counts_df_test_only["disease"],
        ],
        axis=1,
    )
    .groupby("disease")
    .sum()
)

# %%
ax = sns.barplot(
    data=pd.melt(
        genetools.stats.normalize_rows(
            pd.concat(
                [
                    genetools.stats.normalize_rows(
                        specimen_isotype_counts_df_test_only[["IGHD-M", "IGHA", "IGHG"]]
                    ),
                    specimen_isotype_counts_df_test_only["disease"],
                ],
                axis=1,
            )
            .groupby("disease")
            .sum()
        )
        .reset_index()
        .rename(columns={"disease": "Disease"}),
        id_vars=["Disease"],
        value_vars=["IGHD-M", "IGHA", "IGHG"],
        var_name="Isotype",
        value_name="Proportion",
    ),
    x="Disease",
    y="Proportion",
    hue="Isotype",
)
sns.despine(ax=ax)
legend_title = "Isotype"
# place legend outside figure
leg = plt.legend(
    bbox_to_anchor=(1.05, 0.5),
    loc="center left",
    borderaxespad=0.0,
    # no border
    frameon=False,
    # transparent background
    framealpha=0.0,
    # legend title
    title=legend_title,
)
# set legend title to bold - workaround for title_fontproperties missing from old matplotlib versions
leg.set_title(title=legend_title, prop={"weight": "bold", "size": "medium"})
# align legend title left
leg._legend_box.align = "left"

# %%

# %%
# option 4 should match option 3, but cleaner implementation and description:
# for specimens of each disease type, plot average (+/- std) of isotype proportions

# %%
pd.concat(
    [
        genetools.stats.normalize_rows(
            specimen_isotype_counts_df_test_only[["IGHD-M", "IGHA", "IGHG"]]
        ),
        specimen_isotype_counts_df_test_only["disease"],
    ],
    axis=1,
)

# %%
isotype_proportions = pd.melt(
    pd.concat(
        [
            genetools.stats.normalize_rows(
                specimen_isotype_counts_df_test_only[["IGHD-M", "IGHA", "IGHG"]]
            ),
            specimen_isotype_counts_df_test_only["disease"],
        ],
        axis=1,
    ).rename(columns={"disease": "Disease"}),
    id_vars=["Disease"],
    value_vars=["IGHD-M", "IGHA", "IGHG"],
    var_name="Isotype",
    value_name="Proportion",
)
isotype_proportions["Isotype"] = isotype_proportions["Isotype"].replace(
    helpers.isotype_friendly_names
)
isotype_proportions

# %%
ax = sns.barplot(
    data=isotype_proportions,
    x="Isotype",
    y="Proportion",
    hue="Disease",
    palette=helpers.disease_color_palette,
    # Compute 95% confidence intervals around a sample mean by bootstrapping:
    # sampling distribution of mean generated by repeated sampling and recording mean each time.
    # the standard error is basically the standard deviation of many sample means
    # we plot mean +/- 1.96*standard error. gives you average value +/- X at the 95% confidence level.
    ci=95,
    # ci="sd", # instead draw the standard deviation of the observations, instead of bootstrapping to get 95% confidence intervals
    # capsize=.025
)

sns.despine(ax=ax)
legend_title = "Disease"
# place legend outside figure
leg = plt.legend(
    bbox_to_anchor=(1.05, 0.5),
    loc="center left",
    borderaxespad=0.0,
    # no border
    frameon=False,
    # transparent background
    framealpha=0.0,
    # legend title
    title=legend_title,
)
# set legend title to bold - workaround for title_fontproperties missing from old matplotlib versions
leg.set_title(title=legend_title, prop={"weight": "bold", "size": "medium"})
# align legend title left
leg._legend_box.align = "left"

ax.set_title("Average specimen isotype proportions by disease")

fig = ax.get_figure()
genetools.plots.savefig(
    fig, config.paths.output_dir / f"isotype_counts_by_disease.png", dpi=300
)
genetools.plots.savefig(
    fig,
    config.paths.high_res_outputs_dir / f"isotype_counts_by_disease.pdf",
)

# %%
ax = sns.barplot(
    data=isotype_proportions,
    x="Disease",
    y="Proportion",
    hue="Isotype",
    # Compute 95% confidence intervals around a sample mean by bootstrapping:
    # sampling distribution of mean generated by repeated sampling and recording mean each time.
    # the standard error is basically the standard deviation of many sample means
    # we plot mean +/- 1.96*standard error. gives you average value +/- X at the 95% confidence level.
    ci=95,
    # ci="sd", # instead draw the standard deviation of the observations, instead of bootstrapping to get 95% confidence intervals
    # capsize=.025
    hue_order=helpers.isotype_friendly_name_order,
    palette=helpers.isotype_palette,
)

sns.despine(ax=ax)
legend_title = "Isotype"
# place legend outside figure
leg = plt.legend(
    bbox_to_anchor=(1.05, 0.5),
    loc="center left",
    borderaxespad=0.0,
    # no border
    frameon=False,
    # transparent background
    framealpha=0.0,
    # legend title
    title=legend_title,
)
# set legend title to bold - workaround for title_fontproperties missing from old matplotlib versions
leg.set_title(title=legend_title, prop={"weight": "bold", "size": "medium"})
# align legend title left
leg._legend_box.align = "left"

ax.set_title("Average specimen isotype proportions by disease")

fig = ax.get_figure()
genetools.plots.savefig(
    fig, config.paths.output_dir / f"isotype_counts_by_disease.inverted.png", dpi=300
)
genetools.plots.savefig(
    fig,
    config.paths.high_res_outputs_dir / f"isotype_counts_by_disease.inverted.pdf",
)

# %%

# %%
