# %%
import numpy as np
import pandas as pd
from malid import config, io, helpers
from malid.datamodels import healthy_label, GeneLocus, TargetObsColumnEnum
import genetools
from genetools.palette import HueValueStyle
import seaborn as sns

# %%

# %%
# Uses data from vgene_usage_stats.ipynb


def get_dirs(gene_locus: GeneLocus):
    output_dir = (
        config.paths.model_interpretations_for_selected_cross_validation_strategy_output_dir
        / gene_locus.name
    )
    highres_output_dir = (
        config.paths.high_res_outputs_dir_for_cross_validation_strategy
        / "model_interpretations"
        / gene_locus.name
    )

    return output_dir, highres_output_dir


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
totals = {}
for gene_locus in config.gene_loci_used:
    print(gene_locus)
    df, v_gene_cols, _ = import_v_gene_counts(gene_locus=gene_locus)
    totals[gene_locus.name] = df[v_gene_cols].sum(axis=1).astype(int)
totals = pd.DataFrame(totals).fillna(0).astype(int)
totals

# %%
# have some BCR only specimens, as expected
(totals == 0).any(axis=0)

# %%
# have some BCR only specimens, as expected
totals.loc[(totals == 0).any(axis=1)]

# %%
cols = totals.columns
cols

# %%
total = totals.sum(axis=1)
total

# %%

# %%
orig_shape = totals.shape
totals = pd.merge(
    totals,
    helpers.get_all_specimen_info().set_index("specimen_label")[
        ["disease", "study_name", "participant_label", "in_training_set"]
    ],
    left_index=True,
    right_index=True,
    validate="1:1",
    how="inner",
)
assert totals.shape[0] == orig_shape[0]
totals

# %%
assert totals["in_training_set"].all(), "sanity check"

# %%
totals.to_csv(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "selected_clone_count_for_each_specimen.tsv",
    sep="\t",
)

# %%

# %%
# num clones
totals.groupby(["disease", "study_name"], observed=True)[cols].sum()

# %%
# num patients
totals.groupby(["disease", "study_name"], observed=True)[
    "participant_label"
].nunique().to_frame(name="number of individuals")

# %%
# num specimens
totals.groupby(["disease", "study_name"], observed=True).size().to_frame(
    name="number of specimens"
)

# %%

# %%
# make a table of all
df_all = pd.concat(
    [
        # num patients
        totals.groupby(["disease", "study_name"], observed=True)["participant_label"]
        .nunique()
        .to_frame(name="number of individuals"),
        # num specimens
        totals.groupby(["disease", "study_name"], observed=True)
        .size()
        .to_frame(name="number of specimens"),
        # num clones
        totals.groupby(["disease", "study_name"], observed=True)[cols].sum(),
    ],
    axis=1,
)
df_all.to_csv(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "size_of_each_disease_batch.tsv",
    sep="\t",
)
df_all


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # Sanity checks


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
if GeneLocus.BCR in config.gene_loci_used:
    df, v_gene_cols, _ = import_v_gene_counts(gene_locus=GeneLocus.BCR)
    total = df[v_gene_cols].sum(axis=1).astype(int)
    print(total)

# %%
if GeneLocus.BCR in config.gene_loci_used:
    df, v_gene_cols = import_cdr3_length_counts(gene_locus=GeneLocus.BCR)
    total2 = df[v_gene_cols].sum(axis=1).astype(int)
    print(total2)
    assert (total == total2).all()

# %%
if GeneLocus.BCR in config.gene_loci_used:
    specimen_isotype_counts_df = pd.read_csv(
        config.paths.dataset_specific_metadata_for_selected_cross_validation_strategy
        / "isotype_counts_by_specimen.tsv",
        sep="\t",
    )
    specimen_isotype_counts_df = specimen_isotype_counts_df[
        specimen_isotype_counts_df["fold_label"] == "test"
    ]
    assert not specimen_isotype_counts_df["specimen_label"].duplicated().any()
    specimen_isotype_counts_df = specimen_isotype_counts_df.set_index("specimen_label")[
        ["IGHD-M", "IGHA", "IGHG"]
    ]
    total3 = specimen_isotype_counts_df.sum(axis=1)
    print(total3)

    print(set(total.index).symmetric_difference(set(total3.index)))

    assert (total == total3.loc[total.index]).all()

# %%

# %%
if GeneLocus.TCR in config.gene_loci_used:
    df, v_gene_cols, _ = import_v_gene_counts(gene_locus=GeneLocus.TCR)
    total = df[v_gene_cols].sum(axis=1)
    print(total)

# %%
if GeneLocus.TCR in config.gene_loci_used:
    df, v_gene_cols = import_cdr3_length_counts(gene_locus=GeneLocus.TCR)
    total2 = df[v_gene_cols].sum(axis=1)
    print(total2)
    assert (total == total2).all()

# %%

# %%

# %%

# %% [markdown]
# # Plot

# %%
# Create "available loci" column
totals_annot = pd.concat(
    [
        totals,
        totals.apply(
            lambda row: (["BCR"] if "BCR" in row and row["BCR"] > 0 else [])
            + (["TCR"] if "TCR" in row and row["TCR"] > 0 else []),
            axis=1,
        ).rename("Available loci"),
    ],
    axis=1,
)
totals_annot["Available loci"] = totals_annot["Available loci"].str.join(" + ")
totals_annot

# %%
assert totals_annot["in_training_set"].all()

# %%
# same values across all rows for each person
assert (
    totals_annot.groupby("participant_label")["Available loci"].nunique() == 1
).all()

# %%
totals_annot_dedupe_by_participant = totals_annot.groupby("participant_label").head(n=1)
totals_annot_dedupe_by_participant

# %%
fig, ax = genetools.plots.stacked_bar_plot(
    totals_annot_dedupe_by_participant.assign(
        disease=totals_annot_dedupe_by_participant["disease"].replace(
            {
                # Friendlier names:
                "Influenza": "Influenza vaccine",
                "T1D": "Type-1 diabetes",
                # Note there's a space character in front of Healthy to force it to be first in sort order
                "Healthy/Background": " Healthy",
                "Covid19": "Covid-19",
            }
        )
    ),
    index_key="disease",
    hue_key="Available loci",
    palette={
        "BCR + TCR": HueValueStyle(color=sns.color_palette()[0]),
        "BCR": HueValueStyle(color=sns.color_palette()[1], hatch="///"),
        "TCR": HueValueStyle(color=sns.color_palette()[2], hatch="..."),
    },
    figsize=(4, 3),
    # ax=ax,
    normalize=False,
    vertical=False,
    hue_order=[
        x
        for x in ["BCR + TCR", "BCR", "TCR"]
        if x in totals_annot_dedupe_by_participant["Available loci"].unique()
    ],
    axis_label="Patient count",
    legend_title="Data",
    enable_legend=True,
)
genetools.plots.savefig(
    fig,
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "number_of_patients_by_disease.png",
    dpi=300,
)
genetools.plots.savefig(
    fig,
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "number_of_patients_by_disease.pdf",
)

# %%
# Spot check against:
totals_annot.groupby("disease")["Available loci"].value_counts()

# %%
# Spot check against:
totals_annot.groupby("disease").size()

# %%
