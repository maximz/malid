# %% [markdown]
# # Combine metadata, then create necessary columns for ETL pipeline based on similar notebook for Adaptive data.
#
# There are some unique columns in the final exported notebook that aren't present in the Adaptive version:
#
# - `expect_a_read_count_column`: Allow some studies to be exempted from read count column requirements
# - `file_extension`: Allow custom file extensions for some studies. Default is tsv
#
# Also note that `Britanova` is a special-cased study name that triggers unique ETL behavior.
#
# Make sure the study names match the data locations on disk.

# %%
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed

# %%
from malid import config, helpers, etl, get_v_sequence, io, logger
from malid.datamodels import GeneLocus, healthy_label
from malid.sample_sequences import sample_sequences
from malid.trained_model_wrappers import ConvergentClusterClassifier

# %%

# %%

# %% [markdown]
# # get specimen filepaths from specimen metadata list

# %%

# %% [markdown]
# ## covid samples

# %%
covid_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid19_bcr.specimen_metadata_extra.tsv",
    sep="\t",
)
covid_specimens

# %%
covid_specimens.shape

# %%
participant_df = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid19_bcr.participant_metadata.tsv",
    sep="\t",
)
participant_df

# %%
covid_specimens = pd.merge(
    covid_specimens, participant_df, how="left", validate="m:1", on="participant_label"
)
covid_specimens

# %%
covid_specimens.shape

# %%
covid_specimens["disease_subtype"] = (
    covid_specimens["disease"]
    + " - "
    + covid_specimens["study_name"]
    + covid_specimens["is_peak"].replace({True: "", False: " (non-peak)"})
)
covid_specimens["gene_locus"] = GeneLocus.BCR.name
covid_specimens

# %%
covid_specimens = covid_specimens[covid_specimens["is_peak"]]
covid_specimens = covid_specimens[covid_specimens["study_name"] == "Kim"]
covid_specimens

# %%
# Special column:
# No read counts in Kim iReceptor Covid
covid_specimens["expect_a_read_count_column"] = False

# Study name will be prepended, so remove prefix from participant_label
covid_specimens["participant_label"] = covid_specimens["participant_label"].str.replace(
    "Kim_", ""
)

covid_specimens

# %%

# %% [markdown]
# ## healthy specimens

# %%
healthy_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.healthy_bcr.participant_metadata.tsv",
    sep="\t",
)

# process peak samples only
healthy_specimens = healthy_specimens[healthy_specimens["is_peak"] == True]

healthy_specimens["disease_subtype"] = (
    healthy_specimens["disease"]
    + " - "
    + healthy_specimens["study_name"]
    + healthy_specimens["is_peak"].replace({True: "", False: " (non-peak)"})
)

healthy_specimens["gene_locus"] = GeneLocus.BCR.name

# Special columns:
# No read counts in Briney
healthy_specimens["expect_a_read_count_column"] = False
# Unusual file extension
healthy_specimens["file_extension"] = "csv"

healthy_specimens

# %%

# %% [markdown]
# ## healthy TCR specimens

# %%
tcr_healthy_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.healthy_tcr_britanova.participant_metadata.tsv",
    sep="\t",
).assign(
    is_peak=True,
    gene_locus=GeneLocus.TCR.name,
)

# Special column:
# Unusual file extension
tcr_healthy_specimens["file_extension"] = "txt.gz"

tcr_healthy_specimens

# %%

# %% [markdown]
# ## Covid TCR specimens

# %%
tcr_covid_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid_tcr_shomuradova.participant_metadata.tsv",
    sep="\t",
).assign(
    is_peak=True,
    gene_locus=GeneLocus.TCR.name,
)
tcr_covid_specimens["disease_subtype"] = tcr_covid_specimens[
    "disease_subtype"
].str.replace("Covid19 -", "Covid19 - Shomuradova -")

# Special column:
# No read counts in Shomuradova
tcr_covid_specimens["expect_a_read_count_column"] = False

tcr_covid_specimens

# %%

# %% [markdown]
# # Combine

# %%
dfs_external = pd.concat(
    [
        covid_specimens,
        healthy_specimens,
        tcr_healthy_specimens,
        tcr_covid_specimens,
    ],
    axis=0,
)
dfs_external

# %%
dfs_external["disease"].value_counts()

# %%
dfs_external["disease_subtype"].value_counts()

# %%
dfs_external["participant_label"]

# %%
dfs_external["specimen_label"]

# %%

# %%
assert not dfs_external["gene_locus"].isna().any()

# %%
# Make sure special columns are present
dfs_external["expect_a_read_count_column"].fillna(True, inplace=True)
dfs_external["file_extension"].fillna("tsv", inplace=True)
print(dfs_external["expect_a_read_count_column"].value_counts())
print(dfs_external["file_extension"].value_counts())

# %%

# %%
# Columns:
# study_name
# participant_label
# specimen_label: globally unique, but may have several amplifications and replicates.
# amplification_label: globally unique, but may have several replicates.
# replicate_label: globally unique.
# sample_name: not globally unique, but should be unique within each study. used in the fasta header and igblast parsed "id" column.

# %%

# %%

# %%

# %%
assert not dfs_external["study_name"].isna().any()
assert not dfs_external["participant_label"].isna().any()
assert not dfs_external["specimen_label"].isna().any()

# %%
# To be consistent with boydlab columns, we'll add amplification_label, which here will always equal specimen_label.
# See sample_sequences.py for more details on how this gets used.
if "amplification_label" not in dfs_external.columns:
    dfs_external["amplification_label"] = dfs_external["specimen_label"]
else:
    # fill NA
    dfs_external["amplification_label"].fillna(
        dfs_external["specimen_label"], inplace=True
    )

# Fill replicate_label
if "replicate_label" not in dfs_external.columns:
    dfs_external["replicate_label"] = dfs_external["specimen_label"]
else:
    # fill NA
    dfs_external["replicate_label"].fillna(dfs_external["specimen_label"], inplace=True)

# Fill sample_name
if "sample_name" not in dfs_external.columns:
    dfs_external["sample_name"] = dfs_external["specimen_label"]
else:
    # fill NA
    dfs_external["sample_name"].fillna(dfs_external["specimen_label"], inplace=True)

# %%
dfs_external

# %%
# add study prefixes to make these labels unique to study:
for col in [
    "participant_label",
    "specimen_label",
    "amplification_label",
    "replicate_label",
]:
    dfs_external[col] = dfs_external["study_name"] + "_" + dfs_external[col].astype(str)

# %%
dfs_external

# %%
# confirm one entry per replicate label per locus, at most!
# (specimens can have multiple replicates, e.g. cell type subsets that get merged.)
# (participants can have multiple specimens, e.g. separate time points)
assert (dfs_external.groupby(["gene_locus", "replicate_label"]).size() == 1).all()

# %%
dfs_external["participant_label"].unique()

# %%

# %%
dfs_external["sequencing_type"] = "cDNA"

# %%

# %%

# %%
dfs_external.groupby(["sequencing_type", "gene_locus", "disease"], observed=True)[
    "participant_label"
].nunique().to_frame().sort_values("participant_label")

# %%
dfs_external["disease_subtype"].isna().any()

# %%
dfs_external["disease"].isna().any()

# %%
dfs_external["disease_subtype"].fillna(dfs_external["disease"], inplace=True)

# %%
dfs_external.isna().any()[dfs_external.isna().any()]

# %%
dfs_external["disease_subtype"].value_counts()

# %%
dfs_external[dfs_external["disease_subtype"] == healthy_label][
    "study_name"
].value_counts()

# %%
dfs_external.groupby(["gene_locus", "disease", "disease_subtype"], observed=True)[
    "participant_label"
].nunique().to_frame()

# %%
dfs_external.groupby(
    ["gene_locus", "disease", "disease_subtype", "study_name"], observed=True
)["participant_label"].nunique().to_frame()

# %%
dfs_external.groupby("disease")["participant_label"].nunique().sort_values()

# %%

# %%
# Review which replicates are getting combined into which specimens
# dfs_external[dfs_external['replicate_label'] != dfs_external['specimen_label']].groupby('specimen_label')['replicate_label'].unique().tolist()
dfs_external[dfs_external["replicate_label"] != dfs_external["specimen_label"]][
    ["specimen_label", "replicate_label"]
]

# %%
# # Review which replicates are getting combined into which specimens
# replicates_being_merged_into_same_specimen = (
#     dfs_external[dfs_external["replicate_label"] != dfs_external["specimen_label"]]
#     .groupby("specimen_label")["replicate_label"]
#     .unique()
#     .apply(pd.Series)
# )
# # remove rows where single replicate (but just happened to have different label) - no merging happening
# replicates_being_merged_into_same_specimen = replicates_being_merged_into_same_specimen[
#     replicates_being_merged_into_same_specimen.notna().sum(axis=1) > 1
# ]
# replicates_being_merged_into_same_specimen

# %%

# %%

# %%

# %%

# %%
# all available columns, in case-insensitive sorted order
dfs_external.columns.sort_values(key=lambda idx: idx.str.lower())

# %%
# specimen description can come in several fields:
specimen_description_fields = ["timepoint"]

# They are either all NA or one is set. Never have multiple of these set:
assert dfs_external[specimen_description_fields].notna().sum(axis=1).max()

# So we can just take first non-null value (if any) per row from these columns (https://stackoverflow.com/a/37938780/130164):
dfs_external["specimen_description"] = (
    dfs_external[specimen_description_fields].fillna(method="bfill", axis=1).iloc[:, 0]
)
dfs_external["specimen_description"]

# %%

# %%
# Set has_BCR, has_TCR
dfs_external["has_BCR"] = False
dfs_external["has_TCR"] = False
dfs_external.loc[dfs_external["gene_locus"] == "BCR", "has_BCR"] = True
dfs_external.loc[dfs_external["gene_locus"] == "TCR", "has_TCR"] = True

# should always be one or the other:
assert (dfs_external["has_BCR"] != dfs_external["has_TCR"]).all()

print(dfs_external["has_BCR"].value_counts())
print(dfs_external["has_TCR"].value_counts())

# %%

# %%
# Subset to these surviving columns
dfs_external = dfs_external[
    [
        "study_name",
        "sample_name",
        "gene_locus",
        "disease",
        "sequencing_type",
        "disease_subtype",
        "participant_label",
        "specimen_label",
        "amplification_label",
        "replicate_label",
        "sex",
        "age",
        "ethnicity",
        "ethnicity_condensed",
        "specimen_description",
        "has_BCR",
        "has_TCR",
        # specials:
        "expect_a_read_count_column",
        "file_extension",
    ]
].copy()
dfs_external

# %%
all_specimens = dfs_external

# %%

# %% [markdown]
# # Make metadata columns consistent with standard Boydlab pipeline

# %%
all_specimens["sex"].value_counts()

# %%


# %%
all_specimens["ethnicity"].isna().value_counts()

# %%
# Here's who is missing ethnicity:
all_specimens[all_specimens["ethnicity"].isna()]["disease"].value_counts()

# %%
# Here's who is missing ethnicity:
all_specimens[all_specimens["ethnicity"].isna()]["study_name"].value_counts()

# %%

# %%
all_specimens["ethnicity_condensed"].value_counts()

# %%
all_specimens["ethnicity_condensed"].isna().value_counts()

# %%
# Here's who is missing ethnicity_condensed:
all_specimens[all_specimens["ethnicity_condensed"].isna()]["disease"].value_counts()

# %%
# Here's who is missing ethnicity_condensed:
# *Important*: If we see entries here that can be resolved, update the ethnicity_condensed rules above.
all_specimens[all_specimens["ethnicity_condensed"].isna()]["ethnicity"].value_counts()

# %%
# Here's who is missing ethnicity_condensed:
all_specimens[all_specimens["ethnicity_condensed"].isna()]["study_name"].value_counts()

# %%
# Versus total counts
all_specimens["disease"].value_counts()

# %%
all_specimens.groupby(["ethnicity_condensed", "disease"]).size()

# %%

# %%
all_specimens["age"].dropna()

# %%
# Set age_group column as well, just as in assemble_etl_metadata
all_specimens["age"].describe()

# %%
all_specimens["age_group"] = pd.cut(
    all_specimens["age"],
    bins=[0, 20, 30, 40, 50, 60, 70, 80, 100],
    labels=["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"],
    right=False,
)
all_specimens["age_group"].value_counts()

# %%
all_specimens["age_group"].cat.categories

# %%
all_specimens["age"].isna().value_counts()

# %%
all_specimens["age_group"].isna().value_counts()

# %%
for age_group, grp in all_specimens.groupby("age_group"):
    print(age_group, grp["age"].min(), grp["age"].max())

# %%
# Just as in assemble_etl_metadata:
# Null out "age_group" column for extreme ages with small sample size.

# Note that we are not getting rid of these specimens altogether,
# but marking age_group NaN will disable their use for demographics-controlling models

orig_shapes = all_specimens.shape[0], all_specimens["age_group"].isna().sum()
mask = all_specimens["age_group"].isin(["80+"])
if mask.sum() > 0:
    all_specimens.loc[mask, "age_group"] = np.nan
    new_shapes = all_specimens.shape[0], all_specimens["age_group"].isna().sum()

    # sanity checks:
    # - we did not drop any specimens
    assert orig_shapes[0] == new_shapes[0]
    # - but we did null out some age_group entries
    assert orig_shapes[1] < new_shapes[1]
    # - we nulled out the right amount
    assert new_shapes[1] - orig_shapes[1] == mask.sum()

# %%

# %%

# %%
# # Fillna for cohorts that are single-locus
# if "specimen_label_by_locus" not in all_specimens:
#     # in case we had no BCR+TCR combined cohorts that set this field already
#     all_specimens["specimen_label_by_locus"] = all_specimens["specimen_label"]
# else:
#     all_specimens["specimen_label_by_locus"].fillna(
#         all_specimens["specimen_label"], inplace=True
#     )

# %%
# # make sure input fnames exist
# assert all_specimens["fname"].apply(os.path.exists).all()
# %%
all_specimens.shape

# %%
# # confirm all specimen labels are unique within each locus (may have one BCR and one TCR line per specimen)
# # TODO: in the future, allow for replicates of each specimen
# assert not all_specimens["specimen_label_by_locus"].duplicated().any()
# for locus, grp in all_specimens.groupby("gene_locus"):
#     assert not grp["specimen_label"].duplicated().any()

# %%
# # Which specimens are in multiple loci?
# all_specimens[all_specimens["specimen_label"].duplicated(keep=False)]

# %%
all_specimens["study_name"].value_counts()

# %%
all_specimens["disease"].value_counts()

# %%
all_specimens["gene_locus"].value_counts()

# %%
all_specimens["disease_subtype"].value_counts()

# %%
for key, grp in all_specimens.groupby("disease"):
    print(key)
    print(grp["disease_subtype"].value_counts())
    print()

# %%

# %%
for key, grp in all_specimens.groupby("disease"):
    print(key)
    print(grp["specimen_description"].value_counts())
    print()

# %%

# %%
for demographics_column in ["age", "age_group", "sex", "ethnicity_condensed"]:
    print(demographics_column)
    print(all_specimens[demographics_column].value_counts())
    print(all_specimens[demographics_column].isna().value_counts())
    print()

# %%

# %%
all_specimens.drop(columns=["ethnicity"]).to_csv(
    config.paths.metadata_dir / "generated.external_cohorts.tsv",
    sep="\t",
    index=None,
)

# %%

# %%

# %%
all_specimens

# %%
all_specimens["study_name"].value_counts()

# %%

# %%
