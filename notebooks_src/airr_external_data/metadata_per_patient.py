# %%
import sys
import os
import numpy as np
import pandas as pd

# %%
import glob

# %%
from malid import config, helpers
from malid.datamodels import healthy_label

# %%
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# %%

# %% [markdown]
# # produce metadata about external cohorts
#
# produces:
#
# - all cohorts
#     - `metadata/generated.external_cohorts.all_bcr.participant_metadata.tsv`
# - covid
#     - `metadata/generated.external_cohorts.covid19_bcr.specimen_metadata_extra.tsv`
#     - `metadata/generated.external_cohorts.covid19_bcr.participant_metadata.tsv`
# - healthy
#     - `metadata/generated.external_cohorts.healthy_bcr.participant_metadata.tsv`

# %%

# %% [markdown]
# # external covid cohorts
#
# ## load metadata
#
# `repertoire_id`'s are like specimen IDs. they are many-to-one with patient IDs. below, we will introduce actual patient IDs for these external cohorts.

# %%

# %%
df = pd.read_csv(
    config.paths.external_raw_data / "Kim" / "airr_covid19_metadata.tsv",
    sep="\t",
)
df.shape

# %%
df.head()

# %%
df[
    [
        "repertoire_id",
        "study.study_id",
        "subject.subject_id",
        "sample.0.collection_time_point_relative",
        "sample.0.collection_time_point_reference",
    ]
].head()

# %%

# %%

# %%
df["sample.0.collection_time_point_relative"].value_counts()

# %%
df["sample.0.collection_time_point_reference"].value_counts()

# %%
assert not df["sample.0.collection_time_point_relative"].isna().any()
assert not df["sample.0.collection_time_point_reference"].isna().any()

# %%

# %%

# %%

# %%

# %%
df.columns

# %%
print("\n".join(df.columns))

# %%

# %%
df[df.columns[df.columns.str.startswith("study.")]].drop_duplicates()

# %%
study_names = {"PRJNA648677": "Kim"}  # could add more here
study_names

# %%

# %%

# %%
for study_id in study_names.keys():
    display(
        df[df["study.study_id"] == study_id][
            ["repertoire_id", "study.study_id"]
            + df.columns[df.columns.str.startswith("subject.")].tolist()
        ]
        .dropna(how="all", axis=1)
        .drop_duplicates()
    )

# %%

# %%

# %%

# %%

# %%

# %%

# %%
for study_id in study_names.keys():
    display(
        df[df["study.study_id"] == study_id][
            ["repertoire_id", "study.study_id", "subject.subject_id"]
            + df.columns[df.columns.str.startswith("sample.")].tolist()
        ]
        .dropna(how="all", axis=1)
        .drop_duplicates()
    )

# %%

# %%
for study_id in study_names.keys():
    display(
        df[df["study.study_id"] == study_id][
            ["repertoire_id", "study.study_id", "subject.subject_id"]
            + [
                "subject.sex",
                "subject.age_min",
                "subject.race",
                "subject.diagnosis.0.study_group_description",
                "subject.diagnosis.0.disease_diagnosis.label",
            ]
            + ["sample.0.sample_id", "sample.0.collection_time_point_relative"]
        ]
        .dropna(how="all", axis=1)
        .drop_duplicates()
    )

# %%

# %% [markdown]
# ## create patient IDs, and extract some patient-level metadata

# %%
# find the right columns...
for study_id in study_names.keys():
    display(
        df[df["study.study_id"] == study_id][
            ["repertoire_id", "study.study_id", "subject.subject_id"]
            + [
                "subject.sex",
                "subject.age_min",
                "subject.race",
                "subject.diagnosis.0.study_group_description",
                "subject.diagnosis.0.disease_diagnosis.label",
            ]
        ]
        .dropna(how="all", axis=1)
        .drop_duplicates()
    )

# %%

# %%
specimens_df = (
    df[df["study.study_id"].isin(study_names.keys())][
        [
            "repertoire_id",  # the internal repertoire ID
            "study.study_id",
            "subject.subject_id",
            "subject.sex",
            "subject.age_min",
            "subject.race",
            "subject.diagnosis.0.study_group_description",
            "subject.diagnosis.0.disease_diagnosis.label",
            "sample.0.collection_time_point_relative",
            "sample.0.sample_id",
        ]
    ]
    .dropna(how="all", axis=1)
    .drop_duplicates()
    .rename(
        columns={
            "subject.subject_id": "patient_id_within_study",
            "study.study_id": "study_id",
            "subject.sex": "sex",
            "subject.age_min": "age",
            "subject.race": "ethnicity",
            "subject.diagnosis.0.study_group_description": "disease_subtype",
            "subject.diagnosis.0.disease_diagnosis.label": "disease",
            "sample.0.collection_time_point_relative": "timepoint",
            "sample.0.sample_id": "specimen_label",
        }
    )
)
specimens_df["sex"] = specimens_df["sex"].replace({"male": "M", "female": "F"})
specimens_df

# %%
specimens_df["ethnicity"].value_counts()

# %%
specimens_df["ethnicity"].isna().value_counts()

# %%
# create ethnicity_condensed
specimens_df["ethnicity_condensed"] = specimens_df["ethnicity"].replace(
    {"Korean": "Asian", "Chinese": "Asian"}
)
specimens_df["ethnicity_condensed"].value_counts()

# %%
specimens_df["ethnicity_condensed"].isna().value_counts()

# %%
# Are there any non-NaN ethnicity values that we did not remap?
specimens_df[specimens_df["ethnicity_condensed"].isna()]["ethnicity"].value_counts()

# %%

# %%
specimens_df["study_name"] = specimens_df["study_id"].replace(study_names)
specimens_df

# %%
specimens_df["participant_label"] = (
    specimens_df["study_name"].str.strip()
    + "_"
    + specimens_df["patient_id_within_study"].str.strip()
)
specimens_df

# %%
# extract number
specimens_df["timepoint"] = specimens_df["timepoint"].str.extract("(\d+)").astype(int)
specimens_df

# %%
specimens_df["disease"] = specimens_df["disease"].replace({"COVID-19": "Covid19"})

# %%
specimens_df.shape, specimens_df["participant_label"].nunique()

# %%

# %% [markdown]
# ## Look at timepoints, decide which ones are peak

# %%

# %%
specimens_df

# %%
specimens_df.groupby("participant_label").size().sort_values(ascending=False)

# %%
specimens_df.groupby(["participant_label", "timepoint"]).size().sort_values(
    ascending=False
).to_frame("num_replicates").head()

# %%

# %%
# # can't do it this way because this will choose only one row as peak per patient, whereas we want all replicates from the peak timepoint to be marked as peak
# specimens_df['is_peak'] = False
# specimens_df.loc[specimens_df.groupby("participant_label", observed=True)["timepoint"].idxmax(), 'is_peak'] = True
# specimens_df['is_peak'].value_counts()

# %%
specimens_df

# %%
# choose peak timepoints per patient, with constraints on the timepoint range
# and exclude patients known to have mild disease (e.g. from Montague et al study, subjects 1-2 are mild disease)

# reset index to make sure .loc[idxmin] works properly
peak_timepoint_per_patient = (
    specimens_df[
        (specimens_df["timepoint"] >= 10)
        & (specimens_df["timepoint"] <= 45)
        & (specimens_df["disease_subtype"] != "Mild")
    ]
).reset_index(drop=True)


# choose the timepoint closest to day 15
# choose one row per group
peak_timepoint_per_patient["timepoint_diff_from_15"] = (
    peak_timepoint_per_patient["timepoint"] - 15
).abs()
peak_timepoint_per_patient = peak_timepoint_per_patient.loc[
    peak_timepoint_per_patient.groupby("participant_label", observed=True)[
        "timepoint_diff_from_15"
    ].idxmin()
].assign(is_peak=True)
peak_timepoint_per_patient

# %%
# Note that peak timepoint may have many replicates!
specimens_df2 = pd.merge(
    specimens_df,
    peak_timepoint_per_patient[["participant_label", "timepoint", "is_peak"]],
    on=["participant_label", "timepoint"],
    how="left",
)
specimens_df2["is_peak"].fillna(False, inplace=True)
assert specimens_df2.shape[0] == specimens_df.shape[0]
specimens_df = specimens_df2
specimens_df

# %%

# %%
# not all patients have any peak timepoints chosen
specimens_df["participant_label"].nunique(), specimens_df[specimens_df["is_peak"]][
    "participant_label"
].nunique()

# %%
# not all patients have any peak timepoints chosen
set(specimens_df["participant_label"]) - set(
    specimens_df[specimens_df["is_peak"]]["participant_label"]
)

# %%

# %%
# how many replicates chosen as peak per patient (should be more than 1 replicate for many)
specimens_df[specimens_df["is_peak"]].groupby(
    ["participant_label", "timepoint"]
).size().sort_values(ascending=False)

# %%
# which were chosen
specimens_df[specimens_df["is_peak"]][
    ["participant_label", "timepoint", "is_peak"]
].sort_values(["participant_label", "timepoint"])

# %%

# %% [markdown]
# ## export

# %%
specimen_metadata_extra = (
    specimens_df[["specimen_label", "participant_label", "timepoint", "is_peak"]]
    .drop_duplicates()
    .sort_values(["participant_label", "timepoint"])
)
specimen_metadata_extra.to_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid19_bcr.specimen_metadata_extra.tsv",
    sep="\t",
    index=None,
)
specimen_metadata_extra

# %%
participant_df = specimens_df[
    [
        "participant_label",
        "study_id",
        "patient_id_within_study",
        "sex",
        "age",
        "ethnicity_condensed",
        "disease_subtype",
        "disease",
        "study_name",
    ]
].drop_duplicates()
participant_df.to_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid19_bcr.participant_metadata.tsv",
    sep="\t",
    index=None,
)
participant_df

# %%
participant_df.shape, specimen_metadata_extra.shape

# %%

# %%

# %%

# %% [markdown]
# # now for briney-healthy
#
# we already have `patient_id` in `ireceptor_data.briney_healthy_sequences`
#
# soon we will add more replicates i.e. more samples though. see `repertoire_id` TODO comments.
#
# for now, just add some simple metadata to create an "all external participants" metadata file

# %%
briney_patients = pd.DataFrame(
    {
        "specimen_label": [
            "D103_1",
            "326780_1",
            "326650_1",
            "326737_1",
            "327059_1",
            "326907_1",
            "316188_1",
            "326797_1",
        ]
    }
)
briney_patients["participant_label"] = (
    briney_patients["specimen_label"].str.split("_").str[0]
)
briney_patients["study_name"] = "Briney"
briney_patients["disease"] = healthy_label
# all healthy are "peak" and 0 timepoint
briney_patients["is_peak"] = True
briney_patients["timepoint"] = 0
print(briney_patients.shape)
briney_patients

# %%
# Original paper table is wrong: 326907 is listed twice with different values; 326737 is missing. One of the dupes should be 326737.
# Fixed based on:
# https://www.ncbi.nlm.nih.gov/biosample/10331432
# https://www.ncbi.nlm.nih.gov/biosample/10331429
briney_demographics = pd.read_csv(config.paths.metadata_dir / "briney_demographics.csv")
assert not briney_demographics["subject"].duplicated().any()
print(briney_demographics.shape)
briney_demographics["sex"] = briney_demographics["sex"].replace(
    {"male": "M", "female": "F"}
)
briney_demographics

# %%
briney_demographics["ethnicity"].value_counts()

# %%
briney_demographics["ethnicity_condensed"] = briney_demographics["ethnicity"].replace(
    {"African American": "African", "African American / Caucasian": np.nan}
)
briney_demographics["ethnicity_condensed"].value_counts()

# %%
briney_demographics["ethnicity_condensed"].isna().value_counts()

# %%
# Are there any non-NaN ethnicity values that we did not remap?
briney_demographics[briney_demographics["ethnicity_condensed"].isna()][
    "ethnicity"
].value_counts()

# %%
briney_patients = pd.merge(
    briney_patients,
    briney_demographics.set_index("subject"),
    how="left",
    validate="1:1",
    left_on="participant_label",
    right_index=True,
)
briney_patients

# %%

# %%
briney_patients.to_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.healthy_bcr.participant_metadata.tsv",
    sep="\t",
    index=None,
)

# %%

# %%
participant_df_plus_briney = pd.concat(
    [
        participant_df[["participant_label", "disease", "study_name"]],
        briney_patients[["participant_label", "disease", "study_name"]],
    ],
    axis=0,
)
participant_df_plus_briney

# %%
participant_df_plus_briney.to_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.all_bcr.participant_metadata.tsv",
    sep="\t",
    index=None,
)

# %%

# %%

# %%

# %%
