# %% [markdown]
# # Predict specimen disease status from age/sex/ethnicity data only

# %% [markdown]
# And export included specimen lists for other confounder testing.
#
# (Can't do this prediction task for "healthy only" subset - because all y is the same value)
#

# %%

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %matplotlib inline

# %%
from IPython.display import display, Markdown
import genetools

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import sklearn.base
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

# %%
from malid import config, helpers, logger
from malid.datamodels import healthy_label
from malid.external import model_evaluation

# %%

# %%
# rare ancestries already set to ethnicity_condensed=NaN
df = helpers.get_all_specimen_info()
df

# %%
df["age_group_binary"].value_counts()

# %%
df["age_group_binary"].isna().value_counts()

# %%
df["age_group_pediatric"].value_counts()

# %%
df["age_group_pediatric"].isna().value_counts()

# %%
df

# %%

# %% [markdown]
# # Load actual specimens in different CV folds
#
#
# * restrict df to only specimens that made it into folds
# * then mark train_smaller and test fold memberships

# %%
# %%
# filter to selected training specimens
df = df[df["in_training_set"]].copy()
df

# %%
assert not df["test_fold_id"].isna().any()
df["test_fold_id"].value_counts()

# %%

# %%
train_validation_memberships = helpers.get_all_specimen_cv_fold_info()
train_validation_memberships = train_validation_memberships[
    train_validation_memberships["fold_label"] != "test"
]
print(train_validation_memberships["fold_label"].value_counts())
train_validation_memberships

# %%
for (fold_id, fold_label), grp in train_validation_memberships.groupby(
    ["fold_id", "fold_label"], observed=True
):
    colname = f"{fold_label}_fold_{fold_id}"
    df[colname] = False
    df.loc[
        df["specimen_label"].isin(grp["specimen_label"].tolist()),
        colname,
    ] = True

    print(colname)
    print(df[colname].value_counts())
    print()

# %%

# %%
df

# %%

# %%

# %%
df.groupby(["disease", "study_name"]).size()

# %%

# %% [markdown]
# # EDA - specimen counts

# %%
# Here's who is missing age:
df[df["age_group"].isna()]["disease"].value_counts()

# %%
# Versus total counts
df["disease"].value_counts()

# %%
# <20: only in HIV, Healthy, and Lupus
# 70-80: not in HIV
# 80+: only in Covid and Healthy

df.groupby(["age_group", "disease"]).size()

# %%
# Age medians, ranges, and number of N/As by disease
# TODO: show boxplots too
for disease, disease_df in df.groupby("disease"):
    print(
        f"{disease}: median {disease_df['age'].median()} years old, range {disease_df['age'].min()} - {disease_df['age'].max()}, with {disease_df['age'].isna().sum()} NaNs"
    )

# %%

# %%

# %%

# %%
# Here's who is missing ethnicity:
df[df["ethnicity_condensed"].isna()]["disease"].value_counts()

# %%
# Versus total counts
df["disease"].value_counts()

# %%
# Take note:
df.groupby(["ethnicity_condensed", "disease"]).size()

# %%
# HIV is only African!
df.groupby(["disease", "ethnicity_condensed"]).size()

# %%
# Percentage of diseases (excluding N/As) by ancestry - these are percentages by specimens though
# TODO: show as stacked boxplots
for ethnicity, ethnicity_df in df.groupby("ethnicity_condensed"):
    print(f"{ethnicity}: {ethnicity_df.shape[0]} samples total")
    print(
        ethnicity_df["disease"]
        .value_counts(normalize=True)
        .apply(lambda fraction: f"{fraction*100:0.0f}%")
    )
    print()

# %%

# %%

# %%
# Here's who is missing sex:
df[df["sex"].isna()]["disease"].value_counts()

# %%
# Lupus is very sex imbalanced, as expected
# Surprisignly HIV is sex imbalanced too?
df.groupby(["disease", "sex"]).size()

# %%
# Lupus is very sex imbalanced
df.groupby(["sex", "disease"]).size()

# %%
# Percentage of females (and number of N/As) by disease
# TODO: show as stacked boxplots
for disease, disease_df in df.groupby("disease"):
    n_total = disease_df["sex"].dropna().shape[0]
    if n_total == 0:
        print(f"{disease}: no sex information")
        continue

    n_female = disease_df["sex"].value_counts().loc["F"]
    n_nas = disease_df["sex"].isna().sum()
    assert (
        n_total + n_nas == disease_df.shape[0]
    ), "sanity check: should add up to total shape"
    print(
        f"{disease}: {n_female} female out of {n_total} total (not counting {n_nas} NaNs) = {n_female/n_total*100:0.0f}%."
    )

# %%

# %%
# CMV status for healthy individuals
df[df["disease"] == healthy_label]["disease_subtype"].value_counts()

# %%

# %%
# Time points at the specimen level
for (disease, study_name), disease_df in df.groupby(["disease", "study_name"]):
    print(disease, study_name)
    print(disease_df["specimen_time_point"].value_counts())
    # print(disease_df["specimen_time_point_days"].describe()) # TODO
    print()

# %%
# Disease subtypes
for (disease, study_name), disease_df in df.groupby(["disease", "study_name"]):
    print(disease, study_name)
    print(disease_df["disease_subtype"].value_counts())
    print()

# %%
# Number of specimens per person
# If you see "1 n; 2 m", that means n people had 1 peak specimen each and m people had 2 peak specimens each.
for (disease, study_name), disease_df in df.groupby(["disease", "study_name"]):
    print(disease, study_name)
    print(disease_df["participant_label"].value_counts().value_counts())
    print()

# %%

# %% [markdown]
# # EDA - participant counts
#
# List these instead of specimen counts

# %%
df_by_participant = df.groupby("participant_label").first()
df_by_participant.shape, df.shape

# %%
df_by_participant

# %%
# number of participants by disease
df_by_participant["disease"].value_counts()

# %%
# Age medians, ranges, and number of N/As by disease
# TODO: show as boxplots
for disease, disease_df in df_by_participant.groupby("disease"):
    print(
        f"{disease}: median {disease_df['age'].median()} years old, range {disease_df['age'].min()} - {disease_df['age'].max()}, with {disease_df['age'].isna().sum()} NaNs, total {disease_df.shape[0]} participants"
    )

# %%
# healthy controls - extreme ages
df_by_participant[
    (df_by_participant["age"] >= 80) & (df_by_participant["disease"] == healthy_label)
]["age"]

# %%
# healthy controls - extreme ages
df_by_participant[
    (df_by_participant["age"] >= 75) & (df_by_participant["disease"] == healthy_label)
]["age"]

# %%
# healthy controls - extreme ages
df_by_participant[
    (df_by_participant["age"] >= 70) & (df_by_participant["disease"] == healthy_label)
]["age"]

# %%
# healthy controls - extreme ages
df_by_participant[
    (df_by_participant["age"] < 20) & (df_by_participant["disease"] == healthy_label)
]["age"]

# %%

# %%
# Percentage of diseases (excluding N/As) by ancestry
# TODO: show as stacked boxplots
for ethnicity, ethnicity_df in df_by_participant.groupby("ethnicity_condensed"):
    print(f"{ethnicity}: {ethnicity_df.shape[0]} patients total")
    print(
        ethnicity_df["disease"]
        .value_counts(normalize=True)
        .apply(lambda fraction: f"{fraction*100:0.0f}%")
    )
    print()

# %%
# Percentage of ancestry (excluding N/As) by disease
# TODO: show as stacked boxplots
for disease, disease_df in df_by_participant.groupby("disease"):
    print(
        f"{disease}: {disease_df.shape[0]} patients total, including those without known ancestry info."
    )
    print(
        f"Among the {disease_df['ethnicity_condensed'].dropna().shape[0]} patients without N/As:"
    )
    print(
        disease_df["ethnicity_condensed"]
        .dropna()
        .value_counts(normalize=True)
        .apply(lambda fraction: f"{fraction*100:0.0f}%")
    )
    print()

# %%
# Percentage of ancestry (including N/As) by disease
# TODO: show as stacked boxplots
for disease, disease_df in df_by_participant.groupby("disease"):
    print(
        f"{disease}: {disease_df.shape[0]} patients total, including {disease_df.shape[0] - disease_df['ethnicity_condensed'].dropna().shape[0]} without known ancestry info."
    )
    ethnicity_counts = (
        disease_df["ethnicity_condensed"].dropna().value_counts(normalize=False)
    )
    # add N/A counts
    ethnicity_counts.at["Unknown"] = disease_df["ethnicity_condensed"].isna().sum()
    # normalize to sum to 1
    ethnicity_counts = ethnicity_counts / ethnicity_counts.sum()
    print(ethnicity_counts.apply(lambda fraction: f"{fraction*100:0.0f}%"))
    print()

# %%
# Raw counts of ancestry (including N/As) by disease
for disease, disease_df in df_by_participant.groupby("disease"):
    print(
        f"{disease}: {disease_df.shape[0]} patients total, including {disease_df.shape[0] - disease_df['ethnicity_condensed'].dropna().shape[0]} without known ancestry info."
    )
    ethnicity_counts = (
        disease_df["ethnicity_condensed"].dropna().value_counts(normalize=False)
    )
    # add N/A counts
    ethnicity_counts.at["Unknown"] = disease_df["ethnicity_condensed"].isna().sum()
    print(ethnicity_counts)
    print()

# %%

# %%
# Percentage of females (and number of N/As) by disease
# TODO: show as stacked boxplots
for disease, disease_df in df_by_participant.groupby("disease"):
    n_total = disease_df["sex"].dropna().shape[0]
    if n_total == 0:
        print(f"{disease}: no sex information")
        continue

    n_female = disease_df["sex"].value_counts().loc["F"]
    n_nas = disease_df["sex"].isna().sum()
    assert (
        n_total + n_nas == disease_df.shape[0]
    ), "sanity check: should add up to total shape"
    print(
        f"{disease}: {n_female} female out of {n_total} total (not counting {n_nas} NaNs) = {n_female/n_total*100:0.0f}%."
    )

# %%

# %% [markdown]
# ## Participant counts within each batch (each `study_name`)

# %%
# Age medians, ranges, and number of N/As by study name (disease batch)
# TODO: show as boxplots
for (disease, study_name), disease_df in df_by_participant.groupby(
    ["disease", "study_name"]
):
    print(
        f"{disease}, {study_name}: median {disease_df['age'].median()} years old, range {disease_df['age'].min()} - {disease_df['age'].max()}, with {disease_df['age'].isna().sum()} NaNs, total {disease_df.shape[0]} participants"
    )

# %%
# Percentage of batches (excluding N/As) by ancestry
# TODO: show as stacked boxplots
for ethnicity, ethnicity_df in df_by_participant.groupby("ethnicity_condensed"):
    print(f"{ethnicity}: {ethnicity_df.shape[0]} patients total")
    print(
        ethnicity_df["study_name"]
        .value_counts(normalize=True)
        .apply(lambda fraction: f"{fraction*100:0.0f}%")
    )
    print()


# %%
def pretty_print_value_counts(ser):
    return ", ".join(
        [f"{value} {label}" for label, value in zip(ser.index, ser.values)]
    )


# %%
# Percentage of ancestry (including N/As) by disease and study name
# TODO: show as stacked boxplots
for (disease, study_name), disease_df in df_by_participant.groupby(
    ["disease", "study_name"]
):
    print(f"{disease}, {study_name}:")
    print(
        f"{disease_df.shape[0]} patients total, including {disease_df.shape[0] - disease_df['ethnicity_condensed'].dropna().shape[0]} without known ancestry info."
    )
    ethnicity_counts = (
        disease_df["ethnicity_condensed"].dropna().value_counts(normalize=False)
    )
    # add N/A counts
    ethnicity_counts.at["unknown"] = disease_df["ethnicity_condensed"].isna().sum()
    # normalize to sum to 1
    ethnicity_counts = ethnicity_counts / ethnicity_counts.sum()
    print(ethnicity_counts.apply(lambda fraction: f"{fraction*100:0.0f}%"))
    print(
        pretty_print_value_counts(
            ethnicity_counts.apply(lambda fraction: f"{fraction*100:0.0f}%")
        )
    )
    print()

# %%
# Percentage of females (and number of N/As) by disease
# TODO: show as stacked boxplots
for (disease, study_name), disease_df in df_by_participant.groupby(
    ["disease", "study_name"]
):
    n_total = disease_df["sex"].dropna().shape[0]
    if n_total == 0:
        print(f"{disease}, {study_name}: no sex information")
        continue

    n_female = (
        disease_df["sex"].value_counts().reindex(["M", "F"], fill_value=0).loc["F"]
    )
    n_nas = disease_df["sex"].isna().sum()
    assert (
        n_total + n_nas == disease_df.shape[0]
    ), "sanity check: should add up to total shape"
    print(
        f"{disease}, {study_name}: {n_female} female out of {n_total} total (not counting {n_nas} NaNs) = {n_female/n_total*100:0.0f}%."
    )

# %%

# %%
# Time points. Ignore this - makes more sense to do at the specimen level.
for (disease, study_name), disease_df in df_by_participant.groupby(
    ["disease", "study_name"]
):
    print(disease, study_name)
    print(disease_df["specimen_time_point"].value_counts())
    # print(disease_df["specimen_time_point_days"].describe()) # TODO
    print()

# %%

# %%
# Disease subtypes
# (Look at specimen level instead?)
for (disease, study_name), disease_df in df_by_participant.groupby(
    ["disease", "study_name"]
):
    print(disease, study_name)
    print(disease_df["disease_subtype"].value_counts(normalize=False))
    print(
        disease_df["disease_subtype"]
        .value_counts(normalize=True)
        .apply(lambda fraction: f"{fraction*100:0.0f}%")
    )
    print()

# %%

# %% [markdown]
# # Subselect for each task

# %%
assert not df["specimen_label"].duplicated().any()
df = df.set_index("specimen_label")
df

# %%
df["age_group"].value_counts()

# %%
df["age"].describe()

# %%
df[df["age"] >= 70]["age"].sort_values()

# %%
df[df["age"] >= 80]["age"]

# %%
df[df["age"] >= 70]["disease"].value_counts()

# %%
df[df["age"] >= 80]["disease"].value_counts()

# %%
df[df["disease"] == healthy_label]["age_group"].value_counts()

# %%

# %%
print(df.shape)

# In "age_group" column, we have removed extreme ages with small sample size.
df_age = df.dropna(subset=["age_group"]).copy()
print(df_age.shape)

# %%
df_age["age_group"].value_counts()

# %%
assert not df_age["age_group_binary"].isna().any()
df_age["age_group_binary"].value_counts()

# %%
assert not df_age["age_group_pediatric"].isna().any()
df_age["age_group_pediatric"].value_counts()

# %%
df_age["disease"].value_counts()

# %%

# %%
df_age_healthy_only = df_age[df_age["disease"] == healthy_label]
df_age_healthy_only = df_age_healthy_only.dropna(subset=["age_group"])
df.shape, df_age.shape, df_age_healthy_only.shape

# %%
df_age_healthy_only["disease"].value_counts()

# %%
df_age_healthy_only["disease_subtype"].value_counts()

# %%
df_age_healthy_only["age_group"].value_counts()

# %%
assert not df_age_healthy_only["age_group_binary"].isna().any()
df_age_healthy_only["age_group_binary"].value_counts()

# %%
assert not df_age_healthy_only["age_group_pediatric"].isna().any()
df_age_healthy_only["age_group_pediatric"].value_counts()

# %%
df_age_healthy_only["age"].hist(bins=20)

# %%

# %%
df_sex = df.copy()
df_sex = df_sex.dropna(subset=["sex"])
df.shape, df_sex.shape

# %%
df_sex["disease"].value_counts()

# %%
df_sex["sex"].value_counts()

# %%

# %%
df_sex_healthy_only = df_sex[df_sex["disease"] == healthy_label]
df_sex_healthy_only = df_sex_healthy_only.dropna(subset=["sex"])
df.shape, df_sex.shape, df_sex_healthy_only.shape

# %%
df_sex_healthy_only["disease"].value_counts()

# %%
df_sex_healthy_only["disease_subtype"].value_counts()

# %%
df_sex_healthy_only["sex"].value_counts()

# %%

# %%
df_ethnicity = df.copy()
df_ethnicity = df_ethnicity.dropna(subset=["ethnicity_condensed"])
df.shape, df_ethnicity.shape

# %%
df_ethnicity["disease"].value_counts()

# %%
df_ethnicity["ethnicity_condensed"].value_counts()

# %%

# %%
df_ethnicity_healthy_only = df_ethnicity[df_ethnicity["disease"] == healthy_label]
df_ethnicity_healthy_only = df_ethnicity_healthy_only.dropna(
    subset=["ethnicity_condensed"]
)
df.shape, df_ethnicity.shape, df_ethnicity_healthy_only.shape

# %%
df_ethnicity_healthy_only["disease"].value_counts()

# %%
df_ethnicity_healthy_only["disease_subtype"].value_counts()

# %%
df_ethnicity_healthy_only["ethnicity_condensed"].value_counts()

# %%

# %%

# %%
# Get subset with age + sex + ethnicity defined
df_all = df.loc[
    df_age.index.intersection(df_sex.index).intersection(df_ethnicity.index)
].copy()
df.shape, df_all.shape

# %%
df_all["disease"].value_counts()

# %%
df_all

# %%
df_all["participant_label"].nunique()

# %%
df_all.groupby("disease")["participant_label"].nunique()

# %%
df_all.reset_index()["specimen_label"].nunique()

# %%
df_all.reset_index().groupby("disease")["specimen_label"].nunique()

# %%

# %%
df_all_healthy_only = df.loc[
    df_age_healthy_only.index.intersection(df_sex_healthy_only.index).intersection(
        df_ethnicity_healthy_only.index
    )
].copy()
df.shape, df_all.shape, df_all_healthy_only.shape

# %%
df_all_healthy_only["disease"].value_counts()

# %%
df_all_healthy_only

# %%

# %%
df_all_healthy_only["disease_subtype"].value_counts()

# %%
# create a "defined CMV status" subset of "healthy controls with known age/sex/ethnicity" dataset
df_cmv_status = df_all_healthy_only.copy()
print(df_cmv_status["cmv"].isna().sum())

# %%
df_cmv_status = df_cmv_status.dropna(subset=["cmv"])
df_cmv_status["cmv"].value_counts()

# %%

# %%
for name, partdf in zip(
    [
        #         "age_group",
        #         "sex",
        #         "ethnicity_condensed",
        "all",
        #         "age_group_healthy_only",
        #         "sex_healthy_only",
        #         "ethnicity_condensed_healthy_only",
        #         "all_healthy_only",
        "cmv",
    ],
    [
        #         df_age,
        #         df_sex,
        #         df_ethnicity,
        df_all,
        #         df_age_healthy_only,
        #         df_sex_healthy_only,
        #         df_ethnicity_healthy_only,
        #         df_all_healthy_only,
        df_cmv_status,
    ],
):
    fname_out = (
        config.paths.dataset_specific_metadata
        / f"computed_metadata_for_confounder_model.{name}.tsv"
    )
    print(name, fname_out)
    partdf.to_csv(
        fname_out,
        sep="\t",
    )

# %%

# %%
# After filters, how many remaining participants
df_all["participant_label"].nunique(), df_all_healthy_only[
    "participant_label"
].nunique()

# %%
# After filters, how many remaining specimens
df_all.reset_index()["specimen_label"].nunique(), df_all_healthy_only.reset_index()[
    "specimen_label"
].nunique()


# %%

# %%
