# -*- coding: utf-8 -*-
# %% [markdown]
# # Assemble metadata for ETL
#
# We want to put together a participant-label + specimen-label whitelist here, along with the `disease` and `disease_subtype` annotations
#
# Then in `etl.ipynb` we'll simply inner-merge against this, to choose only the requested specimens and to fill in `disease` and `disease_subtype`.

# %%

# %%
import numpy as np
import pandas as pd
from typing import Union

# %%
from malid import config
from malid.datamodels import healthy_label

# %%

# %% [markdown]
# ## Covid

# %%
# Covid-Stanford
covid_stanford_specimens = (
    pd.read_csv(config.paths.metadata_dir / "covid19_stanford.specimens.tsv", sep="\t")
    .rename(
        columns={
            "time_point": "specimen_time_point_override",
            "Age": "age",
            "Gender": "sex",
            "Race": "ethnicity",
            "Ethnicity": "hispanic",
        }
    )
    .assign(study_name="Covid19-Stanford", has_BCR=True, has_TCR=True)
)

covid_stanford_specimens["sex"] = covid_stanford_specimens["sex"].replace(
    {"Male": "M", "Female": "F"}
)


covid_stanford_specimens

# %%
covid_stanford_specimens.groupby(["ethnicity", "hispanic"]).size()

# %%
covid_stanford_specimens["ethnicity"].isna().value_counts()

# %%
covid_stanford_specimens["hispanic"].isna().value_counts()

# %%
# combine the two fields
covid_stanford_specimens["ethnicity"] = (
    covid_stanford_specimens["ethnicity"] + " - " + covid_stanford_specimens["hispanic"]
)
covid_stanford_specimens["ethnicity"].value_counts()

# %%

# %%
covid_seattle_metadata = pd.read_csv(
    config.paths.metadata_dir / "covid19_seattle_metadata.csv"
).assign(disease="Covid19", study_name="Covid19-Seattle", has_BCR=True, has_TCR=False)
covid_seattle_metadata["disease_subtype"] = (
    "Covid19 - " + covid_seattle_metadata["description"]
)
covid_seattle_metadata

# %%

# %%
# specimen metadata, for covid original Cell Host and Microbe PBMCs
# only keep these samples:
covid_specimen_metadata = pd.read_csv(
    config.paths.metadata_dir / "covid19_metadata_serology.csv"
)[
    [
        "participant_label",
        "specimen_label",
        "seroconversion_IgG",
        "days_post_symptom_onset",
    ]
]

# here is the list of the ones we actually have though, which doesn't include three from above - filter those out:
covid_specimen_metadata = pd.merge(
    covid_specimen_metadata,
    pd.read_csv(config.paths.metadata_dir / "covid_participants.txt"),
    how="inner",
    on="participant_label",
).assign(disease="Covid19", study_name="Covid19-buffycoat", has_BCR=True, has_TCR=True)

# add demographics
covid_specimen_demographics = pd.read_csv(
    config.paths.metadata_dir / "covid_cell_host_and_microbe_demographics.tsv", sep="\t"
)
covid_specimen_metadata = pd.merge(
    covid_specimen_metadata,
    covid_specimen_demographics,
    how="inner",
    on="participant_label",
    validate="m:1",
)

covid_specimen_metadata["disease_subtype"] = (
    "Covid19 - "
    + covid_specimen_metadata["seroconversion_IgG"].map(
        {"Yes": "Sero-positive", "No": "Sero-negative"}
    )
    + " ("
    + covid_specimen_metadata["disease_severity"]
    + ")"
)
covid_specimen_metadata["specimen_time_point_override"] = (
    covid_specimen_metadata["days_post_symptom_onset"].astype(str) + " days"
)
covid_specimen_metadata

# %%
covid_specimen_metadata["disease_subtype"].value_counts()

# %%


# %%

# %% [markdown]
# ## HIV

# %%
hiv_specimens = pd.read_csv(
    config.paths.metadata_dir / "hiv_cohort.specimens.tsv", sep="\t"
).assign(disease="HIV", study_name="HIV", has_BCR=True, has_TCR=True)

assert not hiv_specimens["disease_subtype"].isna().any()
hiv_specimens.loc[
    hiv_specimens["disease_subtype"] == "HIV Negative", "disease"
] = healthy_label
hiv_specimens.loc[
    hiv_specimens["disease_subtype"] == "HIV Negative", "disease_subtype"
] = f"{healthy_label} - HIV Negative"

# HIV: allow certain runs only
hiv_specimens["hiv_run_filter"] = True

assert hiv_specimens["ethnicity"].isna().all()
assert not hiv_specimens["description"].isna().any()


hiv_specimens["ethnicity"] = hiv_specimens["description"].replace(
    {
        "Location: Malawi": "African",
        "Location: South Africa": "African",
        "Location: USA": "Unknown",
        "Location: Tanzania": "African",
        "Location: UK": "Unknown",
    }
)
assert not hiv_specimens["ethnicity"].isna().any()

hiv_specimens

# %%
hiv_specimens.groupby(["disease", "disease_subtype", "ethnicity", "description"]).size()

# %%

# %% [markdown]
# ## Healthy donors

# %%
healthy_specimens = pd.read_csv(
    config.paths.metadata_dir / "healthy_human_controls.specimens.tsv",
    sep="\t",
).assign(
    disease=healthy_label,
    study_name="Healthy-StanfordBloodCenter",
    has_BCR=True,
    has_TCR=True,
)
healthy_specimens

# %%
cmv_status = healthy_specimens["disease_subtype"].str.extract("CMV = ([NP])")[0]
cmv_status

# %%
healthy_specimens.loc[cmv_status.isna()]

# %%
cmv_status = cmv_status.replace({"N": "CMV-", "P": "CMV+"})
cmv_status.value_counts()

# %%
# "specimens with defined CMV status" is a subset of "healthy controls with known age/sex/ethnicity" dataset
# All other healthy control subsets and all other cohorts will have cmv status NaN
healthy_specimens["symptoms_cmv"] = cmv_status
healthy_specimens["disease_subtype"] = (
    healthy_label + " - " + cmv_status.fillna("CMV Unknown")
)
healthy_specimens

# %%
healthy_specimens["ethnicity"].isna().any()

# %%
healthy_specimens["ethnicity"].fillna(-1, inplace=True)
assert not healthy_specimens["ethnicity"].isna().any()

# %%
healthy_specimens["ethnicity"] = healthy_specimens["ethnicity"].astype(int)
healthy_specimens["ethnicity"].value_counts()

# %%
ethnicity_map = {
    -1: "Unknown",
    1: "Caucasian",
    3: "Asian",
    6: "India/Arabia/Iran",
    5: "Central/South American",
    15: "Caucasian + Central/South American",
    2: "African American",
    14: "Caucasian + Native American",
}

# %%
# confirm all are in keys
assert all(k in ethnicity_map.keys() for k in healthy_specimens["ethnicity"].values)

# %%
healthy_specimens["ethnicity"] = healthy_specimens["ethnicity"].replace(ethnicity_map)
healthy_specimens["ethnicity"].value_counts()

# %%
healthy_specimens

# %%

# %%
# Also mark which healthy specimens were involved in the resequencing experiments (M477/M482, M479/M484 - IBD runs below)
# These will have multiple amplification labels.
hhc_resequencing = pd.concat(
    [
        pd.read_csv(
            config.paths.metadata_dir
            / "ibd_pre_pandemic_and_resequencing_some_old_hhcs.M477_M482.specimens.tsv",
            sep="\t",
        ),
        pd.read_csv(
            config.paths.metadata_dir
            / "ibd_post_pandemic_and_resequencing_some_more_old_hhcs.M479_M484.specimens.tsv",
            sep="\t",
        ),
    ],
    axis=0,
)[["specimen_label"]].assign(symptoms_healthy_in_resequencing_experiment=True)
# symptoms_healthy_in_resequencing_experiment is True for the matching ones, NaN for all others (since we will combine with many other healthy cohorts)
# (Remember that each row here is a specimen, so this describes both the old and new copy of the specimen. Sequences belonging to each copy can be delineated by the amplification label.)
healthy_specimens = pd.merge(
    healthy_specimens, hhc_resequencing, how="left", on="specimen_label", validate="1:1"
)
print(healthy_specimens["symptoms_healthy_in_resequencing_experiment"].value_counts())
print(
    healthy_specimens["symptoms_healthy_in_resequencing_experiment"]
    .isna()
    .value_counts()
)

# %%
# Also adjust study_name for these specimens.
healthy_specimens.loc[
    healthy_specimens["symptoms_healthy_in_resequencing_experiment"].fillna(False),
    "study_name",
] = "Healthy-StanfordBloodCenter_included-in-resequencing"
healthy_specimens["study_name"].value_counts()

# %%
healthy_specimens[
    healthy_specimens["symptoms_healthy_in_resequencing_experiment"].fillna(False)
]["symptoms_cmv"].value_counts()

# %%
healthy_specimens

# %%

# %% [markdown]
# ## Lupus

# %%
# Lupus - cases and healthy controls
# assign disease and disease_subtype
lupus_specimens = pd.read_csv(
    config.paths.metadata_dir / "lupus_m281redo.specimens.tsv",
    sep="\t",
)
lupus_specimens = lupus_specimens.assign(
    disease="Lupus", study_name="Lupus", has_BCR=True, has_TCR=False
)

lupus_specimens.loc[
    lupus_specimens["disease_subtype"].str.contains(healthy_label), "disease"
] = healthy_label
lupus_specimens.loc[
    lupus_specimens["disease_subtype"].str.contains(healthy_label),
    "disease_subtype",
] = f"{healthy_label} - SLE Negative"

lupus_specimens

# %%
# Add clinical symptom metadata
lupus_specimen_clinical_metadata = pd.read_csv(
    config.paths.metadata_dir / "m281_clinical_metadata_extract.csv",
).set_index("specimen_label")
lupus_specimen_clinical_metadata.rename(
    columns=lambda col: f"symptoms_Lupus_{col}", inplace=True
)
lupus_specimens = pd.merge(
    lupus_specimens,
    lupus_specimen_clinical_metadata,
    how="left",
    left_on="specimen_label",
    right_index=True,
    validate="1:1",
)
lupus_specimens

# %%
# Pediatric lupus
pediatric_lupus_specimens = (
    pd.read_csv(
        config.paths.metadata_dir / "lupus_M447_M448.specimens.tsv",
        sep="\t",
    )
    .rename(columns={"participant_description": "disease_subtype"})
    .assign(study_name="Lupus Pediatric", has_BCR=True, has_TCR=True)
)
pediatric_lupus_specimens["disease"] = pediatric_lupus_specimens["disease"].replace(
    {"SLE": "Lupus"}
)
assert (pediatric_lupus_specimens["disease"] == "Lupus").all()
pediatric_lupus_specimens

# %%
# New adult lupus RNA
new_lupus_rna_specimens = (
    pd.read_csv(
        config.paths.metadata_dir / "adult_lupus_rna_M454_M455.specimens.tsv",
        sep="\t",
    )
    .rename(columns={"participant_description": "disease_subtype"})
    .assign(study_name="New Lupus RNA", has_BCR=True, has_TCR=True)
)
new_lupus_rna_specimens["disease"] = new_lupus_rna_specimens["disease"].replace(
    {"SLE": "Lupus"}
)
new_lupus_rna_specimens = new_lupus_rna_specimens[
    new_lupus_rna_specimens["disease"].isin(["Lupus", healthy_label])
]
new_lupus_rna_specimens

# %%
# New adult lupus Paxgene
new_lupus_paxgene_specimens = (
    pd.read_csv(
        config.paths.metadata_dir / "adult_lupus_paxgene_M456_M457.specimens.tsv",
        sep="\t",
    )
    .rename(columns={"participant_description": "disease_subtype"})
    .assign(study_name="New Lupus Paxgene", has_BCR=True, has_TCR=True)
)
new_lupus_paxgene_specimens["disease"] = new_lupus_paxgene_specimens["disease"].replace(
    {"SLE": "Lupus"}
)
new_lupus_paxgene_specimens = new_lupus_paxgene_specimens[
    new_lupus_paxgene_specimens["disease"].isin(["Lupus", healthy_label])
]
new_lupus_paxgene_specimens

# %%

# %% [markdown]
# ## Healthy children

# %%
healthy_children = (
    pd.read_csv(
        config.paths.metadata_dir / "healthy_children_M464_M463.specimens.tsv",
        sep="\t",
    )
    .rename(columns={"participant_description": "disease_subtype"})
    .assign(study_name="healthy_children", has_BCR=True, has_TCR=True)
)
assert all(healthy_children["disease"] == "Healthy")
healthy_children["disease"] = healthy_label
healthy_children["disease_subtype"] = f"{healthy_label} (children)"
healthy_children

# %%


# %% [markdown]
# ## IBD pre-pandemic

# %%
# Yoni samples pre-pandemic
ibd_pre_pandemic_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "ibd_pre_pandemic_and_resequencing_some_old_hhcs.M477_M482.specimens.without_old_hhcs.tsv",
    sep="\t",
).assign(study_name="IBD pre-pandemic Yoni", has_BCR=True, has_TCR=True)

# Notes:
# 1. This run also resequenced some old HHCs, but we have excluded those from this spreadsheet because they are accounted for elsewhere above.
#
# 2. The TCR specimens were split into a CD4 and a CD8 cell fraction.
# At import time, this spreadsheet has rows with specimen_label "M477-S001" (B cells), "M477-S001_CD4" (CD4 T cells), "M477-S001_CD8" (CD8 T cells), and so on for the other samples.
# But really these are all the same specimen, just split into different cell fractions.
# We will mark these with equivalent specimen_label and amplification_label but separate replicate_label
# Then the ETL notebook will merge these under the no-suffix specimen labels, e.g. "M477-S001".
# Note that we will have duplicate metadata entries, with the CD4 or CD8 suffix inside the replicate_label_override.

suffix_to_cell_type_map = {"_CD8": "CD8 T", "_CD4": "CD4 T"}

# - specimen_label: M477-S008, M477-S008_CD4, M477-S008_CD8
# (stays as is; must merge against original specimen_label in raw participant table.)

# - replicate_label_override: M477-S008, M477-S008_CD4, M477-S008_CD8
ibd_pre_pandemic_specimens["replicate_label_override"] = ibd_pre_pandemic_specimens[
    "specimen_label"
].copy()

# - specimen_label_override: M477-S008, M477-S008, M477-S008
ibd_pre_pandemic_specimens["specimen_label_override"] = ibd_pre_pandemic_specimens[
    "specimen_label"
].copy()
for suffix in suffix_to_cell_type_map.keys():
    # remove each possible suffix
    ibd_pre_pandemic_specimens["specimen_label_override"] = ibd_pre_pandemic_specimens[
        "specimen_label_override"
    ].str.removesuffix(suffix)

# - amplification_label_override: M477-S008, M477-S008, M477-S008
ibd_pre_pandemic_specimens["amplification_label_override"] = ibd_pre_pandemic_specimens[
    "specimen_label_override"
].copy()


# - cell_type
def mark_cell_type(specimen_label) -> Union[str, float]:
    for suffix, cell_type in suffix_to_cell_type_map.items():
        if specimen_label.endswith(suffix):
            # Mark cell type (optional column)
            return cell_type
    return np.nan  # float


ibd_pre_pandemic_specimens["cell_type"] = ibd_pre_pandemic_specimens[
    "replicate_label_override"
].apply(mark_cell_type)

# The consequence of this approach is that the output metadata file will not have one entry per specimen_label_override.
# (It will have one entry per original specimen_label, but each row represents a replicate/fraction that will be merged into the parent specimen)

print(
    ibd_pre_pandemic_specimens.groupby("specimen_label_override")
    .head(n=1)["disease"]
    .value_counts()
)

# Load clinical symptom metadata
ibd_specimen_clinical_metadata = pd.read_csv(
    config.paths.metadata_dir / "all_annot_m477_ibd.tsv", sep="\t"
).set_index("specimen_label")

# First, add the compiled disease subtype info
ibd_pre_pandemic_specimens = pd.merge(
    ibd_pre_pandemic_specimens,
    ibd_specimen_clinical_metadata["disease_subtype"],
    how="left",
    left_on="specimen_label_override",
    right_index=True,
    validate="m:1",
)
assert not ibd_pre_pandemic_specimens["disease_subtype"].isna().any()

# Mark healthy controls as a separate disease subtype
ibd_pre_pandemic_specimens.loc[
    ibd_pre_pandemic_specimens["disease"] == healthy_label,
    "disease_subtype",
] = f"{healthy_label} - IBD Negative"

# Drop the "IND" indeterminate samples.
ibd_pre_pandemic_specimens = ibd_pre_pandemic_specimens[
    ibd_pre_pandemic_specimens["disease"] != "IND"
]

# Now add the other clinical symptom metadata:
ibd_specimen_clinical_metadata_symptom_columns = ibd_specimen_clinical_metadata.drop(
    columns=[
        "Subject ID Yoni",
        "participant_label",
        "FR1",
        "IgA",
        "IgD",
        "IgE",
        "IgG",
        "IgM",
        "Disease",
        "Sex",
        "disease_subtype",
    ]
).rename(columns=lambda col: f"symptoms_IBD_{col}")
ibd_pre_pandemic_specimens = pd.merge(
    ibd_pre_pandemic_specimens,
    ibd_specimen_clinical_metadata_symptom_columns,
    how="left",
    left_on="specimen_label_override",
    right_index=True,
    validate="m:1",
)

ibd_pre_pandemic_specimens

# %%

# %% [markdown]
# ## IBD post-pandemic

# %%
# John Gubatan samples post-pandemic
# Note: This run also resequenced some old HHCs, but we have excluded those from this spreadsheet because they are accounted for elsewhere above.

ibd_post_pandemic_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "ibd_post_pandemic_and_resequencing_some_more_old_hhcs.M479_M484.specimens.without_old_hhcs.tsv",
    sep="\t",
).assign(study_name="IBD post-pandemic Gubatan", has_BCR=True, has_TCR=True)

print(ibd_post_pandemic_specimens["disease"].value_counts())
print(ibd_post_pandemic_specimens["participant_description"].value_counts())
ibd_post_pandemic_specimens[
    "symptoms_IBD_Disease location"
] = ibd_post_pandemic_specimens["participant_description"].str.removeprefix(
    "IBD cohort from Gubatan post-pandemic: "
)
print(ibd_post_pandemic_specimens["symptoms_IBD_Disease location"].value_counts())

print(ibd_post_pandemic_specimens["specimen_time_point"].value_counts())
timepoints_extracted = ibd_post_pandemic_specimens["specimen_time_point"].map(
    {"84 days": 84, "00:00:00": 0}
)
assert not timepoints_extracted.isna().any()
ibd_post_pandemic_specimens["disease_subtype"] = (
    ibd_post_pandemic_specimens["disease"]
    + " - day "
    + timepoints_extracted.astype(str)
)
assert not ibd_post_pandemic_specimens["disease_subtype"].isna().any()
print(ibd_post_pandemic_specimens["disease_subtype"].value_counts())

# no HHCs in this run; otherwise would need to mark HHCs as different disease subtype
assert (ibd_post_pandemic_specimens["disease"] != healthy_label).all()

ibd_post_pandemic_specimens

# %%

# %% [markdown]
# ## Flu vaccine

# %%
# UPenn influenza vaccine response, 2021-22 vaccine year
flu_vaccine_specimens = pd.read_csv(
    config.paths.metadata_dir / "upenn_flu_vaccine.M433_M435_M444.specimens.tsv",
    sep="\t",
).assign(
    study_name="Flu vaccine UPenn 2021",
    disease="Influenza",
    disease_subtype="Influenza vaccine 2021",
    has_BCR=True,
    has_TCR=True,
)
print(flu_vaccine_specimens["specimen_time_point"].value_counts())
timepoints_extracted = flu_vaccine_specimens["specimen_time_point"].map(
    {"28 days": 28, "00:00:00": 0, "7 days": 7, "90 days": 90}
)
assert not timepoints_extracted.isna().any()
flu_vaccine_specimens["disease_subtype"] = (
    flu_vaccine_specimens["disease_subtype"]
    + " - day "
    + timepoints_extracted.astype(str)
)
flu_vaccine_specimens

# %%
flu_vaccine_demographics = pd.read_csv(
    config.paths.metadata_dir / "flu_vaccine.demographics.csv"
).rename(columns={"participant": "participant_alternative_label"})
# participant column has extra zero padding
flu_vaccine_demographics["participant_alternative_label"] = flu_vaccine_demographics[
    "participant_alternative_label"
].str.replace("A21-0", "A21-")
assert not flu_vaccine_demographics["participant_alternative_label"].duplicated().any()
flu_vaccine_demographics.head()

# %%
flu_vaccine_specimens.shape

# %%
flu_vaccine_specimens = pd.merge(
    flu_vaccine_specimens.drop(columns=["sex", "age"]),
    flu_vaccine_demographics,
    on="participant_alternative_label",
    how="left",
    validate="m:1",
)
flu_vaccine_specimens.shape

# %%
flu_vaccine_specimens["age"].isna().value_counts()

# %%
flu_vaccine_specimens["sex"].isna().value_counts()

# %%
flu_vaccine_specimens

# %%

# %%

# %% [markdown]
# ## Type 1 and Type 2 Diabetes

# %%
diabetes_samples = pd.read_csv(
    config.paths.metadata_dir / "diabetes_biobank.M491_M492_M493.specimens.tsv",
    sep="\t",
).assign(
    study_name="Diabetes biobank",
    # disease="Influenza",
    # disease_subtype="Influenza vaccine 2021",
    has_BCR=True,
    has_TCR=True,
)
print(diabetes_samples["disease"].value_counts())
print()

diabetes_samples = diabetes_samples.loc[
    diabetes_samples["disease"].isin(["T1D", "T2D", "Healthy/Background"])
]
print(diabetes_samples["disease"].value_counts())
print()

# Mark healthy controls as a separate disease subtype
diabetes_samples["disease_subtype"] = diabetes_samples["disease"]
diabetes_samples.loc[
    diabetes_samples["disease"] == healthy_label,
    "disease_subtype",
] = f"{healthy_label} - Diabetes Negative"


print(diabetes_samples["participant_description"].value_counts())
print()
# each one is marked "adult" or "pediatric":
assert (
    diabetes_samples["participant_description"].str.contains("adult")
    | diabetes_samples["participant_description"].str.contains("pediatric")
).all()

age_group_pediatric_extracted = (
    diabetes_samples["participant_description"]
    .str.contains("pediatric")
    .map({True: "pediatric", False: "adult"})
)
diabetes_samples["disease_subtype"] = (
    diabetes_samples["disease_subtype"] + " - " + age_group_pediatric_extracted
)
print(diabetes_samples["disease_subtype"].value_counts())
diabetes_samples

# %%

# %%
# Add age metadata

# %%
assert diabetes_samples["age"].isna().all()

# %%
diabetes_demographics = pd.read_csv(
    config.paths.metadata_dir / "diabetes_ages.csv",
)
diabetes_demographics = (
    diabetes_demographics.assign(
        participant_alternative_label="35453-"
        + diabetes_demographics["PI"].astype(str).str.zfill(3)
    )
    .rename(columns={"Age": "age"})
    .drop(columns=["PI"])
)
diabetes_demographics

# %%
assert not diabetes_demographics["participant_alternative_label"].duplicated().any()

# %%
diabetes_demographics.shape

# %%

# %%
diabetes_samples["participant_alternative_label"].value_counts().loc[lambda x: x > 1]

# %%
# Some participants have multiple samples: diabetes_samples['participant_alternative_label'].value_counts().loc[lambda x: x>1]
# Looking into them:
#
# Participant 276 has two samples collected almost at same time = no age difference (keep)
#
# 316 has samples collected over two years apart = age difference (TODO: delete)
# 275 has two samples collected exactly a year apart  = age difference (TODO: delete)
# (Our metadata is at the specimen not patient level, but for modeling simplicity, let's ignore patients with multiple specimens collected over a span of ages for now.)
#
# 202 has no sample collection dates
# 274 has sample collection date for only one sample (TODO: delete age)
diabetes_demographics = diabetes_demographics[
    ~diabetes_demographics["participant_alternative_label"].isin(
        ["35453-274", "35453-202", "35453-316", "35453-275"]
    )
].copy()

# %%
diabetes_demographics.shape

# %%

# %%
diabetes_samples.shape

# %%
diabetes_samples = pd.merge(
    diabetes_samples.drop(columns="age"),
    diabetes_demographics,
    on="participant_alternative_label",
    validate="m:1",
    how="left",
)
diabetes_samples

# %%
diabetes_samples["age"].isna().value_counts()

# %%
# Check ages. Notice that T1D pediatric has an age going up to 24. That's unexpected
for key, grp in diabetes_samples.groupby("disease_subtype"):
    print(key)
    print(grp["age"].describe()[["count", "min", "max"]])
    print()

# %%
# Several "pediatric" have age 18+
# We'll have to set the age_group_pediatric column first based on real age if available, then fillna based on the subtype
tmp = (
    diabetes_samples[diabetes_samples["disease_subtype"] == "T1D - pediatric"]
    .dropna(subset="age")
    .sort_values("age")
)
tmp[tmp["age"] >= 18]

# %%

# %% [markdown]
# # Combine

# %%
# combine
specimens_to_keep = pd.concat(
    [
        covid_stanford_specimens,
        covid_seattle_metadata,
        covid_specimen_metadata,
        hiv_specimens,
        healthy_specimens,
        lupus_specimens,
        pediatric_lupus_specimens,
        new_lupus_rna_specimens,
        new_lupus_paxgene_specimens,
        healthy_children,
        ibd_pre_pandemic_specimens,
        ibd_post_pandemic_specimens,
        flu_vaccine_specimens,
        diabetes_samples,
    ],
    axis=0,
)
specimens_to_keep

# %%
specimens_to_keep.columns

# %%
# If these optional columns don't exist, add them:
# - override columns
override_columns = [
    "participant_label_override",
    "specimen_label_override",
    "specimen_time_point_override",
    "amplification_label_override",
    "replicate_label_override",
]
# - cell_type
specimens_to_keep = specimens_to_keep.reindex(
    columns=set(specimens_to_keep.columns) | set(override_columns) | set({"cell_type"}),
    fill_value=np.nan,
)

# %%
# View total number of unique participants (with overrides applied)
specimens_to_keep["participant_label_override"].fillna(
    specimens_to_keep["participant_label"]
).nunique()

# %%
# Subset
specimens_to_keep = specimens_to_keep[
    [
        "participant_label",
        "specimen_label",
        "disease",
        "disease_subtype",
        "hiv_run_filter",
        "age",
        "ethnicity",
        "sex",
        "study_name",  # a batch ID
        "has_BCR",
        "has_TCR",
        "cell_type",
    ]
    + override_columns
    + sorted(
        list(
            specimens_to_keep.columns[
                specimens_to_keep.columns.str.startswith("symptoms_")
            ]
        )
    )
].copy()
specimens_to_keep

# %%
# make sure key columns have no NaNs
assert (
    not specimens_to_keep[
        [
            "participant_label",
            "specimen_label",
            "disease",
            "disease_subtype",
            "study_name",
            "has_BCR",
            "has_TCR",
        ]
    ]
    .isna()
    .any()
    .any()
)

# %%
# Note: These counts are not number of specimens; we need to dedupe by specimen_label_override to get that.
specimens_to_keep["disease"].value_counts()

# %%
# Dedupe by specimen_label_override
specimens_to_keep_with_override_applied_and_dedupe_by_specimen = (
    specimens_to_keep.copy()
)
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    "specimen_label"
] = specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    "specimen_label_override"
].fillna(
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen["specimen_label"]
)
specimens_to_keep_with_override_applied_and_dedupe_by_specimen = (
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen.groupby(
        "specimen_label"
    )
    .head(n=1)
    .drop(
        columns=[
            "specimen_label_override",
            "amplification_label_override",
            "replicate_label_override",
        ]
    )
)
assert (
    not specimens_to_keep_with_override_applied_and_dedupe_by_specimen["specimen_label"]
    .duplicated()
    .any()
)

print(
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen.shape,
    specimens_to_keep.shape,
)

specimens_to_keep_with_override_applied_and_dedupe_by_specimen

# %%
# Get new counts post-dedupe
specimens_to_keep_with_override_applied_and_dedupe_by_specimen["disease"].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    "disease_subtype"
].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    "study_name"
].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen["has_BCR"].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen["has_BCR"]
]["disease"].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen["has_TCR"].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen["has_TCR"]
]["disease"].value_counts()

# %%

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen["sex"].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    "sex"
].isna().value_counts()

# %%
# Here's who is missing sex:
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen["sex"].isna()
]["disease"].value_counts()

# %%
# Here's who is missing sex:
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen["sex"].isna()
]["study_name"].value_counts()

# %%
# Lupus is very sex imbalanced, as may be expected
specimens_to_keep_with_override_applied_and_dedupe_by_specimen.groupby(
    ["disease", "sex"]
).size()

# %%

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    "ethnicity"
].isna().value_counts()

# %%
# Here's who is missing ethnicity:
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen["ethnicity"].isna()
]["disease"].value_counts()

# %%
# Here's who is missing ethnicity:
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    specimens_to_keep_with_override_applied_and_dedupe_by_specimen["ethnicity"].isna()
]["study_name"].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen[
    "ethnicity"
].value_counts()

# %%
# Condense rare ethnicity names (on the pre-dedupe, real object, not on specimens_to_keep_with_override_applied_and_dedupe_by_specimen)
specimens_to_keep["ethnicity_condensed"] = specimens_to_keep["ethnicity"].replace(
    {
        "Black or African American": "African",
        "Black - non-Hispanic": "African",
        "African American": "African",
        "Black or African American - Not Hispanic or Latino": "African",
        #
        "Central/South American": "Hispanic/Latino",
        "White - Hispanic or Latino": "Hispanic/Latino",
        "White - Hispanic": "Hispanic/Latino",
        "White Hispanic": "Hispanic/Latino",
        "Caucasian + Central/South American": "Hispanic/Latino",
        "Caucasian + Hispanic": "Hispanic/Latino",
        "Other - Hispanic": "Hispanic/Latino",
        "Other - Hispanic or Latino": "Hispanic/Latino",
        "Unknown/Not Reported - Hispanic or Latino": "Hispanic/Latino",
        "Native American + Caucasian + Hispanic": "Hispanic/Latino",
        "Multiracial - Hispanic or Latino": "Hispanic/Latino",
        "Unknown - Hispanic/Latino": "Hispanic/Latino",
        "Black or African American - Hispanic/Latino": "Hispanic/Latino",
        "White - Hispanic/Latino": "Hispanic/Latino",
        #
        "Asian - non-Hispanic": "Asian",
        "Asian - Not Hispanic or Latino": "Asian",
        "Asian - Non-Hispanic/Non-Latino": "Asian",
        "Native Hawaiian or Pacific Islander": "Asian",
        "Native Hawaiian or Pacific Islander - Not Hispanic or Latino": "Asian",
        "Native Hawaiian or Other Pacific Islander - Non-Hispanic/Non-Latino": "Asian",
        "India/Arabia/Iran": "Asian",
        #
        "White": "Caucasian",
        "White - non-Hispanic": "Caucasian",
        "White - Not Hispanic or Latino": "Caucasian",
        "Caucasian + Native American": "Caucasian",
        "Native American": "Caucasian",
        "Native American + Caucasian": "Caucasian",
        "White - Non-Hispanic/Non-Latino": "Caucasian",
        #
        "Unknown/Not Reported": np.nan,
        "Other": np.nan,
        "Unknown": np.nan,
        "Black/white - non-Hispanic": np.nan,
        "Other - Not Hispanic or Latino": np.nan,
        "Don't know - Don't know": np.nan,
        "Unknown - Unknown": np.nan,
        "Unknown - Non-Hispanic/Non-Latino": np.nan,
    }
)

# %%
# Beware, these counts are no longer on the deduped object specimens_to_keep_with_override_applied_and_dedupe_by_specimen
specimens_to_keep["ethnicity_condensed"].value_counts()

# %%
# Note this count is not with deduping:
specimens_to_keep["ethnicity_condensed"].isna().value_counts()

# %%
# Note this count is not with deduping:
# Here's who is missing ethnicity_condensed:
specimens_to_keep[specimens_to_keep["ethnicity_condensed"].isna()][
    "disease"
].value_counts()

# %%
# Note this count is not with deduping:
# Here's who is missing ethnicity_condensed:
# *Important*: If we see entries here that can be resolved, update the ethnicity_condensed rules above.
specimens_to_keep[specimens_to_keep["ethnicity_condensed"].isna()][
    "ethnicity"
].value_counts()

# %%
# Note this count is not with deduping:
# Here's who is missing ethnicity_condensed:
specimens_to_keep[specimens_to_keep["ethnicity_condensed"].isna()][
    "study_name"
].value_counts()

# %%
# Note this count is not with deduping:
# Versus total counts
specimens_to_keep["disease"].value_counts()

# %%
# Note this count is not with deduping:
specimens_to_keep.groupby(["ethnicity_condensed", "disease"]).size()

# %%

# %%
# Back to using specimens_to_keep_with_override_applied_and_dedupe_by_specimen for accurate counts
specimens_to_keep_with_override_applied_and_dedupe_by_specimen["age"].value_counts()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen["age"].hist()

# %%
specimens_to_keep_with_override_applied_and_dedupe_by_specimen["age"].describe()

# %%
# Apply age_group cut to the real object (not the dedupe specimens_to_keep_with_override_applied_and_dedupe_by_specimen)
specimens_to_keep["age_group"] = pd.cut(
    specimens_to_keep["age"],
    bins=[0, 20, 30, 40, 50, 60, 70, 80, 100],
    labels=["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"],
    right=False,
)
# Note this count is not with deduping:
specimens_to_keep["age_group"].value_counts()

# %%
specimens_to_keep["age_group"].cat.categories

# %%
# Note this count is not with deduping:
specimens_to_keep["age"].isna().value_counts()

# %%
# Note this count is not with deduping:
specimens_to_keep["age_group"].isna().value_counts()

# %%
# Note this count is not with deduping:
for age_group, grp in specimens_to_keep.groupby("age_group"):
    print(age_group, grp["age"].min(), grp["age"].max())

# %%

# %%
# Note this count is not with deduping:
# Here's who is missing age:
specimens_to_keep[specimens_to_keep["age_group"].isna()]["disease"].value_counts()

# %%
# Note this count is not with deduping:
# Here's who is missing age:
specimens_to_keep[specimens_to_keep["age_group"].isna()]["study_name"].value_counts()

# %%
# Note this count is not with deduping:
# Versus total counts
specimens_to_keep["disease"].value_counts()

# %%
# Note this count is not with deduping:
# <20: only in HIV, Healthy, and Lupus
# 70-80: not in HIV
# 80+: only in Covid and Healthy

specimens_to_keep.groupby(["age_group", "disease"], observed=True).size()

# %%
# Null out "age_group" column for extreme ages with small sample size.

# Note that <20 is kept but is predominantly pediatric lupus.
# Keeping "70-80" despite small sample size, since there is a mix of disease types in there.
# 80+ is too small though.

# Note that we are not getting rid of these specimens altogether,
# but marking age_group NaN will disable their use for demographics-controlling models

orig_shapes = specimens_to_keep.shape[0], specimens_to_keep["age_group"].isna().sum()
mask = specimens_to_keep["age_group"].isin(["80+"])
specimens_to_keep.loc[mask, "age_group"] = np.nan
new_shapes = specimens_to_keep.shape[0], specimens_to_keep["age_group"].isna().sum()

# sanity checks:
# - we did not drop any specimens
assert orig_shapes[0] == new_shapes[0]
# - but we did null out some age_group entries
assert orig_shapes[1] < new_shapes[1]
# - we nulled out the right amount
assert new_shapes[1] - orig_shapes[1] == mask.sum()

# %%

# %%
# The raw specimen_label is not duplicated.
# However, note that once the specimen_label_override is applied, there may be duplicates.
assert not specimens_to_keep["specimen_label"].duplicated().any()

# %%
# All of these rows will get squashed into a single row per specimen
specimens_to_keep["specimen_label_override"].fillna(
    specimens_to_keep["specimen_label"]
).duplicated(keep=False).sum()

# %%

# %%
specimens_to_keep.drop(columns=["ethnicity"]).to_csv(
    config.paths.metadata_dir / "generated_combined_specimen_metadata.tsv",
    sep="\t",
    index=None,
)

# %%

# %%

# %%

# %%
