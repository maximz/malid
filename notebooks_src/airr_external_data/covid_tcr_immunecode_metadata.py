# %%
from malid import config
import pandas as pd
import numpy as np

# %%

# %% [markdown]
# # Filter ImmuneCODE dataset to "Covid peak-timepoint" samples for external validation.
#
# Goal is no comorbidities with immune effects

# %%

# %%
df = pd.read_csv(
    config.paths.metadata_dir / "adaptive" / "ImmuneCODE-Repertoire-Tags-002.2.tsv",
    sep="\t",
)
df.shape

# %%
df

# %%
df["Dataset"].value_counts()

# %%

# %%
df = df[df["Virus Diseases"] == "COVID-19 Positive"]
df.shape

# %%
df = df[df["days_from_symptom_onset_to_sample"] >= 11]
df.shape

# %%
df = df[df["days_from_symptom_onset_to_sample"] <= 21]
df.shape

# %%
df["Tissue Source"].value_counts()

# %%
df["Dataset"].value_counts()

# %%

# %%
df["diseases"].str.split(";").explode().value_counts()

# %%
df["diseases"].str.startswith("covid--positive").value_counts()

# %%
# remove other diseases
df = df[df["diseases"] == "covid--positive"]
df.shape

# %%
df["Dataset"].value_counts()

# %%

# %% [markdown]
# Filter by other tags, which are not universal to all cohorts:

# %%
df = df[df["covid_category"] != "Recovered"]
df.shape

# %%
df = df[df["covid_category"] != "Exposed"]
df.shape

# %%
df = df[df["cancer_diagnosed"] != True]
df.shape

# %%
df = df[df["cancer_type"].isna()]
df.shape

# %%
df = df[df["describe_autoimmune_diagnoses"].isna()]
df.shape

# %%
df = df[df["describe_autoimmune_medications"].isna()]
df.shape

# %%
df = df[df["describe_cancers"].isna()]
df.shape

# %%
df = df[df["describe_immunosupressants"].replace("None", np.nan).isna()]
df.shape

# %%
df = df[df["describe_other_diagnoses"].isna()]
df.shape

# %%
df = df[df["diabetes_type"].replace("No", np.nan).isna()]
df.shape

# %%
df = df[df["has_hiv"] != True]
df.shape

# %%
df = df[df["is_immunocompromised"] != True]
df.shape

# %%
df = df[df["selected_autoimmune_diagnoses"].replace("None", np.nan).isna()]
df.shape

# %%
df = df[df["selected_other_diagnoses"].replace("None", np.nan).isna()]
df.shape

# %%
df = df[df["uses_autoimmune_medications"].isna()]
df.shape

# %%
df = df[df["uses_immunosuppressant"] != True]
df.shape

# %%

# %%
df["Dataset"].value_counts()

# %%

# %%
# Maybe we should go further and guarantee that not HIV? We have that info for COVID-19-ISB only
# Avoiding this for now.

# %%
# Per the preprint, the -Adaptive cohort may have cDNA in addition to gDNA: https://www.researchsquare.com/article/rs-51964/v1 table 2
# (Although this might just mean that the samples were used for gDNA sequencing and also for cDNA MIRA?)
# The other cohorts are all gDNA, I believe
# To be consistent / safe, we will remove any possibility of cDNA by removing the (already very few) -Adaptive cohort entries
df = df[df["Dataset"] != "COVID-19-Adaptive"].assign(
    sequencing_type="gDNA", locus="TCRB"
)
df.shape

# %%
df["Dataset"].value_counts()

# %%

# %%

# %%
# Set subtype
df[["Dataset", "hospitalized", "icu_admit"]]

# %%
df["icu_admit"] = df["icu_admit"].map({True: "ICU"})
df["hospitalized"] = df["hospitalized"].map({True: "Hospitalized"})
df[["Dataset", "hospitalized", "icu_admit"]]

# %%
df["disease_subtype"] = df[["Dataset", "hospitalized", "icu_admit"]].apply(
    lambda row: " - ".join(row.dropna()), axis=1
)
df["disease_subtype"]

# %%

# %%

# %%

# %%
# Get first row for each subject_id (can have multiple sample_name's)
df = df.groupby("subject_id").head(n=1)
df.shape

# %%

# %%
# Set identifiers
df = df.assign(
    participant_label="ImmuneCode-" + df["subject_id"].astype(str),
    disease="Covid19",
)

df.rename(columns={"sample_name": "specimen_label"}, inplace=True)

assert not df["specimen_label"].duplicated().any()
assert not df["participant_label"].duplicated().any()

# %%
df["sex"] = df["Biological Sex"].str.lower().replace({"male": "M", "female": "F"})

# %%
# extract number of years
df["age"] = df["Age"].str.extract("(\d+)")
df["age"].value_counts()

# %%
df["ethnicity"] = (
    df["Racial Group"].fillna("Unknown") + " - " + df["Ethnic Group"].fillna("Unknown")
)
df["ethnicity"].value_counts()

# %%
df["ethnicity_condensed"] = df["ethnicity"].replace(
    {
        "Caucasian - Unknown": "Caucasian",
        "Hispanic - Unknown": "Hispanic/Latino",
        "Unknown racial group - Hispanic or Latino": "Hispanic/Latino",
        "Caucasian - Non-Hispanic or Latino": "Caucasian",
        "Unknown - Unknown": np.nan,
        "Unknown racial group - Non-Hispanic or Latino": np.nan,
        "Asian or Pacific Islander - Non-Hispanic or Latino": "Asian",
        "Asian or Pacific Islander - Unknown": "Asian",
    }
)
df["ethnicity_condensed"].value_counts()

# %%
df = df[
    [
        "participant_label",
        "specimen_label",
        "disease",
        "disease_subtype",
        "age",
        "sex",
        "ethnicity_condensed",
        "sequencing_type",
        "locus",
    ]
]
df

# %%
df.to_csv(
    config.paths.metadata_dir
    / "adaptive"
    / "generated.immunecode_covid_tcr.specimens.tsv",
    sep="\t",
    index=None,
)

# %%

# %%
