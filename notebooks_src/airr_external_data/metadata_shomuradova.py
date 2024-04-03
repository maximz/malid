# %% [markdown]
# Patient types:
#
# - `convalescent patients (CPs) `
#
# > To study the adaptive immune response to SARS-CoV-2, we recruited 34 CPs. According to the classification developed by the U.S. National Institutes of Health, the patients were categorized as having asymptomatic (n = 2), mild (n = 20), or moderate to severe (n = 12) disease.
# >
# > Peripheral blood was collected between days 17 and 49 (median day 34) after the onset of symptoms or a positive PCR test result.
#
# - `14 healthy volunteers recruited during the COVID-19 pandemic (HD(CoV)) with no symptoms and negative PCR test results`. However might have past exposure?
#    - Later they do say `All tested HD(CoV) sera lacked antibodies against SARS-CoV-2 antigens`
#    - and `IgGs from the HD(CoV) and HD(BB) groups showed no reactivity to the S protein of SARS-CoV-2 or its receptor-binding domain (RBD)`
#    - however **`This might indicate that some HD(CoV) patients were exposed to the virus but rapidly cleared it via T cells without developing a humoral response.`**
# - `10 samples of peripheral blood mononuclear cells (PBMCs) from biobanked healthy hematopoietic stem cell donors (HD(BB)), which were cryopreserved no later than September 2019`. The HD(BB) are not included in the download - why?
# - `10 serum samples from healthy blood donors that were cryopreserved no later than 2017 (HD(S)).` These were not sequenced

# %%

# %%
import pandas as pd
from malid import config

# %%
base_dir = config.paths.external_raw_data / "Shomuradova"

# %%
# specimen labels seen in the AIRR repertoire list
metadata = pd.read_csv(
    base_dir / "Shomuradova_ir_2022-09-24_0105_632e57c42f04a.tsv", sep="\t"
)
metadata = (
    metadata[
        [
            "repertoire_id",
            "subject_id",
            "sample_id",
            "cell_subset",
            "cell_phenotype",
            "medical_history",
            "disease_stage",
            "age_min",
            "sex",
        ]
    ]
    .rename(columns={"age_min": "age"})
    .assign(
        sex=metadata["sex"].replace({"male": "M", "female": "F"}),
        ethnicity_condensed="Caucasian",
    )
)
metadata

# %%
# specimen labels seen in the actual AIRR sequences export
specimen_labels = pd.read_csv(base_dir / "specimen_labels.txt", header=None)
specimen_labels

# %%
# patient diagnosis/type extracted from paper
# we want to ignore HD(CoV) because they may have some Covid exposure. see notes above.
patient_status = pd.read_csv(base_dir / "Shomuradova_patient_metadata.csv")
patient_status

# %%

# %%
metadata_annot = pd.merge(
    metadata,
    patient_status.assign(
        subject_id=patient_status["patient"].str.extract("(\d+)").astype(int)
    ),
    how="left",
    on="subject_id",
    validate="m:1",
)
metadata_annot

# %%
metadata_annot["type"].isna().value_counts()

# %%
metadata_annot["type"].value_counts()

# %%
metadata_annot.groupby("type")["subject_id"].nunique()

# %%
metadata_annot.groupby("disease")["subject_id"].nunique()

# %%

# %%
# HD(BB) not available. Covid19 only.
metadata_filtered = (
    metadata_annot[metadata_annot["disease"] == "Covid19"]
    .rename(columns={"repertoire_id": "specimen_label", "patient": "participant_label"})
    .assign(study_name="Shomuradova")
)
metadata_filtered["disease_subtype"] = (
    metadata_filtered["disease"] + " - " + metadata_filtered["severity"]
)
metadata_filtered

# %%
# if disease_stage is blank (with one space), it means asymptomatic
metadata_filtered[
    metadata_filtered["disease_stage"]
    .mask(metadata_filtered["disease_stage"].str.strip() == "")
    .isna()
]

# %%
# confirm our annotations
metadata_filtered[["medical_history", "severity"]].drop_duplicates()

# %%

# %%

# %%
# extract sample type
# "We analyzed the TCR repertoires of fluorescence-activated cell sorting (FACS)-sorted IFNγ-secreting CD8+/CD4+ cells and MHC-tetramer-positive populations as well as the total fraction of PBMCs by Illumina high-throughput sequencing"
metadata_filtered["sample_type"] = (
    metadata_filtered["sample_id"]
    .str.split("_")
    .str[1:]
    .apply(lambda arr: " ".join(arr))
)
metadata_filtered[["cell_phenotype", "cell_subset", "sample_type"]].drop_duplicates()

# %%
metadata_filtered.groupby(["cell_phenotype", "cell_subset", "sample_type"]).size()

# %%

# %%
# use total PBMC samples only. cell_phenotype and cell_subset must be blank
metadata_filtered = metadata_filtered[
    (metadata_filtered["sample_type"] == "PBMC")
    & (
        metadata_filtered["cell_phenotype"]
        .mask(metadata_filtered["cell_phenotype"].str.strip() == "")
        .isna()
    )
    & (
        metadata_filtered["cell_subset"]
        .mask(metadata_filtered["cell_subset"].str.strip() == "")
        .isna()
    )
]
metadata_filtered

# %%
# one sample left per patient

# %%
metadata_filtered["type"].value_counts()

# %%
metadata_filtered["disease"].value_counts()

# %%
metadata_filtered.groupby("type")["subject_id"].nunique()

# %%
metadata_filtered.groupby("disease")["subject_id"].nunique()

# %%
assert (metadata_filtered.groupby("subject_id").size() == 1).all()

# %%

# %%

# %%

# %%
# filter by subtype? not for now.

# %%
metadata_filtered["disease_subtype"].value_counts()

# %%
metadata_filtered

# %%

# %%

# %%
# export
metadata_export = metadata_filtered[
    [
        "specimen_label",
        "participant_label",
        "disease",
        "study_name",
        "disease_subtype",
        "age",
        "sex",
        "ethnicity_condensed",
    ]
]
metadata_export

# %%
metadata_export.to_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid_tcr_shomuradova.participant_metadata.tsv",
    sep="\t",
    index=None,
)

# %%

# %%

# %%
