# %%
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed

# %%
from malid import config
from malid.datamodels import GeneLocus, healthy_label

# %%

# %%

# %% [markdown]
# ## Instructions for adding more
#
# **Merging cell type subsets from the same sample:** `specimen_label` should be consistent across entries with different `replicate_label`'s.
#
# **If we have external cohorts that are BCR+TCR:**
#
# We should have one row per locus per replicate-of-a-specimen. Often specimens are single-replicate, so we can think of this as: one low per locus per specimen.
#
# Set a `specimen_label_by_locus` column that is the globally-unique specimen label tailored to a particular locus, e.g. `$SPECIMENLABEL-IGH` or `$SPECIMENLABEL-TRB` format. (TODO: update this to be replicate_label_by_locus?)
#
# And set a `specimen_label` column that is equivalent across different-loci rows for that specimen.
#
# The row's `gene_locus` column should be set to the locus of that row (must be the name of a valid `GeneLocus` enum value), and the `fname` column should be set to the path to the file containing the data for that locus.
#
# _NOTE: The above is not so relevant for Adaptive -- we see below that this data is almost entirely TCR, and we will just ignore the BCR._

# %%

# %%

# %% [markdown]
# ## Adaptive Covid TCR specimens

# %%
# This metadata was created in covid_tcr_immunecode_metadata.ipynb. We filtered to peak disease samples.
adaptive_tcr_covid_specimens = (
    pd.read_csv(
        config.paths.metadata_dir
        / "adaptive"
        / "generated.immunecode_covid_tcr.specimens.tsv",
        sep="\t",
    )
    .assign(
        study="immunecode",
        species="Human",
    )
    .rename(columns={"ethnicity_condensed": "ethnicity"})
)
adaptive_tcr_covid_specimens["sample_name"] = adaptive_tcr_covid_specimens[
    "specimen_label"
]

adaptive_tcr_covid_specimens

# %%

# %% [markdown]
# # All others
#
# First a general survey:

# %%
dfs_adaptive = []

cols_desired = [
    "sample_name",
    "locus",
    "sample_tags",
    "counting_method",
    "primer_set",
    "species",
    "study",
]


def _load_metadata_df(study, filter_cols=True):
    df = pd.read_csv(
        config.paths.metadata_dir / "adaptive" / f"{study}.tsv", sep="\t"
    ).assign(study=study)
    if filter_cols:
        # Filter down to any matching cols
        return df[[c for c in cols_desired if c in df.columns]]
    return df


for study in [
    "emerson-2017-natgen",
    "mitchell-2022-jcii",
    "mustjoki-2017-natcomms",
    "ramesh-2015-ci",
    "TCRBv4-control",
    "towlerton-2022-hiv",
]:
    dfs_adaptive.append(_load_metadata_df(study))
dfs_adaptive = pd.concat(dfs_adaptive, axis=0).reset_index(drop=True)
dfs_adaptive

# %%
dfs_adaptive["species"].value_counts()

# %%
dfs_adaptive.groupby(["locus", "primer_set", "counting_method"]).size()

# %%
sample_tags = pd.merge(
    # explode the sample tags
    dfs_adaptive["sample_tags"].str.split(",").explode(),
    # link back to study
    dfs_adaptive["study"],
    left_index=True,
    right_index=True,
    how="left",
    validate="m:1",
).dropna(subset="sample_tags")

sample_tags["sample_tags"] = sample_tags["sample_tags"].str.strip()
sample_tags = sample_tags.drop_duplicates()
sample_tags

# %%
for k, v in sample_tags.groupby("study")["sample_tags"].unique().iteritems():
    print(k)
    print(", ".join(v))
    print("")

# %%

# %% [markdown]
# ## Let's annotate each study manually, one by one.

# %%
dfs_adaptive_manual = []

# %%

# %%
"""
Emerson healthy TCR specimens, CMV+ and CMV-:

ImmuneAccess says not to run "Export Sample", and instead use their dedicated zip file link. That sounds fine, and we could use ImmuneAccess's metadata from the Sample Overview screen as usual...

except ImmuneAccess has somehow lost the "HIP"-prefixed labels for cohort 1, which are necessary to match up to CMV positivity metadata from SI Table 1. Instead ImmuneAccess now labels cohort 1 samples from 1 to N, basically.

But we found an older zip file online that has the correct "HIP-" labels: https://s3-us-west-2.amazonaws.com/publishedproject-supplements/emerson-2017-natgen/emerson-2017-natgen.zip

And someone kindly exported the Sample Overview ImmuneAccess metadata a while back and uploaded it here: https://github.com/jostmey/dkm/blob/7839937413af23203a442d5291e311ccb034dce7/repertoire-classification-problem/dataset/downloads/SampleOverview_08-19-2017_12-59-19_PM.tsv
This is saved in our repo as emerson-2017-natgen.tsv.

From brief inspection, all that has changed is the sample labels are now mangled in ImmuneAccess. And the order is different. Bummer.

We're going to use the older exports, but merge in some important fields from the newer metadata export.

One more important note: We have split the source files into two study names corresponding to the Emerson training and validation cohorts.
The metadata files here are under the master study name, but the source files are under the two split study names.
"HIP" samples: emerson-2017-natgen_train
"Keck" samples: emerson-2017-natgen_validation
"""

df = _load_metadata_df("emerson-2017-natgen", filter_cols=False)
df.sort_values("sample_name")

# %%
df["sample_tags"].str.split(",").explode().str.strip().value_counts().head(n=50)

# %%
# counting_method is not available in the older immuneaccess sampleoverview we found online (see note above)
# we'll use the wrong-labeled one to check this:
df_m_wrong_labels = _load_metadata_df(
    "emerson-2017-natgen.wrong_labels", filter_cols=False
)
df_m_wrong_labels["counting_method"].value_counts()

# %%
# same shape, different orders
df_m_wrong_labels.shape, df.shape

# %%
cols_desired

# %%
df.shape

# %%
df = pd.merge(
    df,
    df_m_wrong_labels[
        [
            "total_templates",
            "total_reads",
            "productive_templates",
            "total_productive_reads",
            "total_rearrangements",
            "productive_rearrangements",
            "counting_method",
            "primer_set",
            "species",
        ]
    ],
    on=[
        "total_templates",
        "total_reads",
        "productive_templates",
        "total_productive_reads",
        "total_rearrangements",
        "productive_rearrangements",
    ],
    how="inner",
    validate="1:1",
)
df.shape

# %%
df = df[cols_desired]
df

# %%
# sanity check
df["counting_method"].value_counts()

# %%
df["sequencing_type"] = "gDNA"
# Split e.g. Keck0116_MC1 into Keck0116
df["specimen_label"] = df["sample_name"]
df["participant_label"] = df["sample_name"].str.split("_").str[0]
assert (df["participant_label"].value_counts() == 1).all()
df["participant_label"].sort_values()

# %%
# let's add metadata from SI Table 1
df_m1 = pd.read_excel(
    config.paths.metadata_dir / "adaptive" / "emerson-2017-natgen.si_table_1.xlsx",
    sheet_name=0,
)
df_m2 = pd.read_excel(
    config.paths.metadata_dir / "adaptive" / "emerson-2017-natgen.si_table_1.xlsx",
    sheet_name=1,
)
df_m1

# %%
df_m2

# %%
df_m = pd.concat([df_m1, df_m2], axis=0)
df_m

# %%
df_m["Known CMV status"].value_counts()

# %%
df_m = df_m.rename(
    columns={
        "Subject ID": "participant_label",
        "Race and ethnicity ": "Ethnicity",
    }
).assign(cmv=df_m["Known CMV status"].map({"+": "CMV+", "-": "CMV-"}))[
    ["participant_label", "Sex", "Age", "Ethnicity", "cmv"]
]
df_m

# %%
df = pd.merge(
    df,
    df_m,
    how="left",
    on="participant_label",
    validate="1:1",
)
df

# %%
df["cmv"].isna().value_counts()

# %%
df["cmv"].value_counts()

# %%
df["disease"] = healthy_label
df["disease_subtype"] = f"{healthy_label} - " + df["cmv"].fillna("CMV Unknown")
df["disease_subtype"].value_counts()

# %%
# One more important note: We have split the source files into two study names corresponding to the Emerson training and validation cohorts.
# The metadata files here are under the master study name, but the source files are under the two split study names.
# "HIP" samples: emerson-2017-natgen_train
# "Keck" samples: emerson-2017-natgen_validation

# So we must update the study name.
df["study"] = (
    df["study"]
    + "_"
    + df["participant_label"]
    .str.startswith("Keck")
    .map({True: "validation", False: "train"})
)
df["study"].value_counts()

# %%
dfs_adaptive_manual.append(df)
df

# %%

# %%
df = _load_metadata_df("mitchell-2022-jcii")

"""
https://insight.jci.org/articles/view/161885
Childhood T1D:

143 new-onset T1D: "As an independent validation, we sequenced and analyzed TCRβ repertoires from a cohort of new-onset T1D patients (n=143)"

25 HHC (4 samples each)
29 children that progress to T1D (4 samples each)
"longitudinal peripheral blood DNA samples at four time points beginning early in life (median age of 1.4 years) from children that progressed to T1D (n=29) and age/sex-matched islet autoantibody negative controls (n=25)"
"""

df["sequencing_type"] = "gDNA"

df

# %%
df["specimen_label"] = df["sample_name"].str.split("_TCRB").str[0]
df["specimen_label"]

# %%
# new onset n=143 cohort is "Denver"
df["sample_name"].str.contains("Denver").value_counts()

# %%
# The other 216 samples
29 * 4 + 25 * 4

# %%
df["sample_tags"].str.split(",").explode().str.strip().value_counts().head(n=20)

# %%

# %%
# Now: import subject IDs, age, sex, ancestry from https://insight.jci.org/articles/view/161885/sd/2
# Notice that the 29 children that progress to T1D don't necessarily have T1D diagnosis prior to the last sample
# So only include the new-onset T1D samples and the 25 healthy control children (with repeated samples)

# %%
control_samples = pd.read_excel(
    config.paths.metadata_dir / "adaptive" / "jci.insight.161885.sdd1.xlsx",
    sheet_name="DAISY Control",
)

# first row per person has Sex/Race/Ethnicity not null
not_null_demographics = control_samples["Sex"].notna()

# double check that all those rows have sample timepoint = 1
assert (control_samples.loc[not_null_demographics, "Sample Timepoint"] == 1).all()


# Assign 'participant_label' as the cumulative sum of 'not_null_demographics', starting from 1
control_samples[
    "participant_label"
] = "control_" + not_null_demographics.cumsum().astype(str)
control_samples["disease"] = healthy_label
control_samples["disease_subtype"] = f"{healthy_label} - T1D negative"

# Transfer demographics to rest of rows per person
# Fill null column values in each group with the first non-null column value in the group (ffill() is the key part, but bfill() saves us if for some reason the first row isn't the one with the entry - not the case for us here but just in case)
for col in ["Sex", "Race", "Ethnicity"]:
    control_samples[col] = (
        control_samples.groupby("participant_label")[col].bfill().ffill()
    )

control_samples

# %%
control_samples = control_samples[
    [
        "Adaptive ID",
        "Age at Visit (years)",
        "Sample Timepoint",
        "Sex",
        "Race",
        "Ethnicity",
        "participant_label",
        "disease",
        "disease_subtype",
    ]
].rename(columns={"Age at Visit (years)": "Age"})
control_samples["Adaptive ID"] = control_samples["Adaptive ID"].astype(str)
control_samples

# %%
control_samples = pd.merge(
    df[~df["sample_name"].str.contains("Denver")],
    control_samples,
    how="inner",
    left_on="specimen_label",
    right_on="Adaptive ID",
    validate="1:1",
).drop(columns="Adaptive ID")
# left with 25*4 as expected
control_samples

# %%
control_samples["participant_label"].value_counts()

# %%

# %%

# %%
new_onset_samples = pd.read_excel(
    config.paths.metadata_dir / "adaptive" / "jci.insight.161885.sdd1.xlsx",
    sheet_name="New-onset T1D",
)
# Left zero pad
new_onset_samples["participant_label"] = "DenverT1D-" + new_onset_samples[
    "Adaptive ID"
].astype(str).str.zfill(3)
new_onset_samples["disease"] = "T1D"
new_onset_samples["disease_subtype"] = "T1D - new onset"
new_onset_samples

# %%
# these are really new onset
new_onset_samples["T1D Duration (days)"].describe()

# %%
new_onset_samples = new_onset_samples[
    [
        "participant_label",
        "Age at Diagnosis (years)",
        "Sex",
        "Race",
        "Ethnicity",
        "disease",
        "disease_subtype",
    ]
].rename(columns={"Age at Diagnosis (years)": "Age"})
new_onset_samples

# %%
df[df["sample_name"].str.contains("Denver")]

# %%
new_onset_samples = pd.merge(
    df[df["sample_name"].str.contains("Denver")],
    new_onset_samples,
    how="inner",
    left_on="specimen_label",
    right_on="participant_label",
    validate="1:1",
)
# left with 143 as expected
new_onset_samples

# %%

# %%
# Recombine
df = pd.concat([control_samples, new_onset_samples], axis=0)
df

# %%
dfs_adaptive_manual.append(df)
df

# %%

# %%

# %%
df = _load_metadata_df("mustjoki-2017-natcomms")

"""
https://www.nature.com/articles/ncomms15869
RA, newly diagnosed. Some seronegative, some seropositive. A few patients have on-treatment followup
See kelkka-2020-jci above for a follow-up study.

Peripheral-blood mononuclear cells (PBMCs) were separated from EDTA blood using Ficoll gradient separation (Ficoll-Paque PLUS, GE Healthcare, cat. no 17-1440-03).
CD4+ and CD8+ cells were separated with magnetic bead sorting using positive selection for both fractions

DNA was extracted from CD4- and CD8-enriched samples or from the PBMC fraction
TCRB deep sequencing was performed from 65 patients, accompanied by sequencing of 20 healthy controls
Genomic DNA was used in all cases.

Totals: 82 newly diagnosed RA patients, among whom 25 patients were sequenced with the immunogene panel. The 20 healthy controls were also included in the immunogene panel.

Table 1: Immunogene panel sequencing was performed on both CD4+ and CD8+ cells of 25 newly diagnosed RA patients and 20 healthy controls (HCs)

More info at https://static-content.springer.com/esm/art%3A10.1038%2Fncomms15869/MediaObjects/41467_2017_BFncomms15869_MOESM218_ESM.pdf SI Table 2
"""

df["sequencing_type"] = "gDNA"
df["participant_label"] = df["sample_name"].str.split("-").str[0]

df

# %%
df["participant_label"].sort_values().unique()

# %%
df["sample_tags"].str.split(",").explode().str.strip().value_counts()

# %%
df = df[~df["sample_tags"].str.contains("Synovial fluid").fillna(False)].copy()

# sanity check
assert not df["sample_name"].str.contains("-SF-").any()

df

# %%

# %%

# %%
# table 1 has the HCs
df1 = pd.read_csv(
    config.paths.metadata_dir / "adaptive" / "mustjoki-2017-natcomms.table_1.csv"
)
df1

# %%
df1 = df1[df1["Patient ID"].str.startswith("HC")]
df1 = df1.rename(columns={"Age at diagnosis": "Age"})[["Patient ID", "Sex", "Age"]]
df1["disease"] = healthy_label
df1["disease_subtype"] = f"{healthy_label} - RA negative"
df1

# %%
# SI table 2 has all the RAs
df2 = pd.read_csv(
    config.paths.metadata_dir / "adaptive" / "mustjoki-2017-natcomms.si_table_2.csv"
)
df2

# %%
df2 = df2.rename(columns={"Age at diagnosis": "Age", "Gender": "Sex"})[
    ["Patient ID", "Sex", "Seropositive", "Age"]
]
df2["Patient ID"] = "RA" + df2["Patient ID"].astype(str)
df2["disease"] = "RA"
df2["disease_subtype"] = "RA - " + df2["Seropositive"].map(
    {"yes": "sero-positive", "no": "sero-negative"}
)
df2

# %%
df = pd.merge(
    df,
    pd.concat([df1, df2], axis=0).drop(columns="Seropositive"),
    how="left",
    left_on="participant_label",
    right_on="Patient ID",
    validate="m:1",
).drop(columns="Patient ID")
df

# %%
assert not df["disease"].isna().any()

# %%
df["participant_label"].value_counts().head(n=10)

# %%
# people with multiple samples
df[
    df["participant_label"].isin(
        df["participant_label"]
        .value_counts()[df["participant_label"].value_counts() > 1]
        .index
    )
].sort_values("sample_name")

# %%
# Remove late timepoints from RA 1, 2, and 3. We want newly diagnosed, treatment naive.
df = df[~df["sample_name"].isin(["RA1-7months", "RA2-18months", "RA3-3years"])]

# %%
# Remove RA patient 1 2's special extra sorts:
# Patient 2 was a 72-year-old female who also had palindromic rheumatism and a previous history of other inflammatory disorders: asthma, lichen ruber and atrophic gastritis. At the time of RA diagnosis, flow cytometric screening identified two unusually large populations of CD8+ T cells: Vβ1+ (14%) and Vβ13.6+ (11%)
df = df[~df["sample_name"].isin(["RA2-CD8+Vb1+"])]

# %%

# %%
# Recheck: people with multiple samples
# We see some CD4 separated out of CD8
df[
    df["participant_label"].isin(
        df["participant_label"]
        .value_counts()[df["participant_label"].value_counts() > 1]
        .index
    )
].sort_values("sample_name")

# %%
df[df["sample_tags"].str.contains("CD4\+").fillna(False)]

# %%
df[df["sample_tags"].str.contains("CD8-").fillna(False)]

# %%
df[df["sample_tags"].str.contains("CD4-").fillna(False)]

# %%
df[df["sample_tags"].str.contains("CD8\+").fillna(False)]

# %%
# Many samples are full PBMC though
df[
    (~df["sample_tags"].str.contains("CD8").fillna(False))
    & (~df["sample_tags"].str.contains("CD4").fillna(False))
]

# %%

# %%
# Mark the CD8/CD4 separations
# keep the cell type inside the replicate label, to be merged within the ETL flow later.
df["specimen_label"] = df["participant_label"]
df["replicate_label"] = df["sample_name"]
df.loc[df["sample_name"] == "RA23-CD4", "replicate_label"] = "RA23_CD4"
df.loc[df["sample_name"] == "RA23", "replicate_label"] = "RA23_CD8"
df.loc[df["sample_name"] == "RA6-CD4", "replicate_label"] = "RA6_CD4"
df.loc[df["sample_name"] == "RA6", "replicate_label"] = "RA6_CD8"

# Recheck: people with multiple samples
# We see some CD4 separated out of CD8
df[
    df["participant_label"].isin(
        df["participant_label"]
        .value_counts()[df["participant_label"].value_counts() > 1]
        .index
    )
].sort_values("sample_name")

# %%

# %%
# Keep full PBMC samples only, for consistency
df = pd.concat(
    [
        df[
            (~df["sample_tags"].str.contains("CD8").fillna(False))
            & (~df["sample_tags"].str.contains("CD4").fillna(False))
            # this would indicate CD8:
            & (~df["sample_tags"].str.contains("HLA MHC Class I").fillna(False))
        ],
        # also keep the ones where we have both fractions - see previous cell
        df[df["replicate_label"].isin(["RA23_CD8", "RA23_CD4", "RA6_CD8", "RA6_CD4"])],
    ],
    axis=0,
)
df = df.sort_values("replicate_label")
df

# %%
dfs_adaptive_manual.append(df)
df


# %%

# %%
df = _load_metadata_df("ramesh-2015-ci")

"""
CVID as a T cell defect, not just B cell
https://www.sciencedirect.com/science/article/abs/pii/S1521661615000042?via%3Dihub
up to 44 CVID, 22 HC? see https://ars.els-cdn.com/content/image/1-s2.0-S1521661615000042-gr6_lrg.jpg for the participant names

Peripheral blood DNA samples of 44 CVID subjects, 15 females and 29 males of ages 9 to 64 years with a mean of 40,
and 22 healthy adult volunteers, 12 females and 10 males of ages 23 to 66 years with a mean of 34

Genomic DNA
"""

df["sequencing_type"] = "gDNA"
df

# %%
df["sample_tags"].str.split(",").explode().str.strip().value_counts().head(n=50)

# %%
# Not sure: What are the last two?
df["sample_name"].unique()

# %%
# Drop those to be safe.
df = df[~df["sample_name"].str.contains("Rundles")].copy()
df

# %%
df["specimen_label"] = df["sample_name"]
df["participant_label"] = df["sample_name"]
df["disease"] = (
    df["sample_name"].str.startswith("N").map({True: healthy_label, False: "CVID"})
)
df["disease_subtype"] = df["disease"].replace(
    {healthy_label: healthy_label + " - CVID negative"}
)

# %%
dfs_adaptive_manual.append(df)
df

# %%

# %%
df = _load_metadata_df("TCRBv4-control")

"""
colorectal cancer, lung cancer, head and neck cancer, and healthy control
58 cancer, 88 hhc, pre-covid
ignore the 11 FFPE samples
add the HHCs, don't do cancer for now
"""

df["sequencing_type"] = "gDNA"

df

# %%
df["sample_tags"].str.split(",").explode().str.strip().value_counts().head(n=50)

# %%
# almost always, a row's sample tag will contain EITHER cancer or healthy
# except for these FFPE rows
df[
    df["sample_tags"].str.contains("Cancer")
    != ~df["sample_tags"].str.contains("Healthy")
]

# %%
# remove FFPE
df = df[~df["sample_tags"].str.contains("FFPE")]

# %%
# All remaining rows either marked Blood or PBMC (or both)
assert (
    df["sample_tags"].str.contains("Blood") | df["sample_tags"].str.contains("PBMC")
).all()

# %%
# now that we filtered to PBMC: a row's sample tag will contain EITHER cancer or healthy, NEVER both
assert (
    df["sample_tags"].str.contains("Cancer")
    != df["sample_tags"].str.contains("Healthy")
).all()

# %%
# filter to healthy only. remove cancer
df = df[df["sample_tags"].str.contains("Healthy")]
df

# %%
# TODO: Pull out age, sex, ancestry
# df['sample_tags'].unique()

# %%
df["disease"] = healthy_label
df["disease_subtype"] = f"{healthy_label} - TCRBv4-control"

# %%
df["participant_label"] = df["sample_name"]
df["participant_label"].unique()

# %%
df["specimen_label"] = df["sample_name"]

# %%
dfs_adaptive_manual.append(df)
df

# %%

# %%
df = _load_metadata_df("towlerton-2022-hiv")

"""
HIV on long term therapy, before and after
30 adults, 192 samples?
30 adults with HIV infection before and after the initiation of ART

A total of 192 cryopreserved PBMC samples serially collected over a mean of 6 (range, 1-12) years from 30 adults with confirmed HIV infection (median, 7 samples per subject) were received from the University of Washington CFAR Biorepository.
PBMC samples collected at 1-4 timepoints before and 2-8 time points after the initiation of ART were available from each subject.
genomic DNA was extracted from unsorted PBMC

The blood samples from PLHIV were obtained between 2004 and 2017: pre-pandemic

Metadata:
https://www.frontiersin.org/articles/10.3389/fimmu.2022.879190/full#supplementary-material
https://github.com/shashidhar22/airr_seq_pipelines/blob/main/metadata/cfar/CFAR_Dean_metadata.xlsx
(identical?)
"""

df["sequencing_type"] = "gDNA"

df

# %%
df["sample_tags"].str.split(",").explode().str.strip().value_counts().head(n=50)

# %%
# each entry has a pre or post ART timepoint identified
df["sample_tags"].str.contains("ART").all()

# %%
sample_tags = df["sample_tags"].str.split(",").explode().str.strip().drop_duplicates()
sample_tags[sample_tags.str.contains("ART")].sort_values()

# %%
df_m = pd.read_excel(
    config.paths.metadata_dir
    / "adaptive"
    / "towlerton-2022-hiv.CFAR_Dean_metadata.xlsx",
    sheet_name="CFAR_Sample_info",
)
df_m

# %%
df.shape

# %%
df = pd.merge(
    df, df_m, left_on="sample_name", right_on="sampleID", validate="1:1", how="left"
).drop(columns="sampleID")
df

# %%
df["timePoint"].value_counts()

# %%
df = df[df["timePoint"] == "Pre-ART"]
df

# %%
df["time_group"].value_counts()

# %%
df = df.rename(
    columns={"patientID": "participant_label", "age_at_collection": "Age", "sex": "Sex"}
)
df["specimen_label"] = df["sample_name"]

# %%
df["disease"] = "HIV"
df["disease_subtype"] = "HIV - before anti-retroviral therapy"
df

# %%
# how many samples per person?
df["participant_label"].value_counts().value_counts()

# %%
dfs_adaptive_manual.append(df)
df

# %%

# %%
# Add additional studies here.

# %%

# %% [markdown]
# # Combine all manually annotated studies

# %%
len(dfs_adaptive_manual)

# %%
# Also add ImmuneCode from the very top
dfs_adaptive_manual.append(adaptive_tcr_covid_specimens)

# %%

# %%
dfs_adaptive_manual = pd.concat(dfs_adaptive_manual, axis=0).reset_index(drop=True)
dfs_adaptive_manual_bak = dfs_adaptive_manual.copy()


# %%
dfs_adaptive_manual = dfs_adaptive_manual_bak.copy()  # for easy reset when debugging
dfs_adaptive_manual


# %%
del df  # unused variable now - let's get it out of scope to be safe

# %%

# %%

# %%

# %%
dfs_adaptive_manual["sequencing_type"].value_counts()

# %%
assert not dfs_adaptive_manual["sequencing_type"].isna().any()

# %%
dfs_adaptive_manual["disease"].value_counts()

# %%
assert not dfs_adaptive_manual["study"].isna().any()
assert not dfs_adaptive_manual["participant_label"].isna().any()
assert not dfs_adaptive_manual["specimen_label"].isna().any()

# %%
# To be consistent with boydlab columns, we'll add amplification_label, which here will always equal specimen_label.
# See sample_sequences.py for more details on how this gets used.
if "amplification_label" not in dfs_adaptive_manual.columns:
    dfs_adaptive_manual["amplification_label"] = dfs_adaptive_manual["specimen_label"]
else:
    # fill NA
    dfs_adaptive_manual["amplification_label"].fillna(
        dfs_adaptive_manual["specimen_label"], inplace=True
    )

# Fill replicate_label
dfs_adaptive_manual["replicate_label"].fillna(
    dfs_adaptive_manual["specimen_label"], inplace=True
)

# %%
# add study prefixes to make these labels unique to study:
for col in [
    "participant_label",
    "specimen_label",
    "amplification_label",
    "replicate_label",
]:
    dfs_adaptive_manual[col] = (
        dfs_adaptive_manual["study"] + "_" + dfs_adaptive_manual[col].astype(str)
    )

# %%
# confirm one entry per replicate label per locus, at most!
# (specimens can have multiple replicates, e.g. cell type subsets that get merged.)
# (participants can have multiple specimens, e.g. separate time points)
assert (dfs_adaptive_manual.groupby(["locus", "replicate_label"]).size() == 1).all()

# %%

# %%

# %%
dfs_adaptive_manual.groupby(["sequencing_type", "locus", "disease"], observed=True)[
    "participant_label"
].nunique().to_frame().sort_values("participant_label")

# %%
dfs_adaptive_manual.groupby(
    ["sequencing_type", "locus", "counting_method", "disease"], observed=True
)["participant_label"].nunique().to_frame()

# %%
dfs_adaptive_manual["disease_subtype"].isna().any()

# %%
dfs_adaptive_manual["disease"].isna().any()

# %%
dfs_adaptive_manual["disease_subtype"].fillna(
    dfs_adaptive_manual["disease"], inplace=True
)

# %%
dfs_adaptive_manual.isna().any()[dfs_adaptive_manual.isna().any()]

# %%
dfs_adaptive_manual["disease_subtype"].value_counts()

# %%
dfs_adaptive_manual[dfs_adaptive_manual["disease_subtype"] == healthy_label][
    "study"
].value_counts()

# %%
dfs_adaptive_manual.groupby(["locus", "disease", "disease_subtype"], observed=True)[
    "participant_label"
].nunique().to_frame()

# %%
dfs_adaptive_manual.groupby(
    ["locus", "counting_method", "disease", "disease_subtype", "study"], observed=True
)["participant_label"].nunique().to_frame()

# %%
dfs_adaptive_manual = dfs_adaptive_manual[
    dfs_adaptive_manual["sequencing_type"] == "gDNA"
]
dfs_adaptive_manual = dfs_adaptive_manual[dfs_adaptive_manual["locus"] == "TCRB"]
dfs_adaptive_manual

# %%
dfs_adaptive_manual.groupby("disease")["participant_label"].nunique().sort_values()

# %%

# %%
# Review which replicates are getting combined into which specimens
# dfs_adaptive_manual[dfs_adaptive_manual['replicate_label'] != dfs_adaptive_manual['specimen_label']].groupby('specimen_label')['replicate_label'].unique().tolist()
dfs_adaptive_manual[
    dfs_adaptive_manual["replicate_label"] != dfs_adaptive_manual["specimen_label"]
][["specimen_label", "replicate_label"]]

# %%
# Review which replicates are getting combined into which specimens
replicates_being_merged_into_same_specimen = (
    dfs_adaptive_manual[
        dfs_adaptive_manual["replicate_label"] != dfs_adaptive_manual["specimen_label"]
    ]
    .groupby("specimen_label")["replicate_label"]
    .unique()
    .apply(pd.Series)
)
# remove rows where single replicate (but just happened to have different label) - no merging happening
replicates_being_merged_into_same_specimen = replicates_being_merged_into_same_specimen[
    replicates_being_merged_into_same_specimen.notna().sum(axis=1) > 1
]
replicates_being_merged_into_same_specimen

# %%
# # Also review the cell_type helper column we made for ourselves:
# (TODO: bring this back)
# dfs_adaptive_manual["cell_type"].value_counts()

# %%

# %%

# %%

# %%
dfs_adaptive_manual["species"].value_counts()

# %%

# %%

# %%

# %%
# all available columns, in case-insensitive sorted order
dfs_adaptive_manual.columns.sort_values(key=lambda idx: idx.str.lower())

# %%
# Symptom metadata columns:
# Add more here if we add more studies
symptoms_columns = [
    "cmv",
]

# %%
# if future studies have participant-level description fields, rename those columns into "participant_description"

# specimen description can come in several fields:
specimen_description_fields = ["time_group", "timePoint"]

# They are either all NA or one is set. Never have multiple of these set:
assert dfs_adaptive_manual[specimen_description_fields].notna().sum(axis=1).max()

# So we can just take first non-null value (if any) per row from these columns (https://stackoverflow.com/a/37938780/130164):
dfs_adaptive_manual["specimen_description"] = (
    dfs_adaptive_manual[specimen_description_fields]
    .fillna(method="bfill", axis=1)
    .iloc[:, 0]
)
dfs_adaptive_manual["specimen_description"].value_counts()

# %%
# the two age columns are never set at the same time, so we can fillna to combine
assert (
    dfs_adaptive_manual["age"].notna().astype(int)
    + dfs_adaptive_manual["Age"].notna().astype(int)
).max() == 1
dfs_adaptive_manual["Age"].fillna(dfs_adaptive_manual["age"], inplace=True)

# %%
# the two sex columns are never set at the same time, so we can fillna to combine
assert (
    dfs_adaptive_manual["sex"].notna().astype(int)
    + dfs_adaptive_manual["Sex"].notna().astype(int)
).max() == 1
dfs_adaptive_manual["Sex"].fillna(dfs_adaptive_manual["sex"], inplace=True)

# %%
# ethnicity and Ethnicity columns are never set at the same time, so we can fillna to combine
assert (
    dfs_adaptive_manual["ethnicity"].notna().astype(int)
    + dfs_adaptive_manual["Ethnicity"].notna().astype(int)
).max() == 1
dfs_adaptive_manual["Ethnicity"].fillna(dfs_adaptive_manual["ethnicity"], inplace=True)

# %%
# Race and Ethnicity are indeed set together, so we need fancier logic to combine
assert (
    dfs_adaptive_manual["Race"].notna().astype(int)
    + dfs_adaptive_manual["Ethnicity"].notna().astype(int)
).max() > 1


def _combine_race_ethnicity_cols(row):
    non_empty = row.dropna()
    if len(non_empty) == 0:
        return np.nan
    return " - ".join(non_empty)


dfs_adaptive_manual["Ethnicity"] = dfs_adaptive_manual[["Race", "Ethnicity"]].apply(
    _combine_race_ethnicity_cols, axis=1
)
dfs_adaptive_manual["Ethnicity"].value_counts()

# %%
# Subset to these surviving columns
dfs_adaptive_manual = dfs_adaptive_manual[
    [
        "study",
        "sample_name",
        "locus",
        "counting_method",
        "disease",
        "sequencing_type",
        "disease_subtype",
        "participant_label",
        "specimen_label",
        "amplification_label",
        "replicate_label",
        # "cell_type",  # optional
        "Sex",
        "Age",
        "Ethnicity",
        "specimen_description",
        # "participant_description",
        "primer_set",
    ]
    + symptoms_columns
].rename(columns={col: f"symptoms_{col}" for col in symptoms_columns})
dfs_adaptive_manual

# %%
all_specimens = dfs_adaptive_manual

# %%

# %% [markdown]
# # Make metadata columns consistent with standard Boydlab pipeline

# %%
all_specimens = all_specimens.rename(
    columns={
        "Age": "age",
        "Sex": "sex",
        "Ethnicity": "ethnicity",
        "study": "study_name",
    }
).assign(has_BCR=False, has_TCR=True)

# %%

# %%
all_specimens["sex"].value_counts()

# %%
# Consolidate
all_specimens["sex"] = (
    all_specimens["sex"].str.upper().str.strip().replace({"MALE": "M", "FEMALE": "F"})
)
all_specimens["sex"].mask(all_specimens["sex"] == "UNKNOWN", inplace=True)

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
all_specimens["ethnicity"].value_counts()

# %%
# Condense rare ethnicity names
all_specimens["ethnicity_condensed"] = (
    all_specimens["ethnicity"]
    .str.strip()
    .replace(
        {
            "Black - Non-Hispanic": "African",
            "Black or African American, Not Hispanic or Latino": "African",
            #
            "White, Hispanic or Latino": "Hispanic/Latino",
            "White - Hispanic": "Hispanic/Latino",
            "Hispanic - Hispanic": "Hispanic/Latino",
            "Unknown, Hispanic or Latino": "Hispanic/Latino",
            "Other, Hispanic or Latino": "Hispanic/Latino",
            #
            "Asian - Non-Hispanic": "Asian",
            "Asian, Not Hispanic or Latino": "Asian",
            #
            "American Indian or Alaska Native, Not Hispanic or Latino": "Native American",
            #
            "Native Hawaiian or other Pacific Islander, Not Hispanic or Latino": "Pacific Islander",
            #
            "White - Non-Hispanic": "Caucasian",
            "White, Not Hispanic or Latino": "Caucasian",
            #
            "Not reported - Not reported": np.nan,
            "Unknown": np.nan,
            "Other, Not Hispanic or Latino": np.nan,
            "Asian, Hispanic or Latino": np.nan,
        }
    )
)

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
# Remove "Unknown"
all_specimens["age"].mask(all_specimens["age"] == "Unknown", inplace=True)

# %%
# Now we can convert to float
all_specimens["age"] = all_specimens["age"].astype(float)

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
all_specimens["locus"].value_counts()

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
    config.paths.metadata_dir / "adaptive" / "generated.adaptive_external_cohorts.tsv",
    sep="\t",
    index=None,
)

# %%

# %%
