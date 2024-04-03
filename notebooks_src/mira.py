# %%
import numpy as np
import pandas as pd
from malid import config, etl, helpers, logger
from malid.datamodels import GeneLocus

# %% [markdown]
# # Load MIRA TCR Covid-19 known binder data, subset, and add our IgBlast parses

# %%
# combine datasets, but store row numbers in each one
df = pd.concat(
    [
        pd.read_csv(
            config.paths.external_raw_data
            / "immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/peptide-detail-ci.csv"
        )
        .assign(source="peptide-cI")
        .rename_axis(index="rownum")
        .reset_index(),
        pd.read_csv(
            config.paths.external_raw_data
            / "immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/peptide-detail-cii.csv"
        )
        .assign(source="peptide-cII")
        .rename_axis(index="rownum")
        .reset_index(),
        pd.read_csv(
            config.paths.external_raw_data
            / "immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/minigene-detail.csv"
        )
        .assign(source="minigene")
        .rename_axis(index="rownum")
        .reset_index(),
    ],
    axis=0,
)

# Merge in subject metadata
subject_metadata = pd.read_csv(
    config.paths.external_raw_data
    / "immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/subject-metadata.csv",
    encoding="unicode_escape",
)
df = pd.merge(
    df,
    subject_metadata,
    how="left",
    on="Experiment",
    validate="m:1",
)
assert not df["Cohort"].isna().any()

# Filter to Covid19 patient data
# We're excluding these cohorts: Healthy (No known exposure), COVID-19-B-Non-Acute, COVID-19-Exposed
df = df[df["Cohort"].isin(["COVID-19-Acute", "COVID-19-Convalescent"])]

# split bioidentity
assert not df["TCR BioIdentity"].isna().any()
df = pd.concat(
    [
        df,
        df["TCR BioIdentity"]
        .str.split("+", expand=True)
        .rename(columns={0: "cdr3_seq_aa_q_trim", 1: "v_gene", 2: "j_gene"}),
    ],
    axis=1,
)


# Trim CDR3: remove ends
# and replace field that's entirely space (or empty) with NaN
df["cdr3_seq_aa_q_trim"] = (
    df["cdr3_seq_aa_q_trim"]
    .str.slice(start=1, stop=-1)
    .replace(r"^\s*$", np.nan, regex=True)
)

# Add length
# Note: Adaptive data can have a length cutoff that will potentially exclude some longer CDR3s
df["cdr3_aa_sequence_trim_len"] = df["cdr3_seq_aa_q_trim"].str.len()

# %%
df

# %%

# %%
# Merge our IgBlast output

# %%
# We ran IgBlast ourselves using the "TCR Nucleotide Sequence" field
# Our IgBlast gives some different V gene calls, but generally doesn't provide CDR3 calls for these short sequences.

# Use the V/J gene calls from our IgBlast.
# Keep sequences called productive by our IgBlast.
# Use Adaptive CDR3 call.

parse_fnames = list(
    (
        config.paths.external_raw_data
        / "immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/splits"
    ).glob(f"*.fasta.part*.fasta.parse.txt.parsed.tsv")
)
len(parse_fnames)

# %%
if len(parse_fnames) == 0:
    raise ValueError(f"No igblast parse files found")

# %%
df_parse = pd.concat([pd.read_csv(fname, sep="\t") for fname in parse_fnames], axis=0)
len(df_parse), len(df)

# %%
# extract fasta ID
df_parse[["specimen_label", "rownum"]] = df_parse["id"].str.split("|", expand=True)
df_parse["rownum"] = df_parse["rownum"].astype(int)
for specimen_label, grp in df_parse.groupby("specimen_label"):
    assert not grp["rownum"].duplicated().any()

# %%
df_parse["specimen_label"].value_counts()

# %%
# create "source" column
df_parse["source"] = df_parse["specimen_label"].replace(
    {
        "peptide-detail-ci": "peptide-cI",
        "peptide-detail-cii": "peptide-cII",
        "minigene-detail": "minigene",
    }
)
df_parse["source"].value_counts()

# %%
orig_shape = df.shape
df = pd.merge(
    df.drop(columns=["v_gene", "j_gene"]),
    df_parse[["source", "rownum", "v_segment", "j_segment", "productive"]],
    left_on=["source", "rownum"],
    right_on=["source", "rownum"],
    how="inner",
    validate="1:1",
)
assert df.shape[0] == min(orig_shape[0], df_parse.shape[0])

# %%
df.head()

# %%

# %%
# Filter to TRB sequencing data by looking at V gene name
df = df[df["v_segment"].str.startswith("TRBV")]

# set isotype flag
df["extracted_isotype"] = "TCRB"

df.shape

# %%
df["productive"].value_counts()

# %%
# productive field is missing for one sequence, no big deal
df["productive"].isna().value_counts()

# %%
df.dropna(subset="productive", inplace=True)
df.shape

# %%
df = df[df["productive"]].copy()
df.shape

# %%
# compute important columns
# note that this converts v_segment, j_segment (with alleles) to v_gene, j_gene columns (no alleles).
df = etl._compute_columns(df=df, gene_locus=GeneLocus.TCR)
df.shape

# %%

# %%
# Deprecated logic for working with Adaptive's own V/J gene calls:

# # Filter to TRB
# df = df[df["v_gene"].str.startswith("TCRBV")]

# # Change from Adaptive V/J gene nomenclature to IMGT
# # See https://tcrdist3.readthedocs.io/en/latest/adaptive.html
# df["v_gene"] = df["v_gene"].apply(lambda vgene: adaptive_to_imgt["human"].get(vgene))
# df["j_gene"] = df["j_gene"].apply(lambda vgene: adaptive_to_imgt["human"].get(vgene))

# # Remove alleles
# df["v_gene"] = df["v_gene"].str.split("*").str[0]
# df["j_gene"] = df["j_gene"].str.split("*").str[0]

# # Drop N/A and duplicates
# df = df.dropna(subset=["v_gene", "j_gene", "cdr3_seq_aa_q_trim"]).drop_duplicates(
#     subset=["v_gene", "j_gene", "cdr3_seq_aa_q_trim"]
# )

# # Make categorical
# df["v_gene"] = df["v_gene"].astype("category")
# df["j_gene"] = df["j_gene"].astype("category")

# %%
# Downselect only to V genes that are in our data
# We will never get matches on the rest. Clustering will always fail.
invalid_v_genes = set(df["v_gene"].unique()) - set(
    helpers.all_observed_v_genes()[GeneLocus.TCR]
)
logger.warning(f"Dropping MIRA V genes that aren't in our data: {invalid_v_genes}")
df = df[df["v_gene"].isin(helpers.all_observed_v_genes()[GeneLocus.TCR])]

# %%

# %% [markdown]
# # Export

# %%
# Drop duplicates
print(df.shape)
df = df.drop_duplicates(subset=["v_gene", "j_gene", "cdr3_seq_aa_q_trim"])
print(df.shape)

# %%
df

# %%
df.columns

# %%
df.to_csv(
    config.paths.external_raw_data
    / "immunecode_all/mira/ImmuneCODE-MIRA-Release002.1"
    / "mira_combined.filtered.tsv",
    sep="\t",
    index=None,
)

# %%
