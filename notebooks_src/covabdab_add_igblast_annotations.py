# %%
import pandas as pd
from malid import config, helpers, logger
from malid.datamodels import GeneLocus

# %%
# Export positives and negatives together.

# %%
df = pd.read_csv(config.paths.base_data_dir / "CoV-AbDab_130623.filtered.tsv", sep="\t")
df["rownum"] = range(df.shape[0])
df

# %%
df_parse = pd.concat(
    [
        pd.read_csv(fname, sep="\t")
        for fname in (config.paths.base_data_dir / "cov_abdab_fasta_split").glob(
            "*.parsed.tsv"
        )
    ],
    axis=0,
)
# extract fasta ID
df_parse[["specimen_label", "rownum"]] = df_parse["query_id"].str.split(
    "|", expand=True
)
df_parse["rownum"] = df_parse["rownum"].astype(int)
for specimen_label, grp in df_parse.groupby("specimen_label"):
    assert not grp["rownum"].duplicated().any()
df_parse

# %%
df["rownum"].describe()

# %%
df_parse["rownum"].describe()

# %%

# %%

# %%
# Merge our IgBlast output
# Our IgBlast gives some different V gene calls, and provides CDR1+2 sequences

# %%
orig_shape = df.shape
df = pd.merge(
    df.drop(columns=["v_gene"]),
    df_parse.rename(
        columns={
            "CDR1-IMGT": "cdr1_seq_aa_q_trim",
            "CDR2-IMGT": "cdr2_seq_aa_q_trim",
            "FR1-IMGT": "fr1_seq_aa_q_trim",
            "FR2-IMGT": "fr2_seq_aa_q_trim",
            "FR3-IMGT": "fr3_seq_aa_q_trim",
            "global_percent_identity": "v_mut",
        }
    )[
        [
            "v_gene",
            "rownum",
            "fr1_seq_aa_q_trim",
            "cdr1_seq_aa_q_trim",
            "fr2_seq_aa_q_trim",
            "cdr2_seq_aa_q_trim",
            "fr3_seq_aa_q_trim",
            "v_mut",
        ]
    ],
    left_on=["rownum"],
    right_on=["rownum"],
    how="inner",
    validate="1:1",
)
assert df.shape[0] == min(orig_shape[0], df_parse.shape[0])

# %%
# convert percent identity to mutation rate
# note: this is the *amino acid mutation rate*, not the nucleotide mutation rate as we compute in main pipeline
# however, this is as good as we can get here.
df["v_mut"] = 1.0 - df["v_mut"] / 100.0
df["v_mut"].describe()

# %%
df.shape[0], orig_shape[0], df_parse.shape[0]

# %%
df.head()

# %%
# sanity check
assert df["v_gene"].str.startswith("IGHV").all()
assert (df["cdr1_seq_aa_q_trim"].str.strip() == df["cdr1_seq_aa_q_trim"]).all()
assert (df["cdr2_seq_aa_q_trim"].str.strip() == df["cdr2_seq_aa_q_trim"]).all()

# %%

# %%
df["v_gene"].value_counts()

# %%
df["v_gene"] = df["v_gene"].str.split("*").str[0].astype("category")

# %%
df["v_gene"].value_counts()

# %%

# %%
# Note: we don't know isotype
# TODO: can we subset to matches in our dataset, and get the true isotype?

# %%

# %%
# Downselect only to V genes that are in our data
# We will never get matches on the rest. Clustering will always fail.
invalid_v_genes = set(df["v_gene"].unique()) - set(
    helpers.all_observed_v_genes()[GeneLocus.BCR]
)
logger.warning(f"Dropping CoV-AbDab V genes that aren't in our data: {invalid_v_genes}")
df = df[df["v_gene"].isin(helpers.all_observed_v_genes()[GeneLocus.BCR])]

# %%

# %% [markdown]
# # Export

# %%
df

# %%
concatenated_parts_available_to_us = (
    df["fr1_seq_aa_q_trim"]
    + df["cdr1_seq_aa_q_trim"]
    + df["fr2_seq_aa_q_trim"]
    + df["cdr2_seq_aa_q_trim"]
    + df["fr3_seq_aa_q_trim"]
    + df["cdr3_seq_aa_q_trim"]
)

# %%

# %%
df.iloc[0]["VHorVHH"]

# %%
concatenated_parts_available_to_us.iloc[0]

# %%

# %%
# Create FR4 field that is VHorVHH minus the concatenation (we verified this with IgBlast manually)
# Basically igblastp does not give us the FR4 region.
# But for our modeling purposes, we can use VHorVHH as the concatenated string.

# %%
df["post_seq_aa_q_trim"] = [
    fullstring.replace(substring, "").strip()
    for fullstring, substring in zip(df["VHorVHH"], concatenated_parts_available_to_us)
]
df["post_seq_aa_q_trim"].head()

# %%

# %%

# %%
# However this approach can sometimes fail, if there's a difference between VHorVHH and the IgBlast fields up to FR3 + CDR3 field in IgBlast
# Below is an example where we see this failure. It looks like the difference is in the CDR3 region, which is provided by CoV-AbDab. Not sure why they have a discrepancy between their own fields.
# Reviewing this particular example in the original CoV-AbDab spreadsheet, we see that the VHorVHH sequence appears several times, and each row has a different CDRH3 annotation. One of the rows has a match between VHorVHH and CDRH3, so it is not flagged.
# We are safe to drop the flagged rows.

# %%
failed_subtractions = df["post_seq_aa_q_trim"] == df["VHorVHH"]
failed_subtractions.value_counts()

# %%
failed_subtractions

# %%
df[failed_subtractions]["post_seq_aa_q_trim"]

# %%
concatenated_parts_available_to_us.loc[failed_subtractions].iloc[0]

# %%
df.loc[failed_subtractions].iloc[0]["VHorVHH"]

# %%
df.loc[failed_subtractions].iloc[0]["CDRH3"]

# %%

# %%
# Drop the flagged rows.
df = df.loc[~failed_subtractions]

# %%
assert not (df["post_seq_aa_q_trim"] == df["VHorVHH"]).any()

# %%
df["post_seq_aa_q_trim"].head()

# %%
df["post_seq_aa_q_trim"].value_counts()

# %%

# %%

# %%
df.columns

# %%

# %%
df["Status"].value_counts()

# %%
assert not df["Status"].isna().any()

# %%

# %%
df.drop(columns="rownum").to_csv(
    config.paths.base_data_dir / "CoV-AbDab_130623.filtered.annotated.tsv",
    sep="\t",
    index=None,
)

# %%
