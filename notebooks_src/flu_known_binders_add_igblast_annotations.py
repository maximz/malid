# %%
import numpy as np
import pandas as pd
from malid import config, helpers, logger
from malid.datamodels import GeneLocus

# %%
from malid import get_v_sequence, etl

# %%

# %%
df = pd.read_csv(
    config.paths.base_data_dir / "flu_known_binders.filtered.tsv", sep="\t"
)
df["rownum"] = range(df.shape[0])
df

# %%
df_parse = pd.concat(
    [
        pd.read_csv(fname, sep="\t")
        for fname in (
            config.paths.base_data_dir / "flu_known_binders_fasta_split"
        ).glob("*.parsed.IgH.tsv")
    ],
    axis=0,
)
# extract fasta ID
df_parse[["specimen_label", "rownum"]] = df_parse["id"].str.split("|", expand=True)
df_parse["rownum"] = df_parse["rownum"].astype(int)
for specimen_label, grp in df_parse.groupby("specimen_label"):
    assert not grp["rownum"].duplicated().any()
df_parse

# %%
df["rownum"].describe()

# %%
df_parse["rownum"].describe()

# %%
df_parse.columns

# %%
# Productive is a bool with NaNs
print(df_parse["productive"].dtype)
print(df_parse["productive"].value_counts())
print(df_parse["productive"].isna().sum())
# Cast to bool fully by filling NaNs
df_parse["productive"] = df_parse["productive"].fillna(False).astype(bool)
df_parse["productive"].dtype

# %%
df_parse["productive"]

# %%

# %%
df.iloc[0]["VH_nuc"]

# %%
df_parse[["cdr3_seq_aa_q", "post_seq_aa_q"]]

# %%
# In default etl for Adaptive TCR, we ran _split_post_seq_into_cdr3_and_fwr4 to post-process the FWR4 to remove the CDR3 prefix, because IgBlast mis-annotated the CDR3+FWR4 as "post_seq" together.
# Splitting them apart is not necessary here, based on our inspection. These appear to be properly parsed full length sequences.

# %%

# %%
# get v_sequence (same way we produce v_sequence in internal pipeline's sort script)
# this will be used to compute v_mut for BCR
(
    df_parse["v_sequence"],
    df_parse["d_sequence"],
    df_parse["j_sequence"],
) = get_v_sequence.complete_sequences(df_parse)

# %%
df_parse["v_sequence"].iloc[0]

# %%

# %%

# %%
df.columns

# %%

# %%
# Merge our IgBlast output

# %%
orig_shape = df.shape
df = pd.merge(
    df.drop(columns=["v_gene", "j_gene"]).rename(
        columns={
            "cdr3_seq_aa_q_trim": "original_cdr3_seq_aa_q_trim",
            "cdr3_aa_sequence_trim_len": "original_cdr3_aa_sequence_trim_len",
        }
    ),
    df_parse[
        [
            "v_segment",
            "j_segment",
            "productive",
            "rownum",
            "pre_seq_aa_q",
            "fr1_seq_aa_q",
            "cdr1_seq_aa_q",
            "fr2_seq_aa_q",
            "cdr2_seq_aa_q",
            "fr3_seq_aa_q",
            "cdr3_seq_aa_q",
            "post_seq_aa_q",  # this is FR4
            "v_sequence",
        ]
    ],
    left_on=["rownum"],
    right_on=["rownum"],
    how="inner",
    validate="1:1",
)
assert df.shape[0] == min(orig_shape[0], df_parse.shape[0])

# %%
df.shape[0], orig_shape[0], df_parse.shape[0]

# %%
df.head()

# %%

# %%

# %%
# Create these columns for consistency.
df["locus"] = "IGH"
df["extracted_isotype"] = np.nan  # We don't know isotype in this situation

# %%

# %%
df.shape

# %%
# Discard any non-productive sequences
# Normally we also discard null isotypes but here we don't know isotype
# df = df.loc[(~pd.isnull(df["extracted_isotype"])) & (df["productive"] == True)]
df = df.loc[df["productive"] == True]
df.shape

# %%
df = etl._compute_columns(df, GeneLocus.BCR)
df.shape

# %%
df.head()

# %%

# %%
# Sanity check

# %%
df[["original_cdr3_seq_aa_q_trim", "cdr3_seq_aa_q_trim"]]

# %%
df[["original_cdr3_aa_sequence_trim_len", "cdr3_aa_sequence_trim_len"]]

# %%
(df["original_cdr3_seq_aa_q_trim"] == df["cdr3_seq_aa_q_trim"]).value_counts()

# %%

# %%
# Drop original CDR3s. We prefer our IgBlast. Anyway they're all consistent.

# %%
df.drop(
    columns=["original_cdr3_aa_sequence_trim_len", "original_cdr3_seq_aa_q_trim"],
    inplace=True,
)

# %%

# %%
# sanity check
assert df["v_gene"].str.startswith("IGHV").all()
assert (df["cdr1_seq_aa_q_trim"].str.strip() == df["cdr1_seq_aa_q_trim"]).all()
assert (df["cdr2_seq_aa_q_trim"].str.strip() == df["cdr2_seq_aa_q_trim"]).all()

# %%

# %%
df["v_gene"].value_counts()

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
df.columns

# %%
df.drop(columns="rownum").to_csv(
    config.paths.base_data_dir / "flu_known_binders.filtered.annotated.tsv",
    sep="\t",
    index=None,
)

# %%
