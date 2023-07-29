# %%
import pandas as pd
from malid import config, helpers, logger
from malid.datamodels import GeneLocus

# %%

# %%
df = pd.read_csv(config.paths.base_data_dir / "CoV-AbDab_260722.filtered.tsv", sep="\t")
df["rownum"] = range(df.shape[0])
df

# %%
df_parse = pd.concat(
    [
        pd.read_csv(fname, sep="\t")
        for fname in (config.paths.base_data_dir / "covabdab_igblast_splits").glob(
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
        columns={"CDR1-IMGT": "cdr1_seq_aa_q_trim", "CDR2-IMGT": "cdr2_seq_aa_q_trim"}
    )[["v_gene", "rownum", "cdr1_seq_aa_q_trim", "cdr2_seq_aa_q_trim"]],
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
# Note: we don't know mutation rate and we don't know isotype
# TODO: improve IgBlast parser to get mutation rate
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
df.columns

# %%
df.drop(columns="rownum").to_csv(
    config.paths.base_data_dir / "CoV-AbDab_260722.filtered.annotated.tsv",
    sep="\t",
    index=None,
)

# %%
