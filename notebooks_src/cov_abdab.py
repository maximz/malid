# %%
import pandas as pd

# %%
from malid import config, helpers
from malid.datamodels import GeneLocus

# %%
# cov-abdab
cov_abdab = pd.read_csv(config.paths.base_data_dir / "CoV-AbDab_260722.csv")
cov_abdab.shape

# %%
cov_abdab.dropna(subset=["Binds to"], inplace=True)
cov_abdab.shape

# %%
cov_abdab = cov_abdab[
    cov_abdab["Binds to"].str.lower().apply(lambda s: "sars-cov2" in s)
]
cov_abdab.shape

# %%
cov_abdab["Binds to"].value_counts()

# %%
# # remove weak binders
# cov_abdab = cov_abdab[
#     ~cov_abdab["Binds to"].str.lower().apply(lambda s: "sars-cov2_wt (weak)" in s)
# ]
# cov_abdab.shape

# %%

# %%
cov_abdab["Binds to"].value_counts()

# %%

# %%
# cov_abdab.dropna(subset=["Neutralising Vs"], inplace=True)
# cov_abdab.shape

# %%
cov_abdab["Neutralising Vs"].isna().value_counts()

# %%
# cov_abdab = cov_abdab[
#     cov_abdab["Neutralising Vs"].str.lower().apply(lambda s: "sars-cov2" in s)
# ]
# cov_abdab.shape

# %%
cov_abdab["Neutralising Vs"].value_counts()

# %%

# %%
cov_abdab["Heavy V Gene"].str.split("(").str[1].value_counts()

# %%
cov_abdab["Heavy V Gene"].isna().value_counts()

# %%
cov_abdab.dropna(subset=["Heavy V Gene"], inplace=True)

# %%
cov_abdab = cov_abdab[cov_abdab["Heavy V Gene"].apply(lambda s: "(Human)" in s)]
cov_abdab.shape

# %%

# %%
cov_abdab["Heavy J Gene"].value_counts()

# %%
cov_abdab["Heavy J Gene"].isna().value_counts()

# %%
cov_abdab = cov_abdab[cov_abdab["Heavy J Gene"] != "ND"]
cov_abdab.shape

# %%
cov_abdab.dropna(subset=["Heavy J Gene"], inplace=True)

# %%
cov_abdab["Heavy J Gene"].value_counts()

# %%

# %%
cov_abdab["Heavy V Gene"].value_counts()

# %%
cov_abdab["Heavy V Gene"] = cov_abdab["Heavy V Gene"].str.split("(").str[0].str.strip()

# %%
cov_abdab["Heavy V Gene"]

# %%
cov_abdab["Heavy J Gene"].value_counts()

# %%
cov_abdab["Heavy J Gene"] = cov_abdab["Heavy J Gene"].str.split("(").str[0].str.strip()

# %%
cov_abdab["Heavy J Gene"]

# %%

# %%
cov_abdab = cov_abdab[cov_abdab["CDRH3"] != "ND"]
cov_abdab.shape

# %%
cov_abdab["CDRH3"]

# %%

# %%
cov_abdab["Ab or Nb"].value_counts()

# %%

# %%
cov_abdab["VHorVHH"].isna().value_counts()

# %%
(cov_abdab["VHorVHH"] != "ND").value_counts()

# %%
cov_abdab = cov_abdab[cov_abdab["VHorVHH"] != "ND"]
cov_abdab.shape

# %%

# %%
cov_abdab.columns

# %%
cov_abdab["Origin"].value_counts()

# %%
cov_abdab["Origin"].value_counts().head(n=25)

# %%
cov_abdab["Origin"].unique()

# %%
# Test "keep human origin only" filter:
[
    origin
    for origin in cov_abdab["Origin"].unique()
    if (
        "human" in origin.lower()
        or "patient" in origin.lower()
        or "vaccinee" in origin.lower()
    )
    and "humanised" not in origin.lower()
]

# %%
# rejects:
[
    origin
    for origin in cov_abdab["Origin"].unique()
    if not (
        (
            "human" in origin.lower()
            or "patient" in origin.lower()
            or "vaccinee" in origin.lower()
        )
        and "humanised" not in origin.lower()
    )
]

# %%

# %%
cov_abdab.shape

# %%
# Apply "keep human origin only" filter:
cov_abdab = cov_abdab[
    cov_abdab["Origin"].apply(
        lambda origin: (
            "human" in origin.lower()
            or "patient" in origin.lower()
            or "vaccinee" in origin.lower()
        )
        and "humanised" not in origin.lower()
    )
]
cov_abdab.shape

# %%

# %%
cov_abdab["Origin"].value_counts()

# %%

# %%

# %%

# %%
cov_abdab["Protein + Epitope"].value_counts()

# %%
cov_abdab["Binds to"].value_counts()

# %%
cov_abdab["Doesn't Bind to"].value_counts()

# %%
cov_abdab["Not Neutralising Vs"].value_counts()

# %%

# %%

# %%
cov_abdab_export = (
    cov_abdab[
        [
            "CDRH3",
            "Heavy J Gene",
            "Heavy V Gene",
            "VHorVHH",
            "Binds to",
            "Doesn't Bind to",
            "Neutralising Vs",
            "Not Neutralising Vs",
            "Protein + Epitope",
            "Origin",
            "Sources",
        ]
    ]
    .rename(columns={"Heavy J Gene": "j_gene", "Heavy V Gene": "v_gene"})
    .reset_index(drop=True)
)
cov_abdab_export

# %% [markdown]
# CDRH3 already has `C` prefix and `W` suffix removed - consistent with our internal data.

# %%

# %%
# compute cdr3_aa_sequence_trim_len

# %%
cov_abdab_export["cdr3_seq_aa_q_trim"] = (
    cov_abdab_export["CDRH3"]
    .str.replace(".", "", regex=False)
    .str.replace("-", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace("*", "", regex=False)
    .str.strip()
    .str.upper()
)
cov_abdab_export["cdr3_aa_sequence_trim_len"] = cov_abdab_export[
    "cdr3_seq_aa_q_trim"
].str.len()
cov_abdab_export

# %%
cov_abdab_export = cov_abdab_export.drop_duplicates(
    subset=["v_gene", "j_gene", "cdr3_seq_aa_q_trim"]
)
cov_abdab_export

# %%

# %%

# %%

# %%
cov_abdab_export["Sources"].value_counts()

# %%

# %%
cov_abdab_export.drop(["Sources"], axis=1).to_csv(
    config.paths.base_data_dir / "CoV-AbDab_260722.filtered.tsv", sep="\t", index=None
)

# %%
