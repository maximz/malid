# %%
import pandas as pd

# %%
from malid import config, helpers
from malid.datamodels import GeneLocus

# %%
# cov-abdab
cov_abdab_all = pd.read_csv(config.paths.base_data_dir / "CoV-AbDab_130623.csv")
cov_abdab_all.shape

# %%

# %% [markdown]
# # Negatives

# %%
cov_abdab_negatives = cov_abdab_all.copy()
cov_abdab_negatives.dropna(subset=["Doesn't Bind to"], inplace=True)
cov_abdab_negatives.shape

# %%
# note: there seem to be a lot of "doesn't bind to sars-cov2" but still "neutralising vs sars-cov2" entries, so we screen those out.
cov_abdab_negatives = cov_abdab_negatives[
    (
        cov_abdab_negatives["Doesn't Bind to"]
        .str.lower()
        .apply(lambda s: "sars-cov2" in s)
    )
    & ~(
        cov_abdab_negatives["Binds to"]
        .fillna("")
        .str.lower()
        .apply(lambda s: "sars-cov2" in s)
    )
    & ~(
        cov_abdab_negatives["Neutralising Vs"]
        .fillna("")
        .str.lower()
        .apply(lambda s: "sars-cov2" in s)
    )
]
cov_abdab_negatives.shape

# %%
# Most of the remainder don't have any any positive binding information
cov_abdab_negatives["Binds to"].isna().value_counts()

# %%
# What do these remainder bind to, if listed?
cov_abdab_negatives["Binds to"].str.split(",|;").explode().value_counts()

# %%
# What don't they bind to?
cov_abdab_negatives["Doesn't Bind to"].str.split(",|;").explode().value_counts()

# %%

# %%
cov_abdab_negatives["Heavy V Gene"].str.split("(").str[1].value_counts()

# %%
cov_abdab_negatives["Heavy V Gene"].isna().value_counts()

# %%
cov_abdab_negatives.dropna(subset=["Heavy V Gene"], inplace=True)

# %%
cov_abdab_negatives = cov_abdab_negatives[
    cov_abdab_negatives["Heavy V Gene"].apply(lambda s: "(Human)" in s)
]
cov_abdab_negatives.shape

# %%

# %%
cov_abdab_negatives["Heavy J Gene"].value_counts()

# %%
cov_abdab_negatives["Heavy J Gene"].isna().value_counts()

# %%
cov_abdab_negatives = cov_abdab_negatives[cov_abdab_negatives["Heavy J Gene"] != "ND"]
cov_abdab_negatives.shape

# %%
cov_abdab_negatives.dropna(subset=["Heavy J Gene"], inplace=True)

# %%
cov_abdab_negatives["Heavy J Gene"].value_counts()

# %%

# %%
cov_abdab_negatives["Heavy V Gene"].value_counts()

# %%
cov_abdab_negatives["Heavy V Gene"] = (
    cov_abdab_negatives["Heavy V Gene"].str.split("(").str[0].str.strip()
)

# %%
cov_abdab_negatives["Heavy V Gene"]

# %%
cov_abdab_negatives["Heavy J Gene"].value_counts()

# %%
cov_abdab_negatives["Heavy J Gene"] = (
    cov_abdab_negatives["Heavy J Gene"].str.split("(").str[0].str.strip()
)

# %%
cov_abdab_negatives["Heavy J Gene"]

# %%

# %%
cov_abdab_negatives = cov_abdab_negatives[cov_abdab_negatives["CDRH3"] != "ND"]
cov_abdab_negatives.shape

# %%
cov_abdab_negatives["CDRH3"]

# %%

# %%
cov_abdab_negatives["Ab or Nb"].value_counts()

# %%

# %%
cov_abdab_negatives["VHorVHH"].isna().value_counts()

# %%
(cov_abdab_negatives["VHorVHH"] != "ND").value_counts()

# %%
cov_abdab_negatives = cov_abdab_negatives[cov_abdab_negatives["VHorVHH"] != "ND"]
cov_abdab_negatives.shape

# %%

# %%
cov_abdab_negatives.columns

# %%
cov_abdab_negatives["Origin"].value_counts()

# %%
# Confirm all are of human origin
assert cov_abdab_negatives["Origin"].str.contains("Human").all()

# %%

# %%
cov_abdab_negatives.shape

# %%

# %%
cov_abdab_negatives["Protein + Epitope"].value_counts()

# %%
cov_abdab_negatives["Binds to"].value_counts()

# %%
cov_abdab_negatives["Doesn't Bind to"].value_counts()

# %%
cov_abdab_negatives["Not Neutralising Vs"].value_counts()

# %%
cov_abdab_negatives["Neutralising Vs"].value_counts()

# %%

# %% [markdown]
# # Positives

# %%
cov_abdab_positives = cov_abdab_all.copy()
cov_abdab_positives = cov_abdab_positives[
    (
        (
            cov_abdab_positives["Binds to"]
            .fillna("")
            .str.lower()
            .apply(lambda s: "sars-cov2" in s)
        )
        | (
            cov_abdab_positives["Neutralising Vs"]
            .fillna("")
            .str.lower()
            .apply(lambda s: "sars-cov2" in s)
        )
    )
    # avoid entries where the binding was selective for a particular strain of SARS-CoV-2:
    & ~(
        cov_abdab_positives["Doesn't Bind to"]
        .fillna("")
        .str.lower()
        .apply(lambda s: "sars-cov2" in s)
    )
]
cov_abdab_positives.shape

# %%
cov_abdab_positives["Binds to"].value_counts()

# %%
# most have positive binding information
cov_abdab_positives["Binds to"].isna().value_counts()

# %%
# neutralizing information is present for only about half
cov_abdab_positives["Neutralising Vs"].isna().value_counts()

# %%
# What do these sequence bind to, if listed?
cov_abdab_positives["Binds to"].str.split(",|;").explode().value_counts()

# %%
# What do these sequence neutralize, if listed?
cov_abdab_positives["Neutralising Vs"].str.split(",|;").explode().value_counts()

# %%

# %%
# # remove weak binders
# cov_abdab_positives = cov_abdab_positives[
#     ~cov_abdab_positives["Binds to"].str.lower().apply(lambda s: "sars-cov2_wt (weak)" in s)
# ]
# cov_abdab_positives.shape

# %%

# %%
cov_abdab_positives["Binds to"].value_counts()

# %%

# %%
# cov_abdab_positives.dropna(subset=["Neutralising Vs"], inplace=True)
# cov_abdab_positives.shape

# %%
cov_abdab_positives["Neutralising Vs"].isna().value_counts()

# %%
# cov_abdab_positives = cov_abdab_positives[
#     cov_abdab_positives["Neutralising Vs"].str.lower().apply(lambda s: "sars-cov2" in s)
# ]
# cov_abdab_positives.shape

# %%
cov_abdab_positives["Neutralising Vs"].value_counts()

# %%

# %%
cov_abdab_positives["Heavy V Gene"].str.split("(").str[1].value_counts()

# %%
cov_abdab_positives["Heavy V Gene"].isna().value_counts()

# %%
cov_abdab_positives.dropna(subset=["Heavy V Gene"], inplace=True)

# %%
cov_abdab_positives = cov_abdab_positives[
    cov_abdab_positives["Heavy V Gene"].apply(lambda s: "(Human)" in s)
]
cov_abdab_positives.shape

# %%

# %%
cov_abdab_positives["Heavy J Gene"].value_counts()

# %%
cov_abdab_positives["Heavy J Gene"].isna().value_counts()

# %%
cov_abdab_positives = cov_abdab_positives[cov_abdab_positives["Heavy J Gene"] != "ND"]
cov_abdab_positives.shape

# %%
cov_abdab_positives.dropna(subset=["Heavy J Gene"], inplace=True)

# %%
cov_abdab_positives["Heavy J Gene"].value_counts()

# %%

# %%
cov_abdab_positives["Heavy V Gene"].value_counts()

# %%
cov_abdab_positives["Heavy V Gene"] = (
    cov_abdab_positives["Heavy V Gene"].str.split("(").str[0].str.strip()
)

# %%
cov_abdab_positives["Heavy V Gene"]

# %%
cov_abdab_positives["Heavy J Gene"].value_counts()

# %%
cov_abdab_positives["Heavy J Gene"] = (
    cov_abdab_positives["Heavy J Gene"].str.split("(").str[0].str.strip()
)

# %%
cov_abdab_positives["Heavy J Gene"]

# %%

# %%
cov_abdab_positives = cov_abdab_positives[cov_abdab_positives["CDRH3"] != "ND"]
cov_abdab_positives.shape

# %%
cov_abdab_positives["CDRH3"]

# %%

# %%
cov_abdab_positives["Ab or Nb"].value_counts()

# %%

# %%
cov_abdab_positives["VHorVHH"].isna().value_counts()

# %%
(cov_abdab_positives["VHorVHH"] != "ND").value_counts()

# %%
cov_abdab_positives = cov_abdab_positives[cov_abdab_positives["VHorVHH"] != "ND"]
cov_abdab_positives.shape

# %%

# %%
cov_abdab_positives.columns

# %%
cov_abdab_positives["Origin"].value_counts()

# %%
cov_abdab_positives["Origin"].value_counts().head(n=25)

# %%
cov_abdab_positives["Origin"].unique()

# %%
# Test "keep human origin only" filter:
[
    origin
    for origin in cov_abdab_positives["Origin"].fillna("").unique()
    if (
        "human" in origin.lower()
        or "patient" in origin.lower()
        or "vaccinee" in origin.lower()
        or "breakthrough infection" in origin.lower()
    )
    and "humanised" not in origin.lower()
    and "phage display" not in origin.lower()
    and "synthetic" not in origin.lower()
]

# %%
# rejects:
[
    origin
    for origin in cov_abdab_positives["Origin"].fillna("").unique()
    if not (
        (
            "human" in origin.lower()
            or "patient" in origin.lower()
            or "vaccinee" in origin.lower()
            or "breakthrough infection" in origin.lower()
        )
        and "humanised" not in origin.lower()
        and "phage display" not in origin.lower()
        and "synthetic" not in origin.lower()
    )
]

# %%

# %%
cov_abdab_positives.shape

# %%
# Apply "keep human origin only" filter:
cov_abdab_positives = cov_abdab_positives[
    cov_abdab_positives["Origin"]
    .fillna("")
    .apply(
        lambda origin: (
            "human" in origin.lower()
            or "patient" in origin.lower()
            or "vaccinee" in origin.lower()
            or "breakthrough infection" in origin.lower()
        )
        and "humanised" not in origin.lower()
        and "phage display" not in origin.lower()
        and "synthetic" not in origin.lower()
    )
]
cov_abdab_positives.shape

# %%

# %%
cov_abdab_positives["Origin"].value_counts()

# %%

# %%

# %%

# %%
cov_abdab_positives["Protein + Epitope"].value_counts()

# %%
cov_abdab_positives["Binds to"].value_counts()

# %%
cov_abdab_positives["Doesn't Bind to"].value_counts()

# %%
# Notice we have not required positives to be neutralizing against SARS-CoV-2.
cov_abdab_positives["Not Neutralising Vs"].value_counts()

# %%

# %% [markdown]
# # Combine positives and negatives, then export

# %%
cov_abdab_export = (
    pd.concat(
        [
            cov_abdab_positives.assign(Status="Positive"),
            cov_abdab_negatives.assign(Status="Negative"),
        ],
        axis=0,
    )[
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
            "Status",
        ]
    ]
    .rename(columns={"Heavy J Gene": "j_gene", "Heavy V Gene": "v_gene"})
    .reset_index(drop=True)
)
cov_abdab_export

# %%
cov_abdab_export["Status"].value_counts()

# %%

# %%

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
# drop duplicate sequences with same V gene, J gene, CDR3 sequence, and positive/negative status
cov_abdab_export = cov_abdab_export.drop_duplicates(
    subset=["v_gene", "j_gene", "cdr3_seq_aa_q_trim", "Status"]
)
cov_abdab_export

# %%

# %%
# However, there are some identical sequences with both positive and negative status!
# These may be due to sequence differences outside the CDR3 (elsewhere in VHorVHH)
cov_abdab_export.groupby(["v_gene", "j_gene", "cdr3_seq_aa_q_trim"]).size()[
    cov_abdab_export.groupby(["v_gene", "j_gene", "cdr3_seq_aa_q_trim"]).size() != 1
]

# %%
# Drop these conflicting sequences:
print(cov_abdab_export.shape)
cov_abdab_export = cov_abdab_export[
    cov_abdab_export.groupby(["v_gene", "j_gene", "cdr3_seq_aa_q_trim"])[
        "Status"
    ].transform(lambda grp: grp.nunique() == 1)
]
print(cov_abdab_export.shape)

# %%
assert (
    not cov_abdab_export[["v_gene", "j_gene", "cdr3_seq_aa_q_trim"]]
    .duplicated(keep=False)
    .any()
)

# %%

# %%

# %%

# %%

# %%
cov_abdab_export["Sources"].value_counts()

# %%

# %%
cov_abdab_export.drop(["Sources"], axis=1).to_csv(
    config.paths.base_data_dir / "CoV-AbDab_130623.filtered.tsv", sep="\t", index=None
)

# %%
