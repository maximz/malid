# %% [markdown]
# # Flu known binders from https://www.biorxiv.org/content/10.1101/2023.09.11.557288v1
#
# but exclude mouse data and human H7N9 vaccine data

# %%
import pandas as pd

# %%
from malid import config, helpers
from malid.datamodels import GeneLocus

# %%
# Flu known binders
df = pd.read_csv(config.paths.base_data_dir / "flu_known_binders.csv")
df.shape

# %%
df.head()

# %%
df.dropna(subset=["Binds to", "VH_nuc"], inplace=True)
df.shape

# %%
df["Binds to"].value_counts()

# %%
df["Specificity"].value_counts()

# %%
df["Specificity"].unique()

# %%
df["Donor Status"].unique()

# %%
df["Donor Status"].value_counts()

# %%
valid_sources = [
    "B-cells;Influenza Human Vaccinee",
    "vaccinated human donors",
    "human vaccinees",
    "Vaccinated with trivalent inactivated influenza vaccine for the 2006-2007 season",
    "Vaccinated with 2015–2016 Flucelvax",
    "Vaccinated with 2015–2016 FluBlok",
    "2015 IIV3 vaccination",
    "Vaccination",
    "Vaccinated",
    "healthy donors 7 d after vaccination with a seasonal influenza vaccine",
    "heathy donor with trivalent inactivated influenza vaccine (2006-2007)",
    "Healthy donor",
    "Infection",
    "Healthy donors",  # this probably means vaccinated with influenza vaccine
    "B-cells;Influenza natural infection or vaccination",
    "ten normal healthy donors",
    "B-cells;Influenza patient",
    "convalescent patients with confirmed infections with pdmH1N1",
]

# %%
# These will be excluded:
set(df["Donor Status"].str.strip()) - set(valid_sources)

# %%
df = df.loc[df["Donor Status"].str.strip().isin(valid_sources)]
df.shape

# %%
df["Donor Status"] = df["Donor Status"].str.strip()

# %%
df["Donor Status"].value_counts()

# %%
df["Binds to"].value_counts()

# %%
df["Specificity"].value_counts()

# %%
# Exclude H5 - H5N1 is not typically included in seasonal flu vaccine
df["Specificity"] = df["Specificity"].str.strip()
df = df.loc[df["Specificity"] != "H5"]
df.shape

# %%
df

# %%

# %%

# %%
df["Heavy_V_gene"].value_counts()

# %%
df["Heavy_V_gene"].isna().value_counts()

# %%
(df["Heavy_V_gene"].str.strip() == "").value_counts()

# %%
df["CDRH3_AA"].isna().value_counts()

# %%
(df["CDRH3_AA"].str.strip() == "").value_counts()

# %%

# %%
df.columns

# %%

# %%
df["CDRH3_AA"].head(n=25)

# %%
df["CDRH3_AA"].tail(n=25)

# %%

# %% [markdown]
# CDRH3 already has `C` prefix and `W` suffix removed - consistent with our internal data.

# %%

# %%
# compute cdr3_aa_sequence_trim_len

# %%
df["cdr3_seq_aa_q_trim"] = (
    df["CDRH3_AA"]
    .str.replace(".", "", regex=False)
    .str.replace("-", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace("*", "", regex=False)
    .str.strip()
    .str.upper()
)
df["cdr3_aa_sequence_trim_len"] = df["cdr3_seq_aa_q_trim"].str.len()
df

# %%

# %%

# %%
df_export = df.rename(
    columns={"Heavy_J_gene": "j_gene", "Heavy_V_gene": "v_gene"}
).reset_index(drop=True)
df_export

# %%

# %%
df_export = df_export.drop_duplicates(subset=["v_gene", "j_gene", "cdr3_seq_aa_q_trim"])
df_export

# %%

# %%

# %%

# %%

# %%
df_export.to_csv(
    config.paths.base_data_dir / "flu_known_binders.filtered.tsv", sep="\t", index=None
)

# %%
