# %%
import pandas as pd
from malid import config
from malid.datamodels import healthy_label

# %%

# %%
base_dir = config.paths.external_raw_data / "Britanova"
external_metadata = pd.read_csv(base_dir / "metadata.txt", sep="\t").assign(
    ethnicity_condensed="Caucasian"
)
# The A* in sample identifier is the batch ID.
# Age and sex data is provided in the metadata.txt file.
# Samples having age "0" are umbilical cord blood samples.
external_metadata

# %%
external_metadata["label"].duplicated().any()

# %%
external_metadata["label"].value_counts()

# %%
external_metadata["label"].str.split("-").str[0].duplicated().any()

# %%
external_metadata["label"].str.split("-").str[0].value_counts()

# %%
# p53, p76 suspicious that years apart. but checks out against paper 2's explanation of two time points.
# the other two must be the two individuals listed in the paper with replicate samples - yes the ages match
external_metadata[
    external_metadata["label"].str.split("-").str[0].duplicated(keep=False)
]

# %%
external_metadata["participant_label"] = (
    external_metadata["label"].str.split("-").str[0]
)
external_metadata

# %%
assert (external_metadata["file_name"] == external_metadata["sample_id"] + ".txt").all()

# %%
external_metadata["..filter.."].value_counts()

# %%

# %%
# get rid of extreme ages like cord blood
external_metadata["age"].describe()

# %%
external_metadata = external_metadata[
    (external_metadata["age"] >= 20) & (external_metadata["age"] <= 80)
]
external_metadata["age"].describe()

# %%

# %%
external_metadata = external_metadata.rename(
    columns={"sample_id": "specimen_label"}
).assign(
    disease=healthy_label,
    study_name="Britanova",
    disease_subtype=f"{healthy_label} - Britanova",
)
external_metadata

# %%

# %%
external_metadata_export = external_metadata[
    [
        "specimen_label",
        "sex",
        "age",
        "ethnicity_condensed",
        "participant_label",
        "disease",
        "study_name",
        "disease_subtype",
    ]
]
external_metadata_export

# %%
# remember, some patients have multiple samples as we investigated above
assert external_metadata_export["participant_label"].duplicated().any()
external_metadata_export[
    external_metadata_export["participant_label"].duplicated(keep=False)
].sort_values("participant_label")

# %%
# but each specimen listed once
assert not external_metadata_export["specimen_label"].duplicated().any()

# %%

# %%
external_metadata_export.to_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.healthy_tcr_britanova.participant_metadata.tsv",
    sep="\t",
    index=None,
)

# %%

# %%
