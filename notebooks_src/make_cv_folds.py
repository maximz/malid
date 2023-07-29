# %% [markdown]
# # Filter to peak timepoints + Make cv divisions.
#
# Already removed specimens with very few sequences or without all isotypes. Already sampled one sequence per clone per isotype per specimen.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from malid import config, helpers, logger

# %%
import dask
import dask.dataframe as dd

# %%

# %%
from dask.distributed import Client

# multi-processing backend
# access dashbaord at http://127.0.0.1:61083
client = Client(
    scheduler_port=61084,
    dashboard_address=":61083",
    n_workers=4,
    processes=True,
    threads_per_worker=8,
    memory_limit="125GB",  # per worker
)
display(client)
# for debugging: client.restart()

# %%
# Don't use fastparquet, because it changes specimen labels like M54-049 to 2049-01-01 00:00:54 -- i.e. it coerces partition names to numbers or dates
df = dd.read_parquet(config.paths.sequences_sampled, engine="pyarrow")
df

# %%
# each partition is a specimen
df.npartitions

# %%
df.columns

# %%

# %%
# groupby participant, specimen, disease - get total sequence count
specimens = (
    df.groupby(
        ["participant_label", "specimen_label", "disease", "disease_subtype"],
        observed=True,
    )
    .size()
    .rename("total_sequence_count")
    .reset_index()
)
specimens

# %%

# %%
specimens = specimens.compute()
specimens

# %%
assert specimens.shape[0] == df.npartitions

# %%
assert not specimens["specimen_label"].duplicated().any()

# %%
# export list of specimens included in this full anndata
# not all specimens survived to this step - some are thrown out in the run_embedding notebooks for not having enough sequences or not having all isotypes
# but these aren't yet filtered to is_peak timepoints
specimens.to_csv(
    config.paths.dataset_specific_metadata / "specimens_kept_in_embedding_anndatas.tsv",
    sep="\t",
    index=None,
)

# %%

# %%

# %%
# filter to is_peak timepoints, in addition to the is_valid / survived_filters filter already here
specimen_metadata = helpers.get_all_specimen_info(
    # CV fold is not available yet, so must set this to False
    add_cv_fold_information=False
).sort_values("disease")
specimen_metadata = specimen_metadata[specimen_metadata["cohort"] == "Boydlab"]
specimen_metadata = specimen_metadata[specimen_metadata["in_training_set"]]

# sanity check the definitions
assert specimen_metadata["is_peak"].all()
assert specimen_metadata["survived_filters"].all()

specimen_metadata

# %%
# merge back to apply the filter
specimens_merged = pd.merge(
    specimens,
    specimen_metadata,
    on=["participant_label", "specimen_label", "disease", "disease_subtype"],
    how="inner",
    validate="1:1",
)
assert specimens_merged.shape[0] == specimen_metadata.shape[0] <= specimens.shape[0]
specimens_merged

# %%
specimens_merged = specimens_merged.assign(cohort="Boydlab")

# %%

# %%

# %%
tmp = (
    specimens_merged.drop_duplicates(["participant_label", "disease_subtype"])[
        "disease_subtype"
    ]
    .value_counts()
    .rename("number of participants")
)

tmp.to_csv(
    f"{config.paths.base_output_dir}/all_data_combined.participants_by_disease_subtype.tsv",
    sep="\t",
)

tmp

# %%
tmp = (
    specimens_merged.groupby(["disease", "cohort"], observed=True)[
        "total_sequence_count"
    ]
    .sum()
    .reset_index(name="number of sequences")
)

tmp.to_csv(
    f"{config.paths.base_output_dir}/all_data_combined.sequences_by_disease_and_cohort.tsv",
    sep="\t",
    index=None,
)

tmp

# %%

# %% [markdown]
# # Make CV splits
#
# **Strategy:**
#
# - Split on patients.
# - Split into (train+validation) / test first with a 2:1 ratio (because 3 folds total). Every patient is in 1 test fold. Then split (train+validation) into train (4/5) and validation (1/5).
# - Also create a "global fold" that has no test set, but has the same 4:1 train/validation ratio.
# - Stratified by disease label
#
# **How to handle varying gene loci:**
#
# * We want to include BCR+TCR samples as well as single-loci (e.g. BCR-only) samples
# * All data is used for single loci models. For example, the BCR sequence model and the BCR-only metamodel will include specimens whether they're BCR+TCR or BCR-only. (Any TCR-only would be excluded)
# * Only BCR+TCR samples are used in BCR+TCR metamodel. (The input BCR models will have been trained on any and all BCR data, but the second stage metamodel will be trained on only samples that have both BCR and TCR components.)
#
# The wrong way to design the cross validation split: split patients up all together, regardless of if they are BCR-only, TCR-only, or BCR+TCR.
#
# Example of why this is wrong: consider just one disease for now; suppose you have a BCR-only set of 3 patients and a BCR+TCR set of a different 3 patients. How do you split those 6 patients into 3 cross validation folds?
#
# The wrong strategy might split as follows:
#
# ```
# # wrong way - possible result
# how many
# included
# patients
#              fold 1              fold 2              fold 3
# BCR/TCR      3 train/0 test      2 train/1 test      1 train/2 test
# BCR only     1 train/2 test      2 train/1 test      3 train/0 test
# ```
#
# The BCR-only vs BCR+TCR are not spread evenly - we have imbalanced folds.
#
# **The right way to split would be: split BCR+TCR patients first, then separately split the BCR-only patients, and combine the resulting folds:**
#
# ```
# # right way - result
# how many
# included
# patients
#              fold 1              fold 2              fold 3
# BCR/TCR      2 train/1 test      2 train/1 test      2 train/1 test
# BCR only     2 train/1 test      2 train/1 test      2 train/1 test
# ```
#
#

# %%

# %%
unique_participants_all = (
    specimens_merged[
        [
            "participant_label",
            "disease",
            "cohort",
            "past_exposure",
            "disease.separate_past_exposures",
            "available_gene_loci",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)
unique_participants_all


# %%

# %%
def make_splits(participants, n_splits):
    # preserve imbalanced class distribution in each fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    folds = list(
        enumerate(
            cv.split(
                participants["participant_label"],
                participants["disease"],
            )
        )
    )
    return folds


# %%

# %%

# %%
fold_participants = []


for gene_locus, unique_participants in unique_participants_all.groupby(
    "available_gene_loci", observed=True, sort=False
):
    print(
        f"{gene_locus}: total number of unique participants {unique_participants.shape[0]}"
    )

    trainbig_test_splits = make_splits(unique_participants, n_splits=config.n_folds)
    assert len(trainbig_test_splits) == config.n_folds

    for (
        fold_id,
        (trainbig_participant_index, test_participant_index),
    ) in trainbig_test_splits:
        trainbig_participants = unique_participants.iloc[
            trainbig_participant_index
        ].reset_index(drop=True)
        test_participants = unique_participants.iloc[
            test_participant_index
        ].reset_index(drop=True)

        # confirm each patient is entirely on one or the other side of the train-big vs test divide
        assert (
            len(
                set(trainbig_participants["participant_label"].unique()).intersection(
                    set(test_participants["participant_label"].unique())
                )
            )
            == 0
        )

        # split trainbig into trainsmaller + validation
        mini_folds = make_splits(trainbig_participants, n_splits=3)
        # unpack first of the splits
        _, (train_smaller_index, validation_index) = mini_folds[0]

        train_smaller_participants = trainbig_participants.iloc[
            train_smaller_index
        ].reset_index(drop=True)
        validation_participants = trainbig_participants.iloc[
            validation_index
        ].reset_index(drop=True)

        # confirm each patient is entirely on one or the other side of the train-smaller vs validation divide
        assert (
            len(
                set(
                    train_smaller_participants["participant_label"].unique()
                ).intersection(
                    set(validation_participants["participant_label"].unique())
                )
            )
            == 0
        )

        # get list of participant labels
        for participants, fold_label in zip(
            [train_smaller_participants, validation_participants, test_participants],
            ["train_smaller", "validation", "test"],
        ):
            fold_participants.append(
                pd.DataFrame(
                    {"participant_label": participants["participant_label"].unique()}
                ).assign(fold_id=fold_id, fold_label=fold_label)
            )

    # also create global fold
    # split entire set into train_smaller + validation, aiming for same % split of those as we did when carving up train_big
    mini_folds = make_splits(unique_participants, n_splits=3)
    # unpack first of the splits
    _, (train_smaller_index, validation_index) = mini_folds[0]

    train_smaller_participants = unique_participants.iloc[
        train_smaller_index
    ].reset_index(drop=True)
    validation_participants = unique_participants.iloc[validation_index].reset_index(
        drop=True
    )

    # confirm each patient is entirely on one or the other side of the train-smaller vs validation divide
    assert (
        len(
            set(train_smaller_participants["participant_label"].unique()).intersection(
                set(validation_participants["participant_label"].unique())
            )
        )
        == 0
    )

    # get list of participant labels
    for participants, fold_label in zip(
        [train_smaller_participants, validation_participants],
        ["train_smaller", "validation"],
    ):
        fold_participants.append(
            pd.DataFrame(
                {"participant_label": participants["participant_label"].unique()}
            ).assign(fold_id=-1, fold_label=fold_label)
        )


fold_participants = pd.concat(fold_participants, axis=0)
fold_participants

# %%
# sanity checks:

# each participant is in each fold, in one group or another
assert all(fold_participants.groupby("participant_label").size() == config.n_folds + 1)

# within the cross validation scheme, each participant is in two non-test sets
assert all(
    fold_participants[
        (fold_participants["fold_id"] != -1)
        & (fold_participants["fold_label"] != "test")
    ]
    .groupby("participant_label")
    .size()
    == config.n_folds - 1
)

# within the cross validation scheme, each participant is in one test set
assert all(
    fold_participants[
        (fold_participants["fold_id"] != -1)
        & (fold_participants["fold_label"] == "test")
    ]
    .groupby("participant_label")
    .size()
    == 1
)

# %%

# %%

# %%
assert (
    "fold_id" not in specimens_merged.columns
    and "fold_label" not in specimens_merged.columns
)

# %%
specimens_by_fold = pd.merge(
    specimens_merged, fold_participants, on="participant_label", how="inner"
)
specimens_by_fold

# %%
assert specimens_by_fold.shape[0] == specimens_merged.shape[0] * (config.n_folds + 1)

# %%
# sanity checks:

# each specimen is in each fold, in one group or another
assert all(specimens_by_fold.groupby("specimen_label").size() == config.n_folds + 1)

# within the cross validation scheme, each specimen is in two non-test sets
assert all(
    specimens_by_fold[
        (specimens_by_fold["fold_id"] != -1)
        & (specimens_by_fold["fold_label"] != "test")
    ]
    .groupby("specimen_label")
    .size()
    == config.n_folds - 1
)

# within the cross validation scheme, each specimen is in one test set
assert all(
    specimens_by_fold[
        (specimens_by_fold["fold_id"] != -1)
        & (specimens_by_fold["fold_label"] == "test")
    ]
    .groupby("specimen_label")
    .size()
    == 1
)

# %%
# sanity checks:

# each participant is in each fold, in one group or another
assert all(
    specimens_by_fold.groupby("participant_label")["fold_id"].nunique()
    == config.n_folds + 1
)

# within the cross validation scheme, each participant is in two non-test sets
assert all(
    specimens_by_fold[
        (specimens_by_fold["fold_id"] != -1)
        & (specimens_by_fold["fold_label"] != "test")
    ]
    .groupby("participant_label")["fold_id"]
    .nunique()
    == config.n_folds - 1
)

# within the cross validation scheme, each participant is in one test set
assert all(
    specimens_by_fold[
        (specimens_by_fold["fold_id"] != -1)
        & (specimens_by_fold["fold_label"] == "test")
    ]
    .groupby("participant_label")["fold_id"]
    .nunique()
    == 1
)

# %%

# %%
for (fold_id, fold_label), grp in specimens_by_fold.groupby(
    ["fold_id", "fold_label"], observed=True
):
    print(f"Fold {fold_id}-{fold_label}:")
    display(
        pd.DataFrame(
            [
                grp.groupby("disease")["participant_label"]
                .nunique()
                .rename("#participants"),
                grp.groupby("disease")["specimen_label"].nunique().rename("#specimens"),
            ]
        )
    )

    print()

# %%

# %%

# %%
# By gene locus
# Nest because can't sort on gene locus column
for (fold_id, fold_label), _grp in specimens_by_fold.groupby(
    ["fold_id", "fold_label"], observed=True
):
    for gene_locus, grp in _grp.groupby(
        ["available_gene_loci"], observed=True, sort=False
    ):
        print(f"Fold {fold_id}-{fold_label}-{gene_locus}:")

        display(
            pd.DataFrame(
                [
                    grp.groupby("disease")["participant_label"]
                    .nunique()
                    .rename("#participants"),
                    grp.groupby("disease")["specimen_label"]
                    .nunique()
                    .rename("#specimens"),
                ]
            )
        )

        print()

# %%

# %%
fold_participants.to_csv(
    config.paths.dataset_specific_metadata
    / "cross_validation_divisions.participants.tsv",
    sep="\t",
    index=None,
)

# %%
specimens_by_fold.to_csv(
    config.paths.dataset_specific_metadata / "cross_validation_divisions.specimens.tsv",
    sep="\t",
    index=None,
)

# %%

# %%

# %%

# %%
