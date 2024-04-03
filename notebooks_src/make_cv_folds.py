# %% [markdown]
# # Filter to selected samples (e.g. peak timepoints) and make cross validation divisions, based on active CrossValidationSplitStrategy
#
# We've already removed specimens with very few sequences or without all isotypes. And we've already sampled one sequence per clone per isotype per specimen.
#
# If you're editing `config.py`'s active CrossValidationSplitStrategy, make sure to run `python scripts/make_dirs.py`.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from malid import config, helpers, logger

# %%
import dask
import dask.dataframe as dd

# %%

# %%
from dask.distributed import Client

# multi-processing backend
# if already opened from another notebook, see https://stackoverflow.com/questions/60115736/dask-how-to-connect-to-running-cluster-scheduler-and-access-total-occupancy
client = Client(
    scheduler_port=config.dask_scheduler_port,
    dashboard_address=config.dask_dashboard_address,
    n_workers=config.dask_n_workers,
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

# %% [markdown]
# # Get all specimens available from ETL - meaning the ones that passed `sample_sequences` filters

# %%

# %%
# groupby participant, specimen, disease - get total sequence count
specimens = (
    df.groupby(
        ["participant_label", "specimen_label", "disease"],
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
# Export list of specimens remaining after QC filtering in sample_sequences.ipynb.
# Not all specimens survived to this step - some are thrown out for not having enough sequences or not having all isotypes.
# However, these aren't yet filtered to is_selected_for_cv_strategy specimens that are particular to the selected cross validation strategy.
specimens.to_csv(
    config.paths.dataset_specific_metadata
    / "specimens_that_survived_qc_filters_in_sample_sequences_notebook.tsv",
    sep="\t",
    index=None,
)

# %%
# TODO: Split the notebooks here. All the above should be run once after sample_sequences.ipynb. All the below gets run separately for each cross validation split strategy.

# %% [markdown]
# # Apply `is_selected_for_cv_strategy` filter using currently active `config.cross_validation_split_strategy`

# %%
config.cross_validation_split_strategy

# %%

# %%
# filter to is_selected_for_cv_strategy specimens, in addition to the is_valid / survived_filters filter already here
specimen_metadata = helpers.get_all_specimen_info(
    # CV fold is not available yet, so must set this to False
    add_cv_fold_information=False
).sort_values("disease")
specimen_metadata = specimen_metadata[specimen_metadata["in_training_set"]]

# sanity check the definitions
assert specimen_metadata["is_selected_for_cv_strategy"].all()
assert specimen_metadata["survived_filters"].all()

specimen_metadata

# %%
# merge back to apply the filter
specimens_merged = pd.merge(
    specimens,
    specimen_metadata,
    on=["participant_label", "specimen_label", "disease"],
    how="inner",
    validate="1:1",
)
assert specimens_merged.shape[0] == specimen_metadata.shape[0] <= specimens.shape[0]
specimens_merged

# %%
specimens_merged["data_source"].value_counts()

# %%

# %%
(
    specimens_merged.drop_duplicates(["participant_label", "disease"])["disease"]
    .value_counts()
    .rename("number of participants")
)

# %%
tmp = (
    specimens_merged.drop_duplicates(["participant_label", "disease_subtype"])[
        "disease_subtype"
    ]
    .value_counts()
    .rename("number of participants")
)

tmp.to_csv(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "all_data_combined.participants_by_disease_subtype.tsv",
    sep="\t",
)

tmp

# %%
tmp = (
    specimens_merged.groupby(["disease", "data_source"], observed=True)[
        "total_sequence_count"
    ]
    .sum()
    .reset_index(name="number of sequences")
)

tmp.to_csv(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "all_data_combined.sequences_by_disease_and_cohort.tsv",
    sep="\t",
    index=None,
)

tmp.sort_values("number of sequences")

# %%

# %% [markdown]
# # Make CV splits
#
# **Strategy:**
#
# ```
# Full data →  [Test | Rest] x 3, producing 3 folds
#
# In each fold:
# Rest →  [ Train | Validation]
# Train → [Train1 | Train2]
# ```
#
# - Split on patients.
# - Split into (train+validation) / test first with a 2:1 ratio (because 3 folds total). Every patient is in 1 test fold. Then split (train+validation) into train (2/3) and validation (1/3).
# - Also create a "global fold" that has no test set, but has the same 2:1 train/validation ratio.
# - Each train set is further subdivided into Train1 and Train2 (used for training model 3 rollup step)
# - All splits are stratified by disease label
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
# **Note: we need to respect `study_names_for_held_out_set` on the CrossValidationSplitStrategy**. If set, that reduces us to a single fold with a pre-determined test set. (We still need to make a random train/validation split that keeps each patient's data segregated to one or the other)

# %%

# %%
unique_participants_all = (
    specimens_merged[
        [
            "participant_label",
            "disease",
            "data_source",
            "past_exposure",
            "disease.separate_past_exposures",
            "available_gene_loci",
            "study_name",
        ]
    ]
    .drop_duplicates(subset=["participant_label"])
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
def make_splits_single(participants: pd.DataFrame, test_proportion: float):
    # Single separation. Returns train and test integer indices into the participants dataframe.
    return train_test_split(
        np.arange(participants.shape[0]),
        test_size=test_proportion,
        random_state=0,
        shuffle=True,
        # preserve imbalanced class distribution in each fold
        stratify=participants["disease"],
    )


# %%

# %%
fold_participants = []


for gene_locus, unique_participants in unique_participants_all.groupby(
    "available_gene_loci", observed=True, sort=False
):
    print(
        f"{gene_locus}: total number of unique participants {unique_participants.shape[0]}"
    )

    if not config.cross_validation_split_strategy.value.is_single_fold_only:
        # Default case
        trainbig_test_splits = make_splits(unique_participants, n_splits=config.n_folds)
    else:
        # Special case: single fold with a pre-determined held-out test set (see study_names_for_held_out_set docs).
        held_out_bool_array = unique_participants["study_name"].isin(
            config.cross_validation_split_strategy.value.study_names_for_held_out_set
        )
        trainbig_test_splits = [
            (0, (np.where(~held_out_bool_array)[0], np.where(held_out_bool_array)[0]))
        ]

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
                set(trainbig_participants["participant_label"]).intersection(
                    set(test_participants["participant_label"])
                )
            )
            == 0
        )

        # split trainbig into trainsmaller + validation
        train_smaller_index, validation_index = make_splits_single(
            trainbig_participants, test_proportion=1 / 3
        )
        train_smaller_participants = trainbig_participants.iloc[
            train_smaller_index
        ].reset_index(drop=True)
        validation_participants = trainbig_participants.iloc[
            validation_index
        ].reset_index(drop=True)

        # confirm each patient is entirely on one or the other side of the train-smaller vs validation divide
        assert (
            len(
                set(train_smaller_participants["participant_label"]).intersection(
                    set(validation_participants["participant_label"])
                )
            )
            == 0
        )

        # split train-smaller into train-smaller1 and train-smaller2
        train_smaller1_indices, train_smaller2_indices = make_splits_single(
            train_smaller_participants, test_proportion=1 / 3
        )
        train_smaller1_participants = train_smaller_participants.iloc[
            train_smaller1_indices
        ]
        train_smaller2_participants = train_smaller_participants.iloc[
            train_smaller2_indices
        ]
        # confirm each patient is entirely on one or the other side of the train-smaller1 vs train-smaller2 divide
        assert (
            len(
                set(train_smaller1_participants["participant_label"]).intersection(
                    set(train_smaller2_participants["participant_label"])
                )
            )
            == 0
        )

        # get list of participant labels
        for participants, fold_label in zip(
            [
                train_smaller_participants,
                train_smaller1_participants,
                train_smaller2_participants,
                validation_participants,
                test_participants,
            ],
            ["train_smaller", "train_smaller1", "train_smaller2", "validation", "test"],
        ):
            fold_participants.append(
                pd.DataFrame(
                    {"participant_label": participants["participant_label"].unique()}
                ).assign(fold_id=fold_id, fold_label=fold_label)
            )

    # also create global fold
    if config.use_global_fold:
        # split entire set into train_smaller + validation, aiming for same % split of those as we did when carving up train_big
        train_smaller_index, validation_index = make_splits_single(
            unique_participants, test_proportion=1 / 3
        )
        train_smaller_participants = unique_participants.iloc[
            train_smaller_index
        ].reset_index(drop=True)
        validation_participants = unique_participants.iloc[
            validation_index
        ].reset_index(drop=True)

        # confirm each patient is entirely on one or the other side of the train-smaller vs validation divide
        assert (
            len(
                set(train_smaller_participants["participant_label"]).intersection(
                    set(validation_participants["participant_label"])
                )
            )
            == 0
        )

        # split train-smaller into train-smaller1 and train-smaller2
        train_smaller1_indices, train_smaller2_indices = make_splits_single(
            train_smaller_participants, test_proportion=1 / 3
        )
        train_smaller1_participants = train_smaller_participants.iloc[
            train_smaller1_indices
        ]
        train_smaller2_participants = train_smaller_participants.iloc[
            train_smaller2_indices
        ]
        # confirm each patient is entirely on one or the other side of the train-smaller1 vs train-smaller2 divide
        assert (
            len(
                set(train_smaller1_participants["participant_label"]).intersection(
                    set(train_smaller2_participants["participant_label"])
                )
            )
            == 0
        )

        # get list of participant labels
        for participants, fold_label in zip(
            [
                train_smaller_participants,
                train_smaller1_participants,
                train_smaller2_participants,
                validation_participants,
            ],
            ["train_smaller", "train_smaller1", "train_smaller2", "validation"],
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

# each participant is in each fold (either in train_smaller, validation, or test - ignore the further subdivisions of train_smaller)
assert all(
    fold_participants[
        ~(fold_participants["fold_label"].isin(["train_smaller1", "train_smaller2"]))
    ]
    .groupby("participant_label")
    .size()
    == config.n_folds_including_global_fold
)

# within the cross validation scheme, each participant is in two non-test sets
# (i.e. shows up either in train_smaller or validation twice).
# ignore the further subdivisions of train_smaller for this check.
assert all(
    fold_participants[
        (fold_participants["fold_id"] != -1)
        & ~(
            fold_participants["fold_label"].isin(
                ["test", "train_smaller1", "train_smaller2"]
            )
        )
    ]
    .groupby("participant_label")
    .size()
    # special case for single fold due to study_names_for_held_out_set restriction
    == (config.n_folds - 1 if config.n_folds > 1 else 1)
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
# sanity check:
# one entry per specimen per fold (either in train_smaller, validation, or test - ignore the further subdivisions of train_smaller)
assert (
    specimens_by_fold[
        ~(specimens_by_fold["fold_label"].isin(["train_smaller1", "train_smaller2"]))
    ].shape[0]
    == specimens_merged.shape[0] * config.n_folds_including_global_fold
)

# %%
# sanity checks:

# each specimen is in each fold, either in train_smaller, validation, or test (ignore the further subdivisions of train_smaller)
assert all(
    specimens_by_fold[
        ~(specimens_by_fold["fold_label"].isin(["train_smaller1", "train_smaller2"]))
    ]
    .groupby("specimen_label")
    .size()
    == config.n_folds_including_global_fold
)

# within the cross validation scheme, each specimen is in two non-test sets
# (i.e. shows up either in train_smaller or validation twice).
# ignore the further subdivisions of train_smaller for this check.
assert all(
    specimens_by_fold[
        (specimens_by_fold["fold_id"] != -1)
        & ~(
            specimens_by_fold["fold_label"].isin(
                ["test", "train_smaller1", "train_smaller2"]
            )
        )
    ]
    .groupby("specimen_label")
    .size()
    # special case for single fold due to study_names_for_held_out_set restriction
    == (config.n_folds - 1 if config.n_folds > 1 else 1)
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
    == config.n_folds_including_global_fold
)

# within the cross validation scheme, each participant is in two non-test sets
assert all(
    specimens_by_fold[
        (specimens_by_fold["fold_id"] != -1)
        & (specimens_by_fold["fold_label"] != "test")
    ]
    .groupby("participant_label")["fold_id"]
    .nunique()
    # special case for single fold due to study_names_for_held_out_set restriction
    == (config.n_folds - 1 if config.n_folds > 1 else 1)
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
        "available_gene_loci", observed=True, sort=False
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
    config.paths.dataset_specific_metadata_for_selected_cross_validation_strategy
    / "cross_validation_divisions.participants.tsv",
    sep="\t",
    index=None,
)

# %%
specimens_by_fold.to_csv(
    config.paths.dataset_specific_metadata_for_selected_cross_validation_strategy
    / "cross_validation_divisions.specimens.tsv.gz",
    sep="\t",
    index=None,
)

# %%

# %%

# %%

# %%
