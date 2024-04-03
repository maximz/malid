# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

from malid import config, helpers, io
from malid.datamodels import (
    map_cross_validation_split_strategy_to_default_target_obs_column,
)
import gc
from kdict import kdict
from IPython.display import display
import itertools
import random
import genetools

# %% [markdown]
# # Overlapping sequences between folds

# %%
sequence_identifying_cols = [
    "v_gene",
    "j_gene",
    "isotype_supergroup",
    "cdr1_seq_aa_q_trim",
    "cdr2_seq_aa_q_trim",
    "cdr3_seq_aa_q_trim",
]
pd.set_option("display.max_columns", 100)
config.gene_loci_used

# %%
target_obs_column = map_cross_validation_split_strategy_to_default_target_obs_column[
    config.cross_validation_split_strategy
]
target_obs_column

# %%


# %%
for gene_locus in config.gene_loci_used:
    fold_details = kdict()
    # use cross validation folds only - because we want to make claims about our cross validation splits:
    for fold_id in config.cross_validation_fold_ids:
        for fold_label in ["train_smaller", "validation", "test"]:
            adata = io.load_fold_embeddings(
                fold_id=fold_id,
                fold_label=fold_label,
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
            )
            df = adata.obs

            n_sequences = df.shape[0]

            participants = set(df["participant_label"].unique())
            n_participants = len(participants)

            specimens = set(df["specimen_label"].unique())
            n_specimens = len(specimens)

            sequences_only = df[sequence_identifying_cols].drop_duplicates()
            n_unique_sequences = sequences_only.shape[0]

            # Slower alternatives, according to %timeit:
            # df.head().apply(joblib.hash, axis=1)
            # sequences_only.head().apply(lambda x: hash(tuple(x)), axis = 1)
            hashed_sequences = pd.util.hash_pandas_object(sequences_only, index=False)

            unique_sequence_hashes = set(hashed_sequences)

            if len(unique_sequence_hashes) != sequences_only.shape[0]:
                raise ValueError("Hash collision")

            fold_details[fold_id, fold_label] = {
                "fold_id": fold_id,
                "fold_label": fold_label,
                "n_sequences": n_sequences,
                "participants": participants,
                "n_participants": n_participants,
                "specimens": specimens,
                "n_specimens": n_specimens,
                "n_unique_sequences": n_unique_sequences,
                "unique_sequence_hashes": unique_sequence_hashes,
            }

            del sequences_only, hashed_sequences, unique_sequence_hashes, df, adata
            io.clear_cached_fold_embeddings()
            gc.collect()

    comparison_data = []
    for fold_id in fold_details.keys(dimensions=0):
        for fold_label1, fold_label2 in itertools.combinations(
            fold_details[fold_id, :].keys(dimensions=1), 2
        ):
            data1 = fold_details[fold_id, fold_label1]
            data2 = fold_details[fold_id, fold_label2]

            unique_sequence_hashes1 = data1["unique_sequence_hashes"]
            unique_sequence_hashes2 = data2["unique_sequence_hashes"]

            comparison_data.append(
                {
                    "fold_id": fold_id,
                    "fold_label1": fold_label1,
                    "fold_label2": fold_label2,
                    "n_sequences1": data1["n_sequences"],
                    "n_sequences2": data2["n_sequences"],
                    "n_participants1": data1["n_participants"],
                    "n_participants2": data2["n_participants"],
                    "n_specimens1": data1["n_specimens"],
                    "n_specimens2": data2["n_specimens"],
                    "n_unique_sequences1": data1["n_unique_sequences"],
                    "n_unique_sequences2": data2["n_unique_sequences"],
                    "n_overlapping_participants": len(
                        data1["participants"].intersection(data2["participants"])
                    ),
                    "n_overlapping_specimens": len(
                        data1["specimens"].intersection(data2["specimens"])
                    ),
                    "n_overlapping_unique_sequence_hashes": len(
                        unique_sequence_hashes1.intersection(unique_sequence_hashes2)
                    ),
                    "jaccard_index_unique_sequence_hashes": float(
                        len(
                            unique_sequence_hashes1.intersection(
                                unique_sequence_hashes2
                            )
                        )
                    )
                    / len(unique_sequence_hashes1.union(unique_sequence_hashes2)),
                }
            )
    comparison_data = pd.DataFrame(comparison_data)
    display(comparison_data)
    print(
        f"{gene_locus}: Average overlap Jaccard index between any two fold labels from the same fold ID: {np.mean(comparison_data['jaccard_index_unique_sequence_hashes'] * 100):0.3f}%"
    )
    print(
        f"{gene_locus}: Average +/- std overlap Jaccard index between any two fold labels from the same fold ID: {np.mean(comparison_data['jaccard_index_unique_sequence_hashes'] * 100):0.3f} +/- {np.std(comparison_data['jaccard_index_unique_sequence_hashes'] * 100):0.3f} %"
    )
    comparison_data.to_csv(
        config.paths.output_dir / f"fold_sequence_overlap_stats.{gene_locus.name}.tsv",
        sep="\t",
        index=None,
    )

    ### Sanity check: compare across folds - should get dupes
    comparison_data_sanity_check = []
    for key1, key2 in itertools.combinations(fold_details.keys(), 2):
        data1 = fold_details[key1]
        data2 = fold_details[key2]

        unique_sequence_hashes1 = data1["unique_sequence_hashes"]
        unique_sequence_hashes2 = data2["unique_sequence_hashes"]

        comparison_data_sanity_check.append(
            {
                "fold_id1": key1[0],
                "fold_id2": key2[0],
                "fold_label1": key1[1],
                "fold_label2": key2[1],
                "n_sequences1": data1["n_sequences"],
                "n_sequences2": data2["n_sequences"],
                "n_participants1": data1["n_participants"],
                "n_participants2": data2["n_participants"],
                "n_specimens1": data1["n_specimens"],
                "n_specimens2": data2["n_specimens"],
                "n_unique_sequences1": data1["n_unique_sequences"],
                "n_unique_sequences2": data2["n_unique_sequences"],
                "n_overlapping_participants": len(
                    data1["participants"].intersection(data2["participants"])
                ),
                "n_overlapping_specimens": len(
                    data1["specimens"].intersection(data2["specimens"])
                ),
                "n_overlapping_unique_sequence_hashes": len(
                    unique_sequence_hashes1.intersection(unique_sequence_hashes2)
                ),
                "jaccard_index_unique_sequence_hashes": float(
                    len(unique_sequence_hashes1.intersection(unique_sequence_hashes2))
                )
                / len(unique_sequence_hashes1.union(unique_sequence_hashes2)),
            }
        )
    comparison_data_sanity_check = pd.DataFrame(comparison_data_sanity_check)
    display(comparison_data_sanity_check)
    print(
        f"{gene_locus}: Average overlap Jaccard index between fold labels from DIFFERENT fold ID: {np.mean(comparison_data_sanity_check[comparison_data_sanity_check['fold_id1'] != comparison_data_sanity_check['fold_id2']]['jaccard_index_unique_sequence_hashes'] * 100):0.3f}%"
    )
    print(
        f"{gene_locus}: Average overlap Jaccard index between fold labels from SAME fold ID: {np.mean(comparison_data_sanity_check[comparison_data_sanity_check['fold_id1'] == comparison_data_sanity_check['fold_id2']]['jaccard_index_unique_sequence_hashes'] * 100):0.3f}%"
    )

    ### Sanity check 2: compare same foldID+label against itself - should see all dupes
    comparison_data_sanity_check = []
    for key in fold_details.keys():
        key1 = key
        key2 = key

        data1 = fold_details[key1]
        data2 = fold_details[key2]

        unique_sequence_hashes1 = data1["unique_sequence_hashes"]
        unique_sequence_hashes2 = data2["unique_sequence_hashes"]

        comparison_data_sanity_check.append(
            {
                "fold_id1": key1[0],
                "fold_id2": key2[0],
                "fold_label1": key1[1],
                "fold_label2": key2[1],
                "n_sequences1": data1["n_sequences"],
                "n_sequences2": data2["n_sequences"],
                "n_participants1": data1["n_participants"],
                "n_participants2": data2["n_participants"],
                "n_specimens1": data1["n_specimens"],
                "n_specimens2": data2["n_specimens"],
                "n_unique_sequences1": data1["n_unique_sequences"],
                "n_unique_sequences2": data2["n_unique_sequences"],
                "n_overlapping_participants": len(
                    data1["participants"].intersection(data2["participants"])
                ),
                "n_overlapping_specimens": len(
                    data1["specimens"].intersection(data2["specimens"])
                ),
                "n_overlapping_unique_sequence_hashes": len(
                    unique_sequence_hashes1.intersection(unique_sequence_hashes2)
                ),
                "jaccard_index_unique_sequence_hashes": float(
                    len(unique_sequence_hashes1.intersection(unique_sequence_hashes2))
                )
                / len(unique_sequence_hashes1.union(unique_sequence_hashes2)),
            }
        )
    comparison_data_sanity_check = pd.DataFrame(comparison_data_sanity_check)
    display(comparison_data_sanity_check)
    print(
        f"{gene_locus}: Average overlap Jaccard index between fold labels from DIFFERENT fold ID: {np.mean(comparison_data_sanity_check[comparison_data_sanity_check['fold_id1'] != comparison_data_sanity_check['fold_id2']]['jaccard_index_unique_sequence_hashes'] * 100):0.3f}%"
    )
    print(
        f"{gene_locus}: Average overlap Jaccard index between fold labels from SAME fold ID: {np.mean(comparison_data_sanity_check[comparison_data_sanity_check['fold_id1'] == comparison_data_sanity_check['fold_id2']]['jaccard_index_unique_sequence_hashes'] * 100):0.3f}%"
    )
    print("*" * 80)

# %%

# %%

# %% [markdown]
# # Compute patient overlaps

# %%
for gene_locus in config.gene_loci_used:
    # get unique sequence hashes by participant
    data_by_participant = {}

    # get test folds. each participant is in 1 test fold only.
    # (don't include global fold because it does not have a test set)
    for fold_id in config.cross_validation_fold_ids:
        adata = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label="test",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )
        df = adata.obs

        for participant_label, participant_grp in df.groupby(
            "participant_label", observed=True
        ):
            sequences_only = participant_grp[
                sequence_identifying_cols
            ].drop_duplicates()
            n_unique_sequences = sequences_only.shape[0]

            # Slower alternatives, according to %timeit:
            # df.head().apply(joblib.hash, axis=1)
            # sequences_only.head().apply(lambda x: hash(tuple(x)), axis = 1)
            hashed_sequences = pd.util.hash_pandas_object(sequences_only, index=False)

            unique_sequence_hashes = set(hashed_sequences)

            if len(unique_sequence_hashes) != n_unique_sequences:
                raise ValueError("Hash collision")

            # Store
            data_by_participant[participant_label] = {
                "participant_label": participant_label,
                "unique_sequence_hashes": unique_sequence_hashes,
                "fold_id": fold_id,
                # 'fold_label': 'test',
                target_obs_column.value.obs_column_name: participant_grp[
                    target_obs_column.value.obs_column_name
                ].iloc[0],
            }

        del df, adata
        io.clear_cached_fold_embeddings()
        gc.collect()

    participant_comparison_data = []
    for participantA, participantB in itertools.combinations(
        data_by_participant.keys(), 2
    ):
        dataA = data_by_participant[participantA]
        dataB = data_by_participant[participantB]

        unique_sequence_hashesA = dataA["unique_sequence_hashes"]
        unique_sequence_hashesB = dataB["unique_sequence_hashes"]

        comparison_result = {
            "participantA": participantA,
            "participantB": participantB,
            "fold_id_A": dataA["fold_id"],
            "fold_id_B": dataB["fold_id"],
            f"{target_obs_column.value.obs_column_name}_A": dataA[
                target_obs_column.value.obs_column_name
            ],
            f"{target_obs_column.value.obs_column_name}_B": dataB[
                target_obs_column.value.obs_column_name
            ],
            "nunique_sequences_A": len(unique_sequence_hashesA),
            "nunique_sequences_B": len(unique_sequence_hashesB),
            "nunique_sequences_intersection": len(
                unique_sequence_hashesA.intersection(unique_sequence_hashesB)
            ),
            "nunique_sequences_union": len(
                unique_sequence_hashesA.union(unique_sequence_hashesB)
            ),
        }

        comparison_result["jaccard_index"] = (
            float(comparison_result["nunique_sequences_intersection"])
            / comparison_result["nunique_sequences_union"]
        )

        participant_comparison_data.append(comparison_result)

    participant_comparison_data = pd.DataFrame(participant_comparison_data)
    display(participant_comparison_data)
    participant_comparison_data.to_csv(
        config.paths.output_dir
        / f"participant_sequence_overlap_stats.{gene_locus.name}.tsv.gz",
        sep="\t",
        index=None,
        float_format="%0.3f",  # save space
    )

    # sanity check that each row is unique combo of participants - by construction
    assert all(
        participant_comparison_data["participantA"]
        != participant_comparison_data["participantB"]
    )

    ## between any two patients
    any_two_patients = (
        participant_comparison_data["jaccard_index"] * 100
    )  # convert to percent
    print(
        f"{gene_locus}: Average +/- std overlap Jaccard index between any two patients: {any_two_patients.mean():0.3f} +/- {any_two_patients.std():0.3f} %"
    )
    print(any_two_patients.max())
    print(any_two_patients.median())
    print(any_two_patients.describe())

    participant_diseases = participant_comparison_data[
        f"{target_obs_column.value.obs_column_name}_A"
    ].unique()

    ## between patients from same disease
    for disease in participant_diseases:
        any_two_patients_same_disease = (
            participant_comparison_data[
                (
                    participant_comparison_data[
                        f"{target_obs_column.value.obs_column_name}_A"
                    ]
                    == disease
                )
                & (
                    participant_comparison_data[
                        f"{target_obs_column.value.obs_column_name}_B"
                    ]
                    == disease
                )
            ]["jaccard_index"]
            * 100
        )  # convert to percent
        print(
            f"{gene_locus}: Average +/- std overlap Jaccard index between any two patients with disease {disease}: {any_two_patients_same_disease.mean():0.3f} +/- {any_two_patients_same_disease.std():0.3f} %"
        )

    ## between patients from different diseases
    for diseaseA, diseaseB in itertools.combinations(participant_diseases, 2):
        # either order accepted
        subset = participant_comparison_data[
            (
                (
                    participant_comparison_data[
                        f"{target_obs_column.value.obs_column_name}_A"
                    ]
                    == diseaseA
                )
                & (
                    participant_comparison_data[
                        f"{target_obs_column.value.obs_column_name}_B"
                    ]
                    == diseaseB
                )
            )
            | (
                (
                    participant_comparison_data[
                        f"{target_obs_column.value.obs_column_name}_A"
                    ]
                    == diseaseB
                )
                & (
                    participant_comparison_data[
                        f"{target_obs_column.value.obs_column_name}_B"
                    ]
                    == diseaseA
                )
            )
        ]
        any_two_patients_disease_pair = (
            subset["jaccard_index"] * 100
        )  # convert to percent
        print(
            f"{gene_locus}: Average +/- std overlap Jaccard index between {diseaseA} and {diseaseB} patients: {any_two_patients_disease_pair.mean():0.3f} +/- {any_two_patients_disease_pair.std():0.3f} %"
        )
    print("*" * 80)


# %% [markdown]
# # Overlapping sequences between specimens from same person


# %%
# Compute:
# Loop over gene loci
for gene_locus in config.gene_loci_used:
    # Load fold embeddings for the current gene locus
    # A way to get *all* data is to combine fold -1's train_smaller and validation sets
    # (The other way is to combine fold 0, 1, and 2's test sets, but those have different language models so unmergeable embeddings - though not relevant here.)

    adata_1 = io.load_fold_embeddings(
        fold_id=-1,
        fold_label="train_smaller",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
    )

    adata_2 = io.load_fold_embeddings(
        fold_id=-1,
        fold_label="validation",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
    )
    df = pd.concat([adata_1.obs, adata_2.obs], axis=0)
    del adata_1, adata_2
    io.clear_cached_fold_embeddings()
    gc.collect()

    # List to store within-patient comparisons (i.e. comparisons between samples from the same patient)
    within_patient_comparisons = []

    # Group data by patient
    for patient, patient_data in df.groupby("participant_label"):
        # If there is more than one sample from the current patient...
        if patient_data.specimen_label.nunique() > 1:
            # Get the unique sample labels for this patient
            unique_samples = patient_data.specimen_label.unique()
            # Loop over pairs of samples from this patient
            for s1 in unique_samples:
                for s2 in unique_samples:
                    # Ignore comparisons between a sample and itself
                    if s1 != s2:
                        # Select the rows corresponding to samples s1 and s2
                        sample_1 = patient_data.loc[
                            patient_data.specimen_label == s1, sequence_identifying_cols
                        ].drop_duplicates()
                        sample_2 = patient_data.loc[
                            patient_data.specimen_label == s2, sequence_identifying_cols
                        ].drop_duplicates()
                        # Calculate the Jaccard index for these two samples
                        hashed_sequences_1 = pd.util.hash_pandas_object(
                            sample_1, index=False
                        )
                        unique_sequence_hashes_1 = set(hashed_sequences_1)
                        hashed_sequences_2 = pd.util.hash_pandas_object(
                            sample_2, index=False
                        )
                        unique_sequence_hashes_2 = set(hashed_sequences_2)
                        n_intersect = len(
                            unique_sequence_hashes_1.intersection(
                                unique_sequence_hashes_2
                            )
                        )
                        n_union = len(
                            unique_sequence_hashes_1.union(unique_sequence_hashes_2)
                        )
                        jaccard_index = n_intersect / n_union
                        # Add the comparison to the list of within-patient comparisons
                        within_patient_comparisons.append(
                            (patient, patient, s1, s2, jaccard_index)
                        )

    # Convert the list of within-patient comparisons to a DataFrame
    within_patient_comparisons = pd.DataFrame(
        within_patient_comparisons,
        columns=[
            "participant1",
            "participant2",
            "specimen1",
            "specimen2",
            "jaccard_index",
        ],
    )

    # Create a column indicating that these are within-patient comparisons
    within_patient_comparisons["type"] = "Samples from same person"

    # Get all unique pairs of samples
    combinations = itertools.combinations(df.specimen_label.unique(), 2)
    # Randomly select 50 pairs of samples
    random_combinations = random.sample(list(combinations), 500)

    # List to store between-patient comparisons (i.e. comparisons between samples from different participants)
    between_patient_comparisons = []
    # Loop over the randomly-selected pairs of samples
    for s1, s2 in random_combinations:
        # Get the patient labels for samples s1 and s2
        p1 = df.loc[df.specimen_label == s1, "participant_label"].iloc[0]
        p2 = df.loc[df.specimen_label == s2, "participant_label"].iloc[0]

        # If the samples are from different patients...
        if p1 != p2:
            # Select the rows corresponding to samples s1 and s2
            sample_1 = df.loc[
                df.specimen_label == s1, sequence_identifying_cols
            ].drop_duplicates()
            sample_2 = df.loc[
                df.specimen_label == s2, sequence_identifying_cols
            ].drop_duplicates()
            # Calculate the Jaccard index for these two samples
            hashed_sequences_1 = pd.util.hash_pandas_object(sample_1, index=False)
            unique_sequence_hashes_1 = set(hashed_sequences_1)
            hashed_sequences_2 = pd.util.hash_pandas_object(sample_2, index=False)
            unique_sequence_hashes_2 = set(hashed_sequences_2)
            n_intersect = len(
                unique_sequence_hashes_1.intersection(unique_sequence_hashes_2)
            )
            n_union = len(unique_sequence_hashes_1.union(unique_sequence_hashes_2))
            jaccard_index = n_intersect / n_union
            # Add the comparison to the list of between-patient comparisons
            between_patient_comparisons.append((p1, p2, s1, s2, jaccard_index))

    # Convert the list of between-patient comparisons to a DataFrame
    between_patient_comparisons = pd.DataFrame(
        between_patient_comparisons,
        columns=[
            "participant1",
            "participant2",
            "specimen1",
            "specimen2",
            "jaccard_index",
        ],
    )
    # Add a column indicating that these are between-patient comparisons
    between_patient_comparisons["type"] = "Samples from different people"

    # Combine the within-patient and between-patient comparisons
    comparison = pd.concat(
        [between_patient_comparisons, within_patient_comparisons], axis=0
    )
    # Export for later plotting
    comparison.to_csv(
        config.paths.output_dir
        / f"sequence_overlap_between_samples.{gene_locus.name}.tsv",
        sep="\t",
        index=None,
    )
    del df
    gc.collect()


# %%

# %%
# Plot:
# Loop over gene loci
for gene_locus in config.gene_loci_used:
    # Reload
    comparison = pd.read_csv(
        config.paths.output_dir
        / f"sequence_overlap_between_samples.{gene_locus.name}.tsv",
        sep="\t",
    )

    fig, ax = plt.subplots()
    sns.boxplot(x="type", y="jaccard_index", data=comparison, ax=ax)
    plt.title(f"{gene_locus.name} sequence overlap between pairs of samples")
    plt.xlabel("Source of two compared samples")
    plt.ylabel("Jaccard index")
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            ax.get_xticklabels(), comparison, "type"
        )
    )

    genetools.plots.savefig(
        fig,
        config.paths.output_dir
        / f"sequence_overlap_between_samples.{gene_locus.name}.png",
        dpi=300,
    )

    selection = comparison[comparison["type"] == "Samples from same person"]
    selection = set(selection["specimen1"].str.split("-").str[0]).union(
        set(selection["specimen2"].str.split("-").str[0])
    )
    print(
        f"{gene_locus}: these runs have participants with multiple samples: {selection}"
    )
    print()

    study_name_per_specimen = helpers.get_all_specimen_info().set_index(
        "specimen_label"
    )["study_name"]
    for key, grp in comparison.groupby("type"):
        vals = grp["jaccard_index"] * 100  # convert to percent
        cohorts_involved = set(
            study_name_per_specimen.loc[comparison["specimen1"]]
        ) | set(study_name_per_specimen.loc[comparison["specimen2"]])
        print(
            f"{gene_locus}, {key}: Avg +/- std of Jaccard index = {vals.mean():0.3f} +/- {vals.std():0.3f} %. n={vals.shape[0]} pairs of samples, from cohorts: {cohorts_involved}"
        )
        print()

# %%
