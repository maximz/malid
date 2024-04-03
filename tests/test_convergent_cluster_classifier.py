from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pytest
import timeit

from malid.datamodels import GeneLocus, TargetObsColumnEnum
from malid.trained_model_wrappers import ConvergentClusterClassifier
from malid import io


def make_data(n_patients_per_disease: int):
    diseases = ["Covid19", "HIV", "Healthy"]
    vgenes = [f"Vgene{i}" for i in range(5)]
    jgenes = [f"Jgene{i}" for i in range(5)]
    seqs = []
    for disease in diseases:
        for participant_id in range(n_patients_per_disease):
            participant_label = f"{disease}_participant_{participant_id}"

            # make disease specific sequences
            seqs.extend(
                [
                    {
                        "v_gene": np.random.choice(vgenes),
                        "j_gene": np.random.choice(jgenes),
                        "cdr3_seq_aa_q_trim": f"cdr3_specific_to_{disease}",
                        "disease": disease,
                        "participant_label": participant_label,
                    }
                    for _ in range(500)
                ]
            )

            # add some background sequences
            seqs.extend(
                [
                    {
                        "v_gene": np.random.choice(vgenes),
                        "j_gene": np.random.choice(jgenes),
                        "cdr3_seq_aa_q_trim": f"nonspecific_cdr3",
                        "disease": disease,
                        "participant_label": participant_label,
                    }
                    for _ in range(500)
                ]
            )
    df = pd.DataFrame.from_records(seqs)
    df["cdr3_aa_sequence_trim_len"] = df["cdr3_seq_aa_q_trim"].str.len()
    # create additional columns so metadata extraction works right
    df["disease_subtype"] = df["disease"]
    df["specimen_label"] = df["participant_label"] + "_specimen1"
    df["fold_id"] = 0
    df["fold_label"] = "train_smaller"
    df = io.label_past_exposures_in_obs(df)

    # in practice, our index is usually pretty weird and not just an increment integer range index
    # so let's make the index here unusual too by permuting the int order
    df.index = np.random.permutation(df.index)

    return df


@pytest.fixture
def df():
    return make_data(n_patients_per_disease=3)


@pytest.fixture
def df_big():
    return make_data(n_patients_per_disease=30)


from genetools.arrays import make_consensus_sequence


def slow_get_cluster_centroids(clustered_df: pd.DataFrame) -> pd.Series:
    # Unoptimized version of get_cluster_centroids
    cluster_centroids = (
        clustered_df.groupby(
            [
                "v_gene",
                "j_gene",
                "cdr3_aa_sequence_trim_len",
                "cluster_id_within_clustering_group",
            ],
            observed=True,
            sort=False,
        )
        .apply(
            lambda grp: make_consensus_sequence(
                grp["cdr3_seq_aa_q_trim"], grp["num_clone_members"]
            )
        )
        .rename("consensus_sequence")
    )
    return cluster_centroids


def test_get_cluster_centroids_timing(df_big):
    df_clustered = ConvergentClusterClassifier._cluster_training_set(
        df=df_big, sequence_identity_threshold=0.9, inplace=False
    )
    # Expressed as lambdas so we can use in timing test below.
    run_fast = lambda: ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=df_clustered
    )
    run_slow = lambda: slow_get_cluster_centroids(clustered_df=df_clustered)

    # Confirm calculations match.
    assert run_fast().equals(run_slow()), "They should give same result"

    # Do a timing test.
    # Expose local functions to timeit, from https://stackoverflow.com/a/46055465/130164
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    n_iters = 10
    fast_timing = timeit.Timer("run_fast()", globals=imports_and_vars).timeit(
        number=n_iters
    )
    slow_timing = timeit.Timer("run_slow()", globals=imports_and_vars).timeit(
        number=n_iters
    )
    assert (
        fast_timing < slow_timing
    ), f"Optimized function should be faster than slow function. Instead, it took fast: {fast_timing}, vs. slow: {slow_timing} seconds for {n_iters} iterations"


def test_get_cluster_centroids(df):
    df_clustered = ConvergentClusterClassifier._cluster_training_set(
        df=df, sequence_identity_threshold=0.9, inplace=False
    )
    # fast
    cluster_centroids = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=df_clustered
    )
    slow_comparison = slow_get_cluster_centroids(clustered_df=df_clustered)
    assert slow_comparison.equals(cluster_centroids)

    cluster_centroids_length_sanity_check = (
        cluster_centroids.str.len().rename("cluster_centroids_length").reset_index()
    )
    assert all(
        cluster_centroids_length_sanity_check["cdr3_aa_sequence_trim_len"]
        == cluster_centroids_length_sanity_check["cluster_centroids_length"]
    )

    assert cluster_centroids.shape == (100,)
    assert cluster_centroids.name == "consensus_sequence"
    assert np.array_equal(
        cluster_centroids.index.names,
        [
            "v_gene",
            "j_gene",
            "cdr3_aa_sequence_trim_len",
            "cluster_id_within_clustering_group",
        ],
    )
    assert cluster_centroids.value_counts().equals(
        pd.Series(
            {
                "cdr3_specific_to_Covid19": 25,
                "nonspecific_cdr3": 25,
                "cdr3_specific_to_HIV": 25,
                "cdr3_specific_to_Healthy": 25,
            }
        )
    )
    assert cluster_centroids.loc["Vgene0", "Jgene0", 20, 0.0] == "cdr3_specific_to_HIV"


def test_get_cluster_centroid_deduplication_optimization_is_not_breaking_frequencies():
    def assign_usual_parameters(dat):
        # pretend all sequences belong to same cluster
        return dat.assign(
            v_gene="v_gene1",
            j_gene="j_gene1",
            cdr3_aa_sequence_trim_len=15,
            cluster_id_within_clustering_group=1.0,
        )

    df1 = assign_usual_parameters(
        pd.DataFrame(
            {
                "cdr3_seq_aa_q_trim": ["seqA", "seqA", "seqA", "seqB"],
                "num_clone_members": [1, 1, 1, 2],
            }
        )
    )
    df2 = assign_usual_parameters(
        pd.DataFrame(
            {
                "cdr3_seq_aa_q_trim": ["seqA", "seqB"],
                # This is how df1 should be aggregated
                "num_clone_members": [3, 2],
            }
        )
    )
    df3 = assign_usual_parameters(
        pd.DataFrame(
            {
                "cdr3_seq_aa_q_trim": ["seqA", "seqB"],
                # This is how df1 might be deduplicated improperly - leading to a wrong consensus sequence
                "num_clone_members": [1, 2],
            }
        )
    )
    for df, expected_answer in zip([df1, df2, df3], ["seqA", "seqA", "seqB"]):
        # fast
        cluster_centroids = ConvergentClusterClassifier._get_cluster_centroids(
            clustered_df=df
        )
        slow_comparison = slow_get_cluster_centroids(clustered_df=df)
        assert slow_comparison.equals(cluster_centroids)
        assert cluster_centroids.values[0] == expected_answer


def slow_assign_sequences_to_known_clusters(
    df: pd.DataFrame,
    cluster_centroids_by_supergroup: Dict[Tuple[str, str, int], pd.DataFrame],
    sequence_identity_threshold: float,
    validate_same_fold_id_and_label: bool = True,
):
    # Performs clustering. Overwrites any earlier cluster assignment columns.

    df = df.copy()  # don't modify original df
    if validate_same_fold_id_and_label:
        ConvergentClusterClassifier._validate_same_fold_id_and_label(df)

    # Create higher-order groups: group by v, j, len
    test_groups = df.groupby(
        ["v_gene", "j_gene", "cdr3_aa_sequence_trim_len"],
        observed=True,
        group_keys=False,
    )

    # Assign each test sequence to a cluster with nearest centroid, using higher-order groups as a starting point
    df[["cluster_id_within_clustering_group", "distance_to_nearest_centroid"]] = (
        test_groups["cdr3_seq_aa_q_trim"].transform(
            lambda test_sequences: ConvergentClusterClassifier.assign_sequences_to_cluster_centroids(
                test_sequences,
                cluster_centroids_by_supergroup,
                sequence_identity_threshold,
            )
        )
        # extract the series of tuples into two columns while preserving the index
        # .to_list() is an alternative here, which may be faster, and seems to guarantee index ordering but not sure?
        .apply(pd.Series)
    )

    # If df["cluster_id_within_clustering_group"].isna(), then this test sequence is not assigned to any predictive cluster

    # Create a globally meaningful "resulting cluster ID" for each row of df (each input sequence from each participant):
    df["global_resulting_cluster_ID"] = df[
        [
            "v_gene",
            "j_gene",
            "cdr3_aa_sequence_trim_len",
            "cluster_id_within_clustering_group",
        ]
    ].apply(tuple, axis=1)

    # If any specimens did not come with num_clone_members annotations for their sequences, set to 1
    if "num_clone_members" not in df.columns:
        df["num_clone_members"] = 1
    df["num_clone_members"].fillna(1, inplace=True)
    return df


def test_assign_sequences_to_known_clusters_timing(df_big):
    # setup the problem, like we do in train_convergent_cluster_classifier
    p_value_threshold = 0.05
    sequence_identity_threshold = 0.9
    gene_locus = GeneLocus.BCR
    target_obs_column = TargetObsColumnEnum.disease

    df_clustered = ConvergentClusterClassifier._cluster_training_set(
        df=df_big,
        sequence_identity_threshold=sequence_identity_threshold,
        inplace=False,
    )
    assert not df_clustered["cluster_id_within_clustering_group"].isna().any()
    cluster_pvalues_per_disease = (
        ConvergentClusterClassifier._compute_fisher_scores_for_clusters(
            df_clustered, target_obs_column
        )
    )
    feature_order = cluster_pvalues_per_disease.columns  # these are the classes
    # filter down
    cluster_pvalues_per_disease = cluster_pvalues_per_disease.loc[
        cluster_pvalues_per_disease.min(axis=1) <= 0.1
    ]
    df_clustered = df_clustered[
        df_clustered["global_resulting_cluster_ID"].isin(
            cluster_pvalues_per_disease.index
        )
    ]
    # get centroids
    cluster_centroids = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=df_clustered
    )
    del df_clustered
    # Merge score columns in. Resulting columns are "consensus_sequence" and class-specific score columns (whose names are in feature_names_order)
    cluster_centroids_with_class_specific_p_values = ConvergentClusterClassifier._merge_cluster_centroids_with_cluster_class_association_scores(
        cluster_centroids, cluster_pvalues_per_disease
    )
    # Start featurize(), given a p-value threshold:
    cluster_centroids_filtered = cluster_centroids_with_class_specific_p_values[
        cluster_centroids_with_class_specific_p_values[feature_order].min(axis=1)
        <= p_value_threshold
    ]
    cluster_centroids_by_supergroup = (
        ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            cluster_centroids_filtered
        )
    )

    # Assign each test sequence to known cluster with nearest centroid: compare slow vs fast implementations

    # Expressed as lambdas so we can use in timing test below.
    run_fast = lambda: ConvergentClusterClassifier._assign_sequences_to_known_clusters(
        df=df_big,
        cluster_centroids_by_supergroup=cluster_centroids_by_supergroup,
        sequence_identity_threshold=sequence_identity_threshold,
        validate_same_fold_id_and_label=True,
    )
    run_slow = lambda: slow_assign_sequences_to_known_clusters(
        df=df_big,
        cluster_centroids_by_supergroup=cluster_centroids_by_supergroup,
        sequence_identity_threshold=sequence_identity_threshold,
        validate_same_fold_id_and_label=True,
    )

    # Confirm calculations match.
    fast_result = run_fast()
    slow_result = run_slow()

    # The columns cluster_id_within_clustering_group and distance_to_nearest_centroid can be NaN.
    # Pandas dataframe .equals allows NaNs in the same position without declaring unequal. That's good.

    # However, NaNs in those columns also get incorporated into a column of tuples called global_resulting_cluster_ID.
    # Since that's a column of tuples, NaNs in the same position are unfortunately not considered equal by .equals.
    # However the tuples are just a determinstic combination of other columns, so we can safely ignore them in the comparison.
    # TODO: this is broken in pandas 2.0.
    assert fast_result.drop(columns="global_resulting_cluster_ID").equals(
        slow_result.drop(columns="global_resulting_cluster_ID")
    ), "They should give same result"

    # Do a timing test.
    # Expose local functions to timeit, from https://stackoverflow.com/a/46055465/130164
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    n_iters = 3
    fast_timing = timeit.Timer("run_fast()", globals=imports_and_vars).timeit(
        number=n_iters
    )
    slow_timing = timeit.Timer("run_slow()", globals=imports_and_vars).timeit(
        number=n_iters
    )
    assert (
        fast_timing < slow_timing
    ), f"Optimized function should be faster than slow function. Instead, it took fast: {fast_timing}, vs. slow: {slow_timing} seconds for {n_iters} iterations"


def test_convergent_cluster_featurizer(df):
    df_clustered = ConvergentClusterClassifier._cluster_training_set(
        df=df, sequence_identity_threshold=0.9, inplace=False
    )
    assert df_clustered.shape[0] == df.shape[0]
    assert not df_clustered["cluster_id_within_clustering_group"].isna().any()

    cluster_pvalues_per_disease = (
        ConvergentClusterClassifier._compute_fisher_scores_for_clusters(
            df_clustered, TargetObsColumnEnum.disease
        )
    )
    assert cluster_pvalues_per_disease.shape == (100, 3)
    feature_names_order = cluster_pvalues_per_disease.columns  # these are the classes
    assert np.array_equal(feature_names_order, ["Covid19", "HIV", "Healthy"])
    assert np.array_equal(
        cluster_pvalues_per_disease.index.names,
        [
            "v_gene",
            "j_gene",
            "cdr3_aa_sequence_trim_len",
            "cluster_id_within_clustering_group",
        ],
    )

    cluster_centroids = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=df_clustered
    )
    assert cluster_centroids.shape == (100,)
    assert cluster_centroids.name == "consensus_sequence"
    assert np.array_equal(
        cluster_centroids.index.names,
        [
            "v_gene",
            "j_gene",
            "cdr3_aa_sequence_trim_len",
            "cluster_id_within_clustering_group",
        ],
    )
    assert cluster_centroids.value_counts().equals(
        pd.Series(
            {
                "cdr3_specific_to_Covid19": 25,
                "nonspecific_cdr3": 25,
                "cdr3_specific_to_HIV": 25,
                "cdr3_specific_to_Healthy": 25,
            }
        )
    )
    assert cluster_centroids.loc["Vgene0", "Jgene0", 20, 0.0] == "cdr3_specific_to_HIV"

    cluster_centroids_scored = ConvergentClusterClassifier._merge_cluster_centroids_with_cluster_class_association_scores(
        cluster_centroids, cluster_pvalues_per_disease
    )
    assert cluster_centroids_scored.shape[0] == cluster_centroids.shape[0]
    assert np.array_equal(
        cluster_centroids_scored.index.names,
        [
            "v_gene",
            "j_gene",
            "cdr3_aa_sequence_trim_len",
            "cluster_id_within_clustering_group",
        ],
    )
    assert np.array_equal(
        cluster_centroids_scored.columns,
        ["consensus_sequence", "Covid19", "HIV", "Healthy"],
    )

    for col in feature_names_order:
        # Now check that the significance test pulls out only truly disease-associated clusters
        disease_specific_clusters = cluster_centroids_scored[
            cluster_centroids_scored[col] <= 0.05
        ]
        assert len(disease_specific_clusters) == 25, col
        assert all(
            disease_specific_clusters["consensus_sequence"] == f"cdr3_specific_to_{col}"
        ), col

    # featurize at specimen level
    featurized = ConvergentClusterClassifier._featurize(
        df=df,
        p_value_threshold=0.05,
        cluster_centroids_with_class_specific_p_values=cluster_centroids_scored,
        sequence_identity_threshold=0.85,
        feature_order=feature_names_order,
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
    )

    assert featurized.X.shape == (9, 3)
    expected_features = pd.DataFrame.from_dict(
        {
            "Covid19_participant_0_specimen1": {
                "Covid19": 25.0,
                "HIV": 0.0,
                "Healthy": 0.0,
            },
            "Covid19_participant_1_specimen1": {
                "Covid19": 25.0,
                "HIV": 0.0,
                "Healthy": 0.0,
            },
            "Covid19_participant_2_specimen1": {
                "Covid19": 25.0,
                "HIV": 0.0,
                "Healthy": 0.0,
            },
            "HIV_participant_0_specimen1": {
                "Covid19": 0.0,
                "HIV": 25.0,
                "Healthy": 0.0,
            },
            "HIV_participant_1_specimen1": {
                "Covid19": 0.0,
                "HIV": 25.0,
                "Healthy": 0.0,
            },
            "HIV_participant_2_specimen1": {
                "Covid19": 0.0,
                "HIV": 25.0,
                "Healthy": 0.0,
            },
            "Healthy_participant_0_specimen1": {
                "Covid19": 0.0,
                "HIV": 0.0,
                "Healthy": 25,
            },
            "Healthy_participant_1_specimen1": {
                "Covid19": 0.0,
                "HIV": 0.0,
                "Healthy": 25,
            },
            "Healthy_participant_2_specimen1": {
                "Covid19": 0.0,
                "HIV": 0.0,
                "Healthy": 25,
            },
        },
        orient="index",
    )
    assert type(featurized.X) == pd.DataFrame and featurized.X.equals(
        expected_features
    ), featurized.X
    assert np.array_equal(
        featurized.y,
        [
            "Covid19",
            "Covid19",
            "Covid19",
            "HIV",
            "HIV",
            "HIV",
            "Healthy",
            "Healthy",
            "Healthy",
        ],
    )
    assert np.array_equal(
        featurized.sample_names,
        [
            "Covid19_participant_0_specimen1",
            "Covid19_participant_1_specimen1",
            "Covid19_participant_2_specimen1",
            "HIV_participant_0_specimen1",
            "HIV_participant_1_specimen1",
            "HIV_participant_2_specimen1",
            "Healthy_participant_0_specimen1",
            "Healthy_participant_1_specimen1",
            "Healthy_participant_2_specimen1",
        ],
    )
    assert (
        len(featurized.abstained_sample_names)
        == len(featurized.abstained_sample_metadata)
        == len(featurized.abstained_sample_y)
        == 0
    )

    ## Sanity check some of the inner logic

    cluster_centroids_filtered = cluster_centroids_scored[
        cluster_centroids_scored[feature_names_order].min(axis=1) <= 0.05
    ]
    assert cluster_centroids_filtered.shape == (75, 4)

    cluster_centroids_by_supergroup = (
        ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            cluster_centroids_filtered
        )
    )
    assert len(cluster_centroids_by_supergroup) == 50
    assert ("Vgene0", "Jgene0", 24) in cluster_centroids_by_supergroup.keys()
    assert np.array_equal(
        cluster_centroids_by_supergroup[("Vgene0", "Jgene0", 24)].columns,
        ["cluster_id_within_clustering_group", "consensus_sequence"],
    )
    assert np.array_equal(
        cluster_centroids_by_supergroup[("Vgene0", "Jgene0", 24)].values,
        np.array(
            [[1.0, "cdr3_specific_to_Covid19"], [2.0, "cdr3_specific_to_Healthy"]],
            dtype="object",
        ),
    )

    original_index = df.index.copy()
    df = ConvergentClusterClassifier._assign_sequences_to_known_clusters(
        df=df,
        cluster_centroids_by_supergroup=cluster_centroids_by_supergroup,
        sequence_identity_threshold=0.85,
    )
    assert df.index.equals(
        original_index
    ), "index should not change after _assign_sequences_to_known_clusters"

    # only nonspecific sequences are not matched to any clusters (because those clusters are not chosen as disease-specific)
    na_counts = {
        key: grp["cluster_id_within_clustering_group"]
        .isna()
        .value_counts()
        .reindex([False, True], fill_value=0)
        .loc[True]
        for key, grp in df.groupby(["cdr3_seq_aa_q_trim"])
    }
    assert na_counts == {
        "cdr3_specific_to_Covid19": 0,
        "cdr3_specific_to_HIV": 0,
        "cdr3_specific_to_Healthy": 0,
        "nonspecific_cdr3": 4500,
    }
