import numpy as np
import pandas as pd
import pytest

from malid.datamodels import GeneLocus, TargetObsColumnEnum
from malid.trained_model_wrappers import ExactMatchesClassifier
from malid import io


@pytest.fixture
def df():
    n_patients_per_disease = 3
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
    return df


def test_exact_matches_featurizer(df):
    results = ExactMatchesClassifier._compute_fisher_scores_for_sequences(
        df, TargetObsColumnEnum.disease
    )
    assert results.shape == (100, 3)
    assert np.array_equal(results.columns, ["Covid19", "HIV", "Healthy"])
    assert np.array_equal(
        results.index.names,
        ["v_gene", "j_gene", "cdr3_aa_sequence_trim_len", "cdr3_seq_aa_q_trim"],
    )

    filtered_results = ExactMatchesClassifier._filter_fisher_scores_to_pvalue_threshold(
        results, p_value_threshold=0.05
    )
    assert filtered_results.shape == (75, 3)

    for col in filtered_results.columns:
        assert all(
            results[results[col] <= 0.05].reset_index()["cdr3_seq_aa_q_trim"]
            == f"cdr3_specific_to_{col}"
        )

    # featurize at specimen level
    featurized = ExactMatchesClassifier._featurize(
        df=df,
        sequences_with_fisher_result=results,
        p_value_threshold=0.05,
        feature_order=["Covid19", "HIV", "Healthy"],
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
    )
    expected_features = pd.DataFrame.from_dict(
        {
            "Covid19_participant_0_specimen1": {
                "Covid19": 500.0,
                "HIV": 0.0,
                "Healthy": 0.0,
            },
            "Covid19_participant_1_specimen1": {
                "Covid19": 500.0,
                "HIV": 0.0,
                "Healthy": 0.0,
            },
            "Covid19_participant_2_specimen1": {
                "Covid19": 500.0,
                "HIV": 0.0,
                "Healthy": 0.0,
            },
            "HIV_participant_0_specimen1": {
                "Covid19": 0.0,
                "HIV": 500.0,
                "Healthy": 0.0,
            },
            "HIV_participant_1_specimen1": {
                "Covid19": 0.0,
                "HIV": 500.0,
                "Healthy": 0.0,
            },
            "HIV_participant_2_specimen1": {
                "Covid19": 0.0,
                "HIV": 500.0,
                "Healthy": 0.0,
            },
            "Healthy_participant_0_specimen1": {
                "Covid19": 0.0,
                "HIV": 0.0,
                "Healthy": 500.0,
            },
            "Healthy_participant_1_specimen1": {
                "Covid19": 0.0,
                "HIV": 0.0,
                "Healthy": 500.0,
            },
            "Healthy_participant_2_specimen1": {
                "Covid19": 0.0,
                "HIV": 0.0,
                "Healthy": 500.0,
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
