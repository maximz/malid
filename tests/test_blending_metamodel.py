import numpy as np
import pandas as pd
from malid.trained_model_wrappers import BlendingMetamodel
from malid.trained_model_wrappers.blending_metamodel import DemographicsFeaturizer


def test_demographics_featurizer_respects_binary_ordered_categorical_column_order():
    # Usually whichever value is seen first is the one that's chosen for binary columns.
    df = pd.DataFrame({"sex": ["F", "M"]})
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_F"]
    )
    df = pd.DataFrame({"sex": ["M", "F"]})
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_M"]
    )

    # But when the binary column has an ordered categorical dtype, the dtype's specified order should be respected.
    df = pd.DataFrame({"sex": ["F", "M"]})
    df["sex"] = df["sex"].astype(
        pd.CategoricalDtype(categories=["M", "F"], ordered=True)
    )
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_F"]
    )
    df = pd.DataFrame({"sex": ["M", "F"]})
    df["sex"] = df["sex"].astype(
        pd.CategoricalDtype(categories=["M", "F"], ordered=True)
    )
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_F"]
    )


def test_convert_feature_name_to_friendly_name():
    for input, expected in [
        # pass through any unmatched
        ("gibberish", "gibberish"),
        (
            "BCR:sequence_model:Healthy/Background",
            "Language embedding (BCR): P(Healthy/Background)",
        ),
        (
            "demographics:ethnicity_condensed_African",
            "Demographics: ethnicity_condensed_African",
        ),
        ("isotype_counts:isotype_proportion:IGHG", "Isotype proportion of IGHG"),
        (
            "interaction|BCR:convergent_cluster_model:HIV|demographics:age",
            "Interaction: [CDR3 clustering (BCR): P(HIV)] x [Demographics: age]",
        ),
        (
            "interaction|TCR:repertoire_stats:Covid19|demographics:ethnicity_condensed_Hispanic/Latino",
            "Interaction: [Repertoire composition (TCR): P(Covid19)] x [Demographics: ethnicity_condensed_Hispanic/Latino]",
        ),
    ]:
        observed = BlendingMetamodel.convert_feature_name_to_friendly_name(input)
        assert (
            observed == expected
        ), f"Expected {input} to be converted to '{expected}', but got '{observed}'"
