from malid.datamodels import (
    SampleWeightStrategy,
    TargetObsColumnEnum,
    combine_classification_option_names,
)
from malid import helpers


def test_disease_restrictions_match_known_disease_names():
    # confirm that each limited_to_disease entry matches a known disease name
    for target_obs_column in TargetObsColumnEnum:
        if target_obs_column.value.limited_to_disease is not None:
            for disease_name in target_obs_column.value.limited_to_disease:
                assert disease_name in helpers.diseases


def test_combine_classification_option_names():
    assert (
        combine_classification_option_names(TargetObsColumnEnum.age_group_healthy_only)
        == "age_group_healthy_only"
    )
    assert (
        combine_classification_option_names(
            TargetObsColumnEnum.age_group_healthy_only,
            sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
        )
        == "age_group_healthy_only_sample_weight_strategy_ISOTYPE_USAGE"
    )
