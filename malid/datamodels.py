from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

# Fix Enum type checking:
if TYPE_CHECKING:
    # Type checking has special-cased Enum, but not aenum: https://github.com/ethanfurman/aenum/issues/10
    # import enum as aenum
    from enum import Enum, auto, Flag
else:
    # The actual runtime import
    from aenum import Enum, auto, Flag

import anndata

from enum_mixins import (
    IterableFlagMixin,
    ValidatableEnumMixin,
    ValidatableFlagMixin,
    NonNullableFlagMixin,
    CustomNameSeparatorFlagMixin,
)

## Constants:

healthy_label = "Healthy/Background"

DEMOGRAPHICS_COLUMNS = ["age", "sex", "ethnicity_condensed"]

# Expand with "age_group", which is a categorical column that is nulled out for rare extreme ages so they don't confound our demographics-controlled models
# (The original "age" column is not nulled out in those cases. Use DEMOGRAPHICS_COLUMNS_EXPANDED when we want to ensure "all demographics are present")
DEMOGRAPHICS_COLUMNS_EXPANDED = DEMOGRAPHICS_COLUMNS + ["age_group"]


## Model options:


@dataclass(frozen=True)
# TODO: rename to ClassificationTarget
class TargetObsColumn:
    """Describes one target obs column.

    These are used as values for TargetObsColumnEnum. Note that the TargetObsColumnEnum option's name will be used as the output name.

    obs_column_name: name of the column in anndata obs
    limited_to_disease: optional List[str]; if set, anndata should be subset to specimens from these diseases only
    confusion_matrix_expanded_column_name: optional alternative column name used for confusion matrix and metrics
    blended_evaluation_column_name: optional alternative column name used for blended (final patient-level prediction) evaluation

    e.g. "age_group_healthy_only" is the enum name (i.e. the output name), but uses the "age_group" obs column, and only for the healthy set of specimens.

    TODO: Rename is_target_binary_for_repertoire_composition_classifier to indicate that this is for disease target only. Or deprecate.
    """

    obs_column_name: str
    is_target_binary_for_repertoire_composition_classifier: bool
    limited_to_disease: Optional[List[str]] = None
    require_metadata_columns_present: Optional[List[str]] = None
    confusion_matrix_expanded_column_name: Optional[str] = None
    blended_evaluation_column_name: Optional[str] = None


# TODO: rename to ClassificationTargetEnum
class TargetObsColumnEnum(ValidatableEnumMixin, Enum):
    """Enum of potential target obs columns.

    To accept this in a function:
    - argument type hint: `target_obs_column: TargetObsColumnEnum = TargetObsColumnEnum.disease`
    - validate: `TargetObsColumnEnum.validate(target_obs_column)`
    - access inner TargetObsColumn value: `target_obs_column.value`

    To loop over all targets: `for target in TargetObsColumnEnum: print(target.name, target.value)`

    See confounder_model.ipynb for how these are created, and helpers.py for how metadata is loaded.
    """

    # Breaking convention of Enum capital names, to match output names:
    # Standard Mal-ID target
    disease = TargetObsColumn(
        obs_column_name="disease",
        # Make model1 predict disease directly, rather than Healthy/Sick
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=None,
        require_metadata_columns_present=None,
        confusion_matrix_expanded_column_name="disease.separate_past_exposures",
        blended_evaluation_column_name="disease.rollup",
    )

    # Also predict disease for all-cohort specimens who have age+sex+ethnicity defined:
    # disease_all_demographics_present is never "healthy cohort only" (it is just "disease" column but only for specimens who have age+sex+ethnicity data and not in spurious cohorts)
    disease_all_demographics_present = TargetObsColumn(
        obs_column_name="disease",
        # Make model1 predict disease directly, rather than Healthy/Sick
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=None,
        require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED,
        confusion_matrix_expanded_column_name="disease.separate_past_exposures",
        blended_evaluation_column_name="disease.rollup",
    )

    # Predict age/sex/ethnicity status for healthy specimens
    # for sequence-based predictions models, we will use healthy cohort only for these variables:
    ethnicity_condensed_healthy_only = TargetObsColumn(
        obs_column_name="ethnicity_condensed",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=[healthy_label],
        require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED,
    )
    age_group_healthy_only = TargetObsColumn(
        obs_column_name="age_group",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=[healthy_label],
        require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED,
    )
    age_group_binary_healthy_only = TargetObsColumn(
        obs_column_name="age_group_binary",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=[healthy_label],
        require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED,
    )
    age_group_pediatric_healthy_only = TargetObsColumn(
        obs_column_name="age_group_pediatric",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=[healthy_label],
        require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED,
    )
    sex_healthy_only = TargetObsColumn(
        obs_column_name="sex",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=[healthy_label],
        require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED,
    )

    # Predict each disease vs healthy separately
    # TODO: auto-generate?
    # But enums should be immutable, right? (aenum disagrees: https://stackoverflow.com/a/35899963/130164)
    # And we don't have helpers.diseases_in_peak_timepoint_dataset() available until after ETL step.
    # For now, just manually list them.
    # TODO: can we extract disease-vs-healthy distinguishing features from multiclass all-diseases model directly?
    covid_vs_healthy = TargetObsColumn(
        obs_column_name="disease",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=["Covid19", healthy_label],
        confusion_matrix_expanded_column_name="disease.separate_past_exposures",
        blended_evaluation_column_name="disease.rollup",
        require_metadata_columns_present=None,
    )
    hiv_vs_healthy = TargetObsColumn(
        obs_column_name="disease",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=["HIV", healthy_label],
        confusion_matrix_expanded_column_name="disease.separate_past_exposures",
        blended_evaluation_column_name="disease.rollup",
        require_metadata_columns_present=None,
    )
    lupus_vs_healthy = TargetObsColumn(
        obs_column_name="disease",
        is_target_binary_for_repertoire_composition_classifier=False,
        limited_to_disease=["Lupus", healthy_label],
        confusion_matrix_expanded_column_name="disease.separate_past_exposures",
        blended_evaluation_column_name="disease.rollup",
        require_metadata_columns_present=None,
    )

    @property
    def value(self) -> TargetObsColumn:
        # Type hint enum values as TargetObsColumn rather than Any: https://github.com/python/typing/issues/535#issuecomment-907352538
        # In the future, maybe there will be typing support for Enum[TargetObsColumn] directly?
        return super().value


class SampleWeightStrategy(ValidatableEnumMixin, Enum):
    """Enum of sample weight strategies. Can only use one at a time."""

    # no sample weighting
    NONE = auto()
    # sample weighting by isotype usage
    ISOTYPE_USAGE = auto()


class GeneLocus(
    IterableFlagMixin,
    ValidatableFlagMixin,
    NonNullableFlagMixin,
    CustomNameSeparatorFlagMixin,
    Flag,
):
    """Define which gene locus we analyze.

    This is a Flag, so we can multi-pack values into a single GeneLocus: e.g. GeneLocus.BCR | GeneLocus.TCR.

    We use a Flag here because we may operate on multiple loci at the same time, like in the metamodel (i.e. combine BCR and TCR base models).
    It's convenient because we can iterate over multi-packed values and we get a determinstic name ("BCR_TCR"), regardless of input order.

    However, all base models operate on a single locus, i.e. they forbid multi-packed GeneLocus entries.
    To be sure you have a single locus, call `GeneLocus.validate_single_value(gene_locus)`.
    """

    __default_separator__ = "_"  # for nice filenames
    BCR = auto()
    TCR = auto()


class UmapSupervisionStrategy(ValidatableEnumMixin, Enum):
    """Enum of UMAP supervision strategies."""

    UNSUPERVISED = auto()
    # fully supervised UMAP using high-confidence subset only
    FULLY_SUPERVISED = auto()
    # partially supervised UMAP with full dataset and only some high-confidence points labeled
    PARTIALLY_SUPERVISED = auto()


class Repertoire(anndata.AnnData):
    # TODO: wrap anndata with class wrapper and expose extension methods
    # write tests to confirm that copy() and implicit view->copy conversion return new Repertoire objects
    pass


def combine_classification_option_names(
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.NONE,
) -> str:
    """useful for generating directory names"""

    if sample_weight_strategy != SampleWeightStrategy.NONE:
        sample_weight_strategy_str = (
            f"_sample_weight_strategy_{sample_weight_strategy.name}"
        )
    else:
        sample_weight_strategy_str = ""

    combined_name = f"{target_obs_column.name}{sample_weight_strategy_str}"
    if "/" in combined_name:
        raise ValueError(f"combined_name cannot contain '/': {combined_name}")
    return combined_name


class CVScorer(ValidatableEnumMixin, Enum):
    """
    Scoring functions for internal nested cross validation.
    """

    MCC = auto()
    AUC = auto()
