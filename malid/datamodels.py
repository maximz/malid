import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Union, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import anndata
from typing_extensions import Self

# Fix Enum type checking:
if TYPE_CHECKING:
    # Type checking has special-cased Enum, but not aenum: https://github.com/ethanfurman/aenum/issues/10
    # import enum as aenum
    from enum import Enum, auto, Flag
else:
    # The actual runtime import
    from aenum import Enum, auto, Flag


from enum_mixins import (
    IterableFlagMixin,
    ValidatableEnumMixin,
    ValidatableFlagMixin,
    NonNullableFlagMixin,
    CustomNameSeparatorFlagMixin,
)

## Constants:

healthy_label = "Healthy/Background"

# Update this with new diseases:
# And make sure to add to the each-disease-vs-healthy classification targets below
diseases = [
    "Covid19",
    "HIV",
    healthy_label,
    "Lupus",
    "Crohn's disease",
    "Ulcerative colitis",
    "Influenza",
    "T1D",
    "T2D",
    # Adaptive only:
    "RA",
    "CVID",  # avoiding the trap of obvious BCR differences, because the Adaptive data is TCR only
]


DEMOGRAPHICS_COLUMNS = ["age", "sex", "ethnicity_condensed"]

# Expand with "age_group", which is a categorical column that is nulled out for rare extreme ages so they don't confound our demographics-controlled models
# (The original "age" column is not nulled out in those cases. Use DEMOGRAPHICS_COLUMNS_EXPANDED when we want to ensure "all demographics are present")
DEMOGRAPHICS_COLUMNS_EXPANDED = DEMOGRAPHICS_COLUMNS + ["age_group"]

## Model options:


class DataSource(ValidatableEnumMixin, Enum):
    """
    Enum of data source names.
    """

    in_house = auto()
    adaptive = auto()
    external_cdna = auto()


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


@dataclass(frozen=True)
class CrossValidationSplitStrategyValue:
    """
    Choose which specimens should be included for each cross validation split strategy. For example, choose "peak timepoint" specimens for the peak-timepoints-only dataset.
    The filtering logic is implemented in helpers.get_all_specimen_info().

    The rule is: match data_sources_keep AND {either diseases_to_keep_all_subtypes OR subtypes_keep},
    then apply filter_specimens_func_by_study_name,
    then filter down further with any filter_out_specimens_funcs_global (this step cannot bring back anything that was already filtered out).

    - data_sources_keep cannot be empty, otherwise no data will be selected.
    - either diseases_to_keep_all_subtypes or subtypes_keep must be provided, otherwise no data will be selected.
    - you must also register the disease names in diseases_to_keep_all_subtypes under datamodels.diseases.
    - filter_specimens_func_by_study_name and filter_out_specimens_funcs_global are both optional.
    """

    data_sources_keep: List[DataSource]

    # Which column name to use for stratification in cross validation
    stratify_by: str = "disease"

    # For these diseases, all disease_subtypes will be kept:
    diseases_to_keep_all_subtypes: List[str] = dataclasses.field(default_factory=list)

    # Also keep these disease_subtypes from any other diseases:
    subtypes_keep: List[str] = dataclasses.field(default_factory=list)

    # Optional: map study name to additional filtering function, e.g. some study names may have special "peak timepoint" filters
    filter_specimens_func_by_study_name: Dict[
        str, Callable[[pd.DataFrame], pd.Index]
    ] = dataclasses.field(default_factory=dict)

    # Optional: which gene loci are supported by this split strategy? Are there any restrictions? (TODO: Move to DataSource?)
    gene_loci_supported: GeneLocus = GeneLocus.BCR | GeneLocus.TCR

    # Optional: exclude study names altogether. Defaults to empty list (no exclusions)
    exclude_study_names: List[str] = dataclasses.field(default_factory=list)

    # Optional: whitelist specific study names. All other study names will be excluded.
    # Defaults to None (no additional whitelisting applied).
    # If this field is provided, exclude_study_names will be ignored.
    include_study_names: Optional[List[str]] = None

    # Optional: apply additional filtering functions to the entire dataset, after the other filters have been applied.
    filter_out_specimens_funcs_global: List[
        Callable[[pd.DataFrame], pd.Index]
    ] = dataclasses.field(default_factory=list)

    # Optional: "leave one cohort out"
    # Should any study names be separated out as a held-out test set?
    # This will disable CV splitting in the traditional sense. What normally happens (when this field is not provided) is that all selected studies are used for training and testing by being divided into several cross validation folds.
    # Instead when this field is provided, we override the divisions:
    # - Only one fold will be created.
    # - We will produce a *single* test set with the indicated study names.
    # - All other studies will be used for training only.
    # - (There will still be a train/validation split for base models vs metamodel, but the key idea is we get a single CV fold)
    # Before this separation happens, all the other filters specified in CrossValidationSplitStrategyValue will be applied. Remaining specimens must meet the other fields' criteria.
    # Each entry in this list should be a valid study_name:
    study_names_for_held_out_set: Optional[List[str]] = None

    def __post_init__(self):
        # Validation:
        # This check runs when the enum is defined at runtime, meaning the first time datamodels is imported (which usually happens very early).
        # So our automated tests will catch these errors.

        # Confirm the enum values are valid.
        GeneLocus.validate(self.gene_loci_supported)
        for data_source in self.data_sources_keep:
            DataSource.validate(data_source)

        # Confirm that either diseases_to_keep_all_subtypes or subtypes_keep is provided
        if len(self.diseases_to_keep_all_subtypes) + len(self.subtypes_keep) == 0:
            raise TypeError(
                "Either diseases_to_keep_all_subtypes or subtypes_keep must be provided for CrossValidationSplitStrategyValue."
            )

    @property
    def is_single_fold_only(self) -> bool:
        """
        Usually we have multiple folds (is_single_fold_only == False), except for one special case: single fold with a pre-determined held-out test set. If so, is_single_fold_only will be True.
        This is the case for cross validation split strategies that have study_names_for_held_out_set defined; that forces a single fold. See study_names_for_held_out_set docs.
        """
        return (
            self.study_names_for_held_out_set is not None
            and len(self.study_names_for_held_out_set) > 0
        )


def acute_disease_choose_most_peak_timepoint(df: pd.DataFrame) -> pd.Index:
    """
    For a single study, choose among the specimens that have passed quality filters:
    Choose the single most peak timepoint specimen for each participant. Ideally closest to day 15.
    But ok with anything from day 10 to day 40 after onset of symptoms.

    Ensure that index is unique so that groupby->idxmin works.
    """
    subset = df[
        (df["specimen_time_point_days"] >= 7) & (df["specimen_time_point_days"] <= 40)
    ].copy()

    # Compute difference from day 15
    subset["diff_from_15"] = (subset["specimen_time_point_days"] - 15).abs()

    # Minimize difference from day 15 for each individual
    selected_index = subset.groupby("participant_label")["diff_from_15"].idxmin()

    return selected_index


def keep_specimens_with_known_disease_severity(df: pd.DataFrame) -> pd.Index:
    """
    For a single study: keep only those specimens that have known disease severity.
    """
    return df[df["disease_severity"].notna()].index


class CrossValidationSplitStrategy(ValidatableEnumMixin, Enum):
    """
    Enum of configurations for which samples are included in the cross-validation training and testing sets.
    See CrossValidationSplitStrategyValue docs for details.
    Be sure to register any new values in the map_cross_validation_split_strategy_to_default_target_obs_column dictionary, and to add new values to available_for_cross_validation_split_strategies in the relevant TargetObsColumnEnum's.
    TODO: Add test that runs each filtering strategy and confirms that the resulting dataset is non-empty and has certain expected entries.
    """

    # Breaking convention of Enum capital names, to match output names:
    in_house_peak_disease_timepoints = CrossValidationSplitStrategyValue(
        data_sources_keep=[
            DataSource.in_house,
        ],
        # For some diseases, do subtype filtering:
        subtypes_keep=[
            # Covid19-buffycoat (Cell Host + Microbe):
            "Covid19 - Sero-positive (ICU)",
            "Covid19 - Sero-positive (Admit)",
            # Covid19-Seattle:
            "Covid19 - Acute 2",
            # Covid19-Stanford:
            "Covid19 - Admit",
            "Covid19 - ICU",
            # Exclude Crohn's and ulcerative colitis remission subtypes (Yoni) or non-day-0 timepoints (Gubatan):
            # "Crohn's disease - Flare",
            # "Ulcerative colitis - Flare",
            # "Crohn's disease - day 0",
            # "Ulcerative colitis - day 0",
            # Flu vaccine:
            "Influenza vaccine 2021 - day 7",  # (day 0 can be added as healthy, pre-vaccination timepoint)
        ],
        # Don't do any disease-subtype filtering for some other diseases — keep all of their subtypes:
        diseases_to_keep_all_subtypes=[
            healthy_label,
            "HIV",
            "Lupus",
            "T1D",
        ],
        filter_specimens_func_by_study_name={
            "Covid19-buffycoat": acute_disease_choose_most_peak_timepoint,
            "Covid19-Stanford": acute_disease_choose_most_peak_timepoint,
            # Covid19-Seattle does not have granular timepoint data
        },
        exclude_study_names=["IBD pre-pandemic Yoni"],
    )

    # Keep all other filtering the same, but create a single CV fold with specific study names in the hold-out test set.
    in_house_peak_disease_leave_one_cohort_out = dataclasses.replace(
        in_house_peak_disease_timepoints,
        study_names_for_held_out_set=[
            "Covid19-buffycoat",
            # The healthy controls that were included in the resequencing experiment have a special study name.
            # (Note that the study name corresponds to both copies of these specimens. The sequences corrresponding to each copy can be distinguished by amplification label.)
            "Healthy-StanfordBloodCenter_included-in-resequencing",
        ],
    )
    # Repeat the above, but leave one lupus cohort out.
    in_house_peak_disease_leave_one_lupus_cohort_out = dataclasses.replace(
        in_house_peak_disease_timepoints,
        study_names_for_held_out_set=[
            "New Lupus Paxgene",
        ],
    )

    adaptive_peak_disease_timepoints = CrossValidationSplitStrategyValue(
        # Adaptive is TCR only
        gene_loci_supported=GeneLocus.TCR,
        data_sources_keep=[
            # Necessary to specify this, otherwise we'll pull in in-house data for these diseases too
            DataSource.adaptive,
        ],
        subtypes_keep=[
            # Among RA cases: Keep seropositive RA only
            "RA - sero-positive",
        ],
        # Don't do any disease-subtype filtering for these diseases. Keep all of their subtypes:
        diseases_to_keep_all_subtypes=[
            "Covid19",
            "CVID",
            healthy_label,
            "HIV",
            "T1D",
        ],
    )

    # Create a single CV fold with specific study names in the train and in the hold-out test sets.
    adaptive_peak_disease_timepoints_leave_some_cohorts_out = CrossValidationSplitStrategyValue(
        # Adaptive is TCR only
        gene_loci_supported=GeneLocus.TCR,
        data_sources_keep=[
            # Necessary to specify this, otherwise we'll pull in in-house data for these diseases too
            DataSource.adaptive,
        ],
        #
        # TODO(refactor): introduce study_names_for_train in make_cv_folds:
        #   replace first argument of (0, (np.where(~held_out_bool_array)[0], np.where(held_out_bool_array)[0])) with something that instead does np.where(included_in_train_bool_array)
        #   also in helpers.py, change df["is_selected_for_cv_strategy"] to filter down if both study_names_for_train and study_names_for_held_out_set are provided.
        #
        # But for now, we will achieve this with a combination of study_names_for_held_out_set and include_study_names.
        # Only these study names will be loaded and passed through following filters:
        # study_names_for_train=['emerson-2017-natgen_train', 'immunecode-NIH', 'immunecode-ISB'],
        study_names_for_held_out_set=[
            "emerson-2017-natgen_validation",
            # TODO(refactor):
            # The Immunecode Covid-19 cohort is divided into study names in helpers.py right now.
            # Move this to ETL metadata creation: change study_name for immunecode to include NIH, Huniv, ISB etc. there.
            "immunecode-HUniv",
        ],
        include_study_names=[
            "emerson-2017-natgen_train",
            "immunecode-NIH",
            "immunecode-ISB",
            "emerson-2017-natgen_validation",
            "immunecode-HUniv",
        ],
        # Do not pick up healthy controls from these studies:
        # "mitchell-2022-jcii"
        # "mustjoki-2017-natcomms"
        # "ramesh-2015-ci"
        # "TCRBv4-control"
        #
        #
        diseases_to_keep_all_subtypes=["Covid19", healthy_label],
    )

    # Define custom cross validation folds to predict lupus nephritis.
    in_house_lupus_nephritis = CrossValidationSplitStrategyValue(
        data_sources_keep=[
            DataSource.in_house,
        ],
        diseases_to_keep_all_subtypes=["Lupus"],
        # Filter down to specimens with known nephritis/no-nephritis status
        filter_out_specimens_funcs_global=[keep_specimens_with_known_disease_severity],
        stratify_by="disease_severity",
    )

    external_data = CrossValidationSplitStrategyValue(
        data_sources_keep=[
            DataSource.external_cdna,  # AIRR format data
        ],
        # List your diseases here:
        diseases_to_keep_all_subtypes=["HIV", "Lupus"],
        # Or mark certain disease subtypes for inclusion:
        subtypes_keep=[
            "Covid19 - Sero-positive (Admit)",  # Other subtypes of this disease will not be included
        ],
        # Adjust depending on which data types you have:
        gene_loci_supported=GeneLocus.BCR | GeneLocus.TCR,
    )


@dataclass(frozen=True)
# TODO: rename to ClassificationTarget
class TargetObsColumn:
    """Describes one target obs column.

    These are used as values for TargetObsColumnEnum. Note that the TargetObsColumnEnum option's name will be used as the output name.

    obs_column_name: name of the column in anndata obs
    limited_to_disease, limited_to_disease_subtype, limited_to_specimen_description: optional List[str]; if set, anndata should be subset to specimens from these diseases/disease_subtypes/specimen_descriptions only
    confusion_matrix_expanded_column_name: optional alternative column name used for confusion matrix and metrics
    blended_evaluation_column_name: optional alternative column name used for blended (final patient-level prediction) evaluation

    e.g. "age_group_healthy_only" is the enum name (i.e. the output name), but uses the "age_group" obs column, and only for the healthy set of specimens.
    """

    obs_column_name: str

    # Deprecated: is_target_binary_for_repertoire_composition_classifier.
    # If true, will revert model 1 to predicting Healthy/Sick instead of predicting target disease class directly.
    # TODO: Rename is_target_binary_for_repertoire_composition_classifier to indicate that this is for disease target only. Or deprecate.
    is_target_binary_for_repertoire_composition_classifier: bool = False

    # For which cross validation split strategies is this classification target available? Will be skipped for the rest.
    available_for_cross_validation_split_strategies: Set[
        CrossValidationSplitStrategy
    ] = dataclasses.field(
        # by default, no restrictions; available for all cross validation split strategies
        # (default_factory is necessary to avoid mutable default value)
        default_factory=lambda: set(CrossValidationSplitStrategy)
    )

    # Filters on sequences or specimens to arrive at a subset of the data to use for this target
    limited_to_disease: Optional[List[str]] = None
    limited_to_disease_subtype: Optional[List[str]] = None
    limited_to_specimen_description: Optional[List[str]] = None
    limited_to_study_name: Optional[List[str]] = None
    # composable (can be multi-value flag):
    limited_to_gene_locus: Optional[GeneLocus] = None
    # Support custom filtering: accept anndata.obs row, return bool indicating whether to keep it
    filter_adata_obs_func: Optional[Callable[[pd.Series], bool]] = None

    require_metadata_columns_present: Optional[List[str]] = None

    confusion_matrix_expanded_column_name: Optional[str] = None
    blended_evaluation_column_name: Optional[str] = None

    # Allow overriding the default p-values tried by the converging clustering classifier
    convergent_clustering_p_values: Optional[Union[List[float], np.ndarray]] = None

    def __post_init__(self):
        # Confirm enum values are valid
        for strategy in self.available_for_cross_validation_split_strategies:
            CrossValidationSplitStrategy.validate(strategy)
        if self.limited_to_gene_locus is not None:
            # this one is optional, may not be provided
            GeneLocus.validate(self.limited_to_gene_locus)


# TODO: rename to ClassificationTarget and ClassificationTargetValue
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
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints,
            CrossValidationSplitStrategy.in_house_peak_disease_leave_one_cohort_out,
            CrossValidationSplitStrategy.adaptive_peak_disease_timepoints,
            CrossValidationSplitStrategy.adaptive_peak_disease_timepoints_leave_some_cohorts_out,
            CrossValidationSplitStrategy.in_house_peak_disease_leave_one_lupus_cohort_out,
            CrossValidationSplitStrategy.external_data,
        },
        limited_to_disease=None,
        require_metadata_columns_present=None,
        confusion_matrix_expanded_column_name="disease.separate_past_exposures",
        blended_evaluation_column_name="disease.rollup",
    )

    # Also predict disease for all-cohort specimens who have age+sex+ethnicity defined
    disease_all_demographics_present = dataclasses.replace(
        disease, require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED
    )

    # Predict age/sex/ethnicity status for healthy specimens
    ethnicity_condensed_healthy_only = TargetObsColumn(
        obs_column_name="ethnicity_condensed",
        # Activate for split strategies that have healthy specimens
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints,
            CrossValidationSplitStrategy.in_house_peak_disease_leave_one_cohort_out,
            CrossValidationSplitStrategy.adaptive_peak_disease_timepoints,
            CrossValidationSplitStrategy.adaptive_peak_disease_timepoints_leave_some_cohorts_out,
        },
        limited_to_disease=[healthy_label],
        require_metadata_columns_present=DEMOGRAPHICS_COLUMNS_EXPANDED,
    )
    age_group_healthy_only = dataclasses.replace(
        ethnicity_condensed_healthy_only,
        obs_column_name="age_group",
    )
    age_group_binary_healthy_only = dataclasses.replace(
        ethnicity_condensed_healthy_only,
        obs_column_name="age_group_binary",
    )
    age_group_pediatric_healthy_only = dataclasses.replace(
        ethnicity_condensed_healthy_only,
        obs_column_name="age_group_pediatric",
    )
    sex_healthy_only = dataclasses.replace(
        ethnicity_condensed_healthy_only,
        obs_column_name="sex",
    )

    # Predict each disease vs healthy separately
    # TODO: auto-generate?
    # But enums should be immutable, right? (aenum disagrees: https://stackoverflow.com/a/35899963/130164)
    # And we don't have helpers.diseases_in_peak_timepoint_dataset() available until after ETL step.
    # For now, just manually list them.
    # TODO: can we extract disease-vs-healthy distinguishing features from multiclass all-diseases model directly?
    covid_vs_healthy = dataclasses.replace(
        disease,
        limited_to_disease=["Covid19", healthy_label],
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints,
            CrossValidationSplitStrategy.adaptive_peak_disease_timepoints,
        },
    )
    hiv_vs_healthy = dataclasses.replace(
        disease,
        limited_to_disease=["HIV", healthy_label],
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints,
            CrossValidationSplitStrategy.adaptive_peak_disease_timepoints,
        },
    )
    t1d_vs_healthy = dataclasses.replace(
        disease,
        limited_to_disease=["T1D", healthy_label],
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints
        },
    )
    # only in in-house:
    lupus_vs_healthy = dataclasses.replace(
        disease,
        limited_to_disease=["Lupus", healthy_label],
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints
        },
    )
    flu_vs_healthy = dataclasses.replace(
        disease,
        limited_to_disease=["Influenza", healthy_label],
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints
        },
    )

    # Lupus symptoms:
    # Many of these, besides nephritis, are only available in BCR-only cohorts for now.
    # Override default range of p-values tried by the convergent clustering classifier with a looser set of p-values,
    # because we saw that the default set of p-values led to featurization failure (all abstentions) or high abstention rates for some subtype classifiers.
    lupus_nephritis = TargetObsColumn(
        obs_column_name="disease_severity",
        limited_to_disease=["Lupus"],
        convergent_clustering_p_values=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        available_for_cross_validation_split_strategies={
            CrossValidationSplitStrategy.in_house_peak_disease_timepoints,
            CrossValidationSplitStrategy.in_house_lupus_nephritis,
        },
    )

    def confirm_compatibility_with_gene_locus(self, gene_locus: GeneLocus) -> None:
        """
        Validate that this target is defined for the given gene locus. Raise an error if not.
        Use this to enforce the optional `limited_to_gene_locus` field.
        """
        if (
            self.value.limited_to_gene_locus is not None
            and gene_locus not in self.value.limited_to_gene_locus
        ):
            raise ValueError(f"Gene locus {gene_locus} is not valid for target {self}")

    def confirm_compatibility_with_cross_validation_split_strategy(
        self, cross_validation_split_strategy: CrossValidationSplitStrategy
    ) -> None:
        """
        Validate that this target is defined for the given gene locus. Raise an error if not.
        Use this to enforce the optional `limited_to_gene_locus` field.
        """
        if (
            cross_validation_split_strategy
            not in self.value.available_for_cross_validation_split_strategies
        ):
            raise ValueError(
                f"Cross validation split strategy {cross_validation_split_strategy} is not valid for target {self}"
            )


# Define a default TargetObsColumn for each cross validation split strategy.
# For example, TargetObsColumnEnum.disease is a good default for our interpretations/analyses, but that's not the case for every cross validation split strategy (where "disease" might not even be a valid target).
# We can't define this as a property of CrossValidationSplitStrategy itself, because TargetObsColumnEnum has not yet been defined by that point.
map_cross_validation_split_strategy_to_default_target_obs_column = {
    CrossValidationSplitStrategy.in_house_peak_disease_timepoints: TargetObsColumnEnum.disease,
    CrossValidationSplitStrategy.in_house_peak_disease_leave_one_cohort_out: TargetObsColumnEnum.disease,
    CrossValidationSplitStrategy.adaptive_peak_disease_timepoints: TargetObsColumnEnum.disease,
    CrossValidationSplitStrategy.adaptive_peak_disease_timepoints_leave_some_cohorts_out: TargetObsColumnEnum.disease,
    CrossValidationSplitStrategy.in_house_lupus_nephritis: TargetObsColumnEnum.lupus_nephritis,
    CrossValidationSplitStrategy.in_house_peak_disease_leave_one_lupus_cohort_out: TargetObsColumnEnum.disease,
    CrossValidationSplitStrategy.external_data: TargetObsColumnEnum.disease,
}


class SampleWeightStrategy(
    IterableFlagMixin,
    ValidatableFlagMixin,
    CustomNameSeparatorFlagMixin,
    Flag,
):
    """Flag Enum of sample weight strategies.
    Can use multiple at a time, e.g. SampleWeightStrategy.ISOTYPE_USAGE | SampleWeightStrategy.CLONE_SIZE to weigh by both isotype usage and clone size.
    SampleWeightStrategy.NONE means no sample weighting. (SampleWeightStrategy.ISOTYPE_USAGE | SampleWeightStrategy.NONE is equivalent to SampleWeightStrategy.ISOTYPE_USAGE alone).
    """

    __default_separator__ = "_"  # for nice filenames

    # no sample weighting (uniform)
    NONE = 0

    # sample weighting by isotype usage
    ISOTYPE_USAGE = auto()

    # Sample weighting by clone size:
    # Incorporate clone size, to avoid long tail of clones crowding out the meaningful sequences.
    # Set weight based on the number of unique VDJ sequences associated with a clone.

    # Note: this may be a little biased by plasma cells, due to sequence diversity from occasional PCR or sequencing errors.
    # In future, consider counting unique VDJ sequences allowing +/- 1 or 2 Hamming distance, constraining them a little more to account for occasional errors.

    # Alternative: weigh by the total number of reads associated with a clone.
    # That would definitely be biased by plasma cells (high RNA expression) and by PCR.
    # In the future, consider gDNA sequencing for a true readout of which clones are expanded.
    CLONE_SIZE = auto()


class EmbedderSequenceContent(ValidatableEnumMixin, Enum):
    """
    Enum of embedder sequence content options, e.g. embed full VDJ, CDR1+2+3, or CDR3 only.

    Even when passing in many sequence regions as context to the language model, we may choose to keep only the embeddings of a particular region.
    For example, we might embed the entire VDJ sequence to capture full context, but train a disease classifier using only the embeddings of the variable (randomly generated) sequence regions.
    (How this works: The language model creates an embedding vector for each amino acid character in the full amino acid sequence, in the context of the rest of the sequence. To create a single vector representing the full sequence, do not average all these embedding vectors. Instead, only average the embedding vectors for the amino acid characters that are part of variable/randomly generated sequence regions.)
    """

    # Embed or fine tune with full sequence, ranging from FR1 to FR4 ("post").
    FR1ThruFR4 = auto()

    # Embed or fine tune with CDR1+2+3 concatenated sequence.
    CDR123 = auto()

    # Embed or fine tune with CDR3 sequence.
    CDR3 = auto()

    # Provide full or CDR1+2+3 concatenated sequence for embedding and fine tuning,
    # but use only CDR3 region (or a subset thereof, i.e. just the center) in pooled embedding.
    # (These could benefit from prioritizing the CDR3 region (or a subset thereof) in the loss function for fine tuning. See cdr3_region_weight_in_loss_computation())
    CDR123_but_only_return_CDR3 = auto()
    CDR123_but_only_return_CDR3_middle = auto()
    FR1ThruFR4_but_only_return_CDR3 = auto()
    FR1ThruFR4_but_only_return_CDR3_middle = auto()

    @property
    def cdr3_region_weight_in_loss_computation(self) -> int:
        """
        In the semi-supervised fine tuning process (masked language model objective), how much should we prioritize CDR3 prediction errors over others in the loss function?
        The goal is to prioritize learning the CDR3 region, not just the framework or CDR1/2 regions that are more determinstic from the V gene.
        """
        # This doesn't seem to work, so we are disabling this for now until we can experiment further.
        return 1

    @property
    def included_sequence_regions(self) -> List[str]:
        """Which sequence regions should be embedded? Returns a list of columns to be concatenated."""
        if self in [
            EmbedderSequenceContent.FR1ThruFR4,
            EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3,
            EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3_middle,
        ]:
            return [
                "fr1_seq_aa_q_trim",
                "cdr1_seq_aa_q_trim",
                "fr2_seq_aa_q_trim",
                "cdr2_seq_aa_q_trim",
                "fr3_seq_aa_q_trim",
                "cdr3_seq_aa_q_trim",
                "post_seq_aa_q_trim",
            ]
        if self in [
            EmbedderSequenceContent.CDR123,
            EmbedderSequenceContent.CDR123_but_only_return_CDR3,
            EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle,
        ]:
            return ["cdr1_seq_aa_q_trim", "cdr2_seq_aa_q_trim", "cdr3_seq_aa_q_trim"]
        if self in [
            EmbedderSequenceContent.CDR3,
        ]:
            return ["cdr3_seq_aa_q_trim"]
        raise ValueError(f"Unknown EmbedderSequenceContent: {self}")

    @property
    def include_v_gene_as_dummy_variable(self) -> bool:
        """
        Modeling feature flags:
        Include V gene identity as dummy variable in model3 depending on which sequence regions we're embedding.
        This is no longer necessary - we saw this always improves model performance. But in the future we could bring this back by checking if self in [EmbedderSequenceContent.CDR3, ...]
        """
        return True


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
    Deviance = auto()


@dataclass(eq=False)
class ObsOnlyAnndata:
    obs: pd.DataFrame
    uns: dict = dataclasses.field(default_factory=dict)

    def __getitem__(self, slice) -> Self:
        # slice pandas dataframe
        return dataclasses.replace(self, obs=self.obs.loc[slice])

    def copy(self) -> Self:
        # trigger pandas clone
        return dataclasses.replace(self, obs=self.obs.copy())

    def __len__(self) -> int:
        return self.obs.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.obs.shape[0], 0)

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def is_view(self) -> bool:
        return self.obs._is_view


GeneralAnndataType = Union[anndata.AnnData, ObsOnlyAnndata]
