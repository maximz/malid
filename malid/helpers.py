import logging
from functools import cache
from typing import Dict, List

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from genetools.palette import HueValueStyle

from malid import config
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    healthy_label,
)

logger = logging.getLogger(__name__)

# Update this with new diseases:
diseases = [
    "Covid19",
    "HIV",
    healthy_label,
    "Lupus",
]

disease_color_palette = {
    disease: color for disease, color in zip(diseases, sc.plotting.palettes.default_20)
}
disease_color_palette["Unknown"] = tuple(val / 255 for val in (163, 163, 163))  # gray
# visualize with: sc.plotting.palettes._plot_color_cycle(helpers.disease_color_palette)

# friendly name for CDR3 segment
cdr3_segment_name: Dict[GeneLocus, str] = {
    GeneLocus.BCR: "CDR-H3",
    GeneLocus.TCR: "CDR3β",
}

# Define colors for study_name field values in peak timepoint set.
# (Verified for completeness by automated tests.)
# The colors for batches of each disease are chosen to be similar. See https://colorhunt.co/palettes/blue
study_name_color_palette = {
    "Covid19-Stanford": "#0078AA",
    "Covid19-Seattle": "#1F4690",
    "Covid19-buffycoat": "#FC0394",
    "HIV": "#ff7f0e",
    "Healthy-StanfordBloodCenter": HueValueStyle(
        color="#279e68", zorder=-15, alpha=0.5
    ),
    "Lupus": "#d62728",
    "Lupus Pediatric": "#aa40fc",
    "New Lupus RNA": "#E4CAC6",
    "New Lupus Paxgene": "#101B3D",
    "healthy_children": HueValueStyle(color="#89CFFD", zorder=-15, alpha=0.5),
}

# module can't have a @property unfortunately
def diseases_in_peak_timepoint_dataset():
    all_specimens = get_all_specimen_info()
    all_specimens = all_specimens[all_specimens["in_training_set"]]
    return all_specimens["disease"].unique().tolist()


## Isotype group whitelist:
# Define which isotype groups are kept. This is also the order in which they are inserted as dummy variables into the model.
# Filter out IgE until we add allergy or parasitic infection cases. IgE is very rare and subject to PCR contamination.
# Filter out gDNA because it a mixture of all isotypes.
isotype_groups_kept: Dict[GeneLocus, List[str]] = {
    GeneLocus.BCR: ["IGHG", "IGHA", "IGHD-M"],
    GeneLocus.TCR: ["TCRB"],
}
isotype_friendly_names = {"IGHG": "IgG", "IGHA": "IgA", "IGHD-M": "IgD/M"}
isotype_friendly_name_order = ["IgD/M", "IgG", "IgA"]
isotype_palette = {
    isotype: color
    for isotype, color in zip(isotype_friendly_name_order, sns.color_palette("Dark2"))
}  # alternative palette to consider: "Set3_r"

# Some V gene names in our IgBlast reference are outdated
v_gene_friendly_names = {
    "IGHV4-b": "IGHV4-38-2",  # https://www.genecards.org/cgi-bin/carddisp.pl?gene=IGHV4-38-2
    "IGHV5-a": "IGHV5-10-1",  # https://www.genecards.org/cgi-bin/carddisp.pl?gene=IGHV5-10-1
}


@cache
def all_observed_v_genes() -> Dict[GeneLocus, np.ndarray]:
    """
    Get list of all V genes observed in our dataset.
    This defines the order of V genes for add_v_gene_dummy_variables_to_embedding_vectors.
    Generated by get_all_v_genes.ipynb.
    """
    return {
        gene_locus: pd.read_csv(
            config.paths.dataset_specific_metadata
            / f"all_v_genes.in_order.{gene_locus.name}.txt"
        )["v_gene"].values
        for gene_locus in GeneLocus
    }


@cache
def all_observed_j_genes() -> Dict[GeneLocus, np.ndarray]:
    """
    Get list of all J genes observed in our dataset.
    Used for subsetting new external cohorts to make sure only our known J genes are included.
    Generated by get_all_v_genes.ipynb.
    """
    return {
        gene_locus: pd.read_csv(
            config.paths.dataset_specific_metadata
            / f"all_j_genes.in_order.{gene_locus.name}.txt"
        )["j_gene"].values
        for gene_locus in GeneLocus
    }


# store up to one per fold (including the final "-1" global fold)


# TODO: this should be a helper method on a Repertoire subclass of AnnData in datamodels.py.
def should_switch_to_raw(repertoire: anndata.AnnData):
    # if target is not the happy path ("disease" / "disease.separate_past_exposures"), we should re-scale the data from raw, because we are starting from a subset of specimens
    # this is recorded if/when we perform subsetting.
    return repertoire.uns.get("adata_was_subset_should_switch_to_raw", False)


def get_obs_column_list_with_target_obs_column_included(
    column_list, target_obs_column: TargetObsColumnEnum
):
    """return modified column_list including target_obs_column (converted into the true obs column name) if not already present."""
    TargetObsColumnEnum.validate(target_obs_column)
    return np.unique(
        list(column_list) + [target_obs_column.value.obs_column_name]
    ).tolist()


# TODO: this should be a helper method on a Repertoire subclass of AnnData in datamodels.py.
def confirm_all_sequences_from_same_specimen(repertoire: anndata.AnnData):
    if repertoire.obs["specimen_label"].nunique() != 1:
        raise ValueError(
            "All sequences must come from one single repertoire, i.e. one specimen_label."
        )
    return True


# TODO: this should be a helper method on a Repertoire subclass of AnnData in datamodels.py.
def anndata_groupby_obs(adata: anndata.AnnData, *groupby_args, **groupby_kwargs):
    """
    Do a groupby on anndata obs, and return anndata chunks

    Example:
    Imagine an anndata with 2 rows with obs "patient = A" and 3 rows with obs "patient = B"
    Call e.g. anndata_groupby_obs(adata, 'patient', observed=True, sort=False)
    This returns ('A', adata[adata.obs['patient'] == 'A']), ('B', adata[adata.obs['patient'] == 'B']) as a generator.
    """
    for key, grp in adata.obs.groupby(*groupby_args, **groupby_kwargs):
        yield key, adata[grp.index, :]


def get_specimen_info(specimen_label):
    """get info for one specimen. see get_all_specimen_info() docs"""
    df = get_all_specimen_info()
    return df[df["specimen_label"] == specimen_label].squeeze()


def get_specimens_for_participant(participant_label):
    """get info for one participant. see get_all_specimen_info() docs"""
    df = get_all_specimen_info()
    return df[df["participant_label"] == participant_label]


def get_all_specimen_cv_fold_info():
    """get cross validation info for each specimen. based on get_all_specimen_info() but supplemented."""
    df = pd.read_csv(
        config.paths.dataset_specific_metadata
        / "cross_validation_divisions.specimens.tsv",
        sep="\t",
    )
    return df


def get_test_fold_id_for_each_participant():
    """returns single test fold ID for each participant (based on where their peak timepoint specimens were placed)"""
    df = get_all_specimen_cv_fold_info()
    df = df[df["fold_label"] == "test"]
    df = df[["participant_label", "fold_id"]].drop_duplicates()
    if df["participant_label"].duplicated().any():
        raise ValueError("Some participant(s) appear in more than one test fold")
    return df.set_index("participant_label")["fold_id"].rename("test_fold_id")


def get_test_fold_id_for_each_specimen():
    """returns single test fold ID for each specimen (based on where the originating participant's peak timepoint specimens were placed)"""
    df = get_all_specimen_cv_fold_info()
    df = df[df["fold_label"] == "test"]
    if df["specimen_label"].duplicated().any():
        raise ValueError("Some specimen(s) appear in more than one test fold")
    return df.set_index("specimen_label")["fold_id"].rename("test_fold_id")


def get_specimens_in_one_cv_fold(fold_id: int, fold_label: str):
    """get specimens in a particular cross validation fold. based on get_all_specimen_cv_fold_info()."""
    df = get_all_specimen_cv_fold_info()
    return df[(df["fold_id"] == fold_id) & (df["fold_label"] == fold_label)].copy()


@cache
def get_all_specimen_info(add_cv_fold_information=True):
    """
    get info for all specimens. one row per specimen.
    only runnable after umap_embedding.subgraphs_for_cv.ipynb step has completed.

    explanation of some columns:
    - survived_filters: is this a valid specimen with sufficient number of sequences and all isotypes present
    - is_peak: was this specimen selected for inclusion in the training set by virtue of being at a peak time point?
    - in_training_set: was this specimen actually included in the training set?

    note that in_training_set implies (is_peak and survived_filters), but is_peak does not imply in_training_set.
    the reason for this is that we skip certain specimens with not enough sequences or not all isotypes present
    we attempt to include all is_peak specimens, but eventually we have an anndata with only the in_training_set specimens.
    so in_training_set is basically "is_peak and is valid"
    """
    # Key metadata values stored in anndata obs
    df = pd.read_csv(
        config.paths.dataset_specific_metadata / "participant_specimen_disease_map.tsv",
        sep="\t",
    ).assign(cohort="Boydlab")

    # Load study_name (required), has_BCR/has_TCR (required), and metadata (where available) for all specimens
    # Also replace some of the anndata obs metadata values, which may be stale
    # (Easier to update _load_etl_metadata()'s contents than to regenerate all anndatas)
    df = pd.merge(
        df.drop(columns=["participant_age", "disease_subtype"]),
        # To update the below list, make sure to update _load_etl_metadata()
        # Also when updating this list, register anything we want to use in io.load_fold_embeddings too.
        _load_etl_metadata()[
            [
                "study_name",
                "available_gene_loci",
                # Overrides:
                "disease_subtype",
                "age",
                "participant_label_override",
                "specimen_time_point_override",
                # Metadata:
                "sex",
                "ethnicity_condensed",
                "age_group",
                "age_group_binary",
                "age_group_pediatric",
                "cmv",
                "disease_severity",
            ]
        ],
        left_on="specimen_label",
        right_index=True,
        how="left",
        validate="1:1",
    )
    # Override some variables, as in etl.ipynb
    df["participant_label"] = df["participant_label_override"].fillna(
        df["participant_label"]
    )
    df["specimen_time_point"] = df["specimen_time_point_override"].fillna(
        df["specimen_time_point"]
    )
    df.drop(
        columns=["participant_label_override", "specimen_time_point_override"],
        inplace=True,
    )

    # Confirm all specimens have study_name annotations
    if df["study_name"].isna().any():
        raise ValueError("Some specimens don't have study_name annotation.")

    # Introduce nullable numeric specimen_time_point_days column:
    # Extract days since symptom onset from timepoint field, where possible
    # Will be NaN for many specimens
    df["specimen_time_point_days"] = pd.Series(dtype=float)  # float is nullable
    # find "day" containing strings. fillna for anything that is not a string.
    mask_days = df["specimen_time_point"].str.contains("day").fillna(False)
    # extract digits from "day" containing strings
    extracted_days = (
        df.loc[mask_days, "specimen_time_point"]
        .str.extract("(\d+)", expand=False)
        .astype(int)
    )
    # confirm correct type
    if type(extracted_days) != pd.Series:
        raise ValueError(
            "Failed to extract days from specimen_time_point: regex extract produced DataFrame, not Series"
        )
    # set extracted days column
    df.loc[mask_days, "specimen_time_point_days"] = extracted_days
    # special case
    df.loc[df["specimen_time_point"] == "00:00:00", "specimen_time_point_days"] = 0

    # not all is_peak specimens actually make it into the embedding anndatas (see docstring for more details)
    # label the ones that do survive
    specimens_kept_in_embedding_anndatas = pd.read_csv(
        config.paths.dataset_specific_metadata
        / "specimens_kept_in_embedding_anndatas.tsv",
        sep="\t",
    )
    df = pd.merge(
        df,
        specimens_kept_in_embedding_anndatas[["specimen_label"]].assign(
            survived_filters=True
        ),
        on="specimen_label",
        how="left",
    )
    df["survived_filters"].fillna(False, inplace=True)

    # Just in case, reset index so that groupby->idxmin/idxmax works
    df = df.reset_index(drop=True).copy()

    # Choose some "pure" specimens for the peak-timepoints-only training set
    # For most diseases, select specific disease-subtypes only. But for certain diseases, keep all their subtypes.
    df["is_peak"] = (df["disease_subtype"].isin(config.subtypes_keep)) | (
        df["disease"].isin(config.diseases_to_keep_all_subtypes)
    )
    # Refine peak further for some study names
    # TODO: do we need a .copy() before the .groupby because we'll be editing as we go?
    for study_name, grp in df[(df["survived_filters"]) & (df["is_peak"])].groupby(
        "study_name", observed=True
    ):
        filtering_function_for_this_study_name = (
            config.study_names_with_special_peak_timepoint_filtering.get(
                study_name, None
            )
        )

        if filtering_function_for_this_study_name is None:
            # No filtering function for this study_name, so keep all is_peak specimens as is
            continue

        # Reset is_peak
        df.loc[grp.index, "is_peak"] = False

        # Update is_peak
        new_is_peak_subset = filtering_function_for_this_study_name(grp)
        df.loc[new_is_peak_subset, "is_peak"] = True

    # compute in_training_set = is_peak and survived_filters
    df["in_training_set"] = df["survived_filters"] & df["is_peak"]

    for col in ["is_peak", "survived_filters", "in_training_set"]:
        if df[col].isna().any():
            raise ValueError(f"Some specimens don't have a value for {col} column.")

    # label past exposures
    # delayed import to avoid circular import
    from malid import io

    df = io.label_past_exposures_in_obs(df)

    # Add test fold ID, where available
    if add_cv_fold_information:
        specimens_in_test_folds = get_test_fold_id_for_each_specimen()
        df = pd.merge(
            df,
            specimens_in_test_folds,
            left_on="specimen_label",
            right_index=True,
            how="left",
            validate="1:1",
        )

    return df


def _load_etl_metadata():
    df = pd.read_csv(
        config.paths.metadata_dir / "generated_combined_specimen_metadata.tsv",
        sep="\t",
    ).set_index("specimen_label")

    def _get_colname(gene_locus: GeneLocus):
        return f"has_{gene_locus.name}"

    # For each gene_locus in config.gene_loci_used, confirm that column exists and is never nan
    for gene_locus in config.gene_loci_used:
        colname = _get_colname(gene_locus)
        if colname not in df.columns:
            raise ValueError(f"Missing {colname} column in specimen metadata.")
        if df[colname].isna().any():
            raise ValueError(f"Some specimens don't have {colname} annotation.")

    # Mapping of column names to GeneLocus for all columns that match with "has_{gene_locus.name}",
    # not just the gene loci used in this config.gene_loci_used
    cols_indicating_gene_locus_presence = {
        _get_colname(gene_locus): gene_locus
        for gene_locus in GeneLocus
        if _get_colname(gene_locus) in df.columns
    }

    # Create list of available gene loci for each specimen
    # Then consolidate into single composite/multi-packed GeneLocus item
    # TODO: Write an automated test for this
    df["available_gene_loci"] = df.apply(
        lambda row: GeneLocus.combine_flags_list_into_single_multiflag_value(
            [
                gene_locus
                for colname, gene_locus in cols_indicating_gene_locus_presence.items()
                if row[colname]
            ]
        ),
        axis=1,
    )

    # age_group deciles already created, but let's make age_group_binary too
    df.loc[df["age"] < 50, "age_group_binary"] = "under 50"
    df.loc[df["age"] >= 50, "age_group_binary"] = "50+"
    df.loc[
        df["age_group"].isna(), "age_group_binary"
    ] = np.nan  # age_group is specially nulled out for rare extreme ages
    assert all(df["age_group"].isna() == df["age_group_binary"].isna())  # sanity check

    # same for age_group_pediatric, which is a +/- 18 years old binary variable
    df.loc[df["age"] < 18, "age_group_pediatric"] = "under 18"
    df.loc[df["age"] >= 18, "age_group_pediatric"] = "18+"
    df.loc[
        df["age_group"].isna(), "age_group_pediatric"
    ] = np.nan  # age_group is specially nulled out for rare extreme ages
    assert all(
        df["age_group"].isna() == df["age_group_pediatric"].isna()
    )  # sanity check

    # Create a "defined CMV status" subset
    # Note: "specimens with defined CMV status" is a subset of "healthy controls with known age/sex/ethnicity" dataset
    # All other healthy control subsets and all other cohorts will have cmv status NaN
    df["cmv"] = df["disease_subtype"].map(
        {
            "Healthy/Background - CMV Negative": "CMV-",
            "Healthy/Background - CMV Positive": "CMV+",
        }
    )

    # Define disease severity
    # For now this is only defined for Covid-19
    # All other Covid subtypes and all other cohorts will have disease_severity nan
    df["disease_severity"] = df["disease_subtype"].map(
        {
            "Covid19 - Sero-negative (ICU)": "ICU",
            "Covid19 - Sero-positive (ICU)": "ICU",
            "Covid19 - Sero-negative (Admit)": "Admit",
            "Covid19 - Sero-positive (Admit)": "Admit",
            "Covid19 - Admit": "Admit",
            "Covid19 - ICU": "ICU",
        }
    )
    # confirm we have Covid severity for the two desired study_names
    assert (
        not df[df["study_name"].isin(["Covid19-buffycoat", "Covid19-Stanford"])][
            "disease_severity"
        ]
        .isna()
        .any()
    ), "Disease severity should not be NaN for Covid19-buffycoat and Covid19-Stanford specimens"
    assert (
        df[~df["study_name"].isin(["Covid19-buffycoat", "Covid19-Stanford"])][
            "disease_severity"
        ]
        .isna()
        .all()
    ), "Disease severity should be NaN except for Covid19-buffycoat and Covid19-Stanford specimens"

    return df


# TODO: these should be helper methods on a Repertoire subclass of AnnData in datamodels.py.
def extract_specimen_metadata_from_anndata(
    adata: anndata.AnnData,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
):
    return extract_specimen_metadata_from_obs_df(
        df=adata.obs, gene_locus=gene_locus, target_obs_column=target_obs_column
    )


def extract_specimen_metadata_from_obs_df(
    df: pd.DataFrame, gene_locus: GeneLocus, target_obs_column: TargetObsColumnEnum
):
    required_cols = [
        "participant_label",
        "specimen_label",
        "disease",
        "disease_subtype",
        "disease.separate_past_exposures",
        "disease.rollup",
        "participant_label",
        "past_exposure",
    ]

    # these columns are used in peak timepoint train/test, but not in off-peak
    optional_cols = [
        # Add demographics and isotype proportion columns for regress_out in metamodel:
        "age",
        "sex",
        "ethnicity_condensed",
        "study_name",
        "disease_severity",
    ] + [f"isotype_proportion:{isotype}" for isotype in isotype_groups_kept[gene_locus]]

    # filter down to optional cols that are present
    optional_cols = [col for col in optional_cols if col in df.columns]

    return (
        df[
            get_obs_column_list_with_target_obs_column_included(
                required_cols + optional_cols,
                target_obs_column,
            )
        ]
        .drop_duplicates()
        .set_index("specimen_label")
    )
