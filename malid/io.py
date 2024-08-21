import gc
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union, overload, Dict

import anndata, anndata.utils
import genetools
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.utils
from pathlib import Path
import scratchcache
from slugify import slugify

from malid import config, helpers
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    healthy_label,
    GeneralAnndataType,
    ObsOnlyAnndata,
)
from environs import Env
import logging

logger = logging.getLogger(__name__)
env = Env()


def load_raw_parquet_sequences_for_fold(
    fold_id: int, fold_label: str, gene_locus: GeneLocus
):
    """load raw sequences - use case: to embed them"""
    specimen_labels = helpers.get_specimens_in_one_cv_fold(
        fold_id=fold_id, fold_label=fold_label
    )
    return load_raw_parquet_sequences_for_specimens(
        specimen_labels=specimen_labels, gene_locus=gene_locus
    )


def load_raw_parquet_sequences_for_specimens(
    specimen_labels: List[str],
    gene_locus: GeneLocus,
    columns: Optional[list] = None,
    fname: Optional[Path] = None,
):
    """load raw sequences for specimen label(s) - use case: to embed them"""
    GeneLocus.validate_single_value(gene_locus)
    if type(specimen_labels) is str:
        # cast to list
        specimen_labels = [specimen_labels]

    if fname is None:
        # default
        fname = config.paths.sequences_sampled
    # defensive cast
    fname = Path(fname)

    # filters = [("specimen_label", "==", specimen_name)]
    filters = [("specimen_label", "in", list(specimen_labels))]

    # Don't use fastparquet, because it changes specimen labels like M54-049 to 2049-01-01 00:00:54 -- i.e. it coerces partition names to numbers or dates
    df = pd.read_parquet(fname, filters=filters, columns=columns, engine="pyarrow")
    if not set(df["specimen_label"].unique()) == set(specimen_labels):
        raise ValueError(
            f"Specimen labels in dataframe do not match specimen labels passed in: {set(df['specimen_label'].unique())} != {set(specimen_labels)}"
        )

    # Filter to desired locus

    return df.loc[
        df["isotype_supergroup"].isin(helpers.isotype_groups_kept[gene_locus])
    ]


def label_past_exposures(adata: GeneralAnndataType) -> GeneralAnndataType:
    # label past exposures
    # TODO: move this to ETL
    # TODO: make this wrapper a decorator.

    if adata.is_view:
        # We cannot pass obs of an anndata view to the label_past_exposures_in_obs function.
        # Otherwise we get this anndata view->copy error:
        # If we call label_past_exposures_in_obs(adata.obs) and adata is a view,
        # then `obs_df["past_exposure"] = False` may lead to `anndata - WARNING - Trying to set attribute `.obs` of view, copying.`.
        # The result is adata is no longer a copy, adata.obs has a past_exposure column,
        # but obs_df is outdated (points to the destroyed view) and does not have a past_exposure column!
        # Then the rest of the code will fail.
        adata = adata.copy()
    adata.obs = label_past_exposures_in_obs(adata.obs)
    return adata


def label_past_exposures_in_obs(obs_df):
    """Same as above, but accepts obs df directly instead of adata."""
    obs_df["past_exposure"] = False
    # obs_df.loc[obs_df["disease_subtype"].str.contains("- Survivor", regex=False), "past_exposure"] = True

    obs_df["disease.separate_past_exposures"] = obs_df["disease"].astype(str) + obs_df[
        "past_exposure"
    ].replace({False: "", True: " - Survivor"})

    # Also label past exposures for rollup
    # creates "disease.rollup" column for final-patient-prediction y labels, where past-exposure individuals are Healthy in the rollup
    obs_df["disease.rollup"] = obs_df["disease.separate_past_exposures"]
    # # Modify anything with df['past_exposure'] flag
    # obs_df.loc[obs_df['past_exposure'], "disease.rollup"]= healthy_label

    return obs_df


# Cache up to 4 fold embeddings in memory
# The time we need 4 in memory is for training metamodels: we load validation and test sets for BCR and TCR.
# But hitting the maximum of 4 in memory is not desirable. It is the user's responsibility to clear out the cache manually (io.clear_cached_fold_embeddings()) when moving to a new fold ID.
@lru_cache(
    maxsize=0 if env.bool("MALID_DISABLE_IN_MEMORY_CACHE", False) else 4
)  # Allow temporarily disabling cache
def _load_anndata(fname: Union[str, Path], gene_locus: GeneLocus) -> anndata.AnnData:
    """cache the sc.read operation in memory"""
    # Remap the filename to a local machine scratch dir filename, then load.
    adata = sc.read(
        scratchcache.local_machine_cache(
            fname=fname, local_machine_cache_dir=config.paths.local_machine_cache_dir
        )
    )

    # Apply fixes prior to saving in cache.
    return _post_anndata_load_fixes(adata, gene_locus)


def _add_anndata_columns(
    adata: GeneralAnndataType, compute_sample_weights: bool = True
) -> GeneralAnndataType:
    """Use _post_anndata_load_fixes() unless there's a particular reason to use this instead."""
    # record original obs columns and uns keys
    adata.uns["original_uns_keys"] = list(adata.uns.keys())
    adata.uns["original_obs_columns"] = adata.obs.columns.tolist()

    # produces 'disease.separate_past_exposures' obs column. TODO: move to ETL
    adata = label_past_exposures(adata)
    adata = fix_gene_names(adata)

    # Create "v_family" column based on "v_gene"
    adata.obs["v_family"] = convert_vgene_to_vfamily(adata.obs["v_gene"])

    if compute_sample_weights:
        # Calculate sample weights to balance out isotype proportions for each specimen
        # (These will only be used if SampleWeightStrategy.ISOTYPE_USAGE in sample_weight_strategy, but we calculate before caching to avoid extra anndata copies)
        adata.obs = genetools.helpers.merge_into_left(
            adata.obs.drop(
                columns=["sample_weight_isotype_rebalance"], errors="ignore"
            ),
            compute_isotype_sample_weights_column(adata).rename(
                "sample_weight_isotype_rebalance"
            ),
        )

        # Calculate sample weights according to clone sizes within each isotype of each specimen.
        # (These will only be used if SampleWeightStrategy.CLONE_SIZE in sample_weight_strategy, but we calculate before caching to avoid extra anndata copies)
        adata.obs = genetools.helpers.merge_into_left(
            adata.obs.drop(columns=["sample_weight_clone_size"], errors="ignore"),
            compute_clone_size_sample_weights_column(adata).rename(
                "sample_weight_clone_size"
            ),
        )

    return adata


def _post_anndata_load_fixes(
    adata: GeneralAnndataType, gene_locus: GeneLocus
) -> GeneralAnndataType:
    """Fixes to apply after loading anndata from disk but before persisting to cache."""
    adata = _add_anndata_columns(adata)

    # Load study_name for all specimens (not just those with demographic data)
    # And load metadata (age, sex, ethnicity, and symptoms or other metadata such as CMV status) for those specimens that have it
    specimen_metadata = helpers.get_all_specimen_info().set_index("specimen_label")
    adata.obs = pd.merge(
        adata.obs.drop(
            columns=[
                # Drop saved-in-raw-data-on-disk versions ahead of the possible metadata overrides below:
                "participant_label",
                "specimen_time_point",
                "participant_age",
                "disease_subtype",
            ],
            # skip dropping any of these columns if they aren't present (possible in CI where we use a reduced dataset?)
            # TODO: this skip happens not just in CI; are specimen_time_point and participant_age real column names?
            errors="ignore",
        ),
        # To update the below list, make sure to also update helpers.get_all_specimen_info()
        specimen_metadata[
            [
                # Available for all
                "study_name",  # Not originally available in the anndata or Parquet; merged in from metadata files at runtime.
                "participant_label",  # Originally available in the anndata or Parquet, but dropped here so it may be overriden.
                "specimen_time_point",  # Similarly, may be overriden here
                "disease_subtype",  # Similarly, may be overriden here
                # (For more info on the overrides, see helpers._load_etl_metadata and helpers.get_all_specimen_info)
                #
                # Available for all (auto-generated):
                "study_name_condensed",
                #
                # Available for some:
                "age",
                "sex",
                "ethnicity_condensed",
                "age_group",
                "age_group_binary",
                "age_group_pediatric",
                "disease_severity",
                "specimen_description",
            ]
            # also include symptoms_* columns
            + list(
                specimen_metadata.columns[
                    specimen_metadata.columns.str.startswith("symptoms_")
                ]
            )
        ],
        left_on="specimen_label",
        right_index=True,
        how="left",
        validate="m:1",
    )
    # confirm all specimens have study_name annotations
    if adata.obs["study_name"].isna().any():
        raise ValueError("Some specimens don't have study_name annotation.")

    # Set a specific order for "sex" so that DemographicsFeaturizers for all folds will reduce to the same dummy variable, regardless of whether sex=M or sex=F comes first in that fold's training set
    # (See DemographicsFeaturizer._get_one_hot_encodings for more details)
    adata.obs["sex"] = adata.obs["sex"].astype(
        pd.CategoricalDtype(categories=["M", "F"], ordered=True)
    )

    # Healthy control resequencing experiment: For these samples, mark whether they are from an old or a new batch
    # This will turn the amplification label prefix into a column suitable to be a classification target.
    # TODO: Add test for this.
    adata.obs.loc[
        adata.obs["symptoms_healthy_in_resequencing_experiment"].notna(),
        "symptoms_healthy_old_vs_new_batch_identifier",
    ] = (
        adata.obs.loc[
            adata.obs["symptoms_healthy_in_resequencing_experiment"].notna(),
            "amplification_label",
        ]
        .str.startswith(("M477", "M479", "M482", "M484"))
        .map({True: "New", False: "Old"})
    )

    # Also add isotype proportions per specimen (for BCR only)
    # Unlike above demographic columns, this info is expected to be available for all specimens.
    if gene_locus == GeneLocus.BCR:
        # make counts dataframe: specimens x isotypes
        isotype_counts = (
            adata.obs.groupby("specimen_label", observed=True)["isotype_supergroup"]
            .value_counts()
            .unstack()
        )
        # filter to only the isotypes we care about for this gene locus
        isotype_counts = isotype_counts[helpers.isotype_groups_kept[gene_locus]]
        # convert to proportions (normalize to sum to 1)
        isotype_counts = genetools.stats.normalize_rows(isotype_counts)
        isotype_counts = isotype_counts.rename(
            columns=lambda colname: f"isotype_proportion:{colname}"
        )
        adata.obs = pd.merge(
            adata.obs,
            isotype_counts,
            left_on="specimen_label",
            right_index=True,
            how="left",
            validate="m:1",
        )
        # confirm all specimens have isotype proportion annotations
        if adata.obs[isotype_counts.columns].isna().any().any():
            raise ValueError("Some specimens don't have isotype count annotations.")

    return adata


# Replace indistinguishable TRBV gene names with the version that we use in our data.
# https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-01008-4
v_gene_renames = {"TRBV12-4": "TRBV12-3", "TRBV6-3": "TRBV6-2"}
v_allele_renames = {
    # Our old IgBLAST can generate TRBV6-2*02 calls, but no CDR1+2 information is available for this allele from get_tcr_v_gene_annotations, because it has been renamed:
    # https://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes&repertoire=genetable&species=human&group=TRBV - see (40)
    "TRBV6-2*02": "TRBV6-2*01",
}


def fix_gene_names(adata: GeneralAnndataType) -> GeneralAnndataType:
    # Standardize TRBV gene names that may be indistinguishable
    # see https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-01008-4
    # TODO: move to ETL
    adata.obs["v_gene"] = adata.obs["v_gene"].replace(v_gene_renames)
    return adata


def convert_vgene_to_vfamily(v_genes: pd.Series) -> pd.Series:
    """
    Create "v_family" column based on "v_gene" column.

    Example formats:
    TRBV2 --> TRBV2
    IGHV1-18 --> IGHV1
    IGHV3-30-3 --> IGHV3
    IGHV1/OR15-1 --> IGHV1
    TRBV29/OR9-2 --> TRBV29
    VH1-67P --> VH1
    """
    results = (
        pd.Series(v_genes)
        .str.split("-")
        .str[0]
        .str.split("/")
        .str[0]
        .rename("v_family")
    )

    # Throw an error if any resulting v_family entries are empty string or NaN
    # (This will also be thrown if any original entries are empty string or NaN)
    if results.mask(results.str.strip() == "").isna().any():
        raise ValueError("Some v_genes could not be converted to v_family")

    return results


def load_fold_embeddings_off_peak(
    fold_id: int,
    gene_locus: GeneLocus,
    sample_weight_strategy: Optional[SampleWeightStrategy] = None,
):
    """
    Load off-peak embeddings for a given fold.
    """
    if sample_weight_strategy is None:
        # Supply default for this parameter
        sample_weight_strategy = config.sample_weight_strategy

    adata = sc.read(
        config.paths.scaled_anndatas_dir
        / gene_locus.name
        / f"off_peak_timepoints.fold.{fold_id}.h5ad"
    )
    adata.obs = adata.obs.assign(fold_id=fold_id, fold_label="expanded_test_set")
    adata = _add_anndata_columns(adata)

    return adata


AVAILABLE_KNOWN_BINDER_EMBEDDINGS: Dict[GeneLocus, List[str]] = {
    # Map from gene locus to list of diseases with available known binder data
    GeneLocus.BCR: ["Covid19", "Influenza"],
    GeneLocus.TCR: ["Covid19"],
}


def load_known_binder_embeddings(
    fold_id: int,
    gene_locus: GeneLocus,
    disease: str,
    known_binder: bool = True,
    sample_weight_strategy: Optional[SampleWeightStrategy] = None,
):
    """
    Load dataset of known binders or non-binders for a particular disease and gene locus.
    If known_binder is True, load known binders. If False, load known non-binders.
    """
    if sample_weight_strategy is None:
        # Supply default for this parameter
        sample_weight_strategy = config.sample_weight_strategy

    # Load known binder sequence embeddings
    adata = joblib.load(
        config.paths.scaled_anndatas_dir
        / gene_locus.name
        / f"known_{'binders' if known_binder else 'nonbinders'}.{slugify(disease)}.embedded.in.all.folds.joblib",
    )[fold_id]

    adata.obs = adata.obs.assign(fold_id=fold_id, fold_label="expanded_test_set")
    adata = _add_anndata_columns(
        adata,
        # We'll set our own below
        compute_sample_weights=False,
    )

    # set all sequence weights to 1 here
    adata.obs["sample_weight_isotype_rebalance"] = 1.0
    adata.obs["sample_weight_clone_size"] = 1.0

    return adata


# Typing for load_fold_embeddings:
# returns GeneralAnndataType if load_obs_only is True. otherwise returns normal anndata.AnnData
@overload
def load_fold_embeddings(
    fold_id,
    fold_label,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    anndatas_base_dir=None,
    sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    load_obs_only=False,
) -> anndata.AnnData:
    pass


@overload
def load_fold_embeddings(
    fold_id,
    fold_label,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    anndatas_base_dir=None,
    sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    load_obs_only=True,
) -> GeneralAnndataType:
    pass


def load_fold_embeddings(
    fold_id,
    fold_label,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    anndatas_base_dir=None,
    sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    load_obs_only=False,
) -> GeneralAnndataType:
    """Load entire fold's embedding anndata.
    Automatically caches several loaded fold embeddings between calls. For example, we might tune multiple models in a row on validation sets; why load the validation sets over and over again for each tuning operation?
    To clear cache, call io.clear_cached_fold_embeddings()

    Note: We duplicate after reloading from cache so that any changes do not affect the cache, and so that we never point to the cache version directly.

    Also supports loading an extra obs column `target_obs_column` to be used as an alternative classification target.
    Subsets to specimens with that obs column defined (so downstream code should revert to raw and rescale X).

    load_obs_only: fast path; doesn't load .X.
    """

    if sample_weight_strategy is None:
        # Supply default for this parameter
        sample_weight_strategy = config.sample_weight_strategy

    TargetObsColumnEnum.validate(target_obs_column)
    SampleWeightStrategy.validate(sample_weight_strategy)
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)

    if anndatas_base_dir is None:
        anndatas_base_dir = config.paths.scaled_anndatas_dir / gene_locus.name

    anndatas_base_dir = Path(anndatas_base_dir)

    mark_that_we_subsetted = False

    # load full anndata
    if load_obs_only:
        # Load .obs without any .X (i.e. no embedder used)
        # Fast path for when you don't need a saved embedding, just the raw sequence dataframe (in .obs)
        # TODO: Cache this too.
        obs = load_raw_parquet_sequences_for_fold(
            fold_id=fold_id, fold_label=fold_label, gene_locus=gene_locus
        )

        # replicate adata.obs_names_make_unique()
        obs.index = anndata.utils.make_index_unique(obs.index.astype(str), join="-")
        adata = ObsOnlyAnndata(obs=obs)

        # Apply post-loading fixes, just like in _load_anndata() in the normal path
        adata = _post_anndata_load_fixes(adata, gene_locus)
    else:
        # Normal (slow) path that loads .X
        def _make_file_path(fold_id, fold_label):
            return anndatas_base_dir / f"fold.{fold_id}.{fold_label}.h5ad"

        if fold_label in ["train_smaller1", "train_smaller2"]:
            # Special case: train_smaller1 and train_smaller2 are further subdivisions of train_smaller.
            # For efficiency, we don't embed them and save them to disk separately.
            # Instead:
            # 1) load train_smaller
            adata = _load_anndata(_make_file_path(fold_id, "train_smaller"), gene_locus)
            # 2) subset
            adata = adata[
                adata.obs["specimen_label"].isin(
                    helpers.get_specimens_in_one_cv_fold(
                        fold_id=fold_id, fold_label=fold_label
                    ).values
                )
            ]
            mark_that_we_subsetted = True
        else:
            # Default path
            adata = _load_anndata(_make_file_path(fold_id, fold_label), gene_locus)

    if target_obs_column.value.obs_column_name not in adata.obs.columns:
        raise ValueError(
            f"Obs column {target_obs_column.value.obs_column_name} (from {target_obs_column}) not in adata.obs.columns. (Update metadata loading?)"
        )
    if adata.obs[target_obs_column.value.obs_column_name].isna().any():
        # Subset to specimens with defined target_obs_column
        adata_shape_orig = adata.shape
        adata_orig_num_specimens = adata.obs["specimen_label"].nunique()
        adata = adata[~adata.obs[target_obs_column.value.obs_column_name].isna()]
        mark_that_we_subsetted = True
        adata_shape_new = adata.shape
        adata_new_num_specimens = adata.obs["specimen_label"].nunique()
        logger.info(
            f"Target {target_obs_column}: filtered to specimens with defined {target_obs_column.value.obs_column_name} column - removed {adata_orig_num_specimens - adata_new_num_specimens} specimens ({adata_shape_orig[0] - adata_shape_new[0]} rows)"
        )

    # Subset to specimens from certain diseases if target_obs_column calls for it.
    if target_obs_column.value.limited_to_disease is not None:
        adata_orig_num_specimens = adata.obs["specimen_label"].nunique()
        adata = adata[
            adata.obs["disease"].isin(target_obs_column.value.limited_to_disease)
        ]
        mark_that_we_subsetted = True
        adata_new_num_specimens = adata.obs["specimen_label"].nunique()
        logger.info(
            f"Target {target_obs_column}: filtering to specimens from diseases {target_obs_column.value.limited_to_disease} only; removed {adata_orig_num_specimens - adata_new_num_specimens} specimens"
        )

    # Subset to specimens from certain disease_subtypes if target_obs_column calls for it.
    if target_obs_column.value.limited_to_disease_subtype is not None:
        adata_orig_num_specimens = adata.obs["specimen_label"].nunique()
        adata = adata[
            adata.obs["disease_subtype"].isin(
                target_obs_column.value.limited_to_disease_subtype
            )
        ]
        mark_that_we_subsetted = True
        adata_new_num_specimens = adata.obs["specimen_label"].nunique()
        logger.info(
            f"Target {target_obs_column}: filtering to specimens from disease_subtypes {target_obs_column.value.limited_to_disease_subtype} only; removed {adata_orig_num_specimens - adata_new_num_specimens} specimens"
        )

    # Subset to specimens from certain specimen_descriptions if target_obs_column calls for it.
    if target_obs_column.value.limited_to_specimen_description is not None:
        adata_orig_num_specimens = adata.obs["specimen_label"].nunique()
        adata = adata[
            adata.obs["specimen_description"].isin(
                target_obs_column.value.limited_to_specimen_description
            )
        ]
        mark_that_we_subsetted = True
        adata_new_num_specimens = adata.obs["specimen_label"].nunique()
        logger.info(
            f"Target {target_obs_column}: filtering to specimens from specimen_descriptions {target_obs_column.value.limited_to_specimen_description} only; removed {adata_orig_num_specimens - adata_new_num_specimens} specimens"
        )

    # Subset to specimens from certain study_names if target_obs_column calls for it.
    if target_obs_column.value.limited_to_study_name is not None:
        adata_orig_num_specimens = adata.obs["specimen_label"].nunique()
        adata = adata[
            adata.obs["study_name"].isin(target_obs_column.value.limited_to_study_name)
        ]
        mark_that_we_subsetted = True
        adata_new_num_specimens = adata.obs["specimen_label"].nunique()
        logger.info(
            f"Target {target_obs_column}: filtering to specimens from study_names {target_obs_column.value.limited_to_study_name} only; removed {adata_orig_num_specimens - adata_new_num_specimens} specimens"
        )

    # Subset adata with a custom function, if target_obs_column calls for it.
    # TODO: Optimize by accepting a pandas query expression?
    if target_obs_column.value.filter_adata_obs_func is not None:
        adata_orig_num_specimens = adata.obs["specimen_label"].nunique()
        adata = adata[
            # Run the function on each row of adata.obs
            adata.obs.apply(
                target_obs_column.value.filter_adata_obs_func, axis=1
            ).to_numpy(dtype=bool)
        ]
        mark_that_we_subsetted = True
        adata_new_num_specimens = adata.obs["specimen_label"].nunique()
        logger.info(
            f"Target {target_obs_column}: filtering with custom filter_adata_obs_func; removed {adata_orig_num_specimens - adata_new_num_specimens} specimens"
        )

    # Filter out specimens that don't have demographic data, if requested by this target_obs_column
    if (
        target_obs_column.value.require_metadata_columns_present is not None
        and len(target_obs_column.value.require_metadata_columns_present) > 0
    ):
        # Filter out an adata row if the corresponding row of obs[required_metadata_cols] has any NaNs
        adata = adata[
            ~adata.obs[target_obs_column.value.require_metadata_columns_present]
            .isna()
            .any(axis=1)
        ]

    if adata.obs_names.duplicated().any():
        raise ValueError("Loaded anndata has duplicated obs_names.")

    if adata.shape[0] == 0:
        raise ValueError("Filters removed all specimens from adata.")

    # We may have applied layers of subsetting.
    # Or we may have applied no subsetting at all, and just have the result straight from cache so far.
    # Either way, we have deliberately avoided making any inplace edits and avoided any copies. We've only done subsetting of the anndata, getting an anndata view.
    # Now, finally, copy the anndata (either view --> new anndata, or anndata --> new anndata) to create the anndata object we'll pass through the rest of our code.
    # Reasons why copying is necessary:
    # 1) We want to make sure we are not affecting any pointers to previous anndata.obs.
    #   This avoids issues where the first load sets the cache, the second load modifies the cached object, and then any pointers to the first load's .obs also contain unexpected updates due to the second call.
    # 2) We also want to make sure that edits to the returned anndata do not affect the cached version and do not manifest in separate future loads.
    adata = adata.copy()

    # We execute this here, instead of before saving to cache (i.e. in _post_anndata_load_fixes()),
    # because "train_smaller1" and "train_smaller2" are special cased above to actually load "train_smaller" from disk and from cache.
    adata.obs["fold_id"] = fold_id
    adata.obs["fold_label"] = fold_label

    if mark_that_we_subsetted:
        # Set this uns item *after* copying, so we don't trigger a copy until now.
        adata.uns["adata_was_subset_should_switch_to_raw"] = True

    # Remove unused disease column categories (i.e. those that were filtered out by the target_obs_column)
    adata.obs["disease"] = adata.obs["disease"].cat.remove_unused_categories()

    gc.collect()

    return adata


def clear_cached_fold_embeddings():
    """Clear LRU cache for load_fold_embeddings(), then run garbage collection"""
    _load_anndata.cache_clear()
    gc.collect()


def compute_isotype_sample_weights_column(adata: GeneralAnndataType) -> pd.Series:
    def _add_isotype_sample_weights_for_single_specimen(isotype_values: pd.Series):
        # Make balanced class weights, just like in sklearn classifiers. See https://datascience.stackexchange.com/a/18722/13342
        # class_weight = "balanced" gives higher weight to minority classes. "balanced" means class weights are total_samples_across_all_classes / (n_classes * np.bincount(y)) -- bincount is value_counts vector of samples in each class
        # so weight of class i = n_samples_total / (n_classes * n_samples_from_class_i)

        # compute_class_weight() gives the same as compute_sample_weight(), but one entry per class, rather than one entry per sample:
        # # sklearn returns array with i-th element being the weight for i-th class. reformat as dictionary mapping class indices (integers) to a weight (float) value
        # classes = np.unique(arr)
        # class_weights = dict(
        #     zip(
        #         classes,
        #         sklearn.utils.class_weight.compute_class_weight(
        #             class_weight="balanced", classes=classes, y=arr
        #         ),
        #     )
        # )

        # Returns one entry per sample, matching the sample's class's weight
        # Note: these weights are not necessarily normalized to sum to 1
        sample_weights = sklearn.utils.class_weight.compute_sample_weight(
            class_weight="balanced",
            y=isotype_values.values,
        )

        # To be safe, normalize weights to sum to 1
        sample_weights /= np.sum(sample_weights)

        return pd.Series(sample_weights, isotype_values.index)

    # split by specimen, calculate, then recombine
    results = []
    for specimen_label, specimen_repertoire_obs in adata.obs.groupby(
        "specimen_label", observed=True
    ):
        results.append(
            _add_isotype_sample_weights_for_single_specimen(
                specimen_repertoire_obs["isotype_supergroup"]
            )
        )
    results = pd.concat(results, axis=0)

    # sanity check
    if results.shape[0] != adata.obs.shape[0]:
        raise ValueError(
            f"Expected to get one weight per sample, but got {results.shape[0]} weights for {adata.obs.shape[0]} samples"
        )
    # return in correct order
    return results.loc[adata.obs_names]


def compute_clone_size_sample_weights_column(adata: anndata.AnnData) -> pd.Series:
    # For each clone (i.e. one row) in one isotype from one specimen,
    # compare how many unique VDJ sequences are part of this clone,
    # versus how many total unique VDJ sequences are part of all clones in this isotype in this specimen.

    # TODO: Does this properly handle clones that appear in multiple isotypes? Do their clone sizes get duplicated?

    return adata.obs.groupby(["specimen_label", "isotype_supergroup"], observed=True)[
        "num_clone_members"
    ].transform(lambda series: series / series.sum())
