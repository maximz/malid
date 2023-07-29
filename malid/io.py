import gc
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import anndata
import genetools
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.utils
from pathlib import Path
import scratchcache

from malid import config, helpers
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    healthy_label,
)
import logging

logger = logging.getLogger(__name__)


def load_raw_parquet_sequences_for_fold(
    fold_id: int, fold_label: str, gene_locus: GeneLocus
):
    """load raw sequences - use case: to embed them"""
    specimen_labels = helpers.get_specimens_in_one_cv_fold(
        fold_id=fold_id, fold_label=fold_label
    )["specimen_label"]
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


def label_past_exposures(adata):
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
@lru_cache(maxsize=4)
def _load_anndata_inner(fname) -> anndata.AnnData:
    # first remap the filename to a local machine scratch dir filename
    adata = sc.read(
        scratchcache.local_machine_cache(
            fname=fname, local_machine_cache_dir=config.paths.local_machine_cache_dir
        )
    )

    # record original obs columns and uns keys
    adata.uns["original_uns_keys"] = list(adata.uns.keys())
    adata.uns["original_obs_columns"] = adata.obs.columns.tolist()

    return adata


def _load_anndata(fname) -> anndata.AnnData:
    """
    cache the sc.read operation in memory
    but duplicate the returned anndata so that any changes do not affect the cache, and so that we never point to the cache version (though it now will be prevented from changing)
    """
    # Call cached function
    adata = _load_anndata_inner(fname)

    # Copy, for two reasons:
    # 1) We want to make sure we are not affecting any pointers to previous anndata.obs.
    #   This avoids issues where the first load sets the cache, the second load modifies the cached object, and then any pointers to the first load's .obs also contain unexpected updates due to the second call.
    # 2) We also want to make sure that edits to the returned anndata do not affect the cached version and do not manifest in separate future loads.
    return adata.copy()


# https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-01008-4
v_gene_renames = {"TRBV12-4": "TRBV12-3", "TRBV6-3": "TRBV6-2"}
v_allele_renames = {
    # Our old IgBLAST can generate TRBV6-2*02 calls, but no CDR1+2 information is available for this allele from get_tcr_v_gene_annotations, because it has been renamed:
    # https://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes&repertoire=genetable&species=human&group=TRBV - see (40)
    "TRBV6-2*02": "TRBV6-2*01",
}


def fix_gene_names(adata):
    # Standardize TRBV gene names that may be indistinguishable
    # see https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-01008-4
    # TODO: move to ETL
    adata.obs["v_gene"] = adata.obs["v_gene"].replace(v_gene_renames)
    return adata


def load_fold_embeddings_off_peak(
    fold_id: int,
    gene_locus: GeneLocus,
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE,
):
    """
    Load off-peak embeddings for a given fold.
    """
    adata = sc.read(
        config.paths.scaled_anndatas_dir
        / gene_locus.name
        / f"off_peak_timepoints.fold.{fold_id}.h5ad"
    )
    adata.obs = adata.obs.assign(fold_id=fold_id, fold_label="expanded_test_set")

    # produces 'disease.separate_past_exposures' obs column. TODO: move to ETL
    adata = label_past_exposures(adata)
    adata = fix_gene_names(adata)

    # load sequence weights for off-peak too
    if sample_weight_strategy == SampleWeightStrategy.ISOTYPE_USAGE:
        # calculate sample weights to balance out isotype proportions for each specimen
        adata.obs = genetools.helpers.merge_into_left(
            adata.obs,
            compute_isotype_sample_weights_column(adata).rename(
                "sample_weight_isotype_rebalance"
            ),
        )

    return adata


def load_known_binder_embeddings(
    fold_id: int,
    gene_locus: GeneLocus,
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE,
):
    # Load known binder sequence embeddings
    adata = joblib.load(
        config.paths.scaled_anndatas_dir
        / gene_locus.name
        / "known_binders.embedded.in.all.folds.joblib",
    )[fold_id]

    adata.obs = adata.obs.assign(fold_id=fold_id, fold_label="expanded_test_set")

    # produces 'disease.separate_past_exposures' obs column. TODO: move to ETL
    adata = label_past_exposures(adata)
    adata = fix_gene_names(adata)

    # set all sequence weights to 1 here
    adata.obs["sample_weight_isotype_rebalance"] = 1.0

    return adata


def load_fold_embeddings(
    fold_id,
    fold_label,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    anndatas_base_dir=None,
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE,
    load_isotype_counts_per_specimen=True,
):
    """Load entire fold's embedding anndata.
    Automatically caches several loaded fold embeddings between calls. For example, we might tune multiple models in a row on validation sets; why load the validation sets over and over again for each tuning operation?
    To clear cache, call helpers.clear_cached_fold_embeddings()

    Also supports loading an extra obs column `target_obs_column` to be used as an alternative classification target.
    Subsets to specimens with that obs column defined (so downstream code should revert to raw and rescale X).
    """

    TargetObsColumnEnum.validate(target_obs_column)
    SampleWeightStrategy.validate(sample_weight_strategy)
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)

    if anndatas_base_dir is None:
        anndatas_base_dir = config.paths.scaled_anndatas_dir / gene_locus.name

    anndatas_base_dir = Path(anndatas_base_dir)

    # load full anndata
    adata = _load_anndata(anndatas_base_dir / f"fold.{fold_id}.{fold_label}.h5ad")
    adata.obs = adata.obs.assign(fold_id=fold_id, fold_label=fold_label)

    adata = label_past_exposures(adata)
    adata = fix_gene_names(adata)

    if sample_weight_strategy == SampleWeightStrategy.ISOTYPE_USAGE:
        # calculate sample weights to balance out isotype proportions for each specimen
        adata.obs = genetools.helpers.merge_into_left(
            adata.obs,
            compute_isotype_sample_weights_column(adata).rename(
                "sample_weight_isotype_rebalance"
            ),
        )

    # Load study_name for all specimens (not just those with demographic data)
    # And load metadata (age, sex, ethnicity, and CMV status) for those specimens that have it
    adata.obs = pd.merge(
        adata.obs.drop(
            columns=[
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
        helpers.get_all_specimen_info().set_index("specimen_label")[
            [
                # Available for all
                "study_name",
                "participant_label",  # May be overriden
                "specimen_time_point",  # May be overriden
                "disease_subtype",  # May be overriden
                # Available for some
                "age",
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

    # Also add isotype proportions per specimen - generated by isotype_stats.ipynb
    # Unlike above demographic columns, this info is expected to be available for all specimens.
    # for BCR only
    # TODO: Generate this on the fly.
    if load_isotype_counts_per_specimen and gene_locus == GeneLocus.BCR:
        try:
            isotype_counts = pd.read_csv(
                config.paths.dataset_specific_metadata
                / "isotype_counts_by_specimen.tsv",
                sep="\t",
            )
            isotype_counts = isotype_counts[
                (isotype_counts["fold_id"] == fold_id)
                & (isotype_counts["fold_label"] == fold_label)
            ]
            isotype_counts = isotype_counts.set_index("specimen_label")[
                helpers.isotype_groups_kept[gene_locus]
            ]
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
        except FileNotFoundError as err:
            logger.warning(
                f"Not marking specimen isotype counts because those annotations weren't found: {err}"
            )

    if target_obs_column.value.obs_column_name not in adata.obs.columns:
        raise ValueError(
            f"Obs column {target_obs_column.value.obs_column_name} (from {target_obs_column}) not in adata.obs.columns. (Update metadata loading?)"
        )
    if adata.obs[target_obs_column.value.obs_column_name].isna().any():
        # Subset to specimens with defined target_obs_column
        adata_shape_orig = adata.shape
        adata_orig_num_specimens = adata.obs["specimen_label"].nunique()
        adata = adata[~adata.obs[target_obs_column.value.obs_column_name].isna()].copy()
        adata.uns["adata_was_subset_should_switch_to_raw"] = True
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
        ].copy()
        adata.uns["adata_was_subset_should_switch_to_raw"] = True
        adata_new_num_specimens = adata.obs["specimen_label"].nunique()
        logger.info(
            f"Target {target_obs_column}: filtering to specimens from diseases {target_obs_column.value.limited_to_disease} only; removed {adata_orig_num_specimens - adata_new_num_specimens} specimens"
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
        ].copy()

    # Remove unused disease column categories
    adata.obs["disease"] = adata.obs["disease"].cat.remove_unused_categories()

    if adata.obs_names.duplicated().any():
        raise ValueError("Loaded anndata has duplicated obs_names.")

    gc.collect()

    return adata


def clear_cached_fold_embeddings():
    """Clear LRU cache for load_fold_embeddings(), then run garbage collection"""
    _load_anndata_inner.cache_clear()
    gc.collect()


def compute_isotype_sample_weights_column(adata: anndata.AnnData) -> pd.Series:
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

        # returns one entry per sample, matching the sample's class's weight
        sample_weights = sklearn.utils.class_weight.compute_sample_weight(
            class_weight="balanced",
            y=isotype_values.values,
        )
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
