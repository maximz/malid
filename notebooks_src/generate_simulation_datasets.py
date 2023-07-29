# -*- coding: utf-8 -*-
# %% [markdown]
# Use previous model to choose the most and least disease specific sequences from real Covid, HIV, and Healthy repertoires:
# - 10 individuals per class
# - 1000 sequences per isotype
# - PCA’ed to 10 components.
#
# Also filter to known important V genes:
# - HIV: V4-34, V4-61, V4-4, V3-20
# - Covid19: V1-24, V3-13, V3-9, V3-53
#
# Signal to noise ratios: 25%, 50%, 75%
#
# Recapitulates what logistic regression excels at.

# %%
from typing import Any, Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata
import genetools
import gc
import shutil
from pathlib import Path
from collections import defaultdict

import malid.external.genetools_scanpy_helpers
from malid import io
from malid.trained_model_wrappers import SequenceClassifier
from malid import config, helpers, logger
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    healthy_label,
)

# %%
# in an effort to generate more convergent sequence clusters, let's filter down V genes
# v_genes_important_to_disease (if disease not listed here, it won't be filtered):
v_gene_filter: Dict[GeneLocus, Dict[str, List[str]]] = {
    GeneLocus.BCR: {
        "HIV": ["IGHV4-34", "IGHV4-61", "IGHV4-4", "IGHV3-20"],
        "Covid19": ["IGHV1-24", "IGHV3-13", "IGHV3-9", "IGHV3-43"],
    },
    GeneLocus.TCR: {},
}

# %%
# If we want to generate based on an older dataset version, we can swap it in here:
dataset_version = config.DATASET_VERSION  # "20220930_expand"

config.paths = config.make_paths(
    embedder=config.embedder, dataset_version=dataset_version
)


# %%

# %%
def copy_metadata(destination_dir: Path):
    destination_dir.mkdir(exist_ok=True, parents=True)
    for fname in config.paths.dataset_specific_metadata.glob("*"):
        shutil.copy2(fname, destination_dir)
        print(fname)


# %%
def _sample_one_specimen(
    adata_specimen: anndata.AnnData,
    specimen_label: str,
    gene_locus: GeneLocus,
    disease: str,
    clf: SequenceClassifier,
    n_sequences_per_patient_per_isotype: int,
    fraction_disease_specific: float,
) -> Union[Dict[str, List[np.ndarray]], None]:
    """For one specimen anndata belonging to one gene locus,
    select and return a subset of obsnames,
    marked with identites healthy, not_disease, or true_disease.
    returns None if there were not enough sequences per isotype."""
    # We are already guaranteed, by the construction of the original dataset, that all isotypes are present for this specimen,
    # but their counts aren't guaranteed.
    # Just in case, we will sanity check that all isotypes are present:
    if set(adata_specimen.obs["isotype_supergroup"].unique()) != set(
        helpers.isotype_groups_kept[gene_locus]
    ):
        logger.warning(
            f"Specimen {specimen_label} from disease {disease} missing some isotypes altogether for {gene_locus} - skipping specimen"
        )
        return None

    obs_names_to_keep_for_this_specimen = defaultdict(list)

    for isotype, adata_subset in helpers.anndata_groupby_obs(
        adata_specimen, "isotype_supergroup", observed=False
    ):
        if adata_subset.shape[0] < n_sequences_per_patient_per_isotype:
            logger.warning(
                f"Specimen {specimen_label}, isotype {isotype}, disease {disease}, {gene_locus}: only had {adata_subset.shape[0]} sequences - skipping specimen"
            )
            # stop looking at this patient - don't include any isotypes staged so far
            return None

        # score the sequences - get predicted probabilities for this disease
        # doesn't matter if we use adjusted decision thresholds because that is just reweighting the entire class by a factor. rankings within the class won't change (unless we renormalize the rows)
        featurized = clf.featurize(adata_subset)
        scores = pd.DataFrame(
            clf.predict_proba(
                featurized.X,
            ),
            index=featurized.sample_names,
            columns=clf.classes_,
        )
        # pull out sequence probability for this disease - still indexed by obsname
        scores = scores[disease]
        if scores.isna().any():
            raise ValueError(f"Specimen {specimen_label}: disease_pr contains NaN")

        # get obsnames in sorted order
        sorted_order = scores.sort_values().index.to_series()

        if disease == healthy_label:
            # take a random smattering of n_sequences_per_patient_per_isotype sequences
            obs_names_to_keep_for_this_specimen["healthy"].append(
                sorted_order.sample(
                    n=n_sequences_per_patient_per_isotype, random_state=0
                ).values
            )
        else:
            # take from top (unlikely to be disease specific)
            obs_names_to_keep_for_this_specimen["not_disease"].append(
                sorted_order.head(
                    n=int(
                        (1 - fraction_disease_specific)
                        * n_sequences_per_patient_per_isotype
                    )
                ).values
            )
            # take from bottom (likely to be disease specific)
            obs_names_to_keep_for_this_specimen["true_disease"].append(
                sorted_order.tail(
                    n=int(
                        fraction_disease_specific * n_sequences_per_patient_per_isotype
                    )
                ).values
            )

    return obs_names_to_keep_for_this_specimen


# %%
def sample_from_fold(
    fold_id: int,
    fold_label: str,
    gene_loci: GeneLocus,
    n_specimens_per_disease: int,
    n_sequences_per_patient_per_isotype: int,
    fraction_disease_specific: float,
    diseases_kept: Optional[List[str]] = None,
    v_genes_kept: Optional[Dict[GeneLocus, Dict[str, List[str]]]] = None,
) -> Dict[GeneLocus, anndata.AnnData]:
    """
    Sample from fold for all gene loci simultaneously,
    so that we have matching specimen lists for all gene loci.
    (in other words, a specimen must pass the relevant BCR *and* TCR filters to be included in either)
    """
    # Load data
    adatas: Dict[GeneLocus, anndata.AnnData] = {}
    clfs: Dict[GeneLocus, SequenceClassifier] = {}
    diseases = None

    for gene_locus in gene_loci:
        adata = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=gene_locus,
            # Require that all participants have all demorgaphic columns defined
            target_obs_column=TargetObsColumnEnum.disease_all_demographics_present,
            sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
        )

        if diseases_kept is not None:
            adata = adata[adata.obs["disease"].isin(diseases_kept)]

        if adata.obs_names.duplicated().any():
            raise ValueError("obs_names had dupes")

        # Get diseases list and confirm all anndatas match it
        if diseases is None:
            diseases = adata.obs["disease"].unique()
        else:
            if set(diseases) != set(adata.obs["disease"].unique()):
                raise ValueError("Disease list mismatch between anndatas.")

        adatas[gene_locus] = adata.copy()
        del adata
        io.clear_cached_fold_embeddings()
        gc.collect()

        # load individual sequence classifier
        clf = SequenceClassifier(
            fold_id=fold_id,
            model_name_sequence_disease="lasso_multiclass",
            fold_label_train="train_smaller",
            gene_locus=gene_locus,
            # Match above
            target_obs_column=TargetObsColumnEnum.disease_all_demographics_present,
            sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
        )
        clfs[gene_locus] = clf

        if not set(diseases) <= set(clf.classes_):
            # all diseases should be in the classifier's classes
            raise ValueError(
                f"Disease list {diseases} should be a subset of (or equal to) clf classes {clf.classes_}"
            )

    obs_names_to_keep_for_all_specimens_by_gene_locus: Dict[
        GeneLocus, Dict[str, List[np.ndarray]]
    ] = {gene_locus: defaultdict(list) for gene_locus in gene_loci}

    # Sample from each disease.
    for disease in diseases:
        # For each gene locus,
        # limit the full anndata to sequences originating from that disease's patients and with particular V genes
        adatas_filtered: Dict[GeneLocus, anndata.AnnData] = {}
        for gene_locus, adata in adatas.items():
            adatas_filtered[gene_locus] = adata[adata.obs["disease"] == disease]
            if (
                v_genes_kept is not None
                and v_genes_kept.get(gene_locus) is not None
                and disease in v_genes_kept[gene_locus]
            ):
                adatas_filtered[gene_locus] = adatas_filtered[gene_locus][
                    adatas_filtered[gene_locus]
                    .obs["v_gene"]
                    .isin(v_genes_kept[gene_locus][disease])
                ]

        # track how many specimens we successfully included already
        n_specimens_kept_from_this_disease = 0
        # track which participants these specimens came from
        participants_represented = set()

        # Get participant+specimen list from first anndata (should match the rest)
        first_anndata = next(iter(adatas_filtered.values()))
        specimen_list = (
            first_anndata.obs[["participant_label", "specimen_label"]]
            .drop_duplicates()
            .values
        )

        # Sample a certain number of sequences per isotype (so we get all isotypes)
        for (participant_label, specimen_label) in specimen_list:
            if n_specimens_kept_from_this_disease >= n_specimens_per_disease:
                # we have enough patients already, stop looking at more specimens to add
                break

            if participant_label in participants_represented:
                logger.warning(
                    f"Specimen {specimen_label} from disease {disease} will be skipped because we already have another specimen from same participant {participant_label}"
                )
                # skip to next specimen
                continue

            # Find which obsnames we'd sample from this specimen in each gene locus dataset
            obs_names_to_keep_for_this_specimen_by_locus: Dict[
                GeneLocus, Union[Dict[str, List[np.ndarray]], None]
            ] = {
                gene_locus: _sample_one_specimen(
                    adata_specimen=adatas_filtered[gene_locus][
                        adatas_filtered[gene_locus].obs["specimen_label"]
                        == specimen_label
                    ],
                    specimen_label=specimen_label,
                    gene_locus=gene_locus,
                    disease=disease,
                    clf=clfs[gene_locus],
                    n_sequences_per_patient_per_isotype=n_sequences_per_patient_per_isotype,
                    fraction_disease_specific=fraction_disease_specific,
                )
                for gene_locus in gene_loci
            }

            # Confirm that this specimen passed filters in each gene locus
            if any(
                obsnames_in_one_locus is None
                for obsnames_in_one_locus in obs_names_to_keep_for_this_specimen_by_locus.values()
            ):
                # Not all isotypes were sampled
                # Skip this specimen for all isotypes and gene loci,
                # so we don't have any specimens missing some isotypes (leads to NaNs in model1 feature matrix)
                # or missing in some gene loci (breaks metamodel).
                logger.warning(
                    f"Specimen {specimen_label} from participant {participant_label}, disease {disease} will not be included due to missing/incomplete isotypes (in one or more gene loci)."
                )
            else:
                for (
                    gene_locus,
                    obsnames_in_one_locus,
                ) in obs_names_to_keep_for_this_specimen_by_locus.items():
                    for (
                        sequence_identity,
                        sequence_obsnames,
                    ) in obsnames_in_one_locus.items():
                        obs_names_to_keep_for_all_specimens_by_gene_locus[gene_locus][
                            sequence_identity
                        ].extend(sequence_obsnames)
                n_specimens_kept_from_this_disease += 1
                participants_represented.add(participant_label)
                logger.info(
                    f"Added specimen {specimen_label} from participant {participant_label}, disease {disease}"
                )

        if n_specimens_kept_from_this_disease != n_specimens_per_disease:
            # Confirm we got the right amount of specimens
            raise ValueError(
                f"We selected only {n_specimens_kept_from_this_disease} specimens from {disease}, rather than desired {n_specimens_per_disease} - fold {fold_id}-{fold_label}, {gene_loci}"
            )
        del adatas_filtered
        gc.collect()

    # Actually perform the sampling and arrive at resulting anndatas
    returned_anndatas: Dict[GeneLocus, anndata.AnnData] = {}
    for gene_locus, adata in adatas.items():
        obs_names_to_keep: Dict[
            str, List[np.ndarray]
        ] = obs_names_to_keep_for_all_specimens_by_gene_locus[gene_locus]

        # Flatten lists of indices from all participants (still separated by healthy/not-disease/true-disease sequence identities)
        obs_names_to_keep_flattened: Dict[str, np.ndarray] = {
            sequence_identity: np.array(
                list_of_np_arrays_for_one_sequence_identity
            ).ravel()
            for sequence_identity, list_of_np_arrays_for_one_sequence_identity in obs_names_to_keep.items()
        }

        # Combine all indices across all sequence identities
        all_obs_names_to_keep = np.hstack(list(obs_names_to_keep_flattened.values()))

        # Return adata at selected indices (across all identities)
        # and undo any scaling
        adata_export = adata[all_obs_names_to_keep, :].raw.to_adata()
        del adata
        gc.collect()

        # Mark identities in obs
        adata_export.obs["sequence_identity_is_true_disease"] = pd.Series(
            dtype=pd.CategoricalDtype(categories=obs_names_to_keep_flattened.keys())
        )
        for sequence_identity, obsnames in obs_names_to_keep_flattened.items():
            adata_export.obs.loc[
                obsnames, "sequence_identity_is_true_disease"
            ] = sequence_identity

        # remove unused labels, if these variables are Categoricals
        adata_export.obs["participant_label"] = (
            adata_export.obs["participant_label"]
            .astype("category")
            .cat.remove_unused_categories()
        )
        adata_export.obs["specimen_label"] = (
            adata_export.obs["specimen_label"]
            .astype("category")
            .cat.remove_unused_categories()
        )

        # no need to pass old PCA info along
        del adata_export.obsm

        returned_anndatas[gene_locus] = adata_export

    del adatas
    gc.collect()
    return returned_anndatas


# %%
def run(
    output_dir_anndatas: Path,
    gene_loci: GeneLocus,
    n_specimens_per_disease: int,
    n_sequences_per_patient_per_isotype: int,
    fraction_disease_specific: float,
    scale_data=False,
    store_raw_pre_scaling=True,
    pca_n_comps: Optional[int] = None,
    diseases_kept: Optional[List[str]] = None,
    v_genes_kept: Optional[Dict[GeneLocus, Dict[str, List[str]]]] = None,
    write_csvs=False,
    include_global_fold=True,
):
    for fold_id in (
        config.all_fold_ids if include_global_fold else config.cross_validation_fold_ids
    ):
        # These transformations will be fit on train_smaller set and applied to others
        # so they start as None and then will be replaced.
        # indexed by gene_locus - i.e. the transformations are different for each gene locus (because coming from different language models)
        scale_transformers: Dict[GeneLocus, Any] = {
            gene_locus: None for gene_locus in gene_loci
        }
        pca_transformers: Dict[GeneLocus, Any] = {
            gene_locus: None for gene_locus in gene_loci
        }

        for fold_label in ["train_smaller", "validation", "test"]:
            if fold_id == -1 and fold_label == "test":
                # skip: global fold does not have a test set
                continue

            # Sample from fold for all gene loci simultaneously,
            # so that we have matching specimen lists for all gene loci.
            # (in other words, a specimen must pass the relevant BCR *and* TCR filters to be included in either)
            adatas_sampled: Dict[GeneLocus, anndata.AnnData] = sample_from_fold(
                fold_id=fold_id,
                fold_label=fold_label,
                gene_loci=gene_loci,
                n_specimens_per_disease=n_specimens_per_disease,
                n_sequences_per_patient_per_isotype=n_sequences_per_patient_per_isotype,
                fraction_disease_specific=fraction_disease_specific,
                diseases_kept=diseases_kept,
                v_genes_kept=v_genes_kept,
            )

            # Now scale, PCA, and export the data separately for each gene locus (because coming from different language models).
            for gene_locus in gene_loci:
                output_dir_anndatas_for_gene_locus = (
                    output_dir_anndatas / gene_locus.name
                )
                output_dir_anndatas_for_gene_locus.mkdir(exist_ok=True, parents=True)

                fname_out = (
                    output_dir_anndatas_for_gene_locus
                    / f"fold.{fold_id}.{fold_label}.h5ad"
                )
                logger.info(f"Fold {fold_id}-{fold_label}, {gene_locus} -> {fname_out}")

                if scale_data:
                    # Scale inplace and set raw (if requested)
                    # Use transformer if available (starts as None)
                    (
                        adatas_sampled[gene_locus],
                        scale_transformers[gene_locus],
                    ) = malid.external.genetools_scanpy_helpers.scale_anndata(
                        adatas_sampled[gene_locus],
                        scale_transformer=scale_transformers[gene_locus],
                        inplace=True,
                        set_raw=store_raw_pre_scaling,
                    )

                # that N x 1900 matrix is too big. let's save a PCA'ed version for our tests.
                if pca_n_comps is not None:
                    # PCA inplace
                    # Use transformer if available (starts as None)
                    (
                        adatas_sampled[gene_locus],
                        pca_transformers[gene_locus],
                    ) = malid.external.genetools_scanpy_helpers.pca_anndata(
                        adatas_sampled[gene_locus],
                        pca_transformer=pca_transformers[gene_locus],
                        n_components=pca_n_comps,
                        inplace=True,
                    )
                    # Replace .X with X_pca
                    adatas_sampled[gene_locus] = anndata.AnnData(
                        X=adatas_sampled[gene_locus].obsm["X_pca"],
                        obs=adatas_sampled[gene_locus].obs,
                        uns=adatas_sampled[gene_locus].uns,
                    )
                    if adatas_sampled[gene_locus].shape[1] != pca_n_comps:
                        raise ValueError(
                            "PCA did not produce the expected number of components"
                        )

                # Some columns like "cmv" may be all NaN in this simulated dataset.
                # This can lead to an anndata / h5py bug:
                # "TypeError: Can't implicitly convert non-string objects to strings
                # Above error raised while writing key 'cmv' of <class 'h5py._hl.group.Group'> to /"
                # This seems caused by adatas_sampled[gene_locus].obs['cmv'].dtype being dtype('O') instead of dtype("float64").
                # We can cast to float:
                #         for col in adatas_sampled[gene_locus].obs.columns:
                #             if adatas_sampled[gene_locus].obs[col].isna().all():
                #                 adatas_sampled[gene_locus].obs[col] = adatas_sampled[gene_locus].obs[col].astype("float")

                # Reduce disk space usage by removing unnecessary obs columns
                adatas_sampled[gene_locus].obs.drop(
                    columns=list(
                        set(adatas_sampled[gene_locus].obs.columns)
                        - set(
                            adatas_sampled[gene_locus].uns.get(
                                "original_obs_columns", []
                            )
                        )
                        # do not delete this column
                        - {"sequence_identity_is_true_disease"}
                    )
                    + [
                        "num_reads",
                        "total_clone_num_reads",
                        "num_clone_members",
                        "cdr1_seq_aa_q_trim",
                        "cdr2_seq_aa_q_trim",
                        "extracted_isotype",
                        "igh_or_tcrb_clone_id",
                        "cdr3_aa_sequence_trim_len",
                        "disease_subtype",
                    ],
                    errors="ignore",
                    inplace=True,
                )
                # Sanity check: make sure we did not drop these columns
                assert "disease" in adatas_sampled[gene_locus].obs.columns
                assert (
                    "sequence_identity_is_true_disease"
                    in adatas_sampled[gene_locus].obs.columns
                )

                # Also remove any uns keys that were added after the original read-from-disk step within load_fold_embeddings
                for key in set(adatas_sampled[gene_locus].uns.keys()) - set(
                    adatas_sampled[gene_locus].uns.get("original_uns_keys", [])
                ):
                    del adatas_sampled[gene_locus].uns[key]

                # Also remove large string index
                # this is a RangeIndex, but after reading back in, these will become strings automatically (ImplicitModificationWarning: Transforming to str index.)
                adatas_sampled[gene_locus].obs_names = range(
                    adatas_sampled[gene_locus].shape[0]
                )

                # Save some space on this field too
                adatas_sampled[gene_locus].obs["v_mut"] = (
                    adatas_sampled[gene_locus].obs["v_mut"].astype(np.float32)
                )

                # Write to disk
                adatas_sampled[gene_locus].write(fname_out, compression="gzip")
                if write_csvs:
                    adatas_sampled[gene_locus].obs.to_csv(
                        output_dir_anndatas_for_gene_locus
                        / f"fold.{fold_id}.{fold_label}.obs.tsv.gz",
                        index=None,
                        sep="\t",
                    )
                    np.savetxt(
                        output_dir_anndatas_for_gene_locus
                        / f"fold.{fold_id}.{fold_label}.X.tsv.gz",
                        adatas_sampled[gene_locus].X,
                        fmt="%0.4f",
                        delimiter="\t",
                    )

            io.clear_cached_fold_embeddings()
            gc.collect()


# %%

# %% [markdown]
# # Generate small simulation dataset for end-to-end test

# %%
copy_metadata(
    destination_dir=config.paths.tests_snapshot_dir / "dataset_specific_metadata"
)

run(
    output_dir_anndatas=config.paths.tests_snapshot_dir / "scaled_anndatas_dir",
    gene_loci=config.gene_loci_used,
    # in each fold:
    n_specimens_per_disease=3,
    # this is the required number of sequences _after_ filtering by V genes. since we are just doing end to end automated tests, we can keep this small
    n_sequences_per_patient_per_isotype=50,
    fraction_disease_specific=0.9,  # signal to noise ratio
    scale_data=False,  # Don't scale, in order to save space. Handled after the fact (but independently for each fold label) in the test suite directly.
    pca_n_comps=2,  # Reduce dimensions for testing. Technically we would want to scale before running PCA, but doesn't matter in this test example
    diseases_kept=[healthy_label, "HIV", "Covid19"],
    # in an effort to generate more convergent sequence clusters, let's filter down V genes
    # v_genes_important_to_disease (if disease not listed here, it won't be filtered):
    v_genes_kept=v_gene_filter,
    write_csvs=False,
    include_global_fold=True,
)


# %%

# %% [markdown]
# # Generate simulation dataset
#
#

# %%
# # copy_metadata(destination_dir=config.paths.simulated_data_dir / "metadata")

# for fraction_disease_specific in [0.25, 0.5, 0.75]:
#     output_dir_anndatas = (
#         config.paths.simulated_data_dir
#         / f"scaled_anndatas_{fraction_disease_specific:0.2f}"
#     )
#     print(output_dir_anndatas)

#     run(
#         output_dir_anndatas=output_dir_anndatas,
#         gene_loci=config.gene_loci_used,
#         # in each fold:
#         n_specimens_per_disease=10,
#         n_sequences_per_patient_per_isotype=100,  # this is the required number of sequences _after_ filtering by V genes
#         fraction_disease_specific=fraction_disease_specific,  # signal to noise ratio
#         scale_data=True,
#         store_raw_pre_scaling=True,
#         pca_n_comps=10,  # or set to None to prevent dimensionality reduction
#         diseases_kept=[healthy_label, "HIV", "Covid19"],
#         # in an effort to generate more convergent sequence clusters, let's filter down V genes
#         # v_genes_important_to_disease (if disease not listed here, it won't be filtered):
#         v_genes_kept=v_gene_filter,
#         write_csvs=True,
#         include_global_fold=False,
#     )


# %%

# %%
