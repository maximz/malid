# %% [markdown]
# 1. Take one real Covid19 patient. Extract any sequences that are known binders (+/- 15% mutation allowed)
# 2. Take one real healthy person. Eliminate any sequences that are known binders to Covid (+/- 15% mutation)
# 3. Mix 1 and 2 at desired signal to noise ratio
#
# This is BCR only.
#

# %%
from typing import List
import numpy as np
import pandas as pd
import anndata
import gc
import shutil

import malid.external.genetools_scanpy_helpers
from malid import io
from malid import config, helpers, logger
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    healthy_label,
)

from malid.trained_model_wrappers import ConvergentClusterClassifier


# %%
# we're looking at BCR only here
gene_locus = GeneLocus.BCR

# %%
# If we want to generate based on an older dataset version, we can swap it in like this:
config.paths = config.make_paths(
    embedder=config.embedder, dataset_version="20220704_filter2"
)

# %%
config.paths.simulated_data_dir

# %%

# %%
## Get CoV-AbDab known binders
# TODO: switch to malid.interpretation provided loader

# set columnns our code expects (fold_label, fold_id, v_gene, j_gene, cdr3_aa_sequence_trim_len, cdr3_seq_aa_q_trim)
covabdab_df = pd.read_csv(
    config.paths.base_data_dir / "CoV-AbDab_310122.filtered.tsv", sep="\t"
).assign(fold_label="reference", fold_id=0)

covabdab_df = ConvergentClusterClassifier._cluster_training_set(
    df=covabdab_df,
    sequence_identity_threshold=0.95,  # very strict - in order to just prevent near-exact dupes
)

# Make cluster centroids for clusters, weighed by number of clone members (number of unique VDJ sequences)
cluster_centroids_df = ConvergentClusterClassifier._get_cluster_centroids(
    clustered_df=covabdab_df
)
# Reshape as dict
cluster_centroids_by_supergroup = (
    ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
        cluster_centroids=cluster_centroids_df
    )
)

# %%
def extract_sequences_from_specimen(
    adata: anndata.AnnData,
    keep_known_binding: bool,
    known_binders_cluster_centroids_by_supergroup,
    # threshold for assigning our seqs (test seqs) to known binder clusters
    clustering_test_assignment_threshold=0.85,
) -> anndata.AnnData:
    """given a specimen anndata,
    extract known binder sequences (+/- 15% mutation) if keep_known_binding,
    or extract sequences that are not known binders (+/- 15% mutation) if not keep_known_binding."""
    test_df = adata.obs.copy()
    # group by v, j, len
    test_groups = test_df.groupby(
        ["v_gene", "j_gene", "cdr3_aa_sequence_trim_len"], observed=True
    )

    test_df[["cluster_id_within_clustering_group", "distance_to_nearest_centroid"]] = (
        test_groups["cdr3_seq_aa_q_trim"].transform(
            lambda test_sequences: ConvergentClusterClassifier.assign_sequences_to_cluster_centroids(
                test_sequences,
                known_binders_cluster_centroids_by_supergroup,
                clustering_test_assignment_threshold,
            )
        )
        # extract the series of tuples into two columns while preserving the index
        # .to_list() is an alternative here, which may be faster, and seems to guarantee index ordering but not sure?
        .apply(pd.Series)
    )

    # create a globally meaningful "resulting cluster ID" for each row of df (each input sequence from each participant):
    test_df["global_resulting_cluster_ID"] = test_df[
        [
            "v_gene",
            "j_gene",
            "cdr3_aa_sequence_trim_len",
            "cluster_id_within_clustering_group",
        ]
    ].apply(tuple, axis=1)

    # which sequences assigned to any predictive clusters?
    unmatched_seqs_bool_vector = test_df["cluster_id_within_clustering_group"].isna()
    if keep_known_binding:
        obsnames = test_df[~unmatched_seqs_bool_vector].index
    else:
        obsnames = test_df[unmatched_seqs_bool_vector].index

    return adata[obsnames]


# %%
def _sample_from_two_populations_to_achieve_mix_ratio(
    population_size_a: int, population_size_b: int, ratio: float
):
    """
    Return how many items to sample from population A and population B to achieve a desired mix ratio.
    We have two populations of size a,b, with a known target signal-to-noise ratio r
    Solve for x,y fractions of each population to sample, given ax/(ax+by) = r (and x,y > 0)
    Also we want to use as much of possible of the available datasets.
    So figure out which dataset is the limiting factor given the ratio; take as much as possible from there.
    """
    # if taking all of a, i.e. x=1, then r(ax+by)=r(a+by)=a.
    x = 1
    y = population_size_a / population_size_b * (1 - ratio) / ratio
    if y > 1:
        # this means that using all of a would require using more than 100% of b - which is impossible.
        # so b is the limiting factor. can't take all of a; let's take all of b.
        # if taking all of b, i.e. y=1, then we have (1-r)(ax+by) = (1-r)(ax+b) = b
        y = 1
        x = population_size_b / population_size_a * ratio / (1 - ratio)
    # round and return how much to sample from each population
    return int(round(x * population_size_a)), int(round(y * population_size_b))


def make_simulated_patient(
    signal_to_noise_ratio: float,
    disease_adata: anndata.AnnData,
    healthy_adata: anndata.AnnData,
) -> anndata.AnnData:
    """create a synthetic disease patient by mixing a real disease patient's known binding sequences and a real healthy donor's sequences that don't match known binders."""
    new_specimen_label = f"{disease_adata.obs['specimen_label'].iloc[0]}_{healthy_adata.obs['specimen_label'].iloc[0]}"
    new_participant_label = f"{disease_adata.obs['participant_label'].iloc[0]}_{healthy_adata.obs['participant_label'].iloc[0]}"

    def _label_anndata(adata, disease_status):
        adata = adata.copy()
        # label the seqs as disease or not
        adata.obs["is_actually_disease"] = disease_status
        # combine the participant labels and specimen labels
        adata.obs["participant_label"] = new_participant_label
        adata.obs["specimen_label"] = new_specimen_label
        return adata

    disease_part = _label_anndata(disease_adata, True)
    healthy_part = _label_anndata(healthy_adata, False)

    # Mix with signal_to_noise_ratio
    n_disease, n_healthy = _sample_from_two_populations_to_achieve_mix_ratio(
        population_size_a=disease_part.shape[0],
        population_size_b=healthy_part.shape[0],
        ratio=signal_to_noise_ratio,
    )
    if n_disease > disease_part.shape[0] or n_healthy > healthy_part.shape[0]:
        # this should not happen
        raise ValueError(
            f"Not enough healthy or disease sequences to achieve desired signal-to-noise ratio {signal_to_noise_ratio}."
        )
    effective_ratio = n_disease / (n_disease + n_healthy)
    logger.info(
        f"Combining {new_participant_label}, {new_specimen_label}: effective signal-to-noise ratio {effective_ratio} ({n_disease} disease and {n_healthy} healthy sequences); desired ratio {signal_to_noise_ratio}."
    )

    # sample and concatenate
    return anndata.concat(
        [
            disease_part[
                np.random.choice(
                    a=disease_part.obs_names, size=n_disease, replace=False
                )
            ],
            healthy_part[
                np.random.choice(
                    a=healthy_part.obs_names, size=n_healthy, replace=False
                )
            ],
        ]
    )


# %%
def validate_passes_thresholds(
    selected_seqs: anndata.AnnData,
    #     n_sequences_per_specimen_per_isotype: int,
    min_number_of_sequences_per_specimen: int,
) -> bool:
    # Sanity check that all isotypes are present:
    if set(selected_seqs.obs["isotype_supergroup"].unique()) != set(
        helpers.isotype_groups_kept[gene_locus]
    ):
        return False

    #     # Sanity check that there are enough sequences from each isotype
    #     for isotype, obs_subset in selected_seqs.obs.groupby(
    #         "isotype_supergroup", observed=True
    #     ):
    #         if obs_subset.shape[0] < n_sequences_per_specimen_per_isotype:
    #             return False

    # Sanity check that there are enough sequences from this specimen
    if selected_seqs.shape[0] < min_number_of_sequences_per_specimen:
        return False

    return True


# %%
def simulate_for_fold(
    adata_fold: anndata.AnnData,
    signal_to_noise_ratio: float,
    n_disease_specimens: int,
    n_healthy_specimens: int,
    #     n_sequences_per_specimen_per_isotype: int,
    min_number_of_sequences_per_specimen: int,
) -> anndata.AnnData:
    selected_disease_samples: List[anndata.AnnData] = []
    selected_healthy_samples: List[anndata.AnnData] = []

    already_sampled_participants = set()

    participants_and_specimens_by_disease = {
        disease: grp.values
        for disease, grp in adata_fold.obs[
            ["disease", "specimen_label", "participant_label"]
        ]
        .drop_duplicates()
        .groupby("disease")
    }

    for (
        _,
        disease_specimen_label,
        source_participant_label,
    ) in participants_and_specimens_by_disease["Covid19"]:
        if len(selected_disease_samples) == n_disease_specimens:
            # have enough
            break
        if source_participant_label in already_sampled_participants:
            # already used another specimen from this patient
            logger.warning(
                f"Skipping disease specimen {disease_specimen_label} because we already included another sample from {source_participant_label}."
            )
            continue
        # sample from a real patient
        known_binding_seqs: anndata.AnnData = extract_sequences_from_specimen(
            adata_fold[adata_fold.obs["specimen_label"] == disease_specimen_label],
            keep_known_binding=True,
            known_binders_cluster_centroids_by_supergroup=cluster_centroids_by_supergroup,
        )
        if validate_passes_thresholds(
            selected_seqs=known_binding_seqs,
            #             n_sequences_per_specimen_per_isotype=n_sequences_per_specimen_per_isotype,
            min_number_of_sequences_per_specimen=min_number_of_sequences_per_specimen,
        ):
            selected_disease_samples.append(known_binding_seqs)
            already_sampled_participants.add(source_participant_label)
            logger.info(
                f"Adding disease specimen {disease_specimen_label} from {source_participant_label}."
            )
        else:
            logger.warning(
                f"Disease specimen {disease_specimen_label} from {source_participant_label} did not pass thresholds."
            )

    for (
        _,
        healthy_specimen_label,
        source_participant_label,
    ) in participants_and_specimens_by_disease[healthy_label]:
        # need one per synthetic disease patient and one per healthy control we create
        if len(selected_healthy_samples) == n_disease_specimens + n_healthy_specimens:
            # have enough
            break
        if source_participant_label in already_sampled_participants:
            # already used another specimen from this patient
            logger.warning(
                f"Skipping healthy specimen {healthy_specimen_label} because we already included another sample from {source_participant_label}."
            )
            continue
        # sample from a real healthy donor
        known_not_binding_seqs = extract_sequences_from_specimen(
            adata_fold[adata_fold.obs["specimen_label"] == healthy_specimen_label],
            keep_known_binding=False,
            known_binders_cluster_centroids_by_supergroup=cluster_centroids_by_supergroup,
        )
        if validate_passes_thresholds(
            selected_seqs=known_not_binding_seqs,
            #             n_sequences_per_specimen_per_isotype=n_sequences_per_specimen_per_isotype,
            min_number_of_sequences_per_specimen=min_number_of_sequences_per_specimen,
        ):
            selected_healthy_samples.append(known_not_binding_seqs)
            already_sampled_participants.add(source_participant_label)
            logger.info(
                f"Adding healthy specimen {healthy_specimen_label} from {source_participant_label}."
            )
        else:
            logger.warning(
                f"Healthy specimen {healthy_specimen_label} from {source_participant_label} did not pass thresholds."
            )

    if len(selected_disease_samples) != n_disease_specimens:
        raise ValueError("Did not select enough disease specimens")
    if len(selected_healthy_samples) != n_disease_specimens + n_healthy_specimens:
        raise ValueError("Did not select enough healthy specimens")

    # make synthetic mixtures for disease patients
    returned_specimens: List[anndata.AnnData] = []
    for _ in range(n_disease_specimens):
        disease_specimen, healthy_specimen = (
            selected_disease_samples.pop(),
            selected_healthy_samples.pop(),
        )
        synthetic_specimen: anndata.AnnData = make_simulated_patient(
            signal_to_noise_ratio, disease_specimen, healthy_specimen
        )
        returned_specimens.append(synthetic_specimen)
    for _ in range(n_healthy_specimens):
        returned_specimens.append(selected_healthy_samples.pop())

    # package up as an anndata
    # undo any scaling
    final_anndata = anndata.concat(returned_specimens).raw.to_adata()
    final_anndata.obs_names_make_unique()

    # remove unused labels, if these variables are Categoricals
    final_anndata.obs["participant_label"] = (
        final_anndata.obs["participant_label"]
        .astype("category")
        .cat.remove_unused_categories()
    )
    final_anndata.obs["specimen_label"] = (
        final_anndata.obs["specimen_label"]
        .astype("category")
        .cat.remove_unused_categories()
    )

    # no need to pass old PCA info along
    del final_anndata.obsm

    return final_anndata


# %%
pca_n_comps = 10

signal_to_noise_ratios = [0.05, 0.10, 0.25]

output_dirs = {
    signal_to_noise_ratio: config.paths.simulated_data_dir
    / f"scaled_anndatas_{signal_to_noise_ratio:0.2f}"
    for signal_to_noise_ratio in signal_to_noise_ratios
}
for output_dir_anndatas in output_dirs.values():
    # Clear out and remove folder if it already exists
    if output_dir_anndatas.exists():
        if not output_dir_anndatas.is_dir():
            raise ValueError(
                f"Output directory {output_dir_anndatas} already xists but is not a directory."
            )
        shutil.rmtree(output_dir_anndatas)

    output_dir_anndatas.mkdir(parents=True, exist_ok=False)
    print(output_dir_anndatas)


for fold_id in config.cross_validation_fold_ids:
    # These transformations will be fit on train_smaller set and applied to others
    # so they start as None and then will be replaced.
    scale_transformers = {ratio: None for ratio in signal_to_noise_ratios}
    pca_transformers = {ratio: None for ratio in signal_to_noise_ratios}

    for fold_label in ["train_smaller", "validation", "test"]:
        if fold_id == -1 and fold_label == "test":
            # skip: global fold does not have a test set
            continue

        adata_fold = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=gene_locus,
            target_obs_column=TargetObsColumnEnum.disease,
            sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
            load_isotype_counts_per_specimen=False,
        )
        for signal_to_noise_ratio in signal_to_noise_ratios:
            # vary fraction disease specific

            # TODO: we can also pull selected_disease_samples,selected_healthy_samples out to be shared between signal_to_noise_ratios
            # (but would need to change pop())
            adata = simulate_for_fold(
                adata_fold=adata_fold,
                signal_to_noise_ratio=signal_to_noise_ratio,
                n_disease_specimens=5,
                n_healthy_specimens=5,
                # n_sequences_per_specimen_per_isotype=100, # removed because expect mostly IgG from Covid patients
                min_number_of_sequences_per_specimen=10,  # TODO: 100
            )

            # Now scale, PCA, and export the data
            # Use transformers if available (starts as None)
            (
                adata,
                scale_transformers[signal_to_noise_ratio],
            ) = malid.external.genetools_scanpy_helpers.scale_anndata(
                adata,
                scale_transformer=scale_transformers[signal_to_noise_ratio],
                inplace=True,
                set_raw=True,
            )
            (
                adata,
                pca_transformers[signal_to_noise_ratio],
            ) = malid.external.genetools_scanpy_helpers.pca_anndata(
                adata,
                pca_transformer=pca_transformers[signal_to_noise_ratio],
                n_components=pca_n_comps,
                inplace=True,
            )
            if adata.obsm["X_pca"].shape[1] != pca_n_comps:
                raise ValueError(
                    "PCA did not produce the expected number of components"
                )

            # Reduce disk space usage by removing unnecessary obs columns
            adata.obs.drop(
                columns=list(
                    set(adata.obs.columns)
                    - set(adata.uns.get("original_obs_columns", []))
                    # do not delete this column
                    - {"is_actually_disease"}
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
            assert "disease" in adata.obs.columns
            assert "is_actually_disease" in adata.obs.columns

            # Also remove any uns keys that were added after the original read-from-disk step within load_fold_embeddings
            for key in set(adata.uns.keys()) - set(
                adata.uns.get("original_uns_keys", [])
            ):
                del adata.uns[key]

            # Also remove large string index
            # this is a RangeIndex, but after reading back in, these will become strings automatically (ImplicitModificationWarning: Transforming to str index.)
            adata.obs_names = range(adata.shape[0])

            # Save some space on this field too
            adata.obs["v_mut"] = adata.obs["v_mut"].astype(np.float32)

            # Write anndata and CSVs to disk
            output_dir_anndatas = output_dirs[signal_to_noise_ratio]
            fname_out = output_dir_anndatas / f"fold.{fold_id}.{fold_label}.h5ad"
            logger.info(f"Fold {fold_id}-{fold_label}, {gene_locus} -> {fname_out}")
            adata.write(fname_out, compression="gzip")
            adata.obs.to_csv(
                output_dir_anndatas / f"fold.{fold_id}.{fold_label}.obs.tsv.gz",
                index=None,
                sep="\t",
            )
            np.savetxt(
                output_dir_anndatas / f"fold.{fold_id}.{fold_label}.X.tsv.gz",
                adata.X,
                fmt="%0.4f",
                delimiter="\t",
            )

            # Replace .X with X_pca
            adata = anndata.AnnData(
                X=adata.obsm["X_pca"],
                obs=adata.obs,
                uns=adata.uns,
            )
            # Writing out the anndata again is unnecessary - already have X_pca in original
            adata.write(
                output_dir_anndatas / f"fold.{fold_id}.{fold_label}.pca.h5ad",
                compression="gzip",
            )
            np.savetxt(
                output_dir_anndatas / f"fold.{fold_id}.{fold_label}.X_pca.tsv.gz",
                adata.X,
                fmt="%0.4f",
                delimiter="\t",
            )

            del adata
            gc.collect()

        # after finishing all desired signal to noise ratios, delete cached dataset
        io.clear_cached_fold_embeddings()
        gc.collect()


# %%

# %%

# %%
