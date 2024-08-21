# %%
import pandas as pd
from typing import Tuple
import anndata
from malid import config, io, helpers, logger
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    map_cross_validation_split_strategy_to_default_target_obs_column,
    healthy_label,
)
from malid.trained_model_wrappers import ConvergentClusterClassifier
from IPython.display import display, Markdown

pd.set_option("display.max_columns", None)

# %% [markdown]
# In this notebook, we:
#
# 1. Check how many BCR Covid-19 clusters Model 2 identifies in the global fold.
# 2. Check how many of the unique sequences in the CoV-AbDab external dataset are matched to a Model 2 Covid-19 cluster. Matching requires having an identical V gene, J gene, and CDR3 length, and having a CDR3 region with at least 85% sequence identity.
#
# _This notebook is based on `notebooks/ranks_of_known_binders_vs_healthy_donor_sequences.ipynb`_

# %%

# %%
# We only support split strategies with default target obs column == TargetObsColumnEnum.disease
assert (
    map_cross_validation_split_strategy_to_default_target_obs_column[
        config.cross_validation_split_strategy
    ]
    == TargetObsColumnEnum.disease
)

# %%
def extract_disease_cluster_list(clf: ConvergentClusterClassifier, disease: str):
    # Get all clusters
    model2_cluster_class_associations = (
        clf.cluster_centroids_with_class_specific_p_values
    )

    # Melt to columns = [cluster_dominant_label, p_value], but first move consensus_sequence into the set of index columns
    # TODO(refactor): this reset_index().set_index() operation is quite slow
    model2_cluster_class_associations = (
        model2_cluster_class_associations.reset_index()
        .set_index(
            list(model2_cluster_class_associations.index.names) + ["consensus_sequence"]
        )
        .melt(
            # preserve index
            ignore_index=False,
            var_name="cluster_dominant_label",
            value_name="p_value",
        )
    )

    # Filter to clusters associated with each class
    model2_cluster_class_associations = model2_cluster_class_associations[
        model2_cluster_class_associations["p_value"] <= clf.p_value_threshold
    ]

    # Filter to Covid predictive cluster centroids only
    disease_clusters_from_model2 = model2_cluster_class_associations[
        model2_cluster_class_associations["cluster_dominant_label"] == disease
    ]

    return disease_clusters_from_model2


# %%

# %%
def score_sequences(adata, disease_clusters_from_model2, gene_locus):
    # Assign each test sequence to known cluster with nearest centroid, if possible
    df = ConvergentClusterClassifier._assign_sequences_to_known_clusters(
        df=adata.obs,
        cluster_centroids_by_supergroup=ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            disease_clusters_from_model2
        ),
        sequence_identity_threshold=config.sequence_identity_thresholds.assign_test_sequences_to_clusters[
            gene_locus
        ],
        validate_same_fold_id_and_label=False,
    )

    # Compute (1 - normalized Hamming distance) to get a proximity score.
    # Set to 0 if there were no centroids that could be compared to (i.e. no predictive clusers with same V/J/CDR3 length)
    df["centroid_proximity_score"] = 1 - df["distance_to_nearest_centroid"].fillna(1)

    return df


# %%

# %%

# %%
fold_id = -1
gene_locus = GeneLocus.BCR
target_obs_column = TargetObsColumnEnum.disease
sample_weight_strategy = config.sample_weight_strategy

# %%

# %%
clf_model2 = ConvergentClusterClassifier(
    fold_id=fold_id,
    model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
        gene_locus
    ],
    fold_label_train="train_smaller1",
    gene_locus=gene_locus,
    target_obs_column=target_obs_column,
)
clf_model2

# %%

# %%
disease = "Covid19"
display(Markdown(f"## {gene_locus}, {disease}"))

# Extract list of predictive clusters from model 2
disease_associated_clusters_from_model2 = extract_disease_cluster_list(
    clf=clf_model2, disease=disease
)

# Load this locus's known binders
known_binders_adata = io.load_known_binder_embeddings(
    fold_id=fold_id,
    gene_locus=gene_locus,
    disease=disease,
    known_binder=True,
    sample_weight_strategy=sample_weight_strategy,
)
if gene_locus == GeneLocus.BCR:
    # set
    known_binders_adata.obs["isotype_supergroup"] = "IGHG"
else:
    known_binders_adata.obs["isotype_supergroup"] = "TCRB"


def _score_with_model2(ad: anndata.AnnData) -> Tuple[pd.DataFrame, float]:
    sequence_distances = score_sequences(
        ad, disease_associated_clusters_from_model2, gene_locus
    )

    # this proportion of sequences had no disease-predictive centroids to compare to,
    # and thus have centroid proximity score = 0.0 by default:
    unscored_sequences_proportion = (
        sequence_distances["distance_to_nearest_centroid"]
        .isna()
        .value_counts(normalize=True)
        .reindex([True, False])
        .loc[True]
    )

    return sequence_distances, unscored_sequences_proportion


# Score known binder sequences
(
    known_binder_sequence_distances,
    proportion_of_known_binder_sequences_with_no_disease_associated_centroids_for_comparison,
) = _score_with_model2(known_binders_adata)

# %%
known_binders_adata.shape

# %%
known_binder_sequence_distances.shape

# %%
proportion_of_known_binder_sequences_with_no_disease_associated_centroids_for_comparison

# %%
known_binder_sequence_distances["distance_to_nearest_centroid"].isna().value_counts()

# %%
known_binder_sequence_distances["distance_to_nearest_centroid"].isna().value_counts(
    normalize=True
)

# %%
known_binder_sequence_distances["distance_to_nearest_centroid"].describe()

# %%
known_binder_sequence_distances["distance_to_nearest_centroid"].hist()

# %%
known_binder_sequence_distances["centroid_proximity_score"].isna().value_counts()

# %%
known_binder_sequence_distances["centroid_proximity_score"].describe()

# %%
known_binder_sequence_distances["centroid_proximity_score"].hist()

# %%

# %%
# How many matches:

# %%
(known_binder_sequence_distances["centroid_proximity_score"] >= 0.85).value_counts()

# %%
known_binder_sequence_distances[
    known_binder_sequence_distances["centroid_proximity_score"] >= 0.85
]["v_gene"].value_counts()

# %%
known_binder_sequence_distances[
    known_binder_sequence_distances["centroid_proximity_score"] >= 0.85
][["v_gene", "j_gene", "cdr3_seq_aa_q_trim"]].sort_values(["v_gene", "j_gene"])

# %%

# %%
known_binder_sequence_distances["v_gene"].value_counts()

# %%

# %%

# %%
disease_associated_clusters_from_model2

# %%
# How many Covid-19 clusters are found by Model 2
disease_associated_clusters_from_model2.shape

# %%
disease_associated_clusters_from_model2.index.get_level_values(
    "v_gene"
).remove_unused_categories().value_counts()

# %%

# %%

# %%

# %%

# %%
