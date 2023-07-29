# %% [markdown]
# Questions:
#
# - How many sequences chosen by ExactMatchesClassifier match the known binders database? (scalar value, for each fold and locus)
# - How many sequences chosen by ExactMatchesClassifier match model2's chosen centroids? (scalar value, for each fold and locus)

# %%

# %%
import numpy as np
import pandas as pd

# %%
from malid import config, interpretation
from malid.datamodels import GeneLocus, TargetObsColumnEnum
from malid.trained_model_wrappers import (
    ConvergentClusterClassifier,
    ExactMatchesClassifier,
)


# %%

# %%
def extract_covid_sequence_list(clf: ExactMatchesClassifier):
    p_value = clf.p_value_threshold
    seqs = clf.sequences_with_fisher_result
    return (
        clf_exact_matches.sequences_with_fisher_result["Covid19"][
            clf_exact_matches.sequences_with_fisher_result["Covid19"]
            <= clf_exact_matches.p_value_threshold
        ]
        .index.to_frame()
        .reset_index(drop=True)
    )


# %%
def extract_covid_cluster_list(clf: ConvergentClusterClassifier):
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
        model2_cluster_class_associations["cluster_dominant_label"] == "Covid19"
    ]

    return disease_clusters_from_model2


# %%
def score_sequences(sequences, cluster_centroids_by_supergroup, gene_locus):
    # Assign each test sequence to known cluster with nearest centroid, if possible
    return ConvergentClusterClassifier._assign_sequences_to_known_clusters(
        df=sequences,
        cluster_centroids_by_supergroup=cluster_centroids_by_supergroup,
        sequence_identity_threshold=config.sequence_identity_thresholds.assign_test_sequences_to_clusters[
            gene_locus
        ],
        validate_same_fold_id_and_label=False,
    )


# %%

# %%

# %%
ConvergentClusterClassifier_modelname = (
    config.metamodel_base_model_names.model_name_convergent_clustering
)
ExactMatchesClassifier_modelname = ConvergentClusterClassifier_modelname
target_obs_column: TargetObsColumnEnum = TargetObsColumnEnum.disease

ConvergentClusterClassifier_modelname

# %%

# %%
for gene_locus in config.gene_loci_used:
    # Load known binders
    (
        known_binders_df,
        known_binder_cluster_centroids_by_supergroup,
    ) = interpretation.load_reference_dataset(gene_locus)

    for fold_id in config.all_fold_ids:
        # Load models
        clf_exact_matches = ExactMatchesClassifier(
            fold_id=fold_id,
            model_name=ExactMatchesClassifier_modelname,
            fold_label_train="train_smaller",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )
        clf_model2 = ConvergentClusterClassifier(
            fold_id=fold_id,
            model_name=ConvergentClusterClassifier_modelname,
            fold_label_train="train_smaller",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )

        # Extract list of predictive sequences from ExactMatchesClassifier
        covid_sequences_from_exact_matches_model = extract_covid_sequence_list(
            clf_exact_matches
        )

        # Extract list of predictive clusters from model 2
        covid_clusters_from_model2 = extract_covid_cluster_list(clf_model2)

        # Assign Covid19-predictive sequences from ExactMatchesClassifier to known binder DB entries (clusters of near-dupe known binder sequences)
        exactmatches_sequences_matched_to_known_binders = score_sequences(
            covid_sequences_from_exact_matches_model,
            known_binder_cluster_centroids_by_supergroup,
            gene_locus,
        )

        # Assign Covid19-predictive sequences from ExactMatchesClassifier to Covid19-predictive clusters from ConvergentClusterClassifier
        exactmatches_sequences_matched_to_model2_clusters = score_sequences(
            covid_sequences_from_exact_matches_model,
            ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
                covid_clusters_from_model2
            ),
            gene_locus,
        )

        print(f"{gene_locus}, fold {fold_id}:")
        print("=" * 30)

        exactmatches_sequences_matched_to_known_binders_summary = (
            exactmatches_sequences_matched_to_known_binders[
                "cluster_id_within_clustering_group"
            ]
            .isna()
            .replace({True: "Unmatched", False: "Matched"})
        )
        print(
            "How many Covid19-predictive sequences chosen by ExactMatchesClassifier match the known binders database? "
            + f"{exactmatches_sequences_matched_to_known_binders_summary.value_counts(normalize=True).reindex(['Unmatched', 'Matched'], fill_value=0).loc['Matched']:0.2%}"
        )
        print(exactmatches_sequences_matched_to_known_binders_summary.value_counts())
        print()

        exactmatches_sequences_matched_to_model2_clusters_summary = (
            exactmatches_sequences_matched_to_model2_clusters[
                "cluster_id_within_clustering_group"
            ]
            .isna()
            .replace({True: "Unmatched", False: "Matched"})
        )
        print(
            "How many Covid19-predictive sequences chosen by ExactMatchesClassifier match Covid19-predictive clusters chosen by ConvergentClusterClassifier? "
            + f"{exactmatches_sequences_matched_to_model2_clusters_summary.value_counts(normalize=True).reindex(['Unmatched', 'Matched'], fill_value=0).loc['Matched']:0.2%}"
        )
        print(exactmatches_sequences_matched_to_model2_clusters_summary.value_counts())
        print()

        print("*" * 60)
        print()

# %%

# %%

# %%

# %%

# %%

# %%
