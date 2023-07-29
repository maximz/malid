# %% [markdown]
# Plot convergent cluster model (model 2) Covid cluster distances for known binders vs healthy donor-originating sequences.
#
# - Plot separately per fold, to avoid any problems of merging ranks across folds that may have different distance scales.
# - y axis: Proximity to nearest Covid-associated cluster chosen by model 2 -- converted to rank
# - x axis: healthy patient sequences (from one fold's test set), vs CoVAbDab/MIRA known binder sequences
#     - CoV-AbDab: we don't know isotype, so we try all isotypes for each sequences and take max `P(Covid)` prediction.
#     - BCR healthy donors: we exclude IgM/D which may be future Covid response.
#
# Notice that we pass all known binder database entries through our model. Not just "matches" according to some threshold. (Previously, known binder discovery relied on matching our sequences to known binder DBs with fixed thresholds that we chose by hand. Instead, we now run all known binder DB entries through our model wholesale, and compare to the model outputs we get for healthy donor sequences.)

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns
import gc

# %%
from malid import config, io, helpers, logger
from malid.datamodels import GeneLocus, TargetObsColumnEnum, SampleWeightStrategy
from malid.trained_model_wrappers import ConvergentClusterClassifier
import genetools
from statannotations.Annotator import Annotator
import sklearn.metrics
import scipy.stats


# %%

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
def plot(
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    mann_whitney_test_method="asymptotic",
    axis_label="Sequence rank\n(percentile of predictive cluster proximity)",
):
    sample_weight_strategy = SampleWeightStrategy.ISOTYPE_USAGE

    clf = ConvergentClusterClassifier(
        fold_id=fold_id,
        model_name=config.metamodel_base_model_names.model_name_convergent_clustering,
        fold_label_train="train_smaller",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
    )

    # Extract list of predictive clusters from model 2
    covid_clusters_from_model2 = extract_covid_cluster_list(clf)

    # Load this locus's known binders
    known_binders_adata = io.load_known_binder_embeddings(
        fold_id=fold_id,
        gene_locus=gene_locus,
    )
    # Score known binder sequences
    known_binder_sequence_distances = score_sequences(
        known_binders_adata, covid_clusters_from_model2, gene_locus
    )
    logger.info(
        f"Known binder sequences for {gene_locus}: {known_binder_sequence_distances['distance_to_nearest_centroid'].isna().value_counts(normalize=True).reindex([True, False]).loc[True]:.2%} of sequences had no Covid-predictive centroids to compare to in fold {fold_id}, and thus have centroid proximity score = 0.0 by default."
    )

    # Load Mal-ID cohort sequences from the test set, to ensure we have not trained on them
    adata = io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label="test",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )

    # Deduplicate identical sequences across specimens/people
    # (duplicated() returns False for first entry and True for all copies of the same entry, so we have to flip this mask)
    adata = adata[
        ~adata.obs.duplicated(
            subset=[
                "v_gene",
                "j_gene",
                "cdr3_seq_aa_q_trim",
                "disease",
                # Even though the below fields are not used by model2,
                # dedupe by them too,
                # to be consistent with the deduping in the model3 rank analysis.
                "cdr1_seq_aa_q_trim",
                "cdr2_seq_aa_q_trim",
                "isotype_supergroup",
            ],
            keep="first",
        )
    ]

    # Get sequences from healthy individuals
    adata_healthy = adata[adata.obs["disease"] == helpers.healthy_label]
    # Remove naive B cells: those that may be poised to respond to SARS-CoV-2 after infection
    adata_healthy = adata_healthy[adata_healthy.obs["isotype_supergroup"] != "IGHD-M"]

    # Score healthy-donor sequences
    healthy_donor_sequence_distances = score_sequences(
        adata_healthy, covid_clusters_from_model2, gene_locus
    )
    logger.info(
        f"Healthy donor sequences in fold {fold_id}-test, {gene_locus}: {healthy_donor_sequence_distances['distance_to_nearest_centroid'].isna().value_counts(normalize=True).reindex([True, False]).loc[True]:.2%} of sequences had no Covid-predictive centroids to compare to, and thus have centroid proximity score = 0.0 by default."
    )

    # Combine known binders + healthy
    healthy_sequences_label = "Sequences from healthy donors"
    known_binders_label = "Known binders"
    combined = pd.concat(
        [
            healthy_donor_sequence_distances[["centroid_proximity_score"]].assign(
                source=healthy_sequences_label
            ),
            known_binder_sequence_distances[["centroid_proximity_score"]].assign(
                source=known_binders_label
            ),
        ],
        axis=0,
    ).reset_index(drop=True)

    # Assign a rank (higher ranks are higher proximity scores)
    combined.sort_values("centroid_proximity_score", ascending=True, inplace=True)

    # Note that there may be a number of sequences with centroid_proximity_score exactly 0 (all sequences that had no possible predictive clusters to match to, because of their V/J/CDR3 length),
    # but they may have different ranks if we use rank_normalize directly, since in that function we are not repeating ranks (all ranks are unique):
    #         combined["rank"] = genetools.stats.rank_normalize(
    #             combined["centroid_proximity_score"]
    #         )
    #         # percentile normalize
    #         combined["rank"] = combined["rank"] / combined.shape[0]

    # Instead, assign the same rank for all sequences with the proximity score = 0.0 by default.
    # First set ranks for all nonzero scores, then fillna for the rest with rank exactly 0.
    # or, set ranks for all nonzero scores, and then fillna for the rest with rank exactly 0
    nonzero_scores = combined[combined["centroid_proximity_score"] > 0.0][
        "centroid_proximity_score"
    ]
    combined["rank"] = genetools.stats.percentile_normalize(nonzero_scores).reindex(
        combined.index, fill_value=0.0
    )

    # Compute AUC of discovering known binders with our rankings
    # (Doesn't matter whether we use rank or the raw distance (e.g. "centroid_proximity_score" column) here, because AUC is just about whether positive examples are assigned higher ranks than negative examples)
    auc = sklearn.metrics.roc_auc_score(
        y_true=combined["source"].replace(
            {healthy_sequences_label: False, known_binders_label: True}
        ),
        y_score=combined["rank"],
    )

    # Compute rank stats
    known_binder_sequence_ranks = combined[combined["source"] == known_binders_label][
        "rank"
    ]
    rank_stats = [
        f"85% of known binder sequences have rank over {known_binder_sequence_ranks.quantile(0.15)*100:0.1f}%"
    ]
    # Invert, either by computing CDF (https://stackoverflow.com/q/26489134/130164) or as follows:
    rank_stats.extend(
        [
            f"{(known_binder_sequence_ranks > 0.80).mean() * 100:0.1f}% of known binder sequences have rank over 80%",
            f"{(known_binder_sequence_ranks > 0.75).mean() * 100:0.1f}% of known binder sequences have rank over 75%",
            f"{(known_binder_sequence_ranks > 0.50).mean() * 100:0.1f}% of known binder sequences have rank over 50%",
        ]
    )

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    order = [healthy_sequences_label, known_binders_label]
    sns.boxplot(data=combined, x="source", y="rank", ax=ax, order=order)
    # Annotate with statistical significance
    annot = Annotator(
        ax=ax,
        pairs=[order],
        data=combined,
        x="source",
        y="rank",
        # This "order" is the reverse of the seaborn plotting order:
        # Specify pair order for the one-sided test that the the known binders have greater ranks than the healthy sequences.
        # (Make sure to be consistent with the scipy equivalent below about which is the "greater" and which is the "less" sample)
        order=list(reversed(order)),
    )
    annot.configure(
        test="Mann-Whitney-gt", text_format="star", loc="outside", verbose=2
    )
    annot.apply_test(method=mann_whitney_test_method)
    ax, test_results_for_annotator = annot.annotate()

    # Reproduce the test ourselves: Wilcoxon rank-sum test, one sided.
    ranks_known_binders = combined[combined["source"] == known_binders_label][
        "rank"
    ].values
    ranks_other = combined[combined["source"] == healthy_sequences_label]["rank"].values

    # The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution.
    # The alternative hypothesis is that values in one sample are more likely to be greater than the values in the other sample.
    significance_test = scipy.stats.mannwhitneyu(
        ranks_known_binders,
        ranks_other,
        alternative="greater",
        method=mann_whitney_test_method,
    )
    # Confirm StatResult matches against scipy's p-value and test statistic
    assert np.allclose(
        significance_test.pvalue, test_results_for_annotator[0].data.pvalue
    )
    assert np.allclose(
        significance_test.statistic, test_results_for_annotator[0].data.stat_value
    )

    # Finish plot
    plt.ylabel(axis_label)
    plt.xlabel(None)
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            ax.get_xticklabels(), combined, "source"
        )
    )
    genetools.plots.wrap_tick_labels(ax)
    plt.ylim(-0.05, 1.05)
    # higher rank means higher confidence
    plt.yticks(
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        labels=[
            "0 (worst)",
            0.25,
            0.50,
            0.75,
            "1 (best)",
        ],
    )

    sns.despine(ax=ax)
    fig.tight_layout()

    return fig, ax, combined, significance_test, auc, rank_stats


# %%
for gene_locus in config.gene_loci_used:
    output_dir = config.paths.model_interpretations_output_dir / gene_locus.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop over all fold IDs except global fold (does not have a test set)
    # (Global fold's validation set is not appropriate: not really held out, because used for model 2 threshold picking and for fine-tuning the language models)
    for fold_id in config.cross_validation_fold_ids:
        fig, ax, df, boxplot_significance_test, auc, rank_stats = plot(
            fold_id=fold_id,
            gene_locus=gene_locus,
            target_obs_column=TargetObsColumnEnum.disease,
        )

        # Make report
        newline = "\n"
        report = f"""AUC of discovering known binders with our rankings based on sequence proximity to nearest model 2 Covid19-predictive cluster centroid,
without a distance threshold but with same V gene, J gene, and CDR3 length: {auc:0.3f}
{newline.join(rank_stats)}

Boxplot p-value: {boxplot_significance_test.pvalue:.2e}, U-statistic={boxplot_significance_test.statistic:.4e}

NOTE: {(df['centroid_proximity_score'] == 0.0).value_counts(normalize=True).reindex([True, False]).loc[True]:0.2%} of sequences had infinite distance from / were unmatchable to any model 2 centroid,
i.e. no compatible V/J/CDR3 length clusters to compute distance from, so these sequences had score and rank set to 0.

The breakdown of infinite-distances by sequence source:
"""
        for key, grp in df.groupby("source"):
            report += f"{key}:\t{(grp['centroid_proximity_score'] == 0.0).value_counts(normalize=True).reindex([True, False]).loc[True]:0.2%} of sequences have infinite distance"
            report += "\n"
        print(f"{gene_locus}, fold {fold_id}")
        print(report)

        # Export
        genetools.plots.savefig(
            fig,
            output_dir
            / f"known_binders_vs_healthy_controls.model2_rank_boxplot.fold_{fold_id}.png",
            dpi=300,
        )
        with open(
            output_dir
            / f"known_binders_vs_healthy_controls.model2_rank_report.fold_{fold_id}.txt",
            "w",
        ) as f:
            f.write(report)
        plt.close(fig)

        # clear cache
        io.clear_cached_fold_embeddings()
        gc.collect()

        print()
        print("*" * 60)
        print()

# %%

# %%
