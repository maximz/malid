# %% [markdown]
# Plot sequence model (model 3) `P(Covid)` rankings for known binders vs healthy donor-originating sequences.
#
# - Plot separately per fold, to avoid any problems of merging ranks across folds that may have different probability scales.
# - y axis: P(covid), converted to rank
# - x axis: healthy patient sequences (from one fold's test set), vs CoVAbDab/MIRA known binder sequences
#     - CoV-AbDab: we don't know isotype, so we try all isotypes for each sequences and take max `P(Covid)` prediction. (Model 3 uses isotype info)
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
from malid.trained_model_wrappers import SequenceClassifier
import genetools
from statannotations.Annotator import Annotator
import sklearn.metrics
import scipy.stats


# %%

# %%

# %%

# %%

# %%
def plot(
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    mann_whitney_test_method="asymptotic",
    axis_label="Sequence prediction confidence\n(percentile of rank)",
):
    sample_weight_strategy = SampleWeightStrategy.ISOTYPE_USAGE

    clf = SequenceClassifier(
        fold_id=fold_id,
        model_name_sequence_disease="lasso_multiclass",
        fold_label_train="train_smaller",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )

    # Load this locus's known binders, embedded for this fold
    known_binders_adata = io.load_known_binder_embeddings(
        fold_id=fold_id,
        gene_locus=gene_locus,
        sample_weight_strategy=sample_weight_strategy,
    )

    # For BCR known binders, we don't know true isotype label
    # Try them all and choose the most Covid-like
    known_binder_probabilities_with_different_isotypes = []
    for possible_isotype in helpers.isotype_groups_kept[gene_locus]:
        known_binders_adata.obs["isotype_supergroup"] = possible_isotype
        featurized_known_binders = clf.featurize(known_binders_adata)
        known_binders_predicted_probabilities = pd.DataFrame(
            clf.predict_proba(featurized_known_binders.X),
            index=featurized_known_binders.sample_names,
            columns=clf.classes_,
        )["Covid19"].rename(f"Covid19_{possible_isotype}")
        known_binder_probabilities_with_different_isotypes.append(
            known_binders_predicted_probabilities
        )
    known_binder_probabilities_with_different_isotypes = pd.concat(
        known_binder_probabilities_with_different_isotypes, axis=1
    )
    known_binders_predicted_probabilities = (
        known_binder_probabilities_with_different_isotypes.max(axis=1).rename("Covid19")
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
                "cdr1_seq_aa_q_trim",
                "cdr2_seq_aa_q_trim",
                "cdr3_seq_aa_q_trim",
                "isotype_supergroup",
                "disease",
            ],
            keep="first",
        )
    ]

    # Get sequences from healthy individuals
    adata_healthy = adata[adata.obs["disease"] == helpers.healthy_label]
    # Remove naive B cells: those that may be poised to respond to SARS-CoV-2 after infection
    adata_healthy = adata_healthy[adata_healthy.obs["isotype_supergroup"] != "IGHD-M"]

    # Score healthy-donor sequences for P(Covid19)
    featurized_healthy = clf.featurize(adata_healthy)
    healthy_predicted_probabilities = pd.DataFrame(
        clf.predict_proba(featurized_healthy.X),
        index=featurized_healthy.sample_names,
        columns=clf.classes_,
    )["Covid19"]

    # Combine known binders + healthy
    healthy_sequences_label = "Sequences from healthy donors"
    known_binders_label = "Known binders"
    combined = pd.concat(
        [
            healthy_predicted_probabilities.to_frame().assign(
                source=healthy_sequences_label
            ),
            known_binders_predicted_probabilities.to_frame().assign(
                source=known_binders_label
            ),
        ],
        axis=0,
    ).reset_index(drop=True)

    # Assign a rank (higher ranks are higher probabilities)
    combined.sort_values("Covid19", ascending=True, inplace=True)
    combined["rank"] = genetools.stats.rank_normalize(combined["Covid19"])
    # percentile normalize
    combined["rank"] = combined["rank"] / combined.shape[0]

    # Compute AUC of discovering known binders with our rankings
    # (Doesn't matter whether we use rank or the raw probability (e.g. "Covid19" column) here, because AUC is just about whether positive examples are assigned higher ranks than negative examples)
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
        rank_stats = "\n".join(rank_stats)
        report = f"""AUC of discovering known binders with our rankings: {auc:0.3f}
{rank_stats}

Boxplot p-value: {boxplot_significance_test.pvalue:.2e}, U-statistic={boxplot_significance_test.statistic:.4e}
"""
        print(f"{gene_locus}, fold {fold_id}")
        print(report)

        # Export
        genetools.plots.savefig(
            fig,
            output_dir
            / f"known_binders_vs_healthy_controls.model3_rank_boxplot.fold_{fold_id}.png",
            dpi=300,
        )
        with open(
            output_dir
            / f"known_binders_vs_healthy_controls.model3_rank_report.fold_{fold_id}.txt",
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
