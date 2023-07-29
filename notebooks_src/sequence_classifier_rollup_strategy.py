# -*- coding: utf-8 -*-
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from malid import io, config, helpers
from malid.datamodels import GeneLocus
from malid.datamodels import TargetObsColumnEnum
from malid.trained_model_wrappers.rollup_sequence_classifier import (
    RollupSequenceClassifier,
)
from malid.datamodels import GeneLocus
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
)
import anndata
import genetools
from sklearn.metrics import classification_report
from malid.external.model_evaluation_scores import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
import scipy.stats
from slugify import slugify

pd.set_option("display.max_columns", 200)

# %%

# %%
# utility functions
def load_val_embeddings(fold_id: int, gene_locus: GeneLocus):
    return io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label="validation",
        gene_locus=gene_locus,
        target_obs_column=TargetObsColumnEnum.disease,
    )


def build_classifier(fold_id: int, gene_locus: GeneLocus) -> RollupSequenceClassifier:
    return RollupSequenceClassifier(
        fold_id=fold_id,
        model_name_sequence_disease="lasso_multiclass",
        fold_label_train="train_smaller",  # what base models were trained on
        gene_locus=gene_locus,
        target_obs_column=TargetObsColumnEnum.disease,
        sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
    )


def evaluate(
    rollup_clf: RollupSequenceClassifier,
    proportiontocut: float,
    validation_embeddings: anndata.AnnData,
    strategy: str,  # e.g. "trimmed_mean"
    fold_id: int,
    gene_locus: GeneLocus,
) -> pd.Series:
    validation_preds = rollup_clf.predict_proba(
        validation_embeddings, proportiontocut=proportiontocut, strategy=strategy
    )
    classes = rollup_clf.classes_
    validation_preds["predicted"] = validation_preds[classes].idxmax(axis=1)
    validation_preds["true_label"] = (
        validation_embeddings.obs[["specimen_label", "disease"]]
        .drop_duplicates()
        .set_index("specimen_label")["disease"]
    )
    labels = validation_preds["true_label"].drop_duplicates().sort_values()
    report = precision_recall_fscore_support(
        validation_preds["true_label"], validation_preds["predicted"], labels=labels
    )
    report = pd.DataFrame(
        report, columns=labels, index=["precision", "recall", "fscore", "support"]
    )
    report = (
        report.loc[["precision", "recall"]].unstack().to_frame().sort_index(level=1).T
    )
    report.columns = ["_".join(col).strip() for col in report.columns.values]
    report = report.iloc[0]
    roc_auc = roc_auc_score(
        validation_preds["true_label"],
        validation_preds[classes],
        multi_class="ovo",
        average="weighted",
        labels=classes,
    )
    report["roc_auc"] = roc_auc
    report["proportiontocut"] = proportiontocut
    report["strategy"] = strategy
    report["fold_id"] = fold_id
    report["gene_locus"] = gene_locus.name

    print("*" * 80)
    print(f"gene_locus: {gene_locus}")
    print(f"proportiontocut: {proportiontocut}")
    print(f"strategy: {strategy}")
    print(f"fold_id: {fold_id}")
    print(f"roc_auc: {roc_auc}")
    print(
        classification_report(
            validation_preds["true_label"], validation_preds["predicted"]
        )
    )
    return report


# %%

# %% [markdown]
# # Visualize data from a single fold (fold 1, BCR)

# %%
validation_embeddings = load_val_embeddings(fold_id=1, gene_locus=GeneLocus.BCR)
validation_embeddings

# %%
rollup_clf = build_classifier(fold_id=1, gene_locus=GeneLocus.BCR)
rollup_clf

# %%


# %% [markdown]
# ## How many sequences exist per sample?
# We can see that the size distribution of sequences is skewed

# %%
specimen_disease = (
    validation_embeddings.obs[["specimen_label", "disease"]]
    .drop_duplicates()
    .set_index("specimen_label")
)
specimen_disease["n_sequences"] = validation_embeddings.obs[
    "specimen_label"
].value_counts()

plt.figure()
sns.displot(
    x="n_sequences",
    hue="disease",
    multiple="stack",
    data=specimen_disease,
    palette="Paired",
)
plt.legend(bbox_to_anchor=(1, 1))
plt.ylabel("count specimens")

plt.figure()
sns.displot(
    x="n_sequences",
    hue="disease",
    multiple="stack",
    data=specimen_disease,
    palette="Paired",
    log_scale=True,
)
plt.legend(bbox_to_anchor=(1, 1))
plt.ylabel("count specimens")


# %%

# %% [markdown]
# ## Plot raw sequence probabilities
#
# Sequence model predicted class probabilities are enriched for sequences from individuals of each disease class. Sequences predicted to have higher disease association are enriched for sequences truly from patients with that disease. The healthy class exhibits a different pattern: sequences with higher “healthy or background” probability (higher x-value) may be healthy sequences within the immune repertoires of disease patients.

# %%
validation_seq_preds = pd.DataFrame(
    rollup_clf.clf_sequence_disease.predict_proba(
        rollup_clf.clf_sequence_disease.featurize(validation_embeddings).X,
    ),
    index=validation_embeddings.obs.specimen_label,
    columns=rollup_clf.clf_sequence_disease.classes_,
)

validation_seq_preds["true_label"] = (
    validation_embeddings.obs[["specimen_label", "disease"]]
    .drop_duplicates()
    .set_index("specimen_label")["disease"]
)

# %%
n = 50000
classes = rollup_clf.clf_sequence_disease.classes_
for i in classes:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # sample n probabilities per disease
    subset = (
        validation_seq_preds.loc[validation_seq_preds.true_label == i]
        .sample(n)
        .reset_index()
    )
    subset["entropy"] = subset[classes].apply(
        lambda row: scipy.stats.entropy(row), axis=1
    )
    sns.histplot(x="entropy", y=i, data=subset, ax=ax[0])

    subset = (
        validation_seq_preds.loc[validation_seq_preds.true_label != i]
        .sample(n)
        .reset_index()
    )
    subset["entropy"] = subset[classes].apply(
        lambda row: scipy.stats.entropy(row), axis=1
    )
    sns.histplot(x="entropy", y=i, data=subset, ax=ax[1])
    ax[0].set_title(f"{i}")
    ax[1].set_title(f"not {i}")


# %%
# weighted sample of sequence level predictions
data = (
    validation_seq_preds.groupby("true_label", observed=True)
    .apply(lambda df: df.sample(100000))
    .reset_index(drop=True)
)
classes = list(rollup_clf.clf_sequence_disease.classes_)
print(classes)


for col in classes:
    fig, ax = plt.subplots(2, 1, sharex=True)

    hue_order = classes.copy()
    hue_order.remove(col)
    hue_order = hue_order + [col]

    sns.histplot(
        x=col,
        hue="true_label",
        data=data,
        multiple="stack",
        ax=ax[0],
        palette=helpers.disease_color_palette,
        hue_order=hue_order,
    )
    sns.histplot(
        x=col,
        hue="true_label",
        data=data,
        multiple="fill",
        ax=ax[1],
        legend=False,
        palette=helpers.disease_color_palette,
        hue_order=hue_order,
    )
    ax[0].get_legend().set_title("True Label")
    sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))
    fig.suptitle(f"{col}")
    ax[1].set_xlabel(f"{col} sequence probability")
    ax[0].set_ylabel("Actual count")
    ax[1].set_ylabel("Normalized count")
    genetools.plots.savefig(
        fig,
        config.paths.output_dir
        / f"sequence_aggregation_probabilties.{slugify(col)}.png",
        dpi=300,
    )


# %%
# to do :
# plot this for TCRs too
# make a true multi-panel figure

# %%

# %% [markdown]
# # Compare rollup strategies (on validation sets from different folds and loci)

# %%
# to do:
# - try inverse-weighted entropy for sequence level prediction

# %%
experiment_summary = []

for gene_locus in config.gene_loci_used:
    print(gene_locus)
    for fold_id in config.cross_validation_fold_ids:
        validation_embeddings = load_val_embeddings(fold_id, gene_locus)
        rollup_clf = build_classifier(fold_id, gene_locus)

        # run weighted_median strategy (no threshold required)
        experiment_summary.append(
            evaluate(
                rollup_clf=rollup_clf,
                proportiontocut=0,
                validation_embeddings=validation_embeddings,
                strategy="weighted_median",
                fold_id=fold_id,
                gene_locus=gene_locus,
            )
        )

        # run trimmed mean and trim bottom only strategies:
        # loop over possible thresholds for cutting
        for proportiontocut in [0, 0.025, 0.05, 0.1]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # run trimmed_mean rollup strategy
                experiment_summary.append(
                    evaluate(
                        rollup_clf=rollup_clf,
                        proportiontocut=proportiontocut,
                        validation_embeddings=validation_embeddings,
                        strategy="trimmed_mean",
                        fold_id=fold_id,
                        gene_locus=gene_locus,
                    )
                )
                # run trim_bottom_only rollup strategy
                experiment_summary.append(
                    evaluate(
                        rollup_clf=rollup_clf,
                        proportiontocut=proportiontocut,
                        validation_embeddings=validation_embeddings,
                        strategy="trim_bottom_only",
                        fold_id=fold_id,
                        gene_locus=gene_locus,
                    )
                )
        # run entropy threshold strategy, loop over different entropy thresholds

        # Above a certain entropy, the entropy threshold applies no filtering, acting as an untrimmed mean
        # That maximum entropy threshold is the entropy of the n-dim vector of all [1/n] entries
        n_classes = len(rollup_clf.classes_)
        maximum_entropy_cutoff = scipy.stats.entropy(np.ones(n_classes) / n_classes)

        for entropy_threshold in [1.6, 1.5, 1.4, 1.3, 1.2]:
            if entropy_threshold >= maximum_entropy_cutoff:
                # Skip if above the maximum entropy threshold
                continue

            experiment_summary.append(
                evaluate(
                    rollup_clf=rollup_clf,
                    proportiontocut=entropy_threshold,
                    validation_embeddings=validation_embeddings,
                    strategy="entropy_threshold",
                    fold_id=fold_id,
                    gene_locus=gene_locus,
                )
            )


# %%
summary_df = (
    pd.concat(experiment_summary, axis=1)
    .T.sort_values("roc_auc", ascending=False)
    .reset_index(drop=True)
)
summary_df.to_csv(
    config.paths.output_dir / "sequence_classifier_rollup_strategy_comparisons.tsv",
    sep="\t",
    index=None,
)
summary_df

# %%

# %%
# Reload from disk - can restart here
summary_df = pd.read_csv(
    config.paths.output_dir / "sequence_classifier_rollup_strategy_comparisons.tsv",
    sep="\t",
)
summary_df

# %%
# Rename for clarity: 0% trimming should be called "untrimmed mean"
summary_df.loc[
    summary_df["strategy"].isin(["trim_bottom_only", "trimmed_mean"])
    & (summary_df["proportiontocut"] == 0),
    "strategy",
] = "untrimmed_mean"


# %%
# sns.boxplot(x='strategy',y='roc_auc', hue='proportiontocut', data=summary_df.loc[summary_df.strategy.isin(['entropy_threshold'])
# plt.legend(bbox_to_anchor=(1,1))
# plt.xticks(rotation=45, ha='right')

# %%
# to do: show proportiontocut range included in each strategy
summary_df_agg = (
    summary_df.groupby(["gene_locus", "strategy"], observed=True)
    .agg({"roc_auc": ["min", "mean", "max", "var"]})
    .round(3)
)
summary_df_agg.to_csv(
    config.paths.output_dir
    / "sequence_classifier_rollup_strategy_comparisons.aggregated.tsv",
    sep="\t",
)
summary_df_agg

# %%
summary_df_agg_simpler = (
    summary_df.groupby(["strategy", "gene_locus"], observed=True)["roc_auc"]
    .agg(["mean", "std"])
    .apply(lambda row: f"{row['mean']:0.3f} +/- {row['std']:0.3f}", axis=1)
    .unstack()
    .sort_values(["TCR", "BCR"], ascending=False)
)
summary_df_agg_simpler.to_csv(
    config.paths.output_dir
    / "sequence_classifier_rollup_strategy_comparisons.aggregated_simpler.tsv",
    sep="\t",
)
summary_df_agg_simpler

# %%
# what's in each strategy?
summary_df.groupby("strategy")["proportiontocut"].unique()

# %%

# %%
plt.figure(figsize=(10, 10))
g = sns.FacetGrid(summary_df, col="gene_locus", row="fold_id", height=3.5)
g.map_dataframe(sns.scatterplot, x="strategy", y="roc_auc", hue="proportiontocut")
g.add_legend(title="proportiontocut")

# %%
# TODO: just use summary_df.loc[summary_df.groupby(["gene_locus", "strategy"], observed=True)["roc_auc"].idxmax()] ?
idx = (
    summary_df.groupby(["gene_locus", "strategy"], observed=True)["roc_auc"].transform(
        max
    )
    == summary_df["roc_auc"]
)
summary_df.loc[idx]


# %%
summary_df.groupby(["gene_locus", "strategy", "proportiontocut"], observed=True).agg(
    {"roc_auc": ["mean", "var", "min", "max"]}
)

# %%
pd.set_option("display.max_rows", 300)
summary_df.sort_values("roc_auc", ascending=False)


# %%

# %%

# %%
