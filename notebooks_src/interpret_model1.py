# %%

# %% [markdown]
# # Can we read model 1 V-J gene use count matrix PCs to see which V genes matter?

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %matplotlib inline
import seaborn as sns
import joblib
import gc
import glob
from pathlib import Path
from matplotlib.ticker import MultipleLocator, PercentFormatter
import genetools
import shap

from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    map_cross_validation_split_strategy_to_default_target_obs_column,
)
from malid.trained_model_wrappers import RepertoireClassifier
from malid import config, helpers, io


# %%

# %%
def interpret(
    gene_locus: GeneLocus, target_obs_column: TargetObsColumnEnum, fold_id: int
):
    clf_rep = RepertoireClassifier(
        fold_id=fold_id,
        model_name="elasticnet_cv",
        fold_label_train="train_smaller",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=config.sample_weight_strategy,
    )
    train_count_matrix_columns = joblib.load(
        clf_rep.models_base_dir
        / f"{clf_rep.fold_label_train}_model.{clf_rep.fold_id}.{clf_rep.fold_label_train}.specimen_vj_gene_counts_columns_joblib"
    )

    isotypes = helpers.isotype_groups_kept[gene_locus]
    # # sanity check:
    # isotypes, clf_rep.steps[0][
    #     1
    # ].named_transformers_.keys(), train_count_matrix_columns.keys()

    for isotype in isotypes:
        print(isotype)
        pca_transformer = (
            clf_rep.steps[0][1]
            .named_transformers_[f"log1p-scale-PCA_{isotype}"]
            .steps[-1][1]
        )

        # PCs x VJ pairs
        components_df = pd.DataFrame(
            pca_transformer.components_, columns=train_count_matrix_columns[isotype]
        )
        # display(components_df)

        # most important features for first PC component
        n_top = 10
        # display(components_df.iloc[0].abs().sort_values(ascending=False).head(n=n_top))

        # V genes in there
        print(
            components_df.iloc[0]
            .abs()
            .sort_values(ascending=False)
            .head(n=n_top)
            .index.str.split("|")
            .str[0]
            .unique()
        )

        # second PC coponent, same thing:
        print(
            components_df.iloc[1]
            .abs()
            .sort_values(ascending=False)
            .head(n=n_top)
            .index.str.split("|")
            .str[0]
            .unique()
        )

        print()
        print("*" * 60)
        print()

    adata = io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label="train_smaller",
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        # Model 1 does not require the embedding .X, so take the fast path and just load .obs:
        load_obs_only=True,
    )
    featurized = clf_rep.featurize(adata)

    # or use clf_rep._inner[:-1] if we want scaling added:
    pca_transformer = clf_rep.steps[0][1]
    transformed = pd.DataFrame(
        pca_transformer.transform(featurized.X),
        index=featurized.X.index,
        columns=pca_transformer.get_feature_names_out(),
    )
    # Show model 1's top 2 PCs of V/J gene use counts for train fold specimens
    for isotype in helpers.isotype_groups_kept[gene_locus]:
        plot_df = pd.concat(
            [
                transformed[
                    [
                        f"log1p-scale-PCA_{isotype}__pca0",
                        f"log1p-scale-PCA_{isotype}__pca1",
                    ]
                ].rename(
                    columns={
                        f"log1p-scale-PCA_{isotype}__pca0": "PC1",
                        f"log1p-scale-PCA_{isotype}__pca1": "PC2",
                    }
                ),
                featurized.metadata[["disease", "study_name"]],
            ],
            axis=1,
        )
        plot_df["Disease and batch"] = (
            plot_df["disease"].astype(str) + " - " + plot_df["study_name"].astype(str)
        )
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=plot_df, x="PC1", y="PC2", hue="Disease and batch", ax=ax, alpha=0.7
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        sns.despine(ax=ax)
        plt.title(f"{isotype} V-J gene use count PCA")


# %%

# %%
# Choose fold:
# Prefer global fold (fold -1), unless the cross validation split strategy is restricted to a single fold.
fold_id = 0 if config.cross_validation_split_strategy.value.is_single_fold_only else -1


# Choose classification target:
target_obs_column = map_cross_validation_split_strategy_to_default_target_obs_column[
    config.cross_validation_split_strategy
]

for gene_locus in config.gene_loci_used:
    print(
        f"{gene_locus}, {target_obs_column}, fold {fold_id} ({config.cross_validation_split_strategy})"
    )
    interpret(
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        fold_id=fold_id,
    )
    io.clear_cached_fold_embeddings()
    gc.collect()
    print()

# %%

# %%

# %% [markdown]
# # Another way to look at top signals

# %%

# %%
fold_id = 1
fold_label = "test"

# %%
adata_bcr = io.load_fold_embeddings(
    fold_id=fold_id,
    fold_label=fold_label,
    gene_locus=GeneLocus.BCR,
    target_obs_column=TargetObsColumnEnum.disease,
    load_obs_only=True,
)
adata_bcr

# %%
clf_bcr = RepertoireClassifier(
    fold_id=fold_id,
    model_name=config.metamodel_base_model_names.model_name_overall_repertoire_composition[
        GeneLocus.BCR
    ],
    fold_label_train="train_smaller",
    gene_locus=GeneLocus.BCR,
    target_obs_column=TargetObsColumnEnum.disease,
)
clf_bcr

# %%
clf_bcr.model_name

# %%

# %%
featurized_data_bcr = clf_bcr.featurize(adata_bcr)
featurized_data_bcr.X

# %%

# %%
transformed_X_bcr = clf_bcr._inner[:-1].transform(featurized_data_bcr.X)
transformed_X_bcr

# %%
featurized_data_bcr.X.shape, transformed_X_bcr.shape

# %%
transformed_X_bcr = pd.DataFrame(
    transformed_X_bcr,
    index=featurized_data_bcr.X.index,
    columns=clf_bcr.named_steps["columntransformer"].get_feature_names_out(),
)

# %%
transformed_X_bcr

# %%

# %%

# %%
explainer_bcr = shap.LinearExplainer(
    (clf_bcr.steps[-1][1].coef_, clf_bcr.steps[-1][1].intercept_),
    transformed_X_bcr,
)

# %%
shap_values_bcr = explainer_bcr.shap_values(X=transformed_X_bcr)
len(shap_values_bcr)

# %%
shap_values_bcr[0].shape

# %%
transformed_X_bcr.shape, clf_bcr.classes_.shape

# %%
clf_bcr.classes_

# %%

# %%
# is featurized_data_bcr.X scaled to sum to 1 per isotype? yes:
assert np.allclose(
    featurized_data_bcr.X[
        featurized_data_bcr.X.columns[
            featurized_data_bcr.X.columns.str.startswith("IGHD-M:pca")
        ]
    ].sum(axis=1),
    1,
)
assert np.allclose(
    featurized_data_bcr.X[
        featurized_data_bcr.X.columns[
            featurized_data_bcr.X.columns.str.startswith("IGHG:pca")
        ]
    ].sum(axis=1),
    1,
)
assert np.allclose(
    featurized_data_bcr.X[
        featurized_data_bcr.X.columns[
            featurized_data_bcr.X.columns.str.startswith("IGHA:pca")
        ]
    ].sum(axis=1),
    1,
)

# %%

# %%
# Extract T1D class coefficients
class_id = np.where(clf_bcr.classes_ == "T1D")[0][0]
coefs_t1d = pd.Series(
    clf_bcr.steps[-1][1].coef_[class_id], index=transformed_X_bcr.columns
).sort_values()
coefs_t1d

# %%
# Find a top T1D feature based on average SHAP values over positive test-set items
class_id = np.where(clf_bcr.classes_ == "T1D")[0][0]
shap_sorted = (
    pd.DataFrame(
        shap_values_bcr[class_id],
        index=featurized_data_bcr.sample_names,
        columns=transformed_X_bcr.columns,
    )
    .loc[featurized_data_bcr.y == "T1D"]
    .mean(axis=0)
    .sort_values()
)
shap_sorted

# %%
# Get top feature
feat = shap_sorted.index[-1]
feat

# %%
# Get loadings: how much each original count feature contributes to each PC
# each row is a top PC
# each column corresponds to an original count feature
prefix = feat.split("__pca")[0]  # e.g. "log1p-scale-PCA_IGHG"
loadings = pd.DataFrame(
    clf_bcr.steps[0][1].named_transformers_[prefix].steps[-1][1].components_,
    columns=clf_bcr.steps[0][1].named_transformers_[prefix].feature_names_in_,
)
# name it consistently
loadings.index = [f"{prefix}__pca{i}" for i in loadings.index]

# %%
# look at top 10 contributors
# We need to decide to look at top positive or top negative loadings - based on the coefficient for this feature:
report_positive_loading = coefs_t1d[feat] > 0
print(feat, coefs_t1d[feat], report_positive_loading)
# If the top feature is a PCA feature:
# Report the top V genes with positive loadings for that PC, meaning their presence drives the PC up.
# That's if the feature has a positive coefficient in the model.
#
# Or, if the PC feature has a negative coefficient in the model, report the V genes with negative loadings, meaning increased gene presence drives the PC down and then the negative coefficient translates it into positive model impact.
# (In other words, for PCA features with negative coefficients, it makes sense to look at the negative loadings, because a V gene with negative loading means a higher V gene presence i.e. count value pushes the PC lower)
#
# This logic only applies to PC features; if we had a v_mut feature with a negative coefficient, there's no interpreting that as leading to positive predictions.

# %%
top_contributors = (
    loadings.loc[feat].sort_values(ascending=report_positive_loading).tail(n=10)
)
# extract V gene names
top_contributor_v_genes = top_contributors.index.str.split("|").str[0].unique().tolist()

# %%
print(feat)
print()
print(top_contributors)
print()
print(top_contributor_v_genes)

# %%

# %%
# Plot those V genes' proportions in T1D vs not T1D.
# If we want to look at V gene level, sum up all the proportions by V gene (within each isotype). because featurized_data_bcr.X is scaled to sum to 1 per isotype

# %%
feat = top_contributors.index[-1]
feat

# %%
feat = feat.split("|")[0]
feat

# %%
fig, ax = plt.subplots(figsize=(6, 4))
plot_df = pd.concat(
    [
        featurized_data_bcr.X[
            featurized_data_bcr.X.columns[
                featurized_data_bcr.X.columns.str.startswith(feat + "|")
            ]
        ]
        .sum(axis=1)
        .rename(feat),
        (featurized_data_bcr.y == "T1D").map({True: "T1D", False: "Not T1D"}),
    ],
    axis=1,
)
order = ["Not T1D", "T1D"]
ax = sns.boxplot(
    data=plot_df,
    x=feat,
    y="disease",
    order=order,
    # Disable outlier markers:
    fliersize=0,
    palette=sns.color_palette("Paired")[:2],
    zorder=1,
    ax=ax,
)
for patch in ax.patches:
    # Set boxplot alpha transparency: https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.3))

np.random.seed(123)  # seed for swarmplot jitter
sns.swarmplot(
    data=plot_df,
    x=feat,
    y="disease",
    order=order,
    hue="disease",
    legend=None,
    linewidth=1,
    edgecolor="gray",
    palette=sns.color_palette("Paired")[:2],
    ax=ax,
    zorder=20,
)

plt.axvline(
    x=plot_df[plot_df["disease"] == "Not T1D"][feat].describe().loc["75%"],
    linestyle="dashed",
    zorder=10,
    linewidth=1,
    # color=sns.color_palette("Paired")[0],
    color="k",
)

# e.g. 'IGHD-M:pca_IGHV1-46' -> ['IGHD-M', 'IGHV1-46']
feat_name_parts = feat.split(":pca_")

# e.g. "IGHV1-46 percentage in IgD/M isotype"
plt.xlabel(
    f"{feat_name_parts[1]} percentage in {helpers.isotype_friendly_names[feat_name_parts[0]]} isotype"
)

ax.xaxis.set_major_formatter(PercentFormatter(1))
plt.xlim(
    0,
)
plt.ylabel("Disease", rotation=0)
ax.set_yticklabels(
    genetools.plots.add_sample_size_to_labels(
        labels=ax.get_yticklabels(),
        data=plot_df,
        hue_key="disease",
    )
)
sns.despine(ax=ax)
genetools.plots.savefig(
    fig,
    clf_bcr.output_base_dir
    / f"{clf_bcr.fold_label_train}_model.top_signal.fold_{clf_bcr.fold_id}.{fold_label}.{clf_bcr.model_name}.png",
    dpi=300,
)

# %%

# %%

# %%

# %%

# %%

# %%

# %%
