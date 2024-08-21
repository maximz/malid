# %% [markdown]
# # Number of model 2 cluster hits per sample

# %%
from malid import helpers, config
from malid.datamodels import *

# %%
from malid.trained_model_wrappers import ConvergentClusterClassifier

# %%
from malid import io

# %%
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

# %%
import genetools

# %%

# %%
target_obs_column = TargetObsColumnEnum.disease

# %%
# Load each fold's model2 and held-out test set
# Count disease-specific model2 cluster hits for each held-out test sample

featurized_datas = {}
for gene_locus in config.gene_loci_used:
    for fold_id in config.cross_validation_fold_ids:
        clf = ConvergentClusterClassifier(
            fold_id=fold_id,
            model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
                gene_locus
            ],
            fold_label_train="train_smaller1",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )
        adata = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label="test",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            load_obs_only=True,
        )
        featurized_datas[(gene_locus, fold_id)] = clf.featurize(adata)
        print(gene_locus, fold_id)
        io.clear_cached_fold_embeddings()

# %%

# %%

# %%

# %%
# Review some of the data manually

# %%
gene_locus = GeneLocus.BCR
fold_id = 0
fd = featurized_datas[gene_locus, 0]

# %%
fd.X

# %%
fd.y

# %%
for disease in fd.y.unique():
    plt.figure()
    plt.hist(fd.X.loc[fd.y == disease, disease])
    plt.title(f"{disease} - {gene_locus} (fold {fold_id})")

# %%

# %%

# %%
# Plot it all

# %%
all_diseases = clf.classes_
all_diseases = all_diseases[all_diseases != healthy_label]
all_diseases

# %%
len(featurized_datas)

# %%
featurized_datas.keys()

# %%

# %%
with sns.plotting_context("paper", font_scale=0.8):
    fig, axarr = plt.subplots(
        nrows=len(all_diseases), ncols=len(featurized_datas), figsize=(12, 15)
    )
    for ix_col, ((gene_locus, fold_id), fd) in enumerate(featurized_datas.items()):
        for ix_row, disease in enumerate(all_diseases):
            ax = axarr[ix_row, ix_col]
            data_for_this_ax = fd.X.loc[fd.y == disease, disease]
            ax.hist(data_for_this_ax)

            if max(data_for_this_ax) > 5:
                ax.set_xlim(
                    -1,
                )
            else:
                ax.set_xlim(
                    -0.05,
                )

            if max(data_for_this_ax) == 0:
                # If the only entry is 0, make the xticks reasonable
                ax.set_xticks([0, 1])
                ax.set_xlim(-0.05, 1.05)

            if ix_row == 0:
                ax.set_title(f"{gene_locus.name}, fold {fold_id}", fontweight="bold")
            if ix_col == 0:
                ax.set_ylabel(
                    "Number of\n" r"$\bf{" + f"{disease}" + r"}$" + " samples",
                    rotation=0,
                    ha="right",
                    va="center",
                )

            sns.despine(ax=ax)

    for ax in axarr[-1, :]:
        ax.set_xlabel("Clusters hit per sample")

    plot_fname = (
        config.paths.convergent_clusters_output_dir
        / f"cluster_hits_per_sample.{target_obs_column.name}.png"
    )
    genetools.plots.savefig(fig, plot_fname, dpi=300)
    print(plot_fname)

# %%

# %%
# How many patients (samples, really) from each disease have at least 1 cluster hit from that disease?
how_many_have_at_least_one_hit = []
for (gene_locus, fold_id), fd in featurized_datas.items():
    for disease in all_diseases:
        how_many_have_at_least_one_hit.append(
            dict(
                gene_locus=gene_locus.name,
                fold_id=fold_id,
                disease=disease,
                count=(fd.X.loc[fd.y == disease, disease] > 0).sum(),
            )
        )
pd.DataFrame(how_many_have_at_least_one_hit).pivot(
    index=["gene_locus", "fold_id"], columns="disease", values="count"
)

# %%
# What percentage of patients (samples, really) from each disease have at least 1 cluster hit from that disease?
how_many_have_at_least_one_hit = []
for (gene_locus, fold_id), fd in featurized_datas.items():
    for disease in all_diseases:
        how_many_have_at_least_one_hit.append(
            dict(
                gene_locus=gene_locus.name,
                fold_id=fold_id,
                disease=disease,
                count=(fd.X.loc[fd.y == disease, disease] > 0).mean(),
            )
        )

# %%
pd.DataFrame(how_many_have_at_least_one_hit).pivot(
    index=["gene_locus", "fold_id"], columns="disease", values="count"
) * 100

# %%

# %%

# %% [markdown]
# # Extract number of predictive clusters for each disease from model 2

# %%
predictive_cluster_counts = {}
for gene_locus in config.gene_loci_used:
    for fold_id in config.cross_validation_fold_ids:
        clftmp = ConvergentClusterClassifier(
            fold_id=fold_id,
            model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
                gene_locus
            ],
            fold_label_train="train_smaller1",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )
        # Get all clusters
        model2_cluster_class_associations = (
            clftmp.cluster_centroids_with_class_specific_p_values
        )

        # Melt to columns = [cluster_dominant_label, p_value], but first move consensus_sequence into the set of index columns
        # TODO(refactor): this reset_index().set_index() operation is quite slow
        model2_cluster_class_associations = (
            model2_cluster_class_associations.reset_index()
            .set_index(
                list(model2_cluster_class_associations.index.names)
                + ["consensus_sequence"]
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
            model2_cluster_class_associations["p_value"] <= clftmp.p_value_threshold
        ]

        # Filter to Covid predictive cluster centroids only
        predictive_cluster_counts[(gene_locus.name, fold_id)] = (
            model2_cluster_class_associations["cluster_dominant_label"]
            .value_counts()
            .to_dict()
        ) | {"p value threshold": clftmp.p_value_threshold}

# Compile
predictive_cluster_counts = pd.DataFrame.from_dict(
    predictive_cluster_counts, orient="index"
).fillna(0)
for col in predictive_cluster_counts.columns:
    if col != "p value threshold":
        predictive_cluster_counts[col] = predictive_cluster_counts[col].astype(int)
predictive_cluster_counts.index.names = ["Sequencing locus", "Fold ID"]
predictive_cluster_counts = predictive_cluster_counts[
    list(clftmp.classes_[clftmp.classes_ != healthy_label]) + ["p value threshold"]
]
predictive_cluster_counts.to_csv(
    config.paths.convergent_clusters_output_dir
    / f"number_of_clusters_chosen.{target_obs_column.name}.tsv",
    sep="\t",
)
predictive_cluster_counts

# %%
# Why are no clusters kept?
# Interpretation: Stricter p-value worked better overall across diseases (held-out performance on train_smaller2).
# In future work, we can consider different p-values for each disease class.

# %%

# %%

# %%

# %% [markdown]
# # Look at coefficients

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
clf_bcr = ConvergentClusterClassifier(
    fold_id=fold_id,
    model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
        GeneLocus.BCR
    ],
    fold_label_train="train_smaller1",
    gene_locus=GeneLocus.BCR,
    target_obs_column=TargetObsColumnEnum.disease,
)
clf_bcr

# %%
clf_bcr._inner

# %%
featurized_data_bcr = clf_bcr.featurize(adata_bcr)
featurized_data_bcr.X

# %%
featurized_data_bcr.X.describe().loc[["min", "max"]]

# %%
# Coefficients table
# Rows are the models
# Columns are the features
coefs_table_bcr = pd.DataFrame(
    clf_bcr.steps[-1][1].coef_,
    index=[f"{c} predictions" for c in clf_bcr.classes_],
    columns=[
        f"{c} cluster hits (scaled around {scale_mean:0.2g})"
        for c, scale_mean in zip(clf_bcr.feature_names_in_, clf_bcr.steps[-2][1].mean_)
    ],
)
coefs_table_bcr

# %%
ax = sns.heatmap(
    coefs_table_bcr,
    center=0,
    cmap="vlag",
    annot=True,
    square=True,
    cbar_kws={"label": "Coefficient", "shrink": 0.8},
)
plt.xticks(rotation=45, ha="right")

for i in range(coefs_table_bcr.shape[0] - 1):
    # Put some dividing lines between the rows
    ax.axhline(y=i + 1, color="#444", linewidth=2)

# Lines around the outer edge of the box
ax.axhline(y=0, color="#444", linewidth=4)
ax.axhline(y=coefs_table_bcr.shape[1], color="#444", linewidth=4)
ax.axvline(x=0, color="#444", linewidth=4)
ax.axvline(x=coefs_table_bcr.shape[0], color="#444", linewidth=4)

plt.ylabel("Class")
plt.xlabel("BCR feature")
genetools.plots.savefig(
    ax.get_figure(),
    clf_bcr.output_base_dir
    / f"{clf_bcr.fold_label_train}_model.coefficients.{clf_bcr.model_name}.fold_{clf_bcr.fold_id}.png",
    dpi=300,
)

# %%

# %%
