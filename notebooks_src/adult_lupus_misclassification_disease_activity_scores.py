# %% [markdown]
# # Connect adult lupus misclassifications to disease activity or treatment
#
# Hypothesis: adult lupus patient misclassifications as healthy (instead of correct prediction as lupus) correlates with lower clinical disease activity scores (SLEDAI). This cohort of patients is on treatment.
#
# Process:
# - 3 types of patient datasets corresponding to 3 types of blood samples: PBMC, RNA, Paxgene
# - PBMC dataset has all the info needed (`specimen_label` is already set, so merging in SLEDAI scores is easy)
# - RNA dataset is missing `specimen_label`. We need to set that by merging on `EXS_ID=specimen_alternative_label`
# - Paxgene dataset update is missing `specimen_label`. We will merge on `BSID` metadata column.
# - Combine the PBMC, RNA, and Paxgene datasets now with specimen labels
# - Compare healthy/lupus predicted label versus SLEDAI score
#
# We do this for BCR only, because that's where the bulk of our adult lupus cohort (the one with SLEDAI scores available) is.


# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# %matplotlib inline
import seaborn as sns
from statannotations.Annotator import Annotator
import scipy.stats as stats
import genetools


from malid import config
import crosseval
from malid.datamodels import (
    TargetObsColumnEnum,
    GeneLocus,
    healthy_label,
)
from malid.trained_model_wrappers import BlendingMetamodel

# %%
sns.set()

# %%

# %% [markdown]
# ## Load data

# %%

# %%
pd.set_option("display.max_columns", None)

# %%
# load SLEDAI scores. only some rows have been matched to our specimen_labels already. we'll have to match the rest by other metadata columns here.
# note that sledai's can be NaN or negative; both are invalid values
sledai_score_df = pd.read_csv(config.paths.metadata_dir / "sledai_scores.csv")
sledai_score_df

# %%
sledai_score_df["SampleType"].value_counts()

# %%
sledai_score_df["SampleType"].isna().value_counts()

# %%
sledai_score_df[sledai_score_df["SLEDAI_TotalScore"].notnull()][
    "SampleType"
].value_counts()

# %%
sledai_score_df.shape

# %%
# Let's go ahead and filter out invalid values: nulls and negatives
sledai_score_df = sledai_score_df[
    (sledai_score_df["SLEDAI_TotalScore"].notnull())
    & (sledai_score_df["SLEDAI_TotalScore"] >= 0)
].copy()
sledai_score_df.shape

# %%
# now that we don't have NaNs, we can cast to int
sledai_score_df["SLEDAI_TotalScore"] = sledai_score_df["SLEDAI_TotalScore"].astype(int)

# %%
# We can also now fill-na on the SLEDAI component columns:
for col in ["SLEDAI_20_LowComplement", "SLEDAI_21_DNAbinding"]:
    sledai_score_df[col] = sledai_score_df[col].fillna(0).astype(int)
sledai_score_df

# %%
# Compute SLEDAI clinical score by subtracting those two component columns, which each have score contribution value 2
sledai_score_df["SLEDAI_ClinicalScore"] = (
    sledai_score_df["SLEDAI_TotalScore"]
    - sledai_score_df["SLEDAI_20_LowComplement"]
    - sledai_score_df["SLEDAI_21_DNAbinding"]
)
sledai_score_df["SLEDAI_ClinicalScore"].describe()

# %%
# sanity checks:
assert all(sledai_score_df["SLEDAI_ClinicalScore"] >= 0)
assert all(
    sledai_score_df["SLEDAI_ClinicalScore"] <= sledai_score_df["SLEDAI_TotalScore"]
)

# %%

# %%

# %%
# PBMC: already matched to specimen_labels
assert sledai_score_df[
    (sledai_score_df.SampleType == "PBMC")
    & (sledai_score_df["specimen_label"].isnull())
].empty
sledai_score_df[sledai_score_df.SampleType == "PBMC"].head(3)

# %%
sample_type_pbmc_df = sledai_score_df[sledai_score_df.SampleType == "PBMC"]
sample_type_pbmc_df.shape

# %%

# %%
# RNA: not matched to specimen_label. We need to join by EXS_ID.
sample_type_rna_df = sledai_score_df[sledai_score_df.SampleType == "RNA"]
sample_type_rna_df.head(3)

# %%
# read adultlupus rna file. merge on specimen_alternative_label
adult_lupus_rna_df_raw = pd.read_csv(
    config.paths.metadata_dir / "adult_lupus_rna_M454_M455.specimens.tsv",
    sep="\t",
)
adult_lupus_rna_df_raw.head()

# %%
# there are duplicate values for specimen_alternative_label' == 'AA01458'
adult_lupus_rna_df_raw["specimen_alternative_label"].value_counts()[
    adult_lupus_rna_df_raw["specimen_alternative_label"].value_counts() > 1
]

# %%
# there are duplicate values for specimen_alternative_label' == 'AA01458'. Let's keep the first value by default
adult_lupus_rna_df_raw.loc[
    adult_lupus_rna_df_raw["specimen_alternative_label"] == "AA01458"
]

# %%
adult_lupus_rna_df_raw.shape

# %%
# keep the first value of specimen_alternative_label
adult_lupus_rna_df = (
    adult_lupus_rna_df_raw.groupby("specimen_alternative_label").first().reset_index()
)
assert all(adult_lupus_rna_df["specimen_alternative_label"].value_counts() == 1)
print(adult_lupus_rna_df.shape)
adult_lupus_rna_df.head(2)

# %%

# %%
sample_type_rna_df.head()

# %%
# merge to get missing specimen_label for sample_type_rna_df dataset
sample_type_rna_df_merged = pd.merge(
    sample_type_rna_df.drop(columns="specimen_label"),
    adult_lupus_rna_df[["specimen_label", "specimen_alternative_label"]],
    left_on="EXS_ID",
    right_on="specimen_alternative_label",
    how="left",
    validate="1:1",
).drop(columns="specimen_alternative_label")

sample_type_rna_df_merged.head(2)

# %%
print(sample_type_rna_df.shape)
print(sample_type_rna_df_merged.shape)

# %%

# %%
# Paxgene: no specimen_labels merged yet. Will merge on BSID.
sample_type_paxgene_df = sledai_score_df[sledai_score_df.SampleType == "Paxgene"]
sample_type_paxgene_df.head()

# %%
# read adult lupus paxgene file
adult_lupus_paxgene_df = pd.read_csv(
    config.paths.metadata_dir / "adult_lupus_paxgene_M456_M457.specimens.tsv",
    sep="\t",
)
adult_lupus_paxgene_df.head()

# %%
assert not sample_type_paxgene_df["BSID"].duplicated().any()

# %%
assert not adult_lupus_paxgene_df["BSID"].duplicated().any()

# %%
sample_type_paxgene_df_merged = pd.merge(
    sample_type_paxgene_df.drop(columns="specimen_label"),
    adult_lupus_paxgene_df,
    on="BSID",
    how="left",
    validate="1:1",
)
sample_type_paxgene_df_merged

# %%

# %%
# combine all the datasets with new specimen_labels together to form complete dataset with SLEDAI scores
combined_df = pd.concat(
    [sample_type_pbmc_df, sample_type_rna_df_merged, sample_type_paxgene_df_merged]
).reset_index(drop=True)
assert combined_df.shape[0] == sledai_score_df.shape[0]
assert (
    sledai_score_df["SampleType"]
    .value_counts()
    .equals(combined_df["SampleType"].value_counts())
)
combined_df

# %%
assert (
    combined_df[
        (combined_df["SLEDAI_TotalScore"].notnull())
        & (combined_df["SLEDAI_TotalScore"] >= 0)
    ].shape[0]
    == combined_df.shape[0]
)

# %%

# %% [markdown]
# ## Load model predictions from disk, based on `analyze_metamodels.ipynb`


# %%
base_model_train_fold_name = "train_smaller"
metamodel_fold_label_train = "validation"

gene_locus = GeneLocus.BCR
assert gene_locus in config.gene_loci_used
target_obs_col = TargetObsColumnEnum.disease
metamodel_flavor = "default"
metamodels_base_dir = BlendingMetamodel._get_metamodel_base_dir(
    gene_locus=gene_locus,
    target_obs_column=target_obs_col,
    metamodel_flavor=metamodel_flavor,
)

fname_prefix = (
    f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model"
)
model_prefix = metamodels_base_dir / fname_prefix
print(model_prefix)

output_base_dir = (
    config.paths.second_stage_blending_metamodel_output_dir
    / gene_locus.name
    / target_obs_col.name
    / metamodel_flavor
)
results_output_prefix = output_base_dir / fname_prefix
print(results_output_prefix)

experiment_set = crosseval.ExperimentSet.load_from_disk(output_prefix=model_prefix)
# Note that default y_true from BlendingMetamodel._featurize() is target_obs_column.value.blended_evaluation_column_name
# Use DROP_INCOMPLETE_FOLDS setting because alternate classification targets might not be well-split in the small validation set of the cross-validation folds that were designed to stratify disease.
# In the cases of some classification targets, we might need to automatically drop folds that have only a single class in the metamodel training data (i.e. in the validation set).
experiment_set_global_performance = experiment_set.summarize(
    remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS
)

# %%
model_global_performance = experiment_set_global_performance.model_global_performances[
    "ridge_cv"
]
individual_classifications = model_global_performance.get_all_entries()
print(individual_classifications.shape)
individual_classifications.head()

# %%
# TODO: Merge in actual lupus predicted probability using this:
model_global_performance.cv_y_preds_proba.shape
# We want to switch from max_predicted_proba to lupus predicted proba.
# But need to filter individual_classifications down to not have abstentions first
# And also need to merge in fold_id so that we can look at probabilities separately for each fold.

# %%
# filter to get y_true = Lupus only
i_filtered = individual_classifications.loc[
    individual_classifications.y_true == "Lupus"
]
print(i_filtered.shape)
print(i_filtered["y_pred"].value_counts())

# %%
# merge with combined_df to get SLEDAI scores
i_filtered_scr = pd.merge(
    i_filtered,
    combined_df[["specimen_label", "SLEDAI_TotalScore", "SLEDAI_ClinicalScore"]],
    on="specimen_label",
    how="inner",  # discard any specimens that didn't have SLEDAIs
    validate="1:1",
)

print(i_filtered_scr.shape)

# %%
i_filtered_healthy_or_lupus = i_filtered_scr.loc[
    i_filtered_scr.y_pred.isin(["Lupus", healthy_label])
]
i_filtered_healthy = i_filtered_scr.loc[i_filtered_scr.y_pred == healthy_label]
i_filtered_lupus = i_filtered_scr.loc[i_filtered_scr.y_pred == "Lupus"]

print(i_filtered_healthy_or_lupus.shape)
print(i_filtered_healthy.shape)
print(i_filtered_lupus.shape)

# %%

# %% [markdown]
# ## Analyze

# %%
# # TODO: Switch to using Pr(lupus), and analyze one fold at a time. See note above
# i_filtered_healthy[["max_predicted_proba", "SLEDAI_TotalScore"]].sort_values(
#     "SLEDAI_TotalScore"
# )

# %%
# # TODO: Switch to using Pr(lupus), and analyze one fold at a time. See note above
# i_filtered_lupus[["max_predicted_proba", "SLEDAI_TotalScore"]].sort_values(
#     "SLEDAI_TotalScore"
# )


# %%
# test for normality (testing hypothesis for using Mann Whitney test)

pred_healthy_sledai = i_filtered_healthy["SLEDAI_TotalScore"]
pred_lupus_sledai = i_filtered_lupus["SLEDAI_TotalScore"]

w1, pvalue1 = stats.shapiro(pred_healthy_sledai)
w2, pvalue2 = stats.shapiro(pred_lupus_sledai)

print(w1, pvalue1)
print(w2, pvalue2)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig.suptitle("SLEDAI_TotalScores")
ax1.hist(pred_healthy_sledai, bins=10, histtype="bar", ec="k")
ax2.hist(pred_lupus_sledai, bins=10, histtype="bar", ec="k")
ax1.set_xlabel("Pred Healthy")
ax2.set_xlabel("Pred Lupus")
plt.show()

# %%
# # TODO: Switch to using Pr(lupus), and analyze one fold at a time. See note above
# sns.scatterplot(
#     data=i_filtered_healthy_or_lupus,
#     hue="y_pred",
#     x="max_predicted_proba",
#     y="SLEDAI_TotalScore",
# )

# %%
# # check if paired test is better

# %%
stats.ttest_ind(
    i_filtered_healthy["SLEDAI_TotalScore"],
    i_filtered_lupus["SLEDAI_TotalScore"],
    nan_policy="omit",
    equal_var=False,
)

# %%
# One sided test for predicted-healthy having *lower* scores than predicted-lupus
with sns.axes_style("ticks"):
    x = "y_pred"
    y = "SLEDAI_TotalScore"
    order = [healthy_label, "Lupus"]
    fig, ax = plt.subplots(figsize=(3, 5))
    ax = sns.boxplot(
        data=i_filtered_healthy_or_lupus,
        x=x,
        y=y,
        order=order,
        # Disable outlier markers:
        fliersize=0,
        palette=sns.color_palette("Paired")[3:5],
    )

    # Force y-axis ticks to be integers since SLEDAIs are not floats. And reduce total number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    for patch in ax.patches:
        # Set boxplot alpha transparency: https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))
    sns.swarmplot(
        data=i_filtered_healthy_or_lupus,
        x=x,
        y=y,
        order=order,
        hue=x,
        legend=None,
        linewidth=1,
        edgecolor="gray",
        palette=sns.color_palette("Paired")[3:5],
        ax=ax,
    )

    # Annotate with statistical significance
    annot = Annotator(
        ax=ax,
        pairs=[order],
        data=i_filtered_healthy_or_lupus,
        x=x,
        y=y,
        order=order,
    )
    annot.configure(
        test="Mann-Whitney-ls", text_format="star", loc="outside", verbose=2
    )
    annot.apply_test(method="asymptotic")
    ax, test_results = annot.annotate()

    # Finish plot
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            ax.get_xticklabels(), i_filtered_healthy_or_lupus, x
        )
    )
    genetools.plots.wrap_tick_labels(ax, wrap_amount=10)
    plt.ylim(
        -0.5,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("SLEDAI total score")
    sns.despine(ax=ax)
    plt.tight_layout()
    genetools.plots.savefig(
        fig, f"{results_output_prefix}.lupus_sledai_vs_predictions.png", dpi=300
    )
    genetools.plots.savefig(
        fig, f"{results_output_prefix}.lupus_sledai_vs_predictions.pdf", dpi=600
    )
    if test_results is not None and len(test_results) > 0:
        with open(
            f"{results_output_prefix}.lupus_sledai_vs_predictions.test_results.txt",
            "w",
        ) as f:
            f.write(test_results[0].data.formatted_output)

# %%

# %%
# Test same with clinical-only score:

# %%
# One sided test for predicted-healthy having *lower* scores than predicted-lupus
with sns.axes_style("ticks"):
    x = "y_pred"
    y = "SLEDAI_ClinicalScore"
    order = [healthy_label, "Lupus"]
    fig, ax = plt.subplots(figsize=(3, 5))
    ax = sns.boxplot(
        data=i_filtered_healthy_or_lupus,
        x=x,
        y=y,
        order=order,
        # Disable outlier markers:
        fliersize=0,
        palette=sns.color_palette("Paired")[3:5],
    )

    # Force y-axis ticks to be integers since SLEDAIs are not floats. And reduce total number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    for patch in ax.patches:
        # Set boxplot alpha transparency: https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))
    sns.swarmplot(
        data=i_filtered_healthy_or_lupus,
        x=x,
        y=y,
        order=order,
        hue=x,
        legend=None,
        linewidth=1,
        edgecolor="gray",
        palette=sns.color_palette("Paired")[3:5],
        ax=ax,
    )

    # Annotate with statistical significance
    annot = Annotator(
        ax=ax,
        pairs=[order],
        data=i_filtered_healthy_or_lupus,
        x=x,
        y=y,
        order=order,
    )
    annot.configure(
        test="Mann-Whitney-ls", text_format="star", loc="outside", verbose=2
    )
    annot.apply_test(method="asymptotic")
    ax, test_results = annot.annotate()

    # Finish plot
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            ax.get_xticklabels(), i_filtered_healthy_or_lupus, x
        )
    )
    genetools.plots.wrap_tick_labels(ax, wrap_amount=10)
    plt.ylim(
        -0.5,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Clinical SLEDAI score")
    sns.despine(ax=ax)
    plt.tight_layout()
    # genetools.plots.savefig(
    #     fig, f"{results_output_prefix}.lupus_sledai_vs_predictions.png", dpi=300
    # )
    # if test_results is not None and len(test_results) > 0:
    #     with open(
    #         f"{results_output_prefix}.lupus_sledai_vs_predictions.test_results.txt",
    #         "w",
    #     ) as f:
    #         f.write(test_results[0].data.formatted_output)

# %%
