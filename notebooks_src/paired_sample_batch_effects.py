# %%
# before any Mal-ID imports, we need to configure our environment to use the special leave-one-cohort-out split
import os

os.environ["MALID_CV_SPLIT"] = "in_house_peak_disease_leave_one_cohort_out"

# %%
import numpy as np
import pandas as pd

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import genetools
from IPython.display import display, Markdown
from typing import List, Tuple, Dict, Optional
import gc

from malid import config, io
from malid.trained_model_wrappers import BlendingMetamodel
import crosseval
from crosseval import FeaturizedData
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    healthy_label,
    CrossValidationSplitStrategy,
)

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# %%

# %%
assert (
    config.cross_validation_split_strategy
    == CrossValidationSplitStrategy.in_house_peak_disease_leave_one_cohort_out
)

# %%

# %% [markdown]
# ## Load data, and figure out how we will separate healthy resequenced samples by replicate.

# %%
adata_bcr = io.load_fold_embeddings(
    fold_id=0,
    fold_label="test",
    gene_locus=GeneLocus.BCR,
    target_obs_column=TargetObsColumnEnum.disease,
)
adata_bcr

# %%
adata_tcr = io.load_fold_embeddings(
    fold_id=0,
    fold_label="test",
    gene_locus=GeneLocus.TCR,
    target_obs_column=TargetObsColumnEnum.disease,
)
adata_tcr

# %%

# %%
adata_bcr.obs.drop_duplicates("specimen_label")["disease"].value_counts()

# %%
# healthy example:
adata_bcr.obs[adata_bcr.obs["specimen_label"] == "M64-081"][
    "amplification_label"
].cat.remove_unused_categories().value_counts()

# %%
# Covid has single amplification
adata_bcr.obs[adata_bcr.obs["specimen_label"] == "M371-S004"][
    "amplification_label"
].nunique()

# %%
# proof: Covid has single amplification, healthy has two amplifications (original and resequencing)
adata_bcr.obs.groupby(["disease", "specimen_label"], observed=True)[
    "amplification_label"
].nunique()

# %%

# %%
# Rename specimens using the amplification label, so we can separate the healthy replicates.

# %%
pd.set_option("display.max_columns", None)
adata_bcr.obs.groupby(["specimen_label", "amplification_label"], observed=True).size()

# %%

# %%
adata_bcr.obs[adata_bcr.obs["disease"] == healthy_label][
    "amplification_label"
].unique().tolist()

# %%
adata_tcr.obs[adata_tcr.obs["disease"] == healthy_label][
    "amplification_label"
].unique().tolist()

# %%
adata_bcr.obs[adata_bcr.obs["disease"] != healthy_label][
    "amplification_label"
].unique().tolist()

# %%
adata_tcr.obs[adata_tcr.obs["disease"] != healthy_label][
    "amplification_label"
].unique().tolist()

# %%

# %%
# Set extracted_replicate_label column based on amplification_label, which is slightly different in BCR vs TCR data
# We will unify them here carefully
adata_tcr.obs["extracted_replicate_label"] = (
    adata_tcr.obs["amplification_label"]
    .str.replace("_cDNA.*TCRB", "")
    .apply(lambda x: f"M66-{x}" if len(x.split("-")) == 2 else x)
)


adata_bcr.obs["extracted_replicate_label"] = (
    adata_bcr.obs["amplification_label"]
    .str.replace("_cDNA_PCR", "")
    .apply(lambda x: f"M66-{x}" if len(x.split("-")) == 2 else x)
)

# %%
# Review results.

# %%
adata_bcr.obs[
    ["amplification_label", "extracted_replicate_label"]
].drop_duplicates().sort_values("extracted_replicate_label")

# %%
adata_tcr.obs[
    ["amplification_label", "extracted_replicate_label"]
].drop_duplicates().sort_values("extracted_replicate_label")

# %%
# This one was present in TCR but not in BCR:
set(adata_tcr.obs.extracted_replicate_label) - set(
    adata_bcr.obs.extracted_replicate_label
)

# %%
# All clear: BCR entries are all in TCR.
set(adata_bcr.obs.extracted_replicate_label) - set(
    adata_tcr.obs.extracted_replicate_label
)

# %%
# Investigate the missing one from above.
# Check what's in BCR for this specimen:
adata_bcr[adata_bcr.obs["extracted_replicate_label"].str.contains("M64-097")].obs[
    ["amplification_label", "extracted_replicate_label"]
].drop_duplicates()

# %%
# Check what's in TCR for this specimen:
adata_tcr[adata_tcr.obs["extracted_replicate_label"].str.contains("M64-097")].obs[
    ["amplification_label", "extracted_replicate_label"]
].drop_duplicates()

# %%
# Just to be safe, check another way..
adata_bcr[adata_bcr.obs["specimen_label"] == "M64-097"].obs[
    ["amplification_label", "extracted_replicate_label"]
].drop_duplicates()

# %%
adata_tcr[adata_tcr.obs["specimen_label"] == "M64-097"].obs[
    ["amplification_label", "extracted_replicate_label"]
].drop_duplicates()

# %%
# Looked into this. This particular replicate (M479-M64-097) failed BCR sequencing.
# Here is read count after demultiplexing:
#
# M479-M64-097_cDNA_PCR_IGA |      0
# M479-M64-097_cDNA_PCR_IGD |      0
# M479-M64-097_cDNA_PCR_IGE |      0
# M479-M64-097_cDNA_PCR_IGG |      2
# M479-M64-097_cDNA_PCR_IGM |      0
#
# But worked fine in TCR:
# M479-M64-097_cDNA_PCR_TCRB_R1 | 220346
#
#
# For consistency, let's exclude this sample from the rest of the leave-one-cohort-out analysis.

# %%

# %%

# %%

# %% [markdown]
# ## Run metamodel on each replicate separately.

# %%
base_model_train_fold_name = "train_smaller"
metamodel_fold_label_train = "validation"
gene_locus = GeneLocus.BCR | GeneLocus.TCR
classification_target = TargetObsColumnEnum.disease
metamodel_flavor = "default"
metamodel_name = "ridge_cv"


# %%
def extract_anndatas_for_replicates(
    adata_bcr,
    adata_tcr,
    sample_identifier="extracted_replicate_label",
):
    """Generator yielding replicate anndatas"""
    # the data is stored as one embedded anndata per locus per participant (i.e. can have multiple specimens)
    all_replicate_labels = set(adata_tcr.obs[sample_identifier]).intersection(
        set(adata_bcr.obs[sample_identifier])
    )
    full_loci_adatas = {GeneLocus.BCR: adata_bcr, GeneLocus.TCR: adata_tcr}

    for extracted_replicate_label in all_replicate_labels:
        adatas_by_locus: Dict[GeneLocus, anndata.AnnData] = {}
        for gene_locus, full_adata in full_loci_adatas.items():
            adata = full_adata[
                full_adata.obs[sample_identifier] == extracted_replicate_label
            ].copy()
            adata.obs = adata.obs.assign(fold_id=-1, fold_label="extra_replicates")
            adata = io._add_anndata_columns(adata)
            adatas_by_locus[gene_locus] = adata

        # For each participant, yield Tuple[sample_identifier string, Dict[GeneLocus, anndata.AnnData]]
        yield extracted_replicate_label, adatas_by_locus


# %%
# Final result containers, for all metamodels
results = crosseval.ExperimentSet()

# Load the metamodels
clf = BlendingMetamodel.from_disk(
    fold_id=0,
    metamodel_name=metamodel_name,
    base_model_train_fold_name=base_model_train_fold_name,
    metamodel_fold_label_train=metamodel_fold_label_train,
    gene_locus=gene_locus,
    target_obs_column=classification_target,
    metamodel_flavor=metamodel_flavor,
)

# %%
clf.output_base_dir.mkdir(parents=True, exist_ok=True)

# %%

# %%
# Apply min-clone-count filters to the replicates (so far applied only to the full specimen)

# %%
specimen_isotype_counts_dfs = []
for gene_locus, adata in ((GeneLocus.BCR, adata_bcr), (GeneLocus.TCR, adata_tcr)):
    specimen_isotype_counts = []
    for (specimen_label, amplification_label), subset_obs in adata.obs.groupby(
        ["specimen_label", "extracted_replicate_label"], observed=True
    ):
        isotype_counts = (
            subset_obs["isotype_supergroup"]
            .cat.remove_unused_categories()
            .value_counts()
        )
        specimen_description = subset_obs[
            ["specimen_label", "extracted_replicate_label"]
        ].iloc[
            0
        ]  # , "disease"
        specimen_isotype_counts.append(
            {
                **isotype_counts.to_dict(),
                **specimen_description.to_dict(),
            }
        )
    specimen_isotype_counts_df = pd.DataFrame(specimen_isotype_counts).set_index(
        ["specimen_label", "extracted_replicate_label"]
    )
    specimen_isotype_counts_df[
        f"total_{gene_locus.name}"
    ] = specimen_isotype_counts_df.sum(axis=1).astype(int)
    specimen_isotype_counts_dfs.append(specimen_isotype_counts_df)
    print(gene_locus)
    display(specimen_isotype_counts_df.head())

# %%
pd.concat(specimen_isotype_counts_dfs, axis=1).isna().any(axis=1).loc[lambda x: x]

# %%
specimen_isotype_counts_df = pd.concat(specimen_isotype_counts_dfs, axis=1)

# Expected: M479-M64-097 (replicate 2 of M64-097) failed BCR sequencing
print(
    "Failed to merge:",
    specimen_isotype_counts_df.isna().any(axis=1).loc[lambda x: x].index,
)

specimen_isotype_counts_df

# %%
specimen_isotype_counts_df = specimen_isotype_counts_df.fillna(0).astype(int)
specimen_isotype_counts_df

# %%
# Run our QC rules on this. Reject a specimens if any of its replicates fail
# (This will include rejecting M64-097 that had no BCR data at all)
from malid.sample_sequences import REQUIRED_CLONE_COUNTS_BY_ISOTYPE

print(REQUIRED_CLONE_COUNTS_BY_ISOTYPE)

rejected_specimens = []
for (
    specimen_label,
    extracted_replicate_label,
), clone_count_by_isotype in specimen_isotype_counts_df.iterrows():
    if (
        clone_count_by_isotype.loc["IGHG"] < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["IGHG"]
        or clone_count_by_isotype.loc["IGHA"] < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["IGHA"]
        or clone_count_by_isotype.loc["IGHD-M"]
        < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["IGHD-M"]
        or clone_count_by_isotype.loc["TCRB"] < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["TCRB"]
    ):
        print(f"Rejecting: {specimen_label}, {extracted_replicate_label}")
        rejected_specimens.append(specimen_label)

# %%
rejected_specimens = list(set(rejected_specimens))
rejected_specimens

# %%
# Save the list to disk
pd.Series(rejected_specimens).sort_values().to_csv(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "rejected_specimens_because_some_replicates_failed_qc.txt",
    index=None,
    header=False,
)

# %%

# %%
# How many rejected out of how many total (note: total includes single-replicate specimens)
len(rejected_specimens), len(
    specimen_isotype_counts_df.index.get_level_values("specimen_label").unique()
)

# %%

# %%
# Plot counts for remaining, non-rejected specimens
specimen_isotype_counts_df_remaining = specimen_isotype_counts_df[
    ~specimen_isotype_counts_df.index.isin(rejected_specimens, level="specimen_label")
]

for gene_locus in config.gene_loci_used:
    # Sort so that higher-count replicate comes before lower-count replicate
    totals_sorted = (
        specimen_isotype_counts_df_remaining.rename(
            columns={f"total_{gene_locus.name}": "total"}
        )
        .reset_index()
        .sort_values(by=["specimen_label", "total"], ascending=[True, False])
    )

    # Group by specimen_label and create the two value columns, one per replicate
    totals_transformed = (
        totals_sorted.groupby("specimen_label")["total"].apply(list).apply(pd.Series)
    )
    totals_transformed.columns = ["total_bigger_replicate", "total_smaller_replicate"]

    # Reset index to make 'specimen_label' a column again
    totals_transformed.reset_index(inplace=True)

    # Drop any rows where one replicate is empty. This removes single-replicate specimens.
    totals_transformed.dropna(axis=0, inplace=True)

    # Melt to long format. And sort so that specimens with lower smaller-replicate counts come first
    totals_long = pd.melt(
        totals_transformed.sort_values(["total_smaller_replicate"]),
        id_vars="specimen_label",
        var_name="Replicate",
        value_name="Clone count",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=totals_long,
        y="specimen_label",
        x="Clone count",
        hue="Replicate",
        palette="viridis",
    )
    plt.title("Value1 and Value2 by specimen_label")
    plt.ylabel("specimen_label")
    plt.xlabel("Clone count")
    plt.title(gene_locus)
    plt.legend(title="Replicate")
    plt.tight_layout()

# %%

# %%
# Remove flagged specimens from the anndatas
print(adata_bcr.shape, adata_tcr.shape)

adata_bcr = adata_bcr[~adata_bcr.obs["specimen_label"].isin(rejected_specimens)]
adata_tcr = adata_tcr[~adata_tcr.obs["specimen_label"].isin(rejected_specimens)]

print(adata_bcr.shape, adata_tcr.shape)

# %%

# %%
# Featurize.
# Create a List[FeaturizedData]. Each extracted replicate will have one FeaturizedData object.
featurized_list: List[FeaturizedData] = []
for extracted_replicate_label, adata_by_locus in extract_anndatas_for_replicates(
    adata_bcr, adata_tcr
):
    # Featurize one replicate.
    # adata_by_locus has input anndatas wrapped as Dict[GeneLocus, anndata.AnnData], allowing use of single-locus or multi-locus metamodel.
    fd = clf.featurize(adata_by_locus)

    # Sanity check: Each FeaturizedData has only a single unique extracted replicate label, or is empty due to abstentions of which there would also only be 1
    assert (
        len(fd.sample_names) + len(fd.abstained_sample_names) == 1
    ), "metadata length plus abstained metadata length should equal 1"

    # We should set the index to be extracted replicate label, and make sure index is unique everywhere.

    # Change metadata index (and sample_names accordingly) to be extracted_replicate_label (which is not yet inside the metadata).
    # So extracted_replicate_label will be the index, and old index (specimen_label) is just another metadata column.
    # Do this for both the main metadata and abstained metadata
    fd.metadata = (
        fd.metadata.assign(extracted_replicate_label=extracted_replicate_label)
        .reset_index()
        .set_index("extracted_replicate_label")
    )
    fd.abstained_sample_metadata = (
        fd.abstained_sample_metadata.assign(
            extracted_replicate_label=extracted_replicate_label
        )
        .reset_index()
        .set_index("extracted_replicate_label")
    )

    fd.sample_names = fd.metadata.index
    fd.abstained_sample_names = fd.abstained_sample_metadata.index

    # Now add it to featurized_list
    featurized_list.append(fd)

    # garbage collect
    del adata_by_locus
    gc.collect()

# %%

# %%
# Combine featurized_list into a single FeaturizedData object
featurized_all: FeaturizedData = FeaturizedData.concat(featurized_list)

# %%
# Sanity check: Each extracted replicate label appears only once in the combined FeaturizedData
# Confirm that all indexes are unique, from metadata and abstained metadata combined
assert (
    not pd.Series(
        np.hstack(
            [
                featurized_all.metadata.index,
                featurized_all.abstained_sample_metadata.index,
            ]
        )
    )
    .duplicated()
    .any()
), "Same extracted replicate label shared by multiple entries in featurized_list"

# %%

# %%
featurized_all.metadata.shape, featurized_all.abstained_sample_metadata.shape

# %%
featurized_all.metadata.head()

# %%
featurized_all.abstained_sample_metadata.head()

# %%
# According to the plots above, the two abstentions, M64-093 and M64-054, had fewer IgH clone counts than all other replicates. They were close to being rejected by the min-clone-count quality filters.

# %%

# %%
# Run model
results = crosseval.ModelSingleFoldPerformance(
    model_name=metamodel_name,
    fold_id=-1,
    y_true=featurized_all.y,
    clf=clf,
    X_test=featurized_all.X,
    fold_label_train="train_smaller",
    fold_label_test="extra_replicates",
    test_metadata=featurized_all.metadata,
    test_abstentions=featurized_all.abstained_sample_y,
    test_abstention_metadata=featurized_all.abstained_sample_metadata,
)

# %%

# %%
results.test_metadata["y_pred"] = results.y_pred

# %%
results.test_metadata.head()

# %%
probas = pd.DataFrame(
    results.y_preds_proba,
    columns=[f"{i}_prob" for i in clf.classes_],
    index=results.test_metadata.index,
)
probas

# %%
summary_results = pd.concat([results.test_metadata, probas], axis=1)
summary_results.head()

# %%

# %%
# Sort by index (extracted_replicate_label in reverse, so that M66 goes before M477) and by specimen_label
summary_results = (
    summary_results.reset_index()
    .sort_values(
        ["specimen_label", "extracted_replicate_label"], ascending=[True, False]
    )
    .set_index("extracted_replicate_label")
)
summary_results

# %%

# %%
# Export summary_results
summary_results.to_csv(
    clf.output_base_dir / f"{clf.model_file_prefix}.healthy_replicates.ridge_cv.tsv",
    sep="\t",
)

# %%

# %%
# Filter summary_results down to specimens with multiple replicates only
replicates = summary_results.loc[
    summary_results["specimen_label"].isin(
        summary_results["specimen_label"].value_counts().loc[lambda x: x > 1].index
    )
].copy()
replicates

# %%
# Which studies did these replicates come from?
replicates.study_name.value_counts()

# %%

# %%
# Important for below code: All specimens with replicates had exactly 2 replicates
assert all(replicates.groupby("specimen_label").size() == 2)

# %%

# %%
# Split into replicate1 and replicate2 dataframes
df1 = (
    replicates.drop_duplicates(subset="specimen_label", keep="first")
    .reset_index()
    .set_index("specimen_label")
)
df2 = (
    replicates.drop_duplicates(subset="specimen_label", keep="last")
    .reset_index()
    .set_index("specimen_label")
    # Make sure they have matching index
    .loc[df1.index]
)
color = df1["disease"]
assert df1.shape[0] == df2.shape[0]

# %%

# %%
# Sanity check the extracted_replicate_label's in each dataframe

# %%
df1

# %%
df2

# %%

# %%
# List of diseases or conditions to compare
columns = [col for col in replicates.columns if "_prob" in col]
columns


# %%
def concordance_correlation_coefficient(y_true, y_pred):
    pearson_corr = pearsonr(y_true, y_pred)[0]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    return (2 * pearson_corr * mean_true * mean_pred) / (
        var_true + var_pred + (mean_true - mean_pred) ** 2
    )


# %%
# For each disease, correlation of P(disease) vector for all replicate1's versus P(disease) vector for all replicate2's
stats_df = {}

for column in columns:
    y_true = df1[column]
    y_pred = df2[column]

    stats_df[column] = {
        "Pearson": pearsonr(y_true, y_pred)[0],
        "Spearman": spearmanr(y_true, y_pred).correlation,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "CCC": concordance_correlation_coefficient(y_true, y_pred),
    }

stats_df = pd.DataFrame.from_dict(stats_df, orient="index")
stats_df

# %%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()

for i, column in enumerate(columns):
    sns.scatterplot(x=df1[column], y=df2[column], ax=axs[i], legend=(i == 0))
    axs[i].plot([0, 1], [0, 1], transform=axs[i].transAxes, color="red", linestyle="--")
    axs[i].set_title(column)
    axs[i].set_xlabel("Replicate 1")
    axs[i].set_ylabel("Replicate 2")

    # Annotate subplot with statistics
    stats = stats_df.loc[column]
    annotation = f"Pearson: {stats['Pearson']:.2f}\nSpearman: {stats['Spearman']:.2f}\nRMSE: {stats['RMSE']:.2f}"
    axs[i].annotate(annotation, xy=(0.05, 0.7), xycoords="axes fraction")

plt.tight_layout()
plt.show()

# %%

# %%
# Plot per sample:

# Determine layout dimensions for subplots
num_samples = len(df1)
nrows = int(np.ceil(np.sqrt(num_samples)))
ncols = int(np.ceil(num_samples / nrows))

# Set base dimensions for each subplot
base_width = 6
base_height = 4

# Calculate total dimensions for the entire figure
fig_width = base_width * ncols
fig_height = base_height * nrows

# Create figure and axis objects
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
axs = axs.ravel()

# Loop through each sample to populate subplots
for idx, sample in enumerate(df1.index):
    ax = axs[idx]

    width = 0.35  # Width of the bars
    r1 = np.arange(len(columns))  # Positions of bars for Replicate 1
    r2 = [x + width for x in r1]  # Positions of bars for Replicate 2

    # Plotting bars for each replicate
    bars1 = ax.bar(
        r1, df1.loc[sample, columns].values, width=width, label="Replicate 1", alpha=0.8
    )
    bars2 = ax.bar(
        r2, df2.loc[sample, columns].values, width=width, label="Replicate 2", alpha=0.8
    )

    # Labeling, title, and legend
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")
    ax.set_title(f"Specimen {sample}")
    ax.set_xticks([r + width / 2 for r in range(len(columns))])
    ax.set_xticklabels(
        [s.replace("_prob", "") for s in columns]
    )  # , rotation=45, ha="right")
    genetools.plots.wrap_tick_labels(
        ax=ax, wrap_x_axis=True, wrap_y_axis=False, wrap_amount=10
    )
    if idx == 0:
        ax.legend()

# Hide any remaining empty subplots
for idx in range(num_samples, nrows * ncols):
    axs[idx].axis("off")

# Adjust layout for better visibility
plt.tight_layout()
plt.show()

# %%

# %%
assert df1.shape[0] == df2.shape[0] == replicates.shape[0] / 2

# %%
# replicates["first_replicate"] = replicates.index.isin(
#     replicates.drop_duplicates(subset="specimen_label", keep="first").index
# )
replicates["first_replicate"] = replicates.index.isin(
    df1["extracted_replicate_label"].values
)
replicates["replicate_num"] = replicates["first_replicate"].apply(
    lambda x: 1 if x else 2
)
replicates[["specimen_label", "first_replicate", "replicate_num"]]

# %%
df = replicates.melt(id_vars=["replicate_num", "specimen_label"], value_vars=columns)
sns.catplot(x="variable", y="value", hue="replicate_num", data=df)
plt.xticks(rotation=45, ha="right")

# %%

# %%
stats_df = {}

for sample in df1.index:
    y_true_sample = df1.loc[sample, columns]
    y_pred_sample = df2.loc[sample, columns]

    stats_df[sample] = {
        "Pearson": pearsonr(y_true_sample, y_pred_sample)[0],
        "Spearman": spearmanr(y_true_sample, y_pred_sample).correlation,
        "RMSE": np.sqrt(mean_squared_error(y_true_sample, y_pred_sample)),
        "CCC": concordance_correlation_coefficient(y_true_sample, y_pred_sample),
    }

stats_df = pd.DataFrame.from_dict(stats_df, orient="index")

sns.displot(x="Pearson", data=stats_df)
plt.title("distribution of per-sample Pearson correlations")
stats_df.sort_values("Pearson")

# %%
# How many specimens had Pearson correlation over 90% between the predicted class probability vectors for their two replicates?
(stats_df.Pearson > 0.9).sum(), stats_df.shape[0]

# %%
with sns.plotting_context(
    "talk",
    # Adjust font_scale as needed for further customization
    font_scale=0.6,
):

    # Determine layout dimensions for subplots
    nrows = int(np.ceil(np.sqrt(num_samples)))
    ncols = int(np.ceil(num_samples / nrows))

    # Set base dimensions for each subplot
    base_width = 6
    base_height = 4

    # Calculate total dimensions for the entire figure
    fig_width = base_width * ncols
    fig_height = base_height * nrows

    # Create figure and axis objects
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    axs = axs.ravel()

    # Loop through each sample to populate subplots
    for idx, sample in enumerate(df1.index):
        ax = axs[idx]

        width = 0.35  # Width of the bars
        r1 = np.arange(len(columns))  # Positions of bars for Replicate 1
        r2 = [x + width for x in r1]  # Positions of bars for Replicate 2

        # Plotting bars for each replicate
        bars1 = ax.bar(
            r1,
            df1.loc[sample, columns].values,
            width=width,
            label="Replicate 1",
            alpha=0.8,
        )
        bars2 = ax.bar(
            r2,
            df2.loc[sample, columns].values,
            width=width,
            label="Replicate 2",
            alpha=0.8,
        )

        # Labeling, title, and legend
        ax.set_xlabel("Disease")
        if idx % ncols == 0:
            ax.set_ylabel("Probability")
        ax.set_title(f"Specimen {sample}")
        ax.set_xticks([r + width / 2 for r in range(len(columns))])
        # ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_xticklabels([s.replace("_prob", "") for s in columns])
        genetools.plots.wrap_tick_labels(
            ax=ax, wrap_x_axis=True, wrap_y_axis=False, wrap_amount=10
        )
        ax.set_ylim(0, 1)

        # Overlaying the statistics from stats_df onto the plot
        stats_text = f"Pearson: {stats_df.loc[sample, 'Pearson']:.2f}\n"
        # stats_text += f"Spearman: {stats_df.loc[sample, 'Spearman']:.2f}\n"
        # stats_text += f"RMSE: {stats_df.loc[sample, 'RMSE']:.2f}\n"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            # bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

        if idx == 0:
            ax.legend()

    # Hide any remaining empty subplots
    for idx in range(num_samples, nrows * ncols):
        axs[idx].axis("off")

    # Adjust layout for better visibility
    plt.tight_layout()
    genetools.plots.savefig(
        fig,
        clf.output_base_dir
        / f"{clf.model_file_prefix}.healthy_replicates.ridge_cv.results_per_sample.png",
        dpi=300,
    )
    plt.show()

# %%

# %%
# This plot relies on Replicate 1 and Replicate 2 labels being set consistently across specimens to the old and new copies, respectively. So the df1, df2 split logic above must be checked carefully.

# Aggregating data
means_df1 = df1[columns].mean()
means_df2 = df2[columns].mean()

# Constructing a DataFrame for aggregated data
agg_data = pd.DataFrame({"Replicate 1": means_df1, "Replicate 2": means_df2})

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))
agg_data.plot(kind="bar", ax=ax)
plt.title("Aggregated Predicted Class Probabilities Comparison")
plt.ylabel("Probability")
plt.xlabel("Disease")
# plt.xticks(rotation=45)
ax.set_xticklabels(
    [lbl.get_text().replace("_prob", "") for lbl in ax.get_xticklabels()], rotation=0
)
genetools.plots.wrap_tick_labels(
    ax=ax, wrap_x_axis=True, wrap_y_axis=False, wrap_amount=10
)
plt.tight_layout()
genetools.plots.savefig(
    fig,
    clf.output_base_dir
    / f"{clf.model_file_prefix}.healthy_replicates.ridge_cv.results_aggregate.png",
    dpi=300,
)

plt.show()

# %%

# %%
(df1["y_pred"] == df2["y_pred"]).value_counts()

# %%
# Where do predictions disagree:
# (The syntax .loc[lambda x: x] or equivalently .loc[lambda x: x == True] simply filters down to true entries)
(df1["y_pred"] != df2["y_pred"]).loc[lambda x: x]

# %%

# %%
# Among samples where the replicates agreed:
# What is the range of Pearson correlations between the two replicates' predicted class probability vectors
stats_df.loc[(df1["y_pred"] == df2["y_pred"]).loc[lambda x: x].index][
    "Pearson"
].describe()

# %%

# %%
