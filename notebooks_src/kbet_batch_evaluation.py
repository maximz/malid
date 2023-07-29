# -*- coding: utf-8 -*-
# %% [markdown]
# # Abandoned attempt with kBET's R package and scib Python wrapper
#
# We first tried to run kBET through R and a Python wrapper.
#
# Resources:
#
# - https://github.com/theislab/scib-pipeline/blob/b2ddba53016fb0574c37fed0c45cc0c490bdac7c/envs/create_conda_environments.sh
# - https://github.com/theislab/scib-pipeline/blob/b2ddba53016fb0574c37fed0c45cc0c490bdac7c/envs/scib-pipeline-R4.0.yml
# - https://github.com/theislab/scib-pipeline/blob/b2ddba53016fb0574c37fed0c45cc0c490bdac7c/envs/env_vars_activate.sh
# - https://github.com/theislab/scib-pipeline/blob/b2ddba53016fb0574c37fed0c45cc0c490bdac7c/scripts/metrics/metrics.py
# - https://github.com/theislab/scib/blob/77ab015254754baa5ca3380bd592bcc9207241de/docs/source/installation.rst
# - https://github.com/theislab/scib/blob/77ab015254754baa5ca3380bd592bcc9207241de/scib/metrics/kbet.py
# - https://github.com/theislab/scib/blob/77ab015254754baa5ca3380bd592bcc9207241de/scib/metrics/metrics.py
# - https://github.com/theislab/kBET/blob/f35171dfb04c7951b8a09ac778faf7424c4b6bc0/R/kBET-utils.R#L49
# - https://github.com/theislab/kBET/blob/f35171dfb04c7951b8a09ac778faf7424c4b6bc0/R/kBET.R#L328
#
#
# Environment setup within our main conda environment:
#
# ```bash
# conda activate cuda-env-py39; # as usual
# # install R and packages
# mamba install -c conda-forge r-base=4.1 r-essentials r-devtools r-stringi rpy2 -y;
# R --version # 4.1.1
# Rscript --version # 4.1.1
# R -e 'install.packages("tidyverse",repos = "http://cran.us.r-project.org")'
# Rscript -e "remotes::install_github('theislab/kBET')"
# pip install 'scib[rpy2] == 1.0.4'
# ```
#
# The test we ran:
#
# ```python
# # set env for rpy2. see https://github.com/theislab/scib-pipeline/blob/main/envs/env_vars_activate.sh
# import os, subprocess
#
# # in notebook, conda environment belonging to the kernel is not active in env vars, so have to set CONDA_PREFIX manually:
# os.environ['CONDA_PREFIX'] = os.path.expandvars("$HOME/anaconda3/envs/cuda-env-py39")
#
# os.environ["CFLAGS"] = subprocess.getoutput("gsl-config --cflags")
# os.environ["LDFLAGS"] = subprocess.getoutput("gsl-config --libs")
# os.environ["LD_LIBRARY_PATH"] = os.path.expandvars("${CONDA_PREFIX}/lib/R/lib/")
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ["R_HOME"] = os.path.expandvars("${CONDA_PREFIX}/lib/R")
# os.environ["R_LIBS"] = ""
#
# import rpy2 # v3.5.1
# import rpy2.robjects as robjects # this will fail if os.environ not set right
# from scib.metrics.kbet import kBET
# print(kBET.__doc__)
#
# # go with "full" type (https://github.com/theislab/scib-pipeline/blob/75ae100cf158191ee9097750e658b2f822cc837b/scripts/metrics/metrics.py#L27)
# type_ = 'full'
# embed = 'X_pca'
# verbose = True
# scaled=False # see kBET.__doc__: make it so that 0 means optimal batch mixing and 1 means low batch mixing, as described in paper
#
# from malid import io
# from malid.datamodels import GeneLocus, TargetObsColumnEnum
# adata=io.load_fold_embeddings(
#     fold_id=0,
#     fold_label='test',
#     gene_locus=GeneLocus.BCR,
#     target_obs_column=TargetObsColumnEnum.disease
# )
# batch_key = "study_name"
# label_key = "disease"
# kbet_scores_df = kBET(
#     adata=adata,
#     batch_key=batch_key,
#     label_key=label_key,
#     type_=type_,
#     embed=embed,
#     scaled=scaled,
#     verbose=verbose,
#     return_df=True
# )
# kbet_scores_df
# final_score = np.nanmean(kbet_scores_df["kBET"])
# final_score = 1 - final_score if scaled else final_score
# final_score
# ```
#
# The problem: very slow compute. Our datasets are ~15x the max dataset size at which the scib benchmarking code disables kBET.

# %%

# %% [markdown]
# # Reimplementation
#
# Instead we will reimplement based on this partial PR: https://github.com/scverse/scanpy/pull/364

# %%

# %% [markdown]
# kBET:
#
# - from https://www.nature.com/articles/s41592-018-0254-1:
#
# > In a dataset with replicates and no batch effects, the proportions of the batch labels in any neighborhood do not differ from the global distribution. In a replicated dataset with batch effects, data points from respective batches tend to cluster with their ‘peers’, and batch label proportions differ considerably between arbitrarily chosen neighborhoods
#
# > Intuitively, a replicated experiment is well mixed if a subset of neighboring samples (e.g., single-cell transcriptomic data points) has the same distribution of batch labels as the full dataset.
# >
# > In contrast, a repetition of the experiment with some bias is expected to yield a skewed distribution of batch labels across the dataset.
# >
# > kBET uses a χ2-based test for random neighborhoods of fixed size to determine whether they are well mixed, followed by averaging of the binary test results to return an overall rejection rate. This result is easy to interpret: low rejection rates imply well-mixed replicates.
#
# - from https://www.nature.com/articles/s41592-021-01336-8:
#
# > The kBET algorithm (v.0.99.6, release 4c9dafa) determines whether the label composition of a k nearest neighborhood of a cell is similar to the expected (global) label composition. The test is repeated for a random subset of cells, and the results are summarized as a rejection rate over all tested neighborhoods.
#
# > We applied kBET separately on the batch variable for each cell identity label. Using the kBET defaults, a k equal to the median of the number of cells per batch within each label was used for this computation. Additionally, we set the minimum and maximum thresholds of k to 10 and 100, respectively. As kNN graphs that have been subset by cell identity labels may no longer be connected, we computed kBET per connected component. If >25% of cells were assigned to connected components too small for kBET computation (smaller than k × 3), we assigned a kBET score of 1 to denote poor batch removal. Subsequently, kBET scores for each label were averaged and subtracted from 1 to give a final kBET score.
#

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests
import anndata
from kdict import kdict
import gc
from IPython.display import display

# %%
from malid import io, helpers, logger, config
from malid.datamodels import GeneLocus, TargetObsColumnEnum

# %%
# Use GPU for kNN construction and queries
import choosegpu

choosegpu.configure_gpu(enable=True)

# Then import relevant methods
from malid.knn import _fit_knn_index, _get_neighbors

# %%

# %%

# %%

# %%

# %%
def kbet(
    adata: anndata.AnnData,
    batch_key: str,
    label_key: str,
    significance_threshold_alpha=0.05,
):
    """
    kBET reimplementation.
    Returns average-rejection-rate averages across labels; average rejection rate by label; all corrected p values for each label (not in same order as anndata obs)
    """
    rejection_rates_by_label = {}
    corrected_p_values_by_label = {}

    # Group by label_key
    # kBET is computed separately for each cell identity label, and only considers connections between cells of the same identity.
    for label, adata_label in helpers.anndata_groupby_obs(
        adata, label_key, observed=True
    ):
        batch_sizes = adata_label.obs[batch_key].value_counts()
        if batch_sizes.index.shape[0] == 1:
            logger.info(
                f"Skipping {label_key}={label} because it has only one {batch_key} batch"
            )
            continue

        # Choose neighborhood size, based on heuristics in kBET and scib
        neighborhood_size = batch_sizes.mean() // 4
        neighborhood_size = min([50, max([10, neighborhood_size])])
        # We also considered these alternative options:
        # "a k equal to the median of the number of cells per batch within each label was used for this computation.
        # Additionally, we set the minimum and maximum thresholds of k to 10 and 100, respectively."
        # We also saw clipping between 10 and int(batch_sizes.mean() * 3/4)
        # None of these make too much sense for our case where we have batch sizes in the tens of thousands
        # Everything will default to 50. (70 was the chosen default in some implementations we saw)

        # But we are keeping this sanity-check on the size from other implementations, though it doesn't seem that important:
        size_max = np.iinfo(np.int32).max
        if neighborhood_size * adata_label.shape[0] >= size_max:
            neighborhood_size = size_max // adata_label.shape[0]

        # Unlike kBET, we are just going to run once with a fixed neighborhood size
        logger.info(
            f"Running kBET on {label_key}={label} with neighborhood size {neighborhood_size} (based on batch sizes {batch_sizes.to_dict()})"
        )

        # Compute separate kNN graph for each identity subset
        # Note: the kBET and scib-benchmarking implementations compute a global kNN graph with cells of all identity labels, then subset it for each cell identity label. This means they have to deal with edge cases when the graph is no longer connected, so it looks like they end up running on each strongly connected component separately.
        # It seems like it’d be easier to just build a separate kNN graph for each cell identity label. However, this might connect cells that would otherwise be disconnected if they live in neighborhoods dominated by other cell identity labels. For the purpose of measuring batch integration within each cell identity label, I’m not sure this is a problem.
        # We will instead build separate kNN graphs for each identity-label subset of cells

        # Build kNN index
        knn_index = _fit_knn_index(X=adata_label.obsm["X_pca"])

        # Query kNN index
        neighbor_links_df = _get_neighbors(
            knn_index=knn_index,
            data_X_contiguous=adata_label.obsm["X_pca"],
            n_neighbors=neighborhood_size,
        )

        # Note when merging:
        # center_id and neighbor_id are integer ilocs (within adata_label), not obs_names / locs

        # prepare to merge by iloc later: create sequence_info that includes the integer index
        sequence_info = adata_label.obs[[batch_key, label_key]].reset_index(drop=True)
        # make columns catgegorical:
        sequence_info[batch_key] = sequence_info[batch_key].astype("category")
        sequence_info[label_key] = sequence_info[label_key].astype("category")

        # Merge in neighbor info
        # neighbor_id is an iloc within adata_label
        neighbor_links_df = pd.merge(
            neighbor_links_df,
            sequence_info.rename(columns=lambda s: f"neighbor_{s}"),
            how="left",
            left_on="neighbor_id",
            right_index=True,
            validate="m:1",
        )

        # merge in center info
        # center_id is a iloc within adata_label
        neighbor_links_df = pd.merge(
            neighbor_links_df,
            sequence_info.rename(columns=lambda s: f"center_{s}"),
            how="left",
            left_on="center_id",
            right_index=True,
            validate="m:1",
        )

        ## Compare label composition of k-nearest neighborhood of each sequence versus the expected (global) label composition

        ## Get expected neighborhood composition from global values: normalized value counts * chosen k neighbors
        expected_frequencies = (batch_sizes / batch_sizes.sum()) * neighborhood_size
        logger.info(
            f"kBET on {label_key}={label} with neighborhood size {neighborhood_size} has expected frequencies: {expected_frequencies.to_dict()}"
        )

        ## Get observed frequencies for each cell

        # First, confirm we computed the right number of neighbors
        total_num_neighbors = (
            neighbor_links_df.groupby(["center_id"], observed=True)
            .size()
            .rename("total_num_neighbors")
        )
        assert all(total_num_neighbors == neighborhood_size)
        # Also sanity check that all identity labels are the same
        assert (
            neighbor_links_df[f"center_{label_key}"]
            == neighbor_links_df[f"neighbor_{label_key}"]
        ).all()
        assert (neighbor_links_df[f"center_{label_key}"] == label).all()

        # Get neighborhood batch frequencies for each cell
        # https://stackoverflow.com/a/39132900/130164
        observed_neighbor_batch_frequencies = (
            neighbor_links_df.groupby(
                ["center_id", f"neighbor_{batch_key}"], observed=True
            )
            .size()
            .unstack(fill_value=0)
        )

        # rearrange order to match
        assert set(observed_neighbor_batch_frequencies.columns) == set(
            expected_frequencies.index
        )
        expected_frequencies = expected_frequencies.loc[
            observed_neighbor_batch_frequencies.columns
        ]

        # Compute test statistic for each cell
        # "kBET uses a χ2-based test for random neighborhoods of fixed size to determine whether they are well mixed,
        # followed by averaging of the binary test results to return an overall rejection rate.
        # This result is easy to interpret: low rejection rates imply well-mixed replicates."
        chisquared_test_statistic, p_values_uncorrected = chisquare(
            f_obs=observed_neighbor_batch_frequencies,
            f_exp=expected_frequencies,
            axis=1,
        )

        # Correct for multiple hypothesis testing
        is_wellmixed_null_hypothesis_rejected, corrected_p_values, _, _ = multipletests(
            pvals=p_values_uncorrected,
            alpha=significance_threshold_alpha,
            method="fdr_tsbh",
        )
        # is_wellmixed_null_hypothesis_rejected is a numpy array of booleans
        rejection_rate = is_wellmixed_null_hypothesis_rejected.mean()

        # Store result for each label
        rejection_rates_by_label[label] = rejection_rate
        # note that corrected_p_values are not in order of obsnames and should thus not be glued onto adata.obs or adata_label.obs
        corrected_p_values_by_label[label] = corrected_p_values

    # Aggregate the results across labels
    # "Subsequently, kBET scores for each label were averaged and subtracted from 1 to give a final kBET score."
    # ^ This subtraction seems to only be in scib benchmarking and the scanpy partial PR.
    # Really we should keep it as rejection rate, so 0 means null hypothesis was not rejected much, suggesting the well-mixed hypothesis holds.

    # Returns average-rejection-rate averages across labels; average rejection rate by label; all corrected p values for each label (not in same order as anndata obs)
    return (
        np.mean(list(rejection_rates_by_label.values())),
        rejection_rates_by_label,
        corrected_p_values_by_label,
    )


# %%

# %%
# kBET will be computed amongst all sequences with same [label_key]
label_key = "disease"

# kBET will compute whether [batch_key] batches are well-mixed within each [label_key]
batch_key = "study_name"

# run on:
fold_label = "test"

# plot labels
xlabel = "kBET χ² test corrected p-value"
ylabel = "Disease"

# %%
kbet_results = kdict()
for gene_locus in config.gene_loci_used:
    for fold_id in config.cross_validation_fold_ids:
        logger.info(f"Running on {gene_locus}, fold {fold_id}-{fold_label}")
        adata = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=gene_locus,
            target_obs_column=TargetObsColumnEnum.disease,
        )
        # PCA was precomputed with 50 dims
        assert adata.obsm["X_pca"].shape[1] == 50
        print(adata.obs.groupby([label_key, batch_key], observed=True).size())

        # Plot kBET on each disease label
        (
            average_rejection_rate_overall,
            average_rejection_rates_by_label,
            corrected_p_values_by_label,
        ) = kbet(
            adata=adata,
            batch_key=batch_key,
            label_key=label_key,
            significance_threshold_alpha=0.05,
        )

        # Save results
        # reformat dict as dataframe
        average_rejection_rates_by_label = (
            pd.Series(average_rejection_rates_by_label)
            .rename("average_rejection_rate")
            .rename_axis(index=label_key)
            .reset_index()
            .assign(fold_id=fold_id, fold_label=fold_label, gene_locus=gene_locus.name)
        )
        kbet_results[gene_locus, fold_id, fold_label] = (
            average_rejection_rate_overall,
            average_rejection_rates_by_label,
        )

        # Print results
        print(
            f"kBET result for {gene_locus}, fold {fold_id}-{fold_label}: {average_rejection_rate_overall:0.5f} average of average rejection rate by label"
        )
        print("Average rejection rates for each label")
        display(average_rejection_rates_by_label)

        # Plot per-sequence corrected p values
        fig, ax = plt.subplots()
        sns.boxplot(
            data=pd.concat(
                [
                    pd.DataFrame(v, columns=[xlabel]).assign(**{ylabel: k})
                    for k, v in corrected_p_values_by_label.items()
                ]
            ),
            x=xlabel,
            y=ylabel,
            ax=ax,
        )
        plt.title(f"Fold {fold_id}, {fold_label} set, {gene_locus}")
        plt.xlim(-0.05, 1.05)

        # Clear cache before moving onto next fold
        io.clear_cached_fold_embeddings()
        gc.collect()

        print("*" * 60)

# %%

# %% [markdown]
# # Review all results

# %%
# Show all results as a table
all_results_overall = pd.DataFrame.from_records(
    dict(
        gene_locus=gene_locus.name,
        fold_id=fold_id,
        fold_label=fold_label,
        average_rejection_rate_overall=average_rejection_rate_overall,
    )
    for (gene_locus, fold_id, fold_label), (
        average_rejection_rate_overall,
        _,
    ) in kbet_results.items()
)
all_results_overall

# %%
all_results_overall.to_csv(
    config.paths.output_dir
    / "kbet_batch_evaluation.overall_results_by_genelocus_and_fold.tsv",
    sep="\t",
    index=None,
)

# %%
all_results_overall = pd.read_csv(
    config.paths.output_dir
    / "kbet_batch_evaluation.overall_results_by_genelocus_and_fold.tsv",
    sep="\t",
)
all_results_overall

# %%
# Remember: 0 means optimal batch mixing and 1 means low batch mixing
# For each locus+fold, we took average across all cells within each label, then average across all labels
# And now we can report average +/- std across folds by locus+disease, or by locus, or overall

# %%
# agg by locus
all_results_overall.groupby("gene_locus")["average_rejection_rate_overall"].agg(
    ["mean", "std"]
)

# %%
# agg over all
all_results_overall["average_rejection_rate_overall"].agg(["mean", "std"])

# %%

# %%

# %%
# All results at per-label specificity (i.e. before we take average over all labels)
all_results_per_label = pd.concat(
    (
        average_rejection_rates_by_label
        for (_, average_rejection_rates_by_label) in kbet_results.values()
    ),
    axis=0,
).reset_index(drop=True)
all_results_per_label

# %%
all_results_per_label.to_csv(
    config.paths.output_dir
    / "kbet_batch_evaluation.per_label_results_by_genelocus_and_fold.tsv",
    sep="\t",
    index=None,
)

# %%
all_results_per_label = pd.read_csv(
    config.paths.output_dir
    / "kbet_batch_evaluation.per_label_results_by_genelocus_and_fold.tsv",
    sep="\t",
)
all_results_per_label

# %%
all_results_per_label.groupby(["gene_locus", "disease"])["average_rejection_rate"].agg(
    ["mean", "std"]
)

# %%
all_results_per_label.groupby(["gene_locus"])["average_rejection_rate"].agg(
    ["mean", "std"]
)

# %%
all_results_per_label["average_rejection_rate"].agg(["mean", "std"])

# %%

# %%

# %%

# %%
