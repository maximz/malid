from typing import Dict, List, Mapping, Optional, Tuple, Any
import logging
from pathlib import Path
from itertools import zip_longest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import seaborn as sns

sns.set_style("ticks")

import itertools
import gc

from kdict import kdict
import genetools
import sklearn.metrics
import scipy.stats
from statannotations.Annotator import Annotator

from malid import config, io, helpers
from malid.trained_model_wrappers import SequenceClassifier, ConvergentClusterClassifier

from malid.datamodels import (
    TargetObsColumnEnum,
    SampleWeightStrategy,
    GeneLocus,
    healthy_label,
)
import string
import matplotlib.transforms as mtransforms

logger = logging.getLogger(__name__)

# TODO: replace with kdict type hint when that's merged


def rank_sequences(
    gene_locus: GeneLocus, target_obs_columns: List[TargetObsColumnEnum]
) -> Mapping[Tuple[TargetObsColumnEnum, Any], pd.DataFrame]:
    """
    For each classification target, for each possible value it takes:
    Get predicted probabilities of that value for all sequences from specimens with that value.

    Example: sequence Covid predictions for all sequences from Covid patients.

    Combines uniques from all our test folds.

    Returns kdict mapping [target_obs_column, target_value] to pandas DataFrame.
    """
    scored = kdict()  # [target_obs_column, target_value] -> list of scores

    # exclude global fold ID, which does not have a test set (and because we want each person once)
    for fold_id in config.cross_validation_fold_ids:
        # Optimization: data for this fold is cached by load_fold_embeddings() between accesses of different target obs columns.
        for target_obs_column in target_obs_columns:
            adata = io.load_fold_embeddings(
                fold_id=fold_id,
                fold_label="test",
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
            )

            # load individual sequence classifier
            try:
                clf = SequenceClassifier(
                    fold_id=fold_id,
                    model_name_sequence_disease="lasso_multiclass",
                    fold_label_train="train_smaller",
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_column,
                    sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
                )
            except FileNotFoundError as err:
                logger.warning(
                    f"Skipping fold {fold_id}, {target_obs_column}, {gene_locus} because sequence classifier was not trained: {err}"
                )
                continue

            target_obs_column_name = target_obs_column.value.obs_column_name
            for target_value, adata_part in helpers.anndata_groupby_obs(
                adata, target_obs_column_name, observed=True
            ):
                # Get sequences originating from specimens with target_obs_column_name = target_value

                if target_value not in clf.classes_:
                    # skip sequences originating from specimens with this target value,
                    # because we will not score these with the present classifier.
                    logger.info(
                        f"For {target_obs_column}, skipping sequences from specimens with {target_obs_column_name}={target_value}. That is not one of classifier's classes: {clf.classes_}"
                    )
                    continue

                # score the sequences
                # doesn't matter if we use adjusted decision thresholds because that is just reweighting the entire class by a factor. rankings within the class won't change (unless we renormalize the rows)
                featurized = clf.featurize(adata_part)
                scores_for_target = pd.DataFrame(
                    clf.predict_proba(
                        featurized.X,
                    ),
                    index=featurized.sample_names,
                    columns=clf.classes_,
                )
                # get predicted probabilities for this target label
                scores_for_specific_target_value = scores_for_target[target_value]

                # combine across folds - get V, J, CDR3 and score.
                # don't store specimen information - aggregate across specimens
                if (target_obs_column, target_value) not in scored:
                    # initialize as if it were a defaultdict(list)
                    scored[target_obs_column, target_value] = []
                scored[target_obs_column, target_value].append(
                    pd.concat(
                        [
                            adata_part.obs[
                                [
                                    "v_gene",
                                    "j_gene",
                                    "cdr1_seq_aa_q_trim",
                                    "cdr2_seq_aa_q_trim",
                                    "cdr3_seq_aa_q_trim",
                                    "cdr3_aa_sequence_trim_len",
                                    "isotype_supergroup",
                                ]
                            ],
                            scores_for_specific_target_value,
                        ],
                        axis=1,
                    )
                )

        # Prepare to move on to next fold: clear cache
        io.clear_cached_fold_embeddings()
        gc.collect()

    # For each target,
    # for each target_value,
    # combine all test set sequences from targetvalue-originating specimens with their targetvalue scores.
    scored_sequence_dfs = kdict(
        {
            (target_obs_column, target_value): pd.concat(scored, axis=0).reset_index(
                drop=True
            )  # reset_index so that later uses of idxmax return unique index values
            for (target_obs_column, target_value), scored in scored.items()
        }
    )

    original_shapes = kdict()
    for (
        target_obs_column,
        target_value,
    ), scored_sequence_df in scored_sequence_dfs.items():
        if scored_sequence_df.isna().any().any():
            raise ValueError(f"Nulls for {target_obs_column}, {target_value}")
        logger.info(
            f"{target_obs_column}, {target_value}: {scored_sequence_df.shape} shape"
        )
        original_shapes[target_obs_column, target_value] = scored_sequence_df.shape[0]
    del scored
    gc.collect()

    ###

    ## Now combine and get mean score for exact Vgene+CDR123+isotype dupes from multiple specimens.
    # they may not be identically scored if they are in different folds (different models):
    # when completely identical sequences are found in separate specimens, they should be scored the same,
    # unless they come from different folds, i.e. might be scored by different models.

    # therefore dedupe. hide specimen label, choose highest score given by any model to a particular isotype-cdr123-v-j combination. i.e. choose best score when seen in any specimen
    # then consider all sequences, even if you see the same sequence in multiple isotypes

    # (afterwards, any remaining variation in predicted probability for same V-J-CDR3 will come from different CDR1/2, or different isotypes.)

    # get full row for maximal entry per group
    # take max predicted probability for sequences with same parameters, which may come from different specimens in different folds
    # alternative: take mean. but max makes more sense here: we want "how good could this sequence possibly be"
    # make sure we ran reset_index() above, so that idxmax returns unique index values: https://stackoverflow.com/a/32459442/130164
    scored_sequence_dfs.update(
        {
            (
                target_obs_column,
                target_value,
            ): all_scored_sequences_for_targetvalue_df.loc[
                all_scored_sequences_for_targetvalue_df.groupby(
                    [
                        "v_gene",
                        "j_gene",
                        "cdr1_seq_aa_q_trim",
                        "cdr2_seq_aa_q_trim",
                        "cdr3_seq_aa_q_trim",
                        "isotype_supergroup",
                    ],
                    observed=True,
                )[target_value].idxmax()
            ]
            for (
                target_obs_column,
                target_value,
            ), all_scored_sequences_for_targetvalue_df in scored_sequence_dfs.items()
        }
    )

    # Check final vs original shape
    for (
        target_obs_column,
        target_value,
    ), all_scored_sequences_for_targetvalue_df in scored_sequence_dfs.items():
        new_shape = all_scored_sequences_for_targetvalue_df.shape[0]
        original_shape = original_shapes[target_obs_column, target_value]
        logger.info(
            f"{target_obs_column}, {target_value} deduplicated from {original_shape} to {new_shape} rows"
        )
        # sanity check
        if new_shape > original_shape:
            raise ValueError(
                f"More rows after deduplication for {target_obs_column}, {target_value}"
            )

    ###

    ## Assign ranks
    for (
        target_obs_column,
        target_value,
    ), all_scored_sequences_for_targetvalue_df in scored_sequence_dfs.items():
        # assign a rank (higher ranks are higher probabilities)
        all_scored_sequences_for_targetvalue_df.sort_values(
            target_value, ascending=True, inplace=True
        )
        all_scored_sequences_for_targetvalue_df[
            "rank"
        ] = genetools.stats.rank_normalize(
            all_scored_sequences_for_targetvalue_df[target_value]
        )
        # percentile normalize
        all_scored_sequences_for_targetvalue_df["rank"] = (
            all_scored_sequences_for_targetvalue_df["rank"]
            / all_scored_sequences_for_targetvalue_df.shape[0]
        )

    return scored_sequence_dfs


def filter_v_genes(
    gene_locus: GeneLocus,
    ranked_sequence_dfs: Mapping[
        Tuple[TargetObsColumnEnum, Any],
        pd.DataFrame,
    ],
) -> Tuple[pd.Series, matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """how rare / common is each V gene? find and exclude very rare V genes."""
    # Get V gene frequencies (using entire data - i.e. TargetObsColumnEnum.disease)
    v_gene_frequencies = {}
    for (_, disease), disease_sequences_df in ranked_sequence_dfs[
        TargetObsColumnEnum.disease, :
    ].items():
        v_gene_frequencies[disease] = disease_sequences_df["v_gene"].value_counts(
            normalize=True
        )
    v_gene_frequencies = pd.DataFrame(v_gene_frequencies)
    v_gene_frequencies_melt = (
        v_gene_frequencies.rename_axis(index="V gene")
        .reset_index()
        .melt(
            id_vars="V gene",
            value_vars=v_gene_frequencies.columns,
            var_name="disease",
            value_name="proportion",
        )
        .fillna(0)
    )

    # # stacked bar plot, each V gene on y axis, how much of each disease it makes up on x-axis as bar
    # # un-normalized, so you can see which V genes are rare vs common
    # fig1, ax = genetools.plots.stacked_bar_plot(
    #     data=v_gene_frequencies_melt,
    #     index_key="V gene",
    #     hue_key="disease",
    #     value_key="proportion",
    #     normalize=False,
    #     figsize=(8, 20),
    #     vertical=False,
    #     palette=helpers.disease_color_palette,
    #     axis_label="proportion of each disease belonging to this V gene",
    # )
    # plt.title(f"All V genes - {gene_locus}")

    # max proportion a given V gene takes up of any disease
    v_gene_max_frequency = v_gene_frequencies.max(axis=1).sort_values()

    # Define criteria for filtering below
    # using median here. or consider setting to a raw value like 0.005? works out identically:
    v_gene_frequency_threshold = v_gene_max_frequency.quantile(0.5)

    fig_unfiltered, axarr = plt.subplots(
        nrows=1,
        ncols=v_gene_frequencies.shape[1],
        figsize=(3 * v_gene_frequencies.shape[1], 12),
        sharex=True,  # Make xlims consistent for better readability
        sharey=False,  # Repeat the V gene in each axis for better readability
    )
    v_gene_order = pd.Series(sorted(v_gene_frequencies.index.unique()))
    for (disease, ax) in zip(sorted(v_gene_frequencies.columns), axarr):
        data = (
            v_gene_frequencies[disease]
            .fillna(0)
            .rename_axis(index="V gene")
            .reset_index()
        )
        # Switch to friendly V gene names
        data["V gene"] = data["V gene"].replace(helpers.v_gene_friendly_names)
        sns.barplot(
            data=data,
            x=disease,
            y="V gene",
            order=v_gene_order.replace(helpers.v_gene_friendly_names),
            ax=ax,
            color=helpers.disease_color_palette[disease],
        )
        # To be kept, a V gene must exceed this dotted line in at least one disease
        ax.axvline(
            x=v_gene_frequency_threshold,
            zorder=10,
            linewidth=2,
            linestyle="dashed",
            color="purple",
        )
        ax.set_title(disease, fontweight="bold")
        ax.set_xlabel("Proportion")
        ax.set_ylabel(None)
        sns.despine(ax=ax)

    axarr[0].set_ylabel("V gene")
    fig_unfiltered.suptitle(f"V gene disease proportions - {gene_locus}")
    plt.tight_layout()

    logger.info(
        f"{gene_locus} criteria for keeping a V gene: in the disease in which it is most prevalent, it must make up at least {v_gene_frequency_threshold*100:0.1f}% of the disease dataset"
    )
    # Which V genes does this leave?
    v_gene_overall_frequency_filter = (
        v_gene_frequencies.max(axis=1) > v_gene_frequency_threshold
    )
    logger.info(
        f"{gene_locus} V genes rejected for rarity: {', '.join(v_gene_overall_frequency_filter[~v_gene_overall_frequency_filter].index)}"
    )

    v_gene_order_filtered = v_gene_order[
        v_gene_order.isin(
            v_gene_overall_frequency_filter[v_gene_overall_frequency_filter].index
        )
    ]
    # v_gene_order_filtered.shape, v_gene_order.shape

    # Now plot V gene disease proportions only for remaining V genes
    # stacked bar plot, each V gene on y axis, how much of each disease it makes up on x-axis as bar
    # normalized: hides V-gene rarity, but accentuates differences between diseases
    data = v_gene_frequencies_melt[
        v_gene_frequencies_melt["V gene"].isin(v_gene_order_filtered)
    ]
    fig_filtered, ax = genetools.plots.stacked_bar_plot(
        data=data.assign(
            **{"V gene": data["V gene"].replace(helpers.v_gene_friendly_names)}
        ).sort_values("V gene"),
        index_key="V gene",
        hue_key="disease",
        value_key="proportion",
        normalize=True,
        figsize=(8, 12),
        vertical=False,
        palette=helpers.disease_color_palette,
        axis_label="Disease makeup of all usage of this V gene",
        legend_title="Disease",
    )
    plt.title(f"Filtered V gene disease proportions - {gene_locus}")

    return v_gene_order_filtered.rename("v_gene"), fig_unfiltered, fig_filtered


###


def plot_v_gene_rankings(
    target_obs_column: TargetObsColumnEnum,
    ranked_sequence_dfs: Mapping[
        Tuple[TargetObsColumnEnum, Any],
        pd.DataFrame,
    ],
    figsize: Tuple[float, float],
    filtered_v_gene_list: Optional[pd.Series] = None,
    exclude_target_values: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """One plot per target value (e.g. per disease), showing V gene rankings for each target value, with different sort orders. Split into rows."""

    # Keep all for this target obs column, or keep all but excluded target values
    if exclude_target_values is None:
        exclude_target_values = []

    selected_targetvalue_sequence_dfs = {
        target_value: ranked_sequence_df
        for (_, target_value), ranked_sequence_df in ranked_sequence_dfs[
            target_obs_column, :
        ].items()
        if target_value not in exclude_target_values
    }

    # support uneven subplot count per row: https://stackoverflow.com/a/53840947/130164
    n_subplots_desired = len(selected_targetvalue_sequence_dfs)
    desired_n_cols = 2
    nrows = int(np.ceil(n_subplots_desired / desired_n_cols))

    # this returns nested axarr for rows
    fig, axarr = plt.subplots(
        nrows=nrows,
        ncols=desired_n_cols,
        # autosize
        figsize=(figsize[0], figsize[1] * nrows),
        sharey=False,  # different yaxes orders for sure.
        sharex=False,  # technically will share X, but disable X so that tick labels aren't auto-hidden (or not even created) for all but final row. see workaround below
        squeeze=False,  # always return 2D axarr, even if only one row
    )

    # flatten array of axes
    axarr_flattened = axarr.flatten(order="C")  # F is col_wise, 'C' is row_wise order

    # remove unused axes, if any
    for ax in axarr_flattened[n_subplots_desired:]:
        fig.delaxes(ax)

    for (target_value, ranked_sequences_df), ax in zip(
        sorted(selected_targetvalue_sequence_dfs.items()), axarr_flattened
    ):
        v_gene_order = (
            ranked_sequences_df.groupby("v_gene", observed=True)["rank"]
            .median()
            .sort_values()
            .index
        )
        # filter down
        if filtered_v_gene_list is not None:
            v_gene_order = v_gene_order[v_gene_order.isin(filtered_v_gene_list)]

        # Switch to friendly V gene names
        v_gene_order_friendly_names = pd.Series(v_gene_order).replace(
            helpers.v_gene_friendly_names
        )

        sns.boxplot(
            data=ranked_sequences_df.assign(
                v_gene=ranked_sequences_df["v_gene"].replace(
                    helpers.v_gene_friendly_names
                )
            ),
            y="v_gene",
            x="rank",
            ax=ax,
            order=v_gene_order_friendly_names,
        )

        # Make y ticks bold
        ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

        # higher rank means higher confidence
        ax.set_xticks(ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(
            labels=[
                "0 (worst)",
                0.25,
                0.50,
                0.75,
                "1 (best)",
            ],
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        ax.set_title(target_value, fontweight="bold", fontsize="x-large")
        sns.despine(ax=ax)

    # Add ylabel to first axes in each row
    for axarr_row in axarr:
        axarr_row[0].set_ylabel("V gene")

    # Add xlabel to last axes in each column (which is not necessarily in the final row)
    for col_ix in range(desired_n_cols):
        # decide whether final row for this column was the penultimate row of the plot:
        # check if [number of desired total subplots] % [number of columns desired] > 0, and whether we are at that cutoff
        # (and handle edge case where we have one row total, in which case don't switch labeling to penultimate row because it doesn't exist)
        final_row_for_this_column = (
            -2
            if (
                col_ix >= n_subplots_desired % desired_n_cols > 0 and axarr.shape[0] > 1
            )
            else -1
        )
        final_ax_for_this_column = axarr[final_row_for_this_column, col_ix]

        final_ax_for_this_column.set_xlabel(
            "Sequence prediction confidence\n(percentile of rank)"
        )

        # Also, accomplish sharex=True, but workaround for final row not having all columns:

        # turn tick labels on for bottommost entry for this column
        for tick_label in final_ax_for_this_column.get_xticklabels():
            tick_label.set_visible(True)

        # turn off tick labels for all axes above this in this column
        for ax in axarr[:final_row_for_this_column, col_ix]:
            for tick_label in ax.get_xticklabels():
                tick_label.set_visible(False)

    # Label subplot panels for figure
    for letter_label, ax in zip(string.ascii_lowercase, axarr_flattened):
        # https://matplotlib.org/3.5.1/gallery/text_labels_and_annotations/label_subplots.html
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.0,
            letter_label,
            transform=ax.transAxes + trans,
            fontsize="x-large",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    return fig


# Subset CDR3 lengths to those that are present in all target values in substantial amounts
def _get_cdr3_lengths_of_reasonable_size(
    ranked_sequences_df: pd.DataFrame, size_threshold: int
):
    sizes = ranked_sequences_df.groupby("cdr3_aa_sequence_trim_len").size()
    sizes = sizes[sizes >= size_threshold]
    return sizes.index.values


def plot_cdr3_length_vs_rank(
    target_obs_column: TargetObsColumnEnum,
    ranked_sequence_dfs: Mapping[
        Tuple[TargetObsColumnEnum, Any],
        pd.DataFrame,
    ],
    cdr3_length_palette: Dict[int, str],
    figsize: Tuple[float, float],
    cdr3_size_threshold=50,
) -> matplotlib.figure.Figure:
    """boxplots. all one row."""
    # warning: this hides different sample sizes per category

    # filter to this target
    selected_ranked_sequence_dfs = ranked_sequence_dfs[target_obs_column, :]
    fig, axarr = plt.subplots(
        nrows=1,
        ncols=len(selected_ranked_sequence_dfs),
        figsize=figsize,
        sharey=True,
        sharex=False,
    )

    # Subset CDR3 lengths to those that are present in all target values in substantial amounts
    cdr3_length_order_subset = sorted(
        list(
            set.intersection(
                *[
                    set(
                        _get_cdr3_lengths_of_reasonable_size(
                            ranked_sequences_df, size_threshold=cdr3_size_threshold
                        )
                    )
                    for ranked_sequences_df in selected_ranked_sequence_dfs.values()
                ]
            )
        )
    )
    # fill in any skips
    cdr3_length_order_subset = np.arange(
        min(cdr3_length_order_subset), max(cdr3_length_order_subset) + 1
    )

    for ((_, target_value), ranked_sequences_df), ax in zip(
        sorted(selected_ranked_sequence_dfs.items()), axarr
    ):
        sns.boxplot(
            data=ranked_sequences_df,
            x="cdr3_aa_sequence_trim_len",
            y="rank",
            orient="v",
            ax=ax,
            order=cdr3_length_order_subset,  # cdr3_length_order,
            palette=cdr3_length_palette,
        )

        # higher rank means higher confidence
        ax.set_yticks(ticks=[0.0, 0.5, 1.0])
        ax.set_yticklabels(
            labels=[
                "0 (worst)",
                0.50,
                "1 (best)",
            ],
        )
        ax.set_xlabel("CDR3 length", fontweight="bold", size="large")
        ax.set_ylabel(None)

        ax.set_title(target_value, fontweight="bold", fontsize="x-large")
        sns.despine(ax=ax)

    axarr[0].set_ylabel("Sequence prediction rank", fontweight="bold", size="large")

    # Label subplot panels for figure
    for letter_label, ax in zip(string.ascii_lowercase, axarr):
        # https://matplotlib.org/3.5.1/gallery/text_labels_and_annotations/label_subplots.html
        # label physical distance to the left and up:
        # trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.05,
            letter_label,
            transform=ax.transAxes,  # + trans,
            fontsize="large",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    return fig


def plot_isotype_use_vs_ranks(
    target_obs_column: TargetObsColumnEnum,
    ranked_sequence_dfs: Mapping[
        Tuple[TargetObsColumnEnum, Any],
        pd.DataFrame,
    ],
    figsize: Tuple[float, float],
) -> matplotlib.figure.Figure:
    """note: see how isotypes were chosen above - same seq may be in multiple isotypes and get diff ranks. we only keep the top version of it. so this is an analysis of isotype used in top-ranked version of any sequence"""

    # filter to this target
    selected_ranked_sequence_dfs = ranked_sequence_dfs[target_obs_column, :]

    # uneven subplot count per row: https://stackoverflow.com/a/53840947/130164
    n_subplots_desired = len(selected_ranked_sequence_dfs)
    desired_n_cols = 3
    nrows = int(np.ceil(n_subplots_desired / desired_n_cols))

    # this returns nested axarr for rows
    fig, axarr = plt.subplots(
        nrows=nrows,
        ncols=desired_n_cols,
        # autosize
        figsize=(figsize[0], figsize[1] * nrows),
        sharey=True,
        sharex=False,
        squeeze=False,  # always return 2D axarr, even if only one row
    )

    # flatten array of axes
    axarr_flattened = axarr.flatten(order="C")  # F is col_wise, 'C' is row_wise order

    # remove unused axes
    for ax in axarr_flattened[n_subplots_desired:]:
        fig.delaxes(ax)

    # within each subplot, we will show comparisons between all pairs of isotypes. but do multiple hypothesis correction across all comparisons from all subplots
    isotype_pairs_to_compare = list(
        itertools.combinations(helpers.isotype_friendly_name_order, 2)
    )
    # print("isotype_pairs_to_compare:", isotype_pairs_to_compare)
    total_num_comparisons = len(isotype_pairs_to_compare) * n_subplots_desired
    # print("total_num_comparisons:", total_num_comparisons)

    # plot
    for ((_, target_value), ranked_sequences_df), ax in zip(
        sorted(selected_ranked_sequence_dfs.items()), axarr_flattened
    ):
        ranked_sequences_df["Isotype"] = ranked_sequences_df[
            "isotype_supergroup"
        ].replace(helpers.isotype_friendly_names)
        sns.boxplot(
            data=ranked_sequences_df,
            x="Isotype",
            y="rank",
            ax=ax,
            order=helpers.isotype_friendly_name_order,
            palette=helpers.isotype_palette,
        )

        # Annotate with statistical significance, two-sided Mann-Whitney test with Bonferroni correction
        # Calculate p-values for each pair, and correct

        # Annotate but only for pairs that are significant:
        # Run test manually, then run annotator only on significant pairs. Specify desired number of corrections. Confirm p-values match.
        # see also https://github.com/trevismd/statannotations/issues/49

        significant_comparisons = kdict()
        # print(target_value)
        for (group1, group2) in isotype_pairs_to_compare:
            # Reproduce the test ourselves: Wilcoxon rank-sum test, two sided.
            # The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution.
            # The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.

            side1 = ranked_sequences_df[ranked_sequences_df["Isotype"] == group1][
                "rank"
            ].values
            side2 = ranked_sequences_df[ranked_sequences_df["Isotype"] == group2][
                "rank"
            ].values
            significance_test = scipy.stats.mannwhitneyu(
                side1, side2, alternative="two-sided"
            )
            # apply bonferroni multiple hypothesis correction
            # thresholding at alpha = 0.05/n_comparisons is like multiplying pvalue by n_comparisons
            # this is consistent with statsmodels.stats.multitest.multipletests used in Annotator:
            # multipletests(np.pad([p_value], (0, n_comparisons-1), mode="constant", constant_values=1), alpha=0.05, method='bonferroni') will multiply p_value by n_comparisons
            corrected_p_value = significance_test.pvalue * total_num_comparisons
            # clip p value to (0,1)
            corrected_p_value = np.clip(corrected_p_value, 0.0, 1.0)
            # print(f"{group1} / {group2}: {corrected_p_value:0.2e}")
            if corrected_p_value <= 0.05:
                significant_comparisons[group1, group2] = corrected_p_value

        # print(significant_comparisons)

        if len(significant_comparisons) > 0:
            annot = Annotator(
                ax=ax,
                pairs=list(
                    significant_comparisons.keys()
                ),  # rather than on full list of pairs (isotype_pairs_to_compare)
                data=ranked_sequences_df,
                x="Isotype",
                y="rank",
                order=helpers.isotype_friendly_name_order,
            )
            annot.configure(
                test="Mann-Whitney",
                comparisons_correction="bonferroni",
                #         correction_format="replace",
                text_format="star",
                loc="inside",
                verbose=2,
            )
            annot.apply_test(num_comparisons=total_num_comparisons)
            ax, test_results = annot.annotate()
            # print()

            # Confirm p-values match what is expected
            if len(test_results) != len(significant_comparisons):
                raise ValueError("Annotator produced incorrect number of test results")
            for res in test_results:
                p_value_calculated_by_us = significant_comparisons[
                    res.data.group1, res.data.group2
                ]
                p_value_calculated_by_library = res.data.pvalue
                if p_value_calculated_by_library != p_value_calculated_by_us:
                    raise ValueError(
                        "P-value mismatch:",
                        res.data.group1,
                        res.data.group2,
                        p_value_calculated_by_library,
                        p_value_calculated_by_us,
                    )

        # add n= for each category
        ax.set_xticklabels(
            genetools.plots.add_sample_size_to_labels(
                ax.get_xticklabels(), ranked_sequences_df, "Isotype"
            )
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(target_value, fontweight="bold")
        sns.despine(ax=ax)

    # Add ylabel to first axes in each row
    for axarr_row in axarr:
        # higher rank means higher confidence
        axarr_row[0].set_ylabel("Sequence prediction confidence\n(percentile of rank)")
        axarr_row[0].set_yticks(ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        axarr_row[0].set_yticklabels(
            labels=[
                "0 (worst)",
                0.25,
                0.50,
                0.75,
                "1 (best)",
            ],
        )

    # Add xlabel to last axes in each column (which is not necessarily in the final row)
    for col_ix in range(desired_n_cols):
        # decide whether final row for this column was the penultimate row of the plot:
        # check if [number of desired total subplots] % [number of columns desired] > 0, and whether we are at that cutoff
        # (and handle edge case where we have one row total, in which case don't switch labeling to penultimate row because it doesn't exist)
        final_row_for_this_column = (
            -2
            if (
                col_ix >= n_subplots_desired % desired_n_cols > 0 and axarr.shape[0] > 1
            )
            else -1
        )
        final_ax_for_this_column = axarr[final_row_for_this_column, col_ix]
        final_ax_for_this_column.set_xlabel("Isotype")

    # Label subplot panels for figure
    for letter_label, ax in zip(string.ascii_lowercase, axarr_flattened):
        # https://matplotlib.org/3.5.1/gallery/text_labels_and_annotations/label_subplots.html
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.0,
            letter_label,
            transform=ax.transAxes + trans,
            fontsize="large",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def plot_all_interpretations(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    ranked_sequence_dfs: Mapping[
        Tuple[TargetObsColumnEnum, Any],
        pd.DataFrame,
    ],
    cdr3_length_palette: Dict[int, str],
    figsize: Tuple[float, float],
    filtered_v_gene_list: Optional[pd.Series] = None,
    exclude_target_values: Optional[List[str]] = None,
    cdr3_size_threshold=100,
) -> matplotlib.figure.Figure:
    GeneLocus.validate_single_value(gene_locus)

    # Keep all for this target obs column, or keep all but excluded target values
    if exclude_target_values is None:
        exclude_target_values = []

    selected_targetvalue_sequence_dfs = {
        target_value: ranked_sequence_df
        for (_, target_value), ranked_sequence_df in ranked_sequence_dfs[
            target_obs_column, :
        ].items()
        if target_value not in exclude_target_values
    }

    ## Design figure

    # We are going to have several rows,
    # each of which is divided into groups of two columns.

    # I.e. each targetvalue will have two columns, we have a certain number of these targetvalues per row,
    # and next set of targetvalues go to next row.

    # For future matplotlib versions, consider this alternative: https://matplotlib.org/stable/tutorials/intermediate/arranging_axes.html#nested-axes-layouts
    # ( Or https://stackoverflow.com/a/46473139/130164 ?)

    # size of each column group: how many columns allocated per targetvalue
    ncols_group_size = 2

    # number of these ncols_group_size-sized column groups: number of targetvalues shown per row
    ncols_ngroups = 2

    targetvalue_order = sorted([k for k in selected_targetvalue_sequence_dfs.keys()])

    # number of total rows
    # (some rows may be incomplete)
    nrows = int(np.ceil(len(targetvalue_order) / ncols_ngroups))

    # Also add spacing between column-groups with an invisible middle axes object: https://stackoverflow.com/a/53643819/130164
    col_widths = []
    for _ in range(ncols_ngroups):
        col_widths.extend([1] * ncols_group_size)
        col_widths.append(0.05)  # separator

    # remove last. no separator after final group in each row.
    col_widths = col_widths[:-1]
    assert len(col_widths) == ncols_ngroups * (ncols_group_size + 1) - 1

    fig, axarr = plt.subplots(
        nrows=nrows,
        ncols=len(col_widths),
        gridspec_kw=dict(width_ratios=col_widths),
        # autosize
        figsize=(figsize[0], figsize[1] * nrows),
        sharex=False,
        sharey=False,
        squeeze=False,  # always return 2D axarr, even if only one row
    )  # , hspace=1, wspace=1

    # get figure's gridspec - it already made the subplots for us
    gs = axarr[0, 0].get_gridspec()

    # # Alternative would be to make a gridspec and subplots manually, something like:
    # fig = plt.figure()
    # gs = gridspec.GridSpec(nrows=2, ncols=4)
    # axarr = kdict()
    # for row_ix in range(2):
    #     for col_ix in range(4):
    #         axarr[row_ix, col_ix] = fig.add_subplot(gs[row_ix, col_ix])

    # Title the groups of columns devoted to each targetvalue
    def iter_over_column_groups(nrows, ncols_ngroups, ncols_group_size):
        for row_ix in range(nrows):
            for ngroup in range(ncols_ngroups):
                group_starting_ix = ngroup * (ncols_group_size + 1)
                yield row_ix, group_starting_ix

    for (row_ix, group_starting_ix), targetvalue in zip_longest(
        iter_over_column_groups(nrows, ncols_ngroups, ncols_group_size),
        targetvalue_order,
    ):
        if targetvalue is not None:
            # Add ghost axes and titles - see https://stackoverflow.com/a/69117807/130164
            ax_ghost = fig.add_subplot(
                gs[row_ix, group_starting_ix : group_starting_ix + ncols_group_size]
            )
            ax_ghost.axis("off")
            ax_ghost.set_title(
                targetvalue, fontweight="bold", fontsize="x-large", y=1.02
            )
        else:
            # we have extra axes beyond the number of targetvalues (i.e. one row is uneven)
            # remove these unused axes
            for ax in axarr[
                row_ix, group_starting_ix : group_starting_ix + ncols_group_size
            ]:
                fig.delaxes(ax)

        # make preceding separator invisible (as long as this isn't the first group in the row)
        preceding_separator_colix = group_starting_ix - 1
        if preceding_separator_colix > 0:
            axarr[row_ix, preceding_separator_colix].set_visible(False)

    # Get axarr for each type of plot.
    # In each group of two cols, first col is V genes, second col is CDR3 lengths.
    axarr_vgenes = [
        axarr[row_ix, group_starting_ix]
        for (row_ix, group_starting_ix) in iter_over_column_groups(
            nrows, ncols_ngroups, ncols_group_size
        )
    ]
    axarr_cdr3_lengths = [
        axarr[row_ix, group_starting_ix + 1]
        for (row_ix, group_starting_ix) in iter_over_column_groups(
            nrows, ncols_ngroups, ncols_group_size
        )
    ]

    # If we had an uneven row: Claw back any now-deleted axes
    axarr_vgenes = axarr_vgenes[: len(targetvalue_order)]
    axarr_cdr3_lengths = axarr_cdr3_lengths[: len(targetvalue_order)]

    assert len(axarr_vgenes) == len(axarr_cdr3_lengths) == len(targetvalue_order)

    ##########

    ## V genes

    for targetvalue, ax in zip(targetvalue_order, axarr_vgenes):
        targetvalue_sequences_df = selected_targetvalue_sequence_dfs[targetvalue]

        v_gene_order = (
            targetvalue_sequences_df.groupby("v_gene", observed=True)["rank"]
            .median()
            .sort_values()
            .index
        )
        # filter down
        if filtered_v_gene_list is not None:
            v_gene_order = v_gene_order[v_gene_order.isin(filtered_v_gene_list)]

        # Switch to friendly V gene names
        v_gene_order_friendly_names = pd.Series(v_gene_order).replace(
            helpers.v_gene_friendly_names
        )

        sns.boxplot(
            data=targetvalue_sequences_df.assign(
                v_gene=targetvalue_sequences_df["v_gene"].replace(
                    helpers.v_gene_friendly_names
                )
            ),
            y="v_gene",
            x="rank",
            ax=ax,
            order=v_gene_order_friendly_names,
        )

        # Make x ticks bold
        ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")  # , rotation=90

        # higher rank means higher confidence
        ax.set_xticks(ticks=[0.0, 0.5, 1.0])
        ax.set_xticklabels(
            labels=[
                "0 (worst)",
                0.50,
                "1 (best)",
            ],
        )
        ax.set_xlabel("Sequence prediction rank", fontweight="bold", size="large")
        ax.set_ylabel(None)  # ("V gene", fontweight="bold", size="large")

        sns.despine(ax=ax)

    # axarr_vgenes[0].set_ylabel("Sequence prediction rank", fontweight="bold", size="large")

    ##########

    ## CDR3 lengths

    # Subset CDR3 lengths to those that are present in all targetvalues in substantial amounts
    cdr3_length_order_subset = sorted(
        list(
            set.intersection(
                *[
                    set(
                        _get_cdr3_lengths_of_reasonable_size(
                            targetvalue_sequences_df, size_threshold=cdr3_size_threshold
                        )
                    )
                    for targetvalue_sequences_df in selected_targetvalue_sequence_dfs.values()
                ]
            )
        )
    )
    # fill in any skips
    cdr3_length_order_subset = np.arange(
        min(cdr3_length_order_subset), max(cdr3_length_order_subset) + 1
    )

    for targetvalue, ax in zip(targetvalue_order, axarr_cdr3_lengths):
        targetvalue_sequences_df = selected_targetvalue_sequence_dfs[targetvalue]

        sns.boxplot(
            data=targetvalue_sequences_df,
            y="cdr3_aa_sequence_trim_len",
            x="rank",
            orient="h",
            ax=ax,
            order=reversed(cdr3_length_order_subset),  # cdr3_length_order,
            palette=cdr3_length_palette,
        )

        # higher rank means higher confidence
        ax.set_xticks(ticks=[0.0, 0.5, 1.0])
        ax.set_xticklabels(
            labels=[
                "0 (worst)",
                0.50,
                "1 (best)",
            ],
        )
        ax.set_ylabel(
            f"{helpers.cdr3_segment_name[gene_locus]} length",
            fontweight="bold",
            size="large",
            rotation=90,
        )
        ax.set_xlabel("Sequence prediction rank", fontweight="bold", size="large")

        # Keep only 1 in every 3 xticklabels active
        from summarynb import chunks

        for three_tick_labels in chunks(ax.get_yticklabels(), 2):
            three_tick_labels[0].set_visible(True)
            for tick_label in three_tick_labels[1:]:
                tick_label.set_visible(False)

        # ax.set_title(targetvalue, fontweight="bold", fontsize="x-large")
        sns.despine(ax=ax)

    # axarr_cdr3_lengths[0].set_ylabel(
    #     "Sequence prediction rank", fontweight="bold", size="large"
    # )

    ####

    # Show xlabels only in last axes in each column (which is not necessarily in the final row)
    for ngroup in range(ncols_ngroups):
        # decide whether final row for this column was the penultimate row of the plot:
        # with the lingo that a "subplot" refers to two axes per targetvalue (there are also separators)
        # and two axes per targetvalue equals one column,
        # check if [number of desired total subplots] % [number of columns desired] > 0, and whether we are at that cutoff
        # (and handle edge case where we have one row total, in which case don't switch labeling to penultimate row because it doesn't exist)
        final_row_for_this_column = (
            -2
            if (
                ngroup >= len(targetvalue_order) % ncols_ngroups > 0
                and axarr.shape[0] > 1
            )
            else -1
        )

        # find this group's starting index into axarr 2nd dimension (accounts for separator axes)
        group_starting_ix = ngroup * (ncols_group_size + 1)

        for colix in [group_starting_ix, group_starting_ix + 1]:
            # Remove x-labels except for last row
            for ax in axarr[:final_row_for_this_column, colix].ravel():
                ax.set_xlabel(None)

    # ###

    # # Only enable y tick labels for first axes in each row - accomplishing sharey=True
    # for row_ix in range(axarr.shape[0]):
    #     # turn tick labels on for bottommost entry for this column
    #     for tick_label in axarr[row_ix, 0].get_yticklabels():
    #         tick_label.set_visible(True)

    #     # turn off tick labels for all axes above this in this column
    #     for ax in axarr[row_ix, 1:]:
    #         for tick_label in ax.get_yticklabels():
    #             tick_label.set_visible(False)

    #####

    # Label subplot panels for figure - only first subplot in each row
    for letter_label, ax in zip(string.ascii_lowercase, axarr_vgenes):
        # https://matplotlib.org/3.5.1/gallery/text_labels_and_annotations/label_subplots.html
        # label physical distance to the left and up:
        # trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.02,
            letter_label,
            transform=ax.transAxes,  # + trans,
            fontsize="x-large",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout(pad=0)  # h_pad=0.01, w_pad=0.1
    return fig


####

# for gene_locus in config.gene_loci_used:
def rank_entire_locus_sequences(
    gene_locus: GeneLocus,
    target_obs_columns: List[TargetObsColumnEnum],
    main_output_base_dir: Path,
    highres_output_base_dir: Path,
    cdr3_size_threshold=100,
):
    # Defensive cast
    main_output_base_dir = Path(main_output_base_dir)
    highres_output_base_dir = Path(highres_output_base_dir)

    # Make directories if needed
    main_output_base_dir.mkdir(parents=True, exist_ok=True)
    highres_output_base_dir.mkdir(parents=True, exist_ok=True)

    # Rank sequences
    ranked_sequence_dfs: Mapping[
        Tuple[TargetObsColumnEnum, Any], pd.DataFrame
    ] = rank_sequences(gene_locus=gene_locus, target_obs_columns=target_obs_columns)
    # # Raw probabilities
    # ranked_sequence_dfs[TargetObsColumnEnum.disease, 'Covid19']["Covid19"].hist()
    # # Rankings (uniform)
    # ranked_sequence_dfs[TargetObsColumnEnum.disease, 'Covid19']['rank'].hist()

    # Eliminate noisy V genes for this locus
    selected_v_genes, fig_unfiltered, fig_filtered = filter_v_genes(
        gene_locus=gene_locus, ranked_sequence_dfs=ranked_sequence_dfs
    )
    genetools.plots.savefig(
        fig_unfiltered,
        highres_output_base_dir / "v_gene_disease_proportions.png",
        dpi=300,
    )
    genetools.plots.savefig(
        fig_unfiltered,
        highres_output_base_dir / "v_gene_disease_proportions.pdf",
    )
    plt.close(fig_unfiltered)

    genetools.plots.savefig(
        fig_filtered,
        highres_output_base_dir / "v_gene_disease_proportions.filtered.png",
        dpi=300,
    )
    genetools.plots.savefig(
        fig_filtered,
        highres_output_base_dir / "v_gene_disease_proportions.filtered.pdf",
    )
    plt.close(fig_filtered)

    # Export the selected V genes
    selected_v_genes.sort_values().to_csv(
        main_output_base_dir / "meaningful_v_genes.txt",
        sep="\t",
        index=None,
    )

    selected_v_genes.replace(helpers.v_gene_friendly_names).sort_values().to_csv(
        main_output_base_dir / "meaningful_v_genes.friendly_names.txt",
        sep="\t",
        index=None,
    )

    # Prepare to plot interpretations:
    # get all unique CDR3 lengths and order them consistently
    cdr3_length_order = np.unique(
        np.hstack(
            [
                ranked_sequences_df["cdr3_aa_sequence_trim_len"].unique()
                for ranked_sequences_df in ranked_sequence_dfs.values()
            ]
        )
    )
    # fill in any skips
    cdr3_length_order = np.arange(min(cdr3_length_order), max(cdr3_length_order) + 1)
    # make colors
    cdr3_length_palette = {
        length: color
        for length, color in zip(
            cdr3_length_order,
            sns.color_palette(
                "Blues_r", n_colors=len(cdr3_length_order)
            ),  # "crest" alternative
        )
    }

    # Downselect to target obs columns that we were able to generate rankings for
    target_obs_columns = ranked_sequence_dfs.keys(dimensions=0)

    for target_obs_column in target_obs_columns:
        # Subdirectories for each classification target
        main_output_dir = main_output_base_dir / target_obs_column.name
        main_output_dir.mkdir(exist_ok=True)

        highres_output_dir = highres_output_base_dir / target_obs_column.name
        highres_output_dir.mkdir(exist_ok=True)

        # Plot all V gene rankings
        fig = plot_v_gene_rankings(
            target_obs_column=target_obs_column,
            ranked_sequence_dfs=ranked_sequence_dfs,
            figsize=(10, 10),
            filtered_v_gene_list=None,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "v_gene_rankings.png",
            dpi=300,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "v_gene_rankings.pdf",
        )
        plt.close(fig)

        # Plot filtered V gene rankings
        fig = plot_v_gene_rankings(
            target_obs_column=target_obs_column,
            ranked_sequence_dfs=ranked_sequence_dfs,
            figsize=(10, 7),
            filtered_v_gene_list=selected_v_genes,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "v_gene_rankings.filtered.png",
            dpi=300,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "v_gene_rankings.filtered.pdf",
        )
        plt.close(fig)

        # Plot filtered V gene rankings, excluding healthy (only if "disease" target)
        fig = plot_v_gene_rankings(
            target_obs_column=target_obs_column,
            ranked_sequence_dfs=ranked_sequence_dfs,
            figsize=(10, 7),
            filtered_v_gene_list=selected_v_genes,
            exclude_target_values=[healthy_label],
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "v_gene_rankings.filtered.without_healthy.png",
            dpi=300,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "v_gene_rankings.filtered.without_healthy.pdf",
        )
        plt.close(fig)

        # Plot CDR3 rankings
        fig = plot_cdr3_length_vs_rank(
            target_obs_column=target_obs_column,
            ranked_sequence_dfs=ranked_sequence_dfs,
            cdr3_length_palette=cdr3_length_palette,
            figsize=(18, 4),
            cdr3_size_threshold=cdr3_size_threshold,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "cdr3_lengths.png",
            dpi=300,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "cdr3_lengths.pdf",
        )
        plt.close(fig)

        # Plot all
        fig = plot_all_interpretations(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            ranked_sequence_dfs=ranked_sequence_dfs,
            cdr3_length_palette=cdr3_length_palette,
            figsize=(16, 11),
            filtered_v_gene_list=selected_v_genes,
            cdr3_size_threshold=cdr3_size_threshold,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "all.png",
            dpi=300,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "all.pdf",
        )
        plt.close(fig)

        # Plot all, without healthy (only if "disease" target)
        fig = plot_all_interpretations(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            ranked_sequence_dfs=ranked_sequence_dfs,
            cdr3_length_palette=cdr3_length_palette,
            figsize=(16, 7),
            filtered_v_gene_list=selected_v_genes,
            exclude_target_values=[healthy_label],
            cdr3_size_threshold=cdr3_size_threshold,
        )
        genetools.plots.savefig(
            fig,
            main_output_dir / "all.without_healthy.png",
            dpi=300,
        )
        genetools.plots.savefig(
            fig,
            highres_output_dir / "all.without_healthy.pdf",
        )
        plt.close(fig)

        # Plot isotype rankings
        if gene_locus == GeneLocus.BCR:
            fig = plot_isotype_use_vs_ranks(
                target_obs_column=target_obs_column,
                ranked_sequence_dfs=ranked_sequence_dfs,
                figsize=(13, 5),
            )
            genetools.plots.savefig(
                fig,
                highres_output_dir / "isotype_usage.png",
                dpi=300,
            )
            genetools.plots.savefig(
                fig,
                highres_output_dir / "isotype_usage.pdf",
            )
            plt.close(fig)

    return ranked_sequence_dfs


###
### Load known binder databases


def _load_covabdab(
    clustering_train_threshold: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], pd.DataFrame]]:
    """Cluster CoV-AbDab CDR3s"""
    train_sequences_df = pd.read_csv(
        config.paths.base_data_dir / "CoV-AbDab_260722.filtered.annotated.tsv", sep="\t"
    )

    old_shape = train_sequences_df.shape[0]
    train_sequences_df = ConvergentClusterClassifier._cluster_training_set(
        df=train_sequences_df,
        sequence_identity_threshold=clustering_train_threshold,
        # skip fold_id and fold_label check
        validate_same_fold=False,
    )
    assert train_sequences_df.shape[0] == old_shape

    # # total number of clusters across all Cov-abdab data
    # train_sequences_df["global_resulting_cluster_ID"].nunique()

    # # a number of cov-abdab sequences were joined into a single cluster
    # train_sequences_df["global_resulting_cluster_ID"].value_counts()

    # # how many cov-abdab sequences were merged
    # (train_sequences_df["global_resulting_cluster_ID"].value_counts() > 1).value_counts()

    # train_sequences_df["global_resulting_cluster_ID"].value_counts()
    # train_sequences_df["global_resulting_cluster_ID"].value_counts().hist(bins=20)

    ## Consider all of these to be "predictive clusters", since they are from Covabdab. I.e. no further filtering.
    ## Make cluster centroids for clusters
    # Since we don't have `num_clone_members`, it was set to 1, so these will not be weighed by number of clone members (number of unique VDJ sequences)
    # train_sequences_df["num_clone_members"].value_counts()

    # Make cluster centroids for predictive clusters, weighed by number of clone members (number of unique VDJ sequences)
    # Except here we have weights=1
    # And all clusters are predictive
    cluster_centroids_df = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=train_sequences_df
    )
    # Reshape as dict
    cluster_centroids_by_supergroup = (
        ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            cluster_centroids=cluster_centroids_df
        )
    )
    return train_sequences_df, cluster_centroids_by_supergroup


def _load_mira(
    clustering_train_threshold: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], pd.DataFrame]]:
    """Cluster MIRA CDR3s"""
    train_sequences_df = pd.read_csv(
        config.paths.external_raw_data
        / "immunecode/mira/ImmuneCODE-MIRA-Release002.1"
        / "mira_combined.filtered.tsv",
        sep="\t",
    )

    old_shape = train_sequences_df.shape[0]
    train_sequences_df = ConvergentClusterClassifier._cluster_training_set(
        df=train_sequences_df,
        sequence_identity_threshold=clustering_train_threshold,
        # skip fold_id and fold_label check
        validate_same_fold=False,
    )
    assert train_sequences_df.shape[0] == old_shape

    # # total number of clusters across all reference data
    # train_sequences_df["global_resulting_cluster_ID"].nunique()

    # # a number of cov-abdab sequences were joined into a single cluster
    # train_sequences_df["global_resulting_cluster_ID"].value_counts()

    # # how many cov-abdab sequences were merged
    # (train_sequences_df["global_resulting_cluster_ID"].value_counts() > 1).value_counts()

    # train_sequences_df["global_resulting_cluster_ID"].value_counts()
    # train_sequences_df["global_resulting_cluster_ID"].value_counts().hist(bins=20)

    ## Consider all of these to be "predictive clusters". No further filtering.

    ## Make cluster centroids for clusters
    # Since we don't have `num_clone_members`, it was set to 1, so this will not be weighed by number of clone members (number of unique VDJ sequences)
    # train_sequences_df["num_clone_members"].value_counts()

    # Make cluster centroids for predictive clusters, weighed by number of clone members (number of unique VDJ sequences)
    # Except here we have weights=1
    # And all clusters are predictive
    cluster_centroids_df = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=train_sequences_df
    )
    # Reshape as dict
    cluster_centroids_by_supergroup = (
        ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            cluster_centroids=cluster_centroids_df
        )
    )
    return train_sequences_df, cluster_centroids_by_supergroup


reference_dataset_name = {GeneLocus.BCR: "CoV-AbDab", GeneLocus.TCR: "MIRA"}


def load_reference_dataset(
    gene_locus: GeneLocus,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], pd.DataFrame]]:
    """Load dataset of known binders (CoV-AbDab for BCR or MIRA for TCR), clustered by CDR3"""
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    if gene_locus == GeneLocus.BCR:
        # sequence identity thresholds for clustering amino acid sequences (across patients):
        # threshold for combining cov-abdab source seqs into clusters
        # very high because we just want to merge near-exact dupes
        return _load_covabdab(clustering_train_threshold=0.95)
    elif gene_locus == GeneLocus.TCR:
        # sequence identity thresholds for clustering amino acid sequences (across patients):
        # threshold for combining reference-dataset source seqs into clusters
        # for TCR this is 1.0 because we want exact matches
        return _load_mira(clustering_train_threshold=1.0)
