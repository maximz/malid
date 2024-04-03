# -*- coding: utf-8 -*-
# %%

# %% [markdown]
# # Analyze metamodel performance on test set, with abstention
#
# > Train patient-level rollup model using existing base models trained on train-smaller set.
#

# %%

# %%
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools
import genetools
from statannotations.Annotator import Annotator

# %matplotlib inline
import seaborn as sns
from IPython.display import display, Markdown

# %%
from malid import config, logger, helpers
from wrap_glmnet import GlmnetLogitNetWrapper
from malid.train import train_metamodel
import crosseval
from malid.datamodels import (
    DataSource,
    TargetObsColumnEnum,
    GeneLocus,
    map_cross_validation_split_strategy_to_default_target_obs_column,
)
from malid.trained_model_wrappers import BlendingMetamodel

# Note: don't import this before seaborn, otherwise the style will be overriden and figures will have a different size:
from malid.trained_model_wrappers.blending_metamodel import MetamodelConfig
from genetools.plots import (
    plot_two_key_color_and_size_dotplot,
    plot_triangular_heatmap,
)
import multiclass_metrics

# %%

# %%


# %%
def _plot_feature_importances(
    plot_df: pd.DataFrame, model_name: str, xlabel: str, xmin_at_zero: bool
):
    """plot feature importances for binary/multiclass random forest or binary linear model,
    where we have one model across all classes, rather than OvR multiclass model"""
    fig, ax = plt.subplots(figsize=(4, plot_df.shape[1] / 2.5))

    try:
        # Convert any metamodel feature names to friendly names,
        # if they have not already been renamed to friendly names when grouping/summing subsets.
        plot_df = plot_df.rename(
            columns=lambda feature_name: BlendingMetamodel.convert_feature_name_to_friendly_name(
                feature_name
            )
        )

        if plot_df.shape[0] == 1:
            # Special case: single entry. Show scatter plot instead of box plot.
            ax.scatter(plot_df.iloc[0].values, plot_df.iloc[0].index)
            # Make spacing and y-axis order similar to default boxplot
            buffer = 0.5
            ax.set_ylim(-buffer, plot_df.shape[1] - 1 + buffer)
            ax.invert_yaxis()
        else:
            # Default: boxplot
            sns.boxplot(data=plot_df, orient="h", ax=ax)

        plt.title(
            f"{model_name} ({plot_df.shape[0]} fold{'s'[:plot_df.shape[0] != 1]})"
        )
        plt.xlabel(xlabel)
        if xmin_at_zero:
            plt.xlim(
                0,
            )
        return fig
    except Exception as err:
        # close figure just in case, some Jupyter does not try to display a broken figure
        plt.close(fig)
        # reraise
        raise err


def _sum_subsets_of_feature_importances(
    df: pd.DataFrame,
    subset_names: Optional[Dict[str, str]],
    drop_empty_subsets: bool = True,
):
    """Sum up feature importances by subsets.
    Subset_names is a dict mapping friendly_subset_name to regex to match columns (we match with "contains" operation).
    Drop_empty_subsets is whether to drop empty subsets (i.e. where no columns match the regex).
    Pass through as-is without summing if subset_names is not provided
    """
    if subset_names is not None:
        # get relevant columns for each subset
        sum_parts = {
            name: df.loc[:, df.columns.str.contains(regex)]
            for name, regex in subset_names.items()
        }

        if drop_empty_subsets:
            # drop subsets where no columns have matched
            sum_parts = {
                name: df_part
                for name, df_part in sum_parts.items()
                if not df_part.empty
            }

        if len(sum_parts) == 0:
            raise ValueError(
                f"Subset names {subset_names} not found in df columns {df.columns}"
            )

        # do the sums
        return pd.DataFrame.from_dict(
            {name: df_part.sum(axis=1) for name, df_part in sum_parts.items()},
            orient="columns",
        )

    # pass through as-is without summing if subset_names is not provided
    return df


def get_feature_importance_subsets_to_plot(
    gene_locus: GeneLocus,
) -> Dict[str, Union[Dict[str, str], None]]:
    model_component_names = [
        ("repertoire_stats", "Repertoire composition"),
        ("convergent_cluster_model", "CDR3 clustering"),
        ("sequence_model", "Language model"),
    ]
    demographics_include = {"Demographics": "^demographics"}
    interactions_include = {
        "Sequence x Demographic feature interactions": "^interaction"
    }
    return {
        "all": None,
        "by_locus": {
            f"{gene_locus_part.name}": f"^{gene_locus_part.name}:*"
            for gene_locus_part in gene_locus
        }
        | demographics_include
        | interactions_include,
        "by_model_component": {
            # Don't match if starts with interaction
            # i.e. match "BCR:sequence_model:Covid19" but not "interaction|BCR:sequence_model:Covid19|demographics:age".
            f"{model_component_friendly_name}": f"^(?:(?!interaction).)*{model_component_name}"
            for model_component_name, model_component_friendly_name in model_component_names
        }
        | demographics_include
        | interactions_include,
        "by_locus_and_model_component": {
            f"{model_component_friendly_name} ({gene_locus_part.name})": f"^{gene_locus_part.name}:{model_component_name}:*"
            for model_component_name, model_component_friendly_name in model_component_names
            for gene_locus_part in gene_locus
        }
        | demographics_include
        | interactions_include,
    }


def plot_multiclass_feature_importances(
    model_name: str,
    raw_coefs_mean: pd.DataFrame,
    raw_coefs_std: Optional[pd.DataFrame],
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    metamodel_flavor: str,
    n_folds: int,
) -> Generator[Tuple[str, plt.Figure], None, None]:
    ## We will plot raw coefs, and also use absvals so we can combine features

    def _sort_plot_features(features_df: pd.DataFrame) -> pd.DataFrame:
        # Arrange feature columns in desired order:
        # 1. BCR : model1 : Covid19
        # 2. BCR : model2 : Covid19
        # 3. BCR : model3 : Covid19
        # 4. TCR : model1 : Covid19
        # 5. TCR : model2 : Covid19
        # 6. TCR : model3 : Covid19
        # 7. BCR : model1 : HIV
        # and so on
        column_order = features_df.columns.to_series().str.split(":", expand=True)
        if column_order.shape[1] >= 3:
            # this is true for the above examples
            column_order = column_order.sort_values([2, 0])
        else:
            # the demographics-only metamodel flavor has feature names with only one single colon
            column_order = column_order.sort_values([0])
        return features_df[column_order.index]

    raw_coefs_mean = _sort_plot_features(raw_coefs_mean)
    if raw_coefs_std is not None:
        raw_coefs_std = _sort_plot_features(raw_coefs_std)

    diverging_color_cmap = "RdBu_r"
    # Cut cmap by 15% from both sides, so that we don't have dark blue and dark red at the extremes, which are hard to distinguish
    # https://stackoverflow.com/a/18926541/130164
    diverging_color_cmap = matplotlib.colormaps.get_cmap(diverging_color_cmap)
    diverging_color_cmap = diverging_color_cmap.from_list(
        name=f"{diverging_color_cmap.name}_truncated",
        colors=diverging_color_cmap(np.linspace(0.15, 0.85, 256)),
        N=256,
    )

    def _plot(
        features_df: pd.DataFrame,
        label: str,
        cmap_diverging: bool,
        require_sum_to_1: bool,
        make_percentage: bool = False,
    ):
        # autosize
        figsize = (features_df.shape[0] * 1.0, features_df.shape[1] / 2.5)

        if require_sum_to_1 and not np.allclose(features_df.sum(axis=1), 1):
            raise ValueError("Sum of feature importances is not 1")
        if make_percentage:
            # Turn fractions into percentages
            if not require_sum_to_1:
                raise ValueError("make_percentage requires require_sum_to_1")
            features_df = features_df * 100

        # Convert any metamodel feature names to friendly names,
        # if they have not already been renamed to friendly names when grouping/summing subsets.
        features_df = features_df.rename(
            columns=lambda feature_name: BlendingMetamodel.convert_feature_name_to_friendly_name(
                feature_name
            )
        )

        fig, ax = plt.subplots(figsize=figsize)

        try:
            # Create dedicated colorbar axis
            colorbar_ax = inset_axes(
                ax,
                width="80%",  # relative unit
                height=0.25,  # in inches
                loc="lower center",
                borderpad=-5,  # create space
            )
            sns.heatmap(
                # plot transpose, so features are on y-axis
                features_df.T,
                center=0 if cmap_diverging else None,
                linewidths=0.5,
                cmap=diverging_color_cmap
                if cmap_diverging
                else "Blues",  # sns.color_palette("vlag", as_cmap=True) is another good diverging
                ax=ax,
                # Put colorbar on bottom
                cbar_kws={"label": label, "orientation": "horizontal"},
                cbar_ax=colorbar_ax,
                # plot all tick labels
                xticklabels=True,
                yticklabels=True,
            )

            # Adjust tick labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            genetools.plots.wrap_tick_labels(
                ax=ax, wrap_x_axis=True, wrap_y_axis=False, wrap_amount=12
            )

            # set global "current axes" back to main axes,
            # so that any calls like plt.title target main ax rather than colorbar_ax
            plt.sca(ax)
            return fig, ax
        except Exception as err:
            # close figure just in case, some Jupyter does not try to display a broken figure
            plt.close(fig)
            # reraise
            raise err

    # Plot mean
    try:
        fig, ax = _plot(
            _sort_plot_features(raw_coefs_mean),
            label="Coefficient mean",
            cmap_diverging=True,
            require_sum_to_1=False,
        )
        ax.set_title(
            f"Feature coefficients, each class versus the rest (mean over {n_folds} folds)"
        )
        yield (f"raw_coefs.mean", fig)
    except Exception as err:
        logger.warning(
            f"Failed to plot {model_name}, {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor} multiclass raw_coefs.mean with error: {err}"
        )

    if raw_coefs_std is not None:
        # Plot std
        try:
            fig, ax = _plot(
                _sort_plot_features(raw_coefs_std),
                label="Coefficient stdev",
                cmap_diverging=False,
                require_sum_to_1=False,
            )
            ax.set_title(
                f"Feature coefficients, each class versus the rest (stdev over {n_folds} folds)"
            )
            yield (f"raw_coefs.stdev", fig)
        except Exception as err:
            logger.warning(
                f"Failed to plot {model_name}, {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor} multiclass raw_coefs.stdev with error: {err}"
            )

        # Plot mean and standard deviation together
        combined = pd.merge(
            raw_coefs_mean.rename_axis(index="class")
            .reset_index()
            .melt(
                id_vars=["class"],
                value_vars=raw_coefs_mean.columns,
                var_name="feature",
                value_name="mean",
            ),
            raw_coefs_std.rename_axis(index="class")
            .reset_index()
            .melt(
                id_vars=["class"],
                value_vars=raw_coefs_std.columns,
                var_name="feature",
                value_name="stdev",
            ),
            on=["class", "feature"],
            how="inner",
            validate="1:1",
        )
        # Convert raw metamodel feature names to friendly names
        combined["feature"] = combined["feature"].apply(
            BlendingMetamodel.convert_feature_name_to_friendly_name
        )

        try:
            fig, ax = plot_two_key_color_and_size_dotplot(
                data=combined,
                x_axis_key="class",
                y_axis_key="feature",
                color_key="mean",
                size_key="stdev",
                color_cmap=diverging_color_cmap,
                color_vcenter=0,
                inverse_size=True,
                color_legend_text="mean",
                size_legend_text="inverse std. dev.",
                min_marker_size=20,
            )
            # TODO: make hierarchical y-axis labels (https://stackoverflow.com/questions/19184484/how-to-add-group-labels-for-bar-charts, https://stackoverflow.com/questions/37934242/hierarchical-axis-labeling-in-matplotlib-python)
            ax.set_title(
                f"Feature coefficients, each class versus the rest (over {n_folds} folds)"
            )
            yield (f"raw_coefs", fig)
        except Exception as err:
            logger.warning(
                f"Failed to plot {model_name}, {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor} multiclass raw_coefs (mean+stdev together) with error: {err}"
            )

    ## Report aggregate feature importance of several features in a linear model
    # e.g. I'd like to say something about how much all the language model features contribute to the metamodel, vs all the CDR3 clustering features.
    # I believe you can [sum feature importances](https://stats.stackexchange.com/questions/311488/summing-feature-importance-in-scikit-learn-for-a-set-of-features) for a set of features in a random forest.
    # for a linear model, I suppose I could take the absolute value of the coefs and sum them for something like "overall effect strength from this set of features".

    # Convert to absolute value, and divide by the sum of absolute values of all coefficients for "percent contribution"
    normalized_coefs = genetools.stats.normalize_rows(np.abs(raw_coefs_mean))
    for fig_name, subset_names in get_feature_importance_subsets_to_plot(
        gene_locus
    ).items():
        # sum up by origin of feature importances and replot.
        try:
            logger.debug(f"{model_name} absval_coefs {fig_name} across folds")
            fig, ax = _plot(
                _sum_subsets_of_feature_importances(
                    df=normalized_coefs, subset_names=subset_names
                ),
                label="Percent contribution",  # "Coefficient absval, percent contribution",
                cmap_diverging=False,
                require_sum_to_1=True,
                make_percentage=True,
            )
            plt.title(
                f"{model_name} feature percent contributions\neach class versus the rest\n(averaged over {n_folds} folds)"
            )
            yield (f"absval_coefs.{fig_name}", fig)
        except Exception as err:
            # Skip broken figures
            # One possible cause is that the feature names for this metamodel flavor don't correspond to what get_feature_importance_subsets_to_plot() is producing.
            logger.warning(
                f"Failed to plot {model_name}, {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor} feature percent contributions for figure name absval_coefs.{fig_name}, subset names {subset_names}: {err}"
            )


def analyze_feature_importances(
    model_name: str,
    model_global_performance: crosseval.ModelGlobalPerformance,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    metamodel_flavor: str,
    highres_results_output_prefix: Path,
    global_fold_classifier: Optional[BlendingMetamodel],
):
    """Get and analyze feature importances."""
    # First, check if model is binary in each fold
    is_binary = all(
        len(per_fold_output.class_names) == 2
        for per_fold_output in model_global_performance.per_fold_outputs.values()
    )
    if (
        global_fold_classifier is not None
        and (len(global_fold_classifier.classes_) == 2) != is_binary
    ):
        # Sanity check
        logger.warning(
            f"Ignoring global fold classifier for {model_name} because cross validation is_binary={is_binary} does not match global fold classes count = {len(global_fold_classifier.classes_)}"
        )
        global_fold_classifier = None

    # Depending on the model type (tree vs linear model; binary vs multiclass), we will retrieve and plot feature importances differently.
    # (Tree models are always a single model across all classes, regardless of whether classification target is binary or multiclass,
    # whereas multiclass linear models may be trained separately for each class.)
    is_tree = model_name in ["rf_multiclass", "xgboost"]
    is_linear_model = any(
        model_name.startswith(s)
        for s in [
            "linearsvm_ovr",
            "lasso_cv",
            "ridge_cv",
            "elasticnet_cv",
        ]
    )

    if is_tree or (is_linear_model and is_binary):
        # Get feature importances for each fold
        feature_importances_cross_validation_df: Union[
            pd.DataFrame, None
        ] = model_global_performance.feature_importances
        if feature_importances_cross_validation_df is None:
            logger.warning(f"No feature importances available for {model_name}")
            # skip this model
            return
        feature_importances_to_plot = [(feature_importances_cross_validation_df, "")]

        if global_fold_classifier is not None:
            global_fold_feature_importances = crosseval._extract_feature_importances(
                global_fold_classifier._inner
            )
            global_fold_feature_names = crosseval._get_feature_names(
                global_fold_classifier._inner
            )
            if global_fold_feature_importances is None:
                logger.warning(
                    f"No feature importances available for {model_name} (global fold)"
                )
            else:
                feature_importances_to_plot.append(
                    (
                        pd.Series(
                            global_fold_feature_importances,
                            index=global_fold_feature_names,
                            name=-1,
                        )
                        .to_frame()
                        .T,
                        "_global_fold",
                    )
                )

        # Plot feature importances.
        for (
            feature_importances,
            overall_name,
        ) in feature_importances_to_plot:
            if is_tree:
                for (
                    fig_name,
                    subset_names,
                ) in get_feature_importance_subsets_to_plot(gene_locus).items():
                    # sum up by origin of feature importances and replot.
                    try:
                        fig = _plot_feature_importances(
                            plot_df=_sum_subsets_of_feature_importances(
                                df=feature_importances,
                                subset_names=subset_names,
                            ),
                            model_name=model_name,
                            xlabel="Feature importance",
                            # Values are all positive for tree models
                            xmin_at_zero=True,
                        )
                        genetools.plots.savefig(
                            fig,
                            f"{highres_results_output_prefix}.feature_importances{overall_name}.{model_name}.{fig_name}.png",
                            dpi=300,
                        )
                        plt.close(fig)
                    except Exception as err:
                        # Skip broken figures
                        # One possible cause is that the feature names for this metamodel flavor don't correspond to what get_feature_importance_subsets_to_plot() is producing.
                        logger.warning(
                            f"Failed to plot {model_name} feature importances{overall_name} for {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor}, with figure name {fig_name} and subset names {subset_names}: {err}"
                        )

            elif is_linear_model:
                for (
                    feature_importances,
                    overall_name,
                ) in feature_importances_to_plot:
                    # TODO: Add normalization of coefficients and summing of subsets (nontrivial for linear model)
                    # For now only plot all the features - don't group by subset.
                    fig = _plot_feature_importances(
                        plot_df=feature_importances,
                        model_name=model_name,
                        xlabel="Feature coefficient",
                        # coefficients are not necessarily positive
                        xmin_at_zero=False,
                    )
                    genetools.plots.savefig(
                        fig,
                        f"{highres_results_output_prefix}.feature_importances{overall_name}.{model_name}.all.png",
                        dpi=300,
                    )
                    plt.close(fig)

    elif is_linear_model and not is_binary:
        # Many OvR models for each class vs the rest
        raw_coefs: Optional[
            Dict[int, pd.DataFrame]
        ] = model_global_performance.multiclass_feature_importances
        if raw_coefs is None:
            logger.warning(
                f"No feature importances available for multiclass {model_name}"
            )
            # skip this model
            return

        ## Combine multiclass feature importances across folds:
        # The coefs are comparable across folds because the inputs to the model were standardized.

        # Create 3D array from these 2D arrays - making sure that the index and column order is the same across folds.
        first_df = next(iter(raw_coefs.values()))
        try:
            raw_coefs_data: np.ndarray = np.array(
                [df.loc[first_df.index][first_df.columns] for df in raw_coefs.values()]
            )
        except Exception as err:
            logger.warning(
                f"Could not combine feature coefficients across folds for multiclass linear model {model_name} ({gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor}), possibly because of missing classes. Skipping feature importance plots with this error: {err}"
            )
            # skip this model
            return

        # Extract mean and standard deviation, and repack in dataframe
        raw_coefs_mean: pd.DataFrame = pd.DataFrame(
            np.mean(raw_coefs_data, axis=0),
            index=first_df.index,
            columns=first_df.columns,
        )
        raw_coefs_std: pd.DataFrame = pd.DataFrame(
            np.std(raw_coefs_data, axis=0),
            index=first_df.index,
            columns=first_df.columns,
        )

        raw_coefs_mean.to_csv(
            f"{highres_results_output_prefix}.feature_importances.{model_name}.raw_coefs_mean.tsv",
            sep="\t",
        )
        raw_coefs_std.to_csv(
            f"{highres_results_output_prefix}.feature_importances.{model_name}.raw_coefs_std.tsv",
            sep="\t",
        )

        for fig_name, fig in plot_multiclass_feature_importances(
            model_name=model_name,
            raw_coefs_mean=raw_coefs_mean,
            raw_coefs_std=raw_coefs_std,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            metamodel_flavor=metamodel_flavor,
            n_folds=len(raw_coefs),
        ):
            fname = f"{highres_results_output_prefix}.feature_importances.{model_name}.{fig_name}.png"
            logger.debug(f"{fig_name} -> {fname}")
            genetools.plots.savefig(
                fig,
                fname,
                dpi=300,
            )
            plt.close(fig)

        if global_fold_classifier is not None:
            # Also plot global fold coefficients on their own. (We will pass them as raw_coefs_mean (without running a mean), with raw_coefs_std set to None)
            global_fold_feature_importances = (
                crosseval._extract_multiclass_feature_importances(
                    global_fold_classifier._inner
                )
            )
            global_fold_feature_names = crosseval._get_feature_names(
                global_fold_classifier._inner
            )
            if global_fold_feature_importances is None:
                logger.warning(
                    f"No feature importances available for multiclass {model_name} (global fold)"
                )
                # skip this model
                return
            global_fold_feature_importances = pd.DataFrame(
                global_fold_feature_importances,
                index=global_fold_classifier.classes_,
                columns=global_fold_feature_names,
            )
            global_fold_feature_importances.to_csv(
                f"{highres_results_output_prefix}.feature_importances.{model_name}.raw_coefs.global_fold.tsv",
                sep="\t",
            )
            for fig_name, fig in plot_multiclass_feature_importances(
                model_name=model_name,
                raw_coefs_mean=global_fold_feature_importances,
                raw_coefs_std=None,
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                metamodel_flavor=metamodel_flavor,
                n_folds=1,
            ):
                fname = f"{highres_results_output_prefix}.feature_importances_global_fold.{model_name}.{fig_name}.png"
                logger.debug(f"{fig_name} -> {fname}")
                genetools.plots.savefig(
                    fig,
                    fname,
                    dpi=300,
                )
                plt.close(fig)
    else:
        logger.warning(
            f"Feature importances not plotted for {model_name}: not a recognized tree or linear model."
        )


def compute_pairwise_scores(
    experiment_set_global_performance: crosseval.ExperimentSetGlobalPerformance,
    models_of_interest: List[str],
) -> Generator[Tuple[str, pd.DataFrame, plt.Figure, plt.Axes], None, None]:
    """
    Given a ExperimentSetGlobalPerformance, compute pairwise scores for all models of interest.
    Returns generator yielding (model_name, table, fig, ax) tuples.
    """

    def pairwise_roc_auc_score_function_factory(classA, classB):
        """Generate a ROC AUC score function for classA vs classB."""

        def score_func(
            y_true: np.ndarray,
            y_score: np.ndarray,
            labels: Optional[np.ndarray] = None,
            **kwargs,
        ):
            # Emulate how multiclass OvO score works:
            # https://github.com/scikit-learn/scikit-learn/blob/00032b09c7feea08edd4486c522c2d962f9d52ec/sklearn/metrics/_base.py#L132

            # y_true may have different entries than y_score/labels
            # so start by extending y_score/labels to match y_true, as we eventually will do in roc_auc_score anyway.
            # otherwise the column_indices lookups below will fail.
            y_score = np.array(y_score)  # defensive cast from pandas dataframe
            y_score, labels = multiclass_metrics._inject_missing_labels(
                y_true=y_true, y_score=y_score, labels=labels
            )

            # defensive cast; otherwise np.argwhere fails in column_indices lookup below.
            labels = np.array(labels)

            # Subset to entries where y_true belongs to one of the classes:
            mask = np.logical_or(y_true == classA, y_true == classB)
            y_true = y_true[mask]
            y_score = y_score[mask, :]

            # Subset to those columns:
            new_label_order = np.array([classA, classB])
            column_indices = np.array(
                [
                    np.argwhere(labels == classA)[0][0],
                    np.argwhere(labels == classB)[0][0],
                ]
            )
            y_score = y_score[:, column_indices]

            # Run ROC AUC score with modified input data
            return multiclass_metrics.roc_auc_score(
                y_true=y_true,
                y_score=y_score,
                labels=new_label_order,
                **kwargs,
            )

        return score_func

    # Get class names.
    # To do so, we first need to extract a model global performance object
    a_model_global_performance = next(
        iter(experiment_set_global_performance.model_global_performances.values())
    )
    # Now get class names (minus unknown/abstain label)
    classes = set(a_model_global_performance.confusion_matrix_label_ordering) - {
        a_model_global_performance.abstain_label
    }
    classes = sorted(classes)

    # Create a dictionary of probability scorers:
    # Treat ROC for every pair of classes as a separate metric.
    # Each metric has a score function. pairwise_roc_auc_score_function_factory generates a score function that extracts test examples for a specific pair of classes and computes the AUC.
    probability_scorers = {}
    friendly_metric_name_subcomponents = {}
    for classA, classB in itertools.combinations(classes, 2):
        friendly_name = f"ROC-AUC {classA} vs {classB}"
        probability_scorers[f"rocauc:{classA}|{classB}"] = (
            pairwise_roc_auc_score_function_factory(classA, classB),
            friendly_name,
            {},
        )
        # Record the two classes that have been stitched into each friendly metric name.
        # The metric name will have an additional suffix added to it when summarized, so let's include that here:
        friendly_metric_name_subcomponents[f"{friendly_name} per fold"] = (
            classA,
            classB,
        )

    # Here is how we could run metrics for a single model:
    # a_model_global_performance.aggregated_per_fold_scores(
    #     with_abstention=False, label_scorers={}, probability_scorers=probability_scorers
    # )

    # Run all these pairwise metrics on all the models
    combined_stats = experiment_set_global_performance.get_model_comparison_stats(
        probability_scorers=probability_scorers, label_scorers={}
    )
    # Isolate columns corresponding to our desired ROC AUC OvO metrics, and rename them back to the source components (pair of class names)
    combined_stats = combined_stats[friendly_metric_name_subcomponents.keys()].rename(
        columns=friendly_metric_name_subcomponents
    )

    for model_name in models_of_interest:
        # For a particular model, create a triangular table of pairwise OvO ROC AUCs
        model_stats = combined_stats.loc[model_name].rename("ROC AUC").to_frame()
        model_stats.index = pd.MultiIndex.from_tuples(
            model_stats.index, names=("classA", "classB")
        )

        # Create a triangular table of pairwise OvO ROC AUCs with all cross validation data.
        triangular_table = (
            model_stats.reset_index()
            .pivot(index="classA", columns="classB", values="ROC AUC")
            .fillna("")
            .T
        )

        # Plot a triangular heatmap with just the mean pairwise ROC AUCs across cross validation folds.
        plot_df = (
            model_stats["ROC AUC"]
            # Remove +/- stddev across cross validation folds
            .str.split(" +/-", regex=False)
            .str[0]
            .astype(float)
            .reset_index()
            .pivot(index="classA", columns="classB", values="ROC AUC")
            .T
            # add each class vs itself (a diagonal of NaNs)
            .reindex(index=classes, columns=classes)
        )
        fig, ax = plot_triangular_heatmap(
            plot_df,
            colorbar_label="ROC AUC",
            figsize=(5, 5),
            # hardcode vmin and vmax so color scales are consistent across metamodel flavors
            vmin=0.8,
            vmax=1.0,
        )
        genetools.plots.wrap_tick_labels(
            ax, wrap_x_axis=True, wrap_y_axis=True, wrap_amount=10
        )
        yield model_name, triangular_table, fig, ax


def compute_per_class_scores(
    experiment_set_global_performance: crosseval.ExperimentSetGlobalPerformance,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a ExperimentSetGlobalPerformance, compute per-class ROC AUC scores (OvR).
    Returns DataFrame with:
        rows = model names,
        columns = class names,
        values = OvR ROC AUC mean and standard deviation across cross validation folds.
    """

    def per_class_ovr_roc_auc_score_function_factory(class_name):
        """Generate an OvR ROC AUC score function for a particular class."""

        def score_func(
            y_true: np.ndarray,
            y_score: np.ndarray,
            labels: np.ndarray,
            **kwargs,
        ):
            # one-vs-rest AUCs for each class (due to average=None and multi_class="ovr")
            # first, run the inject_missing_labels functionality that happens inside crosseval_scores.roc_auc_score,
            # so we can create the Series correctly.
            labels_with_any_missing_added = multiclass_metrics._inject_missing_labels(
                y_true=y_true, y_score=y_score, labels=labels
            )[1]
            aucs = pd.Series(
                multiclass_metrics.roc_auc_score(
                    y_true=y_true,
                    y_score=y_score,
                    labels=labels,
                    average=None,
                    multi_class="ovr",
                    **kwargs,
                ),
                index=labels_with_any_missing_added,
            )
            return aucs[class_name]

        return score_func

    # Get class names.
    # To do so, we first need to extract a model global performance object
    a_model_global_performance = next(
        iter(experiment_set_global_performance.model_global_performances.values())
    )
    # Now get class names (minus unknown/abstain label)
    classes = set(a_model_global_performance.confusion_matrix_label_ordering) - {
        a_model_global_performance.abstain_label
    }

    # Create a dictionary of probability scorers:
    # Treat ROC for each class as a separate metric.
    # Each metric has a score function.
    # per_class_ovr_roc_auc_score_function_factory generates a score function that computes OvR scores for all classes and then returns the one for the desired class.
    probability_scorers = {}
    map_friendly_metric_name_to_class_name = {}
    for class_name in classes:
        friendly_name = f"ROC-AUC {class_name} OvR"
        probability_scorers[f"rocauc:{class_name}"] = (
            per_class_ovr_roc_auc_score_function_factory(class_name),
            friendly_name,
            {},
        )
        # Record the two classes that have been stitched into each friendly metric name.
        # The metric name will have an additional suffix added to it when summarized, so let's include that here:
        map_friendly_metric_name_to_class_name[f"{friendly_name} per fold"] = class_name

    # Here is how we could run metrics for a single model:
    #     a_model_global_performance.aggregated_per_fold_scores(
    #         with_abstention=False,
    #         label_scorers={},
    #         probability_scorers=probability_scorers,
    #     )

    # Run all the metrics on all the models
    per_class_stats = experiment_set_global_performance.get_model_comparison_stats(
        probability_scorers=probability_scorers, label_scorers={}
    )
    # Isolate columns corresponding to our desired ROC AUC OvR metrics, and rename them back to the source class name
    # Use a reindex instead of column subset in case any of the keys are missing (e.g. 'ROC-AUC HIV OvR per fold')
    per_class_stats = per_class_stats.reindex(
        columns=list(map_friendly_metric_name_to_class_name.keys()), fill_value=np.nan
    ).rename(columns=map_friendly_metric_name_to_class_name)

    # Now each column is the OvR ROC AUC for a particular class vs all others (column name = class name),
    # and each row corresponds to one model.

    # Also generate the usual metrics
    typical_stats = experiment_set_global_performance.get_model_comparison_stats()

    return per_class_stats, typical_stats


# %%
base_model_train_fold_name = "train_smaller"
metamodel_fold_label_train = "validation"


def analyze_metamodel_flavor(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    metamodel_flavor: str,
    metamodel_config: MetamodelConfig,
    forced_abstentions: Optional[List[str]] = None,
) -> Optional[List[str]]:
    # should already exist:
    metamodels_base_dir = BlendingMetamodel._get_metamodel_base_dir(
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        metamodel_flavor=metamodel_flavor,
    )

    _output_suffix = Path(gene_locus.name) / target_obs_column.name / metamodel_flavor
    # might not exist yet:
    output_base_dir = (
        config.paths.second_stage_blending_metamodel_output_dir / _output_suffix
    )
    highres_output_base_dir = (
        config.paths.high_res_outputs_dir / "metamodel" / _output_suffix
    )
    output_base_dir.mkdir(parents=True, exist_ok=True)
    highres_output_base_dir.mkdir(parents=True, exist_ok=True)

    fname_prefix = (
        f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model"
    )
    model_prefix = metamodels_base_dir / fname_prefix
    results_output_prefix = output_base_dir / fname_prefix
    highres_results_output_prefix = highres_output_base_dir / fname_prefix

    computed_abstentions = None

    try:
        # Load and summarize
        experiment_set = crosseval.ExperimentSet.load_from_disk(
            output_prefix=model_prefix
        )

        # Note that default y_true from BlendingMetamodel._featurize() is target_obs_column.value.blended_evaluation_column_name
        # Use DROP_INCOMPLETE_FOLDS setting because alternate classification targets might not be well-split in the small validation set of the cross-validation folds that were designed to stratify disease.
        # In the cases of some classification targets, we might need to automatically drop folds that have only a single class in the metamodel training data (i.e. in the validation set).
        experiment_set_global_performance = experiment_set.summarize(
            remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS
        )
        experiment_set_global_performance.export_all_models(
            func_generate_classification_report_fname=lambda model_name: f"{results_output_prefix}.classification_report.test_set_performance.{model_name}.txt",
            func_generate_confusion_matrix_fname=lambda model_name: f"{results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.png",
            dpi=300,
        )
        combined_stats = experiment_set_global_performance.get_model_comparison_stats()
        combined_stats.to_csv(
            f"{results_output_prefix}.compare_model_scores.test_set_performance.tsv",
            sep="\t",
        )
        display(
            Markdown(
                f"## {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor} from {model_prefix} to {results_output_prefix}"
            )
        )
        print(metamodel_config)
        display(combined_stats)

        # Extract list of abstentions.
        # (Abstentions should be the same for all models in this metamodel flavor,
        # because they are all trained from the same featurization which determines the abstentions.)
        any_model_global_performance = next(
            iter(experiment_set_global_performance.model_global_performances.values())
        )
        computed_abstentions = (
            any_model_global_performance.cv_abstentions_metadata[
                "specimen_label"
            ].tolist()
            if any_model_global_performance.cv_abstentions_metadata is not None
            else None
        )

        # Redo, but (potentially) override y_true to pass in e.g. disease with past exposures separated out (delinates past exposures on ground truth axis)
        # For cleaner confusion matrices
        # (But this changes global score metrics)
        experiment_set.summarize(
            global_evaluation_column_name=target_obs_column.value.confusion_matrix_expanded_column_name,
            remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS,
        ).export_all_models(
            func_generate_classification_report_fname=lambda model_name: f"{highres_results_output_prefix}.classification_report.test_set_performance.{model_name}.expanded_confusion_matrix.txt",
            func_generate_confusion_matrix_fname=lambda model_name: f"{highres_results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix.png",
            confusion_matrix_true_label="Patient of origin - expanded",
            dpi=300,
        )

        # Do extra analyses for the default classification target for this cross validation split strategy:
        if (
            target_obs_column
            == map_cross_validation_split_strategy_to_default_target_obs_column[
                config.cross_validation_split_strategy
            ]
        ):
            if target_obs_column == TargetObsColumnEnum.disease:
                # Special case when default classification target is "disease":
                # Redo, but (potentially) override y_true to pass in "disease_subtype" for ground truth axis
                # (But this changes global score metrics)
                experiment_set.summarize(
                    global_evaluation_column_name="disease_subtype",
                    remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS,
                ).export_all_models(
                    func_generate_classification_report_fname=lambda model_name: f"{highres_results_output_prefix}.classification_report.test_set_performance.{model_name}.expanded_confusion_matrix_disease_subtype.txt",
                    func_generate_confusion_matrix_fname=lambda model_name: f"{highres_results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix_disease_subtype.png",
                    confusion_matrix_true_label="Patient of origin - subtype",
                    dpi=300,
                )

            # Also resummarize by a combined variable of disease + ethnicity (or other default classification target + ethnicity)
            # But first, fillna on ethnicity column to change nans to "Unknown"
            experiment_set_modified_ethnicity_metadata_column = (
                # Create a copy of the experiment_set, to not disturb original metadata dataframes
                experiment_set.copy()
            )
            for (
                model_single_fold_performance
            ) in (
                experiment_set_modified_ethnicity_metadata_column.model_outputs.values()
            ):
                # Modify every model_single_fold_performance's metadata: fillna on the ethnicity_condensed column
                for df in [
                    model_single_fold_performance.test_metadata,
                    model_single_fold_performance.test_abstention_metadata,
                ]:
                    if df is None or df.shape[0] == 0:
                        continue
                    df["ethnicity_condensed"].fillna("Unknown", inplace=True)
            experiment_set_modified_ethnicity_metadata_column.summarize(
                global_evaluation_column_name=[
                    crosseval.Y_TRUE_VALUES,
                    "ethnicity_condensed",
                ],
                remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS,
            ).export_all_models(
                func_generate_classification_report_fname=lambda model_name: f"{highres_results_output_prefix}.classification_report.test_set_performance.{model_name}.expanded_confusion_matrix_ethnicity_condensed.txt",
                func_generate_confusion_matrix_fname=lambda model_name: f"{highres_results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix_ethnicity_condensed.png",
                confusion_matrix_true_label="Patient of origin - ancestry",
                dpi=300,
            )

            # Also resummarize by a combined variable of disease + age_group_pediatric (or other default classification target + age_group_pediatric)
            experiment_set_modified_metadata_age_pediatric_column = (
                # Create a copy of the experiment_set, to not disturb original metadata dataframes
                experiment_set.copy()
            )
            for (
                model_single_fold_performance
            ) in (
                experiment_set_modified_metadata_age_pediatric_column.model_outputs.values()
            ):
                # Modify every model_single_fold_performance's metadata: create age_group_pediatric column
                def _mutate_metadata_for_age_pediatric_analysis(df):
                    if df is None or df.shape[0] == 0:
                        # Skip empty dataframes
                        # This can happen if e.g. there are no abstentions. Then test_abstention_metadata will be empty.
                        return df

                    # Enrich metadata to include the age_group_pediatric computed column.
                    # So far only "age" column exists in the model outputs saved to disk by metamodel train.
                    # Note this returns a copy instead of mutating in place.
                    df = helpers.enrich_metadata(df)

                    # Fill NaNs intelligently:
                    if (df["data_source"] == DataSource.in_house).all():
                        # If in-house data:
                        # We know we have very few children cohorts and they are clearly indicated in the study name.
                        # If study name indicates that this is a pediatric cohort, set to "under 18". Otherwise set to 18+.
                        slice_children = df["study_name"].str.contains(
                            "pediatric|children", regex=True, case=False
                        )
                        df.loc[slice_children, "age_group_pediatric"] = df.loc[
                            slice_children, "age_group_pediatric"
                        ].fillna("under 18")
                        df["age_group_pediatric"].fillna("18+", inplace=True)
                    else:
                        df["age_group_pediatric"].fillna("Unknown", inplace=True)

                    return df

                # Apply mutations to each dataframe and update the model_single_fold_performance
                # We wrote it this way rather than a loop to make sure we assign back to the original dataframe objects, so the changes are picked up below.
                model_single_fold_performance.test_metadata = (
                    _mutate_metadata_for_age_pediatric_analysis(
                        model_single_fold_performance.test_metadata
                    )
                )
                model_single_fold_performance.test_abstention_metadata = (
                    _mutate_metadata_for_age_pediatric_analysis(
                        model_single_fold_performance.test_abstention_metadata
                    )
                )
            # Now make new confusion matrices with these metadata changes included
            experiment_set_modified_metadata_age_pediatric_column.summarize(
                global_evaluation_column_name=[
                    crosseval.Y_TRUE_VALUES,
                    "age_group_pediatric",
                ],
                remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS,
            ).export_all_models(
                func_generate_classification_report_fname=lambda model_name: f"{highres_results_output_prefix}.classification_report.test_set_performance.{model_name}.expanded_confusion_matrix_age_group_pediatric.txt",
                func_generate_confusion_matrix_fname=lambda model_name: f"{highres_results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix_age_group_pediatric.png",
                confusion_matrix_true_label="Patient of origin - pediatric vs adult",
                dpi=300,
            )

            # Calculate ROC AUC scores for each class in One vs Rest fashion.
            # To be consistent across metamodel flavors, so we can compare scores apples-to-apples,
            # we will force abstention on samples in the forced_abstentions list,
            # which is computed for the "default" flavor and then passed around.
            experiment_set_summary_with_forced_abstentions_applied = crosseval.ExperimentSet(
                model_outputs={
                    key: model_single_fold_performance.apply_abstention_mask(
                        model_single_fold_performance.test_metadata.index.isin(
                            forced_abstentions if forced_abstentions is not None else []
                        )
                    )
                    for key, model_single_fold_performance in experiment_set.model_outputs.items()
                }
            ).summarize()
            (
                roc_auc_ovr_per_class_scores,
                standard_scores_with_forced_abstentions,
            ) = compute_per_class_scores(
                experiment_set_summary_with_forced_abstentions_applied
            )
            # Export the OvR per-class scores and the standard scores
            roc_auc_ovr_per_class_scores.to_csv(
                f"{highres_results_output_prefix}.per_class_roc_auc_scores_ovr.tsv",
                sep="\t",
            )
            standard_scores_with_forced_abstentions.to_csv(
                f"{highres_results_output_prefix}.standard_scores_with_forced_consistent_abstentions.tsv",
                sep="\t",
            )

            # Calculate pairwise OvO ROC AUC scores for all pairs of classes (i.e. one class vs another class)
            for (
                model_name,
                triangular_table,
                triangular_fig,
                triangular_ax,
            ) in compute_pairwise_scores(
                experiment_set_global_performance=experiment_set_global_performance,
                models_of_interest=config.model_names_to_train,
            ):
                triangular_ax.set_title(
                    f"""{gene_locus}
{target_obs_column}

metamodel flavor: {metamodel_flavor}
model: {model_name}"""
                )
                # Export
                genetools.plots.savefig(
                    triangular_fig,
                    f"{highres_results_output_prefix}.pairwise_roc_auc_scores.{model_name}.png",
                    dpi=300,
                )
                triangular_table.to_csv(
                    f"{highres_results_output_prefix}.pairwise_roc_auc_scores.{model_name}.tsv",
                    sep="\t",
                )
                plt.close(triangular_fig)

        for (
            model_name,
            model_global_performance,
        ) in experiment_set_global_performance.model_global_performances.items():
            if model_name not in config.model_names_to_analyze_extra:
                # Skip further analysis for most models
                continue

            # review classification for each specimen
            individual_classifications = model_global_performance.get_all_entries()
            individual_classifications.to_csv(
                f"{highres_results_output_prefix}.classification_raw_per_specimen.test_set_performance.{model_name}.with_abstention.tsv",
                sep="\t",
                index=None,
            )

            # filter to mistakes (including abstentions)
            mistakes = individual_classifications[
                individual_classifications["y_true"]
                != individual_classifications["y_pred"]
            ]
            mistakes.to_csv(
                f"{highres_results_output_prefix}.classification_errors.test_set_performance.{model_name}.with_abstention.tsv",
                sep="\t",
                index=None,
            )

            # filter further to abstentions
            abstentions = individual_classifications[
                individual_classifications["y_pred"]
                == model_global_performance.abstain_label
            ]
            abstentions.to_csv(
                f"{highres_results_output_prefix}.classification_abstentions.test_set_performance.{model_name}.with_abstention.tsv",
                sep="\t",
                index=None,
            )

            # label correct/incorrect
            individual_classifications["classification_success"] = "Correct"
            individual_classifications.loc[
                individual_classifications["y_true"]
                != individual_classifications["y_pred"],
                "classification_success",
            ] = "Incorrect"

            # Plot difference between top two predicted probabilities, p1 - p2,
            # and difference in logits (log odds) of the top two classes, log(p1/(1-p1)) - log(p2/(1-p2)),
            # to account for the fact that these are probability distributions that sum to 1.
            # (That's the natural log, i.e. log base e.)
            # Alternative considered: difference in log probabilities of top two classes, i.e. log(p1) - log(p2), but that won't distinguish cases like p1=0.5, p2=0.25 from p1=0.4, p2=0.2.
            # difference_between_top_two_predicted_probas was already generated, but we can create the rest ourselves here.
            # TODO: consider other metrics from https://robertmunro.com/uncertainty_sampling_example.html?
            p1, p2 = (
                individual_classifications["max_predicted_proba"],
                individual_classifications["second_highest_predicted_proba"],
            )
            epsilon = 1e-8  # avoid log(0) if p=0 or p=1
            individual_classifications[
                "difference_between_logits_of_top_two_classes"
            ] = (np.log(p1 + epsilon) - np.log(1 - p1 + epsilon)) - (
                np.log(p2 + epsilon) - np.log(1 - p2 + epsilon)
            )
            order = ["Incorrect", "Correct"]
            for metric, label in [
                (
                    "difference_between_top_two_predicted_probas",
                    "Difference between\ntop two predicted probabilities",
                ),
                (
                    "difference_between_logits_of_top_two_classes",
                    "Difference between log odds\nof top two predicted classes",
                ),
            ]:
                # Subset order to actually present values.
                # Avoids seaborn boxplot error (we believe this is v0.13+ only): "Missing x value(s) `"Incorrect"` in classification_success (specified in `order`)"
                filtered_order = [
                    x
                    for x in order
                    if x in individual_classifications["classification_success"].values
                ]
                fig = plt.figure(figsize=(3, 5))
                sns.boxplot(
                    data=individual_classifications,
                    x="classification_success",
                    y=metric,
                    # TODO(seaborn >= 0.13): reenable hue and legend:
                    # hue="classification_success",
                    # legend=False,
                    order=filtered_order,
                    palette=sns.color_palette("Paired")[:2],
                )
                plt.title(f"Blending metamodel {model_name}")
                plt.xlabel("Specimen classification")
                plt.ylabel(label)
                sns.despine()
                genetools.plots.savefig(
                    fig,
                    f"{highres_results_output_prefix}.errors_versus_{metric}.test_set_performance.{model_name}.with_abstention.vertical.simple.png",
                    dpi=300,
                )
                plt.close(fig)

                ##

                # Repeat the plot, but now with scatter points above the boxplot, and with statistical test.
                fig = plt.figure(figsize=(3, 5))
                ax = sns.boxplot(
                    data=individual_classifications,
                    x="classification_success",
                    y=metric,
                    # TODO(seaborn >= 0.13): reenable hue and legend:
                    # hue="classification_success",
                    # legend=False,
                    order=filtered_order,
                    palette=sns.color_palette("Paired")[:2],
                    # Disable outlier markers:
                    fliersize=0,
                    zorder=20,
                )

                for patch in ax.patches:
                    # Set boxplot alpha transparency: https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0.3))
                sns.stripplot(
                    data=individual_classifications,
                    x="classification_success",
                    y=metric,
                    order=filtered_order,
                    hue="classification_success",
                    legend=None,
                    linewidth=0.25,
                    edgecolor="gray",
                    palette=sns.color_palette("Paired")[:2],
                    ax=ax,
                    size=2.5,
                    jitter=0.25,
                    zorder=1,
                    alpha=0.7,
                )

                # Annotate with statistical significance
                # One sided test for Incorrect having *lower* scores than Correct
                test_results = None
                if len(filtered_order) > 1:
                    annot = Annotator(
                        ax=ax,
                        pairs=[filtered_order],
                        data=individual_classifications,
                        x="classification_success",
                        y=metric,
                        order=filtered_order,
                    )
                    annot.configure(
                        test="Mann-Whitney-ls",
                        text_format="star",
                        loc="outside",
                        verbose=2,
                    )
                    annot.apply_test(method="asymptotic")
                    ax, test_results = annot.annotate()

                ax.set_xticklabels(
                    genetools.plots.add_sample_size_to_labels(
                        labels=ax.get_xticklabels(),
                        data=individual_classifications,
                        hue_key="classification_success",
                    )
                )

                plt.xlabel(f"Specimen classification\n{model_name}")
                plt.ylabel(label)
                sns.despine(ax=ax)
                plt.tight_layout()
                genetools.plots.savefig(
                    fig,
                    f"{highres_results_output_prefix}.errors_versus_{metric}.test_set_performance.{model_name}.with_abstention.vertical.png",
                    dpi=300,
                )
                if test_results is not None and len(test_results) > 0:
                    with open(
                        f"{highres_results_output_prefix}.errors_versus_{metric}.test_set_performance.{model_name}.with_abstention.vertical.test_results.txt",
                        "w",
                    ) as f:
                        f.write(test_results[0].data.formatted_output)
                plt.close(fig)

            try:
                # Try to load global fold classifier for analysis, too.
                # It wasn't included in the ExperimentSet, because no .metadata_joblib was generated, since the global fold does not have a test set.
                # Note that this will only process global fold classifiers for models that were trained for at least one cross validation fold.
                global_fold_classifier = BlendingMetamodel.from_disk(
                    fold_id=-1,
                    metamodel_name=model_name,
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_column,
                    base_model_train_fold_name=base_model_train_fold_name,
                    metamodel_fold_label_train=metamodel_fold_label_train,
                    metamodel_flavor=metamodel_flavor,
                    # We are providing a metamodel_config directly to avoid expensive load from disk.
                    # But we are cheating here, because our metamodel_config was created with use_stubs_instead_of_submodels=True
                    # This means that the metamodel itself will be loaded from disk, but it won't be able to make new predictions because the submodels are nonexistent.
                    # This is fine for our purposes here, and saves us a ton of time.
                    metamodel_config=metamodel_config,
                )
            except FileNotFoundError as err:
                # Global fold classifier does not exist,
                # either because it was not trained yet,
                # or because we may be using a single fold 0 (if config.cross_validation_split_strategy.value.is_single_fold_only is True).
                logger.warning(
                    f"No global fold classifier found for {model_name}: {err}"
                )
                global_fold_classifier = None

            analyze_feature_importances(
                model_name=model_name,
                model_global_performance=model_global_performance,
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                metamodel_flavor=metamodel_flavor,
                highres_results_output_prefix=highres_results_output_prefix,
                global_fold_classifier=global_fold_classifier,
            )

            # Plot additional model diagnostics for models with internal cross validation over a range of hyperparameters
            if any(
                model_name.startswith(s)
                for s in ["lasso_cv", "ridge_cv", "elasticnet_cv"]
            ):
                if "_lambda1se" in model_name:
                    # TODO(bugfix): Remove this when we fix the bug in switch_lambda() that doesn't copy n_train_samples_ for plot_cross_validation_curve()
                    # This workaround is ok for now, beacause the lambda1se model plot is going to be identical to the other model plot
                    continue

                def _get_classifiers():
                    # load classifier from disk.
                    for (
                        fold_id,
                        per_fold_performance,
                    ) in model_global_performance.per_fold_outputs.items():
                        yield (fold_id, per_fold_performance.classifier)
                    if global_fold_classifier is not None:
                        # Also include global fold classifier, if it exists
                        yield (-1, global_fold_classifier)

                for fold_id, clf in _get_classifiers():
                    if isinstance(clf, BlendingMetamodel):
                        # Unwrap if it's a BlendingMetamodel
                        clf = clf._inner

                    # it's probably a Pipeline - unwrap it
                    clf = crosseval._get_final_estimator_if_pipeline(clf)

                    if not isinstance(clf, GlmnetLogitNetWrapper):
                        # it should be a GlmnetLogitNetWrapper - skip
                        # this may be a glmnet model inside another wrapper, like an OvR wrapper.
                        logger.info(
                            f"Expected {model_name} for fold {fold_id} to be of type GlmnetLogitNetWrapper, got {type(clf)}. Skipping plot_cross_validation_curve()."
                        )
                        continue

                    # TODO(bugfix): Scorer name should be stored in clf automatically. (Remove the hardcoded scorer_name here once we fix the scorer name bug.)
                    # In internal/nested cross validation, we optimize MCC for metamodel, but AUC for base models. See discussion in core code
                    fig = clf.plot_cross_validation_curve(scorer_name="MCC")
                    genetools.plots.savefig(
                        fig,
                        f"{highres_results_output_prefix}.internal_cross_validation_hyperparameter_diagnostics.{model_name}.fold_{fold_id}.png",
                        dpi=300,
                    )
                    plt.close(fig)

    except Exception as err:
        logger.exception(
            f"{gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor}, config {metamodel_config} failed with error: {err}"
        )

    return computed_abstentions


# %%
def run_analysis(gene_locus: GeneLocus, target_obs_column: TargetObsColumnEnum):
    try:
        flavors = train_metamodel.get_metamodel_flavors(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            fold_id=config.all_fold_ids[0],
            base_model_train_fold_name=base_model_train_fold_name,
            # For the sake of speed, we are choosing not to load all the submodels from disk here.
            use_stubs_instead_of_submodels=True,
        )
    except Exception as err:
        logger.warning(
            f"Failed to generate metamodel flavors for {gene_locus}, {target_obs_column}: {err}"
        )
        return

    abstentions_from_default_flavor = None

    # Process "default" first to get list of abstentions.
    if "default" in flavors:
        abstentions_from_default_flavor = analyze_metamodel_flavor(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            metamodel_flavor="default",
            metamodel_config=flavors["default"],
            forced_abstentions=None,
        )
        logger.info(
            f"abstentions_from_default_flavor = {abstentions_from_default_flavor}"
        )

    # Process remaining flavors
    for metamodel_flavor, metamodel_config in flavors.items():
        if metamodel_flavor == "default":
            # Already processed, skip
            continue
        analyze_metamodel_flavor(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            metamodel_flavor=metamodel_flavor,
            metamodel_config=metamodel_config,
            forced_abstentions=abstentions_from_default_flavor,
        )


# %%

# %%
# Individual gene locus
for gene_locus in config.gene_loci_used:
    print(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    for target_obs_column in config.classification_targets:
        run_analysis(gene_locus=gene_locus, target_obs_column=target_obs_column)

# %%
# Together in combined metamodel
if len(config.gene_loci_used) > 1:
    print(config.gene_loci_used)
    for target_obs_column in config.classification_targets:
        run_analysis(
            gene_locus=config.gene_loci_used, target_obs_column=target_obs_column
        )

# %%

# %%

# %%
