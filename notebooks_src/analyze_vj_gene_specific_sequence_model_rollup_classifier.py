# -*- coding: utf-8 -*-
# %%

# %%
from IPython.display import display, Markdown
from malid import config, logger
import crosseval
from malid.trained_model_wrappers import (
    VJGeneSpecificSequenceModelRollupClassifier,
)
from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SequenceSubsetStrategy,
)
import pandas as pd
import re
from summarynb import show
from typing import Callable, Optional
import numpy as np
import multiclass_metrics
from malid.datamodels import (
    SampleWeightStrategy,
    healthy_label,
    TargetObsColumnEnum,
    GeneLocus,
)

sample_weight_strategy = config.sample_weight_strategy

# %%
config.embedder.name

# %%
# Read this optional flag from environment variables.
from environs import Env

env = Env()
default_sequence_subset_strategy: SequenceSubsetStrategy = (
    config.metamodel_base_model_names.base_sequence_model_subset_strategy
)
sequence_subset_strategy: SequenceSubsetStrategy = env.enum(
    "SEQUENCE_SUBSET_STRATEGY",
    type=SequenceSubsetStrategy,
    ignore_case=True,
    # Pass .name as default here, because matching happens on string name:
    # The internal "if enum_value.name.lower() == value.lower()" will fail unless value is the .name. The enum object itself doesn't have a .lower()
    default=default_sequence_subset_strategy.name,
)
sequence_subset_strategy

# %%
base_model_fold_label_train = "train_smaller1"
rollup_model_fold_label_train = "train_smaller2"

# %%
base_classifier = sequence_subset_strategy.base_model
base_classifier


# %%
def get_available_base_model_names(sequence_models_base_dir):
    # Auto-detect available base sequence model names
    # TODO: Use the list of available base models configured in training script.
    for dirname in (
        sequence_models_base_dir
        / f"rollup_models_specialized_for_{base_classifier.split_short_name}"
    ).glob(f"base_model_*_trained_on_{base_model_fold_label_train}"):
        model_name_search = re.search(
            r"base_model_(.*)_trained_on", str(dirname)
        )  # returns None if nothing found
        if model_name_search:
            yield model_name_search.group(1)


# %%
for gene_locus in config.gene_loci_used:
    for target_obs_column in config.classification_targets:
        try:
            target_obs_column.confirm_compatibility_with_gene_locus(gene_locus)
            target_obs_column.confirm_compatibility_with_cross_validation_split_strategy(
                config.cross_validation_split_strategy
            )
        except Exception as err:
            # Skip invalid combinations
            logger.warning(f"{err}. Skipping.")
            continue

        display(
            Markdown(f"## {gene_locus}, {target_obs_column}, {sample_weight_strategy}")
        )
        sequence_models_base_dir = base_classifier._get_model_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )
        for base_sequence_model_name in get_available_base_model_names(
            sequence_models_base_dir
        ):
            models_base_dir = (
                VJGeneSpecificSequenceModelRollupClassifier._get_model_base_dir(
                    sequence_models_base_dir=sequence_models_base_dir,
                    base_sequence_model_name=base_sequence_model_name,
                    base_model_train_fold_label=base_model_fold_label_train,
                    split_short_name=base_classifier.split_short_name,
                )
            )  # should already exist

            output_base_dir = (
                VJGeneSpecificSequenceModelRollupClassifier._get_output_base_dir(
                    sequence_model_output_base_dir=base_classifier._get_output_base_dir(
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        sample_weight_strategy=sample_weight_strategy,
                    ),
                    base_sequence_model_name=base_sequence_model_name,
                    base_model_train_fold_label=base_model_fold_label_train,
                    split_short_name=base_classifier.split_short_name,
                )
            )  # might not yet exist
            output_base_dir.mkdir(parents=True, exist_ok=True)  # create if needed

            model_output_prefix = (
                models_base_dir / f"{rollup_model_fold_label_train}_model"
            )
            results_output_prefix = (
                output_base_dir / f"{rollup_model_fold_label_train}_model"
            )

            try:
                logger.debug(
                    f"{gene_locus}, {target_obs_column}, {sample_weight_strategy}, base model {base_sequence_model_name} from {model_output_prefix} to {results_output_prefix}"
                )

                ## Load and summarize
                experiment_set = crosseval.ExperimentSet.load_from_disk(
                    output_prefix=model_output_prefix
                )

                # Remove global fold (we trained global fold model, but now get evaluation scores on cross-validation folds only)
                # TODO: make kdict support: del self.model_outputs[:, fold_id]
                for key in experiment_set.model_outputs[:, -1].keys():
                    logger.debug(f"Removing {key} (global fold)")
                    del experiment_set.model_outputs[key]

                experiment_set_global_performance = experiment_set.summarize()
                func_generate_classification_report_fname = (
                    lambda model_name: f"{results_output_prefix}.classification_report.{model_name}.txt"
                )
                func_generate_confusion_matrix_fname = (
                    lambda model_name: f"{results_output_prefix}.confusion_matrix.{model_name}.png"
                )
                experiment_set_global_performance.export_all_models(
                    func_generate_classification_report_fname=func_generate_classification_report_fname,
                    func_generate_confusion_matrix_fname=func_generate_confusion_matrix_fname,
                    dpi=72,
                )

                # Note that the confusion matrix is hard to interpret here:
                # We use OvR classifiers along with different sets of features,
                # so the class probabilities are not necessarily comparable and choosing the class with highest probability as winning label is not appropriate.
                # For this reason, model3-aggregation label evaluation is a fraught exercise. But the AUC is still evaluatable.
                # See malid/train/train_vj_gene_specific_sequence_model_rollup.py for further discussion.

                # Before generating model comparison stats, add a metric that excludes the Healthy class.
                # Rationale: Predicting that sequences are "healthy" is hard to reason about.
                # We may not even want to be taking this into consideration when choosing the version of model 3 that goes into the metamodel.
                # Do this only for main disease classification target.
                if target_obs_column == TargetObsColumnEnum.disease:
                    # Here we add an extra AUC measure that is restricted to disease classes only:
                    # - we subset y_true and y_score by removing rows where y_true == healthy,
                    # - and we also remove healthy class predicted probabilities so those don't get included in the multiclass AUC averaging.
                    def score_wrapper_excluding_a_class(
                        y_true: np.ndarray,
                        y_score: np.ndarray,
                        exclude: str,
                        func: Callable,
                        labels: Optional[np.ndarray] = None,
                        sample_weight: Optional[np.ndarray] = None,
                        **kwargs,
                    ):
                        y_score = np.array(y_score)
                        y_true = np.array(y_true)

                        # Remove examples where ground truth label matches label-to-exclude
                        mask = y_true != exclude
                        y_score = y_score[mask, :]
                        y_true = y_true[mask]
                        if sample_weight is not None:
                            sample_weight = np.array(sample_weight)
                            sample_weight = sample_weight[mask]

                        # Remove predicted classes where label matches label-to-exclude
                        # (probabilities will no longer sum to 1)
                        if labels is not None and exclude in labels:
                            labels = np.array(labels)
                            mask = labels != exclude
                            y_score = y_score[:, mask]
                            labels = labels[mask]

                        # Call original function with modified data
                        return func(
                            y_true,
                            y_score,
                            labels=labels,
                            sample_weight=sample_weight,
                            **kwargs,
                        )

                    # Wire up the new metric, like we do in crosseval.DEFAULT_PROBABILITY_SCORERS
                    custom_probability_scorers = crosseval.DEFAULT_PROBABILITY_SCORERS | {
                        "rocauc_without_healthy": (
                            score_wrapper_excluding_a_class,
                            "ROC-AUC without Healthy (weighted OvO)",
                            {
                                # Label to be removed:
                                "exclude": healthy_label,
                                # It will call this function after modifying the data to remove healthy_label:
                                "func": multiclass_metrics.roc_auc_score,
                                # Standard kwargs, as in crosseval.DEFAULT_PROBABILITY_SCORERS:
                                "average": "weighted",
                                "multi_class": "ovo",
                            },
                        ),
                    }
                else:
                    # pass None to get default values
                    custom_probability_scorers = None
                combined_stats = (
                    experiment_set_global_performance.get_model_comparison_stats(
                        probability_scorers=custom_probability_scorers,
                    )
                )
                combined_stats.to_csv(
                    f"{results_output_prefix}.compare_model_scores.tsv",
                    sep="\t",
                )
                display(Markdown(f"### Base model {base_sequence_model_name}"))
                display(combined_stats)

                # Show the saved confusion matrices
                model_names = combined_stats.index
                # exclude dummies
                model_names = model_names[~model_names.str.startswith("dummy")]
                show(
                    [
                        #         [
                        #             func_generate_classification_report_fname(model_name)
                        #             for model_name in model_names
                        #         ],
                        [
                            func_generate_confusion_matrix_fname(model_name)
                            for model_name in model_names
                        ],
                    ],
                    headers=[
                        f"Rollup model: {model_name}" for model_name in model_names
                    ],
                    max_width=500,
                )

            except Exception as err:
                logger.exception(
                    f"Failed to analyze {gene_locus}, {target_obs_column}, {sample_weight_strategy}, base model {base_sequence_model_name}: {err}"
                )

            print("*" * 80)
        # Done with this target+locus
        display(Markdown(f"---"))

# %%

# %% [markdown]
# # Report on the best-performing model for each locus

# %%
def show_summary(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    top_N_base_models_to_investigate_aggregation_models: int = 3,
):
    display(Markdown(f"## {gene_locus}, {target_obs_column}, {sample_weight_strategy}"))
    # Two ways to sort the results:
    for sort_col_description, sort_col_name in [
        ("normally", "ROC-AUC (weighted OvO) per fold"),
        (
            # This sort column is not always present. It's only added for disease classification target.
            # Below, we will gracefully skip this sort column if it's not present.
            "diseases only without healthy",
            "ROC-AUC without Healthy (weighted OvO) per fold",
        ),
    ]:
        # Re-using code from above to generate output file paths
        sequence_models_base_dir = base_classifier._get_model_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )
        results = {}
        results_single_row = {}
        # Loop over base models
        for base_sequence_model_name in get_available_base_model_names(
            sequence_models_base_dir
        ):
            # Given a base model:
            # Load rollup model performance comparison spreadsheet
            output_base_dir = (
                VJGeneSpecificSequenceModelRollupClassifier._get_output_base_dir(
                    sequence_model_output_base_dir=base_classifier._get_output_base_dir(
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        sample_weight_strategy=sample_weight_strategy,
                    ),
                    base_sequence_model_name=base_sequence_model_name,
                    base_model_train_fold_label=base_model_fold_label_train,
                    split_short_name=base_classifier.split_short_name,
                )
            )
            try:
                df = pd.read_csv(
                    output_base_dir
                    / f"{rollup_model_fold_label_train}_model.compare_model_scores.tsv",
                    sep="\t",
                    index_col=0,
                )
            except FileNotFoundError:
                # If the file is not found, skip to the next base model
                logger.warning(
                    f"Results file not found for {gene_locus}, {target_obs_column}, {sample_weight_strategy}, base model {base_sequence_model_name}. See earlier in this notebook for possible errors generating that file."
                )
                continue

            if sort_col_name not in df.columns:
                # If sort column does not exist in the data, skip to the next sort column
                # (We are actually probably ok to use break instead of continue here: if the sort column is missing, it should be missing for all base models, so we can't do anything with this sort column.
                # But just in case, we use continue to skip to the next sort column only after trying all base models, just in case there's a reason some base models might have the sort column while others don't.)
                continue

            results[base_sequence_model_name] = df.sort_values(
                sort_col_name, ascending=False
            )
            # Keep only the best-performing rollup model for each base model
            # In other words, we choose only one row (best performance) for each base model
            results_single_row[base_sequence_model_name] = results[
                base_sequence_model_name
            ].iloc[0]

        if not results_single_row:
            # If no results were added (due to missing sort columns), continue to next sort_col_name
            continue

        results_single_row = pd.DataFrame(results_single_row).T.sort_values(
            sort_col_name, ascending=False
        )
        # put the sort column first
        displayed_column_order = [sort_col_name] + [
            col for col in results_single_row.columns if col != sort_col_name
        ]
        display(
            Markdown(
                f"### Evaluating {sort_col_description}: best performance for each base model:"
            )
        )
        display(results_single_row[displayed_column_order])

        # Then zoom into the top N performing base models, and show the rollup model options for them.
        for base_model in results_single_row.index[
            :top_N_base_models_to_investigate_aggregation_models
        ]:
            display(
                Markdown(f"#### Base model {base_model} has aggregation model options:")
            )
            # put the sort column first
            displayed_column_order = [sort_col_name] + [
                col for col in results[base_model].columns if col != sort_col_name
            ]
            display(results[base_model].head(n=3)[displayed_column_order])

        display(Markdown(f"---"))


# %%
for gene_locus in config.gene_loci_used:
    for target_obs_column in config.classification_targets:
        show_summary(gene_locus, target_obs_column, sample_weight_strategy)

# %%
