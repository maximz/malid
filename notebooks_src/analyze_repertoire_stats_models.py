# -*- coding: utf-8 -*-
# %%

# %% [markdown]
# # Analyze repertoire stats model performance on validation set

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union

# %matplotlib inline
import seaborn as sns
import genetools
from IPython.display import display, Markdown

# %%
from malid import config, helpers, logger
import crosseval
from malid.trained_model_wrappers import RepertoireClassifier


# %%

# %%

# %% [markdown]
# # Analyze

# %%
for gene_locus in config.gene_loci_used:
    for target_obs_col in config.classification_targets:
        try:
            target_obs_col.confirm_compatibility_with_gene_locus(gene_locus)
            target_obs_col.confirm_compatibility_with_cross_validation_split_strategy(
                config.cross_validation_split_strategy
            )
        except Exception as err:
            # Skip invalid combinations
            logger.warning(f"{err}. Skipping.")
            continue

        models_base_dir = RepertoireClassifier._get_model_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_col,
            sample_weight_strategy=config.sample_weight_strategy,
        )  # should already exist

        output_base_dir = RepertoireClassifier._get_output_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_col,
            sample_weight_strategy=config.sample_weight_strategy,
        )  # might not yet exist
        output_base_dir.mkdir(parents=True, exist_ok=True)  # create if needed

        model_output_prefix = models_base_dir / "train_smaller_model"
        results_output_prefix = output_base_dir / "train_smaller_model"

        try:
            logger.info(
                f"{gene_locus}, {target_obs_col} from {model_output_prefix} to {results_output_prefix}"
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
            experiment_set_global_performance.export_all_models(
                func_generate_classification_report_fname=lambda model_name: f"{results_output_prefix}.classification_report.{model_name}.txt",
                func_generate_confusion_matrix_fname=lambda model_name: f"{results_output_prefix}.confusion_matrix.{model_name}.png",
                dpi=72,
            )
            combined_stats = (
                experiment_set_global_performance.get_model_comparison_stats(sort=True)
            )
            combined_stats.to_csv(
                f"{results_output_prefix}.compare_model_scores.tsv",
                sep="\t",
            )
            display(
                Markdown(
                    f"## {gene_locus}, {target_obs_col} from {model_output_prefix} to {results_output_prefix}"
                )
            )
            display(combined_stats)

            ## Review binary misclassifications: Binary prediction vs ground truth
            # For binary case, make new confusion matrix of actual disease label (y) vs predicted y_binary
            # (But this changes global score metrics)
            if (
                target_obs_col.value.is_target_binary_for_repertoire_composition_classifier
            ):
                # this is a binary healthy/sick classifier
                # re-summarize with different ground truth label
                experiment_set.summarize(
                    global_evaluation_column_name=target_obs_col.value.confusion_matrix_expanded_column_name
                ).export_all_models(
                    func_generate_classification_report_fname=lambda model_name: f"{results_output_prefix}.classification_report.{model_name}.binary_vs_ground_truth.txt",
                    func_generate_confusion_matrix_fname=lambda model_name: f"{results_output_prefix}.confusion_matrix.{model_name}.binary_vs_ground_truth.png",
                    confusion_matrix_pred_label="Predicted binary label",
                    dpi=72,
                )

            ## also create the “coefficient variability” plot, over all the CV folds
            for (
                model_name,
                model_global_performance,
            ) in experiment_set_global_performance.model_global_performances.items():
                # get feature importances for each fold
                feature_importances: Union[
                    pd.DataFrame, None
                ] = model_global_performance.feature_importances

                if feature_importances is not None:
                    # feature importances are available for this model
                    fig = plt.figure(figsize=(9, 9))
                    sns.boxplot(data=feature_importances.abs(), orient="h")
                    plt.title(
                        f"Feature importance (absolute value) variability: {model_name}"
                    )
                    plt.tight_layout()
                    genetools.plots.savefig(
                        fig,
                        f"{results_output_prefix}.feature_importances.{model_name}.png",
                        dpi=72,
                    )
                    plt.close(fig)

        except Exception as err:
            logger.exception(f"{gene_locus}, {target_obs_col} failed with error: {err}")


# %%

# %%

# %%

# %%
