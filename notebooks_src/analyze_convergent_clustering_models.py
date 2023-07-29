# -*- coding: utf-8 -*-
# %%

# %% [markdown]
# # Analyze convergent clustering model performance on validation set

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns
import genetools
from IPython.display import display

# %%
from malid import config, logger
from malid.external import model_evaluation
from malid.datamodels import (
    combine_classification_option_names,
)
from malid.trained_model_wrappers import ConvergentClusterClassifier

# %%

# %%

# %%

# %% [markdown]
# # Analyze

# %%
for gene_locus in config.gene_loci_used:
    for target_obs_col in config.classification_targets:
        models_base_dir = ConvergentClusterClassifier._get_model_base_dir(
            gene_locus=gene_locus, target_obs_column=target_obs_col
        )  # should already exist

        output_base_dir = (
            config.paths.convergent_clusters_output_dir
            / gene_locus.name
            / combine_classification_option_names(target_obs_col)
        )  # might not yet exist
        output_base_dir.mkdir(parents=True, exist_ok=True)  # create if needed

        model_output_prefix = models_base_dir / "train_smaller_model"
        results_output_prefix = output_base_dir / "train_smaller_model"

        try:
            logger.info(
                f"{gene_locus}, {target_obs_col} from {model_output_prefix} to {results_output_prefix}"
            )

            ## Load and summarize
            experiment_set = model_evaluation.ExperimentSet.load_from_disk(
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
                experiment_set_global_performance.get_model_comparison_stats()
            )
            combined_stats.to_csv(
                f"{results_output_prefix}.compare_model_scores.tsv",
                sep="\t",
            )
            print(gene_locus, target_obs_col)
            display(combined_stats)

            # Which p values were chosen (varies by locus, model, and fold)? How many disease-associated sequences found?
            for fold_id in config.cross_validation_fold_ids:
                for model_name in [
                    "lasso_multiclass",
                    "rf_multiclass",
                    "linearsvm_ovr",
                ]:
                    clf = ConvergentClusterClassifier(
                        fold_id=fold_id,
                        model_name=model_name,
                        fold_label_train="train_smaller",
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_col,
                    )
                    p_value = clf.p_value_threshold
                    clusters = clf.cluster_centroids_with_class_specific_p_values
                    feature_names = clf.feature_names_in_
                    print(
                        f"{gene_locus}, fold {fold_id}, {target_obs_col}, {model_name}: best p value = {p_value}. Number of disease associated clusters: {(clusters[feature_names] <= p_value).sum().to_dict()}"
                    )
                print()

        except Exception as err:
            logger.exception(f"{gene_locus}, {target_obs_col} failed with error: {err}")


# %%

# %%

# %%
