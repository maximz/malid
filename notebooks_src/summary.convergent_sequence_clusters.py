# -*- coding: utf-8 -*-
# %%

# %%
from summarynb import show, indexed_csv, table, chunks
from malid.external.summarynb_extras import plaintext, empty
from malid import config, helpers, logger
from malid.datamodels import (
    TargetObsColumnEnum,
    combine_classification_option_names,
)
import pandas as pd
from IPython.display import display, Markdown


# %%

# %% [markdown]
# # Convergent sequence cluster classifiers
#

# %% [markdown]
# ### Distance thresholds
#
# #### Clustering training set:

# %%
config.sequence_identity_thresholds.cluster_amino_acids_across_patients

# %% [markdown]
# #### Assigning test sequences to clusters:

# %%
config.sequence_identity_thresholds.assign_test_sequences_to_clusters

# %%

# %%
fold_label_train, fold_label_validation = config.get_fold_split_labels()

# %%

# %%

# %%
for gene_locus in config.gene_loci_used:
    # map target_obs_column to results_output_prefix
    targets = {}
    for target in config.classification_targets:
        targets[target] = (
            config.paths.convergent_clusters_output_dir
            / gene_locus.name
            / combine_classification_option_names(target)
            / "train_smaller_model"
        )

    print(targets)

    for target_obs_column, results_output_prefix in targets.items():
        display(
            Markdown(
                f"# {gene_locus}, {target_obs_column} trained on {fold_label_train} set"
            )
        )

        display(Markdown(f"## Specimen predictions on {fold_label_validation} set"))
        try:
            ## All results in a table
            all_results = pd.read_csv(
                f"{results_output_prefix}.compare_model_scores.tsv",
                sep="\t",
                index_col=0,
            )
            show(table(all_results), headers=["All results, sorted"])

            models_of_interest = all_results.index

            ## Confusion matrices
            for model_names in chunks(models_of_interest, 4):
                show(
                    [
                        [
                            plaintext(
                                f"{results_output_prefix}.classification_report.{model_name}.txt"
                            )
                            for model_name in model_names
                        ],
                        [
                            f"{results_output_prefix}.confusion_matrix.{model_name}.png"
                            for model_name in model_names
                        ],
                    ],
                    headers=model_names,
                    max_width=500,
                )
        except FileNotFoundError as err:
            logger.warning(f"Not run: {err}")

        display(
            Markdown(
                "## Apply train-smaller model -- Test set performance - With and without tuning on validation set"
            )
        )
        try:
            ## All results in a table
            all_results = pd.read_csv(
                f"{results_output_prefix}.compare_model_scores.test_set_performance.tsv",
                sep="\t",
                index_col=0,
            )
            show(table(all_results), headers=["All results, sorted"])

            models_of_interest = all_results.index

            ## Confusion matrices
            for model_names in chunks(models_of_interest, 4):
                show(
                    [
                        [
                            plaintext(
                                f"{results_output_prefix}.test_set_performance.{model_name}.classification_report.txt"
                            )
                            for model_name in model_names
                        ],
                        [
                            f"{results_output_prefix}.test_set_performance.{model_name}.confusion_matrix.png"
                            for model_name in model_names
                        ],
                    ],
                    max_width=500,
                    headers=model_names,
                )
        except FileNotFoundError as err:
            logger.warning(f"Not run: {err}")

        display(Markdown("---"))

# %%
