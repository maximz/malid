# -*- coding: utf-8 -*-
# %%

# %%
from summarynb import show, indexed_csv, table, chunks
from malid.external.summarynb_extras import plaintext, empty
from malid import config, helpers, logger
import pandas as pd
from IPython.display import display, Markdown
from malid.trained_model_wrappers import ConvergentClusterClassifier


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
# fold_label_train, fold_label_validation = config.get_fold_split_labels()

# Set training and held-out fold names as in scripts/train_convergent_clustering_models.py:
fold_label_train = "train_smaller1"  # Train on portion of train_smaller used to cluster sequences and train patient-level (really, specimen-level) classifier based on cluster hits
fold_label_validation = "train_smaller2"  # Evaluate on portion of train_smaller used to choose best p-value threshold for cluster association with disease, to prune cluster list

# %%

# %%

# %%
for gene_locus in config.gene_loci_used:
    # map target_obs_column to results_output_prefix
    targets = {}
    for target in config.classification_targets:
        targets[target] = (
            ConvergentClusterClassifier._get_output_base_dir(
                gene_locus=gene_locus,
                target_obs_column=target,
            )
            / f"{fold_label_train}_model"
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
                f"## Apply {fold_label_train} model -- Validation set performance - With and without tuning on validation set"
            )
        )
        try:
            ## All results in a table
            all_results = pd.read_csv(
                f"{results_output_prefix}.compare_model_scores.validation_set_performance.tsv",
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
                                f"{results_output_prefix}.validation_set_performance.{model_name}.classification_report.txt"
                            )
                            for model_name in model_names
                        ],
                        [
                            f"{results_output_prefix}.validation_set_performance.{model_name}.confusion_matrix.png"
                            for model_name in model_names
                        ],
                    ],
                    max_width=500,
                    headers=model_names,
                )
        except FileNotFoundError as err:
            logger.warning(f"Not run: {err}")

        display(
            Markdown(
                f"## Apply {fold_label_train} model -- Test set performance - With and without tuning on validation set"
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
