# %%
from summarynb import show, indexed_csv, table, chunks
from malid.external.summarynb_extras import plaintext, empty
from malid import config, logger
import pandas as pd
from IPython.display import display, Markdown
from malid.trained_model_wrappers import RepertoireClassifier

# %%
fold_label_train, fold_label_validation = config.get_fold_split_labels()

# %%

# %%

# %% [markdown]
# # Summary statistics of the repertoire --> classifiers
#
# ## Features for each specimen's repertoire, per isotype:
#
# Derived from same sampled versions of repertoires used for other models:
#
# * Top 15 PCs from V-J gene use counts
# * Median sequence SHM rate
# * Proportion of sequences at least 1% mutated
#
# All standardized to zero mean and unit variance.
#
# _Some models will not have feature importances._

# %%

# %%
for gene_locus in config.gene_loci_used:
    # map target_obs_column to results_output_prefix
    targets = {}
    for target in config.classification_targets:
        targets[target] = (
            RepertoireClassifier._get_output_base_dir(
                gene_locus=gene_locus,
                target_obs_column=target,
                sample_weight_strategy=config.sample_weight_strategy,
            )
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
                        # mistakes
                        [
                            f"{results_output_prefix}.confusion_matrix.{model_name}.binary_vs_ground_truth.png"
                            for model_name in model_names
                        ],
                        # feature importances
                        [
                            f"{results_output_prefix}.feature_importances.{model_name}.png"
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
                        # mistakes
                        [
                            f"{results_output_prefix}.test_set_performance.{model_name}.confusion_matrix.binary_vs_ground_truth.png"
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
