# -*- coding: utf-8 -*-
# %% [markdown]
#
# # Exact matches model
#
# 1. **For each training set sequence,  for each disease class, run Fisher’s exact test on 2x2 contingency table** counting the number of unique participants that are/aren't in a disease class and have/don't have a particular V/J/CDR3.
#
#
# ```
# Contigency table is:
# ---------------------------------------------|---------------------- |
# is other class |                                                     |
# ----------------      # unique participants with this property       -
# is this class  |                                                     |
# ---------------|-----------------------------|---------------------- |
#                | does not have this sequence | has this sequence --- |
# ```
#
# This returns p-values representing how enriched each sequence is in each disease class.
#
# No multiple hypothesis correction, but we do try some low p-value thresholds below that may simulate Bonferroni correction.
#
# 2. **Given a p-value threshold, featurize and fit a model.**
#
# Featurization into `n_specimens x n_diseases` feature matrix:
# - For a specimen, count how many of its sequences are exact matches to [sequences whose Covid-19 specificity p-values <= p-value threshold], [sequences whose HIV specificity p-values <= p-value threshold], etc.
# - Divide by total number of sequences in the specimen
#
#
#
#
# 3. **Evaluate on validation set. Choose p-value threshold that gives highest AUC.**
#
#
# 4. **Evaluate best model on test set, possibly after tuning decision thresholds on validation set.**
#
# Note that we explicitly find Healthy-specific sequences in the featurization.

# %%

# %%

# %% [markdown]
# # Analyze exact matches model performance on validation set

# %%
from IPython.display import display, Markdown
from malid import config, logger, io
from malid.external import model_evaluation
from malid.datamodels import (
    combine_classification_option_names,
    TargetObsColumnEnum,
    GeneLocus,
)
from malid.trained_model_wrappers import ExactMatchesClassifier
import joblib
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
import matplotlib.ticker
from summarynb import show, indexed_csv, table, chunks

# %%

# %% [markdown]
# ## Analyze on validation set

# %%
for gene_locus in config.gene_loci_used:
    for target_obs_col in [
        TargetObsColumnEnum.disease
    ]:  # config.classification_targets:
        models_base_dir = ExactMatchesClassifier._get_model_base_dir(
            gene_locus=gene_locus, target_obs_column=target_obs_col
        )  # should already exist

        output_base_dir = (
            config.paths.exact_matches_output_dir
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

        except Exception as err:
            logger.exception(f"{gene_locus}, {target_obs_col} failed with error: {err}")


# %%

# %%
model_names = combined_stats.index[
    ~combined_stats.index.str.startswith("dummy_")
].tolist()
model_names

# %%

# %% [markdown]
# # Evaluate on test set

# %%
target_obs_column = TargetObsColumnEnum.disease
featurized_all = {}
for gene_locus in config.gene_loci_used:
    print(gene_locus)
    results_output_dir = (
        config.paths.exact_matches_output_dir
        / gene_locus.name
        / combine_classification_option_names(target_obs_column)
    )  # already created above when analyzing validation set

    results = model_evaluation.ExperimentSet()
    for fold_id in config.cross_validation_fold_ids:
        adata_test = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label="test",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )

        # load validation set for tuning
        adata_validation = io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label="validation",
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )

        for model_name in model_names:
            clf = ExactMatchesClassifier(
                fold_id=fold_id,
                model_name=model_name,
                fold_label_train="train_smaller",
                target_obs_column=target_obs_column,
                gene_locus=gene_locus,
            )

            # Featurization may vary by model, because p-value threshold may vary by model
            featurized: model_evaluation.FeaturizedData = clf.featurize(adata_test)
            # store for later analysis of featurized.extras
            featurized_all[(gene_locus, fold_id, model_name)] = featurized

            def _make_result(
                name: str,
                model: model_evaluation.Classifier,
                featurized_test: model_evaluation.FeaturizedData,
            ):
                # For curiosity's sake, what does the model predict for an input row of all 0s?
                row_of_0s: pd.DataFrame = (
                    pd.Series(1, index=model.feature_names_in_).to_frame().T
                )
                row_of_0s_predict = model.predict(row_of_0s)[0]
                row_of_0s_predict_proba = (
                    pd.DataFrame(model.predict_proba(row_of_0s), columns=model.classes_)
                    .iloc[0]
                    .apply(lambda p: f"{p:0.2%}")
                    .to_dict()
                )
                logger.info(
                    f"For a row of all zeros ({row_of_0s.shape}), model {name} on fold {fold_id}, {gene_locus}, {target_obs_column} predicts: {row_of_0s_predict} with probabilities {row_of_0s_predict_proba}"
                )

                # Create performance object
                return model_evaluation.ModelSingleFoldPerformance(
                    model_name=name,
                    fold_id=fold_id,
                    y_true=featurized_test.y,
                    clf=model,
                    X_test=featurized_test.X,
                    fold_label_train="train_smaller",
                    fold_label_test="test",
                    test_metadata=featurized_test.metadata,
                    test_abstentions=featurized_test.abstained_sample_y,
                    test_abstention_metadata=featurized_test.abstained_sample_metadata,
                )

            results.add(_make_result(model_name, clf, featurized))

            try:
                # tune on validation set
                clf_tuned = clf.tune_model_decision_thresholds_to_validation_set(
                    validation_set=adata_validation
                )

                # save out tuned model
                joblib.dump(
                    clf_tuned,
                    clf_tuned.models_base_dir
                    / f"train_smaller_model.{model_name}.decision_thresholds_tuned.{fold_id}.joblib",
                )

                # add tuned model performance
                results.add(
                    _make_result(
                        f"{model_name}_decision_thresholds_tuned", clf_tuned, featurized
                    )
                )
            except Exception as err:
                logger.warning(
                    f"Failed to tune {model_name} on validation set for fold {fold_id}, {gene_locus}, {target_obs_column}"
                )

        # Clear RAM
        del adata_test, adata_validation
        io.clear_cached_fold_embeddings()
        gc.collect()

    # summarize performance across folds and models
    summary = results.summarize()
    combined_stats = summary.get_model_comparison_stats()
    display(combined_stats)

    fname = (
        results_output_dir
        / f"train_smaller_model.compare_model_scores.test_set_performance.tsv"
    )
    combined_stats.to_csv(fname, sep="\t")

    summary.export_all_models(
        func_generate_classification_report_fname=lambda model_name: results_output_dir
        / f"train_smaller_model.test_set_performance.{model_name}.classification_report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: results_output_dir
        / f"train_smaller_model.test_set_performance.{model_name}.confusion_matrix.png",
        dpi=72,
    )

# %%

# %% [markdown]
# # Test set performance - summary

# %%
for gene_locus in config.gene_loci_used:
    display(Markdown(f"## {gene_locus} test set performance"))
    results_output_dir = (
        config.paths.exact_matches_output_dir
        / gene_locus.name
        / combine_classification_option_names(target_obs_column)
    )
    results_df = pd.read_csv(
        results_output_dir
        / "train_smaller_model.compare_model_scores.test_set_performance.tsv",
        sep="\t",
        index_col=0,
    )
    show(table(results_df))
    for model_name_chunks in chunks(results_df.index.sort_values(), 4):
        show(
            [
                [
                    results_output_dir
                    / f"train_smaller_model.test_set_performance.{model_name}.confusion_matrix.png"
                    for model_name in model_name_chunks
                ],
                [
                    results_output_dir
                    / f"train_smaller_model.test_set_performance.{model_name}.classification_report.txt"
                    for model_name in model_name_chunks
                ],
            ],
            headers=model_name_chunks,
        )

# %%

# %% [markdown]
# # Compare to CDR3 clustering

# %%
for gene_locus in config.gene_loci_used:
    results_output_prefix_alternative = (
        config.paths.convergent_clusters_output_dir
        / gene_locus.name
        / combine_classification_option_names(target_obs_column)
        / "train_smaller_model"
    )

    display(
        Markdown(
            f"## Compare to CDR3 clustering: {gene_locus}, {target_obs_column}, eval on test set, with and without tuning on validation set"
        )
    )
    try:
        ## All results in a table
        all_results = pd.read_csv(
            f"{results_output_prefix_alternative}.compare_model_scores.test_set_performance.tsv",
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
                        f"{results_output_prefix_alternative}.test_set_performance.{model_name}.classification_report.txt"
                        for model_name in model_names
                    ],
                    [
                        f"{results_output_prefix_alternative}.test_set_performance.{model_name}.confusion_matrix.png"
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

# %% [markdown]
# # Which p values were chosen (varies by locus, model, and fold)? How many disease-associated sequences found?

# %%
for gene_locus in config.gene_loci_used:
    for fold_id in config.cross_validation_fold_ids:
        for model_name in model_names:
            clf = ExactMatchesClassifier(
                fold_id=fold_id,
                model_name=model_name,
                fold_label_train="train_smaller",
                target_obs_column=TargetObsColumnEnum.disease,
                gene_locus=gene_locus,
            )
            p_value = clf.p_value_threshold
            seqs = clf.sequences_with_fisher_result
            print(
                f"{gene_locus}, fold {fold_id}, {model_name}: p = {p_value}. Number of disease associated sequences: {(seqs <= p_value).sum().to_dict()}"
            )
        print()

# %%

# %%
