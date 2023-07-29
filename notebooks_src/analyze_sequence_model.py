# -*- coding: utf-8 -*-
# %%

# %%

# %%
from IPython.display import display, Markdown
from malid import config, logger
from malid.external import model_evaluation
from malid.trained_model_wrappers import SequenceClassifier
from malid.datamodels import (
    TargetObsColumnEnum,
    SampleWeightStrategy,
    combine_classification_option_names,
)

sample_weight_strategy = SampleWeightStrategy.ISOTYPE_USAGE

# %%

# %%
for gene_locus in config.gene_loci_used:
    for target_obs_column in config.classification_targets:
        models_base_dir = SequenceClassifier._get_model_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )  # should already exist
        model_location = models_base_dir / "train_smaller_model"

        output_dir = (
            config.paths.sequence_models_output_dir
            / gene_locus.name
            / combine_classification_option_names(
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
            )
        )  # might not yet exist
        output_dir.mkdir(exist_ok=True, parents=True)  # create if needed

        try:
            logger.info(
                f"{gene_locus}, {target_obs_column}, {sample_weight_strategy} -> {model_location} -> {output_dir}"
            )

            ## Load and summarize
            experiment_set = model_evaluation.ExperimentSet.load_from_disk(
                output_prefix=model_location
            )

            # Remove global fold (we trained global fold model, but now get evaluation scores on cross-validation folds only)
            # TODO: make kdict support: del self.model_outputs[:, fold_id]
            for key in experiment_set.model_outputs[:, -1].keys():
                logger.debug(f"Removing {key} (global fold)")
                del experiment_set.model_outputs[key]

            experiment_set_global_performance = experiment_set.summarize()
            experiment_set_global_performance.export_all_models(
                func_generate_classification_report_fname=lambda model_name: output_dir
                / f"sequence_model.results_on_validation_set.{model_name}.classification_report.txt",
                func_generate_confusion_matrix_fname=lambda model_name: output_dir
                / f"sequence_model.results_on_validation_set.{model_name}.confusion_matrix.png",
                dpi=72,
            )
            combined_stats = (
                experiment_set_global_performance.get_model_comparison_stats()
            )
            combined_stats.to_csv(
                output_dir
                / "sequence_model.results_on_validation_set.compare_model_scores.tsv",
                sep="\t",
            )

            # Display
            display(
                Markdown(
                    f"# {gene_locus}, {target_obs_column}, {sample_weight_strategy}"
                )
            )
            display(combined_stats)

        except Exception as err:
            logger.exception(
                f"Failed to analyze {gene_locus}, {target_obs_column}, {sample_weight_strategy}: {err}"
            )

        print("*" * 80)

# %%

# %%
