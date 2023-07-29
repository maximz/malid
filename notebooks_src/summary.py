# %%
from pathlib import Path
from summarynb import show
import pandas as pd
from IPython.display import display, Markdown

from malid.external.summarynb_extras import empty
from malid import config, logger
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    combine_classification_option_names,
)
from malid.train import train_metamodel
from malid.trained_model_wrappers import BlendingMetamodel

# %%
model_name_overall_repertoire_composition = (
    config.metamodel_base_model_names.model_name_overall_repertoire_composition
)
model_name_convergent_clustering = (
    config.metamodel_base_model_names.model_name_convergent_clustering
)
model_name_sequence_disease = (
    config.metamodel_base_model_names.model_name_sequence_disease
)
base_model_train_fold_name = "train_smaller"
metamodel_fold_label_train = "validation"

# %%
def summary(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE,
    show_base=True,
    show_metamodel=True,
    show_best_metamodel_only=False,
):
    display(
        Markdown(f"### {gene_locus}, {target_obs_column}, {sample_weight_strategy}")
    )
    try:
        if show_base:
            show(
                [
                    [
                        config.paths.repertoire_stats_classifier_output_dir
                        / gene_locus.name
                        / combine_classification_option_names(
                            target_obs_column=target_obs_column
                        )
                        / f"train_smaller_model.test_set_performance.{model_name_overall_repertoire_composition}.classification_report.txt",
                        config.paths.convergent_clusters_output_dir
                        / gene_locus.name
                        / combine_classification_option_names(
                            target_obs_column=target_obs_column
                        )
                        / f"train_smaller_model.test_set_performance.{model_name_convergent_clustering}.classification_report.txt",
                        config.paths.sequence_models_output_dir
                        / gene_locus.name
                        / "rollup_models"
                        / combine_classification_option_names(
                            target_obs_column=target_obs_column,
                            sample_weight_strategy=sample_weight_strategy,
                        )
                        / f"sequence_prediction_rollup.{model_name_sequence_disease}.train_smaller_model.report.txt",
                    ],
                    [
                        config.paths.repertoire_stats_classifier_output_dir
                        / gene_locus.name
                        / combine_classification_option_names(
                            target_obs_column=target_obs_column
                        )
                        / f"train_smaller_model.test_set_performance.{model_name_overall_repertoire_composition}.confusion_matrix.png",
                        config.paths.convergent_clusters_output_dir
                        / gene_locus.name
                        / combine_classification_option_names(
                            target_obs_column=target_obs_column
                        )
                        / f"train_smaller_model.test_set_performance.{model_name_convergent_clustering}.confusion_matrix.png",
                        config.paths.sequence_models_output_dir
                        / gene_locus.name
                        / "rollup_models"
                        / combine_classification_option_names(
                            target_obs_column=target_obs_column,
                            sample_weight_strategy=sample_weight_strategy,
                        )
                        / f"sequence_prediction_rollup.{model_name_sequence_disease}.train_smaller_model.confusion_matrix.png",
                    ],
                ],
                headers=[
                    f"model1 {model_name_overall_repertoire_composition}",
                    f"model2 {model_name_convergent_clustering}",
                    f"model3-rollup {model_name_sequence_disease}",
                ],
                max_width=400,
            )

            display(Markdown("---"))

        if show_metamodel:
            try:
                flavors = train_metamodel.get_metamodel_flavors(
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_column,
                    fold_id=config.all_fold_ids[0],
                    base_model_train_fold_name=base_model_train_fold_name,
                )
            except Exception as err:
                logger.warning(
                    f"Failed to generate metamodel flavors for {gene_locus}, {target_obs_column}: {err}"
                )
                return
            for metamodel_flavor in flavors.keys():
                display(Markdown(f"#### Metamodel flavor {metamodel_flavor}"))
                _output_suffix = (
                    Path(gene_locus.name)
                    / target_obs_column.name
                    / metamodel_flavor
                    / f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model"
                )
                metamodel_output_prefix = (
                    config.paths.second_stage_blending_metamodel_output_dir
                    / _output_suffix
                )
                metamodel_highres_results_output_prefix = (
                    config.paths.high_res_outputs_dir / "metamodel" / _output_suffix
                )
                metamodel_stats = pd.read_csv(
                    f"{metamodel_output_prefix}.compare_model_scores.test_set_performance.tsv",
                    sep="\t",
                    index_col=0,
                )

                if metamodel_stats.shape[0] == 0:
                    logger.warning(f"Metamodel not run")
                    return

                for metamodel_name in (
                    [metamodel_stats.index[0]]
                    if show_best_metamodel_only
                    else [
                        "linearsvm_ovr",
                        "lasso_cv",
                        "ridge_cv",
                        "elasticnet_cv",
                        "xgboost",
                        "rf_multiclass",
                    ]
                ):
                    show(
                        [
                            [
                                f"{metamodel_output_prefix}.classification_report.test_set_performance.{metamodel_name}.txt",
                                f"{metamodel_highres_results_output_prefix}.errors_versus_difference_between_top_two_probabilities.test_set_performance.{metamodel_name}.with_abstention.vertical.png",
                                f"{metamodel_highres_results_output_prefix}.feature_importances.{metamodel_name}.raw_coefs.png"
                                if metamodel_name != "rf_multiclass"
                                else f"{metamodel_highres_results_output_prefix}.feature_importances.{metamodel_name}.all.png",
                                f"{metamodel_highres_results_output_prefix}.feature_importances.{metamodel_name}.absval_coefs.by_locus_and_model_component.png"
                                if metamodel_name != "rf_multiclass"
                                else f"{metamodel_highres_results_output_prefix}.feature_importances.{metamodel_name}.by_locus_and_model_component.png",
                            ],
                            [
                                f"{metamodel_output_prefix}.confusion_matrix.test_set_performance.{metamodel_name}.png",
                                # This one is only available for "disease":
                                f"{metamodel_highres_results_output_prefix}.confusion_matrix.test_set_performance.{metamodel_name}.expanded_confusion_matrix_disease_subtype.png",
                                f"{metamodel_highres_results_output_prefix}.confusion_matrix.test_set_performance.{metamodel_name}.expanded_confusion_matrix_ethnicity_condensed.png",
                                f"{metamodel_highres_results_output_prefix}.confusion_matrix.test_set_performance.{metamodel_name}.expanded_confusion_matrix_age_group_pediatric.png",
                            ],
                        ],
                        headers=[f"metamodel {metamodel_name}"],
                        max_width=400,
                    )
                    display(Markdown("---"))
    except FileNotFoundError as err:
        logger.warning(f"Not run: {err}")


# %% [markdown]
# # Overall summary

# %%
for target_obs_column in config.classification_targets:
    # Single-loci:
    for gene_locus in config.gene_loci_used:
        summary(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            show_base=True,
            show_metamodel=True,
        )

    # Multi-loci:
    if len(config.gene_loci_used) > 1:
        summary(
            gene_locus=config.gene_loci_used,
            target_obs_column=target_obs_column,
            show_base=False,
            show_metamodel=True,
        )

# %%

# %%
