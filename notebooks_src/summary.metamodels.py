# -*- coding: utf-8 -*-
# %% [markdown]
# # Metamodel performance comparisons
#
# * `disease_all_demographics_present` is `disease` prediction for only those samples with known age+sex+ethnicity.
#
# * `disease_all_demographics_present_regress_out_demographics`: we also ran metamodel after regressing out age+sex+ethnicity from metamodel's feature matrix. i.e. replace each column independently with residual `Y-Yhat` after fitting regression `Y ~ X`, where `Y` is the original column and `X` is the age+sex+ethnicity confounders all together. This is also called "orhogonalizing"  or "decorrelating". If performance suffers after we've decorrelated, then removing the effects of age/sex/ethnicity had a big impact.
#
#
# Here's how we made feature importances for multiclass OvR models:
#
# 1. Get coefs for each class versus the rest. Average them across folds (OK because input features to models are standardized) -> "raw coef" plots of mean and standard deviation across folds
# 2. Using the means: Convert to absolute value. Divide by sum of absolute values for each class -> percent contribution of each feature for a class -> "absval coef" plots
# 3. Sum the percent contributions of a set of features -> "absval coef" plots

# %%

# %%

# %%
from pathlib import Path
from summarynb import show, table, chunks
from malid.external.summarynb_extras import plaintext
from malid import config, logger
from malid.train import train_metamodel
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
)
import pandas as pd
from IPython.display import display, Markdown
from typing import Optional, List


# %%

# %%

# %%

# %%

# %%
def run_summary(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    metamodel_flavor_filter: Optional[List[str]] = None,
):
    base_model_train_fold_name = "train_smaller"
    metamodel_fold_label_train = "validation"
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
    for metamodel_flavor, metamodel_config in flavors.items():
        if (
            metamodel_flavor_filter is not None
            and len(metamodel_flavor_filter) > 0
            and metamodel_flavor not in metamodel_flavor_filter
        ):
            # Skip this metamodel flavor
            continue
        _output_suffix = (
            Path(gene_locus.name)
            / target_obs_column.name
            / metamodel_flavor
            / f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model"
        )
        results_output_prefix = (
            config.paths.second_stage_blending_metamodel_output_dir / _output_suffix
        )
        highres_results_output_prefix = (
            config.paths.high_res_outputs_dir / "metamodel" / _output_suffix
        )

        display(
            Markdown(
                f"# {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor}"
            )
        )
        print(metamodel_config)

        display(
            Markdown(
                "## Trained on validation set, performance on test set - with abstentions"
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
                                f"{results_output_prefix}.classification_report.test_set_performance.{model_name}.txt"
                            )
                            for model_name in model_names
                        ],
                        [
                            f"{results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.png"
                            for model_name in model_names
                        ],
                        #                 [
                        #                     f"{results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix.png"
                        #                     for model_name in model_names
                        #                 ],
                        # This one is only available for "disease":
                        [
                            f"{highres_results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix_disease_subtype.png"
                            for model_name in model_names
                        ],
                        [
                            f"{highres_results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix_ethnicity_condensed.png"
                            for model_name in model_names
                        ],
                        [
                            f"{highres_results_output_prefix}.confusion_matrix.test_set_performance.{model_name}.expanded_confusion_matrix_age_group_pediatric.png"
                            for model_name in model_names
                        ],
                        # diagnostics
                        [
                            f"{highres_results_output_prefix}.errors_versus_difference_between_top_two_predicted_probas.test_set_performance.{model_name}.with_abstention.vertical.png"
                            for model_name in model_names
                        ],
                        [
                            f"{highres_results_output_prefix}.errors_versus_difference_between_logits_of_top_two_classes.test_set_performance.{model_name}.with_abstention.vertical.png"
                            for model_name in model_names
                        ],
                    ],
                    max_width=500,
                    headers=model_names,
                )

            display(Markdown("---"))

            for name, fname in [
                ("cross validation folds", "feature_importances"),
                ("global fold", "feature_importances_global_fold"),
            ]:
                for model_name in ["rf_multiclass", "xgboost"]:
                    show(
                        [
                            f"{highres_results_output_prefix}.{fname}.{model_name}.all.png",
                            f"{highres_results_output_prefix}.{fname}.{model_name}.by_locus.png",
                            f"{highres_results_output_prefix}.{fname}.{model_name}.by_model_component.png",
                            f"{highres_results_output_prefix}.{fname}.{model_name}.by_locus_and_model_component.png",
                        ],
                        max_width=600,
                        max_height=None,
                        headers=[
                            f"{model_name} feature importances ({name}) - all",
                            "by locus",
                            "by model component",
                            "by locus and model component",
                        ],
                    )

                display(Markdown("---"))

                for model_name in [
                    "linearsvm_ovr",
                    "lasso_cv",
                    "ridge_cv",
                    "elasticnet_cv",
                    "lasso_multiclass",
                ]:
                    if Path(
                        f"{highres_results_output_prefix}.{fname}.{model_name}.raw_coefs.mean.png"
                    ).exists():
                        # Case 1: multiclass linear model
                        display(
                            Markdown(
                                f"### Feature importances {model_name} - raw ({name})"
                            )
                        )
                        if fname == "feature_importances":
                            show(
                                [
                                    f"{highres_results_output_prefix}.{fname}.{model_name}.raw_coefs.png",
                                    f"{highres_results_output_prefix}.{fname}.{model_name}.raw_coefs.mean.png",
                                    f"{highres_results_output_prefix}.{fname}.{model_name}.raw_coefs.stdev.png",
                                ],
                                max_width=600,
                                max_height=None,
                                headers=["combined", "mean", "standard deviation"],
                            )
                        elif fname == "feature_importances_global_fold":
                            show(
                                [
                                    f"{highres_results_output_prefix}.{fname}.{model_name}.raw_coefs.mean.png",
                                ],
                                max_width=600,
                                max_height=None,
                                headers=["global fold coefficients"],
                            )

                        display(
                            Markdown(
                                f"### Feature importances {model_name} - normalized absolute values ({name})"
                            )
                        )
                        show(
                            [
                                f"{highres_results_output_prefix}.{fname}.{model_name}.absval_coefs.all.png",
                                f"{highres_results_output_prefix}.{fname}.{model_name}.absval_coefs.by_locus.png",
                                f"{highres_results_output_prefix}.{fname}.{model_name}.absval_coefs.by_model_component.png",
                                f"{highres_results_output_prefix}.{fname}.{model_name}.absval_coefs.by_locus_and_model_component.png",
                            ],
                            max_width=600,
                            max_height=None,
                            headers=[
                                f"Feature coefficients - all",
                                "by locus",
                                "by model component",
                                "by locus and model component",
                            ],
                        )
                    elif Path(
                        f"{highres_results_output_prefix}.{fname}.{model_name}.all.png"
                    ).exists():
                        # Case 2: binary linear model
                        show(
                            [
                                f"{highres_results_output_prefix}.{fname}.{model_name}.all.png",
                            ],
                            max_width=600,
                            max_height=None,
                            headers=[
                                f"{model_name} feature coefficients - all ({name})",
                            ],
                        )
                    else:
                        logger.warning(f"No feature impotrances found for {model_name}")

                    display(Markdown("---"))

            for model_name in [
                "lasso_cv",
                "ridge_cv",
                "elasticnet_cv",
            ]:
                display(
                    Markdown(f"### Hyperparameter tuning diagnostics: {model_name}")
                )
                show(
                    [
                        f"{highres_results_output_prefix}.internal_cross_validation_hyperparameter_diagnostics.{model_name}.fold_{fold_id}.png"
                        for fold_id in config.all_fold_ids
                    ],
                    headers=[f"Fold {fold_id}" for fold_id in config.all_fold_ids],
                    max_width=500,
                )

        except FileNotFoundError as err:
            print(f"Not yet run: {err}")


# %%

# %%
# Individual gene locus
for gene_locus in config.gene_loci_used:
    print(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    for target_obs_column in config.classification_targets:
        run_summary(gene_locus=gene_locus, target_obs_column=target_obs_column)

# %%
# Together in combined metamodel
if len(config.gene_loci_used) > 1:
    print(config.gene_loci_used)
    for target_obs_column in config.classification_targets:
        run_summary(
            gene_locus=config.gene_loci_used, target_obs_column=target_obs_column
        )

# %%

# %%

# %%

# %%

# %%
for gene_locus in config.gene_loci_used:
    run_summary(
        gene_locus=gene_locus,
        target_obs_column=TargetObsColumnEnum.disease,
        metamodel_flavor_filter=["default"],
    )
run_summary(
    gene_locus=config.gene_loci_used,
    target_obs_column=TargetObsColumnEnum.disease,
    metamodel_flavor_filter=["default"],
)

# %%
