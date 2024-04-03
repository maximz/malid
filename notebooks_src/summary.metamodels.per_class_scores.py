# -*- coding: utf-8 -*-
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
    map_cross_validation_split_strategy_to_default_target_obs_column,
)
import pandas as pd
from IPython.display import display, Markdown
from typing import Optional, List


# %%
config.embedder.name

# %%
# get default classification target, e.g. TargetObsColumnEnum.disease
default_target_obs_column = (
    map_cross_validation_split_strategy_to_default_target_obs_column[
        config.cross_validation_split_strategy
    ]
)
default_target_obs_column

# %%

# %% [markdown]
# # Metamodel per-class OvR ROC AUC scores
#
# Abstentions consistent across metamodel flavors from the same gene locus. e.g. the model1-only metamodel is forced to abstain wherever the model2-only metamodel abstained.
#
# Color scales consistent between metamodel names.
#
# Also includes a row for the average-across-class-pairs OvO score we normally report, but again now with consistent abstentions across flavors.
#
# Note that BCR and BCR+TCR are not comparable, because the sample sizes are different. (TCR and BCR+TCR have the same sample size because there are no TCR-only cohorts, but still are not comparable because abstentions are not forced to be identical across GeneLocus settings.)

# %%
for model_name in [
    "lasso_cv",
    "elasticnet_cv",
    "ridge_cv",
    "rf_multiclass",
    "linearsvm_ovr",
]:
    show(
        config.paths.second_stage_blending_metamodel_output_dir
        / f"{default_target_obs_column.name}.roc_auc_per_class.{model_name}.png",
        max_width=1200,
    )

# %%


# %% [markdown]
# # Metamodel pairwise ROC AUC scores
#
# Headers are the multiclass weighted ROC AUC scores


# %%
def pairwise_summary(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
):
    base_model_train_fold_name = "train_smaller"
    metamodel_fold_label_train = "validation"
    display(Markdown(f"## {gene_locus}, {target_obs_column}"))
    try:
        flavors = train_metamodel.get_metamodel_flavors(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            fold_id=config.all_fold_ids[0],
            base_model_train_fold_name=base_model_train_fold_name,
            use_stubs_instead_of_submodels=True,
        )
    except Exception as err:
        logger.warning(
            f"Failed to generate metamodel flavors for {gene_locus}, {target_obs_column}: {err}"
        )
        return
    for metamodel_flavor, metamodel_config in flavors.items():
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

        # Load multiclass ROC AUC scores for these models
        # (show mean across cross validation folds - remove +/- stddev)
        try:
            multiclass_scores = (
                pd.read_csv(
                    f"{results_output_prefix}.compare_model_scores.test_set_performance.tsv",
                    sep="\t",
                    index_col=0,
                )["ROC-AUC (weighted OvO) per fold"]
                .str.split(" +/-", regex=False)
                .str[0]
            )
        except FileNotFoundError as err:
            # This can happen because use_stubs_instead_of_submodels=True above means some non-existent flavors will be generated
            logger.warning(
                f"File not found for {gene_locus}, {target_obs_column}, flavor {metamodel_flavor}: {err}"
            )
            continue

        display(Markdown(f"#### {metamodel_flavor}"))

        model_names = config.model_names_to_train
        show(
            [
                f"{highres_results_output_prefix}.pairwise_roc_auc_scores.{model_name}.png"
                for model_name in model_names
            ],
            max_width=400,
            headers=[
                f"{model_name}: {multiclass_scores.loc[model_name]}"
                for model_name in model_names
            ],
        )


# %%

# %%
pairwise_summary(
    gene_locus=config.gene_loci_used, target_obs_column=default_target_obs_column
)

# %%

# %%
