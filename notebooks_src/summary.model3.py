# %%
from summarynb import show, indexed_csv, plaintext, chunks, table
from malid import config, logger, helpers
from malid.datamodels import (
    TargetObsColumnEnum,
    SampleWeightStrategy,
    combine_classification_option_names,
)
from slugify import slugify
import pandas as pd

# %%
from IPython.display import display, Markdown

# %%
# TODO: bring back sequence level model evaluation too.


# %% [markdown]
# # Predict disease/age/sex/ethnicity from sequences only (language model only)
#
# We train sequence model on train-smaller and eval on validation set. Then we rollup (with optional tuning) and eval on test set. That's below
#
# Note: run on same CV folds but subset to specimens without clear age/sex/ethnicity biases. But we kept same fine-tuned unirep, which was created from train-smaller + validation with more specimens than just this subset. This is OK.

# %%
model_names = [
    "lasso_multiclass",
    "lasso_multiclass-rollup_tuned",
    "dummy_stratified",
    "dummy_most_frequent",
    "dummy_stratified-rollup_tuned",
    "dummy_most_frequent-rollup_tuned",
]
for gene_locus in config.gene_loci_used:
    for target_obs_column in config.classification_targets:
        for sample_weight_strategy in [
            SampleWeightStrategy.ISOTYPE_USAGE,
        ]:
            display(
                Markdown(
                    f"### {gene_locus}, {target_obs_column}, {sample_weight_strategy}"
                )
            )
            try:
                show(
                    [
                        [
                            config.paths.sequence_models_output_dir
                            / gene_locus.name
                            / "rollup_models"
                            / combine_classification_option_names(
                                target_obs_column=target_obs_column,
                                sample_weight_strategy=sample_weight_strategy,
                            )
                            / f"sequence_prediction_rollup.{model_name}.train_smaller_model.report.txt"
                            for model_name in model_names
                        ],
                        [
                            config.paths.sequence_models_output_dir
                            / gene_locus.name
                            / "rollup_models"
                            / combine_classification_option_names(
                                target_obs_column=target_obs_column,
                                sample_weight_strategy=sample_weight_strategy,
                            )
                            / f"sequence_prediction_rollup.{model_name}.train_smaller_model.confusion_matrix.png"
                            for model_name in model_names
                        ],
                    ],
                    headers=model_names,
                    max_width=400,
                )
            except FileNotFoundError as err:
                logger.warning(f"Not run: {err}")

# %%
