# %% [markdown]
# # Tune "train-smaller" repertoire stats model decision thresholds on validation set, and report performance (with and without tuning) on test set
#
# This is run using train-smaller+validation+test, not train+test sets.
#
# Use trained model APIs to do this.

# %%
from malid import config
from malid.train.training_utils import (
    evaluate_original_and_tuned_on_test_set,
    tune_on_validation_set,
)
from malid.trained_model_wrappers import RepertoireClassifier
from malid.datamodels import (
    combine_classification_option_names,
)

# %%

# %%

# %%
# Run
for gene_locus in config.gene_loci_used:
    map_targets_to_output_dir = {
        target_obs_column: (
            config.paths.repertoire_stats_classifier_output_dir
            / gene_locus.name
            / combine_classification_option_names(target_obs_column)
        )  # output base dir should already exist
        for target_obs_column in config.classification_targets
    }
    print(gene_locus)
    clfs = tune_on_validation_set(
        gene_locus=gene_locus,
        targets=map_targets_to_output_dir,
        model_names=[
            "lasso_multiclass",
            "lasso_cv",
            "ridge_cv",
            "elasticnet_cv",
            "rf_multiclass",
            "xgboost",
            "linearsvm_ovr",
        ],
        model_class=RepertoireClassifier,
    )
    evaluate_original_and_tuned_on_test_set(
        clfs=clfs, gene_locus=gene_locus, targets=map_targets_to_output_dir
    )


# %%
