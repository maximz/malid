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

# %%

# %%

# %%
# Run
for gene_locus in config.gene_loci_used:
    map_targets_to_output_dir = {
        target_obs_column: RepertoireClassifier._get_output_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=config.sample_weight_strategy,
        )  # output base dir should already exist
        for target_obs_column in config.classification_targets
    }
    print(gene_locus)
    clfs = tune_on_validation_set(
        gene_locus=gene_locus,
        targets=map_targets_to_output_dir,
        sample_weight_strategy=config.sample_weight_strategy,
        model_names=config.model_names_to_train,
        model_class=RepertoireClassifier,
        # Model 1 does not require the embedding .X, so take the fast path and just load .obs:
        load_obs_only=True,
    )
    evaluate_original_and_tuned_on_test_set(
        clfs=clfs,
        gene_locus=gene_locus,
        targets=map_targets_to_output_dir,
        sample_weight_strategy=config.sample_weight_strategy,
        # Model 1 does not require the embedding .X, so take the fast path and just load .obs:
        load_obs_only=True,
    )


# %%
