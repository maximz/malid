# -*- coding: utf-8 -*-
# %% [markdown]
# # How does model3-rollup perform if allowed the same number of abstentions as model 2?
#
# Model 2's AUC is boosted by abstaining on difficult samples. How high could model 3's AUC get if it also got to abstain on the same number of samples?
#
# Compare model 3's AUC to model 2 more fairly by considering abstentions. If we assume that the N% abstentions are the worst predictions, what would an apples-to-apples AUC comparison be? Take out the worst N% of model 3's predictions (e.g. true 0 but predicted 0.99 -- rank by absolute difference from the truth), and recompute AUC.
#
# Multiclass way to implement this: true labels as one-hot vector, vs predicted probabilities vector; look at the difference of the two vectors: sum of squares or sum of absolute values. Rank by that difference. Take the bottom N%. Drop 'em. Get a new AUC for model 3 with N% of worst predictions removed.

# %%

# %%
import numpy as np
import pandas as pd

from malid import config, logger
from malid.external import model_evaluation
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    combine_classification_option_names,
)

# %%
from IPython.display import display, Markdown

# %%
from malid.trained_model_wrappers import (
    ConvergentClusterClassifier,
    SequenceClassifier,
    RollupSequenceClassifier,
)

# %%
target_obs_column = TargetObsColumnEnum.disease
sample_weight_strategy = SampleWeightStrategy.ISOTYPE_USAGE

# %%
model2_name = config.metamodel_base_model_names.model_name_convergent_clustering
model3_name = config.metamodel_base_model_names.model_name_sequence_disease


# %%

# %%

# %%

# %%
def process(gene_locus: GeneLocus):
    display(Markdown(f"## {gene_locus}"))
    GeneLocus.validate_single_value(gene_locus)

    convergent_cluster_models_base_dir = (
        ConvergentClusterClassifier._get_model_base_dir(
            gene_locus=gene_locus, target_obs_column=target_obs_column
        )
    )

    rollup_models_base_dir = RollupSequenceClassifier._get_model_base_dir(
        sequence_models_base_dir=SequenceClassifier._get_model_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )
    )

    # Load model2 **test set** performance
    model2_experiment_set = model_evaluation.ExperimentSet.load_from_disk(
        output_prefix=convergent_cluster_models_base_dir
        / "test_performance_of_train_smaller_model"
    )
    # Load model3-rollup **test set** performance
    model3_experiment_set = model_evaluation.ExperimentSet.load_from_disk(
        output_prefix=rollup_models_base_dir / "train_smaller_model"
    )

    # Remove global fold (we trained global fold model, but now get evaluation scores on cross-validation folds only)
    for experiment_set in [model2_experiment_set, model3_experiment_set]:
        # TODO: make kdict support: del self.model_outputs[:, fold_id]
        for key in experiment_set.model_outputs[:, -1].keys():
            logger.info(f"Removing {key} (global fold)")
            del experiment_set.model_outputs[key]

    model3_experiment_set_summary_original = model3_experiment_set.summarize()
    print("Original model3-rollup performance:")
    display(
        model3_experiment_set_summary_original.get_model_comparison_stats().loc[
            [model3_name]
        ]
    )
    print()

    model2_experiment_set_summary = model2_experiment_set.summarize()
    print("Original model2 performance:")
    display(
        model2_experiment_set_summary.get_model_comparison_stats().loc[[model2_name]]
    )
    print()
    print(
        f"Model 2 abstained on {model2_experiment_set_summary.model_global_performances[model2_name].abstention_proportion:0.2%} of samples"
    )

    # Sanity check: confirm that "sample_size including abstentions" matches between the two
    assert (
        model3_experiment_set.summarize()
        .get_model_comparison_stats()
        .loc[model3_name]["sample_size including abstentions"]
        == model2_experiment_set_summary.get_model_comparison_stats().loc[model2_name][
            "sample_size including abstentions"
        ]
    )

    # Discarded attempt: Make model 3 abstain on the exact same samples as model 2 did
    #     samples_to_abstain = (
    #         model2_experiment_set_summary.model_global_performances[model2_name]
    #         .cv_abstentions_metadata["specimen_label"]
    #         .values
    #     )

    # Update: Model 2 got to choose which samples to abstain on, so it has that advantage.
    # Let's instead try to remove the "worst offenders" in model 3 to see the best AUC we could possibly get with the same number of abstentions.
    # Not necessarily abstaining on the same exact samples as model 2 anymore

    y_preds_proba = model3_experiment_set_summary_original.model_global_performances[
        model3_name
    ].cv_y_preds_proba

    y_true_one_hot = pd.get_dummies(
        model3_experiment_set_summary_original.model_global_performances[
            model3_name
        ].cv_y_true_without_abstention
    ).reindex(
        columns=y_preds_proba.columns,
        fill_value=0,
    )

    # Get Euclidean distance between each y_true and accompanying y_preds_proba entry (alternative: sklearn cdist, then extract diagonal)
    differences = np.linalg.norm(y_true_one_hot - y_preds_proba, axis=1)
    assert differences.shape[0] == y_preds_proba.shape[0]

    # Get indices of top N highest-difference entries (same number of abstentions as in model2)
    indices_to_abstain_on = differences.argsort()[
        -model2_experiment_set_summary.model_global_performances[
            model2_name
        ].cv_abstentions.shape[0] :
    ]

    # Convert those to specimen labels
    samples_to_abstain = (
        model3_experiment_set_summary_original.model_global_performances[model3_name]
        .cv_metadata.iloc[indices_to_abstain_on]["index"]
        .values
    )

    print(f"Making model3-rollup abstain on: {samples_to_abstain}")
    revised_model3_outputs = model3_experiment_set.model_outputs[model3_name, :].copy()
    for (
        model_name,
        fold_id,
    ), model_single_fold_performance in revised_model3_outputs.items():
        mask = model_single_fold_performance.test_metadata.index.isin(
            samples_to_abstain
        )
        print(f"In fold {fold_id}, switching {mask.sum()} specimens to abstentions")
        revised_model3_outputs[
            model_name, fold_id
        ] = model_single_fold_performance.apply_abstention_mask(mask)

    model3_experiment_set_revised = model_evaluation.ExperimentSet(
        revised_model3_outputs
    )
    model3_experiment_set_revised_summary = model3_experiment_set_revised.summarize()

    # Sanity checks
    assert (
        model3_experiment_set_revised_summary.model_global_performances[
            model3_name
        ].n_abstentions
        == model2_experiment_set_summary.model_global_performances[
            model2_name
        ].n_abstentions
    )
    assert (
        model3_experiment_set_revised_summary.model_global_performances[
            model3_name
        ].sample_size_with_abstentions
        == model2_experiment_set_summary.model_global_performances[
            model2_name
        ].sample_size_with_abstentions
    )

    print()
    print("New model3-rollup performance:")
    display(model3_experiment_set_revised_summary.get_model_comparison_stats())

    # Export model3_experiment_set_revised_summary
    output_dir = (
        config.paths.sequence_models_output_dir
        / gene_locus.name
        / "rollup_models"
        / combine_classification_option_names(
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )
        / "with_abstentions_to_match_model2"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    model3_experiment_set_revised_summary.export_all_models(
        func_generate_classification_report_fname=lambda model_name: output_dir
        / f"sequence_prediction_rollup_with_abstentions_to_match_model2.{model_name}.train_smaller_model.report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: output_dir
        / f"sequence_prediction_rollup_with_abstentions_to_match_model2.{model_name}.train_smaller_model.confusion_matrix.png",
        confusion_matrix_pred_label="Rollup of sequence predictions (with model2's number of abstentions)",
        dpi=300,
    )
    model3_experiment_set_revised_summary.get_model_comparison_stats().to_csv(
        output_dir
        / f"sequence_prediction_rollup_with_abstentions_to_match_model2.train_smaller_model.compare_model_scores.tsv",
        sep="\t",
    )
    print(f"Exported to {output_dir}")


# %%

# %%
for gene_locus in config.gene_loci_used:
    process(gene_locus)
    print()
    print()

# %%
